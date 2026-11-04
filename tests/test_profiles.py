"""Profile resolution uses synthetic homes only; no account or cloud access."""
import argparse
import contextlib
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from colabcola import profiles
from colabcola.lifecycle_io import LifecycleError


class ProfileTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="colabcola-profiles-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.home = self.root / "synthetic-home"
        self.home.mkdir()
        environment = mock.patch.dict(os.environ, {
            "HOME": str(self.home), "USERPROFILE": str(self.home),
        })
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop("COLABCOLA_CONFIG_HOME", None)
        os.environ.pop("COLABCOLA_PROFILE", None)
        home = mock.patch("pathlib.Path.home", return_value=self.home)
        home.start()
        self.addCleanup(home.stop)
        self.legacy = self.home / ".config" / "colabcola"
        self.default = self.home / ".colabcola" / "configs" / "default"

    def test_error_is_lifecycle_error(self):
        self.assertTrue(issubclass(profiles.ProfileError, LifecycleError))

    def test_fresh_default_creates_nothing(self):
        self.assertEqual(profiles.config_home(), self.default)
        self.assertEqual(list(self.home.iterdir()), [])

    def test_existing_named_directory_does_not_switch_default(self):
        other = self.home / ".colabcola" / "configs" / "work"
        other.mkdir(parents=True)
        self.assertEqual(profiles.config_home(), self.default)
        self.assertFalse(self.default.exists())
        self.assertEqual(list(other.iterdir()), [])

    def test_legacy_is_preserved_without_reading_or_migrating(self):
        self.legacy.mkdir(parents=True)
        marker = self.legacy / "marker.txt"
        marker.write_bytes(b"synthetic legacy marker\n")
        before = marker.stat()
        with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("unexpected file read")), \
                mock.patch.object(Path, "read_text", side_effect=AssertionError("unexpected file read")):
            self.assertEqual(profiles.config_home(), self.legacy)
        self.assertEqual(marker.read_bytes(), b"synthetic legacy marker\n")
        self.assertEqual(marker.stat().st_mtime_ns, before.st_mtime_ns)
        self.assertFalse((self.home / ".colabcola").exists())

    def test_environment_profile_wins_over_legacy(self):
        self.legacy.mkdir(parents=True)
        os.environ["COLABCOLA_PROFILE"] = "work"
        expected = self.home / ".colabcola" / "configs" / "work"
        self.assertEqual(profiles.config_home(), expected)
        self.assertFalse(expected.exists())

    def test_environment_home_wins_even_over_invalid_profile(self):
        self.legacy.mkdir(parents=True)
        explicit = self.root / "explicit"
        os.environ["COLABCOLA_CONFIG_HOME"] = str(explicit)
        os.environ["COLABCOLA_PROFILE"] = "../invalid"
        self.assertEqual(profiles.config_home(), explicit)
        self.assertFalse(explicit.exists())

    def test_empty_environment_selectors_are_unset(self):
        os.environ["COLABCOLA_CONFIG_HOME"] = ""
        os.environ["COLABCOLA_PROFILE"] = ""
        self.assertEqual(profiles.config_home(), self.default)
        os.environ["COLABCOLA_PROFILE"] = "work"
        self.assertEqual(profiles.config_home(), profiles.named_home("work"))

    def test_environment_home_expands_user_and_becomes_absolute(self):
        os.environ["COLABCOLA_CONFIG_HOME"] = "~/selected"
        self.assertEqual(profiles.config_home(), self.home / "selected")
        with contextlib.chdir(self.home):
            os.environ["COLABCOLA_CONFIG_HOME"] = "relative"
            self.assertEqual(profiles.config_home(), self.home / "relative")

    def test_names_are_portable_and_bounded(self):
        for name in ("a", "0", "default", "work_2-local", "a" * 64, "com0", "com10"):
            with self.subTest(name=name):
                self.assertEqual(profiles.named_home(name), self.home / ".colabcola" / "configs" / name)
        self.assertEqual(list(self.home.iterdir()), [])

    def test_profile_name_identifies_only_exact_valid_named_children(self):
        self.assertEqual(profiles.profile_name(self.default), "default")
        self.assertEqual(profiles.profile_name(profiles.named_home("work")), "work")
        for path in (self.legacy, self.default / "child", self.default.parent,
                     self.default.with_name("CON"), self.default.with_name("con"),
                     self.default.with_name("bad name"), self.root / "work"):
            with self.subTest(path=path):
                self.assertIsNone(profiles.profile_name(path))
        self.assertEqual(list(self.home.iterdir()), [])

    def test_profile_name_ignores_unrelated_named_root_links_or_access_errors(self):
        named_root = self.default.parent
        for unavailable in (False, True):
            def link_check(path):
                if path == named_root:
                    if unavailable:
                        raise PermissionError("synthetic unavailable named root")
                    return True
                return False
            with self.subTest(unavailable=unavailable):
                with mock.patch.object(Path, "is_symlink", link_check):
                    self.assertIsNone(profiles.profile_name(self.legacy))
                    self.assertIsNone(profiles.profile_name(self.root / "custom"))
        self.assertEqual(list(self.home.iterdir()), [])

    def test_profile_name_still_checks_selected_path_links(self):
        with mock.patch.object(Path, "is_symlink", lambda path: path == self.legacy):
            with self.assertRaisesRegex(profiles.ProfileError, "links or junctions"):
                profiles.profile_name(self.legacy)

    def test_dangling_legacy_link_does_not_select_fresh_default(self):
        with mock.patch.object(Path, "is_symlink", lambda path: path == self.legacy):
            with self.assertRaisesRegex(profiles.ProfileError, "links or junctions"):
                profiles.config_home()
        self.assertEqual(list(self.home.iterdir()), [])

    def test_native_dangling_legacy_symlink_does_not_select_fresh_default(self):
        self.legacy.parent.mkdir(parents=True)
        try:
            self.legacy.symlink_to(self.root / "absent-target", target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            self.skipTest("native symlinks unavailable: " + type(exc).__name__)
        self.assertFalse(self.legacy.exists())
        with self.assertRaisesRegex(profiles.ProfileError, "links or junctions"):
            profiles.config_home()
        self.assertFalse((self.home / ".colabcola").exists())

    @unittest.skipUnless(os.name == "nt" and hasattr(Path, "is_junction"), "native Windows junction test")
    def test_native_dangling_legacy_junction_does_not_select_fresh_default(self):
        target = self.root / "legacy-junction-target"
        target.mkdir()
        self.legacy.parent.mkdir(parents=True)
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(self.legacy), str(target)],
                                capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        self.assertEqual(result.returncode, 0, result.stderr)
        try:
            target.rmdir()
            self.assertFalse(self.legacy.exists())
            self.assertTrue(self.legacy.is_junction())
            with self.assertRaisesRegex(profiles.ProfileError, "links or junctions"):
                profiles.config_home()
            self.assertFalse((self.home / ".colabcola").exists())
        finally:
            self.legacy.rmdir()

    def test_invalid_names_fail_without_creating_anything(self):
        names = (None, 1, "", ".", "..", "../other", "a/b", "a\\b", "A", "a b", "a.",
                 "_a", "-a", "a" * 65, "a\n", "a\0b", "caf\u00e9", "a:b", " a", "a ")
        for name in names:
            with self.subTest(name=name), self.assertRaises(profiles.ProfileError):
                profiles.named_home(name)
        self.assertEqual(list(self.home.iterdir()), [])

    def test_windows_device_aliases_are_rejected_on_every_platform(self):
        names = ["con", "prn", "aux", "nul"]
        names += [prefix + str(i) for prefix in ("com", "lpt") for i in range(1, 10)]
        names += ["CON", "con.txt", "nul.", "aux ", "com1:", "lpt\u00b9", "com\u00b2"]
        for name in names:
            with self.subTest(name=name), self.assertRaises(profiles.ProfileError):
                profiles.named_home(name)

    def test_invalid_environment_profile_is_not_a_default_fallback(self):
        os.environ["COLABCOLA_PROFILE"] = "../other"
        with self.assertRaises(profiles.ProfileError):
            profiles.config_home()
        self.assertEqual(list(self.home.iterdir()), [])

    def test_explicit_profile_overrides_and_restores_both_environment_values(self):
        os.environ["COLABCOLA_CONFIG_HOME"] = str(self.root / "old")
        os.environ["COLABCOLA_PROFILE"] = "old-profile"
        before = dict(os.environ)
        with profiles.selected(profile="work") as path:
            self.assertEqual(path, profiles.named_home("work"))
            self.assertEqual(os.environ["COLABCOLA_CONFIG_HOME"], str(path))
            self.assertNotIn("COLABCOLA_PROFILE", os.environ)
            self.assertEqual(profiles.config_home(), path)
        self.assertEqual(dict(os.environ), before)
        self.assertFalse(path.exists())

    def test_explicit_home_overrides_and_restores_empty_environment_values(self):
        os.environ["COLABCOLA_CONFIG_HOME"] = ""
        os.environ["COLABCOLA_PROFILE"] = ""
        before = dict(os.environ)
        with profiles.selected(config_home="~/explicit") as path:
            self.assertEqual(path, self.home / "explicit")
            self.assertEqual(profiles.config_home(), path)
            self.assertNotIn("COLABCOLA_PROFILE", os.environ)
        self.assertEqual(dict(os.environ), before)

    def test_selection_restores_absent_environment_after_exception(self):
        before = dict(os.environ)
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            with profiles.selected(profile="work"):
                os.environ["COLABCOLA_PROFILE"] = "changed"
                raise RuntimeError("fixture failure")
        self.assertEqual(dict(os.environ), before)

    def test_private_config_homes_includes_active_and_shadowed_custom_roots(self):
        old = self.root / "arbitrary-custom-root"
        os.environ["COLABCOLA_CONFIG_HOME"] = str(old)
        before = dict(os.environ)
        with profiles.selected(profile="work") as selected:
            self.assertEqual(profiles.private_config_homes(), (selected, old))
            self.assertEqual(set(os.environ), set(before))
        self.assertEqual(profiles.private_config_homes(), (old,))
        self.assertEqual(dict(os.environ), before)
        self.assertFalse(old.exists())
        self.assertEqual(list(self.home.iterdir()), [])

    def test_private_config_homes_nested_selections_restore_after_failure(self):
        old = self.root / "original-custom-root"
        inner = self.root / "inner-custom-root"
        os.environ["COLABCOLA_CONFIG_HOME"] = str(old)
        with profiles.selected(profile="outer") as outer:
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                with profiles.selected(config_home=inner):
                    self.assertEqual(profiles.private_config_homes(), (inner, old, outer))
                    raise RuntimeError("synthetic failure")
            self.assertEqual(profiles.private_config_homes(), (outer, old))
        self.assertEqual(profiles.private_config_homes(), (old,))
        os.environ.pop("COLABCOLA_CONFIG_HOME")
        self.assertEqual(profiles.private_config_homes(), (self.default,))

    def test_private_config_homes_deduplicates_repeated_selection(self):
        old = self.root / "custom-root"
        os.environ["COLABCOLA_CONFIG_HOME"] = str(old)
        with profiles.selected(config_home=old):
            with profiles.selected(config_home=old):
                self.assertEqual(profiles.private_config_homes(), (old,))
        self.assertEqual(profiles.private_config_homes(), (old,))

    def test_shadowed_config_path_is_lexical_and_never_inspected(self):
        old = self.home / "old" / "private"
        for unavailable in (False, True):
            def link_check(path):
                if path in (old, old.parent):
                    if unavailable:
                        raise PermissionError("synthetic inaccessible old root")
                    return True
                return False
            with self.subTest(unavailable=unavailable):
                os.environ["COLABCOLA_CONFIG_HOME"] = "~/old/private"
                with mock.patch.object(Path, "is_symlink", link_check):
                    with profiles.selected(profile="work") as selected:
                        self.assertEqual(profiles.private_config_homes(), (selected, old))
                self.assertEqual(os.environ["COLABCOLA_CONFIG_HOME"], "~/old/private")
        self.assertEqual(list(self.home.iterdir()), [])

    def test_empty_ambient_home_does_not_add_exclusion_metadata(self):
        os.environ["COLABCOLA_CONFIG_HOME"] = ""
        with profiles.selected(profile="work") as selected:
            self.assertEqual(profiles.private_config_homes(), (selected,))
        self.assertEqual(profiles.private_config_homes(), (self.default,))

    def test_invalid_explicit_selection_does_not_retain_shadow_metadata(self):
        old = self.root / "original-custom-root"
        os.environ["COLABCOLA_CONFIG_HOME"] = str(old)
        with self.assertRaises(profiles.ProfileError):
            with profiles.selected(profile="../invalid"):
                self.fail("invalid selection entered")
        os.environ.pop("COLABCOLA_CONFIG_HOME")
        self.assertEqual(profiles.private_config_homes(), (self.default,))

    def test_nested_explicit_selections_restore_outer_selection(self):
        with profiles.selected(profile="outer") as outer:
            with profiles.selected(config_home=self.root / "inner") as inner:
                self.assertEqual(profiles.config_home(), inner)
            self.assertEqual(profiles.config_home(), outer)
        self.assertNotIn("COLABCOLA_CONFIG_HOME", os.environ)
        self.assertNotIn("COLABCOLA_PROFILE", os.environ)

    def test_no_explicit_selection_does_not_pin_or_restore_environment(self):
        with profiles.selected() as path:
            self.assertEqual(path, self.default)
            self.assertNotIn("COLABCOLA_CONFIG_HOME", os.environ)
            os.environ["COLABCOLA_PROFILE"] = "session"
            self.assertEqual(profiles.config_home(), profiles.named_home("session"))
        self.assertEqual(os.environ["COLABCOLA_PROFILE"], "session")

    def test_bad_explicit_selectors_leave_environment_unchanged(self):
        os.environ["COLABCOLA_CONFIG_HOME"] = str(self.root / "old")
        os.environ["COLABCOLA_PROFILE"] = "old-profile"
        before = dict(os.environ)
        for arguments in ({"profile": "work", "config_home": self.root}, {"profile": ""},
                          {"profile": "../other"}, {"config_home": ""}, {"config_home": 10}):
            with self.subTest(arguments=arguments), self.assertRaises(profiles.ProfileError):
                with profiles.selected(**arguments):
                    self.fail("invalid selection entered")
            self.assertEqual(dict(os.environ), before)

    def test_link_in_any_selected_path_parent_is_rejected(self):
        target = self.home / "linked" / "not-created"
        for linked in (target, target.parent, self.home):
            with self.subTest(linked=linked):
                with mock.patch.object(Path, "is_symlink", lambda path: path == linked):
                    with self.assertRaisesRegex(profiles.ProfileError, "links or junctions"):
                        with profiles.selected(config_home=target):
                            self.fail("linked selection entered")
        self.assertEqual(list(self.home.iterdir()), [])

    def test_native_symlink_parent_is_rejected(self):
        target = self.root / "target"
        target.mkdir()
        link = self.home / "linked"
        try:
            link.symlink_to(target, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            self.skipTest("native symlinks unavailable: " + type(exc).__name__)
        with self.assertRaisesRegex(profiles.ProfileError, "links or junctions"):
            with profiles.selected(config_home=link / "not-created"):
                self.fail("linked selection entered")
        self.assertEqual(list(target.iterdir()), [])

    @unittest.skipUnless(os.name == "nt" and hasattr(Path, "is_junction"), "native Windows junction test")
    def test_native_windows_junction_parent_is_rejected(self):
        target = self.root / "junction-target"
        target.mkdir()
        link = self.home / "junction"
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                                capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        self.assertEqual(result.returncode, 0, result.stderr)
        try:
            self.assertTrue(link.is_junction())
            with self.assertRaisesRegex(profiles.ProfileError, "links or junctions"):
                with profiles.selected(config_home=link / "not-created"):
                    self.fail("junction selection entered")
            self.assertEqual(list(target.iterdir()), [])
        finally:
            link.rmdir()

    def test_argument_helper_and_duplicate_rejection(self):
        parser = argparse.ArgumentParser()
        profiles.add_profile_argument(parser)
        self.assertIsNone(parser.parse_args([]).profile)
        self.assertEqual(parser.parse_args(["--profile", "work"]).profile, "work")
        self.assertIn("--profile NAME", parser.format_help())
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors), self.assertRaises(SystemExit) as error:
            parser.parse_args(["--profile", "work", "--profile", "work"])
        self.assertEqual(error.exception.code, 2)
        self.assertIn("--profile may be specified only once", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
