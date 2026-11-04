"""Bundled provider identity, isolation and fresh-install metadata contracts."""
import hashlib
import importlib.resources
import json
import os
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest import mock

from colabcola.lifecycle_io import LifecycleError
from colabcola.lifecycle_provider import _verify_bundle, _upstream_bytes, auth_path, dependency_report


class VendorTests(unittest.TestCase):
    def test_every_bundled_file_has_reproducible_upstream_provenance(self):
        manifest = _verify_bundle()
        root = importlib.resources.files("colabcola._vendor")
        for name, provider in manifest["providers"].items():
            self.assertEqual(len(provider["commit"]), 40)
            self.assertIn("LICENSE", provider["files"])
            for filename, record in provider["files"].items():
                if record.get("generated"):
                    continue
                body = root.joinpath(name, filename).read_bytes()
                original = _upstream_bytes(body) if filename.endswith(".py") else body
                self.assertEqual(hashlib.sha256(original).hexdigest(), record["upstream_sha256"], filename)

    def test_ambient_upstream_packages_cannot_replace_bundled_provider(self):
        with mock.patch.dict("sys.modules", {"colab_cli": None, "jupyter_kernel_client": None}), \
             mock.patch("colabcola.lifecycle_provider.credentials", side_effect=AssertionError("credential access")):
            report = dependency_report()
        self.assertTrue(report["provider_source_verified"])
        self.assertEqual(report["distribution"], "bundled_private_modules")

    def test_tampered_license_or_source_fails_before_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {"schema": 1, "providers": {
                "colab_cli": {"commit": "79cd126383e72b1e911f20b7030e4247f7861704", "files": {"LICENSE": {"sha256": "0" * 64}}},
                "jupyter_kernel_client": {"commit": "f18e982c3265df5e923aa9def101ab3fd737e139", "files": {"LICENSE": {"sha256": "0" * 64}}}}}
            (root / "provenance.json").write_text(json.dumps(manifest))
            (root / "colab_cli").mkdir()
            (root / "colab_cli/LICENSE").write_text("modified")
            with mock.patch("colabcola.lifecycle_provider.importlib.resources.files", return_value=root):
                with self.assertRaisesRegex(LifecycleError, "integrity failed"):
                    _verify_bundle()

    def test_explicit_config_home_isolated_from_existing_colab_login(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(os.environ, {"COLABCOLA_CONFIG_HOME": temporary}):
            self.assertEqual(auth_path(), Path(temporary) / "colab-oauth.json")
            self.assertFalse(auth_path().exists())

    def test_runtime_and_traitlets_resolve_private_kernel_classes(self):
        from colabcola._vendor.colab_cli.runtime import ColabRuntime
        from colabcola._vendor import jupyter_kernel_client as jkc
        runtime = ColabRuntime("https://invalid", "placeholder", kernel_id="owned", session_id="session")
        with mock.patch.object(jkc.KernelClient, "start"), \
             mock.patch.object(jkc.KernelHttpManager, "refresh_model"), \
             mock.patch("requests.sessions.Session.request", side_effect=AssertionError("network access")) as network, \
             mock.patch.object(ColabRuntime, "_apply_ws_hook"):
            client = runtime.kernel_client
        network.assert_not_called()
        self.assertIsInstance(client, jkc.KernelClient)
        self.assertEqual(client.kernel_manager_class.__module__, "colabcola._vendor.jupyter_kernel_client.manager")
        self.assertEqual(client._manager.client_factory.__module__, "colabcola._vendor.jupyter_kernel_client.wsclient")
        client._manager = None  # No connection was opened; avoid lazy cleanup.

    def test_default_install_has_public_dependencies_and_no_private_wheel_requirements(self):
        config = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
        dependencies = config["project"]["dependencies"]
        self.assertTrue(any(value.startswith("jupyter-client") for value in dependencies))
        self.assertTrue(any(value.startswith("pydantic") for value in dependencies))
        self.assertFalse(any("colabcola1" in value or "f18e982c" in value or " @ " in value for value in dependencies))
        self.assertEqual(config["project"]["optional-dependencies"]["colab"], [])


if __name__ == "__main__":
    unittest.main()
