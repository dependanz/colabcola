import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/gaussian_avatar"

def module(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLE / "gaussian_avatar" / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

class GaussianAvatarExampleTests(unittest.TestCase):
    def fixture(self, root):
        source = root / "sequence"; source.mkdir()
        flame = root / "flame"; flame.mkdir()
        for name in ("flame2023.pkl", "FLAME_masks.pkl"):
            (flame / name).write_bytes(b"structural test fixture only")
        (source / "canonical_flame_param.npz").write_bytes(b"canonical fixture")
        frames = []
        for i in range(3):
            (source / f"im{i}.png").write_bytes(b"fixture")
            (source / f"param{i}.npz").write_bytes(b"fixture")
            frames.append(dict(file_path=f"im{i}", flame_param_path=f"param{i}.npz", timestep_index=10+i,
                transform_matrix=[[1,0,0,i],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
        for split in ("train", "val", "test"):
            (source / f"transforms_{split}.json").write_text(json.dumps(dict(camera_angle_x=1, frames=frames if split=="train" else [])))
        return source, flame

    def test_subset_is_self_contained_and_remaps_timestamps(self):
        assets = module("assets")
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source, flame=self.fixture(root); out=root/"subset"
            result=assets.subset(source,out,2)
            self.assertEqual(result["copied_files"],5)
            self.assertEqual(assets.inspect(out,flame)["frames"],dict(train=2,val=0,test=0))
            frames=json.loads((out/"transforms_train.json").read_text())["frames"]
            self.assertEqual([f["timestep_index"] for f in frames],[0,1])
            self.assertEqual((out/"canonical_flame_param.npz").read_bytes(), b"canonical fixture")
            self.assertFalse((out/"im2.png").exists())
            with self.assertRaises(ValueError): assets.subset(source,out,2)

    def test_missing_and_escaping_inputs_fail_before_subset_write(self):
        assets=module("assets")
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source,flame=self.fixture(root)
            for bad in ("../outside", "C:/outside", "/outside", "a\\b"):
                with self.assertRaises(ValueError): assets.contained(source,bad)
            (flame/"flame2023.pkl").unlink()
            with self.assertRaises(ValueError): assets.inspect(source,flame)
            data=json.loads((source/"transforms_train.json").read_text())
            data["frames"][0]["file_path"]="../outside"
            (source/"transforms_train.json").write_text(json.dumps(data))
            with self.assertRaises(ValueError): assets.subset(source,root/"subset",2)
            self.assertFalse((root/"subset").exists())

    def test_loader_selection_inputs_are_required_before_copy(self):
        assets = module("assets")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source, flame = self.fixture(root)
            canonical = source / "canonical_flame_param.npz"
            canonical.unlink()
            with self.assertRaisesRegex(ValueError, "canonical_flame_param"):
                assets.inspect(source, flame)
            with self.assertRaises(ValueError): assets.subset(source, root/"out", 2)
            self.assertFalse((root/"out").exists())
            canonical.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "empty canonical"):
                assets.inspect(source, flame)
            canonical.write_bytes(b"fixture")
            (source/"sparse").mkdir()
            with self.assertRaisesRegex(ValueError, "Colmap"):
                assets.inspect(source, flame)

    def test_exhausted_build_budget_does_not_spawn(self):
        pilot=module("pilot")
        with patch.object(pilot.subprocess,"run") as run:
            with self.assertRaises(TimeoutError): pilot.run_command(["pip"],0,Path("unused"))
            run.assert_not_called()

    def test_failed_native_command_keeps_compiler_output(self):
        import time
        pilot = module("pilot")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            argv = [sys.executable, "-c", "import sys; print('compiler diagnostic', file=sys.stderr); sys.exit(7)"]
            with self.assertRaises(subprocess.CalledProcessError):
                pilot.run_command(argv, time.monotonic()+30, root)
            text = (root/"build.log").read_text(encoding="utf-8")
            self.assertIn("compiler diagnostic", text)
            self.assertIn("FAILED exit=7", text)

    def test_compatibility_patch_is_bound_idempotent_and_rejects_unknown_source(self):
        import hashlib
        compat = module("compat")
        original = b"#include <iostream>\n"
        fixed = b"#include <cstdint>\n" + original
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); header = root/compat.HEADER
            header.parent.mkdir(parents=True); header.write_bytes(original)
            with patch.object(compat, "BEFORE", hashlib.sha256(original).hexdigest()), patch.object(compat, "AFTER", hashlib.sha256(fixed).hexdigest()):
                self.assertEqual(compat.apply(root)["state"], "applied")
                self.assertEqual(header.read_bytes(), fixed)
                self.assertEqual(compat.apply(root)["state"], "already_applied")
                header.write_bytes(b"changed source")
                with self.assertRaises(ValueError): compat.apply(root)
                self.assertEqual(header.read_bytes(), b"changed source")

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow preview fixture")
    def test_preview_pairs_exact_pixels_and_rejects_missing_views(self):
        from PIL import Image
        preview = module("preview")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root/"rendered"
            for name, color in (("gt", (255,0,0)), ("renders", (0,255,0))):
                (source/name).mkdir(parents=True)
                Image.new("RGB", (32,48), color).save(source/name/"00000.png")
            with self.assertRaisesRegex(ValueError,"missing"):
                preview.compose(source, root/"bad", 2)
            self.assertFalse((root/"bad").exists())
            records = preview.compose(source, root/"pairs", 1)
            self.assertEqual(len(records),1)
            with Image.open(root/"pairs/pair-00.png") as pair:
                self.assertEqual(pair.getpixel((0,36)), (255,0,0))
                self.assertEqual(pair.getpixel((32,36)), (0,255,0))
            with self.assertRaisesRegex(ValueError,"already exists"):
                preview.compose(source, root/"pairs", 1)

    @unittest.skipUnless(importlib.util.find_spec("PIL") and importlib.util.find_spec("numpy"), "image comparison dependencies")
    def test_comparison_measures_improvement_and_rejects_changed_inputs(self):
        from PIL import Image
        comparison = module("compare")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for step, color in ((200, (100,100,100)), (5000, (20,20,20))):
                for folder in ("gt", "renders"):
                    dest = root/"train"/f"ours_{step}"/folder
                    dest.mkdir(parents=True)
                    Image.new("RGB", (32,48), (0,0,0) if folder=="gt" else color).save(dest/"00000.png")
            report = comparison.compare(root,root/"comparison",views=1)
            self.assertAlmostEqual(report["mean_mse_reduction_fraction"], .96)
            with Image.open(root/"comparison/comparison-00.png") as im:
                self.assertEqual(im.getpixel((64,36)), (20,20,20))
            Image.new("RGB", (32,48), (255,255,255)).save(root/"train/ours_5000/gt/00000.png")
            with self.assertRaisesRegex(ValueError,"inputs"):
                comparison.compare(root,root/"bad",views=1)
            self.assertFalse((root/"bad").exists())

    def test_render_sequence_has_no_training_command(self):
        commands = json.loads((EXAMPLE/"render-commands.json").read_text())
        self.assertEqual([c[2] for c in commands],
                         ["gaussian_avatar.assets", "gaussian_avatar.pilot", "render", "gaussian_avatar.preview"])
        render = commands[2]
        self.assertEqual(render[render.index("--iteration")+1], "200")
        self.assertIn("--skip_val", render)
        self.assertIn("--skip_test", render)

    def test_help_is_dependency_free_and_training_is_bounded(self):
        for name in ("pilot","assets","preview"):
            r=subprocess.run([sys.executable,"-m","gaussian_avatar."+name,"--help"],cwd=EXAMPLE,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
        commands=json.loads((EXAMPLE/"training-commands.json").read_text())
        self.assertEqual([c[2] for c in commands],["gaussian_avatar.assets","gaussian_avatar.pilot","train"])
        train=commands[-1]
        self.assertEqual(train[train.index("--iterations")+1],"200")
        self.assertIn("--bind_to_mesh",train)

if __name__ == "__main__": unittest.main()
