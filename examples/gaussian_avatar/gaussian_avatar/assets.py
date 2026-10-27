"""Validate local licensed assets and make a small tracked-data subset; no downloads."""
import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath
import shutil


def contained(root, name):
    path = Path(name)
    if path.is_absolute() or PureWindowsPath(name).is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"unsafe dataset path: {name}")
    root = root.resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError(f"missing or escaping dataset file: {name}")
    return target


def inspect(source, flame):
    if (source / "sparse").exists():
        raise ValueError("sparse selects the Colmap loader instead of DynamicNerf")
    canonical = contained(source, "canonical_flame_param.npz")
    if canonical.stat().st_size == 0:
        raise ValueError("empty canonical FLAME parameters")
    records = {}
    for name in ("flame2023.pkl", "FLAME_masks.pkl"):
        path = contained(flame, name)
        if path.stat().st_size == 0:
            raise ValueError(f"empty FLAME asset: {name}")
        records[name] = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    counts = {}
    for split in ("train", "val", "test"):
        data = json.loads(contained(source, f"transforms_{split}.json").read_text(encoding="utf-8"))
        frames = data["frames"]
        if not isinstance(frames, list) or (split == "train" and not frames):
            raise ValueError("training frames must be nonempty")
        for frame in frames:
            image = frame["file_path"]
            contained(source, image if ".png" in image else image + ".png")
            contained(source, frame["flame_param_path"])
            if type(frame["timestep_index"]) is not int or frame["timestep_index"] < 0:
                raise ValueError("invalid timestep_index")
            matrix = frame["transform_matrix"]
            if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
                raise ValueError("transform_matrix must be 4x4")
            if not ("camera_angle_x" in frame or "camera_angle_x" in data):
                raise ValueError("camera_angle_x is required")
        counts[split] = len(frames)
    return {"state": "files_validated", "frames": counts, "flame": records,
            "boundary": "Structural check only; no pickle loading, FLAME semantic validation or training."}


def subset(source, output, frame_count):
    if not 2 <= frame_count <= 16:
        raise ValueError("pilot subset must have 2 to 16 frames")
    if output.exists():
        raise ValueError("subset destination already exists")
    data = json.loads(contained(source, "transforms_train.json").read_text(encoding="utf-8"))
    frames = data["frames"][:frame_count]
    if len(frames) < 2:
        raise ValueError("at least two training views required")
    # Reject degenerate camera normalization rather than silently train a zero-radius scene.
    centers = {tuple(f["transform_matrix"][i][3] for i in range(3)) for f in frames}
    if len(centers) < 2:
        raise ValueError("select training views with at least two different camera centers")
    files = {"canonical_flame_param.npz": contained(source, "canonical_flame_param.npz")}
    if files["canonical_flame_param.npz"].stat().st_size == 0 or (source / "sparse").exists():
        raise ValueError("dataset must select DynamicNerf with nonempty canonical parameters")
    times = {v: i for i, v in enumerate(sorted({f["timestep_index"] for f in frames}))}
    selected = []
    for frame in frames:
        frame = dict(frame)
        for field in ("file_path", "flame_param_path"):
            name = frame[field]
            if field == "file_path" and ".png" not in name:
                name += ".png"
            files[name] = contained(source, name)
        frame["timestep_index"] = times[frame["timestep_index"]]
        selected.append(frame)
    output.mkdir(parents=True)
    for name, file in files.items():
        dest = output / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, dest)
    for split in ("train", "val", "test"):
        value = dict(data, frames=selected if split == "train" else [])
        (output / f"transforms_{split}.json").write_text(json.dumps(value, indent=2), encoding="utf-8")
    return {"train_frames": len(frames), "timesteps": len(times), "copied_files": len(files), "validation_frames": 0, "test_frames": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--flame", type=Path, required=True)
    parser.add_argument("--subset-output", type=Path)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--loader-check", type=Path, help="CPU check against this pinned upstream source root")
    args = parser.parse_args()
    result = inspect(args.source, args.flame)
    if args.loader_check:
        from .loader_check import check
        result["loader"] = check(args.source, args.loader_check)
    if args.subset_output:
        result["subset"] = subset(args.source, args.subset_output, args.frames)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
