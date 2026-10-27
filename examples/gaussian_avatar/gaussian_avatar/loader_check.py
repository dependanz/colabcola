"""CPU preflight of pinned upstream loader bodies, without importing CUDA extensions.

Only the source-verified dispatch, camera/mesh readers and their lightweight
geometry helpers are executed. This does not claim a full CUDA Scene construction.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple, Optional

HASHES = {
    "scene/__init__.py": "c212c657878882711c06923c389d4445be3672ae5bc851ddb18a88dbe5e14f20",
    "scene/dataset_readers.py": "75540fa4bc079ad222de976592b4a2d6c9dd9af550e02d7bf6871c56a44b68da",
    "utils/graphics_utils.py": "aef5b24e6a64f9bebc03506ada43806d207abafbbb4f380fcefb4e77f84efa10",
}


def check(source, upstream):
    import math
    import numpy as np
    from PIL import Image
    from tqdm import tqdm
    from .assets import contained
    source, upstream = Path(source).resolve(), Path(upstream).resolve()
    trees = {}
    for name, expected in HASHES.items():
        body = (upstream/name).read_bytes()
        if hashlib.sha256(body).hexdigest() != expected:
            raise ValueError("unsupported upstream loader source: " + name)
        trees[name] = ast.parse(body, filename=name)
    # Validate all archives before the upstream reader's allow_pickle=True call.
    paths = {contained(source, "canonical_flame_param.npz")}
    required = {"shape", "expr", "rotation", "neck_pose", "jaw_pose", "eyes_pose", "translation", "static_offset"}
    for split in ("train", "val", "test"):
        data = json.loads(contained(source, f"transforms_{split}.json").read_text(encoding="utf-8"))
        for frame in data["frames"]:
            path = contained(source, frame["flame_param_path"])
            paths.add(path)
            with np.load(path, allow_pickle=False) as params:
                if not required <= set(params.files):
                    raise ValueError("missing pose parameters: " + str(path))
    for path in paths:
        with np.load(path, allow_pickle=False) as params:
            if not params.files:
                raise ValueError("empty parameter archive")
            for key in params.files:
                value = params[key]
                if value.dtype.kind not in "biufc" or not np.isfinite(value).all():
                    raise ValueError("non-numeric or non-finite pose array: " + key)
    namespace = dict(np=np, math=math, os=os, json=json, Path=Path, Image=Image,
                     NamedTuple=NamedTuple, Optional=Optional, tqdm=tqdm)
    wanted = {"BasicPointCloud", "getWorld2View2", "focal2fov", "fov2focal",
              "CameraInfo", "SceneInfo", "getNerfppNorm", "readCamerasFromTransforms",
              "readMeshesFromTransforms", "readDynamicNerfInfo"}
    for name in ("utils/graphics_utils.py", "scene/dataset_readers.py"):
        selected = [n for n in trees[name].body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted]
        exec(compile(ast.Module(body=selected, type_ignores=[]), name, "exec"), namespace)
    scene = next(n for n in trees["scene/__init__.py"].body if isinstance(n, ast.ClassDef) and n.name == "Scene")
    init = next(n for n in scene.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    dispatch = next(n for n in init.body if isinstance(n, ast.If) and "sceneLoadTypeCallbacks" in ast.unparse(n))
    selected = []
    def dynamic(*args, **kwargs):
        selected.append("DynamicNerf")
        return namespace["readDynamicNerfInfo"](*args, **kwargs)
    def wrong(*args, **kwargs):
        raise ValueError("upstream selected a non-avatar loader")
    namespace.update(args=SimpleNamespace(source_path=str(source), images="images", eval=True,
                                         white_background=True, target_path=""),
                     sceneLoadTypeCallbacks={"DynamicNerf": dynamic, "Colmap": wrong, "Blender": wrong})
    exec(compile(ast.Module(body=[dispatch], type_ignores=[]), "scene/__init__.py", "exec"), namespace)
    info = namespace["scene_info"]
    meshes = {**info.train_meshes, **info.test_meshes}
    if selected != ["DynamicNerf"] or not info.train_cameras or not meshes or 0 not in meshes:
        raise ValueError("nonempty avatar cameras and zero-based pose meshes required")
    if any(c.timestep not in meshes for c in info.train_cameras):
        raise ValueError("camera timestep has no pose mesh")
    radius = float(info.nerf_normalization["radius"])
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("invalid camera normalization radius")
    return {"loader": selected[0], "training_cameras": len(info.train_cameras),
            "pose_meshes": len(meshes), "timesteps": sorted(meshes), "radius": radius,
            "numeric_parameter_archives": len(paths), "source_hashes": HASHES,
            "boundary": "Pinned upstream CPU loader bodies; full CUDA Scene/training not executed."}
