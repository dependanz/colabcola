"""Run with the optional Hyperplus proof environment; does not enter core unittest discovery."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parent
spec = importlib.util.spec_from_file_location("image_fit", ROOT / "train.py")
workload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workload)


@pytest.fixture
def data(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    rng = np.random.default_rng(10)
    files = {}
    for split, count in (("train", 4), ("heldout", 2)):
        path = root / (split + ".npy")
        np.save(path, rng.integers(0, 256, (count, 3, 8, 8), dtype=np.uint8), allow_pickle=False)
        files[path.name] = {"sha256": workload.digest(path), "images": count}
    (root / "manifest.json").write_text(json.dumps({"files": files}), encoding="utf-8")
    return root


def run(data, output, steps, resume=None, extra=()):
    command = [sys.executable, str(ROOT / "train.py"), "--data", str(data),
               "--output", str(output), "--steps", str(steps), "--checkpoint-every", "3",
               "--width", "8", "--latent", "8", "--batch", "2", "--pixels", "16",
               "--threads", "1"]
    command.extend(extra)
    if resume:
        command.extend(("--resume", str(resume)))
    return subprocess.run(command, text=True, capture_output=True, timeout=90)


def checkpoint(output, step):
    return output / f"state-{step:08d}.pt.json"


def exact(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            exact(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            exact(x, y)
    else:
        assert a == b


def test_pixel_centers_and_order():
    torch.testing.assert_close(workload.coordinates(2, 2),
                               torch.tensor([[.25, .25], [.75, .25], [.25, .75], [.75, .75]]))


def test_real_hyperplus_conditioning_and_encoder_gradient():
    torch.manual_seed(4)
    net = workload.model(8, 8)
    images = torch.stack((torch.zeros(3, 8, 8), torch.ones(3, 8, 8)))
    generated = net(images)
    assert not torch.equal(generated.flat[0], generated.flat[1])
    pred = generated(workload.coordinates(8, 8).expand(2, -1, -1))
    assert pred.shape == (2, 64, 3)
    pred.square().mean().backward()
    first_encoder_layer = net.levels[-1][0]
    assert torch.isfinite(first_encoder_layer.weight.grad).all()
    assert first_encoder_layer.weight.grad.norm() > 0


def test_fresh_process_resume_matches_uninterrupted(data, tmp_path):
    full, first, resumed = [tmp_path / name for name in ("full", "first", "resumed")]
    for arguments in ((data, full, 6), (data, first, 3),
                      (data, resumed, 6, checkpoint(first, 3))):
        result = run(*arguments)
        assert result.returncode == 0, result.stderr
    expected = workload.load_checkpoint(checkpoint(full, 6))
    actual = workload.load_checkpoint(checkpoint(resumed, 6))
    exact(expected, actual)
    assert json.loads((resumed / "metrics.json").read_text())["restored_step"] == 3
    assert (resumed / "heldout.png").exists()


def test_corruption_and_changed_dataset_rejected(data, tmp_path):
    first = tmp_path / "first"
    result = run(data, first, 3)
    assert result.returncode == 0, result.stderr
    record = checkpoint(first, 3)
    state_file = first / "state-00000003.pt"
    original = state_file.read_bytes()
    state_file.write_bytes(original[:-10])
    with pytest.raises(ValueError, match="integrity"):
        workload.load_checkpoint(record)
    state_file.write_bytes(original)
    manifest = data / "manifest.json"
    manifest.write_text(manifest.read_text() + "\n", encoding="utf-8")
    result = run(data, tmp_path / "wrong-data", 6, record)
    assert result.returncode != 0
    assert "identity/config/data/environment mismatch" in result.stderr


def test_dataset_content_hash_checked(data):
    path = data / "train.npy"
    path.write_bytes(path.read_bytes() + b"bad")
    with pytest.raises(ValueError, match="dataset hash mismatch"):
        workload.load_data(data)

@pytest.mark.parametrize("shape", [(2, 2), (4, 4), (8, 8), (7, 9)])
def test_spatial_mean_pool_preserves_forward_and_gradient(shape):
    torch.manual_seed(3)
    original = torch.randn(2, 3, *shape, dtype=torch.float64, requires_grad=True)
    replacement = original.detach().clone().requires_grad_(True)
    expected = torch.nn.AdaptiveAvgPool2d((4, 4))(original)
    actual = workload.SpatialMeanPool4()(replacement)
    weights = torch.randn_like(expected)
    (expected * weights).sum().backward()
    (actual * weights).sum().backward()
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(replacement.grad, original.grad, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available in local CPU verification environment")
def test_cuda_hyperplus_backward_with_strict_determinism():
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    net = workload.model(8, 8).cuda()
    images = torch.rand(2, 3, 32, 32, device="cuda")
    xy = workload.coordinates(32, 32).cuda().expand(2, -1, -1)
    net(images, xy).square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


@pytest.mark.parametrize("beta,omega", [(0.5, 1.0), (2.0, 3.0), (14.0, 30.0)])
def test_hosc_formula_and_analytic_derivative(beta, omega):
    x = torch.linspace(-2, 2, 101, dtype=torch.float64, requires_grad=True)
    y = workload.HOSC(beta, omega)(x)
    expected = torch.tanh(beta * torch.sin(omega * x))
    torch.testing.assert_close(y, expected)
    actual = torch.autograd.grad(y.sum(), x)[0]
    derivative = beta * omega * torch.cos(omega * x) * (1 - expected.square())
    torch.testing.assert_close(actual, derivative, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("beta,omega", [(0, 1), (-1, 1), (1, 0), (float("nan"), 1), (1, float("inf"))])
def test_hosc_rejects_invalid_parameters(beta, omega):
    with pytest.raises(ValueError, match="finite and positive"):
        workload.HOSC(beta, omega)


def test_hosc_hyperplus_gradient_and_baseline_initialization():
    torch.manual_seed(4)
    baseline = workload.model(8, 8)
    torch.manual_seed(4)
    net = workload.model(8, 8, activation="hosc", hosc_beta=2, hosc_omega=1)
    assert net.slot_counts == baseline.slot_counts
    for actual, expected in zip(net.parameters(), baseline.parameters()):
        assert torch.equal(actual, expected)
    images = torch.stack((torch.zeros(3, 8, 8), torch.ones(3, 8, 8)))
    pred = net(images, workload.coordinates(8, 8).expand(2, -1, -1))
    pred.square().mean().backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(float(g.square().sum()) for g in grads) > 0


def test_hosc_fresh_process_resume_and_changed_activation_rejected(data, tmp_path):
    extra = ("--activation", "hosc", "--hosc-beta", "2", "--hosc-omega", "1")
    full, first, resumed = [tmp_path / name for name in ("full", "first", "resumed")]
    for output, steps, resume in ((full, 6, None), (first, 3, None), (resumed, 6, checkpoint(first, 3))):
        result = run(data, output, steps, resume, extra)
        assert result.returncode == 0, result.stderr
    exact(workload.load_checkpoint(checkpoint(full, 6)), workload.load_checkpoint(checkpoint(resumed, 6)))
    result = run(data, tmp_path / "wrong-activation", 6, checkpoint(first, 3))
    assert result.returncode != 0 and "identity/config/data/environment mismatch" in result.stderr
    result = run(data, tmp_path / "wrong-beta", 6, checkpoint(first, 3),
                 ("--activation", "hosc", "--hosc-beta", "3"))
    assert result.returncode != 0 and "identity/config/data/environment mismatch" in result.stderr
