"""Tests for automatic GPU device selection (pysurgery.gpu.device).

Overview:
    The GPU engines must run on the best available device without being told, obey
    an explicit request or the ``PYSURGERY_DEVICE`` environment variable, and refuse
    an accelerator that is not there. These tests pass on a CPU-only machine (the
    automatic choice is then the CPU) and on CUDA / MPS machines alike.
"""
import pytest

torch = pytest.importorskip("torch")

from pysurgery.gpu import device as D  # noqa: E402


def test_automatic_choice_prefers_accelerators():
    dev = D.resolve_device(None)
    if torch.cuda.is_available():
        assert dev.type == "cuda"
    elif D._mps_available(torch):
        assert dev.type == "mps"
    else:
        assert dev.type in ("xpu", "cpu")
    assert D.available_devices()[-1] == "cpu"
    assert str(dev) in D.available_devices() or dev.type == "cuda"


def test_explicit_and_environment_overrides(monkeypatch):
    assert D.resolve_device("cpu").type == "cpu"
    monkeypatch.setenv(D.DEVICE_ENV_VAR, "cpu")
    assert D.resolve_device(None).type == "cpu"
    monkeypatch.delenv(D.DEVICE_ENV_VAR)
    assert D.resolve_device(torch.device("cpu")).type == "cpu"


def test_unavailable_accelerator_is_refused():
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            D.resolve_device("cuda")
    if not D._mps_available(torch):
        with pytest.raises(RuntimeError):
            D.resolve_device("mps")


def test_capabilities_and_budget():
    assert D.supports_float64("cpu")
    budget = D.device_memory_budget("cpu", fraction=0.25)
    assert budget >= 64 * 2**20
    info = D.describe_device("cpu")
    assert info.type == "cpu" and info.float64 and info.memory_budget_bytes > 0


def test_importing_pysurgery_gpu_does_not_need_torch_yet():
    import pysurgery.gpu as G
    assert "dual_alpha_complex" in G.__all__ and "gpu_homology" in G.__all__
    with pytest.raises(AttributeError):
        G.not_a_symbol  # noqa: B018
