"""Automatic compute-device selection for pySurgery's GPU engines.

Overview:
    Every GPU routine in :mod:`pysurgery.gpu` accepts a ``device`` argument, and
    every one of them defaults to ``None``, meaning "pick the best device on this
    machine". This module owns that decision so it is made in exactly one place and
    in the same way everywhere. The order of preference is

        1. an explicit ``device`` argument (``"cuda"``, ``"cuda:1"``, ``"mps"``,
           ``"cpu"``, or a ``torch.device``);
        2. the ``PYSURGERY_DEVICE`` environment variable, if set;
        3. CUDA -- and, when several CUDA GPUs are visible, the one with the most
           free memory;
        4. Apple-Silicon MPS;
        5. Intel XPU;
        6. the CPU.

    The GPU routines are written against PyTorch, which is an *optional*
    dependency of pySurgery (``pip install pysurgery[ml]`` or ``pip install
    torch``). Nothing here imports torch until a device is actually requested, so
    importing pySurgery stays cheap on machines without it.

Key Concepts:
    - **Capability, not just presence**: MPS has no ``float64`` and no
      ``torch.linalg.eigh`` kernel. Routines that need either ask
      :func:`supports_float64` and route that part of the work to the CPU instead
      of silently degrading precision (see :mod:`pysurgery.gpu.dual_alpha`).
    - **Memory budget**: dense GPU linear algebra must refuse inputs that would
      not fit rather than thrash or crash; :func:`device_memory_budget` is the
      single estimate every engine uses.

Common Workflows:
    1. **Let pySurgery choose** -> call any GPU routine with ``device=None``.
    2. **Pin a device for a whole session** -> ``export PYSURGERY_DEVICE=cuda:1``.
    3. **Inspect the choice** -> ``describe_device()``.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, List, Optional, Union

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

#: True when PyTorch is importable in this environment.
HAS_TORCH: bool = importlib.util.find_spec("torch") is not None

#: Environment variable that pins the device for every GPU routine.
DEVICE_ENV_VAR: str = "PYSURGERY_DEVICE"

DeviceLike = Union[str, "torch.device", None]


def require_torch():
    """Import and return the ``torch`` module, or fail with an actionable message.

    Returns:
        The imported ``torch`` module.

    Raises:
        ImportError: If PyTorch is not installed.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise ImportError(
            "pySurgery's GPU engines require PyTorch. Install it with "
            "`pip install torch` (or `pip install pysurgery[ml]`)."
        ) from exc
    return torch


def _mps_available(torch) -> bool:
    """Return True when the Apple-Silicon MPS backend is built and usable."""
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available() and mps.is_built())


def _xpu_available(torch) -> bool:
    """Return True when an Intel XPU device is available."""
    xpu = getattr(torch, "xpu", None)
    try:
        return bool(xpu is not None and xpu.is_available())
    except Exception:  # pragma: no cover - vendor runtime quirks
        return False


def _best_cuda_index(torch) -> int:
    """Index of the visible CUDA device with the most free memory.

    With a single visible device this returns ``0`` without touching the driver.
    With several, it queries ``torch.cuda.mem_get_info`` on each (which creates a
    CUDA context on every device probed); set ``PYSURGERY_DEVICE`` to skip the
    probe on shared machines.

    Args:
        torch: The imported torch module.

    Returns:
        The chosen device index.
    """
    count = torch.cuda.device_count()
    if count <= 1:
        return 0
    best, best_free = 0, -1
    for i in range(count):
        try:
            free, _total = torch.cuda.mem_get_info(i)
        except Exception:  # pragma: no cover - device in a bad state
            continue
        if free > best_free:
            best, best_free = i, free
    return best


@lru_cache(maxsize=1)
def _auto_device_string() -> str:
    """The automatically chosen device, as a string (cached per process)."""
    torch = require_torch()
    if torch.cuda.is_available():
        return f"cuda:{_best_cuda_index(torch)}"
    if _mps_available(torch):
        return "mps"
    if _xpu_available(torch):
        return "xpu"
    return "cpu"


def resolve_device(device: DeviceLike = None) -> "torch.device":
    """Resolve a device request to a concrete ``torch.device``.

    What is Being Computed?:
        The device every pySurgery GPU routine runs on, following the preference
        order documented at module level: explicit argument, then the
        ``PYSURGERY_DEVICE`` environment variable, then CUDA (most free memory),
        MPS, XPU and finally the CPU.

    Args:
        device: ``None`` for automatic selection, or anything ``torch.device``
            accepts (``"cuda"``, ``"cuda:1"``, ``"mps"``, ``"cpu"``, a device).

    Returns:
        The resolved ``torch.device``.

    Raises:
        ImportError: If PyTorch is not installed.
        RuntimeError: If an explicitly requested accelerator is not available.

    Example:
        dev = resolve_device()          # e.g. device(type='cuda', index=0)
        dev = resolve_device("cpu")     # force the CPU
    """
    torch = require_torch()
    if device is None:
        env = os.environ.get(DEVICE_ENV_VAR, "").strip()
        device = env if env else _auto_device_string()
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"device {dev} was requested but CUDA is not available")
    if dev.type == "mps" and not _mps_available(torch):
        raise RuntimeError(f"device {dev} was requested but MPS is not available")
    return dev


def available_devices() -> List[str]:
    """List every device the GPU engines could run on, best first.

    Returns:
        Device strings such as ``["cuda:0", "cuda:1", "cpu"]``. Empty when PyTorch
        is not installed.
    """
    if not HAS_TORCH:
        return []
    torch = require_torch()
    out: List[str] = []
    if torch.cuda.is_available():
        out.extend(f"cuda:{i}" for i in range(torch.cuda.device_count()))
    if _mps_available(torch):
        out.append("mps")
    if _xpu_available(torch):
        out.append("xpu")
    out.append("cpu")
    return out


def supports_float64(device: DeviceLike) -> bool:
    """Whether ``device`` computes in IEEE double precision.

    MPS has no ``float64`` at all; every other backend PyTorch ships does.

    Args:
        device: A device (resolved with :func:`resolve_device`).

    Returns:
        True when float64 tensors can live on ``device``.
    """
    return resolve_device(device).type != "mps"


def device_memory_budget(device: DeviceLike, fraction: float = 0.5) -> int:
    """Bytes a single dense allocation on ``device`` may reasonably use.

    What is Being Computed?:
        ``fraction`` of the memory currently available to the device: free device
        memory on CUDA/XPU, the recommended working set minus what the driver
        already holds on MPS, and available physical RAM on the CPU. Used by the
        dense engines to refuse, with a clear message, inputs that would not fit.

    Args:
        device: The device to budget for.
        fraction: Share of the available memory to allow (0 < fraction <= 1).

    Returns:
        A byte count (at least 64 MiB).
    """
    torch = require_torch()
    dev = resolve_device(device)
    avail: Optional[int] = None
    try:
        if dev.type == "cuda":
            avail = int(torch.cuda.mem_get_info(dev)[0])
        elif dev.type == "mps":
            rec = int(torch.mps.recommended_max_memory())
            avail = max(rec - int(torch.mps.driver_allocated_memory()), 0)
        elif dev.type == "xpu":  # pragma: no cover - no XPU in CI
            props = torch.xpu.get_device_properties(dev)
            avail = int(props.total_memory) - int(torch.xpu.memory_allocated(dev))
    except Exception:  # pragma: no cover - driver quirks
        avail = None
    if avail is None:
        avail = _host_available_bytes()
    return max(int(avail * float(fraction)), 64 * 2**20)


def _host_available_bytes() -> int:
    """Available host RAM in bytes (total/4 where 'available' is not exposed)."""
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        pass
    try:  # macOS exposes only the total
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")) // 4
    except (ValueError, OSError, AttributeError):  # pragma: no cover
        return 2 * 2**30


@dataclass(frozen=True)
class DeviceInfo:
    """What pySurgery knows about a compute device.

    Attributes:
        name: The device string (``"cuda:0"``, ``"mps"``, ``"cpu"``).
        type: The device type (``"cuda"``, ``"mps"``, ``"xpu"``, ``"cpu"``).
        float64: Whether double precision is available on the device.
        memory_budget_bytes: The default dense-allocation budget
            (:func:`device_memory_budget` at ``fraction=0.5``).
        detail: Human-readable hardware name when the backend reports one.
    """

    name: str
    type: str
    float64: bool
    memory_budget_bytes: int
    detail: str = ""


def describe_device(device: DeviceLike = None) -> DeviceInfo:
    """Describe the device a GPU routine would run on.

    Args:
        device: ``None`` for the automatic choice, or an explicit device.

    Returns:
        A :class:`DeviceInfo` for the resolved device.

    Example:
        print(describe_device())  # DeviceInfo(name='mps', type='mps', float64=False, ...)
    """
    torch = require_torch()
    dev = resolve_device(device)
    detail = ""
    try:
        if dev.type == "cuda":
            detail = torch.cuda.get_device_name(dev)
        elif dev.type == "mps":
            detail = "Apple Silicon (Metal Performance Shaders)"
    except Exception:  # pragma: no cover
        detail = ""
    name = str(dev) if dev.type != "cuda" or dev.index is not None else f"cuda:{torch.cuda.current_device()}"
    return DeviceInfo(name=name, type=dev.type, float64=dev.type != "mps",
                      memory_budget_bytes=device_memory_budget(dev), detail=detail)
