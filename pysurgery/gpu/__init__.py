"""GPU engines for pySurgery: dual-alpha complexes, their filtrations, and exact homology.

Overview:
    Three pieces, all pure Python on top of PyTorch (an optional dependency), all
    running on the automatically selected device (CUDA, then Apple MPS, then Intel
    XPU, then the CPU -- see :mod:`pysurgery.gpu.device`):

    1. :mod:`pysurgery.gpu.dual_alpha` -- the alpha complex by the dual active-set
       quadratic program of Carlsson & Carlsson, with no Delaunay triangulation, so
       the ambient dimension is not a direct cost. Exact: gray-zone verdicts are
       decided in rational arithmetic.
    2. The alpha **filtration** from one build: every simplex's alpha value comes
       with the complex (:meth:`DualAlphaResult.filtration_values`,
       :meth:`DualAlphaResult.subcomplex`), and
       :class:`~pysurgery.topology.filtration_tools.DualAlphaFiltrationReport` runs
       the full filtration report on it.
    3. :mod:`pysurgery.gpu.homology` -- exact homology on the GPU: ranks over
       ``F_p`` by modular elimination, and integral homology (Betti numbers *and*
       torsion) by unimodular elimination with an exact Smith-normal-form finish on
       the CPU. Also reachable as ``complex.homology(backend="gpu")``.

    Importing this package does not import torch; the first call that needs it does.

Common Workflows:
    1. ``from pysurgery.gpu import dual_alpha_complex``
       ``res = dual_alpha_complex(points, radius=0.5)``
    2. ``res.complex.homology(backend="gpu")``
    3. ``from pysurgery.gpu import describe_device; describe_device()``
"""

from .device import (  # noqa: F401  (light: no torch import)
    DEVICE_ENV_VAR,
    HAS_TORCH,
    DeviceInfo,
    available_devices,
    describe_device,
    device_memory_budget,
    resolve_device,
    supports_float64,
)

_DUAL_ALPHA = {
    "dual_alpha_complex",
    "DualAlphaResult",
    "connectivity_radius",
    "exact_decide",
    "kkt_report",
    "primal_is_feasible",
    "AlphaUndecided",
    "AlphaBudgetExceeded",
}
_HOMOLOGY = {
    "rank_mod_p",
    "smith_invariant_factors",
    "invariant_factors_from_diagonal",
    "peel_unit_singletons",
    "chain_complex_homology",
    "gpu_homology",
    "gpu_betti_numbers",
    "torsion_prime_screen",
    "DenseBudgetExceeded",
}


def __getattr__(name):
    if name in _DUAL_ALPHA:
        from . import dual_alpha
        return getattr(dual_alpha, name)
    if name in _HOMOLOGY:
        from . import homology
        return getattr(homology, name)
    raise AttributeError(f"module 'pysurgery.gpu' has no attribute {name!r}")


__all__ = [
    "DEVICE_ENV_VAR",
    "HAS_TORCH",
    "DeviceInfo",
    "available_devices",
    "describe_device",
    "device_memory_budget",
    "resolve_device",
    "supports_float64",
    *sorted(_DUAL_ALPHA),
    *sorted(_HOMOLOGY),
]
