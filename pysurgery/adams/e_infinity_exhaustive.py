"""Exhaustive E_infinity solver for the Adams spectral sequence.

When the user does not know (or does not want to compute) the actual
differentials d_r at ambiguous bidegrees, this module enumerates every
mathematically-possible E_infinity grid consistent with the ambiguous
flags and reports tight (min, max) bounds per bidegree.

Why this is non-trivial:
    Each ambiguous flag at (r, src, tgt) has a rank k in {0, 1, ..., min(S, T)};
    a different rank assignment kills different cells. The number of joint
    rank assignments grows as product_f (min(S_f, T_f) + 1), exponential in
    the number of flags.

Strategy:
    - 'analytical' (default, fast, exact for bounds): per-cell formulas.
      The MAX of dim E_infinity^{s,t} over all rank assignments is just
      E_2^{s,t} (set every involving rank to 0 — jointly achievable).
      The MIN of dim E_infinity^{s,t} is max(0, E_2^{s,t} - total possible
      rank-in - total possible rank-out), also jointly achievable.

    - 'exhaustive' (slow, gives joint statistics): explicit enumeration.
      Uses Julia for the inner loop when available; Python fallback uses
      Numba JIT; pure Python as last resort.

    - 'sample' (statistical, scales): uniform sampling over rank assignments
      with running min/max tracking. Useful when the analytical bound is
      acceptable but a histogram is desired.

References:
    Adams, J. F. (1958). On the structure and applications of the
        Steenrod algebra. Comment. Math. Helv. 32, 180-214.
    Bruner, R. R. (1993). Ext in the nineties. Contemp. Math. 146, AMS.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Literal, Tuple

from pydantic import BaseModel, ConfigDict, Field

from pysurgery.adams.spectral_sequence import AdamsDifferentialFlag, AdamsE2Page
from pysurgery.core.foundations import CONTRACT_VERSION


# ── Public contract ──────────────────────────────────────────────────────────


class ExhaustiveEInfinityResult(BaseModel):
    """Per-bidegree (min, max) dim E_infinity over all consistent rank choices.

    Attributes:
        e_infinity_min: (s, t) -> minimum dim_{F_p} E_infinity^{s,t}.
        e_infinity_max: (s, t) -> maximum dim_{F_p} E_infinity^{s,t}.
        e_infinity_mean: (s, t) -> average over enumerated combinations
            (only populated by method='exhaustive' or 'sample').
        stem_min: stem n -> sum_{s>0} e_infinity_min^{s, n+s}.
        stem_max: stem n -> sum_{s>0} e_infinity_max^{s, n+s}.
        method: which solver produced the result.
        combinations_explored: number of rank assignments visited.
        combinations_total: full product of (min(S, T) + 1) over flags.
        wall_seconds: elapsed time.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    space_label: str = ""
    prime: Literal[2, 3, 5]

    e_infinity_min: Dict[Tuple[int, int], int]
    e_infinity_max: Dict[Tuple[int, int], int]
    e_infinity_mean: Dict[Tuple[int, int], float] = Field(default_factory=dict)

    stem_min: Dict[int, int]
    stem_max: Dict[int, int]

    method: Literal["analytical", "exhaustive", "sample"]
    combinations_explored: int
    combinations_total: int
    wall_seconds: float

    exact: bool = True
    theorem_tag: str = "adams_e_infinity_exhaustive_v1"
    contract_version: str = CONTRACT_VERSION


# ── Analytical (fast, exact for bounds) ──────────────────────────────────────


def _per_cell_rank_in_out(
    flags: List[AdamsDifferentialFlag],
) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], int]]:
    """Sum of max-rank of all flags incident at each (s, t) as source / target."""
    rank_out: Dict[Tuple[int, int], int] = {}
    rank_in: Dict[Tuple[int, int], int] = {}
    for fl in flags:
        m = min(fl.source_dim, fl.target_dim)
        s_src, t_src = fl.source
        s_tgt, t_tgt = fl.target
        rank_out[(s_src, t_src)] = rank_out.get((s_src, t_src), 0) + m
        rank_in[(s_tgt, t_tgt)] = rank_in.get((s_tgt, t_tgt), 0) + m
    return rank_out, rank_in


def _analytical_bounds(
    page: AdamsE2Page,
) -> ExhaustiveEInfinityResult:
    """Per-cell (min, max) dim E_infinity from ambiguous flags.

    The MAX is achieved by setting every rank to 0 (all differentials zero);
    the MIN by setting every rank to its maximum. Both endpoints are jointly
    consistent in the rank-assignment lattice, so these bounds are tight as
    *bounds*, even though the joint enumeration would be more informative.
    """
    import time as _time

    t0 = _time.time()
    rank_out, rank_in = _per_cell_rank_in_out(page.ambiguous_differentials)

    e_min: Dict[Tuple[int, int], int] = {}
    e_max: Dict[Tuple[int, int], int] = {}
    for (s, t), d in page.e2_grid.items():
        if d <= 0:
            continue
        max_kill = rank_out.get((s, t), 0) + rank_in.get((s, t), 0)
        e_max[(s, t)] = d
        e_min[(s, t)] = max(0, d - max_kill)

    stem_min: Dict[int, int] = {}
    stem_max: Dict[int, int] = {}
    for (s, t), d in e_min.items():
        if s > 0:
            stem_min[t - s] = stem_min.get(t - s, 0) + d
    for (s, t), d in e_max.items():
        if s > 0:
            stem_max[t - s] = stem_max.get(t - s, 0) + d

    combos_total = 1
    for fl in page.ambiguous_differentials:
        combos_total *= (min(fl.source_dim, fl.target_dim) + 1)

    return ExhaustiveEInfinityResult(
        space_label=page.space_label,
        prime=page.prime,
        e_infinity_min=e_min,
        e_infinity_max=e_max,
        e_infinity_mean={},
        stem_min=stem_min,
        stem_max=stem_max,
        method="analytical",
        combinations_explored=2,  # the two extreme assignments
        combinations_total=combos_total,
        wall_seconds=_time.time() - t0,
    )


# ── Exhaustive: Julia first, Python fallback ─────────────────────────────────


def _exhaustive_python(
    page: AdamsE2Page,
    hard_cap: int,
) -> ExhaustiveEInfinityResult:
    """Pure-Python exhaustive enumeration. Returns analytical bounds with
    e_infinity_mean populated by enumeration.
    """
    import time as _time

    t0 = _time.time()
    flags = page.ambiguous_differentials
    # Per-flag range
    ranges = [min(fl.source_dim, fl.target_dim) + 1 for fl in flags]
    combos_total = 1
    for r in ranges:
        combos_total *= r

    if combos_total > hard_cap:
        warnings.warn(
            f"exhaustive enumeration would visit {combos_total} > hard_cap "
            f"{hard_cap}; returning analytical bounds instead.",
            UserWarning,
            stacklevel=2,
        )
        return _analytical_bounds(page)

    # Try Numba JIT if available; fall back to pure-Python loop.
    try:
        return _exhaustive_python_numba(page, ranges)
    except Exception:
        pass

    # Pure-Python loop with running min/max/sum.
    n_flags = len(flags)
    flag_src = [fl.source for fl in flags]
    flag_tgt = [fl.target for fl in flags]

    e_min: Dict[Tuple[int, int], int] = {}
    e_max: Dict[Tuple[int, int], int] = {}
    e_sum: Dict[Tuple[int, int], int] = {}

    # Initialize from E_2 (the all-ranks-zero case).
    for (s, t), d in page.e2_grid.items():
        if d > 0:
            e_min[(s, t)] = d
            e_max[(s, t)] = d
            e_sum[(s, t)] = 0  # accumulated over combinations

    # Mixed-radix counter.
    indices = [0] * n_flags
    e2 = page.e2_grid

    cells: List[Tuple[int, int]] = sorted(
        (k for k, v in e2.items() if v > 0)
    )
    cell_idx = {c: i for i, c in enumerate(cells)}

    explored = 0
    done = False
    while not done:
        # Build candidate grid for this assignment.
        cand = [e2[c] for c in cells]
        for i in range(n_flags):
            k = indices[i]
            if k > 0:
                src = flag_src[i]
                tgt = flag_tgt[i]
                if src in cell_idx:
                    cand[cell_idx[src]] -= k
                if tgt in cell_idx:
                    cand[cell_idx[tgt]] -= k
        # Update running min/max/sum.
        for i, c in enumerate(cells):
            v = max(0, cand[i])
            if v < e_min[c]:
                e_min[c] = v
            if v > e_max[c]:
                e_max[c] = v
            e_sum[c] += v
        explored += 1

        # Increment mixed-radix counter.
        for i in range(n_flags):
            indices[i] += 1
            if indices[i] < ranges[i]:
                break
            indices[i] = 0
        else:
            done = True

    e_mean: Dict[Tuple[int, int], float] = {
        c: e_sum[c] / explored for c in cells
    } if explored > 0 else {}

    stem_min: Dict[int, int] = {}
    stem_max: Dict[int, int] = {}
    for (s, t), d in e_min.items():
        if s > 0:
            stem_min[t - s] = stem_min.get(t - s, 0) + d
    for (s, t), d in e_max.items():
        if s > 0:
            stem_max[t - s] = stem_max.get(t - s, 0) + d

    return ExhaustiveEInfinityResult(
        space_label=page.space_label,
        prime=page.prime,
        e_infinity_min=e_min,
        e_infinity_max=e_max,
        e_infinity_mean=e_mean,
        stem_min=stem_min,
        stem_max=stem_max,
        method="exhaustive",
        combinations_explored=explored,
        combinations_total=combos_total,
        wall_seconds=_time.time() - t0,
    )


def _exhaustive_python_numba(
    page: AdamsE2Page,
    ranges: List[int],
) -> ExhaustiveEInfinityResult:
    """Numba-JIT accelerated exhaustive enumeration."""
    import time as _time

    import numpy as np
    from numba import njit  # type: ignore

    t0 = _time.time()
    flags = page.ambiguous_differentials
    n_flags = len(flags)

    cells: List[Tuple[int, int]] = sorted(
        (k for k, v in page.e2_grid.items() if v > 0)
    )
    n_cells = len(cells)
    cell_idx = {c: i for i, c in enumerate(cells)}

    src_idx = np.zeros(n_flags, dtype=np.int64)
    tgt_idx = np.zeros(n_flags, dtype=np.int64)
    for i, fl in enumerate(flags):
        src_idx[i] = cell_idx.get(fl.source, -1)
        tgt_idx[i] = cell_idx.get(fl.target, -1)

    e2_vec = np.array([page.e2_grid[c] for c in cells], dtype=np.int64)
    radices = np.array(ranges, dtype=np.int64)

    combos_total = int(np.prod(radices.astype(np.float64)))

    @njit(cache=True, fastmath=False)
    def _kernel(
        e2_vec: np.ndarray,
        radices: np.ndarray,
        src_idx: np.ndarray,
        tgt_idx: np.ndarray,
        n_cells: int,
        n_flags: int,
        combos_total: int,
    ):
        e_min = e2_vec.copy()
        e_max = e2_vec.copy()
        e_sum = np.zeros(n_cells, dtype=np.float64)
        cand = np.empty(n_cells, dtype=np.int64)
        indices = np.zeros(n_flags, dtype=np.int64)
        explored = 0
        while True:
            # cand = e2 - rank contributions
            for j in range(n_cells):
                cand[j] = e2_vec[j]
            for i in range(n_flags):
                k = indices[i]
                if k > 0:
                    si = src_idx[i]
                    ti = tgt_idx[i]
                    if si >= 0:
                        cand[si] -= k
                    if ti >= 0:
                        cand[ti] -= k
            for j in range(n_cells):
                v = cand[j]
                if v < 0:
                    v = 0
                if v < e_min[j]:
                    e_min[j] = v
                if v > e_max[j]:
                    e_max[j] = v
                e_sum[j] += v
            explored += 1

            # Increment mixed-radix counter
            done = True
            for i in range(n_flags):
                indices[i] += 1
                if indices[i] < radices[i]:
                    done = False
                    break
                indices[i] = 0
            if done:
                break
        return e_min, e_max, e_sum, explored

    e_min_arr, e_max_arr, e_sum_arr, explored = _kernel(
        e2_vec, radices, src_idx, tgt_idx, n_cells, n_flags, combos_total,
    )

    e_min: Dict[Tuple[int, int], int] = {
        c: int(e_min_arr[i]) for i, c in enumerate(cells)
    }
    e_max: Dict[Tuple[int, int], int] = {
        c: int(e_max_arr[i]) for i, c in enumerate(cells)
    }
    e_mean: Dict[Tuple[int, int], float] = {
        c: float(e_sum_arr[i] / explored) for i, c in enumerate(cells)
    }

    stem_min: Dict[int, int] = {}
    stem_max: Dict[int, int] = {}
    for (s, t), d in e_min.items():
        if s > 0:
            stem_min[t - s] = stem_min.get(t - s, 0) + d
    for (s, t), d in e_max.items():
        if s > 0:
            stem_max[t - s] = stem_max.get(t - s, 0) + d

    return ExhaustiveEInfinityResult(
        space_label=page.space_label,
        prime=page.prime,
        e_infinity_min=e_min,
        e_infinity_max=e_max,
        e_infinity_mean=e_mean,
        stem_min=stem_min,
        stem_max=stem_max,
        method="exhaustive",
        combinations_explored=int(explored),
        combinations_total=int(combos_total),
        wall_seconds=_time.time() - t0,
    )


def _exhaustive_julia(
    page: AdamsE2Page,
) -> ExhaustiveEInfinityResult:
    """Julia-accelerated exhaustive enumeration.

    Delegates the inner enumeration loop to SurgeryBackend.exhaustive_e_inf
    via julia_engine; falls back to caller for handling Julia errors.
    """
    import time as _time
    import numpy as np

    from ..bridge.julia_bridge import julia_engine

    if not julia_engine.available:
        raise RuntimeError("Julia engine not available")
    if not hasattr(julia_engine.backend, "exhaustive_e_inf"):
        raise RuntimeError("Julia backend lacks exhaustive_e_inf kernel")

    t0 = _time.time()
    flags = page.ambiguous_differentials
    n_flags = len(flags)

    cells: List[Tuple[int, int]] = sorted(
        (k for k, v in page.e2_grid.items() if v > 0)
    )
    cell_idx = {c: i for i, c in enumerate(cells)}

    src_idx = np.zeros(n_flags, dtype=np.int64)
    tgt_idx = np.zeros(n_flags, dtype=np.int64)
    radices = np.zeros(n_flags, dtype=np.int64)
    for i, fl in enumerate(flags):
        # 1-based for Julia
        src_idx[i] = cell_idx.get(fl.source, -1) + 1
        tgt_idx[i] = cell_idx.get(fl.target, -1) + 1
        radices[i] = min(fl.source_dim, fl.target_dim) + 1

    e2_vec = np.array([page.e2_grid[c] for c in cells], dtype=np.int64)

    e_min_arr, e_max_arr, e_sum_arr, explored = julia_engine.backend.exhaustive_e_inf(
        e2_vec, radices, src_idx, tgt_idx,
    )

    e_min: Dict[Tuple[int, int], int] = {
        c: int(e_min_arr[i]) for i, c in enumerate(cells)
    }
    e_max: Dict[Tuple[int, int], int] = {
        c: int(e_max_arr[i]) for i, c in enumerate(cells)
    }
    e_mean: Dict[Tuple[int, int], float] = {
        c: float(e_sum_arr[i] / explored) for i, c in enumerate(cells)
    }

    stem_min: Dict[int, int] = {}
    stem_max: Dict[int, int] = {}
    for (s, t), d in e_min.items():
        if s > 0:
            stem_min[t - s] = stem_min.get(t - s, 0) + d
    for (s, t), d in e_max.items():
        if s > 0:
            stem_max[t - s] = stem_max.get(t - s, 0) + d

    combos_total = 1
    for r in radices:
        combos_total *= int(r)

    return ExhaustiveEInfinityResult(
        space_label=page.space_label,
        prime=page.prime,
        e_infinity_min=e_min,
        e_infinity_max=e_max,
        e_infinity_mean=e_mean,
        stem_min=stem_min,
        stem_max=stem_max,
        method="exhaustive",
        combinations_explored=int(explored),
        combinations_total=int(combos_total),
        wall_seconds=_time.time() - t0,
    )


# ── Public façade ────────────────────────────────────────────────────────────


def exhaustive_e_infinity_bounds(
    page: AdamsE2Page,
    *,
    method: Literal["auto", "analytical", "exhaustive"] = "auto",
    backend: Literal["auto", "python", "julia"] = "auto",
    hard_cap: int = 1 << 20,
) -> ExhaustiveEInfinityResult:
    """Per-bidegree (min, max) bounds for E_infinity over all rank assignments.

    What is Being Computed?:
        For each ambiguous d_r flag at (r, src, tgt) with src_dim S and
        tgt_dim T, the rank k of the differential lies in {0, ..., min(S,T)}.
        This routine enumerates over the set of all such joint rank
        assignments (or computes analytical bounds when 'analytical' is
        requested) and returns the (min, max) dim E_infinity per bidegree.

    Args:
        page: An AdamsE2Page produced by adams_e2_page.
        method:
            - 'auto' (default): picks 'exhaustive' if total combinations
              <= hard_cap, else 'analytical'.
            - 'analytical': fast per-cell (min, max) formula; tight as a
              bound, no joint statistics. O(|cells| + |flags|) time.
            - 'exhaustive': enumerates every assignment; populates e_mean
              and exact joint statistics. Time = O(combinations_total · n_cells).
        backend:
            - 'auto' (default): Julia first; Numba; pure Python.
            - 'julia': hard fail if Julia unavailable.
            - 'python': skip Julia; use Numba if importable, else pure Python.
        hard_cap: maximum number of combinations to enumerate before falling
            back to analytical bounds. Default 2**20 ~ 1M.

    Returns:
        ExhaustiveEInfinityResult.

    Notes:
        - Per-cell max is achieved by all-zero rank assignment (all d_r = 0).
        - Per-cell min is achieved by all-max rank assignment.
        - Both extremes are jointly achievable, so 'analytical' bounds are
          tight as bounds even though they don't capture joint distribution.
        - The 'exhaustive' branch only adds joint info (e_mean and the
          knowledge that no intermediate assignment yields tighter bounds).
    """
    flags = page.ambiguous_differentials
    if not flags:
        # No flags = E_2 is already E_infinity.
        return _analytical_bounds(page)

    combos_total = 1
    for fl in flags:
        combos_total *= (min(fl.source_dim, fl.target_dim) + 1)

    if method == "auto":
        method = "exhaustive" if combos_total <= hard_cap else "analytical"

    if method == "analytical":
        return _analytical_bounds(page)

    # method == 'exhaustive'
    if combos_total > hard_cap:
        warnings.warn(
            f"exhaustive enumeration would visit {combos_total} > hard_cap "
            f"{hard_cap}; returning analytical bounds.",
            UserWarning,
            stacklevel=2,
        )
        return _analytical_bounds(page)

    use_julia = (backend == "julia") or (
        backend == "auto" and _try_julia_available()
    )
    if use_julia:
        try:
            return _exhaustive_julia(page)
        except Exception as e:
            if backend == "julia":
                raise
            warnings.warn(
                f"Julia exhaustive_e_inf failed: {e!r}; falling back.",
                UserWarning,
                stacklevel=2,
            )

    return _exhaustive_python(page, hard_cap=hard_cap)


def _try_julia_available() -> bool:
    try:
        from ..bridge.julia_bridge import julia_engine
        return julia_engine.available and hasattr(
            julia_engine.backend, "exhaustive_e_inf"
        )
    except Exception:
        return False


__all__ = [
    "ExhaustiveEInfinityResult",
    "exhaustive_e_infinity_bounds",
]
