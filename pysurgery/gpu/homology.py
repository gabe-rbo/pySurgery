"""Exact homology on the GPU: ranks over F_p, and integral homology with torsion.

Overview:
    Homology of a chain complex is determined by its boundary matrices: over a field
    ``F``, ``beta_k(F) = dim C_k - rank_F d_k - rank_F d_{k+1}``; over ``Z``, the
    free rank uses ``rank_Q`` and the torsion of ``H_{k-1}`` is the list of invariant
    factors ``> 1`` of ``d_k`` (the Smith-normal-form diagonal). This module
    computes both on the GPU, **exactly**:

    * ``rank_F_p`` by Gaussian elimination modulo ``p`` in native integer
      arithmetic (int32 for ``p < 46341``, int64 for ``p < 2^31``), which never
      rounds;
    * the invariant factors over ``Z`` by *unimodular* elimination -- pivots on
      entries ``+-1``, or on entries dividing their whole row and column -- in int64
      with a rigorous overflow guard, followed by an exact arbitrary-precision SNF of
      whatever residual is left (for boundary matrices, usually nothing), on the CPU.

Key Concepts:
    - **Unit-singleton peeling (CPU, exact, no arithmetic)**: a column (or row) with
      a single nonzero entry ``+-1`` splits off: ``A ~ (+-1) (+) A'`` with ``A'`` the
      matrix minus that row and column, *unchanged*. Iterating is simplicial
      collapse at the level of one boundary matrix, costs ``O(nnz)`` and typically
      removes most of a boundary matrix before anything is densified. Over ``F_p``
      every nonzero singleton qualifies. See :func:`peel_unit_singletons`.
    - **Parallel pivot rounds (GPU)**: in each round every column proposes the first
      nonzero row it has; one proposal per row is kept. Sorted by row, the chosen
      pivots form a triangular block, so they can be eliminated one after another
      without re-checking -- no pivot is disturbed by an earlier one. One host
      synchronisation per round, not per pivot.
    - **Overflow guard**: before every rank-1 update over ``Z`` the kernel carries a
      bound ``M`` on every entry; the update is applied only while ``M <= 2^31``
      (so products stay below ``2^62``) and is otherwise masked to a no-op on the
      device, the round stops, and the exact maximum is re-measured. Nothing ever
      wraps around.
    - **Why not CRT / "rank drops mod p"?** Ranks modulo a handful of primes certify
      *which of those primes* carry torsion (a drop is proof), but not that no other
      prime does, and a single "large" prime only bounds ``rank_Q`` from below.
      Unimodular elimination decides everything; :func:`torsion_prime_screen` offers
      the modular screen separately, with its guarantees stated.

Common Workflows:
    1. **Integral homology** -> ``gpu_homology(sc)`` or ``sc.homology(backend="gpu")``.
    2. **Betti numbers over F_p** -> ``gpu_betti_numbers(sc, p=2)``.
    3. **One matrix** -> ``smith_invariant_factors(d2)``, ``rank_mod_p(d2, 3)``.
    4. **Pin the device** -> ``backend="gpu:cuda:1"`` / ``device="cpu"``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numba
import numpy as np
import scipy.sparse as sp

from .device import device_memory_budget, require_torch, resolve_device

torch = require_torch()

__all__ = [
    "rank_mod_p",
    "smith_invariant_factors",
    "SmithResult",
    "invariant_factors_from_diagonal",
    "peel_unit_singletons",
    "boundary_invariants",
    "chain_complex_homology",
    "gpu_homology",
    "gpu_betti_numbers",
    "torsion_prime_screen",
    "TorsionScreen",
    "DenseBudgetExceeded",
]

#: Largest prime modulus handled in int32 (``p^2 < 2^31``).
INT32_PRIME_LIMIT = 46341
#: Exclusive upper bound on a prime modulus (``p^2 < 2^62`` in int64).
PRIME_LIMIT = 2 ** 31
#: Entry bound under which an int64 rank-1 update cannot overflow.
_ENTRY_LIMIT = 2 ** 31
#: Default primes for :func:`torsion_prime_screen`.
DEFAULT_SCREEN_PRIMES: Tuple[int, ...] = (2, 3, 5, 7, 11, 13)
#: Large prime used as the reference rank in the screen (fits int32 arithmetic).
_REFERENCE_PRIME = 46337
#: Refuse to finish residuals larger than this with the exact Python SNF.
_RESIDUAL_CELL_BUDGET = 4_000_000


class DenseBudgetExceeded(MemoryError):
    """The dense block left after peeling would not fit the device memory budget."""


# ──────────────────────────────────────────────────────────────────────────────
# Invariant factors from any diagonalisation
# ──────────────────────────────────────────────────────────────────────────────
def invariant_factors_from_diagonal(entries: Iterable[int]) -> List[int]:
    """Invariant factors ``d_1 | d_2 | ...`` of ``diag(entries)``.

    What is Being Computed?:
        Any diagonal matrix equivalent to ``A`` determines the same abelian group
        ``coker``; its Smith normal form is obtained without factoring by the
        pairwise exchange ``(a_i, a_j) -> (gcd, lcm)`` for ``i < j``. For every prime
        separately this is a selection sort of the exponents, so the result is the
        divisibility chain. ``diag(2, 3)`` becomes ``[1, 6]``.

    Args:
        entries: Nonzero integers (signs ignored; zeros are dropped).

    Returns:
        The nonzero invariant factors, ascending, as Python ints (length = rank).
    """
    a = [abs(int(e)) for e in entries if int(e) != 0]
    ones = sum(1 for x in a if x == 1)
    rest = [x for x in a if x != 1]
    for i in range(len(rest)):
        for j in range(i + 1, len(rest)):
            g = math.gcd(rest[i], rest[j])
            if g != rest[i]:
                rest[i], rest[j] = g, rest[i] // g * rest[j]
    rest.sort()
    return [1] * ones + rest


# ──────────────────────────────────────────────────────────────────────────────
# Unit-singleton peeling (CPU, exact)
# ──────────────────────────────────────────────────────────────────────────────
@numba.njit(cache=True)
def _peel_kernel(n_rows, n_cols, r_ptr, r_idx, r_val, c_ptr, c_idx, c_val, any_nonzero):
    """Queue-driven singleton elimination; ``any_nonzero`` treats every nonzero as a unit."""
    row_alive = np.ones(n_rows, dtype=np.bool_)
    col_alive = np.ones(n_cols, dtype=np.bool_)
    row_cnt = np.empty(n_rows, dtype=np.int64)
    col_cnt = np.empty(n_cols, dtype=np.int64)
    for i in range(n_rows):
        row_cnt[i] = r_ptr[i + 1] - r_ptr[i]
    for j in range(n_cols):
        col_cnt[j] = c_ptr[j + 1] - c_ptr[j]
    col_stack = np.empty(n_cols + r_idx.shape[0] + 1, dtype=np.int64)
    row_stack = np.empty(n_rows + r_idx.shape[0] + 1, dtype=np.int64)
    nc, nr = 0, 0
    for j in range(n_cols):
        if col_cnt[j] == 1:
            col_stack[nc] = j
            nc += 1
    for i in range(n_rows):
        if row_cnt[i] == 1:
            row_stack[nr] = i
            nr += 1
    pivots = 0
    while nc > 0 or nr > 0:
        pi, pj = -1, -1
        if nc > 0:
            nc -= 1
            j = col_stack[nc]
            if col_alive[j] and col_cnt[j] == 1:
                for t in range(c_ptr[j], c_ptr[j + 1]):
                    if row_alive[c_idx[t]]:
                        v = c_val[t]
                        if any_nonzero or v == 1 or v == -1:
                            pi, pj = c_idx[t], j
                        break
        else:
            nr -= 1
            i = row_stack[nr]
            if row_alive[i] and row_cnt[i] == 1:
                for t in range(r_ptr[i], r_ptr[i + 1]):
                    if col_alive[r_idx[t]]:
                        v = r_val[t]
                        if any_nonzero or v == 1 or v == -1:
                            pi, pj = i, r_idx[t]
                        break
        if pi < 0:
            continue
        row_alive[pi] = False
        col_alive[pj] = False
        pivots += 1
        for t in range(r_ptr[pi], r_ptr[pi + 1]):       # the row leaves every column
            l = r_idx[t]
            if col_alive[l]:
                col_cnt[l] -= 1
                if col_cnt[l] == 1:
                    col_stack[nc] = l
                    nc += 1
        for t in range(c_ptr[pj], c_ptr[pj + 1]):       # the column leaves every row
            k = c_idx[t]
            if row_alive[k]:
                row_cnt[k] -= 1
                if row_cnt[k] == 1:
                    row_stack[nr] = k
                    nr += 1
    return pivots, row_alive, col_alive


def _to_csr_int64(matrix) -> sp.csr_matrix:
    """Coerce a matrix-like to a fresh CSR int64 copy, rejecting non-integer entries.

    Always a copy: the in-place clean-ups below (and in callers) must never touch
    the caller's matrix -- a boundary matrix shared with a cached chain complex.
    """
    if sp.issparse(matrix):
        M = sp.csr_matrix(matrix, copy=True)
    else:
        arr = np.asarray(matrix)
        if arr.ndim != 2:
            raise ValueError(f"expected a 2-D matrix, got shape {arr.shape}")
        M = sp.csr_matrix(arr)
    data = M.data
    if data.dtype.kind == "f":
        if not np.all(np.isfinite(data)) or np.any(data != np.round(data)):
            raise ValueError("boundary matrices must have integer entries")
    elif data.dtype.kind == "O":
        if any(abs(int(v)) >= 2 ** 62 for v in data):
            raise ValueError("entries must be below 2^62 in absolute value")
    M = sp.csr_matrix((data.astype(np.int64), M.indices.copy(), M.indptr.copy()), shape=M.shape)
    M.sum_duplicates()
    M.eliminate_zeros()
    return M


def peel_unit_singletons(matrix, *, any_nonzero: bool = False) -> Tuple[int, sp.csr_matrix]:
    """Split off every unit singleton, iteratively; exact and arithmetic-free.

    What is Being Computed?:
        If column ``j`` (or row ``i``) of ``A`` has a single nonzero entry ``u`` and
        ``u`` is a unit, elementary operations that touch nothing else give
        ``A ~ (u) (+) A'`` where ``A'`` is ``A`` with that row and column deleted --
        no entry changes. Repeating until no singleton remains yields ``k`` pivots
        and a residual submatrix ``R`` with ``SNF(A) = diag(1^k) (+) SNF(R)`` and
        ``rank_F(A) = k + rank_F(R)`` over every field ``F``.

    Algorithm:
        A worklist of singleton rows and columns, maintained by live nonzero
        counts (numba, ``O(nnz)`` total).

    Args:
        matrix: Integer matrix (sparse or dense).
        any_nonzero: Treat every nonzero entry as a unit (valid over a field, for a
            matrix already reduced modulo ``p``).

    Returns:
        Tuple ``(k, R)``: the number of pivots split off and the residual (CSR,
        int64, with empty rows and columns removed).
    """
    M = _to_csr_int64(matrix)
    m, n = M.shape
    if M.nnz == 0:
        return 0, sp.csr_matrix((0, 0), dtype=np.int64)
    C = M.tocsc()
    k, ra, ca = _peel_kernel(m, n, M.indptr.astype(np.int64), M.indices.astype(np.int64),
                             M.data, C.indptr.astype(np.int64), C.indices.astype(np.int64),
                             C.data, bool(any_nonzero))
    R = M[np.nonzero(ra)[0]][:, np.nonzero(ca)[0]]
    R = sp.csr_matrix(R)
    R.eliminate_zeros()
    if R.nnz == 0:
        return int(k), sp.csr_matrix((0, 0), dtype=np.int64)
    nzr = np.diff(R.indptr) > 0
    nzc = np.bincount(R.indices, minlength=R.shape[1]) > 0
    R = R[np.nonzero(nzr)[0]][:, np.nonzero(nzc)[0]]
    return int(k), sp.csr_matrix(R, dtype=np.int64)


# ──────────────────────────────────────────────────────────────────────────────
# Dense device kernels
# ──────────────────────────────────────────────────────────────────────────────
def _rank1_sub_(A, c, r) -> None:
    """In place ``A -= c r`` (``c`` a column, ``r`` a row), without an ``m x n`` temporary when possible."""
    try:
        A.addcmul_(c, r, value=-1)
    except (RuntimeError, NotImplementedError, TypeError):
        A.sub_(c * r)


def _compact(A):
    """Drop the zero rows and columns of ``A`` (one host synchronisation)."""
    nz = A != 0
    rows = torch.nonzero(nz.any(1), as_tuple=False).flatten()
    cols = torch.nonzero(nz.any(0), as_tuple=False).flatten()
    if rows.numel() == 0 or cols.numel() == 0:
        return A[:0, :0]
    if rows.numel() == A.shape[0] and cols.numel() == A.shape[1]:
        return A
    return A.index_select(0, rows).index_select(1, cols)


def _first_true_row(nz):
    """Index of the first True entry of every column (0 for an all-False column).

    ``argmax`` returns the first maximal index; a ``uint8`` view keeps the
    temporary at one byte per entry. Backends without a ``uint8`` argmax get the
    ``int32`` view instead.
    """
    try:
        return nz.to(torch.uint8).argmax(0)
    except (RuntimeError, NotImplementedError, TypeError):
        return nz.to(torch.int32).argmax(0)


def _round_pivots(A, admissible=None, limit: int = 512):
    """Pivots for one round: each column's first nonzero row, one column per row.

    Sorted by row, the chosen ``(i_a, j_a)`` satisfy ``A[i_a, j_b] = 0`` for
    ``a < b`` (``i_b`` is the *first* nonzero of column ``j_b`` and ``i_a < i_b``),
    so eliminating them in order never disturbs a later pivot. ``admissible``
    (a boolean mask over the columns' first entries) restricts which proposals
    count -- over ``Z``, only units.

    Returns:
        Tuple ``(rows, cols)`` of device index tensors, sorted by row.
    """
    nz = A != 0
    first = _first_true_row(nz)
    has = nz.any(0)
    if admissible is not None:
        has = has & admissible(first)
    cols = torch.nonzero(has, as_tuple=False).flatten()
    if cols.numel() == 0:
        return cols, cols
    rows = first.index_select(0, cols)
    order = torch.argsort(rows * A.shape[1] + cols)
    rows, cols = rows[order], cols[order]
    keep = torch.ones_like(rows, dtype=torch.bool)
    keep[1:] = rows[1:] != rows[:-1]
    return rows[keep][:limit], cols[keep][:limit]


def _dense_rank_mod_p(A, p: int, round_limit: int = 512) -> int:
    """Rank of a dense matrix over ``F_p`` (entries already in ``[0, p)``)."""
    rank = 0
    while True:
        A = _compact(A)
        if A.numel() == 0:
            return rank
        rows, cols = _round_pivots(A, limit=round_limit)
        t = int(rows.numel())
        piv = A[rows, cols].cpu().tolist()
        inv = torch.as_tensor([pow(int(v), -1, p) for v in piv], dtype=A.dtype, device=A.device)
        for a in range(t):
            i, j = rows[a:a + 1], cols[a:a + 1]
            r = A.index_select(0, i)                    # (1, n)
            c = A.index_select(1, j)                    # (m, 1); c_i * inv = 1
            c = torch.remainder(c * inv[a], p)
            _rank1_sub_(A, c, r)                        # column j -> 0, row i -> 0
            A.remainder_(p)
        rank += t


def _dense_unimodular(A, round_limit: int = 512):
    """Unimodular elimination over ``Z`` on the device.

    Returns:
        Tuple ``(n_unit, extra, R)``: the number of unit pivots, the non-unit
        divisor pivots split off (each ``a`` with ``A ~ (a) (+) A'``), and the
        residual as a host int64 array (possibly empty).
    """
    dev = A.device
    lim = torch.tensor(_ENTRY_LIMIT, dtype=torch.int64, device=dev)
    n_unit = 0
    extra: List[int] = []
    while True:
        A = _compact(A)
        if A.numel() == 0:
            return n_unit, extra, np.zeros((0, 0), dtype=np.int64)
        M = int(A.abs().max())
        if M > _ENTRY_LIMIT:
            break
        rows, cols = _round_pivots(A, admissible=lambda first: A.gather(
            0, first[None, :]).squeeze(0).abs() == 1, limit=round_limit)
        values: List[int]
        if rows.numel() == 0:
            unit = A.abs() == 1
            if bool(unit.any()):                        # a unit that is not first in its column
                flat = int(torch.nonzero(unit.flatten(), as_tuple=False)[0])
                rows = torch.tensor([flat // A.shape[1]], device=dev)
                cols = torch.tensor([flat % A.shape[1]], device=dev)
            else:                                       # a non-unit pivot dividing its row and column
                big = torch.iinfo(torch.int64).max
                mag = torch.where(A != 0, A.abs(), torch.full_like(A, big))
                flat = int(mag.flatten().argmin())
                i, j = flat // A.shape[1], flat % A.shape[1]
                a = int(A[i, j])
                if not (bool((A[i, :] % a == 0).all()) and bool((A[:, j] % a == 0).all())):
                    break
                rows = torch.tensor([i], device=dev)
                cols = torch.tensor([j], device=dev)
        values = A[rows, cols].cpu().tolist()
        bound = torch.tensor(M, dtype=torch.int64, device=dev)
        go = torch.ones((), dtype=torch.bool, device=dev)
        applied = torch.zeros((), dtype=torch.int64, device=dev)
        for a, v in enumerate(values):
            i, j = rows[a:a + 1], cols[a:a + 1]
            r = A.index_select(0, i)
            c = A.index_select(1, j)
            mult = c * v if abs(v) == 1 else torch.div(c, v, rounding_mode="floor")
            mc = torch.minimum(mult.abs().max(), lim)
            mr = torch.minimum(r.abs().max(), lim)
            safe = go & (bound <= lim) & (mult.abs().max() <= lim) & (r.abs().max() <= lim)
            _rank1_sub_(A, mult * safe.to(A.dtype), r)
            bound = bound + mc * mr * safe.to(torch.int64)
            applied = applied + safe.to(torch.int64)
            go = safe
        done = int(applied)
        for v in values[:done]:
            if abs(v) == 1:
                n_unit += 1
            else:
                extra.append(abs(int(v)))
        if done == 0:  # pragma: no cover - the first update of a round is always safe
            break
    return n_unit, extra, A.cpu().numpy()


def _on_device(run, dev, what: str):
    """Run ``run(dev)``; if an accelerator lacks an op or memory, warn and redo it on the CPU.

    The computation is exact on every device, so the CPU rerun returns the same
    answer; only the speed differs. Integer-op coverage on MPS varies with the
    macOS / PyTorch version, which is what this guards against.

    Args:
        run: Callable taking a ``torch.device``.
        dev: The requested device.
        what: Short description for the warning.

    Returns:
        Whatever ``run`` returns.
    """
    if dev.type == "cpu":
        return run(dev)
    try:
        return run(dev)
    except (NotImplementedError, RuntimeError, TypeError) as exc:
        import warnings
        warnings.warn(
            f"{what} failed on {dev} ({type(exc).__name__}: {str(exc)[:160]}); "
            "recomputing on the CPU (the result is exact either way).",
            RuntimeWarning, stacklevel=3,
        )
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return run(torch.device("cpu"))


def _budget_check(shape: Tuple[int, int], itemsize: int, device, max_dense_bytes: Optional[int]):
    """Refuse a dense block that would not fit (matrix plus working copies)."""
    need = int(shape[0]) * int(shape[1]) * itemsize * 3
    budget = device_memory_budget(device) if max_dense_bytes is None else int(max_dense_bytes)
    if need > budget:
        raise DenseBudgetExceeded(
            f"the dense block left after peeling is {shape[0]} x {shape[1]} "
            f"(~{need / 2**30:.2f} GiB with working copies), over the "
            f"{budget / 2**30:.2f} GiB budget on {device}. Pass a larger "
            "`max_dense_bytes`, use a device with more memory, or the sparse CPU "
            "backends (backend='julia' / 'python')."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Public matrix-level API
# ──────────────────────────────────────────────────────────────────────────────
def _check_prime(p: int) -> int:
    """Validate a prime modulus for exact int32/int64 arithmetic."""
    import sympy
    p = int(p)
    if p < 2 or not sympy.isprime(p):
        raise ValueError(f"rank over Z/{p}Z is only a field rank for a prime modulus; got {p}")
    if p >= PRIME_LIMIT:
        raise ValueError(f"prime modulus must be below 2^31 for exact int64 arithmetic; got {p}")
    return p


def rank_mod_p(matrix, p: int, *, device=None, presimplify: bool = True,
               max_dense_bytes: Optional[int] = None) -> int:
    """Rank of an integer matrix over ``F_p``, exactly, on the GPU.

    What is Being Computed?:
        ``rank_{F_p}(A mod p)`` by Gaussian elimination in native integers: int32
        when ``p < 46341`` (so every product ``< p^2 < 2^31``), int64 otherwise.
        No floating point is involved, so the answer is exact.

    Args:
        matrix: Integer matrix (``scipy.sparse`` or dense).
        p: A prime below ``2^31``.
        device: ``None`` for automatic selection.
        presimplify: Peel singletons on the CPU before densifying (exact).
        max_dense_bytes: Memory budget for the dense block (default: half of what
            the device has available).

    Returns:
        The rank over ``F_p``.

    Raises:
        ValueError: If ``p`` is not a prime below ``2^31``.
        DenseBudgetExceeded: If the dense block would not fit.

    Example:
        rank_mod_p(sc.boundary_matrix(2), 2)
    """
    p = _check_prime(p)
    M = _to_csr_int64(matrix)
    M = sp.csr_matrix((np.mod(M.data, p), M.indices, M.indptr), shape=M.shape)  # M is our copy
    M.eliminate_zeros()
    k = 0
    if presimplify and M.nnz:
        k, M = peel_unit_singletons(M, any_nonzero=True)
    if M.nnz == 0:
        return int(k)
    dev = resolve_device(device)
    dt = torch.int32 if p < INT32_PRIME_LIMIT else torch.int64
    dense = M.toarray()

    def run(d):
        _budget_check(M.shape, 4 if dt == torch.int32 else 8, d, max_dense_bytes)
        return _dense_rank_mod_p(torch.as_tensor(dense, dtype=dt, device=d), p)

    return int(k) + _on_device(run, dev, f"rank modulo {p}")


@dataclass
class SmithResult:
    """Invariant factors of an integer matrix, with how they were obtained.

    Attributes:
        factors: The nonzero invariant factors ``d_1 | d_2 | ...`` (ascending).
        n_peeled: Unit pivots split off on the CPU before densifying.
        n_device_pivots: Unit pivots eliminated on the device.
        divisor_pivots: Non-unit pivots split off on the device.
        residual_shape: Shape of the block finished by the exact CPU SNF.
        device: Device the dense phase ran on (``""`` if none was needed).
    """

    factors: List[int]
    n_peeled: int = 0
    n_device_pivots: int = 0
    divisor_pivots: List[int] = field(default_factory=list)
    residual_shape: Tuple[int, int] = (0, 0)
    device: str = ""

    @property
    def rank(self) -> int:
        """Rank over ``Q`` (the number of nonzero invariant factors)."""
        return len(self.factors)

    @property
    def torsion(self) -> List[int]:
        """The invariant factors greater than one."""
        return [d for d in self.factors if d > 1]


def smith_invariant_factors(matrix, *, device=None, presimplify: bool = True,
                            max_dense_bytes: Optional[int] = None,
                            return_details: bool = False):
    """Nonzero invariant factors (Smith normal form diagonal) of an integer matrix.

    What is Being Computed?:
        The diagonal ``d_1 | d_2 | ... | d_r`` of the Smith normal form of ``A``,
        exactly. ``r = rank_Q(A)``; for a boundary matrix ``d_k`` the factors
        ``> 1`` are the torsion coefficients of ``H_{k-1}``.

    Algorithm:
        1. Peel unit singletons on the CPU (exact, no arithmetic).
        2. Densify the residual on the device (int64) and eliminate in parallel
           rounds of unit pivots, then single pivots that divide their row and
           column, under the overflow guard (every step unimodular).
        3. Finish any residual with the exact arbitrary-precision SNF on the CPU.
        4. Canonicalise all diagonal entries into the divisibility chain.

    Args:
        matrix: Integer matrix (``scipy.sparse`` or dense).
        device: ``None`` for automatic selection.
        presimplify: Peel unit singletons first (exact).
        max_dense_bytes: Memory budget for the dense block.
        return_details: Return a :class:`SmithResult` instead of the factor array.

    Returns:
        The factors as an ``int64`` array (``object`` if some exceed int64), or a
        :class:`SmithResult`.

    Raises:
        DenseBudgetExceeded: If the dense block would not fit.
        MemoryError: If the residual left for the exact CPU finish is too large.

    Example:
        smith_invariant_factors(rp2.boundary_matrix(2))   # array([1, ..., 1, 2])
    """
    M = _to_csr_int64(matrix)
    k = 0
    if presimplify and M.nnz:
        k, M = peel_unit_singletons(M)
    res = SmithResult(factors=[], n_peeled=int(k))
    diag: List[int] = [1] * int(k)
    if M.nnz:
        dev = resolve_device(device)
        dense = M.toarray()
        used: List[str] = []

        def run(d):
            _budget_check(M.shape, 8, d, max_dense_bytes)
            used.append(str(d))
            return _dense_unimodular(torch.as_tensor(dense, dtype=torch.int64, device=d))

        n_unit, extra, R = _on_device(run, dev, "unimodular elimination")
        res.device = used[-1]
        res.n_device_pivots = int(n_unit)
        res.divisor_pivots = list(extra)
        diag.extend([1] * int(n_unit))
        diag.extend(extra)
        if R.size:
            R = R[np.any(R != 0, axis=1)][:, np.any(R != 0, axis=0)]
        if R.size:
            res.residual_shape = tuple(int(s) for s in R.shape)
            if R.size > _RESIDUAL_CELL_BUDGET:
                raise MemoryError(
                    f"the residual left for the exact CPU Smith form is {R.shape}; "
                    "too large for the arbitrary-precision finish")
            from pysurgery.algebra.math_core import get_snf_diagonal
            diag.extend(int(x) for x in get_snf_diagonal(R.astype(object)))
    res.factors = invariant_factors_from_diagonal(diag)
    if return_details:
        return res
    try:
        return np.asarray(res.factors, dtype=np.int64)
    except OverflowError:  # pragma: no cover - astronomically large torsion
        return np.asarray(res.factors, dtype=object)


# ──────────────────────────────────────────────────────────────────────────────
# Chain-complex level
# ──────────────────────────────────────────────────────────────────────────────
def _ring(coefficient_ring: str) -> Tuple[str, Optional[int]]:
    """Parse a coefficient ring label with pySurgery's own parser."""
    from pysurgery.topology.complexes import _parse_coefficient_ring
    return _parse_coefficient_ring(coefficient_ring)


def boundary_invariants(matrix, coefficient_ring: str = "Z", *, device=None,
                        max_dense_bytes: Optional[int] = None) -> Tuple[int, List[int]]:
    """Rank and torsion of one boundary matrix over a coefficient ring.

    Args:
        matrix: Integer boundary matrix.
        coefficient_ring: ``"Z"``, ``"Q"`` or ``"Z/pZ"`` with ``p`` prime.
        device: ``None`` for automatic selection.
        max_dense_bytes: Memory budget for the dense block.

    Returns:
        Tuple ``(rank, torsion)``: rank over ``Q`` (for ``Z`` and ``Q``) or over
        ``F_p``, and the invariant factors ``> 1`` (``Z`` only; empty otherwise).

    Raises:
        ValueError: For a composite modulus (use :func:`chain_complex_homology`,
            which applies the universal coefficient theorem).
    """
    kind, p = _ring(coefficient_ring)
    if kind in ("Z", "Q"):
        res = smith_invariant_factors(matrix, device=device, max_dense_bytes=max_dense_bytes,
                                      return_details=True)
        return res.rank, (res.torsion if kind == "Z" else [])
    return rank_mod_p(matrix, int(p), device=device, max_dense_bytes=max_dense_bytes), []


def _is_prime(n: int) -> bool:
    """Primality via sympy."""
    import sympy
    return bool(sympy.isprime(int(n)))


def chain_complex_homology(boundaries: Dict[int, Any], cells: Dict[int, int],
                           coefficient_ring: str = "Z", degrees: Optional[Sequence[int]] = None,
                           *, device=None, max_dense_bytes: Optional[int] = None,
                           _cache: Optional[dict] = None) -> Dict[int, Tuple[int, List[int]]]:
    """Homology ``H_n = ker d_n / im d_{n+1}`` of a chain complex, on the GPU.

    What is Being Computed?:
        For each degree ``n``: ``(rank, torsion)`` with ``rank = dim C_n -
        rank d_n - rank d_{n+1}`` (ranks over ``Q``, or over ``F_p`` for a prime
        modulus) and ``torsion`` the invariant factors ``> 1`` of ``d_{n+1}`` over
        ``Z``. A composite modulus ``Z/nZ`` is derived from integral homology by the
        universal coefficient theorem, exactly as the CPU path does.

    Args:
        boundaries: ``{k: d_k}`` with ``d_k: C_k -> C_{k-1}`` (rows index
            ``(k-1)``-cells).
        cells: ``{k: dim C_k}``.
        coefficient_ring: ``"Z"``, ``"Q"`` or ``"Z/nZ"``.
        degrees: Degrees to compute (default: every degree with cells).
        device: ``None`` for automatic selection.
        max_dense_bytes: Memory budget for each dense block.

    Returns:
        ``{n: (rank, torsion)}``.
    """
    kind, p = _ring(coefficient_ring)
    cache = {} if _cache is None else _cache
    if degrees is None:
        ds = set(int(d) for d in cells)
        for d in boundaries:
            ds.update({int(d), int(d) - 1})
        degrees = sorted(d for d in ds if d >= 0)

    def size(n):
        if n in cells:
            return int(cells[n])
        if boundaries.get(n) is not None:
            return int(boundaries[n].shape[1])
        if boundaries.get(n + 1) is not None:
            return int(boundaries[n + 1].shape[0])
        return 0

    def inv(k, ring):
        key = (int(k), ring)
        if key not in cache:
            mat = boundaries.get(k)
            if mat is None or (sp.issparse(mat) and mat.nnz == 0) or (
                    not sp.issparse(mat) and not np.any(np.asarray(mat))):
                cache[key] = (0, [])
            else:
                cache[key] = boundary_invariants(mat, ring, device=device,
                                                 max_dense_bytes=max_dense_bytes)
        return cache[key]

    out: Dict[int, Tuple[int, List[int]]] = {}
    for n in degrees:
        n = int(n)
        if kind == "ZMOD" and not _is_prime(int(p)):
            from pysurgery.topology.complexes import _composite_mod_uct_decomposition
            r_n, t_n = _integral(n, size, inv)
            _r, t_nm1 = _integral(n - 1, size, inv) if n >= 1 else (0, [])
            out[n] = _composite_mod_uct_decomposition(r_n, t_n, t_nm1, int(p))
            continue
        ring = "Z" if kind in ("Z", "Q") else f"Z/{int(p)}Z"
        c_n = size(n)
        r_n = inv(n, ring)[0]
        r_n1, tors = inv(n + 1, ring)
        out[n] = (max(0, c_n - r_n - r_n1), list(tors) if kind == "Z" else [])
    return out


def _integral(n, size, inv) -> Tuple[int, List[int]]:
    """Integral ``(rank, torsion)`` of degree ``n`` from cached boundary invariants."""
    if n < 0:
        return 0, []
    r_n = inv(n, "Z")[0]
    r_n1, tors = inv(n + 1, "Z")
    return max(0, size(n) - r_n - r_n1), list(tors)


def _chain_data(obj, coefficient_ring: Optional[str]):
    """Boundaries, cells and ring of a complex-like object."""
    ring = coefficient_ring or getattr(obj, "coefficient_ring", "Z")
    if hasattr(obj, "chain_complex") and callable(obj.chain_complex):
        cc = obj.chain_complex()
    elif hasattr(obj, "cellular_chain_complex") and callable(obj.cellular_chain_complex):
        cc = obj.cellular_chain_complex()
    else:
        cc = obj
    return dict(cc.boundaries), dict(cc.cells), ring


def gpu_homology(obj, n: Optional[int] = None, coefficient_ring: Optional[str] = None, *,
                 device=None, max_dense_bytes: Optional[int] = None):
    """Homology of a pySurgery complex on the GPU (exact).

    Args:
        obj: A ``SimplicialComplex``, ``CWComplex`` or ``ChainComplex``.
        n: A single degree, or ``None`` for every degree.
        coefficient_ring: Override the complex's ring (``"Z"``, ``"Q"``, ``"Z/nZ"``).
        device: ``None`` for automatic selection.
        max_dense_bytes: Memory budget for each dense block.

    Returns:
        ``(rank, torsion)`` for a degree, or ``{n: (rank, torsion)}``.

    Example:
        gpu_homology(rp2)            # {0: (1, []), 1: (0, [2]), 2: (0, [])}
    """
    bd, cells, ring = _chain_data(obj, coefficient_ring)
    degrees = None if n is None else [int(n)]
    out = chain_complex_homology(bd, cells, ring, degrees, device=device,
                                 max_dense_bytes=max_dense_bytes)
    return out[int(n)] if n is not None else out


def gpu_betti_numbers(obj, p: Optional[int] = None, *, device=None,
                      max_dense_bytes: Optional[int] = None) -> Dict[int, int]:
    """Betti numbers over ``Q`` (``p=None``) or over ``F_p``, on the GPU.

    Args:
        obj: A ``SimplicialComplex``, ``CWComplex`` or ``ChainComplex``.
        p: ``None`` for rational Betti numbers, or a prime.
        device: ``None`` for automatic selection.
        max_dense_bytes: Memory budget for each dense block.

    Returns:
        ``{n: beta_n}``.
    """
    ring = "Q" if p is None else f"Z/{_check_prime(p)}Z"
    return {d: r for d, (r, _t) in gpu_homology(obj, None, ring, device=device,
                                                  max_dense_bytes=max_dense_bytes).items()}


@dataclass
class TorsionScreen:
    """Result of the modular torsion screen.

    Attributes:
        primes: The primes tested.
        ranks: ``{k: {p: rank_{F_p} d_k}}``.
        reference_rank: ``{k: max_p rank_{F_p} d_k}`` -- a lower bound on
            ``rank_Q d_k``.
        detected: ``{k-1: {p: count}}`` -- primes proven to divide the torsion of
            ``H_{k-1}``, with a lower bound on the number of cyclic ``p``-primary
            summands.
    """

    primes: List[int]
    ranks: Dict[int, Dict[int, int]]
    reference_rank: Dict[int, int]
    detected: Dict[int, Dict[int, int]]


def torsion_prime_screen(obj, primes: Sequence[int] = DEFAULT_SCREEN_PRIMES, *,
                         device=None, max_dense_bytes: Optional[int] = None) -> TorsionScreen:
    """Screen for torsion by comparing ranks modulo several primes.

    What is Being Computed?:
        ``rank_{F_p} d_k`` for each tested prime (plus a large reference prime).
        ``rank_{F_p} d_k`` equals the number of invariant factors not divisible by
        ``p``, so a prime whose rank falls below another prime's rank **provably**
        divides the torsion of ``H_{k-1}``. What the screen cannot prove is the
        converse: untested primes, and primes dividing every factor alike, go
        unseen. For the complete answer use :func:`gpu_homology` over ``Z``.

    Args:
        obj: A complex (anything :func:`gpu_homology` accepts).
        primes: Primes to test.
        device: ``None`` for automatic selection.
        max_dense_bytes: Memory budget for each dense block.

    Returns:
        A :class:`TorsionScreen`.
    """
    bd, _cells, _ring_ = _chain_data(obj, None)
    ps = sorted({_check_prime(q) for q in primes} | {_REFERENCE_PRIME})
    ranks: Dict[int, Dict[int, int]] = {}
    ref: Dict[int, int] = {}
    detected: Dict[int, Dict[int, int]] = {}
    for k in sorted(bd):
        mat = bd[k]
        if mat is None or (sp.issparse(mat) and mat.nnz == 0):
            continue
        ranks[k] = {q: rank_mod_p(mat, q, device=device, max_dense_bytes=max_dense_bytes)
                    for q in ps}
        ref[k] = max(ranks[k].values())
        drops = {q: ref[k] - r for q, r in ranks[k].items() if r < ref[k]}
        if drops:
            detected[k - 1] = drops
    return TorsionScreen(primes=ps, ranks=ranks, reference_rank=ref, detected=detected)
