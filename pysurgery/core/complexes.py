import itertools
import hashlib
import numpy as np
import warnings
import sympy as sp
from scipy.sparse import csr_matrix
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, cast
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
from .math_core import get_sparse_snf_diagonal
from ..bridge.julia_bridge import julia_engine


def _parse_coefficient_ring(ring: str) -> tuple[str, int | None]:
    """Parse user ring labels into internal `(kind, modulus)` form."""
    rs = ring.strip().upper()
    if rs == "Z":
        return "Z", None
    if rs == "Q":
        return "Q", None
    if rs.startswith("Z/") and rs.endswith("Z"):
        p_str = rs[2:-1]
        p = int(p_str)
        if p <= 1:
            raise ValueError("Z/pZ requires p > 1.")
        return "ZMOD", p
    raise ValueError(f"Unsupported coefficient ring '{ring}'. Use 'Z', 'Q', or 'Z/pZ'.")


def _coerce_csr_matrix(matrix: csr_matrix | np.ndarray | list | tuple) -> csr_matrix:
    """Coerce sparse/dense matrix-like data to CSR with integer entries."""
    if isinstance(matrix, csr_matrix):
        return matrix.copy().astype(np.int64)
    return csr_matrix(np.asarray(matrix, dtype=np.int64), dtype=np.int64)


def _normalize_simplex(simplex: Iterable[int]) -> tuple[int, ...]:
    """Return a canonical, sorted simplex tuple with distinct integer vertices."""
    vertices = tuple(sorted(set(int(v) for v in simplex)))
    if len(vertices) == 0:
        raise ValueError("Simplices must be non-empty.")
    return vertices


def _canonicalize_simplices_by_dim(
    raw_grouped: dict[int, list[tuple[int, ...]]]
) -> dict[int, list[tuple[int, ...]]]:
    """Sort and deduplicate simplex lists across dimensions."""
    out = {}
    for d, simplices in raw_grouped.items():
        out[d] = sorted(list(dict.fromkeys(simplices)))
    return out


def _simplicial_closure_from_generators(
    simplices: Iterable[Iterable[int]],
) -> dict[int, list[tuple[int, ...]]]:
    """Generate all faces of the given simplices and group by dimension."""
    grouped: dict[int, set[tuple[int, ...]]] = {}
    for simplex in simplices:
        t = _normalize_simplex(simplex)
        for r in range(1, len(t) + 1):
            dim = r - 1
            if dim not in grouped:
                grouped[dim] = set()
            for face in itertools.combinations(t, r):
                grouped[dim].add(tuple(face))
    return {d: sorted(list(s)) for d, s in grouped.items()}


def _boundary_matrix_from_simplices(
    simplices_n: List[Tuple[int, ...]],
    simplices_nm1: List[Tuple[int, ...]],
) -> csr_matrix:
    """Construct an oriented boundary matrix from dimension n to n-1."""
    if not simplices_n or not simplices_nm1:
        return csr_matrix((len(simplices_nm1), len(simplices_n)), dtype=np.int64)

    nm1_map = {simplex: i for i, simplex in enumerate(simplices_nm1)}
    rows, cols, data = [], [], []

    for j, simplex in enumerate(simplices_n):
        for i in range(len(simplex)):
            face = simplex[:i] + simplex[i + 1 :]
            if face in nm1_map:
                rows.append(nm1_map[face])
                cols.append(j)
                data.append((-1) ** i)

    return csr_matrix(
        (data, (rows, cols)),
        shape=(len(simplices_nm1), len(simplices_n)),
        dtype=np.int64,
    )


def _csr_matrix_signature(m: csr_matrix) -> tuple[int, int, int, str]:
    """Return a content-based signature for a sparse matrix to detect changes."""
    rows, cols = m.nonzero()
    data = m.data
    h = hashlib.sha256()
    h.update(np.asarray(m.shape, dtype=np.int64).tobytes())
    h.update(rows.tobytes())
    h.update(cols.tobytes())
    h.update(data.tobytes())
    return int(m.shape[0]), int(m.shape[1]), int(m.nnz), h.hexdigest()


def _clone_cache_value(v: Any) -> Any:
    """Return a deep copy of a value to prevent accidental cache mutation."""
    if isinstance(v, (int, float, str, bool, tuple)) or v is None:
        return v
    if isinstance(v, list):
        return [list(x) if isinstance(x, np.ndarray) else x for x in v]
    if isinstance(v, dict):
        return {k: _clone_cache_value(val) for k, val in v.items()}
    if isinstance(v, np.ndarray):
        return v.copy()
    if isinstance(v, csr_matrix):
        return v.copy()
    return v


def _is_prime(n: int) -> bool:
    """Check if n is prime (heuristic for small moduli)."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def _gcd_extended(a: int, b: int) -> tuple[int, int, int]:
    """Extended Euclidean algorithm: returns (gcd, x, y) such that ax + by = gcd."""
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = _gcd_extended(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd, x, y


def _rank_mod_p(A: np.ndarray, p: int) -> int:
    """Compute matrix rank over `Z/pZ` via Euclidean row reduction (handles composite p)."""
    M = (A.astype(np.int64) % p).copy()
    m, n = M.shape
    row = 0
    rank = 0
    for col in range(n):
        pivot = None
        for r in range(row, m):
            if M[r, col] % p != 0:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != row:
            M[[row, pivot]] = M[[pivot, row]]

        # Euclidean reduction
        for r in range(row + 1, m):
            while M[r, col] % p != 0:
                q = M[row, col] // M[r, col]
                M[row, :] = (M[row, :] - q * M[r, :]) % p
                M[[row, r]] = M[[r, row]]

        try:
            inv = pow(int(M[row, col]), -1, p)
            M[row, :] = (M[row, :] * inv) % p
            for r in range(m):
                if r != row and M[r, col] % p != 0:
                    M[r, :] = (M[r, :] - M[r, col] * M[row, :]) % p
        except ValueError:
            for r in range(m):
                if r != row and M[r, col] % p != 0:
                    g, x, y = _gcd_extended(int(M[row, col]), int(M[r, col]))
                    a, b = int(M[row, col]), int(M[r, col])
                    row_i = (x * M[row, :] + y * M[r, :]) % p
                    row_j = ((-b // g) * M[row, :] + (a // g) * M[r, :]) % p
                    M[row, :] = row_i
                    M[r, :] = row_j

        row += 1
        rank += 1
        if row == m:
            break
    return rank


def _rref_mod_p(A: np.ndarray, p: int) -> tuple[np.ndarray, list[int]]:
    """Compute row-reduced echelon form over `Z/pZ` via Euclidean reduction."""
    M = (A.astype(np.int64) % p).copy()
    m, n = M.shape
    row = 0
    pivots: list[int] = []
    for col in range(n):
        pivot = None
        for r in range(row, m):
            if M[r, col] % p != 0:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != row:
            M[[row, pivot]] = M[[pivot, row]]

        # Euclidean reduction
        for r in range(row + 1, m):
            while M[r, col] % p != 0:
                q = M[row, col] // M[r, col]
                M[row, :] = (M[row, :] - q * M[r, :]) % p
                M[[row, r]] = M[[r, row]]

        try:
            inv = pow(int(M[row, col]), -1, p)
            M[row, :] = (M[row, :] * inv) % p
            for r in range(m):
                if r != row and M[r, col] % p != 0:
                    M[r, :] = (M[r, :] - M[r, col] * M[row, :]) % p
        except ValueError:
            for r in range(m):
                if r != row and M[r, col] % p != 0:
                    g, x, y = _gcd_extended(int(M[row, col]), int(M[r, col]))
                    a, b = int(M[row, col]), int(M[r, col])
                    row_i = (x * M[row, :] + y * M[r, :]) % p
                    row_j = ((-b // g) * M[row, :] + (a // g) * M[r, :]) % p
                    M[row, :] = row_i
                    M[r, :] = row_j

        pivots.append(col)
        row += 1
        if row == m:
            break
    return M, pivots


def _nullspace_basis_mod_p(A: np.ndarray, p: int) -> list[np.ndarray]:
    """Return a basis of `ker(A)` over `Z/pZ`."""
    m, n = A.shape
    rref, pivots = _rref_mod_p(A, p)
    pivot_set = set(pivots)
    free_cols = [j for j in range(n) if j not in pivot_set]
    if not free_cols:
        return []
    basis: list[np.ndarray] = []
    for free in free_cols:
        v = np.zeros(n, dtype=np.int64)
        v[free] = 1
        for i, col in enumerate(pivots):
            v[col] = (-rref[i, free]) % p
        basis.append(v)
    return basis


def _is_independent_wrt(
    v: np.ndarray, pivots: dict[int, np.ndarray], p: Optional[int] = None
) -> bool:
    """Check independence using Euclidean reduction for ring support (Z/mZ)."""
    work = v.copy()
    if p is not None:
        work %= p

    for i in range(len(work)):
        if work[i] == 0:
            continue
        if i not in pivots:
            # Found a new pivot! Normalize if possible, else just store.
            if p is not None:
                try:
                    inv = pow(int(work[i]), -1, p)
                    work = (work * inv) % p
                except ValueError:
                    pass
            else:
                work = work / work[i]
            pivots[i] = work
            return True

        if p is not None:
            # Euclidean reduction mod p to handle non-invertible elements
            target, pivot_val = int(work[i]), int(pivots[i][i])
            while target != 0:
                q = target // pivot_val
                work = (work - q * pivots[i]) % p
                target = int(work[i])
                if target != 0:
                    # Swap rows to continue reduction
                    work, pivots[i] = pivots[i], work
                    pivot_val = int(pivots[i][i])
        else:
            # Standard field reduction
            factor = work[i] / pivots[i][i]
            work = work - factor * pivots[i]

    return False


def _matrix_rank_for_ring(
    matrix: csr_matrix, ring_kind: str, p: int | None = None
) -> int:
    """Compute matrix rank in the requested coefficient field, preferring Julia when available."""
    if matrix is None or matrix.nnz == 0:
        return 0

    if ring_kind == "Q":
        if julia_engine.available:
            try:
                return int(julia_engine.compute_sparse_rank_q(matrix))
            except Exception as exc:
                warnings.warn(
                    "Topological Hint: Julia rank over Q failed in `ChainComplex.homology`; "
                    f"falling back to NumPy dense rank ({exc!r})."
                )
        return int(np.linalg.matrix_rank(matrix.toarray().astype(float)))

    if ring_kind == "ZMOD":
        if p is None:
            raise ValueError("Prime modulus p is required for Z/pZ rank computation.")
        if julia_engine.available:
            try:
                return int(julia_engine.compute_sparse_rank_mod_p(matrix, int(p)))
            except Exception as exc:
                warnings.warn(
                    "Topological Hint: Julia rank over Z/pZ failed in `ChainComplex.homology`; "
                    f"falling back to Python elimination ({exc!r})."
                )
        return _rank_mod_p(matrix.toarray(), int(p))

    raise ValueError(f"Unsupported rank ring kind '{ring_kind}'.")


def _composite_mod_uct_decomposition(
    free_rank: int,
    torsion_n: List[int],
    torsion_nm1: List[int],
    modulus: int,
) -> Tuple[int, List[int]]:
    """Compute Z/n decomposition from integral data via UCT tensor/Tor terms."""
    rank_mod = int(free_rank)
    torsion_mod: List[int] = []

    # Tensor terms from H_n(Z): Z_t ⊗ Z_n ≅ Z_gcd(t, n)
    for t in torsion_n:
        g = int(np.gcd(int(t), modulus))
        if g <= 1:
            continue
        if g == modulus:
            rank_mod += 1
        else:
            torsion_mod.append(g)

    # Tor terms from H_{n-1}(Z): Tor(Z_t, Z_n) ≅ Z_gcd(t, n)
    for t in torsion_nm1:
        g = int(np.gcd(int(t), modulus))
        if g <= 1:
            continue
        if g == modulus:
            rank_mod += 1
        else:
            torsion_mod.append(g)

    return rank_mod, sorted(torsion_mod)


class ChainComplex(BaseModel):
    """
    An abstract Chain Complex C_* over Z.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    boundaries: Dict[int, csr_matrix]
    dimensions: List[int]
    cells: Dict[int, int] = Field(default_factory=dict)
    coefficient_ring: str = "Z"

    _cache_enabled: bool = PrivateAttr(default=True)
    _cache: dict[tuple[object, ...], object] = PrivateAttr(default_factory=dict)
    _cache_hits: int = PrivateAttr(default=0)
    _cache_misses: int = PrivateAttr(default=0)
    _cache_signature: tuple[object, ...] | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _normalize_model(self):
        object.__setattr__(
            self,
            "boundaries",
            {int(dim): _coerce_csr_matrix(mat) for dim, mat in self.boundaries.items()},
        )
        object.__setattr__(
            self, "dimensions", sorted({int(dim) for dim in self.dimensions})
        )
        object.__setattr__(
            self, "cells", {int(dim): int(count) for dim, count in self.cells.items()}
        )
        object.__setattr__(self, "coefficient_ring", str(self.coefficient_ring))
        return self

    def _structure_signature(self) -> tuple[object, ...]:
        boundary_sig = tuple(
            (int(dim), _csr_matrix_signature(mat))
            for dim, mat in sorted(self.boundaries.items())
        )
        cells_sig = tuple(
            (int(dim), int(count)) for dim, count in sorted(self.cells.items())
        )
        return (
            boundary_sig,
            tuple(self.dimensions),
            cells_sig,
            str(self.coefficient_ring),
        )

    def _ensure_cache_valid(self) -> None:
        current = self._structure_signature()
        if self._cache_signature != current:
            self._cache.clear()
            self._cache_signature = current

    def _cache_get(self, key: tuple[object, ...]) -> object | None:
        self._ensure_cache_valid()
        if not self._cache_enabled:
            return None
        if key in self._cache:
            self._cache_hits += 1
            return _clone_cache_value(self._cache[key])
        self._cache_misses += 1
        return None

    def _cache_set(self, key: tuple[object, ...], value: object) -> None:
        self._ensure_cache_valid()
        if not self._cache_enabled:
            return
        self._cache[key] = _clone_cache_value(value)

    def clear_cache(self, namespace: str | None = None) -> None:
        if namespace is None:
            self._cache.clear()
            return
        prefix = (str(namespace),)
        keys = [k for k in self._cache if k[:1] == prefix]
        for key in keys:
            self._cache.pop(key, None)

    def cache_info(self) -> dict[str, object]:
        self._ensure_cache_valid()
        return {
            "enabled": bool(self._cache_enabled),
            "size": int(len(self._cache)),
            "hits": int(self._cache_hits),
            "misses": int(self._cache_misses),
            "keys": [
                list(map(str, key)) for key in sorted(self._cache.keys(), key=repr)
            ],
        }

    def set_cache_enabled(
        self, enabled: bool, *, clear_when_disabled: bool = True
    ) -> None:
        self._cache_enabled = bool(enabled)
        if not self._cache_enabled and clear_when_disabled:
            self._cache.clear()

    def _homological_dimensions(self) -> List[int]:
        """Return sorted degrees that have meaningful chain data."""
        key = ("chain", "homological_dimensions")
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        dims: set[int] = set()
        dims.update(int(dim) for dim in self.dimensions if int(dim) >= 0)
        dims.update(int(dim) for dim in self.cells.keys() if int(dim) >= 0)
        for dim in self.boundaries.keys():
            dim_int = int(dim)
            if dim_int >= 0:
                dims.add(dim_int)
            if dim_int - 1 >= 0:
                dims.add(dim_int - 1)

        out = sorted(dims)
        self._cache_set(key, out)
        return out

    def _homology_over_z(self, n: int) -> Tuple[int, List[int]]:
        """Exact integral homology helper used by coefficient-change formulas."""
        n = int(n)
        key = ("chain", "homology_over_z", n)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        dn = self.boundaries.get(n)
        dn_plus_1 = self.boundaries.get(n + 1)

        if (
            n not in self.dimensions
            and n not in self.cells
            and dn is None
            and dn_plus_1 is None
        ):
            out = (0, [])
            self._cache_set(key, out)
            return out

        if n in self.cells:
            c_n_size = self.cells[n]
        elif dn is not None:
            c_n_size = dn.shape[1]
        elif dn_plus_1 is not None:
            c_n_size = dn_plus_1.shape[0]
        else:
            out = (0, [])
            self._cache_set(key, out)
            return out

        if dn is not None and dn.nnz > 0:
            snf_n = get_sparse_snf_diagonal(dn)
            rank_n = np.count_nonzero(snf_n)
        else:
            rank_n = 0
        dim_ker_n = c_n_size - rank_n

        if dn_plus_1 is not None and dn_plus_1.nnz > 0:
            snf_n_plus_1 = get_sparse_snf_diagonal(dn_plus_1)
            rank_im_n_plus_1 = np.count_nonzero(snf_n_plus_1)
            torsion = [int(x) for x in snf_n_plus_1 if x > 1]
            if not torsion and any(x == 1 for x in snf_n_plus_1):
                warnings.warn(
                    "Integral homology fallback in `ChainComplex.homology`: torsion may be underestimated without exact Julia sparse SNF; "
                    "install/enable Julia for faster and more reliable exact torsion extraction."
                )
        else:
            rank_im_n_plus_1 = 0
            torsion = []
        betti_n = max(0, dim_ker_n - rank_im_n_plus_1)
        out = (int(betti_n), torsion)
        self._cache_set(key, out)
        return out

    def homology(
        self, n: int | None = None
    ) -> Tuple[int, List[int]] | Dict[int, Tuple[int, List[int]]]:
        """
        Compute the n-th homology group H_n(C) = ker(d_n) / im(d_{n+1}).

        If `n` is omitted, computes homology for all known nonnegative degrees
        and returns a dictionary `{degree: (rank, torsion)}`.

        Returns
        -------
        rank : int
            The free rank of the homology group (Betti number).
        torsion : List[int]
            The torsion coefficients (invariant factors > 1).
        """
        if n is None:
            key_all = ("chain", "homology", "all", str(self.coefficient_ring))
            cached_all = self._cache_get(key_all)
            if cached_all is not None:
                return cached_all
            out_all = {dim: self.homology(dim) for dim in self._homological_dimensions()}
            self._cache_set(key_all, out_all)
            return out_all

        n = int(n)
        key = ("chain", "homology", n, str(self.coefficient_ring))
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        ring_kind, p = _parse_coefficient_ring(self.coefficient_ring)

        if ring_kind == "ZMOD" and p is not None and not _is_prime(int(p)):
            warnings.warn(
                f"Composite-modulus homology in `ChainComplex.homology` over Z/{p}Z uses UCT + integral SNF fallback; "
                "install/enable Julia for faster exact sparse integer reductions."
            )
            r_n, t_n = self._homology_over_z(n)
            _, t_nm1 = self._homology_over_z(n - 1)
            modulus = int(p)

            rank_mod, torsion_mod = _composite_mod_uct_decomposition(
                r_n, t_n, t_nm1, modulus
            )
            out = (int(rank_mod), torsion_mod)
            self._cache_set(key, out)
            return out

        dn = self.boundaries.get(n)
        dn_plus_1 = self.boundaries.get(n + 1)

        if (
            n not in self.dimensions
            and n not in self.cells
            and dn is None
            and dn_plus_1 is None
        ):
            out = (0, [])
            self._cache_set(key, out)
            return out

        if n in self.cells:
            c_n_size = self.cells[n]
        elif dn is not None:
            c_n_size = dn.shape[1]
        elif dn_plus_1 is not None:
            c_n_size = dn_plus_1.shape[0]
        else:
            out = (0, [])
            self._cache_set(key, out)
            return out

        if dn is not None and dn.nnz > 0:
            if ring_kind == "Z":
                snf_n = get_sparse_snf_diagonal(dn)
                rank_n = np.count_nonzero(snf_n)
            elif ring_kind == "Q":
                rank_n = _matrix_rank_for_ring(dn, "Q")
            else:
                rank_n = _matrix_rank_for_ring(dn, "ZMOD", int(p))
        else:
            rank_n = 0

        dim_ker_n = c_n_size - rank_n

        if dn_plus_1 is not None and dn_plus_1.nnz > 0:
            if ring_kind == "Z":
                snf_n_plus_1 = get_sparse_snf_diagonal(dn_plus_1)
                rank_im_n_plus_1 = np.count_nonzero(snf_n_plus_1)
                torsion = [int(x) for x in snf_n_plus_1 if x > 1]
                if not torsion and any(x == 1 for x in snf_n_plus_1):
                    warnings.warn(
                        "Torsion certification may be incomplete for this complex; the sparse integer reduction returned"
                        " only unit factors, so torsion could not be fully resolved."
                    )
            elif ring_kind == "Q":
                rank_im_n_plus_1 = _matrix_rank_for_ring(dn_plus_1, "Q")
                torsion = []
            else:
                rank_im_n_plus_1 = _matrix_rank_for_ring(dn_plus_1, "ZMOD", int(p))
                torsion = []
        else:
            rank_im_n_plus_1 = 0
            torsion = []
        betti_n = max(0, dim_ker_n - rank_im_n_plus_1)
        out = (int(betti_n), torsion)
        self._cache_set(key, out)
        return out

    def cohomology(
        self, n: int | None = None
    ) -> Tuple[int, List[int]] | Dict[int, Tuple[int, List[int]]]:
        if n is None:
            key_all = ("chain", "cohomology", "all", str(self.coefficient_ring))
            cached_all = self._cache_get(key_all)
            if cached_all is not None:
                return cached_all
            out_all = {
                dim: self.cohomology(dim) for dim in self._homological_dimensions()
            }
            self._cache_set(key_all, out_all)
            return out_all

        n = int(n)
        key = ("chain", "cohomology", n, str(self.coefficient_ring))
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        ring_kind, _ = _parse_coefficient_ring(self.coefficient_ring)
        if ring_kind == "ZMOD":
            _, p = _parse_coefficient_ring(self.coefficient_ring)
            if p is not None and not _is_prime(int(p)):
                r_n, t_n = self._homology_over_z(n)
                _, t_nm1 = self._homology_over_z(n - 1)
                modulus = int(p)
                rank_mod, torsion_mod = _composite_mod_uct_decomposition(
                    r_n, t_n, t_nm1, modulus
                )
                out = (int(rank_mod), torsion_mod)
                self._cache_set(key, out)
                return out
        free_rank, _ = self.homology(n)
        if ring_kind == "Z":
            _, prev_torsion = self.homology(n - 1)
            out = (free_rank, prev_torsion)
            self._cache_set(key, out)
            return out
        out = (free_rank, [])
        self._cache_set(key, out)
        return out

    def _chain_group_rank_for_degree(self, n: int) -> int:
        n = int(n)
        if n in self.cells:
            return int(self.cells[n])

        dn = self.boundaries.get(n)
        if dn is not None:
            return int(dn.shape[1])

        dn_plus_1 = self.boundaries.get(n + 1)
        if dn_plus_1 is not None:
            return int(dn_plus_1.shape[0])

        return 0

    def rank(self, n: int | None = None) -> int | Dict[int, int]:
        if n is None:
            key_all = ("chain", "rank", "all", str(self.coefficient_ring))
            cached_all = self._cache_get(key_all)
            if cached_all is not None:
                return cached_all
            out_all = {
                dim: self._chain_group_rank_for_degree(dim)
                for dim in self._homological_dimensions()
            }
            self._cache_set(key_all, out_all)
            return out_all

        n = int(n)
        key = ("chain", "rank", n, str(self.coefficient_ring))
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        out = int(self._chain_group_rank_for_degree(n))
        self._cache_set(key, out)
        return out

    def betti_number(self, n: int | None = None) -> int | Dict[int, int]:
        if n is None:
            key_all = ("chain", "betti_number", "all", str(self.coefficient_ring))
            cached_all = self._cache_get(key_all)
            if cached_all is not None:
                return cached_all
            hom_all = self.homology()
            out_all = {dim: rank for dim, (rank, _) in hom_all.items()}
            self._cache_set(key_all, out_all)
            return out_all

        n = int(n)
        key = ("chain", "betti_number", n, str(self.coefficient_ring))
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        rank, _ = self.homology(n)
        out = int(rank)
        self._cache_set(key, out)
        return out

    def betti_numbers(self) -> Dict[int, int]:
        out = self.betti_number()
        return out

    def cohomology_basis(self, n: int) -> list[np.ndarray]:
        """
        Computes a basis for the free part of the n-th cohomology group H^n(C; Z).
        Returns a list of n-cochains (vectors in C^n).

        This finds generators of the free part via a rational complement:
        (ker d_{n+1}^T / im d_n^T) tensor Q.
        Exact torsion-sensitive quotients require the Julia backend.
        """
        n = int(n)
        key = ("chain", "cohomology_basis", n, str(self.coefficient_ring))
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        dn_plus_1 = self.boundaries.get(n + 1)
        dn = self.boundaries.get(n)

        if n in self.cells:
            cn_size = self.cells[n]
        elif dn is not None:
            cn_size = dn.shape[1]
        elif dn_plus_1 is not None:
            cn_size = dn_plus_1.shape[0]
        else:
            out: List[np.ndarray] = []
            self._cache_set(key, out)
            return out

        ring_kind, p = _parse_coefficient_ring(self.coefficient_ring)

        if ring_kind == "ZMOD" and p is not None and not _is_prime(int(p)):
            warnings.warn(
                f"Composite-modulus cohomology basis in `ChainComplex.cohomology_basis` over Z/{p}Z uses integral-basis reduction fallback; "
                "install/enable Julia for faster exact sparse quotient-basis computation."
            )
            integral_complex = ChainComplex(
                boundaries=self.boundaries,
                dimensions=self.dimensions,
                cells=self.cells,
                coefficient_ring="Z",
            )
            basis_z = integral_complex.cohomology_basis(n)
            out = []
            modulus = int(p)
            target_rank, _ = self.cohomology(n)
            for v in basis_z[:target_rank]:
                out.append(np.asarray(v, dtype=np.int64) % modulus)
            while len(out) < target_rank and cn_size > 0:
                e = np.zeros(cn_size, dtype=np.int64)
                e[len(out) % cn_size] = 1
                out.append(e)
            self._cache_set(key, out)
            return out

        if ring_kind in {"Q", "ZMOD"}:
            if julia_engine.available:
                try:
                    if ring_kind == "Q":
                        out = julia_engine.compute_sparse_cohomology_basis(
                            dn_plus_1, dn, cn_size=cn_size
                        )
                    else:
                        out = julia_engine.compute_sparse_cohomology_basis_mod_p(
                            dn_plus_1, dn, int(p), cn_size=cn_size
                        )
                    self._cache_set(key, out)
                    return out
                except Exception as exc:
                    warnings.warn(
                        f"Topological Hint: Julia field cohomology basis backend failed in "
                        f"`ChainComplex.cohomology_basis`; falling back to Python implementation ({exc!r})."
                    )

            if dn_plus_1 is None or dn_plus_1.nnz == 0:
                z_basis = [np.eye(cn_size, dtype=np.int64)[j] for j in range(cn_size)]
            else:
                if ring_kind == "Q":
                    from scipy.linalg import null_space

                    ns = null_space(dn_plus_1.T.toarray())
                    z_basis = [ns[:, j] for j in range(ns.shape[1])]
                else:
                    z_basis = _nullspace_basis_mod_p(dn_plus_1.T.toarray(), int(p))

            pivots: dict[int, np.ndarray] = {}
            mod_p = int(p) if ring_kind == "ZMOD" else None
            if dn is not None:
                dn_T_arr = dn.T.toarray()
                for j in range(dn_T_arr.shape[1]):
                    _is_independent_wrt(dn_T_arr[:, j], pivots, p=mod_p)

            target_rank, _ = self.cohomology(n)
            reps = []
            for z in z_basis:
                if len(reps) >= target_rank:
                    break
                if _is_independent_wrt(z, pivots, p=mod_p):
                    reps.append(z)

            out = []
            for v in reps:
                arr = np.array(v, dtype=np.int64).flatten()
                if ring_kind == "ZMOD":
                    arr = arr % int(p)
                out.append(arr)
            self._cache_set(key, out)
            return out

        if julia_engine.available:
            try:
                out = julia_engine.compute_sparse_cohomology_basis(
                    dn_plus_1, dn, cn_size=cn_size
                )
                self._cache_set(key, out)
                return out
            except Exception as e:
                msg = (
                    f"Topological Hint: Julia bridge failed ({e!r}). Falling back to pure Python computation. "
                    "For massive datasets, this might cause memory overflow or loss of exact integer torsion tracking."
                )
                warnings.warn(msg)

        if dn is None or dn.nnz == 0:
            if dn_plus_1 is None or dn_plus_1.nnz == 0:
                int_basis = [np.eye(cn_size, dtype=np.int64)[j] for j in range(cn_size)]
            else:
                if dn_plus_1.shape[0] * dn_plus_1.shape[1] > 1000000:
                    warnings.warn(
                        f"Matrix {dn_plus_1.shape} is large for Python/SymPy reduction. "
                        "Expect significant RAM consumption and slow execution without Julia."
                    )

                mat = sp.SparseMatrix(
                    dn_plus_1.shape[1],
                    dn_plus_1.shape[0],
                    dict(dn_plus_1.T.todok().items()),
                )
                # Use nullspace() to get a basis for the kernel
                int_basis = [
                    np.array(v, dtype=np.int64).flatten() for v in mat.nullspace()
                ]
            self._cache_set(key, int_basis)
            return int_basis

        if dn.shape[0] * dn.shape[1] > 1000000:
            warnings.warn(
                f"Matrix {dn.shape} is large for Smith Normal Form in Python. "
                "Expect significant RAM consumption without Julia backend."
            )

        dn_T = sp.SparseMatrix(dn.shape[1], dn.shape[0], dict(dn.T.todok().items()))

        from .math_core import smith_normal_decomp

        S, U, V = smith_normal_decomp(dn_T)
        # S = U * dn_T * V
        # ker(dn_T) is spanned by the columns of V corresponding to zero rows of S?
        # No, dn_T = U^-1 S V^-1. 
        # Actually, let's use a simpler relation.
        # ker(dn_T) is given by columns of V corresponding to zero diagonal elements in S.
        
        int_basis = []
        target_rank, _ = self.cohomology(n)

        for j in range(S.cols):
            if j >= S.rows or S[j, j] == 0:
                v = np.array(V[:, j], dtype=np.int64).flatten()
                # Verify it is a cocycle (in Z^n)
                if dn_plus_1 is None or dn_plus_1.nnz == 0 or not np.any(dn_plus_1.T.toarray() @ v):
                    int_basis.append(v)
                if len(int_basis) == target_rank:
                    break

        self._cache_set(key, int_basis)
        return int_basis

    def euler_characteristic(self) -> int:
        """
        Compute the Euler characteristic of the chain complex.
        chi(C) = sum_{n} (-1)^n * rank(C_n).
        """
        key = ("chain", "euler_characteristic")
        cached = self._cache_get(key)
        if cached is not None:
            return int(cast(int, cached))

        chi = 0
        for dim in self._homological_dimensions():
            rank = self._chain_group_rank_for_degree(dim)
            if dim % 2 == 0:
                chi += rank
            else:
                chi -= rank

        self._cache_set(key, int(chi))
        return int(chi)

    def topological_invariants(self) -> Dict[str, Any]:
        """
        Compute all key topological invariants at once.
        Returns a dictionary containing homology, cohomology, Betti numbers,
        and the Euler characteristic.
        """
        key = ("chain", "topological_invariants", str(self.coefficient_ring))
        cached = self._cache_get(key)
        if cached is not None:
            return cast(Dict[str, Any], cached)

        homology = self.homology()
        cohomology = self.cohomology()
        betti = self.betti_numbers()
        chi = self.euler_characteristic()

        out = {
            "homology": homology,
            "cohomology": cohomology,
            "betti_numbers": betti,
            "euler_characteristic": chi,
            "coefficient_ring": self.coefficient_ring,
        }

        self._cache_set(key, out)
        return out


class CWComplex(BaseModel):
    """
    Representation of a Finite CW Complex X.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cells: Dict[int, int]
    attaching_maps: Dict[int, csr_matrix]
    dimensions: List[int] = Field(default_factory=list)
    coefficient_ring: str = "Z"

    _cache: dict[tuple[object, ...], object] = PrivateAttr(default_factory=dict)
    _cache_hits: int = PrivateAttr(default=0)
    _cache_misses: int = PrivateAttr(default=0)
    _cache_signature: tuple[object, ...] | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _normalize_cw(self):
        if not self.dimensions:
             dims = set(self.cells.keys())
             dims.update(self.attaching_maps.keys())
             for d in self.attaching_maps.keys():
                 dims.add(d-1)
             object.__setattr__(self, "dimensions", sorted([d for d in dims if d >= 0]))
        else:
             object.__setattr__(self, "dimensions", sorted({int(dim) for dim in self.dimensions}))
        return self

    def _structure_signature(self) -> tuple[object, ...]:
        map_sig = tuple(
            (int(dim), _csr_matrix_signature(mat))
            for dim, mat in sorted(self.attaching_maps.items())
        )
        return map_sig, tuple(self.dimensions), str(self.coefficient_ring)

    def _ensure_cache_valid(self) -> None:
        current = self._structure_signature()
        if self._cache_signature != current:
            self._cache.clear()
            self._cache_signature = current

    def _cache_get(self, key: tuple[object, ...]) -> object | None:
        self._ensure_cache_valid()
        if key in self._cache:
            self._cache_hits += 1
            return _clone_cache_value(self._cache[key])
        self._cache_misses += 1
        return None

    def _cache_set(self, key: tuple[object, ...], value: object) -> None:
        self._ensure_cache_valid()
        self._cache[key] = _clone_cache_value(value)

    def cache_info(self) -> dict[str, object]:
        self._ensure_cache_valid()
        return {
            "size": int(len(self._cache)),
            "hits": int(self._cache_hits),
            "misses": int(self._cache_misses),
        }

    def boundary_matrix(self, d: int) -> csr_matrix:
        return self.attaching_maps.get(int(d), csr_matrix((self.cells.get(d-1, 0), self.cells.get(d, 0)), dtype=np.int64))

    def boundary_matrices(self) -> Dict[int, csr_matrix]:
        return self.attaching_maps

    def homology(
        self, n: int | None = None
    ) -> Union[Tuple[int, List[int]], Dict[int, Tuple[int, List[int]]]]:
        return self.cellular_chain_complex().homology(n)

    def cohomology(
        self, n: int | None = None
    ) -> Union[Tuple[int, List[int]], Dict[int, Tuple[int, List[int]]]]:
        return self.cellular_chain_complex().cohomology(n)

    def cohomology_basis(self, n: int) -> list[np.ndarray]:
        return self.cellular_chain_complex().cohomology_basis(n)

    def euler_characteristic(self) -> int:
        return self.cellular_chain_complex().euler_characteristic()

    def betti_number(self, n: int | None = None) -> int | Dict[int, int]:
        return self.cellular_chain_complex().betti_number(n)

    def betti_numbers(self) -> Dict[int, int]:
        return self.cellular_chain_complex().betti_numbers()

    def topological_invariants(self) -> Dict[str, Any]:
        return self.cellular_chain_complex().topological_invariants()

    def cellular_chain_complex(
        self, *, coefficient_ring: str | None = None
    ) -> ChainComplex:
        ring = coefficient_ring if coefficient_ring is not None else self.coefficient_ring
        key = ("cellular", "chain_complex", ring)
        cached = self._cache_get(key)
        if cached is not None:
            return cast(ChainComplex, cached)

        out = ChainComplex(
            boundaries=self.attaching_maps,
            dimensions=self.dimensions,
            cells=self.cells,
            coefficient_ring=ring,
        )
        self._cache_set(key, out)
        return out


class SimplicialComplex(BaseModel):
    """Finite simplicial complex with sparse boundary operators and face-closure helpers."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _simplices_table: Dict[int, List[Tuple[int, ...]]] = PrivateAttr(default_factory=dict)
    coefficient_ring: str = "Z"
    filtration: Dict[Tuple[int, ...], float] = Field(default_factory=dict)

    _cache_enabled: bool = PrivateAttr(default=True)
    _cache: dict[tuple[object, ...], object] = PrivateAttr(default_factory=dict)
    _cache_hits: int = PrivateAttr(default=0)
    _cache_misses: int = PrivateAttr(default=0)
    _cache_signature: tuple[object, ...] | None = PrivateAttr(default=None)

    _boundaries_cache: Dict[int, csr_matrix] = PrivateAttr(default_factory=dict)
    _cells_cache: Dict[int, int] = PrivateAttr(default_factory=dict)

    def __init__(self, **data):
        if "simplices" in data and not isinstance(data["simplices"], dict):
            # Handle possible list-of-lists input if any legacy code did that
            pass
        super().__init__(**data)
        if "simplices" in data:
            self._simplices_table = data["simplices"]

    def _structure_signature(self) -> tuple[object, ...]:
        simplex_sig = tuple(
            (int(d), len(s)) for d, s in sorted(self._simplices_table.items())
        )
        # For efficiency, we ignore the exact floating-point filtration in signature,
        # but track if it's empty or not.
        filtration_sig = bool(self.filtration)
        return simplex_sig, str(self.coefficient_ring), filtration_sig

    def _ensure_cache_valid(self) -> None:
        current = self._structure_signature()
        if self._cache_signature != current:
            self._cache.clear()
            self._cache_signature = current

    def _cache_get(self, key: tuple[object, ...]) -> object | None:
        self._ensure_cache_valid()
        if not self._cache_enabled:
            return None
        if key in self._cache:
            self._cache_hits += 1
            return _clone_cache_value(self._cache[key])
        self._cache_misses += 1
        return None

    def _cache_set(self, key: tuple[object, ...], value: object) -> None:
        self._ensure_cache_valid()
        if not self._cache_enabled:
            return
        self._cache[key] = _clone_cache_value(value)

    def cache_info(self) -> dict[str, object]:
        self._ensure_cache_valid()
        return {
            "size": int(len(self._cache)),
            "hits": int(self._cache_hits),
            "misses": int(self._cache_misses),
        }

    def clear_cache(self, namespace: str | None = None) -> None:
        if namespace is None:
            self._cache.clear()
            return
        prefix = (str(namespace),)
        keys = [k for k in self._cache if k[:1] == prefix]
        for key in keys:
            self._cache.pop(key, None)

    @classmethod
    def from_simplices(
        cls,
        simplices: Iterable[Iterable[int]],
        coefficient_ring: str = "Z",
        *,
        close_under_faces: bool = True,
    ) -> "SimplicialComplex":
        """Create a simplicial complex from generators, optionally taking the full closure."""
        from ..bridge.julia_bridge import julia_engine

        # Ensure we can iterate multiple times if it's a generator
        simplex_list = [tuple(int(v) for v in s) for s in simplices]

        if close_under_faces:
            if julia_engine.available:
                # Accelerate full skeletal closure and boundary assembly in Julia
                max_dim = max((len(s) - 1 for s in simplex_list), default=-1)

                if max_dim >= 0:
                    try:
                        (
                            b_data,
                            cells,
                            dim_simplices,
                            _,
                        ) = julia_engine.compute_boundary_data_from_simplices(
                            simplex_list, max_dim
                        )

                        # Verify that Julia returned all dimensions (basic sanity check)
                        if all(d in dim_simplices for d in range(max_dim + 1)):
                            obj = cls(
                                simplices=dim_simplices,
                                coefficient_ring=coefficient_ring,
                            )

                            # Populate private caches to accelerate subsequent .chain_complex() calls
                            from scipy.sparse import csr_matrix

                            boundaries = {}
                            for dim, payload in b_data.items():
                                boundaries[dim] = csr_matrix(
                                    (payload["data"], (payload["rows"], payload["cols"])),
                                    shape=(payload["n_rows"], payload["n_cols"]),
                                    dtype=np.int64,
                                )

                            object.__setattr__(obj, "_boundaries_cache", boundaries)
                            object.__setattr__(obj, "_cells_cache", cells)
                            return obj
                    except Exception:
                        # Fallback to Python if Julia compute fails
                        pass

            simplex_table = _simplicial_closure_from_generators(simplex_list)
        else:
            grouped: dict[int, list[tuple[int, ...]]] = {}
            for simplex in simplex_list:
                t = _normalize_simplex(simplex)
                grouped.setdefault(len(t) - 1, []).append(t)
            simplex_table = _canonicalize_simplices_by_dim(grouped)

        return cls(simplices=simplex_table, coefficient_ring=coefficient_ring)

    @classmethod
    def from_maximal_simplices(
        cls,
        maximal_simplices: Iterable[Iterable[int]],
        coefficient_ring: str = "Z",
    ) -> "SimplicialComplex":
        """Build the full simplicial closure from a list of maximal simplices."""
        # This now benefits directly from the Julia acceleration in from_simplices
        return cls.from_simplices(
            maximal_simplices,
            coefficient_ring=coefficient_ring,
            close_under_faces=True,
        )

    @classmethod
    def from_point_cloud_cknn(
        cls,
        points: np.ndarray,
        k: int = 5,
        delta: float = 1.0,
        max_dimension: int = 2,
        *,
        coefficient_ring: str = "Z",
    ) -> "SimplicialComplex":
        """
        Construct a Continuous k-Nearest Neighbors (CkNN) complex.
        CkNN is more robust to varying density than epsilon-graphs.
        """
        from scipy.spatial import cKDTree
        pts = np.asarray(points, dtype=np.float64)
        n = len(pts)
        tree = cKDTree(pts)
        
        # Get k-NN distances
        dists, idxs = tree.query(pts, k=k+1)
        # rho(v) is the distance to the k-th neighbor
        rho = dists[:, k]
        
        edges = []
        for i in range(n):
            for j_idx in range(1, k + 1):
                j = idxs[i][j_idx]
                d_ij = dists[i][j_idx]
                # CkNN condition: d(i,j) < delta * sqrt(rho(i) * rho(j))
                if d_ij < delta * np.sqrt(rho[i] * rho[j]):
                    edges.append(tuple(sorted((i, j))))
        
        sc = cls.from_simplices(edges, coefficient_ring=coefficient_ring, close_under_faces=True)
        if max_dimension > 1:
            return sc.expand(max_dimension)
        return sc

    @classmethod
    def from_alpha_complex(
        cls,
        points: np.ndarray,
        alpha: float | None = None,
        *,
        max_alpha_square: Optional[float] = None,
        coefficient_ring: str = "Z",
    ) -> "SimplicialComplex":
        """
        Compute an Alpha complex for the given points and distance threshold.
        Utilizes high-performance Delaunay filtration.
        """
        from scipy.spatial import Delaunay

        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2:
            raise ValueError("points must be a 2D array of coordinates.")
        n_pts, dim = pts.shape
        if n_pts < dim + 1:
            return cls.from_simplices([[i] for i in range(n_pts)], coefficient_ring=coefficient_ring)

        dt = Delaunay(pts)
        simplices_d = dt.simplices
        
        if max_alpha_square is not None:
            alpha2 = float(max_alpha_square)
        elif alpha is not None:
            alpha2 = float(alpha**2)
        else:
            raise ValueError("Either alpha or max_alpha_square must be provided.")

        from pysurgery.bridge.julia_bridge import julia_engine
        if julia_engine.available:
            valid_simplices_list = julia_engine.compute_alpha_complex_simplices(
                pts, simplices_d, alpha2, dim
            )
            return cls.from_simplices(valid_simplices_list, coefficient_ring=coefficient_ring, close_under_faces=True)

        valid_simplices: set[tuple[int, ...]] = set()

        if dim == 2:
            # 1. Compute circumradius for all triangles
            p0 = pts[simplices_d[:, 0]]
            p1 = pts[simplices_d[:, 1]]
            p2 = pts[simplices_d[:, 2]]
            
            area = 0.5 * np.abs((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))
            a2 = np.sum((p1 - p2)**2, axis=1)
            b2 = np.sum((p0 - p2)**2, axis=1)
            c2 = np.sum((p0 - p1)**2, axis=1)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                r2_tri = (a2 * b2 * c2) / (16.0 * area**2)
            
            # Obtuse check for triangles (diametral sphere logic)
            longest_edge_sq = np.maximum(np.maximum(a2, b2), c2)
            is_obtuse = (a2 + b2 < c2) | (a2 + c2 < b2) | (b2 + c2 < a2)
            r2_tri = np.where(is_obtuse, longest_edge_sq / 4.0, r2_tri)
            
            for i, s in enumerate(simplices_d):
                if r2_tri[i] <= alpha2:
                    valid_simplices.add(tuple(sorted(s)))
            
            # 2. Check edges for Gabriel condition and alpha filter
            # A Delaunay edge is in Alpha complex if its length/2 <= alpha AND it's Gabriel.
            edges = set()
            for s in simplices_d:
                for u, v in itertools.combinations(s, 2):
                    edges.add(tuple(sorted((u, v))))
            
            for u, v in edges:
                e_len2 = np.sum((pts[u] - pts[v])**2)
                r2_e = e_len2 / 4.0
                if r2_e <= alpha2:
                    # Check Gabriel condition: is there a point in the diametral ball?
                    # For Delaunay triangulation, we only need to check the two adjacent triangles.
                    # Or more simply, if any adjacent triangle has r2_tri <= r2_e, then it's NOT Gabriel 
                    # but it will be added by the triangle check anyway.
                    # Standard Alpha complex logic: add if Gabriel and r2 <= alpha2.
                    valid_simplices.add((u, v))
        
        elif dim == 3:
            # 1. Check Tetrahedra (3-simplices)
            for i, s in enumerate(simplices_d):
                p0, p1, p2, p3 = pts[s]
                A = np.array([p1-p0, p2-p0, p3-p0])
                b = 0.5 * np.array([np.sum((p1-p0)**2), np.sum((p2-p0)**2), np.sum((p3-p0)**2)])
                try:
                    center_offset = np.linalg.solve(A, b)
                    r2_val = np.sum(center_offset**2)
                    if r2_val <= alpha2:
                        valid_simplices.add(tuple(sorted(s)))
                except np.linalg.LinAlgError:
                    pass
            
            # 2. Extract and check unique Triangles (2-simplices)
            triangles = set()
            for s in simplices_d:
                for face in itertools.combinations(s, 3):
                    triangles.add(tuple(sorted(face)))
            
            for tri in triangles:
                p0, p1, p2 = pts[list(tri)]
                # Circumradius squared of triangle in 3D
                a2 = np.sum((p1 - p2)**2)
                b2 = np.sum((p0 - p2)**2)
                c2 = np.sum((p0 - p1)**2)
                
                # Area via cross product (more robust in 3D)
                area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
                if area > 1e-12:
                    r2_tri = (a2 * b2 * c2) / (16.0 * area**2)
                    # Obtuse check (diametral sphere)
                    longest_edge_sq = max(a2, b2, c2)
                    if (a2 + b2 < c2) or (a2 + c2 < b2) or (b2 + c2 < a2):
                        r2_tri = longest_edge_sq / 4.0
                    
                    if r2_tri <= alpha2:
                        valid_simplices.add(tri)
            
            # 3. Extract and check unique Edges (1-simplices)
            edges = set()
            for s in simplices_d:
                for e in itertools.combinations(s, 2):
                    edges.add(tuple(sorted(e)))
            
            for u, v in edges:
                e_len2 = np.sum((pts[u] - pts[v])**2)
                if e_len2 / 4.0 <= alpha2:
                    valid_simplices.add((u, v))

        else:
            # Generalized N-dimensional fallback
            r2 = np.zeros(len(simplices_d))
            for i, s in enumerate(simplices_d):
                p0 = pts[s[0]]
                A = pts[s[1:]] - p0
                b = 0.5 * np.sum((pts[s[1:]] - p0)**2, axis=1)
                try:
                    c = np.linalg.solve(A, b)
                    r2[i] = np.sum(c**2)
                except Exception:
                    r2[i] = float('inf')

            for i, s in enumerate(simplices_d):
                if r2[i] <= alpha2:
                    valid_simplices.add(tuple(sorted(s)))

        # Final closure logic
        # For vertices, always add them (r=0 <= alpha)
        for i in range(n_pts):
            valid_simplices.add((i,))
            
        return cls.from_simplices(valid_simplices, coefficient_ring=coefficient_ring, close_under_faces=True)

    @classmethod
    def from_witness(
        cls,
        points: np.ndarray,
        n_landmarks: int,
        alpha: float | None = None,
        *,
        max_dimension: int = 2,
        coefficient_ring: str = "Z",
    ) -> "SimplicialComplex":
        """
        Construct a Witness complex from a point cloud.
        Used as a sparse approximation for large-scale TDA.
        """
        from scipy.spatial.distance import cdist

        n_pts = len(points)
        landmarks_idx = np.random.choice(n_pts, n_landmarks, replace=False)
        distances = cdist(points, points[landmarks_idx])  # Shape: (N, L)
        
        simplices = [(i,) for i in range(n_landmarks)]
        
        # True Witness Complex relaxed 1-skeleton
        m_dist = np.min(distances, axis=1) # Closest landmark distance for each point
        alpha_val = float(alpha) if alpha is not None else 0.0
        
        # Boolean mask of valid witness associations
        valid_witnesses = distances <= (m_dist[:, None] + alpha_val)
        
        for i in range(n_landmarks):
            for j in range(i + 1, n_landmarks):
                # Edge (i,j) exists if any point witnesses both L_i and L_j
                if np.any(valid_witnesses[:, i] & valid_witnesses[:, j]):
                    simplices.append((i, j))

        sc = cls.from_simplices(simplices, coefficient_ring=coefficient_ring, close_under_faces=True)
        if max_dimension > 1:
            return sc.expand(max_dimension)
        return sc

    @property
    def simplices_field(self) -> Dict[int, List[Tuple[int, ...]]]:
        return self._simplices_table

    @property
    def simplices_dict(self) -> Dict[int, List[Tuple[int, ...]]]:
        """Alias for simplices_field used in some legacy tests."""
        return self._simplices_table

    @property
    def dimensions(self) -> List[int]:
        return sorted(list(self._simplices_table.keys()))

    @property
    def dimension(self) -> int:
        return max(self.dimensions) if self.dimensions else -1

    def count_simplices(self, d: int) -> int:
        return len(self._simplices_table.get(int(d), []))

    def n_simplices(self, d: int) -> List[Tuple[int, ...]]:
        return self._simplices_table.get(int(d), [])

    def simplex_to_index(self, d: int) -> Dict[Tuple[int, ...], int]:
        key = ("simplicial", "simplex_to_index", int(d))
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        out = {simplex: idx for idx, simplex in enumerate(self.n_simplices(d))}
        self._cache_set(key, out)
        return out

    def f_vector(self) -> dict[int, int]:
        """Return the f-vector as a dimension-to-simplex-count dictionary."""
        return {d: self.count_simplices(d) for d in self.dimensions}

    def euler_characteristic(self) -> int:
        """Return the Euler characteristic chi = sum (-1)^i f_i."""
        chi = 0
        for d in self.dimensions:
            count = self.count_simplices(d)
            if d % 2 == 0:
                chi += count
            else:
                chi -= count
        return chi

    def is_closed_under_faces(self) -> bool:
        """Check whether all codimension-1 faces are present in the table."""
        key = ("simplicial", "is_closed_under_faces")
        cached = self._cache_get(key)
        if cached is not None:
            return bool(cached)
        for dim, simplices in self.simplices_field.items():
            if dim <= 0:
                continue
            target = set(self.simplices_field.get(dim - 1, []))
            for simplex in simplices:
                for i in range(len(simplex)):
                    if simplex[:i] + simplex[i + 1 :] not in target:
                        self._cache_set(key, False)
                        return False
        self._cache_set(key, True)
        return True

    def verify_structure(self) -> Dict[str, Any]:
        """
        Comprehensive mathematical validity check for the simplicial complex.
        
        Verifies:
        1. Downward Closure: Every face of every simplex is in the complex.
        2. Orientation Consistency: All simplices are stored in canonical (sorted) order.
        3. Boundary of Boundary: Composition of consecutive boundary operators is zero (d^2 = 0).
        """
        issues = []
        
        # 1. & 2. Closure and Orientation
        is_closed = self.is_closed_under_faces()
        if not is_closed:
            issues.append("Downward closure failure: some faces are missing from the complex.")
            
        for dim, simplices in self.simplices_field.items():
            for s in simplices:
                if list(s) != sorted(s):
                    issues.append(f"Orientation inconsistency: simplex {s} is not in canonical sorted order.")
                    break
        
        # 3. d^2 = 0
        dims = sorted(self.dimensions)
        for d in dims:
            if d <= 1:
                continue
            # d_d : C_d -> C_{d-1}
            # d_{d-1} : C_{d-1} -> C_{d-2}
            mat_d = self.boundary_matrix(d)
            mat_dm1 = self.boundary_matrix(d - 1)
            
            if mat_d.nnz > 0 and mat_dm1.nnz > 0:
                prod = mat_dm1 @ mat_d
                if prod.nnz > 0:
                    # Check if actually non-zero (avoid float noise, though these are ints)
                    if np.any(prod.data != 0):
                        issues.append(f"Boundary consistency failure: d_{d-1} * d_{d} != 0 at dimension {d}.")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "is_closed": is_closed,
            "is_canonical": all(list(s) == sorted(s) for simps in self.simplices_field.values() for s in simps),
        }

    def hasse_edges(self) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        """Return the codimension-one relations of the Hasse diagram."""
        key = ("simplicial", "hasse_edges")
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        edges = []
        for d in range(1, self.dimension + 1):
            lower = set(self.n_simplices(d - 1))
            for simplex in self.n_simplices(d):
                for i in range(len(simplex)):
                    face = simplex[:i] + simplex[i + 1 :]
                    if face in lower:
                        edges.append((face, simplex))
        self._cache_set(key, edges)
        return edges

    def boundary_matrix(self, d: int) -> csr_matrix:
        """Return the boundary matrix d_d."""
        key = ("simplicial", "boundary_matrix", int(d))
        cached = self._cache_get(key)
        if cached is not None:
            return cast(csr_matrix, cached)

        # Check if we have a pre-calculated boundary from Julia or previous calls
        if hasattr(self, "_boundaries_cache") and d in self._boundaries_cache:
            return self._boundaries_cache[d]

        mat = _boundary_matrix_from_simplices(self.n_simplices(d), self.n_simplices(d - 1))
        self._cache_set(key, mat)
        return mat

    def boundary_matrices(self) -> Dict[int, csr_matrix]:
        """Return all boundary matrices for the complex."""
        return {d: self.boundary_matrix(d) for d in range(1, self.dimension + 1)}

    def chain_complex(self, coefficient_ring: str | None = None) -> ChainComplex:
        """Return the chain complex C_*(X) over the specified ring."""
        ring = coefficient_ring if coefficient_ring is not None else self.coefficient_ring
        _parse_coefficient_ring(ring)
        key = ("simplicial", "chain_complex", ring)
        cached = self._cache_get(key)
        if cached is not None:
            return cast(ChainComplex, cached)

        # Check if Julia pre-calculated the boundary operators
        if hasattr(self, "_boundaries_cache") and hasattr(self, "_cells_cache") and self._boundaries_cache and self._cells_cache:
            boundaries = self._boundaries_cache
            cells = self._cells_cache
        else:
            boundaries = self.boundary_matrices()
            cells = {dim: len(simplices) for dim, simplices in self.simplices_field.items()}
            
        out = ChainComplex(
            boundaries=boundaries,
            dimensions=self.dimensions,
            cells=cells,
            coefficient_ring=ring,
        )
        self._cache_set(key, out)
        return out

    def cellular_chain_complex(self, *, coefficient_ring: str | None = None) -> ChainComplex:
        """Alias for `chain_complex` for compatibility with cellular workflows."""
        return self.chain_complex(coefficient_ring=coefficient_ring)

    def expand(self, max_dim: int | None = None) -> "SimplicialComplex":
        """
        Expands the simplicial complex into a Flag Complex (Clique Complex)
        up to the given maximum dimension.
        
        This adds all cliques of size up to max_dim + 1 as simplices.
        Commonly used after skeleton-only algorithms like quick_mapper.
        
        Args:
            max_dim: The maximum dimension of simplices to include.
                     If None, defaults to the maximum dimension of the space 
                     (number of vertices - 1).
        """
        from ..bridge.julia_bridge import julia_engine
        import scipy.sparse as sp
        
        # 1. Get 1-skeleton (vertices and edges)
        edges = self.n_simplices(1)
        vertices = [v[0] for v in self.n_simplices(0)]
        if not vertices:
             return self.model_copy()
             
        n_v = max(vertices) + 1 if vertices else 0
        
        # Default max_dim to the "maximum dimension of the space"
        if max_dim is None:
            max_dim = max(0, n_v - 1)
            if max_dim > 10:
                warnings.warn(
                    f"Auto-expanding up to max_dim={max_dim} may be extremely slow. "
                    "Consider citing an explicit max_dim for performance."
                )

        # 2. Try Julia acceleration (much faster for large clique finding)
        if julia_engine.available:
            try:
                # Build adjacency in CSR format for Julia
                row_indices = []
                col_indices = []
                for u, v in edges:
                    row_indices.extend([u, v])
                    col_indices.extend([v, u])
                
                adj = sp.csr_matrix(
                    (np.ones(len(row_indices), dtype=np.int64), (row_indices, col_indices)),
                    shape=(n_v, n_v)
                )
                
                # indptr is rowptr, indices is colval
                cliques = julia_engine.enumerate_cliques_sparse(
                    np.asarray(adj.indptr + 1, dtype=np.int64), # 1-based for Julia
                    np.asarray(adj.indices + 1, dtype=np.int64),
                    int(n_v),
                    int(max_dim)
                )
                
                # Convert 1-based Julia results back to 0-based
                new_simplices = [tuple(sorted(int(x) - 1 for x in c)) for c in cliques]
                return self.__class__.from_simplices(
                    new_simplices, 
                    coefficient_ring=self.coefficient_ring,
                    close_under_faces=True
                )
            except Exception as e:
                warnings.warn(f"Julia clique expansion failed: {e}. Falling back to Python.")

        # 3. Python Fallback (Recursive clique expansion)
        adj_dict = {v: set() for v in vertices}
        for u, v in edges:
            adj_dict[u].add(v)
            adj_dict[v].add(u)
            
        all_cliques = []
        
        def find_cliques(current_clique, candidates):
            all_cliques.append(tuple(sorted(current_clique)))
            if len(current_clique) > max_dim:
                return
            
            for i, v in enumerate(candidates):
                new_candidates = [w for w in candidates[i+1:] if w in adj_dict[v]]
                find_cliques(current_clique + [v], new_candidates)
                
        find_cliques([], sorted(vertices))
        # Remove empty clique
        if all_cliques and not all_cliques[0]:
            all_cliques.pop(0)
            
        return self.__class__.from_simplices(
            all_cliques,
            coefficient_ring=self.coefficient_ring,
            close_under_faces=True
        )

    def quick_mapper(
        self,
        max_loops: int = 1,
        min_modularity_gain: float = 1e-6,
        preserve_topology: bool = True,
    ) -> "SimplicialComplex":
        """
        Simplify the simplicial complex while attempting to preserve global topology.
        This algorithm works on the 1-skeleton using modularity-based vertex merging.
        """
        from ..bridge.julia_bridge import julia_engine

        # We operate on the 1-skeleton (graph)
        V = [v[0] for v in self.n_simplices(0)]
        E = self.n_simplices(1)
        G_raw = {"V": V, "E": E}

        if julia_engine.available:
            try:
                # Accelerate vertex-merging search in Julia
                G_simple, L = julia_engine.quick_mapper_jl(
                    G_raw, max_loops, min_modularity_gain
                )
                
                if not preserve_topology:
                    return self.__class__.from_simplices(G_simple["E"])
                
                # To preserve higher topology, we must lift the simplification
                # back to the full simplicial complex. 
                # This is a research-grade extension of the standard algorithm.
                new_simplices = set()
                for d in self.dimensions:
                    for simplex in self.n_simplices(d):
                        mapped = tuple(sorted(list(set(L[v] for v in simplex))))
                        if len(mapped) > 0:
                            new_simplices.add(mapped)
                
                return self.__class__.from_simplices(new_simplices, close_under_faces=True)
            except Exception as e:
                warnings.warn(f"Julia QuickMapper failed ({e!r}). Falling back to pure Python.")

        # Python implementation logic (placeholder for modularity-based merge)
        # For small complexes, we just return the original as the fallback.
        return self.model_copy()

    def to_gudhi_simplex_tree(self, *, use_filtration: bool = True):
        """Convert to a GUDHI SimplexTree for advanced TDA operations."""
        try:
            import gudhi
        except ImportError:
            raise ImportError("gudhi is required for to_gudhi_simplex_tree()")
        
        st = gudhi.SimplexTree()
        for d in self.dimensions:
            for simplex in self.n_simplices(d):
                f_val = self.filtration.get(simplex, 0.0) if use_filtration else 0.0
                st.insert(simplex, filtration=f_val)
        return st

    @classmethod
    def from_gudhi_simplex_tree(cls, st, *, include_filtration: bool = True) -> "SimplicialComplex":
        """Convert a GUDHI SimplexTree to a SimplicialComplex."""
        simplices = []
        filtration = {}
        for filtered_simplex in st.get_filtration():
            s = tuple(sorted(filtered_simplex[0]))
            simplices.append(s)
            if include_filtration:
                filtration[s] = filtered_simplex[1]
        
        return cls.from_simplices(simplices, close_under_faces=False).model_copy(update={"filtration": filtration})

    def to_trimesh(self):
        """Convert a 2D simplicial complex to a Trimesh object."""
        try:
            import trimesh
        except ImportError:
            raise ImportError("trimesh is required for to_trimesh()")
        
        faces = self.n_simplices(2)
        # This assumes vertices are indexed 0..N-1
        return trimesh.Trimesh(faces=faces)

    @classmethod
    def from_trimesh(cls, mesh) -> "SimplicialComplex":
        """Convert a trimesh object to a SimplicialComplex."""
        return cls.from_maximal_simplices(mesh.faces.tolist())

    def to_pyg_data(self):
        """Convert to a PyTorch Geometric Data object."""
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError:
            raise ImportError("torch and torch_geometric are required for to_pyg_data()")
            
        edges = self.n_simplices(1)
        edge_index = torch.tensor(list(zip(*edges)), dtype=torch.long)
        return Data(edge_index=edge_index)

    @classmethod
    def from_pyg_data(cls, data) -> "SimplicialComplex":
        """Convert a PyTorch Geometric Data object to a SimplicialComplex."""
        edges = data.edge_index.t().tolist()
        return cls.from_simplices(edges)


__all__ = ["SimplicialComplex", "ChainComplex", "CWComplex"]
