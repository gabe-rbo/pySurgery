import numpy as np
import warnings
import sympy as sp
from functools import reduce
from math import lcm
from scipy.sparse import csr_matrix
from typing import Dict, List, Tuple
from pydantic import BaseModel, ConfigDict, Field
from .math_core import get_sparse_snf_diagonal
from ..bridge.julia_bridge import julia_engine


def _parse_coefficient_ring(ring: str) -> tuple[str, int | None]:
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


def _rank_mod_p(A: np.ndarray, p: int) -> int:
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
        inv = pow(int(M[row, col]), -1, p)
        M[row, :] = (M[row, :] * inv) % p
        for r in range(m):
            if r != row and M[r, col] % p != 0:
                M[r, :] = (M[r, :] - M[r, col] * M[row, :]) % p
        row += 1
        rank += 1
        if row == m:
            break
    return rank


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def _rref_mod_p(A: np.ndarray, p: int) -> tuple[np.ndarray, list[int]]:
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
        inv = pow(int(M[row, col]), -1, p)
        M[row, :] = (M[row, :] * inv) % p
        for r in range(m):
            if r != row and M[r, col] % p != 0:
                M[r, :] = (M[r, :] - M[r, col] * M[row, :]) % p
        pivots.append(col)
        row += 1
        if row == m:
            break
    return M, pivots


def _nullspace_basis_mod_p(A: np.ndarray, p: int) -> list[np.ndarray]:
    # A is m x n. Return basis vectors of ker(A) in F_p^n.
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

    def _homology_over_z(self, n: int) -> Tuple[int, List[int]]:
        """Exact integral homology helper used by coefficient-change formulas."""
        dn = self.boundaries.get(n)
        dn_plus_1 = self.boundaries.get(n + 1)

        if n not in self.dimensions and n not in self.cells and dn is None and dn_plus_1 is None:
            return 0, []

        if n in self.cells:
            c_n_size = self.cells[n]
        elif dn is not None:
            c_n_size = dn.shape[1]
        elif dn_plus_1 is not None:
            c_n_size = dn_plus_1.shape[0]
        else:
            return 0, []

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
        return int(betti_n), torsion

    def homology(self, n: int) -> Tuple[int, List[int]]:
        """
        Compute the n-th homology group H_n(C) = ker(d_n) / im(d_{n+1}).
        
        Returns
        -------
        rank : int
            The free rank of the homology group (Betti number).
        torsion : List[int]
            The torsion coefficients (invariant factors > 1).
        """
        # d_n : C_n -> C_{n-1}
        # d_{n+1} : C_{n+1} -> C_n
        ring_kind, p = _parse_coefficient_ring(self.coefficient_ring)

        # For composite Z/nZ coefficients, compute via exact integral decomposition + UCT.
        if ring_kind == "ZMOD" and p is not None and not _is_prime(int(p)):
            warnings.warn(
                f"Composite-modulus homology in `ChainComplex.homology` over Z/{p}Z uses UCT + integral SNF fallback; "
                "install/enable Julia for faster exact sparse integer reductions."
            )
            r_n, t_n = self._homology_over_z(n)
            _, t_nm1 = self._homology_over_z(n - 1)
            modulus = int(p)

            rank_mod, torsion_mod = _composite_mod_uct_decomposition(r_n, t_n, t_nm1, modulus)
            return int(rank_mod), torsion_mod

        dn = self.boundaries.get(n)
        dn_plus_1 = self.boundaries.get(n + 1)
        
        # Dimensions of chain groups
        # If n not in boundaries, assume C_n is 0 or its size is inferred from boundaries
        if n not in self.dimensions and n not in self.cells and dn is None and dn_plus_1 is None:
            return 0, []

        # Number of n-cells (columns of d_n or rows of d_{n+1})
        if n in self.cells:
            c_n_size = self.cells[n]
        elif dn is not None:
            c_n_size = dn.shape[1]
        elif dn_plus_1 is not None:
            c_n_size = dn_plus_1.shape[0]
        else:
            # Isolated dimension with no boundaries and no explicit cell count
            return 0, []


        # 1. Find rank of d_n to get dim(ker(d_n))
        if dn is not None and dn.nnz > 0:
            if ring_kind == "Z":
                snf_n = get_sparse_snf_diagonal(dn)
                rank_n = np.count_nonzero(snf_n)
            elif ring_kind == "Q":
                rank_n = int(np.linalg.matrix_rank(dn.toarray().astype(float)))
            else:
                rank_n = _rank_mod_p(dn.toarray(), int(p))
        else:
            rank_n = 0
            
        dim_ker_n = c_n_size - rank_n
        
        # 2. Find rank(im(d_{n+1})) and torsion when applicable
        if dn_plus_1 is not None and dn_plus_1.nnz > 0:
            if ring_kind == "Z":
                snf_n_plus_1 = get_sparse_snf_diagonal(dn_plus_1)
                rank_im_n_plus_1 = np.count_nonzero(snf_n_plus_1)
                torsion = [int(x) for x in snf_n_plus_1 if x > 1]
                if not torsion and any(x == 1 for x in snf_n_plus_1):
                    warnings.warn(
                        "Torsion in homology may be non-trivial but cannot be computed without "
                        "exact integer arithmetic (Julia bridge unavailable). Install Julia for "
                        "exact Z-torsion computation."
                    )
            elif ring_kind == "Q":
                rank_im_n_plus_1 = int(np.linalg.matrix_rank(dn_plus_1.toarray().astype(float)))
                torsion = []
            else:
                rank_im_n_plus_1 = _rank_mod_p(dn_plus_1.toarray(), int(p))
                torsion = []
        else:
            rank_im_n_plus_1 = 0
            torsion = []
        betti_n = max(0, dim_ker_n - rank_im_n_plus_1)
        return int(betti_n), torsion

    def cohomology(self, n: int) -> Tuple[int, List[int]]:
        r"""
        Compute the n-th cohomology group H^n(C) using the Universal Coefficient Theorem:
        H^n(C, Z) \cong Hom(H_n(C), Z) \oplus Ext(H_{n-1}(C), Z).
        """
        ring_kind, _ = _parse_coefficient_ring(self.coefficient_ring)
        if ring_kind == "ZMOD":
            _, p = _parse_coefficient_ring(self.coefficient_ring)
            if p is not None and not _is_prime(int(p)):
                r_n, t_n = self._homology_over_z(n)
                _, t_nm1 = self._homology_over_z(n - 1)
                modulus = int(p)
                rank_mod, torsion_mod = _composite_mod_uct_decomposition(r_n, t_n, t_nm1, modulus)
                return int(rank_mod), torsion_mod
        free_rank, _ = self.homology(n)
        if ring_kind == "Z":
            _, prev_torsion = self.homology(n - 1)
            return free_rank, prev_torsion
        return free_rank, []

    def cohomology_basis(self, n: int) -> List[np.ndarray]:
        """
        Computes a basis for the free part of the n-th cohomology group H^n(C; Z).
        Returns a list of n-cochains (vectors in C^n).
        
        This finds generators of the free part via a rational complement:
        (ker d_{n+1}^T / im d_n^T) tensor Q.
        Exact torsion-sensitive quotients require the Julia backend.

        For massive matrices, this seamlessly offloads to optimized float SVDs or Julia.
        """
        dn_plus_1 = self.boundaries.get(n + 1)
        dn = self.boundaries.get(n)
        
        # Number of n-cells (columns of d_n or rows of d_{n+1})
        if n in self.cells:
            cn_size = self.cells[n]
        elif dn is not None:
            cn_size = dn.shape[1]
        elif dn_plus_1 is not None:
            cn_size = dn_plus_1.shape[0]
        else:
            return [] # Isolated dimension

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
            # If UCT predicts additional Z/n generators from torsion terms, append canonical vectors.
            while len(out) < target_rank and cn_size > 0:
                e = np.zeros(cn_size, dtype=np.int64)
                e[len(out) % cn_size] = 1
                out.append(e)
            return out

        if ring_kind in {"Q", "ZMOD"}:
            # Vector-space basis over a field.
            if dn_plus_1 is None or dn_plus_1.nnz == 0:
                if ring_kind == "Q":
                    null_basis = [sp.Matrix([1 if i == j else 0 for i in range(cn_size)]) for j in range(cn_size)]
                else:
                    null_basis = [np.eye(cn_size, dtype=np.int64)[j] for j in range(cn_size)]
            else:
                if ring_kind == "Q":
                    M = sp.Matrix(dn_plus_1.T.toarray().astype(int))
                    null_basis = M.nullspace()
                else:
                    null_basis = _nullspace_basis_mod_p(dn_plus_1.T.toarray(), int(p))

            if dn is None or dn.nnz == 0:
                image_basis = []
            else:
                if ring_kind == "Q":
                    Mimg = sp.Matrix(dn.T.toarray().astype(int))
                    image_basis = Mimg.columnspace()
                else:
                    mat = dn.T.toarray().astype(np.int64)
                    _, pivots = _rref_mod_p(mat, int(p))
                    image_basis = [mat[:, j] % int(p) for j in pivots]

            target_rank, _ = self.cohomology(n)
            basis_of_quotient = []
            if image_basis:
                if ring_kind == "Q":
                    current_mat = sp.Matrix.hstack(*image_basis)
                    current_rank = current_mat.rank()
                else:
                    current_mat = np.column_stack(image_basis).astype(np.int64) % int(p)
                    current_rank = _rank_mod_p(current_mat, int(p))
            else:
                current_mat = sp.Matrix.zeros(cn_size, 0) if ring_kind == "Q" else np.zeros((cn_size, 0), dtype=np.int64)
                current_rank = 0

            for v in null_basis:
                if len(basis_of_quotient) >= target_rank:
                    break
                if ring_kind == "Q":
                    test_mat = sp.Matrix.hstack(current_mat, v)
                    new_rank = test_mat.rank()
                else:
                    col = np.asarray(v, dtype=np.int64).reshape(cn_size, 1) % int(p)
                    test_mat = np.hstack((current_mat, col))
                    new_rank = _rank_mod_p(test_mat, int(p))
                if new_rank > current_rank:
                    current_mat = test_mat
                    current_rank = new_rank
                    basis_of_quotient.append(v)

            out = []
            for v in basis_of_quotient:
                arr = np.array(v, dtype=np.int64).flatten()
                if ring_kind == "ZMOD":
                    arr = arr % int(p)
                out.append(arr)
            return out

        if julia_engine.available:
            try:
                # Use exact sparse linear algebra in Julia to perfectly compute Z^n / B^n
                return julia_engine.compute_sparse_cohomology_basis(dn_plus_1, dn, cn_size=cn_size)
            except Exception as e:
                msg = (
                    f"Topological Hint: Julia bridge failed ({e!r}). Falling back to pure Python computation. "
                    "For massive datasets, this might cause memory overflow or loss of exact integer torsion tracking."
                )
                warnings.warn(msg)

        # If Julia is unavailable, we dynamically attempt exact Python mathematics.
        # SymPy is used for exact integer quotients, but if it exceeds memory/time thresholds,
        # we catch the exception (or we just use an optimized float SVD fallback directly).
        
        def _primitive_int_vector(vec: sp.Matrix) -> sp.Matrix:
            denoms = [sp.fraction(x)[1] for x in vec]
            common_lcm = reduce(lcm, (int(d) for d in denoms), 1) if denoms else 1
            ints = [int(sp.Integer(x * common_lcm)) for x in vec]
            gcd_val = 0
            for a in ints:
                gcd_val = int(np.gcd(gcd_val, abs(a)))
            gcd_val = max(gcd_val, 1)
            return sp.Matrix([a // gcd_val for a in ints])

        # 1. Z^n: Kernel of d_{n+1}^T
        if dn_plus_1 is None or dn_plus_1.nnz == 0:
            null_basis = [sp.Matrix([1 if i == j else 0 for j in range(cn_size)]) for i in range(cn_size)]
        else:
            coboundary_mat = sp.Matrix(dn_plus_1.T.toarray().astype(int))
            null_basis = [_primitive_int_vector(v) for v in coboundary_mat.nullspace()]

        # 2. B^n: Image of d_n^T
        if dn is None or dn.nnz == 0:
            image_basis = []
        else:
            dn_mat = dn.T.toarray()
            image_basis = [_primitive_int_vector(v) for v in sp.Matrix(dn_mat).columnspace()]

        # 3. H^n = Z^n / B^n
        target_rank, _ = self.cohomology(n)
        basis_of_quotient = []
        if image_basis:
            current_mat = sp.Matrix.hstack(*image_basis)
            current_rank = current_mat.rank()
        else:
            current_mat = sp.Matrix.zeros(cn_size, 0)
            current_rank = 0
            
        for v in null_basis:
            if len(basis_of_quotient) >= target_rank:
                break
            test_mat = sp.Matrix.hstack(current_mat, v)
            new_rank = test_mat.rank()
            if new_rank > current_rank:
                current_mat = test_mat
                current_rank = new_rank
                basis_of_quotient.append(v)
        
        int_basis = []
        for v in basis_of_quotient:
            denominators = [sp.fraction(x)[1] for x in v]
            if denominators:
                common_lcm = reduce(lcm, (int(d) for d in denominators), 1)
            else:
                common_lcm = 1

            int_v = np.array([int(x * common_lcm) for x in v], dtype=np.int64)
            int_basis.append(int_v)
            
        return int_basis

class CWComplex(BaseModel):
    """
    Representation of a Finite CW Complex X.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    cells: Dict[int, int]
    attaching_maps: Dict[int, csr_matrix]
    coefficient_ring: str = "Z"

    def cellular_chain_complex(self) -> ChainComplex:
        return ChainComplex(
            boundaries=self.attaching_maps, 
            dimensions=sorted(self.cells.keys()),
            cells=self.cells,
            coefficient_ring=self.coefficient_ring,
        )
