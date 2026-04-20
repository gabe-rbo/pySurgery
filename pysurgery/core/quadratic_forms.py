import numpy as np
from pydantic import ConfigDict
from typing import List
from .intersection_forms import IntersectionForm
from .exceptions import DimensionError


def arf_invariant_gf2(M: np.ndarray, q: np.ndarray) -> int:
    """Optimized symplectic reduction in O(N^3) by direct Gram matrix updates."""
    M = (np.asarray(M, dtype=np.int64) % 2).copy()
    q_vals = (np.asarray(q, dtype=np.int64).flatten() % 2).copy()

    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise DimensionError("M must be a square matrix over GF(2).")
    n = M.shape[0]
    if len(q_vals) != n:
        raise DimensionError(f"q must have length {n}, got {len(q_vals)}.")
    
    if n == 0:
        return 0

    # Arf invariant is classically defined for nondegenerate quadratic spaces.
    # Here we enforce nondegeneracy of the associated bilinear form matrix M,
    # which is the standard computational assumption for these APIs.
    if _rank_mod_2(M) != n:
        raise DimensionError(
            "Arf invariant is undefined for degenerate GF(2) bilinear forms."
        )

    arf = 0
    active = list(range(n))
    
    while len(active) >= 2:
        found = False
        for idx_i, i in enumerate(active):
            for idx_j, j in enumerate(active):
                if i == j or M[i, j] == 0:
                    continue
                
                # Found a hyperbolic pair (e_i, e_j)
                arf = (arf + q_vals[i] * q_vals[j]) % 2
                
                # Orthogonalize remaining basis by updating the Gram matrix directly
                # v_k = v_k + B(v_k, e_j)e_i + B(v_k, e_i)e_j
                for k in active:
                    if k == i or k == j:
                        continue
                    
                    w_kj = M[k, j]
                    w_ki = M[k, i]
                    
                    if w_kj:
                        M[k, :] = (M[k, :] + M[i, :]) % 2
                        M[:, k] = (M[:, k] + M[:, i]) % 2
                        q_vals[k] = (q_vals[k] + q_vals[i]) % 2
                        
                    if w_ki:
                        M[k, :] = (M[k, :] + M[j, :]) % 2
                        M[:, k] = (M[:, k] + M[:, j]) % 2
                        q_vals[k] = (q_vals[k] + q_vals[j]) % 2
                
                active.pop(max(idx_i, idx_j))
                active.pop(min(idx_i, idx_j))
                found = True
                break
            if found:
                break
        if not found:
            break
            
    return int(arf)


def _rank_mod_2(M: np.ndarray) -> int:
    """Compute matrix rank over GF(2)."""
    A = (np.asarray(M, dtype=np.int64) % 2).copy()
    m, n = A.shape
    row = 0
    rank = 0
    for col in range(n):
        pivot = None
        for r in range(row, m):
            if A[r, col] == 1:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != row:
            A[[row, pivot]] = A[[pivot, row]]
        for r in range(m):
            if r != row and A[r, col] == 1:
                A[r, :] = (A[r, :] + A[row, :]) % 2
        row += 1
        rank += 1
        if row == m:
            break
    return rank


class QuadraticForm(IntersectionForm):
    """
    A quadratic form on an abelian group, which is a refinement of a symmetric bilinear form.
    Specifically, this models the Z/2Z refinements required for L_{4k+2} surgery obstructions
    and the computation of the Arf invariant.

    Attributes
    ----------
    q_refinement : List[int]
        The quadratic mapping q: H -> Z_2 evaluated on the basis elements.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    q_refinement: List[int]

    def arf_invariant(self) -> int:
        """
        Compute the Arf invariant of the quadratic form.
        For a symplectic basis (e_i, f_i) where q(e_i)=a_i and q(f_i)=b_i,
        Arf(q) = sum(a_i * b_i) mod 2.

        This assumes the underlying intersection form matrix represents a symplectic basis.
        """
        return arf_invariant_gf2(
            self.matrix, np.array(self.q_refinement, dtype=np.int64)
        )
