import numpy as np
from pydantic import ConfigDict
from typing import List
from .intersection_forms import IntersectionForm
from .exceptions import DimensionError


def arf_invariant_gf2(M: np.ndarray, q: np.ndarray) -> int:
    """Compute Arf invariant for a GF(2) bilinear form matrix M and refinement vector q."""
    M = np.asarray(M, dtype=np.int64) % 2
    q_vals = np.asarray(q, dtype=np.int64).flatten() % 2

    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise DimensionError("M must be a square matrix over GF(2).")
    n = M.shape[0]
    if len(q_vals) != n:
        raise DimensionError(f"q must have length {n}, got {len(q_vals)}.")
    if n % 2 != 0:
        raise DimensionError("Arf invariant requires even-dimensional ambient space.")

    # Arf invariant is classically defined for nondegenerate quadratic spaces.
    # Here we enforce nondegeneracy of the associated bilinear form matrix M,
    # which is the standard computational assumption for these APIs.
    if _rank_mod_2(M) != n:
        raise DimensionError("Arf invariant is undefined for degenerate GF(2) bilinear forms.")

    basis = np.eye(n, dtype=np.int64)
    active_indices = list(range(n))
    arf = 0

    def eval_q(vec: np.ndarray) -> int:
        lin = np.sum(vec * q_vals)
        cross = 0
        for i in range(n):
            for j in range(i + 1, n):
                cross += vec[i] * vec[j] * M[i, j]
        return int((lin + cross) % 2)

    while len(active_indices) >= 2:
        found = False
        for i_idx, i in enumerate(active_indices):
            for j in active_indices[i_idx + 1 :]:
                if int((basis[i] @ M @ basis[j]) % 2) == 1:
                    e_idx, f_idx = i, j
                    found = True
                    break
            if found:
                break

        if not found:
            break

        e = basis[e_idx]
        f = basis[f_idx]
        arf = (arf + eval_q(e) * eval_q(f)) % 2

        new_active = []
        for k in active_indices:
            if k == e_idx or k == f_idx:
                continue
            v = basis[k]
            v_dot_f = int((v @ M @ f) % 2)
            v_dot_e = int((v @ M @ e) % 2)
            basis[k] = (v - v_dot_f * e - v_dot_e * f) % 2
            new_active.append(k)
        active_indices = new_active

    return int(arf)


def _rank_mod_2(M: np.ndarray) -> int:
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
        return arf_invariant_gf2(self.matrix, np.array(self.q_refinement, dtype=np.int64))

