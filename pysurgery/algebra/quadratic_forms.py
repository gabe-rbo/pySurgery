import numpy as np
from pydantic import ConfigDict
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from pysurgery.topology.complexes import SimplicialComplex

from pysurgery.algebra.intersection_forms import IntersectionForm
from pysurgery.core.exceptions import DimensionError


def arf_invariant_gf2(M: np.ndarray, q: np.ndarray) -> int:
    """Compute the Arf invariant of a quadratic form over GF(2).

    What is Being Computed?:
        The Arf invariant of a nondegenerate quadratic form over the field GF(2). 
        For a quadratic form Q on a vector space V with associated symplectic 
        bilinear form B, the Arf invariant is the value that Q takes most often, 
        or equivalently, Arf(Q) = Σ Q(e_i)Q(f_i) mod 2 for any symplectic 
        basis {e_i, f_i}.

    Algorithm:
        1. Validate that the bilinear form matrix M is square and nondegenerate over GF(2).
        2. Perform an optimized symplectic reduction in O(N^3) by direct Gram matrix updates.
        3. Iteratively find hyperbolic pairs (e_i, e_j) such that M[i, j] = 1.
        4. Accumulate the product of their quadratic refinements: arf = (arf + q(e_i) * q(e_j)) % 2.
        5. Orthogonalize the remaining basis with respect to the selected pair.
        6. Return the final accumulated value modulo 2.

    Preserved Invariants:
        - Arf invariant is a complete invariant of nondegenerate quadratic forms 
          over GF(2) up to isomorphism.
        - In surgery theory, it represents the obstruction to performing surgery 
          on a (4k+2)-dimensional manifold to obtain a homotopy sphere.

    Args:
        M: A square (n, n) bilinear form matrix over GF(2).
        q: A (n,) array of quadratic refinements for each basis element.

    Returns:
        int: The Arf invariant (0 or 1).

    Use When:
        - Computing L-groups L_{4k+2}(Z).
        - Analyzing surgery obstructions for manifolds of dimension 2, 6, 10, etc.
        - Classification of quadratic forms over finite fields.

    Example:
        arf = arf_invariant_gf2(np.array([[0, 1], [1, 0]]), np.array([1, 1])) # 1
    """
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
        sub = M[np.ix_(active, active)]
        # Find off-diagonal 1 in the active submatrix
        off = np.argwhere(np.triu(sub, k=1) == 1)
        if len(off) == 0:
            break

        loc_i, loc_j = off[0]
        i, j = active[loc_i], active[loc_j]

        # Found a hyperbolic pair (e_i, e_j)
        arf = (arf + q_vals[i] * q_vals[j]) % 2

        # Orthogonalize remaining basis by updating the Gram matrix and q_vals
        # Vectorized orthogonalization using outer product mod 2
        remaining = np.array([k for k in active if k not in (i, j)], dtype=np.int64)
        if remaining.size:
            w_kj = M[remaining, j].astype(np.int64)
            w_ki = M[remaining, i].astype(np.int64)

            # Update M[remaining, :] and then symm copy to M[:, remaining]
            # M[k, :] ^= w_kj * M[i, :] ^ w_ki * M[j, :]
            M[remaining, :] ^= (np.outer(w_kj, M[i, :]) % 2).astype(np.int64)
            M[remaining, :] ^= (np.outer(w_ki, M[j, :]) % 2).astype(np.int64)
            M[:, remaining] = M[remaining, :].T

            # Update q_vals for remaining basis
            q_vals[remaining] ^= (w_kj * q_vals[i] + w_ki * q_vals[j]) % 2

        active.pop(loc_j)
        active.pop(loc_i)

    return int(arf)


def _rank_mod_2(M: np.ndarray) -> int:
    """Compute matrix rank over the field GF(2).

    What is Being Computed?:
        The rank of a matrix where all operations are performed modulo 2.

    Algorithm:
        1. Convert input matrix to int64 and take modulo 2.
        2. Perform Gaussian elimination (row reduction) using XOR as addition.
        3. Count the number of pivot elements (linearly independent rows).

    Preserved Invariants:
        - Matrix rank over GF(2).

    Args:
        M: The matrix to compute the rank of.

    Returns:
        int: The rank of the matrix over GF(2).

    Use When:
        - Verifying nondegeneracy of intersection forms over Z/2Z.
        - Linear algebra tasks over the binary field.

    Example:
        r = _rank_mod_2(np.array([[1, 1], [1, 1]])) # 1
    """
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
    """A quadratic form on an abelian group, extending a bilinear intersection form.

    Overview:
        A QuadraticForm represents a quadratic refinement of a bilinear form, 
        typically the intersection form of a manifold. It is essential for 
        computing surgery obstructions in dimensions 4k+2, where the bilinear 
        form alone is insufficient.

    Key Concepts:
        - **Quadratic Refinement**: A map q: V → Z/2Z satisfying q(x+y) = q(x) + q(y) + B(x,y) mod 2.
        - **Arf Invariant**: The primary invariant of the quadratic form over Z/2Z.
        - **Surgery Obstruction**: The element in L_{4k+2}(G) that must vanish for surgery to succeed.

    Common Workflows:
        1. **Initialization** → Provide the bilinear matrix and the quadratic refinements.
        2. **Invariant Analysis** → Compute the arf_invariant().

    Coefficient Ring:
        - Operates over Z/2Z (GF(2)).

    Attributes:
        q_refinement (List[int]): The quadratic mapping q evaluated on the basis elements.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    q_refinement: List[int]

    def arf_invariant(self) -> int:
        """Compute the Arf invariant of the quadratic form.

        What is Being Computed?:
            The Arf invariant of the refinement q, derived from the underlying 
            bilinear matrix and the refinement values on the basis.

        Algorithm:
            Delegates to `arf_invariant_gf2` using the internal matrix and 
            q_refinement list.

        Preserved Invariants:
            - Arf invariant (0 or 1).

        Returns:
            int: The Arf invariant.

        Use When:
            - High-level API for computing Arf invariants from a QuadraticForm object.

        Example:
            qf = QuadraticForm(matrix=M, q_refinement=[1, 0, 1, 1])
            val = qf.arf_invariant()
        """
        return arf_invariant_gf2(
            self.matrix, np.array(self.q_refinement, dtype=np.int64)
        )

    @classmethod
    def from_complex(
        cls,
        K: "SimplicialComplex",
        *,
        backend: str = "auto",
    ) -> "QuadraticForm":
        """Quadratic refinement on H_{n/2}(K; F_2) for n ≡ 2 mod 4. Implements Gap G09 (Arf side)."""
        # First compute the underlying intersection form
        Q_form = IntersectionForm.from_complex(K, backend=backend)
        # By default, use all-0 refinements which is the standard choice without framing info
        q_ref = [0] * Q_form.matrix.shape[0]
        return cls(matrix=Q_form.matrix, dimension=K.dimension, q_refinement=q_ref)

