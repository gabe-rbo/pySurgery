"""
pysurgery/algebra/exact_sequences.py

Mathematical framework for exact sequences of modules and diagrams.
Provides tools for automated exactness verification, element-level lifting,
and diagram chasing.
"""

from __future__ import annotations
import numpy as np
import sympy as sp
from typing import List, Tuple, Optional, Any, Union
from pysurgery.algebra.math_core import smith_normal_decomp


class Morphism:
    """Represents a homomorphism between finitely generated abelian groups (Z-modules).

    Attributes:
        matrix: The transformation matrix representing the map.
        domain_rank: The rank of the source module.
        codomain_rank: The rank of the target module.
    """

    def __init__(self, matrix: Union[np.ndarray, sp.Matrix], domain_rank: int, codomain_rank: int):
        if isinstance(matrix, sp.Matrix):
            self.matrix = np.array(matrix.tolist(), dtype=object)
        else:
            self.matrix = np.array(matrix, dtype=object)
        
        self.domain_rank = domain_rank
        self.codomain_rank = codomain_rank
        self._snf_cache = None

    @property
    def snf_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute and cache the Smith Normal Decomposition S = U * A * V."""
        if self._snf_cache is None:
            # smith_normal_decomp returns (S, U, V)
            self._snf_cache = smith_normal_decomp(self.matrix, compute_u=True, compute_v=True)
        return self._snf_cache

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply the morphism to an element (represented as a coordinate vector)."""
        x_arr = np.array(x, dtype=object)
        return self.matrix @ x_arr

    def kernel_basis(self) -> np.ndarray:
        """Compute an integral basis for the kernel of this morphism.
        
        Returns:
            A matrix whose columns form a basis for the kernel.
        """
        S, U, V = self.snf_data
        # Kernel elements correspond to columns of V where S[i,i] == 0
        diag = np.zeros(min(S.shape), dtype=object)
        for i in range(len(diag)):
            diag[i] = S[i, i]
        
        zero_indices = np.where(diag == 0)[0]
        # Also include any columns beyond the diagonal if it's a wide matrix
        extra_indices = np.arange(len(diag), V.shape[1])
        all_indices = np.concatenate([zero_indices, extra_indices]).astype(int)
        
        return V[:, all_indices]

    def image_basis(self) -> np.ndarray:
        """Compute an integral basis for the image of this morphism.
        
        Returns:
            A matrix whose columns form a basis for the image (in the codomain).
        """
        # The image is spanned by the columns of the matrix. 
        # We find a basis by computing the Smith Normal Form or just 
        # column-reducing to Hermite Normal Form.
        # A quick way to get a basis of the column space over Z:
        # Use smith_normal_decomp(A.T), then the non-zero columns of U_transpose * S
        # are a basis? No, easier:
        # If A = U^-1 S V^-1, then Im(A) = Im(U^-1 S).
        # Since S is diagonal, Im(U^-1 S) is spanned by { s_ii * (i-th column of U^-1) }.
        S, U, V = self.snf_data
        
        # We need U_inv. Since U is unimodular, we can compute it exactly.
        # For simplicity and robustness, we can use SymPy for the inverse of 
        # a unimodular matrix.
        U_inv = np.array(sp.Matrix(U).inv().tolist(), dtype=object)
        
        cols = []
        for i in range(min(S.shape)):
            s_ii = S[i, i]
            if s_ii != 0:
                cols.append(s_ii * U_inv[:, i])
        
        if not cols:
            return np.zeros((self.codomain_rank, 0), dtype=object)
            
        return np.column_stack(cols)

    def lift(self, y: np.ndarray) -> Optional[np.ndarray]:
        """Given y in the codomain, find x in the domain such that f(x) = y.
        
        Returns:
            x such that f(x) = y, or None if no such integral element exists.
        """
        S, U, V = self.snf_data
        # Solve A x = y  =>  U A V (V^-1 x) = U y  =>  S z = b' where z = V^-1 x, b' = U y
        b_prime = U @ np.array(y, dtype=object)
        z = np.zeros(self.domain_rank, dtype=object)
        
        for i in range(min(S.shape)):
            s_ii = S[i, i]
            if s_ii != 0:
                if b_prime[i] % s_ii != 0:
                    return None # No integral lift
                z[i] = b_prime[i] // s_ii
            else:
                if b_prime[i] != 0:
                    return None # Not in image
                z[i] = 0 # Arbitrary choice for kernel component
        
        # x = V z
        return V @ z

    def to_latex(self) -> str:
        """Return LaTeX representation of the matrix."""
        return sp.latex(sp.Matrix(self.matrix))


class ExactSequence:
    """Represents an exact sequence ... -> M_i -> M_{i+1} -> M_{i+2} -> ...
    
    A sequence is exact if Im(f_i) == Ker(f_{i+1}) for all i.
    """

    def __init__(self, modules: List[Any], morphisms: List[Morphism]):
        self.modules = modules
        self.morphisms = morphisms

    def verify_exactness(self, index: int) -> bool:
        """Verify exactness at the module modules[index+1].
        
        Checks if Im(morphisms[index]) == Ker(morphisms[index+1]).
        """
        if index < 0 or index >= len(self.morphisms) - 1:
            raise ValueError("Index out of bounds for exactness check.")
            
        f = self.morphisms[index]
        g = self.morphisms[index+1]
        
        # 1. Check g * f == 0 (Im f subset Ker g)
        gf = g.matrix @ f.matrix
        if np.any(gf != 0):
            return False
            
        # 2. Check Ker g subset Im f
        # We compare the ranks and the volumes (invariant factors) of the lattices.
        # Or more simply: try to lift every basis element of Ker g through f.
        ker_g = g.kernel_basis()
        for i in range(ker_g.shape[1]):
            vec = ker_g[:, i]
            if f.lift(vec) is None:
                return False
                
        return True

    def lift_element(self, element: np.ndarray, morphism_index: int) -> Optional[np.ndarray]:
        """Lift an element from the target of a morphism to its source.
        
        Only works if the element is in the image (which is guaranteed if 
        the sequence is exact and the element is in the kernel of the next map).
        """
        return self.morphisms[morphism_index].lift(element)


class ShortExactSequence(ExactSequence):
    """Represents a short exact sequence 0 -> A -> B -> C -> 0."""

    def __init__(self, A: Any, B: Any, C: Any, f: Morphism, g: Morphism):
        # We prepend and append zero maps implicitly or explicitly
        super().__init__([None, A, B, C, None], [None, f, g, None])
        self.A, self.B, self.C = A, B, C
        self.f, self.g = f, g

    def is_split(self) -> bool:
        """Check if the short exact sequence splits (B ~= A + C)."""
        # A sequence splits if there exists a section s: C -> B such that g * s = id_C
        # or a retraction r: B -> A such that r * f = id_A.
        # We can try to lift the basis of C to B.
        c_rank = self.g.codomain_rank
        # Assuming C is free for simplicity of 'splitting' check in this context.
        # If C has torsion, splitting is more complex (extension class in Ext).
        basis_C = np.eye(self.g.codomain_rank, dtype=object)
        section_cols = []
        for i in range(c_rank):
            lifted = self.g.lift(basis_C[:, i])
            if lifted is None:
                return False
            section_cols.append(lifted)
            
        # If we found integral lifts for all basis elements, it splits.
        return True
