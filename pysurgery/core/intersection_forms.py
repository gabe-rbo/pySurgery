import numpy as np
from scipy.linalg import eigvalsh
from pydantic import BaseModel, ConfigDict
import sympy as sp

from .exceptions import (
    NonSymmetricError,
    DimensionError,
    IsotropicError,
    NonPrimitiveError,
    UnimodularityError
)

class IntersectionForm(BaseModel):
    """
    Representation of a symmetric bilinear form Q on H_{2k}(M, Z).
    
    Attributes
    ----------
    matrix : np.ndarray
        The symmetric matrix of the intersection form.
    dimension : int
        The dimension of the manifold (n = 4k).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    matrix: np.ndarray
    dimension: int

    def __init__(self, **data):
        super().__init__(**data)
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise DimensionError("Intersection form matrix must be square.")
        if not np.allclose(self.matrix, self.matrix.T):
            raise NonSymmetricError("Intersection form matrix must be symmetric.")
        if self.dimension % 2 != 0:
            raise DimensionError("Intersection forms on H_{2k}(M) are usually defined for even-dimensional manifolds.")

    def _eigen_tol(self, eigenvalues: np.ndarray) -> float:
        if len(eigenvalues) == 0:
            return 1e-10
        scale = float(max(1.0, np.max(np.abs(eigenvalues))))
        return max(self.matrix.shape) * np.finfo(float).eps * scale

    def signature(self) -> int:
        """
        Compute the signature of the intersection form (rank+ - rank-).
        
        Returns
        -------
        sig : int
            The signature of the bilinear form.
        """
        # For a symmetric matrix over R, we use eigenvalues to find the signature.
        # This is valid for non-singular forms on M.
        eigenvalues = eigvalsh(self.matrix)
        tol = self._eigen_tol(eigenvalues)
        pos = np.sum(eigenvalues > tol)
        neg = np.sum(eigenvalues < -tol)
        return int(pos - neg)

    def is_even(self) -> bool:
        """
        Check if the form is even (Q(x, x) is even for all x).
        For an integral symmetric matrix, this is true if the diagonal elements are even.
        
        Returns
        -------
        even : bool
            True if the form is even, False if it is odd.
        """
        return all(int(self.matrix[i, i]) % 2 == 0 for i in range(self.matrix.shape[0]))

    def type(self) -> str:
        """
        Return the type of the form (I or II).
        Type II if even, Type I if odd.
        """
        return "II" if self.is_even() else "I"

    def rank(self) -> int:
        """Linear rank of the bilinear form (number of non-zero eigenvalues)."""
        eigenvalues = eigvalsh(self.matrix)
        tol = self._eigen_tol(eigenvalues)
        return int(np.sum(np.abs(eigenvalues) > tol))

    def is_indefinite(self) -> bool:
        """
        A form is indefinite if it has both positive and negative eigenvalues.
        For unimodular forms, this is equivalent to |signature| < rank.
        """
        return abs(self.signature()) < self.rank()

    def classify_z_form(self) -> dict:
        """
        Perform a basic classification of the unimodular form over Z.
        (Rank, Signature, Type).
        """
        return {
            "rank": self.rank(),
            "signature": self.signature(),
            "type": self.type(),
            "even": self.is_even()
        }

    def determinant(self) -> int:
        """
        Compute the determinant using exact arithmetic via SymPy if the matrix is integral.
        """
        sym_matrix = sp.Matrix(self.matrix.astype(int))
        return int(sym_matrix.det())

    def perform_algebraic_surgery(self, x: np.ndarray) -> 'IntersectionForm':
        """
        Perform algebraic surgery on the manifold by surgering out the isotropic class x.
        This corresponds to finding a class y with Q(x, y) = 1, and restricting the form to {x, y}^perp.
        Returns the new IntersectionForm of the surgered manifold.
        """
        x = np.asarray(x, dtype=int).flatten()
        if x.shape[0] != self.matrix.shape[0]:
            raise DimensionError(f"Surgery class 'x' must be a vector in the H_2 basis. "
                                 f"Expected size {self.matrix.shape[0]}, got {x.shape[0]}.")
            
        if np.dot(x, self.matrix @ x) != 0:
            raise IsotropicError(f"Surgery class 'x' must be isotropic (Q(x,x) = 0). Its self-intersection is {np.dot(x, self.matrix @ x)}. "
                                 "Topological translation: The normal bundle of the embedded sphere twists (like a Möbius strip), physically blocking the attachment of the surgery handle $D^3 \\times S^1$.")
            
        if np.gcd.reduce(x) != 1:
            raise NonPrimitiveError("Surgery class 'x' is not primitive (GCD of coordinates > 1). "
                                    "Topological translation: The class is a mathematical multiple of a basis element. Attempting surgery on it would create irremediable singularities in the resulting space.")
            
        x_TQ = x.T @ self.matrix
        
        # We need y such that x_TQ @ y = 1.
        def ext_gcd(a, b):
            if b == 0:
                return a, 1, 0
            x0, x1, y0, y1 = 1, 0, 0, 1
            while b != 0:
                q, a, b = a // b, b, a % b
                x0, x1 = x1, x0 - q * x1
                y0, y1 = y1, y0 - q * y1
            return a, x0, y0
            
        def ext_gcd_array(arr):
            if len(arr) == 1:
                return arr[0], [1]
            g, coeffs = ext_gcd_array(arr[:-1])
            g_final, s, t = ext_gcd(g, arr[-1])
            return g_final, [c * s for c in coeffs] + [t]

        g, y_list = ext_gcd_array(x_TQ.tolist())
        if g not in (1, -1):
            raise UnimodularityError("Intersection form is not unimodular (determinant != +/-1). "
                                     "Topological translation: The Extended Euclidean Algorithm failed to find a dual class 'y' where Q(x,y)=1. The space is not a closed manifold (Poincaré Duality has failed).")
        
        y = np.array(y_list, dtype=int) * g
        
        m = self.matrix.shape[0]
        y_TQ = y.T @ self.matrix
        constraints = sp.Matrix(np.vstack([x_TQ, y_TQ]))
        null_vecs = constraints.nullspace()

        basis_vectors = []
        for v in null_vecs:
            denoms = [sp.fraction(val)[1] for val in v]
            lcm_val = 1
            for d in denoms:
                lcm_val = int(np.lcm(lcm_val, int(d)))
            int_vec = np.array([int(sp.Integer(val * lcm_val)) for val in v], dtype=int)
            gcd_val = 0
            for a in int_vec.tolist():
                gcd_val = int(np.gcd(gcd_val, abs(a)))
            gcd_val = max(gcd_val, 1)
            basis_vectors.append((int_vec // gcd_val).tolist())

        basis_matrix = np.array(basis_vectors, dtype=int)
        
        if basis_matrix.shape[0] == 0:
            return self.__class__(matrix=np.zeros((0, 0), dtype=int), dimension=self.dimension)

        expected_rank = max(0, m - 2)
        if basis_matrix.shape[0] != expected_rank:
            # Keep only an independent set of the expected rank.
            reduced = []
            current = sp.Matrix.zeros(m, 0)
            for row in basis_matrix:
                v = sp.Matrix(row).reshape(m, 1)
                test = sp.Matrix.hstack(current, v)
                if test.rank() > current.rank():
                    reduced.append(row.tolist())
                    current = test
                if len(reduced) == expected_rank:
                    break
            basis_matrix = np.array(reduced, dtype=int)

        new_matrix = basis_matrix @ self.matrix @ basis_matrix.T
        
        return self.__class__(matrix=new_matrix, dimension=self.dimension)
