import numpy as np
from scipy.linalg import eigvalsh
from typing import Tuple, List, Optional
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
        if not np.allclose(self.matrix, self.matrix.T):
            raise NonSymmetricError("Intersection form matrix must be symmetric.")
        if self.dimension % 4 != 0:
            raise DimensionError("Intersection forms on H_{2k}(M) are usually defined for 4k-dimensional manifolds.")

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
        pos = np.sum(eigenvalues > 1e-10)
        neg = np.sum(eigenvalues < -1e-10)
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
        """
        Compute the rank of the form.
        """
        return self.matrix.shape[0]

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
            raise DimensionError(f"Vector x must have size {self.matrix.shape[0]}")
            
        if np.dot(x, self.matrix @ x) != 0:
            raise IsotropicError("Class x is not isotropic (self-intersection is not 0).")
            
        if np.gcd.reduce(x) != 1:
            raise NonPrimitiveError("Class x is not primitive.")
            
        x_TQ = x.T @ self.matrix
        
        # We need y such that x_TQ @ y = 1.
        def ext_gcd(a, b):
            if b == 0: return a, 1, 0
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
            raise UnimodularityError("Form is not unimodular; cannot find dual class y.")
        
        y = np.array(y_list, dtype=int) * g
        
        # Now P(v) = v - [Q(v, y) - Q(y, y) Q(v, x)] x - Q(v, x) y
        # Projects v onto H^perp = {x, y}^perp.
        q_yy = np.dot(y, self.matrix @ y)
        
        # Form matrix P where rows are P(e_i)
        m = self.matrix.shape[0]
        P_cols = []
        for i in range(m):
            v = np.zeros(m, dtype=int)
            v[i] = 1
            q_vy = np.dot(v, self.matrix @ y)
            q_vx = np.dot(v, self.matrix @ x)
            proj_v = v - (q_vy - q_yy * q_vx) * x - q_vx * y
            P_cols.append(proj_v)
            
        # To get a Z-basis for the column space, we apply Hermite Normal Form
        # to the rows (P_cols elements are rows here).
        from sympy.matrices.normalforms import hermite_normal_form
        H = hermite_normal_form(sp.Matrix(P_cols))
        
        # The non-zero rows of H form a basis for H^perp.
        basis_rows = []
        for i in range(H.rows):
            if not all(H[i, j] == 0 for j in range(H.cols)):
                basis_rows.append([int(H[i, j]) for j in range(H.cols)])
                
        basis_matrix = np.array(basis_rows, dtype=int)
        
        if basis_matrix.shape[0] == 0:
            return self.__class__(matrix=np.zeros((0, 0), dtype=int), dimension=self.dimension)
            
        new_matrix = basis_matrix @ self.matrix @ basis_matrix.T
        
        return self.__class__(matrix=new_matrix, dimension=self.dimension)
