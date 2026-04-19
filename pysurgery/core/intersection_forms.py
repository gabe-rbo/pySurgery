import numpy as np
from scipy.linalg import eigvalsh
from pydantic import BaseModel, ConfigDict
import sympy as sp

from .exceptions import (
    NonSymmetricError,
    DimensionError,
    IsotropicError,
    NonPrimitiveError,
    UnimodularityError,
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
        """Validate matrix shape/symmetry and manifold parity constraints."""
        super().__init__(**data)
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise DimensionError("Intersection form matrix must be square.")
        if not np.allclose(self.matrix, self.matrix.T):
            raise NonSymmetricError("Intersection form matrix must be symmetric.")
        if self.dimension % 2 != 0:
            raise DimensionError(
                "Intersection forms on H_{2k}(M) are usually defined for even-dimensional manifolds."
            )

    def _eigen_tol(self, eigenvalues: np.ndarray) -> float:
        """Return a scale-aware tolerance for eigenvalue sign/rank decisions."""
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

    def approx_signature(self, temp: float = 10.0) -> float:
        """
        Compute a differentiable soft-approximation of the signature using JAX.
        Useful for optimization tasks and very large matrices.
        """
        from ..integrations.jax_bridge import HAS_JAX
        if not HAS_JAX:
            # Fallback to a non-JAX approximation if needed, but here we enforce JAX for differentiability
            return float(self.signature())
        
        from ..integrations.jax_bridge import _approximate_signature
        return float(_approximate_signature(self.matrix, temp=temp))

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
            "even": self.is_even(),
        }

    def determinant(self) -> int:
        """
        Compute the determinant using exact arithmetic via SymPy if the matrix is integral.
        """
        sym_matrix = sp.Matrix(self.matrix.astype(int))
        return int(sym_matrix.det())

    def perform_algebraic_surgery(self, x: np.ndarray) -> "IntersectionForm":
        """
        Perform algebraic surgery on the manifold by surgering out the isotropic class x.
        This corresponds to finding a class y with Q(x, y) = 1, and restricting the form to {x, y}^perp.
        Returns the new IntersectionForm of the surgered manifold.
        """
        x = np.asarray(x, dtype=int).flatten()
        if x.shape[0] != self.matrix.shape[0]:
            raise DimensionError(
                f"Surgery class 'x' must be a vector in the H_2 basis. "
                f"Expected size {self.matrix.shape[0]}, got {x.shape[0]}."
            )

        if np.dot(x, self.matrix @ x) != 0:
            raise IsotropicError(
                f"Surgery class 'x' must be isotropic (Q(x,x) = 0). Its self-intersection is {np.dot(x, self.matrix @ x)}. "
                "Topological translation: The normal bundle of the embedded sphere twists (like a Möbius strip), physically blocking the attachment of the surgery handle $D^3 \\times S^1$."
            )

        nonzero_components = x[x != 0]
        if len(nonzero_components) == 0:
            raise NonPrimitiveError(
                "Surgery class 'x' is zero. Cannot perform surgery on the zero class. "
                "Topological translation: The zero class has no geometric surgery representative."
            )
        if int(np.gcd.reduce(nonzero_components.astype(np.int64))) != 1:
            raise NonPrimitiveError(
                "Surgery class 'x' is not primitive (GCD of non-zero coordinates > 1). "
                "Topological translation: The class is a mathematical multiple of a basis element. Attempting surgery on it would create irremediable singularities in the resulting space."
            )

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
            if not arr:
                return 0, []
            n = len(arr)
            if n == 1:
                return arr[0], [1]
            
            curr_g = arr[0]
            s_vals = []
            t_vals = []
            
            for i in range(1, n):
                curr_g, s, t = ext_gcd(curr_g, arr[i])
                s_vals.append(s)
                t_vals.append(t)
            
            coeffs = [0] * n
            multiplier = 1
            for i in range(n - 1, 0, -1):
                coeffs[i] = t_vals[i-1] * multiplier
                multiplier *= s_vals[i-1]
            coeffs[0] = multiplier
            return curr_g, coeffs

        g, y_list = ext_gcd_array(x_TQ.tolist())
        if g not in (1, -1):
            raise UnimodularityError(
                "Intersection form is not unimodular (determinant != +/-1). "
                "Topological translation: The Extended Euclidean Algorithm failed to find a dual class 'y' where Q(x,y)=1. The space is not a closed manifold (Poincaré Duality has failed)."
            )

        y = np.array(y_list, dtype=int) * g

        m = self.matrix.shape[0]
        
        # Exact Integral Projection onto H^perp
        # P(v) = v - (v^T Q y - (v^T Q x)(y^T Q y)) x - (v^T Q x) y
        # We apply P to the identity matrix (standard basis of Z^n)
        identity_mat = np.eye(m, dtype=int)
        y_TQ = y.T @ self.matrix
        y_TQ_y = int(np.dot(y_TQ, y))
        
        # Vectorized projection of all standard basis vectors
        v_TQ_y = identity_mat @ y_TQ.T
        v_TQ_x = identity_mat @ x_TQ.T
        
        c1 = v_TQ_y - v_TQ_x * y_TQ_y
        c2 = v_TQ_x
        
        # P is an m x m matrix whose rows span H^perp exactly over Z
        P = identity_mat - np.outer(c1, x) - np.outer(c2, y)
        
        # Extract a Z-basis of the row space using SymPy's Hermite Normal Form
        # HNF guarantees an exact Z-basis for the lattice generated by the rows.
        P_sym = sp.Matrix(P.tolist())
        from sympy.matrices.normalforms import hermite_normal_form
        H = hermite_normal_form(P_sym.T)
        
        # The non-zero columns of H (transposed back to rows) form the exact Z-basis
        basis_vectors = []
        for i in range(H.shape[1]):
            col = np.array(H[:, i]).astype(int).flatten()
            if np.any(col):
                basis_vectors.append(col)
                
        basis_matrix = np.array(basis_vectors, dtype=int)
        
        if basis_matrix.shape[0] == 0:
            return self.__class__(
                matrix=np.zeros((0, 0), dtype=int), dimension=self.dimension
            )

        new_matrix = basis_matrix @ self.matrix @ basis_matrix.T

        return self.__class__(matrix=new_matrix, dimension=self.dimension)
