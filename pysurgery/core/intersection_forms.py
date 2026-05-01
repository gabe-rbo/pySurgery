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
    """Representation of a symmetric bilinear form Q on H_{2k}(M, Z).

    Overview:
        The IntersectionForm class encapsulates the intersection properties of 
        even-dimensional manifolds. For a 4k-dimensional manifold, the 
        intersection form is a symmetric bilinear form on H_{2k}(M; Z). It is 
        a fundamental invariant for the classification of manifolds, 
        particularly in dimension 4.

    Key Concepts:
        - **Symmetric Bilinear Form**: A map Q: V x V -> Z that is linear in 
          each variable and Q(x, y) = Q(y, x).
        - **Unimodularity**: The form is unimodular if the matrix has determinant +/- 1, 
          corresponding to Poincaré duality on a closed manifold.
        - **Signature**: The difference between the number of positive and 
          negative eigenvalues (rank+ - rank-).
        - **Even/Odd Type**: A form is even (Type II) if Q(x, x) is even for all x, 
          otherwise it is odd (Type I).

    Common Workflows:
        1. **Classification** → Use `classify_z_form()` to get rank, signature, and type.
        2. **Surgery** → Use `perform_algebraic_surgery(x)` to simulate the effect 
           of surgery on an isotropic class.
        3. **Invariants** → Compute `signature()` and `determinant()`.

    Coefficient Ring:
        - ℤ (Integers): The form is defined over the integers to capture the 
          full arithmetic of the intersection lattice.

    Attributes:
        matrix (np.ndarray): The symmetric matrix representing the form.
        dimension (int): The dimension of the manifold (n = 4k).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    matrix: np.ndarray
    dimension: int

    def __init__(self, **data):
        """Validate matrix shape/symmetry and manifold parity constraints.

        Args:
            **data: Field values for the model.

        Raises:
            DimensionError: If matrix is not square or dimension is not even.
            NonSymmetricError: If matrix is not symmetric.
        """
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
        """Return a scale-aware tolerance for eigenvalue sign/rank decisions.

        Args:
            eigenvalues (np.ndarray): The eigenvalues of the matrix.

        Returns:
            float: The computed tolerance.
        """
        if len(eigenvalues) == 0:
            return 1e-10
        scale = float(max(1.0, np.max(np.abs(eigenvalues))))
        return max(self.matrix.shape) * np.finfo(float).eps * scale

    def signature(self) -> int:
        """Compute the signature of the intersection form.

        What is Being Computed?:
            The signature σ(Q) = n₊ - n₋, where n₊ and n₋ are the number of 
            positive and negative eigenvalues of the bilinear form Q.

        Algorithm:
            1. Convert the matrix to a SymPy Matrix for exact arithmetic.
            2. Perform LDL decomposition to find the diagonal elements.
            3. Count positive and negative diagonal entries (Sylvester's Law of Inertia).
            4. Fallback to numerical eigvalsh if symbolic decomposition fails.

        Preserved Invariants:
            - Signature is a cobordism invariant for 4k-dimensional manifolds.
            - Signature is unchanged under stabilization (adding hyperbolic pairs).

        Returns:
            int: The signature σ(Q) of the bilinear form.

        Use When:
            - Classifying 4-manifolds via Freedman's theorem.
            - Checking for Hirzebruch Signature Theorem consistency.

        Example:
            Q = IntersectionForm(matrix=np.array([[1, 0], [0, -1]]), dimension=4)
            print(Q.signature())  # Output: 0
        """
        if self.matrix.size == 0:
            return 0
        
        # Exact calculation via Sylvester's Law of Inertia
        import sympy as sp
        M = sp.Matrix(self.matrix.tolist())
        # LDL decomposition provides the diagonal D over the field of fractions
        try:
            _, D = M.LDLdecomposition()
            pos = sum(1 for i in range(D.rows) if D[i, i] > 0)
            neg = sum(1 for i in range(D.rows) if D[i, i] < 0)
            return pos - neg
        except Exception:
            # Fallback for degenerate cases that LDL might not handle directly
            # by computing SNF or characteristic poly root isolation, 
            # but for non-degenerate surgery forms, LDL is perfect.
            eigenvalues = eigvalsh(self.matrix)
            tol = self._eigen_tol(eigenvalues)
            pos = np.sum(eigenvalues > tol)
            neg = np.sum(eigenvalues < -tol)
            return int(pos - neg)

    def approx_signature(self, temp: float = 10.0) -> float:
        """Compute a differentiable soft-approximation of the signature using JAX.

        Useful for optimization tasks and very large matrices.

        Args:
            temp (float): Temperature parameter for approximation. Defaults to 10.0.

        Returns:
            float: The approximated signature.
        """
        from ..integrations.jax_bridge import HAS_JAX
        if not HAS_JAX:
            # Fallback to a non-JAX approximation if needed, but here we enforce JAX for differentiability
            return float(self.signature())
        
        from ..integrations.jax_bridge import _approximate_signature
        return float(_approximate_signature(self.matrix, temp=temp))

    def is_even(self) -> bool:
        """Check if the form is even (Q(x, x) is even for all x).

        For an integral symmetric matrix, this is true if the diagonal elements are even.

        Returns:
            bool: True if the form is even, False if it is odd.
        """
        return all(int(self.matrix[i, i]) % 2 == 0 for i in range(self.matrix.shape[0]))

    def type(self) -> str:
        """Return the type of the form (I or II).

        Returns:
            str: "II" if even, "I" if odd.
        """
        return "II" if self.is_even() else "I"

    def rank(self) -> int:
        """Linear rank of the bilinear form (number of non-zero eigenvalues).

        Returns:
            int: The rank of the matrix.
        """
        eigenvalues = eigvalsh(self.matrix)
        tol = self._eigen_tol(eigenvalues)
        return int(np.sum(np.abs(eigenvalues) > tol))

    def is_indefinite(self) -> bool:
        """Check if the form is indefinite.

        A form is indefinite if it has both positive and negative eigenvalues.
        For unimodular forms, this is equivalent to |signature| < rank.

        Returns:
            bool: True if indefinite, False otherwise.
        """
        return abs(self.signature()) < self.rank()

    def classify_z_form(self) -> dict:
        """Perform a basic classification of the unimodular form over Z.

        Returns:
            dict: A dictionary containing "rank", "signature", "type", and "even".
        """
        return {
            "rank": self.rank(),
            "signature": self.signature(),
            "type": self.type(),
            "even": self.is_even(),
        }

    def determinant(self) -> int:
        """Compute the determinant using exact arithmetic via SymPy if the matrix is integral.

        Returns:
            int: The determinant of the matrix.
        """
        sym_matrix = sp.Matrix(self.matrix.astype(int))
        return int(sym_matrix.det())

    def perform_algebraic_surgery(self, x: np.ndarray) -> "IntersectionForm":
        """Perform algebraic surgery on the manifold by surgering out the isotropic class x.

        What is Being Computed?:
            The intersection form of the manifold resulting from surgery on the 
            homology class x. Algebraically, this is the restriction of the 
            form Q to the orthogonal complement of the subspace spanned by {x, y}, 
            where Q(x, y) = 1 and Q(x, x) = 0.

        Algorithm:
            1. Validate that x is isotropic (Q(x, x) = 0) and primitive.
            2. Find a dual class y such that Q(x, y) = 1 using the Extended Euclidean Algorithm.
            3. Construct the orthogonal complement H^perp = {v | Q(v, x) = 0 and Q(v, y) = 0}.
            4. Compute an exact ℤ-basis for H^perp using Hermite Normal Form (HNF).
            5. Project the original form onto this new basis.

        Preserved Invariants:
            - Signature is preserved (surgery on S^k in M^{2k} preserves signature if 2k=4).
            - The resulting form remains symmetric and (usually) unimodular if the input was.

        Args:
            x (np.ndarray): The isotropic class to surger out.

        Returns:
            IntersectionForm: The new IntersectionForm of the surgered manifold.

        Raises:
            DimensionError: If x has the wrong size.
            IsotropicError: If x is not isotropic.
            NonPrimitiveError: If x is zero or not primitive.
            UnimodularityError: If the dual class y cannot be found.

        Use When:
            - Modeling the effect of handle attachment on the intersection form.
            - Simplifying the intersection form of a 4-manifold towards a target form.
            - Implementing the algebraic steps of the surgery exact sequence.

        Example:
            # Surgery on a hyperbolic pair H = [[0, 1], [1, 0]] surgering x=[1, 0]
            Q = IntersectionForm(matrix=np.array([[0, 1], [1, 0]]), dimension=4)
            Q_new = Q.perform_algebraic_surgery(np.array([1, 0]))
            # Resulting matrix will be empty (0x0) as we surgered out the only pair.
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
