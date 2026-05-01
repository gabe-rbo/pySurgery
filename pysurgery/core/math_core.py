import numpy as np
import numba
import sympy as sp


@numba.njit
def swap_rows(A, i, j):
    """In-place row swap.

    What is Being Computed?:
        Exchanges two rows in a matrix in-place.

    Algorithm:
        1. Iterates through each column index k.
        2. Swaps elements A[i, k] and A[j, k] using a temporary variable.

    Args:
        A: The matrix to modify.
        i: Index of the first row.
        j: Index of the second row.
    """
    if i == j:
        return
    for k in range(A.shape[1]):
        tmp = A[i, k]
        A[i, k] = A[j, k]
        A[j, k] = tmp

@numba.njit
def swap_cols(A, i, j):
    """In-place column swap.

    What is Being Computed?:
        Exchanges two columns in a matrix in-place.

    Algorithm:
        1. Iterates through each row index k.
        2. Swaps elements A[k, i] and A[k, j] using a temporary variable.

    Args:
        A: The matrix to modify.
        i: Index of the first column.
        j: Index of the second column.
    """
    if i == j:
        return
    for k in range(A.shape[0]):
        tmp = A[k, i]
        A[k, i] = A[k, j]
        A[k, j] = tmp

@numba.njit
def extended_gcd(a, b):
    """Return `(g, x, y)` such that `ax + by = g = gcd(a, b)`.

    What is Being Computed?:
        Computes the greatest common divisor (GCD) of two integers and the coefficients
        of Bézout's identity.

    Algorithm:
        1. Uses the extended Euclidean algorithm.
        2. Iteratively updates remainders and Bézout coefficients until the remainder is zero.

    Args:
        a: First integer.
        b: Second integer.

    Returns:
        Tuple[int, int, int]: (g, x, y) where g is the GCD, and ax + by = g.
    """
    x0, x1, y0, y1 = 1, 0, 0, 1
    while b != 0:
        q, r = divmod(a, b)
        a, b = b, r
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    return a, x0, y0


def smith_normal_decomp(A_in: sp.Matrix | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the Smith Normal Decomposition S = U*A*V for a matrix A.

    What is Being Computed?:
        Computes the Smith Normal Form (SNF) of an integer matrix A, along with the 
        unimodular transformation matrices U and V such that S = U * A * V.

    Algorithm:
        1. Converts input to a NumPy object array for arbitrary-precision integer arithmetic.
        2. Iteratively applies elementary row and column operations (swaps, additions, 
           scaling, and Euclidean reductions) to zero out off-diagonal elements.
        3. Uses the extended Euclidean algorithm to handle non-divisible pivot elements.
        4. Tracks all operations in matrices U and V.

    Preserved Invariants:
        - The diagonal elements of S (invariant factors) are unique up to sign.
        - The product of the first k diagonal elements is the GCD of all k×k minors of A.
        - Unimodular transformations preserve the underlying module structure (H_n calculations).

    Args:
        A_in: The input integer matrix (SymPy Matrix or NumPy array).

    Returns:
        tuple: (S, U, V) where S is the Smith Normal Form and U, V are unimodular matrices.

    Use When:
        - Computing homology groups over ℤ (need torsion and ranks)
        - Solving systems of linear Diophantine equations
        - Identifying the structure of finitely generated abelian groups

    Example:
        S, U, V = smith_normal_decomp(np.array([[2, 4], [4, 8]]))
        # S will be diag(2, 0)
    """
    # Use NumPy object arrays for speed while maintaining arbitrary precision
    if isinstance(A_in, sp.Matrix):
        A = np.array(A_in.tolist(), dtype=object)
    else:
        A = np.array(A_in, dtype=object)
        
    m, n = A.shape
    U = np.eye(m, dtype=object)
    V = np.eye(n, dtype=object)
    
    def pivot(A, U, V, k):
        # Optimized pivot: find first non-zero in row/col k, or then search
        if A[k, k] != 0:
            return True
            
        # Search row k
        for c in range(k + 1, n):
            if A[k, c] != 0:
                A[:, [k, c]] = A[:, [c, k]]
                V[:, [k, c]] = V[:, [c, k]]
                return True
        # Search col k
        for r in range(k + 1, m):
            if A[r, k] != 0:
                A[[k, r], :] = A[[r, k], :]
                U[[k, r], :] = U[[r, k], :]
                return True
                
        # Full submatrix search (rare fallback)
        for r in range(k + 1, m):
            for c in range(k + 1, n):
                if A[r, c] != 0:
                    A[[k, r], :] = A[[r, k], :]
                    U[[k, r], :] = U[[r, k], :]
                    A[:, [k, c]] = A[:, [c, k]]
                    V[:, [k, c]] = V[:, [c, k]]
                    return True
        return False

    for k in range(min(m, n)):
        if not pivot(A, U, V, k):
            break
            
        while True:
            changed = False
            # Clear row k
            for c in range(k + 1, n):
                if A[k, c] != 0:
                    if A[k, c] % A[k, k] != 0:
                        g, s, t = extended_gcd(int(A[k, k]), int(A[k, c]))
                        u, v = -A[k, c] // g, A[k, k] // g
                        
                        col_k = A[:, k].copy()
                        col_c = A[:, c].copy()
                        A[:, k] = s * col_k + t * col_c
                        A[:, c] = u * col_k + v * col_c
                        
                        V_k = V[:, k].copy()
                        V_c = V[:, c].copy()
                        V[:, k] = s * V_k + t * V_c
                        V[:, c] = u * V_k + v * V_c
                        changed = True
                    else:
                        q = A[k, c] // A[k, k]
                        A[:, c] -= q * A[:, k]
                        V[:, c] -= q * V[:, k]
            
            # Clear column k
            for r in range(k + 1, m):
                if A[r, k] != 0:
                    if A[r, k] % A[k, k] != 0:
                        g, s, t = extended_gcd(int(A[k, k]), int(A[r, k]))
                        u, v = -A[r, k] // g, A[k, k] // g
                        
                        row_k = A[k, :].copy()
                        row_r = A[r, :].copy()
                        A[k, :] = s * row_k + t * row_r
                        A[r, :] = u * row_k + v * row_r
                        
                        U_k = U[k, :].copy()
                        U_r = U[r, :].copy()
                        U[k, :] = s * U_k + t * U_r
                        U[r, :] = u * U_k + v * U_r
                        changed = True
                    else:
                        q = A[r, k] // A[k, k]
                        A[r, :] -= q * A[k, :]
                        U[r, :] -= q * U[k, :]
            
            if not changed:
                break
                
        if A[k, k] < 0:
            A[k, :] *= -1
            U[k, :] *= -1

    return A, U, V


def smith_normal_form(A_in: np.ndarray) -> np.ndarray:
    """Compute the Smith Normal Form of an integer matrix A exactly.

    What is Being Computed?:
        Extracts the Smith Normal Form S for an integer matrix.

    Algorithm:
        1. Delegates to `smith_normal_decomp` to perform the full reduction.
        2. Discards the transformation matrices U and V.

    Args:
        A_in: The input integer matrix as a numpy array.

    Returns:
        np.ndarray: The diagonal matrix S.

    Use When:
        - You only need the invariant factors, not the transformation basis
        - Checking equivalence of linear maps over ℤ

    Example:
        S = smith_normal_form(boundary_matrix)
    """
    S, _, _ = smith_normal_decomp(A_in)
    return S


def get_snf_diagonal(A: np.ndarray) -> np.ndarray:
    """Convenience wrapper to extract the invariant factors.

    What is Being Computed?:
        Extracts the non-zero invariant factors (the diagonal of the SNF) from a matrix.

    Algorithm:
        1. Computes the Smith Normal Form using `smith_normal_decomp`.
        2. Takes the absolute values of the diagonal elements.
        3. Filters out zeros to return only the meaningful invariant factors.

    Args:
        A: The input integer matrix.

    Returns:
        np.ndarray: A 1D array of non-zero invariant factors.

    Use When:
        - Directly computing Betti numbers and torsion coefficients
        - Standardized summary of a boundary operator's structure

    Example:
        factors = get_snf_diagonal(d2)
        # factors [1, 1, 2] means torsion is ℤ₂
    """
    S, _, _ = smith_normal_decomp(A)
    diag_len = min(S.shape)
    factors = np.zeros(diag_len, dtype=object)
    for i in range(diag_len):
        factors[i] = abs(S[i, i])
    
    non_zero = factors[factors != 0]
    try:
        return np.array(non_zero, dtype=np.int64)
    except (OverflowError, TypeError, ValueError):
        return np.array(non_zero, dtype=object)


def get_sparse_snf_diagonal(A_sparse, allow_approx: bool = False, backend: str = "auto") -> np.ndarray:
    """Computes the SNF diagonal for sparse matrices.

    What is Being Computed?:
        Computes the invariant factors of a sparse integer matrix, optimized for 
        topological boundary operators.

    Algorithm:
        1. If Julia is available and requested, uses a high-performance sparse engine 
           utilizing "leaf-peeling" pre-processors.
        2. If Python is used, warns about potential memory usage and converts to dense SNF.
        3. Strictly forbids floating-point fallbacks to ensure mathematical integrity.

    Preserved Invariants:
        - Exact integer torsion is maintained, which is critical for surgery obstructions (L-groups).

    Args:
        A_sparse: The input sparse matrix (e.g., scipy.sparse).
        allow_approx: Whether to allow approximate floating-point fallback (False).
        backend: 'auto', 'julia', or 'python'.

    Returns:
        np.ndarray: A 1D array of non-zero invariant factors.

    Use When:
        - Computing homology of large simplicial complexes
        - Speed is required via the Julia bridge
        - Exact torsion data is mandatory

    Example:
        torsion = get_sparse_snf_diagonal(sparse_boundary, backend='julia')
    """
    from ..bridge.julia_bridge import julia_engine
    import warnings

    # Normalize backend
    backend_norm = str(backend).lower().strip()
    use_julia = (backend_norm == "julia") or (backend_norm == "auto" and julia_engine.available)

    m, n = A_sparse.shape

    if use_julia:
        try:
            A_coo = A_sparse.tocoo()
            out = julia_engine.compute_sparse_snf(
                A_coo.row, A_coo.col, A_coo.data, A_sparse.shape
            )
            return out
        except Exception as e:
            if backend_norm == "julia":
                raise e
            msg = f"Topological Hint: Julia backend failed ({e!r}). Falling back to exact Python/SymPy SNF (slower)."
            warnings.warn(msg)

    if allow_approx:
        raise ValueError(
            "Mathematical Violation: `allow_approx=True` using floating-point SVD "
            "destroys exact integer torsion required for surgery obstructions. "
            "Use exact solvers or switch the complex to coefficient_ring='Q' if torsion data is not required."
        )

    if A_sparse.shape[0] * A_sparse.shape[1] > 10000000:
        warnings.warn(
            f"Large sparse matrix {A_sparse.shape} detected. Python fallback will use significant RAM (dense SNF). "
            "Install Julia backend for high-performance sparse exact algebra."
        )

    return (
        get_snf_diagonal(A_sparse.toarray())
        if hasattr(A_sparse, "toarray")
        else get_snf_diagonal(np.asarray(A_sparse))
    )
