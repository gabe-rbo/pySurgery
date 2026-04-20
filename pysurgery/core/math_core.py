import numpy as np
import numba
import sympy as sp
from sympy.matrices.normalforms import smith_normal_form as sympy_smith_normal_form


@numba.njit
def swap_rows(A, i, j):
    """In-place row swap for integer elimination kernels."""
    if i == j:
        return
    for k in range(A.shape[1]):
        tmp = A[i, k]
        A[i, k] = A[j, k]
        A[j, k] = tmp


@numba.njit
def swap_cols(A, i, j):
    """In-place column swap for integer elimination kernels."""
    if i == j:
        return
    for k in range(A.shape[0]):
        tmp = A[k, i]
        A[k, i] = A[k, j]
        A[k, j] = tmp


@numba.njit
def extended_gcd(a, b):
    """Return `(g, x, y)` such that `ax + by = g = gcd(a, b)`."""
    x0, x1, y0, y1 = 1, 0, 0, 1
    while b != 0:
        q, r = divmod(a, b)
        a, b = b, r
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    return a, x0, y0


def smith_normal_decomp(A_in: sp.Matrix) -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    """
    Computes the Smith Normal Decomposition S = U*A*V for a SymPy Matrix A.
    Returns (S, U, V) where U, V are unimodular.
    
    This implementation handles the reduction iteratively over Z.
    """
    A = A_in.copy()
    m, n = A.rows, A.cols
    U = sp.eye(m)
    V = sp.eye(n)
    
    def pivot(A, U, V, k):
        # 1. Find smallest non-zero entry at or below/right of (k,k)
        best_val = -1
        best_pos = (-1, -1)
        for r in range(k, m):
            for c in range(k, n):
                val = abs(A[r, c])
                if val != 0 and (best_val == -1 or val < best_val):
                    best_val = val
                    best_pos = (r, c)
        
        if best_pos == (-1, -1):
            return False

        # Move to (k,k)
        r, c = best_pos
        if r != k:
            A.row_swap(k, r)
            U = U.elementary_row_op('n<->m', row1=k, row2=r)
        if c != k:
            A.col_swap(k, c)
            V = V.elementary_col_op('n<->m', col1=k, col2=c)
            
        return True

    for k in range(min(m, n)):
        if not pivot(A, U, V, k):
            break
            
        while True:
            # Clear row
            changed = False
            for c in range(k + 1, n):
                if A[k, c] % A[k, k] != 0:
                    g, s, t = extended_gcd(int(A[k, k]), int(A[k, c]))
                    # [k, k] = g, [k, c] = 0
                    # T = [[s, t], [-A[k,c]/g, A[k,k]/g]]
                    u, v = -A[k, c] // g, A[k, k] // g
                    
                    # Update A
                    col_k = A[:, k]
                    col_c = A[:, c]
                    A[:, k] = s * col_k + t * col_c
                    A[:, c] = u * col_k + v * col_c
                    
                    # Update V
                    V_k = V[:, k]
                    V_c = V[:, c]
                    V[:, k] = s * V_k + t * V_c
                    V[:, c] = u * V_k + v * V_c
                    
                    changed = True
                else:
                    q = A[k, c] // A[k, k]
                    A[:, c] -= q * A[:, k]
                    V[:, c] -= q * V[:, k]
            
            # Clear column
            for r in range(k + 1, m):
                if A[r, k] % A[k, k] != 0:
                    g, s, t = extended_gcd(int(A[k, k]), int(A[r, k]))
                    u, v = -A[r, k] // g, A[k, k] // g
                    
                    # Update A
                    row_k = A[k, :]
                    row_r = A[r, :]
                    A[k, :] = s * row_k + t * row_r
                    A[r, :] = u * row_k + v * row_r
                    
                    # Update U
                    U_k = U[k, :]
                    U_r = U[r, :]
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
    """
    Compute the Smith Normal Form of an integer matrix A exactly.

    This is the default exact backend used for homology/torsion computations.
    Returns only the diagonal matrix S such that S = U*A*V for unimodular U,V.
    """
    if A_in.dtype != object and not np.issubdtype(A_in.dtype, np.integer):
        A_in = np.asarray(A_in, dtype=object)

    A = sp.Matrix(A_in)
    S = sympy_smith_normal_form(A, domain=sp.ZZ)
    
    try:
        return np.array(S.tolist(), dtype=np.int64)
    except OverflowError:
        return np.array(S.tolist(), dtype=object)


def get_snf_diagonal(A: np.ndarray) -> np.ndarray:
    """Convenience wrapper to extract the invariant factors."""
    S = smith_normal_form(A)
    diag_len = min(S.shape)
    factors = np.zeros(diag_len, dtype=np.int64)
    for i in range(diag_len):
        factors[i] = abs(S[i, i])
    # Preserve the natural SNF order; sorting can destroy the invariant-factor structure.
    return factors[factors != 0]


def get_sparse_snf_diagonal(A_sparse, allow_approx: bool = False) -> np.ndarray:
    """
    Computes the SNF diagonal for sparse matrices.
    Uses exact integer SNF by default, with optional explicit approximate fallback.
    """
    from ..bridge.julia_bridge import julia_engine
    import warnings

    m, n = A_sparse.shape

    if julia_engine.available:
        try:
            A_coo = A_sparse.tocoo()
            return julia_engine.compute_sparse_snf(
                A_coo.row, A_coo.col, A_coo.data, A_sparse.shape
            )
        except Exception as e:
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
