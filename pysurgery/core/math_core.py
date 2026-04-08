import numpy as np
import numba
import sympy as sp
from sympy.matrices.normalforms import smith_normal_form as sympy_smith_normal_form

@numba.njit
def swap_rows(A, i, j):
    if i == j:
        return
    for k in range(A.shape[1]):
        tmp = A[i, k]
        A[i, k] = A[j, k]
        A[j, k] = tmp

@numba.njit
def swap_cols(A, i, j):
    if i == j:
        return
    for k in range(A.shape[0]):
        tmp = A[k, i]
        A[k, i] = A[k, j]
        A[k, j] = tmp

@numba.njit
def extended_gcd(a, b):
    x0, x1, y0, y1 = 1, 0, 0, 1
    while b != 0:
        q, r = divmod(a, b)
        a, b = b, r
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    return a, x0, y0

def smith_normal_form(A_in: np.ndarray) -> np.ndarray:
    """
    Compute the Smith Normal Form of an integer matrix A exactly.

    This is the default exact backend used for homology/torsion computations.
    Returns only the diagonal matrix S such that S = U*A*V for unimodular U,V.
    """
    A = sp.Matrix(np.asarray(A_in, dtype=np.int64))
    S = sympy_smith_normal_form(A, domain=sp.ZZ)
    return np.array(S.tolist(), dtype=np.int64)

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
    import scipy.sparse.linalg as spla
    
    m, n = A_sparse.shape
    
    if julia_engine.available:
        try:
            A_coo = A_sparse.tocoo()
            return julia_engine.compute_sparse_snf(A_coo.row, A_coo.col, A_coo.data, A_sparse.shape)
        except Exception as e:
            if allow_approx:
                msg = (
                    f"Topological Hint: Julia backend failed ({e!r}). Falling back to floating-point SVD for sparse SNF. "
                    "This estimates free ranks but misses exact Z-torsion."
                )
                warnings.warn(
                    msg
                )
            else:
                msg = (
                    f"Topological Hint: Julia backend failed ({e!r}). Falling back to exact Python/SymPy SNF (slower)."
                )
                warnings.warn(
                    msg
                )

    if not allow_approx:
        return get_snf_diagonal(A_sparse.toarray()) if hasattr(A_sparse, "toarray") else get_snf_diagonal(np.asarray(A_sparse))

    A_float = A_sparse.astype(float)
    min_dim = min(m, n)
    try:
        if min_dim <= 500:
            import scipy.linalg as la
            s = la.svdvals(A_float.toarray())
            tol = max(m, n) * np.finfo(float).eps * max(s) if len(s) > 0 else 1e-10
            rank = int(np.sum(s > tol))
            return np.ones(rank, dtype=np.int64)

        k_svd = min(min_dim - 1, 500)
        if k_svd <= 0:
            return np.array([], dtype=np.int64)

        u, s, vt = spla.svds(A_float, k=k_svd, which='LM')
        tol = max(m, n) * np.finfo(float).eps * max(s) if len(s) > 0 else 1e-10
        rank = int(np.sum(s > tol))
        if min_dim > k_svd:
            warnings.warn(
                "Topological Hint: Sparse SVD fallback sampled only part of the spectrum. "
                "Computed rank is a lower bound and torsion is unavailable without exact Julia SNF."
            )
        return np.ones(rank, dtype=np.int64)
    except Exception as e:
        msg = (
            f"Topological Hint: Sparse SVD failed to converge ({e!r}). "
            "The boundary matrix may be too degenerate. Assuming 0 free rank."
        )
        warnings.warn(msg)
        return np.array([], dtype=np.int64)
