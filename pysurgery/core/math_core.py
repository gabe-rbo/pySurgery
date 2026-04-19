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
    import scipy.sparse.linalg as spla

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

    return (
        get_snf_diagonal(A_sparse.toarray())
        if hasattr(A_sparse, "toarray")
        else get_snf_diagonal(np.asarray(A_sparse))
    )
