import numpy as np
import numba

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

@numba.njit
def smith_normal_form(A_in: np.ndarray) -> np.ndarray:
    """
    Compute the Smith Normal Form of an integer matrix A.
    Returns the diagonal matrix S such that S = UAV for unimodular U, V.
    (This implementation returns only S for homology/torsion purposes).
    """
    A = A_in.astype(np.int64).copy()
    rows, cols = A.shape
    curr_diag = 0
    
    while curr_diag < rows and curr_diag < cols:
        # 1. Pivot Selection: Find smallest non-zero element
        pivot_idx = (-1, -1)
        min_val = 0
        
        found = False
        for i in range(curr_diag, rows):
            for j in range(curr_diag, cols):
                if A[i, j] != 0:
                    val = abs(A[i, j])
                    if not found or val < min_val:
                        min_val = val
                        pivot_idx = (i, j)
                        found = True
        
        if not found:
            break
            
        # Move pivot to (curr_diag, curr_diag)
        swap_rows(A, curr_diag, pivot_idx[0])
        swap_cols(A, curr_diag, pivot_idx[1])
        
        # 2. Euclidean Reduction
        changed = True
        while changed:
            changed = False
            
            # Clear row
            for j in range(curr_diag + 1, cols):
                if A[curr_diag, j] != 0:
                    g, s, t = extended_gcd(A[curr_diag, curr_diag], A[curr_diag, j])
                    # Unimodular column transformation:
                    # [A_ii A_ij] * [s  -A_ij/g] = [g 0]
                    #              [t   A_ii/g]
                    u1, u2 = s, -(A[curr_diag, j] // g)
                    v1, v2 = t, (A[curr_diag, curr_diag] // g)
                    
                    for k in range(rows):
                        val_i = A[k, curr_diag]
                        val_j = A[k, j]
                        A[k, curr_diag] = val_i * u1 + val_j * v1
                        A[k, j] = val_i * u2 + val_j * v2
                    changed = True
            
            # Clear column
            for i in range(curr_diag + 1, rows):
                if A[i, curr_diag] != 0:
                    g, s, t = extended_gcd(A[curr_diag, curr_diag], A[i, curr_diag])
                    # Unimodular row transformation
                    u1, u2 = s, -(A[i, curr_diag] // g)
                    v1, v2 = t, (A[curr_diag, curr_diag] // g)
                    
                    for k in range(cols):
                        val_i = A[curr_diag, k]
                        val_curr = A[i, k]
                        A[curr_diag, k] = val_i * u1 + val_curr * v1
                        A[i, k] = val_i * u2 + val_curr * v2
                    changed = True
                    
        # Ensure A[curr_diag, curr_diag] divides all elements in the submatrix
        # (Divisibility condition of SNF)
        div_violation = False
        for i in range(curr_diag + 1, rows):
            for j in range(curr_diag + 1, cols):
                d = A[curr_diag, curr_diag]
                if d != 0 and A[i, j] % d != 0:
                    # Add row i to row curr_diag and restart reduction
                    for k in range(cols):
                        A[curr_diag, k] += A[i, k]
                    div_violation = True
                    break
            if div_violation:
                break
        
        if div_violation:
            continue
            
        curr_diag += 1
        
    return A

def get_snf_diagonal(A: np.ndarray) -> np.ndarray:
    """Convenience wrapper to extract the invariant factors."""
    S = smith_normal_form(A)
    diag_len = min(S.shape)
    factors = np.zeros(diag_len, dtype=np.int64)
    for i in range(diag_len):
        factors[i] = abs(S[i, i])
    return np.sort(factors)

def get_sparse_snf_diagonal(A_sparse) -> np.ndarray:
    """
    Computes the SNF diagonal for sparse matrices.
    Always uses the high-performance exact Julia Sparse SNF backend if available.
    If Julia is not installed, it seamlessly estimates the free rank using sparse iterative SVD.
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
            warnings.warn(f"Topological Hint: Julia backend failed ({e}). Falling back to floating-point SVD for sparse SNF. This will estimate free ranks but misses exact Z-torsion.")
            
    A_float = A_sparse.astype(float)
    k_svd = min(m - 1, n - 1, 500)
    
    if k_svd < min(m, n):
        import scipy.linalg as la
        s = la.svdvals(A_float.toarray())
        tol = max(m, n) * np.finfo(float).eps * max(s) if len(s) > 0 else 1e-10
        rank = np.sum(s > tol)
        return np.ones(rank, dtype=np.int64)
        
    try:
        u, s, vt = spla.svds(A_float, k=k_svd, which='LM')
        tol = max(m, n) * np.finfo(float).eps * max(s) if len(s) > 0 else 1e-10
        rank = np.sum(s > tol)
        return np.ones(rank, dtype=np.int64)
    except Exception as e:
        # If the SVD solver fails to converge on extremely degenerate or sparse matrices,
        # we return an empty array, assuming 0 free rank to prevent a total pipeline crash.
        # This signifies a mathematical failure to extract accurate topological invariants.
        warnings.warn(f"Topological Hint: Sparse SVD failed to converge ({e}). The boundary matrix may be too degenerate. Assuming 0 free rank.")
        return np.array([], dtype=np.int64)
