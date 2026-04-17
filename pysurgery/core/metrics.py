import numpy as np
import scipy.spatial.distance as dist
from typing import Tuple, Optional
from ..bridge.julia_bridge import julia_engine

def orthogonal_procrustes(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Finds orthogonal matrix R aligning B to A, returning (R, B_aligned, disparity)."""
    if julia_engine.available:
        try:
            return julia_engine.orthogonal_procrustes(A, B)
        except Exception:
            pass # Fallback

    try:
        from scipy.spatial.transform import Rotation
        R, disparity = Rotation.align_vectors(A, B)
        R_mat = R.as_matrix()
        return R_mat, B @ R_mat, disparity
    except Exception:
        # SVD fallback
        M = B.T @ A
        U, S, Vt = np.linalg.svd(M)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        B_aligned = B @ R
        disparity = np.linalg.norm(A - B_aligned)
        return R, B_aligned, float(disparity)

def compute_distance_matrix(data: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Computes pairwise distance matrix using Julia (if available) or SciPy."""
    if julia_engine.available:
        try:
            return julia_engine.pairwise_distance_matrix(data, metric)
        except Exception:
            pass # Fallback
            
    if metric == "euclidean":
        return dist.squareform(dist.pdist(data, 'euclidean'))
    elif metric == "manhattan":
        return dist.squareform(dist.pdist(data, 'cityblock'))
    elif metric == "chebyshev":
        return dist.squareform(dist.pdist(data, 'chebyshev'))
    else:
        raise ValueError(f"Unsupported metric: {metric}")

def frechet_distance(curve_a: np.ndarray, curve_b: np.ndarray) -> float:
    """Computes Discrete Fréchet distance between two ordered sequences of points."""
    if julia_engine.available:
        try:
            return julia_engine.frechet_distance(curve_a, curve_b)
        except Exception:
            pass
            
    # Python fallback DP
    n = len(curve_a)
    m = len(curve_b)
    ca = np.full((n, m), -1.0)
    
    def d(i, j):
        return np.linalg.norm(curve_a[i] - curve_b[j])
        
    ca[0, 0] = d(0, 0)
    for i in range(1, n):
        ca[i, 0] = max(ca[i-1, 0], d(i, 0))
    for j in range(1, m):
        ca[0, j] = max(ca[0, j-1], d(0, j))
        
    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(min(ca[i-1, j], ca[i, j-1], ca[i-1, j-1]), d(i, j))
            
    return float(ca[n-1, m-1])

def gromov_wasserstein_distance(
    dist_matrix_A: np.ndarray, 
    dist_matrix_B: np.ndarray, 
    p: Optional[np.ndarray] = None, 
    q: Optional[np.ndarray] = None,
    epsilon: float = 0.01,
    max_iter: int = 100
) -> float:
    """Computes (Entropic) Gromov-Wasserstein distance between two metric spaces."""
    n = dist_matrix_A.shape[0]
    m = dist_matrix_B.shape[0]
    
    if p is None:
        p = np.ones(n) / n
    if q is None:
        q = np.ones(m) / m

    if julia_engine.available:
        try:
            return julia_engine.gromov_wasserstein_distance(dist_matrix_A, dist_matrix_B, p, q, epsilon, max_iter)
        except Exception:
            pass
            
    # Python fallback Sinkhorn algorithm
    T = np.outer(p, q)
    C1 = (dist_matrix_A ** 2) @ p[:, None] @ np.ones((1, m))
    C2 = np.ones((n, 1)) @ q[None, :] @ (dist_matrix_B ** 2).T
    const_C = C1 + C2
    
    for _ in range(max_iter):
        L = const_C - 2 * (dist_matrix_A @ T @ dist_matrix_B.T)
        K = np.exp(-L / epsilon)
        
        u = np.ones(n) / n
        v = np.ones(m) / m
        for _ in range(20):
            v = q / (K.T @ u + 1e-10)
            u = p / (K @ v + 1e-10)
            
        T_new = u[:, None] * K * v[None, :]
        if np.linalg.norm(T_new - T) < 1e-6:
            T = T_new
            break
        T = T_new
        
    gw_dist = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(n):
                for l in range(m):
                    gw_dist += ((dist_matrix_A[i, k] - dist_matrix_B[j, l])**2) * T[i, j] * T[k, l]
                    
    return float(np.sqrt(gw_dist))
