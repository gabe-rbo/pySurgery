import numpy as np
import scipy.spatial.distance as dist
from typing import Tuple, Optional
from ..bridge.julia_bridge import julia_engine

def orthogonal_procrustes(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Finds orthogonal matrix R aligning B to A, returning (R, B_aligned, disparity).

    Args:
        A (np.ndarray): The target point cloud.
        B (np.ndarray): The point cloud to be aligned.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]: A tuple containing the orthogonal
            matrix R, the aligned point cloud B_aligned, and the disparity (error).
    """
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
    """Computes pairwise distance matrix using the optimal singular implementation.

    Standardizes on JAX if available to ensure hardware acceleration for all scales.

    Args:
        data (np.ndarray): The input data points.
        metric (str): The distance metric to use. Defaults to "euclidean".

    Returns:
        np.ndarray: The pairwise distance matrix.

    Raises:
        ValueError: If the metric is not supported.
    """
    from ..integrations.jax_bridge import HAS_JAX
    if HAS_JAX and metric == "euclidean":
        from ..integrations.jax_bridge import jax_pairwise_distance
        return np.asarray(jax_pairwise_distance(data))

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
    """Computes Discrete Fréchet distance between two ordered sequences of points.

    Args:
        curve_a (np.ndarray): The first sequence of points.
        curve_b (np.ndarray): The second sequence of points.

    Returns:
        float: The discrete Fréchet distance.
    """
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
    """Computes (Entropic) Gromov-Wasserstein distance.

    Standardizes on JAX if available for all scales.

    Args:
        dist_matrix_A (np.ndarray): Distance matrix of the first space.
        dist_matrix_B (np.ndarray): Distance matrix of the second space.
        p (Optional[np.ndarray]): Probability distribution on the first space.
        q (Optional[np.ndarray]): Probability distribution on the second space.
        epsilon (float): Regularization parameter. Defaults to 0.01.
        max_iter (int): Maximum number of iterations. Defaults to 100.

    Returns:
        float: The (entropic) Gromov-Wasserstein distance.
    """
    n = dist_matrix_A.shape[0]
    m = dist_matrix_B.shape[0]
    
    if p is None:
        p = np.ones(n) / n
    if q is None:
        q = np.ones(m) / m

    from ..integrations.jax_bridge import HAS_JAX
    if HAS_JAX:
        from ..integrations.jax_bridge import jax_gromov_wasserstein
        return float(jax_gromov_wasserstein(dist_matrix_A, dist_matrix_B, p, q, epsilon, max_iter))

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
        
    # Vectorized final GW distance computation:
    # GW = sum_{i,j,k,l} (C1_ik - C2_jl)^2 * T_ij * T_kl
    #    = sum_{i,j} (C1^2 @ p)_i * T_ij + sum_{k,l} (C2^2 @ q)_l * T_kl - 2 * Tr(C1 @ T @ C2.T @ T.T)
    C1 = dist_matrix_A
    C2 = dist_matrix_B
    term1 = np.sum((C1**2 @ p[:, None]) * np.sum(T, axis=1, keepdims=True))
    term2 = np.sum((C2**2 @ q[:, None]) * np.sum(T, axis=0, keepdims=True).T)
    term3 = 2 * np.sum((C1 @ T @ C2) * T)
    
    gw_dist = max(0.0, term1 + term2 - term3)
    return float(np.sqrt(gw_dist))

def farthest_point_sampling(points: np.ndarray, n_samples: int, initial_idx: int = 0) -> np.ndarray:
    """Subsample a point cloud by greedily picking points that maximize distance to the current set.

    This provides a 'maximal' covering of the underlying manifold.

    Args:
        points (np.ndarray): The input point cloud.
        n_samples (int): The number of landmark points to sample.
        initial_idx (int): The index of the first point to pick. Defaults to 0.

    Returns:
        np.ndarray: The indices of the selected landmark points.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    if n_samples >= n:
        return np.arange(n)
        
    landmarks = np.zeros(n_samples, dtype=np.int64)
    landmarks[0] = initial_idx
    
    # Track min distance from each point to the current set of landmarks
    min_distances = np.linalg.norm(pts - pts[initial_idx], axis=1)
    
    for i in range(1, n_samples):
        # Pick the point that is farthest from all existing landmarks
        new_idx = np.argmax(min_distances)
        landmarks[i] = new_idx
        
        # Update min distances with the new landmark
        dist_to_new = np.linalg.norm(pts - pts[new_idx], axis=1)
        min_distances = np.minimum(min_distances, dist_to_new)
        
    return landmarks
