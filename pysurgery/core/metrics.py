"""Metric computations and point cloud alignment utilities.

Overview:
    This module provides tools for comparing and aligning geometric structures, 
    ranging from simple Euclidean distances to advanced Gromov-Wasserstein 
    transport and Fréchet path comparisons.

Key Concepts:
    - **Point Alignment**: Rigid alignment via Procrustes.
    - **Metric Spaces**: Pairwise distance matrices and subsampling.
    - **Optimal Transport**: Invariant comparison of metric measure spaces.

Common Workflows:
    1. **Preprocessing** -> `compute_distance_matrix()` or `farthest_point_sampling()`.
    2. **Alignment** -> `orthogonal_procrustes()` to align embeddings.
    3. **Comparison** -> `gromov_wasserstein_distance()` or `frechet_distance()` for similarity.
"""

import numpy as np
import scipy.spatial.distance as dist
from typing import Tuple, Optional
from ..bridge.julia_bridge import julia_engine

def orthogonal_procrustes(A: np.ndarray, B: np.ndarray, backend: str = "auto") -> Tuple[np.ndarray, np.ndarray, float]:
    """Find the optimal orthogonal transformation to align two point clouds.

    What is Being Computed?:
        Computes an orthogonal matrix R that minimizes the disparity (sum of 
        squared differences) between point cloud B and target A: 
        min_R ||A - BR||^2 such that R^T R = I.

    Algorithm:
        1. Compute the cross-covariance matrix M = B^T A.
        2. Perform Singular Value Decomposition (SVD): M = U S V^T.
        3. The optimal rotation matrix is R = U V^T.
        4. Handle reflection by checking the determinant of R if necessary.

    Preserved Invariants:
        - **Relative Distances**: Preserves all pairwise distances within B (isometry).
        - **Orientation**: Preserves or reverses orientation depending on the determinant of R.

    Args:
        A: The target point cloud (n_samples, dim).
        B: The point cloud to be aligned (n_samples, dim).
        backend: 'auto', 'julia', or 'python'.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]: (R, B_aligned, disparity) where R is the 
            orthogonal matrix, B_aligned is the transformed B, and disparity is the error.

    Use When:
        - You need to align two representations of the same shape in a common coordinate system.
        - Comparing different embeddings of the same topological space.

    Example:
        R, aligned, err = orthogonal_procrustes(A, B)
    """
    # Normalize backend
    backend_norm = str(backend).lower().strip()
    use_julia = (backend_norm == "julia") or (backend_norm == "auto" and julia_engine.available)

    if use_julia:
        try:
            return julia_engine.orthogonal_procrustes(A, B)
        except Exception as e:
            if backend_norm == "julia":
                raise e

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

def compute_distance_matrix(data: np.ndarray, metric: str = "euclidean", backend: str = "auto") -> np.ndarray:
    """Computes pairwise distance matrix using the optimal implementation.

    What is Being Computed?:
        A symmetric matrix D where D_ij = dist(x_i, x_j) for all pairs of points 
        in the input data.

    Algorithm:
        1. Select backend based on availability (JAX > Julia > Scipy).
        2. Compute distances using vectorized operations or optimized libraries.
        3. Return the dense distance matrix.

    Preserved Invariants:
        - **Metric Properties**: Non-negativity, symmetry, and identity of indiscernibles.
        - **Topology**: The induced topology remains invariant regardless of the backend.

    Args:
        data (np.ndarray): The input data points (n_samples, dim).
        metric (str): The distance metric to use ('euclidean', 'manhattan', 'chebyshev').
        backend: 'auto', 'julia', or 'python'.

    Returns:
        np.ndarray: The pairwise distance matrix (n_samples, n_samples).

    Raises:
        ValueError: If the requested metric is not supported.

    Use When:
        - Constructing Vietoris-Rips complexes.
        - Computing persistent homology.
        - Performing manifold learning or dimensionality reduction.

    Example:
        dist_mat = compute_distance_matrix(cloud, metric='euclidean')
    """
    # Normalize backend
    backend_norm = str(backend).lower().strip()
    use_julia = (backend_norm == "julia") or (backend_norm == "auto" and julia_engine.available)

    from ..integrations.jax_bridge import HAS_JAX
    if backend_norm != "julia" and HAS_JAX and metric == "euclidean":
        from ..integrations.jax_bridge import jax_pairwise_distance
        return np.asarray(jax_pairwise_distance(data))

    if use_julia:
        try:
            return julia_engine.pairwise_distance_matrix(data, metric)
        except Exception as e:
            if backend_norm == "julia":
                raise e
            
    if metric == "euclidean":
        return dist.squareform(dist.pdist(data, 'euclidean'))
    elif metric == "manhattan":
        return dist.squareform(dist.pdist(data, 'cityblock'))
    elif metric == "chebyshev":
        return dist.squareform(dist.pdist(data, 'chebyshev'))
    else:
        raise ValueError(f"Unsupported metric: {metric}")

def frechet_distance(curve_a: np.ndarray, curve_b: np.ndarray, backend: str = "auto") -> float:
    """Computes the Discrete Fréchet distance between two ordered point sequences.

    What is Being Computed?:
        The Fréchet distance d_F(A, B), which measures the similarity between curves 
        that takes into account the location and ordering of the points. It is 
        often called the "dog-man distance".

    Algorithm:
        Computes the discrete version via dynamic programming. The state ca[i, j] 
        represents the Fréchet distance between the prefix curves A[:i] and B[:j].

    Preserved Invariants:
        - **Reparameterization Invariance**: Approximately preserved (exactly in the continuous limit).
        - **Order**: Sensitive to the sequence of points, unlike Hausdorff distance.

    Args:
        curve_a (np.ndarray): First sequence of points (n, dim).
        curve_b (np.ndarray): Second sequence of points (m, dim).
        backend: 'auto', 'julia', or 'python'.

    Returns:
        float: The discrete Fréchet distance value.

    Use When:
        - Comparing paths in a topological space.
        - Matching trajectories or generator loops.
        - Verifying path homotopy representatives metrically.

    Example:
        dist = frechet_distance(path1, path2)
    """
    # Normalize backend
    backend_norm = str(backend).lower().strip()
    use_julia = (backend_norm == "julia") or (backend_norm == "auto" and julia_engine.available)

    if use_julia:
        try:
            return julia_engine.frechet_distance(curve_a, curve_b)
        except Exception as e:
            if backend_norm == "julia":
                raise e
            
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
    max_iter: int = 100,
    backend: str = "auto"
) -> float:
    """Computes the (Entropic) Gromov-Wasserstein distance between metric measure spaces.

    What is Being Computed?:
        The GW distance measures how far two metric measure spaces (X, d_X, μ_X) 
        and (Y, d_Y, μ_Y) are from being isometric. It is invariant to rigid 
        transformations (rotations and translations).

    Algorithm:
        1. Initialize transport plan T.
        2. Iteratively solve the entropic-regularized optimal transport problem 
           using the Sinkhorn algorithm or Julia/JAX acceleration.
        3. Compute the final GW cost from the optimal coupling.

    Preserved Invariants:
        - **Isometry Invariance**: GW(X, Y) = 0 if X and Y are isometric.
        - **Mass Conservation**: Works with probability measures p and q.

    Args:
        dist_matrix_A (np.ndarray): Distance matrix of space X.
        dist_matrix_B (np.ndarray): Distance matrix of space Y.
        p (Optional[np.ndarray]): Weights (measure) for points in X.
        q (Optional[np.ndarray]): Weights (measure) for points in Y.
        epsilon (float): Entropic regularization parameter (higher = smoother/faster).
        max_iter (int): Maximum Sinkhorn/Optimization iterations.
        backend: 'auto', 'julia', or 'python'.

    Returns:
        float: The Gromov-Wasserstein distance value.

    Use When:
        - Comparing point clouds with different numbers of points or in different dimensions.
        - Shape matching where rigid alignment is not sufficient.
        - Analyzing metric structures without a common coordinate system.

    Example:
        gw = gromov_wasserstein_distance(D1, D2, epsilon=0.05)
    """
    # Normalize backend
    backend_norm = str(backend).lower().strip()
    use_julia = (backend_norm == "julia") or (backend_norm == "auto" and julia_engine.available)

    n = dist_matrix_A.shape[0]
    m = dist_matrix_B.shape[0]
    
    if p is None:
        p = np.ones(n) / n
    if q is None:
        q = np.ones(m) / m

    from ..integrations.jax_bridge import HAS_JAX
    if backend_norm != "julia" and HAS_JAX:
        from ..integrations.jax_bridge import jax_gromov_wasserstein
        return float(jax_gromov_wasserstein(dist_matrix_A, dist_matrix_B, p, q, epsilon, max_iter))

    if use_julia:
        try:
            return julia_engine.gromov_wasserstein_distance(dist_matrix_A, dist_matrix_B, p, q, epsilon, max_iter)
        except Exception as e:
            if backend_norm == "julia":
                raise e
            
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

    What is Being Computed?:
        A subset of indices corresponding to landmark points that provide a 
        maximal covering of the point cloud.

    Algorithm:
        1. Start with an initial point.
        2. Iteratively pick the point that has the maximum "minimum distance" 
           to the already selected set.
        3. Update distances and repeat until n_samples are reached.

    Preserved Invariants:
        - **Density Hierarchy**: Points are picked in order of their contribution 
          to covering the space.
        - **Topology**: For sufficiently high n_samples, the landmarks capture 
           the same homotopy type (Nerve Theorem).

    Args:
        points (np.ndarray): The input point cloud (n_total, dim).
        n_samples (int): The number of landmark points to select.
        initial_idx (int): Index of the first seed point.

    Returns:
        np.ndarray: Array of indices of length n_samples.

    Use When:
        - Downsampling large data for faster persistent homology.
        - Selecting centers for RBF interpolation or sparse manifold learning.
        - Ensuring uniform coverage of a geometric object.

    Example:
        indices = farthest_point_sampling(data, n_samples=100)
        landmarks = data[indices]
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
