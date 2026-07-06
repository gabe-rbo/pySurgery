"""Intrinsic-dimension estimators for point clouds and latent embeddings.

Overview:
    This module provides a suite of estimators to determine the intrinsic 
    dimensionality of a dataset. It bridges the gap between approximate 
    point-cloud-based diagnostics and exact homology-based manifold 
    certification. The estimators are designed to fit pySurgery's 
    exact-first architecture by returning explicit diagnostic metadata.

Key Concepts:
    - **Approximate Estimators**: Statistical methods (MLE, TwoNN) for raw point data.
    - **Geometric Estimators**: Local PCA for tangent space dimension.
    - **Exact Certificates**: Homology manifold verification for SimplicialComplexes.
    - **Ensemble Aggregation**: Combining multiple methods for robust consensus.

Common Workflows:
    1. **Point Cloud Analysis** -> `estimate_intrinsic_dimension()` with 'mle' or 'twonn'.
    2. **Manifold Verification** -> `exact_intrinsic_dimension()` on a `SimplicialComplex`.
    3. **Dimension-Aware Surgery** -> Use estimated dimension to parameterize surgery obstructions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union, TYPE_CHECKING
if TYPE_CHECKING:
    from .point_cloud import PointCloud

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.spatial import cKDTree

from pysurgery.topology.complexes import SimplicialComplex

_EPS = 1e-12


class IntrinsicDimensionMethodResult(BaseModel):
    """Result for one intrinsic-dimension estimator.

    Overview:
        Encapsulates the output of a single intrinsic-dimension estimation method,
        providing global and local estimates along with diagnostic metadata and 
        confidence scores.

    Key Concepts:
        - **Global Dimension**: A single scalar representing the estimated dimension of the entire dataset.
        - **Local Dimension**: Point-wise estimates revealing variations in manifold dimension.
        - **Confidence**: Heuristic scoring based on valid point coverage and method stability.

    Attributes:
        method (str): Name of the estimation method (e.g., 'twonn', 'levina_bickel_mle').
        global_dimension (float): The estimated global intrinsic dimension.
        local_dimensions (list[float]): Local dimension estimates for each point.
        neighborhood_size (int): Size of the neighborhood (k) used for estimation.
        valid_points (int): Number of points that yielded a valid estimate.
        ambient_dim (int): Dimension of the ambient space.
        exact (bool): Whether the estimate is theoretically exact (e.g., homology manifold).
        status (str): Status of the estimation ('success', 'inconclusive', etc.).
        scale (float): Characteristic scale of the estimation.
        confidence (float): A heuristic confidence score [0, 1].
        diagnostics (list[str]): List of diagnostic messages.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    method: str
    global_dimension: float
    local_dimensions: list[float] = Field(default_factory=list)
    neighborhood_size: int = 0
    valid_points: int = 0
    ambient_dim: int = 0
    exact: bool = False
    status: str = "success"
    scale: float = 0.0
    confidence: float = 0.0
    diagnostics: list[str] = Field(default_factory=list)

    def decision_ready(self) -> bool:
        """Check if the result is conclusive and usable for decisions.

        Returns:
            bool: True if status is 'success', there are valid points, and the dimension is finite.

        Use When:
            - You need to programmatically decide if the estimation should be trusted.
        """
        return self.status == "success" and self.valid_points > 0 and np.isfinite(self.global_dimension)


class IntrinsicDimensionResult(BaseModel):
    """Aggregated intrinsic-dimension estimate for a point cloud.

    Overview:
        Provides a consolidated view of multiple intrinsic-dimension estimates,
        typically from an ensemble of different methods. It supports bootstrap
        confidence intervals and aggregated diagnostic reporting.

    Common Workflows:
        1. **Estimation** -> `estimate_intrinsic_dimension()` returns this object.
        2. **Validation** -> Check `.decision_ready()` before using the dimension in downstream tasks.
        3. **Ensemble Analysis** -> Compare `.method_estimates` to check for consensus.

    Attributes:
        method (str): Name of the aggregation method (usually 'ensemble').
        global_dimension (float): The aggregated global intrinsic dimension.
        method_estimates (dict[str, float]): Dictionary mapping method names to their global estimates.
        method_results (dict[str, IntrinsicDimensionMethodResult]): Dictionary mapping method names to their full results.
        local_dimensions (list[float]): Combined list of local dimension estimates.
        n_samples (int): Total number of points in the dataset.
        ambient_dim (int): Dimension of the ambient space.
        exact (bool): Whether the result is theoretically exact.
        status (str): Aggregated status of the estimation.
        confidence_interval (Optional[tuple[float, float]]): (low, high) bootstrap confidence interval.
        confidence (float): Aggregated confidence score.
        bootstrap_samples (int): Number of bootstrap samples used.
        diagnostics (list[str]): Combined list of diagnostic messages.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    method: str = "ensemble"
    global_dimension: float
    method_estimates: dict[str, float] = Field(default_factory=dict)
    method_results: dict[str, IntrinsicDimensionMethodResult] = Field(default_factory=dict)
    local_dimensions: list[float] = Field(default_factory=list)
    n_samples: int = 0
    ambient_dim: int = 0
    exact: bool = False
    status: str = "success"
    confidence_interval: Optional[tuple[float, float]] = None
    confidence: float = 0.0
    bootstrap_samples: int = 0
    diagnostics: list[str] = Field(default_factory=list)

    def decision_ready(self) -> bool:
        """Check if the aggregated result is conclusive.

        Returns:
            bool: True if status is 'success' and the global dimension is finite.

        Use When:
            - You need to verify the overall ensemble consensus before proceeding.
        """
        return self.status == "success" and np.isfinite(self.global_dimension)


@dataclass
class _NeighborhoodCache:
    """Internal cache for nearest-neighbor data.

    Overview:
        Stores precomputed distances and indices for nearest neighbors to avoid 
        redundant computations during multiple estimation passes.

    Attributes:
        distances (np.ndarray): (n_points, k) array of distances to neighbors.
        indices (Optional[np.ndarray]): (n_points, k) array of neighbor indices.
        points (Optional[np.ndarray]): The original point cloud coordinates.
        ambient_dim (int): Dimension of the ambient space.
    """
    distances: np.ndarray
    indices: Optional[np.ndarray]
    points: Optional[np.ndarray]
    ambient_dim: int


def _coerce_point_cloud(
    data: np.ndarray | PointCloud | SimplicialComplex | object,
    coordinates: Optional[np.ndarray | PointCloud] = None,
) -> np.ndarray:
    """Coerce a point-cloud-like input to a dense float array.

    Algorithm:
        1. Checks if input is already a numpy array.
        2. If not, attempts to extract `.coordinates` or uses provided coordinates.
        3. Validates dimensions and minimum sample size.

    Args:
        data: Point cloud data or object with coordinates.
        coordinates: Optional explicit coordinates.

    Returns:
        np.ndarray: A dense (n_samples, ambient_dim) float64 array.

    Raises:
        TypeError: If the input type is not supported.
        ValueError: If the dimensions or sample size are invalid.
    """
    from .point_cloud import PointCloud
    if isinstance(data, (np.ndarray, PointCloud)):
        points = np.asarray(data, dtype=np.float64)
    elif coordinates is not None:
        points = np.asarray(coordinates, dtype=np.float64)
    elif hasattr(data, "coordinates") and getattr(data, "coordinates") is not None:
        points = np.asarray(getattr(data, "coordinates"), dtype=np.float64)
    else:
        raise TypeError(
            "Expected a point cloud array, an object with `.coordinates`, or explicit coordinates."
        )

    if points.ndim != 2:
        raise ValueError("Point cloud must be a 2D array of shape (n_samples, ambient_dim).")
    if points.shape[0] < 3:
        raise ValueError("At least 3 points are required for intrinsic-dimension estimation.")
    return points


def _compute_knn_cache(
    points: Optional[np.ndarray | PointCloud],
    *,
    k: int,
    distance_matrix: Optional[np.ndarray] = None,
) -> _NeighborhoodCache:
    """Compute nearest-neighbor distances (and indices when available).

    What is Being Computed?:
        A cache of distances and indices for the k nearest neighbors of each point.

    Algorithm:
        1. If a distance matrix is provided, it sorts and extracts neighbors.
        2. Otherwise, it uses a cKDTree for efficient spatial querying.
        3. Returns a `_NeighborhoodCache` containing the results.

    Args:
        points: (n_samples, ambient_dim) point cloud.
        k: Number of nearest neighbors.
        distance_matrix: Optional precomputed (n_samples, n_samples) distance matrix.

    Returns:
        _NeighborhoodCache: A cache containing distances and indices.

    Raises:
        ValueError: If inputs are inconsistent or invalid.
    """
    if k < 2:
        raise ValueError("k must be at least 2.")

    if distance_matrix is not None:
        D = np.asarray(distance_matrix, dtype=np.float64)
        if D.ndim != 2 or D.shape[0] != D.shape[1]:
            raise ValueError("distance_matrix must be square.")
        n = D.shape[0]
        if n < 3:
            raise ValueError("At least 3 points are required.")
        if k >= n:
            k = n - 1
        work = D.copy()
        np.fill_diagonal(work, np.inf)
        indices = np.argsort(work, axis=1)[:, :k]
        rows = np.arange(n)[:, None]
        distances = work[rows, indices]
        return _NeighborhoodCache(
            distances=distances,
            indices=indices,
            points=points,
            ambient_dim=0 if points is None else int(points.shape[1]),
        )

    if points is None:
        raise ValueError("points are required when no distance_matrix is provided.")
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2:
        raise ValueError("points must be a 2D array.")
    n = pts.shape[0]
    if n < 3:
        raise ValueError("At least 3 points are required.")
    k_eff = min(int(k), n - 1)
    if k_eff < 2:
        raise ValueError("Need at least two neighbors beyond the query point.")
    tree = cKDTree(pts)
    distances, indices = tree.query(pts, k=k_eff + 1)
    distances = np.asarray(distances, dtype=np.float64)[:, 1:]
    indices = np.asarray(indices, dtype=np.int64)[:, 1:]
    return _NeighborhoodCache(
        distances=distances,
        indices=indices,
        points=pts,
        ambient_dim=int(pts.shape[1]),
    )


def _bootstrap_interval(
    values: Sequence[float],
    *,
    n_bootstrap: int = 200,
    confidence: float = 0.95,
    random_state: Optional[int] = None,
) -> Optional[tuple[float, float]]:
    """Compute a bootstrap confidence interval for the mean.

    Algorithm:
        1. Resample the input values with replacement `n_bootstrap` times.
        2. Compute the mean of each bootstrap sample.
        3. Extract the quantiles corresponding to the desired confidence level.

    Args:
        values: Sequence of numeric values.
        n_bootstrap: Number of bootstrap samples.
        confidence: Confidence level (e.g., 0.95).
        random_state: Optional seed for the random generator.

    Returns:
        Optional[tuple[float, float]]: A tuple (low, high) representing the interval, or None if input is empty.
    """
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if vals.size == 0:
        return None
    if n_bootstrap <= 0:
        return None
    alpha = 0.5 * (1.0 - float(confidence))
    rng = np.random.default_rng(random_state)
    boots = []
    for _ in range(int(n_bootstrap)):
        sample = rng.choice(vals, size=vals.size, replace=True)
        boots.append(float(np.mean(sample)))
    low = float(np.quantile(boots, alpha))
    high = float(np.quantile(boots, 1.0 - alpha))
    return low, high


def _aggregate_method_result(
    method: str,
    local_dimensions: Sequence[float],
    *,
    k: int,
    ambient_dim: int,
    diagnostics: Optional[list[str]] = None,
    exact: bool = False,
) -> IntrinsicDimensionMethodResult:
    """Aggregate local estimates into a method result.

    Algorithm:
        1. Filters out non-finite local estimates.
        2. Computes the median as the global dimension.
        3. Calculates a heuristic confidence score based on coverage.

    Args:
        method: Name of the estimation method.
        local_dimensions: Sequence of local estimates.
        k: Neighborhood size used.
        ambient_dim: Dimension of ambient space.
        diagnostics: Optional diagnostic messages.
        exact: Whether the method is exact.

    Returns:
        IntrinsicDimensionMethodResult: An aggregated method result.
    """
    vals = np.asarray([v for v in local_dimensions if np.isfinite(v)], dtype=np.float64)
    if vals.size == 0:
        return IntrinsicDimensionMethodResult(
            method=method,
            global_dimension=float("nan"),
            local_dimensions=[],
            neighborhood_size=int(k),
            valid_points=0,
            ambient_dim=int(ambient_dim),
            exact=exact,
            status="inconclusive",
            diagnostics=list(diagnostics or []) + ["No finite local estimates were produced."],
        )
    global_dim = float(np.median(vals))
    confidence = float(min(1.0, vals.size / max(1.0, float(k))))
    return IntrinsicDimensionMethodResult(
        method=method,
        global_dimension=global_dim,
        local_dimensions=[float(v) for v in vals.tolist()],
        neighborhood_size=int(k),
        valid_points=int(vals.size),
        ambient_dim=int(ambient_dim),
        exact=exact,
        status="success",
        confidence=confidence,
        diagnostics=list(diagnostics or []),
    )


def levina_bickel_mle(
    data: np.ndarray | PointCloud | SimplicialComplex | object,
    k: int = 10,
    *,
    coordinates: Optional[np.ndarray | PointCloud] = None,
    distance_matrix: Optional[np.ndarray] = None,
) -> IntrinsicDimensionMethodResult:
    """Estimate intrinsic dimension with the Levina--Bickel MLE.

    What is Being Computed?:
        The maximum likelihood estimate of the intrinsic dimension based on the 
        distribution of distances to nearest neighbors, assuming a local Poisson 
        process in a k-dimensional ball.

    Algorithm:
        1. Compute distances to the k nearest neighbors for each point.
        2. Apply the MLE formula based on the ratio of distances to the k-th neighbor.
        3. Average local estimates to produce a global dimension.

    Preserved Invariants:
        - Approximates the local Hausdorff dimension.
        - Sensitive to noise and sampling density; not a strict topological invariant.

    Args:
        data: Point cloud data, or an object exposing `.coordinates`.
        k: Number of nearest neighbors to use. Must be at least 2.
        coordinates: Optional explicit coordinates.
        distance_matrix: Optional precomputed distance matrix.

    Returns:
        IntrinsicDimensionMethodResult: The result of the Levina-Bickel estimation.

    Use When:
        - Fast estimation for large point clouds is required.
        - Data is expected to lie on a single manifold with uniform density.
        - You need a baseline MLE estimate.

    Example:
        result = levina_bickel_mle(points, k=10)
        print(f"ID: {result.global_dimension:.2f}")

    Raises:
        ValueError: If k is less than 2.
    """
    points = None if distance_matrix is not None else _coerce_point_cloud(data, coordinates=coordinates)
    cache = _compute_knn_cache(points, k=k, distance_matrix=distance_matrix)
    dist = np.maximum(cache.distances, _EPS)
    k_eff = dist.shape[1]
    if k_eff < 2:
        raise ValueError("Levina-Bickel MLE requires at least two neighbors.")

    rk = np.maximum(dist[:, -1], _EPS)
    logs = np.log(np.maximum(rk[:, None] / dist[:, :-1], 1.0 + _EPS))
    denom = np.sum(logs, axis=1)
    local = np.where(denom > 0.0, (k_eff - 1.0) / denom, np.nan)
    diagnostics = []
    if np.isnan(local).any():
        diagnostics.append("Some neighborhoods were degenerate or had repeated distances.")
    return _aggregate_method_result(
        "levina_bickel_mle",
        local,
        k=k_eff,
        ambient_dim=cache.ambient_dim,
        diagnostics=diagnostics,
    )


def twonn(
    data: np.ndarray | PointCloud | SimplicialComplex | object,
    *,
    coordinates: Optional[np.ndarray | PointCloud] = None,
    distance_matrix: Optional[np.ndarray] = None,
) -> IntrinsicDimensionMethodResult:
    """Estimate intrinsic dimension using the TwoNN method.

    What is Being Computed?:
        Intrinsic dimension via the distribution of the ratio of distances to the 
        first and second nearest neighbors.

    Algorithm:
        1. Compute distances to the two nearest neighbors (r1, r2) for each point.
        2. Compute mu = r2 / r1.
        3. Perform linear regression on the empirical cumulative distribution of mu 
           to find the slope, which corresponds to the intrinsic dimension.

    Preserved Invariants:
        - Dimension is invariant to local density variations (scale-invariant).
        - Robust against non-uniform sampling on the manifold.

    Args:
        data: Point cloud data or object with coordinates.
        coordinates: Optional explicit coordinates.
        distance_matrix: Optional precomputed distance matrix.

    Returns:
        IntrinsicDimensionMethodResult: The result of the TwoNN estimation.

    Use When:
        - Data density is highly non-uniform.
        - You want an estimate robust to varying sampling rates.
        - Quick estimation with only 2 neighbors is preferred.

    Example:
        result = twonn(points)
        print(f"ID: {result.global_dimension:.2f}")
    """
    points = None if distance_matrix is not None else _coerce_point_cloud(data, coordinates=coordinates)
    cache = _compute_knn_cache(points, k=2, distance_matrix=distance_matrix)
    dist = np.maximum(cache.distances[:, :2], _EPS)
    mu = np.maximum(dist[:, 1] / dist[:, 0], 1.0 + _EPS)
    mu = mu[np.isfinite(mu)]
    if mu.size == 0:
        return IntrinsicDimensionMethodResult(
            method="twonn",
            global_dimension=float("nan"),
            local_dimensions=[],
            neighborhood_size=2,
            valid_points=0,
            ambient_dim=cache.ambient_dim,
            status="inconclusive",
            exact=False,
            diagnostics=["No finite two-nearest-neighbor ratios were available."],
        )

    mu_sorted = np.sort(mu)
    n = mu_sorted.size
    F = (np.arange(1, n + 1, dtype=np.float64) - 0.5) / float(n)
    x = np.log(mu_sorted)
    y = np.log(np.maximum(1.0 - F, _EPS))
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    denom = float(np.sum((x - x_mean) ** 2))
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom) if denom > 0 else float("nan")
    global_dim = float(-slope) if np.isfinite(slope) else float("nan")
    local = 1.0 / np.log(mu)
    diagnostics = []
    if np.isnan(local).any() or not np.isfinite(global_dim):
        diagnostics.append("TwoNN encountered degenerate distance ratios.")
    return IntrinsicDimensionMethodResult(
        method="twonn",
        global_dimension=global_dim,
        local_dimensions=[float(v) for v in local[np.isfinite(local)].tolist()],
        neighborhood_size=2,
        valid_points=int(mu.size),
        ambient_dim=cache.ambient_dim,
        exact=False,
        status="success" if np.isfinite(global_dim) else "inconclusive",
        confidence=float(min(1.0, mu.size / 10.0)),
        diagnostics=diagnostics,
    )


def _local_pca_spectrum(
    pts: np.ndarray,
    cache: _NeighborhoodCache,
    *,
    k_out: Optional[int] = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """Shared per-point local-neighborhood covariance eigendecomposition.

    What is Being Computed?:
        For every point, the eigenvalues and eigenvectors of the covariance matrix of its
        own neighborhood (the point itself plus its ``k`` nearest neighbors, centered) --
        the common numerical core behind both ``local_pca_tangent_space_dimension`` (which
        thresholds the *full* eigenvalue spectrum's cumulative-variance curve to get a
        per-point-varying integer dimension) and ``local_pca_tangent_basis`` (which keeps
        only the leading ``k_out`` eigenvectors as a forced, uniform-across-points
        projection basis).

    Algorithm:
        JAX path (available and neighbor indices present): batched SVD of each point's
        centered neighborhood (``jax_local_pca_basis``) -- eigenvalues from the singular
        values squared, eigenvectors from the right singular vectors. Python fallback: a
        per-point ``numpy.linalg.eigh`` on the explicit ``(D, D)`` covariance matrix. Either
        way, eigenvalues are always returned in full (needed for a true explained-variance
        ratio and for cumulative-variance dimension counting), while eigenvectors are
        truncated to the first ``k_out`` columns (or kept in full if ``k_out`` is None) --
        slicing is free once the decomposition is already computed.

    Args:
        pts: (N, D) point coordinates.
        cache: Precomputed neighborhood cache (see ``_compute_knn_cache``).
        k_out: Number of leading eigenvectors to keep per point. ``None`` keeps the full
            available spectrum (``min(k_neighbors + 1, D)``).

    Returns:
        tuple[list[np.ndarray], list[np.ndarray], list[str]]: ``(eigvals_per_point,
        eigvecs_per_point, diagnostics)`` -- ``eigvals_per_point[i]`` is always the *full*
        descending eigenvalue spectrum, ``eigvecs_per_point[i]`` is the matching ``(D,
        n_kept)`` array of eigenvectors truncated to ``k_out`` columns.
    """
    diagnostics: list[str] = []
    ambient_dim = pts.shape[1]

    from ..integrations.jax_bridge import HAS_JAX
    if HAS_JAX and cache.indices is not None:
        from ..integrations.jax_bridge import jax_local_pca_basis
        nbr_idx = np.hstack([np.arange(pts.shape[0])[:, None], cache.indices])
        r_available = min(nbr_idx.shape[1], ambient_dim)
        k_request = r_available if k_out is None else min(int(k_out), r_available)
        bases, eigvals = jax_local_pca_basis(pts, nbr_idx, k_request)
        bases = np.asarray(bases)
        eigvals = np.asarray(eigvals)
        if k_out is not None and k_request < k_out:
            diagnostics.append(
                f"Requested basis dimension {k_out} exceeds the available local rank "
                f"({r_available}); truncated to {r_available} for every point."
            )
        return (
            [eigvals[i] for i in range(eigvals.shape[0])],
            [bases[i] for i in range(bases.shape[0])],
            diagnostics,
        )

    eigvals_list: list[np.ndarray] = []
    eigvecs_list: list[np.ndarray] = []
    for i in range(pts.shape[0]):
        if cache.indices is None:
            diagnostics.append(f"Missing neighbor indices for local PCA; skipped point {i}.")
            eigvals_list.append(np.zeros(0))
            eigvecs_list.append(np.zeros((ambient_dim, 0)))
            continue
        neighborhood = pts[cache.indices[i]]
        cloud = np.vstack([pts[i], neighborhood])
        cloud = cloud - np.mean(cloud, axis=0, keepdims=True)
        cov = np.atleast_2d(np.cov(cloud, rowvar=False))
        w, v = np.linalg.eigh(np.asarray(cov, dtype=np.float64))
        order = np.argsort(w)[::-1]
        eigvals = np.maximum(w[order], 0.0)
        eigvecs = v[:, order]
        k_request = eigvals.shape[0] if k_out is None else min(int(k_out), eigvals.shape[0])
        if k_out is not None and k_request < k_out:
            diagnostics.append(
                f"Point {i}: local rank ({eigvals.shape[0]}) is smaller than the requested "
                f"basis dimension ({k_out}); truncated."
            )
        eigvals_list.append(eigvals)
        eigvecs_list.append(eigvecs[:, :k_request])

    return eigvals_list, eigvecs_list, diagnostics


def local_pca_tangent_space_dimension(
    data: np.ndarray | PointCloud | SimplicialComplex | object,
    k: int = 12,
    *,
    coordinates: Optional[np.ndarray | PointCloud] = None,
    distance_matrix: Optional[np.ndarray] = None,
    variance_threshold: float = 0.9,
    max_dimension: Optional[int] = None,
) -> IntrinsicDimensionMethodResult:
    """Estimate local tangent dimension via PCA on k-neighborhoods.

    What is Being Computed?:
        Local intrinsic dimension by analyzing the spectrum of the covariance 
        matrix of local neighborhoods.

    Algorithm:
        1. For each point, find its k-nearest neighbors.
        2. Centering the neighborhood and compute the local covariance matrix.
        3. Perform Eigenvalue decomposition or use SVD (accelerated by JAX if available).
        4. Count the number of eigenvalues required to reach a cumulative variance threshold.

    Preserved Invariants:
        - Estimates the dimension of the tangent space at each point.
        - Captures the dimension of the linear subspace locally approximating the manifold.

    Args:
        data: Point cloud data or object with coordinates.
        k: Number of neighbors for local PCA.
        coordinates: Optional explicit coordinates.
        distance_matrix: Optional precomputed distance matrix.
        variance_threshold: Cumulative variance threshold to determine dimension.
        max_dimension: Optional cap on the estimated dimension.

    Returns:
        IntrinsicDimensionMethodResult: The result of the local PCA estimation.

    Use When:
        - You need a clear geometric interpretation (tangent space dimension).
        - Analyzing data with noise where a variance threshold is a natural way to separate signal.
        - JAX is available for high-performance batch PCA.

    Example:
        result = local_pca_tangent_space_dimension(points, k=15, variance_threshold=0.95)

    Raises:
        ValueError: If k is less than 2.
    """
    points = _coerce_point_cloud(data, coordinates=coordinates)
    cache = _compute_knn_cache(points, k=k, distance_matrix=distance_matrix)
    pts = cache.points if cache.points is not None else points
    assert pts is not None
    k_eff = cache.distances.shape[1]
    if k_eff < 2:
        raise ValueError("Local PCA requires at least two neighbors.")

    max_dim = max_dimension if max_dimension is not None else pts.shape[1]
    max_dim = max(1, int(max_dim))

    eigvals_list, _eigvecs_list, diagnostics = _local_pca_spectrum(pts, cache)

    local_dims: list[float] = []
    for eigvals in eigvals_list:
        if eigvals.size == 0:
            continue
        total = float(np.sum(eigvals))
        if total <= _EPS:
            continue
        explained = np.cumsum(eigvals) / total
        dim = int(np.searchsorted(explained, float(variance_threshold), side="left") + 1)
        dim = min(max(dim, 1), max_dim)
        local_dims.append(float(dim))

    if not local_dims:
        return IntrinsicDimensionMethodResult(
            method="local_pca",
            global_dimension=float("nan"),
            local_dimensions=[],
            neighborhood_size=int(k_eff),
            valid_points=0,
            ambient_dim=cache.ambient_dim,
            exact=False,
            status="inconclusive",
            diagnostics=diagnostics + ["No valid local PCA neighborhoods were found."],
        )

    return _aggregate_method_result(
        "local_pca",
        local_dims,
        k=k_eff,
        ambient_dim=cache.ambient_dim,
        diagnostics=diagnostics,
    )


class TangentBasisResult(BaseModel):
    """Per-point, fixed-dimension local tangent-space basis from local PCA.

    Overview:
        Bundles the per-point orthonormal tangent-frame basis vectors (the leading
        eigenvectors of each point's local neighborhood covariance) that
        ``pysurgery.geometry.tangential_complex`` projects each point's neighborhood onto
        before running a local, lower-dimensional Delaunay triangulation.

    Attributes:
        bases (np.ndarray): ``(N, D, k)`` array; ``bases[i]`` is an orthonormal ``(D, k)``
            matrix whose columns span point i's estimated k-dimensional tangent space,
            ordered by descending explained variance.
        explained_variance_ratio (np.ndarray): ``(N,)`` array; the fraction of point i's
            local neighborhood variance captured by its k-dimensional basis (1.0 if k spans
            the full local rank).
        neighborhood_size (int): Number of neighbors actually used per point.
        k (int): The forced, uniform basis dimension.
        ambient_dim (int): Dimension of the ambient space.
        diagnostics (list[str]): Human-readable notes (e.g. near-rank-deficient neighborhoods).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bases: np.ndarray
    explained_variance_ratio: np.ndarray
    neighborhood_size: int
    k: int
    ambient_dim: int
    diagnostics: list[str] = Field(default_factory=list)


def local_pca_tangent_basis(
    data: np.ndarray | PointCloud | SimplicialComplex | object,
    k: int,
    *,
    neighborhood_size: int = 12,
    coordinates: Optional[np.ndarray | PointCloud] = None,
    distance_matrix: Optional[np.ndarray] = None,
) -> TangentBasisResult:
    """Estimate a per-point, fixed-dimension local tangent-space basis via local PCA.

    What is Being Computed?:
        For every point, the top-``k`` eigenvectors of its local neighborhood's covariance
        matrix -- an orthonormal basis for the ``k``-dimensional linear subspace that best
        approximates the manifold near that point. Unlike
        ``local_pca_tangent_space_dimension`` (which estimates a *per-point-varying* integer
        dimension from a variance threshold), ``k`` here is a single, caller-fixed value
        applied uniformly to every point -- exactly what the tangential Delaunay complex
        needs (one consistent projection dimension to triangulate in), and what
        ``jax.vmap``'s batched SVD needs for a uniform output shape.

    Algorithm:
        1. Gather each point's ``neighborhood_size`` nearest neighbors
           (``_compute_knn_cache``).
        2. Shared with ``local_pca_tangent_space_dimension`` (``_local_pca_spectrum``):
           center each neighborhood (including the point itself), eigendecompose its
           covariance (JAX batched SVD when available, else a per-point
           ``numpy.linalg.eigh`` loop), sort descending by eigenvalue.
        3. Keep the first ``k`` eigenvectors per point, and record what fraction of that
           point's local variance they explain.

    Preserved Invariants:
        Every point's basis has exactly ``k`` orthonormal columns -- guaranteed by requiring
        ``neighborhood_size + 1 >= k`` and ``ambient_dim >= k`` upfront (a structural,
        data-independent check), rather than silently truncating or zero-padding a
        rank-deficient point's basis, which would silently corrupt downstream triangulation.

    Args:
        data: Point cloud data or object with coordinates.
        k: The fixed tangent-space dimension to estimate for every point.
        neighborhood_size: Number of nearest neighbors per point (excluding the point
            itself); must satisfy ``neighborhood_size + 1 >= k``.
        coordinates: Optional explicit coordinates.
        distance_matrix: Optional precomputed distance matrix.

    Returns:
        TangentBasisResult: The per-point bases and explained-variance diagnostics.

    Raises:
        ValueError: If ``k`` is not in ``[1, ambient_dim]``, or ``neighborhood_size + 1 <
            k``.

    Use When:
        - Building a tangential Delaunay complex (``pysurgery.geometry.tangential_complex``),
          where every point must project onto a basis of the *same* dimension.

    Example:
        result = local_pca_tangent_basis(points, k=2, neighborhood_size=15)
        tangent_coords_i = (points - points[i]) @ result.bases[i]
    """
    points = _coerce_point_cloud(data, coordinates=coordinates)
    cache = _compute_knn_cache(points, k=neighborhood_size, distance_matrix=distance_matrix)
    pts = cache.points if cache.points is not None else points
    assert pts is not None
    ambient_dim = pts.shape[1]
    k = int(k)
    if k < 1 or k > ambient_dim:
        raise ValueError(f"k must be in [1, {ambient_dim}] (ambient dimension), got {k}.")
    k_eff = cache.distances.shape[1]
    if k_eff + 1 < k:
        raise ValueError(
            f"neighborhood_size ({k_eff}) + 1 must be >= k ({k}); increase "
            "neighborhood_size to estimate a rank-k tangent basis."
        )

    eigvals_list, eigvecs_list, diagnostics = _local_pca_spectrum(pts, cache, k_out=k)

    bases = np.stack(eigvecs_list, axis=0)
    explained_variance_ratio = np.array(
        [
            float(np.sum(ev[:k]) / total) if (total := float(np.sum(ev))) > _EPS else 0.0
            for ev in eigvals_list
        ],
        dtype=np.float64,
    )

    return TangentBasisResult(
        bases=bases,
        explained_variance_ratio=explained_variance_ratio,
        neighborhood_size=int(k_eff),
        k=k,
        ambient_dim=int(ambient_dim),
        diagnostics=diagnostics,
    )


def exact_intrinsic_dimension(
    data: SimplicialComplex | object,
) -> IntrinsicDimensionMethodResult:
    """Calculate the exact intrinsic dimension of a manifold using link homology.
    
    What is Being Computed?:
        Determines if a simplicial complex is a homology manifold and, if so, 
        extracts its exact topological dimension.

    Algorithm:
        1. Iterate over all vertices (0-simplices).
        2. For each vertex, compute the homology of its link.
        3. Verify if the links are homology spheres of the same dimension.
        4. Return the unique dimension n such that the complex is locally like R^n.

    Preserved Invariants:
        - **Topological Dimension**: The exact dimension of the manifold.
        - **Homology Manifold Property**: Verified for all local neighborhoods.

    Args:
        data: A SimplicialComplex or object exposing one.

    Returns:
        IntrinsicDimensionMethodResult: The result of the exact homology-based calculation.

    Use When:
        - You have a combinatorial structure (SimplicialComplex) and need a certificate.
        - Exact dimension is required for subsequent surgery theory obstructions.
        - You need to verify if the space is actually a manifold.

    Example:
        result = exact_intrinsic_dimension(sc)
        if result.status == 'success':
            print(f"Certified {int(result.global_dimension)}-manifold")
    """
    if not isinstance(data, SimplicialComplex):
        if hasattr(data, "simplicial_complex"):
            sc = getattr(data, "simplicial_complex")
        else:
            return IntrinsicDimensionMethodResult(
                method="exact_homology",
                global_dimension=float("nan"),
                status="inconclusive",
                exact=True,
                diagnostics=["Input is not a SimplicialComplex; exact calculation aborted."],
            )
    else:
        sc = data

    is_manifold, dim, diagnostics = sc.is_homology_manifold()
    
    diag_list = [f"{k}: {v}" for k, v in diagnostics.items()]
    if not is_manifold:
        diag_list.append("Complex is not a pure homology manifold.")
        status = "inconclusive"
    else:
        status = "success"

    return IntrinsicDimensionMethodResult(
        method="exact_homology",
        global_dimension=float(dim) if dim is not None else float("nan"),
        valid_points=len(sc.n_simplices(0)),
        exact=True,
        status=status,
        confidence=1.0 if is_manifold else 0.0,
        diagnostics=diag_list,
    )


def estimate_intrinsic_dimension(
    data: np.ndarray | PointCloud | SimplicialComplex | object,
    k: int = 10,
    *,
    coordinates: Optional[np.ndarray | PointCloud] = None,
    distance_matrix: Optional[np.ndarray] = None,
    methods: Sequence[str] = ("mle", "twonn", "pca"),
    variance_threshold: float = 0.9,
    max_dimension: Optional[int] = None,
    bootstrap_samples: int = 200,
    confidence: float = 0.95,
    random_state: Optional[int] = 0,
    backend: str = "auto",
) -> IntrinsicDimensionResult:
    """Estimate intrinsic dimension using a small ensemble of estimators.

    What is Being Computed?:
        An aggregated intrinsic dimension estimate by combining multiple numerical 
        and (optionally) exact methods.

    Algorithm:
        1. Run requested estimators (MLE, TwoNN, PCA, Exact).
        2. Collect local and global estimates.
        3. Compute an aggregated global dimension (median of methods).
        4. Perform bootstrap sampling on the local pool to generate confidence intervals.

    Preserved Invariants:
        - Attempts to converge on the true topological dimension of the underlying manifold.

    Args:
        data: Point cloud data or object with coordinates.
        k: Number of neighbors for MLE and PCA methods.
        coordinates: Optional explicit coordinates.
        distance_matrix: Optional precomputed distance matrix.
        methods: List of methods to include ('mle', 'twonn', 'pca', 'exact').
        variance_threshold: Variance threshold for PCA method.
        max_dimension: Cap on the estimated dimension.
        bootstrap_samples: Number of bootstrap samples for confidence interval.
        confidence: Confidence level for the interval.
        random_state: Seed for bootstrap sampling.
        backend: 'auto', 'julia', or 'python'.

    Returns:
        IntrinsicDimensionResult: An aggregated IntrinsicDimensionResult.

    Use When:
        - You need a robust, multi-method consensus on the intrinsic dimension.
        - Confidence intervals are required for statistical significance.
        - You want to automatically use exact methods if the input is a complex.

    Example:
        res = estimate_intrinsic_dimension(data, methods=['mle', 'twonn'])
        print(f"Consensus ID: {res.global_dimension}")
    """
    points = None if distance_matrix is not None else _coerce_point_cloud(data, coordinates=coordinates)
    method_results: dict[str, IntrinsicDimensionMethodResult] = {}
    estimates: dict[str, float] = {}
    local_pool: list[float] = []
    diagnostics: list[str] = []

    methods_norm = list(m.lower() for m in methods)
    is_complex = isinstance(data, SimplicialComplex) or hasattr(data, "simplicial_complex")
    
    if is_complex and "exact" not in methods_norm:
        methods_norm.append("exact")

    if "exact" in methods_norm or "exact_homology" in methods_norm:
        res = exact_intrinsic_dimension(data)
        method_results["exact"] = res
        if res.status == "success":
            estimates["exact"] = float(res.global_dimension)
        diagnostics.extend(res.diagnostics)

    if "mle" in methods_norm or "levina_bickel" in methods_norm:
        res = levina_bickel_mle(
            data,
            k=k,
            coordinates=coordinates,
            distance_matrix=distance_matrix,
        )
        method_results["mle"] = res
        estimates["mle"] = float(res.global_dimension)
        local_pool.extend(res.local_dimensions)
        diagnostics.extend(res.diagnostics)

    if "twonn" in methods_norm or "two_nn" in methods_norm:
        res = twonn(data, coordinates=coordinates, distance_matrix=distance_matrix)
        method_results["twonn"] = res
        estimates["twonn"] = float(res.global_dimension)
        local_pool.extend(res.local_dimensions)
        diagnostics.extend(res.diagnostics)

    if "pca" in methods_norm or "local_pca" in methods_norm:
        res = local_pca_tangent_space_dimension(
            data,
            k=max(3, k),
            coordinates=coordinates,
            distance_matrix=distance_matrix,
            variance_threshold=variance_threshold,
            max_dimension=max_dimension,
        )
        method_results["pca"] = res
        estimates["pca"] = float(res.global_dimension)
        local_pool.extend(res.local_dimensions)
        diagnostics.extend(res.diagnostics)

    finite_estimates = [v for v in estimates.values() if np.isfinite(v)]
    if finite_estimates:
        global_dimension = float(np.median(np.asarray(finite_estimates, dtype=np.float64)))
        status = "success"
    else:
        global_dimension = float("nan")
        status = "inconclusive"

    ci = _bootstrap_interval(
        local_pool if local_pool else finite_estimates,
        n_bootstrap=bootstrap_samples,
        confidence=confidence,
        random_state=random_state,
    )
    confidence_score = float(min(1.0, len(finite_estimates) / max(1.0, float(len(methods_norm)))))
    ambient_dim = int(points.shape[1]) if points is not None else int(distance_matrix.shape[0] if distance_matrix is not None else 0)
    n_samples = int(points.shape[0]) if points is not None else int(distance_matrix.shape[0] if distance_matrix is not None else 0)

    return IntrinsicDimensionResult(
        method="ensemble",
        global_dimension=global_dimension,
        method_estimates=estimates,
        method_results=method_results,
        local_dimensions=[float(v) for v in local_pool],
        n_samples=n_samples,
        ambient_dim=ambient_dim,
        exact=False,
        status=status,
        confidence_interval=ci,
        confidence=confidence_score,
        bootstrap_samples=int(bootstrap_samples),
        diagnostics=diagnostics,
    )


__all__ = [
    "IntrinsicDimensionMethodResult",
    "IntrinsicDimensionResult",
    "TangentBasisResult",
    "estimate_intrinsic_dimension",
    "levina_bickel_mle",
    "local_pca_tangent_space_dimension",
    "local_pca_tangent_basis",
    "twonn",
    "exact_intrinsic_dimension",
]
