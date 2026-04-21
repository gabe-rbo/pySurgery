"""Intrinsic-dimension estimators for point clouds and latent embeddings.

The estimators in this module are numerical diagnostics rather than exact
certificates. They are designed to fit pySurgery's exact-first architecture by
returning explicit status/diagnostic metadata and by isolating approximation in a
separate layer.

Implemented estimators
----------------------
- Levina--Bickel maximum-likelihood estimator
- TwoNN estimator
- Local PCA / tangent-space dimension estimation
- Bootstrap aggregation / ensemble reporting

References
----------
- Levina & Bickel, "Maximum Likelihood Estimation of Intrinsic Dimension".
- Facco et al., "Estimating the intrinsic dimension of datasets by a minimal
  neighborhood information".
- Local PCA / tangent-space dimension estimation from manifold learning literature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.spatial import cKDTree

from .complexes import SimplicialComplex

_EPS = 1e-12


class IntrinsicDimensionMethodResult(BaseModel):
    """Result for one intrinsic-dimension estimator."""

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
        return self.status == "success" and self.valid_points > 0 and np.isfinite(self.global_dimension)


class IntrinsicDimensionResult(BaseModel):
    """Aggregated intrinsic-dimension estimate for a point cloud."""

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
        return self.status == "success" and np.isfinite(self.global_dimension)


@dataclass
class _NeighborhoodCache:
    distances: np.ndarray
    indices: Optional[np.ndarray]
    points: Optional[np.ndarray]
    ambient_dim: int


def _coerce_point_cloud(
    data: np.ndarray | SimplicialComplex | object,
    coordinates: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Coerce a point-cloud-like input to a dense float array."""
    if isinstance(data, np.ndarray):
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
    points: Optional[np.ndarray],
    *,
    k: int,
    distance_matrix: Optional[np.ndarray] = None,
) -> _NeighborhoodCache:
    """Compute nearest-neighbor distances (and indices when available)."""
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
    data: np.ndarray | SimplicialComplex | object,
    k: int = 10,
    *,
    coordinates: Optional[np.ndarray] = None,
    distance_matrix: Optional[np.ndarray] = None,
) -> IntrinsicDimensionMethodResult:
    """Estimate intrinsic dimension with the Levina--Bickel MLE.

    Parameters
    ----------
    data:
        Point cloud data, or an object exposing `.coordinates`.
    k:
        Number of nearest neighbors to use. Must be at least 2.
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
    data: np.ndarray | SimplicialComplex | object,
    *,
    coordinates: Optional[np.ndarray] = None,
    distance_matrix: Optional[np.ndarray] = None,
) -> IntrinsicDimensionMethodResult:
    """Estimate intrinsic dimension using the TwoNN method.

    The global estimate is obtained from the slope of log(1-F(mu)) vs log(mu),
    where mu = r2 / r1.
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


def local_pca_tangent_space_dimension(
    data: np.ndarray | SimplicialComplex | object,
    k: int = 12,
    *,
    coordinates: Optional[np.ndarray] = None,
    distance_matrix: Optional[np.ndarray] = None,
    variance_threshold: float = 0.9,
    max_dimension: Optional[int] = None,
) -> IntrinsicDimensionMethodResult:
    """Estimate local tangent dimension via PCA on k-neighborhoods."""
    points = _coerce_point_cloud(data, coordinates=coordinates)
    cache = _compute_knn_cache(points, k=k, distance_matrix=distance_matrix)
    pts = cache.points if cache.points is not None else points
    assert pts is not None
    k_eff = cache.distances.shape[1]
    if k_eff < 2:
        raise ValueError("Local PCA requires at least two neighbors.")

    local_dims: list[float] = []
    diagnostics: list[str] = []
    max_dim = max_dimension if max_dimension is not None else pts.shape[1]
    max_dim = max(1, int(max_dim))

    from ..integrations.jax_bridge import HAS_JAX
    if HAS_JAX and cache.indices is not None:
        from ..integrations.jax_bridge import jax_local_pca_dimensions
        # neighborhood including point itself
        nbr_idx = np.hstack([np.arange(pts.shape[0])[:, None], cache.indices])
        jax_dims = jax_local_pca_dimensions(pts, nbr_idx, variance_threshold)
        local_dims = np.minimum(jax_dims.astype(float), float(max_dim)).tolist()
    else:
        for i in range(pts.shape[0]):
            if cache.indices is not None:
                nbr_idx = cache.indices[i]
                neighborhood = pts[nbr_idx]
            else:
                # Reconstruct from distances is impossible; skip.
                diagnostics.append("Missing neighbor indices for local PCA; skipped point {}.".format(i))
                continue
            cloud = np.vstack([pts[i], neighborhood])
            cloud = cloud - np.mean(cloud, axis=0, keepdims=True)
            if cloud.shape[0] <= 2:
                continue
            cov = np.cov(cloud, rowvar=False)
            if np.ndim(cov) == 0:
                eigvals = np.array([float(cov)], dtype=np.float64)
            else:
                eigvals = np.linalg.eigvalsh(np.asarray(cov, dtype=np.float64))[::-1]
            eigvals = np.maximum(eigvals, 0.0)
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


def exact_intrinsic_dimension(
    data: SimplicialComplex | object,
) -> IntrinsicDimensionMethodResult:
    """
    Calculate the exact intrinsic dimension of a manifold using link homology.
    
    This method requires the input to be a SimplicialComplex or expose a 
    compatible interface. It verifies if the complex is a homology manifold 
    and returns its dimension with 100% certainty (within the homology regime).
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
    data: np.ndarray | SimplicialComplex | object,
    k: int = 10,
    *,
    coordinates: Optional[np.ndarray] = None,
    distance_matrix: Optional[np.ndarray] = None,
    methods: Sequence[str] = ("mle", "twonn", "pca"),
    variance_threshold: float = 0.9,
    max_dimension: Optional[int] = None,
    bootstrap_samples: int = 200,
    confidence: float = 0.95,
    random_state: Optional[int] = 0,
) -> IntrinsicDimensionResult:
    """Estimate intrinsic dimension using a small ensemble of estimators."""
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
    "estimate_intrinsic_dimension",
    "levina_bickel_mle",
    "local_pca_tangent_space_dimension",
    "twonn",
    "exact_intrinsic_dimension",
]

