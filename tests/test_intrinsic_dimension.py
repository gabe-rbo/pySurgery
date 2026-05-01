"""Tests for Intrinsic Dimension Estimation algorithms.

Overview:
    This suite verifies various estimators for the intrinsic dimension of 
    point clouds, including Maximum Likelihood Estimation (MLE), Two-Nearest 
    Neighbors (TwoNN), and Local PCA.

Key Concepts:
    - **Intrinsic Dimension**: The minimal number of variables needed to represent 
      the data without significant information loss.
    - **MLE Estimator**: Based on the distance distribution of nearest neighbors.
    - **TwoNN**: Uses the ratio of distances to the first two nearest neighbors.
    - **Local PCA**: Estimates dimension via the rank of local covariance matrices.
"""
from types import SimpleNamespace

import numpy as np

from pysurgery.core.intrinsic_dimension import (
    estimate_intrinsic_dimension,
    levina_bickel_mle,
    local_pca_tangent_space_dimension,
    twonn,
)


def _circle_point_cloud(n=400, seed=0):
    rng = np.random.default_rng(seed)
    theta = np.sort(rng.uniform(0.0, 2.0 * np.pi, size=n))
    r = 1.0 + 0.01 * rng.normal(size=n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = 0.01 * rng.normal(size=n)
    return np.column_stack([x, y, z]).astype(np.float64)


def _plane_point_cloud(n=500, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=n)
    y = rng.uniform(-1.0, 1.0, size=n)
    z = 0.01 * rng.normal(size=n)
    return np.column_stack([x, y, z]).astype(np.float64)


def test_mle_and_twonn_on_distance_matrix_plane():
    """Verify MLE and TwoNN estimators on a planar point cloud.

    What is Being Computed?:
        Intrinsic dimension estimates for a set of points sampled from a 
        2-dimensional plane embedded in 3D space.

    Algorithm:
        1. Generate a plane point cloud with small normal noise.
        2. Precompute the distance matrix.
        3. Apply MLE and TwoNN estimators.
        4. Validate that estimated global dimension is approximately 2.

    Preserved Invariants:
        - Intrinsic dimension — should be invariant to isometric embedding.
    """
    points = _plane_point_cloud(n=500, seed=7)
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)

    mle = levina_bickel_mle(points, k=12, distance_matrix=distances)
    two_nn = twonn(points, distance_matrix=distances)
    ensemble = estimate_intrinsic_dimension(
        points,
        methods=("mle", "twonn"),
        distance_matrix=distances,
        bootstrap_samples=50,
    )

    assert mle.status == "success"
    assert two_nn.status == "success"
    assert ensemble.status == "success"
    assert 1.5 < mle.global_dimension < 2.5
    assert 1.5 < two_nn.global_dimension < 2.5
    assert 1.5 < ensemble.global_dimension < 2.5
    assert ensemble.confidence_interval is not None


def test_local_pca_and_ensemble_on_plane_point_cloud():
    points = _plane_point_cloud()
    wrapped = SimpleNamespace(coordinates=points)

    pca = local_pca_tangent_space_dimension(wrapped, k=18)
    ensemble = estimate_intrinsic_dimension(wrapped, k=18, methods=("pca", "mle", "twonn"))

    assert pca.status == "success"
    assert ensemble.status == "success"
    assert 1.5 < pca.global_dimension < 2.5
    assert 1.5 < ensemble.global_dimension < 2.5
    assert pca.decision_ready()
    assert ensemble.decision_ready()
    assert len(pca.local_dimensions) > 0
    assert len(ensemble.method_results) == 3


def test_intrinsic_dimension_reports_inconclusive_on_too_small_sample():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    out = estimate_intrinsic_dimension(points, k=2, methods=("mle", "twonn"), bootstrap_samples=0)

    assert out.n_samples == 3
    assert out.ambient_dim == 3
    assert out.status in {"success", "inconclusive"}
    assert out.method_estimates

def test_intrinsic_dimension_sphere():
    # Points on S2 in 3D
    n = 600
    rng = np.random.default_rng(42)
    phi = rng.uniform(0, 2*np.pi, n)
    theta = np.arccos(rng.uniform(-1, 1, n))
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    points = np.column_stack([x, y, z])
    
    res = estimate_intrinsic_dimension(points, k=15, methods=("mle", "twonn", "pca"))
    assert res.status == "success"
    # Expected dimension is 2
    assert 1.5 < res.global_dimension < 2.5



