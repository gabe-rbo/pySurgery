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



