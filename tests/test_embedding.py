import numpy as np

from pysurgery.core.complexes import SimplicialComplex
from pysurgery.bridge.julia_bridge import julia_engine
import pysurgery.core.embedding as emb
from pysurgery.core.embedding import (
    analyze_embedding,
    check_immersion,
    PLMap,
    project_coordinates,
)


def _tetrahedron_boundary_complex():
    return SimplicialComplex.from_maximal_simplices(
        [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    )


def _regular_tetrahedron_coordinates(extra_dim: int = 0):
    base = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    )
    if extra_dim <= 0:
        return base
    extra = np.zeros((base.shape[0], extra_dim), dtype=np.float64)
    return np.hstack([base, extra])


def test_tetrahedron_boundary_embeds_in_r3():
    sc = _tetrahedron_boundary_complex()
    coords = _regular_tetrahedron_coordinates()

    result = analyze_embedding(sc, coords)

    assert result.status == "success"
    assert result.exact is True
    assert result.embedded is True
    assert result.immersion.immersed is True
    assert result.intersections == []
    assert result.decision_ready()
    assert result.immersion.decision_ready()
    assert result.intersections == []


def test_self_intersecting_surface_is_detected():
    sc = SimplicialComplex.from_maximal_simplices([(0, 1, 2), (3, 4, 5)])
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.25, 0.25, -1.0],
            [0.25, 0.25, 1.0],
            [0.75, 0.75, 0.0],
        ],
        dtype=np.float64,
    )

    result = analyze_embedding(sc, coords)

    assert result.embedded is False
    assert result.status == "impediment"
    assert len(result.intersections) > 0


def test_local_immersion_failure_on_degenerate_triangle():
    sc = _tetrahedron_boundary_complex()
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    pl_map = PLMap.from_source(sc, coords)
    immersion = check_immersion(pl_map)
    result = analyze_embedding(sc, coords)

    assert immersion.immersed is False
    assert result.immersion.immersed is False
    assert result.embedded is False
    assert result.status == "impediment"
    assert result.immersion.simplex_rank_failures


def test_projection_fallback_preserves_tetrahedron_embedding():
    sc = _tetrahedron_boundary_complex()
    coords = _regular_tetrahedron_coordinates(extra_dim=1)

    projected = project_coordinates(coords, target_dimension=3, method="pca")
    result = analyze_embedding(
        sc,
        coords,
        target_dimension=3,
        allow_projection=True,
        projection_method="pca",
    )

    assert projected.points.shape[1] == 3
    assert result.projection_used is True
    assert result.embedded is True
    assert result.status == "success"
    assert result.immersion.immersed is True
    assert result.decision_ready()


def test_embedding_self_intersection_is_deterministic_across_runs():
    sc = SimplicialComplex.from_maximal_simplices([(0, 1, 2), (3, 4, 5)])
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.25, 0.25, -1.0],
            [0.25, 0.25, 1.0],
            [0.75, 0.75, 0.0],
        ],
        dtype=np.float64,
    )
    first = analyze_embedding(sc, coords)
    second = analyze_embedding(sc, coords)
    assert [(w.simplex_a, w.simplex_b, w.kind) for w in first.intersections] == [
        (w.simplex_a, w.simplex_b, w.kind) for w in second.intersections
    ]


def test_embedding_uses_julia_broad_phase_when_available(monkeypatch):
    sc = _tetrahedron_boundary_complex()
    coords = _regular_tetrahedron_coordinates()
    calls = {"count": 0}

    monkeypatch.setattr(emb, "_JULIA_PAIR_BATCH_THRESHOLD", 1, raising=False)
    monkeypatch.setattr(julia_engine, "available", True, raising=False)

    def _fake_pairs(centroids, radii, *, tol):
        calls["count"] += 1
        n = centroids.shape[0]
        return np.array([(i, j) for i in range(n) for j in range(i + 1, n)], dtype=np.int64)

    monkeypatch.setattr(
        julia_engine,
        "compute_broad_phase_pairs",
        _fake_pairs,
        raising=False,
    )

    result = analyze_embedding(sc, coords)
    assert result.status == "success"
    assert calls["count"] >= 1



