from __future__ import annotations

import numpy as np
import pytest

from pysurgery.bridge.julia_bridge import julia_engine
from pysurgery.core import embedding as emb
from pysurgery.core.complexes import SimplicialComplex
from pysurgery.core.embedding import PLMap, analyze_embedding


def test_pl_map_simplex_vertices():
    sc = SimplicialComplex.from_simplices([(0, 1, 2), (2, 3)])
    coords = np.array([[0.0, 0], [1, 0], [0, 1], [1, 1]])
    pl = PLMap.from_source(source=sc, coordinates=coords)
    v = pl.simplex_vertices((0, 1, 2))
    assert v.shape == (3, 2)
    assert np.allclose(v[0], [0, 0])


def test_analyze_embedding_tetrahedron():
    sc = _tetrahedron_boundary_complex()
    coords = _regular_tetrahedron_coordinates()
    result = analyze_embedding(sc, coords)
    assert result.status == "success"
    assert result.embedded is True
    assert result.immersion.immersed is True
    assert len(result.intersections) == 0


def test_analyze_embedding_self_intersection():
    # Two triangles intersecting in R3
    sc = SimplicialComplex.from_simplices([(0, 1, 2), (3, 4, 5)])
    coords = np.array(
        [
            [0.0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0.5, 0.5, -1],
            [0.5, 0.5, 1],
            [-1, -1, 0],
        ]
    )
    result = analyze_embedding(sc, coords)
    assert result.embedded is False
    # There should be at least one witness pair (detects multiple due to faces)
    assert len(result.intersections) >= 1
    # Check that at least one is triangle_triangle
    assert any(w.kind == "triangle_triangle" for w in result.intersections)


def test_analyze_embedding_immersion_fail():
    # Singular map: triangle collapsed to a line
    sc = SimplicialComplex.from_simplices([(0, 1, 2)])
    coords = np.array([[0.0, 0], [1, 0], [2, 0]])
    result = analyze_embedding(sc, coords)
    assert result.immersion.immersed is False
    assert len(result.immersion.local_failures) > 0


def test_analyze_embedding_caching():
    sc = SimplicialComplex.from_simplices([(0, 1, 2), (3, 4, 5)])
    coords = np.array(
        [
            [0.0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0.25, 0.25, -1],
            [0.25, 0.25, 1],
            [0.75, 0.75, 0],
        ],
        dtype=np.float64,
    )
    first = analyze_embedding(sc, coords)
    second = analyze_embedding(sc, coords)
    assert [(w.simplex_a, w.simplex_b, w.kind) for w in first.intersections] == [
        (w.simplex_a, w.simplex_b, w.kind) for w in second.intersections
    ]


def _tetrahedron_boundary_complex():
    return SimplicialComplex.from_simplices([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])


def _regular_tetrahedron_coordinates():
    return np.array(
        [
            [1.0, 1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
        ],
        dtype=np.float64,
    )
