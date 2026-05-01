"""Tests for PL-map embeddings and self-intersection analysis.

Overview:
    This suite verifies the piecewise-linear (PL) embedding analysis for simplicial
    complexes mapped into Euclidean space. It checks for self-intersections,
    immersion failures (singularities), and correctness of the PL map vertex
    extraction.

Key Concepts:
    - **PL Map**: A map from a simplicial complex to R^n that is linear on each simplex.
    - **Embedding**: An injective immersion; a map with no self-intersections and no local singularities.
    - **Immersion**: A map that is locally an embedding (non-degenerate on each simplex).
    - **Self-Intersection**: Points in the target space that are images of distinct points in the source.
"""
from __future__ import annotations

import numpy as np

from pysurgery.core.complexes import SimplicialComplex
from pysurgery.core.embedding import PLMap, analyze_embedding


def test_pl_map_simplex_vertices():
    """Verify vertex coordinate extraction for simplices in a PL map.

    What is Being Computed?:
        The coordinates in R^n for a given simplex's vertices.

    Algorithm:
        1. Create a simplicial complex and a PL map with specified coordinates.
        2. Extract vertices for a 2-simplex.
        3. Verify the resulting shape and coordinate values.

    Preserved Invariants:
        - Geometric realization of the simplicial complex.
    """
    sc = SimplicialComplex.from_simplices([(0, 1, 2), (2, 3)])
    coords = np.array([[0.0, 0], [1, 0], [0, 1], [1, 1]])
    pl = PLMap.from_source(source=sc, coordinates=coords)
    v = pl.simplex_vertices((0, 1, 2))
    assert v.shape == (3, 2)
    assert np.allclose(v[0], [0, 0])


def test_analyze_embedding_tetrahedron():
    """Verify that a standard tetrahedron is correctly identified as an embedding in R3.

    What is Being Computed?:
        Embedding status and intersection witnesses for a tetrahedron.

    Algorithm:
        1. Construct a tetrahedron boundary as a simplicial complex.
        2. Assign regular tetrahedron coordinates in R3.
        3. Run analyze_embedding.

    Preserved Invariants:
        - Topology of S^2 (boundary of tetrahedron).
    """
    sc = _tetrahedron_boundary_complex()
    coords = _regular_tetrahedron_coordinates()
    result = analyze_embedding(sc, coords)
    assert result.status == "success"
    assert result.embedded is True
    assert result.immersion.immersed is True
    assert len(result.intersections) == 0


def test_analyze_embedding_self_intersection():
    """Verify detection of self-intersecting triangles in R3.

    What is Being Computed?:
        Self-intersection witnesses for non-embedded triangles.

    Algorithm:
        1. Define two triangles that pass through each other in R3.
        2. Run analyze_embedding.
        3. Assert that embedded is False and check for triangle-triangle intersection witnesses.

    Preserved Invariants:
        - N/A
    """
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
    """Verify detection of immersion failure (singularities) for collapsed simplices.

    What is Being Computed?:
        Immersion status and local failures.

    Algorithm:
        1. Define a triangle where vertices are collinear (collapsed to a line).
        2. Run analyze_embedding.
        3. Assert that immersed is False.

    Preserved Invariants:
        - Local rank of the PL map.
    """
    # Singular map: triangle collapsed to a line
    sc = SimplicialComplex.from_simplices([(0, 1, 2)])
    coords = np.array([[0.0, 0], [1, 0], [2, 0]])
    result = analyze_embedding(sc, coords)
    assert result.immersion.immersed is False
    assert len(result.immersion.local_failures) > 0


def test_analyze_embedding_caching():
    """Verify that embedding analysis results are stable and correctly cached/recomputed.

    What is Being Computed?:
        Consistency of intersection witnesses across multiple calls.

    Algorithm:
        1. Run analyze_embedding on a set of coordinates.
        2. Run it again on the same coordinates.
        3. Compare the lists of intersections.

    Preserved Invariants:
        - Determinism of intersection detection.
    """
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
    """Helper to build a simplicial complex of a tetrahedron boundary.

    Returns:
        SimplicialComplex: Boundary of a 3-simplex.
    """
    return SimplicialComplex.from_simplices([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])


def _regular_tetrahedron_coordinates():
    """Helper to provide coordinates for a regular tetrahedron centered at origin.

    Returns:
        np.ndarray: (4, 3) array of coordinates.
    """
    return np.array(
        [
            [1.0, 1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
        ],
        dtype=np.float64,
    )
