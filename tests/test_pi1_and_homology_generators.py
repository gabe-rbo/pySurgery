"""Tests for fundamental group extraction and homology generator computation.

Overview:
    This test suite validates the extraction of fundamental group (π₁) presentations 
    from CW complexes and the computation of homology generators from simplicial data.
    It covers spatial traces, presentation simplification, and backend consistency 
    between Python and Julia implementations.

Key Concepts:
    - **Fundamental Group (π₁)**: Group of loops up to homotopy.
    - **Spatial Traces**: Mapping between abstract group generators and specific edges/cells.
    - **Homology Generators**: Cycle representatives for homology classes (H_n).
    - **Julia Backend**: High-performance engine for exact topological computations.

Common Workflows:
    1. Extract π₁ with traces for geometric mapping.
    2. Compute optimal 1-homology bases (shortest cycles).
    3. Validate H_n generators for higher-dimensional complexes.
"""

import numpy as np
import scipy.sparse as sp

from pysurgery.core.complexes import CWComplex
from pysurgery.core.fundamental_group import extract_pi_1, extract_pi_1_with_traces
from pysurgery.core.homology_generators import (
    compute_homology_basis_from_simplices,
    compute_optimal_h1_basis_from_simplices,
)
from pysurgery.bridge.julia_bridge import julia_engine


def test_extract_pi1_with_traces_circle_has_data_grounded_generator():
    """Verify π₁ extraction from a circle returns correct spatial traces.

    What is Being Computed?:
        Extracts the fundamental group and spatial traces for a minimal circle representation.

    Algorithm:
        1. Define a CW complex with one 0-cell and one 1-cell (d1=0).
        2. Execute extract_pi_1_with_traces.
        3. Validate that the generator 'g_0' is correctly mapped to the 1-cell.

    Preserved Invariants:
        - Rank of π₁ (must be 1 for S¹).

    Returns:
        None (performs pytest assertions).
    """
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    cw = CWComplex(attaching_maps={1: d1}, dimensions=[0, 1], cells={0: 1, 1: 1})

    out = extract_pi_1_with_traces(cw)
    assert out.generators == ["g_0"]
    assert len(out.traces) == 1
    tr = out.traces[0]
    assert tr.generator == "g_0"
    assert tr.edge_index == 0
    assert tr.directed_edge_path[0] == (0, 0)


def test_extract_pi_1_with_traces_disc_simplifies_killed_generator():
    """Ensure π₁ extraction correctly simplifies a contractible disc to a trivial group.

    What is Being Computed?:
        The fundamental group of a disc (D²) using automated simplification (Tietze transformations).

    Algorithm:
        1. Construct CW complex for D² (one 0-cell, one 1-cell, one 2-cell killing the loop).
        2. Run extraction with simplify=True and generator_mode='optimized'.
        3. Confirm the resulting presentation is empty (trivial group).

    Preserved Invariants:
        - Homotopy type of a contractible space (π₁ must be trivial).

    Returns:
        None (performs pytest assertions).
    """
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    d2 = sp.csr_matrix(np.array([[1]], dtype=np.int64))
    cw = CWComplex(
        attaching_maps={1: d1, 2: d2}, dimensions=[0, 1, 2], cells={0: 1, 1: 1, 2: 1}
    )

    out = extract_pi_1_with_traces(cw, simplify=True, generator_mode="optimized")
    assert out.generators == []
    assert out.relations == []
    assert out.traces == []
    assert out.generator_mode == "optimized"
    assert out.mode_used == "optimized"
    assert out.optimized_generator_count == 0


def test_extract_pi1_with_traces_raw_mode_keeps_all_generators():
    """Verify that 'raw' mode preserves all generators even if they could be simplified.

    What is Being Computed?:
        A non-simplified presentation of the fundamental group for a contractible space.

    Algorithm:
        1. Construct CW complex for D².
        2. Run extraction with generator_mode='raw'.
        3. Validate that 'g_0' is retained despite being killed by a relation.

    Returns:
        None (performs pytest assertions).
    """
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    d2 = sp.csr_matrix(np.array([[1]], dtype=np.int64))
    cw = CWComplex(
        attaching_maps={1: d1, 2: d2}, dimensions=[0, 1, 2], cells={0: 1, 1: 1, 2: 1}
    )

    out = extract_pi_1_with_traces(cw, simplify=True, generator_mode="raw")
    assert out.generators == ["g_0"]
    assert len(out.traces) == 1
    assert out.traces[0].generator == "g_0"
    assert out.generator_mode == "raw"
    assert out.mode_used == "raw"
    assert out.raw_generator_count == 1
    assert out.reduced_generator_count == 1


def test_extract_pi1_raw_fundamental_group_mode_matches_traces():
    """Verify π₁ extraction in 'raw' mode yields a FundamentalGroup consistent with spatial traces.

    What is Being Computed?:
        Equivalence between the abstract FundamentalGroup presentation and the spatial trace data.

    Algorithm:
        1. Construct D² CW complex.
        2. Extract π₁ in 'raw' mode.
        3. Assert generators and relations match the expected non-simplified form.
    """
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    d2 = sp.csr_matrix(np.array([[1]], dtype=np.int64))
    cw = CWComplex(
        attaching_maps={1: d1, 2: d2}, dimensions=[0, 1, 2], cells={0: 1, 1: 1, 2: 1}
    )

    pi1 = extract_pi_1(cw, simplify=True, generator_mode="raw")
    assert pi1.generators == ["g_0"]
    assert pi1.relations == [["g_0"]]


def test_compute_optimal_h1_basis_from_simplices_square_cycle_rank_one():
    """Verify optimal H₁ basis computation for a square cycle.

    What is Being Computed?:
        The shortest cycle representing the homology class of a square graph.

    Algorithm:
        1. Define simplices for a 4-vertex cycle.
        2. Compute optimal H₁ basis.
        3. Assert rank is at least 1 and generators are 1-dimensional.
    """
    simplices = [
        (0, 1),
        (1, 2),
        (2, 3),
        (0, 3),
    ]
    res = compute_optimal_h1_basis_from_simplices(simplices, num_vertices=4)
    assert res.dimension == 1
    assert res.rank >= 1
    assert len(res.generators) >= 1
    assert all(g.dimension == 1 for g in res.generators)


def test_compute_optimal_h1_basis_from_simplices_filled_triangle_rank_zero():
    """Ensure H₁ basis is empty for a contractible filled triangle.

    What is Being Computed?:
        H₁ homology basis for a 2-simplex (triangle).

    Algorithm:
        1. Define simplices for a triangle plus its interior face (0,1,2).
        2. Compute H₁ basis.
        3. Assert rank is 0.
    """
    simplices = [
        (0, 1),
        (1, 2),
        (0, 2),
        (0, 1, 2),
    ]
    res = compute_optimal_h1_basis_from_simplices(simplices, num_vertices=3)
    assert res.dimension == 1
    assert res.rank == 0
    assert res.generators == []


def test_compute_optimal_h1_basis_from_simplices_julia_path_handles_square_cycle(
    monkeypatch,
):
    """Verify Julia backend integration for optimal H₁ basis computation.

    What is Being Computed?:
        Homology basis via the Julia accelerator bridge.

    Algorithm:
        1. Mock julia_engine.available and its H₁ basis method.
        2. Run basis computation.
        3. Confirm the result identifies 'Julia backend' in its metadata.
    """
    simplices = [
        (0, 1),
        (1, 2),
        (2, 3),
        (0, 3),
    ]

    monkeypatch.setattr(julia_engine, "available", True, raising=False)

    def _fake_julia_optimal(*args, **kwargs):
        assert args[0] == simplices
        assert args[1] == 4
        return [[(0, 1), (1, 2), (2, 3), (0, 3)]]

    monkeypatch.setattr(
        julia_engine,
        "compute_optimal_h1_basis_from_simplices",
        _fake_julia_optimal,
        raising=False,
    )

    res = compute_optimal_h1_basis_from_simplices(simplices, num_vertices=4)
    assert res.dimension == 1
    assert res.rank == 1
    assert len(res.generators) == 1
    assert "Julia backend" in res.message


def test_compute_optimal_h1_basis_python_fallback_when_julia_unavailable(monkeypatch):
    """Ensure graceful fallback to Python when Julia is unavailable.

    What is Being Computed?:
        H₁ basis using the pure-Python fallback engine.

    Algorithm:
        1. Force julia_engine.available = False.
        2. Compute H₁ basis.
        3. Confirm 'Python backend' is used.
    """
    simplices = [
        (0, 1),
        (1, 2),
        (2, 3),
        (0, 3),
    ]

    monkeypatch.setattr(julia_engine, "available", False, raising=False)
    res = compute_optimal_h1_basis_from_simplices(simplices, num_vertices=4)
    assert res.dimension == 1
    assert res.rank >= 1
    assert "Python backend" in res.message


def test_compute_optimal_h1_basis_from_simplices_julia_empty_result_does_not_fallback(
    monkeypatch,
):
    """Verify that an empty Julia result is accepted without unnecessary Python fallback.

    What is Being Computed?:
        Trivial H₁ basis for a contractible space via Julia.

    Algorithm:
        1. Mock Julia to return an empty basis.
        2. Assert that Python fallback functions are NOT called.
    """
    simplices = [
        (0, 1),
        (1, 2),
        (0, 2),
        (0, 1, 2),
    ]

    monkeypatch.setattr(julia_engine, "available", True, raising=False)
    monkeypatch.setattr(
        julia_engine,
        "compute_optimal_h1_basis_from_simplices",
        lambda *args, **kwargs: [],
        raising=False,
    )

    def _unexpected_fallback(*args, **kwargs):
        raise AssertionError(
            "Python fallback should not run when Julia returns an empty basis"
        )

    monkeypatch.setattr(
        "pysurgery.core.homology_generators.generator_cycles_from_simplices",
        _unexpected_fallback,
    )
    monkeypatch.setattr(
        "pysurgery.core.homology_generators.greedy_h1_basis", _unexpected_fallback
    )

    res = compute_optimal_h1_basis_from_simplices(simplices, num_vertices=3)
    assert res.dimension == 1
    assert res.rank == 0
    assert res.generators == []
    assert "Julia backend" in res.message


def test_compute_h0_generators_from_components():
    """Verify H₀ generator computation from connected components.

    What is Being Computed?:
        Basis for H₀(X; ℤ), where generators represent connected components.

    Algorithm:
        1. Define disjoint simplices.
        2. Compute H₀ basis.
        3. Assert rank matches number of components (3 in this case).
    """
    simplices = [
        (0, 1),
        (2, 3),
    ]
    res = compute_homology_basis_from_simplices(simplices, num_vertices=5, dimension=0)
    assert res.dimension == 0
    assert res.rank == 3
    assert all(g.dimension == 0 for g in res.generators)
    reps = sorted(g.support_simplices[0][0] for g in res.generators)
    assert reps == [0, 2, 4]


def test_compute_h2_generators_boundary_of_tetrahedron_rank_one():
    """Verify H₂ generator computation for the boundary of a tetrahedron.

    What is Being Computed?:
        The fundamental 2-cycle of a hollow tetrahedron (S²).

    Algorithm:
        1. Define the four 2-faces of a tetrahedron.
        2. Compute H₂ basis.
        3. Assert rank is 1.
    """
    simplices = [
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 3),
        (1, 2, 3),
    ]
    res = compute_homology_basis_from_simplices(simplices, num_vertices=4, dimension=2)
    assert res.dimension == 2
    assert res.rank == 1
    assert len(res.generators) == 1
    assert len(res.generators[0].support_simplices) == 4


def test_compute_h2_generators_filled_tetrahedron_rank_zero():
    """Ensure H₂ basis is empty for a solid tetrahedron.

    What is Being Computed?:
        H₂ homology for a 3-simplex (ball).

    Algorithm:
        1. Define 2-faces and the 3-face of a tetrahedron.
        2. Compute H₂ basis.
        3. Assert rank is 0.
    """
    simplices = [
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 3),
        (1, 2, 3),
        (0, 1, 2, 3),
    ]
    res = compute_homology_basis_from_simplices(simplices, num_vertices=4, dimension=2)
    assert res.dimension == 2
    assert res.rank == 0
    assert res.generators == []


def test_compute_h2_generators_optimal_mode_metadata():
    """Verify that 'optimal' mode correctly sets metadata in HomologyBasis.

    What is Being Computed?:
        Homology basis with optimality flags.

    Algorithm:
        1. Compute H₂ basis with mode='optimal'.
        2. Assert the 'optimal' flag is True in the result.
    """
    simplices = [
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 3),
        (1, 2, 3),
    ]
    res = compute_homology_basis_from_simplices(
        simplices, num_vertices=4, dimension=2, mode="optimal"
    )
    assert res.rank == 1
    assert res.optimal is True


def test_compute_h1_generators_valid_mode_works_when_julia_unavailable(monkeypatch):
    """Verify that 'valid' mode works correctly in pure-Python.

    What is Being Computed?:
        A non-necessarily optimal but valid H₁ basis.

    Algorithm:
        1. Force Julia unavailable.
        2. Compute H₁ basis with mode='valid'.
        3. Assert basis is correct.
    """
    simplices = [
        (0, 1),
        (1, 2),
        (2, 3),
        (0, 3),
    ]
    monkeypatch.setattr(julia_engine, "available", False, raising=False)
    res = compute_homology_basis_from_simplices(
        simplices, num_vertices=4, dimension=1, mode="valid"
    )
    assert res.dimension == 1
    assert res.rank >= 1
    assert all(g.dimension == 1 for g in res.generators)
