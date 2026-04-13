import numpy as np
import scipy.sparse as sp

from pysurgery.core.complexes import CWComplex
from pysurgery.core.fundamental_group import extract_pi_1_with_traces
from pysurgery.core.homology_generators import (
    compute_homology_basis_from_simplices,
    compute_optimal_h1_basis_from_simplices,
)
from pysurgery.bridge.julia_bridge import julia_engine


def test_extract_pi1_with_traces_circle_has_data_grounded_generator():
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    cw = CWComplex(attaching_maps={1: d1}, dimensions=[0, 1], cells={0: 1, 1: 1})

    out = extract_pi_1_with_traces(cw)
    assert out.generators == ["g_0"]
    assert len(out.traces) == 1
    tr = out.traces[0]
    assert tr.generator == "g_0"
    assert tr.edge_index == 0
    assert tr.directed_edge_path[0] == (0, 0)


def test_extract_pi1_with_traces_disc_simplifies_killed_generator():
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    d2 = sp.csr_matrix(np.array([[1]], dtype=np.int64))
    cw = CWComplex(attaching_maps={1: d1, 2: d2}, dimensions=[0, 1, 2], cells={0: 1, 1: 1, 2: 1})

    out = extract_pi_1_with_traces(cw, simplify=True)
    assert out.generators == []
    assert out.relations == []
    assert out.traces == []


def test_compute_optimal_h1_basis_from_simplices_square_cycle_rank_one():
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


def test_compute_optimal_h1_basis_python_fallback_when_julia_unavailable(monkeypatch):
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


def test_compute_optimal_h1_basis_from_simplices_julia_empty_result_does_not_fallback(monkeypatch):
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
        raise AssertionError("Python fallback should not run when Julia returns an empty basis")

    monkeypatch.setattr("pysurgery.core.homology_generators.generator_cycles_from_simplices", _unexpected_fallback)
    monkeypatch.setattr("pysurgery.core.homology_generators.greedy_h1_basis", _unexpected_fallback)

    res = compute_optimal_h1_basis_from_simplices(simplices, num_vertices=3)
    assert res.dimension == 1
    assert res.rank == 0
    assert res.generators == []
    assert "Julia backend" in res.message


def test_compute_h0_generators_from_components():
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
    simplices = [
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 3),
        (1, 2, 3),
    ]
    res = compute_homology_basis_from_simplices(simplices, num_vertices=4, dimension=2, mode="optimal")
    assert res.rank == 1
    assert res.optimal is True


def test_compute_h1_generators_valid_mode_works_when_julia_unavailable(monkeypatch):
    simplices = [
        (0, 1),
        (1, 2),
        (2, 3),
        (0, 3),
    ]
    monkeypatch.setattr(julia_engine, "available", False, raising=False)
    res = compute_homology_basis_from_simplices(simplices, num_vertices=4, dimension=1, mode="valid")
    assert res.dimension == 1
    assert res.rank >= 1
    assert all(g.dimension == 1 for g in res.generators)


