import numpy as np
import scipy.sparse as sp
import pytest
import sys
import types

try:
    from tests.discrete_surface_data import get_surfaces, get_3_manifolds, to_complex
except ImportError:
    pass
from pysurgery.core.complexes import ChainComplex, CWComplex, SimplicialComplex
import pysurgery.core.complexes as complexes
from pysurgery.bridge.julia_bridge import julia_engine


def test_sphere_homology():
    # S^2: 1 cell of dim 0, 1 cell of dim 2, trivial boundaries
    d1 = sp.csr_matrix(np.zeros((1, 0), dtype=np.int64))
    d2 = sp.csr_matrix(np.zeros((0, 1), dtype=np.int64))
    complex_c = ChainComplex(
        boundaries={1: d1, 2: d2}, dimensions=[0, 1, 2], cells={0: 1, 1: 0, 2: 1}
    )
    assert complex_c.homology(0) == (1, [])
    assert complex_c.homology(1) == (0, [])
    assert complex_c.homology(2) == (1, [])

    assert complex_c.cohomology(0) == (1, [])
    assert complex_c.cohomology(1) == (0, [])
    assert complex_c.cohomology(2) == (1, [])


def test_torus_homology():
    # T^2: 1 cell of dim 0, 2 cells of dim 1, 1 cell of dim 2
    d1 = sp.csr_matrix(np.zeros((1, 2), dtype=np.int64))
    d2 = sp.csr_matrix(np.zeros((2, 1), dtype=np.int64))
    complex_c = ChainComplex(
        boundaries={1: d1, 2: d2}, dimensions=[0, 1, 2], cells={0: 1, 1: 2, 2: 1}
    )
    assert complex_c.homology(0) == (1, [])
    assert complex_c.homology(1) == (2, [])
    assert complex_c.homology(2) == (1, [])


def test_all_dimension_homology_and_cohomology_helpers():
    d1 = sp.csr_matrix(np.zeros((1, 2), dtype=np.int64))
    d2 = sp.csr_matrix(np.zeros((2, 1), dtype=np.int64))
    complex_c = ChainComplex(
        boundaries={1: d1, 2: d2}, dimensions=[0, 1, 2], cells={0: 1, 1: 2, 2: 1}
    )

    assert complex_c.homology() == {0: (1, []), 1: (2, []), 2: (1, [])}
    assert complex_c.cohomology() == {0: (1, []), 1: (2, []), 2: (1, [])}
    assert complex_c.rank(0) == 1
    assert complex_c.rank() == {0: 1, 1: 2, 2: 1}
    assert complex_c.betti_number(0) == 1
    assert complex_c.betti_number() == {0: 1, 1: 2, 2: 1}
    assert complex_c.betti_numbers() == {0: 1, 1: 2, 2: 1}


def test_all_dimension_homology_query_is_cached():
    d1 = sp.csr_matrix(np.zeros((1, 2), dtype=np.int64))
    d2 = sp.csr_matrix(np.zeros((2, 1), dtype=np.int64))
    complex_c = ChainComplex(
        boundaries={1: d1, 2: d2}, dimensions=[0, 1, 2], cells={0: 1, 1: 2, 2: 1}
    )

    hits_before = complex_c.cache_info()["hits"]
    _ = complex_c.homology()
    hits_after_first = complex_c.cache_info()["hits"]
    _ = complex_c.homology()
    hits_after_second = complex_c.cache_info()["hits"]

    assert hits_after_first >= hits_before
    assert hits_after_second > hits_after_first


def test_projective_plane_homology():
    # RP^2: 1 0-cell, 1 1-cell, 1 2-cell, d1 = 0, d2 = 2
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    d2 = sp.csr_matrix(np.array([[2]], dtype=np.int64))
    complex_c = ChainComplex(
        boundaries={1: d1, 2: d2}, dimensions=[0, 1, 2], cells={0: 1, 1: 1, 2: 1}
    )
    assert complex_c.homology(0) == (1, [])
    assert complex_c.homology(1) == (0, [2])
    assert complex_c.homology(2) == (0, [])
    assert complex_c.rank() == {0: 1, 1: 1, 2: 1}
    assert complex_c.betti_number() == {0: 1, 1: 0, 2: 0}


def test_cohomology_basis():
    # Ker d2^T / Im d1^T
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    d2 = sp.csr_matrix(np.array([[2]], dtype=np.int64))
    complex_c = ChainComplex(
        boundaries={1: d1, 2: d2}, dimensions=[0, 1, 2], cells={0: 1, 1: 1, 2: 1}
    )

    basis0 = complex_c.cohomology_basis(0)
    assert len(basis0) == 1

    basis1 = complex_c.cohomology_basis(1)
    assert len(basis1) == 0

    basis2 = complex_c.cohomology_basis(2)
    assert len(basis2) == 0


def test_homology_uses_cells_even_if_dimension_list_missing_degree():
    complex_c = ChainComplex(boundaries={}, dimensions=[], cells={0: 1})
    assert complex_c.homology(0) == (1, [])


def test_homology_over_q_uses_field_rank():
    d1 = sp.csr_matrix(np.array([[2]], dtype=np.int64))
    c = ChainComplex(
        boundaries={1: d1}, dimensions=[0, 1], cells={0: 1, 1: 1}, coefficient_ring="Q"
    )
    # Over Q, multiplication by 2 is invertible as map Q->Q: H_0 = 0, H_1 = 0.
    assert c.homology(0) == (0, [])
    assert c.homology(1) == (0, [])


def test_homology_over_z2_detects_mod2_kernel():
    d1 = sp.csr_matrix(np.array([[2]], dtype=np.int64))
    c = ChainComplex(
        boundaries={1: d1},
        dimensions=[0, 1],
        cells={0: 1, 1: 1},
        coefficient_ring="Z/2Z",
    )
    # Over Z/2, d1=0, so H_0 has rank 1 and H_1 has rank 1.
    assert c.homology(0) == (1, [])
    assert c.homology(1) == (1, [])


def test_cohomology_basis_over_z2():
    d1 = sp.csr_matrix(np.array([[2]], dtype=np.int64))
    c = ChainComplex(
        boundaries={1: d1},
        dimensions=[0, 1],
        cells={0: 1, 1: 1},
        coefficient_ring="Z/2Z",
    )
    b1 = c.cohomology_basis(1)
    assert len(b1) == 1
    assert int(b1[0][0]) in {0, 1}


def test_cohomology_basis_over_q_prefers_julia(monkeypatch):
    d1 = sp.csr_matrix(np.array([[2]], dtype=np.int64))
    c = ChainComplex(
        boundaries={1: d1}, dimensions=[0, 1], cells={0: 1, 1: 1}, coefficient_ring="Q"
    )

    monkeypatch.setattr(julia_engine, "available", True)
    called = {"count": 0}

    def _fake_basis(d_np1, d_n, cn_size=None):
        called["count"] += 1
        assert cn_size == 1
        return [np.array([1], dtype=np.int64)]

    monkeypatch.setattr(julia_engine, "compute_sparse_cohomology_basis", _fake_basis)
    b1 = c.cohomology_basis(1)
    assert called["count"] == 1
    assert len(b1) == 1
    assert int(b1[0][0]) == 1


def test_simplicial_complex_from_maximal_simplex_builds_full_closure():
    sc = SimplicialComplex.from_maximal_simplices([(0, 1, 2)])

    assert sc.dimension == 2
    assert sc.is_closed_under_faces()
    assert sc.f_vector() == {0: 3, 1: 3, 2: 1}
    assert sc.euler_characteristic() == 1

    d1 = sc.boundary_matrix(1)
    d2 = sc.boundary_matrix(2)
    assert d1.shape == (3, 3)
    assert d2.shape == (3, 1)

    cc = sc.chain_complex()
    assert cc.homology(0) == (1, [])
    assert cc.homology(1) == (0, [])
    assert cc.homology(2) == (0, [])


def test_simplicial_complex_can_skip_face_closure_when_explicitly_requested():
    sc = SimplicialComplex.from_simplices(
        [(0, 1), (1, 2), (0, 2)], close_under_faces=False
    )

    assert sc.dimension == 1
    assert sc.f_vector() == {1: 3}
    assert not sc.is_closed_under_faces()


def test_cohomology_basis_over_prime_mod_prefers_julia(monkeypatch):
    d1 = sp.csr_matrix(np.array([[2]], dtype=np.int64))
    c = ChainComplex(
        boundaries={1: d1},
        dimensions=[0, 1],
        cells={0: 1, 1: 1},
        coefficient_ring="Z/2Z",
    )

    monkeypatch.setattr(julia_engine, "available", True)
    called = {"count": 0}

    def _fake_basis_mod_p(d_np1, d_n, p, cn_size=None):
        called["count"] += 1
        assert p == 2
        assert cn_size == 1
        return [np.array([1], dtype=np.int64)]

    monkeypatch.setattr(
        julia_engine, "compute_sparse_cohomology_basis_mod_p", _fake_basis_mod_p
    )
    b1 = c.cohomology_basis(1)
    assert called["count"] == 1
    assert len(b1) == 1
    assert int(b1[0][0]) == 1


def test_cwcomplex_propagates_coefficient_ring():
    cw = CWComplex(cells={0: 1}, attaching_maps={}, coefficient_ring="Q")
    cc = cw.cellular_chain_complex()
    assert cc.coefficient_ring == "Q"


def test_simplicial_cellular_chain_complex_accepts_ring_override():
    sc = SimplicialComplex.from_maximal_simplices([(0, 1, 2)], coefficient_ring="Z")
    cc_q = sc.cellular_chain_complex(coefficient_ring="Q")
    assert cc_q.coefficient_ring == "Q"


def test_cw_cellular_chain_complex_accepts_ring_override():
    cw = CWComplex(cells={0: 1}, attaching_maps={}, coefficient_ring="Z")
    cc_z2 = cw.cellular_chain_complex(coefficient_ring="Z/2Z")
    assert cc_z2.coefficient_ring == "Z/2Z"


def test_simplicial_chain_complex_cache_separates_coefficient_rings():
    sc = SimplicialComplex.from_maximal_simplices([(0, 1, 2)], coefficient_ring="Z")
    cc_z = sc.cellular_chain_complex()
    cc_q = sc.cellular_chain_complex(coefficient_ring="Q")
    cc_z2 = sc.cellular_chain_complex(coefficient_ring="Z/2Z")
    assert cc_z.coefficient_ring == "Z"
    assert cc_q.coefficient_ring == "Q"
    assert cc_z2.coefficient_ring == "Z/2Z"


def test_cwcomplex_normalizes_dimensions_and_boundary_views():
    d1 = sp.csr_matrix(np.array([[0, 1]], dtype=np.int64))
    cw = CWComplex(
        cells={1: 1, 0: 2},
        attaching_maps={1: d1},
        dimensions=[1, 0],
        coefficient_ring="Z",
    )

    assert cw.dimensions == [0, 1]
    assert cw.boundary_matrix(1).shape == (1, 2)
    assert cw.cellular_chain_complex().dimensions == [0, 1]


def test_zmod_supports_composite_modulus():
    c = ChainComplex(
        boundaries={}, dimensions=[0], cells={0: 1}, coefficient_ring="Z/4Z"
    )
    assert c.homology(0) == (1, [])


def test_zmod_composite_modulus_tensor_contribution_from_integral_torsion():
    # Integral model with H_1 = Z_4: one 1-cell and one 2-cell with d2 = [4].
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    d2 = sp.csr_matrix(np.array([[4]], dtype=np.int64))
    c = ChainComplex(
        boundaries={1: d1, 2: d2},
        dimensions=[0, 1, 2],
        cells={0: 1, 1: 1, 2: 1},
        coefficient_ring="Z/4Z",
    )
    # H_1(X; Z/4) gets a Z/4 summand from Z_4 tensor Z_4.
    assert c.homology(1) == (1, [])
    assert c.cohomology(1) == (1, [])


@pytest.mark.parametrize(
    "name, builder, bettis, torsion, euler",
    get_surfaces() if "get_surfaces" in globals() else [],
)
def test_discrete_surface_homology(name, builder, bettis, torsion, euler):
    st = builder()
    complex_c = to_complex(st)
    from pysurgery.bridge.julia_bridge import julia_engine

    for dim in range(3):
        h_rank, h_torsion = complex_c.homology(dim)
        assert h_rank == bettis.get(dim, 0)
        if julia_engine.available:
            assert set(h_torsion) == set(torsion.get(dim, []))

    chi = sum((-1) ** dim * complex_c.cells.get(dim, 0) for dim in complex_c.cells)
    assert chi == euler


@pytest.mark.parametrize(
    "name, builder, bettis, torsion, euler",
    get_3_manifolds() if "get_3_manifolds" in globals() else [],
)
def test_discrete_3_manifold_homology(name, builder, bettis, torsion, euler):
    st = builder()
    complex_c = to_complex(st)
    for dim in range(4):
        h_rank, h_torsion = complex_c.homology(dim)
        assert h_rank == bettis.get(dim, 0)
        assert set(h_torsion) == set(torsion.get(dim, []))


def test_simplicial_boundary_matrix_is_cached(monkeypatch):
    sc = SimplicialComplex.from_maximal_simplices([(0, 1, 2)])
    calls = {"count": 0}
    original = complexes._boundary_matrix_from_simplices

    def _wrapped(source, target):
        calls["count"] += 1
        return original(source, target)

    monkeypatch.setattr(complexes, "_boundary_matrix_from_simplices", _wrapped)
    _ = sc.boundary_matrix(2)
    _ = sc.boundary_matrix(2)
    assert calls["count"] == 1
    info = sc.cache_info()
    assert info["hits"] >= 1


def test_chain_homology_and_cohomology_are_cached():
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    d2 = sp.csr_matrix(np.array([[2]], dtype=np.int64))
    c = ChainComplex(
        boundaries={1: d1, 2: d2},
        dimensions=[0, 1, 2],
        cells={0: 1, 1: 1, 2: 1},
    )
    h1a = c.homology(1)
    h1b = c.homology(1)
    r1a = c.rank(1)
    r1b = c.rank(1)
    b1h_a = c.betti_number(1)
    b1h_b = c.betti_number(1)
    c1a = c.cohomology(1)
    c1b = c.cohomology(1)
    b1a = c.cohomology_basis(1)
    b1b = c.cohomology_basis(1)

    assert h1a == h1b
    assert r1a == r1b
    assert b1h_a == b1h_b
    assert c1a == c1b
    assert len(b1a) == len(b1b)
    info = c.cache_info()
    assert info["hits"] >= 2
    assert any("homology" in " ".join(key) for key in info["keys"])
    assert any("rank" in " ".join(key) for key in info["keys"])
    assert any("betti_number" in " ".join(key) for key in info["keys"])


def test_cache_clear_forces_recompute(monkeypatch):
    d1 = sp.csr_matrix(np.array([[2]], dtype=np.int64))
    c = ChainComplex(
        boundaries={1: d1},
        dimensions=[0, 1],
        cells={0: 1, 1: 1},
    )
    calls = {"count": 0}
    original = complexes.get_sparse_snf_diagonal

    def _wrapped(matrix):
        calls["count"] += 1
        return original(matrix)

    monkeypatch.setattr(complexes, "get_sparse_snf_diagonal", _wrapped)
    _ = c.homology(0)
    _ = c.homology(0)
    assert calls["count"] == 1
    c.clear_cache("chain")
    _ = c.homology(0)
    assert calls["count"] == 2


class _FakeSimplexTree:
    def __init__(self):
        self._entries = []

    def insert(self, simplex, filtration=0.0):
        self._entries.append((tuple(simplex), float(filtration)))

    def get_filtration(self):
        return sorted(self._entries, key=lambda item: (len(item[0]), item[0], item[1]))


def test_simplicial_complex_gudhi_roundtrip(monkeypatch):
    st_in = _FakeSimplexTree()
    st_in.insert([0], 0.0)
    st_in.insert([1], 0.0)
    st_in.insert([2], 0.0)
    st_in.insert([0, 1], 0.1)
    st_in.insert([1, 2], 0.2)
    st_in.insert([0, 2], 0.3)
    st_in.insert([0, 1, 2], 0.5)

    sc = SimplicialComplex.from_gudhi_simplex_tree(st_in, include_filtration=True)
    assert sc.f_vector() == {0: 3, 1: 3, 2: 1}
    assert sc.filtration[(0, 1, 2)] == pytest.approx(0.5)

    fake_gudhi = types.ModuleType("gudhi")
    fake_gudhi.SimplexTree = _FakeSimplexTree
    monkeypatch.setitem(sys.modules, "gudhi", fake_gudhi)

    st_out = sc.to_gudhi_simplex_tree(use_filtration=True)
    out_entries = {tuple(s): f for s, f in st_out.get_filtration()}
    assert (0, 1, 2) in out_entries
    assert out_entries[(0, 1, 2)] == pytest.approx(0.5)


def test_cw_cache_for_boundary_and_chain_complex():
    d1 = sp.csr_matrix(np.array([[0, 1]], dtype=np.int64))
    cw = CWComplex(cells={0: 2, 1: 1}, attaching_maps={1: d1}, dimensions=[0, 1])
    _ = cw.boundary_matrices()
    _ = cw.boundary_matrices()
    _ = cw.cellular_chain_complex()
    _ = cw.cellular_chain_complex()
    info = cw.cache_info()
    assert info["hits"] >= 1


def test_cw_chain_complex_cache_separates_coefficient_rings():
    cw = CWComplex(cells={0: 1}, attaching_maps={}, coefficient_ring="Z")
    cc_z = cw.cellular_chain_complex()
    cc_q = cw.cellular_chain_complex(coefficient_ring="Q")
    assert cc_z.coefficient_ring == "Z"
    assert cc_q.coefficient_ring == "Q"


