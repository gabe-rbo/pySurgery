"""Tests for ChainComplex, CWComplex, and SimplicialComplex.

Overview:
    Validates foundational data structures for topological computation, 
    including boundary matrix construction, homology/cohomology calculation, 
    and skeletal closure properties.

Key Concepts:
    - **ChainComplex**: Linear algebra of boundary operators.
    - **SimplicialComplex**: Combinatorial representation of triangulated spaces.
    - **CWComplex**: Generalization using cell attaching maps.
"""
import numpy as np
import scipy.sparse as sp
import pytest
import sys
import types

from discrete_surface_data import get_surfaces, get_3_manifolds, to_complex
from pysurgery.core.complexes import ChainComplex, CWComplex, SimplicialComplex
import pysurgery.core.complexes as complexes
from pysurgery.bridge.julia_bridge import julia_engine


def test_sphere_homology():
    """Verify homology computation for a minimal S² model.

    What is Being Computed?:
        Computes H_n(S²) for n=0, 1, 2.

    Algorithm:
        1. Define boundary maps for a 0-cell and a 2-cell.
        2. Instantiate ChainComplex.
        3. Assert H₀=ℤ, H₁=0, H₂=ℤ.
    """
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
    get_surfaces(),
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
    get_3_manifolds(),
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
    # Clear private cache to force a new computation
    object.__setattr__(sc, "_boundaries_cache", {})
    sc.clear_cache()

    calls = {"count": 0}
    original = complexes._boundary_matrix_from_simplices_with_maps

    def _wrapped(source, target_map):
        calls["count"] += 1
        return original(source, target_map)

    monkeypatch.setattr(complexes, "_boundary_matrix_from_simplices_with_maps", _wrapped)
    # We must also ensure it doesn't use the Julia-bridge if available for this specific test
    monkeypatch.setattr(complexes.julia_engine, "available", False)

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

    def _wrapped(matrix, **kwargs):
        calls["count"] += 1
        return original(matrix, **kwargs)

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

    def dimension(self):
        if not self._entries:
            return 0
        return max(len(s) - 1 for s, _ in self._entries)


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


def test_chain_complex_topological_invariants():
    # S^1 as a Simplicial Complex
    sc = SimplicialComplex.from_maximal_simplices([(0, 1), (1, 2), (2, 0)])
    cc = sc.chain_complex()
    
    inv = cc.topological_invariants()
    
    assert inv["euler_characteristic"] == 0
    assert inv["betti_numbers"][0] == 1
    assert inv["betti_numbers"][1] == 1
    assert inv["homology"][0] == (1, [])
    assert inv["homology"][1] == (1, [])
    assert inv["cohomology"][0] == (1, [])
    assert inv["cohomology"][1] == (1, [])
    assert inv["coefficient_ring"] == "Z"


def test_euler_characteristic_formula_consistency():
    """
    Verify the Euler-Poincaré formula: 
    sum (-1)^n * c_n = sum (-1)^n * beta_n
    where c_n is the number of n-cells and beta_n is the n-th Betti number.
    """
    # 1. A single point (0-simplex)
    sc_point = SimplicialComplex.from_simplices([[0]])
    cc_point = sc_point.chain_complex()
    chi_cells = cc_point.euler_characteristic()
    betti = cc_point.betti_numbers()
    chi_betti = sum((-1)**n * b for n, b in betti.items())
    assert chi_cells == 1
    assert chi_cells == chi_betti

    # 2. A triangle (2-simplex with closure)
    sc_tri = SimplicialComplex.from_maximal_simplices([(0, 1, 2)])
    cc_tri = sc_tri.chain_complex()
    chi_cells = cc_tri.euler_characteristic()
    betti = cc_tri.betti_numbers()
    chi_betti = sum((-1)**n * b for n, b in betti.items())
    assert chi_cells == 1 # Contractible
    assert chi_cells == chi_betti

    # 3. A circle (S^1)
    sc_s1 = SimplicialComplex.from_maximal_simplices([(0, 1), (1, 2), (2, 0)])
    cc_s1 = sc_s1.chain_complex()
    chi_cells = cc_s1.euler_characteristic()
    betti = cc_s1.betti_numbers()
    chi_betti = sum((-1)**n * b for n, b in betti.items())
    assert chi_cells == 0
    assert chi_cells == chi_betti

    # 4. A sphere (boundary of a tetrahedron, S^2)
    sc_s2 = SimplicialComplex.from_maximal_simplices([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
    cc_s2 = sc_s2.chain_complex()
    chi_cells = cc_s2.euler_characteristic()
    betti = cc_s2.betti_numbers()
    chi_betti = sum((-1)**n * b for n, b in betti.items())
    assert chi_cells == 2
    assert chi_cells == chi_betti

    # 5. A Torus (T^2) via CW complex model
    # T^2 = 1 vertex, 2 edges (a, b), 1 face (aba^-1b^-1)
    # d1: Z^2 -> Z^1 is zero (all edges start/end at same vertex)
    # d2: Z^1 -> Z^2 is zero (a+b-a-b = 0)
    d1 = sp.csr_matrix((1, 2), dtype=np.int64)
    d2 = sp.csr_matrix((2, 1), dtype=np.int64)
    cw_torus = CWComplex(
        cells={0: 1, 1: 2, 2: 1},
        attaching_maps={1: d1, 2: d2},
        dimensions=[0, 1, 2]
    )
    cc_torus = cw_torus.cellular_chain_complex()
    chi_cells = cc_torus.euler_characteristic()
    betti = cc_torus.betti_numbers()
    chi_betti = sum((-1)**n * b for n, b in betti.items())
    assert chi_cells == 0 # 1 - 2 + 1
    assert chi_cells == chi_betti


def test_simplicial_complex_mathematical_validity():
    # 1. Valid Sphere (Boundary of tetrahedron)
    sc = SimplicialComplex.from_maximal_simplices([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
    v = sc.verify_structure()
    assert v["valid"] is True
    assert v["is_closed"] is True
    assert v["is_canonical"] is True

    # 2. Manual invalid construction: missing faces
    # Constructing a 2-simplex manually without its edges in the field
    sc_invalid = SimplicialComplex(simplices={0: [(0,), (1,), (2,)], 1: [], 2: [(0, 1, 2)]})
    v_inv = sc_invalid.verify_structure()
    assert v_inv["is_closed"] is False
    assert v_inv["valid"] is False
    assert any("Downward closure" in issue for issue in v_inv["issues"])

    # 3. Boundary consistency d^2 = 0 check on a 3-manifold
    # A single tetrahedron
    sc_tetra = SimplicialComplex.from_maximal_simplices([(0, 1, 2, 3)])
    v_tetra = sc_tetra.verify_structure()
    assert v_tetra["valid"] is True
    
    # Check d1 * d2 = 0
    d1 = sc_tetra.boundary_matrix(1).toarray()
    d2 = sc_tetra.boundary_matrix(2).toarray()
    np.testing.assert_array_equal(d1 @ d2, 0)

    # Check d2 * d3 = 0
    d3 = sc_tetra.boundary_matrix(3).toarray()
    np.testing.assert_array_equal(d2 @ d3, 0)


def test_quick_mapper_validity_and_euler():
    """
    Test quick_mapper on a mid-size complex (100 vertices), 
    verify its structural validity and Euler-Poincaré consistency.
    """
    import random
    np.random.seed(42)
    random.seed(42)

    # 1. Generate a random geometric graph complex
    n_vertices = 100
    points = np.random.rand(n_vertices, 2)
    
    # Simple proximity-based edges
    edges = []
    threshold = 0.15
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            if np.linalg.norm(points[i] - points[j]) < threshold:
                edges.append((i, j))
    
    sc_orig = SimplicialComplex.from_simplices(edges, close_under_faces=True)
    
    # 2. Run quick_mapper to simplify the topology
    # We use preserve_topology=True to ensure Betti numbers (and thus Chi) remain the same
    # though the complex will have fewer vertices/edges.
    sc_simple, _ = sc_orig.simplify()

    # 3. Verify mathematical validity
    validity = sc_simple.verify_structure()
    assert validity["valid"], f"QuickMapper produced invalid complex: {validity['issues']}"
    assert validity["is_closed"] is True
    assert validity["is_canonical"] is True

    # 4. Verify Euler-Poincaré Formula: Chi(cells) == Chi(Betti)
    cc = sc_simple.chain_complex()
    chi_cells = cc.euler_characteristic()
    
    betti = cc.betti_numbers()
    chi_betti = sum((-1)**n * b for n, b in betti.items())
    
    assert chi_cells == chi_betti, f"Euler formula failed: {chi_cells} != {chi_betti}"

    # 5. If preserve_topology was True, it should match original Chi
    chi_orig = sc_orig.chain_complex().euler_characteristic()
    assert chi_cells == chi_orig, f"Topology not preserved: simplified Chi {chi_cells} != original Chi {chi_orig}"


def test_simplicial_complex_expand():
    # 1. Test on a cycle (S^1 skeleton)
    # A 4-cycle should NOT gain any 2-simplices when expanded, 
    # because it has no 3-cliques.
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    sc = SimplicialComplex.from_simplices(edges)
    sc_expanded = sc.expand(max_dim=2)
    
    assert sc_expanded.count_simplices(2) == 0
    assert sc_expanded.chain_complex().homology(1)[0] == 1 # Still a cycle

    # 2. Test on a complete graph K_4
    # A K_4 should expand into a full tetrahedron (one 3-simplex, four 2-simplices, etc.)
    k4_edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    sc_k4 = SimplicialComplex.from_simplices(k4_edges)
    sc_tetra = sc_k4.expand(max_dim=3)
    
    assert sc_tetra.count_simplices(3) == 1
    assert sc_tetra.count_simplices(2) == 4
    assert sc_tetra.count_simplices(1) == 6
    assert sc_tetra.count_simplices(0) == 4
    
    inv = sc_tetra.chain_complex().topological_invariants()
    assert inv["betti_numbers"][0] == 1
    assert inv["betti_numbers"][1] == 0
    assert inv["betti_numbers"][2] == 0
    assert inv["betti_numbers"][3] == 0
    assert inv["euler_characteristic"] == 1 # Contractible


def test_simplicial_complex_expand_default_max_dim():
    # K_3 triangle graph
    edges = [(0, 1), (1, 2), (2, 0)]
    sc = SimplicialComplex.from_simplices(edges)
    
    # Expand without citing max_dim
    # Default should be max_v = 2
    sc_expanded = sc.expand()
    
    assert sc_expanded.count_simplices(2) == 1
    assert sc_expanded.dimension == 2
    assert sc_expanded.verify_structure()["valid"] is True
