import numpy as np
import scipy.sparse as sp
import pytest

try:
    from tests.discrete_surface_data import get_surfaces, get_3_manifolds, to_complex
except ImportError:
    pass
from pysurgery.core.complexes import ChainComplex, CWComplex
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
