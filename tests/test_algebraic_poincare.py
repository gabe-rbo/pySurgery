import numpy as np
import scipy.sparse as sp
import pytest

from discrete_surface_data import get_surfaces, to_complex
from pysurgery.algebraic_poincare import AlgebraicPoincareComplex
from pysurgery.core.complexes import ChainComplex
from pysurgery.core.exceptions import DimensionError


def test_algebraic_poincare_dual():
    d1 = sp.csr_matrix(np.array([[-1, 1]], dtype=np.int64))
    cc = ChainComplex(boundaries={1: d1}, dimensions=[0, 1], cells={0: 2, 1: 1})
    fund_class = np.array([1], dtype=np.int64)
    apc = AlgebraicPoincareComplex(
        chain_complex=cc, fundamental_class=fund_class, dimension=1
    )

    assert apc.dimension == 1
    dual = apc.dual_complex()
    assert dual.dimensions == [0, 1]


def test_cap_product():
    # S1 case
    d1 = sp.csr_matrix(np.array([[-1, 1]], dtype=np.int64))
    cc = ChainComplex(boundaries={1: d1}, dimensions=[0, 1], cells={0: 2, 1: 1})
    fund_class = np.array([1], dtype=np.int64)
    # cohomology class at dim 0 (2 vertices)
    cohom_class = np.array([1, 1], dtype=np.int64)
    
    # Cap product of [S1] with 1 in H^0 is [S1] in H_1
    # n=1, k=0 -> n-k=1. psi_0: C^0 -> C_1. size(C^0)=2, size(C_1)=1. shape (1, 2)
    psi0 = np.array([[1, 0]], dtype=np.int64)
    apc = AlgebraicPoincareComplex(
        chain_complex=cc,
        fundamental_class=fund_class,
        dimension=1,
        psi={0: psi0}
    )
    # cap_product(cohomology_class, degree k)
    cap = apc.cap_product(cohom_class, 0)
    assert cap.shape == (1,)


def test_cap_product_requires_defined_psi():
    d1 = sp.csr_matrix(np.array([[-1, 1]], dtype=np.int64))
    cc = ChainComplex(boundaries={1: d1}, dimensions=[0, 1], cells={0: 2, 1: 1})
    apc = AlgebraicPoincareComplex(
        chain_complex=cc,
        fundamental_class=np.array([1]),
        dimension=1,
        psi={} # empty psi
    )
    with pytest.raises(DimensionError, match="Diagonal map psi_0 not defined"):
        apc.cap_product(np.array([1, 1]), 0)


def test_cap_product_dimension_mismatch_raises():
    d1 = sp.csr_matrix(np.array([[-1, 1]], dtype=np.int64))
    cc = ChainComplex(boundaries={1: d1}, dimensions=[0, 1], cells={0: 2, 1: 1})
    # n=1, k=0 -> n-k=1. psi_0 shape (1, 2)
    apc = AlgebraicPoincareComplex(
        chain_complex=cc,
        fundamental_class=np.array([1]),
        dimension=1,
        psi={0: np.array([[1, 0]])}
    )
    with pytest.raises(DimensionError):
        apc.cap_product(np.array([1]), 0) # should be length 2


def test_psi_default_is_not_shared_between_instances():
    d1 = sp.csr_matrix(np.array([[-1, 1]], dtype=np.int64))
    cc = ChainComplex(boundaries={1: d1}, dimensions=[0, 1], cells={0: 2, 1: 1})
    apc1 = AlgebraicPoincareComplex(
        chain_complex=cc, fundamental_class=np.array([1]), dimension=1
    )
    apc2 = AlgebraicPoincareComplex(
        chain_complex=cc, fundamental_class=np.array([1]), dimension=1
    )
    assert apc1.psi is not apc2.psi


def test_fundamental_class_length_mismatch_raises():
    d1 = sp.csr_matrix(np.array([[-1, 1]], dtype=np.int64))
    cc = ChainComplex(boundaries={1: d1}, dimensions=[0, 1], cells={0: 2, 1: 1})
    with pytest.raises(DimensionError):
        AlgebraicPoincareComplex(
            chain_complex=cc, fundamental_class=np.array([1, 1]), dimension=1
        )


def test_fundamental_class_must_be_cycle_when_boundary_known():
    # Chain complex with d1: [1, 1] mapping 1 -> 1
    d1 = sp.csr_matrix(np.array([[1, 1]], dtype=np.int64))
    # Dim 1 has 2 cells, dim 0 has 1 cell.
    cc = ChainComplex(boundaries={1: d1}, dimensions=[0, 1], cells={0: 1, 1: 2})
    # Fundamental class [1, 0] is NOT a cycle because d1([1, 0]) = 1 != 0
    with pytest.raises(DimensionError, match="must be a cycle"):
        AlgebraicPoincareComplex(
            chain_complex=cc, fundamental_class=np.array([1, 0]), dimension=1
        )


def test_psi_shape_checked_at_construction():
    d1 = sp.csr_matrix(np.array([[-1, 1]], dtype=np.int64))
    cc = ChainComplex(boundaries={1: d1}, dimensions=[0, 1], cells={0: 2, 1: 1})
    # n=1, k=0 -> n-k=1. psi_0 shape (1, 2)
    with pytest.raises(DimensionError):
        AlgebraicPoincareComplex(
            chain_complex=cc,
            fundamental_class=np.array([1]),
            dimension=1,
            psi={0: np.array([[1], [0]], dtype=np.int64)}, # (2, 1) instead of (1, 2)
        )


@pytest.mark.parametrize(
    "name, builder, bettis, torsion, euler",
    get_surfaces(),
)
def test_poincare_duality_betti(name, builder, bettis, torsion, euler):
    st = builder()
    c = to_complex(st)

    dim_max = c.dimension
    # S2, Torus are orientable. RP2 is non-orientable.
    orientable = name in ["Tetrahedron", "Octahedron", "Icosahedron", "Torus", "S3", "Cube"] or "Torus" in name
    if orientable:
        for k in range(dim_max + 1):
            hk_rank, _ = c.homology(k)
            hnk_rank, _ = c.homology(dim_max - k)
            # modulo torsion, ranks should match
            assert hk_rank == hnk_rank
