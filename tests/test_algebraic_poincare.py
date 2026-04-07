import numpy as np
import scipy.sparse as sp
import pytest
try:
    from tests.discrete_surface_data import get_surfaces, to_complex
except ImportError:
    pass
from pysurgery.algebraic_poincare import AlgebraicPoincareComplex
from pysurgery.core.complexes import ChainComplex

def test_algebraic_poincare_dual():
    d1 = sp.csr_matrix(np.array([[-1, 1]], dtype=np.int64))
    cc = ChainComplex(boundaries={1: d1}, dimensions=[0, 1], cells={0: 2, 1: 1})
    fund_class = np.array([1], dtype=np.int64)
    apc = AlgebraicPoincareComplex(chain_complex=cc, fundamental_class=fund_class, dimension=1)
    
    dual = apc.dual_complex()
    assert 1 in dual.boundaries
    d0_dual = dual.boundaries[1]
    assert d0_dual.shape == (2, 1)
    assert d0_dual.toarray()[0, 0] == -1
    assert d0_dual.toarray()[1, 0] == 1

def test_cap_product():
    d1 = sp.csr_matrix(np.array([[-1, 1]], dtype=np.int64))
    cc = ChainComplex(boundaries={1: d1}, dimensions=[0, 1], cells={0: 2, 1: 1})
    fund_class = np.array([1], dtype=np.int64)
    
    psi_0 = np.array([[1, 0]], dtype=np.int64)
    apc = AlgebraicPoincareComplex(chain_complex=cc, fundamental_class=fund_class, dimension=1, psi={0: psi_0})
    
    cohom_class = np.array([1, 1])
    res = apc.cap_product(cohom_class, 0)
    assert np.array_equal(res, np.array([1]))


@pytest.mark.parametrize("name, builder, bettis, torsion, euler", get_surfaces() if 'get_surfaces' in globals() else [])
def test_poincare_duality_betti(name, builder, bettis, torsion, euler):
    st = builder()
    c = to_complex(st)
    
    dim_max = max(c.cells.keys())
    # S2, Torus are orientable. RP2 is non-orientable.
    orientable = (name in ["Tetrahedron", "Octahedron", "Icosahedron", "Torus", "S3"])
    if orientable:
        for k in range(dim_max + 1):
            hk_rank, _ = c.homology(k)
            hnk_rank, _ = c.homology(dim_max - k)
            # modulo torsion, ranks should match (H_k = H^{n-k} = Free(H_{n-k}) + Tors(H_{n-k-1}))
            # Free(H_k) = Free(H_{n-k})
            assert hk_rank == hnk_rank

