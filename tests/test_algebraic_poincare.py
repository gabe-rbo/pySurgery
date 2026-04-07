import numpy as np
import scipy.sparse as sp
from pysurgery.algebraic_poincare import AlgebraicPoincareComplex
from pysurgery.core.complexes import ChainComplex

def test_algebraic_poincare_dual():
    d1 = sp.csr_matrix(np.array([[-1, 1]], dtype=np.int64))
    cc = ChainComplex(boundaries={1: d1}, dimensions=[0, 1], cells={0: 2, 1: 1})
    fund_class = np.array([1], dtype=np.int64)
    apc = AlgebraicPoincareComplex(chain_complex=cc, fundamental_class=fund_class, dimension=1)
    
    dual = apc.dual_complex()
    assert 0 in dual.boundaries
    d0_dual = dual.boundaries[0]
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
