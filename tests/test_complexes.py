import numpy as np
import scipy.sparse as sp
from pysurgery.core.complexes import ChainComplex

def test_sphere_homology():
    # S^2: 1 cell of dim 0, 1 cell of dim 2, trivial boundaries
    d1 = sp.csr_matrix(np.zeros((1, 0), dtype=np.int64))
    d2 = sp.csr_matrix(np.zeros((0, 1), dtype=np.int64))
    complex_c = ChainComplex(boundaries={1: d1, 2: d2}, dimensions=[0, 1, 2], cells={0: 1, 1: 0, 2: 1})
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
    complex_c = ChainComplex(boundaries={1: d1, 2: d2}, dimensions=[0, 1, 2], cells={0: 1, 1: 2, 2: 1})
    assert complex_c.homology(0) == (1, [])
    assert complex_c.homology(1) == (2, [])
    assert complex_c.homology(2) == (1, [])

def test_projective_plane_homology():
    # RP^2: 1 0-cell, 1 1-cell, 1 2-cell, d1 = 0, d2 = 2
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    d2 = sp.csr_matrix(np.array([[2]], dtype=np.int64))
    complex_c = ChainComplex(boundaries={1: d1, 2: d2}, dimensions=[0, 1, 2], cells={0: 1, 1: 1, 2: 1})
    assert complex_c.homology(0) == (1, [])
    
    from pysurgery.bridge.julia_bridge import julia_engine
    if julia_engine.available:
        assert complex_c.homology(1) == (0, [2])
    else:
        assert complex_c.homology(1)[0] == 0
        
    assert complex_c.homology(2) == (0, [])

def test_cohomology_basis():
    # Ker d2^T / Im d1^T
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    d2 = sp.csr_matrix(np.array([[2]], dtype=np.int64))
    complex_c = ChainComplex(boundaries={1: d1, 2: d2}, dimensions=[0, 1, 2], cells={0: 1, 1: 1, 2: 1})
    
    basis0 = complex_c.cohomology_basis(0)
    assert len(basis0) == 1
    
    basis1 = complex_c.cohomology_basis(1)
    assert len(basis1) == 0

    basis2 = complex_c.cohomology_basis(2)
    assert len(basis2) == 0
