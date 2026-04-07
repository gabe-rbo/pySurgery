import numpy as np
import scipy.sparse as sp
from pysurgery.core.complexes import ChainComplex
from pysurgery.core.fundamental_group import extract_pi_1
from pysurgery.core.complexes import CWComplex

def test_klein_bottle_homology():
    # Klein bottle H_1 = Z + Z_2
    # cell structure: 1 0-cell, 2 1-cells (a,b), 1 2-cell (f)
    # boundary of f is a + b - a + b = 2b
    d1 = sp.csr_matrix(np.zeros((1, 2), dtype=np.int64))
    d2 = sp.csr_matrix(np.array([[0], [2]], dtype=np.int64))
    cc = ChainComplex(boundaries={1: d1, 2: d2}, dimensions=[0, 1, 2], cells={0: 1, 1: 2, 2: 1})
    
    assert cc.homology(0) == (1, [])
    
    from pysurgery.bridge.julia_bridge import julia_engine
    if julia_engine.available:
        r, t = cc.homology(1)
        assert r == 1
        assert t == [2]
    else:
        assert cc.homology(1)[0] == 1
        
    assert cc.homology(2) == (0, [])

def test_genus_2_surface():
    # Genus 2 surface H_1 = Z^4
    # 1 0-cell, 4 1-cells, 1 2-cell
    # boundary of f is a+b-a-b+c+d-c-d = 0
    d1 = sp.csr_matrix(np.zeros((1, 4), dtype=np.int64))
    d2 = sp.csr_matrix(np.zeros((4, 1), dtype=np.int64))
    cc = ChainComplex(boundaries={1: d1, 2: d2}, dimensions=[0, 1, 2], cells={0: 1, 1: 4, 2: 1})
    
    assert cc.homology(0) == (1, [])
    assert cc.homology(1) == (4, [])
    assert cc.homology(2) == (1, [])

def test_klein_bottle_pi_1():
    d1 = sp.csr_matrix(np.zeros((1, 2), dtype=np.int64))
    cw = CWComplex(attaching_maps={1: d1}, dimensions=[0, 1], cells={0: 1, 1: 2})
    pi = extract_pi_1(cw)
    assert len(pi.generators) == 2

def test_torus_pi_1():
    # Torus with 1 vertex, 2 loops (a, b), 1 face
    d1 = sp.csr_matrix(np.zeros((1, 2), dtype=np.int64))
    cw = CWComplex(attaching_maps={1: d1}, dimensions=[0, 1], cells={0: 1, 1: 2})
    pi = extract_pi_1(cw)
    assert len(pi.generators) == 2
    assert pi.generators == ["g_0", "g_1"]

def test_projective_plane_pi_1():
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    d2 = sp.csr_matrix(np.array([[2]], dtype=np.int64))
    cw = CWComplex(attaching_maps={1: d1, 2: d2}, dimensions=[0, 1, 2], cells={0: 1, 1: 1, 2: 1})
    pi = extract_pi_1(cw)
    assert len(pi.generators) == 1
    assert pi.generators == ["g_0"]

