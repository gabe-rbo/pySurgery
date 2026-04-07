import numpy as np
import scipy.sparse as sp
from pysurgery.core.fundamental_group import extract_pi_1
from pysurgery.core.complexes import CWComplex

def test_extract_pi_1_trivial():
    d1 = sp.csr_matrix(np.zeros((1, 0), dtype=np.int64))
    cw = CWComplex(attaching_maps={1: d1}, dimensions=[0], cells={0: 1, 1: 0})
    pi_1 = extract_pi_1(cw)
    assert len(pi_1.generators) == 0
    assert len(pi_1.relations) == 0

def test_extract_pi_1_circle():
    # 1 vertex, 1 loop
    # d1 has 1 row (vertex 0), 1 col (edge 0). Loop boundary is 0.
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    cw = CWComplex(attaching_maps={1: d1}, dimensions=[0, 1], cells={0: 1, 1: 1})
    pi_1 = extract_pi_1(cw)
    assert len(pi_1.generators) == 1
    assert pi_1.generators == ["g_0"]
    assert len(pi_1.relations) == 0

def test_extract_pi_1_unclosed_loop():
    # 2 vertices, 1 edge connecting them
    d1 = sp.csr_matrix(np.array([[-1], [1]], dtype=np.int64))
    cw = CWComplex(attaching_maps={1: d1}, dimensions=[0, 1], cells={0: 2, 1: 1})
    pi_1 = extract_pi_1(cw)
    assert len(pi_1.generators) == 0
    assert len(pi_1.relations) == 0

def test_extract_pi_1_disc():
    # 1 vertex, 1 edge (loop), 1 face attaching to the loop
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    # d2 contains the sequence of edges. 
    # 1 face. Edge 0 is traversed once forward.
    d2 = sp.csr_matrix(np.array([[1]], dtype=np.int64))
    cw = CWComplex(attaching_maps={1: d1, 2: d2}, dimensions=[0, 1, 2], cells={0: 1, 1: 1, 2: 1})
    pi_1 = extract_pi_1(cw)
    assert len(pi_1.generators) == 1
    assert len(pi_1.relations) == 1
    # Relation should be g_0
    assert pi_1.relations[0] == ["g_0"]
