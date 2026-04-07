import numpy as np
import pytest
try:
    from tests.discrete_surface_data import build_torus, to_complex
except ImportError:
    pass
from pysurgery.core.cup_product import alexander_whitney_cup

def test_alexander_whitney_cup():
    simplices_p_plus_q = [(0, 1, 2)]
    simplex_to_idx_p = {(0, 1): 0, (1, 2): 1, (0, 2): 2}
    simplex_to_idx_q = {(0, 1): 0, (1, 2): 1, (0, 2): 2}
    
    alpha = np.array([2, 0, 0], dtype=np.int64) 
    beta = np.array([0, 3, 0], dtype=np.int64)  
    
    cup = alexander_whitney_cup(
        alpha, beta, p=1, q=1,
        simplices_p_plus_q=simplices_p_plus_q,
        simplex_to_idx_p=simplex_to_idx_p,
        simplex_to_idx_q=simplex_to_idx_q
    )
    
    assert cup.shape == (1,)
    assert cup[0] == 6

def test_alexander_whitney_cup_empty():
    cup = alexander_whitney_cup(
        np.array([]), np.array([]), p=1, q=1,
        simplices_p_plus_q=[],
        simplex_to_idx_p={},
        simplex_to_idx_q={}
    )
    assert len(cup) == 0


def test_cup_product_torus():
    try:
        c1 = to_complex(build_torus())
        assert c1.homology(1)[0] == 2
    except NameError:
        pytest.skip("GUDHI not available")

