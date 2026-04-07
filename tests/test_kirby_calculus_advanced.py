import numpy as np
import pytest
from pysurgery.core.kirby_calculus import KirbyDiagram
from pysurgery.core.exceptions import KirbyMoveError

def test_hopf_link_slide():
    linking = np.array([[0, 1], [1, 0]])
    framings = np.array([0, 0])
    diagram = KirbyDiagram(linking_matrix=linking, framings=framings)
    
    new_diag = diagram.handle_slide(source_idx=0, target_idx=1)
    
    expected = np.array([[2, 1], [1, 0]])
    assert np.array_equal(new_diag.linking_matrix, expected)
    assert np.array_equal(new_diag.framings, np.array([2, 0]))

def test_kirby_blowup_signature():
    linking = np.array([[0]])
    diag = KirbyDiagram(linking_matrix=linking, framings=np.array([0]))
    
    diag_plus = diag.blow_up(sign=1)
    assert diag_plus.extract_intersection_form().signature() == 1
    assert diag_plus.extract_intersection_form().rank() == 2
    
    diag_minus = diag.blow_up(sign=-1)
    assert diag_minus.extract_intersection_form().signature() == -1
    assert diag_minus.extract_intersection_form().rank() == 2

def test_invalid_kirby_moves():
    linking = np.array([[0, 1], [1, 0]])
    diagram = KirbyDiagram(linking_matrix=linking, framings=np.array([0, 0]))
    
    with pytest.raises(KirbyMoveError):
        diagram.blow_up(sign=2)
        
    with pytest.raises(KirbyMoveError):
        diagram.handle_slide(source_idx=0, target_idx=0)
