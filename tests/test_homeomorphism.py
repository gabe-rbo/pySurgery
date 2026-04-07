import numpy as np
from pysurgery.homeomorphism import analyze_homeomorphism_4d, surgery_to_remove_impediments
from pysurgery.core.intersection_forms import IntersectionForm

def test_analyze_homeomorphism_4d_indefinite():
    matrix1 = np.array([[0, 1], [1, 0]])
    form1 = IntersectionForm(matrix=matrix1, dimension=4)
    
    matrix2 = np.array([[0, 1], [1, 0]])
    form2 = IntersectionForm(matrix=matrix2, dimension=4)
    
    is_homeo, reason = analyze_homeomorphism_4d(form1, form2)
    assert is_homeo
    assert "SUCCESS" in reason

def test_analyze_homeomorphism_4d_impediment():
    matrix1 = np.array([[0, 1], [1, 0]])
    form1 = IntersectionForm(matrix=matrix1, dimension=4)
    
    matrix2 = np.array([[1, 0], [0, -1]])
    form2 = IntersectionForm(matrix=matrix2, dimension=4)
    
    is_homeo, reason = analyze_homeomorphism_4d(form1, form2)
    assert not is_homeo
    assert "Parity mismatch" in reason

def test_surgery_to_remove_impediments():
    matrix1 = np.array([[1, 0], [0, 1]]) # sig = 2
    form1 = IntersectionForm(matrix=matrix1, dimension=4)
    
    can_remove, reason = surgery_to_remove_impediments(form1, 10)
    assert can_remove
    
    can_remove, reason = surgery_to_remove_impediments(form1, 4)
    assert not can_remove
