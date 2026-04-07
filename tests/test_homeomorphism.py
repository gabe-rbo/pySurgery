import numpy as np
import pytest
try:
    from tests.discrete_surface_data import build_tetrahedron, build_octahedron, build_icosahedron, build_torus, to_complex
    from pysurgery.homeomorphism import analyze_homeomorphism_2d
except ImportError:
    pass
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
    assert can_remove


def test_s2_models_homeomorphism():
    try:
        c1 = to_complex(build_tetrahedron())
        c2 = to_complex(build_octahedron())
        c3 = to_complex(build_icosahedron())
        
        # They should all be homeomorphic to each other
        is_homeo_1, _ = analyze_homeomorphism_2d(c1, c2)
        is_homeo_2, _ = analyze_homeomorphism_2d(c2, c3)
        assert is_homeo_1
        assert is_homeo_2
    except NameError:
        pytest.skip("GUDHI not available")

def test_s2_vs_torus_homeomorphism():
    try:
        c1 = to_complex(build_tetrahedron())
        c2 = to_complex(build_torus())
        
        is_homeo, _ = analyze_homeomorphism_2d(c1, c2)
        assert not is_homeo
    except NameError:
        pytest.skip("GUDHI not available")

