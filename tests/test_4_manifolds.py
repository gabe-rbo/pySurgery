import numpy as np
from pysurgery.core.intersection_forms import IntersectionForm
from pysurgery.core.characteristic_classes import extract_stiefel_whitney_w2 as wu_class, extract_pontryagin_p1 as pontryagin_class
from pysurgery.homeomorphism import analyze_homeomorphism_4d

def test_CP2():
    Q = np.array([[1]])
    form = IntersectionForm(matrix=Q, dimension=4)
    assert form.signature() == 1
    assert not form.is_even()
    assert form.type() == "I"
    assert pontryagin_class(form) == 3
    assert np.array_equal(wu_class(form), np.array([1]))

def test_anti_CP2():
    Q = np.array([[-1]])
    form = IntersectionForm(matrix=Q, dimension=4)
    assert form.signature() == -1
    assert not form.is_even()
    assert form.type() == "I"
    assert pontryagin_class(form) == -3
    assert np.array_equal(wu_class(form), np.array([1]))

def test_S2_times_S2():
    Q = np.array([[0, 1], [1, 0]])
    form = IntersectionForm(matrix=Q, dimension=4)
    assert form.signature() == 0
    assert form.is_even()
    assert form.type() == "II"
    assert pontryagin_class(form) == 0
    assert np.array_equal(wu_class(form), np.array([0, 0]))

def test_K3_surface():
    e8 = np.array([
        [2, -1, 0, 0, 0, 0, 0, 0],
        [-1, 2, -1, 0, 0, 0, 0, 0],
        [0, -1, 2, -1, 0, 0, 0, -1],
        [0, 0, -1, 2, -1, 0, 0, 0],
        [0, 0, 0, -1, 2, -1, 0, 0],
        [0, 0, 0, 0, -1, 2, -1, 0],
        [0, 0, 0, 0, 0, -1, 2, 0],
        [0, 0, -1, 0, 0, 0, 0, 2]
    ])
    h = np.array([[0, 1], [1, 0]])
    from scipy.linalg import block_diag
    k3_matrix = block_diag(-e8, -e8, h, h, h)
    form = IntersectionForm(matrix=k3_matrix, dimension=4)
    
    assert form.rank() == 22
    assert form.signature() == -16
    assert form.is_even()
    assert form.type() == "II"
    assert pontryagin_class(form) == -48
    assert np.all(wu_class(form) == 0)

def test_K3_homeomorphism():
    e8 = np.array([
        [2, -1, 0, 0, 0, 0, 0, 0],
        [-1, 2, -1, 0, 0, 0, 0, 0],
        [0, -1, 2, -1, 0, 0, 0, -1],
        [0, 0, -1, 2, -1, 0, 0, 0],
        [0, 0, 0, -1, 2, -1, 0, 0],
        [0, 0, 0, 0, -1, 2, -1, 0],
        [0, 0, 0, 0, 0, -1, 2, 0],
        [0, 0, -1, 0, 0, 0, 0, 2]
    ])
    h = np.array([[0, 1], [1, 0]])
    from scipy.linalg import block_diag
    k3_matrix = block_diag(-e8, -e8, h, h, h)
    form1 = IntersectionForm(matrix=k3_matrix, dimension=4)
    form2 = IntersectionForm(matrix=k3_matrix.copy(), dimension=4)
    
    is_homeo, reason = analyze_homeomorphism_4d(form1, form2, ks1=0, ks2=0, simply_connected=True)
    assert is_homeo
    assert "SUCCESS" in reason
