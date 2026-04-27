import pytest
from hypothesis import given, settings, strategies as st
import numpy as np
from pysurgery.core.intersection_forms import IntersectionForm
from pysurgery.core.characteristic_classes import extract_pontryagin_p1, verify_hirzebruch_signature

@st.composite
def symmetric_matrices(draw, min_size=1, max_size=8):
    """Strategy to generate random symmetric integer matrices."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    # Generate upper triangular part
    data = draw(st.lists(
        st.integers(min_value=-10, max_value=10),
        min_size=(size * (size + 1)) // 2,
        max_size=(size * (size + 1)) // 2
    ))
    matrix = np.zeros((size, size), dtype=np.int64)
    idx = 0
    for i in range(size):
        for j in range(i, size):
            matrix[i, j] = matrix[j, i] = data[idx]
            idx += 1
    return matrix

@settings(max_examples=100, deadline=None)
@given(symmetric_matrices())
def test_intersection_form_symmetry_property(matrix):
    """Verify that IntersectionForm always maintains a symmetric matrix."""
    form = IntersectionForm(matrix=matrix, dimension=4)
    m = form.matrix
    assert np.all(m == m.T)

@settings(max_examples=100, deadline=None)
@given(symmetric_matrices(min_size=1, max_size=6))
def test_hirzebruch_signature_theorem_property(matrix):
    """Verify 3 * sigma = p1 for generated intersection forms."""
    # This identity is a definition/verification pair in characteristic_classes.py
    # and should always be consistent by construction or theorem.
    form = IntersectionForm(matrix=matrix, dimension=4)
    sig = form.signature()
    p1 = extract_pontryagin_p1(form)
    
    assert 3 * sig == p1
    assert verify_hirzebruch_signature(form, p1)

@settings(max_examples=50, deadline=None)
@given(symmetric_matrices(min_size=2, max_size=4))
def test_hyperbolic_stabilization_signature(matrix):
    """Verify that adding a hyperbolic pair (H) doesn't change signature."""
    form = IntersectionForm(matrix=matrix, dimension=4)
    sig_orig = form.signature()
    
    # Add H = [[0, 1], [1, 0]]
    h = np.array([[0, 1], [1, 0]], dtype=np.int64)
    size = matrix.shape[0]
    new_matrix = np.zeros((size + 2, size + 2), dtype=np.int64)
    new_matrix[:size, :size] = matrix
    new_matrix[size:, size:] = h
    
    form_stabilized = IntersectionForm(matrix=new_matrix, dimension=4)
    assert form_stabilized.signature() == sig_orig
