import numpy as np
import pytest
from hypothesis import given, strategies as st
from pysurgery.core.intersection_forms import IntersectionForm

@given(st.lists(st.integers(min_value=-10, max_value=10), min_size=4, max_size=4))
def test_intersection_form_signature(data):
    # Create a 2x2 symmetric matrix
    matrix = np.array([[data[0], data[1]], [data[1], data[3]]])
    form = IntersectionForm(matrix=matrix, dimension=4)
    
    sig = form.signature()
    assert isinstance(sig, int)
    assert abs(sig) <= 2

def test_even_form_classification():
    # E8 matrix (even, unimodular, rank 8, signature 8)
    # This is just a simplified 2x2 placeholder for the test
    matrix = np.array([[2, 1], [1, 2]])
    form = IntersectionForm(matrix=matrix, dimension=4)
    
    assert form.is_even()
    assert form.type() == "II"
    assert form.signature() == 2

def test_odd_form_classification():
    matrix = np.array([[1, 0], [0, 1]])
    form = IntersectionForm(matrix=matrix, dimension=4)
    
    assert not form.is_even()
    assert form.type() == "I"
    assert form.signature() == 2
