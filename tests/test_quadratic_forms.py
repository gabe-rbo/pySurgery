import numpy as np
import pytest
from pysurgery.core.quadratic_forms import QuadraticForm, arf_invariant_gf2
from pysurgery.core.exceptions import DimensionError

def test_quadratic_form_init():
    matrix = np.array([[0, 1], [1, 0]])
    q = QuadraticForm(matrix=matrix, dimension=2, q_refinement=[1, 1])
    assert q.arf_invariant() == 1

def test_quadratic_form_invalid_dim():
    matrix = np.array([[0, 1], [1, 0]])
    with pytest.raises(DimensionError):
        QuadraticForm(matrix=matrix, dimension=3, q_refinement=[0, 0])

def test_arf_invariant_trivial():
    matrix = np.array([[0, 1], [1, 0]])
    q = QuadraticForm(matrix=matrix, dimension=2, q_refinement=[0, 0])
    assert q.arf_invariant() == 0

def test_arf_invariant_larger():
    matrix = np.array([[0, 1, 0, 0],
                       [1, 0, 0, 0],
                       [0, 0, 0, 1],
                       [0, 0, 1, 0]])
    q = QuadraticForm(matrix=matrix, dimension=2, q_refinement=[1, 1, 0, 0])
    assert q.arf_invariant() == 1
    
    q2 = QuadraticForm(matrix=matrix, dimension=2, q_refinement=[1, 1, 1, 1])
    assert q2.arf_invariant() == 0


def test_arf_invariant_gf2_standalone():
    m = np.array([[0, 1], [1, 0]], dtype=np.int64)
    q = np.array([1, 1], dtype=np.int64)
    assert arf_invariant_gf2(m, q) == 1


def test_arf_invariant_gf2_invalid_shapes():
    with pytest.raises(DimensionError):
        arf_invariant_gf2(np.array([[1, 0, 0], [0, 1, 0]]), np.array([0, 0]))


def test_arf_invariant_gf2_degenerate_form_raises():
    m = np.array([[0, 0], [0, 0]], dtype=np.int64)
    q = np.array([0, 0], dtype=np.int64)
    with pytest.raises(DimensionError):
        arf_invariant_gf2(m, q)


