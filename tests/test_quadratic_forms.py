"""Tests for quadratic forms and Arf invariants.

Overview:
    This module validates the computation of quadratic forms, their refinements, 
    and the Arf invariant over GF(2).

Key Concepts:
    - **Quadratic Form**: A symmetric bilinear form with a quadratic refinement.
    - **Arf Invariant**: A ℤ/2ℤ invariant that classifies non-singular quadratic 
      forms over GF(2).
"""
import numpy as np
import pytest
from pysurgery.core.quadratic_forms import QuadraticForm, arf_invariant_gf2
from pysurgery.core.exceptions import DimensionError


def test_quadratic_form_init():
    """Verify initialization of QuadraticForm objects.

    What is Being Computed?:
        Initializes a QuadraticForm and computes its Arf invariant.

    Algorithm:
        1. Define a hyperbolic plane matrix [[0, 1], [1, 0]].
        2. Provide a quadratic refinement [1, 1].
        3. Assert arf_invariant() returns 1.
    """
    matrix = np.array([[0, 1], [1, 0]])
    q = QuadraticForm(matrix=matrix, dimension=2, q_refinement=[1, 1])
    assert q.arf_invariant() == 1


def test_quadratic_form_invalid_dim():
    """Ensure DimensionError is raised for inconsistent inputs.

    Algorithm:
        Attempts to initialize a 2D matrix as a 3D quadratic form.
    """
    matrix = np.array([[0, 1], [1, 0]])
    with pytest.raises(DimensionError):
        QuadraticForm(matrix=matrix, dimension=3, q_refinement=[0, 0])


def test_arf_invariant_trivial():
    """Test Arf invariant on a trivial (null) refinement.

    Algorithm:
        Checks that a hyperbolic plane with zero refinement has Arf invariant 0.
    """
    matrix = np.array([[0, 1], [1, 0]])
    q = QuadraticForm(matrix=matrix, dimension=2, q_refinement=[0, 0])
    assert q.arf_invariant() == 0


def test_arf_invariant_larger():
    """Test Arf invariant on larger direct sums of hyperbolic planes.

    Algorithm:
        Computes Arf invariant for a 4x4 matrix representing two hyperbolic planes.
    """
    matrix = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    q = QuadraticForm(matrix=matrix, dimension=2, q_refinement=[1, 1, 0, 0])
    assert q.arf_invariant() == 1

    q2 = QuadraticForm(matrix=matrix, dimension=2, q_refinement=[1, 1, 1, 1])
    assert q2.arf_invariant() == 0


def test_arf_invariant_gf2_standalone():
    """Test the standalone arf_invariant_gf2 utility.

    What is Being Computed?:
        Directly calculates the Arf invariant using bitwise operations on GF(2) matrices.
    """
    m = np.array([[0, 1], [1, 0]], dtype=np.int64)
    q = np.array([1, 1], dtype=np.int64)
    assert arf_invariant_gf2(m, q) == 1


def test_arf_invariant_gf2_invalid_shapes():
    """Verify shape validation in arf_invariant_gf2."""
    with pytest.raises(DimensionError):
        arf_invariant_gf2(np.array([[1, 0, 0], [0, 1, 0]]), np.array([0, 0]))


def test_arf_invariant_gf2_degenerate_form_raises():
    """Ensure degenerate forms raise errors in Arf computation."""
    m = np.array([[0, 0], [0, 0]], dtype=np.int64)
    q = np.array([0, 0], dtype=np.int64)
    with pytest.raises(DimensionError):
        arf_invariant_gf2(m, q)
