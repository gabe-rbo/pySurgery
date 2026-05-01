"""Tests for characteristic classes of 4-manifolds.

Overview:
    This suite verifies the computation of Stiefel-Whitney classes (w₂), 
    Wu classes, and Pontryagin classes (p₁) for 4-manifolds, as well as 
    topological properties like Spin structures and the Hirzebruch Signature Theorem.

Key Concepts:
    - **Stiefel-Whitney Class (w₂)**: An obstruction to a Spin structure. For 
      4-manifolds, w₂ is the Poincaré dual of the Wu class.
    - **Pontryagin Class (p₁)**: A characteristic class related to the 
      signature via the Hirzebruch Signature Theorem.
    - **Hirzebruch Signature Theorem**: σ(M) = 1/3 ∫ p₁(M).
    - **Spin Structure**: A manifold is Spin if and only if its second 
      Stiefel-Whitney class w₂ vanishes.
"""
import numpy as np
import pytest
from pysurgery.core.characteristic_classes import (
    extract_stiefel_whitney_w2 as wu_class,
    extract_pontryagin_p1 as pontryagin_class,
    check_spin_structure,
    verify_hirzebruch_signature,
)
from pysurgery.core.intersection_forms import IntersectionForm
from pysurgery.core.exceptions import CharacteristicClassError


def test_wu_class():
    """Verify w₂ computation and Spin structure detection for simple forms.

    What is Being Computed?:
        The Stiefel-Whitney class w₂ for hyperbolic and diagonal forms.

    Algorithm:
        1. Define the hyperbolic form H (even). Verify w₂=0 and Spin=True.
        2. Define the diagonal form I₂ (odd). Verify w₂=[1,1] and Spin=False.

    Preserved Invariants:
        - w₂ = 0 if and only if the intersection form is even (Spin condition).
    """
    Q = np.array([[0, 1], [1, 0]])
    form = IntersectionForm(matrix=Q, dimension=4)
    w2 = wu_class(form)
    assert np.array_equal(w2, np.array([0, 0]))
    assert "admits a Spin structure" in check_spin_structure(form)

    Q2 = np.array([[1, 0], [0, 1]])
    form2 = IntersectionForm(matrix=Q2, dimension=4)
    w2_2 = wu_class(form2)
    assert np.array_equal(w2_2, np.array([1, 1]))
    assert "Non-Spin" in check_spin_structure(form2)

def test_wu_class_E8():
    """Verify w₂ for the E₈ form.

    What is Being Computed?:
        The w₂ invariant for a non-trivial even form.

    Algorithm:
        1. Construct the 8x8 E₈ matrix.
        2. Verify that w₂ vanishes (all zeros).
    """
    # E8 is even, so w2 should be 0.
    E8 = np.array([
        [2, 0, -1, 0, 0, 0, 0, 0],
        [0, 2, 0, -1, 0, 0, 0, 0],
        [-1, 0, 2, -1, 0, 0, 0, 0],
        [0, -1, -1, 2, -1, 0, 0, 0],
        [0, 0, 0, -1, 2, -1, 0, 0],
        [0, 0, 0, 0, -1, 2, -1, 0],
        [0, 0, 0, 0, 0, -1, 2, -1],
        [0, 0, 0, 0, 0, 0, -1, 2]
    ])
    form = IntersectionForm(matrix=E8, dimension=4)
    w2 = wu_class(form)
    assert np.all(w2 == 0)
    assert "admits a Spin structure" in check_spin_structure(form)

def test_hirzebruch_consistency():
    """Verify the Hirzebruch Signature Theorem relation.

    What is Being Computed?:
        The consistency check σ(M) = 1/3 ∫ p₁(M).

    Algorithm:
        1. Define a form with signature 2.
        2. Verify that p₁=6 satisfies the theorem (6/3 = 2).
    """
    Q = np.array([[1, 0], [0, 1]])
    form = IntersectionForm(matrix=Q, dimension=4)
    # signature is 2. p1 integral is 3 * 2 = 6.
    assert verify_hirzebruch_signature(form, 6)
    assert not verify_hirzebruch_signature(form, 7)


def test_pontryagin_class():
    """Verify p₁ computation for a positive definite form.

    What is Being Computed?:
        The first Pontryagin class p₁(M).
    """
    Q = np.array([[1, 0], [0, 1]])
    form = IntersectionForm(matrix=Q, dimension=4)
    p1 = pontryagin_class(form)
    assert p1 == 6


def test_pontryagin_class_negative():
    """Verify p₁ computation for a negative definite form.

    What is Being Computed?:
        The first Pontryagin class p₁(M) with reversed orientation.
    """
    Q = np.array([[-1, 0], [0, -1]])
    form = IntersectionForm(matrix=Q, dimension=4)
    p1 = pontryagin_class(form)
    assert p1 == -6


def test_wu_class_requires_unimodular_form():
    """Ensure that characteristic class computation requires unimodular forms.

    What is Being Computed?:
        Error handling for non-unimodular forms where w₂ is not well-defined.
    """
    q = IntersectionForm(matrix=np.array([[2]]), dimension=4)
    with pytest.raises(CharacteristicClassError):
        wu_class(q)
