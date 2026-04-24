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
    Q = np.array([[1, 0], [0, 1]])
    form = IntersectionForm(matrix=Q, dimension=4)
    # signature is 2. p1 integral is 3 * 2 = 6.
    assert verify_hirzebruch_signature(form, 6)
    assert not verify_hirzebruch_signature(form, 7)


def test_pontryagin_class():
    Q = np.array([[1, 0], [0, 1]])
    form = IntersectionForm(matrix=Q, dimension=4)
    p1 = pontryagin_class(form)
    assert p1 == 6


def test_pontryagin_class_negative():
    Q = np.array([[-1, 0], [0, -1]])
    form = IntersectionForm(matrix=Q, dimension=4)
    p1 = pontryagin_class(form)
    assert p1 == -6


def test_wu_class_requires_unimodular_form():
    q = IntersectionForm(matrix=np.array([[2]]), dimension=4)
    with pytest.raises(CharacteristicClassError):
        wu_class(q)
