import numpy as np
import pytest
from pysurgery.core.characteristic_classes import (
    extract_stiefel_whitney_w2 as wu_class,
    extract_pontryagin_p1 as pontryagin_class,
)
from pysurgery.core.intersection_forms import IntersectionForm
from pysurgery.core.exceptions import CharacteristicClassError


def test_wu_class():
    Q = np.array([[0, 1], [1, 0]])
    form = IntersectionForm(matrix=Q, dimension=4)
    w2 = wu_class(form)
    assert np.array_equal(w2, np.array([0, 0]))

    Q2 = np.array([[1, 0], [0, 1]])
    form2 = IntersectionForm(matrix=Q2, dimension=4)
    w2_2 = wu_class(form2)
    assert np.array_equal(w2_2, np.array([1, 1]))


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
