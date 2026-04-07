import pytest
from pysurgery.core.group_rings import GroupRingElement
from pysurgery.core.exceptions import GroupRingError

def test_group_ring_init():
    el = GroupRingElement({"e": 1, "g_1": -2}, group_order=5)
    assert el.coeffs == {"e": 1, "g_1": -2}

def test_group_ring_add():
    el1 = GroupRingElement({"e": 1, "g_1": -2}, group_order=5)
    el2 = GroupRingElement({"e": -1, "g_1": 3, "g_2": 1}, group_order=5)
    res = el1 + el2
    assert res.coeffs == {"g_1": 1, "g_2": 1}

def test_group_ring_add_mismatch():
    el1 = GroupRingElement({"e": 1}, group_order=5)
    el2 = GroupRingElement({"e": 1}, group_order=3)
    with pytest.raises(GroupRingError):
        _ = el1 + el2

def test_group_ring_involution():
    el = GroupRingElement({"e": 1, "g_1": 2, "g_2": 3}, group_order=5)
    inv = el.involution()
    # (g_1)^-1 = g_4, (g_2)^-1 = g_3
    assert inv.coeffs == {"e": 1, "g_4": 2, "g_3": 3}

def test_group_ring_involution_standard():
    el = GroupRingElement({"e": 1, "g1": 2, "g2": 3}, group_order=5)
    inv = el.involution()
    assert inv.coeffs == {"e": 1, "g4": 2, "g3": 3}

def test_group_ring_involution_non_cyclic():
    el = GroupRingElement({"a": 1, "b": 2}, group_order=None)
    with pytest.raises(GroupRingError):
        el.involution()
