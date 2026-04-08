import pytest
from pysurgery.core.group_rings import GroupRingElement
from pysurgery.core.exceptions import GroupRingError

def test_group_ring_init():
    el = GroupRingElement({"e": 1, "g_1": -2}, group_order=5)
    assert el.coeffs == {"1": 1, "g_1": -2}

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
    assert inv.coeffs == {"1": 1, "g_4": 2, "g_3": 3}

def test_group_ring_involution_standard():
    el = GroupRingElement({"e": 1, "g1": 2, "g2": 3}, group_order=5)
    inv = el.involution()
    assert inv.coeffs == {"1": 1, "g4": 2, "g3": 3}


def test_group_ring_multiply_python_fallback_cyclic(monkeypatch):
    # Uses pure Python fallback when Julia is unavailable; still valid when Julia is available.
    from pysurgery.core import group_rings as gr
    monkeypatch.setattr(gr.julia_engine, "available", False)
    a = GroupRingElement({"1": 1, "g_1": 2}, group_order=5)
    b = GroupRingElement({"1": 3, "g_2": 1}, group_order=5)
    c = a * b
    # (1 + 2g)*(3 + g^2) = 3 + g^2 + 6g + 2g^3
    assert c.coeffs.get("1", 0) == 3
    assert c.coeffs.get("g_2", 0) == 1
    assert c.coeffs.get("g_1", 0) == 6
    assert c.coeffs.get("g_3", 0) == 2


def test_group_ring_mixed_identity_spellings_normalize():
    el = GroupRingElement({"e": 1, "1": 2, "g0": 3, "g_0": 4}, group_order=7)
    assert el.coeffs == {"1": 10}

def test_group_ring_involution_non_cyclic():
    el = GroupRingElement({"a": 1, "b": 2}, group_order=None)
    with pytest.raises(GroupRingError):
        el.involution()


def test_group_ring_generic_group_law_noncyclic():
    # Klein four group V4 = {1,a,b,c}, where a^2=b^2=c^2=1 and ab=c, bc=a, ca=b.
    table = {
        ("1", "1"): "1", ("1", "a"): "a", ("1", "b"): "b", ("1", "c"): "c",
        ("a", "1"): "a", ("a", "a"): "1", ("a", "b"): "c", ("a", "c"): "b",
        ("b", "1"): "b", ("b", "a"): "c", ("b", "b"): "1", ("b", "c"): "a",
        ("c", "1"): "c", ("c", "a"): "b", ("c", "b"): "a", ("c", "c"): "1",
    }
    law = lambda x, y: table[(x, y)]

    x = GroupRingElement({"a": 1, "b": 1}, group_law=law)
    y = GroupRingElement({"a": 1, "1": 1}, group_law=law)
    z = x * y
    # (a+b)(a+1) = aa + a + ba + b = 1 + a + c + b
    assert z.coeffs == {"1": 1, "a": 1, "b": 1, "c": 1}


def test_group_ring_generic_involution_callback():
    inv = {"1": "1", "a": "a", "b": "b", "c": "c"}
    inv_law = lambda g: inv[g]
    law = lambda x, y: x if y == "1" else y if x == "1" else "1"
    el = GroupRingElement({"a": 2, "1": 1}, group_law=law, inverse_law=inv_law)
    bar = el.involution()
    assert bar.coeffs == {"a": 2, "1": 1}

