"""Tests for group ring elements and algebraic operations (addition, multiplication, involution).

Overview:
    This suite verifies the GroupRingElement class, which represents elements 
    of the group ring R[G]. It tests basic arithmetic, normalization of identity 
    spellings, and both cyclic and generic group law implementations.

Key Concepts:
    - **Group Ring (ℤ[G])**: Formal sums of group elements with integer coefficients.
    - **Involution (Anti-automorphism)**: The map sending Σ n_g g to Σ n_g g⁻¹.
    - **Group Law**: The binary operation defining the group structure.
"""

import pytest
from pysurgery.core.group_rings import GroupRingElement
from pysurgery.core.exceptions import GroupRingError


def test_group_ring_init():
    """Verify initialization and coefficient normalization for group ring elements.

    What is Being Computed?:
        Normalization of group element labels (e.g., 'e' to '1').

    Algorithm:
        1. Initialize GroupRingElement with 'e' and 'g_1'.
        2. Assert that 'e' is converted to '1' in the internal coeffs dictionary.

    Preserved Invariants:
        - Total sum of coefficients.
    """
    el = GroupRingElement({"e": 1, "g_1": -2}, group_order=5)
    assert el.coeffs == {"1": 1, "g_1": -2}


def test_group_ring_add():
    """Verify pointwise addition of group ring elements.

    What is Being Computed?:
        The sum x + y for x, y ∈ ℤ[G].

    Algorithm:
        1. Initialize two elements with overlapping supports.
        2. Compute their sum.
        3. Verify the resulting coefficients.

    Preserved Invariants:
        - Addition is commutative and associative.
    """
    el1 = GroupRingElement({"e": 1, "g_1": -2}, group_order=5)
    el2 = GroupRingElement({"e": -1, "g_1": 3, "g_2": 1}, group_order=5)
    res = el1 + el2
    assert res.coeffs == {"g_1": 1, "g_2": 1}


def test_group_ring_add_mismatch():
    """Verify that adding elements from different groups raises an error.

    What is Being Computed?:
        Error handling for incompatible group ring additions.

    Algorithm:
        1. Create two elements with different group_order values.
        2. Attempt to add them.
        3. Assert that GroupRingError is raised.
    """
    el1 = GroupRingElement({"e": 1}, group_order=5)
    el2 = GroupRingElement({"e": 1}, group_order=3)
    with pytest.raises(GroupRingError):
        _ = el1 + el2


def test_group_ring_involution():
    """Verify the algebraic involution (bar map) for cyclic group elements.

    What is Being Computed?:
        The involution x ↦ x̄ where (Σ n_g g)̄ = Σ n_g g⁻¹.

    Algorithm:
        1. Define an element in a cyclic group C₅.
        2. Compute its involution.
        3. Assert that g_k is mapped to g_{5-k}.

    Preserved Invariants:
        - Involution is an additive homomorphism.
        - (x̄)̄ = x.
    """
    el = GroupRingElement({"e": 1, "g_1": 2, "g_2": 3}, group_order=5)
    inv = el.involution()
    # (g_1)^-1 = g_4, (g_2)^-1 = g_3
    assert inv.coeffs == {"1": 1, "g_4": 2, "g_3": 3}


def test_group_ring_involution_standard():
    """Verify involution using standard g{k} naming conventions.

    What is Being Computed?:
        Involution mapping for g1, g2, etc.

    Algorithm:
        1. Define element with g1, g2 labels.
        2. Assert g1 maps to g4 and g2 maps to g3 in C₅.
    """
    el = GroupRingElement({"e": 1, "g1": 2, "g2": 3}, group_order=5)
    inv = el.involution()
    assert inv.coeffs == {"1": 1, "g4": 2, "g3": 3}


def test_group_ring_multiply_python_fallback_cyclic(monkeypatch):
    """Verify group ring multiplication using the pure-Python fallback.

    What is Being Computed?:
        The product x * y for x, y ∈ ℤ[C₅].

    Algorithm:
        1. Force Julia engine to be unavailable.
        2. Compute product of two elements.
        3. Verify result matches the Cauchy product in the group ring.

    Preserved Invariants:
        - Multiplicative identity (1 * x = x).
        - Distributivity over addition.
    """
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
    """Verify that various identity spellings ('e', '1', 'g0', 'g_0') normalize to '1'.

    What is Being Computed?:
        Normalization of multiple identity representations in the group ring.

    Algorithm:
        1. Initialize GroupRingElement with 'e', '1', 'g0', and 'g_0'.
        2. Verify that all coefficients are summed under the key '1'.

    Preserved Invariants:
        - Resulting coefficient of '1' is the sum of input identity coefficients.
    """
    el = GroupRingElement({"e": 1, "1": 2, "g0": 3, "g_0": 4}, group_order=7)
    assert el.coeffs == {"1": 10}


def test_group_ring_involution_non_cyclic():
    """Verify that involution fails when no group structure or order is provided.

    What is Being Computed?:
        Error handling for involution on un-structured group elements.

    Algorithm:
        1. Create element with group_order=None.
        2. Attempt involution and assert GroupRingError.
    """
    el = GroupRingElement({"a": 1, "b": 2}, group_order=None)
    with pytest.raises(GroupRingError):
        el.involution()


def test_group_ring_involution_non_cyclic_with_group_order_still_requires_structure():
    """Verify that involution fails even with group_order if names are not standard g{k}.

    What is Being Computed?:
        Error handling for involution on non-standard labels.

    Algorithm:
        1. Create element with group_order=5 but custom label 'a'.
        2. Attempt involution and assert GroupRingError.
    """
    el = GroupRingElement({"a": 1}, group_order=5)
    with pytest.raises(GroupRingError):
        el.involution()


def test_group_ring_generic_group_law_noncyclic():
    """Verify group ring multiplication using a custom generic group law (V4).

    What is Being Computed?:
        The product in the group ring ℤ[V4] (Klein four-group).

    Algorithm:
        1. Define the multiplication table for V4.
        2. Pass a group_law callback to GroupRingElement.
        3. Compute product and verify against the V4 multiplication rules.

    Preserved Invariants:
        - Algebraic structure of the specified group ring.
    """
    # Klein four group V4 = {1,a,b,c}, where a^2=b^2=c^2=1 and ab=c, bc=a, ca=b.
    table = {
        ("1", "1"): "1",
        ("1", "a"): "a",
        ("1", "b"): "b",
        ("1", "c"): "c",
        ("a", "1"): "a",
        ("a", "a"): "1",
        ("a", "b"): "c",
        ("a", "c"): "b",
        ("b", "1"): "b",
        ("b", "a"): "c",
        ("b", "b"): "1",
        ("b", "c"): "a",
        ("c", "1"): "c",
        ("c", "a"): "b",
        ("c", "b"): "a",
        ("c", "c"): "1",
    }

    def law(x, y):
        return table[(x, y)]

    x = GroupRingElement({"a": 1, "b": 1}, group_law=law)
    y = GroupRingElement({"a": 1, "1": 1}, group_law=law)
    z = x * y
    # (a+b)(a+1) = aa + a + ba + b = 1 + a + c + b
    assert z.coeffs == {"1": 1, "a": 1, "b": 1, "c": 1}


def test_group_ring_generic_involution_callback():
    """Verify involution using a custom inverse_law callback.

    What is Being Computed?:
        Involution x ↦ x̄ via user-provided inverse mapping.

    Algorithm:
        1. Define an inverse mapping dictionary and a trivial group law.
        2. Pass inverse_law and group_law callbacks.
        3. Assert that bar map correctly applies the inverse mapping.

    Preserved Invariants:
        - g⁻¹ is correctly identified for each g in the support.
    """
    inv = {"1": "1", "a": "a", "b": "b", "c": "c"}

    def inv_law(g):
        return inv[g]

    def law(x, y):
        return x if y == "1" else y if x == "1" else "1"

    el = GroupRingElement({"a": 2, "1": 1}, group_law=law, inverse_law=inv_law)
    bar = el.involution()
    assert bar.coeffs == {"a": 2, "1": 1}
