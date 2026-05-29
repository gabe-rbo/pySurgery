"""Tests for the CRT multi-prime synthesizer."""
from __future__ import annotations

import pytest

from pysurgery.homotopy.multi_prime_synthesis import (
    group_string,
    synthesize_torsion,
    torsion_to_p_primary,
)


# ── synthesize_torsion ────────────────────────────────────────────────────────


def test_empty_returns_empty():
    assert synthesize_torsion({}) == []


def test_only_two_primary_passes_through():
    assert synthesize_torsion({2: [4, 2]}) == [4, 2]


def test_two_and_three_collapse_via_crt():
    # Z/4 ⊕ Z/3 ≅ Z/12
    assert synthesize_torsion({2: [4], 3: [3]}) == [12]


def test_two_and_three_at_different_lengths():
    # 2-primary: Z/8 ⊕ Z/2, 3-primary: Z/3
    # → invariants: lcm(8, 3) = 24, lcm(2, 1) = 2
    assert synthesize_torsion({2: [8, 2], 3: [3]}) == [24, 2]


def test_pi_6_s2_signature_24_3_to_invariant_factor_form():
    # 2-primary: Z/4, 3-primary: Z/3 → Z/12
    assert synthesize_torsion({2: [4], 3: [3]}) == [12]


def test_pi_10_s4_signature_2_primary_8_three_primary_three_three():
    # Z/8 ⊕ Z/3 ⊕ Z/3 → invariant factors [24, 3] (after CRT pairing)
    assert synthesize_torsion({2: [8], 3: [3, 3]}) == [24, 3]


def test_z_2_z_2_no_collapse():
    assert synthesize_torsion({2: [2, 2]}) == [2, 2]


def test_unsorted_input_is_sorted_descending():
    assert synthesize_torsion({2: [2, 8]}) == [8, 2]


def test_rejects_non_prime_power():
    with pytest.raises(ValueError):
        # 6 is not a power of 2
        synthesize_torsion({2: [6]})


def test_rejects_trivial_factor_in_input():
    with pytest.raises(ValueError):
        synthesize_torsion({2: [1]})


# ── torsion_to_p_primary ──────────────────────────────────────────────────────


def test_torsion_to_p_primary_roundtrip_z12():
    by_prime = torsion_to_p_primary([12])
    assert by_prime == {2: [4], 3: [3]}


def test_torsion_to_p_primary_24_2():
    by_prime = torsion_to_p_primary([24, 2])
    # 24 = 8·3, 2 = 2 → 2-primary [8, 2], 3-primary [3]
    assert by_prime[2] == [8, 2]
    assert by_prime[3] == [3]


def test_synthesize_inverse_roundtrip():
    for orig in ([12], [24, 2], [2, 2], [60]):
        decomp = torsion_to_p_primary(orig)
        # Convert dict values to list (already are).
        regen = synthesize_torsion(decomp)
        # regen == orig modulo the canonical descending order.
        assert regen == sorted(orig, reverse=True), (orig, decomp, regen)


# ── group_string ──────────────────────────────────────────────────────────────


def test_group_string_trivial():
    assert group_string(0, []) == "0"


def test_group_string_z():
    assert group_string(1, []) == "Z"


def test_group_string_torsion():
    assert group_string(0, [12]) == "Z/12"


def test_group_string_mixed():
    assert group_string(1, [12]) == "Z ⊕ Z/12"
