"""Tests for the verification-only known homotopy table module.

These tests check internal consistency of the tables themselves
(p_primary parts CRT-multiply to the torsion order, group strings parse,
look-ups land on the right entry). They do NOT compare the tables to the
pysurgery algorithm — that comparison happens in the verifier test suite.
"""
from __future__ import annotations

from typing import Iterable

import pytest

from pysurgery.homotopy.known_homotopy_tables import (
    KNOWN_PI_N_CP,
    KNOWN_PI_N_RP,
    KNOWN_PI_N_SPHERES,
    KnownHomotopyEntry,
    KnownHomotopyTable,
    lookup,
)


def _torsion_order(invariant_factors: Iterable[int]) -> int:
    n = 1
    for d in invariant_factors:
        n *= int(d)
    return n


def _p_primary_total_order(p_primary) -> int:
    n = 1
    for p, factors in p_primary.items():
        for d in factors:
            n *= int(d)
    return n


# ── Schema sanity ─────────────────────────────────────────────────────────────


def test_known_homotopy_entry_group_string_trivial():
    e = KnownHomotopyEntry(free_rank=0)
    assert e.group_string() == "0"


def test_known_homotopy_entry_group_string_z():
    e = KnownHomotopyEntry(free_rank=1)
    assert e.group_string() == "Z"


def test_known_homotopy_entry_group_string_z_squared():
    e = KnownHomotopyEntry(free_rank=2)
    assert e.group_string() == "Z^2"


def test_known_homotopy_entry_group_string_torsion_only():
    e = KnownHomotopyEntry(free_rank=0, torsion=(2, 12))
    assert e.group_string() == "Z/2 ⊕ Z/12"


def test_known_homotopy_entry_group_string_mixed():
    e = KnownHomotopyEntry(free_rank=1, torsion=(12,))
    assert e.group_string() == "Z ⊕ Z/12"


# ── Order consistency: |torsion| must equal product over primes of |p_primary| ─


def _all_entries(table: KnownHomotopyTable):
    for n, e in table.entries.items():
        yield n, e


def _table_iter(d):
    for k, tab in d.items():
        for n, e in _table_entries(tab):
            yield (tab.family, k, n, e)


def _table_entries(tab: KnownHomotopyTable):
    return tab.entries.items()


@pytest.mark.parametrize("table_dict,family_name", [
    (KNOWN_PI_N_SPHERES, "S"),
    (KNOWN_PI_N_CP, "CP"),
    (KNOWN_PI_N_RP, "RP"),
])
def test_p_primary_orders_match_torsion_orders(table_dict, family_name):
    """Σ-prime |p-primary| == |torsion|; primes outside p_primary contribute 1."""
    for k, tab in table_dict.items():
        for n, e in tab.entries.items():
            if not e.p_primary:
                # Allowed: no p_primary annotation means caller doesn't claim
                # to have decomposed; nothing to check.
                continue
            t_order = _torsion_order(e.torsion)
            p_order = _p_primary_total_order(e.p_primary)
            assert t_order == p_order, (
                f"{family_name}^{k}, n={n}: |torsion|={t_order} "
                f"!= prod|p-primary|={p_order} "
                f"(torsion={e.torsion}, p_primary={e.p_primary})"
            )


# ── Spot checks: classic values ───────────────────────────────────────────────


def test_pi_3_s2_is_Z():
    e = lookup("S", 2, 3)
    assert e is not None
    assert e.free_rank == 1
    assert e.torsion == ()


def test_pi_4_s2_is_Z_mod_2():
    e = lookup("S", 2, 4)
    assert e is not None
    assert e.free_rank == 0
    assert e.torsion == (2,)


def test_pi_4_s3_is_Z_mod_2():
    e = lookup("S", 3, 4)
    assert e is not None
    assert e.torsion == (2,)


def test_pi_6_s3_is_Z_mod_12():
    e = lookup("S", 3, 6)
    assert e is not None
    assert e.torsion == (12,)
    # 12 = 4·3 in p-primary form.
    assert e.p_primary[2] == (4,)
    assert e.p_primary[3] == (3,)


def test_pi_7_s4_is_Z_plus_Z_mod_12():
    e = lookup("S", 4, 7)
    assert e is not None
    assert e.free_rank == 1
    assert e.torsion == (12,)


def test_pi_n_s1_trivial_for_n_geq_2():
    for n in range(2, 19):
        e = lookup("S", 1, n)
        assert e is not None
        assert e.free_rank == 0 and e.torsion == ()


# ── CP and RP cross-checks ───────────────────────────────────────────────────


def test_cp1_equals_s2():
    s2 = KNOWN_PI_N_SPHERES[2].entries
    cp1 = KNOWN_PI_N_CP[1].entries
    for n in range(2, 11):
        if n in s2:
            assert cp1[n].free_rank == s2[n].free_rank
            assert cp1[n].torsion == s2[n].torsion


def test_pi1_cp_is_zero_for_n_geq_2():
    for k in (2, 3):
        e = lookup("CP", k, 1)
        assert e is not None
        assert e.free_rank == 0 and e.torsion == ()


def test_pi1_rp_n_geq_2_is_z2():
    for k in (2, 3, 4, 5):
        e = lookup("RP", k, 1)
        assert e is not None
        assert e.torsion == (2,)


def test_pi_2_cp_n_is_z():
    for k in (1, 2, 3):
        e = lookup("CP", k, 2)
        assert e is not None
        assert e.free_rank == 1


# ── Lookup error cases ───────────────────────────────────────────────────────


def test_lookup_unknown_family_returns_none():
    assert lookup("BOGUS", 1, 1) is None


def test_lookup_unknown_parameter_returns_none():
    assert lookup("S", 999, 3) is None


def test_lookup_out_of_range_stem_returns_none():
    assert lookup("S", 2, 99) is None
