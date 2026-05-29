"""Tests for the cocycle representative API."""
from __future__ import annotations

import pytest

from pysurgery.adams.spectral_sequence import (
    reduce_fp_cohomology_ring,
    sphere_cohomology_fp,
)
from pysurgery.adams.u_resolution import UnstableResolution
from pysurgery.homology.ext_cocycles import (
    ExtCocycle,
    basis_cocycles,
    evaluate,
    is_cocycle,
)


@pytest.fixture
def res_s3_p2():
    ring = reduce_fp_cohomology_ring(sphere_cohomology_fp(3, prime=2))
    r = UnstableResolution(ring=ring, prime=2, t_max=10)
    r.build(s_max=4)
    return r


def test_basis_cocycle_at_zero_zero_for_s3():
    ring = reduce_fp_cohomology_ring(sphere_cohomology_fp(3, prime=2))
    r = UnstableResolution(ring=ring, prime=2, t_max=10)
    r.build(s_max=4)
    # S^3 reduced has F_0 generator x at degree 3.
    bs = basis_cocycles(r, s=0, t=3)
    assert len(bs) == 1
    cy = bs[0]
    assert cy.prime == 2
    assert cy.s == 0
    assert cy.t == 3
    assert 1 in cy.coefs.values()


def test_basis_cocycle_empty_outside_resolution(res_s3_p2):
    assert basis_cocycles(res_s3_p2, s=0, t=99) == []
    assert basis_cocycles(res_s3_p2, s=99, t=3) == []


def test_basis_cocycle_count_matches_e2_dim_s2_p2():
    ring = reduce_fp_cohomology_ring(sphere_cohomology_fp(2, prime=2))
    r = UnstableResolution(ring=ring, prime=2, t_max=10)
    r.build(s_max=4)
    # E_2^{0, 2} should have dim 1 (the generator x).
    assert len(basis_cocycles(r, s=0, t=2)) == 1
    # E_2^{1, 3} = the kernel of d_0 at degree 3: Sq^1·γ_x in F_0 maps to 0
    # in M (no class at degree 3), generating F_1 at deg 3 with one gen.
    assert len(basis_cocycles(r, s=1, t=3)) == 1


def test_ext_cocycle_is_zero_when_all_coefs_zero():
    cy = ExtCocycle(prime=2, s=1, t=3, coefs={})
    assert cy.is_zero()
    cy2 = ExtCocycle(prime=2, s=1, t=3, coefs={0: 0})
    assert cy2.is_zero()


def test_ext_cocycle_normalized_drops_mod_prime_zeros():
    cy = ExtCocycle(prime=2, s=1, t=3, coefs={0: 4, 1: 3})
    n = cy.normalized()
    # 4 mod 2 = 0 (dropped); 3 mod 2 = 1 (kept).
    assert n.coefs == {1: 1}


def test_evaluate_on_identity_pair_picks_generator_coefficient(res_s3_p2):
    bs = basis_cocycles(res_s3_p2, s=0, t=3)
    cy = bs[0]
    gid = next(iter(cy.coefs))
    # Identity admissible on this gid should evaluate to 1.
    assert evaluate(res_s3_p2, cy, {(gid, ()): 1}) == 1
    # Sq-action on the generator evaluates to 0 (Sq-action elements are
    # not in the dual basis).
    assert evaluate(res_s3_p2, cy, {(gid, (1,)): 1}) == 0


def test_is_cocycle_holds_for_minimal_basis():
    ring = reduce_fp_cohomology_ring(sphere_cohomology_fp(3, prime=2))
    r = UnstableResolution(ring=ring, prime=2, t_max=10)
    r.build(s_max=3)
    for s in range(0, 3):
        for t in range(0, 10):
            for cy in basis_cocycles(r, s, t):
                assert is_cocycle(r, cy), f"Failed at s={s}, t={t}, cy={cy}"
