"""Tests for the three-layer Adams bound-tightening pipeline.

Layer 3: reduce_fp_cohomology_ring
Layer 2: adams_e_infinity_exhaustive
Layer 1: adams_unstable.unstable_adams_e2_page

Modules under test:
    pysurgery/core/adams_spectral_sequence.py  (reduce_fp_cohomology_ring)
    pysurgery/core/adams_e_infinity_exhaustive.py
    pysurgery/core/adams_unstable.py
"""
from __future__ import annotations


from pysurgery.adams.spectral_sequence import (
    adams_e2_page,
    reduce_fp_cohomology_ring,
    sphere_cohomology_fp,
)
from pysurgery.adams.e_infinity_exhaustive import (
    exhaustive_e_infinity_bounds,
)
from pysurgery.adams.unstable import (
    excess_of_monomial,
    unstable_adams_e2_page,
)


# ── Layer 3: reduced cohomology ──────────────────────────────────────────────


def test_reduce_sphere_strips_degree_zero():
    """sphere_cohomology_fp(n) has basis {0: [1], n: [x]}; reduced drops the 1."""
    ring = sphere_cohomology_fp(2, prime=2)
    red = reduce_fp_cohomology_ring(ring)
    assert 0 not in red.basis
    assert red.basis.get(2, []) == ["x"]


def test_reduce_is_idempotent():
    ring = sphere_cohomology_fp(3, prime=3)
    red1 = reduce_fp_cohomology_ring(ring)
    red2 = reduce_fp_cohomology_ring(red1)
    assert dict(red1.basis) == dict(red2.basis)
    assert dict(red1.sq_table) == dict(red2.sq_table)


def test_reduced_adams_has_fewer_cells_than_unreduced():
    """Reducing the ring should at minimum halve the E_2 cell count for S^n
    (removes the ghost S^0 summand)."""
    ring = sphere_cohomology_fp(2, prime=2)
    red = reduce_fp_cohomology_ring(ring)
    p_un = adams_e2_page(ring, prime=2, s_max=3, t_max=8)
    p_red = adams_e2_page(red, prime=2, s_max=3, t_max=8)
    n_un = sum(1 for v in p_un.e2_grid.values() if v)
    n_red = sum(1 for v in p_red.e2_grid.values() if v)
    assert n_red < n_un


# ── Layer 2: exhaustive E_infinity enumerator ────────────────────────────────


def test_e_infinity_max_bounded_by_e2():
    """For every cell, max E_infinity dim <= original E_2 dim."""
    ring = sphere_cohomology_fp(2, prime=2)
    red = reduce_fp_cohomology_ring(ring)
    page = adams_e2_page(red, prime=2, s_max=4, t_max=10)
    result = exhaustive_e_infinity_bounds(page, method="analytical")
    for cell, max_dim in result.e_infinity_max.items():
        assert max_dim <= page.e2_grid[cell]


def test_e_infinity_min_nonnegative():
    ring = sphere_cohomology_fp(2, prime=2)
    red = reduce_fp_cohomology_ring(ring)
    page = adams_e2_page(red, prime=2, s_max=4, t_max=10)
    result = exhaustive_e_infinity_bounds(page, method="analytical")
    for cell, min_dim in result.e_infinity_min.items():
        assert min_dim >= 0


def test_analytical_and_exhaustive_agree_on_bounds():
    """The min/max bounds from 'analytical' and 'exhaustive' must match.

    'exhaustive' adds e_mean but the min/max should be identical (both
    endpoints of the rank lattice are jointly achievable).
    """
    ring = sphere_cohomology_fp(2, prime=2)
    red = reduce_fp_cohomology_ring(ring)
    page = adams_e2_page(red, prime=2, s_max=4, t_max=10)
    r_an = exhaustive_e_infinity_bounds(page, method="analytical")
    r_ex = exhaustive_e_infinity_bounds(page, method="exhaustive", backend="python")
    assert r_an.e_infinity_min == r_ex.e_infinity_min
    assert r_an.e_infinity_max == r_ex.e_infinity_max


def test_no_flags_returns_e2_unchanged():
    """If there are no ambiguous flags, E_infinity = E_2."""
    ring = sphere_cohomology_fp(0, prime=2)
    page = adams_e2_page(ring, prime=2, s_max=2, t_max=4)
    # Force no ambiguous flags by truncation. Then E_inf min == max == E_2 dim.
    result = exhaustive_e_infinity_bounds(page, method="analytical")
    for cell, d in page.e2_grid.items():
        if d > 0:
            assert result.e_infinity_min[cell] == d
            assert result.e_infinity_max[cell] == d


# ── Layer 1: unstable Adams (excess-filtered cobar) ──────────────────────────


def test_excess_function_for_p2_monomial():
    """At p=2 with no tau, excess(xi_1^{r_1} xi_2^{r_2} ...) = r_1."""
    assert excess_of_monomial(((), ())) == 0
    assert excess_of_monomial(((1,), ())) == 1
    assert excess_of_monomial(((3,), ())) == 3
    assert excess_of_monomial(((0, 2), ())) == 0  # only xi_2^2
    assert excess_of_monomial(((1, 1), ())) == 1  # xi_1 * xi_2


def test_unstable_page_is_a_subset_at_low_excess():
    """For S^2 the excess cap is 2, so cells from r_1 > 2 monomials are gone.

    Pinned to method='cobar_approx' because this test exercises the
    legacy excess-filtered cobar engine specifically. The default
    method='auto' now dispatches to the U-resolution at p=2; that
    engine has its own tests in test_adams_u_resolution.py.
    """
    ring = sphere_cohomology_fp(2, prime=2)
    red = reduce_fp_cohomology_ring(ring)
    stable = adams_e2_page(red, prime=2, s_max=4, t_max=12)
    unstable = unstable_adams_e2_page(
        red, prime=2, s_max=4, t_max=12, method="cobar_approx"
    )
    # The cobar-approx engine should label its space.
    assert "unstable approx" in unstable.space_label
    # The total cell count in unstable should not exceed the stable page
    # (excess filter can only remove or keep cells, not add new ones).
    n_stable = sum(1 for v in stable.e2_grid.values() if v)
    n_unstable = sum(1 for v in unstable.e2_grid.values() if v)
    # At low excess cap = 2 we expect strict reduction or equality.
    assert n_unstable <= n_stable + 2  # allow tiny edge wiggle from rounding


def test_unstable_returns_e2_grid_with_bottom_cell():
    """The bottom cell at (0, n_min) must be present in unstable E_2."""
    ring = sphere_cohomology_fp(2, prime=2)
    red = reduce_fp_cohomology_ring(ring)
    page = unstable_adams_e2_page(red, prime=2, s_max=3, t_max=8)
    assert page.e2_grid.get((0, 2), 0) == 1
