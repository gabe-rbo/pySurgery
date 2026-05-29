"""Tests for the (minimal) Yoneda product on Ext_U^{*, *}."""
from __future__ import annotations

import pytest

from pysurgery.adams.spectral_sequence import (
    reduce_fp_cohomology_ring,
    sphere_cohomology_fp,
)
from pysurgery.adams.u_resolution import u_resolution_e2_page
from pysurgery.homology.ext_yoneda_product import (
    ExtElement,
    h0_shift_map,
    h_zero,
    yoneda_product,
)


def test_h_zero_lives_in_1_1_at_p2():
    h0 = h_zero(2)
    assert h0.prime == 2
    assert h0.s == 1 and h0.t == 1


def test_h_zero_lives_in_1_1_at_p3():
    h0 = h_zero(3)
    assert h0.prime == 3
    assert h0.s == 1 and h0.t == 1


def test_h0_times_class_shifts_bidegree():
    x = ExtElement(prime=2, s=2, t=5, coords=((0, 1),))
    h0 = h_zero(2)
    prod = yoneda_product(h0, x)
    assert prod is not None
    assert prod.s == 3 and prod.t == 6


def test_h0_action_is_commutative():
    x = ExtElement(prime=2, s=2, t=5, coords=((0, 1),))
    h0 = h_zero(2)
    a = yoneda_product(h0, x)
    b = yoneda_product(x, h0)
    assert a is not None and b is not None
    assert (a.s, a.t) == (b.s, b.t)


def test_general_yoneda_product_raises():
    x = ExtElement(prime=2, s=2, t=5, coords=((0, 1),))
    y = ExtElement(prime=2, s=3, t=7, coords=((0, 1),))
    with pytest.raises(NotImplementedError):
        yoneda_product(x, y)


def test_prime_mismatch_raises():
    a = ExtElement(prime=2, s=1, t=1, coords=((0, 1),))
    b = ExtElement(prime=3, s=1, t=1, coords=((0, 1),))
    with pytest.raises(ValueError):
        yoneda_product(a, b)


def test_h0_shift_map_on_s2_at_p2():
    """For reduced S^2 at p=2, the U-resolution has h_0-tower starting
    at (s=0, t=2) extending up via h_0-multiplication. The shift map
    should expose nonzero rank between successive cells."""
    red = reduce_fp_cohomology_ring(sphere_cohomology_fp(2, prime=2))
    page = u_resolution_e2_page(red, prime=2, s_max=3, t_max=8)
    mp = h0_shift_map(page)
    # If (0, 2) and (1, 3) both nonzero, the map's rank is min of dims.
    if page.e2_grid.get((0, 2), 0) > 0 and page.e2_grid.get((1, 3), 0) > 0:
        assert mp.get((0, 2), 0) >= 1
