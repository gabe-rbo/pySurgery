"""Tests for the d_2 verdict module."""
from __future__ import annotations


from pysurgery.adams.differentials import (
    compute_d2_via_h0_action,
)
from pysurgery.adams.spectral_sequence import AdamsE2Page


def _mk_page(grid, prime=2, s_max=4, t_max=8) -> AdamsE2Page:
    return AdamsE2Page(
        space_label="test",
        prime=prime,
        s_max=s_max,
        t_max=t_max,
        e2_grid=grid,
        reliable_window=(s_max, t_max),
        status="success",
        reasoning="test",
    )


def test_empty_grid_yields_empty_report():
    page = _mk_page({})
    r = compute_d2_via_h0_action(page)
    assert r.prime == 2
    assert not r.forced_zeros and not r.unresolved


def test_sparse_grid_all_d2_forced_zero():
    """A grid with a single cell at (0, 2) has no possible d_2 target."""
    grid = {(0, 2): 1}
    r = compute_d2_via_h0_action(_mk_page(grid))
    assert len(r.forced_zeros) == 1
    assert r.forced_zeros[0].source == (0, 2)
    assert r.forced_zeros[0].target == (2, 3)
    assert r.forced_zeros[0].target_dim == 0


def test_two_cells_in_d2_pattern_yields_unresolved():
    """Cell at (0, 2) and another at (2, 3) makes d_2 candidate non-trivial."""
    grid = {(0, 2): 1, (2, 3): 1}
    r = compute_d2_via_h0_action(_mk_page(grid))
    # The (0, 2) → (2, 3) differential is non-trivial.
    has_unresolved_at_source = any(
        v.source == (0, 2) and v.target == (2, 3)
        for v in r.unresolved
    )
    assert has_unresolved_at_source


def test_d2_report_prime_propagates():
    page = _mk_page({(0, 1): 1}, prime=3)
    r = compute_d2_via_h0_action(page)
    assert r.prime == 3
