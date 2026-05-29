"""Tests for the h_0-tower extension solver."""
from __future__ import annotations


from pysurgery.adams.extension_solver import (
    detect_h0_towers,
    solve_extensions,
    solve_stem,
)
from pysurgery.adams.spectral_sequence import AdamsE2Page


def _mk_page(grid, prime=2, s_max=6, t_max=12) -> AdamsE2Page:
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


def test_empty_stem_returns_empty_hypothesis():
    page = _mk_page({})
    h = solve_stem(page, 3)
    assert h.most_collapsed_invariant_factors == ()
    assert h.most_split_invariant_factors == ()
    assert h.p_order_upper_bound == 1


def test_free_tower_at_s_zero_is_separated_from_torsion():
    """Stem n=2: cell at (0, 2) and a tower at (1, 3), (2, 4)."""
    grid = {(0, 2): 1, (1, 3): 1, (2, 4): 1}
    page = _mk_page(grid)
    h = solve_stem(page, 2)
    assert h.free_tower is not None and h.free_tower.s_start == 0
    assert len(h.towers) == 1
    assert h.towers[0].s_start == 1 and h.towers[0].length == 2


def test_collapsed_tower_extends_to_single_z_mod_p_power():
    """One tower of length 2 → single Z/4 in the most-collapsed view."""
    grid = {(1, 3): 1, (2, 4): 1}
    page = _mk_page(grid)
    h = solve_stem(page, 2)
    assert h.most_collapsed_invariant_factors == (4,)


def test_split_tower_keeps_each_cell_as_z_mod_p():
    """Same tower of length 2 → (Z/2)^2 in the most-split view."""
    grid = {(1, 3): 1, (2, 4): 1}
    page = _mk_page(grid)
    h = solve_stem(page, 2)
    assert h.most_split_invariant_factors == (2, 2)


def test_two_disjoint_towers_emit_two_summands():
    """Tower at s=1..2 AND tower at s=4..5 at the same stem."""
    grid = {(1, 3): 1, (2, 4): 1, (4, 6): 1, (5, 7): 1}
    page = _mk_page(grid)
    h = solve_stem(page, 2)
    assert len(h.towers) == 2
    # Two collapsed Z/4 summands, sorted descending.
    assert h.most_collapsed_invariant_factors == (4, 4)


def test_p_order_upper_bound_is_product_of_cell_dims():
    grid = {(1, 3): 1, (2, 4): 1}
    page = _mk_page(grid, prime=2)
    h = solve_stem(page, 2)
    assert h.p_order_upper_bound == 4  # 2 * 2


def test_solve_extensions_dispatches_per_prime():
    page_p2 = _mk_page({(1, 3): 1}, prime=2)
    page_p3 = _mk_page({(1, 4): 1}, prime=3)
    out = solve_extensions({2: page_p2, 3: page_p3}, n=2)
    assert set(out) == {2, 3}
    assert out[2].most_collapsed_invariant_factors == (2,)
    assert out[3].most_collapsed_invariant_factors == ()  # stem 2 not at p=3


def test_detect_h0_towers_at_odd_prime():
    page = _mk_page({(1, 4): 1, (2, 5): 1}, prime=3)
    free, torsion = detect_h0_towers(page, 3)
    assert free is None
    assert len(torsion) == 1
    assert torsion[0].length == 2
    assert torsion[0].prime == 3
