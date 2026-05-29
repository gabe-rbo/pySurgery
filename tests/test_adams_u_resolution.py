"""Tests for the minimal free unstable A_p-resolution engine.

Module under test: pysurgery/core/adams_u_resolution.py
"""
from __future__ import annotations


from pysurgery.adams.spectral_sequence import (
    FpCohomologyRing,
    reduce_fp_cohomology_ring,
    sphere_cohomology_fp,
)
from pysurgery.adams.u_resolution import (
    UnstableResolution,
    excess_p2,
    u_resolution_e2_page,
)


# ── Excess function ──────────────────────────────────────────────────────────


def test_excess_p2_empty_is_zero():
    assert excess_p2(()) == 0


def test_excess_p2_known_values():
    # excess(Sq^n) = n
    assert excess_p2((1,)) == 1
    assert excess_p2((3,)) == 3
    assert excess_p2((5,)) == 5
    # excess(Sq^a Sq^b) = a - b
    assert excess_p2((3, 1)) == 2  # 2*3 - (3+1) = 2
    assert excess_p2((4, 2)) == 2  # 2*4 - (4+2) = 2
    assert excess_p2((5, 2, 1)) == 2  # 2*5 - (5+2+1) = 2
    # Excess is never negative for admissibles (by definition).


# ── Sphere S^n ───────────────────────────────────────────────────────────────


def test_sphere_s2_has_bottom_cell():
    """E_2^{0, 2} = 1: the bottom cell of S^2."""
    red = reduce_fp_cohomology_ring(sphere_cohomology_fp(2, prime=2))
    page = u_resolution_e2_page(red, prime=2, s_max=3, t_max=8)
    assert page.e2_grid.get((0, 2), 0) == 1


def test_sphere_s2_first_resolution_step():
    """E_2^{1, 3} = E_2^{1, 4} = 1: Sq^1 ι_2 and Sq^2 ι_2 are kernel
    elements of the augmentation (since Sq^i(x) = 0 for i ≥ 1 on x ∈ deg 2).
    Sq^k for k > 2 has excess > 2, so doesn't act unstably on S^2.
    """
    red = reduce_fp_cohomology_ring(sphere_cohomology_fp(2, prime=2))
    page = u_resolution_e2_page(red, prime=2, s_max=3, t_max=8)
    assert page.e2_grid.get((1, 3), 0) == 1
    assert page.e2_grid.get((1, 4), 0) == 1
    # No (1, t) for t > 4 — the unstable filter excludes Sq^k with k > 2.
    for t in range(5, 9):
        assert page.e2_grid.get((1, t), 0) == 0, (
            f"unexpected E_2^{{1,{t}}} = {page.e2_grid[(1, t)]} on S^2"
        )


def test_sphere_s2_engine_label():
    red = reduce_fp_cohomology_ring(sphere_cohomology_fp(2, prime=2))
    page = u_resolution_e2_page(red, prime=2, s_max=3, t_max=8)
    assert "(U-resolution)" in page.space_label


def test_sphere_s2_status_success():
    red = reduce_fp_cohomology_ring(sphere_cohomology_fp(2, prime=2))
    page = u_resolution_e2_page(red, prime=2, s_max=3, t_max=8)
    assert page.status == "success"


# ── Π-bounds on π_n(S^2) for the Whitehead-kernel stems ──────────────────────


def test_sphere_s2_pi_4_bound():
    """Stem 4: π_4(S^2) = Z/2. The U-resolution bound should reflect this.

    For the 2-primary part, Σ_s dim E_2^{s, 4+s} must be ≥ log_2 |π_4,2|.
    """
    red = reduce_fp_cohomology_ring(sphere_cohomology_fp(2, prime=2))
    page = u_resolution_e2_page(red, prime=2, s_max=6, t_max=14)
    stem_4 = sum(
        page.e2_grid.get((s, 4 + s), 0) for s in range(0, 7)
    )
    # π_4(S^2) = Z/2; 2-primary part has order 2; bound ≥ 1.
    assert stem_4 >= 1, f"stem-4 bound {stem_4} too low (π_4(S^2) = Z/2)"


def test_sphere_s2_pi_5_bound():
    """Stem 5: π_5(S^2) = Z/2."""
    red = reduce_fp_cohomology_ring(sphere_cohomology_fp(2, prime=2))
    page = u_resolution_e2_page(red, prime=2, s_max=6, t_max=14)
    stem_5 = sum(
        page.e2_grid.get((s, 5 + s), 0) for s in range(0, 7)
    )
    assert stem_5 >= 1, f"stem-5 bound {stem_5} too low (π_5(S^2) = Z/2)"


def test_sphere_s2_pi_6_bound():
    """Stem 6: π_6(S^2) = Z/12; 2-primary part is Z/4 with log_2 = 2.

    The U-resolution should produce ≥ 2 cells at total stem 6 to upper-bound
    this. The OLD cobar approximation under-bounded this stem (1 cell), which
    was the Whitehead-kernel mistake. We verify the new engine fixes it.
    """
    red = reduce_fp_cohomology_ring(sphere_cohomology_fp(2, prime=2))
    page = u_resolution_e2_page(red, prime=2, s_max=6, t_max=14)
    stem_6 = sum(
        page.e2_grid.get((s, 6 + s), 0) for s in range(0, 7)
    )
    # Z/4 has 2-primary order 4 = 2^2; bound must be ≥ 2.
    assert stem_6 >= 2, (
        f"stem-6 bound {stem_6} too low; π_6(S^2)_2-prim = Z/4 requires ≥ 2 "
        "(this is the Whitehead-kernel test the cobar-approx engine fails)."
    )


def test_sphere_s2_pi_7_bound():
    """Stem 7: π_7(S^2) = π_7(S^3) = Z/2 (by Freudenthal)."""
    red = reduce_fp_cohomology_ring(sphere_cohomology_fp(2, prime=2))
    page = u_resolution_e2_page(red, prime=2, s_max=6, t_max=14)
    stem_7 = sum(
        page.e2_grid.get((s, 7 + s), 0) for s in range(0, 7)
    )
    assert stem_7 >= 1, f"stem-7 bound {stem_7} too low (π_7(S^2) = Z/2)"


# ── CP^2 (single A_2-generator at degree 2 with Sq^2 hitting x^2) ───────────


def _cp2_fp_ring() -> FpCohomologyRing:
    """H^*(CP^2; F_2) = F_2[x] / x^3, |x| = 2, Sq^2(x) = x^2.

    Sq^4(x^2) = (x^2)^2 = x^4 = 0 in CP^2. All other Sq^i (i ≥ 1) on the
    given basis are zero (by instability and the relations).
    """
    return FpCohomologyRing(
        space_label="CP^2",
        prime=2,
        max_degree=4,
        basis={0: ["1"], 2: ["x"], 4: ["x2"]},
        degree_of={"1": 0, "x": 2, "x2": 4},
        cup_table={("x", "x"): {"x2": 1}},
        sq_table={
            (1, "x"): {},
            (2, "x"): {"x2": 1},
            (3, "x"): {},
            (1, "x2"): {},
            (2, "x2"): {},
            (3, "x2"): {},
            (4, "x2"): {},
        },
        ring_generators=["x"],
    )


def test_cp2_single_a2_generator():
    """CP^2 has one A_2-generator in reduced cohomology: x.

    The class x^2 is in the A_2-orbit of x (via Sq^2). So the resolution
    finds exactly one F_0-generator at degree 2.
    """
    red = reduce_fp_cohomology_ring(_cp2_fp_ring())
    res = UnstableResolution(ring=red, prime=2, t_max=10)
    res.build(s_max=2)
    assert len(res.F[0]) == 1, (
        f"expected 1 A_2-generator of reduced CP^2; found {len(res.F[0])}"
    )
    assert res.F[0][0].degree == 2


def test_cp2_runs_and_has_bottom_cell():
    red = reduce_fp_cohomology_ring(_cp2_fp_ring())
    page = u_resolution_e2_page(red, prime=2, s_max=3, t_max=10)
    assert page.e2_grid.get((0, 2), 0) == 1
    assert page.status == "success"


# ── Trivial input: empty ring ────────────────────────────────────────────────


def test_empty_module_gives_empty_resolution():
    """A ring with no positive-degree basis should yield an empty E_2 grid
    (no A_p-generators of M means no F_s generators at any (s, t > 0)).
    """
    empty = FpCohomologyRing(
        space_label="empty",
        prime=2,
        max_degree=0,
        basis={0: []},
        degree_of={},
        cup_table={},
        sq_table={},
        ring_generators=[],
    )
    page = u_resolution_e2_page(empty, prime=2, s_max=2, t_max=4)
    # All e2_grid cells should be zero (no surviving entries).
    for d in page.e2_grid.values():
        assert d == 0


# ── Determinism: same input → same output ────────────────────────────────────


def test_u_resolution_is_deterministic():
    red = reduce_fp_cohomology_ring(sphere_cohomology_fp(2, prime=2))
    p1 = u_resolution_e2_page(red, prime=2, s_max=3, t_max=8)
    p2 = u_resolution_e2_page(red, prime=2, s_max=3, t_max=8)
    assert dict(p1.e2_grid) == dict(p2.e2_grid)


# ── Odd-prime U-resolution: now wired at p=3 and p=5 ─────────────────────────


def test_odd_prime_p3_runs_and_returns_e2_page():
    """At p=3 the U-resolution must produce an Ext_U^{s,t} page.

    Sphere S^2 over F_3 has a single generator x in degree 2 with x^2 = 0
    (and trivial action: P^1 lands in degree 6, which is below the t cutoff
    here for s = 0). The resolution should not crash and the (0, 2) cell
    must be 1 (the generator itself).
    """
    red = reduce_fp_cohomology_ring(sphere_cohomology_fp(2, prime=3))
    page = u_resolution_e2_page(red, prime=3, s_max=2, t_max=6)
    assert page.prime == 3
    assert page.e2_grid.get((0, 2), 0) == 1


def test_odd_prime_p5_runs_and_returns_e2_page():
    red = reduce_fp_cohomology_ring(sphere_cohomology_fp(2, prime=5))
    page = u_resolution_e2_page(red, prime=5, s_max=2, t_max=6)
    assert page.prime == 5
    assert page.e2_grid.get((0, 2), 0) == 1


# ── Dispatch round-trip: unstable_adams_e2_page should route to U at p=2 ─────


def test_dispatcher_auto_routes_to_u_resolution_at_p2():
    from pysurgery.adams.unstable import unstable_adams_e2_page
    red = reduce_fp_cohomology_ring(sphere_cohomology_fp(2, prime=2))
    page = unstable_adams_e2_page(red, prime=2, s_max=3, t_max=8)
    assert "(U-resolution)" in page.space_label


def test_dispatcher_cobar_approx_explicit():
    from pysurgery.adams.unstable import unstable_adams_e2_page
    red = reduce_fp_cohomology_ring(sphere_cohomology_fp(2, prime=2))
    page = unstable_adams_e2_page(
        red, prime=2, s_max=3, t_max=8, method="cobar_approx"
    )
    assert "(unstable approx)" in page.space_label
