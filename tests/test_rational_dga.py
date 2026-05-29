"""Tests for the RationalDGA layer (accessed via higher_homotopy_groups).

Coverage:
    - Basic polynomial algebra over ℚ
    - Wedge product associativity
    - d² = 0 verification
    - Small cohomology examples (S^3, S^2, ℂP^2)
    - verify_differential_closure and compute_cohomology helpers
"""
from fractions import Fraction

import pytest

from pysurgery.homotopy.higher_homotopy_groups import (
    RationalDGA,
    RationalCohomologyAlgebra,
    DGAElement,
    DegreeError,
    MinimalityError,
    verify_differential_closure,
    compute_cohomology,
    sphere_cohomology,
    complex_projective_space_cohomology,
    sullivan_minimal_model,
)
from pysurgery.homotopy.rational_homotopy import _mon_mul, UNIT


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def empty_dga():
    return RationalDGA()


@pytest.fixture()
def odd3_dga():
    """Minimal model of S^3: one generator x in degree 3, d(x)=0."""
    dga = RationalDGA()
    x = dga.add_generator(3, name="x")
    return dga, x


@pytest.fixture()
def s2_dga():
    """Minimal model of S^2: generators a (deg 2, d=0) and b (deg 3, d=a^2)."""
    dga = RationalDGA()
    a = dga.add_generator(2, name="a")
    b = dga.add_generator(3, name="b")
    a2 = DGAElement(dga, {((a.gid, 2),): Fraction(1)})
    dga.set_differential(b, a2)
    return dga, a, b


# ── Basic polynomial algebra ──────────────────────────────────────────────────


class TestBasicAlgebra:
    def test_add_generator_increments_gid(self, empty_dga):
        dga = empty_dga
        x = dga.add_generator(2, name="x")
        y = dga.add_generator(3, name="y")
        assert y.gid > x.gid

    def test_generator_degree_stored(self, empty_dga):
        dga = empty_dga
        g = dga.add_generator(5, name="g")
        assert g.degree == 5

    def test_generator_parity_even(self, empty_dga):
        dga = empty_dga
        g = dga.add_generator(4, name="g")
        assert g.parity == "even"

    def test_generator_parity_odd(self, empty_dga):
        dga = empty_dga
        g = dga.add_generator(3, name="g")
        assert g.parity == "odd"

    def test_degree_zero_raises(self, empty_dga):
        with pytest.raises(DegreeError):
            empty_dga.add_generator(0)

    def test_negative_degree_raises(self, empty_dga):
        with pytest.raises(DegreeError):
            empty_dga.add_generator(-1)

    def test_coefficients_are_exact_fractions(self, empty_dga):
        dga = empty_dga
        x = dga.add_generator(2, name="x")
        elt = DGAElement(dga, {((x.gid, 1),): Fraction(1, 3)})
        assert isinstance(list(elt.terms.values())[0], Fraction)

    def test_zero_coefficient_is_dropped(self, empty_dga):
        dga = empty_dga
        x = dga.add_generator(2, name="x")
        elt = DGAElement(dga, {((x.gid, 1),): Fraction(0)})
        assert elt.is_zero()

    def test_element_addition(self, empty_dga):
        dga = empty_dga
        x = dga.add_generator(2, name="x")
        mon = ((x.gid, 1),)
        e1 = DGAElement(dga, {mon: Fraction(1)})
        e2 = DGAElement(dga, {mon: Fraction(2)})
        s = e1 + e2
        assert s.terms[mon] == Fraction(3)

    def test_element_negation(self, empty_dga):
        dga = empty_dga
        x = dga.add_generator(2, name="x")
        mon = ((x.gid, 1),)
        e = DGAElement(dga, {mon: Fraction(5)})
        assert (-e).terms[mon] == Fraction(-5)

    def test_element_scale(self, empty_dga):
        dga = empty_dga
        x = dga.add_generator(2, name="x")
        mon = ((x.gid, 1),)
        e = DGAElement(dga, {mon: Fraction(1)})
        assert e.scale(Fraction(3, 2)).terms[mon] == Fraction(3, 2)

    def test_odd_generator_squared_is_zero(self, empty_dga):
        dga = empty_dga
        x = dga.add_generator(3, name="x")
        mon_x2 = ((x.gid, 2),)
        # graded_basis(6) should NOT contain x^2 for odd x.
        basis6 = dga.graded_basis(6)
        assert mon_x2 not in basis6

    def test_even_generator_squared_present(self, empty_dga):
        dga = empty_dga
        x = dga.add_generator(2, name="x")
        mon_x2 = ((x.gid, 2),)
        basis4 = dga.graded_basis(4)
        assert mon_x2 in basis4


# ── Wedge product associativity ───────────────────────────────────────────────


class TestWedgeProduct:
    def test_wedge_two_even_generators_commutes(self, empty_dga):
        dga = empty_dga
        a = dga.add_generator(2, name="a")
        b = dga.add_generator(2, name="b")
        mon_ab, s1 = _mon_mul(((a.gid, 1),), ((b.gid, 1),), dga._g2n)
        mon_ba, s2 = _mon_mul(((b.gid, 1),), ((a.gid, 1),), dga._g2n)
        # Both should give the same canonical monomial.
        assert mon_ab == mon_ba

    def test_wedge_with_unit(self, empty_dga):
        dga = empty_dga
        x = dga.add_generator(3, name="x")
        mon, sign = _mon_mul(UNIT, ((x.gid, 1),), dga._g2n)
        assert mon == ((x.gid, 1),)
        assert sign == 1

    def test_wedge_associativity_three_factors(self, empty_dga):
        """(a ∧ b) ∧ c = a ∧ (b ∧ c) as monomials."""
        dga = empty_dga
        a = dga.add_generator(2, name="a")
        b = dga.add_generator(2, name="b")
        c = dga.add_generator(2, name="c")
        ab, _ = _mon_mul(((a.gid, 1),), ((b.gid, 1),), dga._g2n)
        abc_left, _ = _mon_mul(ab, ((c.gid, 1),), dga._g2n)
        bc, _ = _mon_mul(((b.gid, 1),), ((c.gid, 1),), dga._g2n)
        abc_right, _ = _mon_mul(((a.gid, 1),), bc, dga._g2n)
        assert abc_left == abc_right

    def test_koszul_sign_odd_generators(self, empty_dga):
        """For odd-degree x, y: x ∧ y = -(y ∧ x)."""
        dga = empty_dga
        x = dga.add_generator(1, name="x")
        y = dga.add_generator(1, name="y")
        _, s_xy = _mon_mul(((x.gid, 1),), ((y.gid, 1),), dga._g2n)
        _, s_yx = _mon_mul(((y.gid, 1),), ((x.gid, 1),), dga._g2n)
        assert s_xy * s_yx == -1

    def test_graded_basis_unit_in_degree_zero(self, empty_dga):
        basis0 = empty_dga.graded_basis(0)
        assert basis0 == [UNIT]

    def test_graded_basis_count_two_even_generators(self, empty_dga):
        dga = empty_dga
        dga.add_generator(2, name="a")
        dga.add_generator(2, name="b")
        # Degree 4: a^2, a∧b, b^2
        basis4 = dga.graded_basis(4)
        assert len(basis4) == 3


# ── d² = 0 verification ───────────────────────────────────────────────────────


class TestDSquaredZero:
    def test_trivial_differential_passes(self, odd3_dga):
        dga, _ = odd3_dga
        assert verify_differential_closure(dga)

    def test_s2_model_passes(self, s2_dga):
        dga, _, _ = s2_dga
        assert verify_differential_closure(dga)

    def test_empty_dga_passes(self, empty_dga):
        assert verify_differential_closure(empty_dga)

    def test_degree_check_on_set_differential(self, empty_dga):
        dga = empty_dga
        x = dga.add_generator(4, name="x")
        y = dga.add_generator(2, name="y")
        y2 = DGAElement(dga, {((y.gid, 2),): Fraction(1)})
        # y2 has degree 4, but d(x) must have degree 5 — should raise.
        with pytest.raises(DegreeError):
            dga.set_differential(x, y2)

    def test_minimality_enforced(self, empty_dga):
        dga = empty_dga
        # x has degree 3; d(x) must land in degree 4.
        x = dga.add_generator(3, name="x")
        # y has degree 4 — so d(x) = y is linear (word-length 1) → MinimalityError.
        y = dga.add_generator(4, name="y")
        y_elt = DGAElement(dga, {((y.gid, 1),): Fraction(1)})
        with pytest.raises(MinimalityError):
            dga.set_differential(x, y_elt)

    def test_indecomposables_dim(self, s2_dga):
        dga, a, b = s2_dga
        # Two generators: one in degree 2 (a), one in degree 3 (b).
        assert dga.indecomposables_dim(2) == 1
        assert dga.indecomposables_dim(3) == 1
        assert dga.indecomposables_dim(4) == 0


# ── Small cohomology examples ─────────────────────────────────────────────────


class TestCohomologyExamples:
    def test_odd_sphere_s3_cohomology(self):
        """H^*(S^3; ℚ): ℚ in degrees 0 and 3 only."""
        alg = sphere_cohomology(3)
        r = sullivan_minimal_model(alg, max_degree=6)
        dga = r.minimal_model
        dims = compute_cohomology(dga, max_degree=6)
        assert dims.get(0) == 1
        assert dims.get(3) == 1
        for n in [1, 2, 4, 5, 6]:
            assert dims.get(n, 0) == 0

    def test_even_sphere_s2_cohomology(self):
        """H^*(S^2; ℚ): ℚ in degrees 0 and 2 only (up to degree 6)."""
        alg = sphere_cohomology(2)
        r = sullivan_minimal_model(alg, max_degree=6)
        dga = r.minimal_model
        dims = compute_cohomology(dga, max_degree=6)
        assert dims.get(0) == 1
        assert dims.get(2) == 1
        for n in [1, 4, 5, 6]:
            assert dims.get(n, 0) == 0

    def test_cp2_cohomology(self):
        """H^*(ℂP^2; ℚ): ℚ in degrees 0, 2, 4."""
        alg = complex_projective_space_cohomology(2)
        r = sullivan_minimal_model(alg, max_degree=6)
        dga = r.minimal_model
        dims = compute_cohomology(dga, max_degree=6)
        assert dims.get(0) == 1
        assert dims.get(2) == 1
        assert dims.get(4) == 1
        for n in [1, 3, 5, 6]:
            assert dims.get(n, 0) == 0

    def test_minimal_model_result_exact(self):
        alg = sphere_cohomology(3)
        r = sullivan_minimal_model(alg)
        assert r.exact is True
        assert r.status == "success"
        assert r.decision_ready() is True

    def test_simply_connected_required(self):
        """Sullivan algorithm should decline when β_1 > 0."""
        non_sc = RationalCohomologyAlgebra(betti={0: 1, 1: 1}, max_degree=4)
        r = sullivan_minimal_model(non_sc)
        assert r.status == "inconclusive"

    def test_pi_n_rational_s3(self):
        """π_3(S^3) ⊗ ℚ = ℚ (one generator in degree 3)."""
        alg = sphere_cohomology(3)
        r = sullivan_minimal_model(alg)
        assert r.pi_n_rational.get(3, 0) == 1

    def test_pi_n_rational_s2(self):
        """S^2 has π_2 ⊗ ℚ = ℚ and π_3 ⊗ ℚ = ℚ (Hopf fibration)."""
        alg = sphere_cohomology(2)
        r = sullivan_minimal_model(alg, max_degree=6)
        assert r.pi_n_rational.get(2, 0) == 1
        assert r.pi_n_rational.get(3, 0) == 1

    def test_verify_differential_closure_helper(self):
        alg = sphere_cohomology(2)
        r = sullivan_minimal_model(alg)
        assert verify_differential_closure(r.minimal_model)

    def test_compute_cohomology_returns_dict(self):
        alg = sphere_cohomology(3)
        r = sullivan_minimal_model(alg)
        dims = compute_cohomology(r.minimal_model, max_degree=5)
        assert isinstance(dims, dict)
        assert 0 in dims
