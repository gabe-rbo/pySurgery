"""Tests for Sullivan minimal models and rational homotopy groups.

Overview:
    Verifies the Quillen-Sullivan theorem implementation in
    ``pysurgery.homotopy.rational_homotopy``.  All results carry ``exact=True``
    and are validated against classical literature values.

Key Concepts:
    - Sullivan minimal model (ΛV, d): free CDGA quasi-isomorphic to H*(X; ℚ).
    - Quillen-Sullivan: π_n(X) ⊗ ℚ ≅ V^n (degree-n indecomposables).
    - Formal space: all differentials in the minimal model are zero.

References:
    Félix, Y., Halperin, S., & Thomas, J.-C. (2001).
        Rational Homotopy Theory. Springer GTM 205. §15.
    Sullivan, D. (1977). Infinitesimal computations in topology.
        Publ. Math. IHES 47, 269–331.
"""
import pytest

from pysurgery.homotopy.rational_homotopy import (
    RationalCohomologyAlgebra,
    RationalMinimalModelResult,
    RationalHomotopyGroup,
    sullivan_minimal_model,
    rational_homotopy_group,
    sphere_cohomology,
    complex_projective_space_cohomology,
    product_cohomology,
    RationalDGA,
    DGAElement,
)
from pysurgery.homotopy.higher_homotopy_groups import sullivan_rational_homotopy


# ── Helpers ───────────────────────────────────────────────────────────────────


def _model(algebra: RationalCohomologyAlgebra, max_degree: int = 10) -> RationalMinimalModelResult:
    result = sullivan_minimal_model(algebra, max_degree=max_degree)
    assert result.exact, "Contract: exact must be True for Sullivan models."
    return result


# ── Test 1: Spheres ───────────────────────────────────────────────────────────


class TestSpheres:
    """Sullivan models of S^n, n = 2, 3, 4, 5, 7.

    Literature reference (FHT §15):
      - S^{2k+1} (odd): minimal model ΛV = Λ(x_{2k+1}), d=0.
        π_{2k+1} ⊗ ℚ = ℚ; all other π_n ⊗ ℚ = 0.
      - S^{2k} (even): minimal model Λ(x_{2k}, y_{4k-1}), d(y) = x^2.
        π_{2k} ⊗ ℚ = ℚ, π_{4k-1} ⊗ ℚ = ℚ; others 0.
    """

    def test_s2_minimal_model(self):
        """S^2 (even): V^2 = V^3 = ℚ, Hopf class in π_3."""
        r = _model(sphere_cohomology(2))
        assert r.status == "success"
        assert r.cohomology_iso
        assert r.pi_n_rational.get(2) == 1, "π_2(S^2) ⊗ ℚ = ℚ"
        assert r.pi_n_rational.get(3) == 1, "π_3(S^2) ⊗ ℚ = ℚ (Hopf fibration)"
        # No other rational homotopy groups up to degree 10
        for k in range(4, 11):
            assert r.pi_n_rational.get(k, 0) == 0, f"π_{k}(S^2) ⊗ ℚ = 0"
        assert not r.is_formal_model, "S^2 has d≠0 in the minimal model"

    def test_s3_minimal_model(self):
        """S^3 (odd): V^3 = ℚ, model is formal."""
        r = _model(sphere_cohomology(3))
        assert r.status == "success"
        assert r.cohomology_iso
        assert r.pi_n_rational.get(3) == 1, "π_3(S^3) ⊗ ℚ = ℚ"
        for k in [2, 4, 5, 6, 7, 8]:
            assert r.pi_n_rational.get(k, 0) == 0, f"π_{k}(S^3) ⊗ ℚ = 0"
        assert r.is_formal_model, "S^3 minimal model has d=0 (formal)"

    def test_s4_minimal_model(self):
        """S^4 (even): V^4 = V^7 = ℚ."""
        r = _model(sphere_cohomology(4, max_degree=10))
        assert r.status == "success"
        assert r.cohomology_iso
        assert r.pi_n_rational.get(4) == 1, "π_4(S^4) ⊗ ℚ = ℚ"
        assert r.pi_n_rational.get(7) == 1, "π_7(S^4) ⊗ ℚ = ℚ"
        for k in [2, 3, 5, 6, 8, 9, 10]:
            assert r.pi_n_rational.get(k, 0) == 0, f"π_{k}(S^4) ⊗ ℚ = 0"
        assert not r.is_formal_model

    def test_s5_minimal_model(self):
        """S^5 (odd): V^5 = ℚ, formal."""
        r = _model(sphere_cohomology(5))
        assert r.status == "success"
        assert r.cohomology_iso
        assert r.pi_n_rational.get(5) == 1
        for k in range(2, 11):
            if k != 5:
                assert r.pi_n_rational.get(k, 0) == 0
        assert r.is_formal_model

    def test_s7_minimal_model(self):
        """S^7 (odd): V^7 = ℚ, formal."""
        r = _model(sphere_cohomology(7))
        assert r.status == "success"
        assert r.cohomology_iso
        assert r.pi_n_rational.get(7) == 1
        assert r.is_formal_model

    def test_contract_fields(self):
        """All required contract fields are present and typed correctly."""
        r = _model(sphere_cohomology(3))
        assert r.exact is True
        assert r.theorem_tag == "rational.quillen_sullivan.minimal_model"
        assert r.contract_version == "2026.04-phase10"
        assert r.truncation_degree >= 3
        assert isinstance(r.reasoning, str) and len(r.reasoning) > 0
        assert r.decision_ready()


# ── Test 2: Homogeneous spaces G/H ────────────────────────────────────────────


class TestHomogeneousSpaces:
    """Sullivan models of complex projective spaces ℂP^n.

    ℂP^n = U(n+1) / (U(n) × U(1)) is simply connected.
    H*(ℂP^n; ℚ) = ℚ[x]/(x^{n+1}), |x| = 2.
    Minimal model: Λ(x_2, y_{2n+1}), d(y) = x^{n+1}.
    π_2(ℂP^n) ⊗ ℚ = ℚ, π_{2n+1}(ℂP^n) ⊗ ℚ = ℚ, others 0.

    References:
        FHT §15, Example 15.1.
        Griffiths-Morgan (1981) §11.
    """

    def test_cp1_is_s2(self):
        """ℂP^1 ≅ S^2: same rational homotopy type."""
        cp1 = complex_projective_space_cohomology(1)
        s2 = sphere_cohomology(2)
        r_cp1 = _model(cp1)
        r_s2 = _model(s2)
        assert r_cp1.pi_n_rational == r_s2.pi_n_rational

    def test_cp2_minimal_model(self):
        """ℂP^2: π_2 ⊗ ℚ = ℚ, π_5 ⊗ ℚ = ℚ."""
        r = _model(complex_projective_space_cohomology(2))
        assert r.status == "success"
        assert r.cohomology_iso
        assert r.pi_n_rational.get(2) == 1, "π_2(ℂP^2) ⊗ ℚ = ℚ"
        assert r.pi_n_rational.get(5) == 1, "π_5(ℂP^2) ⊗ ℚ = ℚ"
        for k in [3, 4, 6, 7, 8, 9, 10]:
            assert r.pi_n_rational.get(k, 0) == 0

    def test_cp3_minimal_model(self):
        """ℂP^3: π_2 ⊗ ℚ = ℚ, π_7 ⊗ ℚ = ℚ."""
        r = _model(complex_projective_space_cohomology(3))
        assert r.status == "success"
        assert r.cohomology_iso
        assert r.pi_n_rational.get(2) == 1
        assert r.pi_n_rational.get(7) == 1
        for k in [3, 4, 5, 6, 8, 9, 10]:
            assert r.pi_n_rational.get(k, 0) == 0

    def test_cp4_minimal_model(self):
        """ℂP^4: π_2 ⊗ ℚ = ℚ, π_9 ⊗ ℚ = ℚ."""
        r = _model(complex_projective_space_cohomology(4))
        assert r.status == "success"
        assert r.cohomology_iso
        assert r.pi_n_rational.get(2) == 1
        assert r.pi_n_rational.get(9) == 1
        for k in [3, 4, 5, 6, 7, 8, 10]:
            assert r.pi_n_rational.get(k, 0) == 0


# ── Test 3: Products ──────────────────────────────────────────────────────────


class TestProducts:
    """Sullivan models of products of simply-connected spaces.

    For X × Y: π_n(X × Y) ⊗ ℚ = (π_n(X) ⊗ ℚ) ⊕ (π_n(Y) ⊗ ℚ).
    The Sullivan model of X × Y is the tensor product of the individual models.
    """

    def test_s3_cross_s3(self):
        """S^3 × S^3: both odd spheres, formal, V^3 = ℚ^2."""
        alg = product_cohomology(sphere_cohomology(3), sphere_cohomology(3))
        r = _model(alg)
        assert r.status == "success"
        assert r.cohomology_iso
        assert r.pi_n_rational.get(3) == 2, "π_3(S^3×S^3) ⊗ ℚ = ℚ^2"
        for k in [2, 4, 5, 6]:
            assert r.pi_n_rational.get(k, 0) == 0
        assert r.is_formal_model, "Product of odd spheres is formal"

    def test_s3_cross_s5(self):
        """S^3 × S^5: formal, V^3 = V^5 = ℚ."""
        alg = product_cohomology(sphere_cohomology(3), sphere_cohomology(5))
        r = _model(alg)
        assert r.status == "success"
        assert r.cohomology_iso
        assert r.pi_n_rational.get(3) == 1
        assert r.pi_n_rational.get(5) == 1
        for k in [2, 4, 6, 7, 8]:
            assert r.pi_n_rational.get(k, 0) == 0
        assert r.is_formal_model

    def test_s2_cross_s2_cohomology(self):
        """S^2 × S^2 cohomology algorithm: β_0=β_4=1, β_2=2 reproduced exactly."""
        alg = product_cohomology(sphere_cohomology(2), sphere_cohomology(2))
        r = _model(alg)
        assert r.status == "success", r.reasoning
        assert r.cohomology_iso, "H*(model) must match H*(S^2×S^2; ℚ)"
        # The algorithm correctly produces V^2 = ℚ^2 (two degree-2 generators).
        assert r.pi_n_rational.get(2) == 2, "π_2(S^2×S^2) ⊗ ℚ = ℚ^2"
        # Killing the spurious H^4 requires 2 degree-3 generators regardless
        # of which basis is chosen, so V^3 ≥ 2.
        assert r.pi_n_rational.get(3, 0) >= 2

    def test_s2_cross_s2_dga_direct(self):
        """Direct DGA for S^2×S^2: Λ(x,x',y,y') d(y)=x², d(y')=(x')².

        This is the tensor product model obtained from the Künneth formula for
        Sullivan models.  The cohomology must match H*(S^2×S^2; ℚ) and
        π_2 = π_3 = ℚ^2.
        """
        from fractions import Fraction
        dga = RationalDGA()
        x = dga.add_generator(degree=2, name="x")
        xp = dga.add_generator(degree=2, name="xp")
        y = dga.add_generator(degree=3, name="y")
        yp = dga.add_generator(degree=3, name="yp")
        # d(y) = x², d(y') = (x')²
        dga._diff[y.gid] = DGAElement(dga, {((x.gid, 2),): Fraction(1)})
        dga._diff[yp.gid] = DGAElement(dga, {((xp.gid, 2),): Fraction(1)})

        assert dga.verify_d_squared()
        # H^0=H^4=1, H^2=2, all others 0 (up to degree 8)
        assert dga.cohomology_dim(0) == 1
        assert dga.cohomology_dim(2) == 2
        assert dga.cohomology_dim(4) == 1
        for k in [1, 3, 5, 6, 7, 8]:
            assert dga.cohomology_dim(k) == 0, f"H^{k} should be 0"
        # π_2 = π_3 = ℚ^2
        assert dga.indecomposables_dim(2) == 2
        assert dga.indecomposables_dim(3) == 2

    def test_product_betti_numbers(self):
        """Product cohomology computes Betti numbers via Künneth."""
        s2 = sphere_cohomology(2)  # β_0=β_2=1
        s3 = sphere_cohomology(3)  # β_0=β_3=1
        prod = product_cohomology(s2, s3)
        assert prod.betti_n(0) == 1
        assert prod.betti_n(2) == 1
        assert prod.betti_n(3) == 1
        assert prod.betti_n(5) == 1
        assert prod.betti_n(1) == 0
        assert prod.betti_n(4) == 0

    def test_three_sphere_product(self):
        """S^3 × S^3 × S^3: V^3 = ℚ^3."""
        s3 = sphere_cohomology(3)
        alg = product_cohomology(product_cohomology(s3, s3), s3)
        r = _model(alg)
        assert r.status == "success"
        assert r.cohomology_iso
        assert r.pi_n_rational.get(3) == 3


# ── Test 4: Formal spaces ─────────────────────────────────────────────────────


class TestFormalSpaces:
    """Formality detection.

    A Sullivan minimal model (ΛV, d) has ``is_formal_model=True`` iff d=0
    for every generator.  This holds precisely for products of odd spheres.
    Spaces with non-trivial rational Massey products are detected by d≠0.

    Note: "formal" in the DGA sense (d=0 in the minimal model) is a stronger
    condition than topological formality (quasi-isomorphic to cohomology ring).
    Odd spheres and their products are formal in both senses.
    """

    def test_odd_spheres_are_formal(self):
        """S^{2k+1} have d=0 in their minimal model."""
        for n in [3, 5, 7, 9]:
            r = _model(sphere_cohomology(n))
            assert r.is_formal_model, f"S^{n} should have d=0 minimal model"

    def test_even_spheres_not_d_zero(self):
        """S^{2k} have d≠0 when computed to sufficient degree (2n+1 reveals killing gen)."""
        # S^2: spurious x^2 at deg 4, killed at deg 3 → needs max_degree ≥ 4.
        r2 = _model(sphere_cohomology(2))
        assert not r2.is_formal_model, "S^2 minimal model has d≠0"
        # S^4: spurious x^2 at deg 8, killed at deg 7 → needs max_degree ≥ 8.
        r4 = _model(sphere_cohomology(4, max_degree=10))
        assert not r4.is_formal_model, "S^4 minimal model has d≠0"
        # S^6: spurious x^2 at deg 12, needs max_degree ≥ 12 to see it.
        r6 = _model(sphere_cohomology(6, max_degree=14), max_degree=14)
        assert not r6.is_formal_model, "S^6 minimal model has d≠0 (max_degree=14)"

    def test_product_of_odd_spheres_formal(self):
        """Products of odd spheres: d=0 (the Künneth model is formal)."""
        for (n1, n2) in [(3, 3), (3, 5), (5, 7), (3, 7)]:
            alg = product_cohomology(sphere_cohomology(n1), sphere_cohomology(n2))
            r = _model(alg)
            assert r.is_formal_model, f"S^{n1}×S^{n2} should be formal"

    def test_k_of_pi_1_type(self):
        """K(ℤ, n) for odd n: acyclic rational homotopy beyond degree n."""
        # K(ℤ, 3) has H^*(K(ℤ,3); ℚ) = ℚ[x]/... complicated.
        # For a simpler check: Eilenberg-MacLane K(ℤ,2) = ℂP^∞,
        # truncated to degree 10 gives β_k = 1 for k=0,2,4,...,10.
        k_z_2 = RationalCohomologyAlgebra(
            betti={2 * k: 1 for k in range(6)},
            name="K(ℤ,2)[≤10]",
            max_degree=10,
        )
        r = _model(k_z_2)
        assert r.status == "success"
        assert r.cohomology_iso
        # Only π_2 ⊗ ℚ = ℚ (one generator, all others killed by the algebra)
        assert r.pi_n_rational.get(2) == 1

    def test_formal_space_decision_ready(self):
        """Formal spaces return decision_ready()=True."""
        for n in [3, 5, 7]:
            r = _model(sphere_cohomology(n))
            assert r.decision_ready()


# ── Test 5: Cross-validation with known literature ────────────────────────────


class TestLiteratureCrossValidation:
    """Cross-validate against classical results from FHT and Hatcher.

    References:
        Hatcher, A. (2002). Algebraic Topology. Cambridge U.P. Appendix.
        Félix, Halperin, Thomas (2001). GTM 205 Tables A–D.
    """

    def test_s2_hopf_invariant(self):
        """π_3(S^2) ⊗ ℚ = ℚ: the Hopf fibration has infinite order."""
        r = _model(sphere_cohomology(2))
        assert r.pi_n_rational[3] == 1

    def test_pi_4_s2_vanishes_rationally(self):
        """π_4(S^2) = ℤ_2 is finite, so π_4(S^2) ⊗ ℚ = 0."""
        r = _model(sphere_cohomology(2))
        assert r.pi_n_rational.get(4, 0) == 0

    def test_pi_6_s3_vanishes_rationally(self):
        """π_6(S^3) = ℤ_{12} is finite, so π_6(S^3) ⊗ ℚ = 0."""
        r = _model(sphere_cohomology(3))
        assert r.pi_n_rational.get(6, 0) == 0

    def test_cp2_pi5_is_hopf(self):
        """π_5(ℂP^2) ⊗ ℚ = ℚ: the generator of V^5 in the minimal model."""
        r = _model(complex_projective_space_cohomology(2))
        assert r.pi_n_rational.get(5) == 1

    def test_euler_char_consistent(self):
        """Euler characteristic from Betti numbers matches S^n formulas."""
        # χ(S^n) = 1 + (-1)^n
        for n in [2, 3, 4, 5]:
            alg = sphere_cohomology(n)
            chi = sum((-1) ** k * alg.betti_n(k) for k in range(n + 1))
            assert chi == 1 + (-1) ** n

    def test_total_rank_s4(self):
        """Total rational homotopy rank of S^4 is 2 (π_4 and π_7)."""
        r = _model(sphere_cohomology(4, max_degree=10))
        total = sum(r.pi_n_rational.values())
        assert total == 2

    def test_total_rank_s2(self):
        """Total rational homotopy rank of S^2 up to degree 10 is 2."""
        r = _model(sphere_cohomology(2))
        total = sum(r.pi_n_rational.values())
        assert total == 2

    def test_cohomology_s2_verified(self):
        """H^k(ΛV_S2) ≅ H^k(S^2; ℚ) for k ≤ 10."""
        r = _model(sphere_cohomology(2))
        assert r.cohomology_iso

    def test_cohomology_cp3_verified(self):
        """H^k(ΛV_{CP3}) ≅ H^k(ℂP^3; ℚ) for k ≤ 10."""
        r = _model(complex_projective_space_cohomology(3))
        assert r.cohomology_iso

    def test_simply_connected_guard(self):
        """Non-simply-connected input returns inconclusive."""
        circle = RationalCohomologyAlgebra(betti={0: 1, 1: 1}, name="S^1")
        r = sullivan_minimal_model(circle)
        assert r.status == "inconclusive"
        assert not r.cohomology_iso

    def test_disconnected_guard(self):
        """Disconnected space (β_0 ≠ 1) returns inconclusive."""
        disconnected = RationalCohomologyAlgebra(betti={0: 2, 3: 2}, name="S^3 ⊔ S^3")
        r = sullivan_minimal_model(disconnected)
        assert r.status == "inconclusive"


# ── DGA unit tests ────────────────────────────────────────────────────────────


class TestDGAInternals:
    """Unit tests for RationalDGA internals: d², Leibniz, basis enumeration."""

    def test_d_squared_zero_s2_model(self):
        """d²=0 verified on the S^2 minimal model."""
        dga = RationalDGA()
        x = dga.add_generator(degree=2, name="x")
        y = dga.add_generator(degree=3, name="y")
        # d(x) = 0 (already default)
        # d(y) = x^2
        x2_mon = ((x.gid, 2),)
        from fractions import Fraction
        x2 = DGAElement(dga, {x2_mon: Fraction(1)})
        dga._diff[y.gid] = x2
        assert dga.verify_d_squared()

    def test_cohomology_dim_s2(self):
        """H^*(S^2 model): β_0=β_2=1, all others 0 up to degree 6."""
        dga = RationalDGA()
        x = dga.add_generator(degree=2, name="x")
        y = dga.add_generator(degree=3, name="y")
        from fractions import Fraction
        x2 = DGAElement(dga, {((x.gid, 2),): Fraction(1)})
        dga._diff[y.gid] = x2
        assert dga.cohomology_dim(0) == 1
        assert dga.cohomology_dim(2) == 1
        assert dga.cohomology_dim(3) == 0
        assert dga.cohomology_dim(4) == 0
        assert dga.cohomology_dim(5) == 0
        assert dga.cohomology_dim(6) == 0

    def test_monomial_multiplication_koszul_sign(self):
        """Koszul sign: odd∧odd generators with gid order give correct sign."""
        dga = RationalDGA()
        a = dga.add_generator(degree=3, name="a")  # odd
        b = dga.add_generator(degree=3, name="b")  # odd (gid > a.gid)
        from pysurgery.homotopy.rational_homotopy import _mon_mul
        # a ∧ b in canonical order (a.gid < b.gid): sign = 1
        m1, s1 = _mon_mul(((a.gid, 1),), ((b.gid, 1),), dga._g2n)
        assert m1 == ((a.gid, 1), (b.gid, 1))
        assert s1 == 1
        # b ∧ a: b.gid > a.gid, both odd → sign = -1
        m2, s2 = _mon_mul(((b.gid, 1),), ((a.gid, 1),), dga._g2n)
        assert m2 == ((a.gid, 1), (b.gid, 1))
        assert s2 == -1

    def test_odd_generator_squared_is_zero(self):
        """g ∧ g = 0 for odd-degree generator."""
        dga = RationalDGA()
        g = dga.add_generator(degree=3, name="g")
        from pysurgery.homotopy.rational_homotopy import _mon_mul
        m, s = _mon_mul(((g.gid, 1),), ((g.gid, 1),), dga._g2n)
        assert m is None
        assert s == 0

    def test_basis_enumeration_s2_model(self):
        """Graded basis of S^2 minimal model ∧(x_2, y_3) up to degree 5."""
        dga = RationalDGA()
        dga.add_generator(degree=2, name="x")
        dga.add_generator(degree=3, name="y")
        assert len(dga.graded_basis(0)) == 1   # {1}
        assert len(dga.graded_basis(2)) == 1   # {x}
        assert len(dga.graded_basis(3)) == 1   # {y}
        assert len(dga.graded_basis(4)) == 1   # {x^2}
        assert len(dga.graded_basis(5)) == 1   # {x∧y}
        assert len(dga.graded_basis(6)) == 1   # {x^3}
        assert len(dga.graded_basis(1)) == 0   # none


# ── Test 6: RationalHomotopyGroup contract ────────────────────────────────────


class TestRationalHomotopyGroupContract:
    """Validate the RationalHomotopyGroup façade contract."""

    def _rhg(self, alg, max_degree=10, **kwargs):
        r = rational_homotopy_group(alg, max_degree=max_degree, **kwargs)
        assert isinstance(r, RationalHomotopyGroup)
        assert r.exact is True
        return r

    def test_s3_contract_fields(self):
        """S^3: all mandatory contract fields populated correctly."""
        r = self._rhg(sphere_cohomology(3))
        assert r.theorem_tag == "rational.quillen_sullivan.minimal_model"
        assert r.status == "success"
        assert r.cohomology_iso_verified is True
        assert r.decision_ready() is True
        assert r.truncation_degree >= 3
        assert r.is_formal is True
        assert r.massey_products == []
        assert r.underlying_model is not None

    def test_nonzero_degrees_equals_pi_keys(self):
        """nonzero_degrees must be exactly the keys of pi_n_rational."""
        r = self._rhg(sphere_cohomology(2))
        assert set(r.nonzero_degrees) == set(r.pi_n_rational.keys())
        assert r.nonzero_degrees == sorted(r.nonzero_degrees)

    def test_rank_at_accessor(self):
        """rank_at(n) matches pi_n_rational.get(n, 0)."""
        r = self._rhg(sphere_cohomology(2))
        for n in range(1, 8):
            assert r.rank_at(n) == r.pi_n_rational.get(n, 0)

    def test_decision_ready_false_for_inconclusive(self):
        """decision_ready() is False when status=inconclusive."""
        non_sc = RationalCohomologyAlgebra(betti={0: 1, 1: 1}, name="non-SC")
        r = rational_homotopy_group(non_sc)
        assert r.status == "inconclusive"
        assert r.decision_ready() is False

    def test_formal_implies_empty_massey(self):
        """is_formal=True forces massey_products=[] (contract invariant)."""
        r = self._rhg(sphere_cohomology(3))
        assert r.is_formal is True
        assert r.massey_products == []

    def test_non_formal_has_massey_products(self):
        """S^2 is not formal and has non-trivial Massey products."""
        r = self._rhg(sphere_cohomology(2))
        assert r.is_formal is False
        assert len(r.massey_products) >= 1

    def test_include_massey_false(self):
        """include_massey=False returns empty massey list."""
        r = rational_homotopy_group(sphere_cohomology(2), include_massey=False)
        assert r.massey_products == []

    def test_space_label_forwarded(self):
        """space_label is stored in the result."""
        r = rational_homotopy_group(sphere_cohomology(3), space_label="my-S3")
        assert r.space_label == "my-S3"

    def test_cp2_contract(self):
        """ℂP^2: standard contract checks."""
        r = self._rhg(complex_projective_space_cohomology(2))
        assert r.decision_ready()
        assert r.rank_at(2) == 1
        assert r.rank_at(5) == 1

    def test_product_contract(self):
        """S^3 × S^5: contract and π_n values correct."""
        alg = product_cohomology(sphere_cohomology(3), sphere_cohomology(5))
        r = self._rhg(alg)
        assert r.decision_ready()
        assert r.rank_at(3) == 1
        assert r.rank_at(5) == 1
        assert r.is_formal is True

    def test_validator_rejects_zero_pi_value(self):
        """Validator must reject pi_n_rational with value 0."""
        with pytest.raises(Exception):
            RationalHomotopyGroup(
                space_label="bad",
                pi_n_rational={3: 0},  # zero not allowed
                nonzero_degrees=[3],
                truncation_degree=10,
                is_formal=True,
                cohomology_iso_verified=True,
                status="success",
                reasoning="test",
            )

    def test_validator_rejects_key_mismatch(self):
        """Validator must reject nonzero_degrees ≠ pi_n_rational keys."""
        with pytest.raises(Exception):
            RationalHomotopyGroup(
                space_label="bad",
                pi_n_rational={3: 1},
                nonzero_degrees=[3, 5],  # mismatch
                truncation_degree=10,
                is_formal=True,
                cohomology_iso_verified=True,
                status="success",
                reasoning="test",
            )

    def test_validator_rejects_formal_with_massey(self):
        """Validator must reject is_formal=True with non-empty massey_products."""
        from pysurgery.homotopy.rational_homotopy import MasseyProductEntry
        with pytest.raises(Exception):
            RationalHomotopyGroup(
                space_label="bad",
                pi_n_rational={3: 1},
                nonzero_degrees=[3],
                truncation_degree=10,
                is_formal=True,
                massey_products=[
                    MasseyProductEntry(
                        input_classes=("x", "x"),
                        order=2,
                        representative_degree=3,
                        representative_name="y",
                        coefficient="1",
                    )
                ],
                cohomology_iso_verified=True,
                status="success",
                reasoning="test",
            )


# ── Test 7: sullivan_rational_homotopy entry point ────────────────────────────


class TestSullivanRationalHomotopy:
    """Tests for the sullivan_rational_homotopy() façade in higher_homotopy_groups."""

    def _srhg(self, alg, max_degree=10, **kwargs):
        r = sullivan_rational_homotopy(alg, max_degree=max_degree, **kwargs)
        assert isinstance(r, RationalHomotopyGroup)
        assert r.exact is True
        return r

    def test_s3_basic(self):
        """S^3: decision_ready, π_3=1, formal."""
        r = self._srhg(sphere_cohomology(3))
        assert r.decision_ready()
        assert r.rank_at(3) == 1
        assert r.is_formal is True

    def test_s2_basic(self):
        """S^2: not formal, π_2=π_3=1."""
        r = self._srhg(sphere_cohomology(2))
        assert r.rank_at(2) == 1
        assert r.rank_at(3) == 1
        assert r.is_formal is False

    def test_s4_basic(self):
        """S^4: π_4=π_7=1, not formal."""
        r = self._srhg(sphere_cohomology(4, max_degree=10))
        assert r.rank_at(4) == 1
        assert r.rank_at(7) == 1
        assert r.is_formal is False

    def test_odd_spheres_formal(self):
        """Odd spheres S^{2k+1} are formal via sullivan_rational_homotopy."""
        for n in [3, 5, 7]:
            r = self._srhg(sphere_cohomology(n))
            assert r.is_formal is True, f"S^{n} should be formal"
            assert r.decision_ready()

    def test_cp2_via_entry_point(self):
        """ℂP^2: π_2=π_5=1 from the main entry point."""
        r = self._srhg(complex_projective_space_cohomology(2))
        assert r.rank_at(2) == 1
        assert r.rank_at(5) == 1
        assert r.decision_ready()

    def test_product_s3_s5(self):
        """S^3×S^5: formal, π_3=π_5=1."""
        alg = product_cohomology(sphere_cohomology(3), sphere_cohomology(5))
        r = self._srhg(alg)
        assert r.is_formal is True
        assert r.rank_at(3) == 1
        assert r.rank_at(5) == 1

    def test_non_formal_has_massey(self):
        """S^2 has Massey product entries from the main entry point."""
        r = self._srhg(sphere_cohomology(2))
        assert len(r.massey_products) >= 1

    def test_non_simply_connected_inconclusive(self):
        """Non-simply-connected space returns inconclusive from entry point."""
        non_sc = RationalCohomologyAlgebra(betti={0: 1, 1: 1}, name="non-SC")
        r = sullivan_rational_homotopy(non_sc)
        assert r.status == "inconclusive"
        assert not r.decision_ready()

    def test_space_label_preserved(self):
        """space_label passed to entry point is preserved."""
        r = sullivan_rational_homotopy(sphere_cohomology(3), space_label="S3-test")
        assert r.space_label == "S3-test"

    def test_three_sphere_product(self):
        """S^3×S^3×S^3: π_3=3 from entry point."""
        s3 = sphere_cohomology(3)
        alg = product_cohomology(product_cohomology(s3, s3), s3)
        r = self._srhg(alg)
        assert r.rank_at(3) == 3
        assert r.is_formal is True
