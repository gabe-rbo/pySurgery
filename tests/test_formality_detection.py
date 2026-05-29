"""Tests for formality detection and Massey product extraction.

Overview:
    Verifies ``is_formal_space`` and ``extract_massey_products`` in
    ``pysurgery.homotopy.rational_homotopy``.  All result contracts carry
    ``exact=True`` as required by Phase 2 Proposal 8 Milestone 3.

Key Concepts:
    - d = 0 formality: minimal model (ΛV, d) with d(g) = 0 for every g.
    - Massey products: monomials in d(g) encode k-fold Massey product data.
    - Contract: FormalityResult.exact = True, MasseyProductsResult.exact = True.

References:
    Félix, Y., Halperin, S., & Thomas, J.-C. (2001).
        Rational Homotopy Theory. Springer GTM 205. §12, §15.
    Sullivan, D. (1977). Infinitesimal computations in topology.
        Publ. Math. IHES 47, 269–331.
    Deligne, P., Griffiths, P., Morgan, J., & Sullivan, D. (1975).
        Real homotopy theory of Kähler manifolds. Invent. Math. 29, 245–274.
"""
from __future__ import annotations

from fractions import Fraction

import pytest

from pysurgery.homotopy.rational_homotopy import (
    DGAElement,
    RationalCohomologyAlgebra,
    RationalDGA,
    RationalMinimalModelResult,
    complex_projective_space_cohomology,
    extract_massey_products,
    is_formal_space,
    product_cohomology,
    sphere_cohomology,
    sullivan_minimal_model,
    FORMALITY_THEOREM_TAG,
    MASSEY_THEOREM_TAG,
)
from pysurgery.core.foundations import CONTRACT_VERSION


# ── Helpers ───────────────────────────────────────────────────────────────────


def _model(algebra: RationalCohomologyAlgebra, max_degree: int = 10) -> RationalMinimalModelResult:
    r = sullivan_minimal_model(algebra, max_degree=max_degree)
    assert r.exact is True
    assert r.minimal_model is not None, "sullivan_minimal_model must populate minimal_model"
    return r


# ── Contract field validation ─────────────────────────────────────────────────


class TestContractFields:
    """All result objects carry the required signed-contract fields."""

    def test_formality_result_exact_true(self):
        r = _model(sphere_cohomology(3))
        fr = is_formal_space(r)
        assert fr.exact is True

    def test_formality_result_theorem_tag(self):
        r = _model(sphere_cohomology(3))
        fr = is_formal_space(r)
        assert fr.theorem_tag == FORMALITY_THEOREM_TAG

    def test_formality_result_contract_version(self):
        r = _model(sphere_cohomology(3))
        fr = is_formal_space(r)
        assert fr.contract_version == CONTRACT_VERSION

    def test_formality_result_decision_ready_formal(self):
        r = _model(sphere_cohomology(3))
        fr = is_formal_space(r)
        assert fr.decision_ready()

    def test_formality_result_decision_ready_nonformal(self):
        r = _model(sphere_cohomology(2))
        fr = is_formal_space(r)
        # Non-formal spaces still have a successful, exact result.
        assert fr.exact is True
        assert fr.status == "success"
        assert fr.decision_ready()

    def test_massey_result_exact_true(self):
        r = _model(sphere_cohomology(2))
        mp = extract_massey_products(r)
        assert mp.exact is True

    def test_massey_result_theorem_tag(self):
        r = _model(sphere_cohomology(2))
        mp = extract_massey_products(r)
        assert mp.theorem_tag == MASSEY_THEOREM_TAG

    def test_massey_result_contract_version(self):
        r = _model(sphere_cohomology(2))
        mp = extract_massey_products(r)
        assert mp.contract_version == CONTRACT_VERSION

    def test_massey_result_decision_ready(self):
        r = _model(sphere_cohomology(2))
        mp = extract_massey_products(r)
        assert mp.decision_ready()


# ── Formal spaces ─────────────────────────────────────────────────────────────


class TestFormalSpaces:
    """Odd spheres and their products have d = 0 in the minimal model."""

    def test_s3_is_formal(self):
        r = _model(sphere_cohomology(3))
        fr = is_formal_space(r)
        assert fr.is_formal is True
        assert fr.non_formal_generators == []

    def test_s5_is_formal(self):
        r = _model(sphere_cohomology(5))
        fr = is_formal_space(r)
        assert fr.is_formal is True

    def test_s7_is_formal(self):
        r = _model(sphere_cohomology(7))
        fr = is_formal_space(r)
        assert fr.is_formal is True

    def test_s9_is_formal(self):
        r = _model(sphere_cohomology(9))
        fr = is_formal_space(r)
        assert fr.is_formal is True

    def test_s3_cross_s5_is_formal(self):
        alg = product_cohomology(sphere_cohomology(3), sphere_cohomology(5))
        r = _model(alg)
        fr = is_formal_space(r)
        assert fr.is_formal is True

    def test_s3_cross_s3_cross_s7_is_formal(self):
        s3 = sphere_cohomology(3)
        alg = product_cohomology(product_cohomology(s3, s3), sphere_cohomology(7))
        r = _model(alg)
        fr = is_formal_space(r)
        assert fr.is_formal is True

    def test_formal_space_has_no_massey_products(self):
        """Formal spaces (d=0) produce zero Massey product entries."""
        for n in [3, 5, 7]:
            r = _model(sphere_cohomology(n))
            mp = extract_massey_products(r)
            assert mp.entries == [], f"S^{n} should have no Massey products"
            assert mp.non_formal_count == 0

    def test_direct_dga_formal(self):
        """is_formal_space accepts a bare RationalDGA."""
        dga = RationalDGA()
        dga.add_generator(degree=3, name="a")
        dga.add_generator(degree=5, name="b")
        fr = is_formal_space(dga)
        assert fr.is_formal is True
        assert fr.exact is True


# ── Non-formal spaces ─────────────────────────────────────────────────────────


class TestNonFormalSpaces:
    """Even spheres and ℂP^n have d ≠ 0 in the minimal model."""

    def test_s2_is_not_formal(self):
        r = _model(sphere_cohomology(2))
        fr = is_formal_space(r)
        assert fr.is_formal is False
        assert len(fr.non_formal_generators) >= 1

    def test_s4_is_not_formal(self):
        r = _model(sphere_cohomology(4, max_degree=10))
        fr = is_formal_space(r)
        assert fr.is_formal is False

    def test_s6_is_not_formal(self):
        r = _model(sphere_cohomology(6, max_degree=14), max_degree=14)
        fr = is_formal_space(r)
        assert fr.is_formal is False

    def test_cp2_is_not_formal(self):
        r = _model(complex_projective_space_cohomology(2))
        fr = is_formal_space(r)
        assert fr.is_formal is False

    def test_cp3_is_not_formal(self):
        r = _model(complex_projective_space_cohomology(3))
        fr = is_formal_space(r)
        assert fr.is_formal is False

    def test_nonformal_dga_direct(self):
        """is_formal_space on a DGA with d ≠ 0."""
        dga = RationalDGA()
        x = dga.add_generator(degree=2, name="x")
        y = dga.add_generator(degree=3, name="y")
        dga._diff[y.gid] = DGAElement(dga, {((x.gid, 2),): Fraction(1)})
        fr = is_formal_space(dga)
        assert fr.is_formal is False
        assert "y" in fr.non_formal_generators

    def test_nonformal_generators_listed(self):
        """Non-formal generators are listed by name."""
        r = _model(sphere_cohomology(2))
        fr = is_formal_space(r)
        # The killing generator has d≠0; at least one name should appear.
        assert len(fr.non_formal_generators) > 0
        for name in fr.non_formal_generators:
            assert isinstance(name, str) and len(name) > 0


# ── Massey product accuracy ───────────────────────────────────────────────────


class TestMasseyProductAccuracy:
    """Massey products extracted from the minimal model match literature values.

    S^2 minimal model: Λ(x₂, y₃), d(y₃) = x₂²
      → binary Massey product (x₂, x₂), order 2, representative degree 3.

    ℂP^2 minimal model: Λ(x₂, y₅), d(y₅) = x₂³
      → triple Massey product (x₂, x₂, x₂), order 3, representative degree 5.

    ℂP^3 minimal model: Λ(x₂, y₇), d(y₇) = x₂⁴
      → 4-fold Massey product (x₂, x₂, x₂, x₂), order 4, representative degree 7.
    """

    def test_s2_binary_massey_product(self):
        r = _model(sphere_cohomology(2))
        mp = extract_massey_products(r)
        assert len(mp.entries) >= 1
        # The entry from d(y₃) = x₂² should have order 2.
        orders = [e.order for e in mp.entries]
        assert 2 in orders

    def test_s2_massey_product_representative_degree(self):
        """d(y₃) = x₂²: representative y₃ has degree 3."""
        r = _model(sphere_cohomology(2))
        mp = extract_massey_products(r)
        deg3_entries = [e for e in mp.entries if e.representative_degree == 3]
        assert len(deg3_entries) >= 1

    def test_cp2_triple_massey_product(self):
        """d(y₅) = x₂³ in ℂP^2 minimal model: order-3 Massey product."""
        r = _model(complex_projective_space_cohomology(2))
        mp = extract_massey_products(r)
        triple_entries = [e for e in mp.entries if e.order == 3]
        assert len(triple_entries) >= 1
        # Representative degree should be 5 (the generator that kills x₂³).
        rep_degrees = {e.representative_degree for e in triple_entries}
        assert 5 in rep_degrees

    def test_cp3_fourfold_massey_product(self):
        """d(y₇) = x₂⁴ in ℂP^3 minimal model: order-4 Massey product."""
        r = _model(complex_projective_space_cohomology(3))
        mp = extract_massey_products(r)
        fourfold_entries = [e for e in mp.entries if e.order == 4]
        assert len(fourfold_entries) >= 1
        rep_degrees = {e.representative_degree for e in fourfold_entries}
        assert 7 in rep_degrees

    def test_cp4_fivefold_massey_product(self):
        """d(y₉) = x₂⁵ in ℂP^4 minimal model: order-5 Massey product."""
        r = _model(complex_projective_space_cohomology(4))
        mp = extract_massey_products(r)
        fivefold_entries = [e for e in mp.entries if e.order == 5]
        assert len(fivefold_entries) >= 1

    def test_s4_binary_massey_product(self):
        """d(y₇) = x₄² in S^4 minimal model: order-2 Massey product."""
        r = _model(sphere_cohomology(4, max_degree=10))
        mp = extract_massey_products(r)
        binary_entries = [e for e in mp.entries if e.order == 2]
        assert len(binary_entries) >= 1

    def test_massey_product_input_classes_are_strings(self):
        """All input_classes are tuples of non-empty strings."""
        r = _model(sphere_cohomology(2))
        mp = extract_massey_products(r)
        for e in mp.entries:
            assert isinstance(e.input_classes, tuple)
            assert all(isinstance(s, str) and len(s) > 0 for s in e.input_classes)

    def test_massey_product_coefficients_are_strings(self):
        """Coefficients are stored as strings (exact ℚ, no floats)."""
        r = _model(complex_projective_space_cohomology(2))
        mp = extract_massey_products(r)
        for e in mp.entries:
            assert isinstance(e.coefficient, str)
            # Must be parseable as a Fraction (exact rational).
            Fraction(e.coefficient)

    def test_massey_product_order_equals_input_classes_length(self):
        """entry.order == len(entry.input_classes) for every entry."""
        for alg in [
            sphere_cohomology(2),
            complex_projective_space_cohomology(2),
            complex_projective_space_cohomology(3),
        ]:
            r = _model(alg)
            mp = extract_massey_products(r)
            for e in mp.entries:
                assert e.order == len(e.input_classes), (
                    f"order={e.order} but len(input_classes)={len(e.input_classes)}"
                )

    def test_as_dict_groups_by_input_classes(self):
        """MasseyProductsResult.as_dict() groups entries by input_classes."""
        r = _model(complex_projective_space_cohomology(2))
        mp = extract_massey_products(r)
        d = mp.as_dict()
        assert isinstance(d, dict)
        for key, entries in d.items():
            assert isinstance(key, tuple)
            for e in entries:
                assert e.input_classes == key

    def test_binary_massey_in_synthetic_dga(self):
        """Synthetic DGA: d(c) = a∧b → binary Massey product (a, b)."""
        dga = RationalDGA()
        a = dga.add_generator(degree=2, name="a")
        b = dga.add_generator(degree=2, name="b")
        c = dga.add_generator(degree=3, name="c")
        # d(c) = a∧b (product of two distinct degree-2 generators)
        dga._diff[c.gid] = DGAElement(
            dga, {((a.gid, 1), (b.gid, 1)): Fraction(1)}
        )
        mp = extract_massey_products(dga)
        assert len(mp.entries) == 1
        e = mp.entries[0]
        assert e.order == 2
        assert set(e.input_classes) == {"a", "b"}
        assert e.representative_name == "c"
        assert e.representative_degree == 3

    def test_triple_massey_in_synthetic_dga(self):
        """Synthetic DGA: d(w) = a∧b∧c → triple Massey product (a, b, c)."""
        dga = RationalDGA()
        a = dga.add_generator(degree=2, name="a")
        b = dga.add_generator(degree=2, name="b")
        c = dga.add_generator(degree=3, name="c")
        w = dga.add_generator(degree=6, name="w")
        # d(w) = a∧b∧c: degree 2+2+3=7 = 6+1 ✓
        dga._diff[w.gid] = DGAElement(
            dga, {((a.gid, 1), (b.gid, 1), (c.gid, 1)): Fraction(1)}
        )
        mp = extract_massey_products(dga)
        assert mp.non_formal_count == 1
        e = mp.entries[0]
        assert e.order == 3
        assert set(e.input_classes) == {"a", "b", "c"}
        assert e.representative_name == "w"
        assert e.representative_degree == 6

    def test_formal_space_zero_massey_products(self):
        """Odd sphere: no entries, non_formal_count = 0."""
        r = _model(sphere_cohomology(3))
        mp = extract_massey_products(r)
        assert mp.entries == []
        assert mp.non_formal_count == 0
        assert mp.exact is True

    def test_product_of_odd_spheres_no_massey(self):
        """S^3 × S^5: formal, no Massey products."""
        alg = product_cohomology(sphere_cohomology(3), sphere_cohomology(5))
        r = _model(alg)
        mp = extract_massey_products(r)
        assert mp.entries == []
        assert mp.non_formal_count == 0


# ── Consistency between is_formal_space and extract_massey_products ──────────


class TestConsistency:
    """is_formal_space and extract_massey_products agree with each other."""

    @pytest.mark.parametrize("n", [3, 5, 7])
    def test_formal_implies_no_massey(self, n):
        r = _model(sphere_cohomology(n))
        fr = is_formal_space(r)
        mp = extract_massey_products(r)
        assert fr.is_formal is True
        assert mp.non_formal_count == 0
        assert mp.entries == []

    @pytest.mark.parametrize("n", [2, 4])
    def test_nonformal_implies_massey(self, n):
        max_d = 10 if n <= 4 else 14
        r = _model(sphere_cohomology(n, max_degree=max_d), max_degree=max_d)
        fr = is_formal_space(r)
        mp = extract_massey_products(r)
        assert fr.is_formal is False
        assert mp.non_formal_count > 0
        assert len(mp.entries) > 0

    def test_non_formal_count_matches_non_formal_generators(self):
        """non_formal_count == len(non_formal_generators) for S^2."""
        r = _model(sphere_cohomology(2))
        fr = is_formal_space(r)
        mp = extract_massey_products(r)
        # Each non-formal generator contributes at least one Massey entry.
        assert mp.non_formal_count == len(fr.non_formal_generators)

    def test_direct_dga_both_functions(self):
        """Both functions accept a bare RationalDGA."""
        dga = RationalDGA()
        x = dga.add_generator(degree=2, name="x")
        y = dga.add_generator(degree=3, name="y")
        dga._diff[y.gid] = DGAElement(dga, {((x.gid, 2),): Fraction(1)})

        fr = is_formal_space(dga)
        mp = extract_massey_products(dga)

        assert fr.exact is True
        assert mp.exact is True
        assert fr.is_formal is False
        assert mp.non_formal_count == 1
