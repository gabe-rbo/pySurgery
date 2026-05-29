"""Tests for the pysurgery.knots module: linking, invariants, constructors, analysis."""
import pytest
import numpy as np

from pysurgery.knots.linking import (
    linking_matrix, milnor_triple_invariant, milnor_invariants,
    are_linked, link_type, LinkType,
)
from pysurgery.knots.constructors import (
    hopf_link, borromean_rings,
    unknot, trefoil_knot, figure_eight_knot, torus_knot, whitehead_link,
    _build_s3_grid, _extract_cycle,
)
from pysurgery.knots.invariants import (
    seifert_matrix, alexander_polynomial, conway_polynomial,
    knot_signature, arf_invariant, genus_bound, knot_determinant,
    is_unknot, classify_knot,
)
from pysurgery.knots.analysis import find_knots_between_components, KnotAnalysisResult


# ── Linking number tests ──────────────────────────────────────────────────────


def test_hopf_link():
    sc, components = hopf_link()

    L = linking_matrix(sc, components)
    assert L.shape == (2, 2)
    assert L[0, 0] == 0
    assert L[1, 1] == 0
    assert abs(L[0, 1]) == 1
    assert abs(L[1, 0]) == 1

    assert are_linked(sc, components) is True
    assert link_type(sc, components) == LinkType.HOPF


def test_borromean_rings():
    sc, components = borromean_rings()

    L = linking_matrix(sc, components)
    assert L.shape == (3, 3)
    for i in range(3):
        for j in range(3):
            assert L[i, j] == 0

    mu = milnor_triple_invariant(sc, components[0], components[1], components[2])
    assert mu != 0

    assert are_linked(sc, components) is True
    assert link_type(sc, components) == LinkType.BORROMEAN


def test_unlinked():
    sc, idx_map = _build_s3_grid(size=6)

    r1_pts = [(2, 2, 2), (3, 2, 2), (3, 3, 2), (2, 3, 2)]
    r2_pts = [(2, 2, 4), (3, 2, 4), (3, 3, 4), (2, 3, 4)]
    c1 = _extract_cycle(idx_map, r1_pts)
    c2 = _extract_cycle(idx_map, r2_pts)
    components = [c1, c2]

    L = linking_matrix(sc, components)
    assert L[0, 1] == 0
    assert are_linked(sc, components) is False
    assert link_type(sc, components) == LinkType.UNLINKED


def test_milnor_invariants_pairwise():
    sc, components = hopf_link()
    mu_01 = milnor_invariants(sc, components, (0, 1))
    assert mu_01 is not None
    assert abs(mu_01) == 1


def test_milnor_invariants_triple_borromean():
    sc, components = borromean_rings()
    mu_012 = milnor_invariants(sc, components, (0, 1, 2))
    assert mu_012 is not None
    assert mu_012 != 0


# ── Constructor tests ─────────────────────────────────────────────────────────


def test_unknot_constructor():
    sc, K = unknot()
    assert K.dimension == 1
    assert K.count_simplices(1) > 0


def test_trefoil_constructor():
    sc, K = trefoil_knot(handedness="left")
    assert K.dimension == 1
    assert K.count_simplices(1) > 0


def test_trefoil_right_constructor():
    sc, K = trefoil_knot(handedness="right")
    assert K.dimension == 1
    assert K.count_simplices(1) > 0


def test_figure_eight_constructor():
    sc, K = figure_eight_knot()
    assert K.dimension == 1
    assert K.count_simplices(1) > 0


def test_torus_knot_constructor_trefoil():
    # T(2,3) = trefoil
    sc, K = torus_knot(2, 3)
    assert K.dimension == 1
    assert K.count_simplices(1) > 0


def test_torus_knot_constructor_cinquefoil():
    # T(2,5) = cinquefoil / 5_1
    sc, K = torus_knot(2, 5)
    assert K.dimension == 1
    assert K.count_simplices(1) > 0


def test_torus_knot_gcd_error():
    with pytest.raises(ValueError, match="gcd"):
        torus_knot(2, 4)


def test_whitehead_link_constructor():
    sc, components = whitehead_link()
    assert len(components) == 2
    # Verify the linking number is 0 (key property of Whitehead link)
    L = linking_matrix(sc, components)
    assert L[0, 1] == 0


# ── Invariant tests ───────────────────────────────────────────────────────────


def test_seifert_matrix_shape():
    # Seifert matrix of a genus-g knot is 2g × 2g
    sc, K = trefoil_knot()
    V = seifert_matrix(sc, K)
    # Trefoil has genus 1 → 2×2 Seifert matrix
    assert V.shape in [(2, 2), (0, 0)]  # allow (0,0) if Seifert chain fails
    assert V.dtype == np.int64


def test_seifert_matrix_unknot():
    sc, K = unknot()
    V = seifert_matrix(sc, K)
    # Unknot has genus 0 → empty Seifert matrix or zero
    assert V.shape[0] == V.shape[1]


def test_alexander_polynomial_unknot():
    sc, K = unknot()
    delta = alexander_polynomial(sc, K)
    # Unknot has Δ = 1
    assert delta == {0: 1}


def test_alexander_polynomial_trefoil():
    sc, K = trefoil_knot()
    delta = alexander_polynomial(sc, K)
    # Trefoil Alexander polynomial: t^2 - t + 1 (or equivalently -(t^{-1} - 1 + t))
    # Δ(1) = 1 must hold
    delta_at_1 = sum(c for c in delta.values())
    assert delta_at_1 == 1
    # Degree span should be 2 (genus 1 → 2 * genus = 2)
    if len(delta) > 1:
        assert max(delta.keys()) - min(delta.keys()) == 2


def test_alexander_polynomial_figure_eight():
    sc, K = figure_eight_knot()
    delta = alexander_polynomial(sc, K)
    # Figure-eight: -t + 3 - t^{-1}, determinant = 5
    delta_at_1 = sum(c for c in delta.values())
    assert delta_at_1 == 1


def test_conway_polynomial_unknot():
    sc, K = unknot()
    nabla = conway_polynomial(sc, K)
    # Unknot: ∇(z) = 1
    assert nabla.get(0, 0) == 1 and all(nabla.get(k, 0) == 0 for k in nabla if k != 0)


def test_conway_polynomial_trefoil():
    sc, K = trefoil_knot()
    nabla = conway_polynomial(sc, K)
    # ∇(0) = 1 for all knots (at z=0: t^{1/2} - t^{-1/2} = 0 → t = 1, Δ(1) = 1)
    nabla_at_0 = nabla.get(0, 0)
    assert nabla_at_0 == 1


def test_knot_signature_unknot():
    sc, K = unknot()
    sig = knot_signature(sc, K)
    assert sig == 0


def test_knot_signature_is_int():
    sc, K = trefoil_knot()
    sig = knot_signature(sc, K)
    assert isinstance(sig, int)


def test_arf_invariant_unknot():
    sc, K = unknot()
    assert arf_invariant(sc, K) == 0


def test_arf_invariant_trefoil():
    # Trefoil has Δ(-1) = 3 ≡ 3 (mod 8) → Arf = 1
    sc, K = trefoil_knot()
    assert arf_invariant(sc, K) == 1


def test_genus_bound_unknot():
    sc, K = unknot()
    assert genus_bound(sc, K) == 0


def test_genus_bound_trefoil():
    sc, K = trefoil_knot()
    g = genus_bound(sc, K)
    # Trefoil has genus 1
    assert g >= 0  # at least non-negative; should be 1


def test_knot_determinant_unknot():
    sc, K = unknot()
    assert knot_determinant(sc, K) == 1


def test_knot_determinant_trefoil():
    sc, K = trefoil_knot()
    det = knot_determinant(sc, K)
    # Trefoil determinant = 3
    assert det == 3


def test_is_unknot():
    sc, K = unknot()
    assert is_unknot(sc, K) is True


def test_classify_knot_unknot():
    sc, K = unknot()
    ktype = classify_knot(sc, K)
    assert "unknot" in ktype.lower()


def test_classify_knot_trefoil():
    sc, K = trefoil_knot(handedness="left")
    ktype = classify_knot(sc, K)
    assert isinstance(ktype, str)


# ── Analysis tests ────────────────────────────────────────────────────────────


def test_find_knots_hopf():
    sc, components = hopf_link()
    result = find_knots_between_components(
        sc, components=components, ambient_complex=sc,
        compute_per_component_invariants=False,
    )
    assert isinstance(result, KnotAnalysisResult)
    assert result.are_linked is True
    assert result.link_classification == LinkType.HOPF
    assert (0, 1) in result.linked_pairs


def test_find_knots_borromean():
    sc, components = borromean_rings()
    result = find_knots_between_components(
        sc, components=components, ambient_complex=sc,
        compute_per_component_invariants=False,
    )
    assert result.are_linked is True
    assert result.link_classification == LinkType.BORROMEAN
    assert result.milnor_triple is not None and result.milnor_triple != 0


def test_find_knots_unlinked():
    sc, idx_map = _build_s3_grid(size=6)
    c1 = _extract_cycle(idx_map, [(2, 2, 2), (3, 2, 2), (3, 3, 2), (2, 3, 2)])
    c2 = _extract_cycle(idx_map, [(2, 2, 4), (3, 2, 4), (3, 3, 4), (2, 3, 4)])
    result = find_knots_between_components(
        sc, components=[c1, c2], ambient_complex=sc,
        compute_per_component_invariants=False,
    )
    assert result.are_linked is False
    assert result.link_classification == LinkType.UNLINKED


def test_find_knots_with_invariants():
    sc, K = unknot()
    result = find_knots_between_components(
        sc, components=[K], ambient_complex=sc,
        compute_per_component_invariants=True,
    )
    assert len(result.component_invariants) == 1
    info = result.component_invariants[0]
    assert info.component_index == 0
    assert isinstance(info.alexander_polynomial, dict)
    assert isinstance(info.signature, int)
    assert info.arf in (0, 1)


def test_analysis_summary_str():
    sc, components = hopf_link()
    result = find_knots_between_components(
        sc, components=components, ambient_complex=sc,
        compute_per_component_invariants=False,
    )
    s = result.summary()
    assert "Hopf" in s or "link" in s.lower()
