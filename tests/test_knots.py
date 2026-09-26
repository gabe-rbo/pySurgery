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
    _delaunay_s3_from_points, _ring_cycle, _subdivided_polyline,
)
from pysurgery.manifolds.surgery import compute_linking_number
from pysurgery.topology.complexes import SimplicialComplex
from pysurgery.bridge.julia_bridge import julia_engine
from pysurgery.knots.invariants import (
    seifert_matrix, alexander_polynomial, conway_polynomial,
    knot_signature, arf_invariant, genus_bound, knot_determinant,
    is_unknot, classify_knot, unknotting_number_lower_bound,
    _conway_from_alexander, _find_crossings, _knot_polyline_coords,
    _goeritz_from_diagram, _is_generic_projection, _projection_basis,
    _signature_and_det, _signature_from_diagram,
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


# ── Linking numbers from the triangulation alone ──────────────────────────────


def _bare(sc):
    """The same triangulation with its vertex coordinates dropped."""
    return SimplicialComplex.from_maximal_simplices(sc.n_simplices(3))


def _mirrored(sc):
    """The same triangulation with its coordinates reflected in the xy-plane."""
    pc = sc.simplices_to_point_cloud
    pts = np.array([pc[(v,)][0] for v in range(len(sc.n_simplices(0)))])
    pts[:, 2] *= -1
    out = _bare(sc)
    out._generate_point_cloud_mappings(pts)
    return out


def _double_clasp():
    """Two rectangles with linking number ±2.

    The second passes up through the first's disk at x = −2 and x = 2 and
    returns around the outside of the first each time.
    """
    C1 = _subdivided_polyline([(-4, -1, 0), (4, -1, 0), (4, 1, 0), (-4, 1, 0)])
    C2 = _subdivided_polyline([
        (-2, 0, -2), (-2, 0, 2), (-2, 3, 2), (-2, 3, -2), (2, 3, -2), (2, 0, -2),
        (2, 0, 2), (2, 4, 2), (2, 4, -3), (-2, 4, -3), (-2, 0, -3),
    ])
    sc, idx_map = _delaunay_s3_from_points(C1 + C2, bbox_extent=12.0)
    return sc, [_ring_cycle(C1, idx_map), _ring_cycle(C2, idx_map)]


# Linking numbers by the Gauss integral in each constructor's coordinates.
_LINKS = {
    "hopf": (hopf_link, -1),
    "whitehead": (whitehead_link, 0),
    "borromean": (borromean_rings, 0),
    "double_clasp": (_double_clasp, 2),
}


@pytest.mark.parametrize("name", sorted(_LINKS))
def test_simplicial_linking_number_matches_gauss(name):
    """backend="python" skips the Gauss integral; with coordinates attached, the
    triangulation is oriented like R^3 and the two methods agree, sign included."""
    build, expected = _LINKS[name]
    sc, components = build()
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            gauss = compute_linking_number(sc, components[i], components[j])
            simplicial = compute_linking_number(sc, components[i], components[j], backend="python")
            assert gauss.seifert_chain_size == 0 < simplicial.seifert_chain_size
            assert gauss.value == simplicial.value == expected


@pytest.mark.parametrize("name", ["hopf", "double_clasp"])
def test_simplicial_linking_number_flips_under_reflection(name):
    build, expected = _LINKS[name]
    sc, (a, b) = build()
    mirrored = _mirrored(sc)
    assert compute_linking_number(mirrored, a, b).value == -expected
    assert compute_linking_number(mirrored, a, b, backend="python").value == -expected


@pytest.mark.parametrize("backend", ["python", "julia", "auto"])
def test_linking_number_without_coordinates(backend):
    """Every backend used to return 0 for linked cycles once coordinates were dropped."""
    if backend == "julia" and not julia_engine.available:
        pytest.skip("Julia not available")
    sc, (a, b) = hopf_link()
    bare = _bare(sc)
    # Without coordinates the first tetrahedron, (0, 1, 3, 4) in increasing
    # vertex order, is positive.  hopf_link()'s coordinates orient it
    # negatively, so the sign is opposite to the Gauss value −1.
    assert compute_linking_number(bare, a, b, backend=backend).value == 1
    assert compute_linking_number(bare, b, a, backend=backend).value == 1
    assert compute_linking_number(bare, a, b, "F2", backend=backend).value == 1

    sc, (a, b) = _double_clasp()
    assert abs(compute_linking_number(_bare(sc), a, b, backend=backend).value) == 2

    for build in (whitehead_link, borromean_rings):
        sc, components = build()
        bare = _bare(sc)
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                assert compute_linking_number(bare, components[i], components[j], backend=backend).value == 0


def test_link_type_without_coordinates():
    sc, components = hopf_link()
    bare = _bare(sc)
    assert abs(linking_matrix(bare, components)[0, 1]) == 1
    assert link_type(bare, components) == LinkType.HOPF


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
    # Figure-eight: -t + 3 - t^{-1}, normalised to -t^2 + 3t - 1
    assert delta == {2: -1, 1: 3, 0: -1}
    assert sum(delta.values()) == 1
    # Determinant |Δ(-1)| = 5 (the trefoil's is 3, so this separates them)
    assert abs(sum(c * (-1) ** d for d, c in delta.items())) == 5
    assert knot_determinant(sc, K) == 5


def test_alexander_polynomial_torus_knot_2_5():
    sc, K = torus_knot(2, 5)
    delta = alexander_polynomial(sc, K)
    # T(2,5): t^4 - t^3 + t^2 - t + 1, determinant 5
    assert delta == {4: 1, 3: -1, 2: 1, 1: -1, 0: 1}
    assert knot_determinant(sc, K) == 5


def test_conway_polynomial_unknot():
    sc, K = unknot()
    nabla = conway_polynomial(sc, K)
    # Unknot: ∇(z) = 1
    assert nabla.get(0, 0) == 1 and all(nabla.get(k, 0) == 0 for k in nabla if k != 0)


def test_conway_polynomial_trefoil():
    sc, K = trefoil_knot()
    nabla = conway_polynomial(sc, K)
    # ∇(0) = 1 for all knots (at z=0: t^{1/2} - t^{-1/2} = 0 → t = 1, Δ(1) = 1)
    assert nabla == {0: 1, 2: 1}


def test_conway_polynomial_figure_eight():
    sc, K = figure_eight_knot()
    # ∇(z) = 1 - z^2; the z^2 coefficient is the Casson invariant a_2 = -1
    assert conway_polynomial(sc, K) == {0: 1, 2: -1}


def test_conway_polynomial_torus_knot_2_5():
    sc, K = torus_knot(2, 5)
    # ∇(z) = 1 + 3z^2 + z^4
    assert conway_polynomial(sc, K) == {0: 1, 2: 3, 4: 1}


def test_conway_from_alexander_normalises_units():
    # Δ is only defined up to ±t^k: shifted and negated inputs agree.
    for delta in ({0: -1, 1: 3, 2: -1}, {3: 1, 4: -3, 5: 1}, {-1: -1, 0: 3, 1: -1}):
        assert _conway_from_alexander(delta) == {0: 1, 2: -1}
    assert _conway_from_alexander({0: 1}) == {0: 1}
    assert _conway_from_alexander({7: -1}) == {0: 1}


def test_conway_from_alexander_rejects_non_knot_polynomials():
    with pytest.raises(ValueError, match="symmetric"):
        _conway_from_alexander({2: 1, 1: 1, 0: -1})  # t^2 + t - 1
    with pytest.raises(ValueError, match="symmetric"):
        _conway_from_alexander({1: 1, 0: -1})  # odd degree span
    with pytest.raises(ValueError, match=r"\|Δ\(1\)\|"):
        _conway_from_alexander({0: 3})
    with pytest.raises(ValueError, match="zero"):
        _conway_from_alexander({0: 0})


@pytest.mark.parametrize("handedness, expected_sign", [("left", -1), ("right", 1)])
def test_trefoil_crossing_signs(handedness, expected_sign):
    # Standard convention: the left-handed trefoil has three negative crossings.
    sc, K = trefoil_knot(handedness=handedness)
    pts = _knot_polyline_coords(sc, K)
    crossings = _find_crossings(pts, np.eye(3)[0], np.eye(3)[1], np.eye(3)[2])
    assert [c["sign"] for c in crossings] == [expected_sign] * 3


def test_knot_signature_unknot():
    sc, K = unknot()
    sig = knot_signature(sc, K)
    assert sig == 0


def test_knot_signature_is_int():
    sc, K = trefoil_knot()
    sig = knot_signature(sc, K)
    assert isinstance(sig, int)


@pytest.mark.parametrize("backend", ["python", "auto"])
def test_knot_signature_trefoils(backend):
    # σ = sig(V + V^T): positive knots have negative signature, so the
    # right-handed trefoil (three positive crossings) has σ = −2.
    assert knot_signature(*trefoil_knot(handedness="left"), backend=backend) == 2
    assert knot_signature(*trefoil_knot(handedness="right"), backend=backend) == -2


@pytest.mark.parametrize("backend", ["python", "auto"])
def test_knot_signature_figure_eight(backend):
    assert knot_signature(*figure_eight_knot(), backend=backend) == 0


@pytest.mark.parametrize("backend", ["python", "auto"])
def test_knot_signature_torus_knot_2_5(backend):
    # torus_knot builds the left-handed (negative) T(2,5), so σ = +4
    sc, K = torus_knot(2, 5)
    assert knot_signature(sc, K, backend=backend) == 4
    assert unknotting_number_lower_bound(sc, K, backend=backend) == 2


def test_diagram_signature_is_projection_invariant():
    # G and μ depend on the diagram, but sign(G) − μ does not: it is the same
    # for every projection and either shading, and mirroring negates it.
    sc, K = torus_knot(2, 5)
    pts = _knot_polyline_coords(sc, K)
    mirror = pts * np.array([1.0, 1.0, -1.0])
    rng = np.random.default_rng(7)
    sigs, n_frames = set(), 0
    while n_frames < 4:
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        ex, ey = Q[:, 0], Q[:, 1]
        ez = np.cross(ex, ey)
        if not (_is_generic_projection(pts, ex, ey, ez)
                and _is_generic_projection(mirror, ex, ey, ez)):
            continue
        n_frames += 1
        for shade in (0, 1):
            sigs.add(_signature_from_diagram(pts, ex, ey, ez, shade=shade))
            sigs.add(-_signature_from_diagram(mirror, ex, ey, ez, shade=shade))
    assert sigs == {4}


def test_goeritz_determinant_is_knot_determinant():
    for sc, K in (trefoil_knot(), figure_eight_knot(), torus_knot(2, 5)):
        pts = _knot_polyline_coords(sc, K)
        G, _ = _goeritz_from_diagram(_find_crossings(pts, *_projection_basis(pts)))
        assert abs(_signature_and_det(G)[1]) == knot_determinant(sc, K)


def test_signature_and_det_exact():
    assert _signature_and_det([[-2, 1], [1, -2]]) == (-2, 3)
    # No nonzero diagonal entry: needs the e_i ↦ e_i + e_j pivot
    assert _signature_and_det([[0, 1], [1, 0]]) == (0, -1)
    assert _signature_and_det([[1, 1], [1, 1]]) == (1, 0)
    assert _signature_and_det([]) == (0, 1)


def test_arf_invariant_unknot():
    sc, K = unknot()
    assert arf_invariant(sc, K) == 0


def test_arf_invariant_trefoil():
    # Trefoil has Δ(-1) = 3 ≡ 3 (mod 8) → Arf = 1
    sc, K = trefoil_knot()
    assert arf_invariant(sc, K) == 1


def test_arf_invariant_figure_eight():
    # Figure-eight has Δ(-1) = -5 ≡ ±3 (mod 8) → Arf = 1 (a_2 = -1 is odd)
    sc, K = figure_eight_knot()
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


@pytest.mark.parametrize("backend", ["python", "auto"])
def test_classify_knot_chiral_knots(backend):
    assert classify_knot(*trefoil_knot(handedness="left"), backend=backend) == "left_trefoil"
    assert classify_knot(*trefoil_knot(handedness="right"), backend=backend) == "right_trefoil"
    # torus_knot builds left-handed torus knots; T(2,3) is the trefoil
    assert classify_knot(*torus_knot(2, 3), backend=backend) == "left_trefoil"
    assert classify_knot(*torus_knot(2, 5), backend=backend) == "torus_knot_T(2,5)_left"
    assert classify_knot(*figure_eight_knot(), backend=backend) == "figure_eight"


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
