"""Tests for SimplicialComplex.certify_pl_manifold.

Overview:
    Verifies that ``certify_pl_manifold`` exactly matches ``is_homology_manifold`` at
    dimension <= 2 (where the plan's design review established the two checks coincide --
    see ``is_homology_manifold``'s docstring), is exact at dimension 3 on a genuine closed
    3-manifold (the boundary of a 4-simplex, S^3), and -- the proof this recursion adds real
    value -- catches the hand-built 8-vertex/7-tetrahedron counterexample from that same
    design review: a complex where ``is_homology_manifold`` alone reports a false
    ``(True, 3, {})`` because a vertex's link has the right *aggregate* homology (genuine
    S^2 homology, from being two S^2's wedged at a single point, which is homotopy- but not
    homeomorphically- S^2) without actually being a topological sphere. Also checks that
    dimension >= 4 returns ``exact=False`` with a warning, since ruling out exotic homology
    spheres among vertex links is undecidable in general at that point.
"""
import itertools

import pytest

from pysurgery.topology.complexes import SimplicialComplex


def test_certify_pl_manifold_matches_is_homology_manifold_at_d_le_2():
    """At dimension <= 2, certify_pl_manifold must be identical to is_homology_manifold,
    with exact=True -- both on a genuine closed surface (the boundary of a tetrahedron, S^2)
    and on a known-defective complex (a pinch point: two triangles sharing only a vertex)."""
    # Genuine S^2: boundary of a tetrahedron.
    sphere = SimplicialComplex.from_simplices(
        list(itertools.combinations(range(4), 3)), close_under_faces=True
    )
    is_mani, dim, diag = sphere.is_homology_manifold(backend="julia")
    cert = sphere.certify_pl_manifold(backend="julia")
    assert cert.is_pl_manifold == is_mani == True
    assert cert.dimension == dim == 2
    assert cert.diagnostics == diag == {}
    assert cert.exact is True

    # A genuine pinch point: two triangles sharing only vertex 0 (a known defect). Vertex
    # 0's own link (two disjoint edges) happens to look like a valid *1-manifold* link in
    # isolation (S^0, two points), so the failure this fixture actually trips is the
    # separate "detected link dimension doesn't match the complex's own top dimension"
    # check, not a direct per-vertex diagnostic -- either way, both methods must agree it
    # is not a manifold.
    pinched = SimplicialComplex.from_simplices(
        [(0, 1, 2), (0, 3, 4)], close_under_faces=True
    )
    is_mani2, dim2, diag2 = pinched.is_homology_manifold(backend="julia")
    cert2 = pinched.certify_pl_manifold(backend="julia")
    assert cert2.is_pl_manifold == is_mani2 == False
    assert cert2.dimension == dim2
    assert cert2.diagnostics == diag2
    assert cert2.diagnostics  # non-empty
    assert cert2.exact is True


def test_certify_pl_manifold_boundary_of_4_simplex_is_exact_s3():
    """The boundary of a 4-simplex (5 vertices, all 4-element subsets) is a genuine closed
    3-manifold (S^3); certify_pl_manifold must certify it exactly True."""
    tetrahedra = list(itertools.combinations(range(5), 4))
    sc = SimplicialComplex.from_simplices(tetrahedra, close_under_faces=True)
    assert sc.dimension == 3

    cert = sc.certify_pl_manifold(backend="julia")
    assert cert.is_pl_manifold is True
    assert cert.dimension == 3
    assert cert.diagnostics == {}
    assert cert.exact is True


def _wedge_of_two_spheres_counterexample():
    """The 8-vertex, 7-tetrahedron counterexample from the design review: cone a new vertex
    v=0 over L, where L (on vertices 1-7) is a genuine S^2 (the boundary of the tetrahedron
    on {1,2,3,4}) wedged, at vertex 1 only (no shared edge), with a closed 3-triangle fan on
    new vertices {5,6,7}. L is homotopy-equivalent to S^2 (trivial reduced homology
    otherwise, rank-1 H_2) since wedging with a contractible piece (any cone, including this
    closed fan, is contractible) does not change homotopy type -- so is_homology_manifold's
    aggregate check on Lk(0)=L sees nothing wrong. But L is not actually a topological S^2:
    within L, vertex 1's own link is two disjoint triangles (the S^2 piece's link
    contribution {2,3},{2,4},{3,4} and the fan's {5,6},{6,7},{7,5}), a disconnected 1-complex
    -- a genuine non-manifold pinch is_homology_manifold's aggregate-homology check at the
    top level cannot see, but certify_pl_manifold's recursive per-vertex-link check does.
    """
    tetrahedra = [
        (0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 3, 4), (0, 2, 3, 4),  # v * (S^2 piece)
        (0, 1, 5, 6), (0, 1, 6, 7), (0, 1, 7, 5),                # v * (fan piece)
    ]
    return SimplicialComplex.from_simplices(tetrahedra, close_under_faces=True)


def test_certify_pl_manifold_catches_wedge_of_spheres_counterexample():
    """The 8-vertex/7-tetrahedron counterexample: is_homology_manifold alone reports a false
    (True, 3, {}), but certify_pl_manifold correctly reports exact=True, is_pl_manifold=False."""
    sc = _wedge_of_two_spheres_counterexample()
    assert len(sc.n_simplices(0)) == 8
    assert len(sc.n_simplices(3)) == 7
    assert sc.dimension == 3

    is_mani, dim, diag = sc.is_homology_manifold(backend="julia")
    assert is_mani is True  # the false positive is_homology_manifold's docstring documents
    assert dim == 3
    assert diag == {}

    cert = sc.certify_pl_manifold(backend="julia")
    assert cert.exact is True
    assert cert.is_pl_manifold is False
    assert cert.dimension == 3
    assert cert.diagnostics  # non-empty: the recursion found the defect


def test_certify_pl_manifold_d_ge_4_returns_inexact_with_warning():
    """The boundary of a 5-simplex (6 vertices, all 5-element subsets) is a genuine closed
    4-manifold (S^4); certify_pl_manifold cannot exactly certify at this dimension (ruling
    out exotic homology spheres among vertex links is undecidable in general), so it must
    return exact=False accompanied by a warning, even though the underlying
    is_homology_manifold verdict (which it defers to) is True here."""
    four_simplices = list(itertools.combinations(range(6), 5))
    sc = SimplicialComplex.from_simplices(four_simplices, close_under_faces=True)
    assert sc.dimension == 4

    with pytest.warns(UserWarning, match="cannot exactly certify"):
        cert = sc.certify_pl_manifold(backend="julia")
    assert cert.exact is False
    assert cert.dimension == 4
    assert cert.is_pl_manifold is True  # deferred from is_homology_manifold, not itself proven
