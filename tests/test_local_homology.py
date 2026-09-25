"""Local homology and the homology-manifold certificate, on exact triangulations.

Overview:
    The certificate is a statement about |K| checked at EVERY simplex, so the tests are
    about complexes whose local structure is known in advance: closed manifolds
    (orientable or not), manifolds with boundary, and the singular examples a
    vertex-only check gets wrong -- a pinch point, a branching edge, a dangling edge, a
    cone on a torus, and the wedge-of-spheres vertex link that fools
    ``is_homology_manifold``. Every test runs on the Python and the Julia backend,
    and the two must agree exactly.
"""
import itertools

import pytest

import exact_triangulations as T
from pysurgery.bridge.julia_bridge import julia_engine
from pysurgery.core.exceptions import NotAManifoldError
from pysurgery.topology import local_homology as LH

BACKENDS = ["python"] + (["julia"] if julia_engine.available else [])


@pytest.mark.parametrize("backend", BACKENDS)
def test_closed_manifolds_of_dimension_one_to_three_certify(backend):
    cases = [(T.boundary_of_simplex(2), 1), (T.octahedron()[0], 2), (T.torus(4, 4), 2),
             (T.klein_bottle(5, 5), 2), (T.projective_plane(), 2),
             (T.subdivided_sphere(2)[0], 2), (T.boundary_of_simplex(4), 3)]
    for K, n in cases:
        c = LH.certify_homology_manifold(K, n, backend=backend)
        assert c.is_closed_homology_manifold and c.is_homology_manifold_with_boundary, (n, str(c))
        assert c.implies_pl_manifold and c.n_other == 0 and c.n_acyclic == 0
        assert c.n_simplices == sum(K.count_simplices(d) for d in K.dimensions)
        assert c.pseudomanifold.is_closed_pseudomanifold


@pytest.mark.parametrize("backend", BACKENDS)
def test_manifolds_with_boundary_and_their_boundaries(backend):
    disk = LH.certify_homology_manifold(T.disk(6), 2, backend=backend)
    assert not disk.is_closed_homology_manifold and disk.is_homology_manifold_with_boundary
    B = LH.homology_manifold_boundary(T.disk(6), 2, backend=backend)
    assert (B.count_simplices(0), B.count_simplices(1)) == (6, 6)
    assert T.betti(B) == [1, 1]
    assert (0,) not in B.n_simplices(0)                      # the centre is interior

    mob = LH.homology_manifold_boundary(T.mobius_band(6), 2, backend=backend)
    assert T.betti(mob) == [1, 1]                            # ONE boundary circle

    ball = T.sc([(0, 1, 2, 3)])                              # the solid tetrahedron
    c = LH.certify_homology_manifold(ball, 3, backend=backend)
    assert c.is_homology_manifold_with_boundary and c.implies_pl_manifold
    assert T.betti(LH.homology_manifold_boundary(ball, 3, backend=backend)) == [1, 0, 1]

    path = T.sc([(0, 1), (1, 2), (2, 3)])
    assert sorted(LH.homology_manifold_boundary(path, 1, backend=backend).n_simplices(0)) == [(0,), (3,)]


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_pinch_point_is_found_and_nothing_else_is(backend):
    c = LH.certify_homology_manifold(T.pinched_spheres(), 2, backend=backend)
    assert not c.is_closed_homology_manifold and not c.is_homology_manifold_with_boundary
    assert [t.simplex for t in c.singular] == [(0,)] and c.singular_vertices == [0]
    # the link at the pinch is two disjoint circles: H~_0 = Z, H~_1 = Z^2
    assert c.singular[0].link_reduced_homology == {0: (1, []), 1: (2, [])}


@pytest.mark.parametrize("backend", BACKENDS)
def test_two_planes_through_a_point(backend):
    c = LH.certify_homology_manifold(T.wedge_of_disks(6), 2, backend=backend)
    assert c.singular_vertices == [0] and not c.is_homology_manifold_with_boundary


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_branching_edge_is_singular(backend):
    K = T.sc([(0, 1, 2), (0, 1, 3), (0, 1, 4)])              # three sheets on one edge
    c = LH.certify_homology_manifold(K, 2, backend=backend)
    assert (0, 1) in [t.simplex for t in c.singular]
    assert not c.is_homology_manifold_with_boundary
    assert not LH.pseudomanifold_report(K, 2).non_branching


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_dangling_edge_is_not_mistaken_for_boundary(backend):
    """The free end of a hair has zero local homology, exactly like a boundary point.
    Only the link's dimension and the boundary's consistency tell them apart."""
    K = T.sc([(0, 1, 2), (2, 3)])
    c = LH.certify_homology_manifold(K, 2, backend=backend)
    assert not c.is_homology_manifold_with_boundary
    bad = {t.simplex for t in c.singular}
    assert {(3,), (2, 3)} <= bad, bad
    with pytest.raises(NotAManifoldError):
        LH.homology_manifold_boundary(K, 2, backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_cone_on_a_torus_is_singular_only_at_its_apex(backend):
    base = T.torus(3, 3)
    apex = 9
    cone = T.sc([tuple(s) + (apex,) for s in base.n_simplices(2)])
    c = LH.certify_homology_manifold(cone, 3, backend=backend)
    assert c.singular_vertices == [apex] and len(c.singular) == 1
    assert c.n_acyclic == sum(base.count_simplices(d) for d in base.dimensions)


@pytest.mark.parametrize("backend", BACKENDS)
def test_wedge_of_spheres_vertex_link_is_caught(backend):
    """The 8-vertex counterexample of test_pl_manifold_certificate: the cone on
    (S^2 wedge a closed fan) has a vertex link with the homology of S^2, so the
    vertex-link check passes, but the link is not a homology 2-manifold. Checking every
    simplex finds the edge whose link is two circles."""
    s2 = list(itertools.combinations([1, 2, 3, 4], 3))
    fan = [(1, 5, 6), (1, 6, 7), (1, 5, 7)]                  # a disk, coned at vertex 1
    K = T.sc([(0,) + t for t in s2 + fan])
    c = LH.certify_homology_manifold(K, 3, backend=backend)
    assert not c.is_homology_manifold_with_boundary
    assert (0, 1) in [t.simplex for t in c.singular]


@pytest.mark.parametrize("backend", BACKENDS)
def test_local_homology_is_read_off_the_link(backend):
    D = T.disk(6)
    assert LH.local_homology(D, (0,), backend=backend) == {2: (1, [])}
    assert LH.local_homology(D, (1,), backend=backend) == {}
    # a maximal simplex has the empty link: H~_{-1} = Z, so local homology is Z in degree 2
    assert LH.link_reduced_homology(D, (0, 1, 2), backend=backend) == {-1: (1, [])}
    assert LH.local_homology(D, (0, 1, 2), backend=backend) == {2: (1, [])}
    # a torsion link: the cone on RP^2 has local homology Z/2 at the apex
    rp2 = T.projective_plane()
    cone = T.sc([tuple(s) + (6,) for s in rp2.n_simplices(2)])
    assert LH.local_homology(cone, (6,), backend=backend) == {2: (0, [2])}


@pytest.mark.parametrize("backend", BACKENDS)
def test_in_dimension_four_the_certificate_does_not_claim_a_manifold(backend):
    c = LH.certify_homology_manifold(T.boundary_of_simplex(5), 4, backend=backend)
    assert c.is_closed_homology_manifold and not c.implies_pl_manifold
    assert any(">= 4" in n for n in c.notes)


@pytest.mark.parametrize("backend", BACKENDS)
def test_wrong_dimension_is_other_everywhere(backend):
    c = LH.certify_homology_manifold(T.octahedron()[0], 1, backend=backend)
    assert c.n_other == c.n_simplices and not c.is_homology_manifold_with_boundary


@pytest.mark.skipif(not julia_engine.available, reason="Julia backend unavailable")
def test_python_and_julia_agree_simplex_by_simplex():
    for K in (T.klein_bottle(5, 5), T.pinched_spheres(), T.mobius_band(5),
              T.sc([(0, 1, 2), (2, 3), (0, 1, 4)])):
        a = LH.classify_simplices(K, 2, backend="python")
        b = LH.classify_simplices(K, 2, backend="julia")
        assert a == b


def test_simplicial_complex_methods_delegate():
    K = T.torus(4, 4)
    cert = K.certify_homology_manifold()
    assert cert.is_closed_homology_manifold and cert.n == 2
    assert K.local_homology((0,)) == {2: (1, [])}
    assert K.pseudomanifold_report().is_closed_pseudomanifold
    with pytest.raises(ValueError):
        LH.links_of(K, [(0, 99)])
