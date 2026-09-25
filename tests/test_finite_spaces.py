"""Finite spaces (Stong cores, weak points, McCord, order complexes) and strong collapses.

Overview:
    Each statement is checked against an independent computation of the same object:
    the homology of the face poset against pySurgery's SNF homology of the complex
    (torsion included), Stong-core contractibility of X(K) against the simplicial
    strong collapse of K (Barmak-Minian: K is strong collapsible iff X(K) is
    contractible), and the Python backend against the Julia one.
"""
import itertools

import numpy as np
import pytest

import exact_triangulations as T
from pysurgery.bridge.julia_bridge import julia_engine
from pysurgery.topology.finite_spaces import FiniteSpace, strong_collapse

BACKENDS = ["python"] + (["julia"] if julia_engine.available else [])


@pytest.mark.parametrize("backend", BACKENDS)
def test_stong_cores_decide_contractibility(backend):
    assert FiniteSpace.chain(5).is_contractible(backend=backend)
    assert not FiniteSpace.antichain(3).is_contractible(backend=backend)
    full = T.sc([(0, 1, 2)]).face_poset()
    core, retraction = full.stong_core(backend=backend)
    assert core.n == 1 and full.is_contractible(backend=backend)
    assert set(retraction.values()) == {0}
    circle = T.boundary_of_simplex(2).face_poset()          # 6 points, no beat point
    assert circle.stong_core(backend=backend)[0].n == 6
    assert not circle.is_contractible(backend=backend)


def test_the_finite_space_has_the_homology_of_the_complex():
    for K in (T.projective_plane(), T.torus(3, 3), T.klein_bottle(5, 5)):
        X = K.face_poset()
        want = K.homology(backend="python")
        assert X.homology(backend="python") == want              # weak points removed first
        assert X.homology(reduce_first=False, backend="python") == want
    assert FiniteSpace.antichain(3).betti_numbers(backend="python") == [3]


def test_order_complex_of_face_poset_is_the_barycentric_subdivision():
    K = T.sc([(0, 1, 2)])
    sd = K.face_poset().order_complex()
    # the barycentric subdivision of a triangle: 7 vertices, 12 edges, 6 triangles
    assert [sd.count_simplices(d) for d in range(3)] == [7, 12, 6]


def test_poset_validation_and_relations():
    with pytest.raises(ValueError):
        FiniteSpace.from_relations(2, [(0, 1), (1, 0)])       # a cycle: not a poset
    with pytest.raises(ValueError):
        FiniteSpace.from_downsets([{0, 1}, {0, 1}])           # not antisymmetric (not T0)
    X = FiniteSpace.from_relations(4, [(0, 1), (1, 2), (0, 3)])
    assert X.leq(0, 2) and not X.leq(3, 2)
    assert sorted(X.covers()) == [(0, 1), (0, 3), (1, 2)]


def test_quotient_takes_the_t0_reflection():
    X = FiniteSpace.chain(3)                                  # 0 < 1 < 2
    Q, mapping = X.quotient([[0, 2], [1]])                    # identifies ends: 0~2 > 1 > 0
    assert Q.n == 1 and mapping == [0, 0, 0]
    Q2, m2 = X.quotient([[0], [1, 2]])
    assert Q2.n == 2 and m2[1] == m2[2] != m2[0] and Q2.leq(m2[0], m2[1])


def test_mccord_certificate_is_one_sided():
    point = FiniteSpace.antichain(1)
    chain = FiniteSpace.chain(4)
    assert chain.mccord_certificate(point, [0] * 4).is_weak_equivalence
    circle = T.boundary_of_simplex(2).face_poset()
    cert = circle.mccord_certificate(point, [0] * circle.n)
    assert not cert.is_weak_equivalence and cert.details[0] == (6, 6)
    with pytest.raises(ValueError):                            # not order-preserving
        chain.mccord_certificate(FiniteSpace.chain(2), [1, 0, 0, 0])


@pytest.mark.parametrize("backend", BACKENDS)
def test_strong_collapse_of_cones_spheres_and_disks(backend):
    cone = T.sc([tuple(s) + (9,) for s in T.torus(3, 3).n_simplices(2)])
    res = strong_collapse(cone, backend=backend)
    assert res.is_strong_collapsible and res.n_core_vertices == 1
    assert cone.is_strong_collapsible(backend=backend)
    octa = T.octahedron()[0]
    res = strong_collapse(octa, backend=backend)
    assert not res.is_strong_collapsible and res.n_core_vertices == 6   # already minimal
    assert T.disk(6).is_strong_collapsible(backend=backend)
    assert not T.sc([(0,), (1,)]).is_strong_collapsible(backend=backend)


def _random_complexes(n=24, seed=0):
    rng = np.random.default_rng(seed)
    tris = list(itertools.combinations(range(7), 3))
    for _ in range(n):
        k = int(rng.integers(2, 9))
        pick = rng.choice(len(tris), size=k, replace=False)
        yield T.sc([tris[i] for i in pick])


def test_strong_collapsible_iff_face_poset_contractible():
    """Barmak-Minian: two independent algorithms, one on the complex (dominated
    vertices), one on its face poset (beat points), must agree -- on random complexes
    where both outcomes occur."""
    outcomes = set()
    for K in _random_complexes():
        a = K.is_strong_collapsible(backend="python")
        b = K.face_poset().is_contractible(backend="python")
        assert a == b
        outcomes.add(a)
    assert outcomes == {True, False}


@pytest.mark.skipif(not julia_engine.available, reason="Julia backend unavailable")
def test_python_and_julia_cores_are_identical():
    for K in list(_random_complexes(12, seed=3)) + [T.torus(3, 3), T.disk(7)]:
        a = strong_collapse(K, backend="python")
        b = strong_collapse(K, backend="julia")
        assert a.core_simplices == b.core_simplices and a.removed == b.removed
        X = K.face_poset()
        ca, ra = X.stong_core(backend="python")
        cb, rb = X.stong_core(backend="julia")
        assert ca.labels == cb.labels and ra == rb
