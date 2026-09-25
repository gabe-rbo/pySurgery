"""Lower-star discrete Morse theory and lower-star persistence.

Overview:
    Each statement is checked against an independent computation: the Morse complex
    against pySurgery's SNF homology of the same complex (over Z, torsion included),
    the critical cells against the weak Morse inequalities over F_2 and the Euler
    relation, the persistence pairing against the Z/2 Betti numbers and (when
    installed) against gudhi, and the Python backend against the Julia one.
"""
import numpy as np
import pytest

import exact_triangulations as T
from pysurgery.bridge.julia_bridge import julia_engine
from pysurgery.topology.lower_star import (
    PairClass,
    classify_critical_pairs,
    critical_pair_persistence,
    lower_star_gradient,
    lower_star_persistence,
)

BACKENDS = ["python"] + (["julia"] if julia_engine.available else [])


def _cases():
    rng = np.random.default_rng(0)
    for K in (T.torus(4, 4), T.klein_bottle(5, 5), T.projective_plane(),
              T.subdivided_sphere(1)[0], T.boundary_of_simplex(4)):
        n = max(s[0] for s in K.n_simplices(0)) + 1
        for _ in range(2):
            yield K, rng.uniform(size=n)


def _f2_betti(K):
    """dim H_p(K; F_2) = beta_p + t_p(2) + t_{p-1}(2) (universal coefficients)."""
    h = K.homology(backend="python")
    top = max(h)
    t2 = {p: sum(1 for t in h[p][1] if t % 2 == 0) for p in h}
    return [h[p][0] + t2[p] + t2.get(p - 1, 0) for p in range(top + 1)]


@pytest.mark.parametrize("backend", BACKENDS)
def test_lower_star_gradients_are_acyclic_matchings_with_the_right_homology(backend):
    for K, g in _cases():
        V = lower_star_gradient(K, g, backend=backend)
        V.verify()
        want = K.homology(backend="python")
        got = V.morse_homology(backend="python")
        assert {p: (r, sorted(t)) for p, (r, t) in got.items()} == \
            {p: (r, sorted(t)) for p, (r, t) in want.items()}
        m = V.morse_vector()
        assert all(mp >= bp for mp, bp in zip(m, _f2_betti(K)))
        assert sum((-1) ** p * mp for p, mp in enumerate(m)) == K.euler_characteristic()


def test_the_morse_complex_is_a_pysurgery_chain_complex():
    K = T.klein_bottle(5, 5)
    g = np.random.default_rng(7).uniform(size=25)
    cc = lower_star_gradient(K, g, backend="python").morse_chain_complex()
    assert cc.homology(1, backend="python") == (1, [2])     # H_1(Klein bottle) = Z + Z/2


def test_a_perfect_height_function_on_the_octahedron():
    """The height function z on the octahedron has one minimum and one maximum and
    nothing else: m = (1, 0, 1) = beta."""
    K, V = T.octahedron()
    Vf = lower_star_gradient(K, V[:, 2] + 1e-3 * V[:, 0] + 1e-6 * V[:, 1], backend="python")
    assert Vf.morse_vector() == [1, 0, 1]


def test_vertex_function_validation():
    K = T.sc([(3, 7)])
    with pytest.raises(ValueError):
        lower_star_gradient(K, [0.0, 1.0])                  # vertex 7 has no value
    V = lower_star_gradient(K, {3: 0.0, 7: 1.0}, backend="python")
    assert V.critical_cells() == [(3,)] and V.up == {(7,): (3, 7)}


@pytest.mark.parametrize("backend", BACKENDS)
def test_pair_classes_agree_with_the_morse_boundary(backend):
    for K, g in _cases():
        V = lower_star_gradient(K, g, backend=backend)
        for p in range(1, K.dimension + 1):
            M = V.morse_boundary(p)
            rows, cols = V.critical_cells(p - 1), V.critical_cells(p)
            for rec in classify_critical_pairs(V, p):
                assert rec["class"] in PairClass.ORDER
                assert rec["incidence"] == M[rows.index(rec["tau"]), cols.index(rec["sigma"])]
                if rec["class"] == PairClass.CANCELLABLE:
                    assert rec["paths"] == 1 and abs(rec["incidence"]) == 1
                    region = V.cancellation_region(rec["sigma"], rec["tau"])
                    assert rec["sigma"] in region and rec["tau"] in region


@pytest.mark.parametrize("backend", BACKENDS)
def test_persistence_counts_z2_homology_in_its_essential_classes(backend):
    rng = np.random.default_rng(1)
    for K in (T.projective_plane(), T.klein_bottle(5, 5), T.torus(4, 4)):
        n = max(s[0] for s in K.n_simplices(0)) + 1
        g = rng.uniform(size=n)
        pairs = lower_star_persistence(K, g, backend=backend)
        ess = [0] * (K.dimension + 1)
        for pr in pairs:
            if pr.is_essential:
                ess[pr.dimension] += 1
        assert ess == _f2_betti(K)
        V = lower_star_gradient(K, g, backend=backend)
        for (sigma, tau), pers in critical_pair_persistence(K, V, backend=backend).items():
            assert V.is_critical(sigma) and V.is_critical(tau) and pers >= 0


@pytest.mark.skipif(not julia_engine.available, reason="Julia backend unavailable")
def test_python_and_julia_are_identical():
    for K, g in _cases():
        a = lower_star_gradient(K, g, backend="python")
        b = lower_star_gradient(K, g, backend="julia")
        assert a.up == b.up and a.critical == b.critical
        assert lower_star_persistence(K, g, True, backend="python") == \
            lower_star_persistence(K, g, True, backend="julia")


def test_persistence_matches_gudhi():
    gd = pytest.importorskip("gudhi")
    rng = np.random.default_rng(2)
    for K in (T.projective_plane(), T.torus(4, 4), T.klein_bottle(5, 5)):
        n = max(s[0] for s in K.n_simplices(0)) + 1
        g = rng.uniform(size=n)
        st = gd.SimplexTree()
        for d in K.dimensions:
            for s in K.n_simplices(d):
                st.insert(list(s), filtration=float(max(g[v] for v in s)))
        # the default field of gudhi is Z/11, where RP^2 has beta = (1, 0, 0): pass Z/2
        dgm = st.persistence(homology_coeff_field=2, persistence_dim_max=True)
        theirs = sorted((d, round(b, 12), round(e, 12)) for d, (b, e) in dgm
                        if e == float("inf") or e > b)
        ours = sorted((pr.dimension, round(pr.birth, 12),
                       float("inf") if pr.is_essential else round(pr.death, 12))
                      for pr in lower_star_persistence(K, g, backend="python"))
        assert ours == theirs
