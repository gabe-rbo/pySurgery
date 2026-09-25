"""Tests for the dual active-set alpha complex (pysurgery.gpu.dual_alpha).

Overview:
    The dual-alpha builder never computes a Delaunay triangulation, so every test
    checks it against an authority that shares none of its machinery:

    * pySurgery's own Delaunay-based alpha complex and alpha filtration values
      (``SimplicialComplex.from_alpha_complex(backend="python")``,
      ``alpha_filtration_values``) -- simplex for simplex on points in general
      position, at radii strictly between filtration values;
    * CGAL's exact alpha complex through gudhi, when gudhi is installed;
    * the witness points themselves: every simplex's witness must be equidistant
      from its vertices at the recorded alpha value, with no other point closer;
    * exact identities (translation invariance, a constant power shifting every
      value) and known homotopy types (circle, sphere, grid nerve).

    All inputs are tiny (at most ~120 points), so the suite stays light on memory.
"""
import itertools
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pysurgery.gpu import dual_alpha as DA  # noqa: E402
from pysurgery.gpu.dual_alpha import (  # noqa: E402
    AlphaBudgetExceeded,
    connectivity_radius,
    dual_alpha_complex,
)
from pysurgery.topology.complexes import SimplicialComplex  # noqa: E402
from pysurgery.topology.filtration_values import _miniball_r2, alpha_filtration_values  # noqa: E402


def _circle(n=40, seed=0):
    t = np.sort(np.random.default_rng(seed).uniform(0, 2 * np.pi, n))
    return np.c_[np.cos(t), np.sin(t)]


def _sphere(n=120, seed=0):
    V = np.random.default_rng(seed).normal(size=(n, 3))
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def _radii_between(values, k):
    """k radii strictly between consecutive distinct filtration radii (never a tie)."""
    v = np.unique(np.asarray(values, dtype=float))
    v = v[v > 0]
    mids = (v[:-1] + v[1:]) / 2
    return mids[np.unique(np.linspace(0, len(mids) - 1, k).astype(int))]


def _simplices(res):
    return set(res.complex.simplices)


# ─────────────────────────────────────────────── against the Delaunay alpha complex

@pytest.mark.parametrize("m,n", [(2, 25), (3, 18)])
def test_identical_to_the_delaunay_alpha_complex_at_every_scale(m, n):
    from scipy.spatial import Delaunay
    P = np.random.default_rng(7 + m).uniform(size=(n, m))
    vals = alpha_filtration_values(P, Delaunay(P).simplices, max_dim=m)
    for r in _radii_between(list(vals.values()), 8):
        want = {s for s, v in vals.items() if v <= r}
        got = _simplices(dual_alpha_complex(P, float(r), max_simplices=None))
        assert got == want, (m, r, sorted(got ^ want)[:5])


def test_filtration_values_are_the_alpha_values():
    from scipy.spatial import Delaunay
    P = np.random.default_rng(3).uniform(size=(20, 2))
    vals = alpha_filtration_values(P, Delaunay(P).simplices, max_dim=2)
    res = dual_alpha_complex(P, 1.05 * max(vals.values()), max_simplices=None)
    got = res.filtration_values()
    assert set(got) == set(vals)
    assert max(abs(got[s] - vals[s]) / max(vals[s], 1e-12) for s in vals) < 1e-9
    assert res.complex.filtration == got
    for s, w in got.items():                        # monotone under faces
        for f in itertools.combinations(s, len(s) - 1):
            if f:
                assert got[f] <= w


def test_from_alpha_complex_gpu_backend_matches_python_backend():
    P = np.random.default_rng(11).uniform(size=(30, 2))
    a = SimplicialComplex.from_alpha_complex(P, 0.14, backend="gpu")
    b = SimplicialComplex.from_alpha_complex(P, 0.14, backend="python")
    assert set(a.simplices) == set(b.simplices)
    sc, res = SimplicialComplex.from_dual_alpha_complex(P, 0.14, return_result=True)
    assert set(sc.simplices) == set(b.simplices) and res.radius == 0.14


def test_identical_to_cgal_exact_alpha_complex():
    gd = pytest.importorskip("gudhi")
    rng = np.random.default_rng(7)
    for m, n in ((2, 20), (3, 14), (4, 10)):
        P = rng.uniform(size=(n, m))
        st = gd.AlphaComplex(points=P, precision="exact").create_simplex_tree()
        filt = [(tuple(sorted(s)), f) for s, f in st.get_filtration()]
        for r in _radii_between([np.sqrt(f) for _, f in filt], 5):
            want = {s for s, f in filt if f <= r * r}
            assert _simplices(dual_alpha_complex(P, float(r), max_simplices=None)) == want


def test_the_exact_solver_is_cgal():
    """``exact_decide`` alone on every 2- and 3-subset, against CGAL's exact values."""
    gd = pytest.importorskip("gudhi")
    P = np.random.default_rng(5).uniform(size=(9, 2))
    st = gd.AlphaComplex(points=P, precision="exact").create_simplex_tree()
    filt = {tuple(sorted(s)): f for s, f in st.get_filtration()}
    for r in _radii_between([np.sqrt(f) for f in filt.values()], 3):
        for k in (2, 3):
            for sig in itertools.combinations(range(len(P)), k):
                nb = [i for i in range(len(P)) if i != sig[0]]
                e = DA.exact_decide(P, None, sig[0], nb, [nb.index(v) for v in sig[1:]], float(r))
                assert e["accept"] == (sig in filt and filt[sig] <= r * r), (sig, r)


# ─────────────────────────────────────────────────────────── witnesses and identities

def test_every_witness_realises_its_value_in_the_restricted_voronoi_face():
    for P, r in ((_sphere(80), 0.55), (np.array([[i, j] for i in range(4) for j in range(4)], float), 0.8)):
        res = dual_alpha_complex(P, r)
        for s, y in res.witnesses.items():
            d = np.linalg.norm(P[list(s)] - y, axis=1)
            assert np.allclose(d, d[0], atol=1e-7) and d[0] <= r + 1e-7
            assert abs(d[0] ** 2 - res.weights[s]) < 1e-7
            assert np.min(np.linalg.norm(P - y, axis=1)) >= d[0] - 1e-7
            assert res.weights[s] >= _miniball_r2(P[list(s)]) - 1e-9   # never below the MEB
    assert res.stats["n_exact_rational"] > 0      # the grid's ties went to the exact solver


def test_translation_and_units_do_not_change_the_complex():
    P = np.random.default_rng(2).uniform(size=(35, 3))
    base = _simplices(dual_alpha_complex(P, 0.27))
    assert _simplices(dual_alpha_complex(P + 1e6, 0.27)) == base      # far from the origin
    assert _simplices(dual_alpha_complex(P * 1e-3, 0.27e-3)) == base
    assert _simplices(dual_alpha_complex(P * 1e3, 0.27e3)) == base


def test_a_constant_power_shifts_the_filtration_exactly():
    """With p = c everywhere, Alpha(S, p, a1) = Alpha(S, 0, a1 + c) and w -> w - c."""
    P = np.random.default_rng(4).uniform(size=(25, 2))
    c = 0.01
    weighted = dual_alpha_complex(P, 0.12, power=np.full(len(P), c))
    plain = dual_alpha_complex(P, float(np.sqrt(0.12 ** 2 + c)))
    assert _simplices(weighted) == _simplices(plain)
    for s, w in weighted.weights.items():
        assert abs(w - (plain.weights[s] - c)) < 1e-12
    with pytest.raises(ValueError):
        dual_alpha_complex(P, 0.05, power=np.full(len(P), 1.0)).filtration_values()


def test_subcomplex_is_the_complex_rebuilt_at_that_radius():
    P = _sphere(90, seed=1)
    res = dual_alpha_complex(P, 0.5)
    for r in (0.2, 0.33, 0.45, 0.5):
        assert set(res.subcomplex(r).simplices) == _simplices(dual_alpha_complex(P, r))
    with pytest.raises(ValueError):
        res.subcomplex(0.6)


# ─────────────────────────────────────────────────────────────── known homotopy types

def test_known_homotopy_types():
    assert dual_alpha_complex(_circle(40), 0.4).complex.betti_numbers() == {0: 1, 1: 1}
    assert dual_alpha_complex(_sphere(120), 0.6).complex.betti_numbers() == {0: 1, 1: 0, 2: 1}
    grid = np.array([[i, j] for i in range(3) for j in range(3)], dtype=float)
    assert dual_alpha_complex(grid, 0.6).complex.betti_numbers() == {0: 1, 1: 4}
    K = dual_alpha_complex(grid, 0.8).complex       # co-circular corners share a witness
    assert len(K.n_simplices(3)) == 4 and K.betti_numbers() == {0: 1, 1: 0, 2: 0, 3: 0}


def test_high_ambient_dimension_where_delaunay_is_impossible():
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.normal(size=(30, 2)))
    P = _circle(36) @ Q.T + 1e-4 * rng.normal(size=(36, 30))
    res = dual_alpha_complex(P, 0.35)
    b = res.complex.betti_numbers()
    assert (b[0], b[1]) == (1, 1) and all(v == 0 for d, v in b.items() if d >= 2)
    assert res.stats["m"] == 30


def test_connectivity_radius_is_the_connection_scale():
    P = np.random.default_rng(9).uniform(size=(30, 3)) * 5.0 + 100.0
    r0 = connectivity_radius(P)
    assert dual_alpha_complex(P, r0).complex.num_connected_components() == 1
    assert dual_alpha_complex(P, r0 * (1 - 1e-6)).complex.num_connected_components() >= 2


# ─────────────────────────────────────────────────────── precision and device policy

def test_float32_graph_is_a_certified_superset_even_far_from_the_origin():
    P = np.random.default_rng(3).uniform(size=(40, 3))
    D = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
    for off in (0.0, 1e4):
        Q = (P + off) / 0.25
        g = DA._cech_graph(Q, 1.0, np.zeros(len(Q)), torch.device("cpu"), torch.float32)
        for i in range(len(P)):
            true_nb = set(np.nonzero((D[i] / 0.25 <= 2.0) & (np.arange(len(P)) != i))[0])
            assert true_nb <= set(g[i].tolist())


def test_float32_may_screen_but_never_decides():
    P = np.random.default_rng(3).uniform(size=(45, 3))
    a = dual_alpha_complex(P, 0.3, dtype=torch.float64)
    b = dual_alpha_complex(P, 0.3, dtype=torch.float32)
    assert _simplices(a) == _simplices(b) and b.stats["n_screened"] > 0


def test_the_qp_never_runs_on_mps():
    import inspect
    src = inspect.getsource(DA.dual_alpha_complex)
    assert 'graph_dev.type == "mps"' in src and "qp_device" in inspect.signature(DA.dual_alpha_complex).parameters


# ──────────────────────────────────────────────────────────────── caps and refusals

def test_a_dimension_cap_is_recorded():
    res = dual_alpha_complex(_sphere(60), 0.5, max_dim=1)
    assert res.complex.dimension == 1 and res.truncation_dim == 1
    assert list(res.exact_betti_dimensions()) == [0]


def test_refusals():
    pts = {tuple(a * b for a, b in zip(perm, sg)) for v in ((3, 0, 0), (1, 2, 2))
           for perm in itertools.permutations(v) for sg in itertools.product((1, -1), repeat=3)}
    P = np.array(sorted(pts), dtype=float)          # 30 integer points ON the sphere of radius 3
    with pytest.raises(AlphaBudgetExceeded, match="circumradius"):
        dual_alpha_complex(P, 3.5, max_simplices=1000)
    assert dual_alpha_complex(P, 2.0).complex.betti_numbers() == {0: 1, 1: 0, 2: 1}
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            dual_alpha_complex(_circle(10), bad)
    with pytest.raises(ValueError):
        dual_alpha_complex(np.zeros(5), 1.0)


def test_undecided_candidates_raise_instead_of_guessing():
    """Without the exact solver a gray-zone candidate must raise, never be dropped."""
    grid = np.array([[i, j] for i in range(3) for j in range(3)], dtype=float)
    nb = np.array([1, 3, 4], dtype=np.int64)        # neighbours of vertex 0 on the unit grid
    Dnp = grid[nb] - grid[0]
    B = torch.as_tensor(Dnp @ Dnp.T)
    U = torch.as_tensor(-0.5 * (Dnp * Dnp).sum(1))
    J = torch.as_tensor([[0, 1, 2]])                # the square {0, 1, 3, 4}: co-circular
    stats = dict(n_equality_only=0, n_active_set=0, n_exact_fallback=0, n_undecided=0, n_screened=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(DA.AlphaUndecided):
            DA._decide(B, U, J, 0.5, 8, 64, 1e-11, 1e-9, B, U, stats, exact_fn=None)
