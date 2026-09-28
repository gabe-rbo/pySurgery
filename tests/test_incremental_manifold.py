"""Exact homology-manifold verdicts along filtrations.

The incremental checker must give, at every threshold, the verdict of the definition
as ``certify_homology_manifold`` checks it (every simplex's link, exact integer
homology). Regressions pinned here: a manifold with boundary used to be reported as a
non-manifold, a single triangle as a *closed 1-manifold*, and a branching complex whose
vertex links are all acyclic (three triangles on one edge) as a manifold.
"""

import itertools
import random
from collections import Counter

import numpy as np
import pytest

from pysurgery.topology.complexes import SimplicialComplex
from pysurgery.topology.filtration_tools import AlphaFiltrationReport, RipsFiltrationReport
from pysurgery.topology.incremental_manifold import (
    IncrementalManifoldChecker,
    filtration_manifold_verdicts,
    manifold_verdict,
)
from pysurgery.topology.local_homology import certify_homology_manifold, classify_simplices


# --------------------------------------------------------------------------- #
# triangulations
# --------------------------------------------------------------------------- #
def _closure(tops):
    out = set()
    for t in tops:
        t = tuple(sorted(t))
        for r in range(1, len(t) + 1):
            out.update(itertools.combinations(t, r))
    return out


def _sphere(n):
    """Boundary of the (n+1)-simplex: S^n."""
    return list(itertools.combinations(range(n + 2), n + 1))


def _torus7():
    return [tuple(sorted({i % 7, (i + 1) % 7, (i + 3) % 7})) for i in range(7)] + [
        tuple(sorted({i % 7, (i + 2) % 7, (i + 3) % 7})) for i in range(7)
    ]


_RP2 = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5), (0, 1, 5),
        (1, 2, 4), (2, 3, 5), (1, 3, 4), (2, 4, 5), (1, 3, 5)]
_MOBIUS = [(0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 0), (4, 0, 1)]


def _cone(tops, apex):
    return [tuple(sorted(t + (apex,))) for t in tops]


def _suspension(tops, a, b):
    return _cone(tops, a) + _cone(tops, b)


def _shift(tops, k):
    return [tuple(v + k for v in t) for t in tops]


# name -> (top simplices, is_manifold, dimension, is_closed)
CASES = {
    "triangle": ([(0, 1, 2)], True, 2, False),
    "hexagon disk": ([(6, i, (i + 1) % 6) for i in range(6)], True, 2, False),
    "3-page book": ([(0, 1, 2), (0, 1, 3), (0, 1, 4)], False, 2, False),
    "circle": (_sphere(1), True, 1, True),
    "path": ([(0, 1), (1, 2)], True, 1, False),
    "S2": (_sphere(2), True, 2, True),
    "torus": (_torus7(), True, 2, True),
    "RP2": (_RP2, True, 2, True),
    "mobius": (_MOBIUS, True, 2, False),
    "B3": ([(0, 1, 2, 3)], True, 3, False),
    "S3": (_sphere(3), True, 3, True),
    "S4": (_sphere(4), True, 4, True),
    "B4": ([(0, 1, 2, 3, 4)], True, 4, False),
    "two points": ([(0,), (1,)], True, 0, True),
    # vertex links have S^1 / point homology, but the edge (0, 3) meets 3 triangles
    "S2 wedge D2 at a vertex": (_sphere(2) + [(3, 4, 5), (3, 5, 6), (3, 6, 4)], False, 2, False),
    "two S2 at a vertex": (_sphere(2) + _shift(_sphere(2), 3), False, 2, False),
    # every vertex link is acyclic; the singular set is the free edge
    "D2 and a disjoint edge": ([(0, 1, 2), (3, 4)], False, 2, False),
    "D2 with a dangling edge": ([(0, 1, 2), (2, 3)], False, 2, False),
    "S2 and a disjoint S1": (_sphere(2) + _shift(_sphere(1), 10), False, 2, False),
    # codimension 3: the apex link is a torus (chi 0) / RP^2 (chi 1, closed)
    "cone over torus": (_cone(_torus7(), 20), False, 3, False),
    "suspension of RP2": (_suspension(_RP2, 20, 21), False, 3, False),
    # codimension 4 (exact link homology)
    "suspension of S3": (_suspension(_sphere(3), 10, 11), True, 4, True),
    "suspension of cone over torus": (_suspension(_cone(_torus7(), 20), 30, 31), False, 4, False),
}


def _reference(simplices):
    """(is_manifold, dim, #inclusion-maximal singular simplices, is_closed) by the definition."""
    K = SimplicialComplex.from_simplices(sorted(simplices, key=len), close_under_faces=True)
    d = K.dimension
    cert = certify_homology_manifold(K, d, backend="python")
    other = {s for s, t in classify_simplices(K, d, backend="python").items() if t.kind == "other"}
    maximal = sum(1 for s in other if not any(t != s and set(s) <= set(t) for t in other))
    return cert.is_homology_manifold_with_boundary, d, maximal, cert.is_closed_homology_manifold


@pytest.mark.parametrize("name", sorted(CASES))
def test_one_shot_verdict_matches_definition(name):
    tops, is_mani, dim, closed = CASES[name]
    v = manifold_verdict(tops)
    assert (v.is_manifold, v.dimension, v.is_closed) == (is_mani, dim, closed)
    ref = _reference(_closure(tops))
    assert (v.is_manifold, v.dimension, v.defects, v.is_closed) == ref


def test_defect_count_is_the_singular_frontier():
    # The book: only the spine edge is singular.
    assert manifold_verdict([(0, 1, 2), (0, 1, 3), (0, 1, 4)]).defects == 1
    # Disjoint S^1 next to an S^2: the circle's three edges are maximal and singular.
    assert manifold_verdict(_sphere(2) + _shift(_sphere(1), 10)).defects == 3


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("seed", [0, 1])
def test_every_threshold_matches_definition(name, seed):
    """Random filtration orders on each complex: all sub-complexes K_eps are checked."""
    tops = CASES[name][0]
    rng = random.Random(seed)
    tv = {tuple(sorted(t)): rng.random() for t in tops}
    values = {s: min(v for t, v in tv.items() if set(s) <= set(t)) for s in _closure(tops)}
    eps = sorted(set(values.values()))
    for e, v in zip(eps, filtration_manifold_verdicts(values, eps)):
        sub = [s for s, x in values.items() if x <= e]
        assert (v.is_manifold, v.dimension, v.defects, v.is_closed) == _reference(sub), e


def test_flag_complexes_match_definition():
    """Rips-type filtrations (mostly singular) on random points."""
    rng = np.random.default_rng(7)
    for _ in range(6):
        pts = rng.random((9, 3))
        D = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
        values = {(i,): 0.0 for i in range(len(pts))}
        for r in (2, 3, 4):
            for c in itertools.combinations(range(len(pts)), r):
                values[c] = float(max(D[a, b] for a, b in itertools.combinations(c, 2)))
        eps = sorted(set(values.values()))
        for e, v in zip(eps, filtration_manifold_verdicts(values, eps)):
            sub = [s for s, x in values.items() if x <= e]
            assert (v.is_manifold, v.dimension, v.defects, v.is_closed) == _reference(sub)


def test_thresholds_in_any_order():
    values = {s: 0.0 for s in _closure([(0, 1, 2)])}
    values.update({(3,): 0.0, (2, 3): 1.0, (1, 3): 1.0, (1, 2, 3): 2.0})
    fwd = filtration_manifold_verdicts(values, [0.0, 1.0, 2.0])
    rev = filtration_manifold_verdicts(values, [2.0, 1.0, 0.0])
    assert fwd == rev[::-1]
    assert [v.is_manifold for v in fwd] == [False, False, True]   # D2 + point, D2 + fin, D2


def test_missing_faces_are_added():
    chk = IncrementalManifoldChecker()
    chk.add((0, 1, 2))
    assert len(chk) == 7
    assert chk.verdict() == (True, 2, 0, False)


# --------------------------------------------------------------------------- #
# filtration reports
# --------------------------------------------------------------------------- #
def _book_points():
    r = 0.8
    return np.array([[-0.5, 0, 0], [0.5, 0, 0]]
                    + [[0, r * np.cos(t), r * np.sin(t)] for t in (0, 2 * np.pi / 3, 4 * np.pi / 3)])


@pytest.mark.parametrize(
    "points, expected",
    [
        (np.array([[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3) / 2, 0]]), ("Yes", "No", "2")),
        (np.vstack([[np.cos(a), np.sin(a), 0] for a in np.arange(6) * np.pi / 3] + [[0, 0, 0]]),
         ("Yes", "No", "2")),
        (_book_points(), ("No (1 dft)", "No", "2")),
    ],
    ids=["triangle", "hexagon-disk", "3-page-book"],
)
def test_report_manifold_row(points, expected):
    rep = RipsFiltrationReport(points, epsilons=[1.05], max_dimension=2,
                               backend="python", analyze_manifolds=True)
    row = rep.results[-1]
    assert (row["is_manifold"], row["is_closed"], row["dimension"]) == expected


def test_report_component_rows_use_own_dimension():
    # A triangle far from a square cycle: a 2-disk and a closed 1-manifold.
    tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
    sq = np.array([[10, 0], [11, 0], [11, 1], [10, 1]])
    rep = RipsFiltrationReport(np.vstack([tri, sq]), epsilons=[1.05], max_dimension=2,
                               backend="python", analyze_manifolds=True,
                               track_connected_components=True)
    row = rep.results[-1]
    assert row["is_manifold"].startswith("No")      # not a manifold of one dimension
    infos = Counter(v for v in row["comp_info_map"].values() if v.startswith(("M(", "Non-M")))
    assert infos == Counter({"M(D:2, Bound)": 1, "M(D:1, Closed)": 1})


def _julia_available():
    try:
        from pysurgery.bridge.julia_bridge import julia_engine
        return julia_engine.available
    except Exception:
        return False


class _ForceFusedRips(RipsFiltrationReport):
    _RIPS_FUSED_MIN_POINTS = 0
    _EXPLICIT_MAX_SIMPLICES = 0


class _ForceFusedAlpha(AlphaFiltrationReport):
    _ALPHA_FUSED_MIN_POINTS = 0
    _EXPLICIT_MAX_SIMPLICES = 0


def _component_strings(values, eps, tol=1e-9):
    sub = [s for s, x in values.items() if x <= eps + tol]
    parent = {}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for s in sub:
        for u in s:
            parent.setdefault(u, u)
    for s in sub:
        if len(s) == 2:
            parent[find(s[0])] = find(s[1])
    groups = {}
    for s in sub:
        groups.setdefault(find(s[0]), []).append(s)
    out = Counter()
    for simps in groups.values():
        v = manifold_verdict(simps)
        out[f"M(D:{v.dimension}, {'Closed' if v.is_closed else 'Bound'})" if v.is_manifold
            else f"Non-M ({v.defects} dft)"] += 1
    return out


@pytest.mark.skipif(not _julia_available(), reason="Julia backend unavailable")
@pytest.mark.parametrize("fused_cls, staged_cls", [(_ForceFusedRips, RipsFiltrationReport),
                                                   (_ForceFusedAlpha, AlphaFiltrationReport)])
def test_julia_engine_matches_python_engine(fused_cls, staged_cls):
    """The fused Julia manifold analysis gives the Python verdicts, globally and per component."""
    rng = np.random.default_rng(3)
    pts = np.vstack([rng.random((14, 2)), rng.random((10, 2)) + [3.0, 0.0]])
    fused = fused_cls(pts, max_dimension=2, analyze_manifolds=True,
                      track_connected_components=True)
    staged = staged_cls(pts, max_dimension=2, analyze_manifolds=False, backend="python")
    assert fused.max_sc is None
    eps = [r["epsilon"] for r in fused.results]
    # Julia and Python compute the appearance values separately: compare up to 1e-9.
    ref = filtration_manifold_verdicts(staged._filt, eps, tol=1e-9)
    for row, v in zip(fused.results, ref):
        want = "Yes" if v.is_manifold else f"No ({v.defects} dft)"
        assert row["is_manifold"] == want
        assert row["is_closed"] == ("Yes" if v.is_closed else "No")
        got = Counter(s for s in row["comp_info_map"].values() if s.startswith(("M(", "Non-M")))
        assert got == _component_strings(staged._filt, row["epsilon"])


@pytest.mark.skipif(not _julia_available(), reason="Julia backend unavailable")
@pytest.mark.parametrize("fused_cls, staged_cls", [(_ForceFusedRips, RipsFiltrationReport),
                                                   (_ForceFusedAlpha, AlphaFiltrationReport)])
def test_julia_component_rows_match_python_rows(fused_cls, staged_cls):
    """Row numbering, merges ('Merged (C_k)' then '-') and verdicts agree across paths."""
    rng = np.random.default_rng(5)
    pts = np.vstack([rng.random((12, 2)), rng.random((9, 2)) + [2.5, 0.0],
                     rng.random((6, 2)) + [0.0, 2.5]])
    fused = fused_cls(pts, max_dimension=2, analyze_manifolds=True,
                      track_connected_components=True)
    eps = list(fused.epsilons)
    staged = staged_cls(pts, epsilons=eps, max_dimension=2, analyze_manifolds=True,
                        track_connected_components=True, backend="python")
    assert fused.max_sc is None and staged.max_sc is not None
    md = fused._precomputed_manifolds
    assert len(staged.results) == len(eps) == len(fused._precomputed_components)
    for i, row in enumerate(staged.results):
        assert row["comp_info_map"] == fused._precomputed_components[i]
        assert row["is_manifold"] == ("Yes" if md["is_manifold"][i]
                                      else f"No ({md['failures'][i]} dft)")
    assert any(v.startswith("Merged") for c in fused._precomputed_components for v in c.values())
