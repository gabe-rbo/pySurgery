"""Tests for the connected-component decomposition optimization.

The component-aware `homology`/`cohomology`/`betti_*` routing on
`SimplicialComplex` is a *pure optimization*: for a disjoint union the
homology of the whole block-diagonal complex equals the direct sum of the
component homologies, so the routed result must match the non-decomposed
computation exactly. These tests pin that equivalence, the β₀ fast path, the
explode/partition contract, and cache invalidation on mutation.
"""

import warnings

import pytest

from pysurgery.topology.complexes import SimplicialComplex


def _circle(offset=0):
    a, b, c = offset, offset + 1, offset + 2
    return [(a, b), (b, c), (c, a)]


def _sphere(offset=0):
    """Boundary of an octahedron on 6 vertices -> S^2."""
    a, b, c, d, e, f = (offset + i for i in range(6))
    return [
        (a, c, e), (a, e, d), (a, d, f), (a, f, c),
        (b, c, e), (b, e, d), (b, d, f), (b, f, c),
    ]


def _rp2():
    return [
        (1, 2, 3), (1, 3, 4), (1, 4, 5), (1, 5, 6), (1, 2, 6),
        (2, 3, 5), (3, 4, 6), (4, 5, 2), (5, 6, 3), (6, 2, 4),
    ]


# --------------------------------------------------------------------------- #
# Component counting / connectivity
# --------------------------------------------------------------------------- #

def test_num_components_counts_isolated_vertices():
    # two circles + one isolated vertex
    K = SimplicialComplex.from_simplices(
        _circle(0) + _circle(10) + [(99,)], close_under_faces=True
    )
    assert K.num_connected_components() == 3
    assert K.is_connected() is False


def test_connected_single_component():
    K = SimplicialComplex.from_maximal_simplices(_sphere(0))
    assert K.num_connected_components() == 1
    assert K.is_connected() is True


def test_empty_complex_is_not_connected():
    K = SimplicialComplex.from_simplices([], close_under_faces=True)
    assert K.num_connected_components() == 0
    assert K.is_connected() is False
    assert K.homology() == {}


# --------------------------------------------------------------------------- #
# β₀ fast path and direct-sum homology
# --------------------------------------------------------------------------- #

def test_h0_equals_component_count():
    K = SimplicialComplex.from_simplices(
        _circle(0) + _circle(10) + [(99,)], close_under_faces=True
    )
    assert K.homology(0) == (3, [])
    assert K.betti_number(0) == 3


def test_two_circles_direct_sum():
    K = SimplicialComplex.from_simplices(_circle(0) + _circle(10), close_under_faces=True)
    assert K.homology(0) == (2, [])
    assert K.homology(1) == (2, [])
    assert K.betti_numbers() == {0: 2, 1: 2}


def test_two_spheres_direct_sum():
    K = SimplicialComplex.from_maximal_simplices(_sphere(0) + _sphere(100))
    assert K.homology(0) == (2, [])
    assert K.homology(1) == (0, [])
    assert K.homology(2) == (2, [])
    assert K.cohomology(2) == (2, [])


def test_disjoint_torsion_is_aggregated():
    # RP^2 (H_1 = Z/2) disjoint from a circle (H_1 = Z)
    K = SimplicialComplex.from_maximal_simplices(_rp2() + _circle(100))
    assert K.num_connected_components() == 2
    rank, torsion = K.homology(1)
    assert rank == 1            # the circle contributes Z
    assert torsion == [2]       # RP^2 contributes the Z/2 torsion


# --------------------------------------------------------------------------- #
# Equivalence guard: routed result == non-decomposed full-matrix result
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "maximal",
    [
        _sphere(0) + _sphere(100),                 # 2 components
        _rp2() + _circle(100),                     # torsion + free, 2 components
        _sphere(0) + _circle(50) + _rp2(),         # 3 components, mixed (disjoint labels)
    ],
)
def test_routed_matches_direct(maximal):
    K = SimplicialComplex.from_maximal_simplices(maximal)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct = K.cellular_chain_complex().homology()      # full block-diagonal SNF
        routed = K.homology()                               # component-decomposed
        direct_co = K.cellular_chain_complex().cohomology()
        routed_co = K.cohomology()

    # Compare degree-by-degree with order-independent torsion.
    assert direct.keys() == routed.keys()
    for d in direct:
        dr, dt = direct[d]
        rr, rt = routed[d]
        assert dr == rr
        assert sorted(dt) == sorted(rt)
    for d in direct_co:
        assert direct_co[d][0] == routed_co[d][0]
        assert sorted(direct_co[d][1]) == sorted(routed_co[d][1])


# --------------------------------------------------------------------------- #
# explode / partition contract
# --------------------------------------------------------------------------- #

def test_explode_partitions_all_simplices():
    K = SimplicialComplex.from_maximal_simplices(_sphere(0) + _sphere(100))
    comps = K.explode()
    assert len(comps) == 2
    # vertices are partitioned (disjoint, covering)
    all_v = set(v[0] for v in K.n_simplices(0))
    comp_v = [set(v[0] for v in c.n_simplices(0)) for c in comps]
    assert comp_v[0] | comp_v[1] == all_v
    assert comp_v[0] & comp_v[1] == set()
    # components are ordered by descending vertex count
    assert len(comp_v[0]) >= len(comp_v[1])


def test_connected_components_are_cached_and_shared():
    K = SimplicialComplex.from_simplices(_circle(0) + _circle(10), close_under_faces=True)
    a = K.connected_components()
    b = K.connected_components()
    assert a is b  # same cached list
    assert all(x is y for x, y in zip(K.explode(), a))


def test_component_cache_invalidates_on_mutation():
    K = SimplicialComplex.from_simplices([(0, 1), (2, 3)], close_under_faces=True)
    assert K.num_connected_components() == 2
    first = K.connected_components()
    # bridge the two components
    K.add_simplex((1, 2))
    assert K.num_connected_components() == 1
    second = K.connected_components()
    assert second is not first  # cache was rebuilt after structural change


# --------------------------------------------------------------------------- #
# Phase 1: cross-threshold per-component info memoization in filtration_tools.
# All point clouds are tiny (12 points) with explicit small epsilons -> bounded.
# --------------------------------------------------------------------------- #

import itertools

import numpy as np

from pysurgery.topology.filtration_tools import FiltrationReport, _BaseFiltrationReport


def _two_circles_points():
    """Two disjoint hexagonal circles (12 points); adjacent chord ~1.0."""
    theta = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    c1 = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    c2 = c1 + np.array([5.0, 0.0])
    return np.vstack([c1, c2])


# eps grid where the two circles are *identical* across 1.1 -> 1.2 (cache hits)
# and then merge at 6.0 (a fresh, distinct component).
_EPS = [1.1, 1.2, 6.0]


def test_component_info_cache_is_output_neutral(monkeypatch):
    """The memoization must not change any reported invariant.

    Compare a normal run against one where every component hashes to a unique
    key, so the cache can never hit (i.e. the pre-optimization behavior).
    """
    pts = _two_circles_points()
    cached = FiltrationReport(pts, epsilons=_EPS, track_connected_components=True)

    counter = itertools.count()
    monkeypatch.setattr(
        _BaseFiltrationReport,
        "_component_content_key",
        staticmethod(lambda sub_sc: f"uniq-{next(counter)}"),
    )
    forced_miss = FiltrationReport(pts, epsilons=_EPS, track_connected_components=True)

    assert cached.results == forced_miss.results


def test_component_info_cache_reduces_manifold_checks(monkeypatch):
    """Stable components across thresholds should skip the per-component check."""
    pts = _two_circles_points()
    calls = {"n": 0}
    orig = SimplicialComplex.is_homology_manifold

    def counting(self, *args, **kwargs):
        calls["n"] += 1
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(SimplicialComplex, "is_homology_manifold", counting)

    # Cache active.
    FiltrationReport(pts, epsilons=_EPS, track_connected_components=True)
    cached_calls = calls["n"]

    # Force every key unique -> no cache hits (pre-optimization call count).
    calls["n"] = 0
    counter = itertools.count()
    monkeypatch.setattr(
        _BaseFiltrationReport,
        "_component_content_key",
        staticmethod(lambda sub_sc: f"uniq-{next(counter)}"),
    )
    FiltrationReport(pts, epsilons=_EPS, track_connected_components=True)
    miss_calls = calls["n"]

    assert cached_calls < miss_calls


def test_component_content_key_is_content_based():
    """Same simplices -> same key; one extra simplex -> different key.

    The key now hashes a ``{dim: simplices}`` table (as produced by
    ``component_simplex_tables``), so we feed the complexes' simplex tables.
    """
    a = SimplicialComplex.from_simplices([(0, 1), (1, 2), (2, 0)], close_under_faces=True)
    b = SimplicialComplex.from_simplices([(0, 1), (1, 2), (2, 0)], close_under_faces=True)
    c = SimplicialComplex.from_simplices([(0, 1), (1, 2)], close_under_faces=True)
    key = _BaseFiltrationReport._component_content_key
    assert key(a._simplices_table) == key(b._simplices_table)
    assert key(a._simplices_table) != key(c._simplices_table)


def test_component_simplex_tables_match_explode():
    """The lightweight table view matches connected_components() in order/content."""
    K = SimplicialComplex.from_maximal_simplices(_sphere(0) + _rp2() + _circle(200))
    comps = K.connected_components()
    tables = K.component_simplex_tables()
    assert len(tables) == len(comps)
    for (vset, table), comp in zip(tables, comps):
        # same vertex set and same descending-size ordering
        assert vset == frozenset(v[0] for v in comp.n_simplices(0))
        # same simplices in every dimension
        for d in comp.dimensions:
            assert sorted(table.get(d, [])) == sorted(comp.n_simplices(d))
        # no extra dimensions
        assert set(table.keys()) == set(comp.dimensions)


# --------------------------------------------------------------------------- #
# Phase 2: cross-threshold per-component *torsion* memoization.
# --------------------------------------------------------------------------- #


def test_torsion_memo_is_output_neutral(monkeypatch):
    """compute_torsion via per-component memo must equal the non-memoized path.

    Force every component key unique so the homology cache can never hit; the
    reported torsion must be byte-identical to the memoized run.
    """
    pts = _two_circles_points()
    memo = FiltrationReport(pts, epsilons=_EPS, compute_torsion=True)

    counter = itertools.count()
    monkeypatch.setattr(
        _BaseFiltrationReport,
        "_component_content_key",
        staticmethod(lambda table: f"uniq-{next(counter)}"),
    )
    forced_miss = FiltrationReport(pts, epsilons=_EPS, compute_torsion=True)

    assert [r["torsion"] for r in memo.results] == [r["torsion"] for r in forced_miss.results]
    assert memo.results == forced_miss.results


def test_torsion_memo_reduces_homology_solves(monkeypatch):
    """Stable components across thresholds should skip the per-component SNF."""
    pts = _two_circles_points()
    calls = {"n": 0}
    orig = SimplicialComplex.homology

    def counting(self, *args, **kwargs):
        calls["n"] += 1
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(SimplicialComplex, "homology", counting)

    FiltrationReport(pts, epsilons=_EPS, compute_torsion=True)
    memo_calls = calls["n"]

    calls["n"] = 0
    counter = itertools.count()
    monkeypatch.setattr(
        _BaseFiltrationReport,
        "_component_content_key",
        staticmethod(lambda table: f"uniq-{next(counter)}"),
    )
    FiltrationReport(pts, epsilons=_EPS, compute_torsion=True)
    miss_calls = calls["n"]

    assert memo_calls < miss_calls


def test_aggregate_component_torsion_contract():
    """Single component preserves native order; many components sort the pool."""
    agg = _BaseFiltrationReport._aggregate_component_torsion
    assert agg([]) == {}
    # one component -> native order preserved (no re-sort)
    assert agg([{1: (0, [4, 2])}]) == {1: [4, 2]}
    # several components -> direct sum, pooled torsion sorted per degree
    assert agg([{1: (1, [2])}, {1: (0, [3]), 2: (0, [2])}]) == {1: [2, 3], 2: [2]}
    # degrees with no torsion are dropped
    assert agg([{0: (2, []), 1: (0, [])}]) == {}


# --------------------------------------------------------------------------- #
# Phase 3: incremental union-find slice engine == per-threshold reference path.
# --------------------------------------------------------------------------- #


def test_incremental_slices_match_reference():
    """`_incremental_slices` reproduces `_slice_complex` + component_simplex_tables.

    The incremental union-find engine must yield, at every threshold, the same
    components (same vertex sets, same descending-size order, same simplex
    content) and the same whole-complex table / top dimension as the
    per-threshold rebuild it replaces.
    """
    pts = _two_circles_points()
    rep = FiltrationReport(pts, epsilons=_EPS, track_connected_components=True)
    max_sc, filt = rep.max_sc, rep._filt
    assert max_sc is not None
    coords = getattr(max_sc, "_coordinates", pts)

    inc = rep._incremental_slices(max_sc, filt, need_components=True, need_whole=True)
    for eps in rep.epsilons:
        comp_tables, full_table, cdim = next(inc)
        ref_sub = rep._slice_complex(SimplicialComplex, max_sc, filt, eps, coords)
        ref_tables = ref_sub.component_simplex_tables()

        # same components, in the same order, by vertex set
        assert [vt[0] for vt in comp_tables] == [vt[0] for vt in ref_tables]
        # same simplex content per component (order within a dim is irrelevant)
        for (v_i, t_i), (v_r, t_r) in zip(comp_tables, ref_tables):
            assert v_i == v_r
            assert set(t_i.keys()) == set(t_r.keys())
            for d in t_i:
                assert sorted(t_i[d]) == sorted(t_r[d])
        # whole-complex slice table + top dimension match
        assert {d: sorted(ss) for d, ss in full_table.items()} == \
            {d: sorted(ref_sub.n_simplices(d)) for d in ref_sub.dimensions}
        assert cdim == ref_sub.dimension


def test_incremental_full_report_matches_slice_rebuild(monkeypatch):
    """End-to-end: the report is byte-identical to one forced onto the old path.

    Monkeypatch `_incremental_slices` with a generator that rebuilds each slice
    from scratch via `_slice_complex` (the pre-Phase-3 behavior); the reported
    results, manifold info and torsion must be unchanged.
    """
    pts = _two_circles_points()

    def slice_rebuild(self, max_sc, filt, need_components, need_whole):
        coords = getattr(max_sc, "_coordinates", self.points)
        for eps in self.epsilons:
            sub = self._slice_complex(SimplicialComplex, max_sc, filt, eps, coords)
            comp_tables = sub.component_simplex_tables() if need_components else None
            full_table = sub._simplices_table if need_whole else None
            yield comp_tables, full_table, sub.dimension

    for kwargs in (
        dict(analyze_manifolds=True),                                  # whole only
        dict(track_connected_components=True, analyze_manifolds=False),  # cdim branch
        dict(compute_torsion=True, analyze_manifolds=False),            # cdim branch
        dict(track_connected_components=True, compute_torsion=True),     # all paths
    ):
        incremental = FiltrationReport(pts, epsilons=_EPS, **kwargs)
        monkeypatch.setattr(_BaseFiltrationReport, "_incremental_slices", slice_rebuild)
        reference = FiltrationReport(pts, epsilons=_EPS, **kwargs)
        monkeypatch.undo()
        assert incremental.results == reference.results
