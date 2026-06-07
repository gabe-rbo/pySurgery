"""Fast (Julia twist/clearing) persistence: exactness + torsion.

The fast reducer must return *exactly* the same barcode as the pure-Python
reference reducer (``_z2_persistence_barcode``) -- the optimisation is clearing,
which removes provably-redundant work only. These tests pin that equivalence on a
battery of complexes, and exercise the opt-in ``compute_torsion`` integer-homology
path (including a real Z/2 torsion class from a minimal RP^2 triangulation).
"""

import math
from collections import Counter

import numpy as np
import pytest

from pysurgery.topology.filtration_tools import (
    RipsFiltrationReport,
    CknnFiltrationReport,
    _BaseFiltrationReport,
)


def _bagify(bars):
    """Multiset of a barcode, robust to float noise and inf."""
    return Counter(
        (int(d),
         round(float(b), 9),
         "inf" if math.isinf(de) else round(float(de), 9))
        for (d, b, de) in bars
    )


def _circle(n, r=1.0, seed=0):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    return pts + np.random.default_rng(seed).normal(scale=1e-3, size=pts.shape)


def _blob(n, d=3, seed=1):
    return np.random.default_rng(seed).standard_normal((n, d))


def _two_circles(seed=2):
    a = _circle(14, r=1.0, seed=seed)
    b = _circle(14, r=1.0, seed=seed + 1) + np.array([4.0, 0.0])
    return np.vstack([a, b])


@pytest.mark.parametrize(
    "cls, kwargs, points",
    [
        (RipsFiltrationReport, {}, _circle(16)),
        (RipsFiltrationReport, {"eps_max": 1.6}, _two_circles()),
        (RipsFiltrationReport, {}, _blob(18, d=3)),
        (CknnFiltrationReport, {"k": 6}, _circle(18)),
        (CknnFiltrationReport, {"k": 5}, _blob(16, d=2)),
    ],
)
def test_fast_reducer_matches_python_oracle(cls, kwargs, points):
    """The barcode the engine computes equals the pure-Python reference bar-for-bar."""
    rep = cls(points, max_dimension=2, analyze_manifolds=False, **kwargs)
    fast = rep.barcode  # backend 'auto' -> Julia twist/clearing when available
    oracle = rep._z2_persistence_barcode(rep.max_sc._simplices_table, rep._filt)
    assert _bagify(fast) == _bagify(oracle)


def test_explicit_python_backend_matches_auto():
    """Forcing backend='python' yields the same Betti table as the fast path."""
    points = _two_circles()
    auto = RipsFiltrationReport(points, max_dimension=2, eps_max=1.6,
                                analyze_manifolds=False, backend="auto")
    py = RipsFiltrationReport(points, max_dimension=2, eps_max=1.6,
                              analyze_manifolds=False, backend="python")
    assert _bagify(auto.barcode) == _bagify(py.barcode)
    assert [r["bettis"] for r in auto.results] == [r["bettis"] for r in py.results]


def test_circle_betti_curve_is_correct():
    """A clean circle gives a long-lived b_1 = 1 regime (sanity on the bars)."""
    rep = RipsFiltrationReport(_circle(24), max_dimension=2, analyze_manifolds=False)
    # Some reported threshold must show exactly one 1-cycle and one component.
    assert any(r["bettis"].get(1, 0) == 1 and r["bettis"].get(0, 0) == 1
               for r in rep.results)


# ---------------------------------------------------------------------------
# Fused all-Julia Rips path (build + longest-edge values + reduction in Julia)
# ---------------------------------------------------------------------------
def _julia_available():
    try:
        from pysurgery.bridge.julia_bridge import julia_engine
        return julia_engine.available
    except Exception:
        return False


@pytest.mark.skipif(not _julia_available(), reason="Julia backend unavailable")
@pytest.mark.parametrize(
    "points, max_dim",
    [
        (_circle(16), 2),
        (_two_circles(), 2),
        (_blob(18, d=3), 2),
        (_blob(14, d=2), 3),
    ],
)
def test_fused_rips_kernel_matches_oracle(points, max_dim):
    """The fused Julia kernel's barcode equals the pure-Python oracle bar-for-bar.

    Cross-checks the kernel's longest-edge values *and* its reduction against an
    independently (Python-) built complex reduced by the reference reducer.
    """
    from scipy.spatial.distance import pdist
    from pysurgery.bridge.julia_bridge import julia_engine
    from pysurgery.topology.complexes import SimplicialComplex as SC
    from pysurgery.topology.filtration_values import rips_filtration_values

    eps_max = float(pdist(points).max())
    payload = julia_engine.compute_rips_filtration(points, eps_max, max_dim)

    sc = SC.from_vietoris_rips(points, eps_max, max_dim, backend="python")
    filt = rips_filtration_values(sc._simplices_table, points)
    oracle = _BaseFiltrationReport._z2_persistence_barcode(sc._simplices_table, filt)

    assert _bagify(payload["barcode"]) == _bagify(oracle)
    # The reported total matches the explicit complex's simplex count.
    assert payload["total"] == sum(len(v) for v in sc._simplices_table.values())


class _ForceFusedRips(RipsFiltrationReport):
    """Force the implicit fused path even on tiny clouds (for test coverage)."""

    _RIPS_FUSED_MIN_POINTS = 0
    _MANIFOLD_MAX_SIMPLICES = 0


@pytest.mark.skipif(not _julia_available(), reason="Julia backend unavailable")
def test_fused_routing_implicit_path_matches_python():
    """The report's fused implicit path (max_sc=None) matches the staged python path.

    ``eps_max=3.0`` exceeds the circle's diameter, so both pipelines build the same
    complete complex (no distance can tie the cap and drop an edge). Barcodes are
    compared through ``_bagify`` (9-dp rounding), which absorbs the sub-ULP
    differences between Julia and NumPy distance arithmetic.
    """
    pts = _circle(28)
    fused = _ForceFusedRips(pts, max_dimension=2, analyze_manifolds=False, eps_max=3.0)
    py = RipsFiltrationReport(pts, max_dimension=2, analyze_manifolds=False,
                              backend="python", eps_max=3.0)
    assert fused.max_sc is None                      # complex kept implicit
    assert _bagify(fused.barcode) == _bagify(py.barcode)


# ---------------------------------------------------------------------------
# Phase B: implicit persistent cohomology (Ripser-style) — exact == clique/oracle
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _julia_available(), reason="Julia backend unavailable")
@pytest.mark.parametrize(
    "points, max_dim",
    [
        (_circle(16), 2),
        (_blob(15, d=3), 2),
        (_blob(13, d=3), 3),     # high-dim: where cohomology pulls ahead
        (_two_circles(), 2),
    ],
)
def test_cohomology_kernel_matches_oracle(points, max_dim):
    """The implicit-cohomology kernel's barcode equals the pure-Python oracle.

    Persistent cohomology yields the identical barcode to homology (duality), so
    this is an exactness check, not an approximation."""
    from scipy.spatial.distance import pdist
    from pysurgery.bridge.julia_bridge import julia_engine
    from pysurgery.topology.complexes import SimplicialComplex as SC
    from pysurgery.topology.filtration_values import rips_filtration_values

    eps = float(pdist(points).max()) * 1.01
    payload = julia_engine.compute_rips_cohomology(points, eps, max_dim)

    sc = SC.from_vietoris_rips(points, eps, max_dim, backend="python")
    filt = rips_filtration_values(sc._simplices_table, points)
    oracle = _BaseFiltrationReport._z2_persistence_barcode(sc._simplices_table, filt)
    assert _bagify(payload["barcode"]) == _bagify(oracle)


@pytest.mark.skipif(not _julia_available(), reason="Julia backend unavailable")
def test_cohomology_report_engine_matches_clique():
    """The report's two fused engines (cohomology vs clique) give identical barcodes."""
    pts = _circle(28)
    coh = _ForceFusedRips(pts, max_dimension=2, analyze_manifolds=False, eps_max=3.0,
                          rips_engine="cohomology")
    cliq = _ForceFusedRips(pts, max_dimension=2, analyze_manifolds=False, eps_max=3.0,
                           rips_engine="clique")
    assert coh.max_sc is None and cliq.max_sc is None
    assert _bagify(coh.barcode) == _bagify(cliq.barcode)


def test_rips_engine_auto_selection():
    """Auto picks cohomology only for max_dimension >= 3; explicit choices honoured."""
    r2 = RipsFiltrationReport(_circle(10), max_dimension=2, analyze_manifolds=False)
    r3 = RipsFiltrationReport(_circle(10), max_dimension=3, analyze_manifolds=False)
    assert r2._select_rips_engine() == "clique"
    assert r3._select_rips_engine() == "cohomology"
    r3c = RipsFiltrationReport(_circle(10), max_dimension=3, analyze_manifolds=False,
                               rips_engine="clique")
    assert r3c._select_rips_engine() == "clique"


# ---------------------------------------------------------------------------
# compute_torsion
# ---------------------------------------------------------------------------
_RP2_FACETS = [(0, 1, 2), (0, 1, 3), (0, 2, 4), (0, 3, 5), (0, 4, 5),
               (1, 2, 5), (1, 3, 4), (1, 4, 5), (2, 3, 4), (2, 3, 5)]


class _RP2Report(_BaseFiltrationReport):
    """Engine driven by a fixed RP^2 triangulation (H_1 = Z/2), value = dimension."""

    method_name = "RP2-test"

    def _build_maximal_and_values(self):
        from pysurgery.topology.complexes import SimplicialComplex as SC
        sc = SC.from_simplices(_RP2_FACETS, coefficient_ring=self.coefficient_ring,
                               close_under_faces=True)
        filt = {}
        for d in sc._simplices_table:
            for s in sc.n_simplices(d):
                filt[s] = float(len(s) - 1)
        sc._coordinates = self.points
        return sc, filt


def test_compute_torsion_detects_rp2_z2_torsion():
    """compute_torsion surfaces the exact integer Z/2 class of RP^2."""
    pts = np.zeros((6, 2))
    rep = _RP2Report(pts, epsilons=[2.0], compute_torsion=True, analyze_manifolds=False)
    res = rep.results[-1]                      # threshold where the whole complex is present
    assert res["torsion"].get(1) == [2]        # H_1(RP^2; Z) = Z/2
    md = rep.to_markdown()
    assert "Integer Torsion Report" in md
    assert "Z/2" in md


def test_compute_torsion_default_off_and_torsion_free():
    """Default leaves torsion absent; on a torsion-free cloud the section is empty."""
    pts = _circle(14)
    off = RipsFiltrationReport(pts, max_dimension=2, analyze_manifolds=False)
    assert all(r.get("torsion", {}) == {} for r in off.results)
    assert "Integer Torsion Report" not in off.to_markdown()

    on = RipsFiltrationReport(pts, max_dimension=2, analyze_manifolds=False,
                              compute_torsion=True, n_samples=6)
    assert all(r["torsion"] == {} for r in on.results)         # circle is torsion-free
    assert "Integer Torsion Report" in on.to_markdown()
    # Betti table is unchanged by enabling torsion.
    assert _bagify(on.barcode) == _bagify(off.barcode)
