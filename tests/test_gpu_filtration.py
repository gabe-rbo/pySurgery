"""Tests for the dual-alpha filtration report (DualAlphaFiltrationReport).

Overview:
    The dual-alpha report must produce the same filtration as the Delaunay-based
    ``AlphaFiltrationReport`` wherever both can run -- the identical barcode, bar
    for bar -- and keep working where Delaunay cannot (high ambient dimension).
    Point clouds are tiny, so the tests are light on memory.
"""
import warnings

import numpy as np
import pytest

pytest.importorskip("torch")

from pysurgery.gpu.dual_alpha import connectivity_radius  # noqa: E402
from pysurgery.topology.filtration_tools import (  # noqa: E402
    AlphaFiltrationReport,
    DualAlphaFiltrationReport,
    FiltrationReport,
)


def _bars(report, cap):
    out = []
    for d, b, e in report.barcode:
        if b > cap or e - b <= 1e-12:
            continue
        out.append((d, round(b, 9), round(e, 9) if np.isfinite(e) else np.inf))
    return sorted(out)


def _circle(n=24):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[np.cos(t), np.sin(t)]


@pytest.mark.parametrize("points", [
    _circle(24),
    np.random.default_rng(0).uniform(size=(30, 2)),
    np.random.default_rng(1).uniform(size=(22, 3)),
])
def test_barcode_is_the_delaunay_alpha_barcode(points):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = AlphaFiltrationReport(points, max_dimension=points.shape[1], backend="python")
        cap = max(b for _d, b, _e in ref.barcode) * 1.5 + 1.0     # past every birth
        dual = DualAlphaFiltrationReport(points, eps_max=cap, max_dimension=points.shape[1],
                                         backend="python", max_simplices=None)
    assert _bars(dual, cap) == _bars(ref, cap)
    assert dual.radius_cap == cap


def test_factory_mode_default_cap_and_gpu_torsion():
    P = _circle(20)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = FiltrationReport(P, mode="dual_alpha", compute_torsion=True, backend="gpu")
    assert r.radius_cap == pytest.approx(3.0 * connectivity_radius(P))
    assert r.results[-1]["bettis"] == {0: 1, 1: 1} and r.results[-1]["torsion"] == {}
    assert r.dual_alpha_result is not None and "Dual-Alpha" in str(r)


def test_runs_where_delaunay_cannot():
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.normal(size=(40, 2)))
    P = _circle(30) @ Q.T + 1e-4 * rng.normal(size=(30, 40))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = DualAlphaFiltrationReport(P, eps_max=0.6, max_dimension=2, backend="python")
    b = rep.results[-1]["bettis"]
    assert b.get(0) == 1 and b.get(1) == 1
