"""Tests for Delaunay-Rips and Delaunay-Cech filtration construction.

Overview:
    Verifies ``SimplicialComplex.from_delaunay_rips`` / ``from_delaunay_cech`` and
    the ``DelaunayRipsFiltrationReport`` / ``DelaunayCechFiltrationReport`` report
    classes: correctness of the restricted-Delaunay filtration values, and
    Python/Julia backend agreement for both the plain classmethods and the fused
    Julia filtration+persistence kernels. No prior test in this repo exercised
    either classmethod or report class at all.

Key Concepts:
    - **Delaunay-Rips**: the Delaunay triangulation's faces, each tagged with its
      longest edge (the Vietoris-Rips value), rather than the full Rips complex.
    - **Delaunay-Cech**: the same Delaunay faces, tagged with the radius of their
      own minimum enclosing ball (a Cech value) instead.
"""
import numpy as np
import pytest

from pysurgery.bridge.julia_bridge import julia_engine
from pysurgery.topology.complexes import SimplicialComplex
from pysurgery.topology.filtration_tools import (
    DelaunayCechFiltrationReport,
    DelaunayRipsFiltrationReport,
)


def test_delaunay_rips_circle():
    """30 points on a circle: Delaunay-Rips at a threshold spanning only adjacent
    edges (~0.209) but not the polygon's diagonals (>=1.34) should recover the
    circle's homology (Betti_0=1, Betti_1=1)."""
    t = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    points = np.column_stack([np.cos(t), np.sin(t)])

    sc = SimplicialComplex.from_delaunay_rips(points, threshold=0.22, max_dimension=1)
    assert sc.count_simplices(1) == 30
    cc = sc.chain_complex()
    h1_rank, _ = cc.homology(n=1)
    h0_rank, _ = cc.homology(n=0)
    assert h0_rank == 1
    assert h1_rank == 1


def test_delaunay_cech_circle():
    """Same circle, Delaunay-Cech: a 2-point face's miniball radius is exactly
    half its edge length, so half the Rips threshold recovers the same cycle."""
    t = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    points = np.column_stack([np.cos(t), np.sin(t)])

    sc = SimplicialComplex.from_delaunay_cech(points, threshold=0.11, max_dimension=1)
    assert sc.count_simplices(1) == 30
    cc = sc.chain_complex()
    h1_rank, _ = cc.homology(n=1)
    h0_rank, _ = cc.homology(n=0)
    assert h0_rank == 1
    assert h1_rank == 1


def test_delaunay_cech_edge_is_half_rips():
    """For every edge, the Cech value (miniball radius) is exactly half the Rips
    value (edge length) -- a direct check of the value formulas, not just
    downstream homology."""
    points = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    sc_rips = SimplicialComplex.from_delaunay_rips(points, max_dimension=1)
    sc_cech = SimplicialComplex.from_delaunay_cech(points, max_dimension=1)
    edges = [s for s in sc_rips.filtration if len(s) == 2]
    assert edges
    for edge in edges:
        assert sc_cech.filtration[edge] == pytest.approx(sc_rips.filtration[edge] / 2.0)


def test_delaunay_filtration_values_monotone_under_faces():
    """Appearance values must be monotone under taking faces: value(face) <=
    value(coface), for both Delaunay-Rips and Delaunay-Cech, on a nontrivial 3D
    cloud -- the property the persistence reduction relies on."""
    rng = np.random.default_rng(0)
    points = rng.random((25, 3))
    for method in ("from_delaunay_rips", "from_delaunay_cech"):
        sc = getattr(SimplicialComplex, method)(points, max_dimension=3)
        assert len(sc.filtration) > 0
        for simplex, value in sc.filtration.items():
            if len(simplex) <= 1:
                continue
            for i in range(len(simplex)):
                face = simplex[:i] + simplex[i + 1:]
                assert sc.filtration[face] <= value + 1e-9, (method, face, simplex)


def test_delaunay_rips_report_renders():
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    report = str(DelaunayRipsFiltrationReport(points, max_dimension=2))
    assert "(Method: Delaunay-Rips)" in report


def test_delaunay_cech_report_renders():
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    report = str(DelaunayCechFiltrationReport(points, max_dimension=2))
    assert "(Method: Delaunay-Cech)" in report


@pytest.mark.skipif(not julia_engine.available, reason="Julia backend not available")
class TestDelaunayFiltrationBackendConsistency:
    """Python vs Julia agreement for the Delaunay-Rips/Cech classmethods and the
    fused filtration-report kernels."""

    def test_from_delaunay_rips_consistency(self):
        rng = np.random.default_rng(1)
        points = rng.random((20, 3))
        sc_py = SimplicialComplex.from_delaunay_rips(points, max_dimension=3, backend="python")
        sc_jl = SimplicialComplex.from_delaunay_rips(points, max_dimension=3, backend="julia")
        assert sorted(sc_py.simplices) == sorted(sc_jl.simplices)
        for s in sc_py.filtration:
            assert sc_py.filtration[s] == pytest.approx(sc_jl.filtration[s], abs=1e-9)

    def test_from_delaunay_rips_consistency_with_threshold(self):
        rng = np.random.default_rng(1)
        points = rng.random((20, 3))
        sc_py = SimplicialComplex.from_delaunay_rips(points, threshold=0.5, max_dimension=3, backend="python")
        sc_jl = SimplicialComplex.from_delaunay_rips(points, threshold=0.5, max_dimension=3, backend="julia")
        assert sorted(sc_py.simplices) == sorted(sc_jl.simplices)

    def test_from_delaunay_cech_consistency(self):
        rng = np.random.default_rng(2)
        points = rng.random((20, 3))
        sc_py = SimplicialComplex.from_delaunay_cech(points, max_dimension=3, backend="python")
        sc_jl = SimplicialComplex.from_delaunay_cech(points, max_dimension=3, backend="julia")
        assert sorted(sc_py.simplices) == sorted(sc_jl.simplices)
        for s in sc_py.filtration:
            assert sc_py.filtration[s] == pytest.approx(sc_jl.filtration[s], abs=1e-7)

    def test_from_delaunay_cech_consistency_with_threshold(self):
        rng = np.random.default_rng(2)
        points = rng.random((20, 3))
        sc_py = SimplicialComplex.from_delaunay_cech(points, threshold=0.3, max_dimension=3, backend="python")
        sc_jl = SimplicialComplex.from_delaunay_cech(points, threshold=0.3, max_dimension=3, backend="julia")
        assert sorted(sc_py.simplices) == sorted(sc_jl.simplices)

    @staticmethod
    def _norm_barcode(barcode):
        return sorted(
            (d, round(b, 6), round(dd, 6) if dd != float("inf") else dd)
            for d, b, dd in barcode
        )

    def test_delaunay_rips_fused_filtration_matches_staged(self):
        """The fused Julia build+reduce path (enough points to trigger it in
        DelaunayRipsFiltrationReport) must give the same barcode as the staged
        Python path."""
        rng = np.random.default_rng(3)
        points = rng.random((300, 3))
        rep_py = DelaunayRipsFiltrationReport(points, max_dimension=2, backend="python", n_samples=10)
        rep_jl = DelaunayRipsFiltrationReport(points, max_dimension=2, backend="julia", n_samples=10)
        assert self._norm_barcode(rep_py.barcode) == self._norm_barcode(rep_jl.barcode)

    def test_delaunay_cech_fused_filtration_matches_staged(self):
        rng = np.random.default_rng(4)
        points = rng.random((300, 3))
        rep_py = DelaunayCechFiltrationReport(points, max_dimension=2, backend="python", n_samples=10)
        rep_jl = DelaunayCechFiltrationReport(points, max_dimension=2, backend="julia", n_samples=10)
        assert self._norm_barcode(rep_py.barcode) == self._norm_barcode(rep_jl.barcode)
