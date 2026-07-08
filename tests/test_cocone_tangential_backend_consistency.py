"""Backend-consistency tests for the Cocone/Tight-Cocone and tangential Delaunay complex
Julia kernels: ``estimate_voronoi_poles``, ``cocone_filter``, ``prune_and_walk``, and the
``k == 2`` tangential local-star construction.

Overview:
    Mirrors ``tests/test_backend_consistency_comprehensive.py``'s convention (one method per
    kernel, ``backend="python"`` vs ``backend="julia"``) but lives separately since these
    kernels are exercised through ``pysurgery.geometry`` functions rather than
    ``SimplicialComplex`` classmethods directly, and the last one (tangential local stars) has
    a genuinely different correctness bar -- see ``TestTangentialLocalStarsConsistency``'s
    docstring for why exact parity is not asserted there.
"""
import numpy as np
import pytest
from scipy.spatial import Delaunay

from pysurgery.bridge.julia_bridge import julia_engine
from pysurgery.geometry.reconstruction import (
    _DEFAULT_REACH_FRACTION,
    _DEFAULT_THETA,
    _sentinel_padded_points,
    cocone_filter,
    estimate_voronoi_poles,
    prune_and_walk,
)
from pysurgery.geometry.tangential_complex import (
    _compute_all_local_stars,
    tangential_complex_reconstruction,
)
from pysurgery.geometry.intrinsic_dimension import local_pca_tangent_basis
from pysurgery.topology.complexes import SimplicialComplex


def _jittered_sphere(n=200, seed=1, radius=1.0, noise=0.01):
    """A deterministic Fibonacci-lattice sphere sample, jittered to avoid exact symmetry."""
    rng = np.random.default_rng(seed)
    i = np.arange(n, dtype=np.float64)
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    z = 1.0 - 2.0 * (i + 0.5) / n
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    golden_angle = 2.0 * np.pi * (1.0 - 1.0 / golden_ratio)
    psi = golden_angle * i
    x = np.sin(theta) * np.cos(psi)
    y = np.sin(theta) * np.sin(psi)
    pts = radius * np.column_stack([x, y, z])
    pts += noise * rng.normal(size=pts.shape)
    return pts


def _complex_from_covered_vertices(points, simplices) -> SimplicialComplex:
    """Build a SimplicialComplex from only the vertices actually covered by some simplex."""
    all_simplices = {tuple(sorted(int(x) for x in s)) for s in simplices}
    for s in list(all_simplices):
        for v in s:
            all_simplices.add((v,))
    return SimplicialComplex.from_simplices(all_simplices, close_under_faces=True)


def _nan_close(a, b, atol=1e-7):
    return bool(np.all(np.isclose(a, b, atol=atol, equal_nan=True) | (np.isnan(a) & np.isnan(b))))


@pytest.mark.skipif(not julia_engine.available, reason="Julia backend not available")
class TestEstimateVoronoiPolesConsistency:
    """Verify the per-point argmax-distance pole selection numeric step."""

    def test_sphere_poles_match(self):
        pts = _jittered_sphere(n=60, seed=0, noise=0.01)
        res_py = estimate_voronoi_poles(pts, backend="python")
        res_jl = estimate_voronoi_poles(pts, backend="julia")

        assert _nan_close(res_py.positive_pole, res_jl.positive_pole)
        assert _nan_close(res_py.negative_pole, res_jl.negative_pole)
        assert np.array_equal(res_py.has_negative_pole, res_jl.has_negative_pole)
        assert _nan_close(res_py.normal, res_jl.normal)
        assert _nan_close(res_py.pole_radius, res_jl.pole_radius)
        assert res_py.diagnostics == res_jl.diagnostics

    def test_fewer_than_four_points(self):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        res_py = estimate_voronoi_poles(pts, backend="python")
        res_jl = estimate_voronoi_poles(pts, backend="julia")
        assert res_py.diagnostics == res_jl.diagnostics


@pytest.mark.skipif(not julia_engine.available, reason="Julia backend not available")
class TestCoconeFilterConsistency:
    """Verify the fused facet-map + circumcenter + angle/reach filter kernel."""

    def test_sphere_candidates_match(self):
        pts = _jittered_sphere(n=80, seed=1, noise=0.01)
        poles = estimate_voronoi_poles(pts, backend="python")
        combined = _sentinel_padded_points(pts, 20.0, None)
        tets = Delaunay(combined, qhull_options="QJ").simplices

        res_py = cocone_filter(
            combined, tets, poles.normal, poles.pole_radius,
            theta=_DEFAULT_THETA, reach_fraction=_DEFAULT_REACH_FRACTION, backend="python",
        )
        res_jl = cocone_filter(
            combined, tets, poles.normal, poles.pole_radius,
            theta=_DEFAULT_THETA, reach_fraction=_DEFAULT_REACH_FRACTION, backend="julia",
        )
        assert res_py.n_candidates == res_jl.n_candidates
        # Julia returns lexicographically sorted triangles, matching
        # sorted(facet_to_tets.keys()) exactly -- order-identical, not just set-identical.
        assert res_py.surviving_triangles == res_jl.surviving_triangles

    def test_empty_tetrahedra(self):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        normals = np.tile([0.0, 0.0, 1.0], (4, 1))
        radius = np.ones(4)
        empty_tets = np.zeros((0, 4), dtype=np.int64)
        res_py = cocone_filter(pts, empty_tets, normals, radius, backend="python")
        res_jl = cocone_filter(pts, empty_tets, normals, radius, backend="julia")
        assert res_py.surviving_triangles == res_jl.surviving_triangles == []
        assert res_py.n_candidates == res_jl.n_candidates == 0


@pytest.mark.skipif(not julia_engine.available, reason="Julia backend not available")
class TestPruneAndWalkConsistency:
    """Verify the per-vertex fast-path check + tangent-plane angular walk kernel."""

    def test_spurious_branch_fixture(self):
        """The exact hand-built fixture from
        test_reconstruction.py::test_prune_and_walk_rejects_spurious_triangle_deterministically."""
        v = np.array([0.0, 0.0, 0.0])
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        c = np.array([-1.0, 0.0, 0.0])
        d = np.array([0.0, -1.0, 0.0])
        angle_e = np.deg2rad(315.0)
        e = np.array([np.cos(angle_e), np.sin(angle_e), 0.0])
        points = np.array([v, a, b, c, d, e])
        normals = np.zeros((6, 3))
        normals[0] = np.array([0.0, 0.0, 1.0])
        candidate_triangles = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1), (0, 1, 5)]

        res_py = prune_and_walk(points, candidate_triangles, normals, backend="python")
        res_jl = prune_and_walk(points, candidate_triangles, normals, backend="julia")
        assert res_py.per_vertex_local_simplices[0] == res_jl.per_vertex_local_simplices[0]
        assert 0 not in res_jl.unresolved_vertices

    def test_pinch_point_both_backends_flag_unresolved(self):
        """A vertex whose link graph is two disjoint edges (a genuine pinch/disconnected
        defect) must be flagged unresolved by both backends, not silently dropped."""
        points = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0],
        ])
        normals = np.zeros((5, 3))
        normals[0] = np.array([0.0, 0.0, 1.0])
        candidate_triangles = [(0, 1, 2), (0, 3, 4)]

        res_py = prune_and_walk(points, candidate_triangles, normals, backend="python")
        res_jl = prune_and_walk(points, candidate_triangles, normals, backend="julia")
        assert 0 in res_py.unresolved_vertices
        assert 0 in res_jl.unresolved_vertices
        assert set(res_py.final_triangles) == set(res_jl.final_triangles)

    def test_pipeline_final_triangles_match(self):
        pts = _jittered_sphere(n=70, seed=2, noise=0.02)
        poles = estimate_voronoi_poles(pts, backend="python")
        combined = _sentinel_padded_points(pts, 20.0, None)
        tets = Delaunay(combined, qhull_options="QJ").simplices
        filt = cocone_filter(
            combined, tets, poles.normal, poles.pole_radius,
            theta=_DEFAULT_THETA, reach_fraction=_DEFAULT_REACH_FRACTION, backend="python",
        )

        res_py = prune_and_walk(combined, filt.surviving_triangles, poles.normal, backend="python")
        res_jl = prune_and_walk(combined, filt.surviving_triangles, poles.normal, backend="julia")
        assert set(res_py.final_triangles) == set(res_jl.final_triangles)
        assert set(res_py.unresolved_vertices) == set(res_jl.unresolved_vertices)
        assert res_py.per_vertex_local_simplices == res_jl.per_vertex_local_simplices


@pytest.mark.skipif(not julia_engine.available, reason="Julia backend not available")
class TestTangentialLocalStarsConsistency:
    """Verify the k=2 DelaunayTriangulation.jl-backed local-star construction.

    Note: exact star-for-star parity is NOT the correctness bar here (unlike the three test
    classes above) -- see ``tangential_complex_reconstruction``'s docstring: two different
    triangulation libraries (Qhull vs. DelaunayTriangulation.jl) choosing different diagonals
    on a near-cocircular local neighborhood is expected, routine behavior that
    ``intersect_local_stars``/``moser_tardos_repair`` already reconcile, not a defect. These
    tests check (a) exact parity on an unambiguous (non-cocircular) configuration, (b)
    degenerate-neighborhood handling, and (c) that the end-to-end pipeline still converges to
    a valid manifold using the Julia path -- not bit-for-bit star agreement on ambiguous
    inputs.
    """

    def test_unambiguous_neighborhood_matches_exactly(self):
        points = np.array([
            [0.0, 0.0, 0.0], [2.0, 0.0, 0.1], [0.0, 2.0, -0.1],
            [-2.0, 0.0, 0.05], [0.0, -2.0, -0.05], [1.3, 1.3, 0.0],
        ])
        basis_res = local_pca_tangent_basis(points, k=2, neighborhood_size=5)
        stars_py, failed_py = _compute_all_local_stars(points, basis_res.bases, 5, backend="python")
        stars_jl, failed_jl = _compute_all_local_stars(points, basis_res.bases, 5, backend="julia")
        assert failed_py == failed_jl == []
        for i in range(len(points)):
            assert stars_py[i] == stars_jl[i]

    def test_degenerate_neighborhood_flagged(self):
        flat_idx = np.array([0, 1], dtype=np.int64)  # only 2 points, need >= k+1 = 3
        flat_coords = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        offsets = np.array([0, 2], dtype=np.int64)
        star_flat, star_offsets, ok = julia_engine.compute_tangential_local_stars_2d(
            flat_idx, flat_coords, offsets
        )
        assert not bool(ok[0])
        assert int(star_offsets[-1]) == 0

    def test_end_to_end_sphere_converges_with_julia_backend(self):
        # Same fixture/params as test_tangential_complex.py's own proven
        # test_tangential_complex_sphere_reproduces_correct_betti_numbers, with backend="julia".
        pts = _jittered_sphere(n=200, seed=1, noise=0.01)
        result = tangential_complex_reconstruction(pts, max_repair_rounds=200, backend="julia")
        assert result.k == 2
        assert len(result.unresolved_points) == 0

        sc = _complex_from_covered_vertices(result.points, result.simplices)
        # is_homology_manifold + euler_characteristic, not betti_numbers: SNF-based torsion
        # certification is unnecessary here and measured elsewhere (test_tangential_complex.py)
        # to be far more memory-hungry for no extra information on a manifold already known
        # closed by construction.
        is_mani, dim, diag = sc.is_homology_manifold()
        assert is_mani, diag
        assert dim == 2
        assert sc.is_closed_manifold
        assert sc.euler_characteristic() == 2  # sphere: chi = 2 - 2g, g = 0
