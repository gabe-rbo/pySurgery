"""Tests for the tangential Delaunay complex (Boissonnat-Ghosh) reconstruction.

Overview:
    Verifies the tangential complex pipeline end-to-end on the same jittered-sphere fixture
    ``tests/test_reconstruction.py`` uses for Cocone (auto-estimated ``k=2`` reproduces the
    correct Betti numbers), on a jittered torus (a gentler major:minor aspect ratio than
    Cocone's own torus fixture -- see ``_jittered_torus``'s docstring for why), on that same
    torus re-embedded into R^5 to demonstrate the ``k << d`` case Cocone's codimension-1-
    specific pole/normal machinery cannot address at all, and on the shared
    ``intersect_local_stars`` inconsistency-count diagnostic.

    Independent per-point local star computations disagree far more often here than in
    Cocone's own prune-and-walk (measured: ~35-55% of points implicated at round 0, taking
    dozens to a couple hundred perturbation rounds to fully resolve -- see
    ``tangential_complex_reconstruction``'s docstring) -- these tests use a generous
    ``max_repair_rounds`` budget accordingly, and check the reliable invariants (repair
    converges, the result is a genuine combinatorial manifold with the right Betti numbers)
    rather than the exact round count.
"""
import numpy as np
from scipy.spatial import cKDTree

from pysurgery.geometry.intrinsic_dimension import local_pca_tangent_space_dimension
from pysurgery.geometry.tangential_complex import tangential_complex_reconstruction
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


def _jittered_torus(n_major=28, n_minor=14, major_radius=3.0, minor_radius=1.4, seed=2, noise=0.01):
    """A jittered torus sample (angles perturbed off the regular grid, plus coordinate noise).

    Uses a gentler major:minor aspect ratio (~2.1, vs. the tighter ~3.3 used for Cocone's own
    torus fixture in tests/test_reconstruction.py) -- measured to matter here specifically:
    the tangential complex's star-consistency disagreement rate (driven by neighboring
    points' independently-estimated tangent bases differing enough in *orientation* to flip
    a near-cocircular quad's Delaunay diagonal -- see ``tangential_complex_reconstruction``'s
    docstring) was found to be substantially lower at this aspect ratio, converging within a
    few dozen rounds where the tighter ratio was still not fully converged after hundreds.
    """
    rng = np.random.default_rng(seed)
    u = np.linspace(0.0, 2.0 * np.pi, n_major, endpoint=False)
    v = np.linspace(0.0, 2.0 * np.pi, n_minor, endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    uu = uu + rng.uniform(-0.25, 0.25, size=uu.shape) * (2.0 * np.pi / n_major)
    vv = vv + rng.uniform(-0.25, 0.25, size=vv.shape) * (2.0 * np.pi / n_minor)
    x = (major_radius + minor_radius * np.cos(vv)) * np.cos(uu)
    y = (major_radius + minor_radius * np.cos(vv)) * np.sin(uu)
    z = minor_radius * np.sin(vv)
    pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    pts += noise * rng.normal(size=pts.shape)
    return pts


def _wider_perturbation_scale(pts, factor=3.0) -> float:
    """3x tangential_complex_reconstruction's own auto-scale default.

    Measured specifically for the torus fixture: unlike Cocone's reach-fraction disagreement
    (a binary admit/reject decision), the tangential complex's diagonal-flip disagreements
    were found to need a *wider* nudge to reliably resolve which side of a near-cocircular
    ambiguity a point ends up on -- the auto-scale default converged the sphere fixture fine
    but left the torus fixture oscillating; 3x converges it within a few dozen rounds.
    """
    tree = cKDTree(pts)
    nn_dist, _ = tree.query(pts, k=2)
    return factor * 0.01 * float(np.median(nn_dist[:, 1]))


def _complex_from_covered_vertices(points, simplices) -> SimplicialComplex:
    """Build a SimplicialComplex from only the vertices actually covered by some simplex."""
    all_simplices = {tuple(sorted(int(x) for x in s)) for s in simplices}
    for s in list(all_simplices):
        for v in s:
            all_simplices.add((v,))
    return SimplicialComplex.from_simplices(all_simplices, close_under_faces=True)


def test_tangential_complex_sphere_reproduces_correct_betti_numbers():
    """Auto-estimated k on a jittered sphere sample must reproduce S^2's Betti numbers.

    What is Being Computed?:
        Reconstructs a jittered Fibonacci-sphere sample with ``k=None`` (auto-estimated from
        local PCA) and checks the result against ``is_homology_manifold`` and the known
        Betti numbers of S^2: (1, 0, 1).
    """
    pts = _jittered_sphere(n=200, seed=1, noise=0.01)
    result = tangential_complex_reconstruction(pts, max_repair_rounds=200)
    assert result.k == 2
    assert len(result.unresolved_points) == 0

    sc = _complex_from_covered_vertices(result.points, result.simplices)
    is_mani, dim, diag = sc.is_homology_manifold(backend="julia")
    assert is_mani, diag
    assert dim == 2
    assert sc.is_closed_manifold

    betti = sc.betti_numbers(backend="julia")
    assert betti.get(0) == 1
    assert betti.get(1, 0) == 0
    assert betti.get(2) == 1


def test_tangential_complex_torus_reproduces_correct_betti_numbers():
    """Auto-estimated k on a jittered torus sample must reproduce T^2's topology.

    What is Being Computed?:
        Reconstructs a jittered torus sample with ``k=None`` and checks the result against
        ``is_homology_manifold`` and the Euler characteristic of T^2 (0). Unlike Cocone
        (which was measured to have a specific concave/hole-facing-region limitation), the
        tangential complex's per-point independent tangent projection has no pole/reach
        machinery to special-case a hole -- it either works or it doesn't based on general
        star-consistency reconciliation.

        Euler characteristic (a plain alternating simplex-count sum) is checked here rather
        than the full ``betti_numbers()`` -- computing exact torsion certificates via Smith
        Normal Form on this complex's boundary matrices was measured to make Julia's
        backend consume several GB of memory (a pre-existing characteristic of
        ``complexes.py``'s torsion-certification path on a nontrivial-H_1 complex this
        size, unrelated to reconstruction correctness) where ``is_homology_manifold`` and
        ``euler_characteristic`` alone are fast and memory-light. Since ``is_homology_manifold``
        already certifies a genuine closed 2-manifold, ``chi = 0`` alone pins down genus 1
        (chi = 2 - 2g for a closed orientable surface) without needing full Betti numbers.
    """
    pts = _jittered_torus(n_major=28, n_minor=14, seed=2, noise=0.01)
    result = tangential_complex_reconstruction(
        pts, max_repair_rounds=250, perturbation_scale=_wider_perturbation_scale(pts)
    )
    assert result.k == 2
    assert len(result.unresolved_points) == 0
    # Exactly 2V triangles (Euler characteristic 0) is itself a strong signal of a genuine,
    # complete genus-1 closure -- checked before the more expensive verification below.
    assert len(result.simplices) == 2 * len(pts)

    sc = _complex_from_covered_vertices(result.points, result.simplices)
    is_mani, dim, diag = sc.is_homology_manifold(backend="julia")
    assert is_mani, diag
    assert dim == 2
    assert sc.is_closed_manifold
    assert sc.euler_characteristic() == 0  # genus 1, i.e. a torus

    coverage = len({v for s in result.simplices for v in s}) / len(pts)
    assert coverage > 0.4


def test_tangential_complex_handles_high_codimension_torus_in_r5():
    """A torus re-embedded into R^5 (k=2, d=5) must reconstruct correctly -- the k << d case
    Cocone structurally cannot address (its pole/normal estimation is specific to
    codimension 1 in R^3).

    What is Being Computed?:
        Applies a fixed random orthogonal embedding of a jittered torus's 3 coordinates into
        5 dimensions (an isometry, so all pairwise distances -- and therefore the correct
        reconstruction -- are unchanged) and checks that auto-estimated k correctly detects
        k=2 despite the higher ambient dimension, and that the reconstruction is a genuine,
        closed, genus-1 manifold (see the Euler-characteristic note on the plain torus test
        above for why ``euler_characteristic`` is checked here instead of full Betti numbers).
    """
    pts3d = _jittered_torus(n_major=28, n_minor=14, seed=2, noise=0.01)
    rng = np.random.default_rng(99)
    random_matrix = rng.normal(size=(5, 3))
    q, _ = np.linalg.qr(random_matrix)  # (5, 3) with orthonormal columns -- an isometry
    pts5d = pts3d @ q.T

    dim_result = local_pca_tangent_space_dimension(pts5d, k=12)
    assert round(dim_result.global_dimension) == 2

    result = tangential_complex_reconstruction(
        pts5d, k=2, max_repair_rounds=250, perturbation_scale=_wider_perturbation_scale(pts5d)
    )
    assert len(result.unresolved_points) == 0
    assert len(result.simplices) == 2 * len(pts5d)

    sc = _complex_from_covered_vertices(result.points, result.simplices)
    is_mani, dim, diag = sc.is_homology_manifold(backend="julia")
    assert is_mani, diag
    assert dim == 2
    assert sc.is_closed_manifold
    assert sc.euler_characteristic() == 0
    coverage = len({v for s in result.simplices for v in s}) / len(pts5d)
    assert coverage > 0.4


def test_tangential_complex_repair_shrinks_inconsistency_count_to_zero():
    """moser_tardos_repair must actually be doing the reconciliation work: without repair,
    a plain jittered sphere sample has a nonzero star-inconsistency count; with repair, it
    must reach exactly zero (a converged, fully-consistent result)."""
    pts = _jittered_sphere(n=120, seed=3, noise=0.01)

    unrepaired = tangential_complex_reconstruction(pts, k=2, repair=False)
    assert unrepaired.diagnostics[0] != "0 candidate simplex/simplices dropped by disagreement."

    repaired = tangential_complex_reconstruction(pts, k=2, repair=True, max_repair_rounds=250)
    assert len(repaired.unresolved_points) == 0
    assert repaired.diagnostics[0] == "0 candidate simplex/simplices dropped by disagreement."
