"""Cocone / Tight Cocone surface reconstruction for point clouds in R^3.

Overview:
    Alpha/Rips-style filtrations recover the correct homotopy type of a sampled surface, but
    not manifold structure -- they grow tetrahedra wherever a noisy sample's thickness allows
    it, so ``SimplicialComplex.is_homology_manifold()`` fails almost everywhere except at the
    trivial (disconnected-points) end of the filtration. This module builds a *single*,
    manifold-*certified* triangulated surface instead, following the Cocone algorithm
    (Amenta, Choi, Dey, Leekha) with an optional Tight Cocone (Dey-Goswami) closing pass.
    Reconstruction (with the default ``repair=True``) is reliable on genus-0 (sphere-like)
    surfaces; on a torus -- the motivating case, and directly re-verified against the real
    ``torus_a.csv`` from that experiment -- the concave, hole-facing region is measurably
    harder and repair can converge to a result that, while genuinely defect-free everywhere
    it exists, is fragmented into several disjoint manifold-with-boundary patches rather than
    one closed torus (see ``cocone_reconstruction``'s docstring for what was measured and why,
    and what fixing it properly would require).

Key Concepts:
    - **Poles**: per-point Voronoi-cell vertices farthest from the sample point, in each of
      the two roughly-opposite directions -- the classical Amenta-Bern normal estimator
      (``estimate_voronoi_poles``).
    - **Cocone filter**: keeps a candidate Delaunay triangle only if its own normal direction
      agrees, within a fixed angle, with every one of its vertices' pole-estimated normals,
      *and* its dual Voronoi edge reaches far enough toward the pole (``cocone_filter``).
    - **Prune-and-walk**: the step that actually *forces* a combinatorial 2-manifold --
      per vertex, the surviving candidate triangles must form a single simple cycle *or open
      path* around that vertex (reusing ``pysurgery.geometry.perturbation.is_single_cycle``/
      ``is_single_path_or_cycle`` as a fast path -- a path is a legitimate boundary vertex,
      not a defect); where they don't, a deterministic tangential angular walk keeps exactly
      one consistent umbrella fan (``prune_and_walk``).
    - **Tight Cocone**: an optional, best-effort closing pass that labels every Delaunay
      tetrahedron inside/outside via a wall-respecting flood fill from the convex hull
      (``tight_cocone_close``) -- see that function's docstring for a real limitation on
      point clouds sampled only on a surface (no interior points).

Common Workflows:
    1. **End-to-end reconstruction** -> ``cocone_reconstruction(points)``, or the
       ``SimplicialComplex.from_cocone`` classmethod that wraps it.
    2. **Verification** -> the resulting triangles feed straight into the existing
       ``SimplicialComplex.is_homology_manifold()`` -- no new verifier is needed for these
       2-dimensional reconstructions (see the docstring note on that method for why it is
       already exact, not merely a homology-manifold check, at dimension <= 2). Always check
       the returned Betti numbers against what's expected, not just manifold-ness alone (see
       the genus caveat above).
"""

from __future__ import annotations

import itertools
import warnings
from collections import defaultdict, deque
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.spatial import Delaunay, Voronoi

from ..bridge.julia_bridge import julia_engine
from .perturbation import (
    intersect_local_stars,
    is_single_cycle,
    is_single_path_or_cycle,
    moser_tardos_repair,
)

# Amenta-Choi-Dey-Leekha's original angle-only cocone criterion recommends theta in
# (0, pi/8]. That guidance predates the pole-reach condition cocone_filter also enforces
# here (see its docstring): once the reach check is doing real discrimination work, a
# substantially wider angle tolerance measurably reduces prune-and-walk dead ends and
# cross-vertex disagreement without re-admitting the shallow "just under the surface"
# facets the reach check exists to reject. Empirically tuned (see tests/test_reconstruction.py
# and the parameter sweeps referenced in cocone_reconstruction's docstring) against jittered
# sphere and torus fixtures, and cross-checked against the real torus_a.csv from the
# motivating FutureLab experiment.
_DEFAULT_THETA = np.deg2rad(40.0)
# A stricter reach fraction (e.g. 0.85, which is a better fit for the torus's concave inner
# region specifically -- see cocone_reconstruction's docstring) was measured to make
# moser_tardos_repair fail to converge within a few hundred rounds on plain, unremarkable
# jittered spheres at moderate sample sizes (n in the 80-150 range): sparser sampling makes a
# strict reach threshold reject legitimate near-surface facets outright, a candidate-
# starvation problem no amount of point perturbation can fix (it is not about *where* the
# points are, only about how many of them there are). 0.6 was cross-checked against six
# jittered-sphere fixtures spanning n=80..250 and reliably converges to a fully *closed*
# manifold within a handful of rounds (often zero) on all of them, unlike 0.85 (which failed
# to converge at all on half of them). Callers reconstructing a surface with a persistent
# concave region relative to their own sampling density (the torus case) should still pass a
# higher ``reach_fraction`` explicitly, as cocone_reconstruction's docstring recommends.
_DEFAULT_REACH_FRACTION = 0.6


def _fibonacci_sphere(n: int) -> np.ndarray:
    """Deterministic, roughly-uniform points on the unit sphere (no randomness).

    What is Being Computed?:
        A Fibonacci-lattice point set on ``S^2`` -- used to place bounding sentinels around a
        point cloud before computing its Voronoi diagram, so every original point's Voronoi
        cell is bounded (see ``estimate_voronoi_poles``). Determinism (as opposed to a random
        sphere sample) keeps reconstruction reproducible across runs.

    Args:
        n: Number of points to place.

    Returns:
        np.ndarray: Shape ``(n, 3)`` array of unit vectors.
    """
    i = np.arange(n, dtype=np.float64)
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    z = 1.0 - 2.0 * (i + 0.5) / n
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    golden_angle = 2.0 * np.pi * (1.0 - 1.0 / golden_ratio)
    psi = golden_angle * i
    x = np.sin(theta) * np.cos(psi)
    y = np.sin(theta) * np.sin(psi)
    return np.column_stack([x, y, z])


def _sentinel_padded_points(
    points: np.ndarray, bounding_radius_factor: float, n_sentinels: Optional[int]
) -> np.ndarray:
    """Original points followed by deterministic bounding sentinels (original points first).

    Overview:
        Both ``estimate_voronoi_poles`` (a Voronoi diagram) and ``cocone_filter`` (a Delaunay
        tetrahedralization, whose tetrahedra circumcenters are exactly that Voronoi diagram's
        vertices, by duality) must be built from the *same* augmented point set -- otherwise
        a facet's tetrahedra circumcenters (computed on one triangulation) are not comparable
        to the pole radii they are checked against (computed on a different one), which was
        measured to produce reach ratios up to ~24x, an artifact of the mismatch rather than
        genuine geometry. This helper is the single source of that augmented point set.

    Args:
        points: (N, 3) array of the original point coordinates.
        bounding_radius_factor: Sentinel sphere radius, as a multiple of the data's spread.
        n_sentinels: Number of sentinels; ``None`` defaults to ``max(16, 24)``.

    Returns:
        np.ndarray: Shape ``(N + S, 3)``; rows ``[0, N)`` are the original points, unchanged
        and in their original order; rows ``[N, N + S)`` are the sentinels.
    """
    pts = np.asarray(points, dtype=np.float64)
    centroid = pts.mean(axis=0)
    spread = float(np.max(np.linalg.norm(pts - centroid, axis=1))) if pts.shape[0] else 1.0
    spread = max(spread, 1e-9)
    radius = bounding_radius_factor * spread
    if n_sentinels is None:
        n_sentinels = max(4 * (pts.shape[1] + 1), 24)
    sentinels = centroid + radius * _fibonacci_sphere(n_sentinels)
    return np.vstack([pts, sentinels])


class PoleEstimationResult(BaseModel):
    """Per-point Voronoi pole and normal estimates.

    Overview:
        The Amenta-Bern observation is that a dense sample's Voronoi cells stretch along the
        surface normal; the farthest cell vertex ("positive pole") in each direction gives a
        normal estimate essentially for free from the Voronoi diagram alone.

    Attributes:
        positive_pole (np.ndarray): Shape ``(N, 3)``; the farthest Voronoi-cell vertex from
            each point. Rows are ``nan`` where no pole could be found (see ``diagnostics``).
        negative_pole (np.ndarray): Shape ``(N, 3)``; the farthest cell vertex on the
            opposite side of the tangent plane, where one exists.
        has_negative_pole (np.ndarray): Shape ``(N,)`` bool; whether ``negative_pole[i]`` is
            valid.
        normal (np.ndarray): Shape ``(N, 3)``; unit vector from each point to its positive
            pole. Zero where undefined.
        pole_radius (np.ndarray): Shape ``(N,)``; distance to the positive pole (a local
            feature-size proxy).
        diagnostics (list[str]): Human-readable notes on any point whose pole was degenerate
            or unbounded even after sentinel padding.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    positive_pole: np.ndarray
    negative_pole: np.ndarray
    has_negative_pole: np.ndarray
    normal: np.ndarray
    pole_radius: np.ndarray
    diagnostics: list[str] = Field(default_factory=list)


def estimate_voronoi_poles(
    points: np.ndarray,
    *,
    bounding_radius_factor: float = 20.0,
    n_sentinels: Optional[int] = None,
    backend: str = "auto",
) -> PoleEstimationResult:
    """Estimate per-point surface normals via Voronoi poles (Amenta-Bern).

    What is Being Computed?:
        For each sample point, the farthest vertex of its Voronoi cell in each of two
        roughly-opposite directions ("poles"), giving a normal estimate from the Voronoi
        diagram alone -- no separate normal-estimation pass is needed.

    Algorithm:
        1. Place ``n_sentinels`` points deterministically (``_fibonacci_sphere``, no RNG) on
           a sphere of radius ``bounding_radius_factor`` times the data's spread around its
           centroid, so every original point's Voronoi cell is bounded -- real surface
           samples generically have many points on their own convex hull (any point on the
           "outside" of a non-convex closed surface), so this is handled uniformly rather
           than as a rare special case.
        2. Compute ``scipy.spatial.Voronoi`` of the combined (points + sentinels) set;
           sentinels never appear in the returned poles, mirroring
           ``SimplicialComplex.from_crust_algorithm``'s "combine, then filter back to the
           original range" idiom.
        3. Per point, the positive pole is the farthest cell vertex; the normal is the unit
           vector towards it. The negative pole is the farthest cell vertex on the opposite
           side of the tangent plane through the point, orthogonal to that normal. Given the
           Voronoi diagram from step 2, this step is Julia-accelerated when available (the
           Voronoi diagram itself always stays in SciPy -- no Julia Voronoi implementation
           exists here).

    Args:
        points: (N, 3) array of point coordinates. Only 3D point clouds are supported (the
            Cocone/Tight-Cocone surface-reconstruction case).
        bounding_radius_factor: Sentinel sphere radius, as a multiple of the data's spread
            (max distance from centroid) about its centroid.
        n_sentinels: Number of bounding sentinels; defaults to ``max(16, 24)`` for 3D data.
        backend: ``"auto"`` (default; use Julia when available), ``"julia"`` (require it), or
            ``"python"`` (force the pure-NumPy path).

    Returns:
        PoleEstimationResult: Per-point poles, normals, and diagnostics.

    Raises:
        ValueError: If ``points`` is not an (N, 3) array.

    Use When:
        - As the first stage of Cocone/Tight Cocone reconstruction (``cocone_reconstruction``).
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("estimate_voronoi_poles currently supports (N, 3) point clouds only.")
    n = pts.shape[0]

    diagnostics: list[str] = []
    positive_pole = np.full((n, 3), np.nan)
    negative_pole = np.full((n, 3), np.nan)
    has_negative_pole = np.zeros(n, dtype=bool)
    normal = np.zeros((n, 3))
    pole_radius = np.zeros(n)

    if n < 4:
        diagnostics.append("Fewer than 4 points; Voronoi poles are undefined.")
        return PoleEstimationResult(
            positive_pole=positive_pole, negative_pole=negative_pole,
            has_negative_pole=has_negative_pole, normal=normal, pole_radius=pole_radius,
            diagnostics=diagnostics,
        )

    combined = _sentinel_padded_points(pts, bounding_radius_factor, n_sentinels)
    vor = Voronoi(combined, qhull_options="QJ")

    # Per-point (-1)-filtered cell vertex indices -- cheap ragged-list bookkeeping that stays
    # in Python; only the argmax-distance/opposite-side numeric loop below is Julia-eligible.
    cell_vertex_lists: list[list[int]] = []
    unbounded = np.zeros(n, dtype=bool)
    for i in range(n):
        region = vor.regions[vor.point_region[i]]
        vertex_ids = [v for v in region if v != -1]
        if region and -1 in region:
            unbounded[i] = True
        cell_vertex_lists.append(vertex_ids)

    backend_norm = str(backend).lower().strip()
    use_julia = (backend_norm == "julia") or (backend_norm == "auto" and julia_engine.available)
    status = None
    if use_julia:
        try:
            positive_pole, negative_pole, has_negative_pole, normal, pole_radius, status = (
                julia_engine.compute_voronoi_poles(pts, vor.vertices, cell_vertex_lists)
            )
        except Exception as e:
            if backend_norm == "julia":
                raise
            warnings.warn(f"Julia backend failed for estimate_voronoi_poles, falling back to Python: {e!r}")

    if status is None:
        status = np.zeros(n, dtype=np.int8)
        for i in range(n):
            vertex_ids = cell_vertex_lists[i]
            if not vertex_ids:
                status[i] = 1
                continue

            cell_vertices = vor.vertices[vertex_ids]
            diffs = cell_vertices - pts[i]
            dists = np.linalg.norm(diffs, axis=1)
            pos_idx = int(np.argmax(dists))
            r_plus = float(dists[pos_idx])
            if r_plus < 1e-12:
                status[i] = 2
                continue

            p_plus = cell_vertices[pos_idx]
            n_hat = (p_plus - pts[i]) / r_plus
            positive_pole[i] = p_plus
            normal[i] = n_hat
            pole_radius[i] = r_plus

            side = diffs @ n_hat
            opposite_mask = side < -1e-9 * r_plus
            if np.any(opposite_mask):
                opp_dists = dists[opposite_mask]
                opp_vertices = cell_vertices[opposite_mask]
                neg_idx = int(np.argmax(opp_dists))
                negative_pole[i] = opp_vertices[neg_idx]
                has_negative_pole[i] = True
            else:
                status[i] = 3

    for i in range(n):
        if unbounded[i]:
            diagnostics.append(f"Point {i}: Voronoi cell still unbounded after sentinel padding.")
        if status[i] == 1:
            diagnostics.append(f"Point {i}: no finite Voronoi vertices found; pole undefined.")
        elif status[i] == 2:
            diagnostics.append(f"Point {i}: degenerate Voronoi cell (zero radius); pole undefined.")
        elif status[i] == 3:
            diagnostics.append(f"Point {i}: no negative pole found on the opposite side.")

    return PoleEstimationResult(
        positive_pole=positive_pole, negative_pole=negative_pole,
        has_negative_pole=has_negative_pole, normal=normal, pole_radius=pole_radius,
        diagnostics=diagnostics,
    )


def _tetrahedron_circumcenter(p0, p1, p2, p3) -> np.ndarray:
    """Circumcenter of a tetrahedron.

    Via the linear system of 3 perpendicular-bisector equations relative to
    ``p0`` (solved by least squares; exact for a non-degenerate tetrahedron).
    """
    pts4 = np.array([p0, p1, p2, p3])
    a = 2.0 * (pts4[1:] - pts4[0])
    b = np.sum(pts4[1:] ** 2, axis=1) - np.sum(pts4[0] ** 2)
    center, *_ = np.linalg.lstsq(a, b, rcond=None)
    return center


def _tetrahedra_circumcenters_batch(points: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    """Vectorized ``_tetrahedron_circumcenter`` over every tetrahedron at once.

    What is Being Computed?:
        The same per-tetrahedron perpendicular-bisector linear system as
        ``_tetrahedron_circumcenter``, solved for all tetrahedra in a single batched
        ``np.linalg.solve`` call instead of a Python loop -- this is the dominant cost of
        ``cocone_filter`` on any non-trivial point cloud, and batching it is a large
        constant-factor win with no change in the result.

    Args:
        points: (N, 3) array of point coordinates.
        tetrahedra: (M, 4) int array of tetrahedra (vertex indices into ``points``).

    Returns:
        np.ndarray: Shape ``(M, 3)`` circumcenters.
    """
    tet_pts = points[tetrahedra]  # (M, 4, 3)
    a = 2.0 * (tet_pts[:, 1:, :] - tet_pts[:, :1, :])  # (M, 3, 3)
    b = np.sum(tet_pts[:, 1:, :] ** 2, axis=2) - np.sum(tet_pts[:, :1, :] ** 2, axis=2)  # (M, 3)
    return np.linalg.solve(a, b[..., None])[..., 0]


class CoconeFilterResult(BaseModel):
    """Output of the Cocone angle + pole-reach filter.

    Attributes:
        surviving_triangles (list[tuple[int, int, int]]): Candidate triangles that passed
            the filter at all three vertices.
        n_candidates (int): Total number of distinct candidate triangles considered.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    surviving_triangles: list[tuple[int, int, int]]
    n_candidates: int


def cocone_filter(
    points: np.ndarray,
    tetrahedra: np.ndarray,
    normals: np.ndarray,
    pole_radius: np.ndarray,
    *,
    theta: float = _DEFAULT_THETA,
    reach_fraction: float = _DEFAULT_REACH_FRACTION,
    backend: str = "auto",
) -> CoconeFilterResult:
    """Keep only Delaunay facets whose dual Voronoi edge reaches towards a real pole.

    What is Being Computed?:
        The Cocone membership test: a candidate Delaunay triangle (a face of some Delaunay
        tetrahedron, with all 3 vertices among the original points) survives at one of its
        vertices ``v`` only if BOTH (a) its own (cross-product) normal is within angle
        ``theta`` of ``v``'s pole-estimated normal, and (b) at least one of its 1-2 incident
        tetrahedra's circumcenters -- the endpoints of its dual Voronoi edge, by
        Delaunay-Voronoi duality -- lies at least ``reach_fraction`` of ``v``'s own pole
        radius away from ``v``. A triangle survives overall only if it passes at all three
        of its vertices.

        Condition (b) is not optional: for smoothly-sampled data, many tetrahedra just
        beneath the true surface have facets whose normal *direction* happens to already
        agree with the nearby true surface normal (condition (a) alone), simply because
        nearby points share similar normals -- but their dual edge stays shallow, nowhere
        near the pole. Filtering on direction alone was measured to accept roughly 2x too
        many candidates per vertex on a well-sampled sphere (over 90% of them reaching under
        half of the true pole radius); requiring the dual edge to actually extend out toward
        the pole is what discriminates the true surface layer from these shallow neighbors.

        ``tetrahedra`` is expected to come from the Delaunay triangulation of the *same*
        sentinel-padded point set ``estimate_voronoi_poles`` used (``points`` here is that
        same padded array; ``normals``/``pole_radius`` are indexed over the original points
        only, i.e. rows ``[0, len(normals))`` of ``points``) -- a tetrahedron's circumcenter
        is exactly a vertex of that same Voronoi diagram, by duality, so pole radii and
        circumcenter distances are only comparable when both come from one consistent
        triangulation. A tetrahedron may include a sentinel as its 4th vertex (only its
        *facet* need be all-original); its circumcenter is still used for the reach check.

    Algorithm:
        1. Build the ``facet -> incident tetrahedra`` map from ``tetrahedra``, restricted to
           facets whose 3 vertices are all original points (a facet has 1 incident
           tetrahedron if it's on the tetrahedralization's outer hull, else 2 -- either may
           include a sentinel as its non-facet 4th vertex).
        2. Compute every tetrahedron's circumcenter (``_tetrahedron_circumcenter``).
        3. Vectorized angle filter as before, restricted to the resulting distinct facets.
        4. Per surviving candidate, check the reach condition per vertex directly (a small
           Python loop over the -- already angle-filtered, so far fewer -- candidates).

    Args:
        points: (N + S, 3) array of point coordinates: the original points (rows
            ``[0, N)``, ``N = len(normals)``) followed by any sentinels used to compute
            ``normals``/``pole_radius`` (e.g. via ``_sentinel_padded_points``).
        tetrahedra: (M, 4) int array of Delaunay tetrahedra over all of ``points``
            (e.g. ``Delaunay(points).simplices``).
        normals: (N, 3) array of per-point pole normal estimates (zero rows are treated as
            "no normal available", failing the filter).
        pole_radius: (N,) array of per-point pole radii (``estimate_voronoi_poles``'s
            ``pole_radius``), used as the reach-condition scale.
        theta: Cocone half-angle, in ``(0, pi/8]`` by convention.
        reach_fraction: Minimum fraction of a vertex's own pole radius its facet's dual edge
            must reach for the facet to be considered part of the true surface layer there.
        backend: ``"auto"`` (default; use Julia when available), ``"julia"`` (require it), or
            ``"python"`` (force the pure-NumPy path). Julia fuses steps 1-4 into one call.

    Returns:
        CoconeFilterResult: The surviving candidate triangles.
    """
    pts = np.asarray(points, dtype=np.float64)
    tets = np.asarray(tetrahedra, dtype=np.int64)
    n_original = len(normals)
    if tets.size == 0:
        return CoconeFilterResult(surviving_triangles=[], n_candidates=0)

    backend_norm = str(backend).lower().strip()
    use_julia = (backend_norm == "julia") or (backend_norm == "auto" and julia_engine.available)
    if use_julia:
        try:
            surviving, n_candidates = julia_engine.compute_cocone_filter(
                pts, tets, np.asarray(normals, dtype=np.float64), np.asarray(pole_radius, dtype=np.float64),
                float(theta), float(reach_fraction),
            )
            return CoconeFilterResult(surviving_triangles=surviving, n_candidates=n_candidates)
        except Exception as e:
            if backend_norm == "julia":
                raise
            warnings.warn(f"Julia backend failed for cocone_filter, falling back to Python: {e!r}")

    facet_to_tets: dict = defaultdict(list)
    for ti in range(tets.shape[0]):
        tet_sorted = sorted(int(v) for v in tets[ti])
        for face in itertools.combinations(tet_sorted, 3):
            if face[-1] < n_original:  # sorted ascending: all 3 vertices are original iff so is the largest
                facet_to_tets[face].append(ti)

    if not facet_to_tets:
        return CoconeFilterResult(surviving_triangles=[], n_candidates=0)

    circumcenters = _tetrahedra_circumcenters_batch(pts, tets)

    candidate_triangles = sorted(facet_to_tets.keys())
    tris = np.array(candidate_triangles, dtype=np.int64)

    p0, p1, p2 = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    tri_normals = np.cross(p1 - p0, p2 - p0)
    tri_norms = np.linalg.norm(tri_normals, axis=1)
    valid = tri_norms > 1e-15
    tri_normals_unit = np.zeros_like(tri_normals)
    tri_normals_unit[valid] = tri_normals[valid] / tri_norms[valid, None]

    cos_theta = np.cos(theta)
    ok = valid.copy()
    for col in range(3):
        v_idx = tris[:, col]
        n_v = normals[v_idx]
        has_normal = np.linalg.norm(n_v, axis=1) > 1e-15
        cos_angle = np.abs(np.einsum("ij,ij->i", tri_normals_unit, n_v))
        ok &= has_normal & (cos_angle >= cos_theta)

    for i, tri in enumerate(candidate_triangles):
        if not ok[i]:
            continue
        ccs = circumcenters[facet_to_tets[tri]]
        reaches = True
        for v in tri:
            if pole_radius[v] <= 1e-12:
                reaches = False
                break
            dists = np.linalg.norm(ccs - pts[v], axis=1)
            if float(np.max(dists)) < reach_fraction * pole_radius[v]:
                reaches = False
                break
        ok[i] = reaches

    surviving = [tuple(int(x) for x in tris[i]) for i in range(len(candidate_triangles)) if ok[i]]
    return CoconeFilterResult(surviving_triangles=surviving, n_candidates=len(candidate_triangles))


def _tangent_frame(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two orthonormal vectors spanning the plane orthogonal to ``normal``."""
    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = helper - np.dot(helper, normal) * normal
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


class PruneWalkResult(BaseModel):
    """Output of the prune-and-walk manifold-forcing step.

    Attributes:
        final_triangles (list[tuple[int, int, int]]): Globally-consistent surviving
            triangles (kept by every one of their own vertices' local decisions).
        per_vertex_local_simplices (dict): ``vertex -> set[frozenset[int]]`` of the
            triangles that vertex's own local decision kept -- the input
            ``intersect_local_stars`` was built from; exposed for repair-loop reuse.
        unresolved_vertices (list[int]): Vertices where the walk hit a dead end (no valid
            single umbrella found from the candidates available) rather than a clean cycle.
        diagnostics (list[str]): Human-readable per-vertex notes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    final_triangles: list[tuple[int, int, int]]
    per_vertex_local_simplices: dict
    unresolved_vertices: list[int]
    diagnostics: list[str] = Field(default_factory=list)


def prune_and_walk(
    points: np.ndarray,
    candidate_triangles,
    normals: np.ndarray,
    *,
    backend: str = "auto",
) -> PruneWalkResult:
    """Force a single consistent 2-manifold umbrella at every vertex.

    What is Being Computed?:
        Per vertex ``v``, the candidate triangles incident to ``v`` induce a local "link
        graph" (nodes = candidate neighbors, edges = ``{a, b}`` for each candidate triangle
        ``(v, a, b)``). If that graph is already a single simple cycle or open path
        (``pysurgery.geometry.perturbation.is_single_path_or_cycle``, the fast path -- a
        superset of the criterion that makes ``SimplicialComplex.is_homology_manifold`` exact
        at dimension <= 2), every candidate at ``v`` is kept unchanged. Otherwise, a
        deterministic walk in tangent-plane angular order around ``v`` (using its pole normal
        to define the tangent plane) keeps exactly one consistent umbrella fan, discarding
        the rest -- this is the step that actually *forces* manifold-ness, not merely filters
        towards it. A simplex only survives globally if every one of its own vertices' local
        decisions agrees (``intersect_local_stars``); this is where a "some vertices keep it,
        others don't" conflict becomes visible for repair.

        A vertex link that closes into a full cycle (an interior point) and one that walks
        out to a simple open path (a legitimate *boundary* point -- e.g. a triangle genuinely
        absent because the true surface doesn't extend that far, not a defect) are both
        accepted as a resolved, single umbrella fan, *even if* the walk also leaves some
        candidate edges attached to that fan unvisited -- a spurious extra candidate hanging
        off an otherwise-clean cycle or path is exactly the junk this step exists to discard,
        not a sign it failed. The one thing that does mean a genuine defect: a leftover edge
        whose *both* endpoints the walk never reached at all, i.e. an entire separate
        connected component (two disjoint triangles at one vertex is the classic pinch
        example) that a single linear walk from one starting point structurally cannot
        absorb -- that must not be silently dropped without being flagged. This mirrors
        ``is_homology_manifold``'s own d<=2 exactness (see its docstring): a connected,
        max-degree-<=2 link is *always* a valid manifold link, whether or not it happens to
        close, and only an actually-disconnected or actually-branching link is a defect.

    Algorithm:
        1. Build each vertex's local link graph from ``candidate_triangles``.
        2. Fast path: ``is_single_path_or_cycle`` on that graph; if clean, keep everything.
        3. Otherwise: order neighbors by angle in the tangent plane orthogonal to the
           vertex's pole normal; walk from the lowest-``(angle, index)`` neighbor, at each
           step moving to the neighbor -- among those still connected by a *surviving
           candidate edge* -- with the smallest forward angular step (ties broken by index);
           stop on cycle closure, a repeated edge, or a dead end (bounded by the neighbor
           count, so this always terminates).
        4. The walk is resolved -- not a dead end -- iff it terminated cleanly (cycle closure
           or running out of neighbors, not a repeated edge or a revisited node) *and* every
           leftover, never-visited edge shares at least one endpoint with the fan the walk
           did keep (see above). Only a leftover edge entirely outside the walked fan --
           proof of a genuinely separate component -- is recorded in ``unresolved_vertices``
           (fed to a perturbation-repair loop by callers), rather than silently dropped.

    Args:
        points: (N, 3) array of point coordinates.
        candidate_triangles: Iterable of length-3 index tuples (typically
            ``cocone_filter``'s surviving triangles).
        normals: (N, 3) array of per-point pole normal estimates.
        backend: ``"auto"`` (default; use Julia when available), ``"julia"`` (require it), or
            ``"python"`` (force the pure-NumPy path). Julia parallelizes the per-vertex work
            (``intersect_local_stars`` always stays a single shared Python implementation,
            also used by ``moser_tardos_repair``).

    Returns:
        PruneWalkResult: The globally-consistent surviving triangles plus diagnostics.
    """
    pts = np.asarray(points, dtype=np.float64)

    backend_norm = str(backend).lower().strip()
    use_julia = (backend_norm == "julia") or (backend_norm == "auto" and julia_engine.available)
    if use_julia:
        try:
            tris_list = []
            for tri in candidate_triangles:
                t = tuple(int(x) for x in tri)
                if len(set(t)) != 3:
                    continue
                tris_list.append(t)
            tris_arr = np.array(tris_list, dtype=np.int64) if tris_list else np.zeros((0, 3), dtype=np.int64)

            vertex_ids, status, n_kept_edges, kept_flat, kept_offsets = julia_engine.compute_prune_and_walk(
                pts, tris_arr, np.asarray(normals, dtype=np.float64)
            )

            per_vertex_local: dict = {}
            unresolved: list[int] = []
            diagnostics: list[str] = []
            for idx in range(len(vertex_ids)):
                v = int(vertex_ids[idx])
                lo, hi = int(kept_offsets[idx]), int(kept_offsets[idx + 1])
                tris = set()
                for k in range(lo, hi):
                    a, b, c = (int(kept_flat[3 * k]), int(kept_flat[3 * k + 1]), int(kept_flat[3 * k + 2]))
                    tris.add(frozenset((a, b, c)))
                per_vertex_local[v] = tris
                st = int(status[idx])
                if st == 1:
                    unresolved.append(v)
                    diagnostics.append(f"Vertex {v}: no normal estimate available; cannot walk.")
                elif st == 3:
                    unresolved.append(v)
                    diagnostics.append(
                        f"Vertex {v}: prune-and-walk dead end after {int(n_kept_edges[idx])} step(s)."
                    )

            surviving, _stats = intersect_local_stars(per_vertex_local)
            final_triangles = [tuple(sorted(int(x) for x in t)) for t in surviving]
            return PruneWalkResult(
                final_triangles=final_triangles,
                per_vertex_local_simplices=per_vertex_local,
                unresolved_vertices=unresolved,
                diagnostics=diagnostics,
            )
        except Exception as e:
            if backend_norm == "julia":
                raise
            warnings.warn(f"Julia backend failed for prune_and_walk, falling back to Python: {e!r}")

    vertex_triangles: dict = defaultdict(set)
    for tri in candidate_triangles:
        t = frozenset(int(x) for x in tri)
        if len(t) != 3:
            continue
        for v in t:
            vertex_triangles[v].add(t)

    per_vertex_local: dict = {}
    unresolved: list[int] = []
    diagnostics: list[str] = []

    for v, tris in vertex_triangles.items():
        link_edges = []
        tri_by_edge = {}
        for t in tris:
            others = tuple(sorted(x for x in t if x != v))
            if len(others) != 2:
                continue
            link_edges.append(others)
            tri_by_edge[frozenset(others)] = t

        ok, _cyclic_order = is_single_path_or_cycle(link_edges)
        if ok:
            per_vertex_local[v] = tris
            continue

        n_v = normals[v]
        if not np.any(n_v):
            unresolved.append(int(v))
            diagnostics.append(f"Vertex {v}: no normal estimate available; cannot walk.")
            per_vertex_local[v] = set()
            continue

        u_axis, v_axis = _tangent_frame(n_v)
        neighbors = sorted({x for e in link_edges for x in e})
        angles = {}
        for nb in neighbors:
            d = pts[nb] - pts[v]
            d = d - np.dot(d, n_v) * n_v
            angles[nb] = float(np.arctan2(np.dot(d, v_axis), np.dot(d, u_axis)))

        adjacency: dict = defaultdict(set)
        for a, b in link_edges:
            adjacency[a].add(b)
            adjacency[b].add(a)

        if not neighbors:
            per_vertex_local[v] = set()
            continue

        start = min(neighbors, key=lambda x: (angles[x], x))
        kept_edges = []
        visited_edges = set()
        prev, current = None, start
        walked_nodes = {start}
        clean_termination = False
        for _step in range(len(neighbors) + 1):
            candidates = [x for x in adjacency[current] if x != prev]
            if not candidates:
                clean_termination = True  # ran out of neighbors: a path endpoint
                break

            def _forward_key(x, _current=current):
                delta = (angles[x] - angles[_current]) % (2.0 * np.pi)
                return (delta, x)

            candidates.sort(key=_forward_key)
            nxt = candidates[0]
            edge_key = frozenset((current, nxt))
            if edge_key in visited_edges:
                break  # tried to reuse an edge: genuine ambiguity, not a clean stop
            kept_edges.append((current, nxt))
            visited_edges.add(edge_key)
            if nxt == start:
                clean_termination = True  # closed a full cycle
                break
            if nxt in walked_nodes:
                break  # revisited a node we didn't start at: genuine branching, not clean
            walked_nodes.add(nxt)
            prev, current = current, nxt

        # A clean termination (closed cycle, or simply ran out of neighbors at a path
        # endpoint) means the walk found one unambiguous, consistent umbrella fan --
        # regardless of whether it also left some candidate edges out. Leftover edges
        # attached to a node the walk *did* include (e.g. a spurious extra candidate
        # hanging off an otherwise-clean fan) are exactly the junk prune-and-walk is
        # supposed to discard, not a sign of failure. The one thing that still counts as a
        # genuine, unresolved defect is a leftover edge whose *both* endpoints the walk
        # never reached at all -- an entire separate connected component (e.g. two disjoint
        # triangles, a classic pinch) a single linear walk from one start node structurally
        # cannot absorb, and must not silently drop without flagging it.
        leftover = [e for e in link_edges if frozenset(e) not in visited_edges]
        disconnected_leftover = any(a not in walked_nodes and b not in walked_nodes for a, b in leftover)
        success = clean_termination and not disconnected_leftover
        kept_tris = {tri_by_edge[frozenset(e)] for e in kept_edges if frozenset(e) in tri_by_edge}
        per_vertex_local[v] = kept_tris
        if not success:
            unresolved.append(int(v))
            diagnostics.append(f"Vertex {v}: prune-and-walk dead end after {len(kept_edges)} step(s).")

    surviving, _stats = intersect_local_stars(per_vertex_local)
    final_triangles = [tuple(sorted(int(x) for x in t)) for t in surviving]
    return PruneWalkResult(
        final_triangles=final_triangles,
        per_vertex_local_simplices=per_vertex_local,
        unresolved_vertices=unresolved,
        diagnostics=diagnostics,
    )


class TightCoconeResult(BaseModel):
    """Output of the Tight Cocone closing pass.

    Attributes:
        boundary_triangles (list[tuple[int, int, int]]): The closed surface -- the boundary
            between flood-fill-labeled inside and outside tetrahedra.
        n_wall_mismatch (int): Number of facets where the flood-fill boundary disagrees with
            the input wall set (a nonzero count signals prune-and-walk left a gap).
        diagnostics (list[str]): Human-readable notes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    boundary_triangles: list[tuple[int, int, int]]
    n_wall_mismatch: int
    diagnostics: list[str] = Field(default_factory=list)


def tight_cocone_close(
    points: np.ndarray, wall_triangles, *, max_circumradius_factor: float = 6.0
) -> TightCoconeResult:
    """Best-effort closing pass towards a watertight surface (Dey-Goswami peeling).

    What is Being Computed?:
        Labels every Delaunay tetrahedron inside/outside via a breadth-first flood fill
        seeded from the tetrahedra touching the convex hull (always "outside"), where the
        given ``wall_triangles`` (typically ``prune_and_walk``'s output) are treated as
        impassable -- the flood fill never crosses them. Anything never reached is "inside"
        by exclusion. In principle the output surface is exactly the boundary between
        differently-labeled tetrahedra, which would be closed (watertight) and manifold by
        construction.

        In practice, on a point cloud sampled only on a surface (no interior points), the
        Delaunay tetrahedralization of the whole point set has enough tetrahedra with no
        facet anywhere near the true surface that the flood fill routinely tunnels around
        even a fully closed, already-consistent wall set (measured directly: a 296-triangle
        sphere reconstruction with zero boundary edges of its own still produced 306
        mismatched facets here). ``max_circumradius_factor`` (excluding tetrahedra whose
        circumradius is large relative to the local point spacing from the flood fill
        entirely) measurably reduces, but does not eliminate, this tunnelling. Treat this
        function as experimental / best-effort: it is not currently a reliable watertight-
        closure guarantee, only a diagnostic-producing attempt at one. The primary path to a
        closed surface is ``cocone_reconstruction``'s ``repair=True`` (which resolves
        prune-and-walk's own conflicts directly and was measured to reliably reach zero
        boundary edges on its own for genus-0 inputs), not this function.

    Algorithm:
        Walls-are-impassable, unreached-is-inside, chosen deliberately over a literal
        "cross-and-flip-label" reading of the peeling algorithm: it degrades gracefully
        (a smaller but still consistent closed boundary) rather than producing inconsistent
        double-crossing artifacts when ``wall_triangles`` has gaps -- which it may, before a
        perturbation-repair loop has fully resolved every ``prune_and_walk`` conflict.

    Args:
        points: (N, 3) array of point coordinates.
        wall_triangles: Iterable of length-3 index tuples treated as impassable walls.
        max_circumradius_factor: Tetrahedra whose circumradius exceeds this factor times the
            median nearest-neighbor spacing are excluded from the flood fill entirely (see
            above); set higher for deliberately non-uniform sampling density.

    Returns:
        TightCoconeResult: The closed boundary surface, plus a wall/boundary mismatch count
        as a diagnostic (not a silently-trusted result) -- see the algorithm note above.

    Raises:
        ValueError: If ``points`` is not an (N, 3) array.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("tight_cocone_close currently supports (N, 3) point clouds only.")

    dt = Delaunay(pts, qhull_options="QJ")
    wall_set = {frozenset(int(x) for x in t) for t in wall_triangles}
    n_tet = dt.simplices.shape[0]
    is_outside = np.zeros(n_tet, dtype=bool)
    visited = np.zeros(n_tet, dtype=bool)

    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    nn_dist, _ = tree.query(pts, k=2)
    circumradius_cutoff = max_circumradius_factor * float(np.median(nn_dist[:, 1]))
    circumcenters = _tetrahedra_circumcenters_batch(pts, dt.simplices)
    circumradii = np.linalg.norm(circumcenters - pts[dt.simplices[:, 0]], axis=1)
    passable = circumradii <= circumradius_cutoff

    queue: deque = deque()
    for t in range(n_tet):
        if passable[t] and np.any(dt.neighbors[t] == -1):
            is_outside[t] = True
            visited[t] = True
            queue.append(t)

    while queue:
        t = queue.popleft()
        verts = dt.simplices[t]
        for j in range(4):
            nb = int(dt.neighbors[t][j])
            if nb == -1 or visited[nb] or not passable[nb]:
                continue
            facet = frozenset(int(verts[k]) for k in range(4) if k != j)
            if facet in wall_set:
                continue
            is_outside[nb] = True
            visited[nb] = True
            queue.append(nb)

    boundary = set()
    for t in range(n_tet):
        verts = dt.simplices[t]
        t_out = bool(is_outside[t])
        for j in range(4):
            nb = int(dt.neighbors[t][j])
            if nb == -1:
                continue
            if t_out != bool(is_outside[nb]):
                facet = frozenset(int(verts[k]) for k in range(4) if k != j)
                boundary.add(facet)

    mismatch = boundary.symmetric_difference(wall_set)
    diagnostics = []
    if mismatch:
        diagnostics.append(
            f"{len(mismatch)} facet(s) differ between the prune-and-walk wall set and the "
            "flood-fill inside/outside boundary; the surface may have a gap prune-and-walk "
            "didn't resolve."
        )
    boundary_triangles = [tuple(sorted(f)) for f in boundary]
    return TightCoconeResult(
        boundary_triangles=boundary_triangles, n_wall_mismatch=len(mismatch), diagnostics=diagnostics
    )


class CoconeReconstructionResult(BaseModel):
    """End-to-end Cocone/Tight Cocone reconstruction output.

    Attributes:
        triangles (list[tuple[int, int, int]]): The reconstructed surface's triangles.
        points (np.ndarray): The point coordinates the triangles are indexed against --
            identical to the input unless repair perturbed a few points (see
            ``repair_rounds_used``), in which case this is the (minutely) perturbed array.
        poles (PoleEstimationResult): The pole/normal estimates used for the final result.
        n_candidates (int): Number of candidate triangles considered (Delaunay tetrahedra
            faces) in the final round.
        n_cocone_surviving (int): Number of candidates that passed the Cocone angle filter
            in the final round.
        unresolved_vertices (list[int]): Vertices where prune-and-walk hit a dead end, in
            the final round (empty on a converged repair).
        repair_rounds_used (int): Number of Moser-Tardos perturbation rounds needed.
        diagnostics (list[str]): Combined diagnostics from every stage.
        tight (TightCoconeResult | None): The Tight Cocone closing-pass result, if run.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    triangles: list[tuple[int, int, int]]
    points: np.ndarray
    poles: PoleEstimationResult
    n_candidates: int
    n_cocone_surviving: int
    unresolved_vertices: list[int]
    repair_rounds_used: int = 0
    diagnostics: list[str] = Field(default_factory=list)
    tight: Optional[TightCoconeResult] = None


def _cocone_pipeline_round(
    pts: np.ndarray, *, theta: float, reach_fraction: float,
    bounding_radius_factor: float, n_sentinels: Optional[int],
    backend: str = "auto",
):
    """Run poles -> cocone filter -> prune-and-walk once.

    Used both for a plain (no-repair) reconstruction and as the per-round
    computation inside the repair loop.
    """
    poles = estimate_voronoi_poles(
        pts, bounding_radius_factor=bounding_radius_factor, n_sentinels=n_sentinels, backend=backend
    )
    # Same sentinel-padded point set poles were computed from (see cocone_filter's docstring
    # note on why this consistency matters): a tetrahedron's circumcenter is only comparable
    # to a pole radius when both come from one triangulation.
    combined = _sentinel_padded_points(pts, bounding_radius_factor, n_sentinels)
    dt = Delaunay(combined, qhull_options="QJ")
    filt = cocone_filter(
        combined, dt.simplices, poles.normal, poles.pole_radius,
        theta=theta, reach_fraction=reach_fraction, backend=backend,
    )
    walk = prune_and_walk(pts, filt.surviving_triangles, poles.normal, backend=backend)
    return poles, filt, walk


def _cocone_conflict_vertices(walk: PruneWalkResult) -> list:
    """Identify vertices that cocone_reconstruction's repair loop should perturb.

    Specifically, prune-and-walk dead ends, plus any vertex whose final
    (post ``intersect_local_stars``) link is not a single closed cycle --
    deliberately *stricter* than ``prune_and_walk``'s own ``unresolved_vertices``
    (which also accepts a clean open path -- a legitimate boundary vertex, not a
    defect -- via ``is_single_path_or_cycle``). That leniency is the right
    general-purpose criterion for prune-and-walk in isolation, but reconstruction's
    whole purpose (per this module's docstring) is reaching a *closed* surface
    whenever the sample supports one; a vertex left on the boundary is exactly the
    thing repair should keep trying to resolve, not wave through as already-fine
    (measured: accepting boundary here made repair converge faster but silently
    settle for an open result on several genus-0 fixtures that reach full closure
    just fine with a little more perturbation).

    Losing a candidate to cross-vertex disagreement is *not*, by itself, a conflict:
    ``intersect_local_stars`` dropping a triangle some but not all of its vertices
    wanted is the expected, harmless way most disagreements resolve (see
    ``prune_and_walk``'s docstring); only checking the vertex's own actual final link
    -- rather than "was anything dropped near it" -- avoids flagging that harmless
    case.
    """
    final_set = {frozenset(t) for t in walk.final_triangles}
    final_by_vertex: dict = defaultdict(set)
    for t in final_set:
        for v in t:
            final_by_vertex[v].add(t)

    conflict_vertices = {int(v) for v in walk.unresolved_vertices}
    for v, tris in final_by_vertex.items():
        link_edges = [tuple(sorted(x for x in t if x != v)) for t in tris]
        ok, _ = is_single_cycle(link_edges)
        if not ok:
            conflict_vertices.add(int(v))
    return sorted(conflict_vertices)


def cocone_reconstruction(
    points: np.ndarray,
    *,
    theta: float = _DEFAULT_THETA,
    reach_fraction: float = _DEFAULT_REACH_FRACTION,
    tight: bool = False,
    bounding_radius_factor: float = 20.0,
    n_sentinels: Optional[int] = None,
    repair: bool = True,
    max_repair_rounds: int = 250,
    perturbation_scale: Optional[float] = None,
    seed: int = 0,
    backend: str = "auto",
) -> CoconeReconstructionResult:
    """Reconstruct a manifold-certified triangulated surface from a 3D point cloud.

    What is Being Computed?:
        The full Cocone (optionally Tight Cocone) pipeline: pole/normal estimation ->
        Cocone angle + pole-reach filter -> prune-and-walk -> optional watertight closing
        pass. Unlike alpha/Rips filtrations, this produces one certified reconstruction at a
        single scale (chosen from the sampling density via the pole radii), not a family
        over a parameter -- manifold-ness is not preserved across a distance-threshold sweep.

        Even on well-sampled, noise-free data, independent per-vertex decisions in
        prune-and-walk routinely disagree with each other at a handful of vertices (some
        candidate triangle is kept by 2 of its 3 vertices' own local decisions but not the
        third) -- this is not a bug, it is the expected shape of the problem (the same
        reason the tangential Delaunay complex needs a star-consistency reconciliation
        step). When ``repair=True`` (the default), any vertex left unresolved by
        prune-and-walk or implicated in such a disagreement is perturbed by a tiny bounded
        random shift and the whole pipeline is re-run, via
        ``pysurgery.geometry.perturbation.moser_tardos_repair`` -- "principled, targeted,
        terminating local resampling" in the Moser-Tardos spirit (see that function's
        docstring for the precise scope of that claim), not ad hoc global joggling.

        Known limitation (genus/high-curvature): on a genus-0 (sphere-like) surface,
        ``repair=True`` reliably converges to a correctly-closed manifold with the right
        Betti numbers. On a surface with a persistent concave region relative to the local
        sampling density -- the torus's inner, hole-facing side is the motivating example --
        the Delaunay tetrahedralization of a point cloud sampled only on the surface (no
        interior points) must span the "hole" with tetrahedra that have no facet anywhere
        near the true surface there, and this was measured to make the angle+reach filter
        systematically less selective in exactly that region, independent of tuning ``theta``
        /``reach_fraction`` or of sampling density. ``repair`` still converges -- every
        vertex it keeps ends up with a genuinely valid (non-branching) link -- but the
        troublesome region can end up split across several individually-valid manifold-
        with-boundary patches rather than stitched into one closed torus, i.e. the result may
        be disconnected (``betti[0] > 1``) or open (``not is_closed_manifold``) rather than
        merely the wrong genus (confirmed on both synthetic tori and the real motivating
        dataset). Properly fixing this needs the fuller sliver-exudation / weighted-Delaunay
        reweighting machinery referenced for the Cheng-Dey-Ramos baseline elsewhere in this
        project's design notes, not merely point perturbation; that is out of scope here.
        Treat a connectivity/genus/Betti mismatch on such inputs as a signal to inspect
        ``diagnostics`` and the covered-vertex count, not as a silent success.

    Args:
        points: (N, 3) array of point coordinates.
        theta: Cocone half-angle. See ``_DEFAULT_THETA``'s comment for why the default is
            wider than the angle-only literature convention of ``(0, pi/8]``.
        reach_fraction: Passed through to ``cocone_filter``.
        tight: If True, run the Tight Cocone closing pass for a watertight (closed) surface.
        bounding_radius_factor: Passed through to ``estimate_voronoi_poles``.
        n_sentinels: Passed through to ``estimate_voronoi_poles``.
        repair: If True, resolve residual prune-and-walk conflicts via
            ``moser_tardos_repair`` rather than returning them as diagnostics only.
        max_repair_rounds: Passed through to ``moser_tardos_repair``. Torus-like inputs were
            measured to need on the order of 100-200 rounds, versus a handful for a sphere.
        perturbation_scale: Passed through to ``moser_tardos_repair``. ``None`` (the
            default) picks ``0.01`` times the median nearest-neighbor spacing among
            ``points`` -- combinatorial decisions (which candidate wins a near-tie, whether
            a walk closes) depend on relative point geometry, so a fixed absolute default
            appropriate for unit-scale data would be meaninglessly tiny on, e.g., data with
            coordinates in the thousands, and vice versa.
        seed: Passed through to ``moser_tardos_repair``.
        backend: ``"auto"`` (default; use Julia when available), ``"julia"`` (require it), or
            ``"python"`` (force the pure-NumPy path), passed through to every stage of
            ``_cocone_pipeline_round`` (``estimate_voronoi_poles``, ``cocone_filter``,
            ``prune_and_walk``) on every round, including inside the repair loop.

    Returns:
        CoconeReconstructionResult: The reconstructed triangles and full diagnostics.

    Raises:
        ReconstructionRepairError: If ``repair=True`` and the conflicts do not resolve
            within ``max_repair_rounds`` (propagated from ``moser_tardos_repair``).

    Use When:
        - Reconstructing a genuine 2-manifold surface (e.g. a torus) sampled in R^3, where an
          alpha complex would grow tetrahedra across the sample's noise band instead.
    """
    pts = np.asarray(points, dtype=np.float64)
    state: dict = {}

    if perturbation_scale is None:
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        nn_dist, _ = tree.query(pts, k=2)
        perturbation_scale = 0.01 * float(np.median(nn_dist[:, 1]))

    def _detect_conflicts(pts_current):
        poles, filt, walk = _cocone_pipeline_round(
            pts_current, theta=theta, reach_fraction=reach_fraction,
            bounding_radius_factor=bounding_radius_factor, n_sentinels=n_sentinels,
            backend=backend,
        )
        state["poles"], state["filt"], state["walk"] = poles, filt, walk
        conflict_vertices = _cocone_conflict_vertices(walk)
        return [{"indices": conflict_vertices}] if conflict_vertices else []

    def _rebuild_local(pts_current, offending):
        pass  # _detect_conflicts always recomputes the full (cheap) pipeline fresh

    repair_rounds_used = 0
    if repair:
        repair_result = moser_tardos_repair(
            pts, _detect_conflicts, _rebuild_local,
            max_rounds=max_repair_rounds, perturbation_scale=perturbation_scale, seed=seed,
            stage="cocone_prune_and_walk",
        )
        pts = repair_result.points
        repair_rounds_used = repair_result.rounds_used
    else:
        _detect_conflicts(pts)  # populate `state` without repairing anything

    poles, filt, walk = state["poles"], state["filt"], state["walk"]
    diagnostics = list(poles.diagnostics) + list(walk.diagnostics)
    tight_result = None
    if tight:
        tight_result = tight_cocone_close(pts, walk.final_triangles)
        diagnostics += tight_result.diagnostics
        triangles = tight_result.boundary_triangles
    else:
        triangles = walk.final_triangles

    return CoconeReconstructionResult(
        triangles=triangles,
        points=pts,
        poles=poles,
        n_candidates=filt.n_candidates,
        n_cocone_surviving=len(filt.surviving_triangles),
        unresolved_vertices=walk.unresolved_vertices,
        repair_rounds_used=repair_rounds_used,
        diagnostics=diagnostics,
        tight=tight_result,
    )


__all__ = [
    "PoleEstimationResult",
    "estimate_voronoi_poles",
    "CoconeFilterResult",
    "cocone_filter",
    "PruneWalkResult",
    "prune_and_walk",
    "TightCoconeResult",
    "tight_cocone_close",
    "CoconeReconstructionResult",
    "cocone_reconstruction",
]
