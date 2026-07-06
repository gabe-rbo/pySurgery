"""Tangential Delaunay complex reconstruction (Boissonnat-Ghosh) for point clouds of
arbitrary codimension.

Overview:
    Cocone/Tight Cocone (``pysurgery.geometry.reconstruction``) is specialized to surfaces
    (codimension 1) embedded in R^3, using per-point *poles* to estimate a single normal
    direction. Many point clouds of interest instead have a low intrinsic dimension ``k``
    embedded in a much larger ambient space ``d`` (``k << d``) -- e.g. data manifolds inside
    a neural network's activation space -- where "the normal direction" isn't even a single
    well-defined line. The tangential Delaunay complex handles this general case directly:
    per point, project its own neighborhood onto its own local tangent-space basis
    (``pysurgery.geometry.intrinsic_dimension.local_pca_tangent_basis``), triangulate that
    ``k``-dimensional projection with a plain Qhull Delaunay call, and keep the *star* of
    simplices incident to the point itself. A simplex survives globally only if kept
    unanimously by every one of its own vertices' independently-computed local stars
    (``pysurgery.geometry.perturbation.intersect_local_stars``) -- exactly the same
    cross-vertex reconciliation rule Cocone's prune-and-walk output is checked against, and
    for the same reason: independent local computations routinely disagree at a handful of
    points even on clean data, and that disagreement is resolved by targeted perturbation
    (``pysurgery.geometry.perturbation.moser_tardos_repair``), not silently picked one way.

Key Concepts:
    - **Local tangent projection**: each point's own PCA-estimated ``k``-dimensional basis,
      not a single global embedding -- the basis itself can (and generally does) vary
      smoothly from point to point across a curved manifold.
    - **Local star**: the ``k``-simplices of a point's own local Delaunay triangulation that
      are incident to the point itself, expressed in the ambient point cloud's global vertex
      indices.
    - **Star consistency**: a simplex is only real if every one of its own vertices'
      independent local-star computations agrees it belongs.

Common Workflows:
    1. **End-to-end reconstruction** -> ``tangential_complex_reconstruction(points, k=2)``,
       or the ``SimplicialComplex.from_tangential_complex`` classmethod that wraps it.
    2. **Codimension too high for Cocone** -> any point cloud where ``k`` is much smaller
       than the ambient dimension ``d`` (Cocone's pole/normal machinery is specific to
       codimension 1 in R^3).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.spatial import Delaunay, cKDTree

from .intrinsic_dimension import local_pca_tangent_basis, local_pca_tangent_space_dimension
from .perturbation import intersect_local_stars, moser_tardos_repair


def _local_star(
    neighbor_global_indices: np.ndarray,
    projected: np.ndarray,
) -> tuple[set[frozenset[int]], bool]:
    """Delaunay-triangulate one point's projected neighborhood and keep its star.

    Args:
        neighbor_global_indices: (m,) global point indices of the neighborhood, with the
            center point itself at position 0.
        projected: (m, k) neighborhood coordinates projected onto the center point's local
            tangent basis (row 0 is the center point, at the local origin).

    Returns:
        tuple[set[frozenset[int]], bool]: ``(star, ok)`` -- ``star`` is the set of
        ``k``-simplices (as global-index frozensets) incident to the center point;
        ``ok`` is False if the local Delaunay call itself failed (degenerate/coplanar
        neighborhood), in which case ``star`` is empty.
    """
    k = projected.shape[1]
    if projected.shape[0] < k + 1:
        return set(), False
    try:
        dt = Delaunay(projected, qhull_options="QJ")
    except Exception:
        return set(), False
    star = set()
    for simplex in dt.simplices:
        if 0 in simplex:
            star.add(frozenset(int(neighbor_global_indices[j]) for j in simplex))
    return star, True


def _compute_all_local_stars(
    points: np.ndarray,
    bases: np.ndarray,
    neighborhood_size: int,
) -> tuple[dict[int, set[frozenset[int]]], list[int]]:
    """Compute every point's local star, in parallel (each Qhull call is independent).

    Args:
        points: (N, D) point coordinates.
        bases: (N, D, k) per-point tangent bases (see ``local_pca_tangent_basis``).
        neighborhood_size: Number of nearest neighbors (excluding the point itself) to
            include in each point's local Delaunay triangulation.

    Returns:
        tuple[dict[int, set[frozenset[int]]], list[int]]: ``(local_stars, failed_points)``
        -- ``local_stars[i]`` is point i's own local star; ``failed_points`` lists points
        whose local Delaunay call itself failed (degenerate neighborhood).
    """
    n = points.shape[0]
    tree = cKDTree(points)
    k_query = min(neighborhood_size + 1, n)
    _, all_nbr_idx = tree.query(points, k=k_query)
    all_nbr_idx = np.atleast_2d(all_nbr_idx)

    def _one(i: int):
        nbr_idx = all_nbr_idx[i]
        if nbr_idx[0] != i:
            nbr_idx = np.concatenate(([i], nbr_idx[nbr_idx != i]))
        local_coords = (points[nbr_idx] - points[i]) @ bases[i]
        star, ok = _local_star(nbr_idx, local_coords)
        return i, star, ok

    local_stars: dict[int, set[frozenset[int]]] = {}
    failed_points: list[int] = []
    with ThreadPoolExecutor() as executor:
        for i, star, ok in executor.map(_one, range(n)):
            local_stars[i] = star
            if not ok:
                failed_points.append(i)
    return local_stars, failed_points


def _tangential_conflict_vertices(
    local_stars: dict[int, set[frozenset[int]]],
    surviving: set[frozenset[int]],
    failed_points: list[int],
) -> list[int]:
    """Vertices ``tangential_complex_reconstruction``'s repair loop should perturb: points
    whose own local Delaunay call failed outright, plus every vertex of a simplex some --
    but not all -- of its own vertices' local stars kept."""
    all_candidates: set = set()
    for local in local_stars.values():
        all_candidates.update(local)
    dropped = all_candidates - surviving
    conflict_vertices = {int(v) for v in failed_points}
    for simplex in dropped:
        conflict_vertices.update(int(v) for v in simplex)
    return sorted(conflict_vertices)


class TangentialComplexResult(BaseModel):
    """End-to-end tangential Delaunay complex reconstruction output.

    Attributes:
        simplices (list[tuple[int, ...]]): The reconstructed ``k``-simplices.
        points (np.ndarray): The point coordinates the simplices are indexed against --
            identical to the input unless repair perturbed a few points.
        k (int): The (possibly auto-estimated) tangent-space dimension used.
        neighborhood_size (int): Number of neighbors used per point's local triangulation.
        n_candidates (int): Number of distinct simplices seen by at least one point's local
            star, in the final round.
        unresolved_points (list[int]): Points whose own local Delaunay call failed
            (degenerate neighborhood), in the final round (empty on a converged repair).
        repair_rounds_used (int): Number of Moser-Tardos perturbation rounds needed.
        diagnostics (list[str]): Human-readable notes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    simplices: list[tuple[int, ...]]
    points: np.ndarray
    k: int
    neighborhood_size: int
    n_candidates: int
    unresolved_points: list[int]
    repair_rounds_used: int = 0
    diagnostics: list[str] = Field(default_factory=list)


def tangential_complex_reconstruction(
    points: np.ndarray,
    *,
    k: Optional[int] = None,
    neighborhood_size: Optional[int] = None,
    variance_threshold: float = 0.9,
    repair: bool = True,
    max_repair_rounds: int = 250,
    perturbation_scale: Optional[float] = None,
    seed: int = 0,
) -> TangentialComplexResult:
    """Reconstruct a ``k``-dimensional simplicial complex from a point cloud in R^d.

    What is Being Computed?:
        The Boissonnat-Ghosh tangential Delaunay complex: per point, a local ``k``-simplex
        star computed by projecting the point's own neighborhood onto its own PCA-estimated
        tangent basis and Delaunay-triangulating that projection, then a global complex
        formed by keeping only the simplices every one of their own vertices' independent
        local computations agrees on (``pysurgery.geometry.perturbation.
        intersect_local_stars``). Handles the general ``k << d`` case Cocone's pole/normal
        machinery is not designed for (Cocone is specific to codimension 1 in R^3).

        As with Cocone's prune-and-walk, independent per-point local decisions routinely
        disagree with each other -- this is not a bug, it is the expected shape of the
        problem, and here it is measurably *more* common than in Cocone's codimension-1
        case: two nearby points' independently-PCA-estimated tangent bases can differ
        enough in orientation that a locally near-cocircular configuration (unavoidable in
        any reasonably uniform sample) gets triangulated with a different diagonal by each
        of the two points it could belong to. Measured on a clean, evenly-jittered 200-point
        sphere: ~35-40% of points start out implicated in some disagreement, and it takes on
        the order of 50-100 perturbation rounds (not Cocone's typical single-digit count) to
        fully resolve -- larger ``neighborhood_size`` was measured not to help (the
        disagreement rate stayed roughly constant from 12 to 40 neighbors), consistent with
        this being an orientation-difference effect rather than a too-small-window boundary
        effect. When ``repair=True`` (the default), any point implicated in such a
        disagreement (or whose own local Delaunay call failed outright) is perturbed by a
        tiny bounded random shift and the whole neighborhood/basis/triangulation pipeline is
        recomputed, via ``pysurgery.geometry.perturbation.moser_tardos_repair``.

    Algorithm:
        1. If ``k`` is not given, auto-estimate it via
           ``local_pca_tangent_space_dimension``'s consensus (median) local dimension,
           rounded to the nearest integer and clamped to ``[1, ambient_dim]``.
        2. Compute every point's tangent basis (``local_pca_tangent_basis``, a single fixed
           ``k`` for every point).
        3. Compute every point's local star (``_compute_all_local_stars``, parallelized via
           ``ThreadPoolExecutor`` since each point's local Delaunay call is independent and
           releases the GIL).
        4. Reconcile into one global complex (``intersect_local_stars``); repair any
           resulting disagreement (or outright local-Delaunay failure) via
           ``moser_tardos_repair`` when ``repair=True``.

    Args:
        points: (N, D) array of point coordinates.
        k: Fixed tangent-space dimension for every point. ``None`` auto-estimates it (see
            Algorithm above).
        neighborhood_size: Number of nearest neighbors (excluding the point itself) used in
            each point's local triangulation. ``None`` defaults to
            ``min(N - 1, max(4 * k_or_estimate, 12))``.
        variance_threshold: Passed to the auto-``k`` estimator when ``k`` is None.
        repair: If True, resolve residual star-consistency conflicts via
            ``moser_tardos_repair`` rather than returning them as diagnostics only.
        max_repair_rounds: Passed through to ``moser_tardos_repair``. See the note above --
            this needs a substantially larger budget than Cocone's typical single-digit
            round count.
        perturbation_scale: Passed through to ``moser_tardos_repair``. ``None`` picks
            ``0.01`` times the median nearest-neighbor spacing among ``points`` (see
            ``cocone_reconstruction``'s docstring for the same reasoning), which was
            measured to be enough for a sphere but not for a torus sample: unlike Cocone's
            binary admit/reject reach criterion, a diagonal-flip disagreement here is
            resolved by which side of a near-cocircular configuration a point ends up on,
            and the default scale was measured to leave a torus sample oscillating rather
            than converging, while 3x that default converged the same sample within a few
            dozen rounds. Callers on harder (higher-curvature or more elongated) samples
            should try explicitly widening this rather than only raising
            ``max_repair_rounds``.
        seed: Passed through to ``moser_tardos_repair``.

    Returns:
        TangentialComplexResult: The reconstructed simplices and full diagnostics.

    Raises:
        ReconstructionRepairError: If ``repair=True`` and the conflicts do not resolve
            within ``max_repair_rounds`` (propagated from ``moser_tardos_repair``).

    Use When:
        - Reconstructing a genuine ``k``-manifold sampled in R^d with ``k`` substantially
          smaller than ``d`` (e.g. a curved data manifold inside a high-dimensional
          activation space), where Cocone's codimension-1-specific pole/normal estimation
          does not apply.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2:
        raise ValueError("tangential_complex_reconstruction requires a 2D (N, D) array.")
    n, d = pts.shape

    if k is None:
        probe_neighborhood = neighborhood_size if neighborhood_size is not None else min(n - 1, 12)
        dim_result = local_pca_tangent_space_dimension(
            pts, k=max(2, probe_neighborhood), variance_threshold=variance_threshold
        )
        k = int(round(dim_result.global_dimension)) if np.isfinite(dim_result.global_dimension) else 1
    k = int(max(1, min(k, d)))

    if neighborhood_size is None:
        neighborhood_size = min(n - 1, max(4 * k, 12))
    neighborhood_size = int(min(neighborhood_size, n - 1))
    if neighborhood_size + 1 < k:
        raise ValueError(
            f"neighborhood_size ({neighborhood_size}) + 1 must be >= k ({k}); "
            "increase neighborhood_size or lower k."
        )

    if perturbation_scale is None:
        tree = cKDTree(pts)
        nn_dist, _ = tree.query(pts, k=2)
        perturbation_scale = 0.01 * float(np.median(nn_dist[:, 1]))

    state: dict = {}

    def _detect_conflicts(cur_pts):
        basis_result = local_pca_tangent_basis(cur_pts, k=k, neighborhood_size=neighborhood_size)
        local_stars, failed_points = _compute_all_local_stars(cur_pts, basis_result.bases, neighborhood_size)
        surviving, stats = intersect_local_stars(local_stars)
        state["basis_result"] = basis_result
        state["local_stars"] = local_stars
        state["surviving"] = surviving
        state["stats"] = stats
        state["failed_points"] = failed_points
        conflict_vertices = _tangential_conflict_vertices(local_stars, surviving, failed_points)
        return [{"indices": conflict_vertices}] if conflict_vertices else []

    def _rebuild_local(cur_pts, offending):
        pass  # _detect_conflicts always recomputes the full (independent-per-point) pipeline fresh

    repair_rounds_used = 0
    if repair:
        repair_result = moser_tardos_repair(
            pts, _detect_conflicts, _rebuild_local,
            max_rounds=max_repair_rounds, perturbation_scale=perturbation_scale, seed=seed,
            stage="tangential_star_consistency",
        )
        pts = repair_result.points
        repair_rounds_used = repair_result.rounds_used
    else:
        _detect_conflicts(pts)

    surviving = state["surviving"]
    stats = state["stats"]
    diagnostics = [f"{stats['n_inconsistent']} candidate simplex/simplices dropped by disagreement."]
    if state["failed_points"]:
        diagnostics.append(
            f"{len(state['failed_points'])} point(s) had a degenerate local Delaunay call."
        )

    return TangentialComplexResult(
        simplices=[tuple(sorted(int(x) for x in s)) for s in surviving],
        points=pts,
        k=k,
        neighborhood_size=neighborhood_size,
        n_candidates=stats["n_candidates"],
        unresolved_points=sorted(int(v) for v in state["failed_points"]),
        repair_rounds_used=repair_rounds_used,
        diagnostics=diagnostics,
    )


__all__ = [
    "TangentialComplexResult",
    "tangential_complex_reconstruction",
]
