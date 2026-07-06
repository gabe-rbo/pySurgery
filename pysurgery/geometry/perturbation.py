"""Shared repair infrastructure for manifold-reconstruction constructors.

Overview:
    Both the Cocone/Tight Cocone surface reconstructor and the tangential Delaunay complex
    builder face the same two recurring problems: (1) deciding whether a vertex's local
    candidate-simplex neighborhood is already combinatorially clean, and (2) resolving
    disagreements -- either among a single vertex's own candidates, or between several
    vertices' independently-computed local decisions -- by nudging only the specific points
    involved, rather than a global retry. This module holds that shared machinery so it is
    implemented and tested once, not duplicated between the two constructors.

Key Concepts:
    - **Single-cycle criterion**: a vertex's link graph (candidate neighbors as nodes,
      candidate triangles as edges) is combinatorially clean exactly when it is connected
      with every node at degree 2 -- the same criterion that makes
      ``SimplicialComplex.is_homology_manifold`` exact at dimension <= 2 (a connected,
      max-degree-2 graph with first Betti number 1 cannot be a path, since paths have
      Betti number 0, so it is forced to be a single simple cycle). ``is_single_cycle`` is a
      standalone, independently-tested implementation of this same criterion via plain
      graph connectivity/degree, not a refactor of ``is_homology_manifold`` itself.
    - **Global star consistency**: a simplex is only real if every one of its vertices,
      computed independently, agrees it belongs -- ``intersect_local_stars`` implements this
      rule once, shared by Cocone's prune-and-walk output and the tangential complex's
      per-point local stars.
    - **Moser-Tardos-style repair**: when the above leaves genuine conflicts,
      ``moser_tardos_repair`` perturbs *only* the points named in a live conflict by a small
      bounded random shift and re-checks, bounded by a round budget -- "principled, targeted,
      terminating local resampling" in the spirit of the Moser-Tardos algorithmic Lovász
      Local Lemma, not a full reproduction of the Boissonnat-Dyer-Ghosh certified
      delta-protection proof (a research-paper-level undertaking on its own). This replaces
      ad hoc global joggling (``qhull_options="QJ"``) for the new reconstruction methods only.

Common Workflows:
    1. **Vertex-local cleanliness check** -> ``is_single_cycle(edges)`` as the fast path
       before falling back to an angular prune-and-walk.
    2. **Cross-vertex reconciliation** -> ``intersect_local_stars(per_vertex_local_simplices)``.
    3. **Conflict resolution** -> ``moser_tardos_repair(points, detect_conflicts,
       rebuild_local)``, raising ``ReconstructionRepairError`` on non-convergence.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..core.exceptions import ReconstructionRepairError


def is_single_cycle(edges) -> tuple[bool, Optional[list[int]]]:
    """Decide whether a set of undirected edges forms a single simple cycle.

    What is Being Computed?:
        Whether the graph ``(nodes, edges)`` is connected with every node at degree exactly
        2 -- equivalently, a single simple cycle through every node that appears in ``edges``.

    Algorithm:
        1. Deduplicate ``edges`` to a set of 2-element node pairs (self-loops dropped).
        2. Build the adjacency list; if any node's degree is not exactly 2, fail immediately.
        3. Walk the graph starting from the smallest node, at each step moving to the
           neighbor that isn't where we came from, until the walk either returns to the
           start (recording the full cyclic order) or revisits a non-start node first (which
           can only happen if the edges form more than one disjoint cycle).
        4. Succeed only if the walk closes back to the start *and* visited every node --
           this rules out the graph being a disjoint union of two or more smaller cycles,
           each individually degree-2-everywhere but not spanning.

    Preserved Invariants:
        This is the same criterion that makes ``SimplicialComplex.is_homology_manifold``
        exact (genuine PL-manifold, not merely homology-manifold) at complex dimension <= 2:
        connected + max degree 2 + first Betti number 1 forces a single simple cycle, since a
        connected degree-<=2 graph is either a path (Betti number 0) or a cycle.

    Args:
        edges: An iterable of 2-element node-id pairs (any hashable, orderable type),
            representing an undirected graph.

    Returns:
        tuple[bool, list[int] | None]: ``(True, cyclic_order)`` if the edges form a single
        simple cycle (``cyclic_order`` lists the nodes in walk order, starting from the
        smallest); otherwise ``(False, None)``.

    Example:
        >>> is_single_cycle([(0, 1), (1, 2), (2, 0)])
        (True, [0, 1, 2])
        >>> is_single_cycle([(0, 1), (2, 3)])
        (False, None)
    """
    unique_edges = {frozenset(e) for e in edges if len(frozenset(e)) == 2}
    if not unique_edges:
        return False, None

    adjacency: dict = defaultdict(list)
    for edge in unique_edges:
        u, v = tuple(edge)
        adjacency[u].append(v)
        adjacency[v].append(u)

    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        return False, None

    nodes = sorted(adjacency.keys())
    for node in nodes:
        adjacency[node] = sorted(adjacency[node])

    start = nodes[0]
    order = [start]
    visited = {start}
    prev, current = None, start
    while True:
        a, b = adjacency[current]
        nxt = b if a == prev else a
        if nxt == start:
            break
        if nxt in visited:
            return False, None
        visited.add(nxt)
        order.append(nxt)
        prev, current = current, nxt

    if len(visited) != len(nodes):
        return False, None
    return True, order


def is_single_path_or_cycle(edges) -> tuple[bool, Optional[list[int]]]:
    """Decide whether a set of undirected edges forms one connected simple path or cycle.

    What is Being Computed?:
        Whether the graph ``(nodes, edges)`` is connected with every node at degree <= 2 --
        equivalently, a single simple path (two degree-1 endpoints) or a single simple cycle
        (every node degree 2) spanning every node that appears in ``edges``. Both shapes are
        valid, non-branching vertex links: a cycle is an interior point of a closed surface, a
        path is a legitimate *boundary* point (see ``pysurgery.geometry.reconstruction
        .prune_and_walk``'s docstring for why a boundary is not a defect). Only a disconnected
        graph or a node of degree > 2 (a pinch or a branch -- a genuine non-manifold point) is
        rejected. ``is_single_cycle`` is the strictly-closed special case of this same
        family; this function is the general one used wherever an open boundary is an
        acceptable outcome, not only a closed surface.

    Algorithm:
        1. Deduplicate ``edges`` to a set of 2-element node pairs (self-loops dropped).
        2. Build the adjacency list; if any node's degree exceeds 2, fail immediately.
        3. Walk from a degree-1 endpoint if one exists, otherwise (no endpoint -- every node
           already degree exactly 2, i.e. a candidate pure cycle) from the smallest node;
           at each step move to the neighbor that isn't where we came from, until the walk
           runs out of neighbors (a path endpoint) or returns to the start (a closed cycle).
        4. Succeed only if the walk visited every node -- this rules out the graph being a
           disjoint union of two or more smaller paths/cycles, each individually
           degree-<=2-everywhere but not spanning.

    Args:
        edges: An iterable of 2-element node-id pairs (any hashable, orderable type),
            representing an undirected graph.

    Returns:
        tuple[bool, list[int] | None]: ``(True, walk_order)`` if the edges form a single
        simple path or cycle (``walk_order`` lists the nodes in walk order, starting from an
        endpoint when the shape is a path); otherwise ``(False, None)``.

    Example:
        >>> is_single_path_or_cycle([(0, 1), (1, 2), (2, 0)])
        (True, [0, 1, 2])
        >>> is_single_path_or_cycle([(0, 1), (1, 2)])
        (True, [0, 1, 2])
        >>> is_single_path_or_cycle([(0, 1), (2, 3)])
        (False, None)
    """
    unique_edges = {frozenset(e) for e in edges if len(frozenset(e)) == 2}
    if not unique_edges:
        return False, None

    adjacency: dict = defaultdict(list)
    for edge in unique_edges:
        u, v = tuple(edge)
        adjacency[u].append(v)
        adjacency[v].append(u)

    if any(len(neighbors) > 2 for neighbors in adjacency.values()):
        return False, None

    nodes = sorted(adjacency.keys())
    for node in nodes:
        adjacency[node] = sorted(adjacency[node])

    endpoints = [node for node in nodes if len(adjacency[node]) == 1]
    start = endpoints[0] if endpoints else nodes[0]
    order = [start]
    visited = {start}
    prev, current = None, start
    while True:
        candidates = [n for n in adjacency[current] if n != prev]
        if not candidates:
            break  # ran out of neighbors: a path endpoint
        nxt = candidates[0]
        if nxt == start:
            break  # closed back to the start: a cycle
        if nxt in visited:
            return False, None
        visited.add(nxt)
        order.append(nxt)
        prev, current = current, nxt

    if len(visited) != len(nodes):
        return False, None
    return True, order


def intersect_local_stars(
    per_vertex_local_simplices: dict,
) -> tuple[set, dict]:
    """Reconcile independently-computed per-vertex local simplex sets into one global complex.

    What is Being Computed?:
        The rule that a candidate simplex is only real if every one of its own vertices,
        computed independently, agrees it belongs -- the analogue of GUDHI's
        ``num_inconsistent_simplices()`` diagnostic, computed directly rather than through a
        third-party dependency.

    Args:
        per_vertex_local_simplices: Mapping ``vertex_id -> set[frozenset[int]]`` of the
            simplices that vertex's own local computation (Cocone's prune-and-walk, or the
            tangential complex's local tangent-plane Delaunay star) believes it belongs to.

    Returns:
        tuple[set[frozenset[int]], dict]: ``(surviving_simplices, stats)`` where ``stats``
        has keys ``"n_candidates"`` (total distinct simplices seen by at least one vertex)
        and ``"n_inconsistent"`` (candidates dropped because at least one of their vertices
        did not independently keep them).

    Use When:
        - Merging Cocone's per-vertex prune-and-walk output into one global surface.
        - Merging the tangential complex's per-point local stars into one global complex.
    """
    all_simplices: set = set()
    for local in per_vertex_local_simplices.values():
        all_simplices.update(local)

    surviving = set()
    n_inconsistent = 0
    for simplex in all_simplices:
        if all(simplex in per_vertex_local_simplices.get(v, ()) for v in simplex):
            surviving.add(simplex)
        else:
            n_inconsistent += 1

    return surviving, {"n_candidates": len(all_simplices), "n_inconsistent": n_inconsistent}


class PerturbationRepairResult(BaseModel):
    """Outcome of a ``moser_tardos_repair`` run.

    Overview:
        Bundles the (possibly perturbed) point coordinates with a record of how the repair
        converged, so callers can both use the repaired points and report on the process.

    Attributes:
        converged (bool): Always True on a normal return (non-convergence raises instead).
        rounds_used (int): Number of perturbation rounds actually performed.
        points (np.ndarray): The (possibly perturbed) point coordinates.
        diagnostics (list[str]): Human-readable per-round messages.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    converged: bool
    rounds_used: int
    points: np.ndarray
    diagnostics: list[str] = Field(default_factory=list)


def moser_tardos_repair(
    points: np.ndarray,
    detect_conflicts: Callable[[np.ndarray], list],
    rebuild_local: Callable[[np.ndarray, list], None],
    *,
    max_rounds: int = 50,
    perturbation_scale: float = 1e-6,
    seed: int = 0,
    stage: str = "reconstruction",
) -> PerturbationRepairResult:
    """Resolve local geometric conflicts via bounded, targeted random perturbation.

    What is Being Computed?:
        A terminating repair loop in the spirit of the Moser-Tardos algorithmic Lovász Local
        Lemma: at each round, only the points named by a currently-live conflict are
        perturbed (a small, independent Gaussian shift; the rest of the point cloud is left
        untouched), the caller re-derives whatever local structure depends on those points,
        and conflicts are re-detected. This directly replaces ad hoc global joggling
        (``qhull_options="QJ"``, which perturbs every point with no targeting and no
        termination/quality guarantee) for the reconstruction methods that use it.

    Algorithm:
        1. ``conflicts = detect_conflicts(points)``; if empty, already consistent, return.
        2. Otherwise, collect the union of point indices named across all conflicts. If a
           round budget remains, perturb exactly those points by
           ``rng.normal(size=D) * perturbation_scale`` (a fresh, round-indexed
           ``np.random.default_rng(seed + round)`` -- deterministic and reproducible, no
           global RNG state), call ``rebuild_local(points, offending_indices)`` so the
           caller can recompute whatever local decisions depend on the perturbed points, and
           repeat from step 1.
        3. If the round budget is exhausted while conflicts remain (or a conflict names no
           offending indices, so there is nothing left to target), raise
           ``ReconstructionRepairError`` with the residual conflicts and offending indices
           attached, rather than looping forever or returning an inconsistent result silently.

    Preserved Invariants:
        Perturbations accumulate additively across rounds and are bounded by
        ``rounds_used * perturbation_scale`` in expectation per coordinate -- the process
        cannot drift the data arbitrarily far from its original position, and it always
        either converges within ``max_rounds`` or raises.

    Args:
        points: (N, D) array of point coordinates. Not mutated; a copy is perturbed and
            returned.
        detect_conflicts: Callable ``points -> list_of_conflicts``. Each conflict is
            expected to be a mapping (or object) exposing an ``"indices"`` key (or
            attribute) listing the offending point indices; an empty list means "no
            conflicts, already consistent."
        rebuild_local: Callable ``(points, offending_indices) -> None``, invoked after each
            perturbation so the caller can update whatever cached local structure (poles,
            candidate simplices, local stars) depends on the perturbed points, in place or
            via caller-managed state, before ``detect_conflicts`` is called again.
        max_rounds: Maximum number of perturbation rounds before giving up.
        perturbation_scale: Standard deviation of the per-coordinate Gaussian perturbation,
            in the same units as ``points`` -- callers should scale this to their own data
            (e.g. a small fraction of the typical nearest-neighbor spacing).
        seed: Base seed for the deterministic, round-indexed RNG.
        stage: Label attached to any raised ``ReconstructionRepairError``, identifying which
            repair loop this call represents (e.g. ``"prune_and_walk"``,
            ``"star_consistency"``).

    Returns:
        PerturbationRepairResult: The repaired points and a convergence record.

    Raises:
        ReconstructionRepairError: If conflicts remain after ``max_rounds`` rounds, or a
            conflict cannot be targeted because it names no offending indices.
    """
    pts = np.array(points, dtype=np.float64, copy=True)
    diagnostics: list[str] = []

    def _indices_of(conflict) -> list:
        if isinstance(conflict, dict):
            return list(conflict.get("indices", []))
        return list(getattr(conflict, "indices", []))

    conflicts = detect_conflicts(pts)
    round_idx = 0
    while conflicts:
        offending = sorted({int(i) for c in conflicts for i in _indices_of(c)})
        if not offending:
            raise ReconstructionRepairError(
                message="Conflicts were reported with no offending point indices; "
                "cannot target a repair.",
                stage=stage,
                rounds_attempted=round_idx,
                max_rounds=max_rounds,
                residual_conflicts=conflicts,
                offending_indices=[],
            )
        if round_idx >= max_rounds:
            raise ReconstructionRepairError(
                stage=stage,
                rounds_attempted=round_idx,
                max_rounds=max_rounds,
                residual_conflicts=conflicts,
                offending_indices=offending,
            )
        rng = np.random.default_rng(seed + round_idx)
        pts[offending] += rng.normal(size=(len(offending), pts.shape[1])) * perturbation_scale
        rebuild_local(pts, offending)
        diagnostics.append(
            f"Round {round_idx}: perturbed {len(offending)} point(s) to resolve "
            f"{len(conflicts)} conflict(s)."
        )
        round_idx += 1
        conflicts = detect_conflicts(pts)

    return PerturbationRepairResult(
        converged=True, rounds_used=round_idx, points=pts, diagnostics=diagnostics
    )


__all__ = [
    "is_single_cycle",
    "is_single_path_or_cycle",
    "intersect_local_stars",
    "PerturbationRepairResult",
    "moser_tardos_repair",
]
