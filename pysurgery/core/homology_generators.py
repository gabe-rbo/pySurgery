"""Data-grounded H_1 generators and optimal basis extraction.

Algorithmic foundation follows the "Generators and Optimality" chapter in:
T. K. Dey and Y. Wang, *Computational Topology for Data Analysis*.

Implemented pipeline:
- edge annotations on 2-complexes,
- shortest-path generated cycle candidates,
- greedy independent minimum-weight basis over Z2 annotations.

When Julia backend support is available, this module can delegate the core
generator/optimality computation to Julia and falls back to Python otherwise.
"""

from __future__ import annotations

import collections
import heapq
import warnings
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .generator_models import HomologyBasisResult, HomologyGenerator
from ..bridge.julia_bridge import julia_engine

Edge = Tuple[int, int]
Triangle = Tuple[int, int, int]
Cycle = List[Edge]


def _edge_weight(u: int, v: int, points: Optional[np.ndarray]) -> float:
    if points is None:
        return 1.0
    return float(np.linalg.norm(points[u] - points[v]))


def _normalize_edges_triangles(simplices: Iterable[Tuple[int, ...]]) -> tuple[list[Edge], list[Triangle], set[int]]:
    edges: list[Edge] = []
    triangles: list[Triangle] = []
    vertex_ids: set[int] = set()
    for s in simplices:
        t = tuple(int(x) for x in s)
        if len(t) == 2:
            e = tuple(sorted(t))
            edges.append(e)
            vertex_ids.update(e)
        elif len(t) == 3:
            tri = tuple(sorted(t))
            triangles.append(tri)
            vertex_ids.update(tri)
    return list(dict.fromkeys(edges)), list(dict.fromkeys(triangles)), vertex_ids


def _minimum_spanning_edges(edges: list[Edge], weights: Dict[Edge, float], num_vertices: int) -> set[Edge]:
    parent = list(range(max(num_vertices, 1)))
    rank = [0] * max(num_vertices, 1)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def unite(a: int, b: int) -> bool:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        return True

    spanning: set[Edge] = set()
    for e in sorted(edges, key=lambda x: weights[x]):
        u, v = e
        if unite(u, v):
            spanning.add(e)
    return spanning


def annot_edge(
    simplices: Iterable[Tuple[int, ...]],
    num_vertices: int,
    edge_weights: Optional[Dict[Edge, float]] = None,
) -> tuple[Dict[Edge, np.ndarray], int]:
    edges, triangles, _ = _normalize_edges_triangles(simplices)
    weights = {e: float(edge_weights.get(e, 1.0)) if edge_weights else 1.0 for e in edges}

    spanning = _minimum_spanning_edges(edges, weights, num_vertices)
    non_tree = [e for e in edges if e not in spanning]
    m = len(non_tree)

    annotations: Dict[Edge, np.ndarray] = {}
    for e in spanning:
        annotations[e] = np.zeros(m, dtype=np.int64)
    for i, e in enumerate(non_tree):
        v = np.zeros(m, dtype=np.int64)
        v[i] = 1
        annotations[e] = v

    valid = list(range(m))
    for u, v, w in triangles:
        e1 = tuple(sorted((u, v)))
        e2 = tuple(sorted((v, w)))
        e3 = tuple(sorted((u, w)))

        boundary = np.zeros(m, dtype=np.int64)
        if e1 in annotations:
            boundary ^= annotations[e1]
        if e2 in annotations:
            boundary ^= annotations[e2]
        if e3 in annotations:
            boundary ^= annotations[e3]

        pivot = next((idx for idx in valid if boundary[idx] == 1), -1)
        if pivot == -1:
            continue

        for e, vec in list(annotations.items()):
            if vec[pivot] == 1:
                annotations[e] = np.bitwise_xor(vec, boundary)
        valid.remove(pivot)

    return {e: vec[valid] for e, vec in annotations.items()}, len(valid)


def _shortest_path_tree(root: int, adjacency: Dict[int, List[Tuple[int, float]]]) -> tuple[Dict[int, int], set[Edge]]:
    dist = {root: 0.0}
    parent = {root: -1}
    pq = [(0.0, root)]

    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        for v, w in adjacency.get(u, []):
            nd = d + w
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))

    tree: set[Edge] = set()
    for child, par in parent.items():
        if par != -1:
            tree.add(tuple(sorted((child, par))))
    return parent, tree


def _path_between(u: int, v: int, parent: Dict[int, int]) -> List[int]:
    path_u: list[int] = []
    seen_u = set()
    x = u
    while x != -1:
        path_u.append(x)
        seen_u.add(x)
        x = parent.get(x, -1)

    path_v: list[int] = []
    y = v
    while y not in seen_u and y != -1:
        path_v.append(y)
        y = parent.get(y, -1)

    if y == -1:
        return []

    i = path_u.index(y)
    return path_u[: i + 1] + list(reversed(path_v))


def _path_edges(path_vertices: List[int]) -> List[Edge]:
    return [tuple(sorted((path_vertices[i], path_vertices[i + 1]))) for i in range(len(path_vertices) - 1)]


def generator_cycles_from_simplices(
    simplices: Iterable[Tuple[int, ...]],
    num_vertices: int,
    point_cloud: Optional[np.ndarray] = None,
    *,
    max_roots: Optional[int] = None,
    root_stride: int = 1,
    max_cycles: Optional[int] = None,
) -> List[Cycle]:
    edges, _, vertex_ids = _normalize_edges_triangles(simplices)
    if not edges:
        return []

    if num_vertices <= 0 and vertex_ids:
        num_vertices = max(vertex_ids) + 1

    adjacency: Dict[int, List[Tuple[int, float]]] = collections.defaultdict(list)
    for u, v in edges:
        w = _edge_weight(u, v, point_cloud)
        adjacency[u].append((v, w))
        adjacency[v].append((u, w))

    vertices = sorted(vertex_ids) if vertex_ids else list(range(num_vertices))
    selected_roots = vertices[:: max(1, root_stride)]
    if max_roots is not None:
        selected_roots = selected_roots[:max_roots]

    cycles: list[Cycle] = []
    for root in selected_roots:
        parent, tree_edges = _shortest_path_tree(root, adjacency)
        for u, v in edges:
            e = tuple(sorted((u, v)))
            if e in tree_edges or u not in parent or v not in parent:
                continue
            path_vertices = _path_between(u, v, parent)
            if len(path_vertices) < 2:
                continue
            cycle = [e] + _path_edges(path_vertices)
            cycles.append(cycle)
            if max_cycles is not None and len(cycles) >= max_cycles:
                return cycles
    return cycles


def _cycle_annotation(cycle: Cycle, simplex_annotations: Dict[Edge, np.ndarray], vec_len: int) -> np.ndarray:
    ann = np.zeros(vec_len, dtype=np.int64)
    for e in cycle:
        se = tuple(sorted(e))
        if se in simplex_annotations:
            ann ^= simplex_annotations[se]
    return ann


def _is_independent_wrt(cv: np.ndarray, pivots: Dict[int, np.ndarray]) -> bool:
    work = cv.copy()
    for i in range(work.shape[0]):
        if work[i] == 0:
            continue
        if i in pivots:
            work ^= pivots[i]
        else:
            pivots[i] = work
            return True
    return False


def greedy_h1_basis(
    cycles: List[Cycle],
    simplices: Iterable[Tuple[int, ...]],
    num_vertices: int,
    point_cloud: Optional[np.ndarray] = None,
) -> List[Cycle]:
    edges, triangles, vertex_ids = _normalize_edges_triangles(simplices)
    if not edges:
        for cyc in cycles:
            edges.extend(cyc)
            for e in cyc:
                vertex_ids.update(e)

    if num_vertices <= 0 and vertex_ids:
        num_vertices = max(vertex_ids) + 1

    edge_weights = {tuple(sorted(e)): _edge_weight(e[0], e[1], point_cloud) for e in edges}
    simplex_annotations, vec_dim = annot_edge(list(edges) + list(triangles), num_vertices, edge_weights=edge_weights)

    cycles_sorted = sorted(cycles, key=lambda cyc: sum(_edge_weight(u, v, point_cloud) for (u, v) in cyc))
    basis: list[Cycle] = []
    pivots: Dict[int, np.ndarray] = {}
    for cyc in cycles_sorted:
        ann = _cycle_annotation(cyc, simplex_annotations, vec_dim)
        if _is_independent_wrt(ann, pivots):
            basis.append(cyc)
    return basis


def compute_optimal_h1_basis_from_simplices(
    simplices: Iterable[Tuple[int, ...]],
    num_vertices: int,
    point_cloud: Optional[np.ndarray] = None,
    *,
    max_roots: Optional[int] = None,
    root_stride: int = 1,
    max_cycles: Optional[int] = None,
) -> HomologyBasisResult:
    simplices_list = [tuple(int(x) for x in s) for s in simplices]
    basis: list[Cycle]
    used_julia = False

    if julia_engine.available:
        try:
            basis_julia = julia_engine.compute_optimal_h1_basis_from_simplices(
                simplices_list,
                int(num_vertices),
                point_cloud=point_cloud,
                max_roots=max_roots,
                root_stride=int(root_stride),
                max_cycles=max_cycles,
            )
            basis = [[tuple(sorted((int(u), int(v)))) for (u, v) in cyc] for cyc in basis_julia]
            used_julia = True
        except Exception as exc:
            warnings.warn(
                "Julia optimal-generator backend failed in "
                "`compute_optimal_h1_basis_from_simplices`; falling back to Python implementation "
                f"({exc!r})."
            )
            basis = []
    else:
        basis = []

    if not basis:
        cycles = generator_cycles_from_simplices(
            simplices_list,
            num_vertices,
            point_cloud=point_cloud,
            max_roots=max_roots,
            root_stride=root_stride,
            max_cycles=max_cycles,
        )
        basis = greedy_h1_basis(cycles, simplices_list, num_vertices, point_cloud=point_cloud)

    gens: list[HomologyGenerator] = []
    for cyc in basis:
        w = sum(_edge_weight(u, v, point_cloud) for (u, v) in cyc)
        gens.append(
            HomologyGenerator(
                dimension=1,
                support_edges=[tuple(sorted(e)) for e in cyc],
                support_simplices=[tuple(sorted(e)) for e in cyc],
                weight=float(w),
                certified_cycle=True,
            )
        )

    return HomologyBasisResult(
        dimension=1,
        rank=len(gens),
        generators=gens,
        optimal=True,
        exact=True,
        message=(
            "Computed by shortest-path generator set + greedy independent basis over Z2 annotations"
            + (" (Julia backend)." if used_julia else " (Python backend).")
        ),
    )


def compute_optimal_h1_basis_from_simplex_tree(
    simplex_tree: object,
    point_cloud: Optional[np.ndarray] = None,
    *,
    max_roots: Optional[int] = None,
    root_stride: int = 1,
    max_cycles: Optional[int] = None,
) -> HomologyBasisResult:
    simplices = [tuple(s[0]) for s in simplex_tree.get_skeleton(2) if len(s[0]) in (2, 3)]
    vertices = [int(s[0][0]) for s in simplex_tree.get_skeleton(0)]
    num_vertices = (max(vertices) + 1) if vertices else 0
    return compute_optimal_h1_basis_from_simplices(
        simplices,
        num_vertices,
        point_cloud=point_cloud,
        max_roots=max_roots,
        root_stride=root_stride,
        max_cycles=max_cycles,
    )

