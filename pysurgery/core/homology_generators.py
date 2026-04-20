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
import itertools
import warnings
from typing import TYPE_CHECKING, Dict, Iterable, List, Literal, Optional, Tuple

if TYPE_CHECKING:
    from .complexes import SimplicialComplex

import numpy as np
import scipy.sparse as sp

from .generator_models import HomologyBasisResult, HomologyGenerator
from ..bridge.julia_bridge import julia_engine

Edge = Tuple[int, int]
Triangle = Tuple[int, int, int]
Cycle = List[Edge]


def _all_simplices_by_dim(
    simplices: Iterable[Tuple[int, ...]],
) -> dict[int, list[tuple[int, ...]]]:
    """Build simplicial closure grouped by dimension from input simplices."""
    by_dim_set: dict[int, set[tuple[int, ...]]] = {}
    for s in simplices:
        t = tuple(sorted(int(x) for x in s))
        if len(t) == 0:
            continue
        # Work with the simplicial closure so boundaries are represented on all faces.
        for r in range(1, len(t) + 1):
            d = r - 1
            if d not in by_dim_set:
                by_dim_set[d] = set()
            for face in itertools.combinations(t, r):
                by_dim_set[d].add(tuple(face))
    
    return {d: sorted(list(faces)) for d, faces in by_dim_set.items()}


def _infer_num_vertices(simplices: Iterable[Tuple[int, ...]], num_vertices: int) -> int:
    """Infer vertex count from simplices when not supplied by the caller."""
    if num_vertices > 0:
        return int(num_vertices)
    max_v = -1
    for s in simplices:
        for x in s:
            max_v = max(max_v, int(x))
    return max_v + 1 if max_v >= 0 else 0


def _boundary_mod2_matrix(
    source: list[tuple[int, ...]],
    target: list[tuple[int, ...]],
) -> np.ndarray:
    """
    Builds mod-2 boundary matrix with optional Julia acceleration.
    Falls back to pure Python if Julia unavailable or fails.
    """
    if not target or not source:
        return np.zeros((len(target), len(source)), dtype=np.int64)

    # Prefer Julia acceleration whenever available.
    if julia_engine.available:
        try:
            payload = julia_engine.compute_boundary_mod2_matrix(source, target)
            return sp.csr_matrix(
                (payload["data"], (payload["rows"], payload["cols"])),
                shape=(payload["m"], payload["n"]),
                dtype=np.int64,
            ).toarray()
        except Exception as e:
            warnings.warn(
                f"Topological Hint: Julia mod2 boundary assembly failed ({e!r}). "
                "Falling back to pure Python."
            )

    # Python fallback
    t_idx = {t: i for i, t in enumerate(target)}
    mat = np.zeros((len(target), len(source)), dtype=np.int64)
    for j, s in enumerate(source):
        for i_drop in range(len(s)):
            face = s[:i_drop] + s[i_drop + 1 :]
            row = t_idx.get(face)
            if row is not None:
                mat[row, j] ^= 1
    return mat


def _rref_mod2(A: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Compute row-reduced echelon form over GF(2) with vectorized NumPy operations."""
    M = (np.asarray(A, dtype=np.int8) & 1).copy()
    m, n = M.shape
    row = 0
    pivots: list[int] = []
    for col in range(n):
        if row >= m:
            break
        # Fast vectorized pivot search
        pivot = int(np.argmax(M[row:, col])) + row
        if M[pivot, col] == 0:
            continue
            
        if pivot != row:
            M[[row, pivot], :] = M[[pivot, row], :]
            
        # Fast vectorized row elimination
        mask = M[:, col] == 1
        mask[row] = False
        if np.any(mask):
            M[mask, :] ^= M[row, :]
            
        pivots.append(col)
        row += 1
        
    return M.astype(np.int64), pivots


def _nullspace_basis_mod2(A: np.ndarray) -> list[np.ndarray]:
    """Return a basis for the nullspace of `A` over GF(2)."""
    # A is m x n over F2; basis vectors are in F2^n.
    _, n = A.shape
    rref, pivots = _rref_mod2(A)
    pivot_set = set(pivots)
    free_cols = [j for j in range(n) if j not in pivot_set]
    basis: list[np.ndarray] = []
    for free in free_cols:
        v = np.zeros(n, dtype=np.int64)
        v[free] = 1
        for i, col in enumerate(pivots):
            v[col] = rref[i, free] & 1
        basis.append(v)
    return basis


def _rank_mod2(A: np.ndarray) -> int:
    """Compute matrix rank over GF(2)."""
    _, pivots = _rref_mod2(A)
    return len(pivots)


def _components_h0_generators(
    edges: list[Edge],
    num_vertices: int,
    point_cloud: Optional[np.ndarray] = None,
) -> HomologyBasisResult:
    """Compute H0 generators as one representative per connected component."""
    if num_vertices <= 0:
        return HomologyBasisResult(
            dimension=0,
            rank=0,
            generators=[],
            optimal=True,
            exact=True,
            message="No vertices found; H0 is trivial.",
        )

    parent = list(range(num_vertices))
    rank = [0] * num_vertices

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def unite(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    for u, v in edges:
        if u < 0 or u >= num_vertices or v < 0 or v >= num_vertices:
            raise ValueError(f"Topological invalidity: Edge ({u}, {v}) contains a vertex outside the bounds [0, {num_vertices - 1}].")
        unite(u, v)

    comps: dict[int, list[int]] = {}
    for v in range(num_vertices):
        r = find(v)
        comps.setdefault(r, []).append(v)

    gens: list[HomologyGenerator] = []
    for verts in comps.values():
        rep = min(verts)
        weight = 0.0
        if point_cloud is not None and len(verts) > 1:
            centroid = point_cloud[np.array(verts)].mean(axis=0)
            weight = float(np.linalg.norm(point_cloud[rep] - centroid))
        gens.append(
            HomologyGenerator(
                dimension=0,
                support_edges=[],
                support_simplices=[(int(rep),)],
                weight=weight,
                certified_cycle=True,
            )
        )
    gens.sort(key=lambda g: g.support_simplices[0][0] if g.support_simplices else -1)
    return HomologyBasisResult(
        dimension=0,
        rank=len(gens),
        generators=gens,
        optimal=True,
        exact=True,
        message="Computed H0 generators as connected-component representatives.",
    )


def _weight_k_chain(
    chain: np.ndarray,
    k_simplices: list[tuple[int, ...]],
    point_cloud: Optional[np.ndarray],
) -> float:
    """Compute a geometric/algebraic proxy weight for an active k-chain."""
    active = [k_simplices[i] for i, bit in enumerate(chain) if bit & 1]
    if not active:
        return 0.0
    if point_cloud is None:
        return float(len(active))
    total = 0.0
    for simplex in active:
        if len(simplex) <= 1:
            total += 1.0
            continue
        # Sum edge lengths inside each simplex as a geometric proxy.
        s = 0.0
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                u, v = simplex[i], simplex[j]
                s += float(np.linalg.norm(point_cloud[u] - point_cloud[v]))
        total += s
    return float(total)


def _independent_mod_image(v: np.ndarray, basis_cols: list[np.ndarray]) -> bool:
    """Check if `v` increases span rank modulo 2 over current basis columns."""
    if not basis_cols:
        return bool(np.any(v & 1))
    M = np.column_stack([*(b & 1 for b in basis_cols), v & 1]).astype(np.int64)
    r_prev = _rank_mod2(M[:, :-1])
    r_new = _rank_mod2(M)
    return r_new > r_prev


def _hk_generators_mod2(
    simplices: Iterable[Tuple[int, ...]],
    num_vertices: int,
    dimension: int,
    point_cloud: Optional[np.ndarray],
    mode: Literal["valid", "optimal"],
) -> HomologyBasisResult:
    """Compute H_k representatives over Z/2 via kernel/image quotient."""
    simplices_list = [tuple(int(x) for x in s) for s in simplices]
    by_dim = _all_simplices_by_dim(simplices_list)
    if dimension == 0:
        edges = [tuple(sorted(e)) for e in by_dim.get(1, []) if len(e) == 2]
        return _components_h0_generators(
            edges,
            _infer_num_vertices(simplices_list, num_vertices),
            point_cloud=point_cloud,
        )

    k_simplices = by_dim.get(dimension, [])
    km1_simplices = by_dim.get(dimension - 1, [])
    kp1_simplices = by_dim.get(dimension + 1, [])
    if not k_simplices:
        return HomologyBasisResult(
            dimension=dimension,
            rank=0,
            generators=[],
            optimal=(mode == "optimal"),
            exact=True,
            message=f"No {dimension}-simplices found; H_{dimension} generators are empty.",
        )

    d_k = _boundary_mod2_matrix(k_simplices, km1_simplices)
    d_kp1 = _boundary_mod2_matrix(kp1_simplices, k_simplices)
    z_basis = _nullspace_basis_mod2(d_k)
    if not z_basis:
        return HomologyBasisResult(
            dimension=dimension,
            rank=0,
            generators=[],
            optimal=(mode == "optimal"),
            exact=True,
            message=f"Kernel of d_{dimension} is trivial over Z/2.",
        )

    b_cols = [d_kp1[:, j].astype(np.int64) & 1 for j in range(d_kp1.shape[1])]
    z_candidates = z_basis[:]
    if mode == "optimal":
        z_candidates = sorted(
            z_candidates, key=lambda v: _weight_k_chain(v, k_simplices, point_cloud)
        )

    quotient_basis: list[np.ndarray] = []
    pivots: Dict[int, np.ndarray] = {}
    
    # 1. Pre-fill the incremental pivot dictionary with the image (boundaries of kp1)
    # This correctly represents the subspace we are quotienting by.
    for b in b_cols:
        _is_independent_wrt(b, pivots)
        
    # 2. Extract quotient basis in O(N^2) time per candidate
    for z in z_candidates:
        if _is_independent_wrt(z, pivots):
            quotient_basis.append(z)

    gens: list[HomologyGenerator] = []
    for z in quotient_basis:
        support = [k_simplices[i] for i, bit in enumerate(z) if bit & 1]
        support_edges: list[Edge] = []
        if dimension == 1:
            support_edges = [
                tuple(sorted((int(a), int(b))))
                for (a, b) in support
                if len((a, b)) == 2
            ]
        w = _weight_k_chain(z, k_simplices, point_cloud)
        gens.append(
            HomologyGenerator(
                dimension=dimension,
                support_edges=support_edges,
                support_simplices=[tuple(int(x) for x in s) for s in support],
                weight=float(w),
                certified_cycle=True,
            )
        )

    return HomologyBasisResult(
        dimension=dimension,
        rank=len(gens),
        generators=gens,
        optimal=(mode == "optimal"),
        exact=True,
        message=(
            f"Computed H_{dimension} representatives over Z/2 as ker(d_{dimension}) / im(d_{dimension + 1})"
            + (" using greedy small-support selection." if mode == "optimal" else ".")
        ),
    )


def _edge_weight(u: int, v: int, points: Optional[np.ndarray]) -> float:
    """Return edge length weight (or unit weight when no geometry is provided)."""
    if points is None:
        return 1.0
    return float(np.linalg.norm(points[u] - points[v]))


def _normalize_edges_triangles(
    simplices: Iterable[Tuple[int, ...]],
) -> tuple[list[Edge], list[Triangle], set[int]]:
    """Extract normalized edge/triangle lists plus observed vertex ids."""
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


def _minimum_spanning_edges(
    edges: list[Edge], weights: Dict[Edge, float], num_vertices: int
) -> set[Edge]:
    """Return Kruskal minimum-spanning-forest edges of the 1-skeleton."""
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
    """Compute edge annotations for cycle-space independence over Z/2."""
    edges, triangles, _ = _normalize_edges_triangles(simplices)
    weights = {
        e: float(edge_weights.get(e, 1.0)) if edge_weights else 1.0 for e in edges
    }

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


def _shortest_path_tree(
    root: int, adjacency: Dict[int, List[Tuple[int, float]]]
) -> tuple[Dict[int, int], set[Edge]]:
    """Build a shortest-path tree from one root using Dijkstra."""
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
    """Return tree-path vertices between two nodes from parent pointers."""
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
    """Convert a vertex path to normalized undirected edges."""
    return [
        tuple(sorted((path_vertices[i], path_vertices[i + 1])))
        for i in range(len(path_vertices) - 1)
    ]


def _cycle_weight(
    cycle: Cycle, edge_weights: Dict[Edge, float], point_cloud: Optional[np.ndarray]
) -> float:
    """Compute cycle weight using supplied edge weights or geometric fallback."""
    total = 0.0
    for u, v in cycle:
        e = tuple(sorted((u, v)))
        if e in edge_weights:
            total += edge_weights[e]
        else:
            total += _edge_weight(u, v, point_cloud)
    return float(total)


def generator_cycles_from_simplices(
    simplices: Iterable[Tuple[int, ...]],
    num_vertices: int,
    point_cloud: Optional[np.ndarray] = None,
    *,
    max_roots: Optional[int] = None,
    root_stride: int = 1,
    max_cycles: Optional[int] = None,
) -> List[Cycle]:
    """Generate candidate H1 cycles from simplices via shortest-path trees."""
    edges, _, vertex_ids = _normalize_edges_triangles(simplices)
    return _generator_cycles_from_normalized_edges(
        edges,
        vertex_ids,
        num_vertices,
        point_cloud=point_cloud,
        max_roots=max_roots,
        root_stride=root_stride,
        max_cycles=max_cycles,
    )


def _generator_cycles_from_normalized_edges(
    edges: list[Edge],
    vertex_ids: set[int],
    num_vertices: int,
    point_cloud: Optional[np.ndarray] = None,
    *,
    max_roots: Optional[int] = None,
    root_stride: int = 1,
    max_cycles: Optional[int] = None,
) -> List[Cycle]:
    """Generate cycle candidates from normalized edge lists."""
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


def _cycle_annotation(
    cycle: Cycle, simplex_annotations: Dict[Edge, np.ndarray], vec_len: int
) -> np.ndarray:
    """Assemble the annotation vector of a cycle by XOR-combining edge labels."""
    ann = np.zeros(vec_len, dtype=np.int64)
    for e in cycle:
        se = tuple(sorted(e))
        if se in simplex_annotations:
            ann ^= simplex_annotations[se]
    return ann


def _is_independent_wrt(cv: np.ndarray, pivots: Dict[int, np.ndarray]) -> bool:
    """Check independence against a pivot map over Z/2 and update pivots."""
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
    """Select a greedy independent H1 basis from cycle candidates."""
    edges, triangles, vertex_ids = _normalize_edges_triangles(simplices)
    return _greedy_h1_basis_from_normalized(
        cycles,
        edges,
        triangles,
        vertex_ids,
        num_vertices,
        point_cloud=point_cloud,
    )


def _greedy_h1_basis_from_normalized(
    cycles: List[Cycle],
    edges: list[Edge],
    triangles: list[Triangle],
    vertex_ids: set[int],
    num_vertices: int,
    point_cloud: Optional[np.ndarray] = None,
) -> List[Cycle]:
    """Internal greedy H1 basis selection on normalized complexes."""
    if not edges:
        for cyc in cycles:
            edges.extend(cyc)
            for e in cyc:
                vertex_ids.update(e)

    if num_vertices <= 0 and vertex_ids:
        num_vertices = max(vertex_ids) + 1

    edge_weights = {
        tuple(sorted(e)): _edge_weight(e[0], e[1], point_cloud) for e in edges
    }
    simplex_annotations, vec_dim = annot_edge(
        list(edges) + list(triangles), num_vertices, edge_weights=edge_weights
    )

    cycles_sorted = sorted(
        cycles, key=lambda cyc: _cycle_weight(cyc, edge_weights, point_cloud)
    )
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
    """Compute a data-grounded optimal H1 basis from simplices.

    Uses Julia's optimized backend when available, otherwise falls back to the
    Python shortest-path candidate + greedy-independence pipeline.
    """
    simplices_list = [tuple(int(x) for x in s) for s in simplices]
    basis: Optional[list[Cycle]] = None
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
            basis = [
                [tuple(sorted((int(u), int(v)))) for (u, v) in cyc]
                for cyc in basis_julia
            ]
            used_julia = True
        except Exception as exc:
            warnings.warn(
                "Julia optimal-generator backend failed in "
                "`compute_optimal_h1_basis_from_simplices`; falling back to Python implementation "
                f"({exc!r})."
            )
    if basis is None:
        edges, triangles, vertex_ids = _normalize_edges_triangles(simplices_list)
        cycles = _generator_cycles_from_normalized_edges(
            edges,
            vertex_ids,
            num_vertices,
            point_cloud=point_cloud,
            max_roots=max_roots,
            root_stride=root_stride,
            max_cycles=max_cycles,
        )
        basis = _greedy_h1_basis_from_normalized(
            cycles,
            edges,
            triangles,
            vertex_ids,
            num_vertices,
            point_cloud=point_cloud,
        )

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


def compute_homology_basis_from_simplices(
    simplices: Iterable[Tuple[int, ...]],
    num_vertices: int,
    dimension: int,
    point_cloud: Optional[np.ndarray] = None,
    *,
    mode: Literal["valid", "optimal"] = "valid",
    max_roots: Optional[int] = None,
    root_stride: int = 1,
    max_cycles: Optional[int] = None,
) -> HomologyBasisResult:
    """Compute H_k generator representatives from simplices over Z/2.

    `mode="valid"` returns any independent quotient basis, while
    `mode="optimal"` applies the small-support/short-cycle heuristic.
    """
    if dimension < 0:
        raise ValueError("dimension must be >= 0")
    if mode not in {"valid", "optimal"}:
        raise ValueError("mode must be 'valid' or 'optimal'")

    simplices_list = [tuple(int(x) for x in s) for s in simplices]

    # Keep H1 optimal path backward-compatible and data-grounded.
    if dimension == 1 and mode == "optimal":
        return compute_optimal_h1_basis_from_simplices(
            simplices_list,
            num_vertices,
            point_cloud=point_cloud,
            max_roots=max_roots,
            root_stride=root_stride,
            max_cycles=max_cycles,
        )

    if julia_engine.available:
        try:
            out = julia_engine.compute_homology_basis_from_simplices(
                simplices_list,
                int(num_vertices),
                int(dimension),
                mode=mode,
                point_cloud=point_cloud,
                max_roots=max_roots,
                root_stride=int(root_stride),
                max_cycles=max_cycles,
            )
            gens: list[HomologyGenerator] = []
            for g in out:
                simp = [
                    tuple(int(x) for x in s) for s in g.get("support_simplices", [])
                ]
                edg = [
                    tuple(sorted((int(e[0]), int(e[1]))))
                    for e in g.get("support_edges", [])
                ]
                gens.append(
                    HomologyGenerator(
                        dimension=int(g.get("dimension", dimension)),
                        support_edges=edg,
                        support_simplices=simp,
                        weight=float(g.get("weight", 0.0)),
                        certified_cycle=bool(g.get("certified_cycle", True)),
                    )
                )
            return HomologyBasisResult(
                dimension=dimension,
                rank=len(gens),
                generators=gens,
                optimal=(mode == "optimal"),
                exact=True,
                message=(
                    f"Computed H_{dimension} representatives over Z/2 via Julia backend"
                    + (" (greedy small-support mode)." if mode == "optimal" else ".")
                ),
            )
        except Exception as exc:
            warnings.warn(
                "Julia homology-generator backend failed in `compute_homology_basis_from_simplices`; "
                f"falling back to Python implementation ({exc!r})."
            )

    py_res = _hk_generators_mod2(
        simplices_list,
        num_vertices=num_vertices,
        dimension=dimension,
        point_cloud=point_cloud,
        mode=mode,
    )
    py_res.message = py_res.message + " (Python backend)."
    return py_res


def compute_homology_basis_from_complex(
    complex: "SimplicialComplex",
    dimension: int,
    point_cloud: Optional[np.ndarray] = None,
    *,
    mode: Literal["valid", "optimal"] = "valid",
    max_roots: Optional[int] = None,
    root_stride: int = 1,
    max_cycles: Optional[int] = None,
) -> HomologyBasisResult:
    """Compute H_k generators directly from a SimplicialComplex object."""
    # Get all simplices up to dim + 1
    simplices = []
    for d in range(dimension + 1):
        simplices.extend(list(complex.n_simplices(d)))
    simplices.extend(list(complex.n_simplices(dimension + 1)))

    num_vertices = len(list(complex.n_simplices(0)))
    
    return compute_homology_basis_from_simplices(
        simplices,
        num_vertices,
        dimension,
        point_cloud=point_cloud,
        mode=mode,
        max_roots=max_roots,
        root_stride=root_stride,
        max_cycles=max_cycles,
    )


def compute_optimal_h1_basis_from_complex(
    complex: "SimplicialComplex",
    point_cloud: Optional[np.ndarray] = None,
    *,
    max_roots: Optional[int] = None,
    root_stride: int = 1,
    max_cycles: Optional[int] = None,
) -> HomologyBasisResult:
    """Compute an optimal H1 basis directly from a SimplicialComplex object."""
    # H1 needs 1-simplices and 2-simplices
    simplices = list(complex.n_simplices(1)) + list(complex.n_simplices(2))
    num_vertices = len(list(complex.n_simplices(0)))
    
    return compute_optimal_h1_basis_from_simplices(
        simplices,
        num_vertices,
        point_cloud=point_cloud,
        max_roots=max_roots,
        root_stride=root_stride,
        max_cycles=max_cycles,
    )
