"""Data-grounded H_1 generators and optimal basis extraction.

Algorithmic foundation follows the "Generators and Optimality" chapter in:
T. K. Dey and Y. Wang, *Computational Topology for Data Analysis*.

References:
    Dey, T. K., & Wang, Y. (2022). Computational topology for data analysis. 
    Cambridge University Press.

Implemented pipeline:
- edge annotations on 2-complexes,
- shortest-path generated cycle candidates,
- greedy independent minimum-weight basis over Z2 annotations.

When Julia backend support is available, this module can delegate the core
generator/optimality computation to Julia and falls back to Python otherwise.
"""

from __future__ import annotations

import heapq
import itertools
import numba
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
    """Build simplicial closure grouped by dimension from input simplices.

    Args:
        simplices (Iterable[Tuple[int, ...]]): Input simplices.

    Returns:
        dict[int, list[tuple[int, ...]]]: Simplices grouped by dimension.
    """
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
    """Infer vertex count from simplices when not supplied by the caller.

    Args:
        simplices (Iterable[Tuple[int, ...]]): Input simplices.
        num_vertices (int): Explicit vertex count or 0 to infer.

    Returns:
        int: The inferred or provided vertex count.
    """
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
    backend: str = "auto",
) -> np.ndarray:
    """Builds mod-2 boundary matrix with optional Julia acceleration.

    Falls back to pure Python if Julia unavailable or fails.

    Args:
        source (list[tuple[int, ...]]): The source simplices (d-simplices).
        target (list[tuple[int, ...]]): The target simplices ((d-1)-simplices).
        backend: 'auto', 'julia', or 'python'.

    Returns:
        np.ndarray: The mod-2 boundary matrix.
    """
    if not target or not source:
        return np.zeros((len(target), len(source)), dtype=np.int64)

    # Normalize backend
    backend_norm = str(backend).lower().strip()
    use_julia = (backend_norm == "julia") or (backend_norm == "auto" and julia_engine.available)

    # Prefer Julia acceleration whenever available.
    if use_julia:
        try:
            payload = julia_engine.compute_boundary_mod2_matrix(source, target)
            return sp.csr_matrix(
                (payload["data"], (payload["rows"], payload["cols"])),
                shape=(payload["m"], payload["n"]),
                dtype=np.int64,
            )
        except Exception as e:
            if backend_norm == "julia":
                raise e
            warnings.warn(
                f"Topological Hint: Julia mod2 boundary assembly failed ({e!r}). "
                "Falling back to pure Python."
            )

    # Python fallback
    t_idx = {t: i for i, t in enumerate(target)}
    rows, cols, data = [], [], []
    for j, s in enumerate(source):
        for i_drop in range(len(s)):
            face = s[:i_drop] + s[i_drop + 1 :]
            row = t_idx.get(face)
            if row is not None:
                rows.append(row)
                cols.append(j)
                data.append(1)
    
    return sp.csr_matrix(
        (data, (rows, cols)),
        shape=(len(target), len(source)),
        dtype=np.int64
    )


@numba.njit(cache=True)
def _numba_xor_rows(M, target_row, source_row):
    """Numba-accelerated row XOR for dense GF(2) blocks."""
    for j in range(M.shape[1]):
        M[target_row, j] ^= M[source_row, j]

@numba.njit(cache=True)
def _numba_rref_gf2_kernel(M):
    m, n = M.shape
    row = 0
    pivots = []
    
    for col in range(n):
        if row >= m:
            break
        
        pivot = -1
        for r in range(row, m):
            if M[r, col] == 1:
                pivot = r
                break
        
        if pivot == -1:
            continue

        if pivot != row:
            for j in range(col, n):
                tmp = M[row, j]
                M[row, j] = M[pivot, j]
                M[pivot, j] = tmp

        for r in range(m):
            if r != row and M[r, col] == 1:
                # Optimized XOR row
                for j in range(col, n):
                    M[r, j] ^= M[row, j]

        pivots.append(col)
        row += 1
    return row, pivots

def _rref_mod2(A: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Compute row-reduced echelon form over GF(2) with accelerated operations.

    Args:
        A (np.ndarray): The input matrix.

    Returns:
        tuple[np.ndarray, list[int]]: A tuple containing the RREF matrix and
            the indices of the pivot columns.
    """
    M = (np.asarray(A, dtype=np.int8) & 1).copy()
    
    # We use a Numba kernel for the entire reduction to avoid loop overhead
    final_row, pivots = _numba_rref_gf2_kernel(M)
    
    return M.astype(np.int64), list(pivots)


def _nullspace_basis_mod2(A: sp.spmatrix) -> list[np.ndarray]:
    """Return a basis for the nullspace of `A` over GF(2).

    Args:
        A: The input matrix (sparse or dense).

    Returns:
        list[np.ndarray]: A list of basis vectors for the nullspace.
    """
    # A is m x n over F2; basis vectors are in F2^n.
    M_dense = A.toarray() if hasattr(A, "toarray") else np.asarray(A)
    rref, pivots = _rref_mod2(M_dense)
    m, n = rref.shape
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
    """Compute matrix rank over GF(2).

    Args:
        A (np.ndarray): The input matrix.

    Returns:
        int: The rank of the matrix over GF(2).
    """
    _, pivots = _rref_mod2(A)
    return len(pivots)


def _components_h0_generators(
    edges: list[Edge],
    num_vertices: int,
    point_cloud: Optional[np.ndarray] = None,
) -> HomologyBasisResult:
    """Compute H0 generators as one representative per connected component.

    Args:
        edges (list[Edge]): List of edges in the complex.
        num_vertices (int): Total number of vertices.
        point_cloud (Optional[np.ndarray]): Point cloud coordinates.

    Returns:
        HomologyBasisResult: The computed H0 basis.

    Raises:
        ValueError: If an edge contains an invalid vertex index.
    """
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
    """Compute a geometric/algebraic proxy weight for an active k-chain.

    Args:
        chain (np.ndarray): The chain as a binary vector.
        k_simplices (list[tuple[int, ...]]): List of k-simplices.
        point_cloud (Optional[np.ndarray]): Point cloud coordinates.

    Returns:
        float: The computed weight of the chain.
    """
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
    """Check if `v` increases span rank modulo 2 over current basis columns.

    Args:
        v (np.ndarray): The vector to check.
        basis_cols (list[np.ndarray]): The current basis columns.

    Returns:
        bool: True if independent, False otherwise.
    """
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
    """Compute H_k representatives over Z/2 via kernel/image quotient.

    Args:
        simplices (Iterable[Tuple[int, ...]]): Input simplices.
        num_vertices (int): Total number of vertices.
        dimension (int): Homology dimension to compute.
        point_cloud (Optional[np.ndarray]): Point cloud coordinates.
        mode (Literal["valid", "optimal"]): Computation mode.

    Returns:
        HomologyBasisResult: The computed homology basis.
    """
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

    b_cols = [d_kp1[:, j].toarray().flatten() & 1 for j in range(d_kp1.shape[1])]
    z_candidates = z_basis[:]
    if mode == "optimal":
        z_candidates = sorted(
            z_candidates, key=lambda v: _weight_k_chain(v, k_simplices, point_cloud)
        )

    # Optimized incremental quotient basis
    max_pivots = len(k_simplices)
    pivot_matrix = np.zeros((max_pivots, len(k_simplices)), dtype=np.int8)
    pivot_indices = np.zeros(max_pivots, dtype=np.int64)
    n_pivots = 0
    
    # 1. Pre-fill with the image (boundaries of kp1)
    for b in b_cols:
        _, n_pivots = _is_independent_mod_image_optimized(b, pivot_matrix, pivot_indices, n_pivots)

    quotient_basis = []
    # 2. Extract quotient basis in O(N^2) total (since pivots are pre-packed)
    for z in z_candidates:
        is_indep, n_pivots = _is_independent_mod_image_optimized(z, pivot_matrix, pivot_indices, n_pivots)
        if is_indep:
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
    """Return edge length weight (or unit weight when no geometry is provided).

    Args:
        u (int): First vertex index.
        v (int): Second vertex index.
        points (Optional[np.ndarray]): Point cloud coordinates.

    Returns:
        float: The weight of the edge.
    """
    if points is None:
        return 1.0
    return float(np.linalg.norm(points[u] - points[v]))


def _normalize_edges_triangles(
    simplices: Iterable[Tuple[int, ...]],
) -> tuple[list[Edge], list[Triangle], set[int]]:
    """Extract normalized edge/triangle lists plus observed vertex ids.

    Args:
        simplices (Iterable[Tuple[int, ...]]): Input simplices.

    Returns:
        tuple[list[Edge], list[Triangle], set[int]]: Normalized edges,
            triangles, and vertex IDs.
    """
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
    """Return minimum-spanning-forest edges using SciPy's optimized backend.

    Args:
        edges (list[Edge]): List of edges.
        weights (Dict[Edge, float]): Dictionary of edge weights.
        num_vertices (int): Total number of vertices.

    Returns:
        set[Edge]: The set of edges in the minimum spanning forest.
    """
    if not edges:
        return set()
    
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree

    n = max(num_vertices, 1)
    u_idx = np.array([e[0] for e in edges], dtype=np.int32)
    v_idx = np.array([e[1] for e in edges], dtype=np.int32)
    # Add a small epsilon to weights to ensure zero-weight edges are not dropped by CSR
    w = np.array([weights[e] + 1e-12 for e in edges], dtype=np.float64)
    
    adj = csr_matrix((w, (u_idx, v_idx)), shape=(n, n))
    mst = minimum_spanning_tree(adj)
    
    mst_rows, mst_cols = mst.nonzero()
    spanning = set()
    for u, v in zip(mst_rows, mst_cols):
        spanning.add(tuple(sorted((int(u), int(v)))))
    
    return spanning


def annot_edge(
    simplices: Iterable[Tuple[int, ...]],
    num_vertices: int,
    edge_weights: Optional[Dict[Edge, float]] = None,
) -> tuple[Dict[Edge, np.ndarray], int]:
    """Compute edge annotations for cycle-space independence over Z/2.

    Args:
        simplices (Iterable[Tuple[int, ...]]): Input simplices.
        num_vertices (int): Total number of vertices.
        edge_weights (Optional[Dict[Edge, float]]): Edge weights.

    Returns:
        tuple[Dict[Edge, np.ndarray], int]: Mapping from edges to annotation
            vectors and the final dimension of the annotation space.
    """
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
    """Build a shortest-path tree from one root using Dijkstra.

    Args:
        root (int): The root vertex index.
        adjacency (Dict[int, List[Tuple[int, float]]]): Adjacency list with weights.

    Returns:
        tuple[Dict[int, int], set[Edge]]: Mapping from vertices to parents and
            the set of edges in the shortest-path tree.
    """
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
    """Return tree-path vertices between two nodes from parent pointers.

    Args:
        u (int): Start vertex.
        v (int): End vertex.
        parent (Dict[int, int]): Parent mapping from shortest-path tree.

    Returns:
        List[int]: The list of vertices in the path.
    """
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
    """Convert a vertex path to normalized undirected edges.

    Args:
        path_vertices (List[int]): List of vertices in the path.

    Returns:
        List[Edge]: The list of edges in the path.
    """
    return [
        tuple(sorted((path_vertices[i], path_vertices[i + 1])))
        for i in range(len(path_vertices) - 1)
    ]


def _cycle_weight(
    cycle: Cycle, edge_weights: Dict[Edge, float], point_cloud: Optional[np.ndarray]
) -> float:
    """Compute cycle weight using supplied edge weights or geometric fallback.

    Args:
        cycle (Cycle): The cycle as a list of edges.
        edge_weights (Dict[Edge, float]): Dictionary of edge weights.
        point_cloud (Optional[np.ndarray]): Point cloud coordinates.

    Returns:
        float: The computed weight of the cycle.
    """
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
    """Generate candidate H1 cycles from simplices via shortest-path trees.

    Args:
        simplices (Iterable[Tuple[int, ...]]): Input simplices.
        num_vertices (int): Total number of vertices.
        point_cloud (Optional[np.ndarray]): Point cloud coordinates.
        max_roots (Optional[int]): Maximum number of tree roots.
        root_stride (int): Stride for picking roots. Defaults to 1.
        max_cycles (Optional[int]): Maximum number of cycles to generate.

    Returns:
        List[Cycle]: The list of generated cycles.
    """
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
    """Generate cycle candidates from normalized edge lists using vectorized shortest path trees.

    Args:
        edges (list[Edge]): List of edges.
        vertex_ids (set[int]): Set of vertex IDs.
        num_vertices (int): Total number of vertices.
        point_cloud (Optional[np.ndarray]): Point cloud coordinates.
        max_roots (Optional[int]): Maximum number of roots.
        root_stride (int): Stride for roots.
        max_cycles (Optional[int]): Maximum number of cycles.

    Returns:
        List[Cycle]: The list of generated cycles.
    """
    if not edges:
        return []

    if num_vertices <= 0 and vertex_ids:
        num_vertices = max(vertex_ids) + 1

    vertices = sorted(vertex_ids) if vertex_ids else list(range(num_vertices))
    selected_roots = vertices[:: max(1, root_stride)]
    if max_roots is not None:
        selected_roots = selected_roots[:max_roots]

    if not selected_roots:
        return []

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    u_idx = np.array([e[0] for e in edges], dtype=np.int32)
    v_idx = np.array([e[1] for e in edges], dtype=np.int32)
    w = np.array([_edge_weight(int(u), int(v), point_cloud) for u, v in edges], dtype=np.float64)

    # Create symmetric adjacency matrix
    u_idx_full = np.concatenate([u_idx, v_idx])
    v_idx_full = np.concatenate([v_idx, u_idx])
    w_full = np.concatenate([w, w])
    
    adj = csr_matrix((w_full, (u_idx_full, v_idx_full)), shape=(num_vertices, num_vertices))
    
    # Compute all shortest paths at once with optimized C-backend
    _, predecessors = shortest_path(csgraph=adj, directed=False, indices=selected_roots, return_predecessors=True)

    cycles: list[Cycle] = []
    
    for i, root in enumerate(selected_roots):
        parent_arr = predecessors[i]
        
        # We need parent as a dictionary for _path_between compatibility
        # and tree_edges for cycle filtering.
        parent: Dict[int, int] = {}
        tree_edges: set[Edge] = set()
        for v_id in vertices:
            p = parent_arr[v_id]
            if p >= 0:
                parent[int(v_id)] = int(p)
                if p != v_id:
                    tree_edges.add(tuple(sorted((int(p), int(v_id)))))
        parent[int(root)] = -1
        
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


@numba.njit(cache=True)
def _numba_assemble_annotation(indices, annotations_flat, vec_len):
    ann = np.zeros(vec_len, dtype=np.int64)
    for i in indices:
        if i != -1:
            ann ^= annotations_flat[i]
    return ann

def _cycle_annotation(
    cycle: Cycle, simplex_annotations: Dict[Edge, np.ndarray], vec_len: int
) -> np.ndarray:
    """Assemble the annotation vector of a cycle by XOR-combining edge labels.

    Args:
        cycle (Cycle): The cycle edges.
        simplex_annotations (Dict[Edge, np.ndarray]): Edge annotations.
        vec_len (int): Length of the annotation vectors.

    Returns:
        np.ndarray: The annotation vector for the cycle.
    """
    # Convert cycle to indices for Numba
    edges_flat = list(simplex_annotations.keys())
    edge_to_idx = {e: i for i, e in enumerate(edges_flat)}
    
    # Pre-pack annotations for Numba
    ann_arr = np.array([simplex_annotations[e] for e in edges_flat])
    
    indices = np.array([edge_to_idx.get(tuple(sorted(e)), -1) for e in cycle], dtype=np.int64)
    
    return _numba_assemble_annotation(indices, ann_arr, vec_len)


@numba.njit(cache=True)
def _numba_is_independent_wrt_optimized(v, pivot_matrix, pivot_indices, n_pivots):
    """GF(2) optimized independence check with zero allocation."""
    work = v.copy()
    for i in range(len(work)):
        if work[i] == 0:
            continue
        
        found_pivot = -1
        for idx in range(n_pivots):
            if pivot_indices[idx] == i:
                found_pivot = idx
                break
        
        if found_pivot != -1:
            for j in range(i, len(work)):
                work[j] ^= pivot_matrix[found_pivot, j]
        else:
            return i, work
    return -1, work


def _is_independent_mod_image_optimized(cv, pivot_matrix, pivot_indices, n_pivots):
    """GF(2) optimized independence check."""
    new_idx, final_v = _numba_is_independent_wrt_optimized(
        cv.astype(np.int8), pivot_matrix, pivot_indices, n_pivots
    )
    if new_idx != -1:
        pivot_matrix[n_pivots] = final_v.astype(np.int8)
        pivot_indices[n_pivots] = new_idx
        return True, n_pivots + 1
    return False, n_pivots


def greedy_h1_basis(
    cycles: List[Cycle],
    simplices: Iterable[Tuple[int, ...]],
    num_vertices: int,
    point_cloud: Optional[np.ndarray] = None,
    backend: str = "auto",
) -> List[Cycle]:
    """Select a greedy independent H1 basis from cycle candidates.

    Args:
        cycles (List[Cycle]): List of candidate cycles.
        simplices (Iterable[Tuple[int, ...]]): Input simplices.
        num_vertices (int): Total number of vertices.
        point_cloud (Optional[np.ndarray]): Point cloud coordinates.
        backend: 'auto', 'julia', or 'python'.

    Returns:
        List[Cycle]: The selected basis cycles.
    """
    backend_norm = str(backend).lower().strip()
    if backend_norm == "julia" and not julia_engine.available:
        julia_engine.require_julia()

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
    """Internal greedy H1 basis selection on normalized complexes.

    Args:
        cycles (List[Cycle]): Candidate cycles.
        edges (list[Edge]): Normalized edges.
        triangles (list[Triangle]): Normalized triangles.
        vertex_ids (set[int]): Vertex IDs.
        num_vertices (int): Vertex count.
        point_cloud (Optional[np.ndarray]): Point cloud coordinates.

    Returns:
        List[Cycle]: The selected basis cycles.
    """
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
    
    # Pre-allocate pivot matrix for state-of-the-art performance
    max_pivots = vec_dim
    pivot_matrix = np.zeros((max_pivots, vec_dim), dtype=np.int8)
    pivot_indices = np.zeros(max_pivots, dtype=np.int64)
    n_pivots = 0
    
    for cyc in cycles_sorted:
        ann = _cycle_annotation(cyc, simplex_annotations, vec_dim)
        new_idx, final_v = _numba_is_independent_wrt_optimized(
            ann.astype(np.int8), pivot_matrix, pivot_indices, n_pivots
        )
        if new_idx != -1:
            pivot_matrix[n_pivots] = final_v.astype(np.int8)
            pivot_indices[n_pivots] = new_idx
            n_pivots += 1
            basis.append(cyc)
        if n_pivots >= vec_dim:
            break
    return basis


def compute_optimal_h1_basis_from_simplices(
    simplices: Iterable[Tuple[int, ...]],
    num_vertices: int,
    point_cloud: Optional[np.ndarray] = None,
    *,
    max_roots: Optional[int] = None,
    root_stride: int = 1,
    max_cycles: Optional[int] = None,
    backend: str = "auto",
) -> HomologyBasisResult:
    """Compute a data-grounded optimal H1 basis from simplices.

    Uses Julia's optimized backend when available, otherwise falls back to the
    Python shortest-path candidate + greedy-independence pipeline.

    Args:
        simplices (Iterable[Tuple[int, ...]]): Input simplices.
        num_vertices (int): Total number of vertices.
        point_cloud (Optional[np.ndarray]): Point cloud coordinates.
        max_roots (Optional[int]): Maximum number of roots for candidates.
        root_stride (int): Stride for roots. Defaults to 1.
        max_cycles (Optional[int]): Maximum number of candidates.
        backend: 'auto', 'julia', or 'python'.

    Returns:
        HomologyBasisResult: The computed optimal H1 basis.
    """
    simplices_list = [tuple(int(x) for x in s) for s in simplices]
    basis: Optional[list[Cycle]] = None
    
    # Normalize backend
    backend_norm = str(backend).lower().strip()
    use_julia = (backend_norm == "julia") or (backend_norm == "auto" and julia_engine.available)

    if use_julia:
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
            if backend_norm == "julia":
                raise exc
            warnings.warn(
                f"Topological Hint: Julia optimal H1 basis extraction failed ({exc!r}). "
                "Falling back to Python candidate+greedy pipeline."
            )
            used_julia = False
    else:
        used_julia = False

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
    backend: str = "auto",
) -> HomologyBasisResult:
    """Compute H_k generator representatives from simplices over Z/2.

    `mode="valid"` returns any independent quotient basis, while
    `mode="optimal"` applies the small-support/short-cycle heuristic.

    Args:
        simplices (Iterable[Tuple[int, ...]]): Input simplices.
        num_vertices (int): Total number of vertices.
        dimension (int): Homology dimension.
        point_cloud (Optional[np.ndarray]): Point cloud coordinates.
        mode (Literal["valid", "optimal"]): Computation mode. Defaults to "valid".
        max_roots (Optional[int]): Max roots for H1 optimal candidates.
        root_stride (int): Stride for roots. Defaults to 1.
        max_cycles (Optional[int]): Max candidates for H1 optimal.

    Returns:
        HomologyBasisResult: The computed homology basis.

    Raises:
        ValueError: If dimension < 0 or mode is invalid.
    """
    if dimension < 0:
        raise ValueError("dimension must be >= 0")
    if mode not in {"valid", "optimal"}:
        raise ValueError("mode must be 'valid' or 'optimal'")

    simplices_list = [tuple(int(x) for x in s) for s in simplices]
    
    # Normalize backend
    backend_norm = str(backend).lower().strip()
    use_julia = (backend_norm == "julia") or (backend_norm == "auto" and julia_engine.available)

    # Keep H1 optimal path backward-compatible and data-grounded.
    if dimension == 1 and mode == "optimal":
        return compute_optimal_h1_basis_from_simplices(
            simplices_list,
            num_vertices,
            point_cloud=point_cloud,
            max_roots=max_roots,
            root_stride=root_stride,
            max_cycles=max_cycles,
            backend=backend
        )

    if use_julia:
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
            if backend_norm == "julia":
                raise exc
            warnings.warn(
                f"Julia homology engine was available but failed during computation ({exc!r}). "
                "Falling back to Python/Numba-accelerated engine."
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
    backend: str = "auto",
) -> HomologyBasisResult:
    """Compute H_k generators directly from a SimplicialComplex object.

    Args:
        complex (SimplicialComplex): The simplicial complex.
        dimension (int): Homology dimension.
        point_cloud (Optional[np.ndarray]): Point cloud coordinates.
        mode (Literal["valid", "optimal"]): Computation mode. Defaults to "valid".
        max_roots (Optional[int]): Max roots for H1 optimal.
        root_stride (int): Stride for roots. Defaults to 1.
        max_cycles (Optional[int]): Max candidates for H1 optimal.
        backend (str): 'auto', 'julia', or 'python'.

    Returns:
        HomologyBasisResult: The computed homology basis.
    """
    # Get all simplices up to dim + 1
    simplices = []
    for d in range(dimension + 2):
        simplices.extend(list(complex.n_simplices(d)))

    v_simps = list(complex.n_simplices(0))
    num_vertices = max(v[0] for v in v_simps) + 1 if v_simps else 0
    
    return compute_homology_basis_from_simplices(
        simplices,
        num_vertices,
        dimension,
        point_cloud=point_cloud,
        mode=mode,
        max_roots=max_roots,
        root_stride=root_stride,
        max_cycles=max_cycles,
        backend=backend,
    )


def compute_optimal_h1_basis_from_complex(
    complex: "SimplicialComplex",
    point_cloud: Optional[np.ndarray] = None,
    *,
    max_roots: Optional[int] = None,
    root_stride: int = 1,
    max_cycles: Optional[int] = None,
    backend: str = "auto",
) -> HomologyBasisResult:
    """Compute an optimal H1 basis directly from a SimplicialComplex object.

    Args:
        complex (SimplicialComplex): The simplicial complex.
        point_cloud (Optional[np.ndarray]): Point cloud coordinates.
        max_roots (Optional[int]): Max roots.
        root_stride (int): Stride for roots. Defaults to 1.
        max_cycles (Optional[int]): Max candidates.
        backend (str): 'auto', 'julia', or 'python'.

    Returns:
        HomologyBasisResult: The computed optimal H1 basis.
    """
    # H1 needs 1-simplices and 2-simplices
    simplices = list(complex.n_simplices(1)) + list(complex.n_simplices(2))
    v_simps = list(complex.n_simplices(0))
    num_vertices = max(v[0] for v in v_simps) + 1 if v_simps else 0
    
    return compute_optimal_h1_basis_from_simplices(
        simplices,
        num_vertices,
        point_cloud=point_cloud,
        max_roots=max_roots,
        root_stride=root_stride,
        max_cycles=max_cycles,
        backend=backend,
    )
