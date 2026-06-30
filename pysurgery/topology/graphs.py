"""Graphs as 1-dimensional simplicial complexes.

A graph is, topologically, a 1-dimensional CW/simplicial complex: vertices are
0-cells and edges are 1-cells. :class:`Graph` is therefore a **hybrid** type — it
subclasses :class:`~pysurgery.topology.complexes.SimplicialComplex` so that all
the heavy topological machinery (integer homology, Betti numbers, connected
components, Euler characteristic, π₁) is inherited for free, while adding a
lightweight adjacency/edge-list layer that tolerates the things a *simple*
simplicial 1-complex cannot represent: **parallel edges, self-loops, edge
weights, and orientation**.

The two layers are kept consistent:

  * The inherited ``SimplicialComplex`` carries the *simple* 1-skeleton (parallel
    edges collapsed, loops dropped), so inherited homology is always well
    defined.
  * The graph layer (``_edge_list``) retains the full edge multiset; graph
    algorithms (spanning forest, cycle space, degree, covers) run on it.

The module also exposes the low-level graph routines
:func:`bfs_spanning_forest` and :func:`lca_tree_path` that the π₁ extractor in
:mod:`pysurgery.topology.fundamental_group` reuses, so the spanning-tree logic
lives in exactly one place.

References:
    - Hatcher, A. (2002). Algebraic Topology, §1.A (graphs and free groups).
    - Diestel, R. (2017). Graph Theory, 5th ed.
"""

from __future__ import annotations

import collections
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from pydantic import ConfigDict, PrivateAttr
from scipy.sparse import csr_matrix

from pysurgery.topology.complexes import SimplicialComplex

__all__ = ["Graph", "bfs_spanning_forest", "lca_tree_path"]


# ──────────────────────────────────────────────────────────────────────────────
# Low-level graph routines (shared with the π₁ extractor)
# ──────────────────────────────────────────────────────────────────────────────


def bfs_spanning_forest(
    vertices: Iterable[int],
    adjacency: Dict[int, List[Tuple[int, int, int]]],
) -> Tuple[set, Dict[int, int], Dict[int, int], Dict[int, int]]:
    """Breadth-first spanning forest of an (undirected) graph.

    What is Being Computed?:
        A rooted spanning forest: one BFS tree per connected component. The
        non-tree edges are exactly the complement of the returned tree-edge set
        and form a basis of the graph's cycle space (free π₁ generators).

    Args:
        vertices: Iterable of vertex ids (need not be contiguous).
        adjacency: ``{v: [(neighbor, edge_id, orientation), ...]}``. ``edge_id``
            is an opaque, hashable identifier for the edge (e.g. its index in an
            edge list); ``orientation`` is ``+1`` if the edge points ``v →
            neighbor`` and ``-1`` otherwise. Self-loops appear as a single entry
            ``(v, edge_id, +1)``.

    Returns:
        ``(tree_edge_ids, parent, depth, component_root)`` where ``tree_edge_ids``
        is the set of edge ids used by the forest, and ``parent``/``depth``/
        ``component_root`` are per-vertex dicts (root has ``parent = -1``).
    """
    visited: Dict[int, bool] = {}
    tree_edge_ids: set = set()
    parent: Dict[int, int] = {}
    depth: Dict[int, int] = {}
    component_root: Dict[int, int] = {}

    for start in vertices:
        if visited.get(start):
            continue
        queue = collections.deque([(start, 0)])
        visited[start] = True
        parent[start] = -1
        depth[start] = 0
        component_root[start] = start
        while queue:
            curr, d = queue.popleft()
            for neighbor, edge_id, _orient in adjacency.get(curr, ()):  # noqa: B007
                if neighbor == curr:
                    continue  # self-loop is never a tree edge
                if not visited.get(neighbor):
                    visited[neighbor] = True
                    tree_edge_ids.add(edge_id)
                    parent[neighbor] = curr
                    depth[neighbor] = d + 1
                    component_root[neighbor] = component_root[curr]
                    queue.append((neighbor, d + 1))
    return tree_edge_ids, parent, depth, component_root


def lca_tree_path(
    u: int, v: int, parent: Dict[int, int], depth: Dict[int, int]
) -> List[int]:
    """Vertex path ``u → … → v`` through a rooted spanning forest via LCA.

    Both endpoints must lie in the same tree of the forest described by
    ``parent``/``depth`` (as returned by :func:`bfs_spanning_forest`). Runs in
    ``O(depth)`` by walking both nodes up to their lowest common ancestor.
    """
    path_u: List[int] = []
    path_v: List[int] = []
    curr_u, curr_v = u, v
    while depth[curr_u] > depth[curr_v]:
        path_u.append(curr_u)
        curr_u = parent[curr_u]
    while depth[curr_v] > depth[curr_u]:
        path_v.append(curr_v)
        curr_v = parent[curr_v]
    while curr_u != curr_v:
        path_u.append(curr_u)
        path_v.append(curr_v)
        curr_u = parent[curr_u]
        curr_v = parent[curr_v]
    path_u.append(curr_u)  # common ancestor
    return path_u + list(reversed(path_v))


def _simple_skeleton_table(
    edges: Iterable[Tuple[int, int]], extra_vertices: Optional[Iterable[int]] = None
) -> Dict[int, List[Tuple[int, ...]]]:
    """Build the simple 1-skeleton ``{0: [(v,)...], 1: [(u,v)...]}`` from edges.

    Parallel edges are collapsed and self-loops dropped so that the resulting
    table is a valid simplicial complex.
    """
    verts: set = set(int(v) for v in (extra_vertices or ()))
    simple_edges: set = set()
    for u, v in edges:
        u, v = int(u), int(v)
        verts.add(u)
        verts.add(v)
        if u != v:
            simple_edges.add((u, v) if u < v else (v, u))
    table: Dict[int, List[Tuple[int, ...]]] = {
        0: sorted((v,) for v in verts),
    }
    if simple_edges:
        table[1] = sorted(simple_edges)
    return table


# ──────────────────────────────────────────────────────────────────────────────
# Graph type
# ──────────────────────────────────────────────────────────────────────────────


class Graph(SimplicialComplex):
    """A graph, modelled as a 1-dimensional simplicial complex (hybrid type).

    Overview:
        ``Graph`` *is a* :class:`SimplicialComplex` capped at dimension 1, so
        every topological invariant defined on simplicial complexes
        (``homology``, ``betti_number``, ``connected_components``,
        ``euler_characteristic``, ``fundamental_group``) is available directly.
        On top of the inherited *simple* 1-skeleton it maintains a graph layer
        (``_edge_list``) that may contain parallel edges and self-loops, plus an
        optional direction flag and edge weights.

    Construction:
        Prefer the classmethods :meth:`from_edges`, :meth:`from_adjacency`, and
        :meth:`from_simplicial_complex`. Direct ``Graph(simplices={...})``
        construction works too (the edge list is then derived from the
        1-skeleton).

    Coefficient Ring:
        Homology is exact over ℤ (inherited). For a graph, ``H_0`` is free of
        rank = #components and ``H_1`` is free of rank = the cyclomatic number;
        there is never any torsion and ``H_k = 0`` for ``k ≥ 2``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _directed: bool = PrivateAttr(default=False)
    _edge_list: List[Tuple[int, int]] = PrivateAttr(default_factory=list)
    _edge_weights: Optional[List[float]] = PrivateAttr(default=None)
    _adjacency_cache: Optional[Dict[int, List[Tuple[int, int, int]]]] = PrivateAttr(
        default=None
    )

    def __init__(self, **data):
        """Initialise a graph.

        Accepts the graph-specific keywords ``edges`` (list of ``(u, v)`` or ``(u, v, weight)``,
        possibly with parallel/loop entries), ``directed`` (bool), ``weights``
        (list aligned with ``edges``) and ``num_vertices`` (to declare isolated
        vertices), in addition to the usual ``SimplicialComplex`` keywords.
        """
        edges = data.pop("edges", None)
        directed = bool(data.pop("directed", False))
        weights = data.pop("weights", None)
        num_vertices = data.pop("num_vertices", None)

        parsed_edges = None
        has_weights = False
        
        if edges is not None:
            parsed_edges = []
            parsed_weights = []
            for e in edges:
                if len(e) == 2:
                    parsed_edges.append((int(e[0]), int(e[1])))
                    parsed_weights.append(1.0)
                elif len(e) == 3:
                    parsed_edges.append((int(e[0]), int(e[1])))
                    parsed_weights.append(float(e[2]))
                    has_weights = True
                else:
                    raise ValueError("Edges must be (u, v) or (u, v, weight)")
            
            # If explicit weights were provided via kwarg, they override the parsed ones
            if weights is not None:
                parsed_weights = [float(x) for x in weights]
                has_weights = True
                
            weights = parsed_weights if has_weights else None
            edges = parsed_edges

        if "simplices" not in data and edges is not None:
            extra = range(int(num_vertices)) if num_vertices is not None else None
            data["simplices"] = _simple_skeleton_table(edges, extra)

        super().__init__(**data)

        object.__setattr__(self, "_directed", directed)
        if edges is not None:
            el = edges
        else:
            el = [(int(e[0]), int(e[1])) for e in self.n_simplices(1)]
        object.__setattr__(self, "_edge_list", el)
        
        if weights is not None:
            if len(weights) != len(el):
                raise ValueError(
                    f"weights length ({len(weights)}) must match edge count ({len(el)})."
                )
            object.__setattr__(self, "_edge_weights", weights)

    # ── constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_edges(
        cls,
        edges: Iterable[Tuple[int, int]],
        *,
        num_vertices: Optional[int] = None,
        directed: bool = False,
        weights: Optional[Iterable[float]] = None,
        coefficient_ring: str = "Z",
    ) -> "Graph":
        """Build a graph from an explicit edge list.

        Parallel edges and self-loops are preserved in the graph layer (and
        collapsed/dropped only in the inherited simple 1-skeleton).
        """
        return cls(
            edges=[(int(u), int(v)) for (u, v) in edges],
            num_vertices=num_vertices,
            directed=directed,
            weights=list(weights) if weights is not None else None,
            coefficient_ring=coefficient_ring,
        )

    @classmethod
    def from_adjacency(
        cls,
        adjacency: Dict[int, Iterable[int]],
        *,
        directed: bool = False,
        coefficient_ring: str = "Z",
    ) -> "Graph":
        """Build a graph from an adjacency map ``{v: [neighbors]}``.

        For undirected graphs each unordered pair is emitted once.
        """
        edges: List[Tuple[int, int]] = []
        seen: set = set()
        verts = set(int(v) for v in adjacency)
        for u, nbrs in adjacency.items():
            u = int(u)
            for w in nbrs:
                w = int(w)
                verts.add(w)
                if directed:
                    edges.append((u, w))
                else:
                    key = (u, w) if u <= w else (w, u)
                    if key in seen:
                        continue
                    seen.add(key)
                    edges.append((u, w))
        return cls(
            edges=edges,
            num_vertices=(max(verts) + 1) if verts else 0,
            directed=directed,
            coefficient_ring=coefficient_ring,
        )

    @classmethod
    def from_simplicial_complex(cls, sc: SimplicialComplex) -> "Graph":
        """Extract the 1-skeleton of ``sc`` as a :class:`Graph`."""
        verts = [int(v[0]) for v in sc.n_simplices(0)]
        edges = [(int(e[0]), int(e[1])) for e in sc.n_simplices(1)]
        table = _simple_skeleton_table(edges, verts)
        return cls(simplices=table, coefficient_ring=sc.coefficient_ring)

    def to_simplicial_complex(self) -> SimplicialComplex:
        """Return the underlying simple 1-skeleton as a plain ``SimplicialComplex``."""
        return SimplicialComplex(
            simplices={d: list(s) for d, s in self.simplices_field.items() if d <= 1},
            coefficient_ring=self.coefficient_ring,
        )

    # ── example graphs ───────────────────────────────────────────────────────────

    @classmethod
    def path(cls, n: int) -> "Graph":
        """Path graph P_n on ``n`` vertices (a tree)."""
        return cls.from_edges([(i, i + 1) for i in range(n - 1)], num_vertices=n)

    @classmethod
    def cycle(cls, n: int) -> "Graph":
        """Cycle graph C_n (``n`` vertices, ``n`` edges; b₁ = 1)."""
        if n < 1:
            raise ValueError("cycle needs n >= 1.")
        return cls.from_edges([(i, (i + 1) % n) for i in range(n)], num_vertices=n)

    @classmethod
    def complete(cls, n: int) -> "Graph":
        """Complete graph K_n."""
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        return cls.from_edges(edges, num_vertices=n)

    @classmethod
    def complete_bipartite(cls, m: int, k: int) -> "Graph":
        """Complete bipartite graph K_{m,k}."""
        edges = [(i, m + j) for i in range(m) for j in range(k)]
        return cls.from_edges(edges, num_vertices=m + k)

    @classmethod
    def star(cls, n: int) -> "Graph":
        """Star graph S_n: one centre (vertex 0) joined to ``n`` leaves."""
        return cls.from_edges([(0, i) for i in range(1, n + 1)], num_vertices=n + 1)

    @classmethod
    def bouquet(cls, k: int) -> "Graph":
        """Bouquet / wedge of ``k`` circles: one vertex with ``k`` self-loops (π₁ = F_k)."""
        return cls.from_edges([(0, 0)] * k, num_vertices=1)

    @classmethod
    def grid(cls, rows: int, cols: int) -> "Graph":
        """Rectangular grid graph on ``rows × cols`` vertices."""
        def idx(r, c):
            return r * cols + c
        edges = []
        for r in range(rows):
            for c in range(cols):
                if c + 1 < cols:
                    edges.append((idx(r, c), idx(r, c + 1)))
                if r + 1 < rows:
                    edges.append((idx(r, c), idx(r + 1, c)))
        return cls.from_edges(edges, num_vertices=rows * cols)

    @classmethod
    def petersen(cls) -> "Graph":
        """The Petersen graph (10 vertices, 15 edges, girth 5)."""
        outer = [(i, (i + 1) % 5) for i in range(5)]
        spokes = [(i, i + 5) for i in range(5)]
        inner = [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
        return cls.from_edges(outer + spokes + inner, num_vertices=10)

    # ── graph structure ─────────────────────────────────────────────────────────

    @property
    def directed(self) -> bool:
        """Whether edges carry an orientation."""
        return self._directed

    @property
    def edges(self) -> List[Tuple[int, int]]:
        """The full edge multiset (parallel edges and loops retained)."""
        return list(self._edge_list)

    @property
    def vertices(self) -> List[int]:
        """Sorted list of vertex ids (0-simplices, plus any edge endpoints)."""
        vs = {int(v[0]) for v in self.n_simplices(0)}
        for u, v in self._edge_list:
            vs.add(u)
            vs.add(v)
        return sorted(vs)

    @property
    def num_vertices(self) -> int:
        """Number of vertices."""
        return len(self.vertices)

    @property
    def num_edges(self) -> int:
        """Number of edges in the full multiset (parallel/loop edges counted)."""
        return len(self._edge_list)

    def adjacency(self) -> Dict[int, List[Tuple[int, int, int]]]:
        """Return ``{v: [(neighbor, edge_id, orientation), ...]}`` (cached).

        ``edge_id`` is the edge's index in :attr:`edges`. For undirected graphs
        each non-loop edge contributes a ``+1`` entry at its tail and a ``-1``
        entry at its head; self-loops contribute a single ``+1`` entry.
        """
        if self._adjacency_cache is not None:
            return self._adjacency_cache
        adj: Dict[int, List[Tuple[int, int, int]]] = collections.defaultdict(list)
        for v in self.vertices:
            adj[v]  # ensure isolated vertices are present
        for eid, (u, v) in enumerate(self._edge_list):
            if u == v:
                adj[u].append((u, eid, 1))
                continue
            adj[u].append((v, eid, 1))
            if not self._directed:
                adj[v].append((u, eid, -1))
        out = {k: list(val) for k, val in adj.items()}
        object.__setattr__(self, "_adjacency_cache", out)
        return out

    def neighbors(self, v: int) -> List[int]:
        """Distinct neighbours of ``v``."""
        return sorted({nbr for nbr, _e, _o in self.adjacency().get(int(v), ())})

    def degree(self, v: int) -> int:
        """Degree of ``v`` (each self-loop contributes 2)."""
        v = int(v)
        deg = 0
        for u, w in self._edge_list:
            if u == v:
                deg += 1
            if w == v:
                deg += 1
        return deg

    def incidence_matrix(self, *, simple: bool = True) -> csr_matrix:
        """Signed vertex–edge incidence matrix (the boundary map ``d₁``).

        Args:
            simple: When ``True`` (default) return the inherited simple-skeleton
                ``boundary_matrix(1)``. When ``False`` build the incidence over
                the full edge multiset (one column per parallel edge; loop
                columns are zero).
        """
        if simple:
            return self.boundary_matrix(1)
        verts = self.vertices
        vidx = {v: i for i, v in enumerate(verts)}
        rows: List[int] = []
        cols: List[int] = []
        vals: List[int] = []
        for eid, (u, v) in enumerate(self._edge_list):
            if u == v:
                continue
            rows.append(vidx[v])
            cols.append(eid)
            vals.append(1)
            rows.append(vidx[u])
            cols.append(eid)
            vals.append(-1)
        return csr_matrix(
            (vals, (rows, cols)),
            shape=(len(verts), len(self._edge_list)),
            dtype=np.int64,
        )

    # ── trees and cycles ─────────────────────────────────────────────────────────

    def spanning_forest(self) -> Tuple[set, Dict[int, int], Dict[int, int], Dict[int, int]]:
        """Rooted BFS spanning forest. See :func:`bfs_spanning_forest`."""
        return bfs_spanning_forest(self.vertices, self.adjacency())

    def tree_path(self, u: int, v: int) -> List[int]:
        """Vertex path ``u → v`` through the spanning forest (same component)."""
        _tree, parent, depth, root = self.spanning_forest()
        if root.get(int(u)) != root.get(int(v)):
            raise ValueError(
                f"Vertices {u} and {v} are in different connected components."
            )
        return lca_tree_path(int(u), int(v), parent, depth)

    @property
    def cyclomatic_number(self) -> int:
        """First Betti number of the graph: ``E − V + C`` (= rank of cycle space).

        Computed from the *full* edge multiset, so parallel edges and self-loops
        each add an independent cycle. For a simple graph this equals
        ``betti_number(1)`` of the inherited complex.
        """
        return self.num_edges - self.num_vertices + self.num_connected_components()

    @property
    def is_forest(self) -> bool:
        """True iff the graph has no cycles (cyclomatic number 0)."""
        return self.cyclomatic_number == 0

    @property
    def is_tree(self) -> bool:
        """True iff the graph is a connected forest."""
        return self.is_forest and self.is_connected()

    def fundamental_cycles(self) -> List[List[Tuple[int, int]]]:
        """A cycle-space basis: one directed-edge loop per non-tree edge.

        Each loop is returned as a list of directed edges ``[(a, b), ...]`` that
        starts and ends at the same vertex. Non-tree edges are the parallel
        edges, self-loops, and the single "extra" edge of every independent
        cycle; together they freely generate π₁ of the graph.
        """
        tree_ids, parent, depth, root = self.spanning_forest()
        cycles: List[List[Tuple[int, int]]] = []
        for eid, (u, v) in enumerate(self._edge_list):
            if eid in tree_ids:
                continue
            if u == v:
                cycles.append([(u, u)])
                continue
            # loop = edge (u → v) then tree path v → u
            back = lca_tree_path(v, u, parent, depth)
            directed = [(u, v)] + [
                (back[i], back[i + 1]) for i in range(len(back) - 1)
            ]
            cycles.append(directed)
        return cycles

    def cycle_space_basis(self) -> List[List[Tuple[int, int]]]:
        """Alias for :meth:`fundamental_cycles`."""
        return self.fundamental_cycles()

    # ── structural operations ────────────────────────────────────────────────────

    def contract_edge(self, u: int, v: int) -> "Graph":
        """Return a new graph with the edge ``{u, v}`` contracted to a point.

        All occurrences of ``v`` are relabelled to ``u``; resulting self-loops
        (former parallel edges between ``u`` and ``v``) are dropped, but other
        parallel edges are preserved in the graph layer.
        """
        u, v = int(u), int(v)
        if u == v:
            raise ValueError("Cannot contract a self-loop.")
        relabel = lambda x: u if x == v else x  # noqa: E731
        new_edges: List[Tuple[int, int]] = []
        for a, b in self._edge_list:
            a2, b2 = relabel(a), relabel(b)
            if a2 == b2:
                continue
            new_edges.append((a2, b2))
        remaining = sorted({w for w in self.vertices if w != v})
        # Compress vertex ids so the contracted vertex set stays contiguous-ish.
        return Graph.from_edges(
            new_edges,
            num_vertices=(max(remaining) + 1) if remaining else 0,
            directed=self._directed,
            coefficient_ring=self.coefficient_ring,
        )

    def subdivide_edge(self, u: int, v: int) -> "Graph":
        """Return a new graph with one edge ``{u, v}`` subdivided by a new vertex."""
        u, v = int(u), int(v)
        new_vertex = (max(self.vertices) + 1) if self.vertices else 0
        new_edges: List[Tuple[int, int]] = []
        done = False
        for a, b in self._edge_list:
            if not done and {a, b} == {u, v}:
                new_edges.append((a, new_vertex))
                new_edges.append((new_vertex, b))
                done = True
            else:
                new_edges.append((a, b))
        if not done:
            raise ValueError(f"Edge {{{u}, {v}}} not found.")
        return Graph.from_edges(
            new_edges,
            num_vertices=new_vertex + 1,
            directed=self._directed,
            coefficient_ring=self.coefficient_ring,
        )

    def line_graph(self) -> "Graph":
        """Return the line graph of ``self``.

        Edges of ``self`` become vertices, adjacent iff the original edges share
        an endpoint.
        """
        n = self.num_edges
        # vertex i of the line graph = edge i of self
        endpoints = self._edge_list
        adj_pairs: List[Tuple[int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                if set(endpoints[i]) & set(endpoints[j]):
                    adj_pairs.append((i, j))
        return Graph.from_edges(
            adj_pairs, num_vertices=n, coefficient_ring=self.coefficient_ring
        )

    # ── matrices and spectra ─────────────────────────────────────────────────────

    def adjacency_matrix(self, *, sparse: bool = False):
        """Vertex adjacency matrix ``A`` (ordered by :attr:`vertices`).

        Entry ``A[i, j]`` sums the weights of edges between vertex ``i`` and ``j``
        (parallel edges add up); a self-loop contributes ``2 * weight`` on the
        diagonal so that the row sums equal the degrees and ``L = D − A`` has
        zero row sums.
        """
        verts = self.vertices
        idx = {v: i for i, v in enumerate(verts)}
        n = len(verts)
        dtype = np.float64 if self._edge_weights is not None else np.int64
        A = np.zeros((n, n), dtype=dtype)
        weights = self._edge_weights if self._edge_weights is not None else [1.0] * len(self._edge_list)
        for eid, (u, v) in enumerate(self._edge_list):
            i, j = idx[u], idx[v]
            w = weights[eid]
            if i == j:
                A[i, i] += 2 * w
            else:
                A[i, j] += w
                A[j, i] += w
        if sparse:
            return csr_matrix(A)
        return A

    def degree_matrix(self, *, sparse: bool = False):
        """Diagonal degree matrix ``D`` (self-loops count twice)."""
        verts = self.vertices
        if self._edge_weights is None:
            d = np.array([self.degree(v) for v in verts], dtype=np.int64)
        else:
            A = self.adjacency_matrix()
            d = np.sum(A, axis=1)
        D = np.diag(d)
        return csr_matrix(D) if sparse else D

    def laplacian(self, *, sparse: bool = False):
        """Combinatorial graph Laplacian ``L = D − A`` (multigraph-aware).

        For a simple graph this coincides with the 0-th Hodge Laplacian
        :meth:`hodge_laplacian(0) <pysurgery.topology.complexes.SimplicialComplex.hodge_laplacian>`;
        for a multigraph it uses the full edge multiset.
        """
        L = self.degree_matrix() - self.adjacency_matrix()
        return csr_matrix(L) if sparse else L

    def hodge_laplacian(self, k: int, *, sparse: bool = True):
        """The ``k``-th combinatorial Hodge Laplacian for a (multi)graph.

        Overrides the inherited method to use the full edge multiset.
        For ``k = 0``, returns the multigraph-aware graph Laplacian ``D - A``.
        For ``k = 1``, returns ``d₁ᵀ d₁`` where ``d₁`` is the full incidence matrix.
        For ``k ≥ 2``, returns a zero matrix of appropriate size.
        """
        if k < 0:
            raise ValueError("hodge_laplacian degree must be >= 0.")
        if k == 0:
            return self.laplacian(sparse=sparse)
        elif k == 1:
            bk = self.incidence_matrix(simple=False).astype(float)
            if self._edge_weights is not None:
                from scipy.sparse import diags
                W_half = diags(np.sqrt(self._edge_weights))
                L = W_half @ bk.T @ bk @ W_half
            else:
                L = bk.T @ bk
            L = csr_matrix(L)
            return L if sparse else L.toarray()
        else:
            L = csr_matrix((0, 0), dtype=np.int64)
            return L if sparse else L.toarray()

    def harmonic_forms(self, k: int, backend: str = "auto") -> "np.ndarray":
        """Compute an orthonormal basis for the space of harmonic k-forms (ker L_k) for the multigraph."""
        import numpy as np
        from pysurgery.bridge.julia_bridge import julia_engine
        
        b_k = 0
        n_k = 0
        if k == 0:
            b_k = self.num_connected_components()
            n_k = self.num_vertices
        elif k == 1:
            b_k = self.cyclomatic_number
            n_k = self.num_edges
        else:
            return np.zeros((0, 0), dtype=float)
            
        if b_k == 0:
            return np.zeros((n_k, 0), dtype=float)
            
        use_julia = (backend == "julia") or (backend == "auto" and julia_engine.available)
        L_k = self.hodge_laplacian(k, sparse=True).astype(float)
        
        if use_julia:
            basis = julia_engine.compute_hodge_harmonics(L_k, b_k)
        elif L_k.shape[0] < 500:
            from scipy.linalg import svd
            U, S, Vh = svd(L_k.toarray())
            basis = Vh[-b_k:].T
        else:
            import scipy.sparse.linalg as sla
            vals, vecs = sla.eigsh(L_k, k=b_k, sigma=-1e-5, which='LM', tol=1e-10)
            basis = vecs
            
        if k == 1 and self._edge_weights is not None:
            # Transform basis back to the true harmonic space from the symmetrized space
            W_half = np.sqrt(self._edge_weights)
            basis = basis * W_half[:, None]
            
        return basis
    def hodge_decomposition(self, k: int, chain, backend: str = "auto"):
        """Decompose a k-chain into exact, coexact, and harmonic components."""
        import numpy as np
        from pysurgery.bridge.julia_bridge import julia_engine
        
        chain = np.asarray(chain, dtype=float)
        
        b_k = 0
        n_k = 0
        if k == 0:
            b_k = self.num_connected_components()
            n_k = self.num_vertices
            B_k = csr_matrix((0, n_k), dtype=float)
            B_kp1 = self.incidence_matrix(simple=False).astype(float)
            if self._edge_weights is not None:
                from scipy.sparse import diags
                W = diags(self._edge_weights)
                B_kp1 = B_kp1 @ W
        elif k == 1:
            b_k = self.cyclomatic_number
            n_k = self.num_edges
            B_k = self.incidence_matrix(simple=False).astype(float)
            B_kp1 = csr_matrix((n_k, 0), dtype=float)
            if self._edge_weights is not None:
                from scipy.sparse import diags
                W_half = diags(np.sqrt(self._edge_weights))
                W_inv_half = diags(1.0 / np.sqrt(self._edge_weights))
                B_k = B_k @ W_half
                chain = W_inv_half @ chain
        else:
            raise ValueError(f"Graph only supports hodge decomposition for k=0, 1")
            
        if len(chain) != n_k:
            raise ValueError(f"Chain length {len(chain)} does not match {n_k}")
            
        L_k = self.hodge_laplacian(k, sparse=True).astype(float)
        use_julia = (backend == "julia") or (backend == "auto" and julia_engine.available)
        
        if use_julia:
            alpha, beta, h = julia_engine.compute_hodge_decomposition(B_k, B_kp1, L_k, chain, b_k)
        else:
            if b_k > 0:
                H = self.harmonic_forms(k, backend="python")
                # When computing the harmonic projection for k=1 in Python, H is already in the true space.
                # However, our transformed system needs the projection in the symmetrized space!
                if k == 1 and self._edge_weights is not None:
                    # In python fallback, harmonic_forms(1) returns the true basis.
                    # But we are solving the transformed system, so we need the symmetric basis.
                    H_sym = H * (1.0 / np.sqrt(self._edge_weights))[:, None]
                    h = H_sym @ (H_sym.T @ chain)
                else:
                    h = H @ (H.T @ chain)
            else:
                h = np.zeros(n_k)
                
            rhs = chain - h
            
            if L_k.shape[0] < 500:
                from scipy.linalg import pinv
                x = pinv(L_k.toarray()) @ rhs
            else:
                import scipy.sparse.linalg as sla
                x, _ = sla.cg(L_k, rhs, tol=1e-10)
                
            alpha = B_kp1.T @ x
            beta = B_k @ x

        if k == 1 and self._edge_weights is not None:
            # Transform h back from the symmetrized space
            h = np.sqrt(self._edge_weights) * h
            
        return alpha, beta, h

    def plot_harmonic(self, k: int, h, ax=None, **kwargs):
        """Plot a harmonic k-form over the graph.
        
        For k=0, colors nodes according to their values.
        For k=1, colors edges and adds directed arrows scaled by flow magnitude.
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            import numpy as np
        except ImportError as e:
            raise ImportError("Plotting requires 'networkx' and 'matplotlib'.") from e
            
        if self._directed:
            G = nx.MultiDiGraph()
        else:
            G = nx.MultiGraph()
            
        G.add_nodes_from(self.vertices)
        for u, v in self._edge_list:
            G.add_edge(u, v)
            
        if ax is None:
            ax = plt.gca()
            
        pos = nx.spring_layout(G, seed=42)
        
        h = np.asarray(h)
        
        if k == 0:
            # Color nodes
            node_colors = [h[i] for i in range(len(self.vertices))]
            nx.draw(G, pos, ax=ax, node_color=node_colors, cmap=plt.cm.coolwarm, with_labels=True, **kwargs)
            # add colorbar
            sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
            plt.colorbar(sm, ax=ax)
        elif k == 1:
            # Color edges
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightgray')
            nx.draw_networkx_labels(G, pos, ax=ax)
            
            # draw edges with colors and widths
            edge_colors = []
            edge_widths = []
            for val in h:
                edge_colors.append(val)
                edge_widths.append(1.0 + 3.0 * abs(val) / (max(abs(h)) + 1e-9))
                
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=edge_widths, edge_cmap=plt.cm.coolwarm, arrows=True)
            sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=min(edge_colors), vmax=max(edge_colors)))
            plt.colorbar(sm, ax=ax)
        else:
            raise ValueError("Only k=0 and k=1 are supported for graph plots.")
            
        return ax

    def normalized_laplacian(self):
        """Symmetric normalized Laplacian ``I − D^{-1/2} A D^{-1/2}``.

        Isolated vertices (degree 0) contribute a zero row/column.
        """
        verts = self.vertices
        d = np.array([self.degree(v) for v in verts], dtype=float)
        A = self.adjacency_matrix().astype(float)
        with np.errstate(divide="ignore"):
            dinv = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
        n = len(verts)
        return np.eye(n) - (dinv[:, None] * A * dinv[None, :])

    def hashimoto_matrix(self):
        """Non-backtracking (Hashimoto) edge-adjacency matrix ``B``.

        ``B`` acts on the ``2·|E|`` oriented edges (darts). ``B[d, d'] = 1`` iff
        the head of dart ``d`` is the tail of dart ``d'`` and ``d'`` is not the
        reversal of ``d`` (i.e. no backtracking). The non-zero spectrum of ``B``
        governs the Ihara zeta function, and ``tr(Bᵐ)`` counts closed
        non-backtracking walks of length ``m``.

        Returns:
            ``(B, darts)`` where ``darts[k] = (tail, head, edge_id)`` describes
            oriented edge ``k`` (``k`` and ``k^1`` are reverses).
        """
        darts: List[Tuple[int, int, int]] = []
        for eid, (u, v) in enumerate(self._edge_list):
            darts.append((u, v, eid))   # dart 2*eid
            darts.append((v, u, eid))   # dart 2*eid + 1 (reverse)
        m2 = len(darts)
        B = np.zeros((m2, m2), dtype=np.int64)
        from collections import defaultdict
        out_of: Dict[int, List[int]] = defaultdict(list)
        for k, (t, h, _e) in enumerate(darts):
            out_of[t].append(k)
        for k, (t, h, _e) in enumerate(darts):
            for j in out_of.get(h, ()):
                if j == (k ^ 1):
                    continue  # backtracking
                B[k, j] = 1
        return B, darts

    # ── zeta functions ───────────────────────────────────────────────────────────

    def ihara_zeta_reciprocal(self, u=None):
        """Reciprocal ``ζ_G(u)⁻¹`` of the Ihara zeta via Bass's determinant formula.

        ``ζ_G(u)⁻¹ = (1 − u²)^{m−n} · det(I − A u + Q u²)`` with ``Q = D − I``,
        ``n = |V|``, ``m = |E|`` (Bass 1992). For a number ``u`` the value is
        returned; otherwise a SymPy expression in the symbol ``u``.
        """
        import sympy as sp
        sym = sp.symbols("u") if u is None else u
        n = self.num_vertices
        m = self.num_edges
        A = sp.Matrix(self.adjacency_matrix().tolist())
        degs = [self.degree(v) for v in self.vertices]
        Q = sp.diag(*[d - 1 for d in degs]) if degs else sp.Matrix.zeros(0, 0)
        Ident = sp.eye(n)
        det = (Ident - A * sym + Q * sym**2).det()
        recip = (1 - sym**2) ** (m - n) * det
        return sp.simplify(recip) if u is None else recip

    def ihara_zeta(self, u=None):
        """Ihara zeta function ``ζ_G(u) = 1 / ζ_G(u)⁻¹`` (see :meth:`ihara_zeta_reciprocal`)."""
        import sympy as sp
        recip = self.ihara_zeta_reciprocal(u)
        return sp.simplify(1 / recip) if u is None else 1 / recip

    def bartholdi_zeta_reciprocal(self, u=None, t=None):
        """Reciprocal of the two-variable Bartholdi zeta function.

        ``ζ_G(u, t)⁻¹ = (1 − (1−t)²u²)^{m−n} · det(I − A u + ((1−t)D − (1−t)²I) u²)``
        (Bartholdi 1999). At ``t = 0`` this reduces to the Ihara reciprocal.
        Returns a SymPy expression in symbols ``u, t`` unless both are supplied.
        """
        import sympy as sp
        su = sp.symbols("u") if u is None else u
        st = sp.symbols("t") if t is None else t
        n = self.num_vertices
        m = self.num_edges
        A = sp.Matrix(self.adjacency_matrix().tolist())
        degs = [self.degree(v) for v in self.vertices]
        D = sp.diag(*degs) if degs else sp.Matrix.zeros(0, 0)
        Ident = sp.eye(n)
        Qt = (1 - st) * D - (1 - st) ** 2 * Ident
        det = (Ident - A * su + Qt * su**2).det()
        recip = (1 - (1 - st) ** 2 * su**2) ** (m - n) * det
        symbolic = u is None or t is None
        return sp.simplify(recip) if symbolic else recip

    # ── prime cycles (primitive π₁ conjugacy classes) ────────────────────────────

    def prime_cycles(
        self, max_length: int, *, up_to_inversion: bool = False
    ) -> List[List[Tuple[int, int]]]:
        """Primitive closed non-backtracking cycles up to ``max_length`` edges.

        Each cycle is a primitive (non-power), tailless, non-backtracking closed
        walk, returned once per cyclic-rotation class as a list of directed edges
        ``[(a, b), …]``. These are exactly the **prime cycles** of Ihara's zeta
        function and represent primitive conjugacy classes of π₁(G) (a directed
        cycle and its reverse are distinct, matching ``g`` vs ``g⁻¹``).

        Args:
            max_length: Maximum number of edges in a cycle.
            up_to_inversion: If True, also identify a cycle with its reverse
                (geometric undirected closed geodesics).
        """
        _B, darts = self.hashimoto_matrix()
        head = [h for (_t, h, _e) in darts]
        tail = [t for (t, _h, _e) in darts]
        from collections import defaultdict
        out_of: Dict[int, List[int]] = defaultdict(list)
        for k, t in enumerate(tail):
            out_of[t].append(k)

        def is_primitive(seq: List[int]) -> bool:
            length = len(seq)
            for q in range(1, length):
                if length % q == 0 and seq == seq[:q] * (length // q):
                    return False
            return True

        def canonical(seq: List[int]) -> tuple:
            length = len(seq)
            cands = [tuple(seq[i:] + seq[:i]) for i in range(length)]
            if up_to_inversion:
                inv = [k ^ 1 for k in reversed(seq)]
                cands += [tuple(inv[i:] + inv[:i]) for i in range(length)]
            return min(cands)

        seen: set = set()
        results: List[List[Tuple[int, int]]] = []

        def dfs(path: List[int]):
            last = path[-1]
            # Record if this is a closed, tailless, primitive cycle.
            if head[last] == tail[path[0]] and last != (path[0] ^ 1):
                if is_primitive(path):
                    key = canonical(path)
                    if key not in seen:
                        seen.add(key)
                        results.append([(tail[k], head[k]) for k in path])
            if len(path) >= max_length:
                return
            for nxt in out_of.get(head[last], ()):
                if nxt == (last ^ 1):
                    continue  # no backtracking
                dfs(path + [nxt])

        for start in range(len(darts)):
            dfs([start])
        results.sort(key=lambda c: (len(c), c))
        return results

    # ── visualization ────────────────────────────────────────────────────────────

    def plot(
        self,
        *,
        ax: Optional["Any"] = None,
        with_labels: bool = True,
        node_color: str = "lightblue",
        edge_color: str = "gray",
        **kwargs,
    ) -> "Any":
        """Draw the graph using NetworkX and Matplotlib.

        Args:
            ax: Optional matplotlib axes to draw on. If None, the current axes are used.
            with_labels: Whether to draw node labels.
            node_color: Color of the nodes.
            edge_color: Color of the edges.
            **kwargs: Additional arguments passed to ``networkx.draw()``.

        Returns:
            The matplotlib axes used for drawing.
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError as e:
            raise ImportError(
                "The 'plot' method requires 'networkx' and 'matplotlib'."
            ) from e

        if self._directed:
            G = nx.MultiDiGraph()
        else:
            G = nx.MultiGraph()

        G.add_nodes_from(self.vertices)
        for u, v in self._edge_list:
            G.add_edge(u, v)

        if ax is None:
            ax = plt.gca()

        nx.draw(
            G,
            ax=ax,
            with_labels=with_labels,
            node_color=node_color,
            edge_color=edge_color,
            **kwargs,
        )
        return ax

    def to_dot(self, *, name: str = "G") -> str:
        """Graphviz DOT source for the graph (render with ``dot``/``neato``)."""
        directed = self._directed
        head = "digraph" if directed else "graph"
        link = "->" if directed else "--"
        lines = [f"{head} {name} {{"]
        for v in self.vertices:
            lines.append(f"    {v};")
        for u, v in self._edge_list:
            lines.append(f"    {u} {link} {v};")
        lines.append("}")
        return "\n".join(lines)

    def to_ascii_tree(self, root: Optional[int] = None) -> str:
        """ASCII rendering of the graph as a rooted tree/forest (best for trees).

        Uses the BFS spanning forest; non-tree edges are appended in parentheses,
        so the output is exact for trees and an approximate skeleton otherwise.
        """
        adj = self.adjacency()
        tree_ids, parent, depth, comp_root = self.spanning_forest()
        children: Dict[int, List[int]] = {v: [] for v in self.vertices}
        for v in self.vertices:
            p = parent.get(v, -1)
            if p != -1:
                children[p].append(v)
        for v in children:
            children[v].sort()

        roots = [root] if root is not None else sorted(
            {comp_root.get(v, v) for v in self.vertices}
        )
        out: List[str] = []

        def render(v: int, prefix: str, is_last: bool, is_root: bool):
            connector = "" if is_root else ("└─ " if is_last else "├─ ")
            out.append(f"{prefix}{connector}{v}")
            child_prefix = prefix + ("" if is_root else ("   " if is_last else "│  "))
            kids = children[v]
            for i, c in enumerate(kids):
                render(c, child_prefix, i == len(kids) - 1, False)

        for r in roots:
            render(r, "", True, True)
        # Note any non-tree edges (cycles) that the tree view cannot show.
        non_tree = [
            (u, v) for eid, (u, v) in enumerate(self._edge_list)
            if eid not in tree_ids and u != v
        ]
        if non_tree:
            extras = ", ".join(f"{u}-{v}" for u, v in non_tree)
            out.append(f"(+ non-tree edges: {extras})")
        return "\n".join(out)

    # ── fundamental group fast path ──────────────────────────────────────────────

    def fundamental_group(self, simplify: bool = True, backend: str = "auto"):
        """π₁ of the graph: the free group on a cycle-space basis.

        For a graph π₁ is always free of rank equal to the cyclomatic number; we
        read it off the spanning-forest non-tree edges directly (no 2-cells, no
        Tietze step needed), and attach a geometric trace for each generator.
        """
        from pysurgery.topology.fundamental_group import FundamentalGroup
        from pysurgery.core.generator_models import Pi1GeneratorTrace

        cycles = self.fundamental_cycles()
        generators = [f"g_{i}" for i in range(len(cycles))]
        traces: List[Pi1GeneratorTrace] = []
        for name, directed in zip(generators, cycles):
            vpath = [directed[0][0]] + [b for (_a, b) in directed]
            traces.append(
                Pi1GeneratorTrace.model_construct(
                    generator=name,
                    edge_index=None,
                    component_root=int(directed[0][0]),
                    vertex_path=vpath,
                    directed_edge_path=[(int(a), int(b)) for (a, b) in directed],
                    undirected_edge_path=[
                        tuple(sorted((int(a), int(b)))) for (a, b) in directed
                    ],
                )
            )
        return FundamentalGroup(
            generators=generators,
            relations=[],
            orientation_character={g: 1 for g in generators},
            traces=traces,
        )

    def pi1(self, *, simplify: bool = True, backend: str = "auto"):
        """Alias for :meth:`fundamental_group`."""
        return self.fundamental_group(simplify=simplify, backend=backend)

    # ── covers (delegated to pysurgery.topology.coverings) ───────────────────────

    def universal_cover(self, depth: int = 3) -> "Any":
        """The universal cover of the graph: its (truncated) unrolled tree.

        A connected graph's universal cover is a tree; for a graph with cycles it
        is infinite, so it is unrolled by breadth-first expansion up to ``depth``
        edges from a base vertex.

        Returns:
            A :class:`~pysurgery.topology.coverings.GraphCovering` whose ``cover``
            is the tree; it supports ``lift_walk`` and ``render`` and delegates
            Graph methods (``betti_number``, ``is_tree``, …) to the tree.
        """
        from pysurgery.topology.coverings import graph_universal_cover

        return graph_universal_cover(self, depth=depth)

    def cover(self, voltages: Dict[int, Any]) -> "Any":
        """A finite cover from a voltage assignment on edges.

        Returns:
            A :class:`~pysurgery.topology.coverings.GraphCovering`; it supports
            ``lift_walk`` and delegates Graph methods to the cover graph. See
            :func:`pysurgery.topology.coverings.cover_graph`.
        """
        from pysurgery.topology.coverings import cover_graph

        return cover_graph(self, voltages)
