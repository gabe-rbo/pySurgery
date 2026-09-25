"""pysurgery/knots/seifert_surface.py.

Seifert matrices of knots read off a triangulation of the ambient 3-manifold
alone, with no vertex coordinates.

Pipeline:
  1. Close the ambient complex: cone off each boundary 2-sphere (a 3-ball
     becomes S^3, which is how R^3 minus the ball compactifies), check that
     the result is a connected combinatorial 3-manifold, and orient it
     coherently.
  2. Seifert surface: a minimal-area integral 2-chain F with ∂F = K, from a
     linear program.  The optimal vertices are integral: F ranges over
     F_0 + im ∂_3, and ∂_3 is totally unimodular because every triangle lies
     in exactly two tetrahedra, which induce opposite orientations on it.
     Minimality rules out closed components.  The support of F must be an
     embedded surface with boundary K.  Where it is not (sheets doubled up,
     or touching along an edge or at a vertex) the triangles involved are
     made more expensive and the LP is solved again.
  3. A basis {α_i} of H_1(F; Z), by tree–cotree decomposition.
  4. The positive push-off α⁺ of each basis cycle α is a closed path of
     tetrahedra on the positive side of F along α (a dual 1-cycle).  It
     crosses triangles only, so it is disjoint from every edge path.
  5. lk(α⁺, β) = α⁺ · G for any 2-chain G with ∂G = β, since a dual path
     meets a 2-chain transversally, one triangle at a time.  G comes from a
     unimodular tree–cotree system.
  6. V_ij = lk(α_i⁺, α_j).  V − Vᵀ is the intersection form of F, so
     det(V − Vᵀ) = 1 is checked.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, hstack
from scipy.sparse.linalg import splu

Simplex = Tuple[int, ...]

_LP_ATTEMPTS = 4  # tie-breaking weight draws per triangulation
_REWEIGHT_ROUNDS = 8  # re-solves per draw, penalising non-embedded spots


class SeifertSurfaceError(ValueError):
    """Raised when the triangulation does not yield a Seifert matrix for K."""


def _faces(s: Simplex) -> List[Simplex]:
    return [s[:i] + s[i + 1:] for i in range(len(s))]


class _UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[object, object] = {}

    def find(self, x):
        parent = self.parent
        root = x
        while parent.setdefault(root, root) != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(self, a, b) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[ra] = rb
        return True


def _close_up(tets: List[Simplex], knot_vertex: int) -> Tuple[List[Simplex], int]:
    """Restrict to the component containing the knot and cone off its boundary.

    Returns (tets, n_original): the first n_original tetrahedra are the
    ambient's own, the rest are cones from a new apex over each boundary
    2-sphere.
    """
    uf = _UnionFind()
    for T in tets:
        for v in T[1:]:
            uf.union(T[0], v)
    root = uf.find(knot_vertex)
    tets = sorted({T for T in tets if uf.find(T[0]) == root})

    count: Dict[Simplex, int] = defaultdict(int)
    for T in tets:
        for t in _faces(T):
            count[t] += 1
    if any(c > 2 for c in count.values()):
        raise SeifertSurfaceError(
            "Ambient complex is not a 3-manifold: a triangle lies in more than two tetrahedra"
        )
    boundary = [t for t, c in count.items() if c == 1]
    if not boundary:
        return tets, len(tets)

    # Group the boundary triangles into surfaces glued along edges.
    edge_tris: Dict[Simplex, List[int]] = defaultdict(list)
    for i, t in enumerate(boundary):
        for e in _faces(t):
            edge_tris[e].append(i)
    if any(len(ts) != 2 for ts in edge_tris.values()):
        raise SeifertSurfaceError("Boundary of the ambient complex is not a closed surface")
    uf = _UnionFind()
    for ts in edge_tris.values():
        uf.union(ts[0], ts[1])
    components: Dict[object, List[Simplex]] = defaultdict(list)
    for i, t in enumerate(boundary):
        components[uf.find(i)].append(t)

    apex = max(max(T) for T in tets)
    coned = list(tets)
    for comp in components.values():
        n_v = len({v for t in comp for v in t})
        n_e = len({e for t in comp for e in _faces(t)})
        if n_v - n_e + len(comp) != 2:
            raise SeifertSurfaceError(
                "Ambient boundary has a component that is not a 2-sphere; "
                "only 3-balls (and closed 3-manifolds) can be closed up to S^3"
            )
        apex += 1
        coned.extend(t + (apex,) for t in comp)
    return coned, len(tets)


class _Triangulation:
    """A closed, connected, coherently oriented combinatorial 3-manifold."""

    def __init__(
        self,
        tets: List[Simplex],
        coords: Optional[Dict[int, np.ndarray]] = None,
        n_original: Optional[int] = None,
    ) -> None:
        self.tets = tets
        self.coords = coords or {}
        self.tris: List[Simplex] = sorted({t for T in tets for t in _faces(T)})
        self.edges: List[Simplex] = sorted({e for t in self.tris for e in _faces(t)})
        self.tri_index = {t: j for j, t in enumerate(self.tris)}
        self.edge_index = {e: i for i, e in enumerate(self.edges)}
        self.tet_index = {T: k for k, T in enumerate(tets)}

        # tet_faces[k][i] = (triangle opposite T[i], its sign in ∂[T sorted])
        self.tet_faces: List[List[Tuple[int, int]]] = []
        self.tri_tets: List[List[int]] = [[] for _ in self.tris]
        self.vertex_tets: Dict[int, List[int]] = defaultdict(list)
        for k, T in enumerate(tets):
            faces = []
            for i, t in enumerate(_faces(T)):
                j = self.tri_index[t]
                faces.append((j, -1 if i % 2 else 1))
                self.tri_tets[j].append(k)
            self.tet_faces.append(faces)
            for v in T:
                self.vertex_tets[v].append(k)

        self._check_manifold()
        self.eps = self._orient(len(tets) if n_original is None else n_original)

        rows, cols, vals = [], [], []
        for j, t in enumerate(self.tris):
            for i, e in enumerate(_faces(t)):
                rows.append(self.edge_index[e])
                cols.append(j)
                vals.append(-1 if i % 2 else 1)
        self.B2 = csr_matrix(
            coo_matrix((vals, (rows, cols)), shape=(len(self.edges), len(self.tris)), dtype=np.int64)
        )
        self._tree_cotree = None

    # ── Validation and orientation ──────────────────────────────────────────

    def _check_manifold(self) -> None:
        if any(len(ks) != 2 for ks in self.tri_tets):
            raise SeifertSurfaceError("Closed-up ambient complex is not a 3-manifold")
        # The link of every edge must be one circle and the link of every vertex
        # a connected closed surface with χ = 2, i.e. a 2-sphere.
        edge_uf, vertex_uf = _UnionFind(), _UnionFind()
        for j, t in enumerate(self.tris):
            k1, k2 = self.tri_tets[j]
            for e in _faces(t):
                edge_uf.union((e, k1), (e, k2))
            for v in t:
                vertex_uf.union((v, k1), (v, k2))
        edge_roots: Dict[Simplex, set] = defaultdict(set)
        for k, T in enumerate(self.tets):
            for i in range(4):
                for l in range(i + 1, 4):
                    e = (T[i], T[l])
                    edge_roots[e].add(edge_uf.find((e, k)))
        if any(len(r) != 1 for r in edge_roots.values()):
            raise SeifertSurfaceError("Ambient complex is not a 3-manifold: an edge link is not a circle")
        n_edges_at: Dict[int, int] = defaultdict(int)
        n_tris_at: Dict[int, int] = defaultdict(int)
        for e in self.edges:
            for v in e:
                n_edges_at[v] += 1
        for t in self.tris:
            for v in t:
                n_tris_at[v] += 1
        for v, ks in self.vertex_tets.items():
            if len({vertex_uf.find((v, k)) for k in ks}) != 1 or (
                n_edges_at[v] - n_tris_at[v] + len(ks) != 2
            ):
                raise SeifertSurfaceError(
                    f"Ambient complex is not a 3-manifold: the link of vertex {v} is not a 2-sphere"
                )

    def _orient(self, n_original: int) -> np.ndarray:
        """Coherent tetrahedron orientations ε, relative to sorted vertex order.

        The global sign comes from the vertex coordinates (majority of the
        geometric orientations of the original tetrahedra) or, without them,
        by convention: the lexicographically first original tetrahedron is
        positive.
        """
        n = len(self.tets)
        eps = np.zeros(n, dtype=np.int64)
        eps[0] = 1
        queue = deque([0])
        while queue:
            k = queue.popleft()
            for j, s in self.tet_faces[k]:
                for m in self.tri_tets[j]:
                    if m == k:
                        continue
                    # Neighbours induce opposite orientations on their common face.
                    s_m = next(sm for jm, sm in self.tet_faces[m] if jm == j)
                    want = -eps[k] * s * s_m
                    if eps[m] == 0:
                        eps[m] = want
                        queue.append(m)
                    elif eps[m] != want:
                        raise SeifertSurfaceError("Ambient 3-manifold is not orientable")

        vote = 0
        for k in range(n_original):
            T = self.tets[k]
            if all(v in self.coords for v in T):
                p = np.array([self.coords[v] for v in T])
                vote += int(eps[k]) * int(np.sign(np.linalg.det(p[1:] - p[0])))
        if vote != 0:
            flip = vote < 0
        else:
            flip = eps[self.tet_index[min(self.tets[:n_original])]] < 0
        return -eps if flip else eps

    def face_sign(self, k: int, j: int) -> int:
        """Sign of triangle j in ∂(ε_k · tet k): +1 iff its normal points out of tet k."""
        return int(self.eps[k]) * next(s for jj, s in self.tet_faces[k] if jj == j)

    # ── Chains ──────────────────────────────────────────────────────────────

    def cycle_vector(self, walk: Sequence[int]) -> np.ndarray:
        """Edge vector of the closed walk walk[0] → walk[1] → … → walk[0]."""
        vec = np.zeros(len(self.edges), dtype=np.int64)
        n = len(walk)
        for i in range(n):
            a, b = walk[i], walk[(i + 1) % n]
            e = (a, b) if a < b else (b, a)
            if e not in self.edge_index:
                raise SeifertSurfaceError(f"({a}, {b}) is not an edge of the ambient complex")
            vec[self.edge_index[e]] += 1 if a < b else -1
        return vec

    def min_area_chain(self, b: np.ndarray, w: np.ndarray) -> Optional[np.ndarray]:
        """An integral 2-chain F with ∂F = b minimising Σ w_t |F_t|, or None."""
        n = len(self.tris)
        res = linprog(
            np.concatenate([w, w]),
            A_eq=hstack([self.B2, -self.B2]).tocsr(),
            b_eq=b.astype(np.float64),
            bounds=(0, None),
            method="highs-ds",
        )
        if res.status != 0:
            return None
        x = res.x[:n] - res.x[n:]
        F = np.rint(x).astype(np.int64)
        if np.max(np.abs(x - F), initial=0.0) > 1e-6 or not np.array_equal(self.B2 @ F, b):
            return None
        return F

    def bounding_chains(self, cycles: np.ndarray) -> np.ndarray:
        """Integral 2-chains G with ∂G = cycles[:, i], one column per cycle.

        With T a spanning tree of the 1-skeleton and T* one of the dual graph,
        ∂_2 restricted to (triangles ∉ T*) × (edges ∉ T) is the cellular
        boundary map of a CW structure with one vertex and one 3-cell, so it
        is unimodular exactly when H_1 = H_2 = 0.
        """
        if self._tree_cotree is None:
            adj: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
            for i, (a, c) in enumerate(self.edges):
                adj[a].append((c, i))
                adj[c].append((a, i))
            root = self.edges[0][0]
            seen_v, tree, queue = {root}, set(), deque([root])
            while queue:
                u = queue.popleft()
                for w, i in adj[u]:
                    if w not in seen_v:
                        seen_v.add(w)
                        tree.add(i)
                        queue.append(w)
            seen_t, cotree, queue = {0}, set(), deque([0])
            while queue:
                k = queue.popleft()
                for j, _ in self.tet_faces[k]:
                    for m in self.tri_tets[j]:
                        if m not in seen_t:
                            seen_t.add(m)
                            cotree.add(j)
                            queue.append(m)
            rows = np.array([i for i in range(len(self.edges)) if i not in tree])
            cols = np.array([j for j in range(len(self.tris)) if j not in cotree])
            try:
                lu = splu(self.B2[rows][:, cols].astype(np.float64).tocsc())
            except RuntimeError as exc:
                raise SeifertSurfaceError(
                    "Ambient complex is not a homology 3-sphere, so K need not bound "
                    "a Seifert surface"
                ) from exc
            self._tree_cotree = (rows, cols, lu)

        rows, cols, lu = self._tree_cotree
        sol = lu.solve(cycles[rows].astype(np.float64))
        G = np.zeros((len(self.tris), cycles.shape[1]), dtype=np.int64)
        G[cols] = np.rint(sol).astype(np.int64)
        if np.max(np.abs(sol - G[cols]), initial=0.0) < 1e-6 and np.array_equal(self.B2 @ G, cycles):
            return G
        # The float LU lost precision: fall back to exact LP vertices.
        for c in range(cycles.shape[1]):
            chain = self.min_area_chain(cycles[:, c], np.ones(len(self.tris)))
            if chain is None:
                raise SeifertSurfaceError("Could not find an integral 2-chain bounding a surface cycle")
            G[:, c] = chain
        return G

class _Surface:
    """The support of an integral 2-chain F with ∂F = K, as a candidate Seifert surface."""

    def __init__(self, tri: _Triangulation, F: np.ndarray, knot_edges: set) -> None:
        self.tri = tri
        self.F = F
        self.support = [int(j) for j in np.nonzero(F)[0]]
        self.support_set = set(self.support)
        self.edge_tris: Dict[Simplex, List[int]] = defaultdict(list)
        for j in self.support:
            for e in _faces(tri.tris[j]):
                self.edge_tris[e].append(j)
        self.knot_edges = knot_edges

    def singular_triangles(self) -> set:
        """Triangles where the support fails to be an embedded surface with boundary K.

        The support is embedded iff |F| = 1 on it, every edge lies in two of
        its triangles (one for edges of K), and the link of every vertex in
        it is connected, hence a single arc or circle.  Returns the triangles
        carrying |F| ≠ 1 or meeting an offending edge or vertex.
        """
        bad = {j for j in self.support if abs(int(self.F[j])) != 1}
        for e, js in self.edge_tris.items():
            if len(js) != (1 if e in self.knot_edges else 2):
                bad.update(js)
        if bad:
            return bad
        # Every link vertex now has degree 1 (knot edges) or 2, so the link of a
        # vertex in the surface is a disjoint union of arcs and circles.
        links: Dict[int, List[Tuple[Simplex, int]]] = defaultdict(list)
        for j in self.support:
            t = self.tri.tris[j]
            for i, v in enumerate(t):
                links[v].append((t[:i] + t[i + 1:], j))
        for link_edges in links.values():
            uf = _UnionFind()
            for (a, b), _ in link_edges:
                uf.union(a, b)
            if len({uf.find(a) for (a, _), _ in link_edges}) != 1:
                bad.update(j for _, j in link_edges)
        return bad

    def h1_basis(self) -> List[List[int]]:
        """Closed vertex walks on the surface forming a basis of H_1(F; Z).

        Tree–cotree: with T a spanning tree of the surface's 1-skeleton and C a
        spanning tree of its dual graph (triangles, plus one node capping the
        boundary) that avoids T, the 2g edges in neither close up, with T, a
        basis of H_1 of the capped surface, which is H_1(F) as ∂F is connected.
        """
        edges = sorted(self.edge_tris)
        adj: Dict[int, List[int]] = defaultdict(list)
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)
        root = edges[0][0]
        parent: Dict[int, Optional[int]] = {root: None}
        tree = set()
        queue = deque([root])
        while queue:
            u = queue.popleft()
            for w in adj[u]:
                if w not in parent:
                    parent[w] = u
                    tree.add((u, w) if u < w else (w, u))
                    queue.append(w)
        if len(parent) != len(adj):
            raise SeifertSurfaceError("Seifert surface is disconnected")

        dual = _UnionFind()
        leftover = []
        for e in edges:
            if e in tree:
                continue
            js = self.edge_tris[e]
            if not dual.union(js[0], js[1] if len(js) == 2 else "cap"):
                leftover.append(e)
        chi = len(adj) - len(edges) + len(self.support)
        if len(leftover) != 1 - chi:
            raise SeifertSurfaceError("Tree–cotree decomposition of the Seifert surface failed")

        def to_root(v: int) -> List[int]:
            path = [v]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            return path

        cycles = []
        for a, b in leftover:
            pa, pb = to_root(a), to_root(b)
            on_pa = set(pa)
            lca = next(x for x in pb if x in on_pa)
            # a → b, then along the tree b → lca → a.
            cycles.append([a] + pb[: pb.index(lca) + 1] + pa[1 : pa.index(lca)][::-1])
        return cycles

    def positive_tet(self, e: Simplex) -> int:
        """A tetrahedron containing edge e on the positive side of the surface."""
        tri = self.tri
        j = self.edge_tris[e][0]
        # The normal of F_j·[t_j] points into tet k iff F_j · (sign of t_j in ∂k) = −1.
        return next(k for k in tri.tri_tets[j] if int(self.F[j]) * tri.face_sign(k, j) == -1)

    def pushoff_crossings(self, walk: List[int]) -> np.ndarray:
        """The positive push-off of a closed walk, as signed triangle crossings.

        Along each edge of the walk the push-off sits in a positive-side
        tetrahedron; at each vertex v it moves between consecutive ones through
        tetrahedra around v, crossing only triangles through v that are not in
        the surface.  At a vertex off K this keeps it on the positive side; at
        a vertex of K the complement of the surface in the star is a ball, so
        any such route is homotopic to the push-off.  Entry j is the signed
        number of times the path crosses triangle j along its normal.
        """
        tri = self.tri
        n = len(walk)
        pos = []
        for i in range(n):
            a, b = walk[i], walk[(i + 1) % n]
            pos.append(self.positive_tet((a, b) if a < b else (b, a)))
        crossings = np.zeros(len(tri.tris), dtype=np.int64)
        for i in range(n):
            v, src, dst = walk[i], pos[i - 1], pos[i]
            prev: Dict[int, Optional[Tuple[int, int]]] = {src: None}
            queue = deque([src])
            while queue and dst not in prev:
                k = queue.popleft()
                T = tri.tets[k]
                for idx, (j, _) in enumerate(tri.tet_faces[k]):
                    # Face idx omits T[idx], so it contains v unless T[idx] == v.
                    if T[idx] == v or j in self.support_set:
                        continue
                    for m in tri.tri_tets[j]:
                        if m != k and m not in prev:
                            prev[m] = (k, j)
                            queue.append(m)
            if dst not in prev:
                raise SeifertSurfaceError(f"Push-off is blocked at vertex {v} of the Seifert surface")
            k = dst
            while prev[k] is not None:
                k_prev, j = prev[k]
                crossings[j] += tri.face_sign(k_prev, j)
                k = k_prev
        return crossings


def _integer_det(M: np.ndarray) -> int:
    """Exact determinant of a small integer matrix (fraction-free Bareiss)."""
    A = [[int(x) for x in row] for row in M]
    n = len(A)
    if n == 0:
        return 1
    sign, prev = 1, 1
    for k in range(n - 1):
        if A[k][k] == 0:
            swap = next((i for i in range(k + 1, n) if A[i][k] != 0), None)
            if swap is None:
                return 0
            A[k], A[swap] = A[swap], A[k]
            sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i][j] = (A[i][j] * A[k][k] - A[i][k] * A[k][j]) // prev
        prev = A[k][k]
    return sign * A[n - 1][n - 1]


def _seifert_form(surface: _Surface) -> np.ndarray:
    tri = surface.tri
    cycles = surface.h1_basis()
    m = len(cycles)
    if m == 0:
        return np.zeros((0, 0), dtype=np.int64)
    betas = np.column_stack([tri.cycle_vector(w) for w in cycles])
    G = tri.bounding_chains(betas)
    C = np.array([surface.pushoff_crossings(w) for w in cycles])
    V = (C @ G).astype(np.int64)  # V_ij = α_i⁺ · G_j = lk(α_i⁺, α_j)
    det = _integer_det(V - V.T)
    if det != 1:
        raise SeifertSurfaceError(f"Computed Seifert form has det(V − Vᵀ) = {det} ≠ 1")
    return V


def seifert_matrix_of_triangulation(
    tets: Sequence[Sequence[int]],
    knot_walk: Sequence[int],
    coords: Optional[Dict[int, np.ndarray]] = None,
) -> np.ndarray:
    """Seifert matrix of a knot in a triangulated 3-ball or 3-sphere.

    Args:
        tets: Tetrahedra of the ambient complex.  The component containing
            the knot must be a combinatorial 3-manifold that becomes a
            homology 3-sphere once each boundary 2-sphere is coned off, e.g.
            a triangulated 3-ball or S^3.
        knot_walk: Vertices of the knot K in cyclic order; consecutive vertices
            (and the last and first) must span edges of the ambient complex.
            This orients K; reversing it transposes the result.
        coords: Optional vertex coordinates in R^3.  They are used only to
            orient the ambient complex; without them the lexicographically
            first tetrahedron, with its vertices in increasing order, is
            positively oriented.

    Returns:
        V with V_ij = lk(α_i⁺, α_j) for a basis {α_i} of H_1(F) of a
        minimal-area Seifert surface F in the triangulation.  It is 2h × 2h
        with h the genus of F, which is at least the Seifert genus of K.

    Raises:
        SeifertSurfaceError: if the ambient complex is not a suitable
            3-manifold, or no embedded Seifert surface is found.
    """
    tets = [tuple(sorted(int(v) for v in T)) for T in tets]
    if not tets or any(len(set(T)) != 4 for T in tets):
        raise SeifertSurfaceError("Ambient complex must be a nonempty set of tetrahedra")
    knot = [int(v) for v in knot_walk]
    closed, n_original = _close_up(tets, knot[0])
    tri = _Triangulation(closed, coords, n_original=n_original)
    b = tri.cycle_vector(knot)
    knot_edges = {(min(a, c), max(a, c)) for a, c in zip(knot, knot[1:] + knot[:1])}

    for seed in range(_LP_ATTEMPTS):
        # Unit areas, plus a small perturbation that breaks ties.
        w = 1.0 + 1e-3 * np.random.default_rng(seed).random(len(tri.tris))
        for _ in range(_REWEIGHT_ROUNDS):
            F = tri.min_area_chain(b, w)
            if F is None:
                raise SeifertSurfaceError("K does not bound an integral 2-chain in the ambient complex")
            surface = _Surface(tri, F, knot_edges)
            singular = surface.singular_triangles()
            if not singular:
                return _seifert_form(surface)
            w[list(singular)] *= 2.0
    raise SeifertSurfaceError(
        "No embedded Seifert surface found: the minimal-area 2-chains bounded by K "
        "keep doubling up or touching themselves; a finer triangulation may help"
    )
