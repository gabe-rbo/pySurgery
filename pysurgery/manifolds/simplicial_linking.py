"""pysurgery/manifolds/simplicial_linking.py.

Linking numbers of 1-cycles in a triangulated 3-manifold, read off the
triangulation alone, with no vertex coordinates.

A primal 1-cycle a and a primal 2-chain G are never transverse: they meet
along edges and vertices.  The naive pairing Σ a[σ]·G[τ]·[τ:σ] over edges
σ ⊂ triangles τ is ⟨a, ∂G⟩, which vanishes whenever a and ∂G are disjoint.
It is not an intersection number.  Instead:

  1. Close the ambient complex up.  Cone off each boundary 2-sphere (a 3-ball
     becomes S^3), check that the result is a connected combinatorial
     3-manifold, and orient it coherently.
  2. Replace a by a dual 1-cycle z_a, a closed path of tetrahedra in which
     each step crosses one triangle.
     - For each vertex v of a, fix a hub tetrahedron H_v ∋ v.
     - For each edge u → w of a with coefficient c, take a tetrahedron
       T ⊃ {u, w} and add c·(H_u → T → H_w).
     - The leg H_u → T runs through tetrahedra around u and crosses only
       triangles that contain u; likewise for T → H_w around w.
     z_a is closed because ∂a = 0 at every hub.  It lies in the open star of
     a and is homologous to a there, so it misses every cycle that shares
     no vertex with a.
  3. Find an integral 2-chain G with ∂G = b, by tree–cotree LU.
  4. Then lk(a, b) = z_a · G = Σ_t z_a[t]·G[t], because a dual path meets a
     2-chain transversally, one triangle at a time.

Sign convention: lk is positive when a crosses G along its normal, taken by
the right-hand rule from the orientation of ∂G = b.  With vertex coordinates
the 3-manifold carries the orientation of R^3, and lk agrees with the Gauss
linking integral.  Without them, the lexicographically first tetrahedron,
with its vertices in increasing order, is taken to be positive.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import splu

Simplex = Tuple[int, ...]


class NotAClosable3ManifoldError(ValueError):
    """Raised when the ambient complex cannot be closed up to an oriented 3-manifold."""


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


def component_tets(tets: Sequence[Simplex], vertex: int) -> List[Simplex]:
    """The tetrahedra in the connected component containing `vertex`."""
    uf = _UnionFind()
    for T in tets:
        for v in T[1:]:
            uf.union(T[0], v)
    root = uf.find(vertex)
    return sorted({T for T in tets if uf.find(T[0]) == root})


def _close_up(tets: List[Simplex], apex_base: int) -> List[Simplex]:
    """Cone off each boundary 2-sphere of a connected 3-complex.

    The first len(tets) tetrahedra of the result are the input; the rest are
    cones over each boundary component from new apices apex_base + 1, … .
    Coning off 2-spheres leaves H_1 unchanged, so linking numbers of 1-cycles
    in the closed-up manifold are those in the original complex.
    """
    count: Dict[Simplex, int] = defaultdict(int)
    for T in tets:
        for t in _faces(T):
            count[t] += 1
    if any(c > 2 for c in count.values()):
        raise NotAClosable3ManifoldError("a triangle lies in more than two tetrahedra")
    boundary = [t for t, c in count.items() if c == 1]
    if not boundary:
        return list(tets)

    edge_tris: Dict[Simplex, List[int]] = defaultdict(list)
    for i, t in enumerate(boundary):
        for e in _faces(t):
            edge_tris[e].append(i)
    if any(len(ts) != 2 for ts in edge_tris.values()):
        raise NotAClosable3ManifoldError("the boundary is not a closed surface")
    uf = _UnionFind()
    for ts in edge_tris.values():
        uf.union(ts[0], ts[1])
    components: Dict[object, List[Simplex]] = defaultdict(list)
    for i, t in enumerate(boundary):
        components[uf.find(i)].append(t)

    apex = max(apex_base, max(max(T) for T in tets))
    closed = list(tets)
    for comp in components.values():
        n_v = len({v for t in comp for v in t})
        n_e = len({e for t in comp for e in _faces(t)})
        if n_v - n_e + len(comp) != 2:
            raise NotAClosable3ManifoldError("a boundary component is not a 2-sphere")
        apex += 1
        closed.extend(t + (apex,) for t in comp)
    return closed


class Oriented3Manifold:
    """A closed, connected, coherently oriented combinatorial 3-manifold.

    Built from the tetrahedra of one component of an ambient complex, with
    each boundary 2-sphere coned off.  Cone apices are labelled above
    `apex_base` (default: the largest vertex), so that they do not collide
    with vertices outside `tets`.
    """

    def __init__(
        self,
        tets: Sequence[Sequence[int]],
        coords: Optional[Mapping[int, np.ndarray]] = None,
        apex_base: Optional[int] = None,
    ) -> None:
        original = sorted({tuple(sorted(int(v) for v in T)) for T in tets})
        if not original or any(len(set(T)) != 4 for T in original):
            raise NotAClosable3ManifoldError("the ambient complex has no tetrahedra")
        self.n_original = len(original)
        self.tets: List[Simplex] = _close_up(original, -1 if apex_base is None else apex_base)
        self.coords = dict(coords or {})
        self.tris: List[Simplex] = sorted({t for T in self.tets for t in _faces(T)})
        self.edges: List[Simplex] = sorted({e for t in self.tris for e in _faces(t)})
        self.tri_index = {t: j for j, t in enumerate(self.tris)}
        self.edge_index = {e: i for i, e in enumerate(self.edges)}

        # tet_faces[k][i] = (triangle opposite T[i], its sign in ∂[T sorted])
        self.tet_faces: List[List[Tuple[int, int]]] = []
        self.tri_tets: List[List[int]] = [[] for _ in self.tris]
        self.vertex_tets: Dict[int, List[int]] = defaultdict(list)
        for k, T in enumerate(self.tets):
            faces = []
            for i, t in enumerate(_faces(T)):
                j = self.tri_index[t]
                faces.append((j, -1 if i % 2 else 1))
                self.tri_tets[j].append(k)
            self.tet_faces.append(faces)
            for v in T:
                self.vertex_tets[v].append(k)

        self._check_manifold()
        self.eps = self._orient()

        rows, cols, vals = [], [], []
        for j, t in enumerate(self.tris):
            for i, e in enumerate(_faces(t)):
                rows.append(self.edge_index[e])
                cols.append(j)
                vals.append(-1 if i % 2 else 1)
        self.B2 = csr_matrix(
            coo_matrix((vals, (rows, cols)), shape=(len(self.edges), len(self.tris)), dtype=np.int64)
        )
        self._tree_cotree: Optional[Tuple[np.ndarray, np.ndarray, object]] = None
        self._star_trees: Dict[int, Dict[int, Optional[Tuple[int, int]]]] = {}

    # ── Validation and orientation ──────────────────────────────────────────

    def _check_manifold(self) -> None:
        if any(len(ks) != 2 for ks in self.tri_tets):
            raise NotAClosable3ManifoldError("the closed-up complex is not a 3-manifold")
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
            raise NotAClosable3ManifoldError("an edge link is not a circle")
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
                raise NotAClosable3ManifoldError(f"the link of vertex {v} is not a 2-sphere")

    def _orient(self) -> np.ndarray:
        """Coherent tetrahedron orientations ε, relative to sorted vertex order.

        The global sign is set by the vertex coordinates, as the majority of
        the geometric orientations of the original tetrahedra.  Without
        coordinates, the first original tetrahedron is positive.
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
                        raise NotAClosable3ManifoldError("the ambient 3-manifold is not orientable")

        vote = 0
        for k in range(self.n_original):
            T = self.tets[k]
            if all(v in self.coords for v in T):
                p = np.array([self.coords[v] for v in T], dtype=np.float64)
                vote += int(eps[k]) * int(np.sign(np.linalg.det(p[1:] - p[0])))
        return -eps if vote < 0 else eps

    def face_sign(self, k: int, j: int) -> int:
        """Sign of triangle j in ∂(ε_k · tet k): +1 iff its normal points out of tet k."""
        return int(self.eps[k]) * next(s for jj, s in self.tet_faces[k] if jj == j)

    # ── Chains ──────────────────────────────────────────────────────────────

    def edge_vector(self, chain: Mapping[Simplex, int]) -> np.ndarray:
        """The 1-chain {(u, w): c} (u < w, oriented u → w) as a vector over self.edges."""
        vec = np.zeros(len(self.edges), dtype=np.int64)
        for e, c in chain.items():
            if e not in self.edge_index:
                raise NotAClosable3ManifoldError(f"{e} is not an edge of a tetrahedron")
            vec[self.edge_index[e]] += int(c)
        return vec

    def _build_tree_cotree(self) -> None:
        adj: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for i, (u, w) in enumerate(self.edges):
            adj[u].append((w, i))
            adj[w].append((u, i))
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
        rows = np.array([i for i in range(len(self.edges)) if i not in tree], dtype=np.int64)
        cols = np.array([j for j in range(len(self.tris)) if j not in cotree], dtype=np.int64)
        lu = None
        try:
            lu = splu(self.B2[rows][:, cols].astype(np.float64).tocsc())
        except RuntimeError:  # singular: b_1 > 0
            pass
        self._tree_cotree = (rows, cols, lu)

    @property
    def is_rational_homology_sphere(self) -> bool:
        """True iff the tree–cotree system is nonsingular, i.e. b_1 = 0."""
        if self._tree_cotree is None:
            self._build_tree_cotree()
        return self._tree_cotree[2] is not None

    def bounding_chain(self, b: np.ndarray) -> Optional[np.ndarray]:
        """An integral 2-chain G with ∂G = b by tree–cotree, or None.

        With T a spanning tree of the 1-skeleton and T* one of the dual graph,
        ∂_2 restricted to (triangles ∉ T*) × (edges ∉ T) is the cellular
        boundary map of a CW structure with one vertex and one 3-cell.  It is
        square because χ = 0, unimodular exactly when H_1 = 0, and singular
        when b_1 > 0.  None means that the system gave no integral solution;
        b may still bound over ℤ when H_1 has torsion or b_1 > 0.
        """
        if not self.is_rational_homology_sphere:
            return None
        rows, cols, lu = self._tree_cotree
        sol = lu.solve(b[rows].astype(np.float64))
        G = np.zeros(len(self.tris), dtype=np.int64)
        G[cols] = np.rint(sol).astype(np.int64)
        if np.max(np.abs(sol - G[cols]), initial=0.0) < 1e-6 and np.array_equal(self.B2 @ G, b):
            return G
        return None

    # ── Dual 1-cycles ───────────────────────────────────────────────────────

    def _star_tree(self, v: int) -> Dict[int, Optional[Tuple[int, int]]]:
        """BFS tree of the tetrahedra around v, rooted at v's hub.

        Tetrahedra are joined through triangles that contain v.  Maps each
        tetrahedron to (parent, triangle crossed), or None at the hub.
        """
        tree = self._star_trees.get(v)
        if tree is None:
            hub = min(self.vertex_tets[v])
            tree = {hub: None}
            queue = deque([hub])
            while queue:
                k = queue.popleft()
                T = self.tets[k]
                for idx, (j, _) in enumerate(self.tet_faces[k]):
                    # Face idx omits T[idx], so it contains v unless T[idx] == v.
                    if T[idx] == v:
                        continue
                    for m in self.tri_tets[j]:
                        if m != k and m not in tree:
                            tree[m] = (k, j)
                            queue.append(m)
            self._star_trees[v] = tree
        return tree

    def _add_hub_path(self, v: int, k: int, c: int, crossings: np.ndarray) -> None:
        """Add c times the dual path from v's hub to tetrahedron k ∋ v."""
        tree = self._star_tree(v)
        while tree[k] is not None:
            parent, j = tree[k]
            crossings[j] += c * self.face_sign(parent, j)
            k = parent

    def dual_cycle(self, chain: Mapping[Simplex, int]) -> np.ndarray:
        """Signed triangle crossings of a dual 1-cycle homologous to `chain`.

        `chain` maps edges (u, w), u < w, oriented u → w, to coefficients and
        must be a cycle.  Entry j of the result is the signed number of times
        the dual cycle crosses triangle j along its normal.  The dual cycle
        stays in the open star of the chain's support.
        """
        crossings = np.zeros(len(self.tris), dtype=np.int64)
        for (u, w), c in chain.items():
            if c == 0:
                continue
            shared = set(self.vertex_tets[w])
            T = next((k for k in self.vertex_tets[u] if k in shared), None)
            if T is None:
                raise NotAClosable3ManifoldError(f"({u}, {w}) is not an edge of a tetrahedron")
            self._add_hub_path(u, T, int(c), crossings)   # H_u → T
            self._add_hub_path(w, T, -int(c), crossings)  # T → H_w
        return crossings
