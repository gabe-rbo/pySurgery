r"""Linking numbers and Milnor invariants of links in a triangulated 3-manifold.

Overview:
    Everything here is intrinsic to the triangulation -- no vertex coordinates are used --
    and exact: rational or integer linear algebra, integer answers (or a refusal).

        triangulated_linking_number(M, J, K)      lk(J, K)
        triangulated_milnor_mu123(M, K1, K2, K3)  Milnor's mu-bar(123)
        triangulated_sato_levine(M, K1, K2)       the Sato-Levine invariant beta = -mu-bar(1122)
        component_vertex_cycle(K)                 the orientation convention for components

    All of them are intersection numbers, and an intersection number of simplicial
    chains is only meaningful for chains in general position. Two primal chains of one
    triangulation never are (a primal curve lies inside the primal 2-chains it would have
    to cross; two primal 2-chains share edges), which is why the linking number cannot be
    read off as ``<J, F>`` with ``dF = K`` (that pairing is ``<J, K>``, zero for disjoint
    J, K) and why the Alexander-Whitney front-face/back-face product of two primal
    2-chains is a cup product of cochains, not ``F_1 cap F_2``. Chains of the DUAL cell
    decomposition are transverse to primal chains:

    - a dual 1-cell t* (the segment joining the barycentres of the two tetrahedra on a
      triangle t) crosses t, and only t, at the barycentre of t;
    - a dual 2-cell e* (the polygon around an edge e, through the barycentres of the
      tetrahedra and triangles containing e) meets a primal triangle t containing e in the
      segment from the midpoint of e to the barycentre of t, and misses all other
      triangles.

Key Concepts:
    - **Linking number.** ``lk(J, K) = I(J*, F)``: J* is a dual 1-cycle (a closed path of
      tetrahedra through shared triangles) running inside the open star of J, homotopic
      to J there and hence away from K; F is ANY rational 2-chain with ``dF = K``. The
      value does not depend on F, because two choices differ by a 2-cycle, which bounds
      when ``H_2(M; Q) = 0``, and a closed dual cycle meets a boundary zero times.
    - **Triple linking number.** For a link with vanishing pairwise linking numbers,

          mu-bar(123) = -lk(F_1 cap G_2, K_3),

      where F_1 is a PRIMAL 2-chain with ``dF_1 = K_1`` that avoids the dual push-off K_2*
      of K_2 (it has no triangle crossed by K_2*), and G_2 is a DUAL 2-chain with
      ``dG_2 = K_2*`` that avoids K_1 and K_3. Poincare duality identifies dual 2-chains
      with primal 1-cochains g, with ``dG_2 = K_2*`` becoming ``delta g = beta_2`` (beta_2
      the cochain counting signed crossings of K_2* through each triangle) and
      "G_2 avoids K_i" becoming "g vanishes on the edges of K_i". The two chains are
      transverse, their intersection is the closed curve
      ``A = sum F_1[t] g(e) [t:e] (midpoint(e) -> barycentre(t))``, oriented so that
      (normal of F_1, normal of G_2, tangent) is positive (closed exactly because F_1
      avoids K_2* and G_2 avoids K_1), and A misses K_3. Pushing each segment of A
      into the ring of tetrahedra around its edge e (which avoids K_3 since e is not an
      edge of K_3) turns A into a dual cycle, so ``lk(A, K_3) = I(A*, F_3)`` for any
      2-chain F_3 bounded by K_3. The result depends on none of the choices (F_1, g, F_3,
      the push-off) once all pairwise linking numbers vanish -- the same argument as for
      Seifert surfaces with ``F_i cap K_j`` empty (Cochran, *Derivatives of links*, Mem.
      AMS 427, 1990; Mellor-Melvin, AGT 3, 2003) -- and replacing K_2 by K_2* is a link
      homotopy, under which mu-bar(123) is invariant (Milnor, 1954). The minus sign is
      the Magnus-expansion convention of ``diagrams.diagram_milnor_mu123``.
    - **Sato-Levine invariant.** For two components with lk = 0, ``beta = lk(C, C+)``
      with C = F_1 cap G_2 as above, but now F_1 and G_2 must be EMBEDDED surfaces: C+
      is the push-off of C along their normals, which chains with multiplicities or
      branching do not have. F_1 is a minimal-area 2-chain with coefficients +-1 whose
      support is a surface, G_2 a 1-cochain with values +-1 (its dual 2-cells then form a
      surface), both found by integer programs whose constraints enforce embedding
      along every edge. At the midpoint of an edge e the normal of G_2 points along e,
      so C+ slides, off C, onto the primal path P through the endpoint of e it points
      to, and ``beta = lk(C, P)``. Where the triangulation leaves no room for the
      surfaces it is refined first (stellar subdivisions separating the components,
      then a barycentric subdivision), which changes no invariant.
    - **Push-offs.** K* follows K through the open star of K: for consecutive edges
      ``[x_{j-1}, x_j]``, ``[x_j, x_{j+1}]`` it walks among the tetrahedra around x_j
      from one containing the first edge to one containing the second. Stellar
      subdivisions of the "chords" of each component first make it a FULL subcomplex, so
      no tetrahedron is visited twice and K* is an embedded circle whose complement has
      the homology of the complement of K.
    - **Ambient.** A closed, orientable, connected combinatorial 3-manifold with
      ``H_1(M; Q) = 0`` -- e.g. a triangulated S^3 -- or a triangulated 3-ball (more
      generally, a manifold whose boundary components are 2-spheres, each capped off by a
      cone before computing). Every hypothesis is checked; a failure raises instead of
      returning a number.

Conventions:
    A component is oriented by traversing its lexicographically largest edge ``(u, v)``,
    ``u < v``, from u to v (``component_vertex_cycle`` lists its vertices in that order);
    this is the orientation ``manifolds.surgery.compute_linking_number`` uses. The
    ambient is oriented coherently, with the convention of ``manifolds.simplicial_linking``:
    when vertex coordinates are attached, as the majority of the input tetrahedra orient
    R^3 (so ``lk`` agrees with the Gauss integral and with
    ``diagrams.diagram_linking_number``), otherwise so that the lexicographically first
    input tetrahedron, with its vertices in increasing order, is positive. ``mu-bar(123)`` does not depend on the
    orientation of the ambient (it is unchanged by mirror image), changes sign when one
    component is reversed or two components are swapped, and is normalised to agree with
    ``diagrams.diagram_milnor_mu123`` (the Magnus-expansion convention) on polygons.

Coefficient Ring:
    Q for the chains (exact ``Fraction`` arithmetic); the invariants are integers.
"""

from __future__ import annotations

from collections import defaultdict, deque
from fractions import Fraction
from itertools import combinations
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, hstack, identity

from ..core.exceptions import NotAManifoldError, UndefinedInvariantError
from ..topology.complexes import SimplicialComplex
from .seifert_surface import SeifertSurfaceError

__all__ = [
    "component_vertex_cycle",
    "triangulated_linking_number",
    "triangulated_milnor_mu123",
    "triangulated_sato_levine",
]

Tet = Tuple[int, int, int, int]
Tri = Tuple[int, int, int]
Edge = Tuple[int, int]
Step = Tuple[Tet, Tet, Tri]

# Global sign making lk(F_1 cap G_2, K_3), with F_1 cap G_2 oriented by
# (normal of F_1, normal of G_2, tangent) positive, agree with the Magnus-expansion
# convention of `diagrams.diagram_milnor_mu123`.
_MU123_SIGN = -1

_RANK_PRIME = 2_147_483_647

_REWEIGHT_ROUNDS = 3  # re-solves penalising vertices where a surface touches itself


class _NoRoom(SeifertSurfaceError):
    """The integer program for a surface is infeasible in this triangulation."""


class _NotFound(SeifertSurfaceError):
    """No embedded surface found (it may exist): e.g. the minimal-area ones touch themselves."""


def _faces(simplex: Tuple[int, ...]) -> List[Tuple[Tuple[int, ...], int]]:
    """Codimension-one faces of a sorted simplex with their incidence numbers ``(-1)^i``."""
    return [(simplex[:i] + simplex[i + 1:], -1 if i % 2 else 1) for i in range(len(simplex))]


def _incidence(simplex: Tuple[int, ...], face: Tuple[int, ...]) -> int:
    """The incidence number ``[simplex : face]`` of a sorted simplex and a sorted facet."""
    (i,) = [k for k, v in enumerate(simplex) if v not in face]
    return -1 if i % 2 else 1


def component_vertex_cycle(component: SimplicialComplex) -> List[int]:
    """The vertices of a link component in the order of its orientation.

    A component must be a simple closed curve: a connected 1-dimensional complex whose
    vertices all have degree 2. It is oriented by traversing its lexicographically
    largest edge ``(u, v)``, ``u < v``, from u to v -- the orientation that
    ``manifolds.surgery.compute_linking_number`` assigns to a 1-cycle.

    Args:
        component: The component.

    Returns:
        ``[u, v, ...]``: its vertices in cyclic order (the closing edge is implicit).

    Raises:
        ValueError: If the component is not a simple closed curve.
    """
    if component.n_simplices(2):
        raise ValueError("a link component must be 1-dimensional; this one has triangles")
    edges = sorted(tuple(sorted(e)) for e in component.n_simplices(1))
    if not edges:
        raise ValueError("a link component must be a closed curve; this one has no edges")
    adjacent: Dict[int, List[int]] = defaultdict(list)
    for u, v in edges:
        adjacent[u].append(v)
        adjacent[v].append(u)
    for (v,) in component.n_simplices(0):
        if len(adjacent[v]) != 2:
            raise ValueError(
                f"a link component must be a simple closed curve; vertex {v} has degree "
                f"{len(adjacent[v])}"
            )
    u, v = edges[-1]
    cycle = [u, v]
    while True:
        a, b = adjacent[cycle[-1]]
        nxt = a if a != cycle[-2] else b
        if nxt == u:
            break
        cycle.append(nxt)
    if len(cycle) != len(edges):
        raise ValueError(
            f"a link component must be connected; this one has {len(edges)} edges but the "
            f"closed curve through its largest edge has {len(cycle)}"
        )
    return cycle


class _SparseEchelon:
    """Incremental sparse Gauss-Jordan elimination, exactly over Q or over F_p.

    Equations are added one at a time. Pivot rows are kept fully reduced (no pivot row
    contains another pivot column), so a new row is reduced in one pass, and the
    particular solution with every free variable zero is read off the pivot rows.
    """

    def __init__(self, modulus: Optional[int] = None):
        self.p = modulus
        self.rows: Dict[int, Tuple[dict, object]] = {}
        self.col_rows: Dict[int, Set[int]] = defaultdict(set)

    def _field(self, x):
        return x % self.p if self.p else Fraction(x)

    def _inv(self, x):
        return pow(x, self.p - 2, self.p) if self.p else 1 / x

    def _axpy(self, row: dict, coef, other: dict, owner: Optional[int] = None) -> None:
        """``row -= coef * other``, keeping ``col_rows`` current when row is pivot ``owner``."""
        for k, v in other.items():
            nv = row.get(k, 0) - coef * v
            if self.p:
                nv %= self.p
            if nv:
                if owner is not None and k not in row:
                    self.col_rows[k].add(owner)
                row[k] = nv
            elif k in row:
                del row[k]
                if owner is not None:
                    self.col_rows[k].discard(owner)

    def add(self, row: Dict[int, int], rhs: int = 0) -> bool:
        """Add the equation ``sum row[c] x_c = rhs``.

        Args:
            row: Sparse coefficients.
            rhs: Right-hand side.

        Returns:
            False iff the equation is inconsistent with the ones already added.
        """
        r = {c: self._field(v) for c, v in row.items() if v}
        r = {c: v for c, v in r.items() if v}
        b = self._field(rhs)
        for c in [c for c in r if c in self.rows]:
            coef = r[c]
            prow, pb = self.rows[c]
            self._axpy(r, coef, prow)
            b = b - coef * pb
            if self.p:
                b %= self.p
        if not r:
            return not b
        pivot = min(r, key=lambda c: (len(self.col_rows.get(c, ())), c))
        inv = self._inv(r[pivot])
        r = {c: (v * inv) % self.p if self.p else v * inv for c, v in r.items()}
        b = (b * inv) % self.p if self.p else b * inv
        for q in list(self.col_rows.pop(pivot, ())):
            qrow, qb = self.rows[q]
            coef = qrow[pivot]
            self._axpy(qrow, coef, r, owner=q)
            qb = qb - coef * b
            self.rows[q] = (qrow, qb % self.p if self.p else qb)
        self.rows[pivot] = (r, b)
        for c in r:
            if c != pivot:
                self.col_rows[c].add(pivot)
        return True

    @property
    def rank(self) -> int:
        """Number of independent equations added so far."""
        return len(self.rows)

    def solution(self) -> Dict[int, object]:
        """The solution with every free variable zero (nonzero entries only)."""
        return {c: b for c, (_, b) in self.rows.items() if b}


def _cone_off_boundary(tets: List[Tet]) -> List[Tet]:
    """Cap every boundary 2-sphere of a 3-dimensional pseudomanifold with a cone."""
    tri_tets: Dict[Tri, List[Tet]] = defaultdict(list)
    for tet in tets:
        for tri, _ in _faces(tet):
            tri_tets[tri].append(tet)
    for tri, ts in tri_tets.items():
        if len(ts) > 2:
            raise NotAManifoldError(
                f"triangle {tri} lies in {len(ts)} tetrahedra; a 3-manifold has at most 2"
            )
    boundary = sorted(tri for tri, ts in tri_tets.items() if len(ts) == 1)
    if not boundary:
        return list(tets)
    edge_tris: Dict[Edge, List[Tri]] = defaultdict(list)
    for tri in boundary:
        for e, _ in _faces(tri):
            edge_tris[e].append(tri)
    for e, ts in edge_tris.items():
        if len(ts) != 2:
            raise NotAManifoldError(
                f"the boundary is not a closed surface: edge {e} lies in {len(ts)} boundary "
                "triangles"
            )
    out = list(tets)
    apex = max(v for tet in tets for v in tet) + 1
    seen: Set[Tri] = set()
    for start in boundary:
        if start in seen:
            continue
        piece = [start]
        seen.add(start)
        k = 0
        while k < len(piece):
            for e, _ in _faces(piece[k]):
                for nb in edge_tris[e]:
                    if nb not in seen:
                        seen.add(nb)
                        piece.append(nb)
            k += 1
        n_vertices = len({v for tri in piece for v in tri})
        n_edges = len({e for tri in piece for e, _ in _faces(tri)})
        chi = n_vertices - n_edges + len(piece)
        if chi != 2:
            raise NotAManifoldError(
                f"a boundary component has Euler characteristic {chi}; only boundary "
                "2-spheres can be capped off (the ambient must be a 3-ball, or a rational "
                "homology sphere minus balls)"
            )
        out.extend(tri + (apex,) for tri in piece)
        apex += 1
    return out


def _check_vertex_links(tets: List[Tet]) -> None:
    """Every vertex link of a closed combinatorial 3-manifold is a 2-sphere."""
    link: Dict[int, List[Tri]] = defaultdict(list)
    for tet in tets:
        for i, v in enumerate(tet):
            link[v].append(tet[:i] + tet[i + 1:])
    for v, tris in link.items():
        edge_tris: Dict[Edge, List[Tri]] = defaultdict(list)
        for tri in tris:
            for e, _ in _faces(tri):
                edge_tris[e].append(tri)
        if any(len(ts) != 2 for ts in edge_tris.values()):
            raise NotAManifoldError(f"the link of vertex {v} is not a closed surface")
        seen = {tris[0]}
        queue = [tris[0]]
        while queue:
            tri = queue.pop()
            for e, _ in _faces(tri):
                for nb in edge_tris[e]:
                    if nb not in seen:
                        seen.add(nb)
                        queue.append(nb)
        chi = len({u for tri in tris for u in tri}) - len(edge_tris) + len(tris)
        if len(seen) != len(tris) or chi != 2:
            raise NotAManifoldError(f"the link of vertex {v} is not a 2-sphere")


def _sort_sign(vertices: Sequence[int]) -> Tuple[Tet, int]:
    """The sorted simplex and the sign of the permutation that sorts ``vertices``."""
    v = list(vertices)
    sign = 1
    for i in range(len(v)):
        for j in range(len(v) - 1 - i):
            if v[j] > v[j + 1]:
                v[j], v[j + 1] = v[j + 1], v[j]
                sign = -sign
    return tuple(v), sign


def _coherent_orientation(tets: List[Tet]) -> Dict[Tet, int]:
    """``o(tet)`` with ``o(a)[a:t] + o(b)[b:t] = 0`` on every triangle t, ``o(tets[0]) = +1``."""
    tri_tets: Dict[Tri, List[Tet]] = defaultdict(list)
    for tet in tets:
        for tri, _ in _faces(tet):
            tri_tets[tri].append(tet)
    orient = {tets[0]: 1}
    queue = deque([tets[0]])
    while queue:
        a = queue.popleft()
        for tri, _ in _faces(a):
            for b in tri_tets[tri]:
                if b == a:
                    continue
                want = -orient[a] * _incidence(a, tri) * _incidence(b, tri)
                if b not in orient:
                    orient[b] = want
                    queue.append(b)
                elif orient[b] != want:
                    raise NotAManifoldError("the ambient 3-manifold is not orientable")
    if len(orient) != len(tets):
        raise NotAManifoldError("the ambient 3-manifold is not connected")
    return orient


def _check_rational_homology_sphere(tets: Sequence[Tet]) -> None:
    """``H_1(M; Q) = 0`` (hence ``H_2(M; Q) = 0``): rank d_2 = #edges - #vertices + 1."""
    tris = sorted({tri for tet in tets for tri, _ in _faces(tet)})
    edges = sorted({e for tri in tris for e, _ in _faces(tri)})
    target = len(edges) - len({v for tet in tets for v in tet}) + 1
    column = {e: k for k, e in enumerate(edges)}

    def rank(modulus: Optional[int]) -> int:
        ech = _SparseEchelon(modulus)
        for tri in tris:
            ech.add({column[e]: s for e, s in _faces(tri)})
            if ech.rank == target:
                break
        return ech.rank

    # rank over F_p never exceeds rank over Q, so a full F_p rank certifies it.
    if rank(_RANK_PRIME) < target and rank(None) < target:
        raise UndefinedInvariantError(
            "the ambient 3-manifold has H_1(M; Q) != 0: linking numbers need a rational "
            "homology sphere (or ball)"
        )


def _separate(orient: Dict[Tet, int], cycles: Sequence[List[int]]) -> Dict[Tet, int]:
    """Stellar-subdivide every edge joining two different cycles.

    Afterwards no simplex has vertices on two cycles, so their open stars are disjoint
    and surfaces avoiding one cycle have room near the other. Every new edge contains
    the new vertex, so the joining edges can be collected once.
    """
    owner = {v: k for k, cycle in enumerate(cycles) for v in cycle}
    joining = sorted({e for tet in orient for e in combinations(tet, 2)
                      if e[0] in owner and e[1] in owner and owner[e[0]] != owner[e[1]]})
    fresh = max(v for tet in orient for v in tet) + 1
    for e in joining:
        orient = _stellar(orient, e, fresh)
        fresh += 1
    return orient


def _barycentric(orient: Dict[Tet, int], cycles: Sequence[List[int]]) -> Tuple[Dict[Tet, int], List[List[int]]]:
    """Barycentric subdivision of oriented tetrahedra, and of the cycles in it.

    Vertices keep their labels; the barycentre of every edge, triangle and tetrahedron
    gets a new one. The simplex ``[b(v0), b(v0 v1), b(v0 v1 v2), b(v0 v1 v2 v3)]`` is
    positively oriented relative to ``[v0, v1, v2, v3]`` (the affine map between them
    is triangular with positive diagonal), which carries the orientation over.
    """
    from itertools import permutations

    fresh = max(v for tet in orient for v in tet) + 1
    label: Dict[Tuple[int, ...], int] = {}

    def b(simplex: Tuple[int, ...]) -> int:
        nonlocal fresh
        if len(simplex) == 1:
            return simplex[0]
        if simplex not in label:
            label[simplex] = fresh
            fresh += 1
        return label[simplex]

    out: Dict[Tet, int] = {}
    for tet, o in orient.items():
        for order in permutations(tet):
            _, parity = _sort_sign(order)
            flag = [b(tuple(sorted(order[: k + 1]))) for k in range(4)]
            piece, sign = _sort_sign(flag)
            out[piece] = o * parity * sign
    new_cycles = []
    for cycle in cycles:
        walk: List[int] = []
        for i, v in enumerate(cycle):
            walk += [v, b(tuple(sorted((v, cycle[(i + 1) % len(cycle)]))))]
        new_cycles.append(walk)
    return out, new_cycles


def _stellar(orient: Dict[Tet, int], simplex: Tuple[int, ...], w: int) -> Dict[Tet, int]:
    """Stellar subdivision of ``simplex`` (an edge or a triangle) at a new vertex w.

    A tetrahedron containing the simplex is replaced by the pieces obtained by
    putting w in the place of one vertex of the simplex; a piece keeps the
    orientation of its tetrahedron with w in that place, so the orientation stays
    coherent.
    """
    s = set(simplex)
    out: Dict[Tet, int] = {}
    for tet, o in orient.items():
        if s.issubset(tet):
            for v in simplex:
                piece, sign = _sort_sign([w if u == v else u for u in tet])
                out[piece] = o * sign
        else:
            out[tet] = o
    return out


def _make_full(orient: Dict[Tet, int], cycles: Sequence[List[int]]) -> Dict[Tet, int]:
    """Subdivide until every cycle is a full subcomplex (spans no other simplex).

    A chord (an edge joining two non-consecutive vertices of a cycle) or, for a 3-cycle,
    the triangle it spans, is removed by a stellar subdivision, which introduces no new
    simplex spanned by vertices of any cycle. Takes and returns oriented tetrahedra.
    """
    fresh = max(v for tet in orient for v in tet) + 1
    for cycle in cycles:
        on = set(cycle)
        own = {tuple(sorted((cycle[i], cycle[(i + 1) % len(cycle)]))) for i in range(len(cycle))}
        while True:
            bad: Optional[Tuple[int, ...]] = None
            for tet in orient:
                inside = tuple(v for v in tet if v in on)
                if len(inside) < 2:
                    continue
                bad = next((e for e in combinations(inside, 2) if e not in own), None)
                if bad is None and len(inside) >= 3:
                    bad = inside[:3]
                if bad is not None:
                    break
            if bad is None:
                break
            orient = _stellar(orient, bad, fresh)
            fresh += 1
    return orient


class _TriangulatedLink:
    """A closed oriented combinatorial 3-manifold with link components in its 1-skeleton."""

    def __init__(self, ambient: SimplicialComplex, components: Sequence[SimplicialComplex],
                 refinement: int = 0):
        if ambient.dimension != 3:
            raise ValueError(
                f"the ambient must be a 3-dimensional complex, got dimension {ambient.dimension}"
            )
        original = sorted({tuple(sorted(t)) for t in ambient.n_simplices(3)})
        ambient_edges = {e for tet in original for e in combinations(tet, 2)}
        self.cycles = [component_vertex_cycle(c) for c in components]
        for k, cycle in enumerate(self.cycles):
            for e in self._cycle_edges(cycle):
                if e not in ambient_edges:
                    raise ValueError(
                        f"component {k}: edge {e} is not an edge of a tetrahedron of the ambient"
                    )
        for (i, a), (j, b) in combinations(enumerate(self.cycles), 2):
            shared = set(a) & set(b)
            if shared:
                raise ValueError(f"components {i} and {j} share vertex {min(shared)}")

        capped = _cone_off_boundary(original)
        _check_vertex_links(capped)
        orient = _coherent_orientation(capped)
        if self._geometric_vote(ambient, original, orient) < 0:
            orient = {tet: -o for tet, o in orient.items()}
        _check_rational_homology_sphere(capped)
        # Refinement for embedded surfaces: 1 separates the components, 2 subdivides
        # barycentrically first (which also separates them). Both are PL homeomorphisms.
        if refinement >= 2:
            orient, self.cycles = _barycentric(orient, self.cycles)
        orient = _make_full(orient, self.cycles)
        if refinement >= 1:
            orient = _separate(orient, self.cycles)
        self.orient: Dict[Tet, int] = orient
        self.tets: List[Tet] = sorted(self.orient)

        self.tri_tets: Dict[Tri, List[Tet]] = defaultdict(list)
        self.vertex_tets: Dict[int, List[Tet]] = defaultdict(list)
        for tet in self.tets:
            for tri, _ in _faces(tet):
                self.tri_tets[tri].append(tet)
            for v in tet:
                self.vertex_tets[v].append(tet)
        self.neighbours: Dict[Tet, List[Tuple[Tri, Tet]]] = {tet: [] for tet in self.tets}
        for tri, (a, b) in self.tri_tets.items():
            self.neighbours[a].append((tri, b))
            self.neighbours[b].append((tri, a))
        for nbs in self.neighbours.values():
            nbs.sort()
        self.edge_tris: Dict[Edge, List[Tuple[Tri, int]]] = defaultdict(list)
        for tri in self.tri_tets:
            for e, sign in _faces(tri):
                self.edge_tris[e].append((tri, sign))
        self.vertices = sorted(self.vertex_tets)
        self.edges = sorted(self.edge_tris)

        self._check_coherent()
        self._primal_tree_edges = self._spanning_tree_edges()
        self._dual_tree_tris = self._dual_spanning_tree_triangles()
        self._tree_cotree_lu: Optional[Tuple[List[Edge], List[Tri], object]] = None

    @staticmethod
    def _cycle_edges(cycle: List[int]) -> List[Edge]:
        return [tuple(sorted((cycle[i], cycle[(i + 1) % len(cycle)]))) for i in range(len(cycle))]

    # ── orientation and homology ────────────────────────────────────────────

    @staticmethod
    def _geometric_vote(ambient: SimplicialComplex, original: List[Tet], orient: Dict[Tet, int]) -> int:
        """Sum over the input tetrahedra of ``o(tet)`` times the sign of its coordinate volume.

        The ambient is flipped when it is negative, so that with coordinates it carries
        the orientation of R^3 (flat tetrahedra vote 0), as in
        ``manifolds.simplicial_linking``; without coordinates the first input
        tetrahedron stays positive.
        """
        cloud = ambient.simplices_to_point_cloud
        if not cloud:
            return 0
        vote = 0
        for tet in original:
            try:
                P = np.array([cloud[(v,)][0] for v in tet], dtype=np.float64)
            except (KeyError, IndexError):
                continue
            if P.shape == (4, 3):
                vote += orient[tet] * int(np.sign(np.linalg.det(P[1:] - P[0])))
        return vote

    def _check_coherent(self) -> None:
        for tri, (a, b) in self.tri_tets.items():
            if self.orient[a] * _incidence(a, tri) + self.orient[b] * _incidence(b, tri):
                raise RuntimeError(f"incoherent orientation across triangle {tri}")

    def _spanning_tree_edges(self) -> Set[Edge]:
        adjacent: Dict[int, List[int]] = defaultdict(list)
        for u, v in self.edges:
            adjacent[u].append(v)
            adjacent[v].append(u)
        root = self.vertices[0]
        seen = {root}
        tree: Set[Edge] = set()
        queue = deque([root])
        while queue:
            u = queue.popleft()
            for v in sorted(adjacent[u]):
                if v not in seen:
                    seen.add(v)
                    tree.add((min(u, v), max(u, v)))
                    queue.append(v)
        return tree

    def _dual_spanning_tree_triangles(self) -> Set[Tri]:
        seen = {self.tets[0]}
        tree: Set[Tri] = set()
        queue = deque([self.tets[0]])
        while queue:
            a = queue.popleft()
            for tri, b in self.neighbours[a]:
                if b not in seen:
                    seen.add(b)
                    tree.add(tri)
                    queue.append(b)
        return tree

    # ── chains ──────────────────────────────────────────────────────────────

    def cycle_vector(self, k: int) -> Dict[Edge, int]:
        """Component k as an oriented 1-cycle: +1 on an edge ``(u, v)`` traversed u -> v."""
        cycle = self.cycles[k]
        out: Dict[Edge, int] = {}
        for i, a in enumerate(cycle):
            b = cycle[(i + 1) % len(cycle)]
            out[(min(a, b), max(a, b))] = 1 if a < b else -1
        return out

    def seifert_chain(self, boundary: Dict[Edge, int], avoid: Iterable[Tri] = ()) -> Dict[Tri, Fraction]:
        """A rational 2-chain F with ``dF = boundary`` using no triangle of ``avoid``.

        Only the equations on edges outside a spanning tree are imposed: the residual
        ``dF - boundary`` is a 1-cycle, and a 1-cycle vanishing off a tree vanishes.
        Without ``avoid`` an integral solution is tried first: restricted further to the
        triangles outside a dual spanning tree the system is square, and unimodular when
        ``H_1(M; Z) = 0``, so a floating-point LU solve rounds to it (checked exactly).
        """
        if not avoid:
            F = self._integral_seifert_chain(boundary)
            if F is not None:
                return F
        avoid = set(avoid)
        index = {tri: k for k, tri in enumerate(t for t in self.tri_tets if t not in avoid)}
        tris = list(index)
        ech = _SparseEchelon()
        for e in self.edges:
            if e in self._primal_tree_edges:
                continue
            row = {index[t]: s for t, s in self.edge_tris[e] if t in index}
            if not ech.add(row, boundary.get(e, 0)):
                raise UndefinedInvariantError(
                    "a component bounds no 2-chain in the complement of the others' push-offs"
                )
        return {tris[c]: v for c, v in ech.solution().items()}

    def _integral_seifert_chain(self, boundary: Dict[Edge, int]) -> Optional[Dict[Tri, Fraction]]:
        from scipy.sparse.linalg import splu

        if self._tree_cotree_lu is None:
            rows = [e for e in self.edges if e not in self._primal_tree_edges]
            cols = [t for t in self.tri_tets if t not in self._dual_tree_tris]
            r_of, c_of = {e: i for i, e in enumerate(rows)}, {t: j for j, t in enumerate(cols)}
            entries = [(r_of[e], c_of[t], s) for t in cols for e, s in _faces(t) if e in r_of]
            r, c, v = zip(*entries)
            M = csr_matrix((np.array(v, float), (r, c)), shape=(len(rows), len(cols)))
            try:
                lu = splu(M.tocsc()) if len(rows) == len(cols) else None
            except RuntimeError:
                lu = None
            self._tree_cotree_lu = (rows, cols, lu)
        rows, cols, lu = self._tree_cotree_lu
        if lu is None:
            return None
        x = lu.solve(np.array([boundary.get(e, 0) for e in rows], dtype=np.float64))
        xi = np.rint(x).astype(np.int64)
        if np.max(np.abs(x - xi), initial=0.0) > 1e-6:
            return None
        F = {cols[j]: int(xi[j]) for j in np.nonzero(xi)[0]}
        residual: Dict[Edge, int] = defaultdict(int)
        for t, f in F.items():
            for e, s in _faces(t):
                residual[e] += s * f
        if any(residual.get(e, 0) != boundary.get(e, 0) for e in set(residual) | set(boundary)):
            return None
        return {t: Fraction(f) for t, f in F.items()}

    def bounding_cochain(self, beta: Dict[Tri, int], vanish_on: Iterable[Edge] = ()) -> Dict[Edge, Fraction]:
        """A rational 1-cochain g with ``delta g = beta`` vanishing on ``vanish_on``.

        This is the Poincare dual of a dual 2-chain bounded by the dual cycle whose
        crossing cochain is beta, avoiding the edges in ``vanish_on``. Only the equations
        on triangles outside a dual spanning tree are imposed: the residual is a 2-cocycle,
        and a 2-cocycle vanishing off a dual tree vanishes.
        """
        vanish_on = set(vanish_on)
        index = {e: k for k, e in enumerate(e for e in self.edges if e not in vanish_on)}
        edges = list(index)
        ech = _SparseEchelon()
        for tri in self.tri_tets:
            if tri in self._dual_tree_tris:
                continue
            row = {index[e]: s for e, s in _faces(tri) if e in index}
            if not ech.add(row, beta.get(tri, 0)):
                raise UndefinedInvariantError(
                    "a push-off bounds no dual 2-chain in the complement of the other components"
                )
        return {edges[c]: v for c, v in ech.solution().items()}

    # ── dual push-offs ──────────────────────────────────────────────────────

    def _dual_path(self, start: Tet, allowed: Callable[[Tet], bool],
                   target: Callable[[Tet], bool]) -> List[Tet]:
        """Shortest path of tetrahedra from ``start`` (exempt from ``allowed``) to a target."""
        prev: Dict[Tet, Optional[Tet]] = {start: None}
        queue = deque([start])
        while queue:
            a = queue.popleft()
            for _, b in self.neighbours[a]:
                if b in prev or not allowed(b):
                    continue
                prev[b] = a
                if target(b):
                    path = [b]
                    while prev[path[-1]] is not None:
                        path.append(prev[path[-1]])
                    return path[::-1]
                queue.append(b)
        raise NotAManifoldError("no path of tetrahedra around a vertex of a component")

    def dual_pushoff(self, k: int) -> List[Step]:
        """A dual 1-cycle homotopic to component k inside its open star.

        Returns the closed path as steps ``(tet, next_tet, shared_triangle)``. Around each
        vertex x_j it walks, among the tetrahedra containing x_j but not x_{j-1}, from one
        containing ``[x_{j-1}, x_j]`` to one containing ``[x_j, x_{j+1}]``; the open star
        of x_j is contractible, so the result is homotopic to the component there. As the
        component is full, the pieces around different vertices share only their
        endpoints, so no tetrahedron repeats and the push-off is an embedded circle.
        """
        x = self.cycles[k]
        m = len(x)
        first = min(t for t in self.vertex_tets[x[0]] if x[1] in t)
        path = [first]
        for j in list(range(1, m)) + [0]:
            xj, before, after = x[j], x[j - 1], x[(j + 1) % m]

            def allowed(t, xj=xj, before=before):
                return xj in t and before not in t

            if j == 0:
                def target(t):
                    return t == first
            else:
                def target(t, after=after):
                    return after in t
            path.extend(self._dual_path(path[-1], allowed, target)[1:])
        if len(set(path[:-1])) != len(path) - 1:
            raise RuntimeError("the dual push-off of a full component revisited a tetrahedron")
        return [(a, b, tuple(sorted(set(a) & set(b)))) for a, b in zip(path, path[1:])]

    def crossing_cochain(self, steps: List[Step]) -> Dict[Tri, int]:
        """``beta(t) = I(dual cycle, t)``: signed crossings of each triangle.

        Stepping from tetrahedron a across its face t crosses t positively (tangent,
        then the orientation of t, is positive) iff ``o(a)[a:t] = +1``.
        """
        beta: Dict[Tri, int] = defaultdict(int)
        for a, _, tri in steps:
            beta[tri] += self.orient[a] * _incidence(a, tri)
        return {t: v for t, v in beta.items() if v}

    # ── embedded surfaces ───────────────────────────────────────────────────

    @staticmethod
    def _min_area(A: csr_matrix, b: np.ndarray, groups: Sequence[Tuple[List[int], int, int]],
                  singular: Callable[[np.ndarray], Set[int]], what: str,
                  rounds: int = _REWEIGHT_ROUNDS) -> np.ndarray:
        """Integral x with ``A x = b``, ``|x| <= 1`` and local embedding, of minimal area.

        An integer program (HiGHS; its LP relaxation is tried first and is usually
        integral already): x = p - n with p, n binary and ``p + n <= 1``,
        ``lo <= sum_{j in cols} |x_j| <= hi`` for each ``(cols, lo, hi)`` in ``groups``,
        minimising ``sum w |x|`` for weights w near 1 (tie-breaking perturbation). The
        solution is checked exactly. ``singular(x)`` names the columns where the support
        still fails to be embedded (conditions that are not linear); their weights are
        doubled and the program solved again.
        """
        n = A.shape[1]
        I = identity(n, format="csr", dtype=np.int64)
        constraints = [LinearConstraint(hstack([A, -A]).tocsr(), b, b),
                       LinearConstraint(hstack([I, I]).tocsr(), 0, 1)]
        if groups:
            rows, cols = [], []
            for i, (js, _, _) in enumerate(groups):
                rows.extend([i] * len(js))
                cols.extend(js)
            G = csr_matrix((np.ones(len(cols)), (rows, cols)), shape=(len(groups), n))
            constraints.append(LinearConstraint(hstack([G, G]).tocsr(),
                                                [lo for _, lo, _ in groups], [hi for _, _, hi in groups]))
        w = 1.0 + 1e-3 * np.random.default_rng(0).random(n)

        def solve(integral: bool) -> Optional[np.ndarray]:
            res = milp(np.concatenate([w, w]), constraints=constraints,
                       integrality=np.full(2 * n, 1 if integral else 0), bounds=Bounds(0, 1))
            if res.status == 2:
                raise _NoRoom(f"no {what} exists in this triangulation; a finer triangulation may help")
            if res.x is None or np.max(np.abs(res.x - np.rint(res.x)), initial=0.0) > 1e-6:
                return None
            pn = np.rint(res.x).astype(np.int64)
            x = pn[:n] - pn[n:]
            if not np.array_equal(A @ x, b.astype(np.int64)) or any(
                not lo <= int(np.count_nonzero(x[js])) <= hi for js, lo, hi in groups
            ):
                return None
            return x

        for _ in range(rounds):
            # The LP relaxation first: usually integral already, and infeasible exactly
            # when the integer program is.
            x = solve(integral=False)
            if x is None:
                x = solve(integral=True)
            if x is None:
                raise _NotFound(f"the integer program gave no verified {what}")
            bad = singular(x)
            if not bad:
                return x
            w[list(bad)] *= 2.0
        raise _NotFound(
            f"no embedded {what} found: the minimal-area solutions keep touching themselves "
            "at a vertex; a finer triangulation may help"
        )

    def embedded_seifert_surface(self, boundary: Dict[Edge, int], avoid: Iterable[Tri] = (),
                                 rounds: int = _REWEIGHT_ROUNDS) -> Dict[Tri, int]:
        """An embedded Seifert surface of a component, as a 2-chain with coefficients +-1.

        Minimal-area integral F with ``dF = boundary`` using no triangle of ``avoid``,
        whose support is an embedded surface with boundary the component: every edge in
        none or two of its triangles (exactly one for boundary edges, constraints of the
        integer program), every vertex link connected (checked, then penalised).
        """
        avoid = set(avoid)
        tris = [t for t in self.tri_tets if t not in avoid]
        row = {e: i for i, e in enumerate(self.edges)}
        entries = [(row[e], j, sign) for j, t in enumerate(tris) for e, sign in _faces(t)]
        rows, cols, vals = zip(*entries)
        A = csr_matrix((vals, (rows, cols)), shape=(len(self.edges), len(tris)), dtype=np.int64)
        b = np.array([boundary.get(e, 0) for e in self.edges], dtype=np.float64)

        at_edge: Dict[Edge, List[int]] = defaultdict(list)
        for j, t in enumerate(tris):
            for e, _ in _faces(t):
                at_edge[e].append(j)
        # Along the boundary one sheet, elsewhere none or two.
        groups = [(js, 1, 1) if boundary.get(e) else (js, 0, 2) for e, js in at_edge.items()]

        def singular(F: np.ndarray) -> Set[int]:
            support = [j for j in np.nonzero(F)[0]]
            bad: Set[int] = set()
            link: Dict[int, List[Tuple[Edge, int]]] = defaultdict(list)
            for j in support:
                t = tris[j]
                for i, v in enumerate(t):
                    link[v].append((t[:i] + t[i + 1:], j))
            for pieces in link.values():
                seen = {pieces[0][0][0]}
                grew = True
                while grew:
                    grew = False
                    for (a, c), _ in pieces:
                        if (a in seen) != (c in seen):
                            seen.update((a, c))
                            grew = True
                if any(a not in seen for (a, _), _ in pieces):
                    bad.update(j for _, j in pieces)
            return bad

        F = self._min_area(A, b, groups, singular, "Seifert surface avoiding the other components", rounds)
        return {tris[j]: int(F[j]) for j in np.nonzero(F)[0]}

    def embedded_dual_surface(self, beta: Dict[Tri, int], vanish_on: Iterable[Edge] = ()) -> Dict[Edge, int]:
        """An embedded dual Seifert surface of a dual push-off, as a 1-cochain with values +-1.

        Minimal-area integral g with ``delta g = beta`` vanishing on ``vanish_on``. With
        ``|g| <= 1`` the dual 2-cells of its support form an embedded surface away from the
        push-off (around a triangle with ``delta g = 0`` an even number, 0 or 2, of its
        edges carry g; around a tetrahedron the pieces are triangles or quadrilaterals),
        and along the push-off it is embedded exactly when every crossed triangle has
        one edge in the support (a constraint of the integer program).
        """
        vanish_on = set(vanish_on)
        edges = [e for e in self.edges if e not in vanish_on]
        col = {e: j for j, e in enumerate(edges)}
        tris = list(self.tri_tets)
        entries = [(i, col[e], sign) for i, t in enumerate(tris) for e, sign in _faces(t) if e in col]
        rows, cols, vals = zip(*entries)
        A = csr_matrix((vals, (rows, cols)), shape=(len(tris), len(edges)), dtype=np.int64)
        b = np.array([beta.get(t, 0) for t in tris], dtype=np.float64)
        # Along the push-off one sheet: exactly one edge of each crossed triangle.
        groups = [([col[e] for e, _ in _faces(t) if e in col], 1, 1) for t in tris if beta.get(t)]
        g = self._min_area(A, b, groups, lambda g: set(), "dual Seifert surface avoiding the other components")
        return {edges[j]: int(g[j]) for j in np.nonzero(g)[0]}

    # ── invariants ──────────────────────────────────────────────────────────

    def _potential_around(self, e: Edge, F: Dict[Tri, Fraction]) -> Dict[Tet, Fraction]:
        """``phi(tet)`` = intersection with F of a path around e from a reference tetrahedron.

        Around an edge e with ``(dF)[e] = 0`` the full loop meets F zero times, so phi is
        well defined on the tetrahedra containing e.
        """
        start = min(t for t in self.vertex_tets[e[0]] if e[1] in t)
        phi = {start: Fraction(0)}
        a, came_by = start, None
        while True:
            tri = next(t for t, _ in _faces(a) if e[0] in t and e[1] in t and t != came_by)
            b = next(t for t in self.tri_tets[tri] if t != a)
            value = phi[a] + self.orient[a] * _incidence(a, tri) * F.get(tri, 0)
            if b == start:
                if value:
                    raise RuntimeError(f"the Seifert chain has boundary on edge {e}")
                return phi
            phi[b] = value
            a, came_by = b, tri

    def triple(self, F1: Dict[Tri, Fraction], g2: Dict[Edge, Fraction],
               F3: Dict[Tri, Fraction]) -> Fraction:
        """``lk(F_1 cap G_2, K_3)`` with G_2 the dual 2-chain of the cochain g2.

        ``F_1 cap G_2 = sum_{e < t} F_1[t] g2(e) [t:e] (midpoint(e) -> barycentre(t))``,
        oriented so that (normal of F_1, normal of G_2, tangent) is positive. Its segment
        at (e, t), pushed into the ring of tetrahedra around e, runs from a reference
        tetrahedron of e to a chosen tetrahedron of t; the pieces close up because the
        curve does, and the linking number with ``K_3 = dF_3`` is the intersection number
        of this dual cycle with F_3.
        """
        total = Fraction(0)
        for e, ge in g2.items():
            terms = [(tri, s * F1[tri]) for tri, s in self.edge_tris[e] if F1.get(tri)]
            if not terms:
                continue
            phi = self._potential_around(e, F3)
            total += ge * sum(c * phi[min(self.tri_tets[tri])] for tri, c in terms)
        return total


def _as_integer(value: Fraction, what: str) -> int:
    if value.denominator != 1:
        raise UndefinedInvariantError(
            f"{what} = {value} is not an integer: the ambient is a rational but not an "
            "integral homology sphere"
        )
    return int(value)


def triangulated_linking_number(
    ambient_complex: SimplicialComplex,
    K_a: SimplicialComplex,
    K_b: SimplicialComplex,
) -> int:
    """The linking number of two disjoint knots in a triangulated 3-manifold, exactly.

    What is Being Computed?:
        ``lk(K_a, K_b) = I(K_a*, F)``: the signed number of crossings of a dual push-off
        K_a* (a closed path of tetrahedra homotopic to K_a in its open star) through a
        rational 2-chain F with ``dF = K_b``. See the module docstring for the
        conventions and the hypotheses on the ambient.

    Args:
        ambient_complex: A triangulated rational homology 3-sphere or 3-ball.
        K_a: First component (a simple closed curve in the 1-skeleton).
        K_b: Second component, sharing no vertex with K_a.

    Returns:
        lk(K_a, K_b).

    Raises:
        ValueError: If the dimensions or components are invalid.
        NotAManifoldError: If the ambient is not an orientable combinatorial 3-manifold
            whose boundary components are 2-spheres.
        UndefinedInvariantError: If ``H_1(ambient; Q) != 0``, or lk is not an integer.
    """
    L = _TriangulatedLink(ambient_complex, [K_a, K_b])
    beta = L.crossing_cochain(L.dual_pushoff(0))
    F = L.seifert_chain(L.cycle_vector(1))
    return _as_integer(sum(v * F.get(t, 0) for t, v in beta.items()), "lk")


def triangulated_milnor_mu123(
    ambient_complex: SimplicialComplex,
    K_1: SimplicialComplex,
    K_2: SimplicialComplex,
    K_3: SimplicialComplex,
) -> int:
    r"""Milnor's triple linking number mu-bar(123) of a link in a triangulated 3-manifold.

    What is Being Computed?:
        For three disjoint knots with vanishing pairwise linking numbers,
        ``mu-bar(123) = -lk(F_1 cap G_2, K_3)`` with F_1 a primal Seifert chain of K_1
        avoiding a dual push-off K_2* of K_2 and G_2 a dual Seifert chain of K_2*
        avoiding K_1 and K_3 (module docstring). It is +-1 on the Borromean rings, 0 on
        the unlink, invariant under cyclic permutation of the components and negated by
        a transposition or by reversing one component.

    Algorithm:
        1. Cap boundary spheres with cones; check the result is a closed, connected,
           orientable combinatorial 3-manifold with ``H_1(M; Q) = 0``; stellar-subdivide
           the chords of each component so the components are full subcomplexes.
        2. Build dual push-offs K_i* and rational Seifert chains F_i (``dF_i = K_i``);
           require ``lk(K_i, K_j) = I(K_i*, F_j) = 0``.
        3. Solve ``dF_1 = K_1`` using no triangle crossed by K_2*, and
           ``delta g = beta(K_2*)`` with g = 0 on the edges of K_1 and K_3.
        4. Sum ``-g(e) [t:e] F_1[t] phi_e(t)`` over edges e of triangles t, where
           ``phi_e`` counts crossings of F_3 on the way around e.

    Args:
        ambient_complex: A triangulated rational homology 3-sphere or 3-ball.
        K_1: First component (a simple closed curve in the 1-skeleton).
        K_2: Second component.
        K_3: Third component. The components are pairwise vertex-disjoint.

    Returns:
        mu-bar(123), an integer.

    Raises:
        ValueError: If the dimensions or components are invalid.
        NotAManifoldError: If the ambient is not an orientable combinatorial 3-manifold
            whose boundary components are 2-spheres.
        UndefinedInvariantError: If a pairwise linking number is nonzero (mu-bar(123) is
            then defined only modulo their gcd), or ``H_1(ambient; Q) != 0``.
    """
    L = _TriangulatedLink(ambient_complex, [K_1, K_2, K_3])
    pushoffs = [L.dual_pushoff(k) for k in (0, 1)]
    betas = [L.crossing_cochain(p) for p in pushoffs]
    F2, F3 = (L.seifert_chain(L.cycle_vector(k)) for k in (1, 2))
    lk12, lk13, lk23 = (
        _as_integer(sum(v * F.get(t, 0) for t, v in beta.items()), "lk")
        for beta, F in ((betas[0], F2), (betas[0], F3), (betas[1], F3))
    )
    if lk12 or lk13 or lk23:
        raise UndefinedInvariantError(
            "mu-bar(123) is an integer invariant only when every pairwise linking number "
            f"vanishes; lk12 = {lk12}, lk13 = {lk13}, lk23 = {lk23}"
        )
    F1 = L.seifert_chain(L.cycle_vector(0), avoid={tri for _, _, tri in pushoffs[1]})
    g2 = L.bounding_cochain(
        betas[1], vanish_on=L._cycle_edges(L.cycles[0]) + L._cycle_edges(L.cycles[2])
    )
    return _MU123_SIGN * _as_integer(L.triple(F1, g2, F3), "mu-bar(123)")


def triangulated_sato_levine(
    ambient_complex: SimplicialComplex,
    K_1: SimplicialComplex,
    K_2: SimplicialComplex,
) -> int:
    r"""The Sato-Levine invariant beta of a two-component link in a triangulated 3-manifold.

    What is Being Computed?:
        For a link with ``lk(K_1, K_2) = 0`` take embedded Seifert surfaces F_1, F_2 with
        ``F_1 cap K_2 = F_2 cap K_1 = empty``. Their intersection C is a closed curve,
        framed by either surface, and ``beta = lk(C, C+)`` for the framed push-off C+
        (Sato, 1984). It does not depend on the surfaces, is an isotopy invariant, equals
        ``-mu-bar(1122)`` (Cochran, *Derivatives of links*, Mem. AMS 427, 1990), is +-1 on
        the Whitehead link and 0 on split links and boundary links, changes sign under
        mirror image and not under reversing a component.

    Algorithm:
        1. As for ``triangulated_milnor_mu123``: cap boundary spheres, check the
           ambient, make the components full, push K_1 and K_2 off to dual cycles, and
           require ``lk(K_1, K_2) = 0``.
        2. F_1: a minimal-area PRIMAL surface with ``dF_1 = K_1`` using no triangle
           crossed by K_2*. G_2: a minimal-area DUAL surface (a 1-cochain g with
           ``delta g = beta(K_2*)``, ``|g| <= 1``) vanishing on K_1. Both are integer
           programs whose constraints make the surfaces embedded along every edge;
           vertices where F_1 touches itself are penalised and the program re-solved.
        3. ``C = F_1 cap G_2 = sum F_1[t] g(e) [t:e] (midpoint(e) -> barycentre(t))``.
           At the midpoint of e the normal of G_2 is ``+-e``, so C+ (pushed along it,
           inside F_1) runs parallel to C on the side of the vertex ``e[1]`` or ``e[0]``
           (as g(e) is +1 or -1), and slides within F_1, off C, onto the primal path P
           through those vertices: in each triangle of F_1 met by C, from the vertex of
           the edge C enters by to the vertex of the edge it leaves by.
        4. ``beta = lk(C, P)``, as in the triple linking number: C pushed into the rings
           of tetrahedra around its edges, against a 2-chain bounded by P.
        If the triangulation has no room for the surfaces with K_1 carrying the primal
        one (e.g. every tetrahedron near K_1 reaches K_2), the roles are swapped; failing
        that, both are tried again after stellar-subdividing every edge that joins the
        components, and then after a barycentric subdivision (slow). beta is a
        topological invariant, so none of this changes it.

    Args:
        ambient_complex: A triangulated rational homology 3-sphere or 3-ball.
        K_1: First component (a simple closed curve in the 1-skeleton).
        K_2: Second component, sharing no vertex with K_1.

    Returns:
        beta(K_1 u K_2), an integer; symmetric in the two components.

    Raises:
        ValueError: If the dimensions or components are invalid.
        NotAManifoldError: If the ambient is not an orientable combinatorial 3-manifold
            whose boundary components are 2-spheres.
        UndefinedInvariantError: If ``lk(K_1, K_2) != 0``, or ``H_1(ambient; Q) != 0``.
        SeifertSurfaceError: If even the refined triangulations have no room for the
            embedded surfaces (the integer programs are infeasible or keep touching
            themselves at a vertex); a finer triangulation may help.
    """
    first_error: Optional[SeifertSurfaceError] = None
    no_room: Set[Tuple[int, int]] = set()
    for refinement in (0, 1, 2):
        links: Dict[int, _TriangulatedLink] = {}
        for rounds in (1, _REWEIGHT_ROUNDS):
            for order, pair in enumerate(((K_1, K_2), (K_2, K_1))):  # beta is symmetric
                if (refinement, order) in no_room:
                    continue
                if order not in links:
                    links[order] = _TriangulatedLink(ambient_complex, list(pair), refinement)
                try:
                    return _sato_levine(links[order], rounds)
                except _NoRoom as err:
                    no_room.add((refinement, order))
                    first_error = first_error or err
                except _NotFound as err:
                    first_error = first_error or err
    raise first_error


def _sato_levine(L: _TriangulatedLink, rounds: int = _REWEIGHT_ROUNDS) -> int:
    """The Sato-Levine invariant lk(C, C+), C = F_1 cap G_2, with the components of L in order."""
    push1, push2 = L.dual_pushoff(0), L.dual_pushoff(1)
    beta1, beta2 = L.crossing_cochain(push1), L.crossing_cochain(push2)
    F2 = L.seifert_chain(L.cycle_vector(1))
    lk = _as_integer(sum(v * F2.get(t, 0) for t, v in beta1.items()), "lk")
    if lk:
        raise UndefinedInvariantError(
            f"the Sato-Levine invariant needs lk(K_1, K_2) = 0, got lk = {lk}"
        )
    on_K1 = set(L._cycle_edges(L.cycles[0]))
    F1 = L.embedded_seifert_surface(L.cycle_vector(0), avoid={tri for _, _, tri in push2}, rounds=rounds)
    g2 = L.embedded_dual_surface(beta2, vanish_on=on_K1)

    P: Dict[Edge, int] = defaultdict(int)
    for t, f in F1.items():
        through = sorted(((f * g2[e] * s, e) for e, s in _faces(t) if g2.get(e)), reverse=True)
        if not through:
            continue
        if [c for c, _ in through] != [1, -1]:
            raise RuntimeError(f"F_1 cap G_2 is not a closed 1-manifold at triangle {t}")
        (_, e_in), (_, e_out) = through
        v_in = e_in[1] if g2[e_in] > 0 else e_in[0]
        v_out = e_out[1] if g2[e_out] > 0 else e_out[0]
        corner = set(e_in) & set(e_out)
        if (v_in in corner) != (v_out in corner):
            raise RuntimeError(f"the dual surface is not coherently oriented at triangle {t}")
        if v_in != v_out:
            P[(min(v_in, v_out), max(v_in, v_out))] += 1 if v_in < v_out else -1
    P = {e: c for e, c in P.items() if c}
    F_P = L.seifert_chain(P) if P else {}
    return _as_integer(L.triple(F1, g2, F_P), "the Sato-Levine invariant")
