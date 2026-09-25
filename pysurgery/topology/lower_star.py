r"""Lower-star discrete Morse theory and lower-star persistence of a vertex function.

Overview:
    A scalar field g on the VERTICES of K extends to every simplex by
    ``f(sigma) = max_{v in sigma} g(v)`` -- the lower-star filtration. This module
    computes, exactly:

      * ``lower_star_gradient``   the Robins-Wood-Sheppard (RWS 2011) discrete gradient
                                  (acyclic matching) whose critical cells are in
                                  bijection with the changes of topology of the
                                  filtration; pairings never leave a single lower star
      * ``GradientField``         verification that the matching is acyclic, the Morse
                                  complex (chain complex on the critical cells, boundary
                                  counted with signs along gradient paths) as a pySurgery
                                  ``ChainComplex``, gradient-path counts, cancellation
                                  corridors
      * ``classify_critical_pairs``  which pairs of critical cells cancel, and how
      * ``lower_star_persistence``   persistence pairs over Z/2, with the simplices that
                                  create and destroy every class

    It complements ``SimplicialComplex.discrete_morse_gradient`` (a coreduction-style
    matching with no function) with a gradient driven by, and faithful to, a given
    function -- the setting of Morse theory proper.

Key Concepts:
    - **Morse complex.** The Morse complex has the homology of K exactly -- over Z,
      torsion included -- with far fewer cells (Forman). Its boundary is
      ``d^M(sigma) = sum over gradient paths sigma -> tau of the path weights``: from
      ``[beta : alpha] alpha + sum [beta : alpha'] alpha' = 0`` in the reduced complex,
      a step alpha -> beta -> alpha' has weight ``-[beta : alpha'] [beta : alpha]``.
    - **Cancellation classes** of a pair ``(sigma^p, tau^(p-1))`` of critical cells, by
      the Morse incidence c and the number of gradient paths:

          exactly one gradient path  ->  CANCELLABLE: Forman's cancellation theorem
                                         applies to the gradient field itself
          |c| = 1, several paths     ->  CHAIN_CANCELLABLE: the pair cancels in the Morse
                                         complex (algebraic Morse theory), not in the
                                         gradient field
          c = 0, or |c| > 1          ->  NOT_CANCELLABLE: the two cells cannot cancel
                                         EACH OTHER (c = 0: not algebraically incident;
                                         |c| > 1: cancelling would change torsion). This
                                         says nothing about homology on its own -- either
                                         cell may still cancel against a third.
    - **Weak Morse inequalities and Euler relation.** ``m_p >= beta_p(K; F)`` for every
      field F, and ``sum (-1)^p m_p = chi(K)``.

Common Workflows:
    1. **Morse homology** -> ``lower_star_gradient(K, g).morse_homology()``.
    2. **Barcode with its critical simplices** -> ``lower_star_persistence(K, g)``.
    3. **Which critical pairs cancel** -> ``classify_critical_pairs(V, p)``.

Coefficient Ring:
    The gradient and the Morse complex are over Z (exact signs); persistence is over Z/2.
"""

from __future__ import annotations

import heapq
import warnings
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
from pydantic import BaseModel

from ..bridge.julia_bridge import julia_engine

if TYPE_CHECKING:  # pragma: no cover
    from .complexes import ChainComplex, SimplicialComplex

Simplex = Tuple[int, ...]
VertexFunction = Union[Mapping[int, float], Sequence[float], np.ndarray]

__all__ = [
    "GradientField",
    "lower_star_gradient",
    "PairClass",
    "classify_critical_pairs",
    "PersistencePair",
    "lower_star_filtration",
    "lower_star_persistence",
    "critical_pair_persistence",
]


def _incidence(high: Simplex, low: Simplex) -> int:
    """``[high : low]`` for sorted simplices: ``(-1)^j`` if low omits high[j], else 0."""
    if len(high) != len(low) + 1:
        return 0
    for j in range(len(high)):
        if high[:j] + high[j + 1:] == low:
            return -1 if j % 2 else 1
    return 0


def _all_simplices(K: "SimplicialComplex") -> List[Simplex]:
    """Every simplex of K, sorted by (dimension, vertex tuple).

    This canonical order is shared by both backends.
    """
    return sorted(
        (tuple(int(v) for v in s) for d in K.dimensions for s in K.n_simplices(d)),
        key=lambda s: (len(s), s),
    )


def _vertex_values(K: "SimplicialComplex", g: VertexFunction) -> Dict[int, float]:
    """Normalise g to ``{vertex: value}``; a sequence is indexed by vertex label."""
    verts = [int(s[0]) for s in K.n_simplices(0)]
    if isinstance(g, Mapping):
        missing = [v for v in verts if v not in g]
        if missing:
            raise ValueError(f"the vertex function has no value at vertices {missing[:10]}")
        vals = {int(v): float(g[v]) for v in verts}
    else:
        arr = np.asarray(g, dtype=np.float64).ravel()
        if verts and max(verts) >= len(arr):
            raise ValueError(
                f"the vertex function has {len(arr)} values but the complex has vertex "
                f"label {max(verts)}; pass a dict {{vertex: value}} for sparse labels"
            )
        vals = {v: float(arr[v]) for v in verts}
    if not all(np.isfinite(list(vals.values()))):
        raise ValueError("the vertex function must be finite")
    return vals


class GradientField:
    """A discrete gradient vector field (an acyclic matching) on a simplicial complex.

    Overview:
        ``up[a] = b`` pairs a p-cell a with a (p+1)-cell b (the arrow a -> b); ``down`` is
        the inverse map; every other cell is critical. Built by
        ``lower_star_gradient``, checked by ``verify``.

    Attributes:
        K (SimplicialComplex): The complex.
        values (dict[int, float]): The vertex function.
        up (dict): tail -> head of every arrow.
        down (dict): head -> tail of every arrow.
        critical (set): The critical cells.
    """

    def __init__(self, K: "SimplicialComplex", values: Dict[int, float]):
        self.K = K
        self.values = dict(values)
        self.up: Dict[Simplex, Simplex] = {}
        self.down: Dict[Simplex, Simplex] = {}
        self.critical: Set[Simplex] = set()
        self._cells = _all_simplices(K)
        self._cellset = set(self._cells)
        self._facets: Optional[Dict[Simplex, List[Simplex]]] = None

    # ------------------------------------------------------------------ basics
    def pair(self, low: Simplex, high: Simplex) -> None:
        """Add the arrow ``low -> high``.

        Args:
            low: A p-cell.
            high: A (p+1)-cell having ``low`` as a facet.

        Raises:
            ValueError: If the dimensions do not differ by one.
        """
        if len(high) != len(low) + 1:
            raise ValueError("a matched pair must differ by one dimension")
        self.up[low] = high
        self.down[high] = low

    def facets_of(self, s: Simplex) -> List[Simplex]:
        """The codimension-1 faces of s (all present, K being closed)."""
        return [s[:i] + s[i + 1:] for i in range(len(s))] if len(s) > 1 else []

    def is_critical(self, s: Simplex) -> bool:
        """Whether s is critical."""
        return tuple(s) in self.critical

    def value(self, s: Simplex) -> float:
        """``f(s) = max_v g(v)``: the lower-star filtration value."""
        return float(max(self.values[v] for v in s))

    def critical_cells(self, dim: Optional[int] = None) -> List[Simplex]:
        """Critical cells, sorted by (dimension, vertices), optionally of one dimension.

        Args:
            dim: Restrict to this dimension.

        Returns:
            The critical cells.
        """
        cells = sorted(self.critical, key=lambda s: (len(s), s))
        return cells if dim is None else [c for c in cells if len(c) - 1 == dim]

    def morse_vector(self) -> List[int]:
        """``m_p``, the number of critical p-cells, for p = 0..dim K."""
        out = [0] * (self.K.dimension + 1)
        for c in self.critical:
            out[len(c) - 1] += 1
        return out

    # ------------------------------------------------------------ verification
    def verify(self) -> None:
        """Check this is an acyclic matching on K; raise on failure.

        Every cell must play exactly one role (tail, head or critical), every arrow must
        go from a facet to a cofacet, and no V-path may cycle (checked by an iterative
        three-colour depth-first search, dimension by dimension).

        Raises:
            ValueError: With the offending cell, if the field is not an acyclic matching.
        """
        assigned: Dict[Simplex, int] = {}
        for low, high in self.up.items():
            for s in (low, high):
                assigned[s] = assigned.get(s, 0) + 1
            if self.down.get(high) != low:
                raise ValueError(f"asymmetric pairing {low} <-> {high}")
            if _incidence(high, low) == 0:
                raise ValueError(f"{low} is not a facet of {high}")
        for c in self.critical:
            assigned[c] = assigned.get(c, 0) + 1
        for s in self._cells:
            if assigned.get(s, 0) != 1:
                raise ValueError(f"cell {s} is in {assigned.get(s, 0)} roles, expected exactly 1")
        if set(assigned) != self._cellset:
            raise ValueError("matching refers to cells outside the complex")
        for p in range(self.K.dimension + 1):
            cells_p = [s for s in self._cells if len(s) == p + 1]
            colour: Dict[Simplex, int] = {}
            for start in cells_p:
                if colour.get(start, 0):
                    continue
                stack = [(start, iter(self._vpath_successors(start)))]
                colour[start] = 1
                while stack:
                    node, it = stack[-1]
                    advanced = False
                    for nxt in it:
                        c = colour.get(nxt, 0)
                        if c == 1:
                            raise ValueError(f"gradient field has a V-path cycle at {nxt}")
                        if c == 0:
                            colour[nxt] = 1
                            stack.append((nxt, iter(self._vpath_successors(nxt))))
                            advanced = True
                            break
                    if not advanced:
                        colour[node] = 2
                        stack.pop()

    def _vpath_successors(self, alpha: Simplex) -> List[Simplex]:
        beta = self.up.get(alpha)
        if beta is None:
            return []
        return [t for t in self.facets_of(beta) if t != alpha]

    # ------------------------------------------------------------- Morse complex
    def _flow(self, p: int, signed: bool) -> Dict[Simplex, Dict[Simplex, int]]:
        """For every p-cell, the (signed) tally of gradient paths to critical p-cells.

        Evaluated in reverse topological order of the V-path digraph (no recursion).
        """
        cells = [s for s in self._cells if len(s) == p + 1]
        order: List[Simplex] = []
        colour: Dict[Simplex, int] = {}
        for start in cells:
            if colour.get(start, 0):
                continue
            stack = [(start, iter(self._vpath_successors(start)))]
            colour[start] = 1
            while stack:
                node, it = stack[-1]
                advanced = False
                for nxt in it:
                    if colour.get(nxt, 0) == 0:
                        colour[nxt] = 1
                        stack.append((nxt, iter(self._vpath_successors(nxt))))
                        advanced = True
                        break
                if not advanced:
                    colour[node] = 2
                    order.append(node)
                    stack.pop()
        flow: Dict[Simplex, Dict[Simplex, int]] = {}
        for alpha in order:  # sinks first
            if alpha in self.critical:
                flow[alpha] = {alpha: 1}
                continue
            beta = self.up.get(alpha)
            if beta is None:  # a head: the path stops without reaching a critical cell
                flow[alpha] = {}
                continue
            acc: Dict[Simplex, int] = {}
            s_ab = _incidence(beta, alpha)
            for ap in self.facets_of(beta):
                if ap == alpha:
                    continue
                coef = -_incidence(beta, ap) * s_ab if signed else 1
                for t, c in flow.get(ap, {}).items():
                    acc[t] = acc.get(t, 0) + coef * c
            flow[alpha] = {t: c for t, c in acc.items() if c != 0}
        return flow

    def morse_boundary(self, p: int) -> np.ndarray:
        """Boundary of the Morse complex ``C^M_p -> C^M_{p-1}`` on critical cells.

        Args:
            p: The degree.

        Returns:
            An integer matrix with rows ``critical_cells(p-1)`` and columns
            ``critical_cells(p)``.
        """
        rows = self.critical_cells(p - 1) if p >= 1 else []
        cols = self.critical_cells(p)
        M = np.zeros((len(rows), len(cols)), dtype=np.int64)
        if p < 1 or p > self.K.dimension or not cols or not rows:
            return M
        row_index = {s: i for i, s in enumerate(rows)}
        flow = self._flow(p - 1, signed=True)
        for j, sigma in enumerate(cols):
            for alpha in self.facets_of(sigma):
                s = _incidence(sigma, alpha)
                for tau, c in flow.get(alpha, {}).items():
                    i = row_index.get(tau)
                    if i is not None:
                        M[i, j] += s * c
        return M

    def morse_chain_complex(self, coefficient_ring: str = "Z") -> "ChainComplex":
        """The Morse complex as a pySurgery ``ChainComplex`` (cells = critical cells).

        Args:
            coefficient_ring: Coefficient ring label of the result.

        Returns:
            A ``ChainComplex`` whose homology is that of K.
        """
        from scipy.sparse import csr_matrix

        from .complexes import ChainComplex

        top = self.K.dimension
        boundaries = {p: csr_matrix(self.morse_boundary(p)) for p in range(1, top + 1)}
        cells = {p: len(self.critical_cells(p)) for p in range(top + 1)}
        return ChainComplex(
            dimensions=list(range(top + 1)), boundaries=boundaries, cells=cells,
            coefficient_ring=coefficient_ring,
        )

    def morse_homology(self, backend: str = "auto") -> Dict[int, Tuple[int, List[int]]]:
        """Integer homology from the Morse complex -- the homology of K, exactly.

        Args:
            backend: 'auto', 'julia' or 'python' (for the SNF).

        Returns:
            ``{degree: (rank, torsion)}``.
        """
        cc = self.morse_chain_complex()
        return {p: cc.homology(p, backend=backend) for p in range(self.K.dimension + 1)}

    def gradient_path_counts(self, p: int) -> Dict[Tuple[Simplex, Simplex], int]:
        """Unsigned number of gradient paths from ``d(sigma^p)`` to each critical tau^(p-1).

        This is what Forman's cancellation theorem is stated in terms of: its hypothesis
        is that this count equals 1.

        Args:
            p: The dimension of sigma.

        Returns:
            ``{(sigma, tau): number_of_paths}``.
        """
        out: Dict[Tuple[Simplex, Simplex], int] = {}
        if p < 1 or p > self.K.dimension:
            return out
        flow = self._flow(p - 1, signed=False)
        for sigma in self.critical_cells(p):
            for alpha in self.facets_of(sigma):
                for tau, c in flow.get(alpha, {}).items():
                    out[(sigma, tau)] = out.get((sigma, tau), 0) + c
        return out

    def cancellation_region(self, sigma: Simplex, tau: Simplex) -> Set[Simplex]:
        """Cells on gradient paths from ``d(sigma)`` to tau, plus sigma and tau.

        When the pair is CANCELLABLE (a unique path) this corridor is exactly the one
        Forman's theorem reverses.

        Args:
            sigma: A critical (p+1)-cell.
            tau: A critical p-cell.

        Returns:
            The corridor of cells.

        Raises:
            ValueError: If ``dim(sigma) != dim(tau) + 1``.
        """
        sigma, tau = tuple(sigma), tuple(tau)
        p = len(tau) - 1
        if len(sigma) - 1 != p + 1:
            raise ValueError("expected dim(sigma) = dim(tau) + 1")
        flow = self._flow(p, signed=False)
        reachable: Set[Simplex] = set()
        stack = list(self.facets_of(sigma))
        while stack:
            a = stack.pop()
            if a in reachable:
                continue
            reachable.add(a)
            stack.extend(self._vpath_successors(a))
        region: Set[Simplex] = {sigma, tau}
        for a in reachable:
            if a == tau or tau in flow.get(a, {}):
                region.add(a)
                partner = self.up.get(a)
                if partner is not None:
                    region.add(partner)
        return region

    def __repr__(self) -> str:
        return (
            f"GradientField(cells={len(self._cells)}, pairs={len(self.up)}, "
            f"critical={len(self.critical)}, m={self.morse_vector()})"
        )


# --------------------------------------------------------------------------- #
# Robins - Wood - Sheppard
# --------------------------------------------------------------------------- #


def _rws_python(cells: List[Simplex], values: Dict[int, float]):
    """RWS ProcessLowerStars on every vertex. Returns (pairs, critical) as cell lists."""
    key = {v: (val, v) for v, val in values.items()}  # injective: ties broken by label
    lower_star: Dict[int, List[Simplex]] = {v: [] for v in values}
    for s in cells:
        lower_star[max(s, key=lambda u: key[u])].append(s)
    pairs: List[Tuple[Simplex, Simplex]] = []
    critical: List[Simplex] = []
    for v in sorted(values):
        L = lower_star[v]  # already in (dimension, vertices) order
        if len(L) == 1:
            critical.append((v,))
            continue
        L_set = set(L)
        assigned: Set[Simplex] = set()

        def sort_key(s: Simplex) -> tuple:
            return tuple(sorted((key[u] for u in s if u != v), reverse=True))

        def unpaired_facets(s: Simplex) -> List[Simplex]:
            return [
                t for t in (s[:i] + s[i + 1:] for i in range(len(s)))
                if t in L_set and t not in assigned
            ]

        cofacets: Dict[Simplex, List[Simplex]] = {s: [] for s in L}
        for s in L:
            for i in range(len(s)):
                t = s[:i] + s[i + 1:]
                if t in L_set:
                    cofacets[t].append(s)

        edges = [s for s in L if len(s) == 2]
        delta = min(edges, key=sort_key)
        pairs.append(((v,), delta))
        assigned.update(((v,), delta))
        pq_zero: List[tuple] = []
        pq_one: List[tuple] = []
        for s in edges:
            if s != delta:
                heapq.heappush(pq_zero, (sort_key(s), s))
        for s in cofacets[delta]:
            if len(unpaired_facets(s)) == 1:
                heapq.heappush(pq_one, (sort_key(s), s))

        def push_cofacets(s: Simplex) -> None:
            for beta in cofacets[s]:
                if beta not in assigned and len(unpaired_facets(beta)) == 1:
                    heapq.heappush(pq_one, (sort_key(beta), beta))

        def pop(pq) -> Optional[Simplex]:
            while pq:
                _, s = heapq.heappop(pq)
                if s not in assigned:
                    return s
            return None

        while True:
            alpha = pop(pq_one)
            if alpha is not None:
                free = unpaired_facets(alpha)
                if not free:
                    heapq.heappush(pq_zero, (sort_key(alpha), alpha))
                else:
                    facet = free[0]
                    pairs.append((facet, alpha))
                    assigned.update((facet, alpha))
                    push_cofacets(alpha)
                    push_cofacets(facet)
                continue
            gamma = pop(pq_zero)
            if gamma is None:
                break
            critical.append(gamma)
            assigned.add(gamma)
            push_cofacets(gamma)
    return pairs, critical


def lower_star_gradient(
    K: "SimplicialComplex", g: VertexFunction, backend: str = "auto"
) -> GradientField:
    """Extend a vertex function to a discrete gradient field (Robins-Wood-Sheppard 2011).

    What is Being Computed?:
        An acyclic matching on K whose critical cells are in bijection with the changes
        of homotopy type of the lower-star filtration of g (RWS, *Theory and algorithms
        for constructing discrete Morse complexes from grayscale digital images*, IEEE
        TPAMI 33 (2011)).

    Algorithm:
        1. Make g injective by breaking ties with the vertex label.
        2. Partition K into lower stars (cells whose highest vertex is v).
        3. ``ProcessLowerStars``: in each lower star, pair v with its steepest edge, then
           greedily pair cells with exactly one unpaired facet (priority by the values
           of their other vertices), declaring a cell critical when none is available.
        Lower stars are independent, so the Julia kernel processes them in parallel
        threads; both backends pop cells in the same order and return the same field.

    Preserved Invariants:
        Acyclic by construction (pairings never leave a lower star, and lower stars are
        totally ordered by g); ``verify()`` checks it independently.

    Args:
        K: A simplicial complex.
        g: One finite value per vertex: a dict ``{vertex: value}`` or a sequence indexed
            by vertex label.
        backend: 'auto', 'julia' or 'python'.

    Returns:
        A ``GradientField``.

    Raises:
        ValueError: If g does not cover the vertices or is not finite.

    Example:
        >>> V = lower_star_gradient(K, heights)
        >>> V.verify(); V.morse_homology() == K.homology()
    """
    values = _vertex_values(K, g)
    cells = _all_simplices(K)
    V = GradientField(K, values)
    if not cells:
        return V
    backend_norm = str(backend).lower().strip()
    use_julia = backend_norm == "julia" or (backend_norm == "auto" and julia_engine.available)
    res = None
    if use_julia:
        try:
            res = julia_engine.lower_star_gradient(cells, values)
        except Exception as e:  # pragma: no cover - depends on the Julia runtime
            if backend_norm == "julia":
                raise
            warnings.warn(f"Julia lower-star gradient failed ({e!r}); falling back to Python.")
    if res is None:
        res = _rws_python(cells, values)
    pairs, critical = res
    for low, high in pairs:
        V.pair(low, high)
    V.critical = set(critical)
    return V


# --------------------------------------------------------------------------- #
# pair classes
# --------------------------------------------------------------------------- #


class PairClass:
    """Cancellation classes of a pair of critical cells (see the module docstring)."""

    CANCELLABLE = "cancellable"
    CHAIN_CANCELLABLE = "chain_cancellable"
    NOT_CANCELLABLE = "not_cancellable"

    ORDER = (CANCELLABLE, CHAIN_CANCELLABLE, NOT_CANCELLABLE)


def classify_critical_pairs(V: GradientField, p: int) -> List[dict]:
    """Classify every critical pair ``(sigma^p, tau^(p-1))`` by its cancellability.

    Args:
        V: A gradient field.
        p: The dimension of sigma.

    Returns:
        One record per pair joined by a gradient path or incident in the Morse complex:
        ``{"sigma", "tau", "incidence", "paths", "class"}``.
    """
    if p < 1 or p > V.K.dimension:
        return []
    rows = V.critical_cells(p - 1)
    cols = V.critical_cells(p)
    boundary = V.morse_boundary(p)
    counts = V.gradient_path_counts(p)
    out = []
    for j, sigma in enumerate(cols):
        for i, tau in enumerate(rows):
            c = int(boundary[i, j])
            paths = counts.get((sigma, tau), 0)
            if paths == 0 and c == 0:
                continue
            if paths == 1:
                cls = PairClass.CANCELLABLE
            elif abs(c) == 1:
                cls = PairClass.CHAIN_CANCELLABLE
            else:
                cls = PairClass.NOT_CANCELLABLE
            out.append({"sigma": sigma, "tau": tau, "incidence": c, "paths": int(paths), "class": cls})
    return out


# --------------------------------------------------------------------------- #
# lower-star persistence over Z/2
# --------------------------------------------------------------------------- #


class PersistencePair(BaseModel):
    """One persistence pair of the lower-star filtration, with its simplices.

    Attributes:
        dimension (int): Homological degree of the class.
        birth_simplex (tuple[int, ...]): The simplex creating the class.
        death_simplex (tuple[int, ...] | None): The simplex killing it (None if essential).
        birth (float): Filtration value of the birth simplex.
        death (float | None): Filtration value of the death simplex (None if essential).
    """

    dimension: int
    birth_simplex: Tuple[int, ...]
    death_simplex: Optional[Tuple[int, ...]] = None
    birth: float
    death: Optional[float] = None

    @property
    def persistence(self) -> float:
        """``death - birth`` (inf for an essential class)."""
        return float("inf") if self.death is None else float(self.death - self.birth)

    @property
    def is_essential(self) -> bool:
        """The class never dies."""
        return self.death_simplex is None


def lower_star_filtration(
    K: "SimplicialComplex", g: VertexFunction
) -> List[Tuple[Simplex, float]]:
    """Order K so faces precede cofaces, by ``f(sigma) = max_v g(v)``.

    Ties in f are broken by dimension (a face has no larger f and strictly lower
    dimension, so it never follows a coface), then by the canonical simplex order.

    Args:
        K: A simplicial complex.
        g: The vertex function.

    Returns:
        ``[(simplex, f(simplex)), ...]`` in filtration order.
    """
    values = _vertex_values(K, g)
    cells = _all_simplices(K)
    keyed = [(max(values[v] for v in s), len(s), i) for i, s in enumerate(cells)]
    keyed.sort()
    return [(cells[i], float(f)) for f, _, i in keyed]


def _reduce_z2(filtration: List[Tuple[Simplex, float]]) -> List[Tuple[int, int]]:
    """Standard column reduction over Z/2.

    Returns ``(birth_position, death_position)`` pairs, death -1 for essential classes.
    """
    position = {s: i for i, (s, _) in enumerate(filtration)}
    columns: List[Set[int]] = []
    low_to_col: Dict[int, int] = {}
    for j, (s, _) in enumerate(filtration):
        col = {position[s[:i] + s[i + 1:]] for i in range(len(s))} if len(s) > 1 else set()
        while col:
            low = max(col)
            owner = low_to_col.get(low)
            if owner is None:
                low_to_col[low] = j
                break
            col ^= columns[owner]
        columns.append(col)
    dead = set(low_to_col) | set(low_to_col.values())
    out = [(b, d) for b, d in low_to_col.items()]
    out += [(i, -1) for i in range(len(filtration)) if i not in dead]
    return sorted(out)


def lower_star_persistence(
    K: "SimplicialComplex",
    g: VertexFunction,
    include_zero_persistence: bool = False,
    backend: str = "auto",
) -> List[PersistencePair]:
    """Persistence pairs of the lower-star filtration over Z/2, with their simplices.

    What is Being Computed?:
        The standard persistence pairing (Edelsbrunner-Letscher-Zomorodian) of the
        filtration ``lower_star_filtration(K, g)``, over Z/2. The number of essential
        classes in degree p is ``dim H_p(K; Z/2)`` -- which is NOT the rational Betti
        number when K has 2-torsion (RP^2: essential classes (1, 1, 1)).

    Args:
        K: A simplicial complex.
        g: The vertex function.
        include_zero_persistence: Keep pairs with ``death == birth``.
        backend: 'auto', 'julia' or 'python'.

    Returns:
        The persistence pairs, sorted by (dimension, birth, death).
    """
    filtration = lower_star_filtration(K, g)
    backend_norm = str(backend).lower().strip()
    use_julia = backend_norm == "julia" or (backend_norm == "auto" and julia_engine.available)
    pairs = None
    if use_julia and filtration:
        try:
            pairs = julia_engine.z2_persistence_pairs([s for s, _ in filtration])
        except Exception as e:  # pragma: no cover - depends on the Julia runtime
            if backend_norm == "julia":
                raise
            warnings.warn(f"Julia persistence reduction failed ({e!r}); falling back to Python.")
    if pairs is None:
        pairs = _reduce_z2(filtration)
    out: List[PersistencePair] = []
    for b, d in pairs:
        s, fb = filtration[b]
        if d < 0:
            out.append(PersistencePair(dimension=len(s) - 1, birth_simplex=s, birth=fb))
            continue
        t, fd = filtration[d]
        if fd > fb or include_zero_persistence:
            out.append(PersistencePair(dimension=len(s) - 1, birth_simplex=s, death_simplex=t,
                                       birth=fb, death=fd))
    out.sort(key=lambda p: (p.dimension, p.birth, float("inf") if p.death is None else p.death,
                            p.birth_simplex))
    return out


def critical_pair_persistence(
    K: "SimplicialComplex", V: GradientField, backend: str = "auto"
) -> Dict[Tuple[Simplex, Simplex], float]:
    """Persistence of the pairs in which BOTH simplices are critical for V.

    Args:
        K: The complex.
        V: A lower-star gradient field of K.
        backend: 'auto', 'julia' or 'python'.

    Returns:
        ``{(sigma^(p+1), tau^p): persistence}`` (death simplex first).
    """
    out: Dict[Tuple[Simplex, Simplex], float] = {}
    for pair in lower_star_persistence(K, V.values, include_zero_persistence=True, backend=backend):
        if pair.is_essential:
            continue
        b, d = pair.birth_simplex, pair.death_simplex
        if V.is_critical(b) and V.is_critical(d):
            out[(d, b)] = pair.persistence
    return out

