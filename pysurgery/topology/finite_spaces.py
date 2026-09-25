r"""Finite topological spaces (finite T0 posets), Stong cores and strong collapses.

Overview:
    Alexandrov's correspondence identifies finite T0 spaces with finite posets. The
    convention throughout: open sets are DOWN-sets, so the minimal open set of x is

        U(x) = { y : y <= x },

    and a function between finite T0 spaces is continuous iff it is order-preserving.
    McCord's theorem ties this to simplicial complexes: the face poset X(K) is weakly
    homotopy equivalent to |K|, and the order complex K(X) of a finite space is weakly
    equivalent to X (K(X(K)) is the barycentric subdivision of K).

    Implemented here, all exact:
      * beat points and Stong cores     -> homotopy type; contractibility is DECIDED
                                           (Stong: X is contractible iff its core is a
                                           point)
      * weak points                     -> removal preserves the weak homotopy type
      * quotients and the T0 reflection
      * McCord's fibre criterion        -> a sufficient condition for a map to be a
                                           weak homotopy equivalence
      * the order complex               -> homology, through pySurgery's exact SNF
      * strong collapses of complexes   -> removal of dominated vertices
                                           (Barmak-Minian); K is strong collapsible iff
                                           its strong core is a vertex iff X(K) is
                                           contractible

Key Concepts:
    - **Beat point.** x is a down-beat point when ``U(x) - {x}`` has a maximum, an
      up-beat point when ``F(x) - {x}`` has a minimum. Removing one is a strong
      deformation retraction.
    - **Weak point.** ``U(x) - {x}`` or ``F(x) - {x}`` is contractible; removal is a weak
      homotopy equivalence (Barmak-Minian), so homology and all homotopy groups are
      unchanged.
    - **Dominated vertex.** v is dominated by v' != v when every maximal simplex
      containing v contains v'. Deleting v is a *strong collapse* (Barmak & Minian,
      *Strong homotopy types, nerves and collapses*, Discrete Comput. Geom. 47 (2012));
      the strong core is unique up to isomorphism, and K and L have the same strong
      homotopy type iff X(K) and X(L) are homotopy equivalent. Strong collapsible
      implies collapsible implies contractible.

One asymmetry to keep in mind: McCord's criterion asks for weakly contractible fibres,
and the fibres are tested with Stong cores, which decide CONTRACTIBILITY. A finite
space can be weakly contractible without being contractible (Barmak-Minian's
examples), so a failed fibre test does not prove a map is not a weak equivalence; a
passed one does prove it is.

Common Workflows:
    1. **Face poset of a complex** -> ``FiniteSpace.from_simplicial_complex(K)``.
    2. **Decide contractibility of a finite space** -> ``X.is_contractible()``.
    3. **Certify that |K| is contractible** -> ``is_strong_collapsible(K)``.
    4. **Certify a quotient map** -> ``X.mccord_certificate(Y, mapping)``.

Coefficient Ring:
    Homology through the order complex is over Z (exact SNF, torsion included).
"""

from __future__ import annotations

import warnings
from collections import deque
from typing import TYPE_CHECKING, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, Field

from ..bridge.julia_bridge import julia_engine

if TYPE_CHECKING:  # pragma: no cover
    from .complexes import SimplicialComplex

__all__ = [
    "FiniteSpace",
    "McCordCertificate",
    "StrongCollapseResult",
    "strong_collapse",
    "is_strong_collapsible",
]


def _strongly_connected_components(succ: Sequence[Set[int]], n: int) -> List[int]:
    """Iterative Tarjan; ``comp[v]`` is a valid DAG labelling of the condensation."""
    index = [-1] * n
    low = [0] * n
    on_stack = [False] * n
    comp = [-1] * n
    stack: List[int] = []
    counter = 0
    n_comp = 0
    for root in range(n):
        if index[root] != -1:
            continue
        work: List[Tuple[int, object]] = [(root, iter(succ[root]))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack[root] = True
        while work:
            v, it = work[-1]
            advanced = False
            for w in it:
                if index[w] == -1:
                    index[w] = low[w] = counter
                    counter += 1
                    stack.append(w)
                    on_stack[w] = True
                    work.append((w, iter(succ[w])))
                    advanced = True
                    break
                if on_stack[w]:
                    low[v] = min(low[v], index[w])
            if advanced:
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[v])
            if low[v] == index[v]:
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    comp[w] = n_comp
                    if w == v:
                        break
                n_comp += 1
    return comp


class McCordCertificate(BaseModel):
    """Outcome of McCord's fibre test for a map ``q: X -> Y`` of finite spaces.

    ``is_weak_equivalence`` is a one-sided guarantee: True certifies q as a weak homotopy
    equivalence; False only means the certificate was not obtained, since Stong cores
    decide contractibility while McCord needs weak contractibility.

    Attributes:
        n_target (int): Number of points of Y.
        nontrivial_fibres (list[int]): Points y whose ``q^{-1}(U(y))`` was not certified
            contractible.
        details (dict[int, tuple[int, int]]): ``y -> (|q^{-1}(U(y))|, |core|)``.
    """

    n_target: int
    nontrivial_fibres: List[int] = Field(default_factory=list)
    details: Dict[int, Tuple[int, int]] = Field(default_factory=dict)

    @property
    def is_weak_equivalence(self) -> bool:
        """True certifies the map is a weak homotopy equivalence."""
        return len(self.nontrivial_fibres) == 0

    @property
    def n_nontrivial(self) -> int:
        """How many fibres were not certified contractible."""
        return len(self.nontrivial_fibres)


class FiniteSpace:
    """A finite T0 topological space, stored as a poset on ``{0, ..., n-1}``.

    Overview:
        Each point carries its minimal open set ``U(x)`` (a down-set) and minimal closed
        set ``F(x)`` (an up-set). All constructors verify reflexivity, transitivity and
        antisymmetry (the T0 axiom) unless told the input is already a poset.

    Attributes:
        n (int): Number of points.
        labels (list[str]): A label per point (the simplex, for a face poset).
    """

    def __init__(
        self,
        downs: Sequence[Iterable[int]],
        labels: Optional[Sequence[str]] = None,
        _check: bool = True,
    ):
        self._down: List[FrozenSet[int]] = [frozenset(int(v) for v in d) for d in downs]
        self.n = len(self._down)
        self.labels = list(labels) if labels is not None else [str(i) for i in range(self.n)]
        self._up: List[FrozenSet[int]] = self._compute_ups()
        if _check:
            self._validate()

    # ------------------------------------------------------------ constructors
    @classmethod
    def from_downsets(cls, downs: Sequence[Iterable[int]], labels=None) -> "FiniteSpace":
        """Build from the minimal open set ``U(x)`` of every point.

        Args:
            downs: ``downs[x]`` is the set of y <= x (must contain x).
            labels: Optional point labels.

        Returns:
            The finite space.
        """
        return cls(downs, labels)

    @classmethod
    def from_relations(cls, n: int, relations: Iterable[Tuple[int, int]], labels=None) -> "FiniteSpace":
        """Build from any generating set of strict relations ``(a, b)`` meaning a < b.

        Args:
            n: Number of points.
            relations: Pairs (a, b) with a < b; the transitive closure is taken.
            labels: Optional point labels.

        Returns:
            The finite space.

        Raises:
            ValueError: If the relations are reflexive or contain a cycle.
        """
        direct: List[Set[int]] = [set() for _ in range(n)]
        for a, b in relations:
            if a == b:
                raise ValueError("a strict relation cannot be reflexive")
            direct[b].add(a)
        down: List[Optional[Set[int]]] = [None] * n
        state = [0] * n  # 0 new, 1 on the DFS stack, 2 done
        for root in range(n):
            if state[root]:
                continue
            stack = [(root, iter(direct[root]))]
            state[root] = 1
            while stack:
                x, it = stack[-1]
                advanced = False
                for y in it:
                    if state[y] == 1:
                        raise ValueError("relations contain a cycle; not a poset")
                    if state[y] == 0:
                        state[y] = 1
                        stack.append((y, iter(direct[y])))
                        advanced = True
                        break
                if not advanced:
                    acc = {x}
                    for y in direct[x]:
                        acc |= down[y]
                    down[x] = acc
                    state[x] = 2
                    stack.pop()
        return cls([frozenset(d) for d in down], labels)

    @classmethod
    def from_simplicial_complex(cls, K: "SimplicialComplex") -> "FiniteSpace":
        """The face poset X(K): simplices ordered by inclusion (McCord: X(K) ~_w |K|).

        Args:
            K: A simplicial complex.

        Returns:
            The face poset, labelled by the simplices (in ``K.n_simplices`` order,
            dimension by dimension); ``space.simplices`` lists them.
        """
        simplices = [tuple(int(v) for v in s) for d in K.dimensions for s in K.n_simplices(d)]
        index = {s: i for i, s in enumerate(simplices)}
        downs = []
        for s in simplices:
            faces = []
            for mask in range(1, 1 << len(s)):
                faces.append(index[tuple(s[i] for i in range(len(s)) if mask >> i & 1)])
            downs.append(frozenset(faces))
        X = cls(downs, [str(s) for s in simplices], _check=False)
        X.simplices = simplices
        return X

    @classmethod
    def chain(cls, n: int) -> "FiniteSpace":
        """A totally ordered space ``0 < 1 < ... < n-1`` (contractible)."""
        return cls.from_downsets([frozenset(range(i + 1)) for i in range(n)])

    @classmethod
    def antichain(cls, n: int) -> "FiniteSpace":
        """A discrete space of n points."""
        return cls.from_downsets([frozenset({i}) for i in range(n)])

    # -------------------------------------------------------------- validation
    def _compute_ups(self) -> List[FrozenSet[int]]:
        ups: List[Set[int]] = [set() for _ in range(self.n)]
        for x, d in enumerate(self._down):
            for y in d:
                ups[y].add(x)
        return [frozenset(u) for u in ups]

    def _validate(self) -> None:
        for x in range(self.n):
            if x not in self._down[x]:
                raise ValueError(f"down-set of {x} must contain {x} (reflexivity)")
            for y in self._down[x]:
                if not self._down[y] <= self._down[x]:
                    raise ValueError(f"transitivity fails at {y} <= {x}")
                if y != x and x in self._down[y]:
                    raise ValueError(f"antisymmetry fails between {x} and {y} (not T0)")

    # -------------------------------------------------------------- order data
    def leq(self, x: int, y: int) -> bool:
        """Whether x <= y."""
        return x in self._down[y]

    def U(self, x: int) -> FrozenSet[int]:
        """Minimal open set of x: ``{y : y <= x}``."""
        return self._down[x]

    def U_hat(self, x: int) -> FrozenSet[int]:
        """``U(x) - {x}``."""
        return self._down[x] - {x}

    def F(self, x: int) -> FrozenSet[int]:
        """Minimal closed set of x: ``{y : y >= x}``."""
        return self._up[x]

    def F_hat(self, x: int) -> FrozenSet[int]:
        """``F(x) - {x}``."""
        return self._up[x] - {x}

    def height(self, x: int) -> int:
        """Number of points strictly below x (not the length of a longest chain)."""
        return len(self._down[x]) - 1

    def covers(self) -> List[Tuple[int, int]]:
        """The Hasse diagram: pairs (a, b) with a < b and nothing strictly between.

        Returns:
            The covering relations.
        """
        out = []
        for b in range(self.n):
            below = self._down[b] - {b}
            for a in below:
                if not any(a in self._down[c] for c in below if c != a):
                    out.append((a, b))
        return out

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        return f"FiniteSpace(n={self.n}, relations={sum(len(d) - 1 for d in self._down)})"

    # -------------------------------------------------------------- subspaces
    def subspace(self, elements: Iterable[int]) -> Tuple["FiniteSpace", List[int]]:
        """The induced subspace (a subposet with the restricted order).

        Args:
            elements: The points to keep.

        Returns:
            ``(subspace, original_indices)``.
        """
        elems = sorted(set(int(e) for e in elements))
        pos = {e: i for i, e in enumerate(elems)}
        downs = [frozenset(pos[y] for y in self._down[e] if y in pos) for e in elems]
        return FiniteSpace(downs, [self.labels[e] for e in elems], _check=False), elems

    # ------------------------------------------------------------- beat points
    def beat_point_target(self, x: int) -> Optional[int]:
        """If x is a beat point, the point it retracts onto; else None.

        Args:
            x: A point.

        Returns:
            The maximum of ``U_hat(x)`` (down-beat) or the minimum of ``F_hat(x)``
            (up-beat), or None.
        """
        uh = self.U_hat(x)
        for m in uh:
            if uh <= self._down[m]:
                return m
        fh = self.F_hat(x)
        for m in fh:
            if fh <= self._up[m]:
                return m
        return None

    def is_beat_point(self, x: int) -> bool:
        """Whether x is a beat point."""
        return self.beat_point_target(x) is not None

    def stong_core(self, backend: str = "auto") -> Tuple["FiniteSpace", Dict[int, int]]:
        """Iteratively remove beat points (Stong's core).

        What is Being Computed?:
            The core of X: a subspace with no beat points, reached by removing beat
            points one at a time. It is unique up to homeomorphism, and the retraction
            X -> core is a homotopy equivalence; X is contractible iff the core is a
            point (Stong 1966).

        Algorithm:
            Worklist on adjacency sets: only the neighbours of a removed point can
            change status, so only they are re-queued (Python), or the same loop in the
            ``stong_core_jl`` kernel (Julia).

        Args:
            backend: 'auto', 'julia' or 'python'.

        Returns:
            ``(core, retraction)`` with ``retraction`` mapping every original index to a
            core index.
        """
        if self.n == 0:
            return FiniteSpace([], [], _check=False), {}
        backend_norm = str(backend).lower().strip()
        use_julia = backend_norm == "julia" or (backend_norm == "auto" and julia_engine.available)
        alive_redirect = None
        if use_julia and self.n > 1:
            try:
                alive_redirect = julia_engine.stong_core([sorted(d) for d in self._down])
            except Exception as e:  # pragma: no cover - depends on the Julia runtime
                if backend_norm == "julia":
                    raise
                warnings.warn(f"Julia Stong core failed ({e!r}); falling back to Python.")
        if alive_redirect is None:
            alive_redirect = self._stong_core_python()
        alive, redirect = alive_redirect
        elems = sorted(alive)
        pos = {e: i for i, e in enumerate(elems)}
        core = FiniteSpace(
            [frozenset(pos[y] for y in self._down[e] if y in pos) for e in elems],
            [self.labels[e] for e in elems],
            _check=False,
        )
        retraction: Dict[int, int] = {}
        for x in range(self.n):
            y, seen = x, set()
            while y not in pos:
                if y in seen:  # pragma: no cover - would be a bug
                    raise RuntimeError("retraction chain did not terminate")
                seen.add(y)
                y = redirect[y]
            retraction[x] = pos[y]
        return core, retraction

    def _stong_core_python(self) -> Tuple[Set[int], Dict[int, int]]:
        down = {x: set(self._down[x]) for x in range(self.n)}
        up = {x: set(self._up[x]) for x in range(self.n)}
        alive = set(range(self.n))
        redirect: Dict[int, int] = {}

        def target(x: int) -> Optional[int]:
            uh = down[x] - {x}
            for m in sorted(uh):
                if uh <= down[m]:
                    return m
            fh = up[x] - {x}
            for m in sorted(fh):
                if fh <= up[m]:
                    return m
            return None

        queue = deque(range(self.n))
        queued = set(range(self.n))
        while queue and len(alive) > 1:
            x = queue.popleft()
            queued.discard(x)
            if x not in alive:
                continue
            t = target(x)
            if t is None:
                continue
            redirect[x] = t
            neighbours = (down[x] | up[x]) - {x}
            for y in up[x] - {x}:
                down[y].discard(x)
            for y in down[x] - {x}:
                up[y].discard(x)
            alive.discard(x)
            del down[x], up[x]
            for y in sorted(neighbours):
                if y in alive and y not in queued:
                    queue.append(y)
                    queued.add(y)
        return alive, redirect

    def is_contractible(self, backend: str = "auto") -> bool:
        """Exact, via Stong's theorem: X is contractible iff its core is a point.

        Args:
            backend: 'auto', 'julia' or 'python'.

        Returns:
            Whether X is contractible (the empty space is not).
        """
        if self.n == 0:
            return False
        return self.stong_core(backend=backend)[0].n == 1

    # ------------------------------------------------------------- weak points
    def is_weak_point(self, x: int) -> bool:
        """``U_hat(x)`` or ``F_hat(x)`` contractible: removal is a weak equivalence."""
        for sub in (self.U_hat(x), self.F_hat(x)):
            if sub:
                s, _ = self.subspace(sub)
                if s.is_contractible(backend="python"):
                    return True
        return False

    def reduce_weak_points(self) -> Tuple["FiniteSpace", List[int]]:
        """Remove beat points and weak points until none remain.

        Preserves the weak homotopy type (hence homology and all homotopy groups), so
        it is a legitimate preprocessing step before computing invariants. It is a
        subspace inclusion, not a quotient.

        Returns:
            ``(reduced_space, original_indices_kept)``.
        """
        down = {x: set(self._down[x]) for x in range(self.n)}
        up = {x: set(self._up[x]) for x in range(self.n)}
        alive = set(range(self.n))

        def is_removable(x: int) -> bool:
            uh, fh = down[x] - {x}, up[x] - {x}
            for m in uh:
                if uh <= down[m]:
                    return True
            for m in fh:
                if fh <= up[m]:
                    return True
            for side in (uh, fh):
                if side:
                    sub, _ = self.subspace(side)
                    if sub.is_contractible(backend="python"):
                        return True
            return False

        queue = deque(range(self.n))
        queued = set(range(self.n))
        while queue and len(alive) > 1:
            x = queue.popleft()
            queued.discard(x)
            if x not in alive or not is_removable(x):
                continue
            neighbours = (down[x] | up[x]) - {x}
            for y in up[x] - {x}:
                down[y].discard(x)
            for y in down[x] - {x}:
                up[y].discard(x)
            alive.discard(x)
            del down[x], up[x]
            for y in neighbours:
                if y in alive and y not in queued:
                    queue.append(y)
                    queued.add(y)
        elems = sorted(alive)
        pos = {e: i for i, e in enumerate(elems)}
        reduced = FiniteSpace(
            [frozenset(pos[y] for y in down[e]) for e in elems],
            [self.labels[e] for e in elems],
            _check=False,
        )
        return reduced, elems

    # ------------------------------------------------------- quotients / T0
    def quotient(self, blocks: Sequence[Iterable[int]]) -> Tuple["FiniteSpace", List[int]]:
        """Quotient by a partition, then the T0 reflection.

        What is Being Computed?:
            The quotient topology on a finite space is the Alexandrov topology of the
            preorder generated by the images of the relations; its T0 reflection
            identifies mutually comparable blocks. Both are done by one pass of strongly
            connected components on the block graph (a cycle is exactly a set of blocks
            the T0 reflection identifies), then a bitset closure.

        Args:
            blocks: A partition of the points.

        Returns:
            ``(quotient, mapping)`` with ``mapping[x]`` the image of x; the map is
            order-preserving, hence continuous.

        Raises:
            ValueError: If ``blocks`` is not a partition of the points.
        """
        block_of = [-1] * self.n
        blocks = [list(b) for b in blocks]
        for bi, block in enumerate(blocks):
            for x in block:
                if block_of[x] != -1:
                    raise ValueError(f"element {x} appears in two blocks")
                block_of[x] = bi
        missing = [i for i, b in enumerate(block_of) if b == -1]
        if missing:
            raise ValueError(f"partition does not cover elements {missing[:5]}")
        nb = len(blocks)
        succ: List[Set[int]] = [set() for _ in range(nb)]
        for x in range(self.n):
            bx = block_of[x]
            for y in self._down[x]:
                by = block_of[y]
                if by != bx:
                    succ[by].add(bx)
        comp = _strongly_connected_components(succ, nb)
        nc = max(comp) + 1 if nb else 0
        cond: List[Set[int]] = [set() for _ in range(nc)]
        indegree = [0] * nc
        for a in range(nb):
            for b in succ[a]:
                ca, cb = comp[a], comp[b]
                if ca != cb and cb not in cond[ca]:
                    cond[ca].add(cb)
                    indegree[cb] += 1
        order: List[int] = []
        q = deque(c for c in range(nc) if indegree[c] == 0)
        while q:
            c = q.popleft()
            order.append(c)
            for d in cond[c]:
                indegree[d] -= 1
                if indegree[d] == 0:
                    q.append(d)
        down_bits = [1 << c for c in range(nc)]
        for c in order:
            for d in cond[c]:
                down_bits[d] |= down_bits[c]
        downs = []
        for c in range(nc):
            mask, out = down_bits[c], []
            while mask:
                low = mask & -mask
                out.append(low.bit_length() - 1)
                mask ^= low
            downs.append(frozenset(out))
        members: List[List[int]] = [[] for _ in range(nc)]
        for b in range(nb):
            members[comp[b]].extend(blocks[b])
        labels = []
        for c in range(nc):
            names = sorted(self.labels[x] for x in members[c])
            labels.append("{" + ",".join(names[:3]) + ("..." if len(names) > 3 else "") + "}")
        return FiniteSpace(downs, labels, _check=False), [comp[block_of[x]] for x in range(self.n)]

    # ------------------------------------------------------- McCord certificate
    def mccord_certificate(self, target: "FiniteSpace", mapping: Sequence[int]) -> McCordCertificate:
        """Test McCord's fibre criterion for a map ``q: self -> target``.

        What is Being Computed?:
            For each y in the target, ``q^{-1}(U(y))`` -- the preimage of the MINIMAL
            OPEN SET, not of the point -- is tested for contractibility via Stong cores.
            If every such preimage is contractible, q is a weak homotopy equivalence
            (McCord 1966).

        Args:
            target: The target space.
            mapping: ``mapping[x]`` is the image of x.

        Returns:
            A ``McCordCertificate``.

        Raises:
            ValueError: If ``mapping`` is not defined everywhere or not order-preserving.
        """
        if len(mapping) != self.n:
            raise ValueError("mapping must be defined on every element of the source")
        for x, y in enumerate(mapping):
            for z in self._down[x]:
                if not target.leq(mapping[z], y):
                    raise ValueError(f"map is not order-preserving at {z} <= {x}")
        nontrivial: List[int] = []
        details: Dict[int, Tuple[int, int]] = {}
        for y in range(target.n):
            opens = target.U(y)
            pre = [x for x in range(self.n) if mapping[x] in opens]
            if not pre:
                nontrivial.append(y)
                details[y] = (0, 0)
                continue
            sub, _ = self.subspace(pre)
            core, _ = sub.stong_core(backend="python")
            details[y] = (len(pre), core.n)
            if core.n != 1:
                nontrivial.append(y)
        return McCordCertificate(n_target=target.n, nontrivial_fibres=nontrivial, details=details)

    # ------------------------------------------------------------- order complex
    def order_complex(self, max_dim: Optional[int] = None) -> "SimplicialComplex":
        """The simplicial complex of nonempty chains; ``|K(X)|`` is weakly equivalent to X.

        Args:
            max_dim: Optional cap on the chain length minus one (the result is then a
                skeleton, and its top homology is not that of X).

        Returns:
            The order complex as a pySurgery ``SimplicialComplex`` (vertices = points).
        """
        from .complexes import SimplicialComplex

        chains: List[Tuple[int, ...]] = []
        order = sorted(range(self.n), key=lambda x: (len(self._down[x]), x))

        def extend(chain: Tuple[int, ...], candidates: List[int]) -> None:
            maximal = True
            if max_dim is None or len(chain) - 1 < max_dim:
                for i, c in enumerate(candidates):
                    maximal = False
                    extend(chain + (c,), [d for d in candidates[i + 1:] if self.leq(c, d)])
            if maximal:
                chains.append(chain)

        for i, x in enumerate(order):
            extend((x,), [y for y in order[i + 1:] if self.leq(x, y)])
        return SimplicialComplex.from_simplices(chains, close_under_faces=True)

    # ------------------------------------------------------------------ homology
    def homology(self, reduce_first: bool = True, backend: str = "auto") -> Dict[int, Tuple[int, List[int]]]:
        """Integer homology of X (that of its order complex), exactly.

        Args:
            reduce_first: Remove beat and weak points first (preserves weak homotopy
                type, shrinks the order complex).
            backend: 'auto', 'julia' or 'python' (for the SNF).

        Returns:
            ``{degree: (rank, torsion)}``; empty for the empty space.
        """
        space = self.reduce_weak_points()[0] if reduce_first else self
        if space.n == 0:
            return {}
        return space.order_complex().homology(backend=backend)

    def betti_numbers(self, reduce_first: bool = True, backend: str = "auto") -> List[int]:
        """Betti numbers of X, degree 0 upward.

        Args:
            reduce_first: Remove beat and weak points first.
            backend: 'auto', 'julia' or 'python'.

        Returns:
            The Betti numbers.
        """
        h = self.homology(reduce_first=reduce_first, backend=backend)
        return [int(h[d][0]) for d in sorted(h)]


# --------------------------------------------------------------------------- #
# strong collapses of simplicial complexes (Barmak-Minian)
# --------------------------------------------------------------------------- #


class StrongCollapseResult(BaseModel):
    """Result of ``strong_collapse``.

    Attributes:
        core_simplices (list[tuple[int, ...]]): Maximal simplices of the strong core.
        removed (list[tuple[int, int]]): ``(v, v')`` for every deleted vertex v, dominated
            by v' at the time of its removal, in order.
        n_core_vertices (int): Number of vertices of the core.
        is_strong_collapsible (bool): The core is a single vertex.
    """

    core_simplices: List[Tuple[int, ...]]
    removed: List[Tuple[int, int]]
    n_core_vertices: int
    is_strong_collapsible: bool

    def core(self) -> "SimplicialComplex":
        """The strong core as a ``SimplicialComplex``.

        Returns:
            The core.
        """
        from .complexes import SimplicialComplex

        return SimplicialComplex.from_simplices(self.core_simplices, close_under_faces=True)


def _maximal_simplices_of(K: "SimplicialComplex") -> List[Tuple[int, ...]]:
    from .local_homology import maximal_simplices

    return maximal_simplices(K)


def _strong_collapse_python(maximal: List[Tuple[int, ...]]):
    facets = {i: frozenset(s) for i, s in enumerate(maximal)}
    star: Dict[int, Set[int]] = {}
    for i, s in facets.items():
        for v in s:
            star.setdefault(v, set()).add(i)
    removed: List[Tuple[int, int]] = []

    def dominator(v: int) -> Optional[int]:
        common = None
        for i in star[v]:
            common = set(facets[i]) if common is None else common & facets[i]
            if common is not None and len(common) <= 1:
                return None
        common.discard(v)
        return min(common) if common else None

    queue = deque(sorted(star))
    queued = set(star)
    while queue and len(star) > 1:
        v = queue.popleft()
        queued.discard(v)
        if v not in star:
            continue
        w = dominator(v)
        if w is None:
            continue
        removed.append((v, w))
        touched = set()
        for i in list(star[v]):
            new = facets[i] - {v}
            touched |= new
            # the face lk is contained in a facet through w; keep it only if maximal
            del facets[i]
            for u in new:
                star[u].discard(i)
            if new and not any(new < facets[j] for j in star[w]):
                if not any(new == facets[j] for j in star[w]):
                    facets[i] = frozenset(new)
                    for u in new:
                        star[u].add(i)
        del star[v]
        for u in sorted(touched):
            if u in star and u not in queued:
                queue.append(u)
                queued.add(u)
    core = sorted((tuple(sorted(f)) for f in facets.values()), key=lambda s: (len(s), s))
    return core, removed


def strong_collapse(K: "SimplicialComplex", backend: str = "auto") -> StrongCollapseResult:
    """Strong-collapse K to its strong core by deleting dominated vertices.

    What is Being Computed?:
        A vertex v is dominated by v' != v when every maximal simplex containing v also
        contains v'; deleting v (all simplices through it) is an elementary strong
        collapse. Repeating until no vertex is dominated yields the strong core, unique
        up to isomorphism (Barmak-Minian 2012). K is strong collapsible iff the core is
        a vertex -- equivalently iff the face poset X(K) is contractible, which
        ``FiniteSpace.is_contractible`` decides independently.

    Algorithm:
        Worklist over vertices on the maximal-simplex hypergraph: deleting v replaces
        every facet ``sigma`` through v by ``sigma - v``, which stays a facet only if it
        is not contained in another facet (it always lies in a facet through v'). Only
        the vertices of the modified facets are re-queued.

    Args:
        K: A simplicial complex.
        backend: 'auto', 'julia' or 'python'.

    Returns:
        A ``StrongCollapseResult``.

    Use When:
        - Certifying that |K| is contractible (strong collapsible => collapsible =>
          contractible) without any homotopy-group computation.
        - Shrinking a complex while preserving its strong homotopy type.
    """
    maximal = _maximal_simplices_of(K)
    if not maximal:
        return StrongCollapseResult(core_simplices=[], removed=[], n_core_vertices=0,
                                    is_strong_collapsible=False)
    backend_norm = str(backend).lower().strip()
    use_julia = backend_norm == "julia" or (backend_norm == "auto" and julia_engine.available)
    res = None
    if use_julia:
        try:
            res = julia_engine.strong_collapse(maximal)
        except Exception as e:  # pragma: no cover - depends on the Julia runtime
            if backend_norm == "julia":
                raise
            warnings.warn(f"Julia strong collapse failed ({e!r}); falling back to Python.")
    if res is None:
        res = _strong_collapse_python(maximal)
    core, removed = res
    n_v = len({v for s in core for v in s})
    return StrongCollapseResult(
        core_simplices=core, removed=removed, n_core_vertices=n_v,
        is_strong_collapsible=n_v == 1,
    )


def is_strong_collapsible(K: "SimplicialComplex", backend: str = "auto") -> bool:
    """Whether K strong-collapses to a vertex (hence |K| is contractible).

    Args:
        K: A simplicial complex.
        backend: 'auto', 'julia' or 'python'.

    Returns:
        True iff the strong core of K is a single vertex.
    """
    return strong_collapse(K, backend=backend).is_strong_collapsible

