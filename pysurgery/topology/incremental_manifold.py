r"""Exact homology-manifold verdicts along a filtration, updated incrementally.

Overview:
    A filtration report asks, at every threshold, whether the sub-complex ``K_eps``
    is a homology manifold (closed or with boundary) of dimension ``d = dim K_eps``.
    The answer is the one ``certify_homology_manifold(K_eps, d)`` gives -- the
    definition, checked at every simplex -- but it is maintained as simplices enter,
    so the whole filtration costs one pass over the simplices instead of one
    certificate per threshold.

Key Concepts:
    - **The definition.** ``H_j(|K|, |K| - x) = H~_{j-k-1}(lk sigma)`` for x in an open
      k-simplex sigma. K is a homology d-manifold (with boundary) iff every simplex is
      ``sphere`` (``lk sigma`` has the homology of S^(d-k-1) and that dimension) or
      ``acyclic`` (``lk sigma`` is acyclic, pure of dimension d-k-1). The boundary
      consistency that ``certify_homology_manifold`` also checks is then automatic
      (Mitchell, Proc. AMS 110 (1990); for d <= 3 it follows directly from the
      combinatorial description below).
    - **Where the witnesses are.** Call sigma *singular* if it is neither. Then K is a
      manifold iff no simplex is singular iff no *inclusion-maximal* singular simplex
      exists (a singular simplex of largest dimension has only regular cofaces). So it
      suffices to decide sigma only when every strict coface of sigma is regular. Then
      ``lk sigma`` is itself a homology (d-k-1)-manifold, because the links inside
      ``lk sigma`` are the links in K of those cofaces.
    - **Codimension <= 3 is combinatorial, exactly.** With ``c_i`` the number of
      cofaces of dimension ``k + i`` (the i-1 simplices of the link):

          codim 0   always sphere (empty link = S^-1).
          codim 1   the link is c_1 points: sphere iff c_1 = 2, acyclic iff c_1 = 1.
          codim 2   the link is a graph: sphere iff connected with V - E = 0,
                    acyclic iff connected with V - E = 1 and E >= 1 (a tree that is
                    not a point).
          codim 3   (all strict cofaces regular) the link is a combinatorial surface:
                    sphere iff connected with chi = 2 (S^2), acyclic iff connected with
                    chi = 1 and a boundary edge (the disk; chi = 1 closed is RP^2).

      Connectivity of every link is kept by a union-find per simplex; counts, chi and
      the number of boundary edges of a link are integers updated in O(1).
    - **Codimension >= 4** (only when ``d >= 4``): exact integer reduced homology of the
      link (``reduced_homology_from_simplices``), computed only for simplices whose
      strict cofaces are all regular.
    - **Incremental.** A new simplex tau changes only the links of its faces, so only
      faces of tau are re-decided. ``G(sigma)`` ("sigma and all its cofaces are
      regular") propagates downwards through a per-simplex count of immediate cofaces
      with ``G`` false. A change of ``d`` re-decides everything (at most ``d + 1``
      times per filtration).

Attributes reported per threshold:
    ``is_manifold``, ``dimension`` (``dim K_eps``), ``defects`` (the number of
    inclusion-maximal singular simplices; 0 iff manifold) and ``is_closed`` (a manifold
    with no (d-1)-face lying in exactly one d-simplex).
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from ..algebra.sparse_elimination import reduced_homology_from_simplices

Simplex = Tuple[int, ...]

__all__ = [
    "ManifoldVerdict",
    "IncrementalManifoldChecker",
    "manifold_verdict",
    "filtration_manifold_verdicts",
]


class ManifoldVerdict(NamedTuple):
    """Homology-manifold status of one complex, relative to its own dimension.

    Attributes:
        is_manifold (bool): Every simplex has a sphere or acyclic link of the right
            dimension (the definition of a homology manifold, closed or with boundary).
        dimension (int): ``dim K`` (-1 for the empty complex).
        defects (int): Number of inclusion-maximal singular simplices; 0 iff manifold.
        is_closed (bool): Manifold with empty boundary.
    """

    is_manifold: bool
    dimension: int
    defects: int
    is_closed: bool


class IncrementalManifoldChecker:
    """Exact homology-manifold verdict of a growing simplicial complex.

    Overview:
        Simplices are added one at a time (faces first -- a missing face is added
        implicitly); :meth:`verdict` returns the status of the current complex. The
        verdict equals ``certify_homology_manifold(K, dim K)`` (closed / with boundary)
        and costs, over a whole filtration, ``O(sum_tau 2^(dim tau + 1))`` for
        ``dim K <= 3``; see the module docstring for the method.

    Example:
        >>> chk = IncrementalManifoldChecker()
        >>> for s in [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]:
        ...     chk.add(s)
        >>> chk.verdict()
        ManifoldVerdict(is_manifold=True, dimension=2, defects=0, is_closed=False)
    """

    def __init__(self) -> None:
        self._id: Dict[Simplex, int] = {}
        self._simplex: List[Simplex] = []
        self._dim: List[int] = []
        self._c1: List[int] = []
        self._c2: List[int] = []
        self._c3: List[int] = []
        # Number of (k+2)-cofaces rho with c1(rho) == 1: boundary edges of lk sigma.
        self._be: List[int] = []
        # Union-find over the link's vertices (keyed by the added vertex).
        self._lk_parent: List[Optional[Dict[int, int]]] = []
        self._lk_unions: List[int] = []
        # Immediate cofaces (only needed to read links of codimension >= 4).
        self._cof: List[List[int]] = []
        # Immediate cofaces tau with G(tau) False; A(sigma) <=> _nbad == 0.
        self._nbad: List[int] = []
        self._G: List[bool] = []
        self._witness: List[bool] = []
        self._is_dirty: List[bool] = []
        self._dirty: Dict[int, List[int]] = defaultdict(list)
        self._full_recheck = False
        self._n_witness = 0
        # k -> number of k-simplices lying in exactly one (k+1)-simplex.
        self._n_free: Dict[int, int] = defaultdict(int)
        self.dimension = -1

    # ------------------------------------------------------------------ #
    # growth
    # ------------------------------------------------------------------ #
    def __contains__(self, simplex) -> bool:
        return tuple(sorted(int(v) for v in simplex)) in self._id

    def __len__(self) -> int:
        return len(self._simplex)

    def add(self, simplex: Iterable[int]) -> None:
        """Add a simplex (and, implicitly, any of its faces not yet present).

        Args:
            simplex: Vertex labels (any order).
        """
        s = tuple(sorted(int(v) for v in simplex))
        if not s or s in self._id:
            return
        if len(s) > 1:
            for i in range(len(s)):
                f = s[:i] + s[i + 1:]
                if f not in self._id:
                    self.add(f)
        self._insert(s)

    def add_many(self, simplices: Iterable[Iterable[int]]) -> None:
        """Add simplices in the given order (see :meth:`add`).

        Args:
            simplices: Any iterable of simplices.
        """
        for s in simplices:
            self.add(s)

    def _insert(self, s: Simplex) -> None:
        i = len(self._simplex)
        m = len(s) - 1
        self._id[s] = i
        self._simplex.append(s)
        self._dim.append(m)
        self._c1.append(0)
        self._c2.append(0)
        self._c3.append(0)
        self._be.append(0)
        self._lk_parent.append(None)
        self._lk_unions.append(0)
        self._cof.append([])
        self._nbad.append(0)
        self._G.append(True)
        self._witness.append(False)
        self._is_dirty.append(False)
        if m > self.dimension:
            self.dimension = m
            self._full_recheck = True
        ids = self._id
        self._mark(i)
        if m == 0:
            return
        sset = s
        for r in range(m, 0, -1):          # proper faces of dimension r - 1
            codim = m - r + 1
            for face in itertools.combinations(sset, r):
                j = ids[face]
                self._mark(j)
                if codim == 1:
                    x = next(v for v in sset if v not in face)
                    self._cof[j].append(i)
                    self._bump_c1(j, face)
                    par = self._lk_parent[j]
                    if par is None:
                        par = self._lk_parent[j] = {}
                    par[x] = x
                elif codim == 2:
                    self._c2[j] += 1
                    x, y = (v for v in sset if v not in face)
                    self._lk_union(j, x, y)
                elif codim == 3:
                    self._c3[j] += 1

    def _bump_c1(self, j: int, face: Simplex) -> None:
        old = self._c1[j]
        new = old + 1
        self._c1[j] = new
        k = self._dim[j]
        if old == 1 or new == 1:
            delta = 1 if new == 1 else -1
            self._n_free[k] += delta
            if k >= 2:  # face is a boundary edge of lk(sigma) for its codim-2 faces
                for sub in itertools.combinations(face, k - 1):
                    self._be[self._id[sub]] += delta

    def _lk_union(self, j: int, x: int, y: int) -> None:
        par = self._lk_parent[j]

        def find(a: int) -> int:
            root = a
            while par[root] != root:
                root = par[root]
            while par[a] != root:
                par[a], a = root, par[a]
            return root

        rx, ry = find(x), find(y)
        if rx != ry:
            par[rx] = ry
            self._lk_unions[j] += 1

    def _mark(self, j: int) -> None:
        if not self._is_dirty[j]:
            self._is_dirty[j] = True
            self._dirty[self._dim[j]].append(j)

    # ------------------------------------------------------------------ #
    # local decisions
    # ------------------------------------------------------------------ #
    def _regular(self, j: int, codim: int) -> bool:
        """``lk sigma`` is a sphere or acyclic of dimension codim - 1 (sigma = simplex j).

        For codim >= 3 this is only called when every strict coface is regular.
        """
        if codim == 0:
            return True
        c1 = self._c1[j]
        if codim == 1:
            return c1 == 1 or c1 == 2
        comps = c1 - self._lk_unions[j]
        c2 = self._c2[j]
        if codim == 2:
            return comps == 1 and c2 >= 1 and 0 <= c1 - c2 <= 1
        if codim == 3:
            chi = c1 - c2 + self._c3[j]
            return comps == 1 and (chi == 2 or (chi == 1 and self._be[j] > 0))
        return self._regular_by_homology(j, codim)

    def _regular_by_homology(self, j: int, codim: int) -> bool:
        sigma = self._simplex[j]
        sset = set(sigma)
        seen = set()
        stack = list(self._cof[j])
        link: List[Simplex] = []
        while stack:
            t = stack.pop()
            if t in seen:
                continue
            seen.add(t)
            link.append(tuple(v for v in self._simplex[t] if v not in sset))
            stack.extend(self._cof[t])
        red, ldim = reduced_homology_from_simplices(link)
        want = codim - 1
        if ldim != want:
            return False
        nz = {d: (r, t) for d, (r, t) in red.items() if r or t}
        return not nz or (set(nz) == {want} and nz[want][0] == 1 and not nz[want][1])

    # ------------------------------------------------------------------ #
    # verdict
    # ------------------------------------------------------------------ #
    def _settle(self) -> None:
        if self._full_recheck:
            for j in range(len(self._simplex)):
                self._mark(j)
            self._full_recheck = False
        d = self.dimension
        for k in range(d, -1, -1):
            bucket = self._dirty.pop(k, None)
            if not bucket:
                continue
            for j in bucket:
                self._is_dirty[j] = False
                A = self._nbad[j] == 0
                good = A and self._regular(j, d - k)
                wit = A and not good
                if wit != self._witness[j]:
                    self._witness[j] = wit
                    self._n_witness += 1 if wit else -1
                if good != self._G[j]:
                    self._G[j] = good
                    if k > 0:
                        delta = -1 if good else 1
                        s = self._simplex[j]
                        for i in range(k + 1):
                            f = self._id[s[:i] + s[i + 1:]]
                            self._nbad[f] += delta
                            self._mark(f)
        self._dirty.clear()

    def verdict(self) -> ManifoldVerdict:
        """Status of the current complex relative to its own dimension.

        Returns:
            A :class:`ManifoldVerdict`.
        """
        self._settle()
        d = self.dimension
        ok = self._n_witness == 0
        closed = ok and (d <= 0 or self._n_free.get(d - 1, 0) == 0)
        return ManifoldVerdict(ok, d, self._n_witness, closed)

    def witnesses(self) -> List[Simplex]:
        """The inclusion-maximal singular simplices of the current complex.

        Returns:
            The simplices, sorted by dimension then vertices; empty iff manifold.
        """
        self._settle()
        found = (self._simplex[j] for j in range(len(self._simplex)) if self._witness[j])
        return sorted(found, key=lambda s: (len(s), s))

    def diagnostics(self) -> Dict[int, str]:
        """Failure reasons keyed by vertex, for every vertex of a singular witness.

        Each inclusion-maximal singular simplex sigma (see :meth:`witnesses`) is
        reported at each of its vertices; a vertex on several witnesses keeps the
        reason of the first one in :meth:`witnesses` order.

        Returns:
            ``{vertex: reason}``; empty iff manifold.
        """
        n = self.dimension
        out: Dict[int, str] = {}
        for s in self.witnesses():
            k = len(s) - 1
            m = n - k - 1
            reason = (f"singular {k}-simplex {'-'.join(str(v) for v in s)}: its link is "
                      f"neither a homology {m}-sphere nor acyclic of dimension {m}")
            for v in s:
                out.setdefault(v, reason)
        return out


def manifold_verdict(simplices: Iterable[Iterable[int]]) -> ManifoldVerdict:
    """One-shot :class:`ManifoldVerdict` of the complex generated by ``simplices``.

    Args:
        simplices: Any simplices (closed under faces implicitly).

    Returns:
        The verdict relative to the complex's own dimension.
    """
    chk = IncrementalManifoldChecker()
    chk.add_many(sorted((tuple(sorted(int(v) for v in s)) for s in simplices), key=len))
    return chk.verdict()


def filtration_manifold_verdicts(
    values: Mapping[Simplex, float],
    epsilons: Sequence[float],
    *,
    tol: float = 1e-12,
    evaluate: Optional[Sequence[bool]] = None,
) -> List[Optional[ManifoldVerdict]]:
    """Verdict of ``K_eps = {sigma : value(sigma) <= eps}`` at every threshold.

    Values are made monotone first (a simplex enters no earlier than its faces), so
    ``K_eps`` is a complex even when rounding breaks monotonicity by an ulp.

    Args:
        values: ``{simplex: appearance value}`` of a filtered complex.
        epsilons: Thresholds, in any order (repeats allowed).
        tol: Absolute tolerance: sigma is in ``K_eps`` when ``value <= eps + tol``.
        evaluate: Optional mask; where False, ``None`` is returned for that threshold
            (the simplices up to it are still added).

    Returns:
        One verdict (or ``None``) per threshold, in the order of ``epsilons``.
    """
    eff: Dict[Simplex, float] = {}
    for s in sorted(values, key=len):
        v = float(values[s])
        if len(s) > 1:
            for i in range(len(s)):
                v = max(v, eff.get(s[:i] + s[i + 1:], v))
        eff[s] = v
    order = sorted(eff, key=lambda s: (eff[s], len(s)))
    chk = IncrementalManifoldChecker()
    out: List[Optional[ManifoldVerdict]] = [None] * len(epsilons)
    pos = 0
    for idx in sorted(range(len(epsilons)), key=lambda t: epsilons[t]):
        bound = float(epsilons[idx]) + tol
        while pos < len(order) and eff[order[pos]] <= bound:
            chk.add(order[pos])
            pos += 1
        if evaluate is None or evaluate[idx]:
            out[idx] = chk.verdict()
    return out
