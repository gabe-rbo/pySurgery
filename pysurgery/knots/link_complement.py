r"""The fundamental group of a link complement, homomorphism counts, splitness certificates.

Overview:
    Homology cannot see splitness: by Alexander duality ``H_1(S^3 - L) = Z^k`` for EVERY
    k-component link. The complement's fundamental group does see it. This module reads
    that group off a certified diagram of the polygons and turns decidable finite
    computations on it into certificates:

        wirtinger_presentation(curves)   pi_1(S^3 - L) as a pySurgery ``FundamentalGroup``
        alexander_check(curves, G)       H_1 of the presentation must be Z^k
        count_homomorphisms(G, n)        |Hom(G, S_n)|, exactly (any finitely presented G)
        certify_splitness(curves)        "split" / "non-split" / "undetermined"
        certify_knottedness(curve)       "knotted" / "unknotted" / "undetermined"

Key Concepts:
    - **The Wirtinger presentation** of a CERTIFIED generic diagram
      (``diagrams.knot_diagram``): one generator per arc, one relator per crossing. A
      presentation read off a non-generic projection is a presentation of the wrong
      group, so the diagram is certified first, and ``alexander_check`` -- known in
      advance for every link -- is REQUIRED to pass before any certificate built on the
      presentation is issued.
    - **Certification is asymmetric, because the problem is undecidable.** Deciding
      whether a finitely presented group is a free product is not algorithmic
      (Adian-Rabin); Tietze reduction is sound but incomplete. So:

        SPLIT      a separating plane, found by LP and VERIFIED point by point; or Tietze
                   reduces pi_1 to the free group of rank k with NO relators (a link group
                   is free of rank k exactly for the k-component unlink -- a proof).
        NON-SPLIT  if L = L_S u L_T splits, pi_1 = pi_1(L_S) * pi_1(L_T) and
                   |Hom(pi_1, S_n)| = |Hom(pi_1(L_S), S_n)| |Hom(pi_1(L_T), S_n)| -- exact
                   finite counts. A mismatch for EVERY bipartition proves non-splitness.
        otherwise  "undetermined" -- never a guess.
    - **Knottedness.** UNKNOTTED when Tietze reduces pi_1 to ``<x | > = Z`` (a knot group
      is Z exactly for the unknot, by Dehn's lemma); KNOTTED when |Hom(pi_1, S_n)|
      differs from ``n! = |Hom(Z, S_n)|``.

Common Workflows:
    1. **Group of a link** -> ``link_group(curves)`` (Tietze-simplified).
    2. **Nontriviality of any pi_1** -> ``count_homomorphisms(pi1, n) > 1``.
    3. **Is this link split?** -> ``certify_splitness(curves)``.

Coefficient Ring:
    Group theory over Z; homomorphism counts are exact integers.
"""

from __future__ import annotations

import itertools
import math
import warnings
from bisect import bisect_right
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, Field

from ..bridge.julia_bridge import julia_engine
from ..topology.fundamental_group import FundamentalGroup, simplify_presentation
from . import diagrams as KD

__all__ = [
    "wirtinger_presentation",
    "link_group",
    "alexander_check",
    "HomCount",
    "count_homomorphisms",
    "free_group_hom_count",
    "SplitnessCertificate",
    "separating_plane",
    "certify_splitness",
    "certify_knottedness",
]


# --------------------------------------------------------------------------- #
# the Wirtinger presentation
# --------------------------------------------------------------------------- #


def _tok(g: int, e: int) -> str:
    return f"x{g}" if e > 0 else f"x{g}^-1"


def wirtinger_presentation(
    curves: Sequence[np.ndarray], D: Optional[KD.KnotDiagram] = None, backend: str = "auto"
) -> FundamentalGroup:
    """``pi_1(S^3 - L)`` from a certified diagram: one generator per arc, one relator per crossing.

    What is Being Computed?:
        Arcs are the pieces a component is cut into where it passes UNDER something. At a
        crossing of sign e the over-arc a conjugates the incoming under-arc b into the
        outgoing one c: ``c = a^e b a^-e`` (relator ``c^-1 a^e b a^-e``). The other
        convention gives the image under x -> x^-1, an isomorphic group. One relator is
        a consequence of the others (per connected piece of the diagram), and dropping
        ONE is always valid -- which is what makes the unknot come out as ``<x | > = Z``.
        A component with no undercrossings is a single arc: one generator.

    Args:
        curves: The polygons of the link.
        D: A diagram of the curves (computed when omitted).
        backend: 'auto', 'julia' or 'python' (for the diagram).

    Returns:
        A ``FundamentalGroup`` with generators ``x0, x1, ...`` (not simplified).
    """
    D = KD.knot_diagram(curves, backend=backend) if D is None else D
    k = D.n_components
    unders = {c: D.under_positions(c) for c in range(k)}
    arc_base, total = {}, 0
    for c in range(k):
        arc_base[c] = total
        total += max(1, len(unders[c]))

    def arc_containing(comp: int, pos: float) -> int:
        ps = unders[comp]
        if not ps:
            return arc_base[comp]
        j = bisect_right(ps, pos) - 1  # arc j runs from ps[j] to ps[j+1]
        return arc_base[comp] + (j % len(ps))

    relators: List[List[str]] = []
    for cr in D.crossings:
        ps = unders[cr.under]
        i = ps.index(cr.under_pos)
        n = len(ps)
        a = arc_containing(cr.over, cr.over_pos)
        b = arc_base[cr.under] + ((i - 1) % n)   # incoming: the arc ending here
        c = arc_base[cr.under] + (i % n)         # outgoing: the arc starting here
        e = cr.sign
        word = [_tok(c, -1), _tok(a, e), _tok(b, 1), _tok(a, -e)]
        relators.append(word)
    if relators:
        relators = relators[:-1]
    return FundamentalGroup(generators=[f"x{i}" for i in range(total)], relations=relators)


def link_group(curves: Sequence[np.ndarray], simplify: bool = True, backend: str = "auto") -> FundamentalGroup:
    """``pi_1(S^3 - L)``, Tietze-simplified by default.

    Args:
        curves: The polygons of the link.
        simplify: Apply pySurgery's (sound) Tietze simplification.
        backend: 'auto', 'julia' or 'python'.

    Returns:
        The link group.
    """
    G = wirtinger_presentation(curves, backend=backend)
    return simplify_presentation(G.generators, G.relations) if simplify else G


def alexander_check(curves: Sequence, G: FundamentalGroup) -> dict:
    """H_1 of the presentation must be Z^(number of components) -- Alexander duality.

    The end-to-end self-check of a presentation: the diagram, the arc numbering, the
    over/under determination and the relator convention all have to be right.

    Args:
        curves: The polygons.
        G: A presentation of their link group.

    Returns:
        ``{"rank", "torsion", "expected", "ok"}``.
    """
    rank, torsion = G._abelianization_free_rank_torsion()
    return {"rank": int(rank), "torsion": [int(t) for t in torsion], "expected": len(curves),
            "ok": bool(rank == len(curves) and not torsion)}


# --------------------------------------------------------------------------- #
# counting homomorphisms into S_n -- the decidable window
# --------------------------------------------------------------------------- #


class HomCount(BaseModel):
    """Result of ``count_homomorphisms``.

    Attributes:
        count (int): ``|Hom(G, S_n)|`` (a lower bound only when ``exact`` is False).
        exact (bool): False when the search hit its budget; such a count must never be
            compared for inequality.
        n (int): The symmetric group S_n.
        generators (int): Number of generators of the presentation.
        tried (int): Assignments examined.
    """

    count: int
    exact: bool
    n: int
    generators: int
    tried: int


def _words(G: FundamentalGroup) -> Tuple[List[str], List[List[Tuple[int, int]]]]:
    from ..algebra.exact_algebra import normalize_word_token

    gens = list(G.generators)
    idx = {g: i for i, g in enumerate(gens)}
    words = []
    for r in G.relations:
        w = []
        for t in r:
            t = normalize_word_token(t)
            if t.endswith("^-1"):
                w.append((idx[t[:-3]], -1))
            else:
                w.append((idx[t], 1))
        words.append(w)
    return gens, words


def _search_order(k: int, relators: List[List[Tuple[int, int]]]):
    order, seen = [], set()
    for r in sorted(relators, key=len):  # stable: ties keep presentation order
        for g, _e in r:
            if g not in seen:
                seen.add(g)
                order.append(g)
    constrained = list(order)
    free = [g for g in range(k) if g not in seen]
    pos = {g: i for i, g in enumerate(constrained)}
    relators_at: List[List[int]] = [[] for _ in constrained]
    for ri, r in enumerate(relators):
        gens = {g for g, _e in r}
        if gens:
            relators_at[max(pos[g] for g in gens)].append(ri)
    return constrained, free, relators_at


def _count_python(n, k, relators, budget):
    perms = [tuple(p) for p in itertools.permutations(range(n))]
    ident = tuple(range(n))
    constrained, free, relators_at = _search_order(k, relators)
    inv = {p: tuple(sorted(range(n), key=lambda i: p[i])) for p in perms}
    assign: Dict[int, tuple] = {}
    state = {"count": 0, "tried": 0, "exact": True}

    def holds(r) -> bool:
        cur = ident
        for g, e in r:
            q = assign[g] if e > 0 else inv[assign[g]]
            cur = tuple(cur[q[i]] for i in range(n))
        return cur == ident

    def rec(i: int) -> None:
        if not state["exact"]:
            return
        if i == len(constrained):
            state["count"] += 1
            return
        g = constrained[i]
        checks = [relators[ri] for ri in relators_at[i]]
        for p in perms:
            state["tried"] += 1
            if budget is not None and state["tried"] > budget:
                state["exact"] = False
                return
            assign[g] = p
            if all(holds(r) for r in checks):
                rec(i + 1)
            if not state["exact"]:
                return

    rec(0)
    return state["count"] * math.factorial(n) ** len(free), state["exact"], state["tried"]


def count_homomorphisms(
    G: FundamentalGroup, n: int = 3, budget: Optional[int] = None, backend: str = "auto"
) -> HomCount:
    """``|Hom(G, S_n)|`` for the group presented by G, by exact backtracking.

    What is Being Computed?:
        The number of homomorphisms from G to the symmetric group S_n: assignments of a
        permutation to every generator under which every relator evaluates to the
        identity. A group invariant (independent of the presentation), computable
        exactly although triviality and isomorphism of finitely presented groups are
        not: ``count > 1`` proves G nontrivial, and different counts prove two groups
        non-isomorphic.

    Algorithm:
        Generators are assigned in the order they first appear in the relators (shortest
        relators first); each relator is evaluated as soon as its last generator is
        assigned. Generators in no relator contribute an exact factor ``n!`` each. The
        Julia backend runs the same search, in the same order, compiled.

    Args:
        G: A finitely presented group (e.g. ``SimplicialComplex.fundamental_group()``).
        n: The symmetric group S_n.
        budget: Stop after this many assignments; the count is then only a lower bound
            and ``exact=False`` (a truncated count must never be compared for inequality:
            it would fabricate certificates). ``None`` runs to completion.
        backend: 'auto', 'julia' or 'python'.

    Returns:
        A ``HomCount``.

    Example:
        >>> count_homomorphisms(link_group([trefoil]), 3).count   # 3! = 6 for the unknot
        24
    """
    n = int(n)
    if n < 1:
        raise ValueError("n must be >= 1")
    gens, relators = _words(G)
    k = len(gens)
    if k == 0:
        return HomCount(count=1, exact=True, n=n, generators=0, tried=0)
    b = str(backend).lower().strip()
    use_julia = b == "julia" or (b == "auto" and julia_engine.available)
    res = None
    if use_julia:
        try:
            res = julia_engine.count_homomorphisms(n, k, relators, budget)
        except Exception as e:  # pragma: no cover - depends on the Julia runtime
            if b == "julia":
                raise
            warnings.warn(f"Julia homomorphism count failed ({e!r}); falling back to Python.")
    if res is None:
        res = _count_python(n, k, relators, budget)
    count, exact, tried = res
    return HomCount(count=int(count), exact=bool(exact), n=n, generators=k, tried=int(tried))


def free_group_hom_count(n: int, rank: int) -> int:
    """``|Hom(F_rank, S_n)| = (n!)^rank``."""
    return math.factorial(n) ** rank


# --------------------------------------------------------------------------- #
# certificates
# --------------------------------------------------------------------------- #


class SplitnessCertificate(BaseModel):
    """A splitness / knottedness verdict, with the method that proves it.

    Attributes:
        verdict (str): ``"split"``, ``"non-split"``, ``"knotted"``, ``"unknotted"`` or
            ``"undetermined"``.
        method (str): The certificate.
        detail (dict): The data behind it.
        notes (list[str]): Human-readable explanation.
    """

    verdict: str
    method: str
    detail: dict = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.verdict.upper()} -- {self.method}" + "".join("\n  " + n for n in self.notes)


def separating_plane(A: np.ndarray, B: np.ndarray) -> dict:
    """A hyperplane with A strictly on one side and B on the other, found by LP and VERIFIED.

    The LP's tolerances are not a proof, so the returned plane ``(w, b)`` is re-evaluated
    on every point and counts only if ``min_A (w.a - b)`` and ``-max_B (w.x - b)`` both
    clear a margin far above the rounding of that evaluation. For polygons, separating
    the vertex sets separates the polygons (each lies in the convex hull of its
    vertices).

    Args:
        A: Points of the first set.
        B: Points of the second set.

    Returns:
        ``{"separable", "margin", "normal", "offset"}``.
    """
    from scipy.optimize import linprog

    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    d = A.shape[1]
    ub = np.vstack([np.hstack([-A, np.ones((len(A), 1)), np.ones((len(A), 1))]),
                    np.hstack([B, -np.ones((len(B), 1)), np.ones((len(B), 1))])])
    c = np.zeros(d + 2)
    c[-1] = -1.0
    res = linprog(c, A_ub=ub, b_ub=np.zeros(len(A) + len(B)),
                  bounds=[(-1, 1)] * d + [(None, None), (None, 1)], method="highs")
    if not res.success:
        return {"separable": False, "margin": float("-inf")}
    w, b = res.x[:d], float(res.x[d])
    sA, sB = A @ w - b, B @ w - b
    gap = float(min(sA.min(), -sB.max()))
    scale = float(np.abs(w).sum() * max(np.abs(A).max(), np.abs(B).max()) + abs(b))
    return {"separable": bool(gap > 1e-9 * max(scale, 1e-300)), "margin": gap,
            "normal": w, "offset": b}


def _bipartitions(k: int):
    out = []
    for r in range(1, k // 2 + 1):
        for S in itertools.combinations(range(k), r):
            T = tuple(i for i in range(k) if i not in S)
            if (T, S) not in out:
                out.append((S, T))
    return out


def certify_splitness(
    curves: Sequence[np.ndarray], n_values=(3, 4), budget: Optional[int] = None,
    backend: str = "auto",
) -> SplitnessCertificate:
    """Split, non-split, or honestly undetermined (see the module docstring).

    Args:
        curves: The polygons of the link (at least two).
        n_values: Symmetric groups to count homomorphisms into.
        budget: Optional search budget per count (a truncated count is never used).
        backend: 'auto', 'julia' or 'python'.

    Returns:
        A ``SplitnessCertificate``.
    """
    arr = [np.asarray(c, dtype=np.float64) for c in curves]
    k = len(arr)
    if k < 2:
        return SplitnessCertificate(verdict="undetermined", method="splitness needs >= 2 components")
    bip = _bipartitions(k)
    for S, T in bip:
        sep = separating_plane(np.vstack([arr[i] for i in S]), np.vstack([arr[i] for i in T]))
        if sep["separable"]:
            return SplitnessCertificate(
                verdict="split", method="a separating hyperplane",
                detail={"margin": sep["margin"], "partition": (S, T),
                        "normal": list(map(float, sep["normal"])), "offset": sep["offset"]},
                notes=[f"components {list(S)} | {list(T)} lie in disjoint half-spaces "
                       f"(margin {sep['margin']:.4g})"],
            )
    G = link_group(arr, backend=backend)
    dual = alexander_check(arr, G)
    if not dual["ok"]:
        return SplitnessCertificate(
            verdict="undetermined", method="the presentation failed its Alexander-duality check",
            detail={"alexander": dual},
            notes=[f"H_1 of the presentation is Z^{dual['rank']}"
                   + (f" + torsion {dual['torsion']}" if dual["torsion"] else "")
                   + f", not Z^{k}: this presentation is not of the link group, and no "
                     f"certificate built on it would mean anything. Report the configuration."],
        )
    if len(G.relations) == 0 and len(G.generators) == k:
        return SplitnessCertificate(
            verdict="split", method=f"pi_1 reduced to the free group of rank {k}",
            detail={"generators": k, "relators": 0, "alexander": dual},
            notes=["Tietze reduced the Wirtinger presentation to a free group of rank k with no "
                   "relators; a link group is free of rank k exactly for the k-component unlink, "
                   "so this is a proof (of SPLIT, and more: trivial)."],
        )
    notes: List[str] = []
    survivors: Dict[int, list] = {}
    for n in n_values:
        whole = count_homomorphisms(G, n, budget, backend=backend)
        if not whole.exact:
            notes.append(f"the S_{n} count hit its budget; skipped")
            continue
        matched = []
        for S, T in bip:
            a = count_homomorphisms(link_group([arr[i] for i in S], backend=backend), n, budget, backend=backend)
            b = count_homomorphisms(link_group([arr[i] for i in T], backend=backend), n, budget, backend=backend)
            if not (a.exact and b.exact) or whole.count == a.count * b.count:
                matched.append((S, T))
        survivors[n] = matched
        if not matched:
            return SplitnessCertificate(
                verdict="non-split", method=f"|Hom(-, S_{n})| rules out every bipartition",
                detail={"whole": whole.count, "n": n, "alexander": dual, "bipartitions": bip},
                notes=notes + [f"|Hom(pi_1(S^3 - L), S_{n})| = {whole.count} differs from the "
                               f"free-product count of all {len(bip)} bipartition(s); every "
                               f"count is exact, so this is a proof."],
            )
    return SplitnessCertificate(
        verdict="undetermined", method="no certificate found",
        detail={"alexander": dual, "surviving_bipartitions": survivors},
        notes=notes + ["no separating plane, and at every n tried some bipartition's "
                       "free-product count matched. That is not evidence of splitness."],
    )


def certify_knottedness(
    curve: np.ndarray, n_values=(3, 4), budget: Optional[int] = None, backend: str = "auto",
) -> SplitnessCertificate:
    """One closed polygon: ``"knotted"``, ``"unknotted"`` or ``"undetermined"``.

    UNKNOTTED when Tietze reduces pi_1 to ``<x | > = Z`` (a knot group is Z exactly for
    the unknot, by Dehn's lemma); KNOTTED when ``|Hom(pi_1, S_n)| != n!``. Otherwise
    undetermined: a matching count proves nothing.

    Args:
        curve: ``(n, 3)`` polygon vertices.
        n_values: Symmetric groups to count homomorphisms into.
        budget: Optional search budget per count.
        backend: 'auto', 'julia' or 'python'.

    Returns:
        A ``SplitnessCertificate``.
    """
    G = link_group([curve], backend=backend)
    dual = alexander_check([curve], G)
    if not dual["ok"]:
        return SplitnessCertificate(verdict="undetermined",
                                    method="the presentation failed its Alexander-duality check",
                                    detail={"alexander": dual})
    if len(G.generators) == 1 and not G.relations:
        return SplitnessCertificate(
            verdict="unknotted", method="pi_1 reduced to Z", detail={"generators": 1},
            notes=["Tietze reduced pi_1 to <x | > = Z; a knot group is Z exactly for the "
                   "unknot (Dehn's lemma)."],
        )
    for n in n_values:
        got = count_homomorphisms(G, n, budget, backend=backend)
        if got.exact and got.count != math.factorial(n):
            return SplitnessCertificate(
                verdict="knotted", method=f"|Hom(-, S_{n})| separates pi_1 from Z",
                detail={"whole": got.count, "unknot": math.factorial(n), "n": n},
                notes=[f"|Hom(pi_1, S_{n})| = {got.count} but Z gives {math.factorial(n)}."],
            )
    return SplitnessCertificate(verdict="undetermined", method="no certificate found",
                                notes=["the S_n counts matched Z; that does not prove unknotted"])
