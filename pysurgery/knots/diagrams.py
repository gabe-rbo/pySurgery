r"""Certified knot and link diagrams of closed polygons, and the invariants read off them.

Overview:
    A closed polygon in R^3 has a well-defined knot (or link) type, and every invariant
    here is computed EXACTLY for that polygon -- an integer, not an estimate -- from ONE
    projection whose genericity has been certified:

        knot_diagram(curves)           every crossing of a certified generic projection
        diagram_linking_number(A, B)   sum of the signs of the crossings where A passes
                                       over B
        diagram_linking_matrix(curves) all pairwise linking numbers from one diagram
        writhe(D, c)                   sum of the signs of a component's self-crossings
        casson_a2(curve)               the degree-2 Vassiliev invariant (Casson; the z^2
                                       coefficient of the Conway polynomial), by the
                                       Polyak-Viro arrow-diagram formula
        diagram_milnor_mu123(A, B, C)  Milnor's triple linking number of a pairwise
                                       unlinked triple, by the Magnus expansion
        diagram_milnor_mu(curves, I)   Milnor's mu-bar(I) for any multi-index, by
                                       Milnor's algorithm (e.g. (0, 0, 1, 1): the
                                       Sato-Levine invariant, up to sign)

    It complements the complex-based knot invariants of ``pysurgery.knots.invariants``
    (Seifert matrices of knots inside a triangulated S^3): here the input is just the
    vertex coordinates of the polygons.

Key Concepts:
    - **Certified genericity, not voting.** An invariant takes the same value on every
      generic projection, so one generic projection is enough -- IF it is generic. A
      direction is accepted only after checking, for every pair of segments, that no
      crossing lies within ``tol`` (relative) of a segment endpoint (a vertex over an
      edge is the event that makes a crossing appear or vanish), no two segments overlap
      collinearly, no two crossings coincide (no triple points), and the two strands at
      every crossing are separated in depth -- if they are not, the polygons themselves
      (nearly) intersect, the knot type is undefined, and ``NonGenericConfigurationError``
      is raised. Directions come from a fixed deterministic sequence.
    - **Conventions.** The viewer sits at +u with (e1, e2, u) right-handed; the over
      strand has the larger u-coordinate. A crossing is POSITIVE (right-handed) when
      ``cross2d(t_over, t_under) > 0`` -- the convention under which linking numbers
      agree with the Seifert-surface definition, with
      ``geometric_linking.curve_linking`` and with the Gauss integral. Positions along a
      component are ``segment index + fraction``.

Common Workflows:
    1. **Is this polygon knotted?** -> ``casson_a2(curve)`` (nonzero => knotted), or the
       certificates of ``pysurgery.knots.link_complement``.
    2. **Borromean-type linking** -> ``diagram_milnor_mu123(A, B, C)``.

Coefficient Ring:
    Z.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..bridge.julia_bridge import julia_engine
from ..core.exceptions import NonGenericConfigurationError, UndefinedInvariantError

__all__ = [
    "Crossing",
    "KnotDiagram",
    "projection_frames",
    "knot_diagram",
    "diagram_linking_number",
    "diagram_linking_matrix",
    "writhe",
    "a2_from_diagram",
    "casson_a2",
    "diagram_milnor_mu123",
    "diagram_milnor_mu",
]


class Crossing(BaseModel):
    """One crossing of a knot diagram.

    Attributes:
        over (int): Component of the over strand.
        over_pos (float): Parameter along it (segment index + fraction).
        under (int): Component of the under strand.
        under_pos (float): Parameter along it.
        sign (int): +1 right-handed, -1 left-handed.
        point (tuple[float, float]): Position in the projection plane.
        depth_gap (float): Separation of the strands along the viewing direction.
    """

    model_config = ConfigDict(frozen=True)

    over: int
    over_pos: float
    under: int
    under_pos: float
    sign: int
    point: Tuple[float, float]
    depth_gap: float


class KnotDiagram(BaseModel):
    """All crossings of closed polygons in one certified generic projection.

    Attributes:
        crossings (list[Crossing]): The crossings.
        n_components (int): Number of polygons.
        n_vertices (list[int]): Vertices per polygon.
        frame (tuple): The projection frame ``(e1, e2, u)``.
        tries (int): Number of directions examined.
        rejected (list[str]): Why earlier directions were rejected.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    crossings: List[Crossing]
    n_components: int
    n_vertices: List[int]
    frame: Tuple[np.ndarray, np.ndarray, np.ndarray]
    tries: int
    rejected: List[str] = Field(default_factory=list)

    def between(self, i: int, j: int) -> List[Crossing]:
        """Crossings between components i and j (either over)."""
        return [c for c in self.crossings if {c.over, c.under} == {i, j} and i != j]

    def self_crossings(self, i: int) -> List[Crossing]:
        """Self-crossings of component i."""
        return [c for c in self.crossings if c.over == i and c.under == i]

    def under_positions(self, comp: int) -> List[float]:
        """Sorted positions where component ``comp`` passes under something."""
        return sorted(c.under_pos for c in self.crossings if c.under == comp)


# --------------------------------------------------------------------------- #
# projection directions
# --------------------------------------------------------------------------- #


def _frame(u: np.ndarray):
    u = np.asarray(u, dtype=np.float64)
    u = u / np.linalg.norm(u)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(u, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)  # e1 x e2 = u
    return e1, e2, u


def projection_frames(n: int = 64):
    """A fixed, deterministic sequence of projection frames ``(e1, e2, u)``.

    Irrational directions first (the coordinate axes are exactly the ones hand-built
    ground truths are degenerate along), a fixed-seed generator after that.

    Args:
        n: Number of frames.

    Returns:
        The frames.
    """
    out = []
    golden = (1 + 5 ** 0.5) / 2
    for i in range(min(n, 16)):
        z = 1 - (2 * i + 1) / 32.0
        rho = (1 - z * z) ** 0.5
        th = 2 * np.pi * i / golden + 0.1234567
        out.append(_frame(np.array([rho * np.cos(th), rho * np.sin(th), z + 1e-3 * (i + 1) / 7])))
    rng = np.random.default_rng(20260921)
    while len(out) < n:
        out.append(_frame(rng.normal(size=3)))
    return out


# --------------------------------------------------------------------------- #
# crossings
# --------------------------------------------------------------------------- #


def _cross2d(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _segments(curves):
    starts, ends, comp, local, ncomp = [], [], [], [], []
    for c, C in enumerate(curves):
        n = len(C)
        if n < 3:
            raise ValueError(f"component {c} has {n} vertices; a closed polygon needs >= 3")
        starts.append(C)
        ends.append(np.roll(C, -1, axis=0))
        comp.append(np.full(n, c))
        local.append(np.arange(n))
        ncomp.append(np.full(n, n))
    return (np.vstack(starts), np.vstack(ends), np.concatenate(comp),
            np.concatenate(local), np.concatenate(ncomp))


def _crossing_records_python(S, E, comp, local, ncomp, frame, tol, scale, chunk=512):
    """Raw crossing data of all segment pairs in one projection.

    The status is 0 generic, 1 collinear overlap, 2 crossing at an endpoint, 3 the
    polygons (nearly) meet (then the closest meeting pair is returned).
    """
    e1, e2, u = frame
    P1 = np.stack([S @ e1, S @ e2], 1)
    P2 = np.stack([E @ e1, E @ e2], 1)
    D1, D2 = S @ u, E @ u
    R = P2 - P1
    rnorm = np.linalg.norm(R, axis=1)
    N = len(S)
    rows = []
    collinear = endpoint = False
    meet = None
    for lo in range(0, N, chunk):
        hi = min(lo + chunk, N)
        II = np.arange(lo, hi)[:, None]
        JJ = np.arange(N)[None, :]
        same = comp[II] == comp[JJ]
        adjacent = same & ((np.abs(local[II] - local[JJ]) == 1) |
                           (np.abs(local[II] - local[JJ]) == ncomp[II] - 1))
        ii, jj = np.nonzero((JJ > II) & ~adjacent)
        if ii.size == 0:
            continue
        ii = ii + lo
        r, s = R[ii], R[jj]
        qp = P1[jj] - P1[ii]
        rxs = _cross2d(r, s)
        par = np.abs(rxs) <= 1e-12 * rnorm[ii] * rnorm[jj]
        if par.any():
            k = np.nonzero(par)[0]
            off = np.abs(_cross2d(qp[k], r[k])) / np.maximum(rnorm[ii[k]], 1e-300)
            col = off <= tol * scale
            if col.any():
                kk = k[col]
                rr = (r[kk] ** 2).sum(1)
                a0 = (qp[kk] * r[kk]).sum(1) / rr
                a1 = ((qp[kk] + s[kk]) * r[kk]).sum(1) / rr
                lo_, hi_ = np.minimum(a0, a1), np.maximum(a0, a1)
                if np.any((hi_ >= -tol) & (lo_ <= 1 + tol)):
                    collinear = True
        with np.errstate(divide="ignore", invalid="ignore"):
            t = _cross2d(qp, s) / rxs
            w = _cross2d(qp, r) / rxs
        near = (~par) & (t > -tol) & (t < 1 + tol) & (w > -tol) & (w < 1 + tol)
        if not near.any():
            continue
        tt, ww = t[near], w[near]
        if np.any((tt < tol) | (tt > 1 - tol) | (ww < tol) | (ww > 1 - tol)):
            endpoint = True
            continue
        a, b = ii[near], jj[near]
        da = D1[a] + tt * (D2[a] - D1[a])
        db = D1[b] + ww * (D2[b] - D1[b])
        gap = np.abs(da - db)
        if np.any(gap <= tol * scale) and meet is None:
            k = int(np.argmin(gap))
            meet = (int(a[k]), int(b[k]), float(gap[k]))
        for k in range(len(a)):
            rows.append((int(a[k]), int(b[k]), float(tt[k]), float(ww[k]), float(da[k]), float(db[k])))
    status = 1 if collinear else 2 if endpoint else 3 if meet is not None else 0
    return rows, status, meet


def _build_crossings(rows, S, E, comp, local, frame):
    e1, e2, _ = frame
    P1 = np.stack([S @ e1, S @ e2], 1)
    R = np.stack([E @ e1, E @ e2], 1) - P1
    recs = []
    for a, b, tt, ww, da, db in rows:
        if da > db:
            o, un, po, pu, to, tu = a, b, tt, ww, R[a], R[b]
        else:
            o, un, po, pu, to, tu = b, a, ww, tt, R[b], R[a]
        pt = P1[a] + tt * R[a]
        recs.append(Crossing(
            over=int(comp[o]), over_pos=float(local[o] + po), under=int(comp[un]),
            under_pos=float(local[un] + pu), sign=int(np.sign(_cross2d(to, tu))),
            point=(float(pt[0]), float(pt[1])), depth_gap=float(abs(da - db)),
        ))
    return recs


def knot_diagram(
    curves: Sequence[np.ndarray],
    tol: float = 1e-9,
    max_tries: int = 64,
    frame=None,
    backend: str = "auto",
) -> KnotDiagram:
    """All crossings of closed polygons in the first CERTIFIED generic projection.

    What is Being Computed?:
        The diagram of the link formed by the polygons: every crossing with its over and
        under strands, positions and sign, in the first direction of a fixed sequence
        that passes every genericity test (module docstring).

    Algorithm:
        For each candidate frame, test every non-adjacent pair of segments (Julia: in
        parallel threads) for collinear overlap, endpoint crossings and depth
        separation, then reject triple points; the first frame with none is used.

    Args:
        curves: ``(n_i, 3)`` arrays of polygon vertices in R^3 (closed implicitly).
        tol: Genericity margin, relative to the size of the configuration.
        max_tries: Number of candidate frames.
        frame: Force one frame ``(e1, e2, u)`` (still certified).
        backend: 'auto', 'julia' or 'python'.

    Returns:
        A ``KnotDiagram``.

    Raises:
        NonGenericConfigurationError: If the polygons (nearly) intersect, or no generic
            direction is found (which for disjoint polygons does not happen).
        ValueError: If an input is not an ``(n, 3)`` polygon with ``n >= 3``.
    """
    cs = [np.asarray(c, dtype=np.float64) for c in curves]
    if any(c.ndim != 2 or c.shape[1] != 3 for c in cs):
        raise ValueError("curves must be (n_i, 3) arrays of polygon vertices in R^3")
    S, E, comp, local, ncomp = _segments(cs)
    allpts = np.vstack(cs)
    scale = float(np.linalg.norm(allpts.max(0) - allpts.min(0))) or 1.0
    frames = [frame] if frame is not None else projection_frames(max_tries)
    b = str(backend).lower().strip()
    use_julia = b == "julia" or (b == "auto" and julia_engine.available)
    reasons: List[str] = []
    for k, fr in enumerate(frames):
        res = None
        if use_julia:
            try:
                res = julia_engine.diagram_crossings(S, E, comp, local, ncomp, fr, tol, scale)
            except Exception as e:  # pragma: no cover - depends on the Julia runtime
                if b == "julia":
                    raise
                warnings.warn(f"Julia crossing detection failed ({e!r}); falling back to Python.")
                use_julia = False
        if res is None:
            res = _crossing_records_python(S, E, comp, local, ncomp, fr, tol, scale)
        rows, status, meet = res
        if status == 1:
            reasons.append("two segments overlap collinearly in the projection")
            continue
        if status == 2:
            reasons.append("a crossing lies at a segment endpoint (a vertex projects onto an edge)")
            continue
        if status == 3:
            a, bb, gap = meet
            raise NonGenericConfigurationError(
                f"the polygons (nearly) intersect in R^3: segments {int(comp[a])}:{int(local[a])} "
                f"and {int(comp[bb])}:{int(local[bb])} are {gap:.3g} apart in depth where they "
                f"cross. The knot/link type is undefined there, in every projection."
            )
        recs = _build_crossings(rows, S, E, comp, local, fr)
        if len(recs) > 1:
            from scipy.spatial import cKDTree

            pts = np.array([c.point for c in recs])
            if cKDTree(pts).query_pairs(r=tol * scale):
                reasons.append("two crossings coincide in the plane (a triple point)")
                continue
        return KnotDiagram(crossings=recs, n_components=len(cs), n_vertices=[len(c) for c in cs],
                           frame=tuple(np.asarray(x) for x in fr), tries=k + 1, rejected=reasons)
    raise NonGenericConfigurationError(
        f"no generic projection among {len(frames)} directions: {sorted(set(reasons))}"
    )


# --------------------------------------------------------------------------- #
# invariants
# --------------------------------------------------------------------------- #


def _pair_sum(D: KnotDiagram, i: int, j: int) -> int:
    a = sum(c.sign for c in D.crossings if c.over == i and c.under == j)
    b = sum(c.sign for c in D.crossings if c.over == j and c.under == i)
    if a != b:
        raise NonGenericConfigurationError(
            f"inconsistent diagram between components {i} and {j}: {a} over-crossings "
            f"vs {b} under-crossings"
        )
    return int(a)


def diagram_linking_number(A: np.ndarray, B: np.ndarray, D: Optional[KnotDiagram] = None,
                           backend: str = "auto") -> int:
    """Exact linking number of two disjoint closed polygons from one diagram.

    The sum of the signs of the crossings where A passes OVER B; it must equal the sum
    where B passes over A (checked, as a certificate of the diagram).

    Args:
        A: First polygon.
        B: Second polygon.
        D: A diagram of ``[A, B]`` (computed when omitted).
        backend: 'auto', 'julia' or 'python'.

    Returns:
        The linking number.
    """
    D = knot_diagram([A, B], backend=backend) if D is None else D
    return _pair_sum(D, 0, 1)


def diagram_linking_matrix(curves: Sequence[np.ndarray], backend: str = "auto") -> np.ndarray:
    """Pairwise exact linking numbers of a link's components, from one diagram.

    Args:
        curves: The polygons.
        backend: 'auto', 'julia' or 'python'.

    Returns:
        A symmetric integer matrix with zero diagonal.
    """
    D = knot_diagram(curves, backend=backend)
    k = len(curves)
    L = np.zeros((k, k), dtype=np.int64)
    for i in range(k):
        for j in range(i + 1, k):
            L[i, j] = L[j, i] = _pair_sum(D, i, j)
    return L


def writhe(D: KnotDiagram, component: int = 0) -> int:
    """Sum of the signs of a component's self-crossings (depends on the projection).

    Args:
        D: A diagram.
        component: The component.

    Returns:
        The writhe of that component in this diagram.
    """
    return int(sum(c.sign for c in D.self_crossings(component)))


def a2_from_diagram(D: KnotDiagram, component: int = 0, basepoint: Optional[float] = None) -> int:
    """The Polyak-Viro formula for the Casson invariant, on one component.

    What is Being Computed?:
        Cut the component at a basepoint that is not a crossing passage and rank the
        passages from there. Sum ``sign(a) sign(b)`` over pairs of self-crossings whose
        four passages read ``over(a), under(b), under(a), over(b)`` -- interleaved chords
        with that one arrow configuration (Polyak-Viro, *Gauss diagram formulas for
        Vassiliev invariants*, IMRN 1994). The basepoint does not matter; by default it is
        the start of segment 0, which a certified diagram never has a passage at.

    Args:
        D: A diagram.
        component: The component.
        basepoint: Optional basepoint parameter.

    Returns:
        a2 of that component.
    """
    n = D.n_vertices[component]
    cr = D.self_crossings(component)
    base = 0.0 if basepoint is None else float(basepoint)

    def rank(p):
        return (p - base) % n

    total = 0
    for i in range(len(cr)):
        ri = (rank(cr[i].over_pos), rank(cr[i].under_pos))
        for j in range(i + 1, len(cr)):
            rj = (rank(cr[j].over_pos), rank(cr[j].under_pos))
            a1, a2_ = sorted(ri)
            b1, b2 = sorted(rj)
            if not ((a1 < b1 < a2_ < b2) or (b1 < a1 < b2 < a2_)):
                continue
            X, Y, sX, sY = (ri, rj, cr[i].sign, cr[j].sign) if min(ri) < min(rj) \
                else (rj, ri, cr[j].sign, cr[i].sign)
            if X[0] < X[1] and Y[0] > Y[1]:  # over(X) under(Y) under(X) over(Y)
                total += sX * sY
    return int(total)


def casson_a2(curve: np.ndarray, backend: str = "auto") -> int:
    """The Casson invariant a2 of a closed polygon in R^3, exactly.

    a2 is the z^2 coefficient of the Conway polynomial: 0 on the unknot, 1 on the
    trefoil (either chirality -- it is mirror-blind), -1 on the figure-eight,
    ``(p^2 - 1)(q^2 - 1)/24`` on the (p, q) torus knot. A nonzero value certifies the
    polygon is knotted.

    Args:
        curve: ``(n, 3)`` polygon vertices.
        backend: 'auto', 'julia' or 'python'.

    Returns:
        a2.
    """
    return a2_from_diagram(knot_diagram([curve], backend=backend), 0)


def diagram_milnor_mu123(
    A: np.ndarray, B: np.ndarray, C: np.ndarray, D: Optional[KnotDiagram] = None,
    backend: str = "auto",
) -> int:
    r"""Milnor's triple linking number mu(123) of three disjoint closed polygons.

    What is Being Computed?:
        Defined as an integer exactly when all three pairwise linking numbers vanish;
        they are computed exactly first, and ``UndefinedInvariantError`` is raised
        otherwise (with nonzero pairwise linking the invariant lives only modulo their
        gcd). Method: the Magnus expansion of the longitude of component 3. Read the
        longitude as the word of meridians of the arcs it passes UNDER, left to right,
        each to the power of the crossing sign; mu(123) is the coefficient of
        ``X_1 X_2`` in its expansion (``x_i -> 1 + X_i``). The meridian of an arc of
        component j is the conjugate ``w^-1 x_j w`` by the partial longitude w read
        along component j, and

            w^-1 x_j w = 1 + X_j + (X_j W - W X_j) + O(3),   W = sum_i w^(i) X_i,

        so conjugation contributes at degree 2 -- exactly where mu lives. The direction
        of conjugation is the one consistent with reading the longitude left to right
        (the opposite one is not even an invariant). The overall sign is a convention;
        ``|mu(123)| = 1`` on the Borromean rings is not.

    Args:
        A: Polygon 1.
        B: Polygon 2.
        C: Polygon 3.
        D: A diagram of ``[A, B, C]`` (computed when omitted).
        backend: 'auto', 'julia' or 'python'.

    Returns:
        mu(123).

    Raises:
        UndefinedInvariantError: If a pairwise linking number is nonzero.
    """
    curves = [np.asarray(A, float), np.asarray(B, float), np.asarray(C, float)]
    D = knot_diagram(curves, backend=backend) if D is None else D
    L = {(i, j): _pair_sum(D, i, j) for i in range(3) for j in range(i + 1, 3)}
    if any(L.values()):
        raise UndefinedInvariantError(
            f"mu(123) is an integer invariant only when every pairwise linking number "
            f"vanishes; lk12 = {L[(0, 1)]}, lk13 = {L[(0, 2)]}, lk23 = {L[(1, 2)]}"
        )

    def unders(under_comp, over_comp):
        return sorted((c.under_pos, c.over_pos, c.sign) for c in D.crossings
                      if c.under == under_comp and c.over == over_comp)

    u12 = unders(0, 1)
    u21 = unders(1, 0)

    def conj_exponent(events, pos):
        return sum(sg for p, _, sg in events if p < pos)

    events = []
    for j in (0, 1):
        for pos3, pos_over, sg in unders(2, j):
            w = conj_exponent(u12 if j == 0 else u21, pos_over)
            events.append((pos3, j, sg, w))
    events.sort(key=lambda e: e[0])
    total = 0
    for s, (_, js, es, _) in enumerate(events):
        if js != 0:
            continue
        for _, jt, et, _ in events[s + 1:]:
            if jt == 1:
                total += es * et
    for _, j, eps, w in events:
        total -= eps * (w if j == 1 else -w)
    return int(total)



# --------------------------------------------------------------------------- #
# Milnor invariants of any length, by Milnor's algorithm
# --------------------------------------------------------------------------- #


def _magnus_mul(a: dict, b: dict, degree: int) -> dict:
    """Product in the Magnus algebra Z<<X_0, ..., X_{n-1}>> truncated above ``degree``."""
    out: dict = {}
    for wa, ca in a.items():
        for wb, cb in b.items():
            if len(wa) + len(wb) <= degree:
                w = wa + wb
                out[w] = out.get(w, 0) + ca * cb
    return {w: c for w, c in out.items() if c}


def _magnus_power(a: dict, e: int, degree: int) -> dict:
    """``a^e`` for a series ``a = 1 + A`` (A without constant term), e any integer."""
    one = {(): 1}
    if e < 0:
        A = {w: c for w, c in a.items() if w}
        term, inv = one, dict(one)
        for _ in range(degree):          # (1 + A)^-1 = sum_j (-A)^j
            term = _magnus_mul(term, {w: -c for w, c in A.items()}, degree)
            for w, c in term.items():
                inv[w] = inv.get(w, 0) + c
        a, e = {w: c for w, c in inv.items() if c}, -e
    out = one
    for _ in range(e):
        out = _magnus_mul(out, a, degree)
    return out


def _longitude_expansions(D: KnotDiagram, degree: int) -> List[dict]:
    """Magnus expansions of the 0-framed longitudes of every component, mod degree + 1.

    Wirtinger generators: one meridian per arc (an arc of component c runs between
    consecutive passages of c UNDER something). Reading c from its basepoint, the
    partial longitude w_{c,a} is the product, left to right, of the meridians of the
    over-arcs at its first a under-passages, each to the power of the crossing sign;
    the meridian of arc a of c is ``w_{c,a}^-1 x_c w_{c,a}`` (Milnor, *Isotopy of
    links*, 1957), the convention of ``diagram_milnor_mu123``. Starting from
    ``x_{c,a} = x_c`` and substituting ``degree`` times fixes the expansions of all
    meridians modulo degree + 1, since each substitution is correct one degree
    further. The longitude is the full product times ``x_c^-writhe_c``, which makes
    the exponent sum of x_c zero (the preferred longitude).
    """
    n = D.n_components
    unders: List[List[Crossing]] = [
        sorted((c for c in D.crossings if c.under == j), key=lambda c: c.under_pos) for j in range(n)
    ]
    positions = [[c.under_pos for c in unders[j]] for j in range(n)]

    def arc(j: int, pos: float) -> int:
        m = len(positions[j])
        return sum(1 for p in positions[j] if p < pos) % m if m else 0

    letters = [[(c.over, arc(c.over, c.over_pos), c.sign) for c in unders[j]] for j in range(n)]
    x = [{(): 1, (j,): 1} for j in range(n)]
    meridian = [[x[j]] * max(len(unders[j]), 1) for j in range(n)]

    def partial_longitudes(j: int) -> List[dict]:
        W = [{(): 1}]
        for o, a, s in letters[j]:
            W.append(_magnus_mul(W[-1], _magnus_power(meridian[o][a], s, degree), degree))
        return W

    for _ in range(degree):
        meridian = [
            [
                _magnus_mul(_magnus_mul(_magnus_power(W, -1, degree), x[j], degree), W, degree)
                for W in partial_longitudes(j)[: max(len(unders[j]), 1)]
            ]
            for j in range(n)
        ]
    out = []
    for j in range(n):
        writhe_j = sum(c.sign for c in D.self_crossings(j))
        out.append(_magnus_mul(partial_longitudes(j)[-1], _magnus_power(x[j], -writhe_j, degree), degree))
    return out


def diagram_milnor_mu(
    curves: Sequence[np.ndarray],
    multi_index: Sequence[int],
    D: Optional[KnotDiagram] = None,
    backend: str = "auto",
) -> int:
    r"""Milnor's invariant mu-bar(i_1 ... i_k) of disjoint closed polygons, exactly.

    What is Being Computed?:
        The coefficient of ``X_{i_1} ... X_{i_{k-1}}`` in the Magnus expansion
        (``x_j -> 1 + X_j``) of the 0-framed longitude of component ``i_k``, computed
        by Milnor's algorithm from one certified diagram (``_longitude_expansions``).
        It is an isotopy invariant of the link modulo Delta, the gcd of mu-bar of every
        sequence obtained by deleting at least one index and permuting cyclically
        (Milnor 1957); this function returns it only when Delta = 0 and raises
        otherwise. Length 2 gives the linking number, (i, j, k) the triple linking
        number of ``diagram_milnor_mu123``, and (i, i, j, j) the first invariant beyond
        lk of a two-component link, ``-beta`` for the Sato-Levine invariant beta
        (Cochran, *Derivatives of links*, Mem. AMS 427, 1990): +-1 on the Whitehead
        link. Repeated indices are allowed; the invariants of length >= 3 change sign
        under mirror image exactly when the length is even.

    Args:
        curves: ``(n_i, 3)`` polygons, the link components.
        multi_index: Component indices ``(i_1, ..., i_k)``, 0-based, ``k >= 2``.
        D: A diagram of ``curves`` (computed when omitted).
        backend: 'auto', 'julia' or 'python' (for the diagram).

    Returns:
        mu-bar(i_1 ... i_k).

    Raises:
        UndefinedInvariantError: If Delta != 0, i.e. an invariant obtained by deleting
            indices (and permuting cyclically) is nonzero.
        ValueError: If the multi-index is too short or names a missing component.
    """
    from itertools import combinations
    from math import gcd

    index = tuple(int(i) for i in multi_index)
    if len(index) < 2:
        raise ValueError("a Milnor invariant needs a multi-index of length >= 2")
    if any(i < 0 or i >= len(curves) for i in index):
        raise ValueError(f"multi-index {index} names a component outside 0..{len(curves) - 1}")
    curves = [np.asarray(c, float) for c in curves]
    D = knot_diagram(curves, backend=backend) if D is None else D
    longitudes = _longitude_expansions(D, len(index) - 1)

    def mu(seq: Tuple[int, ...]) -> int:
        return int(longitudes[seq[-1]].get(seq[:-1], 0))

    delta = 0
    for size in range(2, len(index)):
        for kept in combinations(range(len(index)), size):
            seq = tuple(index[i] for i in kept)
            for r in range(size):
                delta = gcd(delta, mu(seq[r:] + seq[:r]))
    if delta:
        raise UndefinedInvariantError(
            f"mu-bar{index} is defined only modulo Delta = {delta}: an invariant obtained by "
            "deleting indices is nonzero"
        )
    return mu(index)
