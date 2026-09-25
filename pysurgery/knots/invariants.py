"""pysurgery/knots/invariants.py.

State-of-the-art knot invariants computed from simplicial complexes.

Invariants implemented:
  - Seifert matrix (minimal-area Seifert surface in the ambient triangulation,
    with positive push-offs through the tetrahedra on its positive side; see
    `pysurgery.knots.seifert_surface`)
  - Alexander polynomial (det(tV - V^T))
  - Conway polynomial (Alexander change of variables)
  - Knot signature (sig(V + V^T))
  - Arf invariant (from Δ(-1) mod 8)
  - Seifert genus bound (half-degree of Alexander)
  - Unknotting number lower bound (|signature|/2)

All invariants have Julia-accelerated paths where beneficial with exact Python fallbacks.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from pysurgery.topology.complexes import SimplicialComplex
from pysurgery.bridge.julia_bridge import julia_engine
from pysurgery.knots.seifert_surface import SeifertSurfaceError, seifert_matrix_of_triangulation


# ── Wirtinger / knot-diagram Alexander polynomial ────────────────────────────


class _DiagramExtractionError(Exception):
    """Raised when a 1-complex cannot be unambiguously read as a polygonal knot."""


def _order_knot_polyline(K: SimplicialComplex) -> List[int]:
    """Return the vertex indices of a closed simple 1-cycle K in cyclic order.

    The complex K must be a single connected closed loop (every vertex has
    degree 2 in the 1-skeleton). Returns vertex indices following an arbitrary
    orientation of the loop.
    """
    edges = [tuple(sorted(e)) for e in K.n_simplices(1)]
    if not edges:
        raise _DiagramExtractionError("K has no 1-simplices")
    adj: Dict[int, List[int]] = {}
    for (a, b) in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    for v, neigh in adj.items():
        if len(neigh) != 2:
            raise _DiagramExtractionError(
                f"K is not a simple closed loop: vertex {v} has degree {len(neigh)}"
            )
    start = min(adj)
    order = [start]
    prev = None
    current = start
    while True:
        nxt = next(n for n in adj[current] if n != prev)
        if nxt == start:
            break
        order.append(nxt)
        prev = current
        current = nxt
        if len(order) > len(adj):
            raise _DiagramExtractionError("Failed to close loop traversal")
    return order


def _knot_polyline_coords(
    ambient_complex: SimplicialComplex, K: SimplicialComplex
) -> Optional[np.ndarray]:
    """Return an (N, 3) array of vertex coordinates along the knot polyline."""
    coords = ambient_complex.simplices_to_point_cloud
    if not coords:
        return None
    order = _order_knot_polyline(K)
    pts: List[np.ndarray] = []
    for v in order:
        key = (v,)
        if key not in coords:
            return None
        p = np.asarray(coords[key][0], dtype=np.float64)
        if p.shape != (3,):
            return None
        pts.append(p)
    return np.asarray(pts, dtype=np.float64)


def _is_generic_projection(pts3: np.ndarray, ex: np.ndarray, ey: np.ndarray, ez: np.ndarray) -> bool:
    """Check that the projection is non-degenerate for a generic knot diagram.

    Requires that no two distinct vertices project to the same (x, y), and that
    `_find_crossings` can resolve every crossing's over/under (no in-segment
    z-coincidence at the crossing point).
    """
    xy = pts3 @ np.column_stack([ex, ey])
    d = xy[:, None, :] - xy[None, :, :]
    np.fill_diagonal(d[..., 0], np.inf)
    np.fill_diagonal(d[..., 1], np.inf)
    if not np.all(np.linalg.norm(d, axis=-1) > 1e-8):
        return False
    # Actually run the crossing detector — it raises on near-degenerate
    # over/under, which is the only genericity property that matters in
    # practice for the Wirtinger pipeline.
    try:
        _find_crossings(pts3, ex, ey, ez)
    except _DiagramExtractionError:
        return False
    return True


def _projection_basis(pts3: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Choose a generic 2D projection basis (e_x, e_y) and a height axis e_z.

    Tries the canonical (x, y, z) axes first — most parametric knot constructors
    are designed so that the xy projection is a clean minimal-crossing diagram.
    Falls back to random orthonormal rotations only if the canonical axes are
    non-generic for this point cloud.
    """
    # First try the canonical basis — knot constructors are designed for this.
    ex = np.array([1.0, 0.0, 0.0])
    ey = np.array([0.0, 1.0, 0.0])
    ez = np.array([0.0, 0.0, 1.0])
    if _is_generic_projection(pts3, ex, ey, ez):
        return ex, ey, ez

    rng = np.random.default_rng(0xC0FFEE)
    for _ in range(64):
        A = rng.normal(size=(3, 3))
        Q, _ = np.linalg.qr(A)
        # e_z = e_x × e_y keeps the frame right-handed; a left-handed frame
        # would read off the mirror diagram and flip every crossing sign.
        ex, ey = Q[:, 0], Q[:, 1]
        ez = np.cross(ex, ey)
        if _is_generic_projection(pts3, ex, ey, ez):
            return ex, ey, ez
    raise _DiagramExtractionError("Could not find generic projection basis")


def _segment_cross_2d(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray,
    eps: float = 1e-9,
) -> Optional[Tuple[float, float]]:
    """Return the interior crossing parameters of two 2D segments, or None.

    If segments (p1, p2) and (p3, p4) cross in their interiors, returns (t, s)
    where the intersection is p1 + t*(p2-p1) = p3 + s*(p4-p3); else None.
    """
    d1 = p2 - p1
    d2 = p4 - p3
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < eps:
        return None
    diff = p3 - p1
    t = (diff[0] * d2[1] - diff[1] * d2[0]) / denom
    s = (diff[0] * d1[1] - diff[1] * d1[0]) / denom
    if eps < t < 1 - eps and eps < s < 1 - eps:
        return float(t), float(s)
    return None


def _find_crossings(
    pts3: np.ndarray, ex: np.ndarray, ey: np.ndarray, ez: np.ndarray,
) -> List[Dict]:
    """Find polyline-segment crossings under the chosen projection.

    For each transverse crossing between segment i (p_i → p_{i+1}) and segment
    j (p_j → p_{j+1}), record:
        over_seg, under_seg, t_over, t_under (parameters along each segment),
        sign in {+1, -1}.

    Crossing sign convention: the diagram is viewed from +e_z (the frame is
    right-handed) and a crossing is positive (right-handed) iff
    det(over tangent, under tangent) > 0, i.e. the under-strand passes from
    the right of the over-strand to its left.  This is the standard
    convention: half the sum of the inter-component crossing signs of a
    two-component link is its Gauss linking number, and the left-handed
    trefoil has three negative crossings.
    """
    n = pts3.shape[0]
    xy = pts3 @ np.column_stack([ex, ey])
    z = pts3 @ ez
    crossings: List[Dict] = []
    for i in range(n):
        p1 = xy[i]
        p2 = xy[(i + 1) % n]
        z1 = z[i]
        z2 = z[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # adjacent segments share an endpoint
            p3 = xy[j]
            p4 = xy[(j + 1) % n]
            z3 = z[j]
            z4 = z[(j + 1) % n]
            cross = _segment_cross_2d(p1, p2, p3, p4)
            if cross is None:
                continue
            t, s = cross
            zi_at = (1 - t) * z1 + t * z2
            zj_at = (1 - s) * z3 + s * z4
            if abs(zi_at - zj_at) < 1e-9:
                raise _DiagramExtractionError(
                    "Projection has near-degenerate over/under at a crossing"
                )
            if zi_at > zj_at:
                over_seg, under_seg = i, j
                t_over, t_under = t, s
                over_tan = p2 - p1
                under_tan = p4 - p3
            else:
                over_seg, under_seg = j, i
                t_over, t_under = s, t
                over_tan = p4 - p3
                under_tan = p2 - p1
            # Right-handed (positive) crossing: det(over, under) > 0.
            det = over_tan[0] * under_tan[1] - over_tan[1] * under_tan[0]
            sign = 1 if det > 0 else -1
            crossings.append({
                "over_seg": int(over_seg),
                "under_seg": int(under_seg),
                "t_over": float(t_over),
                "t_under": float(t_under),
                "sign": int(sign),
            })
    return crossings


def _assign_arcs(n_segs: int, crossings: List[Dict]) -> Tuple[List[List[int]], List[Dict]]:
    """Partition the polyline into arcs split at under-crossings.

    An arc is a maximal run of the polyline not interrupted by an under-crossing.

    Returns:
        arcs:        list of segment-id lists; arcs[k] is the list of polyline
                     segment indices covered (partially or wholly) by arc k.
                     This is informational; the precise arc boundary points are
                     captured in the per-segment under-crossing list below.
        crossings:   the same crossings list, augmented with
                     'over_arc', 'in_arc' (under-incoming), 'out_arc' (under-outgoing).
    """
    # Collect under-crossings per segment, ordered by t_under
    under_per_seg: Dict[int, List[int]] = {i: [] for i in range(n_segs)}
    for k, c in enumerate(crossings):
        under_per_seg[c["under_seg"]].append(k)
    for i in range(n_segs):
        under_per_seg[i].sort(key=lambda k: crossings[k]["t_under"])

    # Walk segments cyclically, breaking arcs at under-crossings.
    # To produce a clean cyclic structure, start at segment 0's leading edge.
    # If segment 0 begins right after an under-crossing of the previous segment,
    # we still start arc 0 here; we'll later identify the in-arc/out-arc of each
    # crossing by traversing this assignment.
    seg_pre_arc: List[int] = [0] * n_segs   # arc at start of segment i
    seg_post_arcs_by_under: List[List[int]] = [[] for _ in range(n_segs)]
    arc_counter = 0
    current_arc = 0
    for i in range(n_segs):
        seg_pre_arc[i] = current_arc
        for ki in under_per_seg[i]:
            arc_counter += 1
            current_arc = arc_counter
            seg_post_arcs_by_under[i].append(current_arc)
    # The last "current_arc" must merge with arc 0 to close the cycle.
    # Relabel arc_counter+? to 0 throughout.
    final_arc = current_arc
    n_arcs = arc_counter + 1  # arcs 0..arc_counter; but final_arc == arc 0 cyclically
    # Build remapping
    remap = {a: a for a in range(n_arcs)}
    if final_arc != 0:
        # Merge final_arc with 0
        remap[final_arc] = 0
        # Apply transitively (only one merge needed since arc IDs are linear)
    def R(a: int) -> int:
        seen = []
        while remap[a] != a and a not in seen:
            seen.append(a)
            a = remap[a]
        return a
    for i in range(n_segs):
        seg_pre_arc[i] = R(seg_pre_arc[i])
        seg_post_arcs_by_under[i] = [R(a) for a in seg_post_arcs_by_under[i]]

    # Now assign per-crossing in_arc, out_arc, over_arc
    for ki, c in enumerate(crossings):
        seg = c["under_seg"]
        # Find which under-crossing index of `seg` this is
        order_list = under_per_seg[seg]
        local_idx = order_list.index(ki)
        # in_arc = pre-arc if local_idx == 0 else post-arc of previous under-crossing
        if local_idx == 0:
            c["in_arc"] = seg_pre_arc[seg]
        else:
            c["in_arc"] = seg_post_arcs_by_under[seg][local_idx - 1]
        c["out_arc"] = seg_post_arcs_by_under[seg][local_idx]
        # over_arc = the arc occupying segment c['over_seg'] at parameter c['t_over']
        oseg = c["over_seg"]
        # Walk under-crossings of oseg in order: arc starts at seg_pre_arc[oseg],
        # then transitions at each under-crossing's t_under.
        cur = seg_pre_arc[oseg]
        for kk in under_per_seg[oseg]:
            if crossings[kk]["t_under"] < c["t_over"]:
                # We've crossed under another arc — arc updates
                # NOTE: kk's local index in under_per_seg[oseg]
                li = under_per_seg[oseg].index(kk)
                cur = seg_post_arcs_by_under[oseg][li]
            else:
                break
        c["over_arc"] = cur

    # Re-densify arc indices
    used = sorted({c["in_arc"] for c in crossings} |
                  {c["out_arc"] for c in crossings} |
                  {c["over_arc"] for c in crossings})
    arc_remap = {a: i for i, a in enumerate(used)}
    for c in crossings:
        c["in_arc"] = arc_remap[c["in_arc"]]
        c["out_arc"] = arc_remap[c["out_arc"]]
        c["over_arc"] = arc_remap[c["over_arc"]]
    arcs_list: List[List[int]] = [[] for _ in range(len(used))]
    return arcs_list, crossings


def _normalize_alexander(poly: Dict[int, int]) -> Dict[int, int]:
    """Pick the canonical representative of a Laurent polynomial up to ±t^k.

    Drops zero coefficients, shifts the lowest degree to 0 and flips the sign
    so the value at t = 1 is non-negative (Δ(1) = 1 for a knot).  The zero
    polynomial is returned as {0: 0}.
    """
    nz = {int(d): int(c) for d, c in poly.items() if c != 0}
    if not nz:
        return {0: 0}
    lo = min(nz)
    sign = -1 if sum(nz.values()) < 0 else 1
    return {d - lo: sign * c for d, c in nz.items()}


def _alexander_from_diagram(crossings: List[Dict]) -> Dict[int, int]:
    """Compute the Alexander polynomial of a knot from its Wirtinger diagram.

    The Alexander matrix is the (#crossings) × (#arcs) matrix of Fox
    derivatives of the Wirtinger relations, abelianised by x_a ↦ t.  With
    meridians oriented by the right-hand rule and paths composed left to
    right, a crossing with over-arc x_k and under-arc x_i (incoming) → x_j
    (outgoing) has the relation

        positive:  x_j = x_k^{-1} x_i x_k,    negative:  x_j = x_k x_i x_k^{-1},

    whose rows, scaled by units so the over-arc entry is 1 − t, are

        positive:  column x_k: 1 − t,   column x_i: −1,   column x_j: t
        negative:  column x_k: 1 − t,   column x_i: t,    column x_j: −1.

    (The opposite meridian convention swaps the two rows, which replaces
    Δ(t) by Δ(t^{-1}) — the same polynomial up to ±t^k.)  Entries add when an
    arc plays two roles at one crossing, e.g. at a kink.

    Every first minor (delete any one row and any one column) equals Δ(t) up
    to ±t^k, so a single minor is computed, exactly over ℤ[t], and the result
    is normalised by `_normalize_alexander`.

    Raises:
        _DiagramExtractionError: if the diagram is inconsistent
            (#arcs ≠ #crossings) or the minor violates |Δ(1)| = 1.
    """
    if not crossings:
        return {0: 1}
    n = len(crossings)
    n_arcs = 1 + max(max(c["over_arc"], c["in_arc"], c["out_arc"]) for c in crossings)
    if n_arcs != n:
        # A valid Wirtinger diagram for a knot satisfies #arcs = #crossings.
        raise _DiagramExtractionError(
            f"Inconsistent diagram: #arcs={n_arcs}, #crossings={n}"
        )

    try:
        import sympy as sp
        from sympy.polys.matrices import DomainMatrix
    except ImportError as exc:
        raise ImportError("sympy is required for Alexander polynomial computation") from exc

    t = sp.Symbol("t")
    R = sp.ZZ[t]
    one_minus_t, t_elt, minus_one = R.convert(1 - t), R.convert(t), R.convert(-1)
    rows = [[R.zero] * n for _ in range(n)]
    for r, c in enumerate(crossings):
        rows[r][c["over_arc"]] += one_minus_t
        if c["sign"] > 0:
            rows[r][c["in_arc"]] += minus_one
            rows[r][c["out_arc"]] += t_elt
        else:
            rows[r][c["in_arc"]] += t_elt
            rows[r][c["out_arc"]] += minus_one

    # Delete the last row and column; the minor's determinant is Δ(t) · ±t^k.
    minor = DomainMatrix([row[:-1] for row in rows[:-1]], (n - 1, n - 1), R)
    delta = _normalize_alexander({m[0]: c for m, c in minor.det().terms()})
    delta_at_1 = sum(delta.values())
    if delta_at_1 != 1:
        raise _DiagramExtractionError(
            f"Wirtinger minor gives |Δ(1)| = {delta_at_1}; a knot has |Δ(1)| = 1"
        )
    return delta


def _alexander_via_wirtinger(
    ambient_complex: SimplicialComplex, K: SimplicialComplex
) -> Optional[Dict[int, int]]:
    """Compute Δ_K(t) from K's polygonal embedding in R^3 via Wirtinger.

    Returns None if no coordinates are attached.  Raises _DiagramExtractionError
    if the polygon cannot be read off (e.g. K is not a simple closed loop).
    """
    pts = _knot_polyline_coords(ambient_complex, K)
    if pts is None:
        return None
    if pts.shape[0] < 3:
        return {0: 1}
    ex, ey, ez = _projection_basis(pts)
    crossings = _find_crossings(pts, ex, ey, ez)
    if not crossings:
        return {0: 1}
    _, annotated = _assign_arcs(pts.shape[0], crossings)
    return _alexander_from_diagram(annotated)


# ── Alexander polynomial helpers ──────────────────────────────────────────────


def _alexander_from_seifert(V: np.ndarray) -> Dict[int, int]:
    """Compute Δ(t) = det(tV - V^T) as {degree: coeff} using symbolic arithmetic."""
    try:
        import sympy as sp
    except ImportError:
        raise ImportError("sympy is required for alexander_polynomial")

    g = V.shape[0]
    if g == 0:
        return {0: 1}

    t = sp.Symbol("t")
    V_sp = sp.Matrix(V.tolist())
    M = t * V_sp - V_sp.T
    poly = sp.expand(M.det())

    if poly == 0:
        return {0: 0}

    poly_obj = sp.Poly(poly, t)
    coeffs = poly_obj.all_coeffs()
    deg = poly_obj.degree()

    result: Dict[int, int] = {}
    for i, c in enumerate(coeffs):
        d = deg - i
        v = int(c)
        if v != 0:
            result[d] = v

    return _normalize_alexander(result)


def _conway_from_alexander(alex_poly: Dict[int, int]) -> Dict[int, int]:
    """Convert a knot's Alexander polynomial to its Conway polynomial.

    Δ is only defined up to ±t^k, so it is first brought to its symmetric
    representative: centred so that Δ(t) = Δ(t^{-1}), with the sign fixed so
    that Δ(1) = +1 (hence ∇(0) = 1).  Writing that representative as
    Δ(t) = a_0 + Σ_{k=1}^g a_k (t^k + t^{-k}):
        ∇(z) = a_0 + Σ_{k=1}^g a_k * T_k(z)

    where T_k(z) = t^k + t^{-k} expressed via z = t^{1/2} - t^{-1/2}:
        T_0 = 2, T_1 = z^2 + 2, T_k = (z^2 + 2)*T_{k-1} - T_{k-2}

    This is the unique polynomial satisfying Δ(t) = ∇(t^{1/2} - t^{-1/2}).

    Raises:
        ValueError: if Δ is not ±t^k times a symmetric polynomial with
            Δ(1) = ±1, i.e. cannot be the Alexander polynomial of a knot.
    """
    poly = {d: c for d, c in alex_poly.items() if c != 0}
    if not poly:
        raise ValueError(f"Δ(t) = {alex_poly} is zero; not the Alexander polynomial of a knot")

    # Center the polynomial so it's symmetric around degree 0 (an odd degree
    # span leaves it lopsided and fails the symmetry check below).
    center = (min(poly) + max(poly)) // 2
    sym = {d - center: c for d, c in poly.items()}
    if any(sym.get(-k) != c for k, c in sym.items()):
        raise ValueError(
            f"Δ(t) = {alex_poly} is not symmetric up to ±t^k; "
            "not the Alexander polynomial of a knot"
        )
    if sum(sym.values()) < 0:
        sym = {k: -c for k, c in sym.items()}
    if sum(sym.values()) != 1:
        raise ValueError(
            f"Δ(t) = {alex_poly} has |Δ(1)| = {sum(sym.values())}; a knot has |Δ(1)| = 1"
        )

    g = max(sym.keys())

    # Build T_k as polynomials in z^2: {power_of_z2: coeff}
    # T_k represents the polynomial T_k(z) = sum_j coef_j * (z^2)^j
    T_prev2: Dict[int, int] = {0: 2}   # T_0 = 2
    T_prev1: Dict[int, int] = {0: 2, 1: 1}  # T_1 = z^2 + 2

    def poly_mul_shift(p: Dict[int, int], shift: int) -> Dict[int, int]:
        """Multiply poly by z^{2*shift}."""
        return {k + shift: v for k, v in p.items()}

    def poly_add(a: Dict[int, int], b: Dict[int, int]) -> Dict[int, int]:
        r: Dict[int, int] = dict(a)
        for k, v in b.items():
            r[k] = r.get(k, 0) + v
        return {k: v for k, v in r.items() if v != 0}

    def poly_scale(p: Dict[int, int], s: int) -> Dict[int, int]:
        return {k: v * s for k, v in p.items() if v * s != 0}

    # T_k: {power_of_z2: coeff}
    T_cache: Dict[int, Dict[int, int]] = {0: T_prev2, 1: T_prev1}

    for k in range(2, g + 1):
        # T_k = (z^2 + 2) * T_{k-1} - T_{k-2}
        # (z^2 + 2) * T_{k-1} = z^2 * T_{k-1} + 2 * T_{k-1}
        #                     = poly_mul_shift(T_{k-1}, 1) + 2 * T_{k-1}
        prev1 = T_cache[k - 1]
        prev2 = T_cache[k - 2]
        term = poly_add(poly_mul_shift(prev1, 1), poly_scale(prev1, 2))
        Tk = poly_add(term, poly_scale(prev2, -1))
        T_cache[k] = Tk

    # ∇(z) = a_0 + Σ_{k=1}^g a_k T_k(z)
    conway: Dict[int, int] = {}
    a0 = sym.get(0, 0)
    if a0 != 0:
        conway[0] = conway.get(0, 0) + a0

    for k in range(1, g + 1):
        a_k = sym.get(k, 0)
        if a_k == 0:
            continue
        for z2_pow, c in T_cache[k].items():
            deg_z = 2 * z2_pow  # T_k gives z^{2*z2_pow} terms
            conway[deg_z] = conway.get(deg_z, 0) + a_k * c

    # Non-empty: the constant term is ∇(0) = Δ(1) = 1.
    return {k: v for k, v in conway.items() if v != 0}


# ── Public API ────────────────────────────────────────────────────────────────


def seifert_matrix(
    ambient_complex: SimplicialComplex,
    K: SimplicialComplex,
    backend: str = "auto",
) -> np.ndarray:
    """Compute a Seifert matrix of knot K in ambient_complex.

    What is Being Computed?:
        The integer matrix V with V[i,j] = lk(α_i^+, α_j), where {α_i} is a
        Z-basis of H_1(F; Z) for a Seifert surface F of K and α_i^+ is the
        positive push-off of α_i off F.  sig(V + V^T) is the knot signature
        and det(tV - V^T) the Alexander polynomial.

    Algorithm:
        Everything is read off the triangulation; vertex coordinates are not
        needed (see `pysurgery.knots.seifert_surface`).
        1. Cone off the boundary 2-sphere of a triangulated 3-ball, giving S^3.
        2. F is a minimal-area integral 2-chain with ∂F = K, found by linear
           programming, whose support is checked to be an embedded surface.
        3. A basis of H_1(F) by tree–cotree decomposition.
        4. Each push-off α_i^+ is a closed path of tetrahedra on the positive
           side of F, and lk(α_i^+, α_j) is its intersection number with any
           2-chain bounded by α_j.

    Orientation:
        The signature changes sign under mirroring, so it depends on the
        orientation of the ambient complex.  With vertex coordinates attached
        this is the orientation of R^3, so positive knots have σ < 0 (e.g.
        σ = −2 for the right-handed trefoil).  Without coordinates the
        lexicographically first tetrahedron, with its vertices in increasing
        order, is taken as positively oriented, and chirality is only
        determined up to this convention.

    Args:
        ambient_complex: Triangulated 3-ball or 3-sphere (more generally, a
            combinatorial 3-manifold that becomes a homology 3-sphere once its
            boundary 2-spheres are coned off).
        K: Knot as a simple closed loop of edges of ambient_complex.
        backend: "auto", "julia", or "python".  The construction is the same
            for every backend.

    Returns:
        np.ndarray of shape (2h, 2h) with dtype int64, where h is the genus of
        the surface found.  h is at least the Seifert genus of K and often
        equal to it.  Returns a (0, 0) array when that surface is a disk (so K
        is the unknot).

    Raises:
        ValueError: if K is not a simple closed loop, or the ambient complex is
            not a suitable 3-manifold (see `SeifertSurfaceError`).
    """
    # Fast geometric path: a coplanar simple polygon embedded in R^3 is the
    # unknot (genus 0), so its Seifert matrix is empty.
    if _is_planar_polygon(ambient_complex, K):
        return np.zeros((0, 0), dtype=np.int64)

    try:
        walk = _order_knot_polyline(K)
    except _DiagramExtractionError as exc:
        raise SeifertSurfaceError(f"K is not a knot: {exc}") from exc
    coords = None
    point_cloud = ambient_complex.simplices_to_point_cloud
    if point_cloud:
        coords = {}
        for (v,) in ambient_complex.n_simplices(0):
            p = np.asarray(point_cloud.get((v,), [[]])[0], dtype=np.float64)
            if p.shape == (3,):
                coords[v] = p
    return seifert_matrix_of_triangulation(ambient_complex.n_simplices(3), walk, coords)


def alexander_polynomial(
    ambient_complex: SimplicialComplex,
    K: SimplicialComplex,
    backend: str = "auto",
) -> Dict[int, int]:
    """Compute the Alexander polynomial Δ_K(t) ∈ ℤ[t, t^{-1}].

    What is Being Computed?:
        Δ_K(t), defined up to units ±t^k.  When vertex coordinates are
        attached, it is read off a knot diagram of K's polygonal embedding
        (Fox calculus on the Wirtinger presentation); otherwise it is
        det(tV - V^T) where V is the Seifert matrix of K.
        Normalized so the lowest degree is 0 and Δ_K(1) = 1.

    Returns:
        dict mapping degree → integer coefficient. E.g. {2: 1, 1: -1, 0: 1} for
        the trefoil (Δ = t^2 - t + 1, equivalently 1 - t + t^2) and
        {2: -1, 1: 3, 0: -1} for the figure-eight knot.

    Properties verified:
        - Δ_K(1) = 1 (knot determinant at t=1)
        - Δ_K(t) = Δ_K(t^{-1}) (symmetry, up to units)
        - deg(Δ_K) = 2 * seifert_genus(K)
    """
    # Planar polygons are unknots → Δ = 1.
    if _is_planar_polygon(ambient_complex, K):
        return {0: 1}

    # Canonical path: Wirtinger from the polygonal embedding.  This produces
    # the exact Alexander polynomial directly from the knot diagram and is
    # independent of the ambient triangulation.
    if ambient_complex.simplices_to_point_cloud:
        try:
            delta = _alexander_via_wirtinger(ambient_complex, K)
            if delta is not None:
                return delta
        except _DiagramExtractionError:
            pass

    V = seifert_matrix(ambient_complex, K, backend=backend)
    if V.shape[0] == 0:
        return {0: 1}

    use_julia = (backend == "julia") or (backend == "auto" and julia_engine.available)
    if use_julia:
        try:
            # The Julia kernel only fixes the sign, so normalise the degree
            # shift the same way as the Python path.
            result = julia_engine.alexander_from_seifert(V)
            if result is not None:
                return _normalize_alexander(result)
        except Exception as e:
            if backend == "julia":
                raise
            import warnings
            warnings.warn(f"Julia alexander_polynomial failed, falling back: {e!r}")
    return _alexander_from_seifert(V)


def conway_polynomial(
    ambient_complex: SimplicialComplex,
    K: SimplicialComplex,
    backend: str = "auto",
) -> Dict[int, int]:
    """Compute the Conway polynomial ∇_K(z) ∈ ℤ[z].

    What is Being Computed?:
        The Conway polynomial satisfying Δ_K(t) = ∇_K(t^{1/2} - t^{-1/2}).
        For knots ∇_K(z) is a polynomial in z^2. ∇_K(0) = 1 for all knots.

    Returns:
        dict mapping degree → coefficient. E.g. {2: 1, 0: 1} for trefoil (1 + z^2)
        and {2: -1, 0: 1} for the figure-eight knot (1 - z^2).

    Raises:
        ValueError: if the computed Δ_K is not symmetric with Δ_K(1) = ±1.
    """
    delta = alexander_polynomial(ambient_complex, K, backend=backend)
    return _conway_from_alexander(delta)


def knot_signature(
    ambient_complex: SimplicialComplex,
    K: SimplicialComplex,
    backend: str = "auto",
) -> int:
    """Compute the knot signature σ(K) = signature(V + V^T).

    What is Being Computed?:
        The signature of the symmetric bilinear form V + V^T where V is a
        Seifert matrix. Equals #positive eigenvalues - #negative eigenvalues.

    Sign convention:
        σ(K) = sig(V + V^T), under which positive knots have negative
        signature (the usual convention, e.g. Rolfsen's and KnotInfo's): the
        right-handed trefoil, with three positive crossings, has σ = −2 and
        the left-handed trefoil σ = +2.  Mirroring K negates σ.  Without
        vertex coordinates the chirality, and with it the sign of σ, is fixed
        only by the orientation convention of `seifert_matrix`.

    Surgery relevance:
        σ(K) is a concordance invariant. |σ(K)|/2 is a lower bound for the
        unknotting number (Nakanishi-Murakami). For a knot bounding a smooth
        disk in B^4, σ(K) = 0.

    Returns:
        int (negative for negative-definite, positive for positive-definite).
    """
    # Planar polygons are unknots → σ = 0.
    if _is_planar_polygon(ambient_complex, K):
        return 0

    V = seifert_matrix(ambient_complex, K, backend=backend)
    if V.shape[0] == 0:
        return 0

    use_julia = (backend == "julia") or (backend == "auto" and julia_engine.available)
    if use_julia:
        try:
            return julia_engine.knot_signature(V)
        except Exception as e:
            if backend == "julia":
                raise
            import warnings
            warnings.warn(f"Julia knot_signature failed, falling back: {e!r}")

    S = V + V.T
    eigs = np.linalg.eigvalsh(S.astype(float))
    pos = int(np.sum(eigs > 1e-10))
    neg = int(np.sum(eigs < -1e-10))
    return pos - neg


def arf_invariant(
    ambient_complex: SimplicialComplex,
    K: SimplicialComplex,
    backend: str = "auto",
) -> int:
    """Compute the Arf invariant of K ∈ {0, 1}.

    What is Being Computed?:
        Arf(K) = 0 if Δ_K(-1) ≡ ±1 (mod 8), 1 if Δ_K(-1) ≡ ±3 (mod 8).
        The Arf invariant detects whether K is "algebraically slice" in a simple sense.

    Returns:
        0 or 1.
    """
    delta = alexander_polynomial(ambient_complex, K, backend=backend)
    delta_minus1 = sum(c * ((-1) ** d) for d, c in delta.items())
    return 0 if abs(delta_minus1) % 8 in (1, 7) else 1


def genus_bound(
    ambient_complex: SimplicialComplex,
    K: SimplicialComplex,
    backend: str = "auto",
) -> int:
    """Compute the degree-based Seifert genus bound g ≥ (1/2) deg(Δ_K).

    What is Being Computed?:
        The genus bound from the Alexander polynomial: g(K) ≥ (max_deg - min_deg) / 2.
        Equality holds for fibred knots.

    Returns:
        Non-negative integer lower bound on the Seifert genus.
    """
    delta = alexander_polynomial(ambient_complex, K, backend=backend)
    if not delta:
        return 0
    return (max(delta.keys()) - min(delta.keys())) // 2


def unknotting_number_lower_bound(
    ambient_complex: SimplicialComplex,
    K: SimplicialComplex,
    backend: str = "auto",
) -> int:
    """Compute |σ(K)|/2 as a lower bound for the unknotting number u(K).

    Returns:
        Non-negative integer.
    """
    sig = knot_signature(ambient_complex, K, backend=backend)
    return abs(sig) // 2


def _is_planar_polygon(
    ambient_complex: SimplicialComplex,
    K: SimplicialComplex,
) -> bool:
    """Fast geometric test for a planar unknot.

    Returns True if K's vertices lie (approximately) in a 2-plane and form a
    simple polygon. Planar simple polygons in R^3 are always unknots.
    """
    coords = ambient_complex.simplices_to_point_cloud
    if not coords:
        return False
    verts: set = set()
    for s in K.n_simplices(1):
        verts.update(s)
    pts = []
    for v in verts:
        key = (v,)
        if key in coords:
            pts.append(coords[key][0])
    if len(pts) < 3:
        return True
    arr = np.asarray(pts, dtype=float)
    # SVD: if smallest singular value ≈ 0, the points are coplanar
    centered = arr - arr.mean(axis=0)
    try:
        _, sv, _ = np.linalg.svd(centered, full_matrices=False)
    except Exception:
        return False
    if sv.size < 3:
        return True
    return bool(sv[-1] < 1e-6 * max(sv[0], 1.0))


def is_unknot(
    ambient_complex: SimplicialComplex,
    K: SimplicialComplex,
    backend: str = "auto",
) -> bool:
    """Test whether K is the unknot using Alexander polynomial and signature.

    A knot is definitely NOT the unknot if Δ_K(t) ≠ 1 or σ(K) ≠ 0.
    If both tests pass, K is likely (not proven) to be the unknot.

    Returns:
        True if K passes all knot invariant tests for the unknot, False otherwise.
    """
    # Geometric fast path: a planar simple polygon embedded in R^3 is always
    # the unknot — skip the expensive Seifert-matrix SNF.
    if _is_planar_polygon(ambient_complex, K):
        return True

    delta = alexander_polynomial(ambient_complex, K, backend=backend)
    # Unknot has Δ = 1
    if delta != {0: 1}:
        return False
    sig = knot_signature(ambient_complex, K, backend=backend)
    return sig == 0


def knot_determinant(
    ambient_complex: SimplicialComplex,
    K: SimplicialComplex,
    backend: str = "auto",
) -> int:
    """Compute the knot determinant det(K) = |Δ_K(-1)|.

    Returns:
        Non-negative integer. 1 for the unknot, 3 for the trefoil, 5 for 5_1, etc.
    """
    delta = alexander_polynomial(ambient_complex, K, backend=backend)
    return abs(sum(c * ((-1) ** d) for d, c in delta.items()))


def classify_knot(
    ambient_complex: SimplicialComplex,
    K: SimplicialComplex,
    backend: str = "auto",
) -> str:
    """Classify K by computing and matching standard knot invariants.

    Returns a string identifying the knot type from a small table, or "unknown".

    Identified knots (by Alexander polynomial, signature, determinant):
        unknot, trefoil (left/right), figure_eight, torus_knot_T(2,5),
        torus_knot_T(2,7), and generic "knot(g=<genus>)" descriptions.
        Chirality follows the sign convention of `knot_signature`: the
        left-handed (negative) knots have σ > 0, the right-handed ones σ < 0.
    """
    delta = alexander_polynomial(ambient_complex, K, backend=backend)
    sig = knot_signature(ambient_complex, K, backend=backend)
    det = abs(sum(c * ((-1) ** d) for d, c in delta.items()))

    KNOT_TABLE = {
        # (frozenset({(deg, coeff)}), sig, det) → name; positive knots have σ < 0
        (frozenset({(0, 1)}), 0, 1): "unknot",
        (frozenset({(2, 1), (1, -1), (0, 1)}), 2, 3): "left_trefoil",
        (frozenset({(2, 1), (1, -1), (0, 1)}), -2, 3): "right_trefoil",
        (frozenset({(2, -1), (1, 3), (0, -1)}), 0, 5): "figure_eight",
        (frozenset({(4, 1), (3, -1), (2, 1), (1, -1), (0, 1)}), 4, 5): "torus_knot_T(2,5)_left",
        (frozenset({(4, 1), (3, -1), (2, 1), (1, -1), (0, 1)}), -4, 5): "torus_knot_T(2,5)_right",
        (frozenset({(6, 1), (5, -1), (4, 1), (3, -1), (2, 1), (1, -1), (0, 1)}), 6, 7): "torus_knot_T(2,7)_left",
        (frozenset({(6, 1), (5, -1), (4, 1), (3, -1), (2, 1), (1, -1), (0, 1)}), -6, 7): "torus_knot_T(2,7)_right",
    }

    key = (frozenset(delta.items()), sig, det)
    if key in KNOT_TABLE:
        return KNOT_TABLE[key]

    g = genus_bound(ambient_complex, K, backend=backend)
    if delta == {0: 1} and sig == 0:
        return "unknot_candidate"
    return f"knot(g≥{g}, det={det}, sig={sig})"
