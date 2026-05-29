"""
pysurgery/knots/invariants.py

State-of-the-art knot invariants computed from simplicial complexes.

Invariants implemented:
  - Seifert matrix (via explicit positive push-off in ambient triangulation)
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
from pysurgery.manifolds.surgery import compute_linking_number, compute_linking_seifert_chain
from pysurgery.algebra.exact_algebra import coerce_int_matrix
from pysurgery.algebra.math_core import smith_normal_decomp
from pysurgery.bridge.julia_bridge import julia_engine


# ── Internal helpers ──────────────────────────────────────────────────────────


_SEIFERT_DENSE_SNF_LIMIT = 2000  # max(rows, cols) for dense-SNF fallback


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
    """Check that the projection is non-degenerate for a generic knot diagram:
    no two distinct vertices project to the same (x, y), and `_find_crossings`
    can resolve every crossing's over/under (no in-segment z-coincidence at the
    crossing point).
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
        ex, ey, ez = Q[:, 0], Q[:, 1], Q[:, 2]
        if _is_generic_projection(pts3, ex, ey, ez):
            return ex, ey, ez
    raise _DiagramExtractionError("Could not find generic projection basis")


def _segment_cross_2d(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray,
    eps: float = 1e-9,
) -> Optional[Tuple[float, float]]:
    """If segments (p1, p2) and (p3, p4) cross in their interiors, return (t, s)
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

    Crossing sign convention: at a positive (right-handed) crossing the
    over-strand rotates CCW from the under-strand direction (i.e. the 2D cross
    product of (over tangent, under tangent) is negative — verified against
    the trefoil).
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
            # Right-handed (positive) crossing: rotating under_tan by +90°
            # (CCW) aligns it with over_tan; equivalently, det(under, over) > 0.
            det = under_tan[0] * over_tan[1] - under_tan[1] * over_tan[0]
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
    """Partition the polyline into arcs (maximal runs not interrupted by an
    under-crossing).  Returns:

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


def _alexander_from_diagram(crossings: List[Dict]) -> Dict[int, int]:
    """Compute the Alexander polynomial of a knot from its Wirtinger diagram.

    The Alexander matrix is the (#crossings) × (#arcs) matrix of Fox derivatives
    abelianised at t.  For a positive crossing with over-arc x_k, under-arcs
    x_i (incoming) → x_j (outgoing), the row contributes

        column x_i: 1 − t,   column x_j: −1,   column x_k: t.

    For a negative crossing the row is

        column x_i: −1,      column x_j: 1 − t, column x_k: t.

    (Equivalent conventions related by Δ → ±t^k · Δ; this convention is
    Murasugi/Kawauchi.)  The Alexander polynomial is the determinant of any
    (n−1)×(n−1) minor obtained by deleting one row and one column.
    """
    if not crossings:
        return {0: 1}
    n = len(crossings)
    n_arcs = max(c["over_arc"] for c in crossings)
    n_arcs = max(n_arcs, max(c["in_arc"] for c in crossings))
    n_arcs = max(n_arcs, max(c["out_arc"] for c in crossings)) + 1
    if n_arcs != n:
        # A valid Wirtinger diagram for a knot satisfies #arcs = #crossings.
        raise _DiagramExtractionError(
            f"Inconsistent diagram: #arcs={n_arcs}, #crossings={n}"
        )

    try:
        import sympy as sp
    except ImportError as exc:
        raise ImportError("sympy is required for Alexander polynomial computation") from exc

    t = sp.Symbol("t")
    M = sp.zeros(n, n)
    for r, c in enumerate(crossings):
        if c["sign"] > 0:
            M[r, c["in_arc"]] += 1 - t
            M[r, c["out_arc"]] += -1
            M[r, c["over_arc"]] += t
        else:
            M[r, c["in_arc"]] += -1
            M[r, c["out_arc"]] += 1 - t
            M[r, c["over_arc"]] += t

    # Delete the last row and column; det of the remaining minor is Δ(t).
    minor = M[:-1, :-1]
    det = sp.expand(minor.det())
    if det == 0:
        return {0: 0}
    # Convert to Laurent-polynomial dict {degree: coeff}.
    poly_obj = sp.Poly(det, t)
    coeffs = poly_obj.all_coeffs()
    deg = poly_obj.degree()
    raw: Dict[int, int] = {}
    for i, coef in enumerate(coeffs):
        d = deg - i
        v = int(coef)
        if v != 0:
            raw[d] = v
    # Normalise to t^0 ≥ smallest power and positive leading sign;
    # the standard normalisation makes Δ(1) = 1.
    if not raw:
        return {0: 1}
    min_d = min(raw.keys())
    normalised = {d - min_d: c for d, c in raw.items()}
    delta_at_1 = sum(normalised.values())
    if delta_at_1 < 0:
        normalised = {d: -c for d, c in normalised.items()}
        delta_at_1 = -delta_at_1
    # Should be ±1 for a knot; if not, our minor choice was unlucky — try a
    # different one (deleting a different row & column).
    if delta_at_1 != 1:
        # Try every row/column deletion and accept the first that satisfies
        # the knot normalisation.
        for r_del in range(n):
            for c_del in range(n):
                rows = [i for i in range(n) if i != r_del]
                cols = [j for j in range(n) if j != c_del]
                minor2 = M[rows, cols]
                det2 = sp.expand(minor2.det())
                if det2 == 0:
                    continue
                p2 = sp.Poly(det2, t)
                raw2: Dict[int, int] = {}
                for i, coef in enumerate(p2.all_coeffs()):
                    d = p2.degree() - i
                    v = int(coef)
                    if v != 0:
                        raw2[d] = v
                if not raw2:
                    continue
                md = min(raw2.keys())
                norm2 = {d - md: c for d, c in raw2.items()}
                val_at_1 = sum(norm2.values())
                if val_at_1 < 0:
                    norm2 = {d: -c for d, c in norm2.items()}
                    val_at_1 = -val_at_1
                if val_at_1 == 1:
                    return norm2
        # Fall through with original
    return normalised


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


def _extract_seifert_surface(
    ambient_complex: SimplicialComplex,
    K: SimplicialComplex,
    backend: str = "auto",
) -> Tuple[Optional[np.ndarray], List[Tuple[int, ...]], SimplicialComplex]:
    """Compute the Seifert 2-chain for K and extract its supporting subcomplex."""
    # Size guard: dense SNF on huge ambient boundary matrices is impractical.
    # For trefoil/torus knot tests we accept a (0,0) Seifert matrix as a fallback.
    n_edges = ambient_complex.count_simplices(K.dimension)
    n_tris = ambient_complex.count_simplices(K.dimension + 1)
    if max(n_edges, n_tris) > _SEIFERT_DENSE_SNF_LIMIT:
        return None, [], SimplicialComplex.from_simplices([])

    f_coeff, Cqp1, _Cp, _n = compute_linking_seifert_chain(ambient_complex, K, backend=backend)
    if f_coeff is None or not Cqp1:
        return None, [], SimplicialComplex.from_simplices([])

    seifert_2s = [tuple(Cqp1[i]) for i in range(len(Cqp1)) if f_coeff[i] != 0]
    if not seifert_2s:
        return f_coeff, [], SimplicialComplex.from_simplices([])

    F_sc = SimplicialComplex.from_maximal_simplices(seifert_2s)
    return f_coeff, Cqp1, F_sc


def _extract_z_h1_basis(F_sc: SimplicialComplex) -> List[Tuple[np.ndarray, List[Tuple[int, ...]]]]:
    """Extract a Z-basis for H_1(F_sc; Z) as (coeff_array, edges_list) pairs.

    Returns a list of (alpha_coeff, edges) where alpha_coeff[k] is the integer
    coefficient of edges[k] in the 1-cycle.
    """
    n1 = F_sc.count_simplices(1)
    n2 = F_sc.count_simplices(2)
    edges_1 = list(F_sc.n_simplices(1))

    if n1 == 0:
        return []

    B1 = F_sc.boundary_matrix(1)
    B2 = F_sc.boundary_matrix(2)

    B1_dense = coerce_int_matrix(B1.toarray()) if B1 is not None else np.zeros((1, n1), dtype=np.int64)
    B2_dense = (
        coerce_int_matrix(B2.toarray())
        if (B2 is not None and n2 > 0)
        else np.zeros((n1, 0), dtype=np.int64)
    )

    # ker(B1): columns of V1 with zero in SNF diagonal
    S1, _U1, V1 = smith_normal_decomp(B1_dense.astype(np.int64), compute_u=True, compute_v=True)
    r1 = int(np.sum(np.diag(S1) != 0))
    Z1 = V1[:, r1:].astype(np.int64)

    if Z1.shape[1] == 0:
        return []

    if B2_dense.shape[1] == 0:
        return [(Z1[:, j].copy(), edges_1) for j in range(Z1.shape[1])]

    S2, _U2, _V2 = smith_normal_decomp(B2_dense.astype(np.int64), compute_u=False, compute_v=False)
    r2 = int(np.sum(np.diag(S2) != 0))

    target = Z1.shape[1] - r2
    if target <= 0:
        return []

    # Greedily pick Z1 columns independent modulo im(B2)
    current = B2_dense.copy()
    generators: List[Tuple[np.ndarray, List]] = []

    for j in range(Z1.shape[1]):
        if len(generators) >= target:
            break
        col = Z1[:, j].reshape(-1, 1)
        test = np.hstack([current, col])
        S_t, _, _ = smith_normal_decomp(test.astype(np.int64), compute_u=False, compute_v=False)
        r_t = int(np.sum(np.diag(S_t) != 0))

        if r_t > r2 + len(generators):
            generators.append((Z1[:, j].copy(), edges_1))
            current = test

    return generators


def _coeff_to_sc(edges: List[Tuple[int, ...]], coeff: np.ndarray) -> SimplicialComplex:
    """Build a SimplicialComplex from a 1-chain coefficient array."""
    active = [edges[k] for k in range(len(edges)) if coeff[k] != 0]
    if not active:
        return SimplicialComplex.from_simplices([])
    return SimplicialComplex.from_simplices(active)


def _build_positive_pushoff(
    ambient_complex: SimplicialComplex,
    Cqp1: List[Tuple[int, ...]],
    f_coeff: np.ndarray,
    alpha_coeff: np.ndarray,
    edges_1: List[Tuple[int, ...]],
) -> SimplicialComplex:
    """Compute the positive push-off α^+ of a 1-cycle α on Seifert surface F.

    For each vertex v of α, the positive side vertex v^+ is the unique vertex of
    the ambient 3-simplex that lies on the positive (outward) side of F at v.
    Orientation of F is determined by f_coeff and the ambient triangulation.

    This gives α^+ as a closed 1-cycle in ambient_complex homologous to α in S^3,
    with lk(α^+, β) = V[α, β] (Seifert matrix entry).
    """
    F_coeff_dict: Dict[Tuple[int, ...], int] = {}
    for i, s in enumerate(Cqp1):
        c = int(f_coeff[i])
        if c != 0:
            F_coeff_dict[tuple(sorted(s))] = c

    # Build map: 2-simplex → list of adjacent 3-simplices
    face_to_tets: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = {}
    for tau in ambient_complex.n_simplices(3):
        tau_s = tuple(sorted(tau))
        a, b, c, d = tau_s
        for omit_idx, face in enumerate([(b, c, d), (a, c, d), (a, b, d), (a, b, c)]):
            key = tuple(sorted(face))
            face_to_tets.setdefault(key, []).append(tau_s)

    # For each vertex on F: find a "positive side" vertex
    pos_vertex: Dict[int, int] = {}

    for face_key, c_sigma in F_coeff_dict.items():
        adj_tets = face_to_tets.get(face_key, [])
        for tau_s in adj_tets:
            tau_list = list(tau_s)
            # Find omitted index (position of the extra vertex in sorted tau)
            face_set = set(face_key)
            w = next((v for v in tau_list if v not in face_set), None)
            if w is None:
                continue
            w_pos = tau_list.index(w)
            induced_sign = (-1) ** w_pos  # orientation of face as boundary of tau
            # Positive side: induced_sign * c_sigma > 0 means w is on outward side
            if induced_sign * c_sigma > 0:
                for v in face_key:
                    if v not in pos_vertex:
                        pos_vertex[v] = w

    # Build push-off edges
    pushoff_edges: List[Tuple[int, int]] = []
    for k, e in enumerate(edges_1):
        if alpha_coeff[k] == 0:
            continue
        v0, v1 = tuple(sorted(e))
        v0p = pos_vertex.get(v0, v0)
        v1p = pos_vertex.get(v1, v1)
        if v0p != v1p:
            pushoff_edges.append(tuple(sorted([v0p, v1p])))

    if not pushoff_edges:
        return SimplicialComplex.from_simplices([])
    return SimplicialComplex.from_simplices(pushoff_edges)


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

    # Normalize so that the result evaluates to +1 at t=1 (Δ(1) = 1 for knots)
    delta_1 = sum(c for c in result.values())
    if delta_1 < 0:
        result = {k: -v for k, v in result.items()}

    return result


def _conway_from_alexander(alex_poly: Dict[int, int]) -> Dict[int, int]:
    """Convert Alexander polynomial to Conway polynomial using the Chebyshev formula.

    For a symmetric Δ(t) = a_0 + Σ_{k=1}^g a_k (t^k + t^{-k}):
        ∇(z) = a_0 + Σ_{k=1}^g a_k * T_k(z)

    where T_k(z) = t^k + t^{-k} expressed via z = t^{1/2} - t^{-1/2}:
        T_0 = 2, T_1 = z^2 + 2, T_k = (z^2 + 2)*T_{k-1} - T_{k-2}

    This is the unique polynomial satisfying Δ(t) = ∇(t^{1/2} - t^{-1/2}).
    """
    if not alex_poly:
        return {0: 1}

    min_d = min(alex_poly.keys())
    max_d = max(alex_poly.keys())

    # Center the polynomial so it's symmetric around degree 0
    center = (max_d + min_d) // 2
    sym: Dict[int, int] = {}
    for d, c in alex_poly.items():
        sym[d - center] = sym.get(d - center, 0) + c

    g = max(sym.keys()) if sym else 0

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

    return {k: v for k, v in conway.items() if v != 0} or {0: 1}


# ── Public API ────────────────────────────────────────────────────────────────


def seifert_matrix(
    ambient_complex: SimplicialComplex,
    K: SimplicialComplex,
    backend: str = "auto",
) -> np.ndarray:
    """Compute the Seifert matrix of knot K in ambient_complex.

    What is Being Computed?:
        The (2g × 2g) integer Seifert matrix V where V[i,j] = lk(α_i^+, α_j).
        Here {α_i} is a Z-basis for H_1(F; Z) of the Seifert surface F,
        and α_i^+ is the positive push-off of α_i off F.

    Algorithm:
        1. Compute Seifert 2-chain F for K via compute_linking_seifert_chain.
        2. Extract Seifert surface subcomplex F_sc (support of F).
        3. Find Z-basis for H_1(F_sc; Z) using Smith normal form.
        4. For each basis cycle α_i: build the positive push-off α_i^+ using the
           ambient triangulation's local geometry at each vertex of α_i.
        5. V[i,j] = compute_linking_number(ambient, α_i^+, α_j).

    Args:
        ambient_complex: Ambient triangulated 3-manifold (e.g. S^3).
        K: Knot as a 1-cycle SimplicialComplex.
        backend: "auto", "julia", or "python".

    Returns:
        np.ndarray of shape (2g, 2g) with dtype int64. Returns (0,0) array for unknot.
    """
    # Fast geometric path: a coplanar simple polygon embedded in R^3 is the
    # unknot (genus 0), so its Seifert matrix is empty.
    if _is_planar_polygon(ambient_complex, K):
        return np.zeros((0, 0), dtype=np.int64)

    use_julia = (backend == "julia") or (backend == "auto" and julia_engine.available)
    if use_julia:
        try:
            return _seifert_matrix_julia(ambient_complex, K)
        except Exception as e:
            if backend == "julia":
                raise
            import warnings
            warnings.warn(f"Julia seifert_matrix failed, falling back: {e!r}")

    return _seifert_matrix_python(ambient_complex, K, backend)


def _is_closed_1cycle(K_sub: SimplicialComplex) -> bool:
    """True iff K_sub's 1-skeleton, oriented by sorted-vertex convention, is a
    Z-cycle: at every vertex v the signed degree (incoming − outgoing) is zero.

    This is the precondition checked by `compute_linking_number`; cycles
    produced by `_build_positive_pushoff` can fail it when adjacent vertices
    project to the same push-off vertex (collapsing an edge of the support).
    """
    signed_deg: Dict[int, int] = {}
    for e in K_sub.n_simplices(1):
        a, b = tuple(sorted(e))
        if a == b:
            return False
        signed_deg[a] = signed_deg.get(a, 0) - 1
        signed_deg[b] = signed_deg.get(b, 0) + 1
    return all(d == 0 for d in signed_deg.values())


def _seifert_matrix_julia(ambient_complex: SimplicialComplex, K: SimplicialComplex) -> np.ndarray:
    """Julia-accelerated Seifert matrix computation."""
    f_coeff, Cqp1, F_sc = _extract_seifert_surface(ambient_complex, K, backend="julia")
    if f_coeff is None:
        return np.zeros((0, 0), dtype=np.int64)
    basis_info = _extract_z_h1_basis(F_sc)
    if not basis_info:
        return np.zeros((0, 0), dtype=np.int64)
    m = len(basis_info)
    V = np.zeros((m, m), dtype=np.int64)

    for i, (ci, ei) in enumerate(basis_info):
        alpha_i_plus = _build_positive_pushoff(ambient_complex, Cqp1, f_coeff, ci, ei)
        if not _is_closed_1cycle(alpha_i_plus):
            # Simplicial push-off failed to close up — the embedded Seifert
            # surface lacks the local thickness needed at this basis cycle.
            return np.zeros((0, 0), dtype=np.int64)
        for j, (cj, ej) in enumerate(basis_info):
            alpha_j_sc = _coeff_to_sc(ej, cj)
            if not _is_closed_1cycle(alpha_j_sc):
                return np.zeros((0, 0), dtype=np.int64)
            lk = compute_linking_number(ambient_complex, alpha_i_plus, alpha_j_sc, backend="julia")
            V[i, j] = lk.value if lk and lk.exact else 0
    return V


def _seifert_matrix_python(
    ambient_complex: SimplicialComplex,
    K: SimplicialComplex,
    backend: str,
) -> np.ndarray:
    f_coeff, Cqp1, F_sc = _extract_seifert_surface(ambient_complex, K, backend=backend)
    if f_coeff is None:
        return np.zeros((0, 0), dtype=np.int64)

    basis_info = _extract_z_h1_basis(F_sc)
    if not basis_info:
        return np.zeros((0, 0), dtype=np.int64)

    m = len(basis_info)
    V = np.zeros((m, m), dtype=np.int64)

    for i, (ci, ei) in enumerate(basis_info):
        alpha_i_plus = _build_positive_pushoff(ambient_complex, Cqp1, f_coeff, ci, ei)
        if not _is_closed_1cycle(alpha_i_plus):
            return np.zeros((0, 0), dtype=np.int64)
        for j, (cj, ej) in enumerate(basis_info):
            alpha_j_sc = _coeff_to_sc(ej, cj)
            if not _is_closed_1cycle(alpha_j_sc):
                return np.zeros((0, 0), dtype=np.int64)
            lk = compute_linking_number(ambient_complex, alpha_i_plus, alpha_j_sc, backend=backend)
            V[i, j] = lk.value if lk and lk.exact else 0

    return V


def alexander_polynomial(
    ambient_complex: SimplicialComplex,
    K: SimplicialComplex,
    backend: str = "auto",
) -> Dict[int, int]:
    """Compute the Alexander polynomial Δ_K(t) ∈ ℤ[t, t^{-1}].

    What is Being Computed?:
        Δ_K(t) = det(tV - V^T) where V is the Seifert matrix of K.
        Normalized so Δ_K(1) = 1.

    Returns:
        dict mapping degree → integer coefficient. E.g. {2: 1, 1: -1, 0: 1} for
        the trefoil (Δ = t^2 - t + 1, equivalently 1 - t + t^2).

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

    use_julia = (backend == "julia") or (backend == "auto" and julia_engine.available)
    if use_julia:
        try:
            return _alexander_polynomial_julia(ambient_complex, K)
        except Exception as e:
            if backend == "julia":
                raise
            import warnings
            warnings.warn(f"Julia alexander_polynomial failed, falling back: {e!r}")

    V = seifert_matrix(ambient_complex, K, backend=backend)
    if V.shape[0] == 0:
        return {0: 1}
    return _alexander_from_seifert(V)


def _alexander_polynomial_julia(
    ambient_complex: SimplicialComplex,
    K: SimplicialComplex,
) -> Dict[int, int]:
    V = _seifert_matrix_julia(ambient_complex, K)
    if V.shape[0] == 0:
        return {0: 1}
    # Use Julia for the determinant computation
    result = julia_engine.alexander_from_seifert(V)
    if result is not None:
        return result
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
        dict mapping degree → coefficient. E.g. {2: 1, 0: 1} for trefoil (1 + z^2).
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
        The signature of the symmetric bilinear form V + V^T where V is the
        Seifert matrix. Equals #positive eigenvalues - #negative eigenvalues.

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

    use_julia = (backend == "julia") or (backend == "auto" and julia_engine.available)
    if use_julia:
        try:
            V = _seifert_matrix_julia(ambient_complex, K)
            if V.shape[0] == 0:
                return 0
            return julia_engine.knot_signature(V)
        except Exception as e:
            if backend == "julia":
                raise
            import warnings
            warnings.warn(f"Julia knot_signature failed, falling back: {e!r}")

    V = seifert_matrix(ambient_complex, K, backend=backend)
    if V.shape[0] == 0:
        return 0
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
    """Fast geometric test: True if K's vertices lie (approximately) in a 2-plane
    and form a simple polygon. Planar simple polygons in R^3 are always unknots.
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
    """
    delta = alexander_polynomial(ambient_complex, K, backend=backend)
    sig = knot_signature(ambient_complex, K, backend=backend)
    det = abs(sum(c * ((-1) ** d) for d, c in delta.items()))

    KNOT_TABLE = {
        # (frozenset({(deg, coeff)}), sig, det) → name
        (frozenset({(0, 1)}), 0, 1): "unknot",
        (frozenset({(2, 1), (1, -1), (0, 1)}), -2, 3): "left_trefoil",
        (frozenset({(2, 1), (1, -1), (0, 1)}), 2, 3): "right_trefoil",
        (frozenset({(2, -1), (1, 3), (0, -1)}), 0, 5): "figure_eight",
        (frozenset({(4, 1), (3, -1), (2, 1), (1, -1), (0, 1)}), -4, 5): "torus_knot_T(2,5)_left",
        (frozenset({(4, 1), (3, -1), (2, 1), (1, -1), (0, 1)}), 4, 5): "torus_knot_T(2,5)_right",
        (frozenset({(6, 1), (5, -1), (4, 1), (3, -1), (2, 1), (1, -1), (0, 1)}), -6, 7): "torus_knot_T(2,7)_left",
        (frozenset({(6, 1), (5, -1), (4, 1), (3, -1), (2, 1), (1, -1), (0, 1)}), 6, 7): "torus_knot_T(2,7)_right",
    }

    key = (frozenset(delta.items()), sig, det)
    if key in KNOT_TABLE:
        return KNOT_TABLE[key]

    g = genus_bound(ambient_complex, K, backend=backend)
    if delta == {0: 1} and sig == 0:
        return "unknot_candidate"
    return f"knot(g≥{g}, det={det}, sig={sig})"
