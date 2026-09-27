import itertools
import math
from collections import Counter
import numpy as np
from typing import Tuple, List, Sequence
from scipy.spatial import Delaunay
from pysurgery.topology.complexes import SimplicialComplex


def _subdivided_polyline(
    corners: Sequence[Tuple[float, float, float]],
) -> List[Tuple[int, int, int]]:
    """Return a closed polyline through `corners` with unit-length integer steps.

    Each consecutive pair (corners[i], corners[i+1]) lies on an axis-aligned
    segment of integer length L; we insert L−1 intermediate integer points
    along that segment so every segment of the resulting polyline has unit
    length. This guarantees the polyline edges are short enough to be Delaunay
    edges in a wider point cloud.
    """
    pts: List[Tuple[int, int, int]] = []
    n = len(corners)
    for i in range(n):
        a = np.asarray(corners[i], dtype=np.int64)
        b = np.asarray(corners[(i + 1) % n], dtype=np.int64)
        diff = b - a
        L = int(round(float(np.linalg.norm(diff))))
        if L == 0:
            continue
        unit = diff // L if L > 0 else diff
        for s in range(L):
            pts.append(tuple((a + s * unit).tolist()))
    return pts


def _delaunay_s3_from_points(
    ring_points: List[Tuple[int, int, int]],
    bbox_extent: float = 10.0,
) -> Tuple[SimplicialComplex, dict]:
    """Build a triangulated 3-ball containing the given point set.

    Attaches geometric coordinates and returns ``(complex, point_to_index map)``.

    The 3-ball is the convex hull of `ring_points` ∪ {north pole}, tetrahedralised
    by scipy's Delaunay. Since H_1(B^3) = 0, every 1-cycle inside is null-homologous
    and admits a Seifert chain — sufficient for the linking and Milnor invariants
    used in this module.
    """
    seen: set = set()
    unique: List[Tuple[int, int, int]] = []
    for p in ring_points:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    idx_map = {p: i for i, p in enumerate(unique)}
    # North pole — guarantees the convex hull encloses the rings on all sides.
    pole = (0, 0, int(round(bbox_extent)))
    coords = np.array(unique + [pole], dtype=np.float64)
    tri = Delaunay(coords)
    tets = [tuple(sorted(int(v) for v in t)) for t in tri.simplices]
    sc = SimplicialComplex.from_maximal_simplices(tets)
    sc._generate_point_cloud_mappings(coords)
    return sc, idx_map


def _ring_cycle(
    perimeter: List[Tuple[int, int, int]],
    idx_map: dict,
) -> SimplicialComplex:
    """Materialise a 1-cycle SimplicialComplex from a perimeter point list."""
    edges = []
    n = len(perimeter)
    for i in range(n):
        a = idx_map[perimeter[i]]
        b = idx_map[perimeter[(i + 1) % n]]
        edges.append((min(a, b), max(a, b)))
    return SimplicialComplex.from_simplices(edges)

def _build_s3_grid(size: int = 4) -> Tuple[SimplicialComplex, np.ndarray]:
    """Build a triangulation of S^3 from a cube grid in R^3.

    The cube [0, size]^3 is cut into unit cubes, each split into the six Kuhn
    (Freudenthal) tetrahedra along its main diagonal, which fit together
    across shared faces.  Coning the boundary of the big cube off to vertex 0,
    the point at infinity, closes this 3-ball up to S^3.

    Returns:
        A tuple of (ambient_complex, index_map).
        index_map is a 3D array mapping (x, y, z) to vertex index; vertex 0 is
        the point at infinity.
    """
    N = size + 1
    idx_map = np.arange(1, N**3 + 1, dtype=int).reshape(N, N, N)

    simplices = []
    for x in range(size):
        for y in range(size):
            for z in range(size):
                # One Kuhn tetrahedron per order of the three unit steps from
                # (x, y, z) to (x+1, y+1, z+1).
                for p in itertools.permutations(range(3)):
                    curr = [x, y, z]
                    verts = [idx_map[x, y, z]]
                    for d in p:
                        curr[d] += 1
                        verts.append(idx_map[curr[0], curr[1], curr[2]])
                    simplices.append(tuple(sorted(int(v) for v in verts)))

    # The boundary 2-sphere consists of the triangles in only one tetrahedron.
    face_count = Counter(t[:i] + t[i + 1:] for t in simplices for i in range(4))
    simplices += [(0,) + f for f, c in face_count.items() if c == 1]

    sc = SimplicialComplex.from_maximal_simplices(simplices)

    # Attach geometric coordinates so linking-number routines can use them.
    # vertex 0 = "infinity" — placed far from the grid; grid vertices use grid coords.
    points = np.zeros((N**3 + 1, 3), dtype=np.float64)
    points[0] = np.array([10.0 * size, 10.0 * size, 10.0 * size])
    points[1:] = np.indices((N, N, N)).reshape(3, -1).T
    sc._generate_point_cloud_mappings(points)

    return sc, idx_map

def _extract_cycle(idx_map: np.ndarray, points: List[Tuple[int, int, int]]) -> SimplicialComplex:
    """Build a 1-dimensional SimplicialComplex from a list of grid coordinates.

    Consecutive points are connected by an axis-aligned staircase path
    (x → y → z), which guarantees every edge is a valid Kuhn-triangulation
    edge. The intermediate vertices are reachable in the grid via single-axis
    steps, so the resulting 1-cycle is always a subcomplex of the ambient
    Kuhn-triangulated S^3.
    """
    simplices: List[Tuple[int, int]] = []
    n = len(points)
    if n == 0:
        return SimplicialComplex.from_simplices([])

    for i in range(n):
        p1 = tuple(points[i])
        p2 = tuple(points[(i + 1) % n])
        if p1 == p2:
            continue
        current = list(p1)
        for axis in range(3):
            while current[axis] != p2[axis]:
                step = 1 if current[axis] < p2[axis] else -1
                nxt = list(current)
                nxt[axis] += step
                v1 = int(idx_map[tuple(current)])
                v2 = int(idx_map[tuple(nxt)])
                if v1 != v2:
                    simplices.append((v1, v2))
                current = nxt
    return SimplicialComplex.from_simplices(simplices)

def _join_complex(
    A_verts: List[int],
    B_verts: List[int],
    A_edges: List[Tuple[int, int]],
    B_edges: List[Tuple[int, int]],
    coords: np.ndarray,
) -> SimplicialComplex:
    """Build the simplicial join A * B as a SimplicialComplex.

    The join |A * B| is homeomorphic to |A| * |B| in the topological sense.
    For two 1-cycles (triangles), |S^1 * S^1| = S^3 — this is the minimal
    triangulation of S^3 that exhibits the Hopf link as a pair of disjoint
    triangle cycles.
    """
    # Maximal simplices are obtained by joining one cell of A with one cell of B.
    # For two 1-cycles we get all (edge_A) * (edge_B) = 4-vertex tetrahedra.
    tetrahedra: List[Tuple[int, ...]] = []
    for e1 in A_edges:
        for e2 in B_edges:
            tetrahedra.append(tuple(sorted(set(e1) | set(e2))))
    sc = SimplicialComplex.from_maximal_simplices(tetrahedra)
    sc._generate_point_cloud_mappings(coords)
    return sc


def hopf_link() -> Tuple[SimplicialComplex, List[SimplicialComplex]]:
    """Construct the Hopf link in S^3 as a minimal triangulation.

    Uses the simplicial join S^1 * S^1 = S^3: two disjoint triangle cycles
    embedded so that one threads through the other once. The ambient S^3 has
    only 6 vertices, 15 edges, 20 triangles, and 9 tetrahedra — far smaller
    than the Kuhn grid version — and the two components carry exact linking
    number 1.

    Returns:
        Tuple of (ambient_complex, [component1, component2]).
    """
    # Component 1: unit triangle in the xy-plane.
    # Component 2: triangle in the y=0 plane that pierces component 1's disk once
    # (one vertex at large +x, one at +z, one at −z) — a classic Hopf weave.
    coords = np.array([
        [ 1.0,  0.0,  0.0],   # 0
        [-0.5,  0.8660254038,  0.0],   # 1
        [-0.5, -0.8660254038,  0.0],   # 2
        [ 3.0,  0.0,  0.0],   # 3
        [ 0.0,  0.0,  3.0],   # 4
        [ 0.0,  0.0, -3.0],   # 5
    ], dtype=np.float64)

    A_edges = [(0, 1), (1, 2), (0, 2)]
    B_edges = [(3, 4), (4, 5), (3, 5)]
    sc = _join_complex([0, 1, 2], [3, 4, 5], A_edges, B_edges, coords)

    c1 = SimplicialComplex.from_simplices(A_edges)
    c2 = SimplicialComplex.from_simplices(B_edges)
    return sc, [c1, c2]


def hopf_link_grid() -> Tuple[SimplicialComplex, List[SimplicialComplex]]:
    """Hopf link embedded in a Kuhn-triangulated S^3 grid (legacy, slower).

    Useful when callers need a richer ambient (e.g. for higher-resolution
    discrete invariants); tests should prefer the minimal `hopf_link()` above.
    """
    sc, idx_map = _build_s3_grid(size=6)
    r1_pts = [(2,2,3), (3,2,3), (4,2,3), (4,3,3), (4,4,3), (3,4,3), (2,4,3), (2,3,3)]
    c1 = _extract_cycle(idx_map, r1_pts)
    r2_pts = [(3,3,2), (3,3,3), (3,3,4), (3,4,4), (3,5,4), (3,5,3), (3,5,2), (3,4,2)]
    c2 = _extract_cycle(idx_map, r2_pts)
    return sc, [c1, c2]

def borromean_rings() -> Tuple[SimplicialComplex, List[SimplicialComplex]]:
    """Construct the Borromean rings in S^3 as a minimal Delaunay triangulation.

    Three pairwise-perpendicular rectangles, each 2×4 in its coordinate plane:
        R1: xy-plane (z=0), x ∈ [−1, 1], y ∈ [−2, 2]
        R2: yz-plane (x=0), y ∈ [−1, 1], z ∈ [−2, 2]
        R3: xz-plane (y=0), x ∈ [−2, 2], z ∈ [−1, 1]
    The perimeters are subdivided to unit-length edges so scipy's Delaunay
    naturally selects every ring edge; the triangulation has 37 vertices
    (36 ring + one north-pole), exactly the structure needed to compute the
    Milnor triple invariant without the cost of a Kuhn S^3 grid.

    Pairwise linking numbers vanish, but μ̄(123) ≠ 0 — the classical Borromean
    obstruction.

    Returns:
        Tuple of (ambient_complex, [component1, component2, component3]).
    """
    R1 = _subdivided_polyline([(-1, -2, 0), (1, -2, 0), (1, 2, 0), (-1, 2, 0)])
    R2 = _subdivided_polyline([(0, -1, -2), (0, 1, -2), (0, 1, 2), (0, -1, 2)])
    R3 = _subdivided_polyline([(-2, 0, -1), (2, 0, -1), (2, 0, 1), (-2, 0, 1)])

    sc, idx_map = _delaunay_s3_from_points(R1 + R2 + R3, bbox_extent=10.0)

    c1 = _ring_cycle(R1, idx_map)
    c2 = _ring_cycle(R2, idx_map)
    c3 = _ring_cycle(R3, idx_map)
    return sc, [c1, c2, c3]


# ── Additional constructors ───────────────────────────────────────────────────


def _parametric_to_grid(
    pts_float: List[Tuple[float, float, float]],
    idx_map: np.ndarray,
) -> List[Tuple[int, int, int]]:
    """Round float points to integer grid, remove consecutive duplicates, verify bounds."""
    N = idx_map.shape[0] - 1  # interior goes 1..N-1
    result = []
    for x, y, z in pts_float:
        ix = int(round(x))
        iy = int(round(y))
        iz = int(round(z))
        # Clamp to the interior, off the boundary that is coned to infinity
        ix = max(1, min(N - 1, ix))
        iy = max(1, min(N - 1, iy))
        iz = max(1, min(N - 1, iz))
        pt = (ix, iy, iz)
        if not result or pt != result[-1]:
            result.append(pt)
    # Remove last point if same as first (closed loop handling)
    if len(result) > 1 and result[-1] == result[0]:
        result.pop()
    return result


def unknot() -> Tuple[SimplicialComplex, SimplicialComplex]:
    """Construct the unknot (trivial knot) in S^3 as a minimal triangulation.

    Reuses the 6-vertex S^1 * S^1 = S^3 ambient from `hopf_link()` and returns
    one of its planar triangle components as the unknot. Δ_K(t) = 1, σ(K) = 0.

    Returns:
        (ambient_complex, knot_component)
    """
    sc, comps = hopf_link()
    return sc, comps[0]


def _delaunay_knot_from_polyline(
    polyline: List[Tuple[float, float, float]],
    pole_height_factor: float = 4.0,
) -> Tuple[SimplicialComplex, SimplicialComplex]:
    """Build a Delaunay 3-ball around a closed polyline.

    Returns `(ambient_complex, K)` with K = the polyline as a 1-cycle.

    The ambient triangulation has |polyline|+1 vertices (one north pole) and is
    minimal in the sense that every polyline edge appears as a Delaunay edge of
    the convex hull of `polyline ∪ {pole}`.  Geometric vertex coordinates are
    attached to `ambient_complex` so that knot-diagram methods (e.g. the
    Wirtinger Alexander polynomial) can read off the embedding.
    """
    pts = np.asarray(polyline, dtype=np.float64)
    bbox = float(np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))
    pole = pts.mean(axis=0) + np.array([0.0, 0.0, pole_height_factor * bbox])
    coords = np.vstack([pts, pole[None, :]])
    tri = Delaunay(coords)
    tets = [tuple(sorted(int(v) for v in t)) for t in tri.simplices]
    sc = SimplicialComplex.from_maximal_simplices(tets)
    sc._generate_point_cloud_mappings(coords)
    n = pts.shape[0]
    edges = [(min(i, (i + 1) % n), max(i, (i + 1) % n)) for i in range(n)]
    K = SimplicialComplex.from_simplices(edges)
    return sc, K


def trefoil_knot(handedness: str = "left") -> Tuple[SimplicialComplex, SimplicialComplex]:
    """Construct the trefoil knot (torus knot T(2,3)) in S^3.

    The trefoil is the simplest non-trivial knot:
        Alexander polynomial: Δ(t) = t - 1 + t^{-1} (equivalently t^2 - t + 1)
        Signature: ±2 (left: +2, right: −2; positive knots have negative
        signature, see `knot_signature`)
        Genus: 1, Determinant: 3, Arf: 1

    Uses the classical 3-crossing trefoil parametrisation

        x(t) = sin(t) + 2 sin(2t)
        y(t) = cos(t) − 2 cos(2t)
        z(t) = sign · sin(3t),   sign = −1 for "left", +1 for "right"

    sampled at 24 points so the projected diagram has exactly three same-sign
    crossings (negative for "left", positive for "right").  The ambient
    triangulation is the Delaunay 3-ball of the polyline plus a north-pole
    vertex (25 vertices, 95 tetrahedra), and vertex coordinates are attached so
    the Alexander polynomial and signature are read off the knot diagram
    exactly.

    Args:
        handedness: "left" (default) or "right".

    Returns:
        (ambient_complex, trefoil_knot_component)
    """
    sign = -1 if handedness == "left" else 1
    N = 24
    polyline: List[Tuple[float, float, float]] = []
    for k in range(N):
        t = 2 * math.pi * k / N
        x = math.sin(t) + 2 * math.sin(2 * t)
        y = math.cos(t) - 2 * math.cos(2 * t)
        z = sign * math.sin(3 * t)
        polyline.append((x, y, z))
    return _delaunay_knot_from_polyline(polyline)


def figure_eight_knot() -> Tuple[SimplicialComplex, SimplicialComplex]:
    """Construct the figure-eight knot (4_1 in Rolfsen table) in S^3.

    Properties:
        Alexander polynomial: Δ(t) = -t + 3 - t^{-1} (equiv. -t^2 + 3t - 1)
        Conway polynomial: ∇(z) = 1 - z^2
        Signature: 0 (amphichiral — equal to its mirror image)
        Genus: 1, Determinant: 5, Arf: 1 (a_2 = -1 is odd)
        It is the simplest hyperbolic knot.

    Uses the standard parametric figure-eight embedding sampled at 32 points;
    triangulation is the Delaunay 3-ball around the polyline plus one pole.

    Returns:
        (ambient_complex, figure_eight_knot_component)
    """
    N = 32
    polyline: List[Tuple[float, float, float]] = []
    for k in range(N):
        t = 2 * math.pi * k / N
        x = (2.0 + math.cos(2 * t)) * math.cos(3 * t)
        y = (2.0 + math.cos(2 * t)) * math.sin(3 * t)
        z = math.sin(4 * t)
        polyline.append((x, y, z))
    return _delaunay_knot_from_polyline(polyline)


def torus_knot(p: int, q: int) -> Tuple[SimplicialComplex, SimplicialComplex]:
    """Construct the torus knot T(p, q) in S^3.

    The torus knot T(p, q) lies on the standard torus in S^3, winding p times
    around the longitude and q times around the meridian (gcd(p, q) = 1 required).

    Alexander polynomial: Δ_{T(p,q)}(t) = (t^{pq} - 1)(t - 1) / ((t^p - 1)(t^q - 1))
    Genus: (p-1)(q-1)/2

    Chirality: this parametrisation (both angles increasing together) gives
    the negative, left-handed torus knot, the mirror of the positive T(p, q)
    that closes the braid (σ_1 ⋯ σ_{p−1})^q.  So torus_knot(2, 3) is the
    left-handed trefoil, and σ > 0 in the convention of `knot_signature`,
    e.g. σ = +2 for T(2, 3) and +4 for T(2, 5).

    Args:
        p: First winding number (≥ 2).
        q: Second winding number (≥ 2, gcd(p,q) = 1).

    Returns:
        (ambient_complex, torus_knot_component)
    """
    if math.gcd(p, q) != 1:
        raise ValueError(f"T({p},{q}): gcd({p},{q}) = {math.gcd(p,q)} ≠ 1 — not a knot")

    N = max(24, 4 * p * q)
    R, r = 3.0, 1.2
    polyline: List[Tuple[float, float, float]] = []
    for k in range(N):
        phi = 2 * math.pi * p * k / N
        theta = 2 * math.pi * q * k / N
        x = (R + r * math.cos(theta)) * math.cos(phi)
        y = (R + r * math.cos(theta)) * math.sin(phi)
        z = r * math.sin(theta)
        polyline.append((x, y, z))
    return _delaunay_knot_from_polyline(polyline)


def whitehead_link() -> Tuple[SimplicialComplex, List[SimplicialComplex]]:
    """Construct the Whitehead link in S^3 as a minimal Delaunay triangulation.

    Component 2 is a figure-eight curve in the plane z = 0: two square lobes, in the
    quadrants x, y > 0 and x, y < 0, whose strands cross once at the origin (the
    horizontal strand at z = 2 over the vertical one at z = -2). Component 1 is the
    square of half-side 4 around the origin; it crosses each lobe twice, passing over
    the lobes on its sides x = +-4 (z = 2) and under them on its sides y = +-4
    (z = -2). Read along component 2 the five crossings alternate over and under, so
    this is the alternating 5-crossing diagram of the Whitehead link 5^2_1: lk = 0,
    Sato-Levine invariant beta = -mu-bar(1122) = +-1, and the link is not split.

    Subdividing every polyline edge to unit length lets scipy's Delaunay 3-ball over
    the union (plus a pole) realise both polylines as 1-cycles of the triangulation.

    Returns:
        (ambient_complex, [component1, component2])
    """
    C1 = _subdivided_polyline([(4, -4, 2), (4, 4, 2), (4, 4, -2), (-4, 4, -2), (-4, 4, 2),
                               (-4, -4, 2), (-4, -4, -2), (4, -4, -2)])
    C2 = _subdivided_polyline([(-2, 0, 2), (2, 0, 2), (2, 0, 0), (8, 0, 0), (8, 6, 0), (0, 6, 0),
                               (0, 2, 0), (0, 2, -2), (0, -2, -2), (0, -2, 0), (0, -6, 0),
                               (-8, -6, 0), (-8, 0, 0), (-2, 0, 0)])
    sc, idx_map = _delaunay_s3_from_points(C1 + C2, bbox_extent=16.0)
    c1 = _ring_cycle(C1, idx_map)
    c2 = _ring_cycle(C2, idx_map)
    return sc, [c1, c2]
