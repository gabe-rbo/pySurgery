"""Per-simplex appearance ("filtration") values for monotone complexes.

Every point-cloud complex we build (Vietoris-Rips, CkNN, Alpha, Delaunay-Rips,
Delaunay-Cech, Witness) is a *monotone* filtration: each simplex enters at a
single parameter value and never leaves. Persistent homology of such a complex
is one boundary-matrix reduction over the simplices ordered by appearance value
(see ``filtration_tools``). This module computes those appearance values; the
only thing that differs between methods is the geometry encoded here.

All functions return values in the same units as the controlling parameter
(a radius / distance), and every value map is monotone under taking faces
(value(face) <= value(coface)), which is exactly what the persistence reduction
requires.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List, Tuple

import numpy as np

Simplex = Tuple[int, ...]


# ──────────────────────────────────────────────────────────────────────────────
# Smallest spheres
# ──────────────────────────────────────────────────────────────────────────────
def _circumsphere(P: np.ndarray) -> Tuple[np.ndarray, float]:
    """Circumscribed sphere of affinely-independent points.

    The sphere passes through every point of ``P`` with its center in their affine
    hull (used for Alpha/Gabriel and as Welzl's boundary solver); it is not
    necessarily the minimum enclosing ball.

    Args:
        P: ``(k, D)`` array of ``k`` affinely-independent points.

    Returns:
        Tuple ``(center, r2)``: the center (length-``D`` array) and squared radius.
    """
    P = np.asarray(P, dtype=np.float64)
    if P.shape[0] == 1:
        return P[0], 0.0
    Q = P[1:] - P[0]                       # (k-1, D)
    rhs = 0.5 * np.sum(Q * Q, axis=1)      # (k-1,)
    G = Q @ Q.T                            # (k-1, k-1)
    a, *_ = np.linalg.lstsq(G, rhs, rcond=None)
    center = P[0] + Q.T @ a
    r2 = float(np.sum((center - P[0]) ** 2))
    return center, r2


def _miniball_r2(P: np.ndarray) -> float:
    """Squared radius of the minimum enclosing ball (Welzl's algorithm).

    ``m`` is tiny here (a simplex has <= ``max_dim + 1`` vertices), so the
    recursion is cheap and the exponential worst case is irrelevant.

    Args:
        P: ``(m, D)`` array of points.

    Returns:
        The squared radius of the smallest ball enclosing every point of ``P``.
    """
    P = np.asarray(P, dtype=np.float64)
    dim = P.shape[1]
    pts: List[np.ndarray] = [P[i] for i in range(P.shape[0])]

    def welzl(pset: List[np.ndarray], boundary: List[np.ndarray]):
        if not pset or len(boundary) == dim + 1:
            if not boundary:
                return None, 0.0
            return _circumsphere(np.asarray(boundary))
        p = pset[-1]
        c, r2 = welzl(pset[:-1], boundary)
        if c is not None and np.sum((p - c) ** 2) <= r2 + 1e-12:
            return c, r2
        return welzl(pset[:-1], boundary + [p])

    _, r2 = welzl(pts, [])
    return float(max(r2, 0.0))


# ──────────────────────────────────────────────────────────────────────────────
# Flag complexes (Rips, CkNN, Delaunay-Rips): value = max over edges
# ──────────────────────────────────────────────────────────────────────────────
def _max_pairwise(simplices_table: Dict[int, List[Simplex]],
                  pair_value) -> Dict[Simplex, float]:
    """Appearance value of every simplex as the max over its edges.

    Computes ``value(sigma) = max`` over vertex pairs ``(u, v)`` of ``sigma`` of
    ``pair_value(u, v)``, vectorised per dimension; vertices get ``0.0``.

    Args:
        simplices_table: dim -> list of simplices (sorted vertex tuples).
        pair_value: Callable mapping index arrays ``(u, v)`` to a value array
            (the per-edge appearance value).

    Returns:
        Dict mapping each simplex to its appearance value.
    """
    out: Dict[Simplex, float] = {}
    for d in sorted(simplices_table.keys()):
        simps = simplices_table[d]
        if not simps:
            continue
        if d == 0:
            for s in simps:
                out[s] = 0.0
            continue
        arr = np.asarray(simps, dtype=np.int64)        # (M, d+1)
        M, k = arr.shape
        maxv = np.zeros(M, dtype=np.float64)
        for a in range(k):
            for b in range(a + 1, k):
                maxv = np.maximum(maxv, pair_value(arr[:, a], arr[:, b]))
        for i in range(M):
            out[simps[i]] = float(maxv[i])
    return out


def rips_filtration_values(simplices_table: Dict[int, List[Simplex]],
                           coords: np.ndarray) -> Dict[Simplex, float]:
    """Vietoris-Rips / Delaunay-Rips appearance values (longest edge).

    Args:
        simplices_table: dim -> list of simplices (sorted vertex tuples).
        coords: (N, D) array of point coordinates.

    Returns:
        Dict mapping each simplex to the maximum pairwise distance among its
        vertices (its longest edge).
    """
    coords = np.asarray(coords, dtype=np.float64)

    def pair_value(u, v):
        diff = coords[u] - coords[v]
        return np.sqrt(np.einsum("ij,ij->i", diff, diff))

    return _max_pairwise(simplices_table, pair_value)


def cknn_filtration_values(simplices_table: Dict[int, List[Simplex]],
                           coords: np.ndarray, rho: np.ndarray) -> Dict[Simplex, float]:
    """CkNN appearance values.

    Edge ``(i, j)`` appears at ``delta = d(i, j) / sqrt(rho_i * rho_j)`` and higher
    simplices at the max over their edges (CkNN is a flag complex in ``delta``).

    Args:
        simplices_table: dim -> list of simplices (sorted vertex tuples).
        coords: (N, D) array of point coordinates.
        rho: (N,) array of per-point local scales (k-th nearest-neighbour
            distances).

    Returns:
        Dict mapping each simplex to its CkNN appearance value.
    """
    coords = np.asarray(coords, dtype=np.float64)
    sqrt_rho = np.sqrt(np.asarray(rho, dtype=np.float64))

    def pair_value(u, v):
        diff = coords[u] - coords[v]
        dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        return dist / (sqrt_rho[u] * sqrt_rho[v])

    return _max_pairwise(simplices_table, pair_value)


# ──────────────────────────────────────────────────────────────────────────────
# Delaunay-Cech: value = minimum enclosing ball radius
# ──────────────────────────────────────────────────────────────────────────────
def cech_filtration_values(simplices_table: Dict[int, List[Simplex]],
                           coords: np.ndarray) -> Dict[Simplex, float]:
    """Cech appearance values (smallest enclosing ball radius).

    The appearance value of a simplex is the radius of the smallest enclosing ball
    of its vertices. Monotone because ``MEB(face) <= MEB(coface)``.

    Args:
        simplices_table: dim -> list of simplices (sorted vertex tuples).
        coords: (N, D) array of point coordinates.

    Returns:
        Dict mapping each simplex to its smallest-enclosing-ball radius.
    """
    coords = np.asarray(coords, dtype=np.float64)
    out: Dict[Simplex, float] = {}
    for d in sorted(simplices_table.keys()):
        for s in simplices_table[d]:
            out[s] = 0.0 if len(s) == 1 else float(np.sqrt(_miniball_r2(coords[list(s)])))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Alpha complex: circumradius if Gabriel, else min over cofaces
# ──────────────────────────────────────────────────────────────────────────────
def alpha_filtration_values(coords: np.ndarray,
                            delaunay_top: Iterable[Iterable[int]],
                            max_dim: int) -> Dict[Simplex, float]:
    """Alpha appearance value for every Delaunay face up to ``max_dim``.

    The faces come from the Delaunay top simplices ``delaunay_top``. A simplex is
    "Gabriel" when its circumscribed sphere is empty of other
    vertices; then its alpha is the circumradius. Otherwise it inherits the
    minimum alpha of its cofaces. By the Delaunay property only the vertices
    opposite a face in its cofacial top simplices can fall inside the sphere, so
    the Gabriel test is local and exact. Values are computed over the full
    Delaunay dimension and then restricted to ``max_dim``.

    Args:
        coords: (N, D) array of point coordinates.
        delaunay_top: Iterable of the Delaunay triangulation's top simplices
            (each a sequence of vertex indices).
        max_dim: Maximum simplex dimension to retain in the result.

    Returns:
        Dict mapping each face (up to ``max_dim``) to its alpha value (a radius).
    """
    coords = np.asarray(coords, dtype=np.float64)
    full_dim = int(coords.shape[1])

    top = [tuple(sorted(int(v) for v in s)) for s in delaunay_top]
    faces_by_dim: Dict[int, set] = {d: set() for d in range(full_dim + 1)}
    opposite: Dict[Simplex, set] = {}     # face -> opposite vertices in cofacial top simplices
    for t in top:
        tset = set(t)
        for d in range(min(full_dim, len(t) - 1) + 1):
            for f in combinations(t, d + 1):
                f = tuple(sorted(f))
                faces_by_dim[d].add(f)
                opposite.setdefault(f, set()).update(tset - set(f))

    circ: Dict[Simplex, Tuple[np.ndarray, float]] = {}
    for d in range(full_dim + 1):
        for f in faces_by_dim[d]:
            circ[f] = _circumsphere(coords[list(f)])

    alpha2: Dict[Simplex, float] = {}
    for d in range(full_dim, -1, -1):                 # high dim first (cofaces before faces)
        for f in faces_by_dim[d]:
            center, r2 = circ[f]
            gabriel = all(
                np.sum((coords[p] - center) ** 2) >= r2 - 1e-12
                for p in opposite[f]
            )
            if gabriel:
                a2 = r2
            else:
                a2 = min((alpha2[tuple(sorted(f + (p,)))]
                          for p in opposite[f]
                          if tuple(sorted(f + (p,))) in alpha2), default=r2)
            alpha2[f] = a2

    return {f: float(np.sqrt(max(a2, 0.0)))
            for f, a2 in alpha2.items()
            if len(f) - 1 <= max_dim}
