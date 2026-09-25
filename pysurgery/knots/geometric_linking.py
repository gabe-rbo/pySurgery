r"""Linking and winding numbers of embedded simplicial cycles in R^m, exactly.

Overview:
    ``pysurgery.knots.linking`` computes linking numbers combinatorially inside an
    ambient triangulated complex, and ``manifolds.surgery`` evaluates the Gauss integral
    of two curves by a Riemann sum (then rounds). This module computes the same
    integers from the EMBEDDING, in any ambient dimension, as intersection counts --
    integers, never integrals:

        definedness(dims, m)          which of linking / a2 / enclosure the geometry admits
        winding_number(x, Z, coords)  degree of a point w.r.t. an (m-1)-cycle in R^m
        simplicial_linking(A, B, m)   linking number of a p-cycle and a q-cycle in R^m,
                                      p + q = m - 1
        curve_linking(c1, c2)         the same for two closed polygons in R^3
        gauss_linking_estimate(...)   the Gauss integral by the midpoint rule -- an
                                      APPROXIMATION, kept because it is smooth in the
                                      coordinates

Key Concepts:
    - **Exact, not integrated.** The linking number of disjoint cycles is an intersection
      number: ``lk(A, B) = I(o * A, B)`` with ``o * A`` the cone on A from a generic apex
      o. Since A is a cycle, ``d(o * A) = A - o * dA = A``, so the cone is a Seifert chain
      bounded by A. Each (p+1)-simplex ``[o, sigma]`` meets each q-simplex of B in at most
      one transverse point (complementary dimensions, generic apex): one linear solve
      per pair, and the answer is an INTEGER. The midpoint-rule Gauss integral it
      replaces converges to the integer and was measured at -1.011 / +1.040 / -1.092 for
      p = 1 / 2 / 3.
    - **Certified genericity, not voting.** The apex (resp. the ray, for winding numbers)
      is accepted only if every intersection is interior to both simplices by a relative
      margin and every linear system is well conditioned; otherwise the next candidate
      of a FIXED deterministic sequence is tried. If B passes (numerically) through A, no
      apex helps -- the cycles are not disjoint, the linking number is not defined, and
      the function raises ``NonGenericConfigurationError``.
    - **Refusals.** Linking of a p- and a q-cycle is defined (and not identically zero)
      exactly when p + q = m - 1; otherwise ``UndefinedInvariantError``. A point on the
      cycle has no winding number.

Conventions:
    A chain is a list of ``(simplex, coefficient)`` pairs; a simplex ``(v0, ..., vk)`` is
    oriented by its edge frame ``(v1 - v0, ..., vk - v0)`` -- the listed vertex order IS
    the orientation (so ``FundamentalCycle.as_pairs()`` plugs in directly). The
    intersection sign of a (p+1)-simplex and a q-simplex is the sign of det[their frames,
    in that order]. For two curves in R^3 this is the classical linking number: it
    agrees with the Seifert-surface count, with the crossing count of
    ``knots.diagrams.linking_number``, and with the Gauss integral
    ``(x - y).(dx x dy) / 4 pi``; in every dimension it has the sign of the generalized
    Gauss integral with ``det[x - y, dA, dB]``. Swapping the cycles gives
    ``lk(B, A) = (-1)^(pq+1) lk(A, B)``. The winding number of a point is the degree of
    ``z -> (z - x)/|z - x|`` on the cycle: +1 inside a counterclockwise circle, +1 inside
    the boundary of a positively oriented simplex.

Common Workflows:
    1. **Two knots in R^3** -> ``curve_linking(c1, c2)``.
    2. **A 2-sphere and a circle in R^4** -> ``simplicial_linking(S, coords, C, coords, 4)``.
    3. **Is a point inside a closed hypersurface?** -> ``winding_number(x, Z, coords)``.

Coefficient Ring:
    Z (every result is an exact integer, or a refusal).
"""

from __future__ import annotations

import warnings
from math import factorial
from typing import List, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, Field
from scipy.special import gammaln

from ..bridge.julia_bridge import julia_engine
from ..core.exceptions import NonGenericConfigurationError, UndefinedInvariantError

Chain = List[Tuple[Tuple[int, ...], int]]

__all__ = [
    "LinkingDefinedness",
    "definedness",
    "linking_definedness",
    "chain_boundary",
    "is_cycle",
    "winding_number",
    "winding_numbers",
    "simplicial_linking",
    "polygon_cycle",
    "curve_linking",
    "gauss_linking_estimate",
    "curve_gauss_integral",
    "sphere_volume",
]

#: Seed of the fixed, deterministic sequence of candidate rays and cone apices.
_GENERIC_SEED = 20260921


# ------------------------------------------------------------- definedness


class LinkingDefinedness(BaseModel):
    """Which invariants the geometry admits, before any of them is run.

    Attributes:
        m (int): Ambient dimension.
        dims (list[float]): The classes' dimensions (possibly fractional estimates).
        linking_ok (bool): p + q = m - 1 (within ``tol``).
        linking_reason (str): Why.
        a2_ok (bool): A filament (n = 1) in R^3 -- the knotting window.
        a2_reason (str): Why.
        enclosure_ok (bool): Codimension exactly 1 (Jordan-Brouwer).
        enclosure_reason (str): Why.
        notes (list[str]): Suggested projections.
    """

    m: int
    dims: List[float]
    linking_ok: bool
    linking_reason: str
    a2_ok: bool
    a2_reason: str
    enclosure_ok: bool
    enclosure_reason: str
    notes: List[str] = Field(default_factory=list)


def definedness(dims: Sequence[float], m: int, tol: float = 0.5) -> LinkingDefinedness:
    """Which of linking / a2 / enclosure is defined for classes of these dimensions in R^m.

    What is Being Computed?:
        Linking of a p- and a q-manifold is defined (and not identically zero) exactly
        when p + q = m - 1; the Casson invariant a2 is a filament-in-R^3 tool; enclosure
        needs codimension 1 (Jordan-Brouwer separation).

    Args:
        dims: One or two class dimensions (a single value is used for both).
        m: The ambient dimension.
        tol: Tolerance for fractional (estimated) dimensions.

    Returns:
        A ``LinkingDefinedness`` report.
    """
    dims = [float(d) for d in dims]
    m = int(m)
    p, q = (dims + dims)[:2]
    need = p + q + 1
    link_ok = abs(m - need) <= tol
    link_reason = f"p + q = {p:.2f} + {q:.2f} = {p + q:.2f}, needs m = {need:.2f}, m = {m}" + (
        "" if link_ok else (
            "; identically ZERO at this codimension" if m > need
            else "; not defined (codimension too small)"
        )
    )
    a2_ok = abs(p - 1) <= tol and m == 3
    a2_reason = "filament in R^3" if a2_ok else (
        f"needs n = 1 and m = 3; have n = {p:.2f}, m = {m}. The knotting window and the "
        f"faithful-projection window are disjoint for n > 1"
    )
    enc_ok = abs(m - (p + 1)) <= tol
    enc_reason = f"codimension {m - p:.2f}; Jordan-Brouwer needs exactly 1" + (
        "" if enc_ok else " -- no separation at this codimension"
    )
    notes = []
    if m > need + tol:
        notes.append(f"project to m = {int(round(need))} for linking")
    if m > p + 1 + tol:
        notes.append(f"project to m = {int(round(p + 1))} for enclosure")
    return LinkingDefinedness(
        m=m, dims=dims, linking_ok=bool(link_ok), linking_reason=link_reason,
        a2_ok=bool(a2_ok), a2_reason=a2_reason, enclosure_ok=bool(enc_ok),
        enclosure_reason=enc_reason, notes=notes,
    )


#: Package-level name of ``definedness`` (exported as ``pysurgery.linking_definedness``).
linking_definedness = definedness


# ----------------------------------------------------------------- helpers


def _chain(cycle: Chain) -> Tuple[np.ndarray, np.ndarray]:
    simp = np.array([list(s) for s, _ in cycle], dtype=np.int64)
    coef = np.array([int(c) for _, c in cycle], dtype=np.int64)
    if simp.ndim != 2:
        raise ValueError("a chain must contain simplices of one dimension")
    return simp, coef


def _sorted_with_sign(face: Sequence[int]) -> Tuple[Tuple[int, ...], int]:
    """Sort a vertex tuple, returning the sign of the sorting permutation."""
    f = list(face)
    sign = 1
    for i in range(len(f)):  # insertion sort, counting transpositions
        j = i
        while j > 0 and f[j - 1] > f[j]:
            f[j - 1], f[j] = f[j], f[j - 1]
            sign = -sign
            j -= 1
    return tuple(f), sign


def chain_boundary(chain: Chain) -> dict:
    """The boundary of a chain, exactly, respecting the listed vertex orders.

    Args:
        chain: ``[(simplex, coefficient), ...]``; each simplex's vertex order is its
            orientation.

    Returns:
        ``{sorted_face: coefficient}`` with zero coefficients dropped.
    """
    acc: dict = {}
    for s, c in chain:
        s = tuple(int(v) for v in s)
        for i in range(len(s)):
            face, sgn = _sorted_with_sign(s[:i] + s[i + 1:])
            acc[face] = acc.get(face, 0) + int(c) * sgn * (-1 if i % 2 else 1)
    return {f: v for f, v in acc.items() if v}


def is_cycle(chain: Chain) -> bool:
    """``d(chain) = 0`` in exact integer arithmetic (chains of dimension >= 1).

    Args:
        chain: ``[(simplex, coefficient), ...]``.

    Returns:
        Whether the chain is a cycle.
    """
    return not chain_boundary(chain)


def _check_nondegenerate(V: np.ndarray, what: str) -> None:
    """Refuse flat simplices.

    Every simplex must span its dimension: a flat simplex has no transverse
    intersections and no orientation, so counts through it mean nothing.
    """
    E = V[:, 1:, :] - V[:, :1, :]
    if E.shape[1] == 0:
        return
    sv = np.linalg.svd(E, compute_uv=False)
    if np.any(sv[:, -1] <= 1e-12 * np.maximum(sv[:, 0], 1e-300)):
        raise ValueError(f"{what} contains a degenerate (flat) simplex")


def _directions(m: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(_GENERIC_SEED)
    out = rng.normal(size=(n, m))
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def _use_julia(backend: str) -> Tuple[bool, str]:
    b = str(backend).lower().strip()
    return (b == "julia" or (b == "auto" and julia_engine.available)), b


# ----------------------------------------------------------- winding number


def _winding_python(x, simp, coef, C, directions, tol) -> Tuple[int, int]:
    """Winding number of one point by the first certified generic ray.

    Returns ``(value, status)``: status 0 ok, 1 the point lies on the cycle, 2 no generic
    ray among the directions.
    """
    m = len(x)
    V = C[simp]
    E = V[:, 1:, :] - V[:, :1, :]
    scale = float(np.abs(C - x).max()) or 1.0
    for d in directions:
        # x + t d = v0 + sum lam_i e_i   <=>   [d, -e_1, ..., -e_{m-1}] [t, lam] = v0 - x
        M = np.concatenate([np.broadcast_to(d, (len(V), 1, m)), -E], axis=1).transpose(0, 2, 1)
        det = np.linalg.det(M)
        norms = np.prod(np.linalg.norm(M, axis=1), axis=1)
        ok = np.abs(det) > 1e-12 * np.maximum(norms, 1e-300)
        if not ok.all():
            # parallel to these simplices' hyperplanes: harmless unless the ray lies
            # (numerically) IN one of them, and then this direction is useless
            _, _, vt = np.linalg.svd(E[~ok], full_matrices=True)
            normal = vt[:, -1, :]
            if np.any(np.abs(np.einsum("ni,ni->n", normal, x - V[~ok, 0, :])) <= tol * scale):
                continue
        sol = np.zeros((len(V), m))
        if ok.any():
            sol[ok] = np.linalg.solve(M[ok], (V[ok, 0, :] - x)[..., None])[..., 0]
        t, lam = sol[:, 0], sol[:, 1:]
        bary = np.concatenate([1 - lam.sum(1, keepdims=True), lam], axis=1)
        inside = ok & np.all(bary > -tol, axis=1)
        if np.any(inside & (np.abs(t) <= tol * scale)):
            return 0, 1
        if np.any(inside & (t > 0) & np.any(bary < tol, axis=1)):
            continue  # the ray grazes a lower-dimensional face: next ray
        hit = inside & (t > 0)
        frame = np.concatenate([np.broadcast_to(d, (len(V), 1, m)), E], axis=1)
        return int(np.sum(coef[hit] * np.sign(np.linalg.det(frame[hit])))), 0
    return 0, 2


def winding_numbers(
    points: np.ndarray,
    cycle: Chain,
    coords: np.ndarray,
    tol: float = 1e-9,
    max_tries: int = 32,
    backend: str = "auto",
    require_cycle: bool = True,
) -> np.ndarray:
    """Degree of every point with respect to a simplicial (m-1)-cycle in R^m, exactly.

    What is Being Computed?:
        For each point x, the signed count of the crossings of ONE certified generic ray
        from x with the cycle: +1 per simplex crossed with ``det[d, frame] > 0``. The ray
        is rejected, and the next of a fixed sequence tried, if it passes within ``tol``
        (relative) of a simplex's boundary or runs inside a simplex's hyperplane.

    Args:
        points: ``(k, m)`` query points (or one point).
        cycle: An (m-1)-cycle ``[(simplex, coefficient), ...]``.
        coords: Vertex coordinates, ``coords[v]`` in R^m.
        tol: Relative genericity margin.
        max_tries: Number of candidate rays.
        backend: 'auto', 'julia' (points in parallel threads) or 'python'.
        require_cycle: Check ``d(cycle) = 0`` over Z first. The even-odd rule
            (``geometry.enclosure.crossing_parity``) turns it off to count crossings of a
            chain that is only a cycle mod 2; the result is then a crossing count, not a
            degree.

    Returns:
        An int array of winding numbers, one per point.

    Raises:
        UndefinedInvariantError: If the cycle has the wrong dimension or a point lies
            (numerically) on the cycle.
        NonGenericConfigurationError: If no generic ray is found.
        ValueError: If the chain is not a cycle or contains a flat simplex.
    """
    P = np.atleast_2d(np.asarray(points, dtype=np.float64))
    C = np.asarray(coords, dtype=np.float64)
    m = P.shape[1]
    simp, coef = _chain(cycle)
    if simp.shape[1] != m:
        raise UndefinedInvariantError(
            f"a winding number in R^{m} needs an ({m - 1})-cycle; got {simp.shape[1] - 1}-simplices"
        )
    if C.shape[1] != m:
        raise ValueError(f"coordinates must be in R^{m}")
    if require_cycle and not is_cycle(cycle):
        raise ValueError("the chain is not a cycle (d != 0, checked exactly)")
    _check_nondegenerate(C[simp], "the cycle")
    dirs = _directions(m, max_tries)
    use_julia, bnorm = _use_julia(backend)
    res = None
    if use_julia:
        try:
            res = julia_engine.winding_numbers(P, simp, coef, C, dirs, tol)
        except Exception as e:  # pragma: no cover - depends on the Julia runtime
            if bnorm == "julia":
                raise
            warnings.warn(f"Julia winding numbers failed ({e!r}); falling back to Python.")
    if res is None:
        res = [_winding_python(x, simp, coef, C, dirs, tol) for x in P]
    out = np.zeros(len(P), dtype=np.int64)
    for i, (w, status) in enumerate(res):
        if status == 1:
            raise UndefinedInvariantError(
                f"point {i} lies on the cycle: its winding number is undefined"
            )
        if status == 2:
            raise NonGenericConfigurationError(f"no generic ray among {max_tries} directions (point {i})")
        out[i] = w
    return out


def winding_number(point: np.ndarray, cycle: Chain, coords: np.ndarray, **kw) -> int:
    """Degree of one point with respect to a simplicial (m-1)-cycle in R^m.

    Args:
        point: A point of R^m.
        cycle: An (m-1)-cycle.
        coords: Vertex coordinates.
        **kw: Passed to ``winding_numbers``.

    Returns:
        The winding number.
    """
    return int(winding_numbers(np.atleast_2d(point), cycle, coords, **kw)[0])


# ------------------------------------------------------- linking, exactly


def _cone_count_python(VA, cA, VB, cB, o, tol, scale, p, chunk=256) -> Tuple[int, int]:
    """Intersection number of the cone o*A with B for one apex.

    Returns ``(total, status)``: status 0 ok, 1 degenerate apex, 2 B passes through A.
    """
    m = VA.shape[2]
    EA = VA - o
    EB = VB[:, 1:, :] - VB[:, :1, :]
    lo_A = np.minimum(VA.min(1), o)
    hi_A = np.maximum(VA.max(1), o)
    lo_B, hi_B = VB.min(1), VB.max(1)
    total, degenerate, through = 0, False, False
    for a0 in range(0, len(VA), chunk):
        a1 = min(a0 + chunk, len(VA))
        ov = np.all((lo_A[a0:a1, None, :] <= hi_B[None] + tol * scale) &
                    (hi_A[a0:a1, None, :] >= lo_B[None] - tol * scale), axis=2)
        ia, ib = np.nonzero(ov)
        if ia.size == 0:
            continue
        ia = ia + a0
        F = np.concatenate([EA[ia], EB[ib]], axis=1)          # rows = frame vectors
        det = np.linalg.det(F)
        norms = np.prod(np.linalg.norm(F, axis=2), axis=1)
        good = np.abs(det) > 1e-12 * np.maximum(norms, 1e-300)
        # x = o + sum lam_i EA_i = VB_0 + sum mu_j EB_j
        M = np.concatenate([EA[ia], -EB[ib]], axis=1).transpose(0, 2, 1)
        rhs = VB[ib, 0, :] - o
        sol = np.full((len(ia), m), np.nan)
        if good.any():
            sol[good] = np.linalg.solve(M[good], rhs[good][..., None])[..., 0]
        lam, mu = sol[:, :p + 1], sol[:, p + 1:]
        bl = np.concatenate([1 - lam.sum(1, keepdims=True), lam], axis=1)  # apex weight first
        bm = np.concatenate([1 - mu.sum(1, keepdims=True), mu], axis=1)
        close = good & np.all(bl > -tol, axis=1) & np.all(bm > -tol, axis=1)
        if np.any(close & (bl[:, 0] < tol)):
            through = True
        if np.any(close & (np.any(bl < tol, axis=1) | np.any(bm < tol, axis=1))):
            degenerate = True
        # A singular pair has no transverse intersection. It is harmless when the two
        # affine hulls are parallel and APART; if they (nearly) meet, this apex is not
        # certified.
        if np.any(~good):
            nb = np.nonzero(~good)[0]
            Uu, sv, _ = np.linalg.svd(M[nb])
            null = sv <= 1e-9 * np.maximum(sv[:, :1], 1e-300)
            comp = np.einsum("kij,ki->kj", Uu, rhs[nb])
            gap = np.sqrt((np.where(null, comp, 0.0) ** 2).sum(1))
            if np.any(gap <= 1e-6 * scale):
                degenerate = True
        total += int(np.sum(cA[ia[close]] * cB[ib[close]] * np.sign(det[close])))
    if through:
        return 0, 2
    if degenerate:
        return 0, 1
    return int(total), 0


def simplicial_linking(
    cycle_A: Chain,
    coords_A: np.ndarray,
    cycle_B: Chain,
    coords_B: np.ndarray,
    m: int,
    tol: float = 1e-9,
    max_tries: int = 16,
    backend: str = "auto",
) -> int:
    """Linking number of a simplicial p-cycle A and q-cycle B in R^m, p + q = m - 1.

    What is Being Computed?:
        The intersection number of B with the cone on A from a certified generic apex
        (see the module docstring) -- an exact integer.

    Algorithm:
        1. Check p + q = m - 1, that A and B are cycles (exactly) and that no simplex is
           flat.
        2. For each apex of a fixed deterministic sequence around A's centroid: for every
           pair (cone simplex, B simplex) whose bounding boxes meet, solve for the
           intersection; accept the apex only if every intersection is interior by a
           relative margin and every singular pair is parallel and apart.
        3. Sum ``c_A c_B sign(det[frames])`` over the intersections.

    Args:
        cycle_A: A p-cycle ``[(simplex, coefficient), ...]``.
        coords_A: Vertex coordinates of A in R^m.
        cycle_B: A q-cycle.
        coords_B: Vertex coordinates of B in R^m.
        m: The ambient dimension.
        tol: Relative genericity margin.
        max_tries: Number of candidate apices.
        backend: 'auto', 'julia' (cone simplices in parallel threads) or 'python'.

    Returns:
        The linking number.

    Raises:
        UndefinedInvariantError: Unless p + q = m - 1 and p, q >= 1.
        ValueError: If A or B is not a cycle, or contains a flat simplex.
        NonGenericConfigurationError: If B (nearly) touches A, or no apex is generic.

    Example:
        >>> simplicial_linking(polygon_cycle(n), hopf_a, polygon_cycle(n), hopf_b, 3)
        -1
    """
    sA, cA = _chain(cycle_A)
    sB, cB = _chain(cycle_B)
    p, q = sA.shape[1] - 1, sB.shape[1] - 1
    if p + q != m - 1:
        raise UndefinedInvariantError(
            f"linking needs p + q = m - 1; have p={p}, q={q}, m={m}. "
            + ("It is identically zero at this codimension." if p + q < m - 1
               else "It is not defined here.")
        )
    if p < 1 or q < 1:
        raise UndefinedInvariantError("p, q >= 1 here; the p = 0 case is `winding_number`")
    if not is_cycle(cycle_A) or not is_cycle(cycle_B):
        raise ValueError("A and B must be cycles (d = 0, checked exactly)")
    XA = np.asarray(coords_A, dtype=np.float64)
    XB = np.asarray(coords_B, dtype=np.float64)
    if XA.shape[1] != m or XB.shape[1] != m:
        raise ValueError(f"coordinates must be in R^{m}")
    VA = XA[sA]
    VB = XB[sB]
    _check_nondegenerate(VA, "cycle A")
    _check_nondegenerate(VB, "cycle B")
    scale = float(max(np.ptp(XA, 0).max(), np.ptp(XB, 0).max())) or 1.0
    centre = XA[np.unique(sA)].mean(0)
    rng = np.random.default_rng(_GENERIC_SEED)
    apices = [centre + 0.05 * scale * (k + 1) * rng.normal(size=m) / np.sqrt(m) for k in range(max_tries)]
    use_julia, bnorm = _use_julia(backend)
    for o in apices:
        res = None
        if use_julia:
            try:
                res = julia_engine.cone_intersection_count(VA, cA, VB, cB, o, tol, scale)
            except Exception as e:  # pragma: no cover - depends on the Julia runtime
                if bnorm == "julia":
                    raise
                warnings.warn(f"Julia cone intersection failed ({e!r}); falling back to Python.")
                use_julia = False
        if res is None:
            res = _cone_count_python(VA, cA, VB, cB, o, tol, scale, p)
        total, status = res
        if status == 2:
            raise NonGenericConfigurationError(
                "B passes (numerically) through A: the cycles are not disjoint, so their "
                "linking number is undefined"
            )
        if status == 0:
            return int(total)
    raise NonGenericConfigurationError(f"no generic cone apex among {max_tries} tries")


def polygon_cycle(n: int, offset: int = 0) -> Chain:
    """The closed polygon on vertices ``offset .. offset+n-1`` as a simplicial 1-cycle.

    Args:
        n: Number of vertices (>= 3).
        offset: Label of the first vertex.

    Returns:
        ``[((offset+i, offset+i+1), 1), ..., ((offset, offset+n-1), -1)]``.
    """
    return [((offset + i, offset + (i + 1) % n) if i < n - 1 else (offset, offset + n - 1),
             1 if i < n - 1 else -1) for i in range(n)]


def curve_linking(c1: np.ndarray, c2: np.ndarray, backend: str = "auto") -> int:
    """Exact linking number of two disjoint closed polygons in R^3.

    Args:
        c1: ``(n1, 3)`` vertices of the first polygon, in order.
        c2: ``(n2, 3)`` vertices of the second polygon, in order.
        backend: 'auto', 'julia' or 'python'.

    Returns:
        The linking number.
    """
    c1 = np.asarray(c1, dtype=np.float64)
    c2 = np.asarray(c2, dtype=np.float64)
    return simplicial_linking(polygon_cycle(len(c1)), c1, polygon_cycle(len(c2)), c2, 3,
                              backend=backend)


# ------------------------------------------------- the Gauss integral, approximately


def sphere_volume(d: int) -> float:
    """Surface area of the unit sphere ``S^(d-1)`` in R^d."""
    return float(2 * np.pi ** (d / 2) / np.exp(gammaln(d / 2)))


def gauss_linking_estimate(
    cycle_A: Chain, coords_A: np.ndarray, cycle_B: Chain, coords_B: np.ndarray, m: int,
    chunk: int = 48,
) -> float:
    """The generalized Gauss linking integral by the midpoint rule -- an APPROXIMATION.

    It converges to ``simplicial_linking`` as the simplices shrink relative to the
    distance between the cycles, and has the same sign convention. Kept because it is a
    smooth function of the coordinates.

    Args:
        cycle_A: A p-cycle.
        coords_A: Its coordinates in R^m.
        cycle_B: A q-cycle.
        coords_B: Its coordinates in R^m.
        m: Ambient dimension, p + q = m - 1.
        chunk: Batch size.

    Returns:
        The estimate (a float).

    Raises:
        UndefinedInvariantError: Unless p + q = m - 1.
    """
    sA, cA = _chain(cycle_A)
    sB, cB = _chain(cycle_B)
    p, q = sA.shape[1] - 1, sB.shape[1] - 1
    if p + q != m - 1:
        raise UndefinedInvariantError(f"linking needs p + q = m - 1; have p={p}, q={q}, m={m}")
    VA = np.asarray(coords_A, dtype=np.float64)[sA]
    VB = np.asarray(coords_B, dtype=np.float64)[sB]
    cs, es = VA.mean(1), VA[:, 1:, :] - VA[:, :1, :]
    ct, et = VB.mean(1), VB[:, 1:, :] - VB[:, :1, :]
    total = 0.0
    for lo in range(0, len(cs), chunk):
        hi = min(lo + chunk, len(cs))
        diff = cs[lo:hi, None, :] - ct[None, :, :]
        r = np.linalg.norm(diff, axis=2)
        M = np.empty((hi - lo, len(ct), m, m))
        M[:, :, 0, :] = diff
        M[:, :, 1:1 + p, :] = es[lo:hi, None]
        M[:, :, 1 + p:, :] = et[None]
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.linalg.det(M) / r ** m
        term[r < 1e-300] = 0.0
        total += float((cA[lo:hi, None] * cB[None, :] * term).sum())
    return total / sphere_volume(m) / (factorial(p) * factorial(q))


def curve_gauss_integral(c1: np.ndarray, c2: np.ndarray) -> float:
    """``(1/4pi) sum (x - y).(dx x dy)/|x - y|^3`` over segment midpoints (approximate).

    Args:
        c1: ``(n1, 3)`` polygon.
        c2: ``(n2, 3)`` polygon.

    Returns:
        The estimate of the classical Gauss linking integral.
    """
    c1 = np.asarray(c1, dtype=np.float64)
    c2 = np.asarray(c2, dtype=np.float64)
    dx = np.roll(c1, -1, axis=0) - c1
    dy = np.roll(c2, -1, axis=0) - c2
    xm, ym = c1 + dx / 2, c2 + dy / 2
    d = xm[:, None, :] - ym[None, :, :]
    num = np.einsum("ijk,ijk->ij", d, np.cross(dx[:, None, :], dy[None, :, :]))
    return float((num / np.linalg.norm(d, axis=2) ** 3).sum() / (4 * np.pi))
