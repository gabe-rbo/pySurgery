r"""Enclosure: does one point set sit inside the region a complex bounds?

Overview:
    A different question from separability. Separability asks for a hyperplane;
    enclosure asks whether a point lies in a BOUNDED component of the complement of a
    complex ``|K_A|`` in R^m. A shell around a ball is perfectly non-separable by any
    linear rule and perfectly enclosed.

Key Concepts:
    - **Codimension is the whole story.** Jordan-Brouwer separation needs a
      hypersurface: m = n + 1. At m > n + 1 the complement of A is connected and every
      point has winding number 0 identically (``geometric_linking.definedness``).
    - **Exact, on the complex.** By Alexander duality the bounded components of
      ``R^m - |K_A|`` are detected by ``H_(m-1)(K_A)``: a point x is enclosed iff some
      (m-1)-cycle of K_A has nonzero winding number around x (the pairing
      ``H_(m-1)(K) x H~_0(R^m - K) -> Z`` is non-degenerate), whatever spanning set of
      cycles is used -- boundaries wind zero around points off ``|K_A|``. Two
      estimators, on different mathematics, required to agree:

        degree   the integer winding number of x around each generator, by one certified
                 generic ray (``knots.geometric_linking.winding_numbers``).
        parity   the mod-2 count of that ray's crossings with every (m-1)-simplex -- the
                 even-odd rule. Needs no homology and no orientation, but means something
                 only on a Z/2 cycle. It is the Z/2 reduction of the SUM of the degrees, so
                 inside nested shells it alternates: a point inside two spheres has
                 degrees (1, 1) and parity 0. Disagreement there is expected.

      The verdict is the degree; disagreements are reported point by point, never
      silently resolved.
    - **The convex hull** is the one check that survives in any codimension: the region
      a closed hypersurface bounds lies in its convex hull, so "enclosed => in the
      hull". Necessary, never sufficient (the centre of a torus's hole is in the hull and
      not enclosed).

Common Workflows:
    1. **Points inside a triangulated surface in R^3** ->
       ``enclosure_report(points, K_A, coords_A, 3)``.
    2. **A necessary condition in any dimension** -> ``in_convex_hull(X_A, points)``.

Coefficient Ring:
    Z for the degree, Z/2 for the parity.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, List, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..knots import geometric_linking as GL

if TYPE_CHECKING:  # pragma: no cover
    from ..topology.complexes import SimplicialComplex

__all__ = [
    "in_convex_hull",
    "hull_fraction",
    "integer_cycle_generators",
    "winding_against_all",
    "is_mod2_cycle",
    "crossing_parity",
    "EnclosureCertificate",
    "enclosure_report",
]


def in_convex_hull(X_A: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Whether each point lies in the convex hull of ``X_A``.

    A feasibility LP per point (HiGHS, feasibility tolerance ~1e-9 -- a point within
    that distance of the hull's boundary can go either way).

    Args:
        X_A: ``(n, m)`` points spanning the hull.
        points: ``(k, m)`` query points.

    Returns:
        A boolean array, one entry per query point.
    """
    from scipy.optimize import linprog

    X_A = np.asarray(X_A, dtype=np.float64)
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    n = len(X_A)
    A_eq = np.vstack([X_A.T, np.ones(n)])
    out = np.zeros(len(points), dtype=bool)
    for i, p in enumerate(points):
        res = linprog(np.zeros(n), A_eq=A_eq, b_eq=np.concatenate([p, [1.0]]),
                      bounds=(0, None), method="highs")
        out[i] = bool(res.status == 0)
    return out


def hull_fraction(X_A: np.ndarray, X_B: np.ndarray) -> float:
    """Fraction of the points of B inside ``conv(A)`` (every point of B is tested).

    Args:
        X_A: Points spanning the hull.
        X_B: Query points.

    Returns:
        The fraction in [0, 1].
    """
    return float(np.mean(in_convex_hull(X_A, X_B)))


def _integer_kernel_basis(K: "SimplicialComplex", p: int) -> List[List]:
    """A Z-basis of the cycle group ``Z_p(K) = ker d_p`` by exact dense SNF (cubic)."""
    from ..algebra.math_core import smith_normal_decomp

    simplices = list(K.n_simplices(p))
    if p == 0:
        return [[(s, 1)] for s in simplices]
    A = np.asarray(K.boundary_matrix(p).toarray(), dtype=object)
    S, _, Vm = smith_normal_decomp(A, compute_u=False, compute_v=True)
    # S = U A V with the nonzero invariant factors first, so A V e_j = 0 exactly for
    # j >= rank, and V unimodular makes those columns a Z-basis of ker A.
    rank = sum(1 for i in range(min(S.shape)) if S[i, i] != 0)
    out = []
    for j in range(rank, len(simplices)):
        col = Vm[:, j]
        out.append([(tuple(simplices[i]), int(col[i])) for i in range(len(simplices)) if col[i] != 0])
    return out


def integer_cycle_generators(K: "SimplicialComplex", p: int) -> List[List]:
    """Integer p-cycles spanning ``H_p(K; Z)`` modulo torsion, each as ``[(simplex, c)]``.

    What is Being Computed?:
        A closed p-pseudomanifold with nothing above dimension p gets its fundamental
        cycles (one per orientable strong component; a breadth-first search, any size).
        Anything else gets a Z-basis of the cycle group ``ker d_p`` from the exact dense
        Smith normal form (cost cubic in the number of p-simplices). The cycles span
        ``H_p`` together with boundaries, which have winding number 0 around every point
        off ``|K|`` -- so they decide enclosure exactly.

    Args:
        K: A simplicial complex.
        p: The degree.

    Returns:
        The cycles.
    """
    from ..core.exceptions import NoFundamentalClassError
    from ..topology.fundamental_cycles import top_homology_basis

    if p < 0 or p > K.dimension:
        return []
    if K.dimension == p and p >= 1:
        try:
            return [c.as_pairs() for c in top_homology_basis(K, p)]
        except NoFundamentalClassError:
            pass  # not a closed pseudomanifold: fall back to the exact SNF kernel
    return _integer_kernel_basis(K, p)


def winding_against_all(
    points: np.ndarray, cycles: List[list], coords: np.ndarray, backend: str = "auto"
) -> np.ndarray:
    """Winding number of each point against each cycle: shape ``(n_points, n_cycles)``.

    Args:
        points: Query points in R^m.
        cycles: (m-1)-cycles.
        coords: Vertex coordinates.
        backend: 'auto', 'julia' or 'python'.

    Returns:
        An integer matrix.
    """
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if not cycles:
        return np.zeros((len(points), 0), dtype=np.int64)
    return np.stack([GL.winding_numbers(points, cyc, coords, backend=backend) for cyc in cycles], axis=1)


def is_mod2_cycle(K: "SimplicialComplex", m: int, simplices=None) -> bool:
    """Do the (m-1)-simplices (all of K's, or the given ones) sum to a cycle over Z/2?

    Args:
        K: A simplicial complex.
        m: The ambient dimension.
        simplices: Optional subset of (m-1)-simplices.

    Returns:
        True iff every (m-2)-face is shared by an even number of them.
    """
    simps = K.n_simplices(m - 1) if simplices is None else [tuple(s) for s in simplices]
    cnt: Counter = Counter()
    for s in simps:
        for i in range(len(s)):
            cnt[tuple(s[:i]) + tuple(s[i + 1:])] += 1
    return bool(cnt) and all(v % 2 == 0 for v in cnt.values())


def crossing_parity(
    points: np.ndarray, simplices: list, coords: np.ndarray, backend: str = "auto"
) -> np.ndarray:
    """The even-odd rule: crossings of one certified generic ray, mod 2.

    Meaningful only when the simplices form a Z/2 cycle (``is_mod2_cycle``).

    Args:
        points: Query points.
        simplices: The (m-1)-simplices.
        coords: Vertex coordinates.
        backend: 'auto', 'julia' or 'python'.

    Returns:
        0/1 per point.
    """
    chain = [(tuple(s), 1) for s in simplices]
    # the signed sum of +-1 over the hit simplices has the parity of their number
    w = GL.winding_numbers(points, chain, coords, backend=backend, require_cycle=False)
    return (np.asarray(w) % 2).astype(np.int64)


class EnclosureCertificate(BaseModel):
    """Result of ``enclosure_report``.

    Attributes:
        defined (bool): Whether the question has an answer here.
        reason (str): Why (or why not).
        m (int): Ambient dimension.
        n_bounding_classes (int): ``rank H_(m-1)(K_A)`` -- the number of bounded regions.
        winding (np.ndarray): ``(n_points, n_generators)`` winding numbers.
        parity (np.ndarray): Even-odd parity per point.
        enclosed (np.ndarray): The verdict, from the degree.
        agreement (float): Fraction of points where degree and parity agree.
        enclosed_fraction (float): Fraction of points enclosed.
        complex_certified (bool | None): K_A certified a closed homology (m-1)-manifold.
        parity_independent (bool): False when parity had to use the generators' support.
        disputed (np.ndarray | None): Points where degree and parity disagree.
        notes (list[str]): Human-readable notes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    defined: bool
    reason: str
    m: int
    n_bounding_classes: int
    winding: np.ndarray
    parity: np.ndarray
    enclosed: np.ndarray
    agreement: float
    enclosed_fraction: float
    complex_certified: Optional[bool] = None
    parity_independent: bool = True
    disputed: Optional[np.ndarray] = None
    notes: List[str] = Field(default_factory=list)

    def __str__(self) -> str:
        if not self.defined:
            return f"enclosure UNDEFINED -- {self.reason}"
        cert = {True: "; complex certified a closed homology manifold",
                False: "; complex NOT certified", None: ""}[self.complex_certified]
        return (
            f"enclosure in R^{self.m}: {self.enclosed_fraction:.1%} of points enclosed by "
            f"{self.n_bounding_classes} bounding cycle(s); degree/parity agreement "
            f"{self.agreement:.1%}" + ("" if self.parity_independent else " (parity not independent here)")
            + cert + "".join("\n  note: " + n for n in self.notes)
        )


def _undefined(reason: str, m: int, n: int) -> EnclosureCertificate:
    return EnclosureCertificate(
        defined=False, reason=reason, m=m, n_bounding_classes=0,
        winding=np.zeros((n, 0), dtype=np.int64), parity=np.zeros(n, dtype=np.int64),
        enclosed=np.zeros(n, dtype=bool), agreement=0.0, enclosed_fraction=0.0,
    )


def enclosure_report(
    points_B: np.ndarray,
    K_A: "SimplicialComplex",
    coords_A: np.ndarray,
    m: int,
    certify: bool = True,
    backend: str = "auto",
) -> EnclosureCertificate:
    """Are the points of B inside the region bounded by the complex K_A, in R^m?

    What is Being Computed?:
        For every point of B, its winding number around every generator of
        ``H_(m-1)(K_A)`` (the verdict: enclosed iff some winding is nonzero) and the
        even-odd crossing parity (an independent check), with every disagreement listed.

    Args:
        points_B: ``(k, m)`` query points.
        K_A: The complex of the enclosing class, with vertex coordinates ``coords_A``.
        coords_A: ``coords_A[v]`` in R^m.
        m: The ambient dimension.
        certify: Also certify K_A as a closed homology (m-1)-manifold (every simplex).
        backend: 'auto', 'julia' or 'python'.

    Returns:
        An ``EnclosureCertificate``; ``defined=False`` (with the reason) when the
        coordinates are not m-dimensional or ``beta_(m-1)(K_A) = 0`` -- then A bounds no
        region, and "nothing is enclosed" would hide the reason.
    """
    points_B = np.atleast_2d(np.asarray(points_B, dtype=np.float64))
    coords_A = np.asarray(coords_A, dtype=np.float64)
    nB = len(points_B)
    if coords_A.shape[1] != m or points_B.shape[1] != m:
        return _undefined(
            f"coordinates are {coords_A.shape[1]}- and {points_B.shape[1]}-dimensional, but m = {m}", m, nB
        )
    b_top = int(K_A.homology(m - 1, backend=backend)[0]) if K_A.dimension >= m - 1 else 0
    if b_top == 0:
        return _undefined(f"beta_{m - 1} of the complex on A is 0: it bounds no region in R^{m}", m, nB)
    notes: List[str] = []
    cert_ok = None
    if certify:
        from ..topology.local_homology import certify_homology_manifold

        mc = certify_homology_manifold(K_A, m - 1, backend=backend)
        cert_ok = bool(mc.is_closed_homology_manifold)
        if not cert_ok:
            notes.append(
                f"A's complex is not a closed homology {m - 1}-manifold ({len(mc.singular)} "
                f"singular, {len(mc.boundary)} boundary simplices); the degree is still exact, "
                f"but the parity rule may disagree on a singular complex"
            )
    cycles = integer_cycle_generators(K_A, m - 1)
    w = winding_against_all(points_B, cycles, coords_A, backend=backend)
    independent = is_mod2_cycle(K_A, m)
    if independent:
        par = crossing_parity(points_B, K_A.n_simplices(m - 1), coords_A, backend=backend)
    else:
        support = sorted({tuple(sorted(s)) for cyc in cycles for s, c in cyc if c % 2})
        if support and is_mod2_cycle(K_A, m, support):
            notes.append(
                f"K_A's {m - 1}-simplices are not a Z/2 cycle (it has {m}-simplices or "
                f"branching), so the parity rule ran on the generators' odd support"
            )
            par = crossing_parity(points_B, support, coords_A, backend=backend)
        else:
            notes.append("no Z/2 cycle to run the parity rule on; parity reported as 0")
            par = np.zeros(nB, dtype=np.int64)
    by_degree = np.any(w != 0, axis=1)
    disputed = by_degree != par.astype(bool)
    agree = float(np.mean(~disputed)) if nB else 1.0
    if agree < 1.0:
        notes.append(
            f"degree and parity disagree on {int(disputed.sum())} of {nB} points; the verdict "
            f"is the degree, the points are listed in `disputed`"
        )
    return EnclosureCertificate(
        defined=True, reason="codimension 1 -- Jordan-Brouwer applies", m=m,
        n_bounding_classes=b_top, winding=w, parity=par, enclosed=by_degree, agreement=agree,
        enclosed_fraction=float(np.mean(by_degree)) if nB else 0.0, complex_certified=cert_ok,
        parity_independent=independent, disputed=disputed, notes=notes,
    )
