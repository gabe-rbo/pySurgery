"""The alpha complex without the Delaunay triangulation (dual active-set QP, GPU).

Overview:
    ``SimplicialComplex.from_alpha_complex`` builds the alpha complex the classical
    way: triangulate with ``scipy.spatial.Delaunay`` and filter. The full Delaunay
    complex has on the order of ``N^ceil(m/2)`` simplices, so that route stops at an
    ambient dimension of about 6. This module implements the algorithm of Carlsson
    & Carlsson, *Computing the alpha complex using dual active set quadratic
    programming* (arXiv:2310.00536), which never builds the Delaunay complex: the
    membership of every candidate simplex is decided by a small convex quadratic
    program attacked through its Lagrangian **dual**, batched on the GPU. The
    implementation is the one validated in TabularTopology
    (``topology/complexes/alpha.py``), wired into pySurgery's complexes.

    Whether ``sigma`` is in ``Alpha(S, r)`` asks "is there a point equidistant (in
    power) from the vertices of ``sigma``, no closer to any other point, and within
    ``r`` of them?". The dual is the right side to attack for two reasons:

    * any dual-feasible ``lambda`` gives a **lower bound** on the primal optimum, so
      the solver stops the moment the bound passes the cutoff -- it rules simplices
      *out*, and ruling out is the common case;
    * ``lambda = 0`` is always dual feasible, so there is no warm-start cost.

    And the property that matters in high dimension (their Section 4.4): the dual
    sees only the Gram matrix ``B`` of the neighbour differences, never the
    coordinates. The ambient width ``m`` enters once per vertex per dimension, when
    ``B`` is formed.

Key Concepts:
    - **Notation (weighted setting)**: points ``S`` in ``R^m``, power ``p: S -> R``,
      cutoff ``a1`` (the unweighted alpha complex at radius ``r`` is ``p = 0``,
      ``a1 = r^2``). For a vertex ``x`` with Cech neighbours ``x_1..x_n``::

          B_ij = (x_i - x)^T (x_j - x)
          U_i  = (p(x_i) - p(x) - |x_i - x|^2) / 2
          c1   = (a1 + p(x)) / 2

      For ``sigma = [x, x_j1, ..., x_jk]`` with ``J = {j1..jk}`` the dual program is
      ``max -1/2 l^T B l + U^T l`` subject to ``l_i >= 0`` for ``i`` not in ``J``.
      Its optimum ``c*`` decides membership (``c* <= c1``), the filtration value
      ``w(sigma) = 2 c* - p(x)`` (a squared radius), and the witness
      ``Phi(sigma) = x - sum_i l*_i (x_i - x)`` (the KKT conditions, eq. 12).
    - **Lazy candidates**: at dimension ``k`` only the ``k``-simplices all of whose
      facets are already accepted are tested (the paper's ``Lazy_{k-1}(X)``); each
      candidate is generated once and tested at its smallest vertex.
    - **Nerve, not triangulation**: on degenerate input (grids, co-spherical points)
      the result is the nerve of the radius-restricted Voronoi cells and may contain
      simplices above the ambient dimension; by the nerve lemma it is homotopy
      equivalent to the union of balls in every case.
    - **Alpha is inside Delaunay-Cech**: ``Alpha(S, r)`` is contained in the
      Delaunay-Cech complex at ``r``, with the same homotopy type, so the two agree
      on Betti numbers, not simplex for simplex.

Exactness:
    Floating point is used only where a verdict has a margin; rational arithmetic
    decides the rest. Every verdict of the float64 solver stands only if it clears a
    margin measured against the size of the terms it was computed from (a witness
    not within ``1e-9`` of the radius, no point outside ``sigma`` within ``1e-9`` of
    being as close as ``sigma``'s vertices, a well-conditioned solve, no rank
    decision). Ties, radii exactly at an alpha value, flat or degenerate simplices
    and anything the active set could not resolve go to :func:`exact_decide`, the
    same dual active set run in exact rational arithmetic on the input floats taken
    at face value -- the role CGAL's exact predicates play for Delaunay. Lower-bound
    rejections need no margin beyond their own evaluation error: the dual objective
    bounds ``c*`` at *any* dual-feasible point.

Devices:
    The Cech graph (one big pairwise-distance pass, genuinely parallel) runs on the
    automatically selected device (:func:`pysurgery.gpu.device.resolve_device`). The
    QP runs there too on CUDA and the CPU. **MPS has neither float64 nor a
    ``torch.linalg.eigh`` kernel**, so on Apple Silicon the graph runs on the GPU in
    float32 -- with a rigorously widened threshold, so no true edge is ever lost --
    and the QP runs on the CPU in float64. ``qp_device`` overrides that split.
    float32, wherever it is used for the QP, is only ever allowed to *screen out*
    candidates on a lower bound that clears its own evaluation error; every
    acceptance is re-decided in float64 from the original coordinates.

Common Workflows:
    1. **Alpha complex at a radius** -> ``dual_alpha_complex(points, r).complex``.
    2. **Alpha filtration up to a radius** -> ``res = dual_alpha_complex(points, r_max)``;
       ``res.filtration_values()`` holds every simplex's alpha value and
       ``res.subcomplex(r)`` is ``Alpha(S, r)`` for any ``r <= r_max`` (ties at
       ``r`` re-decided exactly).
    3. **From a complex** -> ``SimplicialComplex.from_dual_alpha_complex(points, r)``
       or ``SimplicialComplex.from_alpha_complex(points, r, backend="gpu")``.
    4. **Filtration report** -> ``DualAlphaFiltrationReport(points, eps_max=r_max)``.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .device import require_torch, resolve_device

torch = require_torch()

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pysurgery.topology.complexes import SimplicialComplex

Simplex = Tuple[int, ...]

__all__ = [
    "dual_alpha_complex",
    "DualAlphaResult",
    "connectivity_radius",
    "exact_decide",
    "kkt_report",
    "primal_is_feasible",
    "AlphaUndecided",
    "AlphaBudgetExceeded",
]


# ──────────────────────────────────────────────────────────────────────────────
# Errors
# ──────────────────────────────────────────────────────────────────────────────
class AlphaBudgetExceeded(RuntimeError):
    """The candidate set outgrew ``max_simplices`` (almost always: the radius is too large)."""


class AlphaUndecided(RuntimeError):
    """A candidate's QP could not be decided and no exact solver was available.

    Raised instead of guessing. Dropping an undecided candidate is not a
    conservative choice: a missing simplex changes homology exactly as much as a
    spurious one, so there is no safe direction to fail in. The builder always
    supplies the exact solver, so this surfaces only from the internal helpers.
    """


# ──────────────────────────────────────────────────────────────────────────────
# Result
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class DualAlphaResult:
    """The alpha complex at a cap radius, with its whole filtration below the cap.

    Overview:
        ``weights[sigma]`` is the alpha value of ``sigma`` (a *squared* radius, the
        paper's ``w(sigma) = 2 c* - p(x)``), so ``{sigma : weights[sigma] <= r^2}``
        is ``Alpha(S, r)`` for every ``r <= radius``: one build at the cap is the
        entire filtration on ``[0, radius]``.

    Attributes:
        complex: The alpha complex at ``radius`` as a pySurgery
            :class:`~pysurgery.topology.complexes.SimplicialComplex`, with
            ``filtration`` set to the alpha radii (unweighted input) and the input
            coordinates attached.
        weights: Alpha value (squared radius) of every simplex.
        witnesses: The witness point ``Phi(sigma)`` of every simplex (empty when
            built with ``keep_witness=False``).
        radius: The cap radius the complex was built at.
        max_dim: The dimension cap, or ``None`` if every dimension was built.
        points: The input coordinates, ``(N, m)`` float64.
        power: The power (weight) of every point, or ``None`` (unweighted).
        adjacency: Neighbour lists of the Cech graph at the cap (a certified
            superset of the true Cech graph), used for exact re-decisions.
        device: Device the QP ran on.
        graph_device: Device the Cech graph ran on.
        dtype: Working precision of the QP.
        stats: Counters describing how each verdict was reached.
    """

    complex: "SimplicialComplex"
    weights: Dict[Simplex, float]
    witnesses: Dict[Simplex, np.ndarray]
    radius: float
    max_dim: Optional[int]
    points: np.ndarray
    power: Optional[np.ndarray]
    adjacency: List[np.ndarray] = field(repr=False)
    device: str = "cpu"
    graph_device: str = "cpu"
    dtype: str = "float64"
    stats: dict = field(default_factory=dict)

    @property
    def truncation_dim(self) -> Optional[int]:
        """The dimension cap (``None`` when the complex was built in every dimension)."""
        return self.max_dim

    def exact_betti_dimensions(self) -> range:
        """Degrees whose Betti numbers the (possibly truncated) complex determines.

        A complex capped at dimension ``d`` is missing the ``(d+1)``-simplices that
        would kill ``d``-cycles, so ``beta_d`` of the capped complex can be inflated;
        ``beta_k`` for ``k < d`` is exact.

        Returns:
            ``range(0, d)`` for a cap ``d``, or every degree of the complex.
        """
        if self.max_dim is None:
            return range(0, max(self.complex.dimension, 0) + 1)
        return range(0, int(self.max_dim))

    def size_vector(self) -> List[int]:
        """Number of simplices in each dimension ``0..dim``."""
        top = self.complex.dimension
        return [len(self.complex.n_simplices(d)) for d in range(top + 1)] if top >= 0 else []

    def filtration_values(self, squared: bool = False) -> Dict[Simplex, float]:
        """Alpha value of every simplex.

        Args:
            squared: Return squared radii (the native weights) instead of radii.

        Returns:
            Dict mapping each simplex to its alpha value; monotone under faces.

        Raises:
            ValueError: If radii are requested for a weighted complex with negative
                weights (which have no real square root).
        """
        if squared:
            return dict(self.weights)
        neg = [s for s, w in self.weights.items() if w < 0]
        if neg:
            raise ValueError(
                f"{len(neg)} simplices have negative power (e.g. {neg[0]}); a weighted "
                "alpha filtration is only expressible as squared radii (squared=True)."
            )
        return {s: float(np.sqrt(w)) for s, w in self.weights.items()}

    def subcomplex(self, radius: float, *, exact: bool = True,
                   coefficient_ring: Optional[str] = None) -> "SimplicialComplex":
        """``Alpha(S, radius)`` for any ``radius`` up to the cap, read off the weights.

        What is Being Computed?:
            The simplices whose alpha value is at most ``radius^2``. Filtration values
            are float64 (measured against CGAL's exact values: relative error at most
            ``1e-11``), so every simplex whose value lies within ``1e-7`` (relative)
            of ``radius^2`` is re-decided by :func:`exact_decide` at ``radius``, using
            the cap's Cech neighbours (a point that is not a neighbour at the cap is
            too far to constrain the Voronoi face at any smaller radius). Verdicts at
            the cap radius itself are certified; for a certified complex at another
            radius without the band argument, build ``dual_alpha_complex(points, r)``.

        Args:
            radius: The radius; must not exceed the cap radius.
            exact: Re-decide near-ties in exact rational arithmetic (default True).
            coefficient_ring: Ring label of the returned complex (defaults to the
                cap complex's ring).

        Returns:
            The sub-complex at ``radius`` (closed under faces).

        Raises:
            ValueError: If ``radius`` exceeds the cap radius.
        """
        from pysurgery.topology.complexes import SimplicialComplex

        r = float(radius)
        if r > self.radius * (1.0 + 1e-15):
            raise ValueError(
                f"radius {r} is above the cap {self.radius} this filtration was built at; "
                "rebuild with a larger radius."
            )
        thr = r * r
        keep: Dict[Simplex, bool] = {}
        for s, w in self.weights.items():
            keep[s] = w <= thr
        if exact:
            from fractions import Fraction
            r_q2 = Fraction(r) ** 2
            for s, w in self.weights.items():
                if abs(w - thr) > 1e-7 * max(abs(thr), abs(w), 1e-300):
                    continue
                if len(s) == 1:
                    pw = Fraction(0) if self.power is None else Fraction(float(self.power[s[0]]))
                    keep[s] = -pw <= r_q2
                    continue
                nb = self.adjacency[s[0]]
                pos = {int(v): i for i, v in enumerate(nb)}
                e = exact_decide(self.points, self.power, s[0], nb, [pos[v] for v in s[1:]], r)
                keep[s] = bool(e["accept"])
        table: Dict[int, List[Simplex]] = {}
        for s, ok in keep.items():
            if ok:
                table.setdefault(len(s) - 1, []).append(s)
        # an exact filtration is monotone, so the kept set is closed under faces;
        # enforce it anyway so a complex is always returned
        for d in sorted(table)[1:]:
            lower = set(table.get(d - 1, ()))
            table[d] = [s for s in table[d]
                        if all(f in lower for f in itertools.combinations(s, d))]
        table = {d: sorted(v) for d, v in table.items() if v}
        ring = coefficient_ring or self.complex.coefficient_ring
        sc = SimplicialComplex(simplices=table, coefficient_ring=ring)
        sc._coordinates = self.points
        return sc

    def __str__(self) -> str:
        s = self.stats
        return (f"Alpha complex on {s.get('n_points', '?')} points in R^{s.get('m', '?')}, "
                f"radius {self.radius:.4g}\n"
                f"  sizes      {self.size_vector()}\n"
                f"  device     graph {self.graph_device}, QP {self.device} / {self.dtype}\n"
                f"  candidates {s.get('n_candidates', 0)} tested, "
                f"{s.get('n_accepted', 0)} accepted\n"
                f"  resolved   {s.get('n_equality_only', 0)} by the equality phase, "
                f"{s.get('n_active_set', 0)} needed the active set, "
                f"{s.get('n_exact_fallback', 0)} went to the float64 CPU authority, "
                f"{s.get('n_exact_rational', 0)} were decided in exact rational arithmetic")


# ──────────────────────────────────────────────────────────────────────────────
# The Cech graph (a certified superset)
# ──────────────────────────────────────────────────────────────────────────────
def _gamma(k: int, u: float) -> float:
    """Higham's ``gamma_k = k u / (1 - k u)``: the rounding bound of a k-term sum."""
    ku = k * u
    return ku / (1.0 - ku) if ku < 1.0 else float("inf")


def _cech_graph(P: np.ndarray, a1: float, power: np.ndarray, device,
                dtype, chunk_bytes: int = 64 * 2**20) -> List[np.ndarray]:
    """Neighbour lists of a certified SUPERSET of ``Skel_1(Cech(S, p, a1))``.

    ``U_x`` is the ball of radius ``sqrt(a1 + p(x))`` about ``x`` (empty when
    ``a1 + p(x) < 0``), and ``x ~ y`` iff ``|x - y| <= rad(x) + rad(y)``. A
    neighbour wrongly left out would drop its Voronoi constraint and every
    candidate simplex through it, so the test is widened by an a-priori bound on
    every rounding error involved; a neighbour wrongly let in costs only a
    redundant constraint and candidates the QP then rejects.

    The bound covers: centering the cloud (one float64 rounding per coordinate),
    casting it to ``dtype`` (float32 on MPS), and the distance evaluation itself --
    via the Gram matrix in float64 (``|x|^2 + |y|^2 - 2 x.y``, error at most
    ``(2 gamma_m + 8u)(|x|^2 + |y|^2)``) or via direct differences in float32
    (relative error at most ``gamma_{m+2}``). Centering matters: without it the
    Gram formula's error scales with the distance to the origin, not the spacing.

    Args:
        P: ``(N, m)`` float64 coordinates (in the solver's units).
        a1: The power cutoff (a squared radius).
        power: ``(N,)`` float64 powers.
        device: Device for the distance pass.
        dtype: ``torch.float64`` or ``torch.float32``.
        chunk_bytes: Memory cap per chunk of rows.

    Returns:
        One sorted ``int64`` array of neighbours per point (symmetric relation).
    """
    N, m = P.shape
    if N == 0:
        return []
    rad2 = a1 + power
    alive = rad2 >= 0
    rad = np.sqrt(np.clip(rad2, 0.0, None))
    Pc = P - P.mean(axis=0)
    A = float(np.abs(Pc).max()) if Pc.size else 0.0
    u64 = 2.0 ** -53
    u = float(torch.finfo(dtype).eps) / 2.0
    cast = 0.0 if dtype == torch.float64 else u
    delta = 2.0 * (u64 + cast) * A                  # per-coordinate perturbation
    pert = 2.0 * np.sqrt(m) * delta                 # |d(x~, y~) - d(x, y)|
    use_gram = dtype == torch.float64

    X = torch.as_tensor(Pc, dtype=dtype, device=device)
    rad_t = torch.as_tensor(rad, dtype=dtype, device=device)
    alive_t = torch.as_tensor(alive, device=device)
    widen = 1.0 + 16.0 * u + 16.0 * u64
    if use_gram:
        nrm = (X * X).sum(1)
        g_err = 2.0 * (2.0 * _gamma(m, u) + 8.0 * u)
        per_row = N * 8 * 4
    else:
        g_rel = 1.0 + 2.0 * _gamma(m + 2, u)
        per_row = N * m * 4 * 2
    step = int(max(1, min(N, chunk_bytes // max(per_row, 1))))

    rows_all: List[np.ndarray] = []
    cols_all: List[np.ndarray] = []
    for s in range(0, N, step):
        e = min(s + step, N)
        thr = rad_t[s:e, None] + rad_t[None, :] + pert
        thr2 = thr * thr * widen
        if use_gram:
            d2 = nrm[s:e, None] + nrm[None, :] - 2.0 * (X[s:e] @ X.T)
            hit = d2 <= thr2 + g_err * (nrm[s:e, None] + nrm[None, :])
        else:
            diff = X[s:e, None, :] - X[None, :, :]
            d2 = (diff * diff).sum(-1)
            hit = d2 <= thr2 * g_rel
        hit &= alive_t[s:e, None] & alive_t[None, :]
        loc = torch.arange(e - s, device=device)
        hit[loc, loc + s] = False
        idx = torch.nonzero(hit, as_tuple=False).cpu().numpy()
        if idx.size:
            rows_all.append(idx[:, 0].astype(np.int64) + s)
            cols_all.append(idx[:, 1].astype(np.int64))
    if not rows_all:
        return [np.zeros(0, dtype=np.int64) for _ in range(N)]
    r = np.concatenate(rows_all)
    c = np.concatenate(cols_all)
    # symmetrise: the float evaluation of (i, j) and (j, i) may round differently
    key = np.unique(np.concatenate([r * N + c, c * N + r]))
    rr, cc = key // N, key % N
    return np.split(cc, np.searchsorted(rr, np.arange(1, N)))


# ──────────────────────────────────────────────────────────────────────────────
# Certificates (used by the tests and for audits)
# ──────────────────────────────────────────────────────────────────────────────
def kkt_report(B: np.ndarray, U: np.ndarray, J: Sequence[int], lam: np.ndarray,
               tol: float = 1e-7) -> dict:
    """Check whether ``lam`` is the optimum of ``max -1/2 l^T B l + U^T l``, ``l >= 0`` off ``J``.

    What is Being Computed?:
        A certificate check, not a re-solve -- it shares no machinery with the
        solver it audits. For a convex QP the KKT conditions are necessary and
        sufficient, so three residuals settle the question outright (with
        ``s = -D^T lam``, hence ``B lam = -D s``): primal feasibility
        ``D_J s = -U_J`` and ``D_i s <= -U_i``; dual feasibility ``lam_i >= 0`` off
        ``J``; complementary slackness ``lam_i (D_i s + U_i) = 0``.

    Args:
        B: ``(n, n)`` Gram matrix.
        U: ``(n,)`` linear term.
        J: Indices of the equality constraints.
        lam: Candidate multipliers.
        tol: Relative tolerance (residuals are scaled by the problem's magnitude).

    Returns:
        Dict with ``ok`` and each residual, plus the dual ``value`` at ``lam``.
    """
    n = len(U)
    Jset = set(int(j) for j in J)
    Ds = -(B @ lam)
    resid = Ds + U
    scale = max(float(np.abs(U).max()) if n else 0.0, float(np.abs(Ds).max()) if n else 0.0,
                1e-300)
    eq = float(np.abs(resid[list(Jset)]).max()) if Jset else 0.0
    ineq = float(max(0.0, resid.max())) if n else 0.0
    sign = float(max(0.0, -min([lam[i] for i in range(n) if i not in Jset], default=0.0)))
    slack = float(np.abs(lam * resid).max()) if n else 0.0
    ok = (eq <= tol * scale and ineq <= tol * scale
          and sign <= tol and slack <= tol * scale)
    return dict(ok=bool(ok), equality=eq, inequality=ineq, sign=sign, slack=slack,
                value=float(-0.5 * lam @ B @ lam + U @ lam), scale=scale)


def primal_is_feasible(D: np.ndarray, U: np.ndarray, J: Sequence[int]) -> bool:
    """Decide by linear programming whether any ``s`` has ``D_J s = -U_J`` and ``D_i s <= -U_i``.

    Audits the solver's *infeasible* verdicts, which the KKT check cannot reach (an
    infeasible primal has no optimum to certify). ``scipy.optimize.linprog`` knows
    nothing about alpha complexes, which is the point.

    Args:
        D: ``(n, m)`` neighbour differences ``x_i - x``.
        U: ``(n,)`` linear term.
        J: Indices of the equality constraints.

    Returns:
        True when the Voronoi face is nonempty.
    """
    from scipy.optimize import linprog
    Jset = sorted(set(int(j) for j in J))
    Jc = [i for i in range(len(U)) if i not in set(Jset)]
    m = D.shape[1]
    res = linprog(c=np.zeros(m), A_ub=D[Jc] if Jc else None,
                  b_ub=-U[Jc] if Jc else None,
                  A_eq=D[Jset] if Jset else None, b_eq=-U[Jset] if Jset else None,
                  bounds=[(None, None)] * m, method="highs")
    return bool(res.status == 0)


# ──────────────────────────────────────────────────────────────────────────────
# The exact (rational) solver
# ──────────────────────────────────────────────────────────────────────────────
def _fsolve(A, b):
    """Solve ``A x = b`` exactly (square, Fractions, nonsingular) by Gauss-Jordan."""
    n = len(A)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        piv = next(i for i in range(c, n) if M[i][c] != 0)
        if piv != c:
            M[c], M[piv] = M[piv], M[c]
        inv = 1 / M[c][c]
        M[c] = [v * inv for v in M[c]]
        for i in range(n):
            if i != c and M[i][c] != 0:
                f = M[i][c]
                M[i] = [a - f * bb for a, bb in zip(M[i], M[c])]
    return [M[i][n] for i in range(n)]


def exact_decide(points: np.ndarray, power: Optional[np.ndarray], x: int,
                 nbrs: Sequence[int], J: Sequence[int], radius: float,
                 max_iter: Optional[int] = None) -> dict:
    """Decide EXACTLY whether ``sigma = {x} + {nbrs[j] : j in J}`` is in ``Alpha(points, radius)``.

    What is Being Computed?:
        The same dual active set (Goldfarb-Idnani) as the batched solver, run in
        rational arithmetic on the float64 input taken at face value (every float
        is a rational), so every test -- a residual's sign, a dependence
        (``|z|^2 == 0``), the comparison with the radius -- is decided exactly,
        ties included. It is the authority for the verdicts the floating-point
        path cannot make with a margin. Goldfarb-Idnani terminates for a strictly
        convex primal, which this is (``min |y - x|^2 / 2``), so ``max_iter`` is
        only a guard.

    Args:
        points: ``(N, m)`` input coordinates.
        power: ``(N,)`` powers, or ``None`` for the unweighted complex.
        x: The vertex the QP is centred on (any vertex of ``sigma``).
        nbrs: Indices of the points whose Voronoi constraints are imposed.
        J: Positions in ``nbrs`` of the other vertices of ``sigma``.
        radius: The radius (``a1 = radius^2``).
        max_iter: Iteration guard.

    Returns:
        Dict ``(accept, c, lam, reason)``: ``c`` is ``c*`` (or the lower bound that
        ruled ``sigma`` out) as a Fraction in the input's units, ``lam`` the
        multipliers over ``nbrs``.

    Raises:
        AlphaUndecided: If the iteration guard is hit (it must not be).
    """
    from fractions import Fraction as Q
    P = np.asarray(points, dtype=np.float64)
    m = P.shape[1]
    nb = [int(v) for v in nbrs]
    n = len(nb)
    fx = [Q(float(v)) for v in P[x]]
    D = [[Q(float(P[j, k])) - fx[k] for k in range(m)] for j in nb]
    if power is None:
        pw_x, pw = Q(0), [Q(0)] * n
    else:
        pw_x, pw = Q(float(power[x])), [Q(float(power[j])) for j in nb]
    U = [(pw[i] - pw_x - sum(d * d for d in D[i])) / 2 for i in range(n)]
    r = Q(float(radius))
    c1 = (r * r + pw_x) / 2
    cache: Dict[Tuple[int, int], object] = {}

    def B(i, j):
        key = (i, j) if i <= j else (j, i)
        v = cache.get(key)
        if v is None:
            v = sum(a * b for a, b in zip(D[i], D[j]))
            cache[key] = v
        return v

    def solve_W(W, rhs):
        return _fsolve([[B(a, b) for b in W] for a in W], rhs) if W else []

    lam = [Q(0)] * n
    Jl = [int(j) for j in J]
    W: List[int] = []
    implied = set()
    # the equality block: independent rows enter, dependent rows must be implied
    for j in Jl:
        rj = solve_W(W, [B(w, j) for w in W])
        z2 = B(j, j) - sum(a * B(w, j) for a, w in zip(rj, W))
        if z2 == 0:
            if U[j] != sum(a * U[w] for a, w in zip(rj, W)):
                return dict(accept=False, c=None, lam=lam,
                            reason="no point is equidistant from the vertices")
            implied.add(j)
        else:
            W.append(j)
    eq = set(W)
    for w, v in zip(W, solve_W(W, [U[w] for w in W])):
        lam[w] = v
    limit = max_iter or 50 * (n + m + 10)
    for _ in range(limit):
        Bl = [sum(lam[w] * B(i, w) for w in W) for i in range(n)]
        dual = sum(lam[w] * U[w] for w in W) - sum(lam[w] * Bl[w] for w in W) / 2
        if dual > c1:
            return dict(accept=False, c=dual, lam=lam, reason="the dual bound passed the radius")
        p, best = -1, Q(0)
        inW = set(W)
        for i in range(n):
            if i in inW or i in implied:
                continue
            ri = U[i] - Bl[i]
            if ri > best:
                p, best = i, ri
        if p < 0:
            return dict(accept=bool(dual <= c1), c=dual, lam=lam, reason="optimal")
        resid_p = best
        while True:
            rr = solve_W(W, [B(w, p) for w in W])
            z2 = B(p, p) - sum(a * B(w, p) for a, w in zip(rr, W))
            t2 = resid_p / z2 if z2 > 0 else None
            t1, blk = None, None
            for a, w in zip(rr, W):
                if w not in eq and a > 0:
                    q = lam[w] / a
                    if t1 is None or q < t1:
                        t1, blk = q, w
            if t2 is None and t1 is None:
                return dict(accept=False, c=None, lam=lam,
                            reason="the Voronoi face is empty (primal infeasible)")
            t = t1 if (t2 is None or (t1 is not None and t1 < t2)) else t2
            for a, w in zip(rr, W):
                lam[w] -= t * a
            lam[p] += t
            resid_p -= t * z2
            if t2 is not None and t == t2:
                W.append(p)
                break
            lam[blk] = Q(0)
            W.remove(blk)
    raise AlphaUndecided(f"the exact solver did not terminate in {limit} iterations "
                         f"(it must, for a strictly convex primal): report this input")


# ──────────────────────────────────────────────────────────────────────────────
# Batched equality phase
# ──────────────────────────────────────────────────────────────────────────────
def _eigh(M):
    """Run ``torch.linalg.eigh``, on the CPU when the device has no kernel for it.

    MPS does not implement ``aten::_linalg_eigh``. Torch's own advice is the global
    ``PYTORCH_ENABLE_MPS_FALLBACK=1``, which silently changes where every
    unimplemented op runs -- not a library's decision to make. This bounces the one
    call instead, and only when it has to; the builder keeps the QP off MPS, so on a
    normal run it never fires.
    """
    try:
        return torch.linalg.eigh(M)
    except (NotImplementedError, RuntimeError) as exc:
        if M.device.type == "cpu":
            raise
        del exc
        w, Qm = torch.linalg.eigh(M.cpu())
        return w.to(M.device), Qm.to(M.device)


def _batched_pinv_solve(M, rhs, rtol: float):
    """Minimum-norm solve of ``M x = rhs`` for a batch of symmetric ``M``, with a consistency flag.

    ``ok=False`` means ``rhs`` left the range of ``M`` -- for us, that the simplex
    has no equidistant point at all. Both tests are relative (the eigenvalue cutoff
    to ``M``'s largest eigenvalue, the consistency test to the size of ``rhs``), so
    verdicts do not depend on the ratio of the point spacing to the radius.
    """
    tiny = torch.finfo(M.dtype).tiny
    w, Qm = _eigh(M)
    scale = torch.clamp(w.abs().amax(dim=-1, keepdim=True), min=tiny)
    keep = w.abs() > rtol * scale
    coef = torch.einsum("bij,bi->bj", Qm, rhs)
    resid = torch.linalg.vector_norm(torch.where(keep, torch.zeros_like(coef), coef), dim=-1)
    inv = torch.where(keep, 1.0 / torch.where(keep, w, torch.ones_like(w)), torch.zeros_like(w))
    x = torch.einsum("bij,bj->bi", Qm, inv * coef)
    ok = resid <= 1e-6 * rhs.abs().amax(dim=-1)
    return x, ok


def _pinv_detail(M, rhs, rtol: float):
    """Run :func:`_batched_pinv_solve` plus the gray-zone facts (relative residual, rank cut)."""
    tiny = torch.finfo(M.dtype).tiny
    w, _Q = _eigh(M)
    scale = torch.clamp(w.abs().amax(dim=-1, keepdim=True), min=tiny)
    cut = (w.abs() <= rtol * scale).any(dim=-1)
    x, ok = _batched_pinv_solve(M, rhs, rtol)
    rel = torch.linalg.vector_norm(torch.einsum("bij,bj->bi", M, x) - rhs, dim=-1) / \
        torch.clamp(rhs.abs().amax(dim=-1), min=tiny)
    return x, ok, rel, cut


# ──────────────────────────────────────────────────────────────────────────────
# Goldfarb-Idnani, batched
# ──────────────────────────────────────────────────────────────────────────────
def _active_set(B, U, Jpos, c1: float, w_max: int, max_iter: int, tol: float,
                lam_init=None, bound_margin: float = 0.0, diag: Optional[dict] = None):
    """Batched dual active set for ``max -1/2 l^T B l + U^T l``, ``l >= 0`` off ``J``.

    One Goldfarb-Idnani step per iteration over the whole batch. The equality
    block ``J`` is seeded into the working set and never leaves it; inequality
    members enter when violated and leave when their multiplier would go negative.
    The running dual objective is a monotone lower bound on ``c*``, which licenses
    rejection the moment it passes ``c1``.

    States returned per row: ``0`` unfinished (caller falls back); ``1`` optimal;
    ``2`` ruled out on the lower bound; ``3`` working set overflowed or a
    dependence test was too close to call (caller falls back); ``4`` ruled out
    because the violated constraint is (numerically) in the span of the working set
    with nothing able to leave -- certified by the bound ``c* >= dual + resid_p^2 /
    (2 |z|^2)``, not by the rank test alone. States 2 and 4 are kept apart because
    only 2 stays sound in low precision; float32 keeps its own 2s and nothing else.
    All tolerances are relative to the problem's own scale ``s^2`` (the largest
    squared neighbour distance). With ``diag``, ``diag["min_ratio"]`` receives the
    smallest ``|z|^2 / |d_p|^2`` of the constraints that entered the working set,
    i.e. how ill-conditioned the final solve was.

    Returns:
        Tuple ``(state, c, lam)``.
    """
    dev, dt = B.device, B.dtype
    nb, k = Jpos.shape
    n = B.shape[0]
    big = float(np.sqrt(torch.finfo(dt).max))
    rtol = 1e-12 if dt == torch.float64 else 1e-6
    eps_dep = 1e-10 if dt == torch.float64 else 1e-5      # relative dependence test
    s2 = float(torch.clamp(torch.diagonal(B).max(), min=torch.finfo(dt).tiny)) if n else 1.0
    tol_v = tol * s2 ** 0.5          # constraint residuals ~ (spacing) x (witness distance <= 1)

    wid = torch.zeros((nb, w_max), dtype=torch.long, device=dev)
    wid[:, :k] = Jpos
    wcnt = torch.full((nb,), k, dtype=torch.long, device=dev)
    # The equality block starts IN the working set and already satisfied. Seeding
    # lambda at zero instead would make y = x, which satisfies every Voronoi
    # inequality trivially, and every candidate would be declared optimal at once.
    if lam_init is None:
        lam = torch.zeros((nb, n), dtype=dt, device=dev)
        if k:
            lamJ, _ = _batched_pinv_solve(B[Jpos[:, :, None], Jpos[:, None, :]], U[Jpos], rtol)
            lam.scatter_(1, Jpos, lamJ)
    else:
        lam = lam_init.clone()
    pend = torch.full((nb,), -1, dtype=torch.long, device=dev)     # constraint being added
    state = torch.zeros((nb,), dtype=torch.long, device=dev)
    in_w = torch.zeros((nb, n), dtype=torch.bool, device=dev)
    in_w.scatter_(1, Jpos, True)

    min_ratio = torch.full((nb,), float("inf"), dtype=dt, device=dev)
    slots = torch.arange(w_max, device=dev)[None, :]
    # inactive slots are padded with s^2 I, not I: padding at the wrong scale would
    # set the eigenvalue cutoff of every working-set solve
    eye = s2 * torch.eye(w_max, dtype=dt, device=dev)[None, :, :]
    c = torch.zeros((nb,), dtype=dt, device=dev)

    # Only the rows still running are carried into each iteration, so the batch
    # does not pay for its slowest member.
    live = torch.arange(nb, device=dev)
    for _ in range(max_iter):
        if live.numel() == 0:
            break
        lam_l, wid_l, wcnt_l = lam[live], wid[live], wcnt[live]
        nl = live.numel()
        act = slots[:, :w_max] < wcnt_l[:, None]
        rows = torch.arange(nl, device=dev)

        Blam = lam_l @ B
        resid = U[None, :] - Blam                     # = A y - V, satisfied when <= 0
        dual = -0.5 * (lam_l * Blam).sum(1) + (U[None, :] * lam_l).sum(1)
        c[live] = dual
        # dual feasible => `dual` is a LOWER bound on c*; passing c1 rules sigma out.
        # In float32 the bound must also clear its own evaluation error.
        if dt == torch.float64:
            done2 = dual > c1 + bound_margin + tol
        else:
            al = lam_l.abs()
            err = 64 * torch.finfo(dt).eps * ((al * U.abs()[None, :]).sum(1)
                                               + (al * (al @ B.abs())).sum(1))
            done2 = dual - err > c1 + bound_margin + tol
        state[live[done2]] = 2

        has_p = pend[live] >= 0
        masked = resid.masked_fill(in_w[live], -big)
        maxviol, argp = masked.max(dim=1)
        done1 = (~done2) & (~has_p) & (maxviol <= tol_v)
        state[live[done1]] = 1
        start_add = (~done2) & (~has_p) & (maxviol > tol_v)
        pend[live[start_add]] = argp[start_add]

        keep = ~(done1 | done2)
        if not bool(keep.any()):
            live = live[keep]
            continue
        sub = live[keep]
        dual_k = dual[keep]
        lam_l, wid_l, wcnt_l = lam_l[keep], wid_l[keep], wcnt_l[keep]
        resid, act = resid[keep], act[keep]
        nl = sub.numel()
        rows = torch.arange(nl, device=dev)
        pcur = pend[sub]

        # r = B_WW^+ B_{W,p};  z2 = |z|^2 = B_pp - r . B_{W,p}
        Bww = B[wid_l[:, :, None], wid_l[:, None, :]]
        Bww = torch.where(act[:, :, None] & act[:, None, :], Bww, eye.expand(nl, -1, -1))
        Bwp = torch.gather(B.index_select(1, pcur).T, 1, wid_l) * act
        r, _ = _batched_pinv_solve(Bww, Bwp, rtol)
        r = r * act
        Bpp = B[pcur, pcur]
        z2 = Bpp - (r * Bwp).sum(1)
        # the dependence threshold grows with |r|^2, the size of the rounding error
        # in z2 = B_pp - r.B_Wp, so that "z2 <= thr" bounds the TRUE z2 too
        thr = eps_dep * Bpp * (1.0 + (r * r).sum(1))
        dep = z2 <= thr                               # p (numerically) in span(W)

        resid_p = torch.gather(resid, 1, pcur[:, None]).squeeze(1)
        t2 = torch.where(dep, torch.full_like(z2, big),
                         resid_p / torch.where(dep, torch.ones_like(z2), z2))
        lam_w = torch.gather(lam_l, 1, wid_l)
        ineq = act & (slots[:, :w_max] >= k)   # the equalities never block, never leave
        ratio = torch.where(ineq & (r > tol), lam_w / torch.clamp(r, min=tol),
                            torch.full_like(r, big))
        t1, argblock = ratio.min(dim=1)

        # p is dependent on W (up to eps_dep) and nothing can be dropped. Exactly
        # dependent: the primal is infeasible. Nearly dependent: the step would raise
        # the dual by at least resid_p^2 / (2 eps_dep B_pp). Either way c* exceeds
        # the current dual by that much -- decisive when it passes c1, otherwise not.
        dead = dep & (t1 >= big / 2)
        jump = resid_p ** 2 / (2.0 * torch.clamp(thr, min=torch.finfo(dt).tiny))
        sure = dead & (dual_k + jump > c1 + bound_margin + tol)
        state[sub[sure]] = 4
        state[sub[dead & ~sure]] = 3

        t = torch.where(dead, torch.zeros_like(t1), torch.minimum(t1, t2))
        lam_l.scatter_add_(1, wid_l, -t[:, None] * r * act)
        lam_l.scatter_add_(1, pcur[:, None], t[:, None])
        lam[sub] = lam_l

        full = (~dead) & (t2 <= t1)
        if bool(full.any()):
            ratio_in = z2 / torch.clamp(Bpp, min=torch.finfo(dt).tiny)
            min_ratio[sub[full]] = torch.minimum(min_ratio[sub[full]], ratio_in[full])
        overflow = full & (wcnt_l >= w_max)
        state[sub[overflow]] = 3
        do_put = full & (wcnt_l < w_max)
        if bool(do_put.any()):
            wid_l[rows[do_put], wcnt_l[do_put]] = pcur[do_put]
            in_w[sub[do_put], pcur[do_put]] = True
            wcnt_l = torch.where(do_put, wcnt_l + 1, wcnt_l)
        pend[sub[full]] = -1

        part = (~dead) & (~full)               # the blocking inequality leaves, p stays
        if bool(part.any()):
            bidx = wid_l[rows, argblock]
            lam[sub[part], bidx[part]] = 0.0
            in_w[sub[part], bidx[part]] = False
            wid_l[rows[part], argblock[part]] = wid_l[rows[part], (wcnt_l - 1)[part]]
            wcnt_l = torch.where(part, wcnt_l - 1, wcnt_l)

        wid[sub] = wid_l
        wcnt[sub] = wcnt_l
        live = sub[state[sub] == 0]

    Blam = lam @ B
    c = torch.where(state == 0, c, -0.5 * (lam * Blam).sum(1) + (U[None, :] * lam).sum(1))
    if diag is not None:
        diag["min_ratio"] = min_ratio
    return state, c, lam


# ──────────────────────────────────────────────────────────────────────────────
# Main entry
# ──────────────────────────────────────────────────────────────────────────────
def _working_dtype(device, dtype=None):
    """float64 wherever it exists; float32 only on MPS, which has no float64."""
    if dtype is not None:
        return dtype
    return torch.float32 if device.type == "mps" else torch.float64


def dual_alpha_complex(points, radius: float, max_dim: Optional[int] = None, *,
                       power: Optional[np.ndarray] = None,
                       device=None, qp_device=None, dtype=None,
                       w_max: Optional[int] = None, max_iter: int = 64,
                       max_simplices: Optional[int] = 2_000_000,
                       recheck_rel: Optional[float] = None,
                       batch: int = 1024, keep_witness: bool = True,
                       coefficient_ring: str = "Z") -> DualAlphaResult:
    """Build ``Alpha(S, radius)`` up to dimension ``max_dim`` by Algorithm 1 of Carlsson & Carlsson.

    What is Being Computed?:
        The d-skeleton of the (weighted) alpha complex -- the nerve of the
        radius-restricted (power) Voronoi diagram -- together with every simplex's
        alpha value and witness point, without computing a Delaunay triangulation.

    Algorithm:
        1. Rescale the cloud by the power of two nearest the radius (exact in
           binary floating point), so the solver's tolerances see ``a1`` in
           ``[1/2, 2]``; results are returned in the caller's units.
        2. Build a certified superset of the Cech graph on the selected device.
        3. For ``k = 1, 2, ...``: generate the lazy candidates (every facet already
           accepted), group them by their smallest vertex (one shared Gram matrix
           per vertex), and decide each with the batched equality phase, the
           batched dual active set, the float64 authority and -- for the gray
           zone -- the exact rational solver.
        4. Stop when no candidate survives or ``max_dim`` is reached; enforce face
           monotonicity of the float weights (a rounding-level correction).

    Preserved Invariants:
        - The complex is identical to the exact alpha complex (validated simplex for
          simplex against CGAL's exact alpha complex in TabularTopology), and is
          homotopy equivalent to the union of the radius-``radius`` balls.
        - ``weights`` is monotone under faces, so every sub-level set is a complex.

    Args:
        points: ``(N, m)`` coordinates (array or ``PointCloud``).
        radius: The cap radius (``a1 = radius^2``); must be positive.
        max_dim: Top dimension to build; ``None`` builds until no candidate remains
            (the honest default: a complex capped at ``d`` has an inflated
            ``beta_d``, see :meth:`DualAlphaResult.exact_betti_dimensions`).
        power: Optional ``(N,)`` weights for the weighted alpha complex.
        device: Device for the Cech graph (and the QP unless on MPS); ``None``
            selects automatically (see :mod:`pysurgery.gpu.device`).
        qp_device: Override the QP device (default: ``device``, or the CPU on MPS).
        dtype: Override the QP working precision (float32 only screens).
        w_max: Working-set width cap (default ``min(n, max(m + k + 2, 8), 64)``);
            overflow is re-solved, never guessed.
        max_iter: Active-set iterations per batch before falling back.
        max_simplices: Refuse with :class:`AlphaBudgetExceeded` beyond this many
            simplices (``None`` disables the guard).
        recheck_rel: Relative screening margin (default ``1e-9`` in float64).
        batch: Candidates per batched solve (memory is proportional to it; does
            not change any verdict).
        keep_witness: Store the witness point of every simplex.
        coefficient_ring: Ring label of the returned complex.

    Returns:
        A :class:`DualAlphaResult`.

    Raises:
        ValueError: On malformed points or a non-positive radius.
        AlphaBudgetExceeded: If the candidate set outgrows ``max_simplices``.

    Use When:
        - The ambient dimension is too high for a Delaunay triangulation.
        - You want the alpha filtration up to a radius, with exact membership.

    Example:
        res = dual_alpha_complex(points, 0.5)
        res.complex.betti_numbers()
        res.subcomplex(0.3)             # Alpha(S, 0.3), no rebuild
    """
    from pysurgery.topology.complexes import SimplicialComplex

    P_in = np.array(np.asarray(points, dtype=np.float64), dtype=np.float64, copy=True)
    if P_in.ndim != 2:
        raise ValueError(f"points must be (N, m); got {P_in.shape}")
    if not np.all(np.isfinite(P_in)):
        raise ValueError("points must be finite")
    if not (float(radius) > 0):
        raise ValueError(f"radius must be positive; got {radius}")
    if max_dim is not None and int(max_dim) < 0:
        raise ValueError(f"max_dim must be >= 0; got {max_dim}")
    N, m = P_in.shape
    graph_dev = resolve_device(device)
    # The Cech graph is one big pairwise pass -- genuinely parallel, and every op it
    # needs exists everywhere. The QP is many small eigendecompositions, and
    # `aten::_linalg_eigh` is NOT implemented on MPS (nor is float64). So on MPS the
    # graph runs on the GPU and the QP on the CPU in float64; CUDA keeps both.
    dev = torch.device(qp_device) if qp_device is not None else (
        torch.device("cpu") if graph_dev.type == "mps" else graph_dev)
    dt = _working_dtype(dev, dtype)
    pw_in = None if power is None else np.asarray(power, dtype=np.float64).reshape(-1)
    if pw_in is not None and pw_in.shape[0] != N:
        raise ValueError(f"power must have one entry per point ({N}); got {pw_in.shape[0]}")
    pw_full = np.zeros(N) if pw_in is None else pw_in

    # Work in units of the radius: an active-set method compares residuals against
    # fixed tolerances, so its verdicts are not scale invariant. Dividing by a POWER
    # OF TWO is exact, so the rescaled cloud is the input, not a rounded copy.
    scale = float(2.0 ** np.round(np.log2(float(radius))))
    P = P_in / scale
    pw_np = pw_full / (scale * scale)
    a1 = (float(radius) / scale) ** 2

    gdt = _working_dtype(graph_dev, None)
    batch0 = int(batch)
    Xt = torch.as_tensor(P, dtype=dt, device=dev)
    pwt = torch.as_tensor(pw_np, dtype=dt, device=dev)
    tol = 1e-11 if dt == torch.float64 else 1e-5
    if recheck_rel is None:
        recheck_rel = 1e-9 if dt == torch.float64 else 1e-3

    try:
        adj = _cech_graph(P, a1, pw_np, graph_dev, gdt)
    except (NotImplementedError, RuntimeError, TypeError) as exc:
        if graph_dev.type == "cpu":
            raise
        warnings.warn(f"the Cech graph failed on {graph_dev} ({type(exc).__name__}: "
                      f"{str(exc)[:160]}); recomputing it on the CPU in float64.",
                      RuntimeWarning, stacklevel=2)
        graph_dev, gdt = torch.device("cpu"), torch.float64
        adj = _cech_graph(P, a1, pw_np, graph_dev, gdt)
    stats = dict(n_points=N, m=m, radius=float(radius), graph_device=str(graph_dev),
                 n_candidates=0, n_accepted=0,
                 n_equality_only=0, n_active_set=0, n_exact_fallback=0, n_undecided=0,
                 n_exact_rational=0, n_screened=0, n_monotone_fixes=0,
                 max_degree=int(max(len(a) for a in adj)) if N else 0,
                 dim_sizes=[])

    weight: Dict[Simplex, float] = {}
    witness: Dict[Simplex, np.ndarray] = {}

    # k = 0. U_x is nonempty iff -p(x) <= a1; the witness is x itself, w = -p(x).
    alive = [i for i in range(N) if a1 + pw_np[i] >= 0]
    accepted: set = set()
    for i in alive:
        weight[(i,)] = 0.0 - float(pw_full[i])      # 0.0 - p, never -0.0
        if keep_witness:
            witness[(i,)] = P_in[i].copy()
        accepted.add((i,))
    stats["dim_sizes"].append(len(alive))
    stats["n_accepted"] += len(alive)

    nbr = [set(a.tolist()) for a in adj]
    prev: List[Simplex] = [(i,) for i in alive]
    k = 1
    while prev and (max_dim is None or k <= int(max_dim)):
        cands = _candidates(prev, nbr, k, set(prev))
        if not cands:
            break
        # A co-spherical sample above its own circumradius makes EVERY subset a
        # simplex -- the centre witnesses all of them. That is a statement about
        # the radius, not a solver failure, so it is said out loud.
        if max_simplices is not None and len(cands) + len(accepted) > max_simplices:
            raise AlphaBudgetExceeded(
                f"dimension {k} has {len(cands)} candidates on top of "
                f"{len(accepted)} accepted simplices, past the {max_simplices} "
                f"budget. Either cap `max_dim`, raise `max_simplices`, or -- most "
                f"likely -- lower the radius: at r above the sample's own "
                f"circumradius (1.0 for a unit sphere) the alpha complex IS the "
                f"full simplex on every vertex.")
        stats["n_candidates"] += len(cands)
        # A GPU allocation failure halves the batch and retries; below 16 the QP moves
        # to the CPU in float64. Neither changes a verdict: the batch only groups one
        # vertex's candidates, and every verdict is margin-checked or exact.
        while True:
            snapshot = dict(stats)
            try:
                acc, wgt, wit = _test_dimension(cands, P, pw_np, Xt, pwt, adj, a1, k, m, dev, dt,
                                                w_max, max_iter, tol, recheck_rel, batch, stats,
                                                exact_ctx=(P_in, pw_in, float(radius), scale),
                                                keep_witness=keep_witness)
                break
            except RuntimeError as exc:
                if dev.type == "cpu" or not _is_alloc_error(exc):
                    raise
                stats.clear()
                stats.update(snapshot)
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                stats["gpu_alloc_retries"] = stats.get("gpu_alloc_retries", 0) + 1
                if batch // 2 >= 16:
                    batch //= 2
                else:
                    warnings.warn(f"dual alpha: out of memory on {dev} even at batch {batch}; "
                                  "continuing the QP on the CPU in float64.",
                                  RuntimeWarning, stacklevel=2)
                    dev, dt, batch = torch.device("cpu"), torch.float64, batch0
                    Xt = torch.as_tensor(P, dtype=dt)
                    pwt = torch.as_tensor(pw_np, dtype=dt)
                    tol, recheck_rel = 1e-11, 1e-9
        weight.update({t: v * scale * scale for t, v in wgt.items()})
        if keep_witness:
            witness.update({t: v * scale for t, v in wit.items()})
        stats["dim_sizes"].append(len(acc))
        stats["n_accepted"] += len(acc)
        prev = acc
        accepted.update(acc)
        k += 1

    _enforce_monotone(weight, stats)

    table: Dict[int, List[Simplex]] = {}
    for s in accepted:
        table.setdefault(len(s) - 1, []).append(s)
    table = {d: sorted(v) for d, v in table.items()}
    sc = SimplicialComplex(simplices=table, coefficient_ring=coefficient_ring)
    sc._coordinates = P_in
    if all(w >= 0 for w in weight.values()):
        sc.filtration = {s: float(np.sqrt(w)) for s, w in weight.items()}
    return DualAlphaResult(complex=sc, weights=weight,
                           witnesses=witness, radius=float(radius),
                           max_dim=None if max_dim is None else int(max_dim),
                           points=P_in, power=pw_in, adjacency=adj,
                           device=str(dev), graph_device=str(graph_dev),
                           dtype=str(dt).replace("torch.", ""), stats=stats)


def _is_alloc_error(exc: BaseException) -> bool:
    """Whether ``exc`` is one of the forms a GPU out-of-memory event takes."""
    if isinstance(exc, getattr(torch, "OutOfMemoryError", ())):
        return True
    msg = f"{type(exc).__name__}: {exc}"
    return any(marker in msg for marker in (
        "out of memory", "CUSOLVER_STATUS_ALLOC_FAILED", "CUSOLVER_STATUS_INTERNAL_ERROR",
        "CUBLAS_STATUS_ALLOC_FAILED", "cudaErrorMemoryAllocation", "MPS backend out of memory"))


def _enforce_monotone(weight: Dict[Simplex, float], stats: dict) -> None:
    """Raise every coface's weight to at least its facets' (in place).

    Exact alpha values are monotone under faces (``V_coface`` is a subset of
    ``V_face``, so its minimum power is larger). The float64 values carry
    rounding of order ``1e-12`` of their size, which can invert a tie between a
    face and a coface; a sub-level set would then not be a complex. This restores
    monotonicity and records how many values moved; a move beyond rounding
    level is warned about, since it would indicate a solver defect.

    Args:
        weight: Simplex -> squared alpha value, modified in place.
        stats: Counter dict (``n_monotone_fixes``, ``max_monotone_fix_rel``).
    """
    by_dim: Dict[int, List[Simplex]] = {}
    for s in weight:
        by_dim.setdefault(len(s) - 1, []).append(s)
    fixes, worst = 0, 0.0
    for d in sorted(by_dim):
        if d == 0:
            continue
        for s in by_dim[d]:
            f = max(weight[t] for t in itertools.combinations(s, d))
            w = weight[s]
            if f > w:
                worst = max(worst, (f - w) / max(abs(f), 1e-300))
                weight[s] = f
                fixes += 1
    stats["n_monotone_fixes"] = fixes
    stats["max_monotone_fix_rel"] = worst
    if worst > 1e-8:
        warnings.warn(
            f"dual alpha: {fixes} filtration values were raised to their faces' values "
            f"(largest relative change {worst:.3g}), more than rounding explains.",
            stacklevel=3,
        )


def _candidates(prev: List[Simplex], nbr, k: int, prev_set: set) -> List[Simplex]:
    """Generate ``Sigma_k = (Lazy_{k-1}(X))_k``: every k-simplex whose facets are all accepted.

    Each candidate is generated exactly once, as ``tau + (u,)`` with ``tau`` the
    candidate minus its largest vertex, so no deduplication pass is needed.
    """
    out: List[Simplex] = []
    for tau in prev:
        common = None
        for v in tau:
            common = nbr[v] if common is None else (common & nbr[v])
            if not common:
                break
        if not common:
            continue
        for u in common:
            if u <= tau[-1]:
                continue
            sig = tau + (u,)
            if all(f in prev_set for f in itertools.combinations(sig, k)):
                out.append(sig)
    return out


def _exact_callback(exact_ctx, x, nbrs, Jrows):
    """Build the ``exact_fn`` handed to :func:`_decide` for one vertex's batch.

    It re-decides rows with :func:`exact_decide` on the ORIGINAL input and returns
    ``c`` in the solver's units (the input's divided by the internal power-of-two
    scale, squared).
    """
    if exact_ctx is None:
        return None
    P0, pw0, rad, scale = exact_ctx

    def exact_fn(rows):
        from fractions import Fraction
        s2 = Fraction(float(scale)) ** 2
        out = []
        for i in rows:
            e = exact_decide(P0, pw0, x, nbrs, Jrows[i], rad)
            c = None if e["c"] is None else float(e["c"] / s2)
            out.append((e["accept"], c, [float(v) for v in e["lam"]]))
        return out
    return exact_fn


def _test_dimension(cands, P, pw_np, Xt, pwt, adj, a1, k, m_dim, dev, dt, w_max,
                    max_iter, tol, recheck_rel, batch, stats, exact_ctx=None,
                    keep_witness=True):
    """Run the QP for every candidate, grouped by the candidate's smallest vertex.

    ``exact_ctx`` = (original points, original power or None, radius, internal
    scale): what the exact solver needs to re-decide a gray-zone candidate from the
    input as given, and to report ``c`` in the solver's units.

    Returns:
        Tuple ``(accepted, weight, witness)`` in the solver's units.
    """
    by_vertex: Dict[int, List[Simplex]] = {}
    for s in cands:
        by_vertex.setdefault(s[0], []).append(s)

    accepted: List[Simplex] = []
    weight: Dict[Simplex, float] = {}
    witness: Dict[Simplex, np.ndarray] = {}

    for x, group in by_vertex.items():
        nb = adj[x]
        if len(nb) == 0:
            continue
        pos = {int(v): i for i, v in enumerate(nb)}
        # At the optimum the active constraints span at most m directions, so a
        # working set much wider than m + k is dead weight in the batched eigh.
        # Overflow is state 3 and a float64 retry at twice the width, never a
        # wrong answer, so this is safe to tune.
        wm = int(min(len(nb), max(m_dim + k + 2, 8), 64)) if w_max is None else w_max
        Jrows = []
        keep_group = []
        for s in group:
            try:
                Jrows.append([pos[v] for v in s[1:]])
                keep_group.append(s)
            except KeyError:
                continue                        # a vertex left the Cech neighbourhood
        if not Jrows:
            continue
        group = keep_group

        nbt = torch.as_tensor(np.asarray(nb, dtype=np.int64), device=dev)
        D = Xt.index_select(0, nbt) - Xt[x]
        B = D @ D.T
        U = 0.5 * (pwt.index_select(0, nbt) - pwt[x] - (D * D).sum(1))
        px = float(pw_np[x])
        c1 = (a1 + px) / 2.0
        # The authority path is rebuilt from the ORIGINAL float64 coordinates, not
        # cast up from the working tensor: a float64 solve on float32 inputs is
        # still a float32 answer.
        idxn = np.asarray(nb, dtype=np.int64)
        Dnp = P[idxn] - P[x]
        Bx = torch.as_tensor(Dnp @ Dnp.T, dtype=torch.float64)
        Ux = torch.as_tensor(0.5 * (pw_np[idxn] - px - (Dnp * Dnp).sum(1)),
                             dtype=torch.float64)

        for lo in range(0, len(group), batch):
            sub = group[lo:lo + batch]
            Jb = Jrows[lo:lo + batch]
            Jpos = torch.as_tensor(np.asarray(Jb, dtype=np.int64), device=dev)
            res = _decide(B, U, Jpos, c1, wm, max_iter, tol, recheck_rel,
                          Bx, Ux, stats, exact_fn=_exact_callback(exact_ctx, x, idxn, Jb))
            for i, s in enumerate(sub):
                if not res["accept"][i]:
                    continue
                accepted.append(s)
                weight[s] = float(2.0 * res["c"][i] - px)
                if keep_witness:
                    witness[s] = P[x] - (res["lam"][i] @ Dnp)
    accepted.sort()
    return accepted, weight, witness


def _state_to_how(st):
    """Map an active-set state to how the verdict was reached (see :func:`_decide`)."""
    return torch.where(st == 1, 2, torch.where(st == 2, 1, torch.where(st == 4, 4, 0)))


def _decide(B, U, Jpos, c1, w_max, max_iter, tol, recheck_rel, Bx, Ux, stats,
            exact_fn=None):
    """Decide a batch: equality phase, active set, float64 authority, then the exact solver.

    Which tier may produce a VERDICT depends on the precision. float32 may only
    screen simplices out on a lower bound; everything else is re-decided in float64
    on the CPU. A float64 verdict stands only when it clears a margin. A candidate
    in the gray zone -- a witness at (numerically) the radius, a tie with a point
    outside sigma, a rank decision on a (nearly) degenerate simplex -- or one that
    did not resolve, is decided by :func:`exact_decide` through ``exact_fn``.
    Without ``exact_fn`` such a candidate raises :class:`AlphaUndecided`.

    ``how`` records how each verdict was reached: 0 unresolved, 1 a lower bound
    passed c1, 2 the optimum c* (vs c1), 3 no point is equidistant from sigma's
    vertices, 4 the dependence bound of :func:`_active_set`.

    Returns:
        Dict with numpy arrays ``accept``, ``c`` and ``lam``.
    """
    dev, dt = B.device, B.dtype
    nb_, k = Jpos.shape
    n = B.shape[0]
    rtol = 1e-12 if dt == torch.float64 else 1e-6
    big = float(np.sqrt(torch.finfo(dt).max))
    tol_v = tol * float(torch.clamp(torch.diagonal(B).max(), min=torch.finfo(dt).tiny)) ** 0.5
    exact = dt == torch.float64
    margin = recheck_rel * max(1.0, abs(c1))

    # --- equality phase: the witness of the unrestricted circumcentre ---------
    Bjj = B[Jpos[:, :, None], Jpos[:, None, :]]                 # [nb, k, k]
    Uj = U[Jpos]
    lamJ, consistent, rel_eq, cut_eq = _pinv_detail(Bjj, Uj, rtol)
    # the dual objective AT lamJ -- a valid lower bound on c* for any lamJ
    c0 = (lamJ * Uj).sum(1) - 0.5 * (lamJ * torch.einsum("bij,bj->bi", Bjj, lamJ)).sum(1)
    lam = torch.zeros((nb_, n), dtype=dt, device=dev)
    lam.scatter_(1, Jpos, lamJ)
    resid = U[None, :] - lam @ B
    in_j = torch.zeros((nb_, n), dtype=torch.bool, device=dev)
    in_j.scatter_(1, Jpos, True)
    maxviol = resid.masked_fill(in_j, -big).max(dim=1).values

    accept = torch.zeros((nb_,), dtype=torch.bool, device=dev)
    c_out = c0.clone()
    lam_out = lam
    how = torch.zeros((nb_,), dtype=torch.long)
    rel64 = torch.zeros((nb_,), dtype=torch.float64)
    cut64 = torch.zeros((nb_,), dtype=torch.bool)
    ratio64 = torch.full((nb_,), float("inf"), dtype=torch.float64)

    # The one verdict float32 may reach on its own: c0 is a LOWER bound on c*
    # (lambda = (lambda_J, 0) is dual feasible), so c0 > c1 rules sigma out.
    if exact:
        screened = consistent & (c0 > c1 + margin)
    else:
        aJ = lamJ.abs()
        err = 64 * torch.finfo(dt).eps * ((aJ * Uj.abs()).sum(1) + (
            aJ * torch.einsum("bij,bj->bi", Bjj.abs(), aJ)).sum(1))
        screened = consistent & (c0 - err > c1 + margin)
    stats["n_screened"] += int(screened.sum())
    how[screened.cpu()] = 1

    fallback = []
    if exact:
        rel64, cut64 = rel_eq.double().cpu(), cut_eq.cpu()
        how[(~consistent).cpu()] = 3
        clean = consistent & (maxviol <= tol_v)   # circumcentre already lies in V_sigma
        accept = clean & (~screened) & (c0 <= c1)
        how[(clean & (~screened)).cpu()] = 2
        stats["n_equality_only"] += int((clean & (~screened)).sum())
        idx = torch.nonzero((~clean) & (~screened) & consistent, as_tuple=False).flatten()
    else:
        # the active set runs purely as a screen: only its lower-bound rejections
        # (state 2, widened by `margin`) are kept, and nothing is accepted here
        idx = torch.nonzero(~screened, as_tuple=False).flatten()

    if idx.numel():
        stats["n_active_set"] += int(idx.numel())
        for lo in range(0, idx.numel(), 512):
            sl = idx[lo:lo + 512]
            ok_rows = consistent[sl]
            dg: dict = {}
            st, c_as, lam_as = _active_set(B, U, Jpos[sl], c1, w_max, max_iter, tol,
                                           lam_init=lam[sl],
                                           bound_margin=0.0 if exact else margin, diag=dg)
            if exact:
                ratio64[sl.cpu()] = dg["min_ratio"].double().cpu()
            if not exact:
                # a row whose equality block was inconsistent in float32 has no
                # meaningful dual start; it goes to the float64 authority as is
                st = torch.where(ok_rows, st, torch.zeros_like(st))
            c_out[sl] = c_as
            lam_out[sl] = lam_as
            if exact:
                accept[sl] = (st == 1) & (c_as <= c1)
                how[sl.cpu()] = _state_to_how(st).cpu()
                fallback.append(sl[(st != 1) & (st != 2) & (st != 4)])
            else:
                stats["n_screened"] += int((st == 2).sum())
                how[sl.cpu()] = torch.where(st == 2, 1, 0).cpu()
                fallback.append(sl[st != 2])

    # --- float64 CPU authority ----------------------------------------------
    fb = [f for f in fallback if f.numel()]
    if fb:
        fb = torch.unique(torch.cat(fb)).cpu()
        stats["n_exact_fallback"] += int(fb.numel())
        J64 = Jpos[fb.to(dev)].cpu()
        lamJ64, ok64, r64, c_64 = _pinv_detail(Bx[J64[:, :, None], J64[:, None, :]], Ux[J64], 1e-12)
        rel64[fb], cut64[fb] = r64, c_64
        how[fb[~ok64]] = 3
        accept[fb[~ok64].to(dev)] = False
        rows = fb[ok64]
        if rows.numel():
            J_ok = J64[ok64]
            lam0 = torch.zeros((rows.numel(), n), dtype=torch.float64)
            lam0.scatter_(1, J_ok, lamJ64[ok64])
            dg = {}
            st, c_ex, lam_ex = _active_set(Bx, Ux, J_ok, c1, min(n, 2 * w_max),
                                           8 * max_iter, 1e-11, lam_init=lam0, diag=dg)
            ratio64[rows] = dg["min_ratio"].double()
            rd = rows.to(dev)
            accept[rd] = ((st == 1) & (c_ex <= c1)).to(dev)
            c_out[rd] = c_ex.to(dt).to(dev)
            lam_out[rd] = lam_ex.to(dt).to(dev)
            how[rows] = _state_to_how(st)

    # --- the gray zone ---------------------------------------------------------
    # Margins are measured in float64 against the size of the terms the decisive
    # numbers are made of BEFORE they cancel (|B| |lambda|): on a flat simplex
    # lambda is huge and B lambda ~ U is tiny, so measuring after cancellation
    # would call a rounding artefact a margin. 1e-9 is seven orders above float64.
    lam64 = lam_out.double().cpu()
    c64 = c_out.double().cpu()
    Bl = lam64 @ Bx
    res64 = Ux[None, :] - Bl
    absBl = lam64.abs() @ Bx.abs()                          # |lambda| |B|
    cscale = ((lam64.abs() * Ux.abs()[None, :]).sum(1) + (lam64.abs() * absBl).sum(1))
    cscale = torch.clamp(cscale, min=max(abs(c1), 1e-300))
    near_radius = (c64 - c1).abs() <= 1e-9 * cscale
    inJ = torch.zeros((nb_, n), dtype=torch.bool)
    inJ.scatter_(1, Jpos.cpu(), True)
    off = (~inJ) & (lam64 == 0)
    vscale = torch.clamp(Ux.abs()[None, :] + absBl, min=1e-300)   # per constraint
    tie = (res64 > -1e-9 * vscale).masked_fill(~off, False).any(1)
    # "no equidistant point" (how 3) is always a rank decision, and a cut
    # eigenvalue can be a genuine one of a very flat simplex, so it is never a
    # float verdict. Optima and dependence verdicts reached through an
    # ill-conditioned solve go to the exact solver too. Lower bounds are exempt.
    ill = (rel64 > 1e-10) | (ratio64 < 1e-6)
    gray = ((how == 0)
            | ((how == 1) & near_radius)
            | ((how == 2) & (near_radius | tie | cut64 | ill))
            | (how == 3)
            | ((how == 4) & ill))
    gi = torch.nonzero(gray, as_tuple=False).flatten().tolist()
    if gi:
        stats["n_undecided"] += int((how == 0).sum())
        if exact_fn is None:
            raise AlphaUndecided(
                f"{len(gi)} candidate simplices could not be decided in floating point "
                f"with a margin, and no exact solver was supplied.")
        stats["n_exact_rational"] = stats.get("n_exact_rational", 0) + len(gi)
        for i, (acc_i, c_i, lam_i) in zip(gi, exact_fn(gi)):
            accept[i] = bool(acc_i)
            c_out[i] = float(c_i) if c_i is not None else float("inf")
            lam_out[i] = torch.as_tensor(np.asarray(lam_i, dtype=np.float64), dtype=dt, device=dev)

    return dict(accept=accept.cpu().numpy(), c=c_out.double().cpu().numpy(),
                lam=lam_out.double().cpu().numpy())


# ──────────────────────────────────────────────────────────────────────────────
# The connectivity radius (exact EMST, on the device)
# ──────────────────────────────────────────────────────────────────────────────
def connectivity_radius(points, device=None) -> float:
    """Half the longest edge of the Euclidean minimum spanning tree.

    What is Being Computed?:
        The smallest ``r`` at which the union of the radius-``r`` balls -- hence the
        alpha complex -- is connected. Computed by Prim's algorithm on the dense
        distance matrix, one row at a time (``O(N)`` memory, ``O(N^2 m)`` work) on
        the selected device, in float64 (on MPS, which has no float64, on the CPU).

        The value is rounded UP by a bound on the evaluation error (each squared
        distance is a sum of ``m`` squared, correctly rounded differences, with
        relative error at most ``gamma_{m+2}``), so the alpha complex at the
        returned radius -- whose membership is decided exactly -- is guaranteed to
        be connected. Differences are taken between the original coordinates
        (rounding relative to the distance, not to the distance from the origin).

    Args:
        points: ``(N, m)`` coordinates.
        device: ``None`` for the automatic choice.

    Returns:
        The connectivity radius (``0.0`` for fewer than two points).

    Example:
        r0 = connectivity_radius(points)   # Alpha(points, r0) is connected
    """
    P = np.asarray(points, dtype=np.float64)
    N = P.shape[0]
    if N < 2:
        return 0.0
    m = P.shape[1]
    dev = resolve_device(device)
    if dev.type == "mps":
        dev = torch.device("cpu")
    X = torch.as_tensor(P, dtype=torch.float64, device=dev)
    inf = torch.tensor(float("inf"), dtype=torch.float64, device=dev)
    dist = torch.full((N,), float("inf"), dtype=torch.float64, device=dev)
    in_tree = torch.zeros(N, dtype=torch.bool, device=dev)
    longest = torch.zeros((), dtype=torch.float64, device=dev)
    i = torch.zeros(1, dtype=torch.long, device=dev)
    dist[0] = 0.0
    for _ in range(N):
        # the argmin stays on the device: no host synchronisation per step
        i = torch.where(in_tree, inf, dist).argmin().view(1)
        longest = torch.maximum(longest, dist.index_select(0, i).squeeze(0))
        in_tree.index_fill_(0, i, True)
        d = X - X.index_select(0, i)
        dist = torch.minimum(dist, (d * d).sum(1))
    u = 2.0 ** -53
    return float(torch.sqrt(longest)) / 2.0 * (1.0 + 2.0 * _gamma(m + 4, u))
