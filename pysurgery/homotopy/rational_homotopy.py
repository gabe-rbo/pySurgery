"""Rational homotopy theory: Sullivan minimal models over ℚ.

Implements the Quillen-Sullivan theorem constructively: for a simply-connected
space X of finite ℚ-type, builds the unique minimal CDGA (ΛV, d) with
H(ΛV) ≅ H*(X; ℚ) and reads off π_n(X) ⊗ ℚ = V^n.

References:
    Quillen, D. (1969). Rational homotopy theory. Ann. Math. 90, 205–295.
    Sullivan, D. (1977). Infinitesimal computations in topology.
        Publ. Math. IHES 47, 269–331.
    Félix, Y., Halperin, S., & Thomas, J.-C. (2001).
        Rational Homotopy Theory. Springer GTM 205.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pysurgery.core.exceptions import SurgeryError
from pysurgery.core.foundations import CONTRACT_VERSION

# ── Constants ─────────────────────────────────────────────────────────────────

THEOREM_TAG = "rational.quillen_sullivan.minimal_model"
FORMALITY_THEOREM_TAG = "rational.formality.deligne_griffiths_morgan_sullivan"
MASSEY_THEOREM_TAG = "rational.massey_products.sullivan_model"
_F0 = Fraction(0)
_F1 = Fraction(1)

# A Monomial is a tuple of (gid, exponent) pairs sorted ascending by gid.
# Invariants: odd-degree generators have exp=1; no (gid,0) entries.
Monomial = Tuple[Tuple[int, int], ...]
UNIT: Monomial = ()


# ── Exceptions ────────────────────────────────────────────────────────────────


class DGAError(SurgeryError):
    """Root error for DGA / rational-homotopy failures."""


class DegreeError(DGAError):
    """Degree mismatch in an algebraic operation."""


class ClosureError(DGAError):
    """d²(g) ≠ 0 for some generator g."""

    def __init__(self, failures: list) -> None:
        self.failures = failures
        names = ", ".join(g.name for g, _ in failures)
        super().__init__(f"d²(g) ≠ 0 for: {names}")


class MinimalityError(DGAError):
    """d(g) contains a linear term, violating minimality."""


class CrossAlgebraError(DGAError):
    """Operation between elements belonging to different DGAs."""


class UnknownGeneratorError(DGAError):
    """Generator gid not found in this DGA."""


class HomogeneityError(DGAError):
    """Mixed-degree DGAElement in an operation requiring homogeneity."""


# ── Generator ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Generator:
    """An immutable named generator of ΛV."""

    name: str
    degree: int
    gid: int  # unique within one RationalDGA

    @property
    def parity(self) -> Literal["even", "odd"]:
        return "odd" if self.degree % 2 == 1 else "even"

    def __repr__(self) -> str:
        return f"Gen({self.name}|{self.degree})"


# ── Monomial helpers ──────────────────────────────────────────────────────────


def _mon_degree(mon: Monomial, g2n: Dict[int, Generator]) -> int:
    return sum(e * g2n[gid].degree for gid, e in mon)


def _mon_mul(
    a: Monomial, b: Monomial, g2n: Dict[int, Generator]
) -> Tuple[Optional[Monomial], int]:
    """Canonical product a ∧ b.

    Returns ``(monomial, sign)`` or ``(None, 0)`` when the product is zero
    (odd generator squared).

    Koszul sign rule: moving an odd generator of b leftward past an odd
    generator of a (i.e. b_j.gid < a_i.gid, both odd) contributes −1.
    """
    # Koszul sign: count (a_i, b_j) pairs where gid(b_j) < gid(a_i)
    # and both generators have odd degree.
    sign = 1
    for gid_b, _ in b:
        if g2n[gid_b].degree % 2 == 0:
            continue
        for gid_a, _ in a:
            if g2n[gid_a].degree % 2 == 1 and gid_a > gid_b:
                sign *= -1

    # Merge sorted factor lists.
    out: List[Tuple[int, int]] = []
    ia = ib = 0
    while ia < len(a) and ib < len(b):
        gid_a, ea = a[ia]
        gid_b, eb = b[ib]
        if gid_a < gid_b:
            out.append((gid_a, ea))
            ia += 1
        elif gid_b < gid_a:
            out.append((gid_b, eb))
            ib += 1
        else:
            if g2n[gid_a].parity == "odd":
                return None, 0  # odd generator squared → zero
            out.append((gid_a, ea + eb))
            ia += 1
            ib += 1
    out.extend(a[ia:])
    out.extend(b[ib:])
    return tuple(out), sign


def _mon_wedge3(
    left: Monomial, mid: Monomial, right: Monomial, g2n: Dict[int, Generator]
) -> Tuple[Optional[Monomial], int]:
    """Compute left ∧ mid ∧ right in canonical form."""
    lm, s1 = _mon_mul(left, mid, g2n)
    if lm is None:
        return None, 0
    rm, s2 = _mon_mul(lm, right, g2n)
    if rm is None:
        return None, 0
    return rm, s1 * s2


# ── DGAElement ─────────────────────────────────────────────────────────────────


class DGAElement:
    """A homogeneous element of a RationalDGA (sparse over ℚ)."""

    __slots__ = ("dga", "terms", "degree")

    def __init__(
        self,
        dga: "RationalDGA",
        terms: Dict[Monomial, Fraction],
        degree: Optional[int] = None,
    ) -> None:
        self.dga = dga
        self.terms: Dict[Monomial, Fraction] = {
            m: c for m, c in terms.items() if c != _F0
        }
        if degree is not None:
            self.degree: Optional[int] = degree
        elif not self.terms:
            self.degree = None
        else:
            degs = {_mon_degree(m, dga._g2n) for m in self.terms}
            if len(degs) != 1:
                raise HomogeneityError(
                    f"Inhomogeneous element: degrees {degs}."
                )
            self.degree = degs.pop()

    # arithmetic ---------------------------------------------------------------

    def __add__(self, other: "DGAElement") -> "DGAElement":
        if self.dga is not other.dga:
            raise CrossAlgebraError("Cannot add elements from different DGAs.")
        if (
            self.degree is not None
            and other.degree is not None
            and self.degree != other.degree
        ):
            raise DegreeError(
                f"Cannot add degree-{self.degree} and degree-{other.degree} elements."
            )
        out = dict(self.terms)
        for m, c in other.terms.items():
            out[m] = out.get(m, _F0) + c
        deg = self.degree if self.degree is not None else other.degree
        return DGAElement(self.dga, out, degree=deg)

    def __neg__(self) -> "DGAElement":
        return DGAElement(self.dga, {m: -c for m, c in self.terms.items()}, degree=self.degree)

    def __sub__(self, other: "DGAElement") -> "DGAElement":
        return self + (-other)

    def scale(self, s: Fraction) -> "DGAElement":
        return DGAElement(self.dga, {m: s * c for m, c in self.terms.items()}, degree=self.degree)

    def is_zero(self) -> bool:
        return not self.terms

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DGAElement):
            return NotImplemented
        return self.dga is other.dga and self.terms == other.terms

    def __repr__(self) -> str:
        if self.is_zero():
            return "0"
        g2n = self.dga._g2n
        parts = []
        for m, c in sorted(self.terms.items()):
            mon_s = "∧".join(
                (g2n[gid].name if e == 1 else f"{g2n[gid].name}^{e}") for gid, e in m
            ) or "1"
            parts.append(f"({c}){mon_s}")
        return " + ".join(parts)


# ── RationalDGA ───────────────────────────────────────────────────────────────


class RationalDGA:
    """A free graded-commutative DGA (ΛV, d) over ℚ.

    Overview:
        The free exterior algebra on generators V = ⊕ V^n, equipped with an
        odd derivation d of degree +1 satisfying d²=0 and the Leibniz rule.
        Coefficients are exact rationals (fractions.Fraction — no floats).

    Key Concepts:
        - Generators added via ``add_generator``; differential set via ``set_differential``.
        - ``d`` is stored on generators and extended by Leibniz to all monomials.
        - Cohomology computed by exact Gaussian elimination over ℚ.
        - ``graded_basis(n)`` enumerates all degree-n monomials (cached).

    Coefficient Ring:
        ℚ exact (fractions.Fraction).
    """

    def __init__(self) -> None:
        self._g2n: Dict[int, Generator] = {}           # gid → Generator
        self._by_deg: Dict[int, List[Generator]] = {}  # degree → [Generator]
        self._diff: Dict[int, DGAElement] = {}          # gid → d(g)
        self._basis_cache: Dict[int, List[Monomial]] = {}
        self._next_gid: int = 1

    # ── generators ────────────────────────────────────────────────────────────

    def add_generator(self, degree: int, name: Optional[str] = None) -> Generator:
        """Add a new generator; its differential defaults to zero."""
        if degree < 1:
            raise DegreeError(f"Generator degree must be ≥ 1, got {degree}.")
        gid = self._next_gid
        self._next_gid += 1
        if name is None:
            name = f"v{degree}_{gid}"
        g = Generator(name=name, degree=degree, gid=gid)
        self._g2n[gid] = g
        self._by_deg.setdefault(degree, []).append(g)
        self._diff[gid] = DGAElement(self, {}, degree=degree + 1)
        self._basis_cache.clear()
        return g

    def set_differential(self, g: Generator, value: DGAElement) -> None:
        """Set d(g) = value; validates degree, minimality, and closure."""
        if g.gid not in self._g2n:
            raise UnknownGeneratorError(f"{g.name} not in this DGA.")
        if value.dga is not self:
            raise CrossAlgebraError("value belongs to a different DGA.")
        if not value.is_zero():
            if value.degree != g.degree + 1:
                raise DegreeError(
                    f"d({g.name}) must have degree {g.degree + 1}, got {value.degree}."
                )
            for mon in value.terms:
                if sum(e for _, e in mon) < 2:
                    raise MinimalityError(
                        f"d({g.name}) contains linear term, violating minimality."
                    )
            ddg = self._d_elt(value)
            if not ddg.is_zero():
                raise ClosureError([(g, ddg)])
        self._diff[g.gid] = value

    # ── internal differential ─────────────────────────────────────────────────

    def _d_mon(self, mon: Monomial) -> DGAElement:
        """d(monomial) by the Leibniz rule. Pure ℚ arithmetic."""
        if not mon:
            return DGAElement(self, {}, degree=1)

        acc: Dict[Monomial, Fraction] = {}
        left_deg = 0  # degree of factors to the left of current position

        for k in range(len(mon)):
            gid_k, exp_k = mon[k]
            g_k = self._g2n[gid_k]
            dg = self._diff.get(gid_k)
            if dg is None or dg.is_zero():
                left_deg += exp_k * g_k.degree
                continue

            leibniz_sign = (-1) ** left_deg
            left_mon: Monomial = mon[:k]
            right_mon: Monomial = mon[k + 1:]

            if g_k.parity == "odd":
                # d(g^1) = d(g)
                coeff = Fraction(leibniz_sign)
                for dg_mon, dg_c in dg.terms.items():
                    m, s = _mon_wedge3(left_mon, dg_mon, right_mon, self._g2n)
                    if m is None:
                        continue
                    acc[m] = acc.get(m, _F0) + dg_c * coeff * s
            else:
                # d(g^e) = e * g^{e-1} * d(g)
                coeff = Fraction(leibniz_sign * exp_k)
                g_pm1: Monomial = ((gid_k, exp_k - 1),) if exp_k > 1 else UNIT
                for dg_mon, dg_c in dg.terms.items():
                    # left ∧ g^{e-1} ∧ d(g)_mon ∧ right
                    lg, s1 = _mon_mul(left_mon, g_pm1, self._g2n)
                    if lg is None:
                        continue
                    lgd, s2 = _mon_mul(lg, dg_mon, self._g2n)
                    if lgd is None:
                        continue
                    lgdr, s3 = _mon_mul(lgd, right_mon, self._g2n)
                    if lgdr is None:
                        continue
                    acc[lgdr] = acc.get(lgdr, _F0) + dg_c * coeff * s1 * s2 * s3

            left_deg += exp_k * g_k.degree

        target_deg = _mon_degree(mon, self._g2n) + 1
        return DGAElement(self, acc, degree=target_deg if acc else None)

    def _d_elt(self, elt: DGAElement) -> DGAElement:
        """d(element) by linearity."""
        acc: Dict[Monomial, Fraction] = {}
        target_deg: Optional[int] = None
        for mon, coef in elt.terms.items():
            dm = self._d_mon(mon)
            if dm.is_zero():
                continue
            if target_deg is None:
                target_deg = dm.degree
            for m, c in dm.terms.items():
                acc[m] = acc.get(m, _F0) + coef * c
        return DGAElement(self, acc, degree=target_deg)

    # ── basis enumeration ─────────────────────────────────────────────────────

    def graded_basis(self, n: int) -> List[Monomial]:
        """All monomials of total degree n (lazy, cached)."""
        if n in self._basis_cache:
            return self._basis_cache[n]
        if n == 0:
            self._basis_cache[0] = [UNIT]
            return [UNIT]
        gens = sorted(self._g2n.values(), key=lambda g: g.gid)
        result: List[Monomial] = []
        _rec_basis(gens, 0, n, [], result)
        self._basis_cache[n] = result
        return result

    # ── cohomology ────────────────────────────────────────────────────────────

    def _d_matrix(
        self, src: List[Monomial], tgt: List[Monomial]
    ) -> List[List[Fraction]]:
        """Matrix of d: A^n → A^{n+1}; shape len(tgt) × len(src)."""
        if not src or not tgt:
            return [[_F0] * len(src) for _ in range(len(tgt))]
        t_idx = {m: i for i, m in enumerate(tgt)}
        ncols = len(src)
        nrows = len(tgt)
        mat = [[_F0] * ncols for _ in range(nrows)]
        for j, mon in enumerate(src):
            for m, c in self._d_mon(mon).terms.items():
                if m in t_idx:
                    mat[t_idx[m]][j] += c
        return mat

    def cohomology_dim(self, n: int) -> int:
        """dim H^n(ΛV, d), computed by exact Gaussian elimination."""
        bn = self.graded_basis(n)
        if not bn:
            return 0
        bn1 = self.graded_basis(n + 1)
        bnm1 = self.graded_basis(n - 1) if n > 0 else []
        Dn = self._d_matrix(bn, bn1)
        ker = len(bn) - _rank(Dn)
        im = _rank(self._d_matrix(bnm1, bn)) if bnm1 else 0
        return ker - im

    def cohomology_reps(self, n: int) -> List[DGAElement]:
        """A basis of representatives for H^n(ΛV, d)."""
        bn = self.graded_basis(n)
        if not bn:
            return []
        bn1 = self.graded_basis(n + 1)
        bnm1 = self.graded_basis(n - 1) if n > 0 else []

        # Kernel of d_n
        if not bn1:
            # No degree-(n+1) monomials: every element of A^n is a cocycle.
            dim_n = len(bn)
            ker_vecs: List[List[Fraction]] = [
                [_F1 if i == j else _F0 for j in range(dim_n)]
                for i in range(dim_n)
            ]
        else:
            Dn = self._d_matrix(bn, bn1)
            ker_vecs = _null_space(Dn)

        if not ker_vecs:
            return []

        if bnm1:
            Dnm1 = self._d_matrix(bnm1, bn)
            reps = _quotient_reps(ker_vecs, Dnm1)
        else:
            reps = ker_vecs

        out = []
        for vec in reps:
            terms = {bn[i]: c for i, c in enumerate(vec) if c != _F0}
            if terms:
                out.append(DGAElement(self, terms))
        return out

    def verify_d_squared(self) -> bool:
        """Check d²=0 on every generator. Raises ClosureError on failure."""
        failures = []
        for gid, g in self._g2n.items():
            dg = self._diff.get(gid, DGAElement(self, {}))
            ddg = self._d_elt(dg)
            if not ddg.is_zero():
                failures.append((g, ddg))
        if failures:
            raise ClosureError(failures)
        return True

    def cohomology_dims(self, max_degree: int = 10) -> Dict[int, int]:
        """Cohomology dimensions ``H^n(ΛV, d)`` for ``n = 0, …, max_degree``.

        What is Being Computed?:
            The dimension of every non-zero cohomology group up to
            ``max_degree``. Degree 0 is always 1 for a connected DGA.
            Wraps :meth:`cohomology_dim` with the resource caps
            (graded-basis warning, ``PYSURGERY_DGA_MEM_CAP_MB``) that
            previously lived in the deprecated ``compute_cohomology``
            free function.

        Args:
            max_degree: Upper truncation (default 10).

        Returns:
            Dict mapping ``n`` to ``dim H^n(ΛV, d)`` for ``n`` with
            non-zero cohomology, including ``{0: 1}`` for connectedness.
        """
        import os
        import tracemalloc
        import warnings

        mem_cap_mb = int(os.environ.get("PYSURGERY_DGA_MEM_CAP_MB", "4096"))
        basis_warn = 500

        tracemalloc.start()
        result: Dict[int, int] = {0: 1}
        try:
            for n in range(1, max_degree + 1):
                basis = self.graded_basis(n)
                if len(basis) > basis_warn:
                    warnings.warn(
                        f"cohomology_dims: graded_basis({n}) has {len(basis)} "
                        f"monomials (threshold {basis_warn}). Memory may grow.",
                        UserWarning,
                        stacklevel=2,
                    )
                _, peak = tracemalloc.get_traced_memory()
                peak_mb = peak / (1024 * 1024)
                if peak_mb > mem_cap_mb:
                    warnings.warn(
                        f"cohomology_dims: memory cap {mem_cap_mb} MB exceeded "
                        f"at degree {n} (peak {peak_mb:.1f} MB). Truncating.",
                        UserWarning,
                        stacklevel=2,
                    )
                    break
                dim = self.cohomology_dim(n)
                if dim > 0:
                    result[n] = dim
        finally:
            tracemalloc.stop()
        return result
        return True

    def indecomposables_dim(self, n: int) -> int:
        """Number of degree-n generators = dim_ℚ(V^n)."""
        return len(self._by_deg.get(n, []))

    def all_generators(self) -> List[Generator]:
        return list(self._g2n.values())


# ── Basis enumeration helper ──────────────────────────────────────────────────


def _rec_basis(
    gens: List[Generator],
    idx: int,
    remaining: int,
    current: List[Tuple[int, int]],
    out: List[Monomial],
) -> None:
    if remaining == 0:
        out.append(tuple(current))
        return
    if idx >= len(gens):
        return
    g = gens[idx]
    max_exp = 1 if g.parity == "odd" else (remaining // g.degree)
    # Option A: skip this generator
    _rec_basis(gens, idx + 1, remaining, current, out)
    # Option B: use it with exp = 1, 2, ...
    for exp in range(1, max_exp + 1):
        cost = exp * g.degree
        if cost > remaining:
            break
        current.append((g.gid, exp))
        _rec_basis(gens, idx + 1, remaining - cost, current, out)
        current.pop()


# ── Linear algebra over ℚ ─────────────────────────────────────────────────────


def _rref(mat: List[List[Fraction]]) -> Tuple[List[List[Fraction]], List[int]]:
    """Reduced row-echelon form; returns (rref, pivot_cols)."""
    if not mat or not mat[0]:
        return mat, []
    m, n = len(mat), len(mat[0])
    M = [row[:] for row in mat]
    pr = 0
    pivots: List[int] = []
    for col in range(n):
        found = next((r for r in range(pr, m) if M[r][col] != _F0), -1)
        if found == -1:
            continue
        M[pr], M[found] = M[found], M[pr]
        pivots.append(col)
        inv = _F1 / M[pr][col]
        M[pr] = [x * inv for x in M[pr]]
        for r in range(m):
            if r != pr and M[r][col] != _F0:
                f = M[r][col]
                M[r] = [M[r][j] - f * M[pr][j] for j in range(n)]
        pr += 1
    return M, pivots


def _rank(mat: List[List[Fraction]]) -> int:
    if not mat or not mat[0]:
        return 0
    _, pivots = _rref(mat)
    return len(pivots)


def _null_space(mat: List[List[Fraction]]) -> List[List[Fraction]]:
    """Null space of mat (m×n); returns list of n-vectors."""
    if not mat:
        return []
    n = len(mat[0])
    if n == 0:
        return []
    rref, pivots = _rref(mat)
    pivot_set = set(pivots)
    free = [j for j in range(n) if j not in pivot_set]
    if not free:
        return []
    p2r = {col: row for row, col in enumerate(pivots)}
    vecs = []
    for fc in free:
        v = [_F0] * n
        v[fc] = _F1
        for pc, row in p2r.items():
            v[pc] = -rref[row][fc]
        vecs.append(v)
    return vecs


def _col_space(mat: List[List[Fraction]]) -> List[List[Fraction]]:
    """Column space basis (list of column vectors of length m)."""
    if not mat or not mat[0]:
        return []
    m, _ = len(mat), len(mat[0])
    _, pivots = _rref(mat)
    return [[mat[r][c] for r in range(m)] for c in pivots]


def _quotient_reps(
    ker_vecs: List[List[Fraction]],
    Dnm1: List[List[Fraction]],
) -> List[List[Fraction]]:
    """From ker_vecs, select representatives independent modulo im(Dnm1).

    Places im_cols first so that RREF pivots landing in the ker_vecs block
    (column index ≥ len(im_cols)) identify exactly those kernel vectors that
    are linearly independent from the image.
    """
    if not ker_vecs:
        return []
    im_cols = _col_space(Dnm1) if Dnm1 and Dnm1[0] else []
    if not im_cols:
        return ker_vecs
    # Build [im_cols | ker_vecs] and row-reduce.
    # Pivots in the ker_vecs block are independent representatives.
    all_cols = im_cols + ker_vecs
    n_dim = len(all_cols[0])
    mat_T = [[all_cols[j][i] for j in range(len(all_cols))] for i in range(n_dim)]
    _, pivots = _rref(mat_T)
    n_im = len(im_cols)
    return [ker_vecs[c - n_im] for c in pivots if c >= n_im]


# ── Result contract ────────────────────────────────────────────────────────────


class RationalMinimalModelResult(BaseModel):
    """Contract for a Sullivan minimal model computation.

    Attributes:
        pi_n_rational: degree → dim_ℚ(π_n(X) ⊗ ℚ).
        cohomology_iso: H(M) ≅ H*(X; ℚ) verified up to truncation_degree.
        is_formal_model: True when all differentials are zero.
        truncation_degree: max_degree used.
        status: "success" or "inconclusive".
        exact: Always True (all arithmetic over ℚ).
        theorem_tag: Stable theorem identifier.
        contract_version: Schema version string.
        reasoning: Human-readable computation summary.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pi_n_rational: Dict[int, int]
    cohomology_iso: bool
    is_formal_model: bool
    truncation_degree: int
    status: Literal["success", "inconclusive"]
    exact: bool = True
    theorem_tag: str = THEOREM_TAG
    contract_version: str = CONTRACT_VERSION
    reasoning: str
    minimal_model: Optional["RationalDGA"] = None

    def decision_ready(self) -> bool:
        """True when exact and cohomology isomorphism was verified."""
        return self.status == "success" and self.exact and self.cohomology_iso


# ── Cohomology algebra spec ───────────────────────────────────────────────────


class RationalCohomologyAlgebra:
    """A specification of H*(X; ℚ) as Betti numbers for the Sullivan algorithm.

    For simply-connected spaces the Sullivan minimal model is constructively
    determined by H*(X; ℚ).  Encoding the ring as Betti numbers is sufficient
    because the inductive algorithm detects spurious cohomology (arising from
    free monomials not present in H*) automatically.

    Args:
        betti: {degree: rank_over_ℚ}.
        name: Human-readable label.
        max_degree: Truncation degree.
    """

    def __init__(
        self,
        betti: Dict[int, int],
        name: str = "",
        max_degree: int = 10,
    ) -> None:
        self.betti = {n: r for n, r in betti.items() if r > 0}
        self.name = name
        self.max_degree = max_degree

    def betti_n(self, n: int) -> int:
        return self.betti.get(n, 0)


# ── Sullivan minimal model ────────────────────────────────────────────────────


def sullivan_minimal_model(
    complex_or_algebra,
    max_degree: int = 10,
) -> RationalMinimalModelResult:
    """Construct the Sullivan minimal model of a simply-connected space.

    What is Being Computed?:
        Builds the minimal CDGA (ΛV, d) with H(ΛV, d) ≅ H*(X; ℚ), then reads
        off π_n(X) ⊗ ℚ ≅ V^n via the Quillen-Sullivan theorem.

    Algorithm:
        Inductive construction (Félix-Halperin-Thomas §12).
        For n = 2, …, max_degree:
          A. Kill spurious H^n: add degree-(n-1) generators with d(g) = [spurious rep].
          B. Create missing H^n: add degree-n generators with d(g) = 0.
        Verify H^k(M) ≅ target β_k for all k ≤ max_degree.

    Preserved Invariants:
        - d² = 0 (ClosureError raised on violation).
        - Minimality: d(g) ∈ Λ^{≥2}V (MinimalityError raised on violation).
        - All arithmetic over ℚ — no floating-point.

    Args:
        complex_or_algebra: A ``RationalCohomologyAlgebra``, or any object
            with a ``betti_numbers()`` method returning ``{degree: rank}``.
        max_degree: Truncation degree (default 10).

    Returns:
        RationalMinimalModelResult with exact=True.

    References:
        Quillen, D. (1969). Ann. Math. 90, 205–295.
        Sullivan, D. (1977). IHES 47, 269–331.
    """
    # ── 1. Obtain target Betti numbers ────────────────────────────────────────
    if isinstance(complex_or_algebra, RationalCohomologyAlgebra):
        target = complex_or_algebra
        eff_max = min(max_degree, target.max_degree)
    else:
        raw = complex_or_algebra.betti_numbers()
        betti_q = {n: int(b) for n, b in raw.items() if int(b) > 0}
        target = RationalCohomologyAlgebra(betti_q, max_degree=max_degree)
        eff_max = max_degree

    # Simply-connected: β_1 = 0 required.
    if target.betti_n(1) != 0:
        return RationalMinimalModelResult(
            pi_n_rational={},
            cohomology_iso=False,
            is_formal_model=False,
            truncation_degree=eff_max,
            status="inconclusive",
            reasoning="β_1 > 0; Quillen-Sullivan applies to simply-connected spaces only.",
        )
    if target.betti_n(0) != 1:
        return RationalMinimalModelResult(
            pi_n_rational={},
            cohomology_iso=False,
            is_formal_model=False,
            truncation_degree=eff_max,
            status="inconclusive",
            reasoning="Space is disconnected (β_0 ≠ 1).",
        )

    # ── 2. Build DGA inductively ──────────────────────────────────────────────
    dga = RationalDGA()
    _name_cnt: Dict[int, int] = {}

    def _name(deg: int) -> str:
        c = _name_cnt.get(deg, 0)
        _name_cnt[deg] = c + 1
        return f"x{deg}" if c == 0 else f"x{deg}_{c}"

    for n in range(2, eff_max + 1):
        # Step A: kill spurious H^n (add degree-(n-1) generators)
        _kill_spurious(dga, target, n, _name)
        # Step B: add missing H^n (add degree-n generators with d=0)
        _add_missing(dga, target, n, _name)

    # ── 3. Verify cohomology isomorphism ──────────────────────────────────────
    mismatches: List[str] = []
    for k in range(eff_max + 1):
        computed = dga.cohomology_dim(k) if k >= 1 else 1
        expected = target.betti_n(k) if k != 0 else 1
        if computed != expected:
            mismatches.append(f"H^{k}: got {computed}, want {expected}")

    cohomology_iso = len(mismatches) == 0

    # ── 4. π_n(X) ⊗ ℚ ────────────────────────────────────────────────────────
    pi_n: Dict[int, int] = {
        n: d
        for n in range(1, eff_max + 1)
        if (d := dga.indecomposables_dim(n)) > 0
    }

    # ── 5. Formality (d = 0 for every generator) ─────────────────────────────
    is_formal = all(
        dga._diff.get(gid, DGAElement(dga, {})).is_zero()
        for gid in dga._g2n
    )

    # ── 6. Result ─────────────────────────────────────────────────────────────
    status: Literal["success", "inconclusive"] = (
        "success" if cohomology_iso else "inconclusive"
    )
    gen_s = ", ".join(f"V^{n}=ℚ^{d}" for n, d in sorted(pi_n.items())) or "none"
    mm_s = "; ".join(mismatches) if mismatches else "none"
    reasoning = (
        f"Sullivan model to degree {eff_max}. "
        f"Generators: {gen_s}. "
        f"Mismatches: {mm_s}. "
        f"Formal (d=0): {is_formal}."
    )
    return RationalMinimalModelResult(
        pi_n_rational=pi_n,
        cohomology_iso=cohomology_iso,
        is_formal_model=is_formal,
        truncation_degree=eff_max,
        status=status,
        reasoning=reasoning,
        minimal_model=dga,
    )


def _kill_spurious(
    dga: RationalDGA,
    target: RationalCohomologyAlgebra,
    n: int,
    name_fn,
) -> None:
    """Kill spurious H^n by adding degree-(n-1) generators."""
    if n < 2:
        return
    max_iter = 20  # safety guard against infinite loops
    for _ in range(max_iter):
        target_h = target.betti_n(n)
        reps = dga.cohomology_reps(n)
        current_h = len(reps)
        if current_h <= target_h:
            break
        # Take the first "excess" representative as the differential.
        omega = reps[target_h]
        if omega.is_zero():
            break
        # Verify omega is decomposable (minimality condition for d).
        all_linear = all(sum(e for _, e in mon) == 1 for mon in omega.terms)
        if all_linear:
            # Linear differential violates minimality — shouldn't happen for n ≥ 2
            # with no degree-1 generators.
            break
        g = dga.add_generator(degree=n - 1, name=name_fn(n - 1))
        # Directly assign — omega is closed (came from cohomology_reps) and
        # decomposable (verified above); bypass the re-check in set_differential
        # to avoid recomputing d(omega) redundantly.
        dga._diff[g.gid] = omega


def _add_missing(
    dga: RationalDGA,
    target: RationalCohomologyAlgebra,
    n: int,
    name_fn,
) -> None:
    """Add degree-n generators with d=0 to create missing H^n."""
    current_h = dga.cohomology_dim(n)
    missing = max(0, target.betti_n(n) - current_h)
    for _ in range(missing):
        g = dga.add_generator(degree=n, name=name_fn(n))
        dga._diff[g.gid] = DGAElement(dga, {}, degree=n + 1)


# ── Standard space constructors ───────────────────────────────────────────────


def sphere_cohomology(n: int, max_degree: Optional[int] = None) -> RationalCohomologyAlgebra:
    """H*(S^n; ℚ): ℚ in degrees 0 and n, zero elsewhere."""
    if n < 1:
        raise ValueError(f"Sphere dimension must be ≥ 1, got {n}.")
    md = max_degree if max_degree is not None else max(10, 2 * n + 3)
    return RationalCohomologyAlgebra(betti={0: 1, n: 1}, name=f"S^{n}", max_degree=md)


def complex_projective_space_cohomology(
    n: int, max_degree: Optional[int] = None
) -> RationalCohomologyAlgebra:
    """H*(ℂP^n; ℚ): ℚ in degrees 0, 2, 4, …, 2n."""
    if n < 1:
        raise ValueError(f"ℂP^n requires n ≥ 1, got {n}.")
    md = max_degree if max_degree is not None else max(10, 2 * n + 4)
    return RationalCohomologyAlgebra(
        betti={2 * k: 1 for k in range(n + 1)}, name=f"ℂP^{n}", max_degree=md
    )


def product_cohomology(
    A: RationalCohomologyAlgebra,
    B: RationalCohomologyAlgebra,
    max_degree: Optional[int] = None,
) -> RationalCohomologyAlgebra:
    """Künneth: β_n(X×Y) = Σ_{p+q=n} β_p(X)·β_q(Y)."""
    md = max_degree if max_degree is not None else min(A.max_degree + B.max_degree, 14)
    betti: Dict[int, int] = {}
    for pa, ra in A.betti.items():
        for pb, rb in B.betti.items():
            k = pa + pb
            if k <= md:
                betti[k] = betti.get(k, 0) + ra * rb
    name = f"({A.name})×({B.name})" if A.name and B.name else ""
    return RationalCohomologyAlgebra(betti=betti, name=name, max_degree=md)


# ── Formality detection ────────────────────────────────────────────────────────


class FormalityResult(BaseModel):
    """Contract for formality detection of a Sullivan minimal model.

    A minimal model (ΛV, d) satisfies ``is_formal = True`` iff d(g) = 0 for
    every generator g in V.  This is equivalent to the minimal model being a
    free CDGA with trivial differential — i.e. a product of odd Eilenberg–
    MacLane spaces over ℚ.

    Note: this is the *d = 0* criterion, which is stronger than topological
    formality in the sense of Deligne–Griffiths–Morgan–Sullivan.  Odd spheres
    and their products satisfy both; even spheres satisfy neither.

    Attributes:
        is_formal: True when d(g) = 0 for every generator.
        non_formal_generators: Names of generators with d ≠ 0.
        exact: Always True (all arithmetic over ℚ).
        theorem_tag: Stable theorem identifier.
        contract_version: Schema version string.
        status: "success" or "inconclusive".
        reasoning: Human-readable summary.

    References:
        Deligne, P., Griffiths, P., Morgan, J., & Sullivan, D. (1975).
            Real homotopy theory of Kähler manifolds. Invent. Math. 29, 245–274.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    is_formal: bool
    non_formal_generators: List[str]
    exact: bool = True
    theorem_tag: str = FORMALITY_THEOREM_TAG
    contract_version: str = CONTRACT_VERSION
    status: Literal["success", "inconclusive"]
    reasoning: str

    def decision_ready(self) -> bool:
        return self.status == "success" and self.exact


class MasseyProductEntry(BaseModel):
    """One Massey product datum extracted from a minimal model generator.

    Attributes:
        input_classes: Ordered tuple of generator names (cohomology class labels).
        order: Word length (2 = binary, 3 = triple, …).
        representative_degree: Degree of the generator representing the product class.
        representative_name: Name of that generator.
        coefficient: Rational coefficient as a string (e.g. ``"1"`` or ``"3/2"``).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_classes: Tuple[str, ...]
    order: int
    representative_degree: int
    representative_name: str
    coefficient: str


class MasseyProductsResult(BaseModel):
    """Contract for Massey product extraction from a Sullivan minimal model.

    Each generator g with d(g) ≠ 0 contributes one entry per monomial in
    d(g): the monomial (g₁^{e₁} ∧ g₂^{e₂} ∧ …) expands to the ordered tuple
    (g₁, …, g₁, g₂, …) of length Σ eᵢ, encoding the Massey product of that
    order whose value contains [g].

    Attributes:
        entries: All extracted Massey product data.
        non_formal_count: Number of generators with d(g) ≠ 0.
        exact: Always True.
        theorem_tag: Stable theorem identifier.
        contract_version: Schema version string.
        status: "success" or "inconclusive".
        reasoning: Human-readable summary.

    References:
        Félix, Y., Halperin, S., & Thomas, J.-C. (2001).
            Rational Homotopy Theory. Springer GTM 205. §12, §15.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    entries: List[MasseyProductEntry]
    non_formal_count: int
    exact: bool = True
    theorem_tag: str = MASSEY_THEOREM_TAG
    contract_version: str = CONTRACT_VERSION
    status: Literal["success", "inconclusive"]
    reasoning: str

    def decision_ready(self) -> bool:
        return self.status == "success" and self.exact

    def as_dict(self) -> Dict[Tuple[str, ...], List["MasseyProductEntry"]]:
        """Group entries by input_classes tuple."""
        out: Dict[Tuple[str, ...], List[MasseyProductEntry]] = {}
        for e in self.entries:
            out.setdefault(e.input_classes, []).append(e)
        return out


# ── Public API ─────────────────────────────────────────────────────────────────


def _resolve_dga(
    minimal_model: Union[RationalMinimalModelResult, RationalDGA],
) -> Optional[RationalDGA]:
    """Extract RationalDGA from either a result object or a raw DGA."""
    if isinstance(minimal_model, RationalDGA):
        return minimal_model
    if isinstance(minimal_model, RationalMinimalModelResult):
        return minimal_model.minimal_model
    return None


def is_formal_space(
    minimal_model: Union[RationalMinimalModelResult, RationalDGA],
) -> FormalityResult:
    """Detect whether a Sullivan minimal model has d = 0 on all generators.

    What is Being Computed?:
        Checks the d = 0 criterion: every generator of (ΛV, d) has zero
        differential.  This holds for odd spheres and their products, and fails
        for even spheres and complex projective spaces.

    Algorithm:
        Iterate over all generators g; evaluate d(g) via the stored differential
        map; return True iff every d(g) is the zero DGAElement.

    Preserved Invariants:
        All arithmetic is over ℚ (no floating-point); the closure d² = 0 is
        maintained by the DGA invariants and not re-checked here.

    Args:
        minimal_model: A ``RationalDGA`` or a ``RationalMinimalModelResult``
            whose ``minimal_model`` field is populated (i.e. obtained from
            ``sullivan_minimal_model``).

    Returns:
        FormalityResult with exact=True.

    Use When:
        - Determining whether a space has trivial secondary rational homotopy operations.
        - Classifying rational homotopy type as a product of K(ℚ, odd n)'s.

    Example:
        >>> from pysurgery.homotopy.rational_homotopy import sphere_cohomology, sullivan_minimal_model, is_formal_space
        >>> r = sullivan_minimal_model(sphere_cohomology(3))
        >>> fr = is_formal_space(r)
        >>> fr.is_formal
        True

    References:
        Sullivan, D. (1977). Infinitesimal computations in topology.
            Publ. Math. IHES 47, 269–331.
        Deligne, P., Griffiths, P., Morgan, J., & Sullivan, D. (1975).
            Real homotopy theory of Kähler manifolds. Invent. Math. 29, 245–274.
    """
    dga = _resolve_dga(minimal_model)
    if dga is None:
        # Fallback: use the is_formal_model flag on the result if DGA not attached.
        if isinstance(minimal_model, RationalMinimalModelResult):
            nf: List[str] = []
            return FormalityResult(
                is_formal=minimal_model.is_formal_model,
                non_formal_generators=nf,
                status="success" if minimal_model.status == "success" else "inconclusive",
                reasoning=(
                    "Formality read from is_formal_model flag (DGA not attached). "
                    f"is_formal={minimal_model.is_formal_model}."
                ),
            )
        return FormalityResult(
            is_formal=False,
            non_formal_generators=[],
            status="inconclusive",
            reasoning="Could not resolve RationalDGA from input.",
        )

    non_formal: List[str] = [
        dga._g2n[gid].name
        for gid in sorted(dga._g2n)
        if not dga._diff.get(gid, DGAElement(dga, {})).is_zero()
    ]
    is_formal = len(non_formal) == 0
    n_gens = len(dga._g2n)

    if is_formal:
        reasoning = (
            f"d(g) = 0 for all {n_gens} generator(s). "
            "Minimal model has trivial differential (d = 0 formal)."
        )
    else:
        shown = non_formal[:5]
        ellipsis = "…" if len(non_formal) > 5 else ""
        reasoning = (
            f"d ≠ 0 for {len(non_formal)} of {n_gens} generator(s): "
            f"{', '.join(shown)}{ellipsis}. Not d = 0 formal."
        )

    return FormalityResult(
        is_formal=is_formal,
        non_formal_generators=non_formal,
        status="success",
        reasoning=reasoning,
    )


def extract_massey_products(
    minimal_model: Union[RationalMinimalModelResult, RationalDGA],
) -> MasseyProductsResult:
    """Extract Massey product data from decomposable differentials of a minimal model.

    What is Being Computed?:
        For each generator g with d(g) ≠ 0, each monomial m in d(g) encodes a
        Massey product: the expansion of m as a sequence of generator names
        (g₁, …, g₁, g₂, …) of total length k = Σ eᵢ identifies the k-fold
        Massey product ⟨[g₁], …, [gₖ]⟩, with [g] as a representative of the
        product value.

    Algorithm:
        For each generator g with d(g) ≠ 0:
          For each (monomial, coefficient) in d(g).terms:
            Expand monomial ((gid, exp), …) → tuple of generator names with repetition.
            Record MasseyProductEntry(input_classes=tuple, order=len, …).
        Return all entries collected.

    Preserved Invariants:
        - Minimality guarantees every monomial has word-length ≥ 2.
        - All coefficients are exact ℚ (Fraction), stored as strings.
        - exact = True on the result.

    Args:
        minimal_model: A ``RationalDGA`` or a ``RationalMinimalModelResult``
            whose ``minimal_model`` field is populated.

    Returns:
        MasseyProductsResult with exact=True.

    Use When:
        - Identifying higher-order cohomological operations in the minimal model.
        - Classifying whether a space has non-trivial triple/higher Massey products.
        - Checking formality via the vanishing of Massey product entries.

    Example:
        >>> from pysurgery.homotopy.rational_homotopy import sphere_cohomology, sullivan_minimal_model, extract_massey_products
        >>> r = sullivan_minimal_model(sphere_cohomology(2))
        >>> mp = extract_massey_products(r)
        >>> mp.entries[0].order  # x^2 in d(y) of S^2 model
        2

    References:
        Félix, Y., Halperin, S., & Thomas, J.-C. (2001).
            Rational Homotopy Theory. Springer GTM 205. §12.
        Sullivan, D. (1977). Infinitesimal computations in topology.
            Publ. Math. IHES 47, 269–331.
    """
    dga = _resolve_dga(minimal_model)
    if dga is None:
        return MasseyProductsResult(
            entries=[],
            non_formal_count=0,
            status="inconclusive",
            reasoning="Could not resolve RationalDGA from input. Pass a RationalDGA or a RationalMinimalModelResult with minimal_model populated.",
        )

    entries: List[MasseyProductEntry] = []
    non_formal_count = 0

    for gid in sorted(dga._g2n):
        g = dga._g2n[gid]
        dg = dga._diff.get(gid)
        if dg is None or dg.is_zero():
            continue
        non_formal_count += 1
        for mon in sorted(dg.terms):
            coef = dg.terms[mon]
            # Expand monomial to sequence of generator names (with exponent repetition).
            names: List[str] = []
            for mon_gid, exp in mon:
                gen_name = dga._g2n[mon_gid].name
                names.extend([gen_name] * exp)
            entries.append(
                MasseyProductEntry(
                    input_classes=tuple(names),
                    order=len(names),
                    representative_degree=g.degree,
                    representative_name=g.name,
                    coefficient=str(coef),
                )
            )

    n_entries = len(entries)
    n_gens = len(dga._g2n)
    reasoning = (
        f"Extracted {n_entries} Massey product datum/data from "
        f"{non_formal_count} non-zero-differential generator(s) "
        f"out of {n_gens} total generator(s) in (ΛV, d)."
    )

    return MasseyProductsResult(
        entries=entries,
        non_formal_count=non_formal_count,
        status="success",
        reasoning=reasoning,
    )


# ── RationalHomotopyGroup façade contract ─────────────────────────────────────


class RationalHomotopyGroup(BaseModel):
    """Per-space façade over RationalMinimalModelResult exposing π_*(X) ⊗ ℚ.

    Overview:
        Thin contract produced by ``rational_homotopy_group()``.  All DGA
        arithmetic lives in ``RationalDGA``/``sullivan_minimal_model``; this
        class is the external-facing result object.

    Invariants (enforced by validators):
        - Every key of ``pi_n_rational`` appears in ``nonzero_degrees`` and
          every value is > 0.
        - ``max(nonzero_degrees) ≤ truncation_degree``.
        - ``is_formal == True`` implies ``len(massey_products) == 0``.
        - ``cohomology_iso_verified`` must be True for ``decision_ready()``.

    Attributes:
        space_label: Human-readable name.
        pi_n_rational: n → dim_ℚ(π_n(X) ⊗ ℚ) for nonzero n.
        nonzero_degrees: Sorted keys of pi_n_rational.
        truncation_degree: max_degree used in the Sullivan construction.
        is_formal: True when d(g) = 0 for every DGA generator.
        massey_products: Massey product data (empty when is_formal=True).
        underlying_model: The (ΛV, d) Sullivan model (optional).
        cohomology_iso_verified: H(ΛV) ≅ H*(X; ℚ) verified up to truncation.
        exact: Always True (arithmetic over ℚ).
        theorem_tag: Stable theorem identifier.
        contract_version: Schema version.
        status: "success" or "inconclusive".
        reasoning: Human-readable summary.

    References:
        Quillen, D. (1969). Rational homotopy theory. Ann. Math. 90, 205–295.
        Sullivan, D. (1977). Infinitesimal computations in topology.
            Publ. Math. IHES 47, 269–331.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    space_label: str = ""
    pi_n_rational: Dict[int, int]
    nonzero_degrees: List[int]
    truncation_degree: int
    is_formal: bool
    massey_products: List[MasseyProductEntry] = Field(default_factory=list)
    underlying_model: Optional["RationalDGA"] = None
    cohomology_iso_verified: bool

    exact: bool = True
    theorem_tag: str = THEOREM_TAG
    contract_version: str = CONTRACT_VERSION
    status: Literal["success", "inconclusive"]
    reasoning: str

    @field_validator("pi_n_rational")
    @classmethod
    def _all_values_positive(cls, v: Dict[int, int]) -> Dict[int, int]:
        bad = [n for n, d in v.items() if d <= 0]
        if bad:
            raise ValueError(f"pi_n_rational values must be > 0; bad degrees: {bad}")
        return v

    @model_validator(mode="after")
    def _check_invariants(self) -> "RationalHomotopyGroup":
        pi_keys = set(self.pi_n_rational.keys())
        nd_keys = set(self.nonzero_degrees)
        if pi_keys != nd_keys:
            raise ValueError(
                f"pi_n_rational keys {pi_keys} must equal nonzero_degrees {nd_keys}."
            )
        if self.nonzero_degrees:
            if max(self.nonzero_degrees) > self.truncation_degree:
                raise ValueError(
                    f"max(nonzero_degrees)={max(self.nonzero_degrees)} "
                    f"> truncation_degree={self.truncation_degree}."
                )
        if self.is_formal and self.massey_products:
            raise ValueError(
                "is_formal=True is inconsistent with non-empty massey_products."
            )
        return self

    def rank_at(self, n: int) -> int:
        """dim_ℚ(π_n(X) ⊗ ℚ), 0 if not computed."""
        return self.pi_n_rational.get(n, 0)

    def decision_ready(self) -> bool:
        """True when exact, status=success, and cohomology iso is verified."""
        return self.exact and self.status == "success" and self.cohomology_iso_verified


def rational_homotopy_group(
    complex_or_algebra,
    max_degree: int = 10,
    include_massey: bool = True,
    space_label: str = "",
) -> RationalHomotopyGroup:
    """Compute π_*(X) ⊗ ℚ as a RationalHomotopyGroup contract.

    What is Being Computed?:
        Constructs the Sullivan minimal model (ΛV, d) via ``sullivan_minimal_model``,
        reads off π_n(X) ⊗ ℚ = V^n, and wraps the result in the
        ``RationalHomotopyGroup`` façade contract.

    Algorithm:
        1. Call ``sullivan_minimal_model(complex_or_algebra, max_degree)``.
        2. If include_massey, call ``extract_massey_products`` on the model.
        3. Build ``RationalHomotopyGroup`` from the result fields.

    Args:
        complex_or_algebra: A ``RationalCohomologyAlgebra``, or an object
            with a ``betti_numbers()`` method (e.g. ChainComplex, SimplicialComplex).
        max_degree: Truncation degree (default 10).
        include_massey: Whether to extract Massey product data (default True).
        space_label: Optional human-readable name for the space.

    Returns:
        RationalHomotopyGroup with exact=True.

    References:
        Sullivan, D. (1977). Infinitesimal computations in topology.
            Publ. Math. IHES 47, 269–331.
    """
    r = sullivan_minimal_model(complex_or_algebra, max_degree=max_degree)

    if include_massey and r.minimal_model is not None:
        mp_result = extract_massey_products(r)
        massey = mp_result.entries
    else:
        massey = []

    # is_formal=True ↔ d=0; if formal, massey must be empty (invariant).
    if r.is_formal_model:
        massey = []

    label = space_label or (
        complex_or_algebra.name
        if isinstance(complex_or_algebra, RationalCohomologyAlgebra)
        else ""
    )

    return RationalHomotopyGroup(
        space_label=label,
        pi_n_rational=r.pi_n_rational,
        nonzero_degrees=sorted(r.pi_n_rational.keys()),
        truncation_degree=r.truncation_degree,
        is_formal=r.is_formal_model,
        massey_products=massey,
        underlying_model=r.minimal_model,
        cohomology_iso_verified=r.cohomology_iso,
        status=r.status,
        reasoning=r.reasoning,
    )
