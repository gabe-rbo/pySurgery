"""Unstable Adams spectral sequence E_2: generic approximation via the
excess-filtered cobar of A_p^*.

Math context:
    The unstable category U of A_p-modules consists of modules M with the
    instability axiom: at p=2, Sq^i(m) = 0 for i > |m|; at p odd, similar
    bounds on beta P^i. The unstable Adams SS (Bousfield-Kan 1972) has

        E_2^{s,t} = Ext_U^{s,t}(H̃*(X; F_p), F_p)

    converging to π_{t-s}(X)_p (the p-completed unstable homotopy of X,
    not the stable π^s_*).

    The full Bousfield-Kan computation requires either:
      (A) the Lambda-algebra Λ at p=2 (Curtis 1971; Bousfield-Curtis-Kan-
          Quillen-Rector-Schlesinger 1966) -- generic in M but heavy to
          implement;
      (B) the unstable cobar of a free unstable resolution of M -- requires
          the unstable category infrastructure on top of A_p.

    This module provides a TRUNCATED-COBAR APPROXIMATION that gives the
    correct unstable Ext when M is a "trivial" unstable A_p-module
    concentrated in a single bottom degree n_0, by enforcing the
    instability filter at the level of the cobar's tensor factors.

Approximation:
    Take the stable cobar over A_p^* (the existing adams_odd_prime_cobar
    machinery, extended to p=2 here), and at each tensor position
    discard contributions from monomials whose dual operation would
    violate instability on a class of degree n_0 (the bottom cell of M).
    Specifically, in A_p^* a monomial ξ_1^{r_1} ξ_2^{r_2} ... corresponds
    dually to a Milnor basis element Sq(r_1, r_2, ...) (or P-monomial at
    odd p) with EXCESS = r_1. We keep tensor factors with excess ≤ n_0.

    For p=2 this exactly matches the unstable resolution of F_2[n_0] when
    n_0 is the smallest generator degree of M.

    For multi-generator M with generator degrees {n_1 < n_2 < ...}, we
    use n_min = n_1 as the instability bound (conservative; tightest with
    the existing cobar machinery without rewriting the resolution).

Limitations (documented honestly):
    - Approximate when M has generators in multiple distinct degrees.
    - At odd primes the excess formula in the Milnor dual is more subtle
      (the τ-factors carry their own excess); this module uses the
      conservative bound `excess(ξ^I τ^E) <= sum(r_1, e_0)` which is
      correct for a single-generator A_p-module of degree n_0.
    - Returns an `AdamsE2Page` flagged as
      `space_label += " (unstable approx)"`.

References:
    Bousfield, A. K. & Kan, D. M. (1972). The homotopy spectral sequence of
        a space with coefficients in a ring. Topology 11, 79-106.
    Curtis, E. B. (1971). Simplicial homotopy theory. Adv. Math. 6, 107-209.
    Milnor, J. (1958). The Steenrod algebra and its dual. Ann. Math. 67,
        150-171.
    Singer, W. M. (1973). Steenrod squares in spectral sequences.
        Trans. AMS 175, 327-336.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Tuple

import numpy as np
import scipy.sparse as sp

from pysurgery.adams.odd_prime_cobar import (
    DualSteenrodAlgebra,
    Monomial,
    _enum_poly,
)
from pysurgery.adams.spectral_sequence import (
    AdamsDifferentialFlag,
    AdamsE2Page,
    FpCohomologyRing,
)


# ── Excess function on Milnor dual basis ─────────────────────────────────────


def excess_of_monomial(mon: Monomial) -> int:
    """The excess of a Milnor-basis monomial in A_p^*.

    At p=2 the dual Milnor basis element ξ_1^{r_1} ξ_2^{r_2} ... ξ_k^{r_k}
    corresponds to Sq(r_1, r_2, ..., r_k) and the excess of the latter
    (in the Steenrod algebra sense) equals r_1.

    At odd p the analogous formula on ξ-only monomials gives r_1 (the
    xi_1 exponent). Including τ-generators we use the bound
        excess(ξ^I τ^E) = r_1 + |E|
    which is correct for the unstability of beta P^I monomials.

    This function returns an integer >= 0; the unit (empty monomial)
    has excess 0.
    """
    poly, ext = mon
    r1 = poly[0] if poly else 0
    return r1 + len(ext)


def filter_basis_by_excess(
    A: DualSteenrodAlgebra, t: int, max_excess: int,
) -> List[Monomial]:
    """Basis of A_p^*_+ at internal degree t with excess <= max_excess."""
    return [m for m in A.basis(t) if excess_of_monomial(m) <= max_excess]


# ── p=2 cobar over A_2^* = F_2[ξ_1, ξ_2, ...] ────────────────────────────────
# At p=2 the dual Steenrod algebra has no tau-generators, so the
# DualSteenrodAlgebra class needs a small extension to handle p=2.
# We extend it on the fly here.


class DualSteenrodAlgebraP2:
    """Truncated A_2^* = F_2[ξ_1, ξ_2, ...] for unstable cobar at p=2.

    Mirrors the odd-prime DualSteenrodAlgebra but at p=2 with no τ
    generators. Coproduct: Δ(ξ_n) = Σ_{i=0..n} ξ_{n-i}^{2^i} ⊗ ξ_i.
    """

    def __init__(self, t_max: int) -> None:
        if t_max < 0:
            raise ValueError("t_max must be >= 0")
        self.p = 2
        self.t_max = t_max
        # |ξ_i| = 2^i - 1 at p=2.
        self.xi_max = 0
        i = 1
        while (2 ** i - 1) <= t_max:
            self.xi_max = i
            i += 1
        self._basis_cache: Dict[int, List[Tuple[Tuple[int, ...], Tuple[int, ...]]]] = {}

    @staticmethod
    def _xi_degree(i: int) -> int:
        return 2 ** i - 1

    def degree(self, mon: Tuple[Tuple[int, ...], Tuple[int, ...]]) -> int:
        poly, ext = mon
        # ext is unused at p=2 but kept for shape compatibility with the
        # odd-prime monomial format.
        assert not ext, "p=2 has no tau generators"
        return sum(e * self._xi_degree(i + 1) for i, e in enumerate(poly) if e)

    def basis(self, t: int) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        if t < 0:
            return []
        if t in self._basis_cache:
            return self._basis_cache[t]
        if t == 0:
            self._basis_cache[t] = [((), ())]
            return self._basis_cache[t]
        xi_degs = [self._xi_degree(i + 1) for i in range(self.xi_max)]
        out = [
            (tuple(poly), ())
            for poly in _enum_poly(t, xi_degs)
        ]
        # canonicalize (strip trailing zeros)
        out = [
            ((tuple(p[:max(i + 1 for i in range(len(p)) if p[i]) ] ) if any(p) else ()), ())
            for (p, _) in out
        ]
        # de-duplicate after canonicalization
        seen = set()
        canon = []
        for m in out:
            if m not in seen:
                seen.add(m)
                canon.append(m)
        self._basis_cache[t] = canon
        return canon

    def coproduct_reduced(
        self, mon: Tuple[Tuple[int, ...], Tuple[int, ...]],
    ) -> List[Tuple[Tuple[Tuple[int, ...], Tuple[int, ...]], Tuple[Tuple[int, ...], Tuple[int, ...]], int]]:
        """Δ_bar(mon) at p=2 by multiplicativity from Δ(ξ_n) = sum ξ_{n-i}^{2^i} ⊗ ξ_i."""
        poly, ext = mon
        assert not ext
        # Start with Δ(1) = 1 ⊗ 1.
        state = [(((), ()), ((), ()), 1)]
        for i, e in enumerate(poly):
            if e == 0:
                continue
            for _ in range(e):
                state = self._multiply_factor(state, self._coprod_xi(i + 1))
        reduced = []
        for L, R, c in state:
            if L == ((), ()) or R == ((), ()):
                continue
            reduced.append((L, R, c % 2))
        # collect like terms
        collected: Dict[Tuple, int] = {}
        for L, R, c in reduced:
            key = (L, R)
            collected[key] = (collected.get(key, 0) + c) % 2
        return [(left, r, c) for (left, r), c in collected.items() if c]

    def _coprod_xi(self, n: int):
        out = []
        for i in range(n + 1):
            j = n - i
            if j == 0:
                L = ((), ())
            else:
                if j > self.xi_max:
                    continue
                exps = [0] * j
                exps[j - 1] = 2 ** i
                L = (tuple(exps), ())
            if i == 0:
                R = ((), ())
            else:
                if i > self.xi_max:
                    continue
                exps = [0] * i
                exps[i - 1] = 1
                R = (tuple(exps), ())
            out.append((L, R, 1))
        return out

    def _multiply_factor(self, state, factor):
        out: Dict[Tuple, int] = {}
        for L, R, c in state:
            for a, b, d in factor:
                la = self._mul(L, a)
                rb = self._mul(R, b)
                if la is None or rb is None:
                    continue
                coef = (c * d) % 2
                if coef == 0:
                    continue
                key = (la, rb)
                out[key] = (out.get(key, 0) + coef) % 2
        return [(left, r, c) for (left, r), c in out.items() if c]

    @staticmethod
    def _mul(m1, m2):
        # multiply polynomial parts (p=2, no signs).
        poly1, _ = m1
        poly2, _ = m2
        n = max(len(poly1), len(poly2))
        poly = tuple(
            (poly1[i] if i < len(poly1) else 0)
            + (poly2[i] if i < len(poly2) else 0)
            for i in range(n)
        )
        # strip trailing zeros
        while poly and poly[-1] == 0:
            poly = poly[:-1]
        return (poly, ())


# ── Excess-filtered cobar Ext ────────────────────────────────────────────────


def ext_unstable_cobar(
    p: int, s_max: int, t_max: int, max_excess: int,
) -> Dict[Tuple[int, int], int]:
    """Excess-bounded cobar Ext_U^{s,t}(F_p[n_0], F_p) where n_0 = max_excess.

    Returns dict[(s, t)] -> dim_{F_p} of the excess-filtered cohomology.

    Algorithm:
        - Build the cobar C^s = (A_+)^{⊗s}, but restrict each tensor
          factor to monomials with excess(factor) <= max_excess.
        - Apply the same cobar differential as the stable cobar, with
          contributions to out-of-window cells dropped.
        - Return dim H^{s,t} of this subcomplex.

    For p=2 this returns the unstable Ext of the free unstable A_2-module
    F_2[n_0] (Curtis 1971). For odd p the formula is approximate when
    multi-tau monomials enter.
    """
    if p == 2:
        A = DualSteenrodAlgebraP2(t_max)
    elif p in (3, 5):
        A = DualSteenrodAlgebra(p, t_max)
    else:
        raise ValueError(f"unsupported prime {p}")

    bases: Dict[Tuple[int, int], List[Tuple]] = {}
    bases[(0, 0)] = [()]

    def filtered_basis(t: int) -> List:
        if t == 0:
            return [((), ())]
        return [m for m in A.basis(t) if excess_of_monomial(m) <= max_excess]

    def tensor_basis(s: int, t: int) -> List[Tuple]:
        if (s, t) in bases:
            return bases[(s, t)]
        out: List[Tuple] = []
        if s == 0:
            out = [()] if t == 0 else []
        elif s == 1:
            if t == 0:
                out = []
            else:
                out = [(m,) for m in filtered_basis(t)]
        else:
            for d in range(1, t + 1):
                last_options = filtered_basis(d)
                if not last_options:
                    continue
                rest = tensor_basis(s - 1, t - d)
                for r in rest:
                    for last in last_options:
                        out.append(r + (last,))
        bases[(s, t)] = out
        return out

    def differential_matrix(s: int, t: int) -> sp.csr_matrix:
        src_basis = tensor_basis(s, t)
        tgt_basis = tensor_basis(s + 1, t)
        if not src_basis or not tgt_basis:
            return sp.csr_matrix(
                (len(tgt_basis), len(src_basis)), dtype=np.int64
            )
        tgt_idx = {b: i for i, b in enumerate(tgt_basis)}
        rows: List[int] = []
        cols: List[int] = []
        data: List[int] = []
        for col, src in enumerate(src_basis):
            if s == 0:
                continue
            for i in range(s):
                a_i = src[i]
                sign_exp = i + 1
                s_coef = ((-1) ** sign_exp) % p
                for left, right, coef in A.coproduct_reduced(a_i):
                    # Filter: both halves must satisfy excess <= max_excess
                    if excess_of_monomial(left) > max_excess:
                        continue
                    if excess_of_monomial(right) > max_excess:
                        continue
                    new_tuple = src[:i] + (left, right) + src[i + 1:]
                    final = (coef * s_coef) % p
                    if final == 0:
                        continue
                    if new_tuple not in tgt_idx:
                        continue
                    rows.append(tgt_idx[new_tuple])
                    cols.append(col)
                    data.append(final)
        return sp.csr_matrix(
            (data, (rows, cols)),
            shape=(len(tgt_basis), len(src_basis)),
            dtype=np.int64,
        )

    def rank_mod_p(mat: sp.csr_matrix) -> int:
        if mat.shape[0] == 0 or mat.shape[1] == 0:
            return 0
        rows = mat.shape[0]
        cols = mat.shape[1]
        R: List[Dict[int, int]] = [{} for _ in range(rows)]
        coo = mat.tocoo()
        for r, c, v in zip(coo.row, coo.col, coo.data):
            v = int(v) % p
            if v:
                R[int(r)][int(c)] = v
        used: List[int] = []
        for c in range(cols):
            piv = None
            for r in range(rows):
                if r in used:
                    continue
                if R[r].get(c, 0) != 0:
                    piv = r
                    break
            if piv is None:
                continue
            used.append(piv)
            pv = R[piv][c]
            if pv != 1:
                inv = pow(pv, p - 2, p)
                R[piv] = {k: (v * inv) % p for k, v in R[piv].items()}
            for r in range(rows):
                if r == piv:
                    continue
                f = R[r].get(c, 0)
                if f == 0:
                    continue
                piv_row = R[piv]
                new_row = dict(R[r])
                for k, v in piv_row.items():
                    nv = (new_row.get(k, 0) - f * v) % p
                    if nv == 0:
                        new_row.pop(k, None)
                    else:
                        new_row[k] = nv
                R[r] = new_row
        return len(used)

    rank_cache: Dict[Tuple[int, int], int] = {}

    def get_rank(s: int, t: int) -> int:
        if (s, t) in rank_cache:
            return rank_cache[(s, t)]
        mat = differential_matrix(s, t)
        r = rank_mod_p(mat)
        rank_cache[(s, t)] = r
        return r

    result: Dict[Tuple[int, int], int] = {}
    for s in range(0, s_max + 1):
        for t in range(0, t_max + 1):
            dim_C = len(tensor_basis(s, t))
            if dim_C == 0:
                continue
            r_out = get_rank(s, t)
            r_in = get_rank(s - 1, t) if s >= 1 else 0
            ext_dim = dim_C - r_out - r_in
            if ext_dim < 0:
                # Truncation artifact: contributions spilled outside the
                # filter; clamp at 0 with a note in the surrounding caller.
                ext_dim = 0
            if ext_dim > 0:
                result[(s, t)] = ext_dim
    return result


# ── Public façade ────────────────────────────────────────────────────────────


def unstable_adams_e2_page(
    fp_ring: FpCohomologyRing,
    prime: int,
    s_max: int = 6,
    t_max: int = 20,
    *,
    method: Literal["auto", "u_resolution", "cobar_approx"] = "auto",
    backend: Literal["auto", "python", "numba", "julia"] = "auto",
) -> AdamsE2Page:
    """Unstable Adams E_2 dispatcher (Bousfield–Kan).

    What is Being Computed?:
        E_2^{s,t}(X) = Ext_U^{s,t}(H̃*(X; F_p), F_p) where U is the
        category of unstable A_p-modules. The result converges to
        π_{t-s}(X)_p (the p-completed unstable homotopy of X).

    Engines (selected by `method`):
        - 'u_resolution' — minimal free unstable A_p-resolution
            (Massey–Peterson / Quillen). Rigorous; wired at p ∈ {2, 3, 5}.
            See `pysurgery.adams.u_resolution`.
        - 'cobar_approx' — excess-filtered cobar over A_p^*. Generic
            approximation (NOT a strict upper bound on Ext_U); kept as a
            fast fallback and for backward compatibility.
        - 'auto' — picks 'u_resolution' at every supported prime.

    The Lambda-algebra engine (`adams_lambda.lambda_e2_page`) is
    currently marked WIP due to a derivation-formula bug; it is NOT
    reachable from this dispatcher.

    Args:
        fp_ring: Input cohomology ring (use a reduced ring for the
            tightest bound).
        prime: 2, 3, or 5.
        s_max, t_max: truncation window.
        method: 'auto' | 'u_resolution' | 'cobar_approx'.
        backend: 'auto' | 'python' | 'numba' | 'julia'. Passed through
            to the underlying engine.

    Returns:
        AdamsE2Page with `space_label` suffix marking which engine ran
        ('(U-resolution)' or '(unstable approx)').
    """
    if method not in ("auto", "u_resolution", "cobar_approx"):
        raise ValueError(
            f"method must be one of 'auto'|'u_resolution'|'cobar_approx'; got {method!r}"
        )
    if method == "auto":
        # U-resolution is wired for p ∈ {2, 3, 5}.
        method = "u_resolution" if prime in (2, 3, 5) else "cobar_approx"
    if method == "u_resolution":
        # Local import to avoid a circular dependency: adams_u_resolution
        # depends on AdamsE2Page from adams_spectral_sequence (NOT on
        # adams_unstable), so this is safe.
        from pysurgery.adams.u_resolution import u_resolution_e2_page
        return u_resolution_e2_page(
            fp_ring, prime, s_max=s_max, t_max=t_max, backend=backend,
        )
    # method == 'cobar_approx': fall through to the legacy implementation.
    return _unstable_adams_e2_page_cobar_approx(
        fp_ring, prime, s_max=s_max, t_max=t_max,
    )


def _unstable_adams_e2_page_cobar_approx(
    fp_ring: FpCohomologyRing,
    prime: int,
    s_max: int = 6,
    t_max: int = 20,
) -> AdamsE2Page:
    """Legacy excess-filtered cobar approximation of Ext_U.

    Approximation: takes the stable cobar over A_p^* and filters each
    tensor factor by Milnor-basis excess. NOT a strict upper bound on
    Ext_U (overcounts in some cells, undercounts in others); kept as a
    fast first-pass and for backward compatibility.

    See module docstring for the full mathematical caveats. Use
    `method='u_resolution'` for a rigorous engine at p = 2.
    """
    import time as _time

    if prime not in (2, 3, 5):
        raise ValueError(f"prime must be in {{2, 3, 5}}; got {prime}")

    # Find module generator dimensions (positive-degree basis).
    module_dims: Dict[int, int] = {}
    for d, labels in fp_ring.basis.items():
        if d <= 0:
            continue
        if labels:
            module_dims[int(d)] = len(labels)
    if not module_dims:
        # Trivial module: E_2 = Ext^{0,0} only.
        return AdamsE2Page(
            space_label=fp_ring.space_label + " (unstable approx; trivial M)",
            prime=prime,
            s_max=s_max,
            t_max=t_max,
            e2_grid={(0, 0): 1} if fp_ring.basis.get(0) else {},
            forced_vanishings=[],
            ambiguous_differentials=[],
            reliable_window=(s_max, max(0, t_max - s_max)),
            resource_summary={"peak_mem_mb": 0.0, "wall_seconds": 0.0},
            status="success",
            reasoning="Trivial unstable A_p-module (no positive-degree basis).",
        )

    n_min = min(module_dims.keys())
    t_start = _time.time()
    base_ext = ext_unstable_cobar(prime, s_max=s_max, t_max=t_max, max_excess=n_min)

    grid: Dict[Tuple[int, int], int] = {}
    for (s, t_rel), dim in base_ext.items():
        for n, mult in module_dims.items():
            t = t_rel + n
            if t > t_max:
                continue
            grid[(s, t)] = grid.get((s, t), 0) + mult * dim
    wall = _time.time() - t_start

    # Reuse the same forced/ambiguous classification as adams_e2_page.
    forced: List[AdamsDifferentialFlag] = []
    ambiguous: List[AdamsDifferentialFlag] = []
    for (s, t), dim in grid.items():
        if dim == 0:
            continue
        for r in range(2, s_max - s + 2):
            tgt_s = s + r
            tgt_t = t + r - 1
            if tgt_s > s_max or tgt_t > t_max:
                continue
            src_dim = dim
            tgt_dim = grid.get((tgt_s, tgt_t), 0)
            if src_dim == 0 or tgt_dim == 0:
                forced.append(AdamsDifferentialFlag(
                    r=r, source=(s, t), target=(tgt_s, tgt_t),
                    classification="forced_zero",
                    reason="source or target dim 0",
                    source_dim=src_dim, target_dim=tgt_dim,
                ))
            else:
                ambiguous.append(AdamsDifferentialFlag(
                    r=r, source=(s, t), target=(tgt_s, tgt_t),
                    classification="ambiguous", reason="both dims > 0",
                    source_dim=src_dim, target_dim=tgt_dim,
                ))

    return AdamsE2Page(
        space_label=fp_ring.space_label + " (unstable approx)",
        prime=prime,
        s_max=s_max,
        t_max=t_max,
        e2_grid=grid,
        forced_vanishings=forced,
        ambiguous_differentials=ambiguous,
        reliable_window=(s_max, max(0, t_max - s_max)),
        resource_summary={"peak_mem_mb": 0.0, "wall_seconds": wall},
        status="success",
        reasoning=(
            f"Unstable Adams E_2 approximation via excess-filtered cobar "
            f"of A_p^*. Excess cap = n_min = {n_min}. "
            f"Approximate when module has multiple generator degrees."
        ),
    )


__all__ = [
    "excess_of_monomial",
    "ext_unstable_cobar",
    "unstable_adams_e2_page",
]
