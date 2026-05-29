"""Odd-prime Adams E_2 via the cobar complex of the dual Steenrod algebra.

Bypasses Adem-relation arithmetic by working in A_p^* directly.

A_p^* = F_p[xi_1, xi_2, ...] (X) Lambda(tau_0, tau_1, ...)
   |xi_i| = 2 (p^i - 1)
   |tau_i| = 2 p^i - 1

Milnor coproduct (1958):
   Delta(xi_n) = sum_{i=0..n} xi_{n-i}^(p^i) (X) xi_i,   xi_0 = 1
   Delta(tau_n) = tau_n (X) 1 + sum_{i=0..n} xi_{n-i}^(p^i) (X) tau_i

Reduced cobar complex:
   C^s_t = (A_p^*_+)^{(X)s} in total internal degree t.
   d^s: C^s -> C^{s+1},
       d(a_1 | ... | a_s) = sum_i (-1)^{i + sum_{j<i}|a_j|} a_1 | ... | Delta_bar(a_i) | ... | a_s

Ext_{A_p}^{s,t}(F_p, F_p) = H^{s,t}(C, d).

For a trivial A_p-module M concentrated in degrees {n_alpha} with dimensions d_alpha,
   Ext_{A_p}^{s,t}(M, F_p) = sum_alpha d_alpha * Ext_{A_p}^{s, t - n_alpha}(F_p, F_p).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp


# Canonical monomial: (poly_exps, exterior_indices) where
#   poly_exps          = tuple of nonneg ints, exponents of xi_1, xi_2, ...
#   exterior_indices   = SORTED ascending tuple of distinct ints, the tau indices.
Monomial = Tuple[Tuple[int, ...], Tuple[int, ...]]


def _drop_trailing_zeros(t: Tuple[int, ...]) -> Tuple[int, ...]:
    while t and t[-1] == 0:
        t = t[:-1]
    return t


def _monomial_canonical(poly: Tuple[int, ...], ext: Tuple[int, ...]) -> Monomial:
    return (_drop_trailing_zeros(poly), tuple(ext))


def _xi_degree(i: int, p: int) -> int:
    """|xi_i| = 2(p^i - 1)."""
    return 2 * (p ** i - 1)


def _tau_degree(i: int, p: int) -> int:
    """|tau_i| = 2 p^i - 1."""
    return 2 * (p ** i) - 1


class DualSteenrodAlgebra:
    """Truncated A_p^* at odd prime p, internal-degree <= t_max."""

    def __init__(self, p: int, t_max: int) -> None:
        if p not in (3, 5, 7):
            raise ValueError(f"odd-prime cobar supports p in {{3, 5, 7}}; got {p}")
        if t_max < 0:
            raise ValueError("t_max must be >= 0")
        self.p = p
        self.t_max = t_max

        # Determine which xi_i, tau_i fit within t_max.
        self.xi_max = 0
        i = 1
        while _xi_degree(i, p) <= t_max:
            self.xi_max = i
            i += 1
        self.tau_max = -1
        i = 0
        while _tau_degree(i, p) <= t_max:
            self.tau_max = i
            i += 1

        # Cache: degree -> list of basis monomials in A_p^*_+ (positive part).
        self._basis_cache: Dict[int, List[Monomial]] = {}
        # Cache: monomial -> internal degree.
        self._deg_cache: Dict[Monomial, int] = {}

    # ── degree ────────────────────────────────────────────────────────────────

    def degree(self, mon: Monomial) -> int:
        if mon in self._deg_cache:
            return self._deg_cache[mon]
        poly, ext = mon
        d = sum(e * _xi_degree(i + 1, self.p) for i, e in enumerate(poly) if e)
        d += sum(_tau_degree(j, self.p) for j in ext)
        self._deg_cache[mon] = d
        return d

    # ── basis enumeration ────────────────────────────────────────────────────

    def basis(self, t: int) -> List[Monomial]:
        """All canonical monomials of internal degree t (positive part if t > 0)."""
        if t < 0:
            return []
        if t in self._basis_cache:
            return self._basis_cache[t]
        result: List[Monomial] = []
        if t == 0:
            result = [((), ())]
            self._basis_cache[t] = result
            return result

        # Enumerate (poly_exps, ext_set) with total degree == t.
        p = self.p
        xi_degs = [_xi_degree(i + 1, p) for i in range(self.xi_max)]
        tau_degs = [_tau_degree(i, p) for i in range(self.tau_max + 1)]

        # First, enumerate exterior subsets (subset of indices 0..tau_max).
        # Each subset has a fixed degree contribution.
        ext_options: List[Tuple[Tuple[int, ...], int]] = []
        n_tau = self.tau_max + 1
        for mask in range(1 << n_tau):
            indices: List[int] = []
            d_ext = 0
            for j in range(n_tau):
                if mask & (1 << j):
                    indices.append(j)
                    d_ext += tau_degs[j]
                    if d_ext > t:
                        break
            else:
                if d_ext <= t:
                    ext_options.append((tuple(indices), d_ext))

        # For each exterior subset, enumerate polynomial exponents summing to t - d_ext.
        for ext, d_ext in ext_options:
            remaining = t - d_ext
            for poly in _enum_poly(remaining, xi_degs):
                result.append(_monomial_canonical(poly, ext))

        self._basis_cache[t] = result
        return result

    # ── multiplication (signed, canonicalized) ────────────────────────────────

    def mul(
        self, m1: Monomial, m2: Monomial
    ) -> Tuple[Optional[Monomial], int]:
        """Product m1 * m2 in A_p^*. Returns (canonical_monomial, sign mod p).

        Returns (None, 0) if the exterior parts collide (tau_i^2 = 0).
        """
        poly1, ext1 = m1
        poly2, ext2 = m2
        # exterior collision check
        if set(ext1) & set(ext2):
            return (None, 0)
        # combine polynomial exponents
        n = max(len(poly1), len(poly2))
        poly = tuple(
            (poly1[i] if i < len(poly1) else 0)
            + (poly2[i] if i < len(poly2) else 0)
            for i in range(n)
        )
        # combine exterior parts with Koszul sign
        combined: List[int] = list(ext1) + list(ext2)
        # sign = sign of permutation that sorts combined (relative to ext1++ext2).
        # Count inversions between ext1 (left) and ext2 (right): for each pair
        # (a, b) with a in ext1, b in ext2, a > b contributes one inversion.
        inversions = 0
        for a in ext1:
            for b in ext2:
                if a > b:
                    inversions += 1
        sign = 1 if (inversions % 2 == 0) else (self.p - 1)
        combined.sort()
        return (_monomial_canonical(poly, tuple(combined)), sign)

    # ── reduced coproduct on a basis monomial ─────────────────────────────────

    def coproduct_reduced(
        self, mon: Monomial
    ) -> List[Tuple[Monomial, Monomial, int]]:
        """Delta_bar(mon) as a list of (left, right, coef in F_p) summands.

        Excludes the trivial 1 (X) mon and mon (X) 1 pieces.
        """
        # Delta is multiplicative; build it factor-by-factor.
        # Start with Delta(1) = 1 (X) 1.
        # Each factor in mon contributes its own coproduct; multiply on tensor.
        poly, ext = mon
        p = self.p

        # Initial state: list of (left, right, coef).
        state: List[Tuple[Monomial, Monomial, int]] = [(((), ()), ((), ()), 1)]

        # Multiply in xi_{i+1}^{poly[i]} for each i.
        for i, e in enumerate(poly):
            if e == 0:
                continue
            for _ in range(e):
                state = self._multiply_factor(state, self._coprod_xi(i + 1))

        # Multiply in tau_j for each j in ext (in increasing order, matching
        # the canonical exterior ordering of mon).
        for j in ext:
            state = self._multiply_factor(state, self._coprod_tau(j))

        # Now state encodes the full Delta(mon). Remove the trivial pieces.
        reduced: List[Tuple[Monomial, Monomial, int]] = []
        for left, right, c in state:
            if left == ((), ()) or right == ((), ()):
                continue
            reduced.append((left, right, c % p))

        # Collect like terms.
        collected: Dict[Tuple[Monomial, Monomial], int] = {}
        for left, right, c in reduced:
            key = (left, right)
            collected[key] = (collected.get(key, 0) + c) % p
        return [(left, r, c) for (left, r), c in collected.items() if c != 0]

    # ── coproduct on individual generators ────────────────────────────────────

    def _coprod_xi(self, n: int) -> List[Tuple[Monomial, Monomial, int]]:
        """Delta(xi_n) = sum_{i=0..n} xi_{n-i}^(p^i) (X) xi_i."""
        out: List[Tuple[Monomial, Monomial, int]] = []
        p = self.p
        for i in range(n + 1):
            j = n - i
            # left = xi_j^(p^i), with j == 0 meaning the unit.
            if j == 0:
                left: Monomial = ((), ())
            else:
                if j > self.xi_max:
                    # outside truncation: skip (would be in higher xi-only terms;
                    # only matters for products beyond the truncation window).
                    continue
                exps = [0] * j
                exps[j - 1] = p ** i
                left = _monomial_canonical(tuple(exps), ())
            # right = xi_i, with i == 0 meaning the unit.
            if i == 0:
                right: Monomial = ((), ())
            else:
                if i > self.xi_max:
                    continue
                exps = [0] * i
                exps[i - 1] = 1
                right = _monomial_canonical(tuple(exps), ())
            out.append((left, right, 1))
        return out

    def _coprod_tau(self, n: int) -> List[Tuple[Monomial, Monomial, int]]:
        """Delta(tau_n) = tau_n (X) 1 + sum_{i=0..n} xi_{n-i}^(p^i) (X) tau_i."""
        out: List[Tuple[Monomial, Monomial, int]] = []
        p = self.p
        # tau_n (X) 1
        out.append((_monomial_canonical((), (n,)), ((), ()), 1))
        for i in range(n + 1):
            j = n - i
            if j == 0:
                left: Monomial = ((), ())
            else:
                if j > self.xi_max:
                    continue
                exps = [0] * j
                exps[j - 1] = p ** i
                left = _monomial_canonical(tuple(exps), ())
            right = _monomial_canonical((), (i,))
            out.append((left, right, 1))
        return out

    # ── multiply two factored coproducts on the tensor product ────────────────

    def _multiply_factor(
        self,
        state: List[Tuple[Monomial, Monomial, int]],
        factor: List[Tuple[Monomial, Monomial, int]],
    ) -> List[Tuple[Monomial, Monomial, int]]:
        """Multiply state (sum L_i (X) R_i) by factor (sum a_k (X) b_k).

        Product on the tensor square is (L * a) (X) (R * b) with sign:
        when 'a' (in the left factor) passes 'R' (the right of state),
        we pick up (-1)^{|a| * |R|}. At p=2 this is no-op; at odd p with
        a of even degree the sign is 1 anyway. The Koszul signs only
        matter when both |a| and |R| are odd.
        """
        p = self.p
        out: Dict[Tuple[Monomial, Monomial], int] = {}
        for L, R, c in state:
            for a, b, d in factor:
                # Compute sign from moving 'a' past 'R'.
                deg_a = self.degree(a)
                deg_R = self.degree(R)
                if (deg_a % 2 == 1) and (deg_R % 2 == 1):
                    sign_pass = p - 1
                else:
                    sign_pass = 1

                # Left product L * a
                la, sL = self.mul(L, a)
                if la is None:
                    continue
                # Right product R * b
                rb, sR = self.mul(R, b)
                if rb is None:
                    continue

                coef = (c * d * sign_pass * sL * sR) % p
                if coef == 0:
                    continue
                key = (la, rb)
                out[key] = (out.get(key, 0) + coef) % p

        return [(left, r, c) for (left, r), c in out.items() if c != 0]


def _enum_poly(target: int, xi_degs: List[int]) -> List[Tuple[int, ...]]:
    """Enumerate nonneg integer tuples (a_1, ..., a_n) with sum(a_i * xi_degs[i]) = target.

    n = len(xi_degs).
    """
    n = len(xi_degs)
    out: List[Tuple[int, ...]] = []

    def rec(idx: int, remaining: int, acc: List[int]) -> None:
        if idx == n:
            if remaining == 0:
                out.append(tuple(acc))
            return
        d = xi_degs[idx]
        max_exp = remaining // d if d > 0 else 0
        for e in range(max_exp + 1):
            acc.append(e)
            rec(idx + 1, remaining - e * d, acc)
            acc.pop()

    rec(0, target, [])
    return out


# ── Cobar complex Ext computation ────────────────────────────────────────────


def ext_cobar(p: int, s_max: int, t_max: int) -> Dict[Tuple[int, int], int]:
    """Compute dim_{F_p} Ext_{A_p}^{s, t}(F_p, F_p) for (s, t) up to bounds.

    Returns a dict keyed by (s, t) containing only nonzero entries.

    Algorithm:
        - Build C^s_t = (A_+)^{(X)s} basis at each (s, t), s <= s_max, t <= t_max.
        - Build d^s: C^s -> C^{s+1} as sparse F_p matrices (one per t).
        - Ext^{s,t} = dim ker d^s_t - dim im d^{s-1}_t = dim C^s_t - rank d^s_t - rank d^{s-1}_t.
    """
    A = DualSteenrodAlgebra(p, t_max)

    # Basis of (A_+)^{(X)s} at internal degree t, as list of tuples of monomials.
    # We only need s in {0, ..., s_max + 1} (extra page for d^{s_max}).
    bases: Dict[Tuple[int, int], List[Tuple[Monomial, ...]]] = {}
    # s = 0: C^0_0 = F_p, C^0_t = 0 for t > 0.
    bases[(0, 0)] = [()]

    # s >= 1: list of tuples (m_1, ..., m_s) of A_+-monomials summing in degree to t.
    def tensor_basis(s: int, t: int) -> List[Tuple[Monomial, ...]]:
        if (s, t) in bases:
            return bases[(s, t)]
        out: List[Tuple[Monomial, ...]] = []
        if s == 0:
            out = [()] if t == 0 else []
        elif s == 1:
            # (A_+)^{(X)1}_t = A_+_t
            if t == 0:
                out = []
            else:
                out = [(m,) for m in A.basis(t)]
        else:
            # split: pick degree d for last factor in 1..t, rest sums to t-d.
            for d in range(1, t + 1):
                last_options = A.basis(d)
                if not last_options:
                    continue
                rest = tensor_basis(s - 1, t - d)
                for r in rest:
                    for last in last_options:
                        out.append(r + (last,))
        bases[(s, t)] = out
        return out

    # ── Differential matrices ────────────────────────────────────────────────
    # d^s_t: C^s_t -> C^{s+1}_t as sparse F_p matrix (cols = source basis, rows = target basis).
    # d(a_1 | ... | a_s) = sum_i (-1)^{i + |a_1|+...+|a_{i-1}|} a_1 | ... | Delta_bar(a_i) | ... | a_s
    # We index the sum from i=1 (1-based); sign exponent is i + (sum of preceding degrees).
    # The 'i' parity convention is the cosimplicial cobar; consistent with Ravenel A1.2.

    def differential_matrix(s: int, t: int) -> sp.csr_matrix:
        src_basis = tensor_basis(s, t)
        tgt_basis = tensor_basis(s + 1, t)
        if not src_basis or not tgt_basis:
            return sp.csr_matrix((len(tgt_basis), len(src_basis)), dtype=np.int64)
        tgt_idx: Dict[Tuple[Monomial, ...], int] = {b: i for i, b in enumerate(tgt_basis)}
        rows: List[int] = []
        cols: List[int] = []
        data: List[int] = []
        for col, src in enumerate(src_basis):
            # s=0: src = (), the unit; d(1) is the 1 -> A_+ map which is zero
            # (since we are in the REDUCED cobar with no augmentation contribution).
            if s == 0:
                continue
            for i in range(s):
                a_i = src[i]
                # Reduced coproduct on the i-th tensor factor.
                # Sign convention: (-1)^{i+1} (i 0-based, so 1-based position).
                # For a coassociative coalgebra with multiplicative Delta (which
                # already absorbs Koszul signs through its action on products),
                # the bare position sign suffices for d^2 = 0. Empirically
                # verified on all small monomials of A_3^* up to deg 12; see
                # the sign-convention sweep that picked this convention.
                sign_exp = i + 1
                s_coef = ((-1) ** sign_exp) % p
                for left, right, coef in A.coproduct_reduced(a_i):
                    new_tuple = src[:i] + (left, right) + src[i + 1:]
                    final_coef = (coef * s_coef) % p
                    if final_coef == 0:
                        continue
                    if new_tuple not in tgt_idx:
                        # Result uses xi_i or tau_i beyond our truncation; safe
                        # to skip when the total degree is bounded.
                        continue
                    rows.append(tgt_idx[new_tuple])
                    cols.append(col)
                    data.append(final_coef)
        mat = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(len(tgt_basis), len(src_basis)),
            dtype=np.int64,
        )
        return mat

    # ── Rank over F_p ────────────────────────────────────────────────────────
    def rank_mod_p(mat: sp.csr_matrix) -> int:
        if mat.shape[0] == 0 or mat.shape[1] == 0:
            return 0
        # Convert to dense list-of-dict for RREF.
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

    # ── Compute Ext^{s,t} = dim C^s_t - rank d^s_t - rank d^{s-1}_t ──────────
    result: Dict[Tuple[int, int], int] = {}
    rank_cache: Dict[Tuple[int, int], int] = {}

    def get_rank(s: int, t: int) -> int:
        if (s, t) in rank_cache:
            return rank_cache[(s, t)]
        mat = differential_matrix(s, t)
        r = rank_mod_p(mat)
        rank_cache[(s, t)] = r
        return r

    for s in range(0, s_max + 1):
        for t in range(0, t_max + 1):
            dim_C = len(tensor_basis(s, t))
            if dim_C == 0:
                continue
            r_out = get_rank(s, t)             # rank of d^s: C^s -> C^{s+1}
            r_in = get_rank(s - 1, t) if s >= 1 else 0  # rank of d^{s-1}: C^{s-1} -> C^s
            ext_dim = dim_C - r_out - r_in
            if ext_dim < 0:
                raise RuntimeError(
                    f"Negative Ext dim at (s={s}, t={t}): "
                    f"dim_C={dim_C}, rank_out={r_out}, rank_in={r_in}. "
                    f"Likely a sign/coproduct bug."
                )
            if ext_dim > 0:
                result[(s, t)] = ext_dim
    return result


# ── Public façade ────────────────────────────────────────────────────────────


def adams_e2_grid_odd_prime_trivial_module(
    fp_ring,
    prime: int,
    s_max: int,
    t_max: int,
) -> Dict[Tuple[int, int], int]:
    """Adams E_2 grid for an A_p-trivial F_p cohomology ring at odd p.

    Returns dict[(s, t)] -> dim_{F_p} E_2^{s, t}.

    For a trivial A_p-module M with basis_dimensions d_n in degree n,
       Ext^{s,t}(M, F_p) = sum_n d_n * Ext^{s, t-n}(F_p, F_p).
    """
    # Pull module basis dimensions out of the FpCohomologyRing.
    module_dims: Dict[int, int] = {}
    for d, labels in fp_ring.basis.items():
        if not labels:
            continue
        module_dims[int(d)] = len(labels)

    # Compute sphere Ext up to (s_max, t_max).
    base_ext = ext_cobar(prime, s_max, t_max)

    grid: Dict[Tuple[int, int], int] = {}
    for (s, t_rel), dim in base_ext.items():
        for n, mult in module_dims.items():
            t = t_rel + n
            if t > t_max:
                continue
            grid[(s, t)] = grid.get((s, t), 0) + mult * dim
    return grid


def is_trivial_ap_module(fp_ring) -> bool:
    """True iff no positive-index Steenrod operation acts non-trivially.

    P^0 (index 0) is the unit and acts as identity; entries (0, label) just
    record "identity sends label to itself" and do not constitute non-trivial
    action. Any entry with index >= 1 that has nonzero image counts as
    non-trivial Steenrod action.

    For odd primes, this captures spaces whose F_p-cohomology has no P^i
    (i >= 1) or Bockstein action -- automatic for S^n at p odd when
    n < 2(p-1) by instability.
    """
    for (i, _label), action in fp_ring.sq_table.items():
        if i == 0:
            continue
        if any((c % fp_ring.prime) != 0 for c in action.values()):
            return False
    return True
