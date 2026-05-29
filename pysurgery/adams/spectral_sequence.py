"""Adams spectral sequence E_2 page computation.

Overview:
    Implements the mod-p Adams spectral sequence E_2 page,
    E_2^{s,t} = Ext_{A_p}^{s,t}(H^*(X; F_p), F_p), via a minimal free
    resolution over the Steenrod algebra. Forced-zero and ambiguous
    differentials d_r are classified (not computed).

Layer architecture (see RFC-adams-v2):
    Layer 1 — SteenrodAlgebra (Adem rewriter, admissible basis).
    Layer 2 — FpCohomologyRing + SteenrodAction (Cartan/instability).
    Layer 3 — ExtComputer (minimal free A-resolution, sparse F_p RREF).
    Layer 4 — adams_e2_page() entry point + AdamsE2Page contract.

Resource Guarantees:
    - All linear-algebra storage is scipy.sparse (no dense materialisation).
    - Polynomial degree truncation: t_max ≤ 50 (T_HARD).
    - Memory cap: tracemalloc peak ≤ 0.8 · available heap → graceful truncate.
    - Admissible-basis cap: 5_000 per internal degree → AdamsCombinatorialError.
    - Prime restriction: p ∈ {2, 3, 5}.

References:
    Milnor, J. (1958). The Steenrod algebra and its dual. Ann. Math. 67, 150-171.
    Adams, J. F. (1958). On the structure and applications of the Steenrod algebra.
        Comment. Math. Helv. 32, 180-214.
    Bruner, R. R. (1993). Ext in the nineties. Contemp. Math. 146, AMS.
    May, J. P. (1981). The work of J. F. Adams. Bull. AMS 7(1).
    Wang, J. S. P. (1967). On the cohomology of the mod-2 Steenrod algebra.
        Illinois J. Math. 11, 480-490.
"""
from __future__ import annotations

import time
import tracemalloc
import warnings
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pysurgery.core.foundations import CONTRACT_VERSION
from pysurgery.core.theorem_tags import ADAMS_E2_EXT_STEENROD


# ── Hard caps and constants ────────────────────────────────────────────────

ADM_HARD_CAP: int = 5_000
T_HARD: int = 50
S_HARD: int = 20
DEFAULT_MEM_CAP_MB: int = 4096


# ── Type aliases ────────────────────────────────────────────────────────────

AdmissibleSequence = Tuple[int, ...]
SteenrodElement = Dict[AdmissibleSequence, int]


# ── Exceptions ──────────────────────────────────────────────────────────────


class AdamsResourceError(Exception):
    """Raised when an Adams computation exceeds resource caps."""


class AdamsCombinatorialError(Exception):
    """Raised when admissible-basis enumeration exceeds the hard cap."""


# ════════════════════════════════════════════════════════════════════════════
# Layer 0 — arithmetic helpers
# ════════════════════════════════════════════════════════════════════════════


def _binom_mod_p(n: int, k: int, p: int) -> int:
    """Binomial(n, k) mod p via Lucas's theorem.

    Returns 0 if k < 0 or n < 0 or k > n. For p == 2 uses the bitwise
    identity C(n,k) mod 2 = 1 iff (k & ~n) == 0.
    """
    if k < 0 or n < 0 or k > n:
        return 0
    if p == 2:
        return 1 if (k & ~n) == 0 else 0
    result = 1
    while n > 0 or k > 0:
        a = n % p
        b = k % p
        if b > a:
            return 0
        num = 1
        den = 1
        for i in range(b):
            num = (num * (a - i)) % p
            den = (den * (i + 1)) % p
        result = (result * num * pow(den, p - 2, p)) % p
        n //= p
        k //= p
    return result


def _vec_add_mod_p(a: Dict[int, int], b: Dict[int, int], p: int) -> Dict[int, int]:
    """In-place-style add of sparse F_p vectors keyed by index."""
    out = dict(a)
    for k, v in b.items():
        nv = (out.get(k, 0) + v) % p
        if nv == 0:
            out.pop(k, None)
        else:
            out[k] = nv
    return out


def _vec_scale_mod_p(a: Dict[int, int], c: int, p: int) -> Dict[int, int]:
    c = c % p
    if c == 0:
        return {}
    return {k: (v * c) % p for k, v in a.items()}


def _sparse_fp_kernel(mat: sp.csr_matrix, p: int) -> List[Dict[int, int]]:
    """Compute a basis of the null space of `mat` over F_p.

    Algorithm:
        Pivoted Gauss elimination on a sparse copy of `mat`. Pivots on the
        smallest column with a nonzero pivot row. Returns the kernel basis
        as a list of sparse F_p-vectors (Dict[col_idx, coef]).

    Storage:
        Matrix is converted to LIL for in-place row updates, then kernel
        vectors are returned as sparse dicts. No dense materialisation.

    Args:
        mat: rows × cols sparse matrix, entries reduced mod p.
        p: prime characteristic.

    Returns:
        List of length (cols - rank) of sparse F_p vectors (one per kernel
        basis element). Each vector is a Dict[col_idx, coef in F_p].
    """
    if mat.shape[0] == 0 or mat.shape[1] == 0:
        # Empty domain or empty matrix → kernel is everything (or nothing)
        if mat.shape[1] == 0:
            return []
        # zero rows → kernel is full
        return [{j: 1} for j in range(mat.shape[1])]
    rows, cols = mat.shape
    # Build a list-of-dict representation for fast row ops
    R: List[Dict[int, int]] = []
    coo = mat.tocoo()
    for _ in range(rows):
        R.append({})
    for r, c, v in zip(coo.row, coo.col, coo.data):
        v = int(v) % p
        if v != 0:
            R[int(r)][int(c)] = v
    # RREF
    pivot_col_to_row: Dict[int, int] = {}
    used_rows: List[int] = []
    for c in range(cols):
        # find a row not yet used with a nonzero in column c
        pivot_r = None
        for r in range(rows):
            if r in used_rows:
                continue
            if R[r].get(c, 0) != 0:
                pivot_r = r
                break
        if pivot_r is None:
            continue
        # normalize the pivot row (multiply by inverse of pivot value)
        pv = R[pivot_r][c]
        if pv != 1:
            inv = pow(pv, p - 2, p)
            R[pivot_r] = {k: (v * inv) % p for k, v in R[pivot_r].items()}
        # eliminate from all other rows
        for r in range(rows):
            if r == pivot_r:
                continue
            f = R[r].get(c, 0)
            if f == 0:
                continue
            # row[r] -= f · row[pivot_r]
            piv_row = R[pivot_r]
            new_row = dict(R[r])
            for k, v in piv_row.items():
                nv = (new_row.get(k, 0) - f * v) % p
                if nv == 0:
                    new_row.pop(k, None)
                else:
                    new_row[k] = nv
            R[r] = new_row
        pivot_col_to_row[c] = pivot_r
        used_rows.append(pivot_r)
    # free columns are those not in pivot_col_to_row
    pivot_cols = set(pivot_col_to_row.keys())
    free_cols = [c for c in range(cols) if c not in pivot_cols]
    kernel: List[Dict[int, int]] = []
    for fc in free_cols:
        # set free var fc = 1; for each pivot col pc with row pr, x_pc = -R[pr][fc]
        vec: Dict[int, int] = {fc: 1}
        for pc, pr in pivot_col_to_row.items():
            coef = R[pr].get(fc, 0)
            if coef != 0:
                vec[pc] = (-coef) % p
        kernel.append(vec)
    return kernel


# ════════════════════════════════════════════════════════════════════════════
# Layer 1 — Steenrod algebra A_p
# ════════════════════════════════════════════════════════════════════════════


class SteenrodAlgebra:
    """Mod-p Steenrod algebra (p ∈ {2, 3, 5}).

    Overview:
        Sparse representation of A_p as an F_p-vector space, with a
        memoised Adem-rewriter that reduces arbitrary monomials in
        Sq^i (p=2) or P^i / β (p odd) to admissible monomials.

    Key Concepts:
        - Admissible monomial (mod 2): tuple (i_1, …, i_k) with
          i_j ≥ 2 i_{j+1}, i_k ≥ 1.  () = Sq^0 = 1.
        - Admissible monomial (mod p odd): (e_0, s_1, e_1, …, s_k, e_k) with
          e_i ∈ {0, 1}, s_i ≥ p · s_{i+1} + e_i, s_k ≥ 1.
        - Element representation: ``Dict[AdmissibleSequence, int]`` —
          admissible-monomial → coefficient in F_p (sparse).

    Coefficient Ring:
        F_p, with p ∈ {2, 3, 5}.

    Attributes:
        prime (int): characteristic p ∈ {2, 3, 5}.
        max_t (int): truncation degree for cached basis enumeration.

    References:
        Milnor, J. (1958). The Steenrod algebra and its dual.
            Ann. Math. 67, 150-171.
        Steenrod, N. & Epstein, D. (1962). Cohomology Operations.
            Princeton Univ. Press.
    """

    def __init__(self, prime: int = 2, max_t: int = 30):
        if prime not in (2, 3, 5):
            raise ValueError(
                f"Steenrod algebra only supported for prime ∈ {{2, 3, 5}}; got {prime}."
            )
        if max_t < 0:
            raise ValueError("max_t must be ≥ 0.")
        if max_t > T_HARD:
            warnings.warn(
                f"SteenrodAlgebra max_t={max_t} exceeds hard cap {T_HARD}; clamping.",
                stacklevel=2,
            )
            max_t = T_HARD
        self.prime = prime
        self.max_t = max_t
        self._adm_basis_cache: Dict[int, List[AdmissibleSequence]] = {}
        self._pair_cache: Dict[Tuple, SteenrodElement] = {}

    # ── Constructors ───────────────────────────────────────────────────────

    def one(self) -> SteenrodElement:
        """The identity element Sq^0 = 1."""
        return {(): 1}

    def zero(self) -> SteenrodElement:
        return {}

    def Sq(self, i: int) -> SteenrodElement:
        """The single Steenrod square Sq^i (mod 2 only)."""
        if self.prime != 2:
            raise ValueError("Sq^i is mod-2 only; use P(i) or beta() for odd p.")
        if i < 0:
            raise ValueError("Sq^i requires i ≥ 0.")
        if i == 0:
            return self.one()
        return {(i,): 1}

    def P(self, i: int) -> SteenrodElement:
        """The Steenrod power P^i (mod p, p odd)."""
        if self.prime == 2:
            raise ValueError("P^i is for odd primes; use Sq(i) for p=2.")
        if i < 0:
            raise ValueError("P^i requires i ≥ 0.")
        if i == 0:
            return self.one()
        # Encoding for odd p admissible: (e_0, s_1, e_1, …, s_k, e_k)
        # P^i corresponds to (0, i, 0).
        return {(0, i, 0): 1}

    def beta(self) -> SteenrodElement:
        """The Bockstein β (mod p, p odd)."""
        if self.prime == 2:
            raise ValueError("β is for odd primes only.")
        return {(1,): 1}

    # ── Algebra operations ─────────────────────────────────────────────────

    def add(self, a: SteenrodElement, b: SteenrodElement) -> SteenrodElement:
        out = dict(a)
        for k, v in b.items():
            nv = (out.get(k, 0) + v) % self.prime
            if nv == 0:
                out.pop(k, None)
            else:
                out[k] = nv
        return out

    def is_zero(self, a: SteenrodElement) -> bool:
        return all((v % self.prime) == 0 for v in a.values())

    def degree_of(self, seq: AdmissibleSequence) -> int:
        """Internal degree of an admissible monomial."""
        if self.prime == 2:
            return sum(seq)
        # Odd: (e_0, s_1, e_1, ...) → e_0 + Σ (2(p-1) s_i + e_i)
        if len(seq) == 0:
            return 0
        if len(seq) == 1:
            return int(seq[0])
        p = self.prime
        d = int(seq[0])  # e_0
        # subsequent come in (s, e) pairs
        i = 1
        while i + 1 < len(seq) + 1:
            if i >= len(seq):
                break
            s = int(seq[i])
            e = int(seq[i + 1]) if i + 1 < len(seq) else 0
            d += 2 * (p - 1) * s + e
            i += 2
        return d

    # ── Admissible basis enumeration ───────────────────────────────────────

    def admissible_basis(self, t: int) -> List[AdmissibleSequence]:
        """All admissible monomials of internal degree t (cached).

        Raises AdamsCombinatorialError if size exceeds ADM_HARD_CAP.
        """
        if t < 0:
            return []
        if t in self._adm_basis_cache:
            return self._adm_basis_cache[t]
        if self.prime == 2:
            result = self._admissible_basis_mod2(t)
        else:
            result = self._admissible_basis_odd(t)
        if len(result) > ADM_HARD_CAP:
            raise AdamsCombinatorialError(
                f"|admissible_basis({t})| = {len(result)} > {ADM_HARD_CAP}; "
                "lower t_max or change prime."
            )
        self._adm_basis_cache[t] = result
        return result

    def _admissible_basis_mod2(self, t: int) -> List[AdmissibleSequence]:
        if t == 0:
            return [()]

        results: List[AdmissibleSequence] = []

        def gen(remaining: int, max_first: int):
            if remaining == 0:
                yield ()
                return
            for i1 in range(1, min(remaining, max_first) + 1):
                for tail in gen(remaining - i1, i1 // 2):
                    yield (i1,) + tail

        for seq in gen(t, t):
            results.append(seq)
        return results

    def _admissible_basis_odd(self, t: int) -> List[AdmissibleSequence]:
        p = self.prime
        results: List[AdmissibleSequence] = []
        # gen yields tail (s_1, e_1, s_2, e_2, …, s_k, e_k) summing to remaining
        # under the constraint s_1 ≤ max_s.
        def gen(remaining: int, max_s: int):
            if remaining < 0:
                return
            if remaining == 0:
                yield ()
                return
            for s in range(1, max_s + 1):
                cost = 2 * (p - 1) * s
                if cost > remaining:
                    break
                # e_1 = 0
                rest0 = remaining - cost
                next_max_0 = s // p  # s ≥ p s' + 0 → s' ≤ s/p
                if rest0 == 0:
                    yield (s, 0)
                else:
                    for tail in gen(rest0, next_max_0):
                        yield (s, 0) + tail
                # e_1 = 1
                rest1 = remaining - cost - 1
                if rest1 < 0:
                    continue
                next_max_1 = (s - 1) // p  # s ≥ p s' + 1 → s' ≤ (s-1)/p
                if rest1 == 0:
                    yield (s, 1)
                else:
                    for tail in gen(rest1, next_max_1):
                        yield (s, 1) + tail

        for e0 in (0, 1):
            if e0 > t:
                continue
            r = t - e0
            if r == 0:
                if e0 == 0:
                    results.append((0,))  # identity element
                else:
                    results.append((1,))  # just β
                continue
            # full body
            for body in gen(r, r):
                results.append((e0,) + body)
        return results

    def is_admissible(self, seq: AdmissibleSequence) -> bool:
        if self.prime == 2:
            if len(seq) == 0:
                return True
            if seq[-1] < 1:
                return False
            return all(seq[i] >= 2 * seq[i + 1] for i in range(len(seq) - 1))
        # odd: (e_0, s_1, e_1, ..., s_k, e_k)
        if len(seq) == 0:
            return True
        if len(seq) == 1:
            return seq[0] in (0, 1)
        if seq[0] not in (0, 1):
            return False
        if (len(seq) - 1) % 2 != 0:
            return False
        p = self.prime
        # pairs (s_i, e_i) at indices 1, 2, 3, 4, …
        for i in range(1, len(seq) - 2, 2):
            s_curr = seq[i]
            e_curr = seq[i + 1]
            s_next = seq[i + 2]
            if e_curr not in (0, 1):
                return False
            if s_curr < p * s_next + e_curr:
                return False
        # last s_k ≥ 1
        if seq[-2] < 1:
            return False
        if seq[-1] not in (0, 1):
            return False
        return True

    # ── Adem rewriter ──────────────────────────────────────────────────────

    def to_admissible(self, raw_sequence) -> SteenrodElement:
        """Reduce an arbitrary monomial to a sum of admissible monomials.

        Args:
            raw_sequence: Tuple/list/tuple representing a product of
                Sq^i (mod 2) or (e_0, s_1, e_1, …) (mod p odd).

        Returns:
            Sparse F_p-linear combination of admissible monomials.
        """
        if self.prime == 2:
            return self._to_admissible_mod2(tuple(raw_sequence))
        return self._to_admissible_odd(tuple(raw_sequence))

    def _to_admissible_mod2(self, seq: AdmissibleSequence) -> SteenrodElement:
        # remove Sq^0 = 1 entries (zero exponents)
        seq = tuple(i for i in seq if i != 0)
        if not seq:
            return {(): 1}
        if any(i < 0 for i in seq):
            raise ValueError(f"Negative exponent in {seq}.")
        if self.is_admissible(seq):
            return {seq: 1}
        # find leftmost violation
        for j in range(len(seq) - 1):
            if seq[j] < 2 * seq[j + 1]:
                a, b = seq[j], seq[j + 1]
                replacement = self._adem_pair_mod2(a, b)
                result: SteenrodElement = {}
                for new_pair, coef in replacement.items():
                    new_seq = seq[:j] + new_pair + seq[j + 2 :]
                    sub = self._to_admissible_mod2(new_seq)
                    for s, c in sub.items():
                        nv = (result.get(s, 0) + coef * c) % 2
                        if nv == 0:
                            result.pop(s, None)
                        else:
                            result[s] = nv
                return result
        return {seq: 1}

    def _adem_pair_mod2(self, a: int, b: int) -> SteenrodElement:
        """Adem expansion of Sq^a Sq^b (mod 2). Memoised.

        For 0 < a < 2b:
            Sq^a Sq^b = Σ_{c=0..⌊a/2⌋} C(b-c-1, a-2c)·Sq^{a+b-c} Sq^c.
        For a ≥ 2b, the pair is admissible.
        For a == 0, Sq^0 Sq^b = Sq^b.
        """
        key = ("p2", a, b)
        if key in self._pair_cache:
            return dict(self._pair_cache[key])
        if a == 0:
            res = {(b,): 1} if b > 0 else {(): 1}
            self._pair_cache[key] = res
            return dict(res)
        if b == 0:
            res = {(a,): 1}
            self._pair_cache[key] = res
            return dict(res)
        if a >= 2 * b:
            res = {(a, b): 1}
            self._pair_cache[key] = res
            return dict(res)
        result: SteenrodElement = {}
        for c in range(0, a // 2 + 1):
            coef = _binom_mod_p(b - c - 1, a - 2 * c, 2)
            if coef == 0:
                continue
            if c == 0:
                seq = (a + b,)
            else:
                seq = (a + b - c, c)
            result[seq] = (result.get(seq, 0) + coef) % 2
        result = {k: v for k, v in result.items() if v != 0}
        self._pair_cache[key] = dict(result)
        return result

    # ── Odd-prime helpers: ops-list representation ─────────────────────────
    #
    # For odd primes we use an internal "ops list" form: a list whose entries
    # are either ("β",) or ("P", k) (with k >= 1). The encoded admissible
    # tuple form (e_0, s_1, e_1, s_2, e_2, …, s_k, e_k) maps to the unique
    # ops list with β^0 entries dropped and intervening β's recorded.

    @staticmethod
    def _seq_to_ops_odd(seq: AdmissibleSequence) -> List[Tuple]:
        """Convert an encoded odd-prime tuple to an ops list.

        Assumes the canonical interleaved form: positions 0, 2, 4, ... are
        β-bits (0 or 1) and positions 1, 3, 5, ... are P-exponents (≥ 0).
        """
        if not seq:
            return []
        ops: List[Tuple] = []
        # Position 0: β-bit
        if seq[0] == 1:
            ops.append(("β",))
        elif seq[0] not in (0, 1):
            # treat any nonzero β-bit ≥ 2 as zero (handled by caller); we still
            # decompose to keep the structure usable.
            ops.append(("β",))
            ops.append(("β",))
        # positions 1, 2, 3, 4, ... → P, β, P, β, ...
        for i in range(1, len(seq)):
            v = int(seq[i])
            if i % 2 == 1:
                # P-exponent slot
                if v != 0:
                    ops.append(("P", v))
            else:
                # β-bit slot
                if v == 1:
                    ops.append(("β",))
                elif v >= 2:
                    # β^k for k >= 2 is zero; record two β's and the caller will detect
                    ops.append(("β",))
                    ops.append(("β",))
        return ops

    @staticmethod
    def _ops_to_seq_odd(ops: List[Tuple]) -> Optional[AdmissibleSequence]:
        """Convert an ops list to canonical encoded tuple form.

        Returns None if the ops list is zero (contains β·β with nothing in between).
        Drops adjacent β's by zero-detection at call sites; assumes caller has
        already verified β^2 has been collapsed.
        """
        # Pattern: (β?, P, β?, P, β?, ...). Walk through, collecting groups.
        # Output: (e_0, s_1, e_1, s_2, e_2, ..., s_k, e_k).
        n = len(ops)
        if n == 0:
            return ()
        out: List[int] = []
        # e_0: leading β?
        i = 0
        if ops[0] == ("β",):
            # Check no double β at front
            if n >= 2 and ops[1] == ("β",):
                return None  # β^2 = 0
            out.append(1)
            i = 1
        else:
            out.append(0)
        # Now expect P-blocks separated by β-bits.
        while i < n:
            if ops[i][0] != "P":
                # double β or other malformed → zero
                return None
            s = ops[i][1]
            out.append(s)
            i += 1
            # β-bit
            if i < n and ops[i] == ("β",):
                # check next isn't also β
                if i + 1 < n and ops[i + 1] == ("β",):
                    return None
                out.append(1)
                i += 1
            else:
                out.append(0)
        return tuple(out)

    def _adem_odd_PP(self, a: int, b: int) -> Dict[Tuple, int]:
        """Adem relation for P^a P^b (no Bockstein between), a < p b.

        Returns dict mapping ops-list tuples (canonical form) to F_p coefficients.
        Boundary: if a >= p*b, returns the admissible singleton.
        """
        p = self.prime
        if a == 0:
            # P^0 = 1, so P^a P^b = P^b
            return {(("P", b),): 1} if b > 0 else {(): 1}
        if b == 0:
            return {(("P", a),): 1}
        if a >= p * b:
            return {(("P", a), ("P", b)): 1}
        result: Dict[Tuple, int] = {}
        for t in range(0, a // p + 1):
            coef = _binom_mod_p((p - 1) * (b - t) - 1, a - p * t, p)
            if coef == 0:
                continue
            sign = 1 if (a + t) % 2 == 0 else p - 1  # (-1)^{a+t} in F_p
            c = (sign * coef) % p
            if c == 0:
                continue
            new_a = a + b - t
            new_b = t
            ops: List[Tuple] = []
            if new_a > 0:
                ops.append(("P", new_a))
            if new_b > 0:
                ops.append(("P", new_b))
            key = tuple(ops)
            result[key] = (result.get(key, 0) + c) % p
        return {k: v for k, v in result.items() if v != 0}

    def _adem_odd_PbP(self, a: int, b: int) -> Dict[Tuple, int]:
        """Adem relation for P^a β P^b, a ≤ p b.

        Returns dict mapping ops-list tuples to F_p coefficients.
        Boundary: if a > p*b, returns the admissible singleton.
        """
        p = self.prime
        if a == 0:
            # P^0 β P^b = β P^b
            ops_a: List[Tuple] = [("β",)]
            if b > 0:
                ops_a.append(("P", b))
            return {tuple(ops_a): 1}
        if b == 0:
            # P^a β P^0 = P^a β
            ops_b: List[Tuple] = [("P", a), ("β",)]
            return {tuple(ops_b): 1}
        if a > p * b:
            return {(("P", a), ("β",), ("P", b)): 1}
        result: Dict[Tuple, int] = {}
        # First sum: (-1)^{a+t} · C((p-1)(b-t), a-pt) · β P^{a+b-t} P^t
        for t in range(0, a // p + 1):
            coef = _binom_mod_p((p - 1) * (b - t), a - p * t, p)
            if coef == 0:
                continue
            sign = 1 if (a + t) % 2 == 0 else p - 1
            c = (sign * coef) % p
            if c == 0:
                continue
            new_a = a + b - t
            new_b = t
            ops: List[Tuple] = [("β",)]
            if new_a > 0:
                ops.append(("P", new_a))
            if new_b > 0:
                ops.append(("P", new_b))
            key = tuple(ops)
            result[key] = (result.get(key, 0) + c) % p
        # Second sum (prefixed by minus): - Σ (-1)^{a+t-1} C((p-1)(b-t)-1, a-pt-1) P^{a+b-t} β P^t
        # The combined sign is - · (-1)^{a+t-1} = (-1)^{a+t}.
        for t in range(0, (a - 1) // p + 1):
            if a - p * t - 1 < 0:
                continue
            coef = _binom_mod_p((p - 1) * (b - t) - 1, a - p * t - 1, p)
            if coef == 0:
                continue
            sign = 1 if (a + t) % 2 == 0 else p - 1
            c = (sign * coef) % p
            if c == 0:
                continue
            new_a = a + b - t
            new_b = t
            ops: List[Tuple] = []
            if new_a > 0:
                ops.append(("P", new_a))
            ops.append(("β",))
            if new_b > 0:
                ops.append(("P", new_b))
            key = tuple(ops)
            result[key] = (result.get(key, 0) + c) % p
        return {k: v for k, v in result.items() if v != 0}

    def _find_first_violation_odd(self, ops: List[Tuple]) -> Optional[int]:
        """Return the index of the first P^a in ops that begins a non-admissible
        pair (P^a P^b with a < p b, or P^a β P^b with a ≤ p b). Returns None
        if ops is admissible (no rewriting needed at the P-P-pair level).
        """
        p = self.prime
        for i, op in enumerate(ops):
            if op[0] != "P":
                continue
            a = op[1]
            # find next P op
            j = i + 1
            beta_between = False
            while j < len(ops) and ops[j][0] != "P":
                if ops[j] == ("β",):
                    beta_between = True
                j += 1
            if j >= len(ops):
                return None
            b = ops[j][1]
            if beta_between:
                if a <= p * b:
                    return i
            else:
                if a < p * b:
                    return i
        return None

    def _reduce_ops_odd(self, ops: Tuple[Tuple, ...]) -> SteenrodElement:
        """Recursively reduce an ops tuple to admissible-encoded SteenrodElement.

        Memoised. Detects β^2 = 0 and zero-exponent P drops.
        """
        # Normalize: drop ("P", 0), collapse β·β to zero.
        norm: List[Tuple] = []
        for op in ops:
            if op[0] == "P" and op[1] == 0:
                continue
            if op == ("β",) and norm and norm[-1] == ("β",):
                # β^2 = 0
                return {}
            norm.append(op)
        ops_t = tuple(norm)
        cache_key = ("ops_odd", self.prime, ops_t)
        if cache_key in self._pair_cache:
            return dict(self._pair_cache[cache_key])
        # Convert to encoded form if admissible
        violation_idx = self._find_first_violation_odd(list(ops_t))
        if violation_idx is None:
            encoded = self._ops_to_seq_odd(list(ops_t))
            if encoded is None:
                res: SteenrodElement = {}
            else:
                res = {encoded: 1}
            self._pair_cache[cache_key] = dict(res)
            return res
        # Apply the Adem relation at violation_idx
        ops_list = list(ops_t)
        a = ops_list[violation_idx][1]
        # find the next P and detect β between
        j = violation_idx + 1
        beta_between = False
        while j < len(ops_list) and ops_list[j][0] != "P":
            if ops_list[j] == ("β",):
                beta_between = True
            j += 1
        b = ops_list[j][1]
        prefix = tuple(ops_list[:violation_idx])
        suffix = tuple(ops_list[j + 1 :])
        if beta_between:
            expansion = self._adem_odd_PbP(a, b)
        else:
            expansion = self._adem_odd_PP(a, b)
        result: SteenrodElement = {}
        p = self.prime
        for replacement, coef in expansion.items():
            new_ops = prefix + replacement + suffix
            sub = self._reduce_ops_odd(new_ops)
            for k, v in sub.items():
                nv = (result.get(k, 0) + coef * v) % p
                if nv == 0:
                    result.pop(k, None)
                else:
                    result[k] = nv
        self._pair_cache[cache_key] = dict(result)
        return result

    def _to_admissible_odd(self, seq: AdmissibleSequence) -> SteenrodElement:
        """Reduce odd-prime monomial to a sum of admissible monomials.

        Accepts either:
          - A canonical admissible-encoded tuple (e_0, s_1, e_1, ..., s_k, e_k).
          - The empty tuple () = unit.
          - Singletons (0,) = unit, (1,) = β.

        Implementation: parse into an ops list, then apply Adem relations
        recursively at the leftmost non-admissible P-P or P-β-P pair.
        """
        if not seq:
            return {(): 1}
        if self.is_admissible(seq):
            return {seq: 1}
        # Parse into ops list.
        ops = self._seq_to_ops_odd(seq)
        return self._reduce_ops_odd(tuple(ops))

    def _concat_seqs_odd(
        self, a: AdmissibleSequence, b: AdmissibleSequence
    ) -> Optional[AdmissibleSequence]:
        """Concatenate two canonical odd-prime encoded tuples by merging the
        trailing β-bit of `a` with the leading β-bit of `b`.

        Returns the merged tuple, or None if the merge produces β^2 = 0.
        """
        if not a:
            return b
        if not b:
            return a
        merged = a[-1] + b[0]
        if merged >= 2:
            return None
        return a[:-1] + (merged,) + b[1:]

    def concat_admissibles(
        self, a: AdmissibleSequence, b: AdmissibleSequence
    ) -> SteenrodElement:
        """Concatenate two admissible monomials and reduce to admissibles.

        Prime-aware: at p=2, raw tuple concatenation works; at odd primes
        the trailing/leading β-bits are merged.
        """
        if self.prime == 2:
            return self.to_admissible(a + b)
        merged = self._concat_seqs_odd(a, b)
        if merged is None:
            return {}
        return self.to_admissible(merged)

    def mul(self, a: SteenrodElement, b: SteenrodElement) -> SteenrodElement:
        """Multiply two SteenrodElements; result reduced to admissibles."""
        result: SteenrodElement = {}
        for sa, ca in a.items():
            for sb, cb in b.items():
                reduced = self.concat_admissibles(sa, sb)
                coef = (ca * cb) % self.prime
                for k, v in reduced.items():
                    nv = (result.get(k, 0) + coef * v) % self.prime
                    if nv == 0:
                        result.pop(k, None)
                    else:
                        result[k] = nv
        return result


# ════════════════════════════════════════════════════════════════════════════
# Layer 2 — F_p cohomology ring + Steenrod action
# ════════════════════════════════════════════════════════════════════════════


class FpCohomologyRing(BaseModel):
    """A graded F_p-cohomology ring with optional Sq-action data.

    Overview:
        Encodes H^*(X; F_p) as a graded F_p-vector space with cup product
        and (optional) Steenrod action data.  Used as input to the Adams
        E_2 page computation.

    Key Concepts:
        - basis: per-degree list of basis-class labels.
        - cup_table: bilinear cup product on labels → label-coefficient dict.
        - sq_table: explicit (i, generator)-action data; everything else is
          derived via the instability axiom and Cartan formula.

    Coefficient Ring:
        F_p with p ∈ {2, 3, 5}.

    Attributes:
        space_label (str): Human-readable name (e.g., "S^3", "CP^2", "RP^5").
        prime (int): Coefficient prime.
        max_degree (int): Top non-zero internal degree.
        basis (Dict[int, List[str]]): Degree → list of class labels.
        cup_table (Dict): (label_a, label_b) → {label_c: coef} for cup.
        sq_table (Dict): (i, label) → {label: coef} for Sq^i (mod 2)
            or P^i (mod p odd).
        ring_generators (List[str]): Labels of polynomial-algebra generators.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    space_label: str = ""
    prime: Literal[2, 3, 5] = 2
    max_degree: int
    basis: Dict[int, List[str]] = Field(default_factory=dict)
    degree_of: Dict[str, int] = Field(default_factory=dict)
    cup_table: Dict[Tuple[str, str], Dict[str, int]] = Field(default_factory=dict)
    sq_table: Dict[Tuple[int, str], Dict[str, int]] = Field(default_factory=dict)
    ring_generators: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_basis(self) -> "FpCohomologyRing":
        for d, labels in self.basis.items():
            for label in labels:
                if label in self.degree_of and self.degree_of[label] != d:
                    raise ValueError(
                        f"Label {label} has inconsistent degrees "
                        f"{self.degree_of[label]} vs {d}."
                    )
                self.degree_of[label] = d
        return self

    def betti_numbers(self) -> Dict[int, int]:
        """Return dimensions per degree for compatibility with Sullivan models."""
        return {d: len(labels) for d, labels in self.basis.items()}

    def fp_dim(self, t: int) -> int:
        """F_p dimension of H^t."""
        return len(self.basis.get(t, []))


class SteenrodAction:
    """Action of A_p on an FpCohomologyRing.

    Overview:
        Computes Sq^i(x) (or P^i(x) / β(x)) on every basis class of the
        cohomology ring via:
            1. Explicit ``sq_table`` entries (for ring generators).
            2. Cartan formula:  Sq^n(xy) = Σ_{i+j=n} Sq^i(x)·Sq^j(y).
            3. Instability:  Sq^i(x) = 0 if i > deg(x); Sq^{deg(x)}(x) = x²
               (mod 2); P^i(x) = 0 if 2i > deg(x) (mod p odd).

    Coefficient Ring:
        F_p, with p ∈ {2, 3, 5}.

    References:
        Milnor, J. (1958). The Steenrod algebra and its dual.
        Steenrod, N. & Epstein, D. (1962). Cohomology Operations.
    """

    def __init__(self, algebra: SteenrodAlgebra, ring: FpCohomologyRing):
        if algebra.prime != ring.prime:
            raise ValueError(
                f"prime mismatch: algebra={algebra.prime} vs ring={ring.prime}."
            )
        self.algebra = algebra
        self.ring = ring
        self._action_cache: Dict[Tuple[AdmissibleSequence, str], Dict[str, int]] = {}

    # ── Action of a single Sq^i / P^i / β on a basis class ─────────────────

    def _apply_basic_op(self, op_idx: int, label: str) -> Dict[str, int]:
        """Apply a single basic operation (Sq^i for p=2, or coded op for p odd)
        to the basis class `label`. Returns a sparse dict label → coef.
        """
        # check sq_table cache (user-provided)
        key = (op_idx, label)
        if key in self.ring.sq_table:
            return dict(self.ring.sq_table[key])
        d = self.ring.degree_of[label]
        p = self.ring.prime
        # Instability axioms for p=2
        if p == 2:
            if op_idx == 0:
                return {label: 1}
            if op_idx > d:
                return {}
            if op_idx == d:
                # Sq^{|x|}(x) = x²; compute via cup table
                if (label, label) in self.ring.cup_table:
                    return dict(self.ring.cup_table[(label, label)])
                return {}
        # No further default: must be in sq_table or derivable from generators
        # via Cartan; we attempt Cartan if label is decomposable.
        cartan = self._try_cartan_decomposition(op_idx, label, p)
        if cartan is not None:
            return cartan
        raise ValueError(
            f"Action of op_idx={op_idx} on label '{label}' (deg {d}) is not "
            "specified in sq_table and cannot be derived from generators."
        )

    def _try_cartan_decomposition(
        self, op_idx: int, label: str, p: int
    ) -> Optional[Dict[str, int]]:
        """If `label` is a cup product of two factors with a rule in
        cup_table, apply Cartan: Sq^n(xy) = Σ Sq^i(x)·Sq^j(y).

        Returns None when no decomposition is found.
        """
        # search cup_table for a representation label = x · y (with coef 1)
        for (la, lb), cdict in self.ring.cup_table.items():
            if cdict.get(label, 0) == 1 and la in self.ring.degree_of and lb in self.ring.degree_of:
                # avoid trivial/self-decomposition into the same label
                if la == label or lb == label:
                    continue
                # Cartan: Σ_{i+j = op_idx} Sq^i(la)·Sq^j(lb)  [mod 2]
                # For odd p with P, same formula on P; β is a derivation.
                if p == 2:
                    result: Dict[str, int] = {}
                    for i in range(0, op_idx + 1):
                        j = op_idx - i
                        sq_i_la = self._apply_basic_op(i, la)
                        sq_j_lb = self._apply_basic_op(j, lb)
                        for ka, ca in sq_i_la.items():
                            for kb, cb in sq_j_lb.items():
                                prod = self.ring.cup_table.get((ka, kb), {})
                                for kc, cc in prod.items():
                                    nv = (result.get(kc, 0) + ca * cb * cc) % 2
                                    if nv == 0:
                                        result.pop(kc, None)
                                    else:
                                        result[kc] = nv
                    return result
                # odd p Cartan kept minimal: only support op_idx encoding P^i (no β here)
                # We keep this conservative.
                return None
        return None

    def _apply_admissible(
        self, seq: AdmissibleSequence, label: str
    ) -> Dict[str, int]:
        """Apply admissible monomial seq to basis class label (right-to-left)."""
        cache_key = (seq, label)
        if cache_key in self._action_cache:
            return dict(self._action_cache[cache_key])
        p = self.ring.prime
        if not seq:
            res = {label: 1}
            self._action_cache[cache_key] = dict(res)
            return res
        # Apply rightmost operator first
        if p == 2:
            head = seq[0]
            tail = seq[1:]
            # First apply tail to label, then head to the result
            tail_img = self._apply_admissible(tail, label)
            result: Dict[str, int] = {}
            for k, c in tail_img.items():
                step = self._apply_basic_op(head, k)
                for kk, cc in step.items():
                    nv = (result.get(kk, 0) + c * cc) % 2
                    if nv == 0:
                        result.pop(kk, None)
                    else:
                        result[kk] = nv
            self._action_cache[cache_key] = dict(result)
            return result
        # odd-prime path is intentionally limited
        raise NotImplementedError("Odd-prime action not fully implemented.")

    def apply(self, sq_elt: SteenrodElement, label: str) -> Dict[str, int]:
        """Apply a Steenrod element (sum of admissibles) to a basis class."""
        result: Dict[str, int] = {}
        p = self.ring.prime
        for seq, coef in sq_elt.items():
            img = self._apply_admissible(seq, label)
            for k, v in img.items():
                nv = (result.get(k, 0) + coef * v) % p
                if nv == 0:
                    result.pop(k, None)
                else:
                    result[k] = nv
        return result

    def action_matrix(
        self, seq: AdmissibleSequence, t_in: int
    ) -> sp.csr_matrix:
        """The matrix of seq:  H^{t_in} → H^{t_in + |seq|} over F_p.

        Returns:
            scipy.sparse.csr_matrix of shape (rows=fp_dim(t_out), cols=fp_dim(t_in)).
        """
        t_out = t_in + self.algebra.degree_of(seq)
        cols = self.ring.fp_dim(t_in)
        rows = self.ring.fp_dim(t_out)
        if cols == 0 or rows == 0:
            return sp.csr_matrix((rows, cols), dtype=np.int64)
        in_basis = self.ring.basis.get(t_in, [])
        out_basis = self.ring.basis.get(t_out, [])
        out_idx = {lbl: i for i, lbl in enumerate(out_basis)}
        data, row_inds, col_inds = [], [], []
        for j, lbl in enumerate(in_basis):
            img = self._apply_admissible(seq, lbl)
            for klbl, c in img.items():
                if c == 0:
                    continue
                if klbl not in out_idx:
                    continue  # higher-degree image clipped
                data.append(int(c))
                row_inds.append(out_idx[klbl])
                col_inds.append(j)
        mat = sp.csr_matrix(
            (data, (row_inds, col_inds)), shape=(rows, cols), dtype=np.int64
        )
        return mat


# ════════════════════════════════════════════════════════════════════════════
# Public: steenrod_squares_matrix
# ════════════════════════════════════════════════════════════════════════════


def steenrod_squares_matrix(
    cohomology_ring: FpCohomologyRing,
    prime: int = 2,
    max_i: Optional[int] = None,
) -> Dict[Tuple[int, int], sp.csr_matrix]:
    """Build matrices for Sq^i (mod 2) or P^i (mod p odd) on each H^j.

    What is Being Computed?:
        For each (i, j) with i ≤ max_i and j ≤ ring.max_degree, returns the
        sparse matrix of the linear map  Sq^i : H^j(X; F_p) → H^{i+j}(X; F_p).

    Algorithm:
        1. Build a SteenrodAlgebra at the requested prime.
        2. Build a SteenrodAction binding the algebra to the input ring.
        3. For each basis class x of degree j, compute Sq^i(x) via Cartan
           and the explicit sq_table on ring generators.
        4. Pack column-by-column into a scipy.sparse.csr_matrix.

    Preserved Invariants:
        - All matrices are scipy.sparse.csr_matrix (never dense).
        - Entries are reduced mod p.

    Args:
        cohomology_ring: An FpCohomologyRing.
        prime: The coefficient prime (must match ring.prime).
        max_i: Optional upper bound on i; defaults to ring.max_degree.

    Returns:
        Dict[(i, j) → csr_matrix] of shape (fp_dim(i+j), fp_dim(j)).

    Use When:
        - Building input data for the Adams E_2 page.
        - Verifying instability or Cartan-derived actions on test rings.

    Example:
        >>> ring = real_projective_space_fp(4, prime=2)
        >>> mats = steenrod_squares_matrix(ring, prime=2)
        >>> mats[(1, 1)].toarray().tolist()  # Sq^1: H^1 → H^2
        [[1]]
    """
    if prime != cohomology_ring.prime:
        raise ValueError(
            f"prime={prime} doesn't match cohomology_ring.prime={cohomology_ring.prime}."
        )
    if prime != 2:
        raise NotImplementedError(
            "steenrod_squares_matrix supports prime=2 only."
        )
    if max_i is None:
        max_i = cohomology_ring.max_degree
    algebra = SteenrodAlgebra(prime=prime, max_t=max_i + cohomology_ring.max_degree)
    action = SteenrodAction(algebra, cohomology_ring)
    out: Dict[Tuple[int, int], sp.csr_matrix] = {}
    for j in range(0, cohomology_ring.max_degree + 1):
        if cohomology_ring.fp_dim(j) == 0:
            continue
        for i in range(0, max_i + 1):
            if i + j > cohomology_ring.max_degree:
                continue
            seq: AdmissibleSequence = () if i == 0 else (i,)
            mat = action.action_matrix(seq, j)
            assert sp.issparse(mat), "steenrod_squares_matrix produced non-sparse output."
            out[(i, j)] = mat
    return out


# ════════════════════════════════════════════════════════════════════════════
# Layer 3 — minimal free resolution + Ext computation
# ════════════════════════════════════════════════════════════════════════════


class _MinimalResolution:
    """Minimal free A-resolution of an A-module M up to (s_max, t_max).

    Represents F_• → M with each F_s = ⊕_α A · g_{s,α}, generators
    g_{s,α} of internal degree n_{s,α}. The differentials d_s are stored
    as dictionaries g_{s,α} → SteenrodElement-vector in F_{s-1}.

    This is the Bruner-style layout: minimality is enforced by always
    picking generators in the quotient by A_{>0}·(prev kernel).
    """

    def __init__(self, action: SteenrodAction, s_max: int, t_max: int):
        self.action = action
        self.algebra = action.algebra
        self.ring = action.ring
        self.p = action.ring.prime
        self.s_max = s_max
        self.t_max = t_max
        # generators[s] is a list of dicts:
        #   {"degree": n, "image": Dict[label_in_F_{s-1}, coef]}
        # where for s=0 the "image" lives in M.
        self.generators: List[List[Dict]] = []

    # ── Helper: vectors over F_p in flattened "F_s as F_p-vec" indexing ────

    def _free_module_index(self, s: int, t: int) -> List[Tuple[int, AdmissibleSequence]]:
        """Basis of (F_s)_t as an F_p-vector space.

        Each basis element is (alpha, admissible_seq) where alpha is the
        index of a generator of F_s and admissible_seq has internal degree
        t - n_{s,α}.
        """
        if s < 0 or s >= len(self.generators):
            return []
        result = []
        for alpha, gen in enumerate(self.generators[s]):
            n = gen["degree"]
            if t - n < 0:
                continue
            for adm in self.algebra.admissible_basis(t - n):
                result.append((alpha, adm))
        return result

    # ── Build F_0: minimal generators of M ─────────────────────────────────

    def build_F0(self) -> None:
        """F_0: pick minimal generators of M = H*(X; F_p) as A-module.

        Algorithm:
            - For each degree t from 0 to t_max, compute A_>0 · M ∩ M_t as a
              span of {Sq^I(x) : 0 < |I| ≤ t, x ∈ M_{t-|I|}, I admissible}.
            - Quotient M_t by this span; basis of the quotient = new generators
              in degree t.
            - For each new generator, its image in M is the original basis class.
        """
        ring = self.ring
        F0_gens: List[Dict] = []
        # We'll express M_t basis as just labels (already a basis).
        # A_>0 · M ∩ M_t spanned by {action_matrix((i,), t-i) · e_x : i in 1..t}.
        a_pos_M: Dict[int, List[Dict[int, int]]] = {}
        for t in range(0, self.t_max + 1):
            spans: List[Dict[int, int]] = []
            for i in range(1, t + 1):
                if ring.fp_dim(t - i) == 0:
                    continue
                # admissibles of degree i (only single-Sq used here is enough
                # because admissibles of degree i are spanned by single Sq^i? No —
                # admissibles of degree i are {Sq^j Sq^k …}, but their images on
                # M_{t-i} are computed directly).
                for adm in self.algebra.admissible_basis(i):
                    if not adm:
                        continue
                    mat = self.action.action_matrix(adm, t - i)
                    if mat.nnz == 0:
                        continue
                    coo = mat.tocoo()
                    # each column of mat is the image of one M_{t-i} basis vec
                    cols_to_vecs: Dict[int, Dict[int, int]] = {}
                    for r, c, v in zip(coo.row, coo.col, coo.data):
                        v = int(v) % self.p
                        if v == 0:
                            continue
                        cols_to_vecs.setdefault(int(c), {})[int(r)] = v
                    for vec in cols_to_vecs.values():
                        spans.append(vec)
            a_pos_M[t] = spans
            # Now compute M_t / span. Build a matrix span × M_t basis indices.
            n_M_t = ring.fp_dim(t)
            if n_M_t == 0:
                continue
            # flatten span into rows of a sparse matrix
            data, row_inds, col_inds = [], [], []
            for ridx, vec in enumerate(spans):
                for cidx, v in vec.items():
                    data.append(int(v))
                    row_inds.append(ridx)
                    col_inds.append(cidx)
            n_rows = len(spans)
            if n_rows == 0:
                # all basis vectors of M_t are minimal generators
                quotient_indices = list(range(n_M_t))
            else:
                span_mat = sp.csr_matrix(
                    (data, (row_inds, col_inds)),
                    shape=(n_rows, n_M_t),
                    dtype=np.int64,
                )
                # find basis of the span via row echelon; then complement.
                pivots = _row_echelon_pivots(span_mat, self.p)
                quotient_indices = [j for j in range(n_M_t) if j not in pivots]
            for qi in quotient_indices:
                lbl = ring.basis[t][qi]
                # Image in M: the basis vector e_lbl
                F0_gens.append({"degree": t, "image": {lbl: 1}})
        self.generators.append(F0_gens)

    # ── Build F_s (s ≥ 1): generators of ker(d_{s-1}) modulo A_>0·(prev) ────

    def build_F_s(self, s: int) -> None:
        """Construct F_s and the differential d_s: F_s → F_{s-1}.

        Approach:
            For each internal degree t = 0..t_max:
                1. Build the linear map  d_{s-1}: (F_{s-1})_t → (F_{s-2})_t
                   (or → M_t when s-1 = 0) as a sparse F_p matrix.
                2. Compute ker((F_{s-1})_t → ⋯) as F_p-subspace.
                3. Subtract A_>0 · K_{<t} ∩ K_t to find new generators.
                4. For each new generator, record its image in F_{s-1}.
        """
        if s == 0:
            self.build_F0()
            return
        prev = self.generators[s - 1]
        # Compute kernel of d_{s-1} per internal degree
        # We track: ker_at_t[t] = list of sparse F_p vectors in basis (F_{s-1})_t
        # plus a list of new generators (already-included K vectors) for "A_>0·K".
        new_gens: List[Dict] = []
        # K_subspace_basis_at_t[t]: list of vectors that are linear combos
        # currently in our submodule K = ker(d_{s-1}).
        # A_pos_K_at_t[t]: span of A_>0 · K_<t that lands in degree t (basis vectors).
        K_at_t: Dict[int, List[Dict[int, int]]] = {}
        for t in range(0, self.t_max + 1):
            # 1. Build the basis of (F_{s-1})_t as F_p-vec, indexed by (alpha, adm)
            basis_F_sm1 = self._free_module_index(s - 1, t)
            n_basis = len(basis_F_sm1)
            if n_basis == 0:
                K_at_t[t] = []
                continue
            # 2. Build d_{s-1} from each basis vector: (alpha, adm) → action.
            # The differential of g_{s-1,alpha} is recorded; admissible · g sends
            # to admissible-action on its image.
            target_basis_index, target_dim = self._target_basis_indexer(s - 1, t)
            data, row_inds, col_inds = [], [], []
            for col_idx, (alpha, adm) in enumerate(basis_F_sm1):
                # image of adm · g_{s-1, alpha} under d_{s-1}
                g_image = prev[alpha]["image"]  # in F_{s-2} or M if s-1==0
                # apply adm to g_image:
                # g_image is dict label_or_(beta,adm') → coef
                # We apply admissible 'adm' on top; mul in A then act.
                applied = self._apply_adm_to_F_image(adm, g_image, s - 1)
                # convert applied → column entries in target basis
                for key, c in applied.items():
                    if c == 0:
                        continue
                    if key not in target_basis_index:
                        continue  # truncated beyond t
                    data.append(int(c))
                    row_inds.append(target_basis_index[key])
                    col_inds.append(col_idx)
            d_mat = sp.csr_matrix(
                (data, (row_inds, col_inds)),
                shape=(target_dim, n_basis),
                dtype=np.int64,
            )
            # 3. Kernel
            kernel_basis = _sparse_fp_kernel(d_mat, self.p)
            K_at_t[t] = list(kernel_basis)
            # 4. Subtract A_>0 · (existing new_gens at degree < t) from kernel
            # to find truly-new generators.
            # We assemble image of admissible-action on each new generator into
            # a span of "already-covered" kernel directions.
            already_covered: List[Dict[int, int]] = []
            for gi, ng in enumerate(new_gens):
                n_g = ng["degree"]
                if n_g >= t:
                    continue
                diff = t - n_g
                for adm in self.algebra.admissible_basis(diff):
                    if not adm:
                        continue
                    # adm · g_gi as an element of (F_{s-1})_t at basis (alpha, adm') ...
                    # but new_gens are generators of F_s, not F_{s-1}.
                    # Their kernel-vector in (F_{s-1})_n_g lifts to admissible-action
                    # vector in (F_{s-1})_t.
                    lifted = self._lift_via_admissible(adm, ng["kernel_vec"], s - 1, n_g, t)
                    if lifted:
                        already_covered.append(lifted)
            # quotient kernel_basis by span(already_covered)
            new_kernel_dirs = _quotient_basis(kernel_basis, already_covered, n_basis, self.p)
            for vec in new_kernel_dirs:
                # convert vec (sparse dict over basis_F_sm1 indices) to image-form
                # for use as the differential of a new generator g_s.
                image_in_F_sm1: Dict[Tuple[int, AdmissibleSequence], int] = {}
                for col_idx, c in vec.items():
                    alpha, adm = basis_F_sm1[col_idx]
                    key = (alpha, adm)
                    image_in_F_sm1[key] = (image_in_F_sm1.get(key, 0) + int(c)) % self.p
                # remove zeros
                image_in_F_sm1 = {k: v for k, v in image_in_F_sm1.items() if v != 0}
                new_gens.append({
                    "degree": t,
                    "image": image_in_F_sm1,
                    "kernel_vec": dict(vec),  # for further A_>0 propagation
                })
        self.generators.append(new_gens)

    # ── Helpers used by build_F_s ──────────────────────────────────────────

    def _target_basis_indexer(
        self, s_minus_1: int, t: int
    ) -> Tuple[Dict, int]:
        """Build a key→index map for the codomain of d_{s_minus_1}."""
        if s_minus_1 == 0:
            # codomain is M_t
            ring = self.ring
            in_basis = ring.basis.get(t, [])
            return ({lbl: i for i, lbl in enumerate(in_basis)}, len(in_basis))
        # codomain is (F_{s_minus_1 - 1})_t
        basis = self._free_module_index(s_minus_1 - 1, t)
        return ({(alpha, adm): i for i, (alpha, adm) in enumerate(basis)}, len(basis))

    def _apply_adm_to_F_image(
        self,
        adm: AdmissibleSequence,
        image: Dict,
        prev_s: int,
    ) -> Dict:
        """Apply admissible 'adm' to an image vector in F_{prev_s - 1} (or M).

        The image has the form:
            - if prev_s == 0: dict label → coef (lives in M).
            - if prev_s > 0:  dict (alpha, admissible) → coef (lives in F_{prev_s-1}).
        """
        if prev_s == 0:
            # apply adm to each label, multiply, accumulate in M_t basis
            result: Dict[str, int] = {}
            for lbl, c in image.items():
                step = self.action._apply_admissible(adm, lbl)
                for klbl, cc in step.items():
                    nv = (result.get(klbl, 0) + c * cc) % self.p
                    if nv == 0:
                        result.pop(klbl, None)
                    else:
                        result[klbl] = nv
            return result
        # prev_s > 0: image is in F_{prev_s - 1}. We act on the admissible component
        # by left-multiplication in A. Concatenate adm with adm', reduce.
        result2: Dict[Tuple[int, AdmissibleSequence], int] = {}
        for (alpha, adm_inner), c in image.items():
            product_seq = adm + adm_inner
            reduced = self.algebra.to_admissible(product_seq)
            for r_seq, r_coef in reduced.items():
                key = (alpha, r_seq)
                nv = (result2.get(key, 0) + c * r_coef) % self.p
                if nv == 0:
                    result2.pop(key, None)
                else:
                    result2[key] = nv
        return result2

    def _lift_via_admissible(
        self,
        adm: AdmissibleSequence,
        kernel_vec: Dict[int, int],
        s_minus_1: int,
        from_t: int,
        to_t: int,
    ) -> Dict[int, int]:
        """Given a kernel vector at (F_{s-1})_{from_t}, lift via adm-action to
        a vector at (F_{s-1})_{to_t}.

        kernel_vec is keyed by indices into _free_module_index(s_minus_1, from_t).
        Result is keyed by indices into _free_module_index(s_minus_1, to_t).
        """
        from_basis = self._free_module_index(s_minus_1, from_t)
        to_basis = self._free_module_index(s_minus_1, to_t)
        to_index = {pair: i for i, pair in enumerate(to_basis)}
        result: Dict[int, int] = {}
        for col_idx, c in kernel_vec.items():
            if c == 0:
                continue
            alpha, adm_inner = from_basis[col_idx]
            product = adm + adm_inner
            reduced = self.algebra.to_admissible(product)
            for r_seq, r_coef in reduced.items():
                key = (alpha, r_seq)
                if key not in to_index:
                    continue
                idx = to_index[key]
                nv = (result.get(idx, 0) + c * r_coef) % self.p
                if nv == 0:
                    result.pop(idx, None)
                else:
                    result[idx] = nv
        return result

    # ── Read off Ext^{s,t} as the count of generators of F_s in degree t ────

    def ext_dimensions(self) -> Dict[Tuple[int, int], int]:
        out: Dict[Tuple[int, int], int] = {}
        for s, gens in enumerate(self.generators):
            for g in gens:
                t = g["degree"]
                out[(s, t)] = out.get((s, t), 0) + 1
        return out


def _row_echelon_pivots(mat: sp.csr_matrix, p: int) -> List[int]:
    """Return indices of pivot columns in F_p-RREF of mat. No dense conversion."""
    if mat.shape[0] == 0 or mat.shape[1] == 0:
        return []
    rows, cols = mat.shape
    # Build sparse rows as dicts
    R: List[Dict[int, int]] = []
    coo = mat.tocoo()
    for _ in range(rows):
        R.append({})
    for r, c, v in zip(coo.row, coo.col, coo.data):
        v = int(v) % p
        if v != 0:
            R[int(r)][int(c)] = v
    used = set()
    pivots: List[int] = []
    for c in range(cols):
        pr = None
        for r in range(rows):
            if r in used:
                continue
            if R[r].get(c, 0) != 0:
                pr = r
                break
        if pr is None:
            continue
        pv = R[pr][c]
        if pv != 1:
            inv = pow(pv, p - 2, p)
            R[pr] = {k: (v * inv) % p for k, v in R[pr].items()}
        for r in range(rows):
            if r == pr or r in used:
                continue
            f = R[r].get(c, 0)
            if f == 0:
                continue
            new_row = dict(R[r])
            for k, v in R[pr].items():
                nv = (new_row.get(k, 0) - f * v) % p
                if nv == 0:
                    new_row.pop(k, None)
                else:
                    new_row[k] = nv
            R[r] = new_row
        used.add(pr)
        pivots.append(c)
    return pivots


def _quotient_basis(
    candidates: List[Dict[int, int]],
    covered: List[Dict[int, int]],
    n_cols: int,
    p: int,
) -> List[Dict[int, int]]:
    """Return candidates whose span modulo span(covered) has new dimensions.

    Implementation: stack covered + candidates as rows of a sparse matrix,
    perform F_p-RREF, identify which candidate-row positions are independent
    (i.e., have a pivot beyond the covered block).

    Returns the candidate sparse vectors that form a basis of the quotient.
    """
    if not candidates:
        return []
    rows = covered + candidates
    n_covered = len(covered)
    # Build a sparse matrix (rows × n_cols)
    data, row_inds, col_inds = [], [], []
    for ri, vec in enumerate(rows):
        for ci, v in vec.items():
            v = int(v) % p
            if v == 0:
                continue
            data.append(v)
            row_inds.append(ri)
            col_inds.append(ci)
    if not data:
        # everything zero → no new directions
        return []
    M = sp.csr_matrix(
        (data, (row_inds, col_inds)),
        shape=(len(rows), n_cols),
        dtype=np.int64,
    )
    # perform Gaussian elimination, track which original-rows survive
    R: List[Dict[int, int]] = []
    coo = M.tocoo()
    for _ in range(len(rows)):
        R.append({})
    for r, c, v in zip(coo.row, coo.col, coo.data):
        v = int(v) % p
        if v != 0:
            R[int(r)][int(c)] = v
    survivors: List[int] = []  # row indices that contributed a new pivot
    used = set()
    for c in range(n_cols):
        pr = None
        for r in range(len(rows)):
            if r in used:
                continue
            if R[r].get(c, 0) != 0:
                pr = r
                break
        if pr is None:
            continue
        pv = R[pr][c]
        if pv != 1:
            inv = pow(pv, p - 2, p)
            R[pr] = {k: (val * inv) % p for k, val in R[pr].items()}
        for r in range(len(rows)):
            if r == pr or r in used:
                continue
            f = R[r].get(c, 0)
            if f == 0:
                continue
            new_row = dict(R[r])
            for k, val in R[pr].items():
                nv = (new_row.get(k, 0) - f * val) % p
                if nv == 0:
                    new_row.pop(k, None)
                else:
                    new_row[k] = nv
            R[r] = new_row
        used.add(pr)
        if pr >= n_covered:
            survivors.append(pr - n_covered)
    return [candidates[i] for i in sorted(set(survivors))]


# ════════════════════════════════════════════════════════════════════════════
# Layer 4 — public API: AdamsE2Page contract + adams_e2_page()
# ════════════════════════════════════════════════════════════════════════════


class AdamsE2Bidegree(BaseModel):
    """One bidegree on the E_2 page."""

    s: int
    t: int
    dim: int
    generators: List[str] = Field(default_factory=list)


class AdamsDifferentialFlag(BaseModel):
    """Classification of d_r at one bidegree: forced-zero or ambiguous."""

    r: int
    source: Tuple[int, int]
    target: Tuple[int, int]
    classification: Literal["forced_zero", "ambiguous"]
    reason: str
    source_dim: int
    target_dim: int


class AdamsE2Page(BaseModel):
    """E_2 page of the Adams spectral sequence converging to π_*^s(X)_p.

    Overview:
        Holds the bigraded F_p-vector space E_2^{s,t} = Ext_{A_p}^{s,t}(H^*(X; F_p), F_p)
        together with classifications of the differentials d_r: E_r^{s,t} → E_r^{s+r, t+r-1}.

    Invariants:
        - All e2_grid keys (s, t) satisfy 0 ≤ s ≤ s_max, 0 ≤ t ≤ t_max.
        - All dims ≥ 0.
        - Each forced_vanishings entry has source_dim == 0 OR target_dim == 0.
        - Each ambiguous_differentials entry has source_dim > 0 AND target_dim > 0.
        - status == "truncated" iff resolution did not stabilise within the window.

    Attributes:
        space_label (str): Name of the space.
        prime (int): coefficient prime ∈ {2, 3, 5}.
        s_max (int): homological-degree bound.
        t_max (int): internal-degree bound.
        e2_grid (Dict): (s, t) → F_p-dimension of E_2^{s,t}.
        e2_named (List[AdamsE2Bidegree]): optional named generators.
        forced_vanishings (List[AdamsDifferentialFlag]): d_r forced to zero.
        ambiguous_differentials (List[AdamsDifferentialFlag]): d_r could be nonzero.
        reliable_window (Tuple[int, int]): (s_max, t_max - (s_max - s)) effective.
        resource_summary (Dict[str, float]): peak memory MB, wall seconds.

    References:
        Adams, J. F. (1958). On the structure and applications of the Steenrod algebra.
        May, J. P. (1981). The work of J. F. Adams. Bull. AMS 7(1).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    space_label: str = ""
    prime: Literal[2, 3, 5]
    s_max: int
    t_max: int

    e2_grid: Dict[Tuple[int, int], int]
    e2_named: List[AdamsE2Bidegree] = Field(default_factory=list)
    forced_vanishings: List[AdamsDifferentialFlag] = Field(default_factory=list)
    ambiguous_differentials: List[AdamsDifferentialFlag] = Field(default_factory=list)

    reliable_window: Tuple[int, int]
    resource_summary: Dict[str, float] = Field(default_factory=dict)

    exact: bool = True
    theorem_tag: str = ADAMS_E2_EXT_STEENROD
    contract_version: str = CONTRACT_VERSION
    status: Literal["success", "truncated", "inconclusive"]
    reasoning: str

    @field_validator("e2_grid")
    @classmethod
    def _check_grid_nonneg(
        cls, v: Dict[Tuple[int, int], int]
    ) -> Dict[Tuple[int, int], int]:
        for (s, t), d in v.items():
            if d < 0 or s < 0 or t < 0:
                raise ValueError(f"Bad entry e2_grid[{s}, {t}] = {d}.")
        return v

    @model_validator(mode="after")
    def _check_diffs(self) -> "AdamsE2Page":
        for fl in self.forced_vanishings:
            if fl.classification != "forced_zero":
                raise ValueError("forced_vanishings must have classification='forced_zero'.")
            if fl.source_dim != 0 and fl.target_dim != 0:
                raise ValueError(
                    "forced_vanishings entries must have source_dim==0 or target_dim==0."
                )
        for fl in self.ambiguous_differentials:
            if fl.classification != "ambiguous":
                raise ValueError(
                    "ambiguous_differentials must have classification='ambiguous'."
                )
            if fl.source_dim <= 0 or fl.target_dim <= 0:
                raise ValueError(
                    "ambiguous_differentials entries must have source_dim>0 and target_dim>0."
                )
        return self

    def e2_dim(self, s: int, t: int) -> int:
        return self.e2_grid.get((s, t), 0)

    def stem(self, n: int) -> Dict[int, int]:
        """All (s, t) bidegrees with t - s = n; values are dims."""
        return {s: self.e2_grid[(s, t)] for (s, t) in self.e2_grid if t - s == n}

    def decision_ready(self) -> bool:
        return self.exact and self.status == "success"


def adams_e2_page(
    cohomology_ring: FpCohomologyRing,
    prime: int = 2,
    s_max: int = 6,
    t_max: int = 20,
    memory_cap_mb: int = DEFAULT_MEM_CAP_MB,
) -> AdamsE2Page:
    """Compute the mod-p Adams E_2 page Ext_{A_p}^{s,t}(H^*(X; F_p), F_p).

    What is Being Computed?:
        E_2^{s,t} = Ext_{A_p}^{s,t}(H^*(X; F_p), F_p), the input page of
        the Adams spectral sequence converging (p-completed) to π_{t-s}^s(X).
        s = homological degree (resolution depth), t = internal grading,
        n = t - s = stem.

    Algorithm:
        1. Validate the cohomology ring and prime ∈ {2, 3, 5}.
        2. Build a SteenrodAlgebra at the prime; bind a SteenrodAction to
           the ring (Cartan + instability).
        3. Construct a minimal free A-resolution F_• → H^*(X; F_p) up to
           (s_max, t_max), tracking generators per (s, t).
        4. Read off Ext^{s,t} = (#generators of F_s in internal degree t).
        5. Classify each potential d_r: target dim 0 or source dim 0 →
           forced_zero; otherwise → ambiguous.

    Resource Guarantees:
        - All linear algebra is sparse; assertions enforce csr-only matrices.
        - t_max is clamped to T_HARD = 50.
        - Peak memory monitored via tracemalloc; if it crosses memory_cap_mb,
          the resolution truncates and returns status="truncated".
        - Admissible-basis enumeration size cap (5_000) raises
          AdamsCombinatorialError on pathological input.

    Args:
        cohomology_ring: An FpCohomologyRing (built from a known space, or
            via canned helpers like sphere_cohomology_fp / cp_n_cohomology_fp /
            rp_n_cohomology_fp).
        prime: Coefficient prime, ∈ {2, 3, 5}; must match cohomology_ring.prime.
        s_max: Maximum homological degree to compute (clamped to S_HARD).
        t_max: Maximum internal degree (clamped to T_HARD).
        memory_cap_mb: Soft memory ceiling; computation truncates if exceeded.

    Returns:
        AdamsE2Page contract with exact=True (or status="truncated"/"inconclusive").

    Use When:
        - Computing Ext over A_p for the input to the Adams spectral sequence.
        - Cross-validating with known stable homotopy stems for spheres,
          ℂP^n, ℝP^n, or other simply-connected spaces.

    Example:
        >>> ring = sphere_cohomology_fp(2, prime=2)
        >>> page = adams_e2_page(ring, prime=2, s_max=2, t_max=6)
        >>> page.e2_dim(0, 0)
        1

    References:
        Adams, J. F. (1958). On the structure and applications of the
            Steenrod algebra. Comment. Math. Helv. 32, 180-214.
        Bruner, R. R. (1993). Ext in the nineties. Contemp. Math. 146, AMS.
    """
    if prime not in (2, 3, 5):
        raise ValueError(f"Adams E_2 supports prime ∈ {{2, 3, 5}}; got {prime}.")
    if prime != cohomology_ring.prime:
        raise ValueError(
            f"prime={prime} doesn't match cohomology_ring.prime={cohomology_ring.prime}."
        )
    if t_max > T_HARD:
        warnings.warn(
            f"t_max={t_max} exceeds hard cap {T_HARD}; clamping.", stacklevel=2
        )
        t_max = T_HARD
    if s_max > S_HARD:
        warnings.warn(
            f"s_max={s_max} exceeds hard cap {S_HARD}; clamping.", stacklevel=2
        )
        s_max = S_HARD
    if t_max < 0 or s_max < 0:
        raise ValueError("s_max and t_max must be ≥ 0.")
    if prime != 2:
        # Odd-prime dispatch via cobar of the dual Steenrod algebra A_p^*.
        # The cobar route currently handles cohomology rings that are
        # *trivial* A_p-modules (no recorded P^i / beta action). For S^n at
        # odd p this is automatic by instability when n < 2(p-1); we extend
        # to any ring whose sq_table is empty/zero.
        from pysurgery.adams.odd_prime_cobar import (
            adams_e2_grid_odd_prime_trivial_module,
            is_trivial_ap_module,
        )

        if not is_trivial_ap_module(cohomology_ring):
            return AdamsE2Page(
                space_label=cohomology_ring.space_label,
                prime=prime,
                s_max=s_max,
                t_max=t_max,
                e2_grid={},
                forced_vanishings=[],
                ambiguous_differentials=[],
                reliable_window=(s_max, max(0, t_max - s_max)),
                resource_summary={"peak_mem_mb": 0.0, "wall_seconds": 0.0},
                status="inconclusive",
                reasoning=(
                    f"Odd-prime (p={prime}) Adams E_2 for non-trivial "
                    f"A_p-modules is not implemented; the cobar dispatch "
                    f"only handles modules with sq_table identically zero."
                ),
            )

        t_start = time.time()
        e2_grid = adams_e2_grid_odd_prime_trivial_module(
            cohomology_ring, prime=prime, s_max=s_max, t_max=t_max,
        )
        wall = time.time() - t_start

        # Build forced-vanishing / ambiguous flags exactly as in the p=2
        # path: for each (s, t) with dim > 0 inside the window, classify
        # each potential d_r at (r in {2..s_max+1}).
        forced: List[AdamsDifferentialFlag] = []
        ambiguous: List[AdamsDifferentialFlag] = []
        for (s, t), dim in e2_grid.items():
            if dim == 0:
                continue
            for r in range(2, s_max - s + 2):
                tgt_s = s + r
                tgt_t = t + r - 1
                if tgt_s > s_max or tgt_t > t_max:
                    continue
                src_dim = dim
                tgt_dim = e2_grid.get((tgt_s, tgt_t), 0)
                if src_dim == 0 or tgt_dim == 0:
                    forced.append(AdamsDifferentialFlag(
                        r=r,
                        source=(s, t),
                        target=(tgt_s, tgt_t),
                        classification="forced_zero",
                        reason="source or target dim 0",
                        source_dim=src_dim,
                        target_dim=tgt_dim,
                    ))
                else:
                    ambiguous.append(AdamsDifferentialFlag(
                        r=r,
                        source=(s, t),
                        target=(tgt_s, tgt_t),
                        classification="ambiguous",
                        reason="both dims > 0",
                        source_dim=src_dim,
                        target_dim=tgt_dim,
                    ))

        return AdamsE2Page(
            space_label=cohomology_ring.space_label,
            prime=prime,
            s_max=s_max,
            t_max=t_max,
            e2_grid=e2_grid,
            forced_vanishings=forced,
            ambiguous_differentials=ambiguous,
            reliable_window=(s_max, max(0, t_max - s_max)),
            resource_summary={
                "peak_mem_mb": 0.0,  # cobar avoids tracemalloc for now
                "wall_seconds": wall,
            },
            status="success",
            reasoning=(
                f"Odd-prime (p={prime}) Adams E_2 via cobar of A_p^* "
                f"on a trivial A_p-module."
            ),
        )

    tracemalloc.start()
    t_start = time.time()
    status: Literal["success", "truncated", "inconclusive"] = "success"
    reason_parts: List[str] = []
    truncated = False
    try:
        # Build algebra + action
        # We need admissibles up to internal-degree t_max
        algebra = SteenrodAlgebra(prime=prime, max_t=t_max)
        action = SteenrodAction(algebra, cohomology_ring)
        # Sanity: build action_matrix for each generator * Sq^i pair to bake the cache.
        # Build the resolution
        resolution = _MinimalResolution(action, s_max, t_max)
        for s in range(0, s_max + 1):
            # memory check
            _, peak = tracemalloc.get_traced_memory()
            peak_mb = peak / (1024 * 1024)
            if peak_mb > memory_cap_mb:
                warnings.warn(
                    f"adams_e2_page: peak memory {peak_mb:.1f} MB > cap {memory_cap_mb} MB; "
                    f"truncating at s={s}.",
                    stacklevel=2,
                )
                truncated = True
                reason_parts.append(
                    f"truncated at s={s} due to memory cap {memory_cap_mb} MB."
                )
                break
            try:
                resolution.build_F_s(s)
            except AdamsCombinatorialError as e:
                truncated = True
                reason_parts.append(f"combinatorial cap at s={s}: {e}")
                break
            # Early termination if the resolution stabilises (no new gens for 3 consecutive s)
            if s >= 3:
                empty_in_a_row = 0
                for ss in range(s - 2, s + 1):
                    if not resolution.generators[ss]:
                        empty_in_a_row += 1
                if empty_in_a_row == 3:
                    reason_parts.append(
                        f"resolution stabilised at s={s} (3 empty layers)."
                    )
                    break
        # Read off Ext
        ext = resolution.ext_dimensions()
        # Build e2_grid restricted to (s, t) within window
        e2_grid: Dict[Tuple[int, int], int] = {}
        for (s, t), dim in ext.items():
            if 0 <= s <= s_max and 0 <= t <= t_max and dim > 0:
                e2_grid[(s, t)] = dim
        # Always include (0, 0) presence if M_0 has a generator → dim ≥ 1
        # (Handled by ext.)
        # Classify differentials d_r
        forced: List[AdamsDifferentialFlag] = []
        ambiguous: List[AdamsDifferentialFlag] = []
        s_max_reliable = s_max
        # reliable window per s: t ≤ t_max - (s_max - s)
        for r in range(2, s_max + 1):
            for (s, t), source_dim in e2_grid.items():
                target_s = s + r
                target_t = t + r - 1
                if target_s > s_max or target_t > t_max:
                    continue
                # restrict to reliable window
                t_max_reliable_target = t_max - (s_max - target_s)
                if target_t > t_max_reliable_target:
                    continue
                target_dim = e2_grid.get((target_s, target_t), 0)
                if source_dim == 0 or target_dim == 0:
                    forced.append(
                        AdamsDifferentialFlag(
                            r=r,
                            source=(s, t),
                            target=(target_s, target_t),
                            classification="forced_zero",
                            reason=(
                                "target dim = 0" if target_dim == 0
                                else "source dim = 0"
                            ),
                            source_dim=int(source_dim),
                            target_dim=int(target_dim),
                        )
                    )
                else:
                    ambiguous.append(
                        AdamsDifferentialFlag(
                            r=r,
                            source=(s, t),
                            target=(target_s, target_t),
                            classification="ambiguous",
                            reason="both dims > 0",
                            source_dim=int(source_dim),
                            target_dim=int(target_dim),
                        )
                    )
        _, peak = tracemalloc.get_traced_memory()
        peak_mb = peak / (1024 * 1024)
        wall = time.time() - t_start
        if truncated:
            status = "truncated"
        return AdamsE2Page(
            space_label=cohomology_ring.space_label,
            prime=prime,
            s_max=s_max,
            t_max=t_max,
            e2_grid=e2_grid,
            forced_vanishings=forced,
            ambiguous_differentials=ambiguous,
            reliable_window=(s_max_reliable, max(0, t_max - s_max)),
            resource_summary={
                "peak_mem_mb": float(peak_mb),
                "wall_seconds": float(wall),
            },
            status=status,
            reasoning=(
                "Computed via minimal free A-resolution + sparse F_p RREF. "
                + " ".join(reason_parts)
            ).strip(),
        )
    finally:
        tracemalloc.stop()


# ════════════════════════════════════════════════════════════════════════════
# Canned cohomology rings
# ════════════════════════════════════════════════════════════════════════════


def sphere_cohomology_fp(n: int, prime: int = 2) -> FpCohomologyRing:
    """H^*(S^n; F_p) — F_p in degrees 0 and n only."""
    if n < 0:
        raise ValueError("Sphere dimension must be ≥ 0.")
    if prime not in (2, 3, 5):
        raise ValueError(f"prime ∈ {{2,3,5}}; got {prime}.")
    basis = {0: ["1"]}
    cup = {("1", "1"): {"1": 1}}
    sq_table: Dict[Tuple[int, str], Dict[str, int]] = {(0, "1"): {"1": 1}}
    if n == 0:
        # S^0: reduced cohomology trivial; unreduced has F_2 in degree 0.
        return FpCohomologyRing(
            space_label="S^0",
            prime=prime,
            max_degree=0,
            basis=basis,
            cup_table=cup,
            sq_table=sq_table,
            ring_generators=[],
        )
    basis[n] = ["x"]
    cup[("1", "x")] = {"x": 1}
    cup[("x", "1")] = {"x": 1}
    cup[("x", "x")] = {}  # x² = 0 in S^n
    sq_table[(0, "x")] = {"x": 1}
    # Sq^i(x) = 0 for 0 < i ≤ n−1 trivially via instability bound (no place to land)
    # except Sq^n(x) = x² = 0 in S^n.
    for i in range(1, n + 1):
        sq_table[(i, "x")] = {}
    return FpCohomologyRing(
        space_label=f"S^{n}",
        prime=prime,
        max_degree=n,
        basis=basis,
        cup_table=cup,
        sq_table=sq_table,
        ring_generators=["x"],
    )


def cp_n_cohomology_fp(n: int, prime: int = 2) -> FpCohomologyRing:
    """H^*(CP^n; F_p) = F_p[x]/(x^{n+1}) with deg(x) = 2."""
    if n < 0:
        raise ValueError("n must be ≥ 0.")
    if prime not in (2, 3, 5):
        raise ValueError(f"prime ∈ {{2,3,5}}; got {prime}.")
    max_deg = 2 * n
    basis: Dict[int, List[str]] = {0: ["1"]}
    cup: Dict[Tuple[str, str], Dict[str, int]] = {("1", "1"): {"1": 1}}
    for k in range(1, n + 1):
        lbl = f"x{k}"
        basis[2 * k] = [lbl]
    # cup table: x^a · x^b = x^{a+b} if a+b ≤ n else 0
    labels = {0: "1"}
    for k in range(1, n + 1):
        labels[k] = f"x{k}"
    for a in range(0, n + 1):
        for b in range(0, n + 1):
            la = labels[a]
            lb = labels[b]
            if a + b <= n:
                cup[(la, lb)] = {labels[a + b]: 1}
            else:
                cup[(la, lb)] = {}
    # Steenrod action mod 2: Sq^i(x_k) = C(k, i/2) · x_{k + i/2} for i even, else 0.
    # In terms of labels: with x = x1 (deg 2), x_k = x^k.
    # Sq^i(x^k) = sum_{j+...} ... actually with x = x1 of degree 2:
    # Sq^0(x^k) = x^k, Sq^1(x^k) = 0 (since deg 1 doesn't reach), Sq^2(x^k) = k·x^{k+1} mod 2,
    # Sq^{2k}(x^k) = x^{2k} (instability), Sq^i(x^k) = 0 for i odd.
    # Use Cartan: Sq^i(x · x^{k-1}) = Σ_{a+b=i} Sq^a(x)·Sq^b(x^{k-1}).
    # For x = x1: Sq^0(x) = x, Sq^2(x) = x², Sq^i(x) = 0 for i ∉ {0, 2}.
    sq_table: Dict[Tuple[int, str], Dict[str, int]] = {(0, "1"): {"1": 1}}
    if n >= 1:
        sq_table[(0, "x1")] = {"x1": 1}
        if n >= 2:
            sq_table[(2, "x1")] = {"x2": 1}  # x²
        else:
            sq_table[(2, "x1")] = {}  # x² = 0 in CP^1
        for i in range(1, max_deg + 1):
            if i == 2 and n >= 1:
                continue
            if i == 0:
                continue
            sq_table.setdefault((i, "x1"), {})
    # higher x_k actions are derived via Cartan automatically by SteenrodAction;
    # but for stability, precompute them:
    # We'll let SteenrodAction derive them lazily via Cartan.
    return FpCohomologyRing(
        space_label=f"CP^{n}",
        prime=prime,
        max_degree=max_deg,
        basis=basis,
        cup_table=cup,
        sq_table=sq_table,
        ring_generators=["x1"] if n >= 1 else [],
    )


def rp_n_cohomology_fp(n: int, prime: int = 2) -> FpCohomologyRing:
    """H^*(RP^n; F_2) = F_2[x]/(x^{n+1}) with deg(x) = 1."""
    if prime != 2:
        raise NotImplementedError(
            "rp_n_cohomology_fp implemented for prime=2 only (RP^n has no nontrivial mod-p odd cohomology beyond degree 1)."
        )
    if n < 1:
        raise ValueError("n must be ≥ 1.")
    basis: Dict[int, List[str]] = {0: ["1"]}
    labels = {0: "1"}
    for k in range(1, n + 1):
        lbl = f"x{k}"
        basis[k] = [lbl]
        labels[k] = lbl
    cup: Dict[Tuple[str, str], Dict[str, int]] = {}
    for a in range(0, n + 1):
        for b in range(0, n + 1):
            la = labels[a]
            lb = labels[b]
            if a + b <= n:
                cup[(la, lb)] = {labels[a + b]: 1}
            else:
                cup[(la, lb)] = {}
    # Steenrod action: Sq^i(x^k) = C(k, i) x^{i+k} mod 2 (truncated to ≤ n).
    sq_table: Dict[Tuple[int, str], Dict[str, int]] = {(0, "1"): {"1": 1}}
    for k in range(1, n + 1):
        sq_table[(0, labels[k])] = {labels[k]: 1}
        for i in range(1, n + 1):
            target_idx = i + k
            if target_idx > n:
                sq_table[(i, labels[k])] = {}
                continue
            coef = _binom_mod_p(k, i, 2)
            if coef == 0:
                sq_table[(i, labels[k])] = {}
            else:
                sq_table[(i, labels[k])] = {labels[target_idx]: 1}
    return FpCohomologyRing(
        space_label=f"RP^{n}",
        prime=prime,
        max_degree=n,
        basis=basis,
        cup_table=cup,
        sq_table=sq_table,
        ring_generators=["x1"],
    )


def reduce_fp_cohomology_ring(ring: FpCohomologyRing) -> FpCohomologyRing:
    """Return the *reduced* F_p cohomology ring of any FpCohomologyRing.

    What is Being Computed?:
        Strips the degree-0 unit summand F_p · 1 from an FpCohomologyRing,
        returning the ring whose basis is the positive-degree part only.
        Cup-product entries involving "1" are removed (they are identity);
        Steenrod-action entries keyed at i=0 are removed too (P^0 = id).

    Why this matters for Adams:
        Feeding the *un*-reduced ring H^*(X; F_p) = F_p (deg 0) ⊕ H̃^*(X; F_p)
        into adams_e2_page computes the Adams SS of the *disjoint-basepoint*
        suspension X_+ = X ⊔ {*}, which converges to π^s_*(X_+)_p
        = π^s_*(S^0)_p ⊕ π^s_*(X)_p. The extra π^s_*(S^0)_p summand is the
        "ghost S^0" inflating every t-s stem. Feeding the *reduced* ring
        kills that ghost and converges directly to π^s_*(X)_p.

    Algorithm:
        1. Drop basis[0] from the basis dict.
        2. Drop cup_table entries that mention any degree-0 label.
        3. Drop sq_table entries keyed (0, *) — these record P^0 = identity.
        4. Drop ring_generators entries that live in degree 0 (none normally).

    Args:
        ring: An FpCohomologyRing.

    Returns:
        FpCohomologyRing with degree-0 trivial summand stripped. Idempotent.

    Use When:
        - You want the Adams SS to converge to π^s_*(X)_p without a ghost S^0.
        - You want a tighter upper bound on p-primary torsion of π^s_n(X).

    Example:
        >>> ring = sphere_cohomology_fp(2, prime=2)
        >>> red = reduce_fp_cohomology_ring(ring)
        >>> sorted(red.basis.keys())
        [2]
    """
    deg0_labels = set(ring.basis.get(0, ()))
    new_basis = {d: list(labels) for d, labels in ring.basis.items() if d != 0}
    new_cup: Dict[Tuple[str, str], Dict[str, int]] = {}
    for (a, b), val in ring.cup_table.items():
        if a in deg0_labels or b in deg0_labels:
            continue
        new_cup[(a, b)] = dict(val)
    new_sq: Dict[Tuple[int, str], Dict[str, int]] = {}
    for (i, lbl), val in ring.sq_table.items():
        if i == 0:
            continue
        if lbl in deg0_labels:
            continue
        new_sq[(i, lbl)] = dict(val)
    new_gens = [g for g in ring.ring_generators if g not in deg0_labels]
    return FpCohomologyRing(
        space_label=(ring.space_label + " (reduced)") if ring.space_label else "(reduced)",
        prime=ring.prime,
        max_degree=ring.max_degree,
        basis=new_basis,
        degree_of={lbl: d for lbl, d in ring.degree_of.items() if d != 0},
        cup_table=new_cup,
        sq_table=new_sq,
        ring_generators=new_gens,
    )


__all__ = [
    "AdamsCombinatorialError",
    "AdamsDifferentialFlag",
    "AdamsE2Bidegree",
    "AdamsE2Page",
    "AdamsResourceError",
    "AdmissibleSequence",
    "FpCohomologyRing",
    "SteenrodAction",
    "SteenrodAlgebra",
    "SteenrodElement",
    "adams_e2_page",
    "cp_n_cohomology_fp",
    "reduce_fp_cohomology_ring",
    "rp_n_cohomology_fp",
    "sphere_cohomology_fp",
    "steenrod_squares_matrix",
]
