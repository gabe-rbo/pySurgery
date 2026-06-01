"""Free unstable A_p-resolution engine (Massey–Peterson / Quillen).

Math context:
    For an unstable A_p-module M, a free unstable resolution

        ... → F_2 → F_1 → F_0 → M → 0

    by free unstable modules computes

        Ext_U^{s,t}(M, F_p)
            = (number of A_p-generators of F_s at internal degree t),

    when the resolution is taken to be MINIMAL. This is the E_2 page of
    the (Bousfield–Kan) unstable Adams spectral sequence converging to
    π_{t-s}(X)_p whenever H̃*(X; F_p) = M.

    Computationally each F_s is a direct sum of free unstable cells
    A_p · γ_α / (instability), one per A_p-generator γ_α of F_s. The
    boundary d_s : F_s → F_{s-1} is determined by its action on
    generators (A_p-linearity propagates the rest).

    The minimal resolution is built degree-by-degree: at each (s, t)
    we compute the kernel of d_{s-1} : F_{s-1}[t] → F_{s-2}[t], subtract
    the subspace already covered by Sq-propagation of lower-degree
    F_s-generators, and the QUOTIENT supplies the new F_s-generators.

    This is generic — it accepts any FpCohomologyRing as M, requires
    only the basis and the Sq-action (no Λ-algebra or Adem-Curtis
    relations on the dual side). The Steenrod algebra primitives reuse
    `pysurgery.adams.spectral_sequence.SteenrodAlgebra`.

Slice scope:
    This file is Slice 3 of the unstable-Adams plan. Currently the
    p = 2 case is wired (covers S^n, CP^n, RP^n, and any F_2-cohomology
    ring whose A_2-module structure is supplied via FpCohomologyRing).
    Odd-prime extension (p ∈ {3, 5}) is a marked future block: the
    existing SteenrodAlgebra has the odd-prime admissibles, only the
    instability formula and the action need wiring.

Conventions:
    - Admissible Sq^I = Sq^{i_1} Sq^{i_2} … Sq^{i_k} with i_j ≥ 2 i_{j+1}, i_k ≥ 1.
    - Internal degree |Sq^I| = Σ i_j.
    - Excess(I) = 2 i_1 − |I| = i_1 − (i_2 + … + i_k).
    - Instability axiom: Sq^I(x) = 0 if excess(I) > |x|. Equivalently,
      Sq^I · γ at γ of degree d is nonzero in the free unstable A_2-cell
      A_2 · γ iff excess(I) ≤ d.

Public API:
    UnstableResolution(ring, prime, t_max)   — incremental resolution.
    u_resolution_e2_page(ring, prime, …)     — façade producing an AdamsE2Page.

References:
    Massey, W. S. & Peterson, F. P. (1967). The mod-2 cohomology of certain
        fibre spaces. Mem. AMS 74.
    Quillen, D. G. (1969). Rational homotopy theory. Ann. Math. 90, 205-295.
    Schwartz, L. (1994). Unstable Modules over the Steenrod Algebra and
        Sullivan's Fixed Point Set Conjecture. Univ. of Chicago Press.
    Steenrod, N. & Epstein, D. (1962). Cohomology Operations. Princeton.
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Set, Tuple

import numpy as np
import scipy.sparse as sp

from pysurgery.adams.spectral_sequence import (
    AdamsDifferentialFlag,
    AdamsE2Page,
    FpCohomologyRing,
    SteenrodAlgebra,
    _sparse_fp_kernel,
)


# ── Types ────────────────────────────────────────────────────────────────────

# An admissible Sq^I is the tuple form used by SteenrodAlgebra: at p=2,
# (i_1, ..., i_k) with i_j ≥ 2·i_{j+1} and i_k ≥ 1; the empty tuple is Sq^0 = 1.
AdmissibleSeq = Tuple[int, ...]

# A generator of F_s is identified by an integer id. Each carries:
#   id        — unique within F_s.
#   degree    — internal degree of γ_id (its "bottom cell").
#   embedding — image of γ_id under d_s : F_s → F_{s-1}, as a sparse
#               map { (parent_gen_id, admissible_I) → coef in F_p }.
@dataclass
class UGenerator:
    """One A_p-generator of F_s in a free unstable resolution."""

    gid: int
    degree: int
    # Embedding into F_{s-1}; for s = 0 this is the image in M instead.
    # Keyed by (parent_id, admissible_I) → F_p coefficient. For s = 0,
    # `parent_id` is the basis-label index in M (encoded via the
    # `label_index` map kept by UnstableResolution).
    embedding: Dict[Tuple[int, AdmissibleSeq], int] = field(default_factory=dict)


# ── Excess and admissibility helpers ─────────────────────────────────────────


def excess_p2(seq: AdmissibleSeq) -> int:
    """Excess of an admissible Sq-monomial at p = 2.

    excess((i_1, i_2, …, i_k)) = i_1 − (i_2 + i_3 + … + i_k) = 2·i_1 − |I|.

    The empty admissible (Sq^0 = 1) has excess 0.

    Back-compat alias: prefer `_excess_p(seq, prime)` for new code.
    """
    if not seq:
        return 0
    return 2 * seq[0] - sum(seq)


def _excess_p(seq: AdmissibleSeq, prime: int) -> int:
    """Excess of an admissible monomial at any supported prime.

    p = 2: excess((i_1, …, i_k)) = 2·i_1 − Σ i_j.
    p odd: encoded admissible is (e_0, s_1, e_1, …, s_k, e_k) per
        `SteenrodAlgebra._to_admissible_odd` (adams_spectral_sequence.py:291).
        The standard formula (Steenrod-Epstein, Schwartz 1994 §1) is

            excess = 2·s_1 + e_0 − 2(p−1)·(s_2 + … + s_k) − (e_1 + … + e_k).

        Empty tuple (identity) and (0,) both have excess 0.
        (1,) — the standalone β — has excess 1.
    """
    if prime == 2:
        if not seq:
            return 0
        return 2 * seq[0] - sum(seq)
    # Odd prime.
    if not seq:
        return 0
    if len(seq) == 1:
        # Either (0,) = identity (excess 0) or (1,) = β (excess 1).
        return int(seq[0])
    # Standard odd-p admissible: (e_0, s_1, e_1, …, s_k, e_k); length = 1+2k.
    e_0 = int(seq[0])
    s_1 = int(seq[1])
    # Indices into seq: pairs (s_i, e_i) at (2i-1, 2i) for i = 1..k.
    # Higher-index s_i for i ≥ 2: seq[3], seq[5], … i.e. seq[3::2].
    # All e_i for i ≥ 1: seq[2], seq[4], … i.e. seq[2::2].
    sum_s_higher = sum(int(x) for x in seq[3::2])
    sum_e_post_first = sum(int(x) for x in seq[2::2])
    return 2 * s_1 + e_0 - 2 * (prime - 1) * sum_s_higher - sum_e_post_first


# ── Apply a Steenrod operation to an element of M ────────────────────────────


def _apply_op_to_label(
    A: SteenrodAlgebra,
    ring: FpCohomologyRing,
    op: AdmissibleSeq,
    label: str,
) -> Dict[str, int]:
    """Apply an admissible Steenrod operation to a single basis label of M.

    At p = 2: `op` is (i_1, …, i_k) and we apply Sq^{i_1} ∘ … ∘ Sq^{i_k}.
    At odd p: `op` is the encoded admissible (e_0, s_1, e_1, …, s_k, e_k) per
        `SteenrodAlgebra._to_admissible_odd`. We apply
            β^{e_0} ∘ P^{s_1} ∘ β^{e_1} ∘ P^{s_2} ∘ … ∘ P^{s_k} ∘ β^{e_k}
        by walking the tuple right-to-left.

    `ring.sq_table[(i, label)]` provides the P^i action at odd p (integer
    indices = power, exactly as at p = 2 for Sq^i). The Bockstein (β) is
    looked up at `ring.sq_table[(-1, label)]`; if absent, β is assumed
    zero on `label` (true for S^n, CP^n, RP^n at odd primes, where the
    relevant degrees miss the Bockstein-target classes).
    """
    state: Dict[str, int] = {label: 1}
    if not op:
        return state
    prime = A.prime
    if prime == 2:
        # Walk Sq^{i_k}, …, Sq^{i_1} right-to-left.
        for i in reversed(op):
            if i == 0:
                continue
            new_state: Dict[str, int] = {}
            for lbl, c in state.items():
                row = ring.sq_table.get((int(i), lbl), {})
                for tgt, coef in row.items():
                    v = (new_state.get(tgt, 0) + c * coef) % prime
                    if v:
                        new_state[tgt] = v
                    else:
                        new_state.pop(tgt, None)
            state = new_state
            if not state:
                return {}
        return state
    # Odd-prime branch. op = (e_0, s_1, e_1, …, s_k, e_k).
    # Special case: length 1 means just β^{e_0} or identity.
    if len(op) == 1:
        if int(op[0]) == 0:
            return state
        return _apply_beta(ring, state, prime)
    # General case. Process pairs (s_i, e_i) for i = k, k-1, …, 1, then
    # apply β^{e_0}. Encoded admissible has e_0 at index 0, then
    # (s_1, e_1, s_2, e_2, …, s_k, e_k) at indices 1, 2, 3, … (length = 1 + 2k).
    e_0 = int(op[0])
    # k = (len(op) − 1) / 2.
    k = (len(op) - 1) // 2
    for i in range(k, 0, -1):
        s_i = int(op[2 * i - 1])
        e_i = int(op[2 * i])
        # Apply β^{e_i} on the right side of P^{s_i}.
        if e_i:
            state = _apply_beta(ring, state, prime)
            if not state:
                return {}
        # Apply P^{s_i}.
        if s_i > 0:
            new_state: Dict[str, int] = {}
            for lbl, c in state.items():
                row = ring.sq_table.get((int(s_i), lbl), {})
                for tgt, coef in row.items():
                    v = (new_state.get(tgt, 0) + c * coef) % prime
                    if v:
                        new_state[tgt] = v
                    else:
                        new_state.pop(tgt, None)
            state = new_state
            if not state:
                return {}
    # Finally apply β^{e_0}.
    if e_0:
        state = _apply_beta(ring, state, prime)
    return state


def _apply_beta(
    ring: FpCohomologyRing,
    state: Dict[str, int],
    prime: int,
) -> Dict[str, int]:
    """Apply the Bockstein β to a sparse F_p-linear combination of labels.

    Looks up `ring.sq_table[(-1, label)]` for each label. If absent, β
    is treated as zero on that label (the default for spaces whose
    Bockstein is structurally zero on the chosen basis: S^n, CP^n, RP^n
    at odd primes all satisfy this).
    """
    new_state: Dict[str, int] = {}
    for lbl, c in state.items():
        row = ring.sq_table.get((-1, lbl), {})
        for tgt, coef in row.items():
            v = (new_state.get(tgt, 0) + c * coef) % prime
            if v:
                new_state[tgt] = v
            else:
                new_state.pop(tgt, None)
    return new_state


# Back-compat alias for callers using the old name (p = 2 only).
def _apply_sq_to_label(
    A: SteenrodAlgebra,
    ring: FpCohomologyRing,
    op: AdmissibleSeq,
    label: str,
) -> Dict[str, int]:
    """Back-compat alias for `_apply_op_to_label`.

    Some external callers may still import this name; the body now
    delegates to the generic implementation. Raises if `A.prime != 2`
    so the old "p=2 only" contract is preserved at the call site.
    """
    if A.prime != 2:
        raise NotImplementedError(
            "_apply_sq_to_label is the p=2 alias; use _apply_op_to_label "
            "for odd primes."
        )
    return _apply_op_to_label(A, ring, op, label)


# ── The minimal free unstable resolution ─────────────────────────────────────


class UnstableResolution:
    """Build a minimal free unstable A_p-resolution of an FpCohomologyRing.

    Algorithm (p = 2; odd p marked for future):
        1. Read M from `ring`. Each basis label is a vector-space basis
           element with its Sq-action specified in `ring.sq_table`.
        2. F_0: enumerate A_p-generators of M. A label γ at degree d is
           an A_p-generator iff γ is not in the image of Sq^I · (label of
           lower degree). This is determined greedily by walking M from
           the bottom up.
        3. For s ≥ 1: at each internal degree t, compute the kernel of
           d_{s-1} : F_{s-1}[t] → F_{s-2}[t] as a sparse F_2-subspace.
           Subtract the subspace already covered by Sq-propagation of
           lower-degree F_s-generators. The quotient supplies the new
           F_s-generators at (s, t).
        4. Record each generator's embedding so that d_s is fully
           determined for the next round.

    The resolution is computed up to a user-specified (s_max, t_max)
    truncation window.

    Public attributes after build():
        F: List[List[UGenerator]]      — F[s] is the generator list of F_s.
        label_index: Dict[str, int]    — stable index for each M-label.
    """

    def __init__(self, ring: FpCohomologyRing, prime: int, t_max: int):
        if prime not in (2, 3, 5):
            raise ValueError(
                f"UnstableResolution: prime must be in {{2, 3, 5}}; got {prime}."
            )
        self.ring = ring
        self.prime = prime
        self.t_max = t_max
        self.A = SteenrodAlgebra(prime=prime, max_t=t_max + 4)
        self._adem_pair_cache: Dict[Tuple[AdmissibleSeq, AdmissibleSeq], Dict[AdmissibleSeq, int]] = {}
        # Stable index of each label in M (for F_0 embedding keys).
        labels: List[Tuple[int, str]] = sorted(
            ((d, lbl) for d, labs in ring.basis.items() for lbl in labs),
            key=lambda x: (x[0], x[1]),
        )
        self.label_index: Dict[str, int] = {lbl: i for i, (_, lbl) in enumerate(labels)}
        self.label_at: Dict[int, Tuple[int, str]] = {
            i: (d, lbl) for i, (d, lbl) in enumerate(labels)
        }
        # The resolution itself: F[s] is the list of F_s-generators, in
        # creation order. Each UGenerator carries its embedding.
        self.F: List[List[UGenerator]] = []
        # gid counter PER s.
        self._next_gid: List[int] = []

    # ── F_0: A_p-generators of M ─────────────────────────────────────────────

    def _build_f0(self) -> None:
        """Determine the A_p-generators of M by walking degrees bottom-up."""
        gens: List[UGenerator] = []
        # span_at[d] : dict from M-label at degree d → F_2-coef, accumulated
        # as the span of (Sq-action on already-chosen generators) at degree d.
        span_at: Dict[int, List[Dict[str, int]]] = {}
        for d in sorted(self.ring.basis.keys()):
            labels = self.ring.basis.get(d, [])
            if not labels:
                continue
            # Vectors at degree d that are already covered by Sq actions on
            # earlier generators.
            covered = span_at.get(d, [])
            # Reduce: pivot indices in the labels.
            covered_pivots: Set[str] = set()
            for vec in covered:
                if not vec:
                    continue
                # Identify a leading label.
                lbl = min(vec.keys())
                covered_pivots.add(lbl)
            # Any label not already a pivot is a new A_2-generator of M.
            for lbl in labels:
                if lbl in covered_pivots:
                    continue
                gid = len(gens)
                # Embedding: γ_α ↦ 1 · label_α in M (using label-index encoding).
                emb = {(self.label_index[lbl], ()): 1}
                gens.append(UGenerator(gid=gid, degree=d, embedding=emb))
                # Propagate Sq actions on this new generator to all higher
                # degrees ≤ t_max.
                for j in range(1, self.t_max - d + 1):
                    # admissibles of degree j with excess ≤ d
                    for op in self._admissibles_excess_le(j, d):
                        result = _apply_op_to_label(self.A, self.ring, op, lbl)
                        if not result:
                            continue
                        tgt_d = d + j
                        span_at.setdefault(tgt_d, []).append(result)
        self.F.append(gens)
        self._next_gid.append(len(gens))

    # ── Admissible Sq^I enumeration with excess filter ───────────────────────

    def _admissibles_excess_le(self, t: int, max_excess: int) -> List[AdmissibleSeq]:
        """All admissible Sq^I with |I| = t and excess(I) ≤ max_excess.

        The empty admissible (excess 0) is included if t == 0.
        """
        if t == 0:
            return [()] if max_excess >= 0 else []
        out: List[AdmissibleSeq] = []
        for op in self.A.admissible_basis(t):
            if _excess_p(op, self.prime) <= max_excess:
                out.append(op)
        return out

    # ── Sq^I · Sq^J (Adem product), cached ───────────────────────────────────

    def _adem_product(
        self, seq_I: AdmissibleSeq, J: AdmissibleSeq
    ) -> Dict[AdmissibleSeq, int]:
        """Reduce Sq^I · Sq^J to a sum of admissibles, cached."""
        if not seq_I:
            return {J: 1}
        if not J:
            return {seq_I: 1}
        key = (seq_I, J)
        if key in self._adem_pair_cache:
            return self._adem_pair_cache[key]
        # Prime-aware composition: at p=2 this is raw concat → to_admissible;
        # at odd p, concat_admissibles merges the trailing/leading β-bits
        # (β·β = 0 kills the term) and then Adem-reduces.
        result = self.A.concat_admissibles(seq_I, J)
        # Drop any non-admissibles or zero coefs (defensive).
        clean: Dict[AdmissibleSeq, int] = {}
        for seq, c in result.items():
            cc = c % self.prime
            if cc:
                clean[seq] = cc
        self._adem_pair_cache[key] = clean
        return clean

    # ── F_s basis at internal degree t ───────────────────────────────────────

    def _basis_at(self, s: int, t: int) -> List[Tuple[int, AdmissibleSeq]]:
        """Basis of F_s at internal degree t.

        Returns the list of (gid, admissible_I) pairs with
            deg(γ_gid) + |I| = t   and   excess(I) ≤ deg(γ_gid).

        gid runs over generators of F_s (must already exist), I is in
        admissible Sq-form (empty tuple is Sq^0).
        """
        if s >= len(self.F):
            return []
        out: List[Tuple[int, AdmissibleSeq]] = []
        for g in self.F[s]:
            if g.degree > t:
                continue
            j = t - g.degree
            for op in self._admissibles_excess_le(j, g.degree):
                out.append((g.gid, op))
        return out

    # ── Boundary d_s : F_s[t] → F_{s-1}[t] as sparse F_p matrix ──────────────

    def _boundary_matrix(self, s: int, t: int) -> Tuple[
        sp.csr_matrix,
        List[Tuple[int, AdmissibleSeq]],   # column basis = F_s[t]
        List[Tuple[int, AdmissibleSeq]],   # row basis    = F_{s-1}[t]
    ]:
        """Build d_s : F_s[t] → F_{s-1}[t] as a sparse matrix mod p.

        Each F_s basis element Sq^I · γ_α maps to Sq^I · embedding(γ_α),
        with the result expanded in the F_{s-1}[t] basis via Adem
        reductions.
        """
        col_basis = self._basis_at(s, t)
        # For s = 0 the target is M, not F_{-1}. Build the M-basis at degree t
        # using the (label_index, Sq^0) encoding _d_on uses.
        if s >= 1:
            row_basis = self._basis_at(s - 1, t)
        else:
            row_basis = [
                (self.label_index[lbl], ())
                for lbl in self.ring.basis.get(t, [])
            ]
        row_index = {key: i for i, key in enumerate(row_basis)}
        rows: List[int] = []
        cols: List[int] = []
        data: List[int] = []
        for j, (gid, seq_I) in enumerate(col_basis):
            image = self._d_on(s, gid, seq_I)
            for key, coef in image.items():
                if coef == 0:
                    continue
                ri = row_index.get(key)
                if ri is None:
                    continue
                rows.append(ri)
                cols.append(j)
                data.append(int(coef))
        if not rows:
            return (
                sp.csr_matrix(
                    (len(row_basis), len(col_basis)), dtype=np.int8
                ),
                col_basis,
                row_basis,
            )
        mat = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(len(row_basis), len(col_basis)),
            dtype=np.int8,
        )
        # Reduce mod p (collapse duplicates).
        mat.sum_duplicates()
        mat.data = (mat.data % self.prime).astype(np.int8)
        mat.eliminate_zeros()
        return mat, col_basis, row_basis

    # ── d_s on a SINGLE element Sq^I · γ_α (used during F_{s+1} build) ───────

    def _d_on(
        self, s: int, gid: int, seq_I: AdmissibleSeq
    ) -> Dict[Tuple[int, AdmissibleSeq], int]:
        """Image of Sq^I · γ_α under d_s : F_s → F_{s-1}.

        For s = 0 the image lives in M and is keyed by (label_index, Sq^0).
        Returns a sparse dict.
        """
        if s < 0 or s >= len(self.F):
            return {}
        g = next((gg for gg in self.F[s] if gg.gid == gid), None)
        if g is None:
            return {}
        out: Dict[Tuple[int, AdmissibleSeq], int] = {}
        for (parent_id, J), c in g.embedding.items():
            if c == 0:
                continue
            product = self._adem_product(seq_I, J)
            for K, ck in product.items():
                coef = (c * ck) % self.prime
                if coef == 0:
                    continue
                # Excess filter at the target.
                if s == 0:
                    d_target = self.label_at[parent_id][0]
                    if K and _excess_p(K, self.prime) > d_target:
                        continue
                    # In M, Sq^K · label is the label sum at degree
                    # deg(label) + |K|.
                    label = self.label_at[parent_id][1]
                    img = _apply_op_to_label(self.A, self.ring, K, label)
                    for tgt_label, ic in img.items():
                        ti = self.label_index.get(tgt_label)
                        if ti is None:
                            continue
                        key = (ti, ())
                        cur = out.get(key, 0)
                        out[key] = (cur + coef * ic) % self.prime
                else:
                    parent_gen = next(
                        (gg for gg in self.F[s - 1] if gg.gid == parent_id),
                        None,
                    )
                    if parent_gen is None:
                        continue
                    if K and _excess_p(K, self.prime) > parent_gen.degree:
                        continue
                    key = (parent_id, K)
                    cur = out.get(key, 0)
                    out[key] = (cur + coef) % self.prime
        # Drop zeros.
        return {k: v for k, v in out.items() if v}

    # ── Find new F_s generators at degree t ──────────────────────────────────

    def _find_new_generators_at(
        self, s: int, t: int
    ) -> List[Dict[Tuple[int, AdmissibleSeq], int]]:
        """At F_s[t]: find A_p-indecomposable kernel elements of d_{s-1}.

        Returns a list of "new" generators (each given by its embedding
        in F_{s-1}). The number of these is exactly dim Ext^{s, t}.
        """
        if s == 0:
            # F_0 generators are computed greedily in _build_f0; this
            # method is for s ≥ 1.
            return []
        # 1. Build d_{s-1} : F_{s-1}[t] → F_{s-2}[t] (or → M when s-1 = 0).
        mat, col_basis, row_basis = self._boundary_matrix(s - 1, t)
        if not col_basis:
            return []
        # 2. Compute the kernel as a sub-vector-space of F_{s-1}[t].
        kernel_vecs = _sparse_fp_kernel(mat, self.prime)
        if not kernel_vecs:
            return []
        # 3. Compute the subspace already covered by Sq-propagation of
        # existing F_s-generators of degree < t.
        if s < len(self.F):
            existing = self.F[s]
        else:
            existing = []
        propagated: List[Dict[int, int]] = []
        col_index = {key: j for j, key in enumerate(col_basis)}
        for g in existing:
            if g.degree >= t:
                continue
            j = t - g.degree
            for op in self._admissibles_excess_le(j, g.degree):
                # The vector in F_{s-1}[t] corresponding to d_s(Sq^op γ_g)
                # is given by _d_on(s, g.gid, op), but we want the vector
                # WITHIN F_{s-1}[t] viewed as a column. The kernel basis is
                # already in F_{s-1}[t] column space. The "propagation"
                # subspace is generated by d_s(Sq^op γ_g) viewed in F_{s-1}[t].
                image = self._d_on(s, g.gid, op)
                vec: Dict[int, int] = {}
                for key, c in image.items():
                    ji = col_index.get(key)
                    if ji is None:
                        continue
                    vec[ji] = (vec.get(ji, 0) + c) % self.prime
                if vec:
                    propagated.append(vec)
        # 4. The new generators correspond to a basis of (kernel) modulo
        # (propagated). Compute via a joint RREF: stack propagated then
        # kernel rows; the rows that introduce new pivots beyond the
        # propagated rows are the "new" generators.
        new_gen_vecs: List[Dict[int, int]] = []
        # rref with priority on propagated.
        rows_list: List[Dict[int, int]] = [dict(v) for v in propagated] + [
            dict(v) for v in kernel_vecs
        ]
        n_prop = len(propagated)
        pivots_used: Dict[int, int] = {}  # pivot col → row index
        for ri, row in enumerate(rows_list):
            r = dict(row)
            # Reduce by existing pivots.
            for pc in sorted(pivots_used):
                if pc in r:
                    coef = r[pc]
                    pr = pivots_used[pc]
                    piv_row = rows_list[pr]
                    # r -= coef * piv_row  (mod p; for p=2 this is just XOR)
                    for k, v in piv_row.items():
                        nv = (r.get(k, 0) - coef * v) % self.prime
                        if nv == 0:
                            r.pop(k, None)
                        else:
                            r[k] = nv
            if not r:
                continue
            # Choose leading column.
            pivot_col = min(r.keys())
            # Normalize.
            piv_val = r[pivot_col]
            if piv_val != 1:
                inv = pow(piv_val, self.prime - 2, self.prime)
                r = {k: (v * inv) % self.prime for k, v in r.items()}
            rows_list[ri] = r
            pivots_used[pivot_col] = ri
            # If this row came from kernel (ri >= n_prop), it adds a new
            # F_s generator.
            if ri >= n_prop:
                # The new generator's embedding in F_{s-1} is given by
                # interpreting `r` (column-indexed) in terms of col_basis.
                emb: Dict[Tuple[int, AdmissibleSeq], int] = {}
                for ji, c in r.items():
                    emb[col_basis[ji]] = c
                new_gen_vecs.append(emb)
        return new_gen_vecs

    # ── Driver: build the resolution up to (s_max, t_max) ───────────────────

    def build(self, s_max: int) -> None:
        """Compute F_0 through F_{s_max} up to internal degree t_max."""
        if not self.F:
            self._build_f0()
        for s in range(1, s_max + 1):
            if s >= len(self.F):
                self.F.append([])
                self._next_gid.append(0)
            for t in range(0, self.t_max + 1):
                new_embs = self._find_new_generators_at(s, t)
                for emb in new_embs:
                    gid = self._next_gid[s]
                    self._next_gid[s] += 1
                    self.F[s].append(UGenerator(gid=gid, degree=t, embedding=emb))

    # ── Read-out: Ext dimensions ─────────────────────────────────────────────

    def ext_grid(self, s_max: int) -> Dict[Tuple[int, int], int]:
        """Dimensions of the minimal resolution = Ext^{s,t}_U(M, F_p)."""
        out: Dict[Tuple[int, int], int] = {}
        for s in range(min(s_max + 1, len(self.F))):
            for g in self.F[s]:
                key = (s, g.degree)
                out[key] = out.get(key, 0) + 1
        return out


# ── Public façade ───────────────────────────────────────────────────────────


def u_resolution_e2_page(
    fp_ring: FpCohomologyRing,
    prime: int,
    *,
    s_max: int = 6,
    t_max: int = 20,
    backend: Literal["auto", "python", "numba", "julia"] = "auto",
) -> AdamsE2Page:
    """Unstable Adams E_2 via a minimal free unstable A_p-resolution.

    What is being computed:
        E_2^{s,t}(X) = Ext_U^{s,t}(H̃*(X; F_p), F_p),

        where U is the category of unstable A_p-modules. The minimal
        resolution dimensions give Ext directly.

    Algorithm:
        1. Build M from `fp_ring`.
        2. Determine A_p-generators of M (F_0 generators) bottom-up.
        3. For s = 1, …, s_max: at each internal degree t, find new F_s
           generators as A_p-indecomposable kernel elements of d_{s-1}.
        4. Ext^{s, t} = number of F_s-generators at degree t.

    Backend dispatch (Slice 4 will wire Julia/Numba):
        Currently uses the python path; backend='auto'|'python' both
        accepted. 'numba'|'julia' raise to signal not-yet-wired.

    Args:
        fp_ring: input cohomology ring. Use the REDUCED ring for a
            tight Ext_U; the un-reduced ring includes the unit and
            yields the disjoint-basepoint X_+ Ext (which contains the
            ghost S^0 cell).
        prime: 2 (Slice 3). Odd primes pending.
        s_max: maximum homological degree s in the truncation window.
        t_max: maximum internal degree t in the truncation window.
        backend: see above.

    Returns:
        AdamsE2Page with status='success' and a `space_label` suffix
        '(U-resolution)' marking the engine.
    """
    if prime not in (2, 3, 5):
        raise ValueError(
            f"u_resolution_e2_page: prime must be in {{2, 3, 5}}; got {prime}."
        )
    if backend in ("numba", "julia"):
        raise NotImplementedError(
            f"u_resolution_e2_page: backend={backend!r} not yet wired in Slice 3; "
            "use backend='auto' or 'python'."
        )

    t_start = _time.time()
    res = UnstableResolution(ring=fp_ring, prime=prime, t_max=t_max)
    res.build(s_max=s_max)
    grid_raw = res.ext_grid(s_max=s_max)
    wall = _time.time() - t_start

    # Project to (s ≤ s_max, t ≤ t_max).
    grid: Dict[Tuple[int, int], int] = {
        (s, t): d for (s, t), d in grid_raw.items() if s <= s_max and t <= t_max
    }

    # Differential flags for downstream consumers (E_∞ enumerator etc.).
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

    n_gens = sum(len(res.F[s]) for s in range(len(res.F)))
    return AdamsE2Page(
        space_label=fp_ring.space_label + " (U-resolution)",
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
            "Unstable Adams E_2 via minimal free unstable A_p-resolution "
            "(Massey–Peterson / Quillen). Computed up to "
            f"(s_max, t_max) = ({s_max}, {t_max}). "
            f"Resolution has {n_gens} generators across {len(res.F)} stages. "
            "Rigorous (no excess-filter approximation)."
        ),
    )


__all__ = [
    "AdmissibleSeq",
    "UGenerator",
    "UnstableResolution",
    "u_resolution_e2_page",
    "excess_p2",
    "_excess_p",
    "_apply_op_to_label",
    "_apply_beta",
]
