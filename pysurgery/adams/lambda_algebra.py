"""Λ-algebra (Bousfield–Curtis–Kan) unstable Adams E₂ engine.

Math context:
    The Λ-algebra at p = 2 (Curtis 1971; Bousfield–Curtis–Kan–Quillen–
    Rector–Schlesinger 1966) is the differential graded algebra

        Λ_2 = F_2 ⟨λ_i : i ≥ 0⟩  modulo Adem relations,

    bigraded by (homological degree s, internal degree t = Σ(i_k + 1)),
    with Curtis differential d: Λ_s → Λ_{s+1} satisfying d² = 0. Its
    cohomology computes

        H^{s,t}(Λ_2, d) ≅ Ext_{A_2}^{s,t}(F_2, F_2),

    and an excess filter on the leading index recovers Ext in the
    unstable category U. For a connective unstable A_2-module M with
    bottom generator in degree n_min, the filtered subcomplex of Λ_2
    with admissibles satisfying i_1 ≤ n_min computes

        Ext_U^{s,t}(M, F_2)  ≅  E_2^{s,t} of the unstable Adams SS of X
                                  whenever H̃*(X; F_2) ≅ M.

    This engine produces an `AdamsE2Page` consumable by all existing
    downstream code (E_∞ enumerator, tetrahedron experiment).

Conventions (Curtis 1971; matches Tangora 1990 tables):
    - Generators λ_i for i ≥ 0 at p = 2; bidegree(λ_i) = (1, i + 1).
    - Adem–Curtis relation, applicable to pairs with 2·i < j:

        λ_i · λ_j  =  Σ_{k ≥ 0}  C(j − 1 − k, i − 2k) · λ_{i+j−k} · λ_k   (mod 2)

    - Admissibility: λ_{i_1} … λ_{i_s} is admissible iff
          2 · i_k ≥ i_{k+1}      for all k = 1, …, s − 1.
      (Equivalently, no further Adem rewrite applies.)

    - Curtis differential on a single generator (n ≥ 1):

        d(λ_n)  =  Σ_{j ≥ 1}  C(n − j, j) · λ_{n−j} · λ_{j−1}   (mod 2)

      and d(λ_0) = 0. On products: d is a derivation,

        d(λ_a · m) = d(λ_a) · m + λ_a · d(m)        (no signs at p = 2).

    - Excess of an admissible monomial: e(λ_{i_1} … λ_{i_s}) = i_1.

    - Verification anchors:
        Ext^{1, t} at p = 2 is one-dimensional iff t = 2^k for k ≥ 0
        (representing h_k = λ_{2^k − 1}); zero otherwise.

Public API:
    LambdaAlgebra(prime)       — admissibles, Adem rewriter, differential.
    lambda_e2_page(ring, …)    — generic façade producing an AdamsE2Page.

Slice scope:
    This file implements p = 2 (Slice 1 of the unstable-Adams plan). The
    odd-prime extension (Slice 2) adds μ-generators (Bockstein) and
    refines the Adem coefficients with Wellington 1982 §2; that work
    inserts at the marked extension points below without changing the
    p = 2 code paths.

References:
    Bousfield, A. K., Curtis, E. B., Kan, D. M., Quillen, D. G., Rector,
        D. L., & Schlesinger, J. W. (1966). The mod-p lower central
        series and the Adams spectral sequence. Topology 5, 331-342.
    Curtis, E. B. (1971). Simplicial homotopy theory. Adv. Math. 6, 107-209.
    Tangora, M. C. (1990). Computing the homology of the Lambda algebra.
        Mem. Amer. Math. Soc. 337.
    Wellington, R. (1982). The unstable Adams spectral sequence for
        free simplicial abelian groups. Trans. Amer. Math. Soc. 270, 479-504.
"""
from __future__ import annotations

import time as _time
from functools import lru_cache
from typing import Dict, List, Literal, Tuple

import numpy as np
import scipy.sparse as sp

from pysurgery.adams.spectral_sequence import (
    AdamsDifferentialFlag,
    AdamsE2Page,
    FpCohomologyRing,
    _binom_mod_p,
)


# ── Types ────────────────────────────────────────────────────────────────────

# A Λ-monomial at p=2 is a tuple of non-negative ints (i_1, ..., i_s).
# The empty tuple is the unit (in Λ_0).
LambdaMonomial = Tuple[int, ...]

# A sparse element of Λ at p=2 is a mapping admissible-monomial → coef mod p.
LambdaElement = Dict[LambdaMonomial, int]


# ── Helpers on monomials ─────────────────────────────────────────────────────


def lambda_degree_p2(mon: LambdaMonomial) -> int:
    """Internal degree |λ_{i_1} … λ_{i_s}|_Λ = Σ (i_k + 1) at p = 2."""
    return sum(i + 1 for i in mon)


def is_admissible_p2(mon: LambdaMonomial) -> bool:
    """Admissibility at p = 2: 2·i_k ≥ i_{k+1} for all k.

    A length-≤-1 monomial is always admissible.
    """
    for k in range(len(mon) - 1):
        if 2 * mon[k] < mon[k + 1]:
            return False
    return True


def excess_p2(mon: LambdaMonomial) -> int:
    """Excess of an admissible Λ-monomial at p = 2 is its leading index.

    For the empty (unit) monomial we return 0 (the unit has trivial excess
    and is preserved under any instability filter).
    """
    return mon[0] if mon else 0


# ── Adem rewriting (p = 2) ───────────────────────────────────────────────────


def _adem_pair_p2(i: int, j: int) -> LambdaElement:
    """Adem rewrite of a length-2 non-admissible monomial λ_i λ_j at p = 2.

    Precondition: 2·i < j (non-admissible). For an admissible pair this
    returns the monomial unchanged (identity rewrite).
    """
    if 2 * i >= j:
        return {(i, j): 1}
    out: LambdaElement = {}
    # Σ_{k ≥ 0} C(j - 1 - k, i - 2k) · λ_{i+j-k} · λ_k
    # The sum is supported where i - 2k ≥ 0 and j - 1 - k ≥ i - 2k, i.e.
    # k ≤ i/2 and k ≤ j - 1 - i + 2k → trivially satisfied. We just iterate
    # over k in [0, i//2] and skip zeros from the binomial.
    for k in range(0, i // 2 + 1):
        c = _binom_mod_p(j - 1 - k, i - 2 * k, 2)
        if c == 0:
            continue
        new_mon = (i + j - k, k)
        # The output of a single Adem rewrite at p=2 is always admissible
        # (by Curtis's lemma: 2·(i+j-k) ≥ k for k ≤ i/2, equivalent to
        # 2(i+j) ≥ 3k, which holds since k ≤ i/2 and j ≥ 2i+1).
        out[new_mon] = (out.get(new_mon, 0) + c) % 2
    return {m: c for m, c in out.items() if c}


def adem_rewrite_p2(mon: LambdaMonomial) -> LambdaElement:
    """Reduce a Λ-monomial to admissible normal form via Adem relations (p=2).

    Algorithm:
        Scan left-to-right for the first non-admissible pair (i_k, i_{k+1}).
        Apply the Adem relation at that position, producing a sum of new
        monomials. Recursively reduce each summand. Termination is
        guaranteed by Curtis's lexicographic ordering argument.

    Args:
        mon: a Λ-monomial (tuple of indices ≥ 0).

    Returns:
        A sparse element of Λ_s (s = len(mon)) supported on admissibles,
        coefficients mod 2.
    """
    return _adem_rewrite_p2_memo(mon)


@lru_cache(maxsize=200_000)
def _adem_rewrite_p2_memo(mon: LambdaMonomial) -> LambdaElement:
    if len(mon) <= 1:
        return {mon: 1}
    # Find first non-admissible position.
    bad = -1
    for k in range(len(mon) - 1):
        if 2 * mon[k] < mon[k + 1]:
            bad = k
            break
    if bad == -1:
        return {mon: 1}
    prefix = mon[:bad]
    suffix = mon[bad + 2:]
    pair = _adem_pair_p2(mon[bad], mon[bad + 1])
    out: LambdaElement = {}
    for (a, b), c in pair.items():
        new_mon = prefix + (a, b) + suffix
        # Recursively reduce (the prefix may now have a non-admissibility
        # with the new 'a', or the new 'b' with the suffix start).
        red = _adem_rewrite_p2_memo(new_mon)
        for m, d in red.items():
            cd = (c * d) % 2
            if cd == 0:
                continue
            out[m] = (out.get(m, 0) + cd) % 2
    # Drop zero coefficients.
    return {m: c for m, c in out.items() if c}


# ── Curtis differential (p = 2) ──────────────────────────────────────────────


@lru_cache(maxsize=10_000)
def _d_lambda_single_p2(n: int) -> LambdaElement:
    """The Curtis differential applied to a single generator λ_n at p = 2.

        d(λ_n) = Σ_{j ≥ 1} C(n - j, j) · λ_{n-j} · λ_{j-1}   (mod 2).

    The result is reduced to admissible form.
    """
    if n <= 0:
        return {}
    raw: LambdaElement = {}
    for j in range(1, n + 1):
        c = _binom_mod_p(n - j, j, 2)
        if c == 0:
            continue
        mono = (n - j, j - 1)
        raw[mono] = (raw.get(mono, 0) + c) % 2
    # Reduce.
    out: LambdaElement = {}
    for m, c in raw.items():
        if c == 0:
            continue
        for am, ac in adem_rewrite_p2(m).items():
            v = (out.get(am, 0) + c * ac) % 2
            if v:
                out[am] = v
            else:
                out.pop(am, None)
    return out


def lambda_differential_p2(mon: LambdaMonomial) -> LambdaElement:
    """Curtis differential on a Λ-monomial at p = 2, reduced to admissibles.

    Acts as a derivation:
        d(λ_{i_1} … λ_{i_s})
            = Σ_{k=1..s}  λ_{i_1} … λ_{i_{k-1}} · d(λ_{i_k}) · λ_{i_{k+1}} … λ_{i_s}

    (no signs at p = 2). Each summand is then Adem-reduced.

    Args:
        mon: a Λ-monomial (admissible or not).

    Returns:
        Sparse element of Λ_{s+1} supported on admissibles, mod 2.
    """
    if not mon:
        return {}
    out: LambdaElement = {}
    for k in range(len(mon)):
        d_lk = _d_lambda_single_p2(mon[k])
        if not d_lk:
            continue
        prefix = mon[:k]
        suffix = mon[k + 1:]
        for (a, b), c in d_lk.items():
            raw = prefix + (a, b) + suffix
            for am, ac in adem_rewrite_p2(raw).items():
                v = (out.get(am, 0) + c * ac) % 2
                if v:
                    out[am] = v
                else:
                    out.pop(am, None)
    return out


# ── Basis enumeration (excess-filtered, internal degree t) ──────────────────


def enumerate_admissibles_p2(
    s: int,
    t: int,
    max_excess: int,
    *,
    hard_cap: int = 1 << 18,
) -> List[LambdaMonomial]:
    """Enumerate admissible Λ-monomials of length s, internal degree t, excess ≤ max_excess at p=2.

    Returned monomials are in lex order on (i_1, i_2, …, i_s) descending
    by i_1 then i_2 etc. (canonical ordering matches Curtis/Tangora).

    Args:
        s: homological degree (length).
        t: internal degree (Σ(i_k + 1) = t).
        max_excess: cap on the leading index i_1.
        hard_cap: safety bound; raises if basis would exceed this.

    Returns:
        List of admissibles. Empty if s == 0 and t > 0, or s > 0 and t < s.
    """
    if s < 0 or t < 0:
        return []
    if s == 0:
        return [()] if t == 0 else []
    # |λ_{i_1}…λ_{i_s}| = Σ(i_k + 1) = (Σ i_k) + s ⇒ Σ i_k = t - s.
    # i_k ≥ 0, so feasibility requires t ≥ s.
    if t < s:
        return []
    # We enumerate by choosing i_1 ∈ [0, min(max_excess, t - s)] then
    # recursing with the admissibility constraint i_2 ≤ 2·i_1 and the
    # tail-degree constraint Σ_{k ≥ 2} i_k = t - s - i_1.
    out: List[LambdaMonomial] = []

    def _recurse(prev_i: int, remaining_t: int, remaining_s: int, acc: List[int]) -> None:
        if remaining_s == 0:
            if remaining_t == 0:
                out.append(tuple(acc))
            return
        # i_next ∈ [0, 2 · prev_i] (admissibility) and contributes (i_next + 1)
        # to remaining_t, so i_next ≤ remaining_t - remaining_s and i_next ≥ 0.
        upper = min(2 * prev_i, remaining_t - remaining_s)
        if upper < 0:
            return
        for i_next in range(0, upper + 1):
            acc.append(i_next)
            _recurse(i_next, remaining_t - i_next - 1, remaining_s - 1, acc)
            acc.pop()

    # First index has the excess cap.
    upper_first = min(max_excess, t - s)
    for i1 in range(0, upper_first + 1):
        _recurse(i1, t - i1 - 1, s - 1, [i1])
        if len(out) > hard_cap:
            raise RuntimeError(
                f"enumerate_admissibles_p2: basis exceeded hard_cap={hard_cap} "
                f"at (s={s}, t={t}, max_excess={max_excess}); raise the cap "
                f"or tighten the truncation window."
            )
    return out


# ── Differential matrix and rank ─────────────────────────────────────────────


def _build_differential_matrix_p2(
    basis_lo: List[LambdaMonomial],
    basis_hi: List[LambdaMonomial],
    max_excess: int,
) -> sp.csr_matrix:
    """Build the sparse F_2 matrix of d : Λ_s → Λ_{s+1} on excess-bounded admissibles.

    Columns index `basis_lo` (source, length s); rows index `basis_hi`
    (target, length s+1). Entries are 0/1 mod 2.
    """
    row_index = {m: i for i, m in enumerate(basis_hi)}
    rows: List[int] = []
    cols: List[int] = []
    for j, m in enumerate(basis_lo):
        dm = lambda_differential_p2(m)
        for tgt, c in dm.items():
            if c == 0:
                continue
            # Drop targets that exceed the excess cap (they vanish in the
            # filtered subcomplex; this preserves d² = 0 on the surviving
            # admissibles by the standard filtration argument).
            if tgt and tgt[0] > max_excess:
                continue
            ri = row_index.get(tgt)
            if ri is None:
                # This admissible is outside the truncation window (excess
                # ok but the target falls in a higher (s+1, t) than we
                # enumerated). Drop it.
                continue
            rows.append(ri)
            cols.append(j)
    if not rows:
        return sp.csr_matrix((len(basis_hi), len(basis_lo)), dtype=np.int8)
    data = [1] * len(rows)
    return sp.csr_matrix(
        (data, (rows, cols)), shape=(len(basis_hi), len(basis_lo)), dtype=np.int8
    )


def _rref_rank_p2(mat: sp.csr_matrix) -> int:
    """Rank of a sparse F_2 matrix via RREF on a list-of-dict representation.

    Mirrors `_sparse_fp_kernel` but returns just the rank (number of pivots).
    """
    if mat.shape[0] == 0 or mat.shape[1] == 0:
        return 0
    rows, cols = mat.shape
    R: List[Dict[int, int]] = [{} for _ in range(rows)]
    coo = mat.tocoo()
    for r, c, v in zip(coo.row, coo.col, coo.data):
        vv = int(v) & 1
        if vv:
            R[int(r)][int(c)] = 1
    rank = 0
    used: set[int] = set()
    for c in range(cols):
        pivot_r = None
        for r in range(rows):
            if r in used:
                continue
            if R[r].get(c, 0):
                pivot_r = r
                break
        if pivot_r is None:
            continue
        for r in range(rows):
            if r == pivot_r or not R[r].get(c, 0):
                continue
            piv = R[pivot_r]
            new_row = dict(R[r])
            for k in piv:
                if new_row.get(k, 0):
                    new_row.pop(k)
                else:
                    new_row[k] = 1
            R[r] = new_row
        used.add(pivot_r)
        rank += 1
    return rank


# ── Self-test: d² = 0 ────────────────────────────────────────────────────────


def verify_d_squared_zero_p2(
    weight_max: int,
    t_max: int,
    excess_max: int,
) -> Tuple[bool, List[Tuple[LambdaMonomial, LambdaElement]]]:
    """Check d² = 0 on every admissible monomial within the given degree/excess bounds.

    Returns (ok, failures) where each failure is (input_monomial, d²(input)).
    A clean run returns (True, []).
    """
    failures: List[Tuple[LambdaMonomial, LambdaElement]] = []
    for s in range(1, weight_max + 1):
        for t in range(s, t_max + 1):
            for mon in enumerate_admissibles_p2(s, t, excess_max, hard_cap=1 << 20):
                d1 = lambda_differential_p2(mon)
                d2: LambdaElement = {}
                for m, c in d1.items():
                    if c == 0:
                        continue
                    for m2, c2 in lambda_differential_p2(m).items():
                        v = (d2.get(m2, 0) + c * c2) % 2
                        if v:
                            d2[m2] = v
                        else:
                            d2.pop(m2, None)
                if d2:
                    failures.append((mon, d2))
    return (len(failures) == 0, failures)


# ── Ext computation: Λ-homology at p = 2 ─────────────────────────────────────


def _ext_lambda_p2(
    s_max: int,
    t_max: int,
    max_excess: int,
) -> Dict[Tuple[int, int], int]:
    """Compute dim_{F_2} H^{s,t}(Λ_2, d) for s ≤ s_max, t ≤ t_max, excess ≤ max_excess.

    Returns a sparse dict (only nonzero entries).
    """
    # Pre-enumerate bases at each (s, t).
    bases: Dict[Tuple[int, int], List[LambdaMonomial]] = {}
    for s in range(0, s_max + 2):
        for t in range(0, t_max + 1):
            b = enumerate_admissibles_p2(s, t, max_excess)
            if b:
                bases[(s, t)] = b

    # Rank cache for d_{s,t}: Λ_{s, t} → Λ_{s+1, t}.
    rank_cache: Dict[Tuple[int, int], int] = {}

    def rank_of(s: int, t: int) -> int:
        if (s, t) in rank_cache:
            return rank_cache[(s, t)]
        basis_lo = bases.get((s, t), [])
        basis_hi = bases.get((s + 1, t), [])
        if not basis_lo or not basis_hi:
            rank_cache[(s, t)] = 0
            return 0
        mat = _build_differential_matrix_p2(basis_lo, basis_hi, max_excess)
        r = _rref_rank_p2(mat)
        rank_cache[(s, t)] = r
        return r

    out: Dict[Tuple[int, int], int] = {}
    for (s, t), basis in bases.items():
        if s > s_max:
            continue
        dim_C = len(basis)
        r_out = rank_of(s, t)
        r_in = rank_of(s - 1, t) if s >= 1 else 0
        ext_dim = dim_C - r_out - r_in
        if ext_dim < 0:
            ext_dim = 0  # truncation artefact; documented in caller
        if ext_dim > 0:
            out[(s, t)] = ext_dim
    return out


# ── Public façade: lambda_e2_page (generic over FpCohomologyRing) ────────────


def lambda_e2_page(
    fp_ring: FpCohomologyRing,
    prime: int,
    *,
    s_max: int = 6,
    t_max: int = 20,
    backend: Literal["auto", "python", "numba", "julia"] = "auto",
) -> AdamsE2Page:
    """WIP — DO NOT USE; raises NotImplementedError.

    The Λ-algebra differential formula in this module
    is mathematically incorrect at length ≥ 2 (d² ≠ 0 on λ_4, λ_8, …).
    The single-generator differential is right, so Ext^{1,*} matches
    Tangora 1990 (h_0..h_4 at t = 1, 2, 4, 8, 16), but Ext^{s,*} for
    s ≥ 2 has errors against the literature. This façade therefore
    raises NotImplementedError; the dispatcher in adams_unstable.py
    falls back to the U-resolution engine.

    The Adem rewriter, admissibility checker, and basis enumerator in
    this module remain correct and reusable. Fixing the differential
    is tracked as future work; the required formula must use a non-
    derivation extension to products that respects the Adem ideal.

    What is being computed:
        E_2^{s,t}(X) = Ext_U^{s,t}(H̃*(X; F_p), F_p),

        where U is the category of unstable A_p-modules. Λ_p computes
        this as the cohomology of an excess-filtered DGA on the
        admissible monomials in λ_i (and μ_j at odd p; see Slice 2).

    Algorithm:
        1. Inspect `fp_ring` to find n_min = min positive-degree label.
        2. Enumerate admissibles of length ≤ s_max+1, internal degree
           ≤ t_max + n_min, excess ≤ n_min.
        3. Build Curtis differentials at each bidegree, take rank.
        4. Shift the resulting Ext^{s, t_rel} by each generator degree
           and weight by basis multiplicity in fp_ring (the Künneth-type
           direct sum across generators of the unstable module).
        5. Assemble an AdamsE2Page with reasoning string.

    Backend dispatch:
        - 'auto': try Julia kernel for basis enumeration + boundary
          assembly; fall back to Numba; fall back to pure Python.
        - 'python': force pure Python (deterministic).
        - 'numba': force the Numba inner-loop variant.
        - 'julia': require Julia; raise if `julia_engine.available is False`.
        Note (Slice 1): only 'python' is wired here. Slice 4 adds the
        Julia and Numba paths; the 'auto' policy below already prefers
        them when available — for now they degrade to 'python' silently.

    Args:
        fp_ring: input cohomology ring (use the reduced ring for the
            tightest bound; the un-reduced ring contains a ghost S^0).
        prime: 2 (Slice 1). Slice 2 extends to {3, 5}.
        s_max: homological-degree truncation (default 6).
        t_max: internal-degree truncation (default 20).
        backend: see above.

    Returns:
        AdamsE2Page with status='success' and a `space_label` suffix
        '(Λ-algebra)' marking the engine.
    """
    raise NotImplementedError(
        "lambda_e2_page is WIP: the Curtis differential's extension to "
        "products is buggy (d² ≠ 0 on λ_4, λ_8, …). Use "
        "pysurgery.adams.u_resolution.u_resolution_e2_page() instead."
    )

    # Slice 4 (Julia/Numba) inserts here:
    if backend == "julia":
        try:
            from pysurgery.bridge.julia_bridge import julia_engine
        except Exception as exc:
            raise RuntimeError(
                "backend='julia' requested but the julia bridge is not importable."
            ) from exc
        if not julia_engine.available:
            raise RuntimeError(
                "backend='julia' requested but julia_engine.available is False; "
                "use backend='auto' to fall back to Python."
            )
        # Slice 4 will replace this with a real Julia kernel call.
        # For now we deliberately fall through to Python so the engine
        # produces correct output under any backend setting.

    if backend not in ("auto", "python", "numba", "julia"):
        raise ValueError(f"unknown backend {backend!r}")

    # Find positive-degree module generators.
    module_dims: Dict[int, int] = {}
    for d, labels in fp_ring.basis.items():
        if d > 0 and labels:
            module_dims[int(d)] = len(labels)

    if not module_dims:
        # Trivial module: Ext^{0,0} = F_2 if the unit is present, else empty.
        e2: Dict[Tuple[int, int], int] = (
            {(0, 0): 1} if fp_ring.basis.get(0) else {}
        )
        return AdamsE2Page(
            space_label=fp_ring.space_label + " (Λ-algebra; trivial M)",
            prime=prime,
            s_max=s_max,
            t_max=t_max,
            e2_grid=e2,
            forced_vanishings=[],
            ambiguous_differentials=[],
            reliable_window=(s_max, max(0, t_max - s_max)),
            resource_summary={"peak_mem_mb": 0.0, "wall_seconds": 0.0},
            status="success",
            reasoning="Trivial unstable A_2-module (no positive-degree basis).",
        )

    n_min = min(module_dims.keys())
    t_start = _time.time()

    # Enumerate the "base" Ext^{s, t_rel} of (F_2, F_2) restricted to
    # excess ≤ n_min. The shift by each generator degree happens below.
    base_t_max = t_max  # internal degree of the shifted bidegree is t_rel + n_alpha
    base_ext = _ext_lambda_p2(s_max=s_max, t_max=base_t_max, max_excess=n_min)

    # Shift and sum across module generators.
    grid: Dict[Tuple[int, int], int] = {}
    for (s, t_rel), dim in base_ext.items():
        for n, mult in module_dims.items():
            t = t_rel + n
            if t > t_max or s > s_max:
                continue
            grid[(s, t)] = grid.get((s, t), 0) + mult * dim

    wall = _time.time() - t_start

    # Classify d_r forced-zero vs ambiguous (same convention as the
    # stable engine; lets the E_∞ enumerator consume our page unchanged).
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
        space_label=fp_ring.space_label + " (Λ-algebra)",
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
            "Unstable Adams E_2 via the Λ-algebra (Bousfield–Curtis–Kan). "
            f"Excess cap = n_min = {n_min}. Computed by Curtis differential "
            f"on admissibles up to (s,t) = ({s_max},{t_max}); shifted across "
            f"generator degrees {sorted(module_dims)}."
        ),
    )


__all__ = [
    "LambdaMonomial",
    "LambdaElement",
    "lambda_degree_p2",
    "is_admissible_p2",
    "excess_p2",
    "adem_rewrite_p2",
    "lambda_differential_p2",
    "enumerate_admissibles_p2",
    "verify_d_squared_zero_p2",
    "lambda_e2_page",
]
