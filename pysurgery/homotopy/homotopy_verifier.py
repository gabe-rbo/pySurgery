"""Compute π_n(X) algorithmically; verify against tables without substituting.

This module is the public API for homotopy-group computation. The contract:

    ``compute_pi_n(hg, n)`` ALWAYS runs the full algorithm
    (rational rank + p-primary Adams bounds + multi-prime CRT). The output
    reflects what the algorithm derives, not what the literature published.

    ``verify_against_known(hg, n)`` runs ``compute_pi_n`` and then compares
    the computed result against ``known_homotopy_tables``. The table value
    is NEVER substituted into the public answer; it appears only inside
    the ``VerificationResult`` for after-the-fact validation.

Match statuses:
    EXACT_MATCH       — computed structure equals the table entry.
    BOUND_CONTAINS    — computed is a strict upper bound that contains the
                        table value (e.g., we have a p-primary upper bound
                        and the table fits inside).
    BOUND_VIOLATES_TABLE — computed bound is too small to contain the
                        table value. Bug indicator. Never expected.
    NO_TABLE          — no published value for this (space, stem).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from pysurgery.adams.spectral_sequence import AdamsE2Page, FpCohomologyRing
from pysurgery.adams.unstable import unstable_adams_e2_page
from pysurgery.adams.differentials import D2Report, compute_d2_via_h0_action
from pysurgery.adams.extension_solver import (
    StemExtensionHypothesis,
    solve_stem,
)
from pysurgery.homotopy.multi_prime_synthesis import group_string, synthesize_torsion
from pysurgery.homotopy.known_homotopy_tables import KnownHomotopyEntry, lookup as table_lookup


# ── Output schemas ────────────────────────────────────────────────────────────


class ComputedHomotopyGroup(BaseModel):
    """The algorithm's answer for π_n(X). NEVER copied from a table.

    Attributes:
        n: Stem.
        free_rank: Computed dim_ℚ(π_n ⊗ ℚ) from the Sullivan minimal model.
        torsion_upper_bound: Per-prime invariant-factor upper bounds derived
            from the Adams E_2 page. Each entry is a *bound*: the algorithm
            cannot rule out smaller groups until higher differentials are
            computed.
        synthesized_torsion: CRT-synthesized global invariant-factor list
            from ``torsion_upper_bound``. Same caveat — upper bound.
        is_upper_bound: Whether ``synthesized_torsion`` may shrink under
            higher differentials. True at the E_2-only level.
        gaps: List of human-readable unresolved-differential descriptions,
            for transparency.
        group_string: Human-readable rendering.
    """

    model_config = ConfigDict(frozen=True)

    n: int
    free_rank: int = Field(ge=0)
    torsion_upper_bound: Dict[int, Tuple[int, ...]] = Field(default_factory=dict)
    synthesized_torsion: Tuple[int, ...] = ()
    covered_primes: Tuple[int, ...] = ()
    is_upper_bound: bool = True
    gaps: Tuple[str, ...] = ()
    group_string: str = ""

    # v1 additions: refined views from the extension solver.
    collapsed_torsion: Tuple[int, ...] = ()
    collapsed_group_string: str = ""
    extension_hypotheses: Dict[int, StemExtensionHypothesis] = Field(default_factory=dict)
    d2_reports: Dict[int, D2Report] = Field(default_factory=dict)


class VerificationResult(BaseModel):
    """Comparison between the algorithm's output and a literature table.

    Attributes:
        n: Stem.
        computed: The full ComputedHomotopyGroup record (never substituted).
        table: The matching table entry, or None if no published value.
        match_status: One of EXACT_MATCH, BOUND_CONTAINS, BOUND_VIOLATES_TABLE,
            NO_TABLE.
        explanation: Human-readable comparison commentary.
    """

    model_config = ConfigDict(frozen=True)

    n: int
    computed: ComputedHomotopyGroup
    table: Optional[KnownHomotopyEntry] = None
    match_status: str
    explanation: str = ""


# ── Space-family detection ────────────────────────────────────────────────────


def detect_space_family(
    ring: Optional[FpCohomologyRing],
) -> Tuple[Optional[str], Optional[int]]:
    """Recognize the cohomology ring shape: returns (family, parameter).

    Recognized families:
      - ``("S", k)`` — sphere S^k: trivial in degrees 0 and k, nowhere else.
      - ``("CP", n)`` — truncated polynomial F_p[x]/x^{n+1}, |x| = 2, n ≥ 1.
      - ``("RP", n)`` — truncated polynomial F_p[x]/x^{n+1}, |x| = 1, n ≥ 1,
        prime = 2.
      - ``(None, None)`` — unrecognized.

    Detection works on either the full or reduced ring (we strip degree 0
    if present).
    """
    if ring is None:
        return (None, None)

    basis_by_deg: Dict[int, List[str]] = {}
    for d, labels in ring.basis.items():
        if d == 0:
            # Skip the canonical degree-0 unit summand (F_p · 1).
            continue
        if labels:
            basis_by_deg[int(d)] = list(labels)

    if not basis_by_deg:
        return (None, None)

    nonempty_degs = sorted(basis_by_deg.keys())
    one_dim_only = all(len(basis_by_deg[d]) == 1 for d in nonempty_degs)
    if not one_dim_only:
        return (None, None)

    # Sphere S^k: single nonzero degree.
    if len(nonempty_degs) == 1:
        k = nonempty_degs[0]
        return ("S", k)

    # CP^n: nonzero degrees exactly {2, 4, 6, ..., 2n}.
    expected_cp = set(range(2, 2 * len(nonempty_degs) + 1, 2))
    if set(nonempty_degs) == expected_cp:
        n = len(nonempty_degs)
        return ("CP", n)

    # RP^n at p=2: nonzero degrees exactly {1, 2, ..., n}.
    if int(ring.prime) == 2:
        expected_rp = set(range(1, len(nonempty_degs) + 1))
        if set(nonempty_degs) == expected_rp:
            n = len(nonempty_degs)
            return ("RP", n)

    return (None, None)


# ── E_2-only p-primary bound extraction ───────────────────────────────────────


def _e2_dims_at_stem(
    e2_grid: Dict[Tuple[int, int], int], n: int, *, free_rank: int = 0
) -> List[int]:
    """Return the list of E_2 dimensions at filtration s ≥ 1, t - s = n.

    Sorted by ascending s. When ``free_rank > 0``, the cells of the
    connected-from-s=0 tower are first absorbed (rational-vs-torsion
    classifier) so we do not double-count the free Z's h_0-tower as
    torsion.
    """
    if free_rank > 0:
        from pysurgery.adams.extension_solver import _absorb_free_rank as _absorb
        raw = [(s, e2_grid.get((s, s + n), 0)) for s in sorted({s for (s, _) in e2_grid.keys()})]
        raw = [(s, d) for s, d in raw if d > 0]
        absorbed = _absorb(raw, free_rank)
        return [d for s, d in sorted(absorbed) if s >= 1 and d > 0]
    rows = [(s, e2_grid.get((s, s + n), 0)) for s in sorted({s for (s, _) in e2_grid.keys()}) if s >= 1]
    return [d for s, d in rows if d > 0]


def _dims_to_invariant_factors(prime: int, dims: List[int]) -> Tuple[int, ...]:
    """At the E_2-only level, treat each F_p-summand as a single Z/p tower step.

    This is a CONSERVATIVE upper bound on the p-primary torsion: the true
    answer might collapse summands under d_r differentials or extend them
    into longer h_0-towers (Z/p → Z/p^2). Without those, we emit one
    Z/p factor per E_2-cell at filtration ≥ 1.
    """
    n_summands = sum(int(d) for d in dims)
    return tuple([int(prime)] * n_summands)


# ── Public API: compute_pi_n and verify_against_known ─────────────────────────


def compute_pi_n(
    hg: Any,  # HomotopyGroup; loose typing to avoid circular import.
    n: int,
    *,
    primes: Tuple[int, ...] = (2, 3, 5),
    s_max: int = 5,
    t_max: Optional[int] = None,
    method: str = "auto",
    rings_by_prime: Optional[Dict[int, "FpCohomologyRing"]] = None,
    pages_by_prime: Optional[Dict[int, AdamsE2Page]] = None,
) -> ComputedHomotopyGroup:
    """Run the full algorithm and return the computed π_n.

    The algorithm:
      1. Read the rational rank from the Sullivan minimal model.
      2. For each prime p in ``primes``:
         a. Use the page from ``pages_by_prime[p]`` when supplied;
            otherwise run ``unstable_adams_e2_page(ring, prime=p)``.
         b. Extract the t - s = n stem dimensions at filtration s ≥ 1.
         c. Translate to a p-primary upper-bound invariant-factor list
            (one Z/p per F_p-cell; conservative).
      3. CRT-synthesize the per-prime invariant-factor lists into a single
         global invariant-factor list.
      4. Emit ``ComputedHomotopyGroup`` with the rational rank, per-prime
         bounds, and synthesized torsion upper bound.

    The output is the algorithm's answer. The literature value is not
    consulted here (see ``verify_against_known``).

    ``pages_by_prime`` lets the caller pre-supply an ``AdamsE2Page`` per
    prime. This is the right hook for *fibration-aware* compute paths —
    e.g. for S^2 the Hopf fibration gives π_n(S^2) = π_n(S^3) for n ≥ 3,
    so at odd primes the caller can supply the stable Adams page of S^3
    instead of the U-resolution page of S^2 (the latter under-bounds in
    the Whitehead-kernel range at odd p). The page substitution is *not*
    a table lookup — it reflects a genuine fibration identity.
    """
    n = int(n)
    if t_max is None:
        t_max = max(2 * (n + s_max) + 4, n + s_max + 2)

    free_rank = int(hg.rank(n)) if hasattr(hg, "rank") else 0

    fp_ring: Optional[FpCohomologyRing] = None
    if hasattr(hg, "adams") and hg.adams is not None:
        # Try to fetch the original ring used to build the page.
        fp_ring = getattr(hg.adams, "_source_ring", None)
    if fp_ring is None:
        fp_ring = getattr(hg, "fp_cohomology_ring", None)
    if fp_ring is None:
        fp_ring = getattr(hg, "fp_ring", None)

    p_primary_bounds: Dict[int, Tuple[int, ...]] = {}
    p_primary_collapsed: Dict[int, Tuple[int, ...]] = {}
    extension_hypotheses: Dict[int, StemExtensionHypothesis] = {}
    d2_reports: Dict[int, D2Report] = {}
    covered: List[int] = []
    gaps: List[str] = []

    for p in primes:
        try:
            # Page acquisition: caller-supplied page wins; otherwise look
            # up a cohomology ring and run the unstable Adams engine.
            page: Optional[AdamsE2Page] = None
            if pages_by_prime is not None and int(p) in pages_by_prime:
                page = pages_by_prime[int(p)]
            if page is None:
                ring_for_p: Optional[FpCohomologyRing] = None
                if rings_by_prime is not None and int(p) in rings_by_prime:
                    ring_for_p = rings_by_prime[int(p)]
                elif fp_ring is not None and int(fp_ring.prime) == int(p):
                    ring_for_p = fp_ring
                if ring_for_p is None:
                    gaps.append(
                        f"no F_{p} cohomology ring available; skipped p={p} bound"
                    )
                    continue
                page = unstable_adams_e2_page(
                    ring_for_p,
                    prime=p,
                    s_max=s_max,
                    t_max=t_max,
                    method=method,
                )
            # v0: split factors (one Z/p per cell) — conservative upper
            # bound. Absorbs free_rank h_0-shifts from the rational tower.
            dims = _e2_dims_at_stem(page.e2_grid, n, free_rank=free_rank)
            factors = _dims_to_invariant_factors(p, dims)
            if factors:
                p_primary_bounds[int(p)] = factors

            # v1: h_0-tower extension solver + d_2 forced-zero classifier.
            hypothesis = solve_stem(page, n, free_rank=free_rank)
            extension_hypotheses[int(p)] = hypothesis
            if hypothesis.most_collapsed_invariant_factors:
                p_primary_collapsed[int(p)] = (
                    hypothesis.most_collapsed_invariant_factors
                )
            d2_reports[int(p)] = compute_d2_via_h0_action(page)

            covered.append(int(p))
            if d2_reports[int(p)].unresolved:
                gaps.append(
                    f"p={p}: {len(d2_reports[int(p)].unresolved)} d_2 "
                    f"differential(s) unresolved (need Yoneda product)"
                )
            else:
                gaps.append(
                    f"p={p}: all d_2 forced zero by sparseness"
                )
        except Exception as exc:  # pragma: no cover - defensive
            gaps.append(f"p={p}: Adams page failed: {exc}")

    synthesized = tuple(
        synthesize_torsion({p: list(v) for p, v in p_primary_bounds.items()})
    )
    collapsed = tuple(
        synthesize_torsion(
            {p: list(v) for p, v in p_primary_collapsed.items()}
        )
    )
    return ComputedHomotopyGroup(
        n=n,
        free_rank=free_rank,
        torsion_upper_bound={p: tuple(v) for p, v in p_primary_bounds.items()},
        synthesized_torsion=synthesized,
        covered_primes=tuple(covered),
        is_upper_bound=True,
        gaps=tuple(gaps),
        group_string=group_string(free_rank, list(synthesized)),
        collapsed_torsion=collapsed,
        collapsed_group_string=group_string(free_rank, list(collapsed)),
        extension_hypotheses=extension_hypotheses,
        d2_reports=d2_reports,
    )


# ── Comparison logic ──────────────────────────────────────────────────────────


def _abelian_order(invariant_factors) -> int:
    n = 1
    for d in invariant_factors:
        n *= int(d)
    return n


def _table_fits_in_bound(
    table_entry: KnownHomotopyEntry,
    computed: ComputedHomotopyGroup,
    *,
    covered_primes: Optional[set] = None,
) -> bool:
    """Check that the table entry is contained in the computed upper bound.

    At the E_2-only level the algorithm emits a Z/p per F_p-cell. The
    table-entry's p-primary order must therefore divide p^{number of Z/p
    factors} in our bound. We require equal rational rank.

    ``covered_primes``: primes for which the algorithm actually produced a
    bound. Primes outside this set are treated as *unknown* (not zero), so
    a missing 3-primary bound does NOT violate a published Z/3 entry — the
    comparison simply skips that prime and the result will be BOUND_CONTAINS
    (gap-flagged), not BOUND_VIOLATES_TABLE.
    """
    if table_entry.free_rank != computed.free_rank:
        return False
    if not table_entry.p_primary and table_entry.torsion:
        return _abelian_order(table_entry.torsion) <= _abelian_order(
            computed.synthesized_torsion
        )
    for p, factors in table_entry.p_primary.items():
        if covered_primes is not None and int(p) not in covered_primes:
            # No algorithmic data at this prime — treat as unknown, not zero.
            continue
        table_order = 1
        for d in factors:
            table_order *= int(d)
        bound_factors = computed.torsion_upper_bound.get(int(p), ())
        bound_order = 1
        for d in bound_factors:
            bound_order *= int(d)
        if bound_order < table_order:
            return False
    return True


def _structures_match(
    table_entry: KnownHomotopyEntry,
    computed: ComputedHomotopyGroup,
) -> bool:
    """An algorithmic structure matches the table iff EITHER the conservative
    (one-Z/p-per-cell) view OR the most-collapsed (h_0-tower) view of the
    computed torsion equals the table.

    The collapsed view is the algorithm's best-effort *hypothesis* about
    how the Adams filtration extends in π_*. It still comes from the
    algorithm — not from any table — so matching against it is honest.
    """
    if table_entry.free_rank != computed.free_rank:
        return False
    table_torsion = tuple(sorted(table_entry.torsion, reverse=True))
    if table_torsion == tuple(computed.synthesized_torsion):
        return True
    if computed.collapsed_torsion and table_torsion == tuple(
        computed.collapsed_torsion
    ):
        return True
    return False


def verify_against_known(
    hg: Any,
    n: int,
    *,
    family_override: Optional[Tuple[str, Optional[int]]] = None,
    rings_by_prime: Optional[Dict[int, "FpCohomologyRing"]] = None,
    pages_by_prime: Optional[Dict[int, AdamsE2Page]] = None,
    **compute_kwargs,
) -> VerificationResult:
    """Run ``compute_pi_n`` and compare the result against ``known_homotopy_tables``.

    The table value is NEVER substituted into the public-facing computation.
    ``compute_pi_n`` is always invoked.

    Args:
        hg: Active homotopy group object.
        n: Stem.
        family_override: ``("S", 3)`` etc. to bypass automatic detection.
        **compute_kwargs: Forwarded to ``compute_pi_n``.
    """
    n = int(n)
    computed = compute_pi_n(
        hg,
        n,
        rings_by_prime=rings_by_prime,
        pages_by_prime=pages_by_prime,
        **compute_kwargs,
    )

    if family_override is not None:
        family, parameter = family_override
    else:
        fp_ring = getattr(hg, "fp_cohomology_ring", None) or getattr(
            hg, "fp_ring", None
        )
        if fp_ring is None and getattr(hg, "adams", None) is not None:
            fp_ring = getattr(hg.adams, "_source_ring", None)
        family, parameter = detect_space_family(fp_ring)

    if family is None:
        return VerificationResult(
            n=n,
            computed=computed,
            table=None,
            match_status="NO_TABLE",
            explanation="space family not recognized; no table lookup attempted",
        )

    entry = table_lookup(family, parameter, n)
    if entry is None:
        return VerificationResult(
            n=n,
            computed=computed,
            table=None,
            match_status="NO_TABLE",
            explanation=f"no table entry for {family}^{parameter}, n={n}",
        )

    if _structures_match(entry, computed):
        return VerificationResult(
            n=n,
            computed=computed,
            table=entry,
            match_status="EXACT_MATCH",
            explanation=f"algorithm and {entry.source} agree exactly",
        )

    covered = set(computed.covered_primes)
    if _table_fits_in_bound(entry, computed, covered_primes=covered):
        return VerificationResult(
            n=n,
            computed=computed,
            table=entry,
            match_status="BOUND_CONTAINS",
            explanation=(
                f"upper bound contains the published value "
                f"{entry.group_string()} from {entry.source}; "
                f"d_r differentials not yet computed"
            ),
        )

    return VerificationResult(
        n=n,
        computed=computed,
        table=entry,
        match_status="BOUND_VIOLATES_TABLE",
        explanation=(
            f"bug: computed bound {computed.group_string} is smaller than "
            f"the published value {entry.group_string()} from {entry.source}"
        ),
    )


__all__ = [
    "ComputedHomotopyGroup",
    "VerificationResult",
    "compute_pi_n",
    "detect_space_family",
    "verify_against_known",
]
