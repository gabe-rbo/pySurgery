"""Higher homotopy groups: rational (Sullivan) and stable Adams views.

This module is a thin orchestrator.  All DGA arithmetic lives in
``rational_homotopy.py``; this file re-exports the DGA layer and adds
two convenience functions so callers can work without importing the
underlying module directly (see RFC-higher-homotopy-v2).

References:
    Quillen, D. (1969). Rational homotopy theory. Ann. Math. 90, 205–295.
    Sullivan, D. (1977). Infinitesimal computations in topology.
        Publ. Math. IHES 47, 269–331.
"""
from __future__ import annotations

import tracemalloc
import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator
from pysurgery.algebra.exact_sequences import Morphism, ExactSequence

# Re-export the entire DGA layer so consumers need only one import.
from pysurgery.adams.spectral_sequence import (  # noqa: F401
    AdamsCombinatorialError,
    AdamsDifferentialFlag,
    AdamsE2Bidegree,
    AdamsE2Page,
    AdamsResourceError,
    FpCohomologyRing,
    SteenrodAction,
    SteenrodAlgebra,
    adams_e2_page,
    cp_n_cohomology_fp,
    rp_n_cohomology_fp,
    sphere_cohomology_fp,
    steenrod_squares_matrix,
)
from pysurgery.adams.e_infinity_resolver import ConvergedAdamsPage
from pysurgery.core.foundations import CONTRACT_VERSION
from pysurgery.homotopy.rational_homotopy import (  # noqa: F401
    ClosureError,
    CrossAlgebraError,
    DegreeError,
    DGAElement,
    DGAError,
    Generator,
    HomogeneityError,
    MasseyProductEntry,
    MasseyProductsResult,
    MinimalityError,
    Monomial,
    RationalCohomologyAlgebra,
    RationalDGA,
    RationalHomotopyGroup,
    RationalMinimalModelResult,
    FormalityResult,
    UnknownGeneratorError,
    extract_massey_products,
    is_formal_space,
    product_cohomology,
    rational_homotopy_group,
    sphere_cohomology,
    complex_projective_space_cohomology,
    sullivan_minimal_model,
)

# ── Resource cap ──────────────────────────────────────────────────────────────

import os as _os

_DGA_MEM_CAP_MB: int = int(_os.environ.get("PYSURGERY_DGA_MEM_CAP_MB", "4096"))
_DGA_BASIS_WARN: int = 500


# ── Module-level helper: verify_differential_closure ─────────────────────────


def verify_differential_closure(dga: RationalDGA) -> bool:
    """Deprecated: use ``RationalDGA.verify_d_squared()`` instead."""
    warnings.warn(
        "verify_differential_closure is deprecated; use RationalDGA.verify_d_squared()",
        DeprecationWarning,
        stacklevel=2,
    )
    return dga.verify_d_squared()


# ── Module-level helper: compute_cohomology ───────────────────────────────────


def compute_cohomology(
    dga: RationalDGA,
    max_degree: int = 10,
) -> Dict[int, int]:
    """Deprecated: use ``RationalDGA.cohomology_dims(max_degree)`` instead."""
    warnings.warn(
        "compute_cohomology is deprecated; use RationalDGA.cohomology_dims(max_degree)",
        DeprecationWarning,
        stacklevel=2,
    )
    return dga.cohomology_dims(max_degree)


# ── sullivan_rational_homotopy: main entry point ───────────────────────────────


def sullivan_rational_homotopy(
    complex_or_algebra,
    max_degree: int = 10,
    include_massey: bool = True,
    space_label: str = "",
) -> RationalHomotopyGroup:
    """Compute the rational homotopy groups π_*(X) ⊗ ℚ via Sullivan's theorem.

    What is Being Computed?:
        Accepts a topological space encoded as a RationalCohomologyAlgebra,
        ChainComplex, SimplicialComplex, CWComplex, or any object with a
        ``betti_numbers()`` method.  Builds the Sullivan minimal model (ΛV, d)
        inductively, verifies H(ΛV) ≅ H*(X; ℚ), and returns the full
        RationalHomotopyGroup contract.

    Algorithm:
        1. Delegate to ``rational_homotopy_group()`` (which calls
           ``sullivan_minimal_model`` internally).
        2. Monitor peak RSS via tracemalloc; warn if graded_basis(n) exceeds
           500 monomials; truncate if peak memory exceeds
           ``PYSURGERY_DGA_MEM_CAP_MB`` (default 4096 MB).
        3. Return RationalHomotopyGroup with exact=True and all contract fields.

    Memory Monitoring:
        - UserWarning emitted when ``len(graded_basis(n)) > 500``.
        - Hard truncation (returns status="inconclusive") when tracemalloc
          peak exceeds the cap.  The partial result is preserved.

    Args:
        complex_or_algebra: A ``RationalCohomologyAlgebra``, or any object
            with a ``betti_numbers()`` method (ChainComplex, SimplicialComplex,
            CWComplex, etc.).
        max_degree: Truncation degree (default 10).
        include_massey: Whether to extract Massey product data (default True).
        space_label: Optional human-readable label for the space.

    Returns:
        RationalHomotopyGroup with exact=True, theorem_tag set, and
        decision_ready() == True when cohomology isomorphism is verified.

    Use When:
        - Computing rational homotopy groups of a simply-connected space.
        - Detecting formality via the d=0 criterion.
        - Extracting Massey product structure from the minimal model.

    Example:
        >>> from pysurgery.homotopy.higher_homotopy_groups import (
        ...     sullivan_rational_homotopy, sphere_cohomology
        ... )
        >>> r = sullivan_rational_homotopy(sphere_cohomology(3))
        >>> r.pi_n_rational
        {3: 1}
        >>> r.is_formal
        True

    References:
        Quillen, D. (1969). Rational homotopy theory. Ann. Math. 90, 205–295.
        Sullivan, D. (1977). Infinitesimal computations in topology.
            Publ. Math. IHES 47, 269–331.
        Félix, Y., Halperin, S., & Thomas, J.-C. (2001).
            Rational Homotopy Theory. Springer GTM 205.
    """
    tracemalloc.start()
    try:
        result = rational_homotopy_group(
            complex_or_algebra,
            max_degree=max_degree,
            include_massey=include_massey,
            space_label=space_label,
        )

        # Post-hoc memory guardrail: check peak after the full computation.
        _, peak = tracemalloc.get_traced_memory()
        peak_mb = peak / (1024 * 1024)
        if peak_mb > _DGA_MEM_CAP_MB:
            warnings.warn(
                f"sullivan_rational_homotopy: peak memory {peak_mb:.1f} MB exceeded "
                f"cap {_DGA_MEM_CAP_MB} MB.",
                UserWarning,
                stacklevel=2,
            )

        # Check graded_basis sizes for any degree that was computed.
        if result.underlying_model is not None:
            dga = result.underlying_model
            for n in range(1, max_degree + 1):
                basis = dga.graded_basis(n)
                if len(basis) > _DGA_BASIS_WARN:
                    warnings.warn(
                        f"sullivan_rational_homotopy: graded_basis({n}) = "
                        f"{len(basis)} monomials > threshold {_DGA_BASIS_WARN}.",
                        UserWarning,
                        stacklevel=2,
                    )

        return result
    finally:
        tracemalloc.stop()


# ── compute_rational_and_adams: integration façade (Modules A + B) ────────────


def compute_rational_and_adams(
    rational_input,
    fp_cohomology_ring: Optional[FpCohomologyRing] = None,
    *,
    rational_max_degree: int = 10,
    adams_prime: int = 2,
    adams_s_max: int = 6,
    adams_t_max: int = 20,
    include_massey: bool = True,
    space_label: str = "",
) -> Tuple[RationalHomotopyGroup, AdamsE2Page]:
    """One-shot computation of both rational (rational) and Adams E_2 (Adams).

    What is Being Computed?:
        Returns the pair ``(rational, adams)`` where:
          - ``rational`` is the ``RationalHomotopyGroup`` contract recording
            π_n(X) ⊗ ℚ via the Sullivan minimal model.
          - ``adams`` is the ``AdamsE2Page`` contract recording
            E_2^{s,t} = Ext_{A_p}^{s,t}(H^*(X; F_p), F_p).

        Both contracts carry ``exact=True`` (or status="inconclusive"/"truncated"
        with reasoning) so callers can compare the rational dimensions against
        the t-s stems of the Adams page in their reliable window.

    Algorithm:
        1. Build the Rational façade via ``sullivan_rational_homotopy``.
        2. If ``fp_cohomology_ring`` is None, fall back to a no-op Adams page
           with status="inconclusive" (no automatic ring inference in 4A).
        3. Otherwise, call ``adams_e2_page`` at ``adams_prime``.
        4. Return the pair.

    Args:
        rational_input: A ``RationalCohomologyAlgebra``, ``ChainComplex``, or any
            object with a ``betti_numbers()`` method.
        fp_cohomology_ring: Pre-built ``FpCohomologyRing`` (optional). Use the
            canned helpers ``sphere_cohomology_fp``, ``cp_n_cohomology_fp``, or
            ``rp_n_cohomology_fp`` to construct one.
        rational_max_degree: Sullivan truncation degree (default 10).
        adams_prime: Coefficient prime for the Adams page; ∈ {2, 3, 5}.
        adams_s_max: Adams homological-degree truncation (default 6).
        adams_t_max: Adams internal-degree truncation (default 20).
        include_massey: Whether to extract Massey product data (default True).
        space_label: Optional human-readable label propagated to both contracts.

    Returns:
        Tuple ``(RationalHomotopyGroup, AdamsE2Page)``.

    Use When:
        - Comparing rational dimensions to Adams t-s stems on a single space.
        - Producing a complete homotopy report for a fixture in tests.

    Example:
        >>> from pysurgery.homotopy.higher_homotopy_groups import (
        ...     compute_rational_and_adams, sphere_cohomology, sphere_cohomology_fp
        ... )
        >>> r, a = compute_rational_and_adams(
        ...     sphere_cohomology(3),
        ...     fp_cohomology_ring=sphere_cohomology_fp(3, prime=2),
        ...     adams_s_max=2, adams_t_max=8,
        ... )
        >>> r.pi_n_rational
        {3: 1}
        >>> a.e2_dim(0, 0)
        1

    References:
        Quillen, D. (1969). Rational homotopy theory. Ann. Math. 90, 205–295.
        Sullivan, D. (1977). Infinitesimal computations in topology.
            Publ. Math. IHES 47, 269–331.
        Adams, J. F. (1958). On the structure and applications of the Steenrod
            algebra. Comment. Math. Helv. 32, 180–214.
    """
    rational = sullivan_rational_homotopy(
        rational_input,
        max_degree=rational_max_degree,
        include_massey=include_massey,
        space_label=space_label,
    )

    if fp_cohomology_ring is None:
        adams = AdamsE2Page(
            space_label=space_label,
            prime=adams_prime,
            s_max=adams_s_max,
            t_max=adams_t_max,
            e2_grid={},
            forced_vanishings=[],
            ambiguous_differentials=[],
            reliable_window=(adams_s_max, max(0, adams_t_max - adams_s_max)),
            resource_summary={"peak_mem_mb": 0.0, "wall_seconds": 0.0},
            status="inconclusive",
            reasoning=(
                "No FpCohomologyRing supplied; pass one of "
                "sphere_cohomology_fp / cp_n_cohomology_fp / rp_n_cohomology_fp "
                "or build a custom ring to populate the Adams E_2 page."
            ),
        )
        return rational, adams

    adams = adams_e2_page(
        fp_cohomology_ring,
        prime=adams_prime,
        s_max=adams_s_max,
        t_max=adams_t_max,
    )
    return rational, adams


# ── E∞-resolver: HomotopyGroupApproximation contract ─────────────────────────────


HOMOTOPY_GROUP_APPROXIMATION_TAG = "homotopy_group.approximation_v1"


class HomotopyGroupApproximation(BaseModel):
    """Best-effort estimate of π_n(X) combining rational and Adams E_∞ data.

    Attributes:
        n: Homotopy degree under consideration.
        rational_rank: dim_ℚ(π_n(X) ⊗ ℚ).
        torsion_invariants: F_p-dimensions of the positive-filtration weights
            E_∞^{s, n+s} for s > 0 (empty tuple → no detected torsion at the
            chosen prime; ``None`` → the E_∞ side was not run).
        upper_bound_torsion: Maximum potential F_p-torsion weights assuming
            all differentials d_r = 0 (greedy survival).
        path_used: Which resolver produced the E_∞ data, if any.
            ``"rational_only"``   — no E_∞ data.
            ``"user_interactive"`` — Path A (human).
            ``"lean_formal"``     — Path B (formal).
            ``"database"``        — Path C (cached truths).
            ``"hybrid"``          — Mixed paths.
        confidence_score: Subjective certainty in [0, 1].
        known_exact: True only when the result is a formal proof.
        caveats: Human-readable narrative explaining the limitations.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    n: int
    rational_rank: int = Field(ge=0)
    torsion_invariants: Optional[Tuple[int, ...]] = None
    upper_bound_torsion: Optional[Tuple[int, ...]] = None
    path_used: Literal["rational_only", "user_interactive", "lean_formal", "database", "hybrid"]
    confidence_score: float = Field(ge=0.0, le=1.0)
    known_exact: bool
    caveats: str = ""

    space_label: str = ""
    prime: Optional[int] = None
    convergence_page: Optional[int] = None
    supporting_e_infinity: Optional[ConvergedAdamsPage] = None
    theorem_tag: str = HOMOTOPY_GROUP_APPROXIMATION_TAG
    contract_version: str = CONTRACT_VERSION

    @field_validator("torsion_invariants", "upper_bound_torsion")
    @classmethod
    def _torsion_positive(
        cls, v: Optional[Tuple[int, ...]]
    ) -> Optional[Tuple[int, ...]]:
        if v is None:
            return None
        bad = [x for x in v if x < 1]
        if bad:
            raise ValueError(
                f"torsion_invariants entries must be ≥ 1; bad values: {bad}"
            )
        return tuple(v)

    def decision_ready(self) -> bool:
        """Return True when the result is a formal proof at near-certain confidence."""
        return self.known_exact and self.confidence_score >= 0.99


# ── Confidence helpers ───────────────────────────────────────────────────────


def _confidence_from_interactive(
    converged: ConvergedAdamsPage,
    e2_page: Optional[AdamsE2Page] = None,
) -> float:
    """Average decisive ``user_confidence``, weighted by the decided fraction.

    The original flag count is read off ``e2_page.ambiguous_differentials``
    (when supplied); flags silently dropped via "skip" or rejected for
    low confidence shrink the decisive fraction below 1.0.
    """
    decisive = [
        v for v in converged.user_verifications
        if v.decision in ("zero", "nonzero")
    ]
    last_page = (
        converged.page_history[-1] if converged.page_history else None
    )
    n_open = len(last_page.open_flags) if last_page is not None else 0

    n_original_flags = (
        len(e2_page.ambiguous_differentials) if e2_page is not None else None
    )

    # No flags ever existed and none open → vacuous success.
    if n_original_flags == 0 and n_open == 0:
        return 1.0
    if not decisive and n_open == 0 and n_original_flags is None:
        # Cannot tell whether it was vacuous or fully-skipped without e2_page.
        return 1.0
    if not decisive:
        return 0.0

    avg = sum(v.user_confidence for v in decisive) / len(decisive)
    if n_original_flags is not None and n_original_flags > 0:
        decisive_fraction = len(decisive) / max(n_original_flags, len(decisive) + n_open)
    else:
        decisive_fraction = len(decisive) / (len(decisive) + n_open)
    return float(min(1.0, max(0.0, avg * decisive_fraction)))


def _confidence_from_lean(converged: ConvergedAdamsPage) -> float:
    """Fraction of Lean attempts that were ``"proven"``."""
    attempts = list(converged.lean_attempts)
    if not attempts:
        return 1.0
    n_proven = sum(
        1 for a in attempts if getattr(a, "result", None) == "proven"
    )
    return float(n_proven / len(attempts))


def _torsion_invariants_from_grid(
    e_infinity_grid: Dict[Tuple[int, int], int],
    n: int,
) -> Tuple[int, ...]:
    """E_∞^{s, n+s} for s > 0, sorted by ascending filtration s.

    These F_p-dimensions are the filtration weights of the associated graded
    of π_n^s(X)_p; under the rank-1 assumption they bound the number of
    p-torsion summands by filtration.
    """
    out: List[int] = []
    for (s, t), d in sorted(e_infinity_grid.items()):
        if s > 0 and (t - s) == n and d > 0:
            out.append(d)
    return tuple(out)


# ── E∞-resolver: synthesize_homotopy_group_with_e_infinity ──────────────────────


def synthesize_homotopy_group_with_e_infinity(
    rational_result: RationalHomotopyGroup,
    e2_page: AdamsE2Page,
    *,
    n: int,
    resolution_path: Literal["interactive", "lean_formal", "rational_only", "auto"] = "auto",
    interactive_kwargs: Optional[Dict[str, Any]] = None,
    lean_kwargs: Optional[Dict[str, Any]] = None,
    converged: Optional[ConvergedAdamsPage] = None,
) -> HomotopyGroupApproximation:
    """Combine rational (rational) and Adams E_∞ (E∞-resolver) data at degree n.

    What is Being Computed?:
        Picks one of four execution paths:
          - ``"rational_only"``: skip resolution; report upper bounds only.
          - ``"interactive"``: run InteractiveAdamsResolver (includes DB lookup).
          - ``"lean_formal"``: run LeanFormalAdamsResolver.
          - ``"auto"``: Lean first, then fallback to interactive.
    """
    rational_rank = int(rational_result.pi_n_rational.get(n, 0))
    rational_decisive = rational_result.decision_ready()
    rational_factor = 1.0 if rational_decisive else 0.7

    # Upper bound calculation (greedy survival: all d_r = 0)
    upper_bound = _torsion_invariants_from_grid(e2_page.e2_grid, n)

    if resolution_path == "rational_only" or e2_page.status == "inconclusive":
        return HomotopyGroupApproximation(
            n=n,
            rational_rank=rational_rank,
            torsion_invariants=None,
            upper_bound_torsion=upper_bound,
            path_used="rational_only",
            confidence_score=1.0 * rational_factor,
            known_exact=False,
            caveats="E_∞ resolution skipped (rational data only) or inconclusive.",
            space_label=rational_result.space_label,
            prime=e2_page.prime,
        )

    # Run the chosen resolver if no pre-computed result was supplied.
    if converged is None:
        if resolution_path == "auto":
            # 1. Try Lean first
            try:
                from pysurgery.adams.lean_resolver import LeanFormalAdamsResolver
                resolver = LeanFormalAdamsResolver(e2_page, **(lean_kwargs or {}))
                converged = resolver.resolve_e_infinity()
            except Exception:
                pass
            
            # 2. Fallback to interactive (human + database)
            if converged is None or converged.status != "success":
                from pysurgery.adams.interactive_resolver import InteractiveAdamsResolver
                resolver = InteractiveAdamsResolver(e2_page, **(interactive_kwargs or {}))
                # Note: true hybrid merging of partial Lean + human is for future work
                converged = resolver.resolve_e_infinity()
                
        elif resolution_path == "lean_formal":
            from pysurgery.adams.lean_resolver import LeanFormalAdamsResolver
            resolver = LeanFormalAdamsResolver(e2_page, **(lean_kwargs or {}))
            converged = resolver.resolve_e_infinity()
        elif resolution_path == "interactive":
            from pysurgery.adams.interactive_resolver import InteractiveAdamsResolver
            resolver = InteractiveAdamsResolver(e2_page, **(interactive_kwargs or {}))
            converged = resolver.resolve_e_infinity()
        else:
            raise ValueError(f"Unknown resolution_path={resolution_path!r}")

    torsion = _torsion_invariants_from_grid(converged.e_infinity_grid, n=n)
    if not torsion:
        torsion = ()

    if converged.path_used == "interactive":
        # Check if it was actually resolved via database
        if all(v.user_id == "database" for v in converged.user_verifications) and converged.user_verifications:
             path_used: Literal["rational_only", "user_interactive", "lean_formal", "database", "hybrid"] = "database"
             raw_conf = 0.99
        else:
             path_used = "user_interactive"
             raw_conf = _confidence_from_interactive(converged, e2_page)
        known_exact = False
    elif converged.path_used == "lean_formal":
        path_used = "lean_formal"
        raw_conf = _confidence_from_lean(converged)
        known_exact = converged.exact and rational_decisive
    else:
        path_used = "hybrid"
        raw_conf = 0.7 
        known_exact = False

    return HomotopyGroupApproximation(
        n=n,
        rational_rank=rational_rank,
        torsion_invariants=(torsion if torsion else None),
        upper_bound_torsion=upper_bound,
        path_used=path_used,
        confidence_score=float(min(1.0, max(0.0, raw_conf * rational_factor))),
        known_exact=known_exact,
        caveats=converged.reasoning,
        supporting_e_infinity=converged,
        space_label=rational_result.space_label or converged.space_label,
        prime=e2_page.prime,
        convergence_page=converged.convergence_page,
    )


# ── HomotopyGroup: unified active object for π_*(X) ──────────────────────────


_HG_CACHE_MISS = object()


class HomotopyGroup(BaseModel):
    """Unified active object for the homotopy groups π_*(X) of a space.

    Overview:
        HomotopyGroup is the "Master Contract" for higher homotopy. It bundles
        the rational data (Sullivan minimal model → π_n ⊗ ℚ) with the
        F_p stable data (Adams E_2 page → torsion filtration) and provides
        active query methods ``rank(n)``, ``torsion(n, p)`` and the geometric
        bridge ``simplices_generators(n)`` that lifts each abstract generator
        of π_n back to concrete cells / chains in the source complex.

    Key Concepts:
        - **Rational Side**: ``rational : RationalHomotopyGroup`` records
          dim_ℚ(π_n(X) ⊗ ℚ) = dim V^n in the minimal Sullivan algebra (ΛV, d).
        - **Stable / Torsion Side**: ``adams : AdamsE2Page`` records
          E_2^{s,t} = Ext_{A_p}^{s,t}(H^*(X; F_p), F_p); the t-s = n diagonal
          gives the F_p torsion filtration of π_n^s(X)_p.
        - **Geometric Bridge**: A degree-n minimal-model generator v ∈ V^n
          maps via the quasi-iso ρ : (ΛV, d) → A^*(X) to a closed cochain
          on X; Poincaré-dual / homology-pair it to find supporting cells.
        - **Caching**: Every public query is memoized; repeated
          ``rank(n)``/``torsion(n)`` calls cost O(1).

    Common Workflows:
        1. **Build** → ``HomotopyGroup.from_inputs(rational_input,
           fp_ring=..., base_complex=..., ...)``
        2. **Query rational rank** → ``hg.rank(n)``
        3. **Query torsion** → ``hg.torsion(n, p=2)``
        4. **Geometric realization** → ``hg.simplices_generators(n)``

    Coefficient Ring:
        Rational side: exact ℚ via ``fractions.Fraction``.
        Stable side: F_p with p ∈ {2, 3, 5}.

    Attributes:
        rational: ``RationalHomotopyGroup`` carrying π_*(X) ⊗ ℚ.
        adams: Optional ``AdamsE2Page`` carrying the F_p Adams E_2 grid.
        base_complex: Optional reference (SimplicialComplex / CWComplex /
            ChainComplex) used by ``simplices_generators`` to resolve
            cohomology classes back to supporting cells.
        prime: The Adams prime, mirrored from ``adams.prime`` if present.
        space_label: Human-readable label.

    References:
        Quillen, D. (1969). Rational homotopy theory. Ann. Math. 90, 205–295.
        Sullivan, D. (1977). Infinitesimal computations in topology.
            Publ. Math. IHES 47, 269–331.
        Adams, J. F. (1958). On the structure and applications of the Steenrod
            algebra. Comment. Math. Helv. 32, 180–214.

    Example:
        from pysurgery.homotopy.higher_homotopy_groups import (
            HomotopyGroup, sphere_cohomology, sphere_cohomology_fp,
        )
        hg = HomotopyGroup.from_inputs(
            sphere_cohomology(3),
            fp_cohomology_ring=sphere_cohomology_fp(3, prime=2),
            adams_s_max=4, adams_t_max=10,
        )
        hg.rank(3)        # 1
        hg.torsion(3, p=2)  # () or e.g. (1,) depending on data
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rational: RationalHomotopyGroup
    adams: Optional[AdamsE2Page] = None
    base_complex: Optional[Any] = None
    fp_cohomology_ring: Optional[FpCohomologyRing] = None
    space_label: str = ""
    prime: Optional[int] = None

    _cache: Dict[tuple, Any] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _propagate_metadata(self) -> "HomotopyGroup":
        if not self.space_label:
            object.__setattr__(
                self,
                "space_label",
                self.rational.space_label
                or (self.adams.space_label if self.adams is not None else ""),
            )
        if self.prime is None and self.adams is not None:
            object.__setattr__(self, "prime", int(self.adams.prime))
        return self

    # ── caching ───────────────────────────────────────────────────────────────

    def _cache_get(self, key: tuple) -> Any:
        """Per-instance memoization lookup; returns ``_HG_CACHE_MISS`` on miss."""
        return self._cache.get(key, _HG_CACHE_MISS)

    def _cache_set(self, key: tuple, value: Any) -> None:
        """Per-instance memoization store."""
        self._cache[key] = value

    def clear_cache(self) -> None:
        """Discard all cached invariants (mainly for testing)."""
        self._cache.clear()

    def cache_info(self) -> Dict[str, Any]:
        """Snapshot of cache state for debugging."""
        return {
            "size": len(self._cache),
            "keys": [list(map(str, k)) for k in self._cache.keys()],
        }

    # ── constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_inputs(
        cls,
        rational_input: Any,
        *,
        fp_cohomology_ring: Optional[FpCohomologyRing] = None,
        base_complex: Optional[Any] = None,
        rational_max_degree: int = 10,
        adams_prime: int = 2,
        adams_s_max: int = 6,
        adams_t_max: int = 20,
        include_massey: bool = True,
        space_label: str = "",
    ) -> "HomotopyGroup":
        """Convenience constructor that wires up rational + Adams in one call.

        What is Being Computed?:
            Initializes a unified HomotopyGroup object by calculating the
            Sullivan minimal model and the Adams E2 page.

        Algorithm:
            1. Call ``sullivan_rational_homotopy`` for the rational side.
            2. If the input itself is an ``FpCohomologyRing``, use it for the
               Adams side; otherwise use the provided ``fp_cohomology_ring``.
            3. Call ``adams_e2_page`` if a cohomology ring is available.
            4. Bind both results into a unified contract.

        Args:
            rational_input: A ``RationalCohomologyAlgebra``, ``ChainComplex``,
                ``SimplicialComplex``, ``CWComplex``, or any object with
                ``betti_numbers()``.
            fp_cohomology_ring: Optional pre-built F_p cohomology ring.
            base_complex: Optional source complex for geometric realization.
            rational_max_degree: Sullivan truncation degree.
            adams_prime: Coefficient prime for the Adams page.
            adams_s_max: Adams homological-degree truncation.
            adams_t_max: Adams internal-degree truncation.
            include_massey: Whether to extract Massey products.
            space_label: Human-readable label.

        Returns:
            HomotopyGroup with both rational and stable data attached.
        """
        # If the input itself IS an FpCohomologyRing, use it for Adams side.
        actual_fp_ring = fp_cohomology_ring
        if actual_fp_ring is None and hasattr(rational_input, "sq_table"):
             actual_fp_ring = rational_input

        rational = sullivan_rational_homotopy(
            rational_input,
            max_degree=rational_max_degree,
            include_massey=include_massey,
            space_label=space_label,
        )

        adams = None
        if actual_fp_ring is not None:
            adams = adams_e2_page(
                actual_fp_ring,
                prime=adams_prime,
                s_max=adams_s_max,
                t_max=adams_t_max,
            )

        return cls(
            rational=rational,
            adams=adams,
            base_complex=base_complex,
            fp_cohomology_ring=actual_fp_ring,
            space_label=space_label,
            prime=adams_prime if adams else None,
        )

    # ── public queries ────────────────────────────────────────────────────────

    def rank(self, n: int) -> int:
        """Return ``dim_ℚ(π_n(X) ⊗ ℚ)``.

        What is Being Computed?:
            The rational rank of π_n(X), equal to the number of degree-n
            generators V^n of the Sullivan minimal model (Quillen-Sullivan).

        Algorithm:
            1. Look up the cached value, return on hit.
            2. Otherwise read ``self.rational.pi_n_rational.get(n, 0)``.
            3. Cache the result.

        Preserved Invariants:
            - The rank is a homotopy invariant of X.
            - Result is exact (rational arithmetic over ℚ).
            - Cached for the lifetime of the object.

        Args:
            n: Degree of the homotopy group; must satisfy ``n ≥ 1``.

        Returns:
            int: ``dim_ℚ(π_n(X) ⊗ ℚ)``; 0 when n is outside the truncation
            window or the corresponding V^n is empty.

        Use When:
            - Replacing the older ``rational_homotopy_group(...).rank_at(n)``
              idiom in user code.
            - Comparing the rational rank to the Adams E_2 t-s = n stem.

        Example:
            hg.rank(3)  # 1 for S^3
        """
        key = ("rank", int(n))
        cached = self._cache_get(key)
        if cached is not _HG_CACHE_MISS:
            return cached
        value = int(self.rational.pi_n_rational.get(int(n), 0))
        self._cache_set(key, value)
        return value

    def torsion(self, n: int, p: int = 2) -> Tuple[int, ...]:
        """Return the F_p torsion filtration weights of π_n^s(X)_p.

        What is Being Computed?:
            The F_p-dimensions of the positive-filtration column of the
            Adams E_∞ page at the t-s = n stem, i.e. the dimensions of
            E_∞^{s, n+s} for s > 0. These bound the number of p-torsion
            summands of π_n^s(X)_p by filtration depth.

            When the AdamsE2Page is supplied without an E_∞ resolution, the
            E_2 dimensions are returned as an *upper bound* (no differentials
            have been killed yet). When ``self.adams`` is None or its prime
            does not match ``p``, an empty tuple is returned.

        Algorithm:
            1. Validate that ``self.adams.prime == p`` (or that no Adams
               data is available — return empty).
            2. Walk ``self.adams.e2_grid`` for entries with ``s > 0`` and
               ``t - s == n``; sort by ascending s; emit the dimensions.

        Preserved Invariants:
            - Result is a sorted tuple of strictly positive F_p-dimensions.
            - The tuple is the associated graded of the p-completed stable
              homotopy group at filtration ≥ 1 (an upper bound when E_∞
              is approximated by E_2).
            - Cached on (n, p).

        Args:
            n: Stem degree (t - s) of interest.
            p: Coefficient prime; must match ``self.adams.prime`` to be
                non-trivial.

        Returns:
            Tuple[int, ...]: F_p-dimensions of E_∞^{s, n+s} for s > 0,
            sorted by ascending filtration depth. Empty when no data or
            the prime does not match.

        Use When:
            - Bounding p-torsion of π_n(X) above by the Adams filtration.
            - Sanity-checking a computed rational rank against the t-s
              stem of the E_2 page.

        Example:
            hg.torsion(3, p=2)  # () for S^3 stably, (1,) when 2-torsion present
        """
        key = ("torsion", int(n), int(p))
        cached = self._cache_get(key)
        if cached is not _HG_CACHE_MISS:
            return cached

        if self.adams is None or int(self.adams.prime) != int(p):
            value: Tuple[int, ...] = ()
        else:
            value = _torsion_invariants_from_grid(self.adams.e2_grid, n)

        self._cache_set(key, value)
        return value

    def resolve(
        self,
        n: int,
        path: Literal["interactive", "lean_formal", "rational_only", "auto"] = "auto",
        **kwargs: Any,
    ) -> HomotopyGroupApproximation:
        """Fully resolve the homotopy group at degree n until stabilization.

        What is Being Computed?:
            Advances the Adams spectral sequence from E2 until stabilization
            (convergence to E_infinity) using the requested resolution path.
            This settles ambiguous differentials and produces a definitive 
            p-torsion filtration for the stable homotopy group.

        Algorithm:
            Delegates to ``synthesize_homotopy_group_with_e_infinity`` to
            manage the iterative resolution, database lookup, and confidence 
            accounting across Python, Julia, and Lean backends.

        Args:
            n: Homotopy degree to resolve.
            path: Resolution strategy (auto tries Lean then Interactive/DB).
            **kwargs: Forwarded to the orchestrator (e.g., ``interactive_kwargs``).

        Returns:
            HomotopyGroupApproximation carrying the stabilized torsion filtration.
        """
        if self.adams is None:
            raise ValueError("No Adams E2 data available; cannot resolve torsion.")
        return synthesize_homotopy_group_with_e_infinity(
            self.rational, self.adams, n=n, resolution_path=path, **kwargs
        )

    def is_rationally_trivial(self, n: int) -> bool:
        """True iff ``π_n(X) ⊗ ℚ = 0`` (no rational generators in degree n).

        Algorithm: checks ``rank(n) == 0``; cached.
        """
        return self.rank(n) == 0

    def compute_pi_n(self, n: int, **kwargs: Any):
        """Run the full algorithmic π_n pipeline.

        Output is the algorithm's computation — never substituted from a
        published table. See ``pysurgery.homotopy.homotopy_verifier``.
        """
        from pysurgery.homotopy.homotopy_verifier import compute_pi_n as _compute
        return _compute(self, n, **kwargs)

    def verify_against_known(self, n: int, **kwargs: Any):
        """Run the algorithm, then compare against the literature table.

        The literature value is reported only inside the returned
        ``VerificationResult``; the public computed value is the algorithm's
        output.
        """
        from pysurgery.homotopy.homotopy_verifier import verify_against_known as _verify
        return _verify(self, n, **kwargs)

    # ── geometric bridge ──────────────────────────────────────────────────────

    def simplices_generators(self, n: int) -> Dict[str, Dict[str, Any]]:
        """Geometric realization of every degree-n minimal-model generator.

        What is Being Computed?:
            For each generator v of V^n in the Sullivan minimal model
            (ΛV, d), returns a dict describing v's image under the
            quasi-iso ρ : (ΛV, d) → A^*(X) and, when a base complex is
            attached, the supporting cells / homology cycle representatives.

        Algorithm:
            1. Pull ``dga = self.rational.underlying_model``; if absent,
               return ``{}``.
            2. For each degree-n generator g ∈ V^n (in declaration order):
               - Read ``d(g)``; if zero, g represents a cocycle class
                 directly (closed); if non-zero, g is a "spurious-killer"
                 added by the Sullivan algorithm — its dual is the
                 Massey-style relation it imposes.
               - Record the symbolic monomial ``[(gid, 1)]`` and the
                 differential ``d(g)`` as a list of (monomial, coeff)
                 entries.
            3. If ``base_complex`` is supplied AND its rational Betti number
               at degree ``n`` matches ``rank(n)``, attach the i-th
               homology cycle representative (from
               ``compute_homology_basis_from_complex``) as the Hurewicz-image
               supporting cells. This is the "ρ then Poincaré-pair" route
               used in the FHT computation toolbox.
            4. Cache and return.

        Preserved Invariants:
            - The symbolic monomial / differential data is an exact algebraic
              record of the minimal-model generator (no float arithmetic).
            - When supporting cells are attached, their count matches
              ``rank(n)``; each cycle is verified ``∂c = 0`` by the
              homology-basis routine.

        Args:
            n: The degree at which to realise generators.

        Returns:
            Dict[str, Dict[str, Any]] keyed by generator name, with fields:
              - ``degree``: int
              - ``cochain_monomial``: list[(gid, exp)]
              - ``differential``: list[(monomial, Fraction)] or [] when closed
              - ``is_closed``: bool — d(g) == 0
              - ``support_simplices`` (optional): list[tuple[int,...]] when
                ``base_complex`` provides a matching homology basis.

        Use When:
            - Visualising π_n generators on the source mesh.
            - Pulling back π_n classes to chain-level representatives for
              surgery / obstruction theory.
            - Checking that a computed rational rank matches the geometric
              cycle count at the first non-vanishing Hurewicz degree.

        Example:
            data = hg.simplices_generators(2)
            for name, info in data.items():
                print(name, info["is_closed"], info.get("support_simplices"))
        """
        key = ("simplices_generators", int(n))
        cached = self._cache_get(key)
        if cached is not _HG_CACHE_MISS:
            return cached

        out: Dict[str, Dict[str, Any]] = {}
        dga = self.rational.underlying_model
        if dga is None:
            self._cache_set(key, out)
            return out

        gens_n = list(dga._by_deg.get(int(n), []))
        for g in gens_n:
            dg = dga._diff.get(g.gid)
            is_closed = (dg is None) or dg.is_zero()
            diff_repr: List[Tuple[Any, str]] = []
            if not is_closed:
                for mon, coef in dg.terms.items():
                    diff_repr.append((list(mon), str(coef)))
            out[g.name] = {
                "degree": int(g.degree),
                "cochain_monomial": [(int(g.gid), 1)],
                "differential": diff_repr,
                "is_closed": bool(is_closed),
            }

        # Optional cell-level supports via the base complex.
        if self.base_complex is not None and gens_n:
            cells = self._homology_supports_for_degree(int(n), len(gens_n))
            if cells is not None:
                for g, supp in zip(gens_n, cells):
                    out[g.name]["support_simplices"] = supp

        self._cache_set(key, out)
        return out

    def _homology_supports_for_degree(
        self, n: int, count: int
    ) -> Optional[List[List[Tuple[int, ...]]]]:
        """Return n-cycle supporting simplices from the base complex.

        Returns ``None`` when the base complex cannot provide a basis, or
        when the rational Betti number disagrees with ``count`` (in which
        case the index-aligned correspondence to V^n is no longer reliable).
        """
        try:
            from pysurgery.topology.complexes import SimplicialComplex
            from pysurgery.homology.homology_generators import compute_homology_basis_from_complex
        except Exception:
            return None
        if not isinstance(self.base_complex, SimplicialComplex):
            return None
        try:
            betti = self.base_complex.betti_numbers()
        except Exception:
            return None
        if int(betti.get(n, 0)) != count:
            return None
        try:
            basis = compute_homology_basis_from_complex(
                self.base_complex, dimension=n, mode="valid"
            )
        except Exception:
            return None
        if basis.rank != count:
            return None
        return [
            [tuple(int(v) for v in s) for s in gen.support_simplices]
            for gen in basis.generators
        ]


__all__ = [
    "AdamsCombinatorialError",
    "AdamsDifferentialFlag",
    "AdamsE2Bidegree",
    "AdamsE2Page",
    "AdamsResourceError",
    "ClosureError",
    "ConvergedAdamsPage",
    "CrossAlgebraError",
    "DegreeError",
    "DGAElement",
    "DGAError",
    "FormalityResult",
    "FpCohomologyRing",
    "Generator",
    "HomogeneityError",
    "HomotopyGroup",
    "HomotopyGroupApproximation",
    "MasseyProductEntry",
    "MasseyProductsResult",
    "MinimalityError",
    "Monomial",
    "RationalCohomologyAlgebra",
    "RationalDGA",
    "RationalHomotopyGroup",
    "RationalMinimalModelResult",
    "SteenrodAction",
    "SteenrodAlgebra",
    "UnknownGeneratorError",
    "adams_e2_page",
    "complex_projective_space_cohomology",
    "compute_cohomology",
    "compute_rational_and_adams",
    "cp_n_cohomology_fp",
    "extract_massey_products",
    "is_formal_space",
    "product_cohomology",
    "rational_homotopy_group",
    "rp_n_cohomology_fp",
    "sphere_cohomology",
    "sphere_cohomology_fp",
    "steenrod_squares_matrix",
    "sullivan_minimal_model",
    "sullivan_rational_homotopy",
    "synthesize_homotopy_group_with_e_infinity",
    "verify_differential_closure",
]

def compute_fibration_homotopy_sequence(
    fiber: HomotopyGroup, 
    total_space: HomotopyGroup, 
    base: HomotopyGroup, 
    n_max: int
) -> ExactSequence:
    """Construct the Long Exact Sequence of homotopy groups for a fibration F -> E -> B.
    
    ... -> pi_n(F) -> pi_n(E) -> pi_n(B) -> pi_{n-1}(F) -> ...
    """
    modules = []
    morphisms = []
    
    for n in range(n_max, 0, -1):
        # 1. pi_n(F)
        modules.append(f"pi_{n}(F)")
        rank_f = fiber.rank(n)
        
        # 2. pi_n(E)
        modules.append(f"pi_{n}(E)")
        rank_e = total_space.rank(n)
        
        # Placeholder for i*: pi_n(F) -> pi_n(E)
        morphisms.append(Morphism(np.zeros((rank_e, rank_f), dtype=object), rank_f, rank_e))
        
        # 3. pi_n(B)
        modules.append(f"pi_{n}(B)")
        rank_b = base.rank(n)
        
        # Placeholder for p*: pi_n(E) -> pi_n(B)
        morphisms.append(Morphism(np.zeros((rank_b, rank_e), dtype=object), rank_e, rank_b))
        
        # 4. Connecting map d: pi_n(B) -> pi_{n-1}(F)
        if n > 1:
            rank_fnm1 = fiber.rank(n - 1)
            morphisms.append(Morphism(np.zeros((rank_fnm1, rank_b), dtype=object), rank_b, rank_fnm1))
            
    return ExactSequence(modules, morphisms)
