"""Sullivan minimal models — verification-layer integration layer.

Overview:
    Bridges the rational-homotopy primitives in
    :mod:`pysurgery.homotopy.rational_homotopy` with the existing verification-layer
    infrastructure of pySurgery: chain complexes, algebraic Poincaré
    complexes, and spectral sequences.  This module exposes a single
    high-level entry point — :func:`sullivan_rational_homotopy` — which
    accepts any of the standard verification-layer inputs and returns a strict,
    exact Pydantic contract describing π_n(X) ⊗ ℚ.

    All arithmetic flows through ``rational_homotopy`` (exact ℚ via
    :class:`fractions.Fraction`); no floating-point operations are
    introduced here, so every emitted contract carries ``exact=True``.

Key Concepts:
    - **Sullivan minimal model**: A free graded-commutative DGA
      ``(ΛV, d)`` over ℚ quasi-isomorphic to the rational cochain
      algebra of X.  When X is simply connected of finite ℚ-type the
      Quillen–Sullivan theorem gives ``π_n(X) ⊗ ℚ ≅ V^n``.
    - **verification-layer entry points**:
        * :class:`pysurgery.topology.complexes.ChainComplex`
        * :class:`pysurgery.topology.complexes.SimplicialComplex`
        * :class:`pysurgery.topology.complexes.CWComplex`
        * :class:`pysurgery.AlgebraicPoincareComplex`
        * :class:`RationalCohomologyAlgebra` or a plain
          ``Mapping[int, int]`` of rational Betti numbers.
    - **Spectral-sequence cross-check**: The companion routine
      :func:`cross_validate_with_serre` uses the Serre exact-couple
      framework to recover the rational total-space cohomology of a
      product fibration and feeds the result back into the Sullivan
      pipeline.  Agreement is a structural integrity check on the
      verification-layer hookup.

Coefficient Ring:
    ℚ exclusively.  Inputs over ℤ or ℤ/pℤ are tensored to ℚ via
    ``betti_numbers``; torsion is intentionally discarded because it
    is invisible to rational homotopy.

References:
    Félix, Y., Halperin, S., & Thomas, J.-C. (2001).
        Rational Homotopy Theory. Springer GTM 205.
    Quillen, D. (1969). Rational homotopy theory.
        Annals of Mathematics, 90(2), 205–295.
    Sullivan, D. (1977). Infinitesimal computations in topology.
        Publ. Math. IHES, 47, 269–331.
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Union,
)

from pydantic import BaseModel, ConfigDict, Field

from pysurgery.topology.complexes import ChainComplex, CWComplex, SimplicialComplex
from pysurgery.core.exceptions import SurgeryError
from pysurgery.core.foundations import CONTRACT_VERSION
from pysurgery.homotopy.rational_homotopy import (
    DGAElement,
    DGAError,
    Generator,
    RationalCohomologyAlgebra,
    RationalDGA,
    RationalMinimalModelResult,
    complex_projective_space_cohomology,
    product_cohomology,
    sphere_cohomology,
    sullivan_minimal_model,
)

__all__ = [
    "RationalDGA",
    "Generator",
    "DGAElement",
    "DGAError",
    "RationalCohomologyAlgebra",
    "RationalHomotopyGroup",
    "RationalHomotopyGroupAtDegree",
    "RationalHomotopyProfile",
    "SullivanIntegrationError",
    "sullivan_rational_homotopy",
    "cross_validate_with_serre",
    "sphere_cohomology",
    "complex_projective_space_cohomology",
    "product_cohomology",
]


# ── Constants ────────────────────────────────────────────────────────────────

PHASE2_THEOREM_TAG = "rational.sullivan_models.phase2_integration"
PI_N_THEOREM_TAG = "rational.quillen_sullivan.pi_n"


# ── Exceptions ────────────────────────────────────────────────────────────────


class SullivanIntegrationError(SurgeryError):
    """verification-layer hookup error (unsupported source, invalid Betti profile, …)."""


# ── Pydantic contracts ────────────────────────────────────────────────────────


class RationalHomotopyGroupAtDegree(BaseModel):
    """Rational homotopy group π_n(X) ⊗ ℚ at a single degree.

    Attributes:
        degree: Homotopy degree n ≥ 1.
        rank:   ``dim_ℚ(π_n(X) ⊗ ℚ)`` = number of degree-n indecomposables
                in the Sullivan minimal model.
        generator_names: Names of the V^n generators (debugging aid).
        theorem_tag: Stable identifier of the underlying theorem.
        contract_version: Result-schema version string.
        exact: Always ``True`` (ℚ arithmetic, no float).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    degree: int = Field(ge=1)
    rank: int = Field(ge=0)
    generator_names: Tuple[str, ...] = ()
    theorem_tag: str = PI_N_THEOREM_TAG
    contract_version: str = CONTRACT_VERSION
    exact: Literal[True] = True

    def decision_ready(self) -> bool:
        """Always ``True`` — every group in this contract is exact."""
        return self.exact


RationalHomotopyGroup = RationalHomotopyGroupAtDegree


class RationalHomotopyProfile(BaseModel):
    """Profile of π_n(X) ⊗ ℚ across all degrees up to a truncation bound.

    Attributes:
        groups: Tuple of :class:`RationalHomotopyGroupAtDegree`, one per non-zero
            rational homotopy degree (sorted ascending by ``degree``).
        truncation_degree: Maximum degree explored by the Sullivan algorithm.
        cohomology_iso: ``True`` iff H(ΛV, d) ≅ H*(X; ℚ) up to truncation.
        is_formal: ``True`` iff every generator of (ΛV, d) has zero differential
            (the d = 0 strong-formality criterion).
        source: Which verification-layer entry point produced this profile.
        status: ``"success"`` or ``"inconclusive"``.
        reasoning: Human-readable summary suitable for logging.
        theorem_tag: Stable theorem identifier for this contract.
        contract_version: Result-schema version string.
        exact: Always ``True``.
    """

    model_config = ConfigDict(extra="forbid")

    groups: Tuple[RationalHomotopyGroup, ...] = ()
    truncation_degree: int = Field(ge=0)
    cohomology_iso: bool
    is_formal: bool
    source: Literal[
        "chain_complex",
        "cw_complex",
        "simplicial_complex",
        "algebraic_poincare",
        "cohomology_algebra",
        "betti_mapping",
        "spectral_sequence",
    ]
    status: Literal["success", "inconclusive"]
    reasoning: str
    theorem_tag: str = PHASE2_THEOREM_TAG
    contract_version: str = CONTRACT_VERSION
    exact: Literal[True] = True

    def by_degree(self) -> Dict[int, int]:
        """Return ``{n: dim π_n(X) ⊗ ℚ}`` as a plain dict."""
        return {g.degree: g.rank for g in self.groups}

    def decision_ready(self) -> bool:
        """``True`` iff the run succeeded *and* H(ΛV) ≅ H*(X; ℚ) was verified."""
        return self.exact and self.status == "success" and self.cohomology_iso


# ── Helpers ───────────────────────────────────────────────────────────────────


def _betti_from_complex(
    obj: Union[ChainComplex, CWComplex, SimplicialComplex],
    max_degree: int,
) -> Dict[int, int]:
    """Extract ℚ-Betti numbers from any verification-layer complex up to ``max_degree``.

    Algorithm:
        Calls ``obj.betti_numbers()`` (rank over the underlying ring; for ℤ
        and ℚ this coincides with ``dim_ℚ H_n(X; ℚ)``).  The result is
        filtered to non-negative degrees ≤ ``max_degree`` with non-zero rank.

    Preserved Invariants:
        Torsion is silently discarded — rational homotopy theory cannot see
        it, by Quillen–Sullivan.
    """
    raw = obj.betti_numbers()
    return {
        int(n): int(b)
        for n, b in raw.items()
        if 0 <= int(n) <= max_degree and int(b) > 0
    }


def _validate_simply_connected(betti: Mapping[int, int]) -> Optional[str]:
    """Return a reason string if the input cannot be simply connected; else None."""
    if betti.get(0, 0) != 1:
        return f"β_0 = {betti.get(0, 0)} ≠ 1; space is disconnected."
    if betti.get(1, 0) != 0:
        return (
            f"β_1 = {betti.get(1, 0)} > 0; Quillen–Sullivan applies only to "
            "simply-connected spaces."
        )
    return None


def _profile_from_result(
    result: RationalMinimalModelResult,
    *,
    source: str,
    extra_reasoning: str = "",
) -> RationalHomotopyProfile:
    """Convert a :class:`RationalMinimalModelResult` to a verification-layer profile."""
    dga = result.minimal_model

    groups: List[RationalHomotopyGroupAtDegree] = []
    for n in sorted(result.pi_n_rational):
        rank = result.pi_n_rational[n]
        if dga is not None:
            names = tuple(g.name for g in dga.all_generators() if g.degree == n)
        else:
            names = ()
        groups.append(
            RationalHomotopyGroupAtDegree(degree=n, rank=rank, generator_names=names)
        )

    reasoning = result.reasoning
    if extra_reasoning:
        reasoning = f"{extra_reasoning} {reasoning}"

    return RationalHomotopyProfile(
        groups=tuple(groups),
        truncation_degree=result.truncation_degree,
        cohomology_iso=result.cohomology_iso,
        is_formal=result.is_formal_model,
        source=source,  # type: ignore[arg-type]
        status=result.status,
        reasoning=reasoning,
    )


def _resolve_to_algebra(
    source: Any,
    max_degree: int,
) -> Tuple[RationalCohomologyAlgebra, str, str]:
    """Normalise any supported verification-layer input to a ``RationalCohomologyAlgebra``.

    Returns:
        ``(algebra, source_tag, extra_reasoning)``.

    Raises:
        SullivanIntegrationError: On unsupported types or invalid Betti data.
    """
    # Lazy import to avoid the top-level ``algebraic_poincare`` <→ core cycle.
    from pysurgery.homology.algebraic_poincare import AlgebraicPoincareComplex

    if isinstance(source, RationalCohomologyAlgebra):
        return source, "cohomology_algebra", ""

    if isinstance(source, AlgebraicPoincareComplex):
        cc = source.chain_complex
        n = source.dimension
        primal = _betti_from_complex(cc, max_degree)
        dual = _betti_from_complex(source.dual_complex(), max_degree)
        # Poincaré symmetry over ℚ: β_k = β_{n-k} for 0 ≤ k ≤ n.
        pd_violations: List[str] = []
        for k in range(n + 1):
            bk = primal.get(k, 0)
            bnk = primal.get(n - k, 0)
            if bk != bnk:
                pd_violations.append(f"β_{k}={bk} vs β_{n-k}={bnk}")
        # Cross-check primal vs. dual via H^k = H_{n-k}.
        dual_violations: List[str] = []
        for k in range(n + 1):
            if primal.get(k, 0) != dual.get(n - k, 0):
                dual_violations.append(
                    f"H_{k} rank {primal.get(k, 0)} ≠ H^{n-k} rank "
                    f"{dual.get(n - k, 0)}"
                )
        notes: List[str] = []
        if pd_violations:
            notes.append("PD-symmetry mismatch: " + "; ".join(pd_violations) + ".")
        if dual_violations:
            notes.append(
                "Primal/dual rank mismatch: " + "; ".join(dual_violations) + "."
            )
        extra = " ".join(notes)
        algebra = RationalCohomologyAlgebra(
            betti=primal,
            name=f"PoincareComplex(dim={n})",
            max_degree=max_degree,
        )
        return algebra, "algebraic_poincare", extra

    if isinstance(source, ChainComplex):
        primal = _betti_from_complex(source, max_degree)
        algebra = RationalCohomologyAlgebra(
            betti=primal, name="ChainComplex", max_degree=max_degree
        )
        return algebra, "chain_complex", ""

    if isinstance(source, CWComplex):
        primal = _betti_from_complex(source, max_degree)
        algebra = RationalCohomologyAlgebra(
            betti=primal, name="CWComplex", max_degree=max_degree
        )
        return algebra, "cw_complex", ""

    if isinstance(source, SimplicialComplex):
        primal = _betti_from_complex(source, max_degree)
        algebra = RationalCohomologyAlgebra(
            betti=primal, name="SimplicialComplex", max_degree=max_degree
        )
        return algebra, "simplicial_complex", ""

    if isinstance(source, Mapping):
        try:
            primal = {int(n): int(b) for n, b in source.items() if int(b) > 0}
        except (TypeError, ValueError) as exc:
            raise SullivanIntegrationError(
                f"Mapping input must be Dict[int, int]; got {source!r}."
            ) from exc
        algebra = RationalCohomologyAlgebra(
            betti=primal, name="BettiMapping", max_degree=max_degree
        )
        return algebra, "betti_mapping", ""

    raise SullivanIntegrationError(
        f"Unsupported source type: {type(source).__name__!r}. "
        "Expected ChainComplex, SimplicialComplex, CWComplex, "
        "AlgebraicPoincareComplex, RationalCohomologyAlgebra, or "
        "Mapping[int, int]."
    )


# ── Public API ────────────────────────────────────────────────────────────────


def sullivan_rational_homotopy(
    source: Union[
        ChainComplex,
        CWComplex,
        SimplicialComplex,
        "AlgebraicPoincareComplex",  # noqa: F821 (forward-ref via lazy import)
        RationalCohomologyAlgebra,
        Mapping[int, int],
    ],
    max_degree: int = 10,
) -> RationalHomotopyProfile:
    """Compute π_n(X) ⊗ ℚ via the Sullivan minimal model algorithm.

    What is Being Computed?:
        For a simply-connected space X of finite ℚ-type given through any
        verification-layer input (chain complex, CW complex, simplicial complex,
        algebraic Poincaré complex, cohomology algebra spec, or a raw
        Betti mapping), constructs the Sullivan minimal model (ΛV, d) up
        to ``max_degree`` and reads off ``dim_ℚ(π_n(X) ⊗ ℚ) = dim_ℚ V^n``.

    Algorithm:
        1. Resolve ``source`` to a :class:`RationalCohomologyAlgebra` by
           extracting ``betti_numbers()`` (for chain-style inputs) or by
           direct unwrapping (for cohomology-algebra / mapping inputs).
           Algebraic Poincaré inputs additionally run a structural
           PD-symmetry sanity check via the dual complex.
        2. Hand the algebra to :func:`sullivan_minimal_model` (the canonical
           Quillen–Sullivan engine in :mod:`rational_homotopy`).
        3. Wrap the resulting :class:`RationalMinimalModelResult` into a
           strict :class:`RationalHomotopyProfile` Pydantic contract,
           recording exactness, formality, and the originating verification-layer
           source kind.

    Preserved Invariants:
        - All arithmetic is exact ℚ (no floating-point).
        - The minimal model satisfies ``d² = 0`` and Sullivan minimality
          ``d(g) ∈ Λ^{≥2} V``; both are enforced inside the engine.
        - For simply-connected inputs the resulting ``H(ΛV, d) ≅ H*(X; ℚ)``
          isomorphism is verified up to ``max_degree``.

    Args:
        source: Any of:
            * :class:`ChainComplex` (any coefficient ring; rationalised
              via Betti numbers),
            * :class:`CWComplex` / :class:`SimplicialComplex`,
            * :class:`AlgebraicPoincareComplex`,
            * :class:`RationalCohomologyAlgebra`,
            * ``Mapping[int, int]`` of Betti numbers.
        max_degree: Truncation bound for the Sullivan algorithm.

    Returns:
        :class:`RationalHomotopyProfile` with ``exact=True``.

    Use When:
        - Computing rational π_n via verification-layer inputs without manually
          building a ``RationalCohomologyAlgebra``.
        - Cross-validating Poincaré-duality data against rational
          homotopy invariants.
        - Verifying that a chain-level model captures the rational
          homotopy type of a target space.

    Example:
        >>> from pysurgery.homotopy.sullivan_models import (
        ...     sullivan_rational_homotopy, sphere_cohomology,
        ... )
        >>> profile = sullivan_rational_homotopy(sphere_cohomology(3))
        >>> profile.by_degree()
        {3: 1}
        >>> profile.is_formal
        True

    References:
        Quillen, D. (1969). Annals of Mathematics, 90, 205–295.
        Sullivan, D. (1977). Publ. Math. IHES, 47, 269–331.
    """
    if max_degree < 2:
        raise SullivanIntegrationError(
            f"max_degree must be ≥ 2 (Sullivan starts at degree 2); got {max_degree}."
        )

    algebra, source_tag, extra_reasoning = _resolve_to_algebra(source, max_degree)

    # Quick simply-connected check on the resolved Betti data.  We avoid
    # raising — we still want to return a contract so callers can read the
    # ``status='inconclusive'`` and ``reasoning`` fields.
    sc_problem = _validate_simply_connected(algebra.betti)
    if sc_problem is not None:
        return RationalHomotopyProfile(
            groups=(),
            truncation_degree=max_degree,
            cohomology_iso=False,
            is_formal=False,
            source=source_tag,  # type: ignore[arg-type]
            status="inconclusive",
            reasoning=f"{extra_reasoning} {sc_problem}".strip(),
        )

    result = sullivan_minimal_model(algebra, max_degree=max_degree)
    return _profile_from_result(
        result, source=source_tag, extra_reasoning=extra_reasoning
    )


def cross_validate_with_serre(
    base_betti: Mapping[int, int],
    fibre_betti: Mapping[int, int],
    max_degree: int = 10,
) -> Tuple[RationalHomotopyProfile, Dict[Tuple[int, int], int]]:
    """Cross-validate Sullivan output against the Serre spectral sequence.

    What is Being Computed?:
        For the trivial fibration ``F → B × F → B`` (with B simply
        connected) the rational Serre SS collapses at E^2 — every
        differential vanishes — and we recover

            H_n(B × F; ℚ)  =  ⊕_{p+q=n}  H_p(B; ℚ) ⊗ H_q(F; ℚ)        (Künneth).

        This routine builds the
        :class:`pysurgery.spectral.spectral_sequences.SerreSpectralSequence`
        from the rational Betti numbers of B and F, drives it to
        convergence, sums the E^∞ ranks along total degree to recover the
        Künneth Betti profile of the total space, and re-feeds that
        profile into :func:`sullivan_rational_homotopy`.

    Algorithm:
        1. Wrap each Betti rank as a :class:`SpectralEntry` over ℚ.
        2. Construct ``SerreSpectralSequence(coefficient_ring='Q')``.
        3. Call ``converge()``.  No user differentials are supplied, so
           E^2 = E^∞.
        4. Aggregate the E^∞ ranks by total degree p + q to obtain the
           total-space Betti numbers.
        5. Run ``sullivan_rational_homotopy`` on that profile.

    Preserved Invariants:
        - All ranks are over ℚ; the exact-couple solver works in pure
          rank arithmetic over the rational field.
        - The collapse at E^2 is automatic: no transgressive
          differentials exist when both base and fibre have free
          rational cohomology and the action is trivial.

    Args:
        base_betti: ℚ-Betti numbers of B (must satisfy β_0 = 1, β_1 = 0).
        fibre_betti: ℚ-Betti numbers of F.
        max_degree: Truncation bound for the Sullivan run on B × F.

    Returns:
        Tuple ``(profile, e_infinity_ranks)`` where ``profile`` is a
        :class:`RationalHomotopyProfile` for B × F and
        ``e_infinity_ranks`` maps each bidegree ``(p, q)`` to its E^∞
        rank.

    Example:
        >>> from pysurgery.homotopy.sullivan_models import cross_validate_with_serre
        >>> # S^3 × S^5 via trivial fibration
        >>> base = {0: 1, 3: 1}
        >>> fibre = {0: 1, 5: 1}
        >>> profile, e_inf = cross_validate_with_serre(base, fibre, max_degree=10)
        >>> profile.by_degree()
        {3: 1, 5: 1}
        >>> profile.is_formal
        True

    References:
        Serre, J.-P. (1951). Annals of Mathematics, 54(3), 425–505.
        McCleary, J. (2001). A user's guide to spectral sequences.
            Cambridge University Press, ch. 5.
    """
    from pysurgery.spectral.spectral_sequences import SerreSpectralSequence, SpectralEntry

    base = {int(k): SpectralEntry(rank=int(v)) for k, v in base_betti.items() if v > 0}
    fibre = {
        int(k): SpectralEntry(rank=int(v)) for k, v in fibre_betti.items() if v > 0
    }

    ss = SerreSpectralSequence(
        base_homology=base,
        fibre_homology=fibre,
        coefficient_ring="Q",
        max_pages=max(8, max_degree + 2),
    )
    convergence = ss.converge()

    e_inf_ranks: Dict[Tuple[int, int], int] = {
        (int(p), int(q)): int(entry.rank)
        for (p, q), entry in convergence.e_infinity.items()
    }

    total_betti: Dict[int, int] = {}
    for (p, q), r in e_inf_ranks.items():
        if r <= 0:
            continue
        n = p + q
        total_betti[n] = total_betti.get(n, 0) + r

    profile = sullivan_rational_homotopy(total_betti, max_degree=max_degree)
    profile_with_source = profile.model_copy(update={"source": "spectral_sequence"})
    return profile_with_source, e_inf_ranks
