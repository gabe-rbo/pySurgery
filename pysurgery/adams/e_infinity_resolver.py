"""Shared types for the E∞ resolver (see RFC-einf-v2).

Overview:
    Defines the contracts shared by both Path A (interactive) and Path B (Lean
    formal): ``UserVerifiedDifferential``, ``ResolvingPage``, ``ConvergedAdamsPage``,
    ``ResolverProtocol``, and the domain-specific exceptions.

Layer architecture:
    Consumed by ``interactive_resolver.py`` (Path A) and ``lean_resolver.py``
    (Path B).  ``higher_homotopy_groups.py`` reads ``ConvergedAdamsPage`` to
    build ``HomotopyGroupApproximation``.

References:
    Adams, J. F. (1958). On the structure and applications of the Steenrod algebra.
        Comment. Math. Helv. 32, 180–214.
    Bruner, R. R. (1993). Ext in the nineties. Contemp. Math. 146, AMS.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Tuple
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from pysurgery.adams.spectral_sequence import AdamsDifferentialFlag
from pysurgery.core.exceptions import SurgeryError
from pysurgery.core.foundations import CONTRACT_VERSION
from pysurgery.core.theorem_tags import (
    ADAMS_EINF_INTERACTIVE,
)

def _decision_key(r: int, src: Tuple[int, int], tgt: Tuple[int, int]) -> str:
    """Stable key for an Adams differential decision (r, source, target)."""
    return f"d{r}_{src[0]}_{src[1]}->{tgt[0]}_{tgt[1]}"


# ── Exceptions ───────────────────────────────────────────────────────────────


class InteractiveConsistencyError(SurgeryError):
    """Raised when a new user verification contradicts existing ones.

    Overview:
        Carries the conflicting pair so the caller can surface both sides
        to the user before deciding which to retract.
    """


class LeanEnvironmentError(SurgeryError):
    """Raised when the Lean 4 toolchain cannot be found or invoked."""


# ── UserVerifiedDifferential ─────────────────────────────────────────────────


class UserVerifiedDifferential(BaseModel):
    """A single human-supplied verdict on one Adams differential.

    Overview:
        Records the user's decision (zero / nonzero / skip), the raw input
        for auditing, provenance metadata, and a confidence score.  The
        ``exact`` field is permanently ``False``; human input never constitutes
        a formal proof.

    Attributes:
        r: Adams page index of the differential (E_r → E_{r+1}).
        bidegree_source: (s, t) of the source.
        bidegree_target: (s+r, t+r-1) of the target.
        decision: The user's verdict.
        decision_input_raw: Verbatim stdin echo for audit trail.
        timestamp: UTC time of the decision.
        user_id: OS username or override.
        proof_reference: Paper / formula / personal note.
        user_confidence: Subjective probability ∈ [0, 1].
        exact: Always False — user input is not a formal proof.
    """

    r: int
    bidegree_source: Tuple[int, int]
    bidegree_target: Tuple[int, int]

    decision: Literal["zero", "nonzero", "skip"]
    decision_input_raw: str

    timestamp: datetime
    user_id: str
    proof_reference: str
    user_confidence: float = Field(ge=0.0, le=1.0)

    exact: bool = False
    theorem_tag: str = ADAMS_EINF_INTERACTIVE
    contract_version: str = CONTRACT_VERSION

    def decision_ready(self) -> bool:
        return self.decision in ("zero", "nonzero") and self.user_confidence > 0.0


# ── ResolvingPage ─────────────────────────────────────────────────────────────


class ResolvingPage(BaseModel):
    """Snapshot of E_r during iterative resolution.

    Overview:
        An internal scaffold tracking the grid dimensions at page level r,
        which flags are still unresolved (``open_flags``), and the accumulated
        decisions across all pages up to r (``closed_decisions``).

    Attributes:
        r: Current page index.
        grid: (s, t) → F_p-dimension of E_r^{s,t}.
        open_flags: Ambiguous d_r flags not yet decided at this level.
        closed_decisions: Accumulated decisions from all levels <= r.
            Key: (r, source, target).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    r: int
    grid: Dict[Tuple[int, int], int]
    open_flags: List[AdamsDifferentialFlag] = Field(default_factory=list)
    closed_decisions: Dict[Tuple[int, Tuple[int, int], Tuple[int, int]], Literal["zero", "nonzero"]] = Field(
        default_factory=dict
    )


# ── ConvergedAdamsPage ────────────────────────────────────────────────────────


class ConvergedAdamsPage(BaseModel):
    """The fully resolved E_∞ page of the Adams spectral sequence.

    Overview:
        Produced by ``InteractiveAdamsResolver`` (Path A) or
        ``LeanFormalAdamsResolver`` (Path B).  Carries the complete audit
        trail: page history, user verifications, and Lean proof attempts.

    Attributes:
        e_infinity_grid: (s, t) → F_p-dimension of E_∞^{s,t}.
        page_history: Sequence of ResolvingPage snapshots E_2, E_3, …, E_∞.
        convergence_page: Smallest r with E_r = E_∞.
        user_verifications: List of UserVerifiedDifferential (Path A).
        lean_attempts: List of Any (Path B).
        path_used: Which resolver produced this page.
        exact: True iff path_used == "lean_formal" AND every flag is proven.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    space_label: str
    prime: Literal[2, 3, 5]
    s_max: int
    t_max: int

    e_infinity_grid: Dict[Tuple[int, int], int]
    page_history: List[ResolvingPage] = Field(default_factory=list)
    convergence_page: int

    user_verifications: List[UserVerifiedDifferential] = Field(default_factory=list)
    lean_attempts: List[Any] = Field(default_factory=list)

    path_used: Literal["none", "interactive", "lean_formal", "hybrid"]
    status: Literal["success", "truncated", "undecidable", "inconclusive"]
    reasoning: str

    exact: bool
    theorem_tag: str
    contract_version: str = CONTRACT_VERSION

    def decision_ready(self) -> bool:
        return self.status == "success"


# ── ResolverProtocol ─────────────────────────────────────────────────────────


@runtime_checkable
class ResolverProtocol(Protocol):
    """Structural interface shared by Path A and Path B resolvers."""

    def identify_ambiguous_differentials(
        self, page: ResolvingPage
    ) -> list[AdamsDifferentialFlag]: ...

    def compute_next_page(
        self,
        page: ResolvingPage,
        verifications: list[Any],
    ) -> ResolvingPage: ...

    def resolve_e_infinity(self) -> ConvergedAdamsPage: ...


__all__ = [
    "ConvergedAdamsPage",
    "InteractiveConsistencyError",
    "LeanEnvironmentError",
    "ResolvingPage",
    "ResolverProtocol",
    "UserVerifiedDifferential",
]
