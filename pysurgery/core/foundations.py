from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field


CONTRACT_VERSION = "2026.04-phase10"


class CoverageMatrixEntry(BaseModel):
    """One scoped theorem-support claim in the package coverage matrix.

    Attributes:
        dimension_class (str): The dimension class (e.g., "2D", "3D").
        pi_family (str): The fundamental group family.
        theorem (str): The name of the theorem.
        theorem_tag (str): A unique tag for the theorem.
        status (str): Support status ("exact", "partial", or "unsupported").
        required_inputs (list[str]): List of required input types.
        notes (list[str]): Additional notes and caveats.
    """

    dimension_class: str
    pi_family: str
    theorem: str
    theorem_tag: str
    status: Literal["exact", "partial", "unsupported"]
    required_inputs: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class AnalyzerContract(BaseModel):
    """Contract metadata attached to analyzer outputs and witness payloads.

    Attributes:
        analyzer (str): The name of the analyzer.
        theorem (str): The theorem being applied.
        theorem_tag (str): The unique tag for the theorem.
        contract_version (str): The version of the contract.
        required_inputs (list[str]): List of required inputs for the contract.
        exactness_policy (str): The policy governing exact statements.
    """

    analyzer: str
    theorem: str
    theorem_tag: str
    contract_version: str = CONTRACT_VERSION
    required_inputs: list[str] = Field(default_factory=list)
    exactness_policy: str = (
        "Exact statements require exact certificates and theorem hypotheses."
    )


COVERAGE_MATRIX: list[CoverageMatrixEntry] = [
    CoverageMatrixEntry(
        dimension_class="2D",
        pi_family="closed surfaces",
        theorem="Classification of Closed Surfaces",
        theorem_tag="surface.classification.closed",
        status="exact",
        required_inputs=["H_1", "H_2", "orientability witness"],
        notes=["Cup/cohomology-ring witnesses are optional strengthening inputs."],
    ),
    CoverageMatrixEntry(
        dimension_class="3D",
        pi_family="homology spheres with pi_1=1",
        theorem="Poincare Conjecture / Geometrization",
        theorem_tag="3d.poincare.geometrization",
        status="exact",
        required_inputs=[
            "homology-sphere checks",
            "pi_1 witness",
            "decision-ready 3-manifold recognition certificate",
        ],
        notes=[
            "Poincare branch is exact for trivial pi_1 homology spheres; broader recognition requires explicit certificate intake."
        ],
    ),
    CoverageMatrixEntry(
        dimension_class="4D",
        pi_family="simply connected",
        theorem="Freedman classification",
        theorem_tag="4d.freedman.simply_connected",
        status="exact",
        required_inputs=[
            "unimodular intersection form",
            "Kirby-Siebenmann invariants",
            "decision-ready definite-lattice isometry certificate (if bounded search does not produce witness)",
        ],
        notes=[
            "Indefinite branch is intrinsic; definite branch supports internal search and external decision-ready isometry certificates."
        ],
    ),
    CoverageMatrixEntry(
        dimension_class="n>=5",
        pi_family="pi=1",
        theorem="s-Cobordism / surgery classification",
        theorem_tag="highdim.scobordism.surgery",
        status="exact",
        required_inputs=[
            "Exact Whitehead certificate",
            "Certified Wall obstruction state",
            "Decision-ready homotopy-completion certificate",
        ],
        notes=[
            "Exact high-dimensional success requires an exact and validated completion certificate."
        ],
    ),
    CoverageMatrixEntry(
        dimension_class="n>=5",
        pi_family="pi with nontrivial products",
        theorem="Wall L-theory over group rings",
        theorem_tag="highdim.wall.group_ring",
        status="exact",
        required_inputs=[
            "group-ring decomposition data",
            "representation-theoretic inputs",
            "decision-ready product-group assembly certificate",
        ],
        notes=[
            "Z-factor recursive splitting and factor-wise surrogate decomposition are implemented; exact product-group success requires assembly certification."
        ],
    ),
]


def coverage_status_counts() -> dict[str, int]:
    """Return the count of matrix entries by support status.

    Returns:
        dict[str, int]: A dictionary mapping status strings to their counts.
    """
    counts = Counter(entry.status for entry in COVERAGE_MATRIX)
    return dict(counts)


def get_contract(
    analyzer: str,
    theorem: str,
    theorem_tag: str,
    required_inputs: list[str] | None = None,
) -> AnalyzerContract:
    """Create a normalized analyzer contract record.

    Args:
        analyzer (str): The name of the analyzer.
        theorem (str): The name of the theorem.
        theorem_tag (str): The unique tag for the theorem.
        required_inputs (list[str] | None): Optional list of required inputs.

    Returns:
        AnalyzerContract: The generated analyzer contract.
    """
    return AnalyzerContract(
        analyzer=analyzer,
        theorem=theorem,
        theorem_tag=theorem_tag,
        required_inputs=list(required_inputs or []),
    )
