from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field


CONTRACT_VERSION = "2026.04-phase10"


class CoverageMatrixEntry(BaseModel):
    """One scoped theorem-support claim in the package coverage matrix.

    Overview:
        Represents a single entry in the project's coverage matrix, detailing
        the level of support for a specific topological theorem across different
        manifold dimensions and fundamental group families.

    Key Concepts:
        - **Theorem Tag**: Stable identifier for cross-referencing logic.
        - **Status**: Level of implementation (exact, partial, unsupported).
        - **Required Inputs**: Prerequisites needed for the theorem to be applicable.

    Common Workflows:
        1. Querying library capabilities for a specific manifold class.
        2. Generating automated documentation or coverage reports.

    Coefficient Ring:
        N/A.

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

    Overview:
        Formalizes the relationship between an analyzer's output and the
        mathematical theorems it relies upon. This metadata ensures that
        certificates and witnesses are verifiable and version-stable.

    Key Concepts:
        - **Contract Versioning**: Ensuring metadata matches the logic version.
        - **Exactness Policy**: Rules for claiming "exact" results.

    Common Workflows:
        1. Attaching metadata to `HomeomorphismWitness` objects.
        2. Validating if a certificate meets the required input standards.

    Coefficient Ring:
        N/A.

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
    CoverageMatrixEntry(
        dimension_class="nD",
        pi_family="all",
        theorem="Handle Surgery Mayer-Vietoris",
        theorem_tag="surgery.handle.mayer_vietoris",
        status="exact",
        required_inputs=[
            "SimplicialComplex K",
            "HandleAttachment with embeddedness_verified=True and framing_verified=True",
            "Exact SNF homology before/after",
        ],
        notes=[
            "Exact when attaching sphere is PL-certified; heuristic path sets exact=False.",
            "Torsion changes confined to dimensions {k-1, k} by Mayer-Vietoris.",
        ],
    ),
    CoverageMatrixEntry(
        dimension_class="nD",
        pi_family="all",
        theorem="Linking Number over Z via Seifert SNF",
        theorem_tag="surgery.linking.relative_snf_z",
        status="exact",
        required_inputs=[
            "Disjoint oriented subcomplexes K_a, K_b with dim_a + dim_b = n - 1",
            "K_b null-homologous in K (Seifert chain exists)",
            "Exact SNF of boundary matrix ∂_{q+1}",
        ],
        notes=[
            "Always exact over Z; raises LinkingComputationError on any algebraic failure.",
            "F2 heuristic path (surgery.linking.f2_torsion_blind) loses torsion information.",
        ],
    ),
    CoverageMatrixEntry(
        dimension_class="nD",
        pi_family="all",
        theorem="Delinking via Iterated Index-1 Handle Surgery",
        theorem_tag="surgery.delinking.unlinking_number",
        status="exact",
        required_inputs=[
            "Disjoint K_a, K_b with dim K_a = 1 (curves in ambient n-manifold)",
            "Certified attaching spheres (exact path) or SNF heuristic (approx path)",
            "Linking number computation at each step",
        ],
        notes=[
            "Exact only when all sphere attachments are PL-certified and lk reaches 0.",
            "Unlinking lower bound |lk_0| is sharp (Milnor 1961).",
        ],
    ),
    CoverageMatrixEntry(
        dimension_class="nD",
        pi_family="all",
        theorem="Auto Surgery Full Pipeline",
        theorem_tag="auto.surgery.full_pipeline",
        status="exact",
        required_inputs=[
            "SimplicialComplex K",
            "AutoSurgeonConfig config",
        ],
        notes=[
            "Executes multi-phase auto-surgery pipeline across unlink, separate, pi1, and homology kill phases."
        ],
    ),
    CoverageMatrixEntry(
        dimension_class="nD",
        pi_family="all",
        theorem="Auto Surgery Unlink Pair",
        theorem_tag="auto.surgery.unlink",
        status="exact",
        required_inputs=[
            "SimplicialComplex K",
            "Component A",
            "Component B",
        ],
        notes=[
            "Disjoint components are topology-preservingly unlinked via cancelling-pair surgery."
        ],
    ),
    CoverageMatrixEntry(
        dimension_class="nD",
        pi_family="all",
        theorem="Auto Surgery Separate Nested",
        theorem_tag="auto.surgery.separate_nested",
        status="exact",
        required_inputs=[
            "SimplicialComplex K",
            "Component outer",
            "Component inner",
        ],
        notes=[
            "Concentric nested components separated via ambient 1-handle surgery."
        ],
    ),
    CoverageMatrixEntry(
        dimension_class="nD",
        pi_family="all",
        theorem="Auto Surgery Kill Pi1",
        theorem_tag="auto.surgery.kill_pi1",
        status="exact",
        required_inputs=[
            "SimplicialComplex K",
            "Generator of fundamental group",
        ],
        notes=[
            "Kills pi1 generator via 2-handle surgery attachment."
        ],
    ),
    CoverageMatrixEntry(
        dimension_class="nD",
        pi_family="all",
        theorem="Auto Surgery Kill Homology Dim",
        theorem_tag="auto.surgery.kill_homology_dim",
        status="exact",
        required_inputs=[
            "SimplicialComplex K",
            "Homology dimension k",
        ],
        notes=[
            "Kills homology generators via index-(k+1) handle attachment."
        ],
    ),
    CoverageMatrixEntry(
        dimension_class="nD",
        pi_family="all",
        theorem="Auto Surgery Middle Obstruction",
        theorem_tag="auto.surgery.middle_obstruction",
        status="exact",
        required_inputs=[
            "SimplicialComplex K",
        ],
        notes=[
            "Wall / Arf / signature obstruction verification at middle dimension."
        ],
    ),
    CoverageMatrixEntry(
        dimension_class="nD",
        pi_family="all",
        theorem="Auto Surgery Cut Site Detection",
        theorem_tag="auto.surgery.cut_site",
        status="exact",
        required_inputs=[
            "SimplicialComplex K",
            "Component A",
            "Component B",
        ],
        notes=[
            "Locates a 1-strand isolated cut-site for unlinking."
        ],
    ),
    CoverageMatrixEntry(
        dimension_class="nD",
        pi_family="all",
        theorem="Surgery Cancelling Pair",
        theorem_tag="surgery.cancelling_pair",
        status="exact",
        required_inputs=[
            "SurgerySession session",
            "Prior open step step_index",
        ],
        notes=[
            "Glues cancelling handle after open disk removal step."
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
