from dataclasses import dataclass, field
from typing import Literal, Tuple

import numpy as np

from .core.complexes import ChainComplex
from .core.fundamental_group import FundamentalGroup, GroupPresentation
from .core.intersection_forms import IntersectionForm
from .core.k_theory import WhiteheadGroup
from .core.foundations import CONTRACT_VERSION
from .core.theorem_tags import infer_theorem_tag
from .homeomorphism import (
    DefiniteLatticeIsometryCertificate,
    HomeomorphismResult,
    HomotopyCompletionCertificate,
    HomotopyEquivalenceWitnessHook,
    ProductAssemblyCertificate,
    ThreeManifoldRecognitionCertificate,
    analyze_homeomorphism_1d_result,
    analyze_homeomorphism_2d_result,
    analyze_homeomorphism_3d_result,
    analyze_homeomorphism_4d_result,
    analyze_homeomorphism_high_dim_result,
    _search_integer_isometry,
)
from .structure_set import NormalInvariantsResult, SurgeryExactSequenceResult
from .wall_groups import ObstructionResult


WitnessKind = Literal[
    "discrete_classification_1d",
    "surface_classification",
    "poincare_conjecture",
    "freedman_indefinite",
    "freedman_definite_isometry",
    "s_cobordism_certificate",
]


@dataclass
class HomeomorphismWitness:
    """Structured certificate describing how a homeomorphism was obtained."""

    dimension: int
    theorem: str
    kind: WitnessKind
    description: str
    exact: bool
    evidence: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    certificates: dict[str, object] = field(default_factory=dict)
    explicit_map: np.ndarray | None = None
    theorem_tag: str | None = None
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.theorem_tag is None:
            self.theorem_tag = infer_theorem_tag(self.theorem)


@dataclass
class HomeomorphismWitnessResult:
    """Result object returned by witness builders."""

    status: Literal["success", "inconclusive", "surgery_required"]
    reasoning: str
    witness: HomeomorphismWitness | None = None
    theorem: str | None = None
    theorem_tag: str | None = None
    contract_version: str = CONTRACT_VERSION
    source_result: HomeomorphismResult | None = None
    missing_data: list[str] = field(default_factory=list)

    def to_legacy_tuple(self) -> Tuple[bool | None, str]:
        if self.witness is not None:
            return True, self.reasoning
        return None, self.reasoning


def _build_from_result(
    result: HomeomorphismResult,
    *,
    dimension: int,
    kind: WitnessKind,
    theorem: str,
    explicit_map: np.ndarray | None = None,
    extra_certificates: dict[str, object] | None = None,
) -> HomeomorphismWitnessResult:
    if result.status != "success" or result.is_homeomorphic is not True:
        return HomeomorphismWitnessResult(
            status="inconclusive",
            reasoning=result.reasoning,
            witness=None,
            theorem=result.theorem or theorem,
            theorem_tag=result.theorem_tag,
            source_result=result,
            missing_data=list(result.missing_data),
        )

    if not result.exact:
        return HomeomorphismWitnessResult(
            status="inconclusive",
            reasoning=(
                "INCONCLUSIVE: The classification result is not exact enough to build a certified homeomorphism witness."
            ),
            witness=None,
            theorem=result.theorem or theorem,
            theorem_tag=result.theorem_tag,
            source_result=result,
            missing_data=["Exact classification certificate"],
        )

    certificates = dict(result.certificates)
    if extra_certificates:
        certificates.update(extra_certificates)
    if explicit_map is not None:
        certificates["explicit_map"] = explicit_map

    witness = HomeomorphismWitness(
        dimension=dimension,
        theorem=result.theorem or theorem,
        theorem_tag=result.theorem_tag,
        kind=kind,
        description=result.reasoning,
        exact=True,
        evidence=list(result.evidence),
        assumptions=list(result.assumptions),
        certificates=certificates,
        explicit_map=explicit_map,
    )
    return HomeomorphismWitnessResult(
        status="success",
        reasoning=result.reasoning,
        witness=witness,
        theorem=result.theorem or theorem,
        theorem_tag=result.theorem_tag,
        source_result=result,
    )


def _obstruction_state_payload(
    obstruction: ObstructionResult | None,
) -> dict[str, object]:
    if obstruction is None:
        return {
            "available": False,
            "computable": False,
            "exact": False,
            "obstructs": None,
            "zero_certified": False,
            "value": None,
            "modulus": None,
            "pi": None,
            "dimension": None,
        }
    return {
        "available": True,
        "computable": bool(obstruction.computable),
        "exact": bool(obstruction.exact),
        "obstructs": obstruction.obstructs,
        "zero_certified": bool(obstruction.zero_certified),
        "value": obstruction.value,
        "modulus": obstruction.modulus,
        "pi": obstruction.pi,
        "dimension": obstruction.dimension,
    }


def build_1d_homeomorphism_witness(
    c1: ChainComplex,
    c2: ChainComplex,
    allow_approx: bool = False,
) -> HomeomorphismWitnessResult:
    result = analyze_homeomorphism_1d_result(
        c1,
        c2,
        allow_approx=allow_approx,
    )
    return _build_from_result(
        result,
        dimension=1,
        kind="discrete_classification_1d",
        theorem="Classification of 1-Manifolds",
    )


def build_surface_homeomorphism_witness(
    c1: ChainComplex,
    c2: ChainComplex,
    allow_approx: bool = False,
    *,
    cohomology_signature_1: dict | None = None,
    cohomology_signature_2: dict | None = None,
    cohomology_ring_signature_1: dict | None = None,
    cohomology_ring_signature_2: dict | None = None,
    cup_product_signature_1: dict | None = None,
    cup_product_signature_2: dict | None = None,
) -> HomeomorphismWitnessResult:
    result = analyze_homeomorphism_2d_result(
        c1,
        c2,
        allow_approx=allow_approx,
        cohomology_signature_1=cohomology_signature_1,
        cohomology_signature_2=cohomology_signature_2,
        cohomology_ring_signature_1=cohomology_ring_signature_1,
        cohomology_ring_signature_2=cohomology_ring_signature_2,
        cup_product_signature_1=cup_product_signature_1,
        cup_product_signature_2=cup_product_signature_2,
    )
    extra = {
        "cohomology_signature_1": cohomology_signature_1,
        "cohomology_signature_2": cohomology_signature_2,
        "cohomology_ring_signature_1": cohomology_ring_signature_1,
        "cohomology_ring_signature_2": cohomology_ring_signature_2,
        "cup_product_signature_1": cup_product_signature_1,
        "cup_product_signature_2": cup_product_signature_2,
    }
    return _build_from_result(
        result,
        dimension=2,
        kind="surface_classification",
        theorem="Classification of Closed Surfaces",
        extra_certificates={k: v for k, v in extra.items() if v is not None},
    )


def build_3d_homeomorphism_witness(
    c1: ChainComplex,
    c2: ChainComplex,
    allow_approx: bool = False,
    *,
    pi1_1: FundamentalGroup | None = None,
    pi1_2: FundamentalGroup | None = None,
    cohomology_signature_1: dict | None = None,
    cohomology_signature_2: dict | None = None,
    cohomology_ring_signature_1: dict | None = None,
    cohomology_ring_signature_2: dict | None = None,
    cup_product_signature_1: dict | None = None,
    cup_product_signature_2: dict | None = None,
    recognition_certificate: ThreeManifoldRecognitionCertificate | dict | None = None,
) -> HomeomorphismWitnessResult:
    result = analyze_homeomorphism_3d_result(
        c1,
        c2,
        allow_approx=allow_approx,
        pi1_1=pi1_1,
        pi1_2=pi1_2,
        cohomology_signature_1=cohomology_signature_1,
        cohomology_signature_2=cohomology_signature_2,
        cohomology_ring_signature_1=cohomology_ring_signature_1,
        cohomology_ring_signature_2=cohomology_ring_signature_2,
        cup_product_signature_1=cup_product_signature_1,
        cup_product_signature_2=cup_product_signature_2,
        recognition_certificate=recognition_certificate,
    )
    extra = {
        "pi1_1": pi1_1,
        "pi1_2": pi1_2,
        "cohomology_signature_1": cohomology_signature_1,
        "cohomology_signature_2": cohomology_signature_2,
        "cohomology_ring_signature_1": cohomology_ring_signature_1,
        "cohomology_ring_signature_2": cohomology_ring_signature_2,
        "cup_product_signature_1": cup_product_signature_1,
        "cup_product_signature_2": cup_product_signature_2,
        "recognition_certificate": recognition_certificate,
    }
    return _build_from_result(
        result,
        dimension=3,
        kind="poincare_conjecture",
        theorem="Poincaré Conjecture / Geometrization",
        extra_certificates={k: v for k, v in extra.items() if v is not None},
    )


def build_4d_homeomorphism_witness(
    m1: IntersectionForm,
    m2: IntersectionForm,
    ks1: int | None = None,
    ks2: int | None = None,
    *,
    simply_connected: bool | None = None,
    definite_lattice_isometry_certificate: DefiniteLatticeIsometryCertificate
    | dict
    | None = None,
) -> HomeomorphismWitnessResult:
    result = analyze_homeomorphism_4d_result(
        m1,
        m2,
        ks1=ks1,
        ks2=ks2,
        simply_connected=simply_connected,
        definite_lattice_isometry_certificate=definite_lattice_isometry_certificate,
    )
    if result.status != "success" or result.is_homeomorphic is not True:
        mapped_status: Literal["inconclusive", "surgery_required"] = (
            "surgery_required"
            if result.status == "surgery_required"
            else "inconclusive"
        )
        return HomeomorphismWitnessResult(
            status=mapped_status,
            reasoning=result.reasoning,
            witness=None,
            theorem=result.theorem,
            theorem_tag=result.theorem_tag,
            source_result=result,
            missing_data=list(result.missing_data),
        )

    q1 = np.asarray(m1.matrix, dtype=np.int64)
    q2 = np.asarray(m2.matrix, dtype=np.int64)

    explicit_map = None
    if np.array_equal(q1, q2):
        explicit_map = np.eye(q1.shape[0], dtype=np.int64)
    elif result.certificates.get("isometry_matrix") is not None:
        explicit_map = np.asarray(
            result.certificates.get("isometry_matrix"), dtype=np.int64
        )
    elif not m1.is_indefinite():
        explicit_map = _search_integer_isometry(q1, q2, max_entry=2)
        if explicit_map is None and q1.shape[0] <= 3:
            explicit_map = _search_integer_isometry(q1, q2, max_entry=3)

    certificates = dict(result.certificates)
    certificates.update(
        {
            "intersection_form_1": m1,
            "intersection_form_2": m2,
            "ks1": ks1,
            "ks2": ks2,
            "simply_connected": simply_connected,
            "definite_lattice_isometry_certificate_input": definite_lattice_isometry_certificate,
        }
    )
    if explicit_map is not None:
        certificates["isometry_matrix"] = np.asarray(
            explicit_map, dtype=np.int64
        ).tolist()

    if explicit_map is None and not m1.is_indefinite():
        return HomeomorphismWitnessResult(
            status="inconclusive",
            reasoning="INCONCLUSIVE: The 4D classification is successful, but no explicit lattice isometry certificate was found within the witness builder's bounded search.",
            witness=None,
            theorem=result.theorem,
            source_result=result,
            missing_data=["Explicit lattice isometry certificate"],
        )

    kind: WitnessKind = (
        "freedman_indefinite" if m1.is_indefinite() else "freedman_definite_isometry"
    )
    witness = HomeomorphismWitness(
        dimension=4,
        theorem=result.theorem or "Freedman classification",
        theorem_tag=result.theorem_tag,
        kind=kind,
        description=result.reasoning,
        exact=True,
        evidence=list(result.evidence),
        assumptions=list(result.assumptions),
        certificates=certificates,
        explicit_map=explicit_map,
    )
    return HomeomorphismWitnessResult(
        status="success",
        reasoning=result.reasoning,
        witness=witness,
        theorem=result.theorem,
        theorem_tag=result.theorem_tag,
        source_result=result,
    )


def build_high_dim_homeomorphism_witness(
    c1: ChainComplex,
    c2: ChainComplex,
    dim: int,
    allow_approx: bool = False,
    *,
    pi1: FundamentalGroup | None = None,
    pi_group: str | GroupPresentation | None = None,
    whitehead_group: WhiteheadGroup | None = None,
    wall_obstruction: ObstructionResult | None = None,
    wall_form: IntersectionForm | None = None,
    cohomology_signature_1: dict | None = None,
    cohomology_signature_2: dict | None = None,
    cohomology_ring_signature_1: dict | None = None,
    cohomology_ring_signature_2: dict | None = None,
    cup_product_signature_1: dict | None = None,
    cup_product_signature_2: dict | None = None,
    normal_invariants_1: NormalInvariantsResult | None = None,
    normal_invariants_2: NormalInvariantsResult | None = None,
    surgery_sequence: SurgeryExactSequenceResult | None = None,
    homotopy_equivalence_witness: object | None = None,
    homotopy_witness_hook: HomotopyEquivalenceWitnessHook | dict | None = None,
    homotopy_completion_certificate: HomotopyCompletionCertificate | dict | None = None,
    product_assembly_certificate: ProductAssemblyCertificate | dict | None = None,
) -> HomeomorphismWitnessResult:
    result = analyze_homeomorphism_high_dim_result(
        c1,
        c2,
        dim=dim,
        allow_approx=allow_approx,
        pi1=pi1,
        pi_group=pi_group,
        whitehead_group=whitehead_group,
        wall_obstruction=wall_obstruction,
        wall_form=wall_form,
        cohomology_signature_1=cohomology_signature_1,
        cohomology_signature_2=cohomology_signature_2,
        cohomology_ring_signature_1=cohomology_ring_signature_1,
        cohomology_ring_signature_2=cohomology_ring_signature_2,
        cup_product_signature_1=cup_product_signature_1,
        cup_product_signature_2=cup_product_signature_2,
        normal_invariants_1=normal_invariants_1,
        normal_invariants_2=normal_invariants_2,
        surgery_sequence=surgery_sequence,
        homotopy_equivalence_witness=homotopy_equivalence_witness,
        homotopy_witness_hook=homotopy_witness_hook,
        homotopy_completion_certificate=homotopy_completion_certificate,
        product_assembly_certificate=product_assembly_certificate,
    )
    if result.status != "success" or result.is_homeomorphic is not True:
        mapped_status: Literal["inconclusive", "surgery_required"] = (
            "surgery_required"
            if result.status == "surgery_required"
            else "inconclusive"
        )
        return HomeomorphismWitnessResult(
            status=mapped_status,
            reasoning=result.reasoning,
            witness=None,
            theorem=result.theorem,
            theorem_tag=result.theorem_tag,
            source_result=result,
            missing_data=list(result.missing_data),
        )

    certificates = dict(result.certificates)
    certificates.update(
        {
            "pi1": pi1,
            "pi_group": pi_group,
            "whitehead_group_input": whitehead_group,
            "wall_obstruction_input": wall_obstruction,
            "wall_form": wall_form,
            "cohomology_signature_1": cohomology_signature_1,
            "cohomology_signature_2": cohomology_signature_2,
            "cohomology_ring_signature_1": cohomology_ring_signature_1,
            "cohomology_ring_signature_2": cohomology_ring_signature_2,
            "cup_product_signature_1": cup_product_signature_1,
            "cup_product_signature_2": cup_product_signature_2,
            "normal_invariants_1_input": normal_invariants_1,
            "normal_invariants_2_input": normal_invariants_2,
            "surgery_sequence_input": surgery_sequence,
            "homotopy_equivalence_witness_input": homotopy_equivalence_witness,
            "homotopy_witness_hook_input": homotopy_witness_hook,
            "homotopy_completion_certificate_input": homotopy_completion_certificate,
            "product_assembly_certificate_input": product_assembly_certificate,
        }
    )
    resolved_wall = certificates.get("wall_obstruction")
    if isinstance(resolved_wall, ObstructionResult):
        certificates["wall_obstruction_state"] = _obstruction_state_payload(
            resolved_wall
        )
    else:
        certificates["wall_obstruction_state"] = _obstruction_state_payload(None)

    seq = certificates.get("surgery_sequence")
    if isinstance(seq, SurgeryExactSequenceResult):
        if hasattr(seq.l_n_state, "to_legacy_dict"):
            certificates["surgery_sequence_l_n_state"] = seq.l_n_state.to_legacy_dict()
        else:
            certificates["surgery_sequence_l_n_state"] = dict(seq.l_n_state)
        if hasattr(seq.l_n_plus_1_state, "to_legacy_dict"):
            certificates["surgery_sequence_l_n_plus_1_state"] = (
                seq.l_n_plus_1_state.to_legacy_dict()
            )
        else:
            certificates["surgery_sequence_l_n_plus_1_state"] = dict(
                seq.l_n_plus_1_state
            )

    certificates = {k: v for k, v in certificates.items() if v is not None}

    witness = HomeomorphismWitness(
        dimension=dim,
        theorem=result.theorem or "s-Cobordism / surgery classification",
        theorem_tag=result.theorem_tag,
        kind="s_cobordism_certificate",
        description=result.reasoning,
        exact=result.exact,
        evidence=list(result.evidence),
        assumptions=list(result.assumptions),
        certificates=certificates,
        explicit_map=None,
    )
    return HomeomorphismWitnessResult(
        status="success",
        reasoning=result.reasoning,
        witness=witness,
        theorem=result.theorem,
        theorem_tag=result.theorem_tag,
        source_result=result,
    )


def build_homeomorphism_witness(
    c1: ChainComplex | None = None,
    c2: ChainComplex | None = None,
    *,
    dim: int | None = None,
    m1: IntersectionForm | None = None,
    m2: IntersectionForm | None = None,
    ks1: int | None = None,
    ks2: int | None = None,
    simply_connected: bool | None = None,
    definite_lattice_isometry_certificate: DefiniteLatticeIsometryCertificate
    | dict
    | None = None,
    pi1_1: FundamentalGroup | None = None,
    pi1_2: FundamentalGroup | None = None,
    pi1: FundamentalGroup | None = None,
    pi_group: str | GroupPresentation | None = None,
    whitehead_group: WhiteheadGroup | None = None,
    wall_obstruction: ObstructionResult | None = None,
    wall_form: IntersectionForm | None = None,
    cohomology_signature_1: dict | None = None,
    cohomology_signature_2: dict | None = None,
    cohomology_ring_signature_1: dict | None = None,
    cohomology_ring_signature_2: dict | None = None,
    cup_product_signature_1: dict | None = None,
    cup_product_signature_2: dict | None = None,
    normal_invariants_1: NormalInvariantsResult | None = None,
    normal_invariants_2: NormalInvariantsResult | None = None,
    surgery_sequence: SurgeryExactSequenceResult | None = None,
    homotopy_equivalence_witness: object | None = None,
    homotopy_witness_hook: HomotopyEquivalenceWitnessHook | dict | None = None,
    homotopy_completion_certificate: HomotopyCompletionCertificate | dict | None = None,
    recognition_certificate: ThreeManifoldRecognitionCertificate | dict | None = None,
    product_assembly_certificate: ProductAssemblyCertificate | dict | None = None,
    allow_approx: bool = False,
) -> HomeomorphismWitnessResult:
    """Dispatch to the appropriate witness builder based on the supplied data."""
    if m1 is not None or m2 is not None:
        if m1 is None or m2 is None:
            return HomeomorphismWitnessResult(
                status="inconclusive",
                reasoning="INCONCLUSIVE: 4D witness building requires both intersection forms.",
                witness=None,
                theorem="Freedman classification",
                missing_data=["Both 4D intersection forms"],
            )
        return build_4d_homeomorphism_witness(
            m1,
            m2,
            ks1=ks1,
            ks2=ks2,
            simply_connected=simply_connected,
            definite_lattice_isometry_certificate=definite_lattice_isometry_certificate,
        )

    if dim is None:
        return HomeomorphismWitnessResult(
            status="inconclusive",
            reasoning="INCONCLUSIVE: A dimension must be supplied when no intersection forms are provided.",
            witness=None,
            theorem=None,
            missing_data=["dimension"],
        )

    if c1 is None or c2 is None:
        return HomeomorphismWitnessResult(
            status="inconclusive",
            reasoning="INCONCLUSIVE: Chain complexes are required for 2D, 3D, or high-dimensional witness building.",
            witness=None,
            theorem=None,
            missing_data=["Both chain complexes"],
        )

    if dim == 1:
        return build_1d_homeomorphism_witness(
            c1,
            c2,
            allow_approx=allow_approx,
        )

    if dim == 2:
        return build_surface_homeomorphism_witness(
            c1,
            c2,
            allow_approx=allow_approx,
            cohomology_signature_1=cohomology_signature_1,
            cohomology_signature_2=cohomology_signature_2,
            cohomology_ring_signature_1=cohomology_ring_signature_1,
            cohomology_ring_signature_2=cohomology_ring_signature_2,
            cup_product_signature_1=cup_product_signature_1,
            cup_product_signature_2=cup_product_signature_2,
        )

    if dim == 3:
        return build_3d_homeomorphism_witness(
            c1,
            c2,
            allow_approx=allow_approx,
            pi1_1=pi1_1,
            pi1_2=pi1_2,
            cohomology_signature_1=cohomology_signature_1,
            cohomology_signature_2=cohomology_signature_2,
            cohomology_ring_signature_1=cohomology_ring_signature_1,
            cohomology_ring_signature_2=cohomology_ring_signature_2,
            cup_product_signature_1=cup_product_signature_1,
            cup_product_signature_2=cup_product_signature_2,
            recognition_certificate=recognition_certificate,
        )

    if dim >= 5:
        return build_high_dim_homeomorphism_witness(
            c1,
            c2,
            dim,
            allow_approx=allow_approx,
            pi1=pi1,
            pi_group=pi_group,
            whitehead_group=whitehead_group,
            wall_obstruction=wall_obstruction,
            wall_form=wall_form,
            cohomology_signature_1=cohomology_signature_1,
            cohomology_signature_2=cohomology_signature_2,
            cohomology_ring_signature_1=cohomology_ring_signature_1,
            cohomology_ring_signature_2=cohomology_ring_signature_2,
            cup_product_signature_1=cup_product_signature_1,
            cup_product_signature_2=cup_product_signature_2,
            normal_invariants_1=normal_invariants_1,
            normal_invariants_2=normal_invariants_2,
            surgery_sequence=surgery_sequence,
            homotopy_equivalence_witness=homotopy_equivalence_witness,
            homotopy_witness_hook=homotopy_witness_hook,
            homotopy_completion_certificate=homotopy_completion_certificate,
            product_assembly_certificate=product_assembly_certificate,
        )

    return HomeomorphismWitnessResult(
        status="inconclusive",
        reasoning=(
            f"INCONCLUSIVE: Witness construction is only implemented for dimensions 2, 3, 4, and n>=5; received {dim}."
        ),
        witness=None,
        theorem=None,
        missing_data=["Supported dimension (2, 3, 4, or >=5)"],
    )
