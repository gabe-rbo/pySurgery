"""Homeomorphism witness construction and certificate management."""

from dataclasses import dataclass, field
from typing import Literal, Tuple, List, Dict, Optional, Any, Union

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
    """Structured certificate describing how a homeomorphism was obtained.

    Attributes:
        dimension: The dimension of the manifolds.
        theorem: The name of the theorem used for classification.
        kind: The specific kind of witness.
        description: A human-readable description of the result.
        exact: Whether the result is mathematically exact.
        evidence: List of evidence strings supporting the result.
        assumptions: List of assumptions made during analysis.
        certificates: Dictionary of detailed mathematical certificates.
        explicit_map: Optional numpy array representing the explicit map.
        theorem_tag: Optional tag for the theorem used.
        contract_version: The version of the witness contract.
    """

    dimension: int
    theorem: str
    kind: WitnessKind
    description: str
    exact: bool
    evidence: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    certificates: Dict[str, object] = field(default_factory=dict)
    explicit_map: Optional[np.ndarray] = None
    theorem_tag: Optional[str] = None
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        """Post-initialization to infer theorem tag if missing."""
        if self.theorem_tag is None:
            self.theorem_tag = infer_theorem_tag(self.theorem)


@dataclass
class HomeomorphismWitnessResult:
    """Result object returned by witness builders.

    Attributes:
        status: The status of the witness building ("success", "inconclusive", "surgery_required").
        reasoning: Human-readable reasoning for the status.
        witness: The built HomeomorphismWitness, if successful.
        theorem: The theorem used.
        theorem_tag: Tag for the theorem.
        contract_version: Version of the contract.
        source_result: The original HomeomorphismResult.
        missing_data: List of data that was missing for a successful witness.
    """

    status: Literal["success", "inconclusive", "surgery_required"]
    reasoning: str
    witness: Optional[HomeomorphismWitness] = None
    theorem: Optional[str] = None
    theorem_tag: Optional[str] = None
    contract_version: str = CONTRACT_VERSION
    source_result: Optional[HomeomorphismResult] = None
    missing_data: List[str] = field(default_factory=list)

    def to_legacy_tuple(self) -> Tuple[Optional[bool], str]:
        """Convert the result to a legacy (bool | None, reasoning) tuple.

        Returns:
            Tuple[Optional[bool], str]: A tuple where the first element is True if a witness exists,
                None otherwise, and the second element is the reasoning string.
        """
        if self.witness is not None:
            return True, self.reasoning
        return None, self.reasoning


def _build_from_result(
    result: HomeomorphismResult,
    *,
    dimension: int,
    kind: WitnessKind,
    theorem: str,
    explicit_map: Optional[np.ndarray] = None,
    extra_certificates: Optional[Dict[str, object]] = None,
) -> HomeomorphismWitnessResult:
    """Internal helper to build a witness result from a classification result.

    Args:
        result: The classification result.
        dimension: The manifold dimension.
        kind: The kind of witness to build.
        theorem: The theorem name.
        explicit_map: Optional explicit map data.
        extra_certificates: Optional additional certificates.

    Returns:
        HomeomorphismWitnessResult: A HomeomorphismWitnessResult instance.
    """
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
    obstruction: Optional[ObstructionResult],
) -> Dict[str, Any]:
    """Convert an ObstructionResult to a structured payload dictionary.

    Args:
        obstruction: The obstruction result, or None.

    Returns:
        Dict[str, Any]: A dictionary containing the obstruction state.
    """
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
    backend: str = "auto",
) -> HomeomorphismWitnessResult:
    """Build a homeomorphism witness for 1D manifolds.

    Args:
        c1: First chain complex.
        c2: Second chain complex.
        allow_approx: Whether to allow approximate/heuristic analysis.
        backend: 'auto', 'julia', or 'python'.

    Returns:
        HomeomorphismWitnessResult: A HomeomorphismWitnessResult.
    """
    result = analyze_homeomorphism_1d_result(
        c1,
        c2,
        allow_approx=allow_approx,
        backend=backend,
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
    cohomology_signature_1: Optional[Dict] = None,
    cohomology_signature_2: Optional[Dict] = None,
    cohomology_ring_signature_1: Optional[Dict] = None,
    cohomology_ring_signature_2: Optional[Dict] = None,
    cup_product_signature_1: Optional[Dict] = None,
    cup_product_signature_2: Optional[Dict] = None,
    backend: str = "auto",
) -> HomeomorphismWitnessResult:
    """Build a homeomorphism witness for closed surfaces.

    Args:
        c1: First chain complex.
        c2: Second chain complex.
        allow_approx: Whether to allow approximate analysis.
        cohomology_signature_1: Optional H^* signature for M1.
        cohomology_signature_2: Optional H^* signature for M2.
        cohomology_ring_signature_1: Optional ring signature for M1.
        cohomology_ring_signature_2: Optional ring signature for M2.
        cup_product_signature_1: Optional cup products for M1.
        cup_product_signature_2: Optional cup products for M2.

    Returns:
        HomeomorphismWitnessResult: A HomeomorphismWitnessResult.
    """
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
    pi1_1: Optional[FundamentalGroup] = None,
    pi1_2: Optional[FundamentalGroup] = None,
    cohomology_signature_1: Optional[Dict] = None,
    cohomology_signature_2: Optional[Dict] = None,
    cohomology_ring_signature_1: Optional[Dict] = None,
    cohomology_ring_signature_2: Optional[Dict] = None,
    cup_product_signature_1: Optional[Dict] = None,
    cup_product_signature_2: Optional[Dict] = None,
    recognition_certificate: Optional[Union[ThreeManifoldRecognitionCertificate, Dict]] = None,
    backend: str = "auto",
) -> HomeomorphismWitnessResult:
    """Build a homeomorphism witness for 3D manifolds.

    Args:
        c1: First chain complex.
        c2: Second chain complex.
        allow_approx: Whether to allow approximate analysis.
        pi1_1: Optional fundamental group for M1.
        pi1_2: Optional fundamental group for M2.
        cohomology_signature_1: Optional H^* signature for M1.
        cohomology_signature_2: Optional H^* signature for M2.
        cohomology_ring_signature_1: Optional ring signature for M1.
        cohomology_ring_signature_2: Optional ring signature for M2.
        cup_product_signature_1: Optional cup products for M1.
        cup_product_signature_2: Optional cup products for M2.
        recognition_certificate: Optional 3D recognition certificate.
        backend: 'auto', 'julia', or 'python'.

    Returns:
        HomeomorphismWitnessResult: A HomeomorphismWitnessResult.
    """
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
        backend=backend,
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
    ks1: Optional[int] = None,
    ks2: Optional[int] = None,
    *,
    simply_connected: Optional[bool] = None,
    definite_lattice_isometry_certificate: Optional[Union[DefiniteLatticeIsometryCertificate, Dict]] = None,
    backend: str = "auto",
) -> HomeomorphismWitnessResult:
    """Build a homeomorphism witness for 4D manifolds.

    Args:
        m1: Intersection form of M1.
        m2: Intersection form of M2.
        ks1: Optional Kirby-Siebenmann invariant for M1.
        ks2: Optional Kirby-Siebenmann invariant for M2.
        simply_connected: Whether the manifolds are simply connected.
        definite_lattice_isometry_certificate: Optional isometry certificate for definite forms.
        backend: 'auto', 'julia', or 'python'.

    Returns:
        HomeomorphismWitnessResult: A HomeomorphismWitnessResult.
    """
    result = analyze_homeomorphism_4d_result(
        m1,
        m2,
        ks1=ks1,
        ks2=ks2,
        simply_connected=simply_connected,
        definite_lattice_isometry_certificate=definite_lattice_isometry_certificate,
        backend=backend,
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
    pi1: Optional[FundamentalGroup] = None,
    pi_group: Optional[Union[str, GroupPresentation]] = None,
    whitehead_group: Optional[WhiteheadGroup] = None,
    wall_obstruction: Optional[ObstructionResult] = None,
    wall_form: Optional[IntersectionForm] = None,
    cohomology_signature_1: Optional[Dict] = None,
    cohomology_signature_2: Optional[Dict] = None,
    cohomology_ring_signature_1: Optional[Dict] = None,
    cohomology_ring_signature_2: Optional[Dict] = None,
    cup_product_signature_1: Optional[Dict] = None,
    cup_product_signature_2: Optional[Dict] = None,
    normal_invariants_1: Optional[NormalInvariantsResult] = None,
    normal_invariants_2: Optional[NormalInvariantsResult] = None,
    surgery_sequence: Optional[SurgeryExactSequenceResult] = None,
    homotopy_equivalence_witness: Optional[object] = None,
    homotopy_witness_hook: Optional[Union[HomotopyEquivalenceWitnessHook, Dict]] = None,
    homotopy_completion_certificate: Optional[Union[HomotopyCompletionCertificate, Dict]] = None,
    product_assembly_certificate: Optional[Union[ProductAssemblyCertificate, Dict]] = None,
    backend: str = "auto",
) -> HomeomorphismWitnessResult:
    """Build a homeomorphism witness for high-dimensional manifolds (n >= 5).

    Args:
        c1: First chain complex.
        c2: Second chain complex.
        dim: Manifold dimension.
        allow_approx: Whether to allow approximate analysis.
        pi1: Optional fundamental group.
        pi_group: Optional pi_1 group presentation or descriptor.
        whitehead_group: Optional Whitehead group for torsion analysis.
        wall_obstruction: Optional L-group surgery obstruction result.
        wall_form: Optional intersection form for L-group evaluation.
        cohomology_signature_1: Optional H^* signature for M1.
        cohomology_signature_2: Optional H^* signature for M2.
        cohomology_ring_signature_1: Optional ring signature for M1.
        cohomology_ring_signature_2: Optional ring signature for M2.
        cup_product_signature_1: Optional cup products for M1.
        cup_product_signature_2: Optional cup products for M2.
        normal_invariants_1: Optional normal invariants for M1.
        normal_invariants_2: Optional normal invariants for M2.
        surgery_sequence: Optional surgery exact sequence results.
        homotopy_equivalence_witness: Optional witness of homotopy equivalence.
        homotopy_witness_hook: Optional hook for external homotopy witnesses.
        homotopy_completion_certificate: Optional completion certificate.
        product_assembly_certificate: Optional assembly certificate.

    Returns:
        HomeomorphismWitnessResult: A HomeomorphismWitnessResult.
    """
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
        backend=backend,
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
    c1: Optional[ChainComplex] = None,
    c2: Optional[ChainComplex] = None,
    *,
    dim: Optional[int] = None,
    m1: Optional[IntersectionForm] = None,
    m2: Optional[IntersectionForm] = None,
    ks1: Optional[int] = None,
    ks2: Optional[int] = None,
    simply_connected: Optional[bool] = None,
    definite_lattice_isometry_certificate: Optional[Union[DefiniteLatticeIsometryCertificate, Dict]] = None,
    pi1_1: Optional[FundamentalGroup] = None,
    pi1_2: Optional[FundamentalGroup] = None,
    pi1: Optional[FundamentalGroup] = None,
    pi_group: Optional[Union[str, GroupPresentation]] = None,
    whitehead_group: Optional[WhiteheadGroup] = None,
    wall_obstruction: Optional[ObstructionResult] = None,
    wall_form: Optional[IntersectionForm] = None,
    cohomology_signature_1: Optional[Dict] = None,
    cohomology_signature_2: Optional[Dict] = None,
    cohomology_ring_signature_1: Optional[Dict] = None,
    cohomology_ring_signature_2: Optional[Dict] = None,
    cup_product_signature_1: Optional[Dict] = None,
    cup_product_signature_2: Optional[Dict] = None,
    normal_invariants_1: Optional[NormalInvariantsResult] = None,
    normal_invariants_2: Optional[NormalInvariantsResult] = None,
    surgery_sequence: Optional[SurgeryExactSequenceResult] = None,
    homotopy_equivalence_witness: Optional[object] = None,
    homotopy_witness_hook: Optional[Union[HomotopyEquivalenceWitnessHook, Dict]] = None,
    homotopy_completion_certificate: Optional[Union[HomotopyCompletionCertificate, Dict]] = None,
    recognition_certificate: Optional[Union[ThreeManifoldRecognitionCertificate, Dict]] = None,
    product_assembly_certificate: Optional[Union[ProductAssemblyCertificate, Dict]] = None,
    allow_approx: bool = False,
    backend: str = "auto",
) -> HomeomorphismWitnessResult:
    """Dispatch to the appropriate witness builder based on the supplied data.

    Args:
        c1: First chain complex.
        c2: Second chain complex.
        dim: Manifold dimension.
        m1: Intersection form for M1 (4D).
        m2: Intersection form for M2 (4D).
        ks1: Kirby-Siebenmann invariant for M1 (4D).
        ks2: Kirby-Siebenmann invariant for M2 (4D).
        simply_connected: Whether manifolds are simply connected.
        definite_lattice_isometry_certificate: Isometry certificate for 4D.
        pi1_1: Fundamental group for M1 (3D).
        pi1_2: Fundamental group for M2 (3D).
        pi1: Fundamental group (high-D).
        pi_group: Group descriptor (high-D).
        whitehead_group: Whitehead group (high-D).
        wall_obstruction: L-group obstruction (high-D).
        wall_form: Intersection form for L-group (high-D).
        cohomology_signature_1: H^* signature for M1.
        cohomology_signature_2: H^* signature for M2.
        cohomology_ring_signature_1: Ring signature for M1.
        cohomology_ring_signature_2: Ring signature for M2.
        cup_product_signature_1: Cup products for M1.
        cup_product_signature_2: Cup products for M2.
        normal_invariants_1: Normal invariants for M1 (high-D).
        normal_invariants_2: Normal invariants for M2 (high-D).
        surgery_sequence: Surgery sequence (high-D).
        homotopy_equivalence_witness: Homotopy witness (high-D).
        homotopy_witness_hook: Witness hook (high-D).
        homotopy_completion_certificate: Completion certificate (high-D).
        recognition_certificate: 3D recognition certificate.
        product_assembly_certificate: Product assembly certificate (high-D).
        allow_approx: Whether to allow approximate analysis.
        backend: 'auto', 'julia', or 'python'.

    Returns:
        HomeomorphismWitnessResult: A HomeomorphismWitnessResult instance.
    """
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
            backend=backend,
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
            backend=backend,
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
            backend=backend,
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
            backend=backend,
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
            backend=backend,
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
