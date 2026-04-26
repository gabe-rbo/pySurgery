from dataclasses import dataclass, field
import re
from typing import Literal, Tuple
from .core.intersection_forms import IntersectionForm
from .core.complexes import ChainComplex, _parse_coefficient_ring
from .core.exceptions import DimensionError
from .core.fundamental_group import (
    FundamentalGroup,
    GroupPresentation,
    simplify_presentation,
    infer_standard_group_descriptor,
)
from .core.pi1_group_ring_scaffold import evaluate_phase2_readiness
from .core.k_theory import WhiteheadGroup, compute_whitehead_group
from .core.foundations import CONTRACT_VERSION
from .structure_set import (
    NormalInvariantsResult,
    SurgeryExactSequenceResult,
    StructureSet,
)
from .core.theorem_tags import infer_theorem_tag
from .wall_groups import ObstructionResult, WallGroupL
from .bridge.julia_bridge import julia_engine
import warnings
import itertools
import numpy as np
import sympy as sp


@dataclass
class HomeomorphismResult:
    """Structured decision object used by dimension-aware analyzers."""

    status: Literal["success", "impediment", "inconclusive", "surgery_required"]
    is_homeomorphic: bool | None
    reasoning: str
    theorem: str | None = None
    evidence: list[str] = field(default_factory=list)
    missing_data: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    certificates: dict[str, object] = field(default_factory=dict)
    exact: bool = True
    theorem_tag: str | None = None
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.theorem_tag is None:
            self.theorem_tag = infer_theorem_tag(self.theorem)

    def to_legacy_tuple(self) -> Tuple[bool | None, str]:
        return self.is_homeomorphic, self.reasoning


@dataclass
class HighDimDecisionStage:
    id: str
    title: str
    outcome: Literal["passed", "failed", "inconclusive", "skipped"]
    detail: str = ""
    exact: bool = True
    data: dict[str, object] = field(default_factory=dict)

    def to_legacy_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "outcome": self.outcome,
            "detail": self.detail,
            "exact": self.exact,
            "data": dict(self.data),
        }


@dataclass
class HighDimDecisionDag:
    dimension: int
    theorem: str
    stages: list[HighDimDecisionStage] = field(default_factory=list)
    contract_version: str = CONTRACT_VERSION

    def to_legacy_dict(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "theorem": self.theorem,
            "contract_version": self.contract_version,
            "stages": [s.to_legacy_dict() for s in self.stages],
        }


@dataclass
class HomotopyEquivalenceWitnessHook:
    provided: bool
    source: str
    exact: bool
    summary: str
    payload: dict[str, object] = field(default_factory=dict)

    def to_legacy_dict(self) -> dict[str, object]:
        return {
            "provided": self.provided,
            "source": self.source,
            "exact": self.exact,
            "summary": self.summary,
            "payload": dict(self.payload),
        }


@dataclass
class HomotopyCompletionCertificate:
    """Formal certificate of a homotopy-equivalence completion.

    Attributes:
        provided: Whether a certificate was provided.
        source: Source of the certificate.
        exact: Whether the certificate is exact.
        validated: Whether the certificate has been formally validated.
        equivalence_type: The type of equivalence certified.
        summary: Human-readable summary.
        assumptions: Mathematical assumptions for the certificate.
        payload: Raw certificate data.
    """

    provided: bool
    source: str
    exact: bool
    validated: bool
    equivalence_type: str = "homotopy_equivalence"
    summary: str = ""
    assumptions: list[str] = field(default_factory=list)
    payload: dict[str, object] = field(default_factory=dict)

    def decision_ready(self) -> bool:
        return bool(self.provided and self.exact and self.validated)

    def to_legacy_dict(self) -> dict[str, object]:
        return {
            "provided": self.provided,
            "source": self.source,
            "exact": self.exact,
            "validated": self.validated,
            "equivalence_type": self.equivalence_type,
            "summary": self.summary,
            "assumptions": list(self.assumptions),
            "payload": dict(self.payload),
            "decision_ready": self.decision_ready(),
        }


@dataclass
class ThreeManifoldRecognitionCertificate:
    provided: bool
    source: str
    exact: bool
    validated: bool
    method: str = "geometrization_recognition"
    summary: str = ""
    assumptions: list[str] = field(default_factory=list)
    payload: dict[str, object] = field(default_factory=dict)

    def decision_ready(self) -> bool:
        return bool(self.provided and self.exact and self.validated)

    def to_legacy_dict(self) -> dict[str, object]:
        return {
            "provided": self.provided,
            "source": self.source,
            "exact": self.exact,
            "validated": self.validated,
            "method": self.method,
            "summary": self.summary,
            "assumptions": list(self.assumptions),
            "payload": dict(self.payload),
            "decision_ready": self.decision_ready(),
        }


@dataclass
class ProductAssemblyCertificate:
    provided: bool
    source: str
    exact: bool
    validated: bool
    summary: str = ""
    assumptions: list[str] = field(default_factory=list)
    payload: dict[str, object] = field(default_factory=dict)

    def decision_ready(self) -> bool:
        return bool(self.provided and self.exact and self.validated)

    def to_legacy_dict(self) -> dict[str, object]:
        return {
            "provided": self.provided,
            "source": self.source,
            "exact": self.exact,
            "validated": self.validated,
            "summary": self.summary,
            "assumptions": list(self.assumptions),
            "payload": dict(self.payload),
            "decision_ready": self.decision_ready(),
        }


@dataclass
class DefiniteLatticeIsometryCertificate:
    provided: bool
    source: str
    exact: bool
    validated: bool
    isometry_matrix: list[list[int]] | None = None
    summary: str = ""
    assumptions: list[str] = field(default_factory=list)
    payload: dict[str, object] = field(default_factory=dict)

    def decision_ready(self) -> bool:
        return bool(
            self.provided
            and self.exact
            and self.validated
            and self.isometry_matrix is not None
        )

    def to_legacy_dict(self) -> dict[str, object]:
        return {
            "provided": self.provided,
            "source": self.source,
            "exact": self.exact,
            "validated": self.validated,
            "isometry_matrix": self.isometry_matrix,
            "summary": self.summary,
            "assumptions": list(self.assumptions),
            "payload": dict(self.payload),
            "decision_ready": self.decision_ready(),
        }


def _normalize_torsion(torsion: list[int]) -> list[int]:
    return sorted(abs(int(x)) for x in torsion if abs(int(x)) > 1)


def _freeze_value(value: object) -> object:
    """Return a hashable, canonical representation for nested witness data."""
    if isinstance(value, dict):
        return tuple(
            (str(k), _freeze_value(v))
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted((_freeze_value(v) for v in value), key=repr))
    if isinstance(value, np.ndarray):
        return _freeze_value(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    return value


def _normalize_homotopy_witness_hook(
    homotopy_equivalence_witness: object | None,
    homotopy_witness_hook: HomotopyEquivalenceWitnessHook | dict | None,
) -> HomotopyEquivalenceWitnessHook:
    if isinstance(homotopy_witness_hook, HomotopyEquivalenceWitnessHook):
        return homotopy_witness_hook
    if isinstance(homotopy_witness_hook, dict):
        payload = dict(homotopy_witness_hook.get("payload") or {})
        if homotopy_equivalence_witness is not None and "witness" not in payload:
            payload["witness"] = _freeze_value(homotopy_equivalence_witness)
        return HomotopyEquivalenceWitnessHook(
            provided=bool(homotopy_witness_hook.get("provided", True)),
            source=str(homotopy_witness_hook.get("source", "hook")),
            exact=bool(homotopy_witness_hook.get("exact", False)),
            summary=str(
                homotopy_witness_hook.get(
                    "summary", "Homotopy witness hook metadata provided."
                )
            ),
            payload=payload,
        )
    if homotopy_equivalence_witness is not None:
        return HomotopyEquivalenceWitnessHook(
            provided=True,
            source="explicit_witness",
            exact=False,
            summary="Homotopy-equivalence witness payload supplied; treated as non-exact unless wrapped by an explicit exact hook.",
            payload={"witness": _freeze_value(homotopy_equivalence_witness)},
        )
    return HomotopyEquivalenceWitnessHook(
        provided=False,
        source="none",
        exact=False,
        summary="No homotopy-equivalence witness hook provided.",
        payload={},
    )


def _normalize_homotopy_completion_certificate(
    homotopy_completion_certificate: HomotopyCompletionCertificate | dict | None,
    hook_state: HomotopyEquivalenceWitnessHook,
    homotopy_equivalence_witness: object | None,
) -> HomotopyCompletionCertificate:
    if isinstance(homotopy_completion_certificate, HomotopyCompletionCertificate):
        return homotopy_completion_certificate
    if isinstance(homotopy_completion_certificate, dict):
        payload = dict(homotopy_completion_certificate.get("payload") or {})
        if homotopy_equivalence_witness is not None and "witness" not in payload:
            payload["witness"] = _freeze_value(homotopy_equivalence_witness)
        assumptions = list(homotopy_completion_certificate.get("assumptions") or [])
        eq_type = str(
            homotopy_completion_certificate.get(
                "equivalence_type", "homotopy_equivalence"
            )
        )
        if eq_type not in {
            "homotopy_equivalence",
            "simple_homotopy_equivalence",
            "h_cobordism",
            "s_cobordism",
        }:
            eq_type = "homotopy_equivalence"
        return HomotopyCompletionCertificate(
            provided=bool(homotopy_completion_certificate.get("provided", True)),
            source=str(homotopy_completion_certificate.get("source", "certificate")),
            exact=bool(homotopy_completion_certificate.get("exact", False)),
            validated=bool(homotopy_completion_certificate.get("validated", False)),
            equivalence_type=eq_type,
            summary=str(
                homotopy_completion_certificate.get(
                    "summary", "Typed homotopy-completion certificate provided."
                )
            ),
            assumptions=assumptions,
            payload=payload,
        )

    # Backward-compatible bridge from phase-5 hook semantics.
    if hook_state.provided:
        return HomotopyCompletionCertificate(
            provided=True,
            source=hook_state.source,
            exact=hook_state.exact,
            validated=hook_state.exact,
            equivalence_type="homotopy_equivalence",
            summary=hook_state.summary,
            assumptions=[],
            payload=dict(hook_state.payload),
        )
    return HomotopyCompletionCertificate(
        provided=False,
        source="none",
        exact=False,
        validated=False,
        equivalence_type="homotopy_equivalence",
        summary="No homotopy-completion certificate supplied.",
        assumptions=[],
        payload={},
    )


def _normalize_3d_recognition_certificate(
    recognition_certificate: ThreeManifoldRecognitionCertificate | dict | None,
) -> ThreeManifoldRecognitionCertificate:
    if isinstance(recognition_certificate, ThreeManifoldRecognitionCertificate):
        return recognition_certificate
    if hasattr(recognition_certificate, "to_recognition_certificate"):
        try:
            candidate = recognition_certificate.to_recognition_certificate()
            if isinstance(candidate, ThreeManifoldRecognitionCertificate):
                return candidate
        except Exception:
            pass
    if isinstance(recognition_certificate, dict):
        return ThreeManifoldRecognitionCertificate(
            provided=bool(recognition_certificate.get("provided", True)),
            source=str(recognition_certificate.get("source", "certificate")),
            exact=bool(recognition_certificate.get("exact", False)),
            validated=bool(recognition_certificate.get("validated", False)),
            method=str(
                recognition_certificate.get("method", "geometrization_recognition")
            ),
            summary=str(
                recognition_certificate.get(
                    "summary", "3-manifold recognition certificate provided."
                )
            ),
            assumptions=list(recognition_certificate.get("assumptions") or []),
            payload=dict(recognition_certificate.get("payload") or {}),
        )
    return ThreeManifoldRecognitionCertificate(
        provided=False,
        source="none",
        exact=False,
        validated=False,
        method="geometrization_recognition",
        summary="No 3-manifold recognition certificate provided.",
        assumptions=[],
        payload={},
    )


def _normalize_product_assembly_certificate(
    product_assembly_certificate: ProductAssemblyCertificate | dict | None,
) -> ProductAssemblyCertificate:
    if isinstance(product_assembly_certificate, ProductAssemblyCertificate):
        return product_assembly_certificate
    if isinstance(product_assembly_certificate, dict):
        return ProductAssemblyCertificate(
            provided=bool(product_assembly_certificate.get("provided", True)),
            source=str(product_assembly_certificate.get("source", "certificate")),
            exact=bool(product_assembly_certificate.get("exact", False)),
            validated=bool(product_assembly_certificate.get("validated", False)),
            summary=str(
                product_assembly_certificate.get(
                    "summary", "Product-group assembly certificate provided."
                )
            ),
            assumptions=list(product_assembly_certificate.get("assumptions") or []),
            payload=dict(product_assembly_certificate.get("payload") or {}),
        )
    return ProductAssemblyCertificate(
        provided=False,
        source="none",
        exact=False,
        validated=False,
        summary="No product-group assembly certificate provided.",
        assumptions=[],
        payload={},
    )


def _normalize_definite_lattice_isometry_certificate(
    certificate: DefiniteLatticeIsometryCertificate | dict | None,
) -> DefiniteLatticeIsometryCertificate:
    if isinstance(certificate, DefiniteLatticeIsometryCertificate):
        return certificate
    if isinstance(certificate, dict):
        matrix = certificate.get("isometry_matrix")
        if matrix is not None:
            matrix = [[int(x) for x in row] for row in matrix]
        return DefiniteLatticeIsometryCertificate(
            provided=bool(certificate.get("provided", True)),
            source=str(certificate.get("source", "certificate")),
            exact=bool(certificate.get("exact", False)),
            validated=bool(certificate.get("validated", False)),
            isometry_matrix=matrix,
            summary=str(
                certificate.get(
                    "summary", "Definite-lattice isometry certificate provided."
                )
            ),
            assumptions=list(certificate.get("assumptions") or []),
            payload=dict(certificate.get("payload") or {}),
        )
    return DefiniteLatticeIsometryCertificate(
        provided=False,
        source="none",
        exact=False,
        validated=False,
        isometry_matrix=None,
        summary="No definite-lattice isometry certificate provided.",
        assumptions=[],
        payload={},
    )


def _canonical_coefficient_ring(raw_ring: object | None) -> str | None:
    """Returns a canonical string representation of a coefficient ring.

    Args:
        raw_ring: The raw ring descriptor.

    Returns:
        Canonical ring name (e.g., 'Z', 'Z/2Z', 'Q').
    """
    if raw_ring is None:
        return None
    ring = str(raw_ring).strip()
    if not ring:
        return None
    try:
        kind, p = _parse_coefficient_ring(ring)
    except Exception:
        return ring.upper()
    if kind == "ZMOD" and p is not None:
        return f"Z/{p}Z"
    return kind


def _parse_ring_certificate(
    certificate: dict | None,
    theorem: str,
    label: str,
) -> tuple[dict[str, object], HomeomorphismResult | None]:
    """Normalize a cohomology-ring witness into a canonical, comparable form."""
    if certificate is None:
        return {}, None
    if not isinstance(certificate, dict):
        return {}, HomeomorphismResult(
            status="inconclusive",
            is_homeomorphic=None,
            reasoning="INCONCLUSIVE: {} ring certificate must be a dictionary.".format(
                label
            ),
            theorem=theorem,
            missing_data=["Structured ring certificate for {}".format(label)],
            exact=False,
        )

    payload = dict(certificate)
    normalized: dict[str, object] = {}
    normalized["coefficient_ring"] = _canonical_coefficient_ring(
        payload.get("coefficient_ring")
    )

    basis = payload.get("basis")
    if basis is None:
        basis = payload.get("generators")
    if basis is not None:
        normalized["basis"] = _freeze_value(basis)

    unit = payload.get("unit")
    if unit is None:
        unit = payload.get("one")
    if unit is not None:
        normalized["unit"] = _freeze_value(unit)

    products = payload.get("products")
    if products is None:
        products = payload.get("cup_products")
    if products is None:
        products = payload.get("multiplication")

    standard_keys = {
        "coefficient_ring",
        "basis",
        "generators",
        "unit",
        "one",
        "products",
        "cup_products",
        "multiplication",
        "groups",
        "cohomology",
        "H",
        "notes",
        "name",
        "label",
    }
    if products is None and not any(
        k in payload for k in {"groups", "cohomology", "H"}
    ):
        products = {k: v for k, v in payload.items() if k not in standard_keys}

    if products is not None:
        normalized["products"] = _freeze_value(products)

    if "groups" in payload or "cohomology" in payload or "H" in payload:
        groups = payload.get("groups")
        if groups is None:
            groups = payload.get("cohomology")
        if groups is None:
            groups = payload.get("H")
        normalized["groups"] = _freeze_value(groups)

    return normalized, None


def _check_cohomology_equivalence(
    c1: ChainComplex,
    c2: ChainComplex,
    max_dim: int,
    theorem: str,
    allow_approx: bool,
    cohomology_signature_1: dict | None = None,
    cohomology_signature_2: dict | None = None,
) -> HomeomorphismResult | None:
    ring1 = _canonical_coefficient_ring(c1.coefficient_ring)
    ring2 = _canonical_coefficient_ring(c2.coefficient_ring)
    if ring1 != ring2:
        return HomeomorphismResult(
            status="inconclusive",
            is_homeomorphic=None,
            reasoning=(
                "INCONCLUSIVE: Cohomology comparison requires a shared coefficient ring; "
                "received %r vs %r." % (c1.coefficient_ring, c2.coefficient_ring)
            ),
            theorem=theorem,
            missing_data=["Common coefficient ring for cohomology invariants"],
        )

    if (cohomology_signature_1 is None) != (cohomology_signature_2 is None):
        return HomeomorphismResult(
            status="inconclusive",
            is_homeomorphic=None,
            reasoning=(
                "INCONCLUSIVE: Manual cohomology signatures must be supplied for both manifolds."
            ),
            theorem=theorem,
            missing_data=["Cohomology signature for both manifolds"],
        )

    def _parse_cohomology_signature(
        signature: dict | None,
        label: str,
    ) -> tuple[
        dict[int, tuple[int, list[int]]], str | None, HomeomorphismResult | None
    ]:
        if signature is None:
            return {}, None, None
        if not isinstance(signature, dict):
            return (
                {},
                None,
                HomeomorphismResult(
                    status="inconclusive",
                    is_homeomorphic=None,
                    reasoning=f"INCONCLUSIVE: {label} cohomology signature must be a dictionary.",
                    theorem=theorem,
                    missing_data=[f"Valid cohomology signature for {label}"],
                    exact=False,
                ),
            )

        ring = _canonical_coefficient_ring(signature.get("coefficient_ring"))
        groups = signature.get("groups")
        if groups is None:
            groups = signature.get("cohomology")
        if groups is None:
            groups = signature.get("H")
        if groups is None:
            groups = {
                k: v
                for k, v in signature.items()
                if k not in {"coefficient_ring", "name", "label", "notes"}
            }

        if not isinstance(groups, dict):
            return (
                {},
                ring,
                HomeomorphismResult(
                    status="inconclusive",
                    is_homeomorphic=None,
                    reasoning=f"INCONCLUSIVE: {label} cohomology signature must map degrees to group data.",
                    theorem=theorem,
                    missing_data=[f"Degree-indexed cohomology data for {label}"],
                    exact=False,
                ),
            )

        normalized: dict[int, tuple[int, list[int]]] = {}
        for degree_key, entry in groups.items():
            if isinstance(degree_key, int):
                degree = degree_key
            else:
                match = re.search(r"(\d+)$", str(degree_key).strip())
                if match is None:
                    return (
                        {},
                        ring,
                        HomeomorphismResult(
                            status="inconclusive",
                            is_homeomorphic=None,
                            reasoning=f"INCONCLUSIVE: {label} cohomology signature contains an unparseable degree key {degree_key!r}.",
                            theorem=theorem,
                            missing_data=[f"Parseable cohomology degrees for {label}"],
                            exact=False,
                        ),
                    )
                degree = int(match.group(1))

            if isinstance(entry, dict):
                if "rank" not in entry:
                    return (
                        {},
                        ring,
                        HomeomorphismResult(
                            status="inconclusive",
                            is_homeomorphic=None,
                            reasoning=f"INCONCLUSIVE: {label} cohomology signature for degree {degree} is missing a rank.",
                            theorem=theorem,
                            missing_data=[f"Rank for H^{degree}({label})"],
                            exact=False,
                        ),
                    )
                rank = int(entry["rank"])
                torsion = entry.get("torsion", [])
            elif isinstance(entry, (tuple, list)) and len(entry) == 2:
                rank = int(entry[0])
                torsion = entry[1]
            elif isinstance(entry, (int, np.integer)):
                rank = int(entry)
                torsion = []
            else:
                return (
                    {},
                    ring,
                    HomeomorphismResult(
                        status="inconclusive",
                        is_homeomorphic=None,
                        reasoning=f"INCONCLUSIVE: {label} cohomology signature for degree {degree} must be a dict, pair, or integer rank.",
                        theorem=theorem,
                        missing_data=[f"Structured H^{degree} data for {label}"],
                        exact=False,
                    ),
                )

            torsion_list = _normalize_torsion(list(torsion or []))
            normalized[degree] = (rank, torsion_list)

        return normalized, ring, None

    sig1, ring1, sig_issue_1 = _parse_cohomology_signature(
        cohomology_signature_1, "first manifold"
    )
    sig2, ring2, sig_issue_2 = _parse_cohomology_signature(
        cohomology_signature_2, "second manifold"
    )

    if sig_issue_1 is not None:
        return sig_issue_1
    if sig_issue_2 is not None:
        return sig_issue_2

    if ring1 is not None and ring2 is not None and ring1 != ring2:
        return HomeomorphismResult(
            status="inconclusive",
            is_homeomorphic=None,
            reasoning=(
                "INCONCLUSIVE: Manual cohomology signatures specify different coefficient rings; "
                f"received {ring1!r} vs {ring2!r}."
            ),
            theorem=theorem,
            missing_data=["Matching coefficient ring in manual cohomology signatures"],
        )

    if sig1 or sig2:
        for n in range(max_dim + 1):
            if n in sig1 and n in sig2 and sig1[n] != sig2[n]:
                return HomeomorphismResult(
                    status="impediment",
                    is_homeomorphic=False,
                    reasoning=(
                        f"IMPEDIMENT: Manual cohomology signatures differ in degree {n} "
                        f"(Rank: {sig1[n][0]} vs {sig2[n][0]}, Torsion: {sig1[n][1]} vs {sig2[n][1]})."
                    ),
                    theorem=theorem,
                    evidence=[f"Manual H^{n} mismatch"],
                )

    for n in range(max_dim + 1):
        computed_1: tuple[int, list[int]] | None = None
        computed_2: tuple[int, list[int]] | None = None
        try:
            computed_1 = c1.cohomology(n)
        except Exception as e:
            if allow_approx:
                warnings.warn(
                    f"Topological Hint: Cohomology extraction failed for the first manifold at dimension {n} ({e!r})."
                )
            if n not in sig1:
                return HomeomorphismResult(
                    status="inconclusive",
                    is_homeomorphic=None,
                    reasoning=f"INCONCLUSIVE: Exact cohomology extraction failed at dimension {n} for the first manifold ({e!r}).",
                    theorem=theorem,
                    missing_data=[f"Exact H^{n} for the first manifold"],
                    exact=False,
                )
        try:
            computed_2 = c2.cohomology(n)
        except Exception as e:
            if allow_approx:
                warnings.warn(
                    f"Topological Hint: Cohomology extraction failed for the second manifold at dimension {n} ({e!r})."
                )
            if n not in sig2:
                return HomeomorphismResult(
                    status="inconclusive",
                    is_homeomorphic=None,
                    reasoning=f"INCONCLUSIVE: Exact cohomology extraction failed at dimension {n} for the second manifold ({e!r}).",
                    theorem=theorem,
                    missing_data=[f"Exact H^{n} for the second manifold"],
                    exact=False,
                )

        if computed_1 is None:
            computed_1 = sig1.get(n)
        if computed_2 is None:
            computed_2 = sig2.get(n)

        if computed_1 is None or computed_2 is None:
            return HomeomorphismResult(
                status="inconclusive",
                is_homeomorphic=None,
                reasoning=f"INCONCLUSIVE: Cohomology data is incomplete in degree {n}.",
                theorem=theorem,
                missing_data=[f"H^{n} for both manifolds"],
                exact=False,
            )

        r1, t1 = computed_1
        r2, t2 = computed_2
        t1n = _normalize_torsion(t1)
        t2n = _normalize_torsion(t2)
        if sig1.get(n) is not None and computed_1 != sig1[n]:
            return HomeomorphismResult(
                status="impediment",
                is_homeomorphic=False,
                reasoning=(
                    f"IMPEDIMENT: The first manifold's computed H^{n} disagrees with the supplied manual signature "
                    f"(computed rank/torsion {computed_1}, manual {sig1[n]})."
                ),
                theorem=theorem,
                evidence=[f"Manual H^{n} mismatch for the first manifold"],
            )
        if sig2.get(n) is not None and computed_2 != sig2[n]:
            return HomeomorphismResult(
                status="impediment",
                is_homeomorphic=False,
                reasoning=(
                    f"IMPEDIMENT: The second manifold's computed H^{n} disagrees with the supplied manual signature "
                    f"(computed rank/torsion {computed_2}, manual {sig2[n]})."
                ),
                theorem=theorem,
                evidence=[f"Manual H^{n} mismatch for the second manifold"],
            )

        if r1 != r2 or t1n != t2n:
            return HomeomorphismResult(
                status="impediment",
                is_homeomorphic=False,
                reasoning=(
                    f"IMPEDIMENT: Cohomology groups differ in dimension {n} "
                    f"(Rank: {r1} vs {r2}, Torsion: {t1n} vs {t2n})."
                ),
                theorem=theorem,
                evidence=[f"H^{n} mismatch"],
            )

    return None


def _check_cup_product_compatibility(
    cohomology_ring_signature_1: dict | None,
    cohomology_ring_signature_2: dict | None,
    cup_product_signature_1: dict | None,
    cup_product_signature_2: dict | None,
    theorem: str,
) -> HomeomorphismResult | None:
    ring_1, issue_1 = _parse_ring_certificate(
        cohomology_ring_signature_1
        if cohomology_ring_signature_1 is not None
        else cup_product_signature_1,
        theorem,
        "first manifold",
    )
    ring_2, issue_2 = _parse_ring_certificate(
        cohomology_ring_signature_2
        if cohomology_ring_signature_2 is not None
        else cup_product_signature_2,
        theorem,
        "second manifold",
    )

    if issue_1 is not None:
        return issue_1
    if issue_2 is not None:
        return issue_2

    if not ring_1 and not ring_2:
        return None
    if bool(ring_1) != bool(ring_2):
        return HomeomorphismResult(
            status="inconclusive",
            is_homeomorphic=None,
            reasoning=(
                "INCONCLUSIVE: Cohomology-ring comparison requires cup-product signatures or structured witnesses "
                "for both manifolds."
            ),
            theorem=theorem,
            missing_data=["Cohomology-ring witness for both manifolds"],
        )

    if (
        ring_1.get("coefficient_ring") is not None
        and ring_2.get("coefficient_ring") is not None
        and ring_1.get("coefficient_ring") != ring_2.get("coefficient_ring")
    ):
        return HomeomorphismResult(
            status="inconclusive",
            is_homeomorphic=None,
            reasoning=(
                "INCONCLUSIVE: Cohomology-ring comparison requires a shared coefficient ring; "
                "received {} vs {}.".format(
                    ring_1.get("coefficient_ring"), ring_2.get("coefficient_ring")
                )
            ),
            theorem=theorem,
            missing_data=["Matching coefficient ring in ring witnesses"],
        )

    if ring_1 != ring_2:
        return HomeomorphismResult(
            status="impediment",
            is_homeomorphic=False,
            reasoning="IMPEDIMENT: Cohomology-ring witnesses differ (cup-product incompatibility; basis/unit/product mismatch).",
            theorem=theorem,
            evidence=["Ring witness mismatch"],
        )
    return None


def _det_int_small(M: np.ndarray) -> int:
    n = M.shape[0]
    if n == 0:
        return 1
    if n == 1:
        return int(M[0, 0])
    if n == 2:
        return int(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])
    return int(sp.Matrix(M.tolist()).det())


def _presentation_key(
    pi1: FundamentalGroup,
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    simplified = simplify_presentation(
        list(pi1.generators), [list(rel) for rel in pi1.relations]
    )
    rels = tuple(tuple(tok for tok in rel) for rel in simplified.relations)
    return tuple(simplified.generators), rels


def _infer_pi_group_descriptor(
    pi1: FundamentalGroup | None, pi_group: str | GroupPresentation | None
) -> str | GroupPresentation | None:
    if pi_group is not None:
        return (
            pi_group.normalized()
            if isinstance(pi_group, GroupPresentation)
            else str(pi_group).strip()
        )
    if pi1 is None:
        return None
    return infer_standard_group_descriptor(pi1)


def _homology_sphere_like(c: ChainComplex, dim: int) -> bool | None:
    try:
        r0, t0 = c.homology(0)
        if r0 != 1 or _normalize_torsion(t0):
            return False
        for n in range(1, dim):
            r, t = c.homology(n)
            if r != 0 or _normalize_torsion(t):
                return False
        rdim, tdim = c.homology(dim)
        return rdim == 1 and _normalize_torsion(tdim) == []
    except Exception:
        return None


def _search_integer_isometry(
    Q1: np.ndarray, Q2: np.ndarray, max_entry: int = 2
) -> np.ndarray | None:
    """Search for U in GL_n(Z) with U^T Q1 U = Q2.

    For definite forms, use an exact finite lattice search (optionally accelerated by Julia).
    For non-definite forms, use a bounded brute-force fallback.

    Args:
        Q1: First symmetric integer matrix.
        Q2: Second symmetric integer matrix.
        max_entry: Maximum entry value for search (fallback mode).

    Returns:
        The isometry matrix U if found, None otherwise.
    """
    if (
        Q1.ndim != 2
        or Q2.ndim != 2
        or Q1.shape[0] != Q1.shape[1]
        or Q2.shape[0] != Q2.shape[1]
    ):
        return None
    if Q1.shape != Q2.shape:
        return None

    q1 = np.asarray(Q1, dtype=np.int64)
    q2 = np.asarray(Q2, dtype=np.int64)
    n = q1.shape[0]

    # Exact branch for definite forms: finite search by lattice-vector norms/pairings.
    eig1 = np.linalg.eigvalsh(q1.astype(float))
    eig2 = np.linalg.eigvalsh(q2.astype(float))
    tol = (
        max(q1.shape)
        * np.finfo(float).eps
        * max(1.0, float(np.max(np.abs(np.concatenate([eig1, eig2])))))
    )
    pos1 = bool(np.all(eig1 > tol))
    neg1 = bool(np.all(eig1 < -tol))
    pos2 = bool(np.all(eig2 > tol))
    neg2 = bool(np.all(eig2 < -tol))

    if (pos1 and pos2) or (neg1 and neg2):
        if julia_engine.available:
            try:
                candidate = julia_engine.integral_lattice_isometry(q1, q2)
                if candidate is not None and np.array_equal(
                    candidate.T @ q1 @ candidate, q2
                ):
                    return candidate
            except Exception:
                # Fall through to Python exact solver.
                pass

        q1_def = q1 if pos1 else -q1
        q2_def = q2 if pos2 else -q2
        lam_min = float(np.min(np.linalg.eigvalsh(q1_def.astype(float))))
        if lam_min <= 0:
            return None

        diag_targets = [int(q2_def[i, i]) for i in range(n)]
        if any(t <= 0 for t in diag_targets):
            return None

        radii = [int(np.floor(np.sqrt(t / lam_min))) + 1 for t in diag_targets]
        search_radius = max(radii) if radii else 0
        values = range(-search_radius, search_radius + 1)

        # Enumerate all vectors of a prescribed quadratic norm in the ambient lattice.
        vectors_by_norm: dict[int, list[np.ndarray]] = {
            t: [] for t in set(diag_targets)
        }
        max_vectors_per_norm = 20000
        for entries in itertools.product(values, repeat=n):
            v = np.array(entries, dtype=np.int64)
            qv = int(v @ q1_def @ v)
            bucket = vectors_by_norm.get(qv)
            if bucket is None:
                continue
            bucket.append(v)
            if len(bucket) > max_vectors_per_norm:
                return None

        for t in diag_targets:
            if not vectors_by_norm.get(t):
                return None

        order = list(range(n))
        order.sort(key=lambda j: len(vectors_by_norm[diag_targets[j]]))
        cols: list[np.ndarray] = [np.zeros(n, dtype=np.int64) for _ in range(n)]
        chosen_original_indices: list[int] = []

        def backtrack(pos: int) -> np.ndarray | None:
            if pos == n:
                U = np.column_stack(cols)
                if abs(_det_int_small(U)) == 1 and np.array_equal(U.T @ q1 @ U, q2):
                    return U
                return None

            j = order[pos]
            target_norm = diag_targets[j]
            for v in vectors_by_norm[target_norm]:
                pair_ok = True
                for prev_pos, i in enumerate(chosen_original_indices):
                    if int(cols[i] @ q1_def @ v) != int(q2_def[i, j]):
                        pair_ok = False
                        break
                if not pair_ok:
                    continue
                cols[j] = v
                chosen_original_indices.append(j)
                out = backtrack(pos + 1)
                if out is not None:
                    return out
                chosen_original_indices.pop()

            return None

        return backtrack(0)

    # High-performance lattice isomorphism check for definite forms.
    # We prune the search by only considering vectors v such that v^T Q1 v = Q2_jj.
    # This is equivalent to finding all vectors in the lattice of Q1 with a specific norm.
    diag_targets = np.diag(Q2).tolist()
    
    # 1. Pre-calculate all vectors of required norms in Q1
    # For speed in Python, we only do this for small search radii.
    lam_min = np.min(np.linalg.eigvalsh(Q1))
    if lam_min <= 1e-10:
        return None
        
    vectors_by_norm: dict[int, list[np.ndarray]] = {t: [] for t in set(diag_targets)}
    
    # Estimate search box
    max_t = max(diag_targets)
    r = int(np.floor(np.sqrt(max_t / lam_min))) + 1
    if r > 3 and n > 3: # Safety cap for Python-side brute force
         return None
         
    values = range(-r, r + 1)
    max_total_vectors = 10000
    count = 0
    for entries in itertools.product(values, repeat=n):
        v = np.array(entries, dtype=np.int64)
        norm = int(v.T @ Q1 @ v)
        if norm in vectors_by_norm:
            vectors_by_norm[norm].append(v)
            count += 1
            if count > max_total_vectors:
                return None

    if any(not v for v in vectors_by_norm.values()):
        return None

    # 2. Backtracking search for the isometry matrix
    # We choose columns one by one such that U[:, i].T Q1 U[:, j] = Q2[i, j]
    cols: list[np.ndarray] = [np.zeros(n, dtype=np.int64) for _ in range(n)]
    
    def backtrack(idx: int) -> np.ndarray | None:
        if idx == n:
            U = np.column_stack(cols)
            if abs(int(round(np.linalg.det(U)))) == 1:
                if np.array_equal(U.T @ Q1 @ U, Q2):
                    return U
            return None
            
        target_norm = diag_targets[idx]
        for v in vectors_by_norm[target_norm]:
            # Check consistency with already chosen columns
            ok = True
            for prev in range(idx):
                if int(v.T @ Q1 @ cols[prev]) != int(Q2[idx, prev]):
                    ok = False
                    break
            if ok:
                cols[idx] = v
                res = backtrack(idx + 1)
                if res is not None:
                    return res
        return None

    return backtrack(0)


def analyze_homeomorphism_1d_result(
    c1: ChainComplex,
    c2: ChainComplex,
    allow_approx: bool = False,
) -> HomeomorphismResult:
    """
    Analyzes the potential for homeomorphism between two 1-dimensional manifolds.
    
    All closed 1-manifolds are disjoint unions of circles (S^1).
    Two closed 1-manifolds are homeomorphic if and only if they have the same
    number of components (rank of H_0).
    """
    try:
        h0_1 = c1.homology(0)
        h0_2 = c2.homology(0)
        h1_1 = c1.homology(1)
        h1_2 = c2.homology(1)
    except Exception as e:
        if allow_approx:
             warnings.warn(f"1D homology extraction failed: {e}. Inconclusive result.")
        return HomeomorphismResult(
            status="inconclusive",
            is_homeomorphic=None,
            reasoning=f"INCONCLUSIVE: Homology extraction failed for 1D analysis: {e}",
            missing_data=["H_0", "H_1"],
        )

    # For a connected closed 1-manifold (S^1), H_0 = Z and H_1 = Z.
    # For k components, H_0 = Z^k and H_1 = Z^k.
    if h0_1 != h0_2:
        return HomeomorphismResult(
            status="impediment",
            is_homeomorphic=False,
            reasoning=f"IMPEDIMENT: Number of components (H_0 rank) differs: {h0_1[0]} vs {h0_2[0]}.",
            evidence=[f"H_0_1={h0_1}", f"H_0_2={h0_2}"],
        )
    
    if h1_1 != h1_2:
        return HomeomorphismResult(
            status="impediment",
            is_homeomorphic=False,
            reasoning=f"IMPEDIMENT: First homology (H_1) differs: {h1_1} vs {h1_2}.",
            evidence=[f"H_1_1={h1_1}", f"H_1_2={h1_2}"],
        )

    return HomeomorphismResult(
        status="success",
        is_homeomorphic=True,
        reasoning="SUCCESS: 1-manifolds are homeomorphic (same number of S^1 components).",
        theorem="Classification of 1-Manifolds",
        evidence=[f"H_0 match (rank={h0_1[0]})", f"H_1 match (rank={h1_1[0]})"],
    )

def analyze_homeomorphism_1d(
    c1: ChainComplex,
    c2: ChainComplex,
    allow_approx: bool = False,
) -> Tuple[bool | None, str]:
    return analyze_homeomorphism_1d_result(
        c1, c2, allow_approx=allow_approx
    ).to_legacy_tuple()


def analyze_homeomorphism_2d_result(
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
) -> HomeomorphismResult:
    """
    Analyzes the potential for homeomorphism between two 2-dimensional manifolds (surfaces).

    Based on the Classification of Closed Surfaces:
    Two closed surfaces are homeomorphic if and only if they have:
    1. The same orientability (H_2 = Z vs H_2 = 0).
    2. The same Euler characteristic (or genus).

    Returns
    -------
    is_homeomorphic : bool
    reasoning : str
    """
    try:
        r2_1, _ = c1.homology(2)
        r2_2, _ = c2.homology(2)
    except Exception as e:
        if allow_approx:
            warnings.warn(
                f"Topological Hint: H_2 homology extraction failed ({e!r}). Exact classification disabled."
            )
        return HomeomorphismResult(
            status="inconclusive",
            is_homeomorphic=None,
            reasoning=f"INCONCLUSIVE: Exact H_2 computation failed ({e!r}).",
            theorem="Classification of Closed Surfaces",
            missing_data=["Exact H_2"],
            exact=False,
        )

    orientable_1 = r2_1 == 1
    orientable_2 = r2_2 == 1

    if orientable_1 != orientable_2:
        return HomeomorphismResult(
            status="impediment",
            is_homeomorphic=False,
            reasoning=f"IMPEDIMENT: Orientability mismatch. Manifold 1 is {'Orientable' if orientable_1 else 'Non-Orientable'}, Manifold 2 is {'Orientable' if orientable_2 else 'Non-Orientable'}.",
            theorem="Classification of Closed Surfaces",
            evidence=[f"H_2 ranks: {r2_1} vs {r2_2}"],
        )

    try:
        r1_1, t1_1 = c1.homology(1)
        r1_2, t1_2 = c2.homology(1)
    except Exception as e:
        if allow_approx:
            warnings.warn(
                f"Topological Hint: H_1 homology extraction failed ({e!r}). Exact classification disabled."
            )
        return HomeomorphismResult(
            status="inconclusive",
            is_homeomorphic=None,
            reasoning=f"INCONCLUSIVE: Exact H_1 computation failed ({e!r}).",
            theorem="Classification of Closed Surfaces",
            missing_data=["Exact H_1"],
            exact=False,
        )

    if r1_1 != r1_2:
        return HomeomorphismResult(
            status="impediment",
            is_homeomorphic=False,
            reasoning=f"IMPEDIMENT: Genus mismatch. H_1 rank differs ({r1_1} vs {r1_2}).",
            theorem="Classification of Closed Surfaces",
            evidence=[f"H_1 ranks: {r1_1} vs {r1_2}"],
        )

    # Check torsion in H_1 (relevant for non-orientable surfaces like RP^2 vs Klein Bottle)
    t1_1n = _normalize_torsion(t1_1)
    t1_2n = _normalize_torsion(t1_2)
    if t1_1n != t1_2n:
        return HomeomorphismResult(
            status="impediment",
            is_homeomorphic=False,
            reasoning=f"IMPEDIMENT: Torsion in H_1 differs ({t1_1n} vs {t1_2n}).",
            theorem="Classification of Closed Surfaces",
            evidence=["Invariant-factor torsion mismatch in H_1"],
        )

    coho_check = _check_cohomology_equivalence(
        c1,
        c2,
        max_dim=2,
        theorem="Classification of Closed Surfaces",
        allow_approx=allow_approx,
        cohomology_signature_1=cohomology_signature_1,
        cohomology_signature_2=cohomology_signature_2,
    )
    if coho_check is not None:
        return coho_check

    cup_check = _check_cup_product_compatibility(
        cohomology_ring_signature_1,
        cohomology_ring_signature_2,
        cup_product_signature_1,
        cup_product_signature_2,
        theorem="Classification of Closed Surfaces",
    )
    if cup_check is not None:
        return cup_check

    return HomeomorphismResult(
        status="success",
        is_homeomorphic=True,
        reasoning="SUCCESS: Homeomorphism established via the Classification Theorem of Closed Surfaces using exact H_1/H_2 invariants.",
        theorem="Classification of Closed Surfaces",
        evidence=["Orientability match", "H_1 rank match", "H_1 torsion match"],
    )


def analyze_homeomorphism_2d(
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
) -> Tuple[bool | None, str]:
    return analyze_homeomorphism_2d_result(
        c1,
        c2,
        allow_approx=allow_approx,
        cohomology_signature_1=cohomology_signature_1,
        cohomology_signature_2=cohomology_signature_2,
        cohomology_ring_signature_1=cohomology_ring_signature_1,
        cohomology_ring_signature_2=cohomology_ring_signature_2,
        cup_product_signature_1=cup_product_signature_1,
        cup_product_signature_2=cup_product_signature_2,
    ).to_legacy_tuple()


def analyze_homeomorphism_3d_result(
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
) -> HomeomorphismResult:
    """
    Analyzes the potential for homeomorphism between two 3-dimensional manifolds.

    Warning: 3-manifolds are classified by Thurston's Geometrization (Perelman, 2003).
    Algebraic topology alone (homology) is insufficient to prove homeomorphism in general
    (e.g., Poincare homology spheres have the same homology as S^3 but different fundamental groups).
    """
    rec_cert = _normalize_3d_recognition_certificate(recognition_certificate)
    # Check basic homology equivalence (exact-only for certifying statements).
    for n in range(4):
        try:
            r_1, t_1 = c1.homology(n)
            r_2, t_2 = c2.homology(n)
        except Exception as e:
            if allow_approx:
                warnings.warn(
                    f"Topological Hint: Homology extraction failed at dimension {n} ({e!r}). Exact classification disabled."
                )
            return HomeomorphismResult(
                status="inconclusive",
                is_homeomorphic=None,
                reasoning=f"INCONCLUSIVE: Exact homology extraction failed at dimension {n} ({e!r}).",
                theorem="Geometrization / 3-manifold recognition",
                missing_data=[f"Exact H_{n}"],
                exact=False,
            )

        t_1n = _normalize_torsion(t_1)
        t_2n = _normalize_torsion(t_2)
        if r_1 != r_2 or t_1n != t_2n:
            return HomeomorphismResult(
                status="impediment",
                is_homeomorphic=False,
                reasoning=f"IMPEDIMENT: Homology groups differ in dimension {n} (Rank: {r_1} vs {r_2}, Torsion: {t_1n} vs {t_2n}).",
                theorem="Geometrization / 3-manifold recognition",
                evidence=[f"H_{n} mismatch"],
            )

    if (pi1_1 is None) != (pi1_2 is None):
        return HomeomorphismResult(
            status="inconclusive",
            is_homeomorphic=None,
            reasoning="INCONCLUSIVE: Fundamental-group data supplied for only one manifold.",
            theorem="Geometrization / 3-manifold recognition",
            missing_data=["Matched pi_1 data for both manifolds"],
        )

    if (
        pi1_1 is not None
        and pi1_2 is not None
        and _presentation_key(pi1_1) != _presentation_key(pi1_2)
    ):
        # We cannot conclude False just because presentations differ (group isomorphism is undecidable).
        # However, we can check the abelianization as a necessary invariant.
        try:
            h1_1 = _normalize_torsion(c1.homology(1)[1])
            h1_2 = _normalize_torsion(c2.homology(1)[1])
            r1_1 = c1.homology(1)[0]
            r1_2 = c2.homology(1)[0]
            if r1_1 != r1_2 or h1_1 != h1_2:
                return HomeomorphismResult(
                    status="impediment",
                    is_homeomorphic=False,
                    reasoning=f"IMPEDIMENT: First homology (abelianization of pi_1) differs: H_1_1=(rank {r1_1}, torsion {h1_1}), H_1_2=(rank {r1_2}, torsion {h1_2}).",
                    theorem="Geometrization / 3-manifold recognition",
                    evidence=["pi_1 abelianization mismatch"],
                )
        except Exception:
            pass

        return HomeomorphismResult(
            status="inconclusive",
            is_homeomorphic=None,
            reasoning="INCONCLUSIVE: Fundamental group presentations differ, but their isomorphism is undecidable. Further geometric or representation-theoretic analysis is required.",
            theorem="Geometrization / 3-manifold recognition",
            missing_data=["Certified pi_1 isomorphism witness"],
        )

    coho_check = _check_cohomology_equivalence(
        c1,
        c2,
        max_dim=3,
        theorem="Geometrization / 3-manifold recognition",
        allow_approx=allow_approx,
        cohomology_signature_1=cohomology_signature_1,
        cohomology_signature_2=cohomology_signature_2,
    )
    if coho_check is not None:
        return coho_check

    cup_check = _check_cup_product_compatibility(
        cohomology_ring_signature_1,
        cohomology_ring_signature_2,
        cup_product_signature_1,
        cup_product_signature_2,
        theorem="Geometrization / 3-manifold recognition",
    )
    if cup_check is not None:
        return cup_check

    s1 = _homology_sphere_like(c1, 3)
    s2 = _homology_sphere_like(c2, 3)
    if s1 and s2:
        if pi1_1 is None:
            return HomeomorphismResult(
                status="inconclusive",
                is_homeomorphic=None,
                reasoning="INCONCLUSIVE: Both are homology-sphere candidates, but pi_1 data is required before applying Poincare/Geometrization conclusions.",
                theorem="Poincare Conjecture / Geometrization",
                missing_data=["pi_1 for both manifolds"],
            )
        pi_desc_1 = _infer_pi_group_descriptor(pi1_1, None)
        pi_desc_2 = _infer_pi_group_descriptor(pi1_2, None)
        if pi_desc_1 == "1" and pi_desc_2 == "1":
            return HomeomorphismResult(
                status="success",
                is_homeomorphic=True,
                reasoning="SUCCESS: Both manifolds satisfy the homology-sphere conditions and have trivial pi_1, so they are homeomorphic by the Poincaré conjecture.",
                theorem="Poincaré Conjecture / Geometrization",
                evidence=[
                    "Homology sphere checks passed",
                    "Trivial pi_1 for both manifolds",
                ],
                assumptions=["Closed connected 3-manifold hypotheses must hold"],
            )
        if pi_desc_1 is not None and pi_desc_2 is not None and pi_desc_1 == pi_desc_2:
            if rec_cert.decision_ready():
                return HomeomorphismResult(
                    status="success",
                    is_homeomorphic=True,
                    reasoning=(
                        "SUCCESS: Homology-sphere side conditions plus a decision-ready 3-manifold recognition "
                        "certificate support homeomorphism in this API model."
                    ),
                    theorem="Geometrization / 3-manifold recognition",
                    evidence=[
                        "Homology sphere checks passed",
                        "Matching pi_1 descriptor",
                        "Decision-ready 3D recognition certificate",
                    ],
                    assumptions=["Closed connected 3-manifold hypotheses must hold"]
                    + rec_cert.assumptions,
                    certificates={
                        "three_manifold_recognition_certificate": rec_cert.to_legacy_dict()
                    },
                    exact=True,
                )
            return HomeomorphismResult(
                status="inconclusive",
                is_homeomorphic=None,
                reasoning="INCONCLUSIVE: Homology-sphere and matching pi_1 evidence is promising; a full 3-manifold recognition pipeline is still required in this API.",
                theorem="Poincare Conjecture / Geometrization",
                missing_data=["Decision-ready 3-manifold recognition certificate"],
                certificates={
                    "three_manifold_recognition_certificate": rec_cert.to_legacy_dict()
                },
                assumptions=["Closed connected 3-manifold hypotheses must hold"],
            )

    if rec_cert.provided:
        if rec_cert.decision_ready():
            return HomeomorphismResult(
                status="success",
                is_homeomorphic=True,
                reasoning=(
                    "SUCCESS: Homology/fundamental-group compatibility checks pass and a decision-ready "
                    "3-manifold recognition certificate completes classification."
                ),
                theorem="Geometrization / 3-manifold recognition",
                evidence=[
                    "Homology checks passed",
                    "Decision-ready 3D recognition certificate",
                ],
                assumptions=rec_cert.assumptions,
                certificates={
                    "three_manifold_recognition_certificate": rec_cert.to_legacy_dict()
                },
                exact=True,
            )
        return HomeomorphismResult(
            status="inconclusive",
            is_homeomorphic=None,
            reasoning=(
                "INCONCLUSIVE: A 3-manifold recognition certificate was provided, but it is not decision-ready "
                "(exact + validated are required)."
            ),
            theorem="Geometrization / 3-manifold recognition",
            missing_data=["Decision-ready 3-manifold recognition certificate"],
            certificates={
                "three_manifold_recognition_certificate": rec_cert.to_legacy_dict()
            },
            exact=False,
        )

    return HomeomorphismResult(
        status="inconclusive",
        is_homeomorphic=None,
        reasoning="INCONCLUSIVE: Manifolds are homology equivalent. In 3D, full homeomorphism recognition requires geometric/fundamental-group analysis beyond homology alone.",
        theorem="Geometrization / 3-manifold recognition",
        missing_data=["Certified geometric or group-theoretic recognition witness"],
    )


def analyze_homeomorphism_3d(
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
    recognition_certificate: ThreeManifoldRecognitionCertificate | dict | None = None,
) -> Tuple[bool | None, str]:
    return analyze_homeomorphism_3d_result(
        c1,
        c2,
        allow_approx=allow_approx,
        cohomology_signature_1=cohomology_signature_1,
        cohomology_signature_2=cohomology_signature_2,
        cohomology_ring_signature_1=cohomology_ring_signature_1,
        cohomology_ring_signature_2=cohomology_ring_signature_2,
        cup_product_signature_1=cup_product_signature_1,
        cup_product_signature_2=cup_product_signature_2,
        recognition_certificate=recognition_certificate,
    ).to_legacy_tuple()


def analyze_homeomorphism_high_dim_result(
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
) -> HomeomorphismResult:
    """
    Analyzes homeomorphism for high-dimensional manifolds (n >= 5) using the s-Cobordism Theorem
    and Smale's Generalized Poincare Conjecture (1961).
    """

    def _is_nontrivial_product_descriptor(desc: object | None) -> bool:
        if desc is None:
            return False
        raw = str(desc)
        if "x" not in raw.lower():
            return False
        factors = [f.strip() for f in raw.split("x") if f.strip()]
        nontrivial = [f for f in factors if f != "1"]
        return len(nontrivial) > 1

    theorem_label = "s-Cobordism / surgery classification"
    decision_stages: list[HighDimDecisionStage] = []
    hook_state = _normalize_homotopy_witness_hook(
        homotopy_equivalence_witness, homotopy_witness_hook
    )
    completion_state = _normalize_homotopy_completion_certificate(
        homotopy_completion_certificate,
        hook_state,
        homotopy_equivalence_witness,
    )
    assembly_state = _normalize_product_assembly_certificate(
        product_assembly_certificate
    )

    def _record_stage(
        sid: str,
        title: str,
        outcome: Literal["passed", "failed", "inconclusive", "skipped"],
        detail: str = "",
        *,
        exact: bool = True,
        data: dict[str, object] | None = None,
    ) -> None:
        decision_stages.append(
            HighDimDecisionStage(
                id=sid,
                title=title,
                outcome=outcome,
                detail=detail,
                exact=exact,
                data=dict(data or {}),
            )
        )

    def _with_phase5_metadata(result: HomeomorphismResult) -> HomeomorphismResult:
        certs = dict(result.certificates)
        certs["decision_dag"] = HighDimDecisionDag(
            dimension=dim,
            theorem=result.theorem or theorem_label,
            stages=list(decision_stages),
        ).to_legacy_dict()
        certs["homotopy_witness_hook"] = hook_state.to_legacy_dict()
        certs["homotopy_completion_certificate"] = completion_state.to_legacy_dict()
        certs["product_assembly_certificate"] = assembly_state.to_legacy_dict()
        if homotopy_equivalence_witness is not None:
            certs["homotopy_equivalence_witness"] = _freeze_value(
                homotopy_equivalence_witness
            )
        result.certificates = certs
        return result

    _record_stage(
        "hook_intake",
        "Homotopy Witness Intake",
        "passed" if hook_state.provided else "skipped",
        hook_state.summary,
        exact=hook_state.exact,
    )
    _record_stage(
        "homotopy_certificate_intake",
        "Homotopy Completion Certificate Intake",
        "passed" if completion_state.provided else "skipped",
        completion_state.summary,
        exact=completion_state.exact,
        data={
            "validated": completion_state.validated,
            "equivalence_type": completion_state.equivalence_type,
            "decision_ready": completion_state.decision_ready(),
        },
    )
    _record_stage(
        "product_assembly_intake",
        "Product Assembly Certificate Intake",
        "passed" if assembly_state.provided else "skipped",
        assembly_state.summary,
        exact=assembly_state.exact,
        data={
            "validated": assembly_state.validated,
            "decision_ready": assembly_state.decision_ready(),
        },
    )

    if dim < 5:
        _record_stage(
            "dimension_guard",
            "Dimension Guard",
            "failed",
            f"Received dimension {dim}; requires n>=5.",
        )
        raise DimensionError(
            f"Function called on {dim}D. The s-Cobordism theorem and Wall's high-dimensional surgery framework strictly apply to n >= 5, where the 'Whitney Trick' guarantees enough room to untangle handles."
        )

    # Check Homology Equivalence
    for n in range(dim + 1):
        try:
            r_1, t_1 = c1.homology(n)
            r_2, t_2 = c2.homology(n)
        except Exception as e:
            if allow_approx:
                warnings.warn(
                    f"Topological Hint: Homology extraction failed at dimension {n} ({e!r}). Exact classification disabled."
                )
            _record_stage(
                "homology_check",
                "Homology Equivalence",
                "inconclusive",
                f"Failed at H_{n}: {e!r}",
                exact=False,
            )
            return _with_phase5_metadata(
                HomeomorphismResult(
                    status="inconclusive",
                    is_homeomorphic=None,
                    reasoning=f"INCONCLUSIVE: Exact homology extraction failed at dimension {n} ({e!r}).",
                    theorem="s-Cobordism / surgery classification",
                    missing_data=[f"Exact H_{n}"],
                    exact=False,
                )
            )

        t_1n = _normalize_torsion(t_1)
        t_2n = _normalize_torsion(t_2)
        if r_1 != r_2 or t_1n != t_2n:
            _record_stage(
                "homology_check",
                "Homology Equivalence",
                "failed",
                f"Mismatch at H_{n}.",
            )
            return _with_phase5_metadata(
                HomeomorphismResult(
                    status="impediment",
                    is_homeomorphic=False,
                    reasoning=f"IMPEDIMENT: Homology mismatch in dimension {n} (Rank: {r_1} vs {r_2}, Torsion: {t_1n} vs {t_2n}).",
                    theorem="s-Cobordism / surgery classification",
                    evidence=[f"H_{n} mismatch"],
                )
            )

    _record_stage(
        "homology_check",
        "Homology Equivalence",
        "passed",
        "Exact homology groups match in all degrees.",
    )

    coho_check = _check_cohomology_equivalence(
        c1,
        c2,
        max_dim=dim,
        theorem="s-Cobordism / surgery classification",
        allow_approx=allow_approx,
        cohomology_signature_1=cohomology_signature_1,
        cohomology_signature_2=cohomology_signature_2,
    )
    if coho_check is not None:
        _record_stage(
            "cohomology_check",
            "Cohomology Equivalence",
            "failed" if coho_check.status == "impediment" else "inconclusive",
            coho_check.reasoning,
            exact=coho_check.exact,
        )
        return _with_phase5_metadata(coho_check)

    _record_stage(
        "cohomology_check",
        "Cohomology Equivalence",
        "passed",
        "Cohomology groups are compatible.",
    )

    cup_check = _check_cup_product_compatibility(
        cohomology_ring_signature_1,
        cohomology_ring_signature_2,
        cup_product_signature_1,
        cup_product_signature_2,
        theorem="s-Cobordism / surgery classification",
    )
    if cup_check is not None:
        _record_stage(
            "cup_product_check",
            "Cup-Product Compatibility",
            "failed" if cup_check.status == "impediment" else "inconclusive",
            cup_check.reasoning,
            exact=cup_check.exact,
        )
        return _with_phase5_metadata(cup_check)

    _record_stage(
        "cup_product_check",
        "Cup-Product Compatibility",
        "passed",
        "No cup-product incompatibility detected.",
    )

    descriptor = _infer_pi_group_descriptor(pi1, pi_group)
    if _is_nontrivial_product_descriptor(descriptor):
        theorem_label = "Wall L-theory over group rings"
    phase2_readiness = evaluate_phase2_readiness(
        pi1, str(descriptor) if descriptor is not None else None
    )
    if descriptor is None:
        _record_stage(
            "pi_descriptor",
            "pi_1 Descriptor",
            "inconclusive",
            "No supported descriptor inferred.",
        )
        return _with_phase5_metadata(
            HomeomorphismResult(
                status="inconclusive",
                is_homeomorphic=None,
                reasoning=f"INCONCLUSIVE: Homology matches in {dim}D, but pi_1/group-ring descriptor is missing. s-Cobordism requires Whitehead and Wall obstruction checks.",
                theorem=theorem_label,
                missing_data=[
                    "pi_1 or supported pi-group descriptor",
                    "Whitehead torsion",
                    "Wall obstruction",
                ],
                certificates={"phase2_readiness": phase2_readiness},
            )
        )

    _record_stage(
        "pi_descriptor",
        "pi_1 Descriptor",
        "passed",
        f"Descriptor resolved as {descriptor}.",
    )

    wh = whitehead_group
    if wh is None:
        if pi1 is not None:
            wh = compute_whitehead_group(pi1)
        elif descriptor == "1":
            wh = WhiteheadGroup(rank=0, description="Wh(1)=0")

    if wh is None:
        _record_stage(
            "whitehead_check",
            "Whitehead Torsion",
            "inconclusive",
            "Could not infer/provide Whitehead data.",
        )
        return _with_phase5_metadata(
            HomeomorphismResult(
                status="inconclusive",
                is_homeomorphic=None,
                reasoning="INCONCLUSIVE: Whitehead torsion was not provided and cannot be inferred from available data.",
                theorem=theorem_label,
                missing_data=["Whitehead torsion Wh(pi_1)"],
            )
        )

    if not wh.computable:
        _record_stage(
            "whitehead_check",
            "Whitehead Torsion",
            "inconclusive",
            "Whitehead computation unavailable.",
            exact=wh.exact,
        )
        return _with_phase5_metadata(
            HomeomorphismResult(
                status="inconclusive",
                is_homeomorphic=None,
                reasoning=f"INCONCLUSIVE: Whitehead torsion computation failed ({wh.description}).",
                theorem=theorem_label,
                missing_data=["Computable Wh(pi_1)"],
                assumptions=wh.assumptions,
                exact=wh.exact,
            )
        )

    if not wh.exact:
        _record_stage(
            "whitehead_check",
            "Whitehead Torsion",
            "inconclusive",
            "Whitehead certificate is heuristic.",
            exact=False,
        )
        return _with_phase5_metadata(
            HomeomorphismResult(
                status="inconclusive",
                is_homeomorphic=None,
                reasoning="INCONCLUSIVE: Whitehead torsion data is only heuristic; exact s-cobordism classification requires an exact Whitehead certificate.",
                theorem=theorem_label,
                missing_data=["Exact Whitehead torsion certificate"],
                assumptions=wh.assumptions,
                exact=False,
            )
        )

    if wh.rank > 0:
        _record_stage(
            "whitehead_check",
            "Whitehead Torsion",
            "failed",
            f"Non-zero Whitehead rank {wh.rank}.",
        )
        return _with_phase5_metadata(
            HomeomorphismResult(
                status="surgery_required",
                is_homeomorphic=False,
                reasoning=f"SURGERY_REQUIRED: Whitehead torsion obstruction detected (rank >= {wh.rank}).",
                theorem=theorem_label,
                evidence=[wh.description],
                assumptions=wh.assumptions,
                exact=wh.exact,
            )
        )

    _record_stage(
        "whitehead_check",
        "Whitehead Torsion",
        "passed",
        "Exact Whitehead obstruction vanishes.",
    )

    if normal_invariants_1 is not None and normal_invariants_1.dimension != dim:
        _record_stage(
            "normal_invariants",
            "Normal Invariants",
            "inconclusive",
            "First normal-invariant dimension mismatch.",
        )
        return _with_phase5_metadata(
            HomeomorphismResult(
                status="inconclusive",
                is_homeomorphic=None,
                reasoning=(
                    f"INCONCLUSIVE: normal_invariants_1 has dimension {normal_invariants_1.dimension}, "
                    f"but the manifold dimension is {dim}."
                ),
                theorem=theorem_label,
                missing_data=[
                    "Dimension-compatible normal invariants for the first manifold"
                ],
            )
        )

    if normal_invariants_2 is not None and normal_invariants_2.dimension != dim:
        _record_stage(
            "normal_invariants",
            "Normal Invariants",
            "inconclusive",
            "Second normal-invariant dimension mismatch.",
        )
        return _with_phase5_metadata(
            HomeomorphismResult(
                status="inconclusive",
                is_homeomorphic=None,
                reasoning=(
                    f"INCONCLUSIVE: normal_invariants_2 has dimension {normal_invariants_2.dimension}, "
                    f"but the manifold dimension is {dim}."
                ),
                theorem=theorem_label,
                missing_data=[
                    "Dimension-compatible normal invariants for the second manifold"
                ],
            )
        )

    structure_set = None
    computed_normal_invariants_1 = normal_invariants_1
    computed_normal_invariants_2 = normal_invariants_2
    sequence_l_n_obstruction = wall_obstruction
    sequence_l_n_plus_1_obstruction = None
    if descriptor == "1":
        try:
            sequence_l_n_plus_1_obstruction = WallGroupL(
                dimension=dim + 1, pi="1"
            ).compute_obstruction_result(wall_form)
        except Exception:
            sequence_l_n_plus_1_obstruction = None
    if descriptor == "1":
        try:
            structure_set = StructureSet(dimension=dim, fundamental_group="1")
            if computed_normal_invariants_1 is None:
                computed_normal_invariants_1 = (
                    structure_set.compute_normal_invariants_result(c1)
                )
            if computed_normal_invariants_2 is None:
                computed_normal_invariants_2 = (
                    structure_set.compute_normal_invariants_result(c2)
                )
            if surgery_sequence is None and computed_normal_invariants_1 is not None:
                surgery_sequence = structure_set.evaluate_exact_sequence_result(
                    normal_invariants=computed_normal_invariants_1,
                    l_n_obstruction=sequence_l_n_obstruction,
                    l_n_plus_1_obstruction=sequence_l_n_plus_1_obstruction,
                )
        except Exception as e:
            if allow_approx:
                warnings.warn(
                    f"Topological Hint: Structure-set witness computation failed ({e!r}). Continuing with obstruction-only classification."
                )

    _record_stage(
        "structure_set",
        "Structure Set Pipeline",
        "passed" if surgery_sequence is not None else "skipped",
        "Structure-set certificate computed."
        if surgery_sequence is not None
        else "Structure-set certificate not provided/computed.",
    )

    if surgery_sequence is not None:
        if surgery_sequence.dimension != dim:
            _record_stage(
                "surgery_sequence",
                "Surgery Exact Sequence",
                "inconclusive",
                "Dimension mismatch in provided sequence certificate.",
            )
            return _with_phase5_metadata(
                HomeomorphismResult(
                    status="inconclusive",
                    is_homeomorphic=None,
                    reasoning=(
                        f"INCONCLUSIVE: Surgery exact-sequence certificate has dimension {surgery_sequence.dimension}, "
                        f"but the manifold dimension is {dim}."
                    ),
                    theorem=theorem_label,
                    missing_data=[
                        "Dimension-compatible surgery exact-sequence certificate"
                    ],
                )
            )
        if not surgery_sequence.computable:
            _record_stage(
                "surgery_sequence",
                "Surgery Exact Sequence",
                "inconclusive",
                "Surgery sequence marked non-computable.",
                exact=surgery_sequence.exact,
            )
            return _with_phase5_metadata(
                HomeomorphismResult(
                    status="inconclusive",
                    is_homeomorphic=None,
                    reasoning=f"INCONCLUSIVE: Surgery exact-sequence certificate is not computable ({surgery_sequence.analysis[:1]}).",
                    theorem=theorem_label,
                    missing_data=["Computable surgery exact-sequence certificate"],
                    assumptions=["Supplied surgery certificate is not exact"],
                    exact=surgery_sequence.exact,
                )
            )
        if (
            surgery_sequence.normal_invariants is not None
            and computed_normal_invariants_1 is not None
            and surgery_sequence.normal_invariants != computed_normal_invariants_1
        ):
            _record_stage(
                "surgery_sequence",
                "Surgery Exact Sequence",
                "failed",
                "Normal invariants mismatch against supplied sequence.",
            )
            return _with_phase5_metadata(
                HomeomorphismResult(
                    status="impediment",
                    is_homeomorphic=False,
                    reasoning="IMPEDIMENT: Supplied surgery exact-sequence certificate is inconsistent with the computed normal invariants for the first manifold.",
                    theorem=theorem_label,
                    evidence=[
                        "Normal invariants mismatch against supplied surgery sequence"
                    ],
                )
            )

    wall = wall_obstruction
    if wall is None:
        try:
            wall = WallGroupL(dimension=dim, pi=descriptor).compute_obstruction_result(
                wall_form
            )
        except Exception as e:
            _record_stage(
                "wall_obstruction",
                "Wall Obstruction",
                "inconclusive",
                f"Wall evaluation failed: {e!r}",
            )
            return _with_phase5_metadata(
                HomeomorphismResult(
                    status="inconclusive",
                    is_homeomorphic=None,
                    reasoning=f"INCONCLUSIVE: Wall obstruction evaluation failed ({e!r}).",
                    theorem=theorem_label,
                    missing_data=["Computable Wall L-group obstruction"],
                )
            )

    if not wall.computable:
        _record_stage(
            "wall_obstruction",
            "Wall Obstruction",
            "inconclusive",
            "Wall obstruction not computable.",
            exact=wall.exact,
        )
        return _with_phase5_metadata(
            HomeomorphismResult(
                status="inconclusive",
                is_homeomorphic=None,
                reasoning=f"INCONCLUSIVE: Wall obstruction not computable ({wall.message}).",
                theorem=theorem_label,
                missing_data=[f"Computable L_{dim}({wall.pi}) obstruction"],
                assumptions=wall.assumptions,
                exact=wall.exact,
            )
        )

    if not wall.exact:
        missing = ["Exact Wall obstruction certificate"]
        if not getattr(wall, "assembly_certified", False):
            missing.append("Certified product-group assembly map witness")
        _record_stage(
            "wall_obstruction",
            "Wall Obstruction",
            "inconclusive",
            "Wall obstruction is heuristic.",
            exact=False,
        )
        return _with_phase5_metadata(
            HomeomorphismResult(
                status="inconclusive",
                is_homeomorphic=None,
                reasoning="INCONCLUSIVE: Wall obstruction data is only heuristic; exact surgery classification requires an exact Wall certificate.",
                theorem=theorem_label,
                missing_data=missing,
                assumptions=wall.assumptions,
                exact=False,
            )
        )

    if wall.obstructs is True:
        obstruction_desc = (
            f"value={wall.value}" if wall.value is not None else "direct-sum element"
        )
        _record_stage(
            "wall_obstruction",
            "Wall Obstruction",
            "failed",
            f"Certified non-zero obstruction ({obstruction_desc}).",
        )
        return _with_phase5_metadata(
            HomeomorphismResult(
                status="surgery_required",
                is_homeomorphic=False,
                reasoning=f"SURGERY_REQUIRED: Non-zero Wall obstruction detected in L_{dim}({wall.pi}) ({obstruction_desc}).",
                theorem=theorem_label,
                evidence=[
                    "Whitehead obstruction vanishes",
                    f"Wall obstruction state obstructs={wall.obstructs}, zero_certified={wall.zero_certified}",
                ],
                assumptions=wall.assumptions,
                exact=wall.exact,
            )
        )

    if not wall.zero_certified:
        _record_stage(
            "wall_obstruction",
            "Wall Obstruction",
            "inconclusive",
            "Wall element computed but vanishing is uncertified.",
        )
        return _with_phase5_metadata(
            HomeomorphismResult(
                status="inconclusive",
                is_homeomorphic=None,
                reasoning=(
                    "INCONCLUSIVE: Wall obstruction is computable, but vanishing is not certified "
                    "(e.g., direct-sum/mixed summands without scalar collapse)."
                ),
                theorem=theorem_label,
                missing_data=["Certified vanishing Wall obstruction element"],
                assumptions=wall.assumptions,
                exact=wall.exact,
            )
        )

    _record_stage(
        "wall_obstruction",
        "Wall Obstruction",
        "passed",
        "Exact Wall obstruction is certified zero.",
    )

    if _is_nontrivial_product_descriptor(descriptor) and not bool(
        getattr(wall, "assembly_certified", False)
    ):
        _record_stage(
            "product_assembly",
            "Product-Group Assembly Certification",
            "inconclusive",
            "Wall decomposition is not assembly-certified; decision-ready product assembly certificate required.",
            exact=False,
        )
        if not assembly_state.decision_ready():
            return _with_phase5_metadata(
                HomeomorphismResult(
                    status="inconclusive",
                    is_homeomorphic=None,
                    reasoning=(
                        "INCONCLUSIVE: Product-group Wall data is not assembly-certified and no decision-ready "
                        "product assembly certificate was supplied."
                    ),
                    theorem=theorem_label,
                    missing_data=["Decision-ready product-group assembly certificate"],
                    assumptions=wall.assumptions + assembly_state.assumptions,
                    exact=False,
                )
            )
        _record_stage(
            "product_assembly",
            "Product-Group Assembly Certification",
            "passed",
            f"Decision-ready product assembly certificate accepted from {assembly_state.source}.",
            exact=True,
        )

    base_certificates = {
        "whitehead_group": wh,
        "wall_obstruction": wall,
        "wall_assembly_state": {
            "decomposition_kind": getattr(wall, "decomposition_kind", "scalar"),
            "assembly_certified": bool(getattr(wall, "assembly_certified", False)),
        },
        "phase2_readiness": phase2_readiness,
        "normal_invariants_1": computed_normal_invariants_1,
        "normal_invariants_2": computed_normal_invariants_2,
        "surgery_sequence": surgery_sequence,
    }

    if completion_state.provided:
        if not completion_state.decision_ready():
            _record_stage(
                "homotopy_completion",
                "Homotopy-Equivalence Completion",
                "inconclusive",
                "Certificate provided, but it is not decision-ready (exact + validated required).",
                exact=False,
                data={
                    "source": completion_state.source,
                    "equivalence_type": completion_state.equivalence_type,
                    "validated": completion_state.validated,
                },
            )
            _record_stage(
                "final_classification",
                "Final Classification",
                "inconclusive",
                "Homotopy completion evidence is present but does not satisfy decision-ready criteria.",
                exact=False,
            )
            return _with_phase5_metadata(
                HomeomorphismResult(
                    status="inconclusive",
                    is_homeomorphic=None,
                    reasoning=(
                        "INCONCLUSIVE: A homotopy-completion certificate was supplied, but it is not decision-ready "
                        "(exact + validated are required for certified high-dimensional completion)."
                    ),
                    theorem=theorem_label,
                    missing_data=[
                        "Decision-ready homotopy-completion certificate (exact and validated)"
                    ],
                    evidence=[
                        "Homology match",
                        "Wh(pi_1)=0",
                        f"L_{dim}(pi_1) obstruction vanishes",
                    ],
                    assumptions=wall.assumptions
                    + wh.assumptions
                    + completion_state.assumptions,
                    certificates=base_certificates,
                    exact=False,
                )
            )

        _record_stage(
            "homotopy_completion",
            "Homotopy-Equivalence Completion",
            "passed",
            f"Decision-ready homotopy-completion certificate accepted from {completion_state.source}.",
            exact=True,
            data={
                "equivalence_type": completion_state.equivalence_type,
                "validated": completion_state.validated,
            },
        )
        _record_stage(
            "final_classification",
            "Final Classification",
            "passed",
            "Exact high-dimensional surgery pipeline completed with a decision-ready homotopy-completion certificate.",
            exact=True,
        )
        return _with_phase5_metadata(
            HomeomorphismResult(
                status="success",
                is_homeomorphic=True,
                reasoning=(
                    f"SUCCESS: Homology equivalence, Wh(pi_1)=0, vanishing Wall obstruction, and an exact "
                    f"validated homotopy-completion certificate certify homeomorphism in {dim}D under the modeled s-cobordism pipeline."
                ),
                theorem=theorem_label,
                evidence=[
                    "Homology match",
                    "Wh(pi_1)=0",
                    f"L_{dim}(pi_1) obstruction vanishes",
                    "Decision-ready homotopy-completion certificate",
                ],
                assumptions=wall.assumptions
                + wh.assumptions
                + completion_state.assumptions,
                certificates=base_certificates,
                exact=wall.exact
                and wh.exact
                and completion_state.exact
                and completion_state.validated,
            )
        )

    _record_stage(
        "homotopy_completion",
        "Homotopy-Equivalence Completion",
        "skipped",
        "No explicit homotopy-equivalence witness supplied.",
        exact=False,
    )

    s1 = _homology_sphere_like(c1, dim)
    s2 = _homology_sphere_like(c2, dim)
    if s1 is None or s2 is None:
        _record_stage(
            "homology_sphere_side_conditions",
            "Homology-Sphere Side Conditions",
            "inconclusive",
            "Could not verify side conditions exactly.",
        )
        return _with_phase5_metadata(
            HomeomorphismResult(
                status="inconclusive",
                is_homeomorphic=None,
                reasoning="INCONCLUSIVE: Could not verify homology-sphere side conditions exactly.",
                theorem="s-Cobordism / surgery classification",
                missing_data=["Exact homology-sphere verification"],
            )
        )

    if s1 and s2 and descriptor == "1":
        _record_stage(
            "final_classification",
            "Final Classification",
            "passed",
            "All currently modeled high-dimensional obstructions vanish with side conditions satisfied.",
        )
        return _with_phase5_metadata(
            HomeomorphismResult(
                status="success",
                is_homeomorphic=True,
                reasoning=f"SUCCESS: Homology-sphere conditions, Wh(pi_1)=0, and vanishing Wall obstruction support homeomorphism in {dim}D under s-cobordism/generalized Poincare hypotheses.",
                theorem="s-Cobordism / generalized Poincare",
                evidence=[
                    "Homology sphere checks passed",
                    "Wh(pi_1)=0",
                    f"L_{dim}(pi_1) obstruction vanishes",
                ],
                assumptions=[
                    "Closed connected manifold hypotheses",
                    "Input normal-map/surgery model is valid",
                ],
                certificates=base_certificates,
                exact=wall.exact and wh.exact,
            )
        )

    _record_stage(
        "final_classification",
        "Final Classification",
        "inconclusive",
        "Missing explicit homotopy-equivalence completion witness in API pipeline.",
    )
    return _with_phase5_metadata(
        HomeomorphismResult(
            status="inconclusive",
            is_homeomorphic=None,
            reasoning=f"INCONCLUSIVE: Homology equivalence holds in {dim}D and computed surgery obstructions vanish, but this API has no explicit homotopy-equivalence witness to complete classification.",
            theorem=theorem_label,
            evidence=[
                "Homology match",
                "Wh(pi_1)=0",
                f"L_{dim}(pi_1) obstruction vanishes",
            ],
            assumptions=wall.assumptions + wh.assumptions,
            certificates=base_certificates,
            exact=wall.exact and wh.exact,
        )
    )


def analyze_homeomorphism_high_dim(
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
) -> Tuple[bool | None, str]:
    return analyze_homeomorphism_high_dim_result(
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
    ).to_legacy_tuple()


def analyze_homeomorphism_4d_result(
    m1: IntersectionForm,
    m2: IntersectionForm,
    ks1: int | None = None,
    ks2: int | None = None,
    *,
    simply_connected: bool | None = None,
    definite_lattice_isometry_certificate: DefiniteLatticeIsometryCertificate
    | dict
    | None = None,
) -> HomeomorphismResult:
    """
    Analyzes the potential for homeomorphism between two simply-connected 4-manifolds.

    Based on Freedman's Classification Theorem:
    Two such manifolds are homeomorphic if and only if:
    1. Their intersection forms are isomorphic over Z.
    2. Their Kirby-Siebenmann invariants match.

    Returns
    -------
    is_homeomorphic : bool
    reasoning : str
    """
    if m1.dimension != 4 or m2.dimension != 4:
        raise DimensionError(
            f"Freedman's Classification Theorem strictly governs simply-connected 4-manifolds via intersection forms. "
            f"Received manifolds of dimensions {m1.dimension} and {m2.dimension}. Hint: Use 2D, 3D, or high_dim analyzers instead."
        )

    if simply_connected is None:
        return HomeomorphismResult(
            status="inconclusive",
            is_homeomorphic=None,
            reasoning="INCONCLUSIVE: Simply-connectedness was not supplied; Freedman classification cannot be applied safely.",
            theorem="Freedman classification",
            missing_data=["Verification that both manifolds are simply-connected"],
        )

    if not simply_connected:
        return HomeomorphismResult(
            status="inconclusive",
            is_homeomorphic=None,
            reasoning="INCONCLUSIVE: Input marked as non-simply-connected; this analyzer currently covers the simply-connected Freedman branch.",
            theorem="Freedman classification",
            missing_data=["Non-simply-connected 4D surgery pipeline"],
        )

    n = int(np.asarray(m1.matrix).shape[0])
    if m1.rank() != n or m2.rank() != n:
        return HomeomorphismResult(
            status="inconclusive",
            is_homeomorphic=None,
            reasoning="INCONCLUSIVE: Degenerate intersection form detected; unimodular non-degenerate forms are required here.",
            theorem="Freedman classification",
            missing_data=["Non-degenerate unimodular intersection forms"],
        )

    try:
        det1 = abs(int(m1.determinant()))
        det2 = abs(int(m2.determinant()))
    except Exception as e:
        return HomeomorphismResult(
            status="inconclusive",
            is_homeomorphic=None,
            reasoning=f"INCONCLUSIVE: Exact determinant/unimodularity check failed ({e!r}).",
            theorem="Freedman classification",
            missing_data=["Exact determinant for both forms"],
        )

    if det1 != 1 or det2 != 1:
        return HomeomorphismResult(
            status="inconclusive",
            is_homeomorphic=None,
            reasoning=f"INCONCLUSIVE: Intersection forms are not unimodular (|det|={det1} vs {det2}).",
            theorem="Freedman classification",
            missing_data=["Unimodular forms required for this branch"],
        )

    # Impediment 1: Rank
    if m1.rank() != m2.rank():
        return HomeomorphismResult(
            status="impediment",
            is_homeomorphic=False,
            reasoning=f"IMPEDIMENT: Ranks differ ({m1.rank()} vs {m2.rank()}). Homeomorphism is impossible.",
            theorem="Freedman classification",
        )

    # Impediment 2: Signature
    if m1.signature() != m2.signature():
        return HomeomorphismResult(
            status="impediment",
            is_homeomorphic=False,
            reasoning=f"IMPEDIMENT: Signatures differ ({m1.signature()} vs {m2.signature()}). The L_4(1) surgery obstruction is non-zero.",
            theorem="Freedman classification",
        )

    # Impediment 3: Parity (Type)
    if m1.type() != m2.type():
        return HomeomorphismResult(
            status="impediment",
            is_homeomorphic=False,
            reasoning=f"IMPEDIMENT: Parity mismatch (Type {m1.type()} vs Type {m2.type()}).",
            theorem="Freedman classification",
        )

    if ks1 is None or ks2 is None:
        return HomeomorphismResult(
            status="inconclusive",
            is_homeomorphic=None,
            reasoning="INCONCLUSIVE: Kirby-Siebenmann invariants were not supplied.",
            theorem="Freedman classification",
            missing_data=["ks1", "ks2"],
        )

    # Impediment 4: Kirby-Siebenmann Invariant
    if ks1 != ks2:
        return HomeomorphismResult(
            status="impediment",
            is_homeomorphic=False,
            reasoning=f"IMPEDIMENT: Kirby-Siebenmann invariants differ ({ks1} vs {ks2}). These manifolds are homotopically equivalent but topologically distinct.",
            theorem="Freedman classification",
        )

    # Case: Indefinite forms (classified by rank, signature, parity)
    if m1.is_indefinite():
        return HomeomorphismResult(
            status="success",
            is_homeomorphic=True,
            reasoning=(
                "SUCCESS: Homeomorphism established via Freedman's theorem for indefinite unimodular forms "
                f"(rank={m1.rank()}, signature={m1.signature()}, type={m1.type()}, KS={ks1})."
            ),
            theorem="Freedman classification",
            evidence=[
                "Indefinite unimodular forms classified by rank/signature/type",
                "Matching KS invariant",
            ],
            certificates={
                "isometry_search_mode": "indefinite_classification",
                "isometry_search_certified": True,
                "solver_bounds": None,
            },
        )

    # Case: Definite forms (require lattice isomorphism)
    Q1 = np.asarray(m1.matrix, dtype=np.int64)
    Q2 = np.asarray(m2.matrix, dtype=np.int64)
    cert = _normalize_definite_lattice_isometry_certificate(
        definite_lattice_isometry_certificate
    )
    if np.array_equal(Q1, Q2):
        return HomeomorphismResult(
            status="success",
            is_homeomorphic=True,
            reasoning="SUCCESS: Definite intersection forms match exactly as integer lattices.",
            theorem="Freedman classification",
            evidence=["Exact matrix equality"],
            certificates={
                "isometry_search_mode": "matrix_equality",
                "isometry_search_certified": True,
                "solver_bounds": [2, 3],
            },
        )

    U = _search_integer_isometry(Q1, Q2, max_entry=2)
    if U is None and Q1.shape[0] <= 3:
        U = _search_integer_isometry(Q1, Q2, max_entry=3)
    if U is not None:
        return HomeomorphismResult(
            status="success",
            is_homeomorphic=True,
            reasoning="SUCCESS: Definite lattice isomorphism certificate found (U^T Q1 U = Q2).",
            theorem="Freedman classification",
            evidence=["Explicit unimodular isometry witness"],
            certificates={
                "isometry_search_mode": "bounded_search",
                "isometry_search_certified": True,
                "solver_bounds": [2, 3],
                "isometry_matrix": np.asarray(U, dtype=np.int64).tolist(),
            },
        )

    if cert.decision_ready():
        try:
            Uc = np.asarray(cert.isometry_matrix, dtype=np.int64)
            valid = (
                Uc.ndim == 2
                and Uc.shape == Q1.shape
                and abs(_det_int_small(Uc)) == 1
                and np.array_equal(Uc.T @ Q1 @ Uc, Q2)
            )
        except Exception:
            valid = False
            Uc = None
        if valid and Uc is not None:
            return HomeomorphismResult(
                status="success",
                is_homeomorphic=True,
                reasoning="SUCCESS: Definite lattice isomorphism certified via decision-ready external isometry witness.",
                theorem="Freedman classification",
                evidence=[
                    "Decision-ready definite-lattice certificate",
                    "Explicit unimodular isometry witness",
                ],
                assumptions=cert.assumptions,
                certificates={
                    "isometry_search_mode": "external_certificate",
                    "isometry_search_certified": True,
                    "solver_bounds": [2, 3],
                    "isometry_matrix": np.asarray(Uc, dtype=np.int64).tolist(),
                    "definite_lattice_isometry_certificate": cert.to_legacy_dict(),
                },
            )
        return HomeomorphismResult(
            status="inconclusive",
            is_homeomorphic=None,
            reasoning="INCONCLUSIVE: Provided definite-lattice certificate is decision-ready but does not verify U^T Q1 U = Q2.",
            theorem="Freedman classification",
            missing_data=["Valid explicit unimodular isometry matrix"],
            certificates={
                "definite_lattice_isometry_certificate": cert.to_legacy_dict()
            },
        )

    return HomeomorphismResult(
        status="inconclusive",
        is_homeomorphic=None,
        reasoning="INCONCLUSIVE: No bounded-search unimodular lattice-isometry certificate found for definite forms.",
        theorem="Freedman classification",
        missing_data=["Decision-ready definite-lattice isometry certificate"],
        certificates={
            "isometry_search_mode": "bounded_search",
            "isometry_search_certified": False,
            "solver_bounds": [2, 3],
            "definite_lattice_isometry_certificate": cert.to_legacy_dict(),
        },
    )


def analyze_homeomorphism_4d(
    m1: IntersectionForm,
    m2: IntersectionForm,
    ks1: int | None = None,
    ks2: int | None = None,
    *,
    simply_connected: bool | None = None,
    definite_lattice_isometry_certificate: DefiniteLatticeIsometryCertificate
    | dict
    | None = None,
) -> Tuple[bool | None, str]:
    return analyze_homeomorphism_4d_result(
        m1,
        m2,
        ks1=ks1,
        ks2=ks2,
        simply_connected=simply_connected,
        definite_lattice_isometry_certificate=definite_lattice_isometry_certificate,
    ).to_legacy_tuple()


def surgery_to_remove_impediments(
    m: IntersectionForm, target_sig: int
) -> Tuple[bool, str]:
    """Analyzes if surgery can be used to remove the 'impediment' to a target signature.

    Args:
        m: The current intersection form.
        target_sig: The target signature.

    Returns:
        A tuple of (can_fix, surgery_plan).
    """
    sig_diff = m.signature() - target_sig
    if sig_diff == 0:
        return (
            True,
            "Signatures already match. No signature-adjustment surgery required; parity, KS, Wh(pi_1), and Wall obstructions may still need checks.",
        )
    # Blow-up with CP^2 or -CP^2 changes signature by +/-1 and rank by 1.
    n_blowups = abs(sig_diff)
    blowup_type = "CP^2" if sig_diff < 0 else "(-CP^2)"
    return True, (
        f"PLAN: Connected sum with {n_blowups} copies of {blowup_type} "
        f"changes signature by {-sig_diff}. "
        "This only addresses signature-level obstructions; complete homeomorphism analysis may still require "
        "Kirby-Siebenmann agreement, Whitehead-torsion vanishing, and Wall L-group obstruction checks."
    )
