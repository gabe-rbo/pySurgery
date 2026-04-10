from .core.complexes import ChainComplex, CWComplex
from .core.intersection_forms import IntersectionForm
from .core.quadratic_forms import QuadraticForm, arf_invariant_gf2
from .core.group_rings import GroupRingElement
from .core.fundamental_group import FundamentalGroup, GroupPresentation, extract_pi_1, extract_pi_1_with_traces, simplify_presentation, infer_standard_group_descriptor
from .core.generator_models import Pi1GeneratorTrace, Pi1PresentationWithTraces, HomologyGenerator, HomologyBasisResult
from .core.homology_generators import (
    annot_edge,
    generator_cycles_from_simplices,
    greedy_h1_basis,
    compute_optimal_h1_basis_from_simplices,
    compute_optimal_h1_basis_from_simplex_tree,
)
from .core.characteristic_classes import extract_stiefel_whitney_w2, check_spin_structure, extract_pontryagin_p1, verify_hirzebruch_signature
from .core.kirby_calculus import KirbyDiagram
from .core.k_theory import WhiteheadGroup, compute_whitehead_group
from .core.foundations import CONTRACT_VERSION, AnalyzerContract, CoverageMatrixEntry, COVERAGE_MATRIX, coverage_status_counts
from .core.theorem_tags import infer_theorem_tag
from .core.exact_algebra import coerce_int_matrix, normalize_word_token, validate_group_descriptor
from .core.pi1_group_ring_scaffold import Pi1Evidence, GroupRingContext, Phase2Readiness, evaluate_phase2_readiness
from .algebraic_poincare import AlgebraicPoincareComplex
from .algebraic_surgery import AlgebraicSurgeryComplex
from .structure_set import StructureSet, NormalInvariantsResult, SurgeryExactSequenceResult, LObstructionState
from .wall_groups import WallGroupL, ObstructionResult, LDirectSummand, LDirectSumElement
from .bridge.julia_bridge import JuliaBridge
from .homeomorphism import (
    DefiniteLatticeIsometryCertificate,
    HighDimDecisionDag,
    HighDimDecisionStage,
    HomotopyCompletionCertificate,
    HomotopyEquivalenceWitnessHook,
    ProductAssemblyCertificate,
    ThreeManifoldRecognitionCertificate,
    HomeomorphismResult,
    analyze_homeomorphism_2d,
    analyze_homeomorphism_2d_result,
    analyze_homeomorphism_3d,
    analyze_homeomorphism_3d_result,
    analyze_homeomorphism_4d,
    analyze_homeomorphism_4d_result,
    analyze_homeomorphism_high_dim,
    analyze_homeomorphism_high_dim_result,
    surgery_to_remove_impediments
)
from .homeomorphism_witness import (
    HomeomorphismWitness,
    HomeomorphismWitnessResult,
    build_homeomorphism_witness,
    build_3d_homeomorphism_witness,
    build_4d_homeomorphism_witness,
    build_high_dim_homeomorphism_witness,
    build_surface_homeomorphism_witness,
)
from . import integrations

__version__ = "1.0.0"
__all__ = [
    "ChainComplex",
    "CWComplex",
    "IntersectionForm",
    "QuadraticForm",
    "arf_invariant_gf2",
    "GroupRingElement",
    "FundamentalGroup",
    "GroupPresentation",
    "extract_pi_1",
    "extract_pi_1_with_traces",
    "simplify_presentation",
    "infer_standard_group_descriptor",
    "Pi1GeneratorTrace",
    "Pi1PresentationWithTraces",
    "HomologyGenerator",
    "HomologyBasisResult",
    "annot_edge",
    "generator_cycles_from_simplices",
    "greedy_h1_basis",
    "compute_optimal_h1_basis_from_simplices",
    "compute_optimal_h1_basis_from_simplex_tree",
    "extract_stiefel_whitney_w2",
    "check_spin_structure",
    "extract_pontryagin_p1",
    "verify_hirzebruch_signature",
    "KirbyDiagram",
    "WhiteheadGroup",
    "compute_whitehead_group",
    "CONTRACT_VERSION",
    "AnalyzerContract",
    "CoverageMatrixEntry",
    "COVERAGE_MATRIX",
    "coverage_status_counts",
    "infer_theorem_tag",
    "coerce_int_matrix",
    "normalize_word_token",
    "validate_group_descriptor",
    "Pi1Evidence",
    "GroupRingContext",
    "Phase2Readiness",
    "evaluate_phase2_readiness",
    "AlgebraicPoincareComplex",
    "AlgebraicSurgeryComplex",
    "StructureSet",
    "NormalInvariantsResult",
    "SurgeryExactSequenceResult",
    "LObstructionState",
    "WallGroupL",
    "ObstructionResult",
    "LDirectSummand",
    "LDirectSumElement",
    "JuliaBridge",
    "HomeomorphismResult",
    "DefiniteLatticeIsometryCertificate",
    "HighDimDecisionStage",
    "HighDimDecisionDag",
    "HomotopyCompletionCertificate",
    "HomotopyEquivalenceWitnessHook",
    "ThreeManifoldRecognitionCertificate",
    "ProductAssemblyCertificate",
    "analyze_homeomorphism_2d",
    "analyze_homeomorphism_2d_result",
    "analyze_homeomorphism_3d",
    "analyze_homeomorphism_3d_result",
    "analyze_homeomorphism_4d",
    "analyze_homeomorphism_4d_result",
    "analyze_homeomorphism_high_dim",
    "analyze_homeomorphism_high_dim_result",
    "surgery_to_remove_impediments",
    "HomeomorphismWitness",
    "HomeomorphismWitnessResult",
    "build_homeomorphism_witness",
    "build_surface_homeomorphism_witness",
    "build_3d_homeomorphism_witness",
    "build_4d_homeomorphism_witness",
    "build_high_dim_homeomorphism_witness",
    "integrations"
]
