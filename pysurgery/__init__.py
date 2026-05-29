"""pysurgery: High-performance library for exact computational algebraic topology and surgery theory.

Overview:
    pysurgery provides a comprehensive suite of tools for computing discrete 
    topological invariants and performing manifold classification using the 
    framework of surgery theory. It leverages exact integer arithmetic for 
    homology and intersection forms, and integrates with Julia and JAX for 
    high-performance accelerators.

Key Concepts:
    - **Surgery Theory**: The classification of manifolds via the surgery exact sequence.
    - **Poincaré Duality**: Exact computation of dualities and intersection forms.
    - **Witnessing Model**: Certified homeomorphism decisions with explicit evidence.
    - **Backend Agnosticism**: Unified API for Python, Julia, and JAX backends.

Common Workflows:
    1. **Invariant Extraction** → homology(), fundamental_group(), intersection_form().
    2. **Manifold Comparison** → build_homeomorphism_witness(), analyze_homeomorphism().
    3. **Surgery Obstructions** → WallGroupL.compute_obstruction_result().

Coefficient Ring:
    Supports 'Z' (integers), 'Q' (rationals), and 'Z/pZ' (mod p).

Attributes:
    __version__ (str): Current version of the library.
    __all__ (list): Exported public symbols.
"""

import os

from .bridge import JuliaBridge

# CRITICAL FIX for Segmentation Faults:
# Set this environment variable before any submodule (like julia_bridge) imports `juliacall`.
# This ensures that when juliacall initializes the C-level libjulia runtime, 
# the required signal handlers are correctly bound for multi-threaded execution.
if "PYTHON_JULIACALL_HANDLE_SIGNALS" not in os.environ:
    os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

from .topology.complexes import (
    ChainComplex,
    CWComplex,
    SimplicialComplex,
    DynamicComplex,
)
from .topology.fundamental_group import (
    FundamentalGroup,
    GroupPresentation,
    extract_pi_1,
    extract_pi_1_with_traces,
    simplify_presentation,
    infer_standard_group_descriptor,
)
from .topology.pi1_group_ring_scaffold import (
    Pi1Evidence,
    GroupRingContext,
    Phase2Readiness,
    evaluate_phase2_readiness,
)
from .topology.persistent_homology import (
    Barcode,
    BarcodeResult,
    compute_barcodes_exact,
    compute_zigzag_persistence,
)
from .topology.temporal_topology import (
    TemporalBarcode,
    BifurcationEvent,
    TemporalAnalysisResult,
    analyze_temporal_evolution,
)
from .core.exceptions import (
    LinkingComputationError,
    BettiTrackingError,
    SurgeryError,
    DimensionError,
    LadderProgressError,
)
from .core.generator_models import (
    Pi1GeneratorTrace,
    Pi1PresentationWithTraces,
    HomologyGenerator,
    HomologyBasisResult,
)
from .core.foundations import (
    CONTRACT_VERSION,
    AnalyzerContract,
    CoverageMatrixEntry,
    COVERAGE_MATRIX,
    coverage_status_counts,
)
from .core.theorem_tags import infer_theorem_tag

from .geometry.embedding import (
    EmbeddingResult,
    ImmersionResult,
    PLMap,
    ProjectionResult,
    SelfIntersectionReport,
    SimplexIntersectionWitness,
    analyze_embedding,
    check_immersion,
    detect_self_intersections,
    jitter_coordinates,
    project_coordinates,
)
from .geometry.intrinsic_dimension import (
    IntrinsicDimensionMethodResult,
    IntrinsicDimensionResult,
    estimate_intrinsic_dimension,
    levina_bickel_mle,
    local_pca_tangent_space_dimension,
    twonn,
)
from .geometry.geometrization_3d import (
    GeometrizationResult,
    NormalSurfaceCandidate,
    PieceDecomposition,
    Triangulated3Manifold,
    analyze_geometrization,
    crush_normal_surface,
    jsj_decomposition,
    normal_surface_candidates,
    normal_surface_matching_matrix,
    prime_decomposition,
)
from .geometry.uniformization import (
    SurfaceMesh,
    SurfaceUniformizationResult,
    circle_packing_uniformization,
    cotangent_laplacian,
    discrete_ricci_flow,
    surface_target_curvature,
    uniformize_surface,
    vertex_gaussian_curvature,
)
from .geometry.characteristic_classes import (
    extract_stiefel_whitney_w2,
    check_spin_structure,
    extract_pontryagin_p1,
    verify_hirzebruch_signature,
)
from .geometry.gauss_bonnet import (
    verify_gauss_bonnet_2d,
    verify_chern_gauss_bonnet_4d,
    chern_gauss_bonnet_integral_expected,
)
from .geometry.immersion_obstructions import (
    NonImmersibilityWitness,
    ImmersibilityInconclusive,
    StructuralObstruction,
    immersion_obstruction_analysis,
)
from .algebra.intersection_forms import IntersectionForm
from .algebra.quadratic_forms import QuadraticForm, arf_invariant_gf2
from .algebra.group_rings import GroupRingElement
from .algebra.exact_algebra import (
    coerce_int_matrix,
    normalize_word_token,
    validate_group_descriptor,
)
from .algebra.k_theory import WhiteheadGroup, compute_whitehead_group
from .homology.homology_generators import (
    hk_generators_z,
    generator_cycles_from_simplices,
    greedy_h1_basis,
    compute_optimal_h1_basis_from_simplices,
    compute_optimal_h1_basis_from_complex,
    compute_homology_basis_from_simplices,
    compute_homology_basis_from_complex,
    annot_edge,
)
from .homology.algebraic_poincare import AlgebraicPoincareComplex
from .homology.controlled_cohomology import (
    FiniteGroupOrderResult,
    UniversalCoverResult,
    TwistedChainResult,
    ControlledCohomologyResult,
    TwistedIntersectionFormResult,
    TwistedObstructionResult,
    FiniteGroupRing,
    TwistedRepresentation,
    UniversalCover,
    TwistedChainComplex,
    compute_controlled_cohomology,
    compute_twisted_intersection_form,
    compute_twisted_obstruction,
)
from .manifolds.kirby_calculus import KirbyDiagram
from .manifolds.handle_decompositions import (
    Handle,
    HandleDecomposition,
    cw_complex_to_handle_decomposition,
)
from .manifolds.rational_surgery import (
    RationalObstruction,
    PLocalObstruction,
    PrimeLocalReport,
    compute_l_group_rational,
    prime_local_obstruction_report,
)
from .manifolds.surgery import (
    perform_handle_surgery,
    perform_algebraic_surgery,
    perform_rational_surgery,
    perform_p_local_surgery,
)
from .homotopy.rational_homotopy import (
    RationalDGA,
    RationalCohomologyAlgebra,
    RationalMinimalModelResult,
    RationalHomotopyGroup,
    FormalityResult,
    MasseyProductsResult,
    sullivan_minimal_model,
    is_formal_space,
    extract_massey_products,
    rational_homotopy_group,
)
from .homotopy.higher_homotopy_groups import (
    HomotopyGroup,
    HomotopyGroupApproximation,
    compute_rational_and_adams,
    synthesize_homotopy_group_with_e_infinity,
)
from .adams.spectral_sequence import (
    SteenrodAlgebra,
    AdamsE2Page,
    adams_e2_page,
)
from .adams.e_infinity_resolver import (
    ConvergedAdamsPage,
    UserVerifiedDifferential,
)
from .adams.interactive_resolver import InteractiveAdamsResolver
from .adams.lean_resolver import LeanFormalAdamsResolver, LeanProofAttempt
from .spectral.spectral_sequences import SerreSpectralSequence

# Top-level modules that remain in the root
from .surgery import AlgebraicSurgeryComplex
from .structure_set import (
    StructureSet,
    NormalInvariantsResult,
    SurgeryExactSequenceResult,
    LObstructionState,
)
from .wall_groups import (
    WallGroupL,
    ObstructionResult,
    LDirectSummand,
    LDirectSumElement,
)
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
    surgery_to_remove_impediments,
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
from .auto_surgery import (
    GeneratorCycle,
    CutSite,
    ComponentInfo,
    LinkedPair,
    Pi1Killer,
    HomologyKiller,
    NestedPair,
    UnlinkPass,
    UnlinkReport,
    NestReport,
    Pi1KillStep,
    Pi1KillReport,
    HKillStep,
    HKillReport,
    ObstructionReport,
    AutoSurgeryReport,
    AutoSurgeryError,
    NonManifoldComponentError,
    NoCutSiteError,
    NoAttachingSphereError,
    MiddleDimensionObstructed,
    HomologyManifoldNotPLWarning,
    compute_pi1_generators_as_cycles,
    detect_components_with_status,
    detect_linked_pairs,
    detect_nested_pairs,
    auto_unlink_pair,
    auto_separate_nested,
    auto_kill_pi1,
    auto_kill_homology_dim,
    auto_check_middle_obstruction,
    AutoSurgeonConfig,
)

from . import integrations

__version__ = "2.0.6"
                                
def __getattr__(name):
    if name == "JuliaBridge":
        from .bridge.julia_bridge import JuliaBridge as _JB
        return _JB
    raise AttributeError(f"module 'pysurgery' has no attribute {name!r}")

__all__ = [
    "ChainComplex",
    "CWComplex",
    "SimplicialComplex",
    "DynamicComplex",
    "EmbeddingResult",
    "ImmersionResult",
    "PLMap",
    "ProjectionResult",
    "SelfIntersectionReport",
    "SimplexIntersectionWitness",
    "analyze_embedding",
    "check_immersion",
    "detect_self_intersections",
    "jitter_coordinates",
    "project_coordinates",
    "IntrinsicDimensionMethodResult",
    "IntrinsicDimensionResult",
    "estimate_intrinsic_dimension",
    "levina_bickel_mle",
    "local_pca_tangent_space_dimension",
    "twonn",
    "GeometrizationResult",
    "NormalSurfaceCandidate",
    "PieceDecomposition",
    "Triangulated3Manifold",
    "analyze_geometrization",
    "crush_normal_surface",
    "jsj_decomposition",
    "normal_surface_candidates",
    "normal_surface_matching_matrix",
    "prime_decomposition",
    "SurfaceMesh",
    "SurfaceUniformizationResult",
    "circle_packing_uniformization",
    "cotangent_laplacian",
    "discrete_ricci_flow",
    "surface_target_curvature",
    "uniformize_surface",
    "vertex_gaussian_curvature",
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
    "compute_optimal_h1_basis_from_complex",
    "compute_homology_basis_from_simplices",
    "compute_homology_basis_from_complex",
    "hk_generators_z",
    "extract_stiefel_whitney_w2",
    "check_spin_structure",
    "extract_pontryagin_p1",
    "verify_hirzebruch_signature",
    "verify_gauss_bonnet_2d",
    "verify_chern_gauss_bonnet_4d",
    "chern_gauss_bonnet_integral_expected",
    "KirbyDiagram",
    "WhiteheadGroup",
    "compute_whitehead_group",
    "FiniteGroupOrderResult",
    "UniversalCoverResult",
    "TwistedChainResult",
    "ControlledCohomologyResult",
    "TwistedIntersectionFormResult",
    "TwistedObstructionResult",
    "FiniteGroupRing",
    "TwistedRepresentation",
    "UniversalCover",
    "TwistedChainComplex",
    "compute_controlled_cohomology",
    "compute_twisted_intersection_form",
    "compute_twisted_obstruction",
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
    "perform_handle_surgery",
    "perform_algebraic_surgery",
    "perform_rational_surgery",
    "perform_p_local_surgery",
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
    "RationalDGA",
    "RationalCohomologyAlgebra",
    "RationalMinimalModelResult",
    "RationalHomotopyGroup",
    "FormalityResult",
    "MasseyProductsResult",
    "sullivan_minimal_model",
    "is_formal_space",
    "extract_massey_products",
    "rational_homotopy_group",
    "SteenrodAlgebra",
    "AdamsE2Page",
    "adams_e2_page",
    "HomotopyGroup",
    "HomotopyGroupApproximation",
    "compute_rational_and_adams",
    "synthesize_homotopy_group_with_e_infinity",
    "Handle",
    "HandleDecomposition",
    "cw_complex_to_handle_decomposition",
    "Barcode",
    "BarcodeResult",
    "compute_barcodes_exact",
    "compute_zigzag_persistence",
    "TemporalBarcode",
    "BifurcationEvent",
    "TemporalAnalysisResult",
    "analyze_temporal_evolution",
    "NonImmersibilityWitness",
    "ImmersibilityInconclusive",
    "StructuralObstruction",
    "immersion_obstruction_analysis",
    "RationalObstruction",
    "PLocalObstruction",
    "PrimeLocalReport",
    "compute_l_group_rational",
    "prime_local_obstruction_report",
    "ConvergedAdamsPage",
    "UserVerifiedDifferential",
    "InteractiveAdamsResolver",
    "LeanFormalAdamsResolver",
    "LeanProofAttempt",
    "integrations",
    "GeneratorCycle",
    "CutSite",
    "ComponentInfo",
    "LinkedPair",
    "Pi1Killer",
    "HomologyKiller",
    "NestedPair",
    "UnlinkPass",
    "UnlinkReport",
    "NestReport",
    "Pi1KillStep",
    "Pi1KillReport",
    "HKillStep",
    "HKillReport",
    "ObstructionReport",
    "AutoSurgeryReport",
    "AutoSurgeryError",
    "NonManifoldComponentError",
    "NoCutSiteError",
    "NoAttachingSphereError",
    "MiddleDimensionObstructed",
    "HomologyManifoldNotPLWarning",
    "compute_pi1_generators_as_cycles",
    "detect_components_with_status",
    "detect_linked_pairs",
    "detect_nested_pairs",
    "auto_unlink_pair",
    "auto_separate_nested",
    "auto_kill_pi1",
    "auto_kill_homology_dim",
    "auto_check_middle_obstruction",
    "AutoSurgeonConfig",
    "LinkingComputationError",
    "BettiTrackingError",
    "SerreSpectralSequence",
    "SurgeryError",
    "DimensionError",
    "LadderProgressError",
]
