from .core.complexes import ChainComplex, CWComplex
from .core.intersection_forms import IntersectionForm
from .core.quadratic_forms import QuadraticForm, arf_invariant_gf2
from .core.group_rings import GroupRingElement
from .core.fundamental_group import FundamentalGroup, GroupPresentation, extract_pi_1, simplify_presentation
from .core.characteristic_classes import extract_stiefel_whitney_w2, check_spin_structure, extract_pontryagin_p1, verify_hirzebruch_signature
from .core.kirby_calculus import KirbyDiagram
from .core.k_theory import WhiteheadGroup, compute_whitehead_group
from .algebraic_poincare import AlgebraicPoincareComplex
from .algebraic_surgery import AlgebraicSurgeryComplex
from .structure_set import StructureSet, NormalInvariantsResult, SurgeryExactSequenceResult
from .wall_groups import WallGroupL, ObstructionResult
from .bridge.julia_bridge import JuliaBridge
from .homeomorphism import (
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
    "simplify_presentation",
    "extract_stiefel_whitney_w2",
    "check_spin_structure",
    "extract_pontryagin_p1",
    "verify_hirzebruch_signature",
    "KirbyDiagram",
    "WhiteheadGroup",
    "compute_whitehead_group",
    "AlgebraicPoincareComplex",
    "AlgebraicSurgeryComplex",
    "StructureSet",
    "NormalInvariantsResult",
    "SurgeryExactSequenceResult",
    "WallGroupL",
    "ObstructionResult",
    "JuliaBridge",
    "HomeomorphismResult",
    "analyze_homeomorphism_2d",
    "analyze_homeomorphism_2d_result",
    "analyze_homeomorphism_3d",
    "analyze_homeomorphism_3d_result",
    "analyze_homeomorphism_4d",
    "analyze_homeomorphism_4d_result",
    "analyze_homeomorphism_high_dim",
    "analyze_homeomorphism_high_dim_result",
    "surgery_to_remove_impediments",
    "integrations"
]
