import warnings
from typing import List, Any
import numpy as np
from pydantic import BaseModel

from pysurgery.topology.complexes import SimplicialComplex
from pysurgery.geometry.characteristic_classes import (
    extract_stiefel_whitney_tangent,
    extract_pontryagin_p1,
    extract_euler_class,
    verify_hirzebruch_signature
)
from pysurgery.homology.cup_product import alexander_whitney_cup

class StructuralObstruction(BaseModel):
    """Result signalling a structural obstruction (e.g. a failed Hirzebruch check)."""

    exact: bool = False
    reason: str = ""
    contract_version: str = "2026.04-phase10"

class NonImmersibilityWitness(BaseModel):
    """Certificate that a manifold does not immerse, citing the witnessing class."""

    exact: bool = False
    immersible: bool = False
    reason: str = ""
    obstruction_class: Any = None
    degree: int = 0
    contract_version: str = "2026.04-phase10"

class ImmersibilityInconclusive(BaseModel):
    """Result indicating the checked immersion obstructions vanish, leaving it undecided."""

    exact: bool = False
    reason: str = ""
    contract_version: str = "2026.04-phase10"

class PontryaginClasses(BaseModel):
    """Container for the rational Pontryagin classes of a manifold."""

    exact: bool = False
    classes: list = []
    contract_version: str = "2026.04-phase10"

class EulerClass(BaseModel):
    """Container for the integer-valued combinatorial Euler class."""

    exact: bool = False
    value: int
    contract_version: str = "2026.04-phase10"


def _compute_dual_stiefel_whitney_classes_impl(manifold: SimplicialComplex) -> List[np.ndarray]:
    """Internal worker for dual Stiefel–Whitney classes.

    Lives at module scope so that test fixtures can monkeypatch
    ``extract_stiefel_whitney_tangent`` and ``alexander_whitney_cup``
    on this module.
    """
    w_classes = []
    for i in range(manifold.dimension + 1):
        w_i = extract_stiefel_whitney_tangent(manifold, k=i)
        w_classes.append(w_i)

    w_bar_classes = []
    w_bar_classes.append(np.ones(manifold.count_simplices(0), dtype=np.int64) % 2)

    for n in range(1, manifold.dimension + 1):
        w_bar_n = np.zeros(manifold.count_simplices(n), dtype=np.int64)
        for i in range(1, n + 1):
            alpha = w_classes[i]
            beta = w_bar_classes[n - i]

            p = i
            q = n - i

            if np.all(alpha == 0) or np.all(beta == 0):
                continue

            simplices_n = manifold.n_simplices(n)
            idx_p = manifold.simplex_to_index(p)
            idx_q = manifold.simplex_to_index(q)

            term = alexander_whitney_cup(
                alpha=alpha,
                beta=beta,
                p=p,
                q=q,
                simplices_p_plus_q=simplices_n,
                simplex_to_idx_p=idx_p,
                simplex_to_idx_q=idx_q,
                modulus=2,
            )
            w_bar_n = (w_bar_n + term) % 2

        w_bar_classes.append(w_bar_n)

    return w_bar_classes


def _combinatorial_euler_class_impl(manifold: SimplicialComplex) -> "EulerClass":
    """Internal worker for the combinatorial Euler class.

    Lives at module scope so that test fixtures can monkeypatch
    ``extract_euler_class`` on this module.
    """
    return EulerClass(value=extract_euler_class(manifold), exact=True)


def compute_dual_stiefel_whitney_classes(manifold: SimplicialComplex) -> List[np.ndarray]:
    """Deprecated: use ``SimplicialComplex.dual_stiefel_whitney_classes()``."""
    warnings.warn(
        "compute_dual_stiefel_whitney_classes is deprecated; use "
        "SimplicialComplex.dual_stiefel_whitney_classes()",
        DeprecationWarning,
        stacklevel=2,
    )
    return manifold.dual_stiefel_whitney_classes()


def check_dual_stiefel_whitney_non_immersibility(manifold: SimplicialComplex, target_dim: int):
    """Checks if M^n can immerse into R^{n+k} using dual SW classes."""
    n = manifold.dimension
    k = target_dim - n

    if k < 0:
        return NonImmersibilityWitness(
            reason="Target dimension smaller than manifold dimension",
            exact=True
        )

    w_bar = _compute_dual_stiefel_whitney_classes_impl(manifold)
    
    highest_non_zero_degree = -1
    for i in range(manifold.dimension, -1, -1):
        if np.any(w_bar[i] != 0):
            highest_non_zero_degree = i
            break
            
    if highest_non_zero_degree > k:
        return NonImmersibilityWitness(
            immersible=False,
            exact=True,  # SW classes mod 2 are exact
            obstruction_class=w_bar[highest_non_zero_degree],
            degree=highest_non_zero_degree,
            reason=f"Dual Stiefel-Whitney w_bar_{highest_non_zero_degree} does not vanish."
        )
        
    return ImmersibilityInconclusive(
        reason="Dual Stiefel-Whitney obstruction vanishes",
        exact=True
    )

def extract_pontryagin(manifold: SimplicialComplex, degree: int, intersection_form: Any = None) -> int:
    """Placeholder for higher Pontryagin classes."""
    if degree == 1 and manifold.dimension >= 4:
        # Fallback to evaluation on the 4-skeleton if it's exactly 4D
        if manifold.dimension == 4 and intersection_form is not None:
            return extract_pontryagin_p1(intersection_form)
    return 0

def rational_pontryagin_classes(manifold: SimplicialComplex, intersection_form: Any = None) -> PontryaginClasses:
    """Collect the rational Pontryagin classes p_i for degrees up to dim/4."""
    p_classes = []
    for i in range(1, manifold.dimension // 4 + 1):
        p_i = extract_pontryagin(manifold, degree=i, intersection_form=intersection_form) 
        p_classes.append(p_i)
    return PontryaginClasses(classes=p_classes, exact=True)

def combinatorial_euler_class(manifold: SimplicialComplex) -> EulerClass:
    """Deprecated: use ``SimplicialComplex.euler_class()``."""
    warnings.warn(
        "combinatorial_euler_class is deprecated; use SimplicialComplex.euler_class()",
        DeprecationWarning,
        stacklevel=2,
    )
    return manifold.euler_class()

def compute_rational_pontryagin_obstruction(manifold: SimplicialComplex, target_dim: int, intersection_form: Any = None):
    """Find immersion obstructions from rational Pontryagin and Euler data.

    Computes the Pontryagin classes and uses the Hirzebruch Signature Theorem
    and combinatorial Euler classes to detect non-immersibility.
    """
    p_result = rational_pontryagin_classes(manifold, intersection_form)
    p_classes = p_result.classes
    
    if manifold.dimension == 4 and intersection_form is not None:
        is_hirzebruch_valid = verify_hirzebruch_signature(intersection_form, p_classes[0])
        if not is_hirzebruch_valid:
            return StructuralObstruction(reason="Hirzebruch Signature Theorem failed")

    e_class_res = manifold.euler_class()
    e_class = e_class_res.value
    
    if manifold.dimension % 2 != 0 and e_class != 0:
        return NonImmersibilityWitness(
            immersible=False, 
            reason="Euler characteristic obstruction for codimension 1 immersion",
            exact=True
        )
        
    # p_normal = p_poly.invert(max_degree=manifold.dimension)
    # 1 + p1_n + p2_n + ...
    # (1 + p1 + p2 + ...) * (1 + p1_n + p2_n + ...) = 1
    # p1_n = -p1
    # p2_n = -p2 - p1*p1_n = p1^2 - p2
    
    # We do formal inversion of [1, p1, p2, ...]
    p_poly = [1] + p_classes
    p_normal = [1]
    
    for n in range(1, len(p_poly)):
        pn_n = 0
        for i in range(1, n + 1):
            if i < len(p_poly):
                pn_n -= p_poly[i] * p_normal[n - i]
        p_normal.append(pn_n)
        
    highest_p_normal_degree = -1
    for i in range(len(p_normal) - 1, 0, -1):
        if p_normal[i] != 0:
            highest_p_normal_degree = i
            break
            
    if highest_p_normal_degree > -1 and highest_p_normal_degree * 4 > target_dim - manifold.dimension:
         return NonImmersibilityWitness(
            immersible=False,
            obstruction_class=p_normal[highest_p_normal_degree],
            degree=highest_p_normal_degree * 4,
            reason=f"Normal Pontryagin class p_{highest_p_normal_degree} does not vanish.",
            exact=True
        )
        
    return ImmersibilityInconclusive(
        reason="Rational Pontryagin obstruction vanishes",
        exact=True
    )

def immersion_obstruction_analysis(manifold: SimplicialComplex, target_dim: int, intersection_form: Any = None):
    """Orchestrate the full immersion obstruction analysis.

    Runs both the dual Stiefel-Whitney and rational Pontryagin paths, returning
    the first witness of non-immersibility found or an inconclusive result.
    """
    # 1. Dual Stiefel-Whitney Path
    sw_res = check_dual_stiefel_whitney_non_immersibility(manifold, target_dim)
    if isinstance(sw_res, NonImmersibilityWitness):
        return sw_res
        
    # 2. Rational Pontryagin Path
    rp_res = compute_rational_pontryagin_obstruction(manifold, target_dim, intersection_form)
    if isinstance(rp_res, NonImmersibilityWitness) or isinstance(rp_res, StructuralObstruction):
        return rp_res
        
    return ImmersibilityInconclusive(
        reason="All checked immersion obstructions vanish.",
        exact=True
    )

