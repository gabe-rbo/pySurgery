from dataclasses import dataclass, field
from typing import Literal, Tuple
from .core.intersection_forms import IntersectionForm
from .core.complexes import ChainComplex
from .core.exceptions import DimensionError
from .core.fundamental_group import FundamentalGroup, GroupPresentation
from .core.k_theory import WhiteheadGroup, compute_whitehead_group
from .wall_groups import ObstructionResult, WallGroupL
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
    exact: bool = True

    def to_legacy_tuple(self) -> Tuple[bool | None, str]:
        return self.is_homeomorphic, self.reasoning


def _normalize_torsion(torsion: list[int]) -> list[int]:
    return sorted(abs(int(x)) for x in torsion if abs(int(x)) > 1)


def _check_cohomology_equivalence(
    c1: ChainComplex,
    c2: ChainComplex,
    max_dim: int,
    theorem: str,
    allow_approx: bool,
) -> HomeomorphismResult | None:
    if c1.coefficient_ring != c2.coefficient_ring:
        return HomeomorphismResult(
            status="inconclusive",
            is_homeomorphic=None,
            reasoning=(
                "INCONCLUSIVE: Cohomology comparison requires a shared coefficient ring; "
                f"received {c1.coefficient_ring!r} vs {c2.coefficient_ring!r}."
            ),
            theorem=theorem,
            missing_data=["Common coefficient ring for cohomology invariants"],
        )

    for n in range(max_dim + 1):
        try:
            r1, t1 = c1.cohomology(n)
            r2, t2 = c2.cohomology(n)
        except Exception as e:
            if allow_approx:
                warnings.warn(
                    f"Topological Hint: Cohomology extraction failed at dimension {n} ({e!r}). Exact classification disabled."
                )
            return HomeomorphismResult(
                status="inconclusive",
                is_homeomorphic=None,
                reasoning=f"INCONCLUSIVE: Exact cohomology extraction failed at dimension {n} ({e!r}).",
                theorem=theorem,
                missing_data=[f"Exact H^{n}"],
                exact=False,
            )

        t1n = _normalize_torsion(t1)
        t2n = _normalize_torsion(t2)
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
    cup_product_signature_1: dict | None,
    cup_product_signature_2: dict | None,
    theorem: str,
) -> HomeomorphismResult | None:
    if cup_product_signature_1 is None and cup_product_signature_2 is None:
        return None
    if (cup_product_signature_1 is None) != (cup_product_signature_2 is None):
        return HomeomorphismResult(
            status="inconclusive",
            is_homeomorphic=None,
            reasoning=(
                "INCONCLUSIVE: Cohomology-ring comparison requires cup-product signatures "
                "for both manifolds."
            ),
            theorem=theorem,
            missing_data=["Cup-product signature for both manifolds"],
        )
    if cup_product_signature_1 != cup_product_signature_2:
        return HomeomorphismResult(
            status="impediment",
            is_homeomorphic=False,
            reasoning="IMPEDIMENT: Cohomology-ring signatures differ (cup-product incompatibility).",
            theorem=theorem,
            evidence=["Cup-product signature mismatch"],
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


def _presentation_key(pi1: FundamentalGroup) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    rels = tuple(tuple(tok for tok in rel) for rel in pi1.relations)
    return tuple(pi1.generators), rels


def _infer_pi_group_descriptor(pi1: FundamentalGroup | None, pi_group: str | GroupPresentation | None) -> str | GroupPresentation | None:
    if pi_group is not None:
        return pi_group
    if pi1 is None:
        return None
    if not pi1.generators:
        return "1"
    if not pi1.relations and len(pi1.generators) == 1:
        return "Z"
    return None


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


def _search_integer_isometry(Q1: np.ndarray, Q2: np.ndarray, max_entry: int = 2) -> np.ndarray | None:
    """Bounded search for U in GL_n(Z) with U^T Q1 U = Q2 (small-rank fallback)."""
    if Q1.ndim != 2 or Q2.ndim != 2 or Q1.shape[0] != Q1.shape[1] or Q2.shape[0] != Q2.shape[1]:
        return None
    if Q1.shape != Q2.shape:
        return None
    n = Q1.shape[0]
    if n > 4:
        return None
    values = range(-max_entry, max_entry + 1)
    for entries in itertools.product(values, repeat=n * n):
        U = np.array(entries, dtype=np.int64).reshape((n, n))
        det = _det_int_small(U)
        if abs(det) != 1:
            continue
        if np.array_equal(U.T @ Q1 @ U, Q2):
            return U
    return None

def analyze_homeomorphism_2d_result(
    c1: ChainComplex,
    c2: ChainComplex,
    allow_approx: bool = False,
    *,
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
            warnings.warn(f"Topological Hint: H_2 homology extraction failed ({e!r}). Exact classification disabled.")
        return HomeomorphismResult(
            status="inconclusive",
            is_homeomorphic=None,
            reasoning=f"INCONCLUSIVE: Exact H_2 computation failed ({e!r}).",
            theorem="Classification of Closed Surfaces",
            missing_data=["Exact H_2"],
            exact=False,
        )

    orientable_1 = (r2_1 == 1)
    orientable_2 = (r2_2 == 1)
    
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
            warnings.warn(f"Topological Hint: H_1 homology extraction failed ({e!r}). Exact classification disabled.")
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
    )
    if coho_check is not None:
        return coho_check

    cup_check = _check_cup_product_compatibility(
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
    cup_product_signature_1: dict | None = None,
    cup_product_signature_2: dict | None = None,
) -> Tuple[bool | None, str]:
    return analyze_homeomorphism_2d_result(
        c1,
        c2,
        allow_approx=allow_approx,
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
    cup_product_signature_1: dict | None = None,
    cup_product_signature_2: dict | None = None,
) -> HomeomorphismResult:
    """
    Analyzes the potential for homeomorphism between two 3-dimensional manifolds.
    
    Warning: 3-manifolds are classified by Thurston's Geometrization (Perelman, 2003).
    Algebraic topology alone (homology) is insufficient to prove homeomorphism in general 
    (e.g., Poincare homology spheres have the same homology as S^3 but different fundamental groups).
    """
    # Check basic homology equivalence (exact-only for certifying statements).
    for n in range(4):
        try:
            r_1, t_1 = c1.homology(n)
            r_2, t_2 = c2.homology(n)
        except Exception as e:
            if allow_approx:
                warnings.warn(f"Topological Hint: Homology extraction failed at dimension {n} ({e!r}). Exact classification disabled.")
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

    if pi1_1 is not None and pi1_2 is not None and _presentation_key(pi1_1) != _presentation_key(pi1_2):
        return HomeomorphismResult(
            status="impediment",
            is_homeomorphic=False,
            reasoning="IMPEDIMENT: Fundamental groups differ; manifolds are not homeomorphic.",
            theorem="Geometrization / 3-manifold recognition",
            evidence=["pi_1 presentation mismatch"],
        )

    coho_check = _check_cohomology_equivalence(
        c1,
        c2,
        max_dim=3,
        theorem="Geometrization / 3-manifold recognition",
        allow_approx=allow_approx,
    )
    if coho_check is not None:
        return coho_check

    cup_check = _check_cup_product_compatibility(
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
        if not pi1_1.generators and not pi1_2.generators:
            return HomeomorphismResult(
                status="inconclusive",
                is_homeomorphic=None,
                reasoning="INCONCLUSIVE: Homology-sphere and trivial pi_1 evidence is promising; a full 3-manifold recognition pipeline is still required in this API.",
                theorem="Poincare Conjecture / Geometrization",
                assumptions=["Closed connected 3-manifold hypotheses must hold"],
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
    cup_product_signature_1: dict | None = None,
    cup_product_signature_2: dict | None = None,
) -> Tuple[bool | None, str]:
    return analyze_homeomorphism_3d_result(
        c1,
        c2,
        allow_approx=allow_approx,
        cup_product_signature_1=cup_product_signature_1,
        cup_product_signature_2=cup_product_signature_2,
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
    cup_product_signature_1: dict | None = None,
    cup_product_signature_2: dict | None = None,
) -> HomeomorphismResult:
    """
    Analyzes homeomorphism for high-dimensional manifolds (n >= 5) using the s-Cobordism Theorem 
    and Smale's Generalized Poincare Conjecture (1961).
    """
    if dim < 5:
        raise DimensionError(f"Function called on {dim}D. The s-Cobordism theorem and Wall's high-dimensional surgery framework strictly apply to n >= 5, where the 'Whitney Trick' guarantees enough room to untangle handles.")
        
    # Check Homology Equivalence
    for n in range(dim + 1):
        try:
            r_1, t_1 = c1.homology(n)
            r_2, t_2 = c2.homology(n)
        except Exception as e:
            if allow_approx:
                warnings.warn(f"Topological Hint: Homology extraction failed at dimension {n} ({e!r}). Exact classification disabled.")
            return HomeomorphismResult(
                status="inconclusive",
                is_homeomorphic=None,
                reasoning=f"INCONCLUSIVE: Exact homology extraction failed at dimension {n} ({e!r}).",
                theorem="s-Cobordism / surgery classification",
                missing_data=[f"Exact H_{n}"],
                exact=False,
            )

        t_1n = _normalize_torsion(t_1)
        t_2n = _normalize_torsion(t_2)
        if r_1 != r_2 or t_1n != t_2n:
            return HomeomorphismResult(
                status="impediment",
                is_homeomorphic=False,
                reasoning=f"IMPEDIMENT: Homology mismatch in dimension {n} (Rank: {r_1} vs {r_2}, Torsion: {t_1n} vs {t_2n}).",
                theorem="s-Cobordism / surgery classification",
                evidence=[f"H_{n} mismatch"],
            )

    coho_check = _check_cohomology_equivalence(
        c1,
        c2,
        max_dim=dim,
        theorem="s-Cobordism / surgery classification",
        allow_approx=allow_approx,
    )
    if coho_check is not None:
        return coho_check

    cup_check = _check_cup_product_compatibility(
        cup_product_signature_1,
        cup_product_signature_2,
        theorem="s-Cobordism / surgery classification",
    )
    if cup_check is not None:
        return cup_check

    descriptor = _infer_pi_group_descriptor(pi1, pi_group)
    if descriptor is None:
        return HomeomorphismResult(
            status="inconclusive",
            is_homeomorphic=None,
            reasoning=f"INCONCLUSIVE: Homology matches in {dim}D, but pi_1/group-ring descriptor is missing. s-Cobordism requires Whitehead and Wall obstruction checks.",
            theorem="s-Cobordism / surgery classification",
            missing_data=["pi_1 or supported pi-group descriptor", "Whitehead torsion", "Wall obstruction"],
        )

    wh = whitehead_group
    if wh is None:
        if pi1 is not None:
            wh = compute_whitehead_group(pi1)
        elif descriptor == "1":
            wh = WhiteheadGroup(rank=0, description="Wh(1)=0")

    if wh is None:
        return HomeomorphismResult(
            status="inconclusive",
            is_homeomorphic=None,
            reasoning="INCONCLUSIVE: Whitehead torsion was not provided and cannot be inferred from available data.",
            theorem="s-Cobordism / surgery classification",
            missing_data=["Whitehead torsion Wh(pi_1)"],
        )

    if not wh.computable:
        return HomeomorphismResult(
            status="inconclusive",
            is_homeomorphic=None,
            reasoning=f"INCONCLUSIVE: Whitehead torsion computation failed ({wh.description}).",
            theorem="s-Cobordism / surgery classification",
            missing_data=["Computable Wh(pi_1)"],
            assumptions=wh.assumptions,
            exact=wh.exact,
        )

    if wh.rank > 0:
        return HomeomorphismResult(
            status="surgery_required",
            is_homeomorphic=False,
            reasoning=f"SURGERY_REQUIRED: Whitehead torsion obstruction detected (rank >= {wh.rank}).",
            theorem="s-Cobordism / surgery classification",
            evidence=[wh.description],
            assumptions=wh.assumptions,
            exact=wh.exact,
        )

    wall = wall_obstruction
    if wall is None:
        try:
            wall = WallGroupL(dimension=dim, pi=descriptor).compute_obstruction_result(wall_form)
        except Exception as e:
            return HomeomorphismResult(
                status="inconclusive",
                is_homeomorphic=None,
                reasoning=f"INCONCLUSIVE: Wall obstruction evaluation failed ({e!r}).",
                theorem="s-Cobordism / surgery classification",
                missing_data=["Computable Wall L-group obstruction"],
            )

    if not wall.computable:
        return HomeomorphismResult(
            status="inconclusive",
            is_homeomorphic=None,
            reasoning=f"INCONCLUSIVE: Wall obstruction not computable ({wall.message}).",
            theorem="s-Cobordism / surgery classification",
            missing_data=[f"Computable L_{dim}({wall.pi}) obstruction"],
            assumptions=wall.assumptions,
            exact=wall.exact,
        )

    if wall.value is not None and int(wall.value) != 0:
        return HomeomorphismResult(
            status="surgery_required",
            is_homeomorphic=False,
            reasoning=f"SURGERY_REQUIRED: Non-zero Wall obstruction detected in L_{dim}({wall.pi}) (value={wall.value}).",
            theorem="s-Cobordism / surgery classification",
            evidence=["Whitehead obstruction vanishes", f"Wall obstruction value {wall.value}"],
            assumptions=wall.assumptions,
            exact=wall.exact,
        )

    s1 = _homology_sphere_like(c1, dim)
    s2 = _homology_sphere_like(c2, dim)
    if s1 is None or s2 is None:
        return HomeomorphismResult(
            status="inconclusive",
            is_homeomorphic=None,
            reasoning="INCONCLUSIVE: Could not verify homology-sphere side conditions exactly.",
            theorem="s-Cobordism / surgery classification",
            missing_data=["Exact homology-sphere verification"],
        )

    if s1 and s2 and descriptor == "1":
        return HomeomorphismResult(
            status="success",
            is_homeomorphic=True,
            reasoning=f"SUCCESS: Homology-sphere conditions, Wh(pi_1)=0, and vanishing Wall obstruction support homeomorphism in {dim}D under s-cobordism/generalized Poincare hypotheses.",
            theorem="s-Cobordism / generalized Poincare",
            evidence=["Homology sphere checks passed", "Wh(pi_1)=0", f"L_{dim}(pi_1) obstruction vanishes"],
            assumptions=["Closed connected manifold hypotheses", "Input normal-map/surgery model is valid"],
            exact=wall.exact and wh.exact,
        )

    return HomeomorphismResult(
        status="inconclusive",
        is_homeomorphic=None,
        reasoning=f"INCONCLUSIVE: Homology equivalence holds in {dim}D and computed surgery obstructions vanish, but this API has no explicit homotopy-equivalence witness to complete classification.",
        theorem="s-Cobordism / surgery classification",
        evidence=["Homology match", "Wh(pi_1)=0", f"L_{dim}(pi_1) obstruction vanishes"],
        assumptions=wall.assumptions + wh.assumptions,
        exact=wall.exact and wh.exact,
    )


def analyze_homeomorphism_high_dim(
    c1: ChainComplex,
    c2: ChainComplex,
    dim: int,
    allow_approx: bool = False,
    *,
    cup_product_signature_1: dict | None = None,
    cup_product_signature_2: dict | None = None,
) -> Tuple[bool | None, str]:
    return analyze_homeomorphism_high_dim_result(
        c1,
        c2,
        dim=dim,
        allow_approx=allow_approx,
        cup_product_signature_1=cup_product_signature_1,
        cup_product_signature_2=cup_product_signature_2,
    ).to_legacy_tuple()

def analyze_homeomorphism_4d_result(
    m1: IntersectionForm,
    m2: IntersectionForm,
    ks1: int | None = None,
    ks2: int | None = None,
    *,
    simply_connected: bool | None = None,
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
        raise DimensionError(f"Freedman's Classification Theorem strictly governs simply-connected 4-manifolds via intersection forms. "
                             f"Received manifolds of dimensions {m1.dimension} and {m2.dimension}. Hint: Use 2D, 3D, or high_dim analyzers instead.")

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
            evidence=["Indefinite unimodular forms classified by rank/signature/type", "Matching KS invariant"],
        )

    # Case: Definite forms (require lattice isomorphism)
    Q1 = np.asarray(m1.matrix, dtype=np.int64)
    Q2 = np.asarray(m2.matrix, dtype=np.int64)
    if np.array_equal(Q1, Q2):
        return HomeomorphismResult(
            status="success",
            is_homeomorphic=True,
            reasoning="SUCCESS: Definite intersection forms match exactly as integer lattices.",
            theorem="Freedman classification",
            evidence=["Exact matrix equality"],
        )

    U = _search_integer_isometry(Q1, Q2, max_entry=2)
    if U is not None:
        return HomeomorphismResult(
            status="success",
            is_homeomorphic=True,
            reasoning="SUCCESS: Definite lattice isomorphism certificate found (U^T Q1 U = Q2).",
            theorem="Freedman classification",
            evidence=["Explicit unimodular isometry witness"],
        )

    return HomeomorphismResult(
        status="inconclusive",
        is_homeomorphic=None,
        reasoning="INCONCLUSIVE: No bounded-search unimodular lattice-isometry certificate found for definite forms.",
        theorem="Freedman classification",
        missing_data=["Exact lattice-isometry solver for definite forms"],
    )


def analyze_homeomorphism_4d(
    m1: IntersectionForm,
    m2: IntersectionForm,
    ks1: int | None = None,
    ks2: int | None = None,
    *,
    simply_connected: bool | None = None,
) -> Tuple[bool | None, str]:
    return analyze_homeomorphism_4d_result(
        m1,
        m2,
        ks1=ks1,
        ks2=ks2,
        simply_connected=simply_connected,
    ).to_legacy_tuple()

def surgery_to_remove_impediments(m: IntersectionForm, target_sig: int) -> Tuple[bool, str]:
    """
    Analyzes if surgery can be used to remove the 'impediment' to a target signature.
    """
    sig_diff = m.signature() - target_sig
    if sig_diff == 0:
        return True, "Signatures already match. No signature-adjustment surgery required; parity, KS, Wh(pi_1), and Wall obstructions may still need checks."
    # Blow-up with CP^2 or -CP^2 changes signature by +/-1 and rank by 1.
    n_blowups = abs(sig_diff)
    blowup_type = "CP^2" if sig_diff < 0 else "(-CP^2)"
    return True, (
        f"PLAN: Connected sum with {n_blowups} copies of {blowup_type} "
        f"changes signature by {-sig_diff}. "
        "This only addresses signature-level obstructions; complete homeomorphism analysis may still require "
        "Kirby-Siebenmann agreement, Whitehead-torsion vanishing, and Wall L-group obstruction checks."
    )