import numpy as np
from enum import Enum
from typing import List, Optional, Tuple
from pysurgery.core.exceptions import UndefinedInvariantError
from pysurgery.topology.complexes import SimplicialComplex
from pysurgery.manifolds.surgery import compute_linking_number
from pysurgery.knots.seifert_surface import SeifertSurfaceError
from pysurgery.knots.triangulated_linking import triangulated_milnor_mu123, triangulated_sato_levine

class LinkType(Enum):
    """Enumeration of classified link types for multi-component links."""

    UNLINKED = "Unlinked"
    HOPF = "Hopf"
    BORROMEAN = "Borromean"
    WHITEHEAD = "Whitehead"        # 2 components, lk = 0, Sato-Levine beta = -mu(1122) != 0
    TORUS_LINK = "TorusLink"       # T(p,q) with p,q sharing a factor
    UNLINKED_KNOTTED = "UnlinkedKnotted"  # components unlinked but individually knotted
    UNKNOWN = "Unknown"

def simplices_to_chain(ambient_complex: SimplicialComplex, simplices: List[Tuple[int, ...]], dim: int) -> np.ndarray:
    """Converts a list of simplices into a chain coefficient vector.
    
    Args:
        ambient_complex: The ambient simplicial complex.
        simplices: List of simplices (tuples of vertices).
        dim: The dimension of the simplices.
        
    Returns:
        np.ndarray: A 1D array of length equal to the number of dim-simplices in the ambient complex,
                    where the i-th entry is the coefficient of the i-th simplex.
    """
    n_simplices = ambient_complex.count_simplices(dim)
    chain = np.zeros(n_simplices, dtype=np.int64)
    if n_simplices == 0:
        return chain
        
    ambient_simplices = ambient_complex.n_simplices(dim)
    simplex_to_idx = {tuple(s): i for i, s in enumerate(ambient_simplices)}
    
    for s in simplices:
        s_tuple = tuple(sorted(s))
        if s_tuple in simplex_to_idx:
            # We assume a coefficient of +1 for provided simplices.
            # To handle orientation, the user should provide a signed chain directly if needed,
            # but for standard unoriented inputs (Z2), +1 is sufficient.
            # For Z coefficients, we assume the provided simplices are consistently oriented.
            chain[simplex_to_idx[s_tuple]] += 1
            
    return chain

def linking_matrix(ambient_complex: SimplicialComplex, components: List[SimplicialComplex], backend: str = "auto") -> np.ndarray:
    """Computes the symmetric pairwise linking matrix L.
    
    Args:
        ambient_complex: The ambient simplicial complex (e.g. S^3).
        components: A list of 1-cycle SimplicialComplexes.
        backend: Computation backend passed to the linking-number routine
            ("auto", "julia", or "python").

    Returns:
        np.ndarray: The symmetric linking matrix where L[i, j] = lk(K_i, K_j).
    """
    n = len(components)
    L = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(i + 1, n):
            lk_result = compute_linking_number(ambient_complex, components[i], components[j], coefficient_ring="Z", backend=backend)
            val = lk_result.value if lk_result and lk_result.exact else 0
            L[i, j] = val
            L[j, i] = val
            
    return L

def milnor_triple_invariant(
    ambient_complex: SimplicialComplex,
    K_a: SimplicialComplex,
    K_b: SimplicialComplex,
    K_c: SimplicialComplex,
    backend: str = "auto"
) -> int:
    r"""Computes Milnor's triple linking number μ̄(123) of three link components.

    What is Being Computed?:
        For three disjoint knots in a triangulated rational homology 3-sphere or 3-ball
        whose pairwise linking numbers vanish, μ̄(123) = −lk(F_a ∩ G_b, K_c), where F_a
        is a PRIMAL Seifert chain of K_a avoiding a dual push-off K_b* of K_b, and G_b is
        a DUAL Seifert chain of K_b* avoiding K_a and K_c (the sign is the convention of
        ``diagrams.diagram_milnor_mu123``). Primal and dual chains are transverse, so
        F_a ∩ G_b is an honest closed curve, disjoint from K_c. It is ±1
        on the Borromean rings and 0 on the unlink, invariant under cyclic permutation
        of the components, and negated by a transposition or by reversing a component.
        The computation is exact and intrinsic to the triangulation (no vertex
        coordinates are used); see ``pysurgery.knots.triangulated_linking``.

    Args:
        ambient_complex: The ambient triangulated 3-manifold (closed, or with 2-sphere
            boundary components).
        K_a: First component (a simple closed curve in the 1-skeleton).
        K_b: Second component.
        K_c: Third component. The components are pairwise vertex-disjoint.
        backend: Accepted for API compatibility; the computation is exact rational
            arithmetic in Python.

    Returns:
        int: The Milnor triple invariant μ̄(123).

    Raises:
        UndefinedInvariantError: If a pairwise linking number is nonzero (μ̄(123) is
            then only defined modulo their gcd), or ``H_1(ambient; Q) != 0``.
        NotAManifoldError: If the ambient is not an orientable combinatorial
            3-manifold whose boundary components are 2-spheres.
        ValueError: If the ambient is not 3-dimensional, or a component is not a simple
            closed curve in its 1-skeleton disjoint from the others.
    """
    return triangulated_milnor_mu123(ambient_complex, K_a, K_b, K_c)


def sato_levine_invariant(
    ambient_complex: SimplicialComplex,
    K_a: SimplicialComplex,
    K_b: SimplicialComplex,
    backend: str = "auto",
) -> int:
    r"""Computes the Sato–Levine invariant β of a two-component link, β = −μ̄(1122).

    What is Being Computed?:
        For two disjoint knots with lk(K_a, K_b) = 0 in a triangulated rational homology
        3-sphere or 3-ball, take embedded Seifert surfaces F_a, F_b with
        F_a ∩ K_b = F_b ∩ K_a = ∅; β = lk(C, C⁺) for their intersection curve C and its
        push-off along either surface. It is the first invariant beyond lk of a
        two-component link: ±1 on the Whitehead link, 0 on split and boundary links,
        symmetric in the components, unchanged by reversing one, negated by mirror
        image. Computed exactly and intrinsically, with F_a PRIMAL and F_b DUAL so that
        they are transverse; see ``pysurgery.knots.triangulated_linking``.

    Args:
        ambient_complex: The ambient triangulated 3-manifold.
        K_a: First component (a simple closed curve in the 1-skeleton).
        K_b: Second component, sharing no vertex with K_a.
        backend: Accepted for API compatibility; the computation is exact.

    Returns:
        int: β(K_a ∪ K_b).

    Raises:
        UndefinedInvariantError: If lk(K_a, K_b) ≠ 0, or ``H_1(ambient; Q) != 0``.
        SeifertSurfaceError: If no embedded surfaces are found even after refining.
        NotAManifoldError: If the ambient is not an orientable combinatorial 3-manifold
            whose boundary components are 2-spheres.
        ValueError: If a component is not a simple closed curve disjoint from the other.
    """
    return triangulated_sato_levine(ambient_complex, K_a, K_b)


def milnor_invariants(
    ambient_complex: SimplicialComplex,
    components: List[SimplicialComplex],
    multi_index: Tuple[int, ...],
    backend: str = "auto",
) -> Optional[int]:
    """Compute a Milnor μ̄ invariant for the given multi-index.

    Supported multi-indices (0-based component indices):
        (i, j)     — the pairwise linking number lk(K_i, K_j)
        (i, j, k)  — i, j, k distinct: the triple invariant μ̄(ijk), via
                     milnor_triple_invariant
        (i, i, j), (i, j, i), (i, j, j), ... — length 3 with a repeated index. These
                     vanish whenever lk(K_i, K_j) = 0: by Milnor's shuffle relation
                     2 μ̄(iij) = 0 (the two shuffles of (i) with (i)) and
                     μ̄(ijj) + μ̄(jij) = 0, and the rest follow by cyclic symmetry.
        (i, i, j, j) and its cyclic permutations — −β, the Sato–Levine invariant
                     (sato_levine_invariant); (i, j, i, j) and (j, i, j, i) — 2β, by the
                     shuffle relation μ̄(1212) + 2 μ̄(1122) = 0. Defined when
                     lk(K_i, K_j) = 0.

    Args:
        ambient_complex: Ambient 3-manifold.
        components: Link components.
        multi_index: Tuple of component indices (0-based) forming the multi-index.
        backend: Computation backend.

    Returns:
        int if computable, None if indeterminate (a lower-order invariant is nonzero)
        or not supported.
    """
    if len(multi_index) == 2:
        i, j = multi_index
        lk = compute_linking_number(
            ambient_complex, components[i], components[j], backend=backend
        )
        return lk.value if lk and lk.exact else None

    if len(multi_index) == 3:
        distinct = sorted(set(multi_index))
        if len(distinct) == 3:
            i, j, k = multi_index
            try:
                return milnor_triple_invariant(
                    ambient_complex, components[i], components[j], components[k], backend=backend
                )
            except UndefinedInvariantError:
                return None
        if len(distinct) == 2:
            i, j = distinct
            lk = compute_linking_number(
                ambient_complex, components[i], components[j], backend=backend
            )
            return 0 if lk and lk.exact and lk.value == 0 else None

    if len(multi_index) == 4 and len(set(multi_index)) == 2:
        i, j = sorted(set(multi_index))
        seq = tuple(multi_index)
        rotations = {seq[r:] + seq[:r] for r in range(4)}
        if (i, j, i, j) in rotations:
            factor = 2
        elif (i, i, j, j) in rotations:
            factor = -1
        else:
            return None
        try:
            return factor * sato_levine_invariant(
                ambient_complex, components[i], components[j], backend=backend
            )
        except (UndefinedInvariantError, SeifertSurfaceError):
            return None

    return None


def are_linked(
    ambient_complex: SimplicialComplex,
    components: List[SimplicialComplex],
    backend: str = "auto",
) -> bool:
    """Determine whether linking is detected among a set of components.

    Algorithm:
        1. Check all pairwise linking numbers lk(K_i, K_j) — detects most links.
        2. For 2 components with lk = 0: compute the Sato–Levine invariant
           β = −μ̄(1122) to detect Whitehead-type links.
        3. For 3 components with all lk = 0: compute the Milnor triple invariant
           μ̄(123) to detect Borromean-type links.

    True certifies that the components are linked. False means none of these invariants
    detects linking; it does not certify a split link: linked two-component links with
    lk = β = 0 and three-component links with lk = μ̄(123) = 0 exist (linked at higher
    order), and β is skipped when no embedded Seifert surfaces are found.

    Args:
        ambient_complex: Ambient simplicial complex.
        components: List of component subcomplexes (1-cycles).
        backend: Computation backend passed to the linking and Milnor-invariant
            routines ("auto", "julia", or "python").

    Returns:
        bool: True if any linking is detected.
    """
    n = len(components)

    # Step 1: pairwise linking numbers
    for i in range(n):
        for j in range(i + 1, n):
            lk_result = compute_linking_number(
                ambient_complex, components[i], components[j], backend=backend
            )
            if lk_result and lk_result.exact and lk_result.value != 0:
                return True

    # Step 2: Sato–Levine invariant for 2 components (Whitehead-type linking)
    if n == 2:
        try:
            if sato_levine_invariant(ambient_complex, components[0], components[1], backend=backend):
                return True
        except (UndefinedInvariantError, SeifertSurfaceError):
            pass

    # Step 3: Milnor triple invariant for 3 components
    if n == 3:
        try:
            mu = milnor_triple_invariant(
                ambient_complex, components[0], components[1], components[2], backend=backend
            )
            if mu != 0:
                return True
        except UndefinedInvariantError:
            pass

    return False


def link_type(
    ambient_complex: SimplicialComplex,
    components: List[SimplicialComplex],
    backend: str = "auto",
) -> LinkType:
    """Classify the link type of a set of components.

    Classification hierarchy:
        UNLINKED          — no linking detected by the pairwise linking numbers, the
                            Sato–Levine invariant (2 components) or μ̄(123)
                            (3 components); this does not certify a split link (see
                            ``are_linked``)
        HOPF              — 2 components with |lk| = 1
        WHITEHEAD         — 2 components with lk = 0 and Sato–Levine β = −μ̄(1122) ≠ 0
        BORROMEAN         — 3 components, pairwise lk = 0, μ̄(123) ≠ 0
        UNLINKED_KNOTTED  — no linking detected, but a component is knotted
        UNKNOWN           — linked but not classified

    Args:
        ambient_complex: Ambient simplicial complex.
        components: List of 1-cycle subcomplexes.
        backend: Computation backend passed to the underlying linking and
            Milnor-invariant routines ("auto", "julia", or "python").

    Returns:
        LinkType enum value.
    """
    from pysurgery.knots.invariants import is_unknot

    n = len(components)
    if n == 0:
        return LinkType.UNLINKED

    # Compute pairwise linking numbers once
    lk_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            lk_result = compute_linking_number(
                ambient_complex, components[i], components[j], backend=backend
            )
            v = lk_result.value if lk_result and lk_result.exact else 0
            lk_matrix[i, j] = v
            lk_matrix[j, i] = v

    any_nonzero_lk = np.any(lk_matrix != 0)

    # ── 2-component classification ────────────────────────────────────────────
    if n == 2:
        lk_val = lk_matrix[0, 1]
        if abs(lk_val) == 1:
            return LinkType.HOPF
        if abs(lk_val) > 1:
            return LinkType.UNKNOWN
        try:
            if sato_levine_invariant(ambient_complex, components[0], components[1], backend=backend):
                return LinkType.WHITEHEAD
        except (UndefinedInvariantError, SeifertSurfaceError):
            pass

    # ── 3-component classification ────────────────────────────────────────────
    if n == 3 and not any_nonzero_lk:
        try:
            mu = milnor_triple_invariant(
                ambient_complex, components[0], components[1], components[2], backend=backend
            )
            if mu != 0:
                return LinkType.BORROMEAN
        except UndefinedInvariantError:
            pass

    # ── Generic linked case ───────────────────────────────────────────────────
    if any_nonzero_lk:
        return LinkType.UNKNOWN

    # ── All pairwise linking = 0: check individual knottedness ───────────────
    any_knotted = False
    for comp in components:
        try:
            if not is_unknot(ambient_complex, comp, backend=backend):
                any_knotted = True
                break
        except Exception:
            pass

    if any_knotted:
        return LinkType.UNLINKED_KNOTTED

    return LinkType.UNLINKED
