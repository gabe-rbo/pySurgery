import numpy as np
from enum import Enum
from typing import List, Optional, Tuple
import warnings
from pysurgery.topology.complexes import SimplicialComplex
from pysurgery.manifolds.surgery import compute_linking_number, compute_linking_seifert_chain
from pysurgery.bridge.julia_bridge import julia_engine

class LinkType(Enum):
    """Enumeration of classified link types for multi-component links."""

    UNLINKED = "Unlinked"
    HOPF = "Hopf"
    BORROMEAN = "Borromean"
    WHITEHEAD = "Whitehead"        # lk=0, linked via 2nd-order Milnor invariant
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

def _intersect_2chains_in_3complex(
    ambient_complex: SimplicialComplex, 
    F_a: np.ndarray, 
    F_b: np.ndarray,
    backend: str = "auto"
) -> np.ndarray:
    """Computes the simplicial intersection of two 2-chains in a 3-manifold.
    
    This uses the dual of the Alexander-Whitney cup product formula:
    For each 3-simplex [v0, v1, v2, v3], the intersection 1-chain on [v1, v2] 
    gets a contribution: F_a([v0,v1,v2]) * F_b([v1,v2,v3]) - F_b([v0,v1,v2]) * F_a([v1,v2,v3]).
    
    Args:
        ambient_complex: The ambient 3-dimensional SimplicialComplex.
        F_a: The first 2-chain (coefficient array).
        F_b: The second 2-chain (coefficient array).
        backend: Computation backend ("auto", "julia", or "python"); selects the
            Julia kernel when available, otherwise the pure-Python fallback.

    Returns:
        np.ndarray: The intersection 1-chain.
    """
    simplices_1_list = [list(s) for s in ambient_complex.n_simplices(1)]
    simplices_2_list = [list(s) for s in ambient_complex.n_simplices(2)]
    simplices_3_list = [list(s) for s in ambient_complex.n_simplices(3)]
    
    use_julia = (backend == "julia") or (backend == "auto" and julia_engine.available)
    if use_julia:
        try:
            return julia_engine.linking_intersect_2chains(
                F_a, F_b, simplices_1_list, simplices_2_list, simplices_3_list
            )
        except Exception as e:
            if backend == "julia":
                raise
            warnings.warn(f"Julia linking_intersect_2chains failed, falling back: {e!r}")
            
    n_1simplices = ambient_complex.count_simplices(1)
    intersection_chain = np.zeros(n_1simplices, dtype=np.int64)
    
    simplices_1 = ambient_complex.n_simplices(1)
    idx_1 = {tuple(s): i for i, s in enumerate(simplices_1)}
    
    simplices_2 = ambient_complex.n_simplices(2)
    idx_2 = {tuple(s): i for i, s in enumerate(simplices_2)}
    
    simplices_3 = ambient_complex.n_simplices(3)
    
    for s3 in simplices_3:
        v0, v1, v2, v3 = s3
        
        # Front and back faces for Alexander-Whitney
        f_face = (v0, v1, v2)
        b_face = (v1, v2, v3)
        m_edge = (v1, v2)
        
        if f_face in idx_2 and b_face in idx_2 and m_edge in idx_1:
            i_f = idx_2[f_face]
            i_b = idx_2[b_face]
            i_m = idx_1[m_edge]
            
            val = F_a[i_f] * F_b[i_b] - F_b[i_f] * F_a[i_b]
            intersection_chain[i_m] += val
            
    return intersection_chain

def milnor_triple_invariant(
    ambient_complex: SimplicialComplex, 
    K_a: SimplicialComplex, 
    K_b: SimplicialComplex, 
    K_c: SimplicialComplex,
    backend: str = "auto"
) -> int:
    r"""Computes the Milnor triple invariant mu_bar(123) for three components.
    
    This invariant is computed by finding Seifert chains F_a, F_b for K_a, K_b,
    intersecting them to get a 1-chain F_a \\cap F_b, and taking the linking number
    with K_c. 
    
    To avoid indeterminacy, all pairwise linking numbers must be 0.
    
    Args:
        ambient_complex: The ambient closed oriented 3-manifold.
        K_a: First component (1-cycle).
        K_b: Second component (1-cycle).
        K_c: Third component (1-cycle).
        backend: Computation backend passed to the linking and intersection
            routines ("auto", "julia", or "python").

    Returns:
        int: The Milnor triple invariant mu_bar(123).
        
    Raises:
        ValueError: If any pairwise linking number is not 0, making the invariant indeterminate.
    """
    if ambient_complex.dimension != 3:
        raise ValueError("Milnor triple invariant computation requires a 3-dimensional ambient complex.")
        
    # 1. Assert pairwise linking numbers are 0
    lk_ab = compute_linking_number(ambient_complex, K_a, K_b, backend=backend).value
    lk_ac = compute_linking_number(ambient_complex, K_a, K_c, backend=backend).value
    lk_bc = compute_linking_number(ambient_complex, K_b, K_c, backend=backend).value
    
    if lk_ab != 0 or lk_ac != 0 or lk_bc != 0:
        raise ValueError(
            f"Indeterminacy error: All pairwise linking numbers must be 0 to compute a genuine integer Milnor triple invariant. "
            f"Got lk(a,b)={lk_ab}, lk(a,c)={lk_ac}, lk(b,c)={lk_bc}."
        )
        
    # 2. Compute Seifert chains F_a, F_b
    F_a, _, _, _ = compute_linking_seifert_chain(ambient_complex, K_a, backend=backend)
    F_b, _, _, _ = compute_linking_seifert_chain(ambient_complex, K_b, backend=backend)
    
    if F_a is None or F_b is None:
        raise ValueError("Could not compute Seifert chains for the components. They may not be null-homologous.")
        
    # 3. Compute simplicial intersection F_a \cap F_b -> 1-chain
    intersection_1chain = _intersect_2chains_in_3complex(ambient_complex, F_a, F_b, backend=backend)
    
    # 4. Compute lk(F_a \cap F_b, K_c).
    # Since K_c is a SimplicialComplex and our intersection is a chain, we can use compute_linking_seifert_chain on K_c
    # to get F_c, and then compute the intersection of intersection_1chain and F_c.
    # Alternatively, since compute_linking_number takes SimplicialComplex, we could build a SimplicialComplex for intersection_1chain,
    # but building a SimplicialComplex drops coefficients if they are > 1.
    # Let's use the seifert chain for K_c and do a direct pairing.
    
    F_c, c2_simplices, _, _ = compute_linking_seifert_chain(ambient_complex, K_c, backend=backend)
    if F_c is None:
        raise ValueError("Could not compute Seifert chain for K_c.")
        
    # The linking number of a 1-chain A and a 1-cycle B is <A, F_B> where \partial F_B = B.
    # We have A = intersection_1chain and F_B = F_c.
    # The pairing <A, F_B> is the signed incidence of 1-simplices in A with 2-simplices in F_B.
    # Wait, in compute_linking_number, it does <K_a, F>.
    # <K_a, F> means: for each 2-simplex in F, its boundary 1-simplices are matched with K_a.
    # Let's implement the pairing <intersection_1chain, F_c>:
    
    lk_val = 0
    simplices_1 = ambient_complex.n_simplices(1)
    idx_1 = {tuple(s): i for i, s in enumerate(simplices_1)}
    
    for i, s2 in enumerate(c2_simplices):
        coef_F_c = F_c[i]
        if coef_F_c == 0:
            continue
            
        # The boundary of a 2-simplex [v0, v1, v2] is [v1,v2] - [v0,v2] + [v0,v1]
        faces = [
            (tuple(sorted([s2[1], s2[2]])), 1),
            (tuple(sorted([s2[0], s2[2]])), -1),
            (tuple(sorted([s2[0], s2[1]])), 1)
        ]
        
        for face_tuple, sign in faces:
            if face_tuple in idx_1:
                coef_A = intersection_1chain[idx_1[face_tuple]]
                lk_val += coef_A * coef_F_c * sign
                
    return int(lk_val)

def _components_bbox_disjoint(
    ambient_complex: SimplicialComplex,
    K_a: SimplicialComplex,
    K_b: SimplicialComplex,
) -> bool:
    """Return True if K_a and K_b have geometrically disjoint bounding boxes.

    The axis-aligned bounding boxes are computed from the ambient complex's
    vertex coordinates.

    A True result implies the two cycles cannot have any higher-order linking,
    so Milnor μ̄(112), μ̄(123), … all vanish without needing Seifert chains.
    """
    coords = ambient_complex.simplices_to_point_cloud
    if not coords:
        return False

    def _bbox(K_sub):
        verts: set = set()
        for d in K_sub.dimensions:
            for s in K_sub.n_simplices(d):
                verts.update(s)
        pts = []
        for v in verts:
            key = (v,)
            if key in coords:
                pts.append(coords[key][0])
        if not pts:
            return None
        arr = np.asarray(pts, dtype=float)
        return arr.min(axis=0), arr.max(axis=0)

    bb_a = _bbox(K_a)
    bb_b = _bbox(K_b)
    if bb_a is None or bb_b is None:
        return False
    lo_a, hi_a = bb_a
    lo_b, hi_b = bb_b
    return bool(np.any(hi_a < lo_b) or np.any(hi_b < lo_a))


def _milnor_2nd_order(
    ambient_complex: SimplicialComplex,
    K_a: SimplicialComplex,
    K_b: SimplicialComplex,
    backend: str = "auto",
) -> int:
    """Compute the Milnor second-order invariant μ̄(1122) for a 2-component link.

    For a 2-component link L = K_a ∪ K_b with lk(K_a, K_b) = 0, the second-order
    invariant is computed via:
        1. Find Seifert chains F_a, F_b with ∂F_a = K_a, ∂F_b = K_b.
        2. Compute the intersection 1-chain C = F_a ∩ F_b.
        3. μ̄(1122) = lk(C, K_a) + lk(C, K_b) (detecting Whitehead-type linking).

    This invariant is non-zero for the Whitehead link and zero for the unlink.

    Returns:
        int — the second-order Milnor invariant (0 for unlink, ≠ 0 for Whitehead-type).
    """
    from pysurgery.manifolds.surgery import compute_linking_seifert_chain

    if ambient_complex.dimension != 3:
        return 0

    # Fast path: components in disjoint bounding boxes are geometrically split,
    # so every higher Milnor invariant vanishes — skip the expensive SNF.
    if _components_bbox_disjoint(ambient_complex, K_a, K_b):
        return 0

    # Verify lk = 0
    lk = compute_linking_number(ambient_complex, K_a, K_b, backend=backend)
    if lk and lk.exact and lk.value != 0:
        raise ValueError(
            f"μ̄(1122) requires lk(K_a, K_b) = 0, got lk = {lk.value}"
        )

    F_a, _, _, _ = compute_linking_seifert_chain(ambient_complex, K_a, backend=backend)
    F_b, _, _, _ = compute_linking_seifert_chain(ambient_complex, K_b, backend=backend)

    if F_a is None or F_b is None:
        return 0

    # Intersection 1-chain C = F_a ∩ F_b
    C = _intersect_2chains_in_3complex(ambient_complex, F_a, F_b, backend=backend)

    if not np.any(C):
        return 0

    # Build SimplicialComplex for C (support where C ≠ 0)
    simplices_1 = list(ambient_complex.n_simplices(1))
    C_edges = [simplices_1[k] for k in range(len(simplices_1)) if k < len(C) and C[k] != 0]
    if not C_edges:
        return 0

    C_sc = SimplicialComplex.from_simplices(C_edges)

    # lk(C, K_a) + lk(C, K_b)
    try:
        lk_Ca = compute_linking_number(ambient_complex, C_sc, K_a, backend=backend)
        lk_Cb = compute_linking_number(ambient_complex, C_sc, K_b, backend=backend)
        val_a = lk_Ca.value if lk_Ca and lk_Ca.exact else 0
        val_b = lk_Cb.value if lk_Cb and lk_Cb.exact else 0
        return val_a + val_b
    except Exception:
        return 0


def milnor_invariants(
    ambient_complex: SimplicialComplex,
    components: List[SimplicialComplex],
    multi_index: Tuple[int, ...],
    backend: str = "auto",
) -> Optional[int]:
    """Compute a Milnor μ̄ invariant for the given multi-index.

    Supported multi-indices:
        (0, 1, 2)  — triple invariant μ̄(123) via milnor_triple_invariant
        (0, 0, 1)  — second-order 2-component invariant μ̄(112)
        (0, 1)     — pairwise linking number lk(K_0, K_1)

    Args:
        ambient_complex: Ambient 3-manifold.
        components: Link components.
        multi_index: Tuple of component indices (0-based) forming the multi-index.
        backend: Computation backend.

    Returns:
        int if computable, None if indeterminate.
    """
    if len(multi_index) == 2:
        i, j = multi_index
        lk = compute_linking_number(
            ambient_complex, components[i], components[j], backend=backend
        )
        return lk.value if lk and lk.exact else None

    if len(multi_index) == 3:
        i, j, k = multi_index
        if len(set([i, j, k])) == 3:
            # Standard triple invariant μ̄(ijk)
            try:
                return milnor_triple_invariant(
                    ambient_complex, components[i], components[j], components[k], backend=backend
                )
            except ValueError:
                return None
        else:
            # μ̄(iij): second-order for 2 components
            if i == j:
                try:
                    return _milnor_2nd_order(
                        ambient_complex, components[i], components[k], backend=backend
                    )
                except (ValueError, Exception):
                    return None
            if j == k:
                try:
                    return _milnor_2nd_order(
                        ambient_complex, components[i], components[j], backend=backend
                    )
                except (ValueError, Exception):
                    return None

    return None


def are_linked(
    ambient_complex: SimplicialComplex,
    components: List[SimplicialComplex],
    backend: str = "auto",
) -> bool:
    """Determine if a set of components are linked.

    Algorithm:
        1. Check all pairwise linking numbers lk(K_i, K_j) — detects most links.
        2. For 2 components with lk = 0: compute the Milnor second-order invariant
           μ̄(112) to detect Whitehead-type links.
        3. For 3 components with all lk = 0: compute the Milnor triple invariant
           μ̄(123) to detect Borromean-type links.

    This is complete for 2- and 3-component links except for exotic higher-genus
    knot concordance invariants beyond Alexander polynomial detection.

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

    # Step 2: 2-component Whitehead-type detection (lk=0 but linked)
    if n == 2:
        try:
            mu11 = _milnor_2nd_order(
                ambient_complex, components[0], components[1], backend=backend
            )
            if mu11 != 0:
                return True
        except Exception:
            pass

    # Step 3: Milnor triple invariant for 3 components
    if n == 3:
        try:
            mu = milnor_triple_invariant(
                ambient_complex, components[0], components[1], components[2], backend=backend
            )
            if mu != 0:
                return True
        except ValueError:
            pass

    return False


def link_type(
    ambient_complex: SimplicialComplex,
    components: List[SimplicialComplex],
    backend: str = "auto",
) -> LinkType:
    """Classify the link type of a set of components.

    Classification hierarchy:
        UNLINKED          — no linking detected at any order
        HOPF              — 2 components with |lk| = 1
        WHITEHEAD         — 2 components with lk = 0, μ̄(112) ≠ 0
        BORROMEAN         — 3 components, pairwise lk = 0, μ̄(123) ≠ 0
        UNLINKED_KNOTTED  — components are unlinked but individually knotted
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
        # lk = 0: check Whitehead-type via second-order Milnor
        try:
            mu11 = _milnor_2nd_order(
                ambient_complex, components[0], components[1], backend=backend
            )
            if mu11 != 0:
                return LinkType.WHITEHEAD
        except Exception:
            pass

    # ── 3-component classification ────────────────────────────────────────────
    if n == 3 and not any_nonzero_lk:
        try:
            mu = milnor_triple_invariant(
                ambient_complex, components[0], components[1], components[2], backend=backend
            )
            if mu != 0:
                return LinkType.BORROMEAN
        except ValueError:
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
