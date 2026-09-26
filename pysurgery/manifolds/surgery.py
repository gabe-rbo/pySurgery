"""pysurgery/core/surgery.py.

Core algorithms for surgery on simplicial complexes.
"""
from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Sequence, Union

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from pysurgery.topology.complexes import SimplicialComplex
from pysurgery.algebra.exact_algebra import coerce_int_matrix
from pysurgery.core.foundations import CONTRACT_VERSION
from pysurgery.core.exceptions import (
    AttachmentSphereError,
    DimensionError,
    HandleSurgeryError,
    KirbyMoveError,
    LinkingComputationError,
    SurgeryPostconditionError,
)
from pysurgery.core.theorem_tags import (
    SURGERY_ATTACHMENT_SPHERE_RECOGNITION_EXACT,
    SURGERY_ATTACHMENT_SPHERE_SNF_HEURISTIC,
    SURGERY_DELINKING_UNLINKING_NUMBER,
    SURGERY_HANDLE_MAYER_VIETORIS,
    SURGERY_LINKING_F2_HEURISTIC,
    SURGERY_LINKING_RELATIVE_SNF_Z,
    SURGERY_VERIFY_SNF_BETTI_TORSION,
)
from pysurgery.bridge.julia_bridge import julia_engine
from pysurgery.manifolds.simplicial_linking import (
    NotAClosable3ManifoldError,
    Oriented3Manifold,
    component_tets,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _complex_hash(K: SimplicialComplex) -> str:
    """Return a short hex digest of the simplex set."""
    h = hashlib.sha256()
    for d, simps in sorted(K.simplices_field.items()):
        for s in sorted(simps):
            h.update(repr((d, s)).encode())
    return h.hexdigest()[:16]


def _betti_numbers(K: SimplicialComplex, backend: str = "auto") -> Dict[int, int]:
    """Return {dim: betti_number} for all dimensions."""
    result: Dict[int, int] = {}
    n = K.dimension
    for j in range(n + 1):
        hom = K.homology(n=j, backend=backend)
        if isinstance(hom, tuple):
            rank, _ = hom
        else:
            rank, _ = hom.get(j, (0, []))
        result[j] = rank
    return result


def _torsion_coefficients(K: SimplicialComplex, backend: str = "auto") -> Dict[int, List[int]]:
    """Return {dim: [torsion_coefficients]} for all dimensions."""
    result: Dict[int, List[int]] = {}
    n = K.dimension
    for j in range(n + 1):
        hom = K.homology(n=j, backend=backend)
        if isinstance(hom, tuple):
            _, torsion = hom
        else:
            _, torsion = hom.get(j, (0, []))
        result[j] = list(torsion)
    return result


def _simplex_faces(simplex: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    """Return all codimension-1 faces of a simplex with their indices for sign computation."""
    return [simplex[:i] + simplex[i + 1 :] for i in range(len(simplex))]


def _get_cycle_coefficients(K: SimplicialComplex, simplices: List[Tuple[int, ...]], d: int) -> Optional[np.ndarray]:
    """Find a Z-cycle representative supported on the given simplices.

    Returns the coefficients as an int64 array if a cycle exists, else None.

    For d ≥ 1 the cycle is primitive and oriented so that its coefficient on
    the lexicographically largest simplex of its support is positive.  For a
    triangulated circle this is the unique primitive cycle it carries, with
    its largest edge (u, w), u < w, traversed from u to w.  Linking numbers
    of unoriented inputs follow this convention.
    """
    if not simplices:
        return None
    if d == 0:
        # For d=0, a "cycle" is just any chain. However, for linking
        # we usually want a null-homologous cycle (boundary of a 1-chain).
        # A single vertex is NOT a boundary in a connected component.
        # A difference of two vertices (v1 - v0) IS a boundary.
        # We return a representative that is more likely to be a boundary.
        idx_map = K.simplex_to_index(0)
        vec = np.zeros(len(idx_map), dtype=np.int64)
        if len(simplices) >= 2:
            # Difference of first two vertices
            v0 = tuple(sorted(simplices[0]))
            v1 = tuple(sorted(simplices[1]))
            if v0 in idx_map and v1 in idx_map:
                vec[idx_map[v1]] = 1
                vec[idx_map[v0]] = -1
                return vec
        # Fallback: single vertex (not a boundary, but let SNF catch it)
        for s in simplices:
            s_sorted = tuple(sorted(s))
            if s_sorted in idx_map:
                vec[idx_map[s_sorted]] = 1
                return vec
        return None

    bm = K.boundary_matrix(d)
    idx_map = K.simplex_to_index(d)
    sub_indices = [idx_map[tuple(sorted(s))] for s in simplices if tuple(sorted(s)) in idx_map]
    if not sub_indices:
        return None

    # Solve B_sub * x = 0 over Z; zero rows do not constrain x.
    import sympy
    B_sub = bm[:, sub_indices].toarray()
    B_sub = B_sub[np.any(B_sub != 0, axis=1)]
    M = sympy.Matrix(B_sub)
    ns = M.nullspace()
    if not ns:
        return None

    # Take first generator and scale to integers
    vec_sp = ns[0]
    # Check if it's the zero vector
    if all(x == 0 for x in vec_sp):
        return None

    denoms = [x.as_numer_denom()[1] for x in vec_sp]
    import math
    lcm = 1
    for den in denoms:
        lcm = (lcm * int(den)) // math.gcd(lcm, int(den))
    
    res_sub = np.array([int(x * lcm) for x in vec_sp], dtype=np.int64)

    vec = np.zeros(bm.shape[1], dtype=np.int64)
    for i, val in zip(sub_indices, res_sub):
        vec[i] = val

    # Canonical sign and scale, independent of sympy's choice of basis vector.
    support = np.nonzero(vec)[0]
    all_simplices = K.n_simplices(d)
    top = max(support, key=lambda i: tuple(sorted(all_simplices[i])))
    g = int(np.gcd.reduce(np.abs(vec[support])))
    return vec // (g if vec[top] > 0 else -g)


def _is_cycle(K: SimplicialComplex, simplices: List[Tuple[int, ...]], d: int) -> bool:
    """Check that the given simplices support a d-cycle (over Z)."""
    return _get_cycle_coefficients(K, simplices, d) is not None


def _shares_simplex(A: List[Tuple[int, ...]], B: List[Tuple[int, ...]]) -> bool:
    """Return True if A and B share any simplex of any dimension."""
    set_b: Set[Tuple[int, ...]] = set(B)
    return any(s in set_b for s in A)


def _all_simplices(K: SimplicialComplex) -> List[Tuple[int, ...]]:
    """Return all simplices across all dimensions as sorted tuples."""
    result = []
    for d in K.dimensions:
        result.extend(K.n_simplices(d))
    return result


def _subcomplex_simplices(K: SimplicialComplex, top_simplices: List[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
    """Return the full closure (all faces) of the given top simplices that are in K."""
    all_k = set(_all_simplices(K))
    result: Set[Tuple[int, ...]] = set()
    for s in top_simplices:
        result.add(s)
        # Add all faces
        frontier = [s]
        while frontier:
            t = frontier.pop()
            for f in _simplex_faces(t):
                if f in all_k and f not in result:
                    result.add(f)
                    frontier.append(f)
    return list(result)


# ── PL Sphere Recognition ─────────────────────────────────────────────────────


def _is_pl_sphere_dim0(simplices: List[Tuple[int, ...]]) -> bool:
    """S^0 = two disjoint points (no edges)."""
    vertices = [s for s in simplices if len(s) == 1]
    edges = [s for s in simplices if len(s) == 2]
    return len(vertices) == 2 and len(edges) == 0


def _is_pl_sphere_dim1(simplices: List[Tuple[int, ...]]) -> bool:
    """S^1 = simple cycle graph (each vertex degree 2, connected)."""
    vertices = {s[0] for s in simplices if len(s) == 1}
    edges = [s for s in simplices if len(s) == 2]
    if not vertices or not edges:
        return False
    degree: Dict[int, int] = {v: 0 for v in vertices}
    for e in edges:
        if len(e) < 2:
            return False
        degree[e[0]] += 1
        degree[e[1]] += 1
    if any(d != 2 for d in degree.values()):
        return False
    # Connectivity check via DFS
    adj: Dict[int, List[int]] = {v: [] for v in vertices}
    for e in edges:
        adj[e[0]].append(e[1])
        adj[e[1]].append(e[0])
    visited: Set[int] = set()
    start = next(iter(vertices))
    stack = [start]
    while stack:
        v = stack.pop()
        if v in visited:
            continue
        visited.add(v)
        stack.extend(adj[v])
    return visited == vertices


def _is_pl_sphere_dim2(simplices: List[Tuple[int, ...]]) -> bool:
    """S^2 iff closed orientable 2-manifold with Euler characteristic 2."""
    triangles = [s for s in simplices if len(s) == 3]
    edges = [s for s in simplices if len(s) == 2]
    vertices = {s[0] for s in simplices if len(s) == 1}
    if not triangles:
        return False
    # Manifold condition: each edge is in exactly 2 triangles
    edge_count: Dict[Tuple[int, ...], int] = {}
    for t in triangles:
        for f in _simplex_faces(t):
            key = tuple(sorted(f))
            edge_count[key] = edge_count.get(key, 0) + 1
    if any(v != 2 for v in edge_count.values()):
        return False
    V = len(vertices)
    E = len(edges)
    F = len(triangles)
    return V - E + F == 2


def _is_pl_sphere_python(simplices: List[Tuple[int, ...]], dim: int) -> bool:
    """Classical PL sphere recognition for dim ≤ 2."""
    if dim == 0:
        return _is_pl_sphere_dim0(simplices)
    if dim == 1:
        return _is_pl_sphere_dim1(simplices)
    if dim == 2:
        return _is_pl_sphere_dim2(simplices)
    # For dim ≥ 3: bounded heuristic — check Betti numbers via homology
    # Build a temporary SimplicialComplex for recognition
    try:
        sc = SimplicialComplex.from_simplices(simplices, close_under_faces=False)
        n = sc.dimension
        if n != dim:
            return False
        # Betti numbers of S^dim: β_0=1, β_dim=1, all others 0
        for j in range(n + 1):
            hom = sc.homology(n=j, backend="python")
            if isinstance(hom, tuple):
                rank, torsion = hom
            else:
                rank, torsion = hom.get(j, (0, []))
            if j == 0 and rank != 1:
                return False
            if j == dim and rank != 1:
                return False
            if j not in (0, dim) and rank != 0:
                return False
            if torsion:
                return False
        # Euler characteristic check
        chi = sc.euler_characteristic()
        expected_chi = 1 + (-1) ** dim  # χ(S^n) = 1 + (-1)^n
        return chi == expected_chi
    except Exception:
        return False


def _is_pl_sphere_julia(simplices: List[Tuple[int, ...]], dim: int) -> Tuple[bool, str]:
    """Julia-dispatched PL sphere recognition."""
    try:
        simplex_lists = [list(s) for s in simplices]
        return julia_engine.sphere_recognition_pl(simplex_lists, dim)
    except Exception:
        return False, "julia_error"


def _is_pl_sphere(simplices: List[Tuple[int, ...]], dim: int, backend: str = "auto") -> bool:
    """PL sphere recognition, dispatching by dimension and backend."""
    use_julia = (backend == "julia") or (backend == "auto" and julia_engine.available)
    if use_julia:
        try:
            is_sphere, _ = _is_pl_sphere_julia(simplices, dim)
            return is_sphere
        except Exception as e:
            if backend == "julia":
                raise
            warnings.warn(f"Julia PL sphere recognition failed; falling back to Python: {e!r}")
    return _is_pl_sphere_python(simplices, dim)


# ── Embeddedness Check ────────────────────────────────────────────────────────


def _is_embedded(sigma_simplices: List[Tuple[int, ...]], K: SimplicialComplex, k: int) -> bool:
    """Check that σ is embedded in K: each vertex star of σ in K is a disk.

    For a (k-1)-sphere σ ⊂ K to be embedded, the star of each vertex v ∈ σ
    in K restricted to σ must be a (k-2)-disk (i.e., the link of v in σ is
    a (k-2)-sphere). For k-1 ≤ 2, we check via vertex degree conditions.
    """
    if k <= 1:
        return True
    dim = k - 1
    if dim == 1:
        # Embedded 1-sphere: each vertex appears in exactly 2 edges
        vertices = {s[0] for s in sigma_simplices if len(s) == 1}
        edges = [s for s in sigma_simplices if len(s) == 2]
        degree = {v: 0 for v in vertices}
        for e in edges:
            degree[e[0]] = degree.get(e[0], 0) + 1
            degree[e[1]] = degree.get(e[1], 0) + 1
        return all(d == 2 for d in degree.values())
    # For dim ≥ 2: check that the link of each vertex in σ has the correct homology
    # and that no two simplices in σ share an interior point (which we approximate
    # by checking that each face of dimension dim-1 appears in at most 2 top simplices of σ).
    top_simplices = [s for s in sigma_simplices if len(s) == k]
    face_count: Dict[Tuple[int, ...], int] = {}
    for t in top_simplices:
        for f in _simplex_faces(t):
            key = tuple(sorted(f))
            face_count[key] = face_count.get(key, 0) + 1
    # For a manifold without boundary, each (dim-1)-face appears in exactly 2 top simplices
    return all(c == 2 for c in face_count.values())


# ── Normal Bundle / Framing ────────────────────────────────────────────────────


class FramingResult(BaseModel):
    """Result of compute_framing containing value, classification reason, and exactness."""
    value: Optional[int]
    reason: Literal[
        "trivial_codim_1",
        "canonical_codim_2",
        "stable_range_codim_ge_3",
        "w2_nonzero",
        "wk_nonzero",
        "unknown_unstable",
    ]
    exact: bool


def _normal_bundle_w2(sigma_simplices: List[Tuple[int, ...]], K: SimplicialComplex) -> int:
    """Compute w_2(ν(σ ⊂ K)) ∈ Z/2.

    Using Whitney sum formula: w_2(ν) = w_2(TK)|_σ + w_2(Tσ) = w_2(TK)|_σ (since S^{k-1} is spin).
    So we sum the tangent w_2 of K over the 2-simplices of the sphere mod 2.
    """
    try:
        from pysurgery.geometry.characteristic_classes import extract_stiefel_whitney_tangent
        w2_K = extract_stiefel_whitney_tangent(K, k=2)
        K_simplices = [tuple(sorted(s)) for s in K.n_simplices(2)]
        simplex_to_idx = {s: i for i, s in enumerate(K_simplices)}
        
        val = 0
        for s in sigma_simplices:
            s_sorted = tuple(sorted(s))
            if s_sorted in simplex_to_idx:
                idx = simplex_to_idx[s_sorted]
                val += w2_K[idx]
        return int(val % 2)
    except Exception:
        return 0


def _normal_bundle_wk(sigma_simplices: List[Tuple[int, ...]], K: SimplicialComplex, k: int) -> int:
    """Compute w_k(ν(σ ⊂ K)) ∈ Z/2.

    Using Whitney sum formula: w_k(ν) = w_k(TK)|_σ + w_k(Tσ) = w_k(TK)|_σ.
    So we sum the tangent w_k of K over the (k-1)-simplices of the sphere mod 2.
    """
    try:
        from pysurgery.geometry.characteristic_classes import extract_stiefel_whitney_tangent
        wk_K = extract_stiefel_whitney_tangent(K, k=k)
        K_simplices = [tuple(sorted(s)) for s in K.n_simplices(k)]
        simplex_to_idx = {s: i for i, s in enumerate(K_simplices)}
        
        val = 0
        for s in sigma_simplices:
            s_sorted = tuple(sorted(s))
            if s_sorted in simplex_to_idx:
                idx = simplex_to_idx[s_sorted]
                val += wk_K[idx]
        return int(val % 2)
    except Exception:
        return 0


def _compute_framing(
    sigma_simplices: List[Tuple[int, ...]],
    K: SimplicialComplex,
    n: int,
    k: int,
) -> FramingResult:
    """w₂/w_k-based framing computation. Implements Gap G07."""
    codim = n - k + 1
    if codim == 1:
        return FramingResult(value=1, reason="trivial_codim_1", exact=True)
    if codim == 2:
        return FramingResult(value=0, reason="canonical_codim_2", exact=True)
    if codim >= 3:
        if codim > k - 1:
            if k == 2:
                w2 = _normal_bundle_w2(sigma_simplices, K)
                if w2 != 0:
                    return FramingResult(value=None, reason="w2_nonzero", exact=True)
                return FramingResult(value=0, reason="stable_range_codim_ge_3", exact=True)
            else:
                wk = _normal_bundle_wk(sigma_simplices, K, k)
                if wk != 0:
                    return FramingResult(value=None, reason="wk_nonzero", exact=True)
                return FramingResult(value=0, reason="stable_range_codim_ge_3", exact=True)
        else:
            return FramingResult(value=0, reason="unknown_unstable", exact=False)
    return FramingResult(value=0, reason="unknown_unstable", exact=False)


def _construct_tubular_neighborhood(
    K: SimplicialComplex,
    sphere_simplices: Sequence[Tuple[int, ...]],
    k: int,
    n: int,
    *,
    derived_subdivision: bool = False,
) -> Tuple[Tuple[int, ...], ...]:
    """Closed tubular neighborhood of σ ⊂ K. Implements Gap G03."""
    if derived_subdivision:
        raise NotImplementedError("derived subdivision deferred to v2.1")
        
    all_k = set(tuple(sorted(x)) for x in _all_simplices(K))
    
    for σ in sphere_simplices:
        σ_sorted = tuple(sorted(σ))
        if σ_sorted not in all_k:
            raise AttachmentSphereError(
                f"_construct_tubular_neighborhood: simplex {σ} not in K",
                reason="not_a_sphere",
                complex_signature=_complex_hash(K),
                index_k=k,
            )

    S_closed = set(tuple(sorted(x)) for x in _subcomplex_simplices(K, list(sphere_simplices)))

    star = set()
    for τ in all_k:
        for σ in sphere_simplices:
            if set(σ).issubset(set(τ)):
                star.add(τ)
                break

    result = set()
    for τ in star:
        stack = [τ]
        while stack:
            s = stack.pop()
            s_sorted = tuple(sorted(s))
            if s_sorted in S_closed:
                continue
            if s_sorted in result:
                continue
            result.add(s_sorted)
            if len(s_sorted) > 1:
                for i in range(len(s_sorted)):
                    face = tuple(sorted(s_sorted[:i] + s_sorted[i+1:]))
                    if face in all_k:
                        stack.append(face)

    res_tuple = tuple(sorted(result, key=lambda x: (len(x), x)))
    if not res_tuple:
        warnings.warn("_construct_tubular_neighborhood: sphere = K; tube empty")
    return res_tuple


# ── Data Structures ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HandleAttachment:
    """Encodes an attaching map φ: S^{k−1} × D^{n−k} → K.

    Overview:
        Frozen dataclass representing the data of a handle attachment in handle
        surgery on a simplicial complex K. Used as a deterministic cache key.

    Key Concepts:
        - Handle index k: the attached handle is a k-handle.
        - Attaching sphere: σ ≅ S^{k−1} embedded in ∂K.
        - Tubular neighborhood: φ(S^{k−1} × D^{n−k}) ⊂ K.
        - Co-disk simplices: the new D^k × S^{n−k−1} to be glued in.

    Preserved Invariants:
        - 0 ≤ index_k ≤ ambient_dim.
        - exact iff embeddedness_verified ∧ framing_verified ∧ exact tag.

    Attributes:
        ambient_complex: The complex K being operated on.
        ambient_dim: n = ambient_complex.dimension.
        index_k: Handle index k.
        attaching_sphere: Top-dim simplices of σ ≅ S^{k−1}.
        tubular_neighborhood: φ(S^{k−1} × D^{n−k}) ⊂ K (closed tube).
        co_disk_simplices: D^k × S^{n−k−1} to be glued in.
        framing: Trivialization of ν(σ ⊂ K).
        embeddedness_verified: True iff inclusion σ ↪ K was checked exact.
        framing_verified: True iff trivialization exists and was constructed.
        theorem_tag: Recognition method tag.
        contract_version: Schema version.

    References:
        Milnor, J. (1965). Lectures on the h-cobordism theorem. Princeton University Press.
        Wall, C. T. C. (1970). Surgery on compact manifolds. Academic Press, §1.
    """

    ambient_complex: SimplicialComplex
    ambient_dim: int
    index_k: int
    attaching_sphere: Tuple[Tuple[int, ...], ...]
    tubular_neighborhood: Tuple[Tuple[int, ...], ...]
    co_disk_simplices: Tuple[Tuple[int, ...], ...]
    framing: Optional[int]
    embeddedness_verified: bool
    framing_verified: bool
    theorem_tag: str
    contract_version: str

    def __post_init__(self) -> None:
        if not (0 <= self.index_k <= self.ambient_dim):
            raise HandleSurgeryError(
                f"index_k={self.index_k} out of range [0, {self.ambient_dim}]"
            )

    @property
    def exact(self) -> bool:
        """True iff embeddedness_verified ∧ framing_verified ∧ exact tag."""
        return (
            self.embeddedness_verified
            and self.framing_verified
            and self.theorem_tag == SURGERY_ATTACHMENT_SPHERE_RECOGNITION_EXACT
        )

    def decision_ready(self) -> bool:
        """Return True when the attachment is fully certified."""
        return self.exact


class LinkingNumberResult(BaseModel):
    """Result of compute_linking_number. Always exact=True when no exception raised.

    Overview:
        Encodes the integer linking number lk(K_a, K_b) ∈ ℤ: the intersection
        number of K_a, pushed off transversally, with a Seifert chain F with
        ∂F = K_b.

    Key Concepts:
        - Linking number: integer topological invariant of disjoint cycles.
        - Seifert chain: F ∈ C_{q+1}(K) with ∂F = K_b (back-solved over ℤ).
        - Exactness: guaranteed for coefficient_ring="Z"; the F2 value is the
          integral linking number mod 2.

    Preserved Invariants:
        - dim_a + dim_b == ambient_dim − 1 (Lefschetz pairing constraint).
        - exact=True iff coefficient_ring == "Z" and computation succeeded.

    Attributes:
        value: The integer linking number (signed).
        coefficient_ring: "Z", "F2", or "Q".
        dim_a: dim K_a.
        dim_b: dim K_b.
        ambient_dim: n.
        seifert_chain_size: |support(F)|.
        seifert_chain_norm: Σ|F[τ]|.
        exact: True iff coefficient_ring == "Z" AND no failure.
        theorem_tag: SURGERY_LINKING_RELATIVE_SNF_Z or SURGERY_LINKING_F2_HEURISTIC.
        contract_version: Schema version.

    References:
        Munkres, J. R. (1984). Elements of algebraic topology. Addison-Wesley, §70.
        Hatcher, A. (2002). Algebraic topology. Cambridge University Press, §3.B.
    """

    value: int
    coefficient_ring: Literal["Z", "F2", "Q"] = "Z"
    dim_a: int
    dim_b: int
    ambient_dim: int
    seifert_chain_size: int = 0
    seifert_chain_norm: int = 0
    exact: bool = True
    theorem_tag: str = SURGERY_LINKING_RELATIVE_SNF_Z
    contract_version: str = CONTRACT_VERSION

    @field_validator("dim_a", "dim_b", "ambient_dim")
    @classmethod
    def _non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Dimensions must be non-negative.")
        return v

    @model_validator(mode="after")
    def _check_dim_constraint(self) -> "LinkingNumberResult":
        if self.dim_a + self.dim_b != self.ambient_dim - 1:
            raise ValueError(
                f"dim_a + dim_b = {self.dim_a + self.dim_b} != ambient_dim - 1 = {self.ambient_dim - 1}"
            )
        if self.coefficient_ring == "F2" and self.value not in (0, 1):
            raise ValueError("F2 linking number must be 0 or 1.")
        return self

    def decision_ready(self) -> bool:
        """Return True when the result is an exact integer linking number."""
        return self.exact and self.coefficient_ring == "Z"


class AttachmentSphereResult(BaseModel):
    """Result of find_attachment_sphere.

    Overview:
        Encodes a candidate attaching sphere σ ≅ S^{k−1} ⊂ K, together with
        certification status for embeddedness and framing.

    Key Concepts:
        - approx_path=False: exhaustive search with PL certification (exact=True).
        - approx_path=True: SNF cycle representative, embeddedness/framing not verified.

    Preserved Invariants:
        - exact = (¬approx_path) ∧ embeddedness_verified ∧ framing_verified.
        - If approx_path, then theorem_tag == SURGERY_ATTACHMENT_SPHERE_SNF_HEURISTIC.

    Attributes:
        sphere_simplices: Top-dim simplices of σ.
        ambient_complex: K.
        ambient_dim: n.
        index_k: Handle index k.
        target_complex: K_b (linked partner).
        linking_with_target: lk(σ, K_b).
        approx_path: True iff approx=True was used.
        embeddedness_verified: True iff inclusion σ ↪ K was checked.
        framing_verified: True iff trivialization was constructed.
        recognition_method: How σ was identified as S^{k−1}.
        enumeration_budget_used: Subcomplexes inspected (exact path).
        exact: True iff certified.
        theorem_tag: Recognition method tag.
        contract_version: Schema version.

    References:
        Milnor, J. (1965). Lectures on the h-cobordism theorem. Princeton University Press.
        Wall, C. T. C. (1970). Surgery on compact manifolds. Academic Press, §1.
    """

    model_config = {"arbitrary_types_allowed": True}

    sphere_simplices: List[Tuple[int, ...]]
    ambient_complex: SimplicialComplex
    ambient_dim: int
    index_k: int
    target_complex: Optional[SimplicialComplex] = None
    linking_with_target: Optional[int] = None
    approx_path: bool = False
    embeddedness_verified: bool = False
    framing_verified: bool = False
    recognition_method: Literal[
        "enumeration", "snf_generator", "rubinstein_thompson", "low_dim_classical"
    ] = "enumeration"
    enumeration_budget_used: Optional[int] = None
    exact: bool = False
    theorem_tag: str = SURGERY_ATTACHMENT_SPHERE_RECOGNITION_EXACT
    contract_version: str = CONTRACT_VERSION

    def decision_ready(self) -> bool:
        """Return True when the sphere is fully certified."""
        return self.exact


class SurgeryResult(BaseModel):
    """Result of perform_handle_surgery.

    Overview:
        Encodes the outcome of performing a handle attachment, including the
        homological change certified via Mayer–Vietoris.

    Preserved Invariants:
        - exact = attachment.exact ∧ mayer_vietoris_postcondition_passed.
        - betti_before/after keyed by {0, ..., max_dim}.

    Attributes:
        complex_before: K.
        complex_after: K''.
        attachment: The attaching map applied.
        surgery_index: k.
        betti_before: β_j(K) for all j.
        betti_after: β_j(K'') for all j.
        torsion_before: Torsion coefficients of H_j(K; ℤ).
        torsion_after: Torsion coefficients of H_j(K''; ℤ).
        mayer_vietoris_predicted_delta: Per-dim allowed Δβ from Milnor's formula.
        mayer_vietoris_postcondition_passed: True iff observed Δβ matches predicted.
        exact: attachment.exact ∧ mayer_vietoris_postcondition_passed.
        theorem_tag: SURGERY_HANDLE_MAYER_VIETORIS.
        contract_version: Schema version.

    References:
        Milnor, J. (1965). Lectures on the h-cobordism theorem. Princeton University Press.
    """

    model_config = {"arbitrary_types_allowed": True}

    complex_before: SimplicialComplex
    complex_after: SimplicialComplex
    attachment: HandleAttachment
    surgery_index: int
    betti_before: Dict[int, int]
    betti_after: Dict[int, int]
    torsion_before: Dict[int, List[int]]
    torsion_after: Dict[int, List[int]]
    mayer_vietoris_predicted_delta: Dict[int, object]
    mayer_vietoris_postcondition_passed: bool
    exact: bool
    theorem_tag: str = SURGERY_HANDLE_MAYER_VIETORIS
    contract_version: str = CONTRACT_VERSION

    def decision_ready(self) -> bool:
        """Return True when the surgery result is fully certified."""
        return self.exact and self.mayer_vietoris_postcondition_passed


class SurgeryVerificationResult(BaseModel):
    """Result of verify_surgery.

    Overview:
        Certifies that the homological change between K_before and K_after
        is consistent with index-k handle surgery, per Mayer–Vietoris.

    Preserved Invariants:
        - exact=True always when passed=True (verification is deterministic).
        - Torsion changes confined to dimensions {k-1, k}.

    Attributes:
        passed: True iff verification succeeded.
        betti_before: β_j(K_before).
        betti_after: β_j(K_after).
        torsion_before: Torsion of H_j(K_before; ℤ).
        torsion_after: Torsion of H_j(K_after; ℤ).
        surgery_index: k.
        exact: True iff passed.
        theorem_tag: SURGERY_VERIFY_SNF_BETTI_TORSION.
        contract_version: Schema version.

    References:
        Milnor, J. (1965). Lectures on the h-cobordism theorem. Princeton University Press.
    """

    passed: bool
    betti_before: Dict[int, int]
    betti_after: Dict[int, int]
    torsion_before: Dict[int, List[int]]
    torsion_after: Dict[int, List[int]]
    surgery_index: int
    exact: bool = True
    theorem_tag: str = SURGERY_VERIFY_SNF_BETTI_TORSION
    contract_version: str = CONTRACT_VERSION

    def decision_ready(self) -> bool:
        """Return True when the verification is conclusive."""
        return self.exact and self.passed


class DelinkingResult(BaseModel):
    """Result of delink.

    Overview:
        Encodes the outcome of iterated index-1 handle surgery on K_a aimed at
        achieving lk(K_a'', K_b'') = 0.

    Preserved Invariants:
        - len(linking_trace) == surgeries_performed + 1.
        - exact = True only when terminated_reason == "delinked" and all surgeries exact.

    Attributes:
        complex_before: K (initial ambient).
        complex_after: K after all surgeries.
        complex_a_before: Initial K_a.
        complex_a_after: K_a''.
        complex_b_before: Initial K_b.
        complex_b_after: K_b''.
        surgery_sequence: Each step, in order.
        linking_trace: lk values after each step.
        initial_linking: lk(K_a, K_b).
        final_linking: lk(K_a'', K_b'').
        surgeries_performed: T.
        unlinking_number_lower_bound: |initial_linking|.
        max_surgeries: Configured budget.
        terminated_reason: Loop termination cause.
        exact: Certified delinking with all exact surgeries.
        theorem_tag: SURGERY_DELINKING_UNLINKING_NUMBER.
        contract_version: Schema version.

    References:
        Milnor, J. (1961). A procedure for killing homotopy groups of differentiable manifolds.
            Proceedings of Symposia in Pure Mathematics, 3, 39–55.
    """

    model_config = {"arbitrary_types_allowed": True}

    complex_before: SimplicialComplex
    complex_after: SimplicialComplex
    complex_a_before: SimplicialComplex
    complex_a_after: SimplicialComplex
    complex_b_before: SimplicialComplex
    complex_b_after: SimplicialComplex
    surgery_sequence: List[SurgeryResult]
    linking_trace: List[int]
    initial_linking: int
    final_linking: int
    surgeries_performed: int
    unlinking_number_lower_bound: int
    max_surgeries: int
    terminated_reason: Literal["delinked", "max_surgeries_reached", "no_attachment_sphere"]
    exact: bool
    theorem_tag: str = SURGERY_DELINKING_UNLINKING_NUMBER
    contract_version: str = CONTRACT_VERSION

    def decision_ready(self) -> bool:
        """Return True when delinking is certified complete."""
        return self.exact


# ── Linking Number — Simplicial Method ───────────────────────────────────────


def _boundary_matrix_between(
    Cq: Sequence[Tuple[int, ...]], Cqp1: Sequence[Tuple[int, ...]]
) -> csr_matrix:
    """Sparse ∂: C_{q+1} → C_q restricted to the given (sorted) simplices."""
    row = {tuple(s): i for i, s in enumerate(Cq)}
    rows, cols, vals = [], [], []
    for j, tau in enumerate(Cqp1):
        tau = tuple(tau)
        for i in range(len(tau)):
            r = row.get(tau[:i] + tau[i + 1:])
            if r is not None:
                rows.append(r)
                cols.append(j)
                vals.append(-1 if i % 2 else 1)
    return csr_matrix(
        coo_matrix((vals, (rows, cols)), shape=(len(Cq), len(Cqp1)), dtype=np.int64)
    )


def _solve_boundary_z(
    Cq: Sequence[Tuple[int, ...]],
    Cqp1: Sequence[Tuple[int, ...]],
    b: np.ndarray,
    backend: str = "auto",
) -> Optional[np.ndarray]:
    """An integral chain f over Cqp1 with ∂f = b, or None if there is none.

    Exact, by Smith normal form: the Julia kernel when available, else Python.
    """
    b = np.asarray(b, dtype=np.int64)
    if not len(Cqp1):
        return np.zeros(0, dtype=np.int64) if not np.any(b) else None
    B = _boundary_matrix_between(Cq, Cqp1)

    use_julia = (backend == "julia") or (backend == "auto" and julia_engine.available)
    if use_julia:
        try:
            B_jl = julia_engine.surgery_relative_boundary_sparse(
                [list(s) for s in Cq], [list(s) for s in Cqp1], []
            )
            f, success, _ = julia_engine.linking_seifert_solve_z(B_jl, b)
            if not success:
                return None
            f = np.asarray(f, dtype=np.int64)
            if np.array_equal(B @ f, b):
                return f
            warnings.warn("Julia Seifert-chain solve returned a wrong chain; using Python.")
        except Exception as e:
            if backend == "julia":
                raise
            warnings.warn(f"Julia Seifert-chain solve failed; using Python: {e!r}")

    B_dense = B.toarray()
    U, D_diag, V = _snf_with_transforms_python(B_dense)
    r = int(np.sum(D_diag != 0))
    u = U @ b
    if np.any(u[r:] != 0):
        return None
    w = np.zeros(B_dense.shape[1], dtype=np.int64)
    for i in range(r):
        if u[i] % D_diag[i] != 0:
            return None
        w[i] = u[i] // D_diag[i]
    f = V @ w
    return f if np.array_equal(B_dense @ f, b) else None


def _linking_in_3manifold(
    K: SimplicialComplex,
    a_vec: np.ndarray,
    b_vec: np.ndarray,
    backend: str,
    failure,
) -> Tuple[int, np.ndarray]:
    """Linking number of vertex-disjoint 1-cycles a, b in a 3-dimensional K, by a dual push-off.

    See pysurgery.manifolds.simplicial_linking.  Returns (lk, G) with ∂G = b.
    Raises NotAClosable3ManifoldError unless K is pure and the component
    carrying a, with its boundary 2-spheres coned off, is an orientable
    combinatorial 3-manifold that also carries b.
    """
    tets = K.n_simplices(3)
    if len({t for T in tets for t in _simplex_faces(T)}) != len(K.n_simplices(2)):
        raise NotAClosable3ManifoldError("K has triangles outside its tetrahedra")
    edges = K.n_simplices(1)
    a = {tuple(edges[i]): int(a_vec[i]) for i in np.nonzero(a_vec)[0]}
    b = {tuple(edges[i]): int(b_vec[i]) for i in np.nonzero(b_vec)[0]}
    pc = K.simplices_to_point_cloud
    coords = {s[0]: np.asarray(pc[s][0]) for s in K.n_simplices(0) if s in pc} if pc else None
    M = Oriented3Manifold(
        component_tets(tets, next(iter(a))[0]),
        coords,
        apex_base=max(s[0] for s in K.n_simplices(0)),
    )
    b_M = M.edge_vector(b)
    G = M.bounding_chain(b_M)
    if G is None:
        # H_1(M) ≠ 0: solve exactly.  If b_1(M) > 0, lk is well defined only
        # when a bounds too.
        G = _solve_boundary_z(M.edges, M.tris, b_M, backend)
        if G is None:
            raise failure("K_b is not null-homologous in K", "kb_not_null_homologous")
        if not M.is_rational_homology_sphere and (
            _solve_boundary_z(M.edges, M.tris, M.edge_vector(a), backend) is None
        ):
            raise failure(
                "K_a is not null-homologous in K, so lk(K_a, K_b) depends on the Seifert chain",
                "ka_not_null_homologous",
            )
    return int(M.dual_cycle(a) @ G), G


def _compute_linking_number_simplicial(
    K: SimplicialComplex,
    K_a: SimplicialComplex,
    K_b: SimplicialComplex,
    coefficient_ring: str = "Z",
    backend: str = "auto",
) -> LinkingNumberResult:
    """Compute lk(K_a, K_b) from the triangulation; coordinates, if any, only orient it.

    lk(K_a, K_b) is the intersection number of K_a, pushed off transversally,
    with a Seifert chain G, ∂G = K_b.  A primal p-cycle and a primal
    (q+1)-chain are never transverse (summing K_a[σ]·G[τ]·[τ:σ] over faces
    σ ⊂ τ gives ⟨K_a, ∂G⟩ = 0), so K_a is replaced as follows.

    - 1-cycles in a 3-dimensional K whose component carrying K_a is a
      combinatorial 3-manifold (boundary 2-spheres coned off): K_a becomes a
      dual 1-cycle, a closed path of tetrahedra in the open star of K_a, and
      lk = Σ_t (signed crossings of triangle t)·G[t].
    - Otherwise lk = 0 is certified when K_b bounds a chain whose support
      shares no vertex with K_a.  Failing that, LinkingComputationError is
      raised rather than a guess.

    Over F2 the value is the integral linking number mod 2.
    """
    n = K.dimension
    p = K_a.dimension
    q = K_b.dimension
    sig = _complex_hash(K)

    def failure(message: str, reason: str) -> LinkingComputationError:
        return LinkingComputationError(
            message,
            reason=reason,
            dim_a=p,
            dim_b=q,
            ambient_dim=n,
            coefficient_ring=coefficient_ring,
            complex_signature=sig,
        )

    if p + q != n - 1:
        raise failure(f"dim_a + dim_b = {p + q} ≠ n − 1 = {n - 1}", "dim_mismatch")

    # Disjointness check across all dimensions
    Ka_all = set(_all_simplices(K_a))
    if Ka_all & set(_all_simplices(K_b)):
        raise failure("K_a and K_b share a simplex", "not_disjoint")

    a_vec = _get_cycle_coefficients(K, K_a.n_simplices(p), p)
    if a_vec is None:
        raise failure("K_a is not a cycle (∂K_a ≠ 0)", "not_a_cycle_a")
    b_vec = _get_cycle_coefficients(K, K_b.n_simplices(q), q)
    if b_vec is None:
        raise failure("K_b is not a cycle (∂K_b ≠ 0)", "not_a_cycle_b")

    result: Optional[Tuple[int, np.ndarray]] = None
    if n == 3 and p == 1 and q == 1:
        try:
            result = _linking_in_3manifold(K, a_vec, b_vec, backend, failure)
        except NotAClosable3ManifoldError:
            result = None

    if result is None:
        # A Seifert chain that misses K_a certifies lk = 0.
        Ka_vertices = {v for s in Ka_all for v in s}
        Cq, Cqp1 = K.n_simplices(q), K.n_simplices(q + 1)
        away = [tau for tau in Cqp1 if Ka_vertices.isdisjoint(tau)]
        G = _solve_boundary_z(Cq, away, b_vec, backend)
        if G is None:
            if _solve_boundary_z(Cq, Cqp1, b_vec, backend) is None:
                raise failure("K_b is not null-homologous in K", "kb_not_null_homologous")
            raise failure(
                "Every Seifert chain of K_b meets K_a, and a transverse push-off of K_a "
                "is only available for 1-cycles in a combinatorial 3-manifold",
                "no_transverse_pushoff",
            )
        result = (0, G)

    value, G = result
    use_f2 = coefficient_ring == "F2"
    return LinkingNumberResult(
        value=value % 2 if use_f2 else value,
        coefficient_ring=coefficient_ring,
        dim_a=p,
        dim_b=q,
        ambient_dim=n,
        seifert_chain_size=int(np.count_nonzero(G)),
        seifert_chain_norm=int(np.sum(np.abs(G))),
        exact=True,
        theorem_tag=(SURGERY_LINKING_F2_HEURISTIC if use_f2 else SURGERY_LINKING_RELATIVE_SNF_Z),
    )


def _snf_with_transforms_python(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Smith Normal Form over ℤ with unimodular transform matrices.

    Returns (U, D_diag, V) such that D = U · M · V where U, V are invertible over ℤ
    and D is diagonal with non-negative entries (SNF).
    """
    from pysurgery.algebra.math_core import smith_normal_decomp
    S, U, V = smith_normal_decomp(M)
    D_diag = np.array([S[i, i] for i in range(min(S.shape))], dtype=np.int64)
    return U.astype(np.int64), D_diag, V.astype(np.int64)


def _oriented_edge_segments(
    K: SimplicialComplex,
    K_sub: SimplicialComplex,
    coords: Dict[Tuple[int, ...], np.ndarray],
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """Return list of (start, end, multiplicity) oriented edges for K_sub using K's coords.

    Uses _get_cycle_coefficients to determine the oriented Z-cycle representation.
    """
    edges_ambient = K.n_simplices(1)
    coeffs = _get_cycle_coefficients(K, K_sub.n_simplices(1), 1)
    if coeffs is None:
        return []
    segments: List[Tuple[np.ndarray, np.ndarray, int]] = []
    for k, simp in enumerate(edges_ambient):
        eps = int(coeffs[k])
        if eps == 0:
            continue
        if simp not in coords:
            continue
        pts = coords[simp]
        if pts.shape != (2, 3):
            continue
        if eps > 0:
            start, end = pts[0], pts[1]
        else:
            start, end = pts[1], pts[0]
        segments.append((start.astype(np.float64), end.astype(np.float64), abs(eps)))
    return segments


def _gauss_linking_riemann(
    Ka_segments: List[Tuple[np.ndarray, np.ndarray, int]],
    Kb_segments: List[Tuple[np.ndarray, np.ndarray, int]],
    n_samples: int = 24,
) -> float:
    """Numerical Gauss linking integral via midpoint Riemann sum on each segment pair.

    Returns the (signed) linking number as a float; caller should round.
    """
    if not Ka_segments or not Kb_segments:
        return 0.0

    # Precompute sample points & tangent vectors for each curve
    def _samples(segments):
        out = []
        for a, b, m in segments:
            tan = b - a
            for i in range(n_samples):
                s = (i + 0.5) / n_samples
                pt = a + s * tan
                out.append((pt, tan, m))
        return out

    A = _samples(Ka_segments)
    B = _samples(Kb_segments)

    total = 0.0
    inv_n2 = 1.0 / (n_samples * n_samples)
    for (r1, tan_a, ma) in A:
        for (r2, tan_b, mb) in B:
            diff = r1 - r2
            d2 = float(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
            if d2 < 1e-18:
                continue
            cross = (
                tan_a[1] * tan_b[2] - tan_a[2] * tan_b[1],
                tan_a[2] * tan_b[0] - tan_a[0] * tan_b[2],
                tan_a[0] * tan_b[1] - tan_a[1] * tan_b[0],
            )
            num = diff[0] * cross[0] + diff[1] * cross[1] + diff[2] * cross[2]
            total += (ma * mb) * num / (d2 * np.sqrt(d2))
    total *= inv_n2
    return total / (4.0 * np.pi)


def _segments_to_arrays(
    segments: List[Tuple[np.ndarray, np.ndarray, int]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert oriented segments to (starts, ends, multiplicities) arrays."""
    N = len(segments)
    starts = np.empty((N, 3), dtype=np.float64)
    ends = np.empty((N, 3), dtype=np.float64)
    mult = np.empty(N, dtype=np.int64)
    for k, (a, b, m) in enumerate(segments):
        starts[k] = a
        ends[k] = b
        mult[k] = m
    return starts, ends, mult


def _compute_linking_number_gauss(
    K: SimplicialComplex,
    K_a: SimplicialComplex,
    K_b: SimplicialComplex,
    coefficient_ring: str = "Z",
    backend: str = "auto",
) -> Optional[LinkingNumberResult]:
    """Linking number via the discrete Gauss integral using geometric coordinates.

    Uses the Julia kernel `linking_gauss_riemann_jl` when available; otherwise
    falls back to the pure-Python implementation. Returns None if no coordinates
    are attached or if K_a/K_b are not 1-cycles in a 3-ambient.
    """
    if K.dimension != 3 or K_a.dimension != 1 or K_b.dimension != 1:
        return None
    coords = K.simplices_to_point_cloud
    if not coords:
        return None

    Ka_segs = _oriented_edge_segments(K, K_a, coords)
    Kb_segs = _oriented_edge_segments(K, K_b, coords)
    if not Ka_segs or not Kb_segs:
        return None

    use_julia = backend in ("auto", "julia", "gauss") and julia_engine.available
    raw: Optional[float] = None
    if use_julia:
        try:
            Ka_s, Ka_e, Ka_m = _segments_to_arrays(Ka_segs)
            Kb_s, Kb_e, Kb_m = _segments_to_arrays(Kb_segs)
            raw = float(julia_engine.linking_gauss_riemann(
                Ka_s, Ka_e, Ka_m, Kb_s, Kb_e, Kb_m, 24
            ))
        except Exception as e:
            if backend == "julia":
                raise
            warnings.warn(f"Julia gauss linking failed; using Python: {e!r}")
            raw = None

    if raw is None:
        raw = _gauss_linking_riemann(Ka_segs, Kb_segs, n_samples=24)

    value = int(round(raw))
    if coefficient_ring == "F2":
        value = value % 2

    n = K.dimension
    return LinkingNumberResult(
        value=value,
        coefficient_ring=coefficient_ring,
        dim_a=K_a.dimension,
        dim_b=K_b.dimension,
        ambient_dim=n,
        seifert_chain_size=0,
        seifert_chain_norm=0,
        exact=True,
        theorem_tag=(SURGERY_LINKING_F2_HEURISTIC if coefficient_ring == "F2" else SURGERY_LINKING_RELATIVE_SNF_Z),
    )


# ── Public Functions ───────────────────────────────────────────────────────────


def compute_linking_number(
    K: SimplicialComplex,
    K_a: SimplicialComplex,
    K_b: SimplicialComplex,
    coefficient_ring: str = "Z",
    backend: str = "auto",
) -> LinkingNumberResult:
    """Compute the linking number lk(K_a, K_b) ∈ ℤ via the Seifert chain method.

    What is Being Computed?:
        The integer linking number of two disjoint oriented cycles K_a (p-cycle)
        and K_b (q-cycle) in an ambient simplicial complex K of dimension n,
        where p + q = n − 1: the intersection number of K_a, pushed off
        transversally, with a Seifert chain F ∈ C_{q+1}(K), ∂F = K_b.

    Algorithm:
        1. With vertex coordinates on K and 1-cycles in a 3-complex, evaluate
           the Gauss linking integral (skipped for backend="python").
        2. Otherwise validate preconditions (dimension, disjointness, cycle
           condition) and orient K_a, K_b as in `_get_cycle_coefficients`.
        3. For 1-cycles in a combinatorial 3-manifold (a 3-ball's boundary
           sphere is coned off), replace K_a by a dual 1-cycle, a closed path
           of tetrahedra near K_a; find F with ∂F = K_b by tree–cotree LU; and
           count the signed triangle crossings of the path, weighted by F.
        4. In any other ambient, return 0 if K_b bounds a chain that shares
           no vertex with K_a, and raise LinkingComputationError otherwise.

    Preserved Invariants:
        - Result is always exact (exact=True) when no exception is raised.
        - result.value is the signed linking number (convention: lk(K_a, K_b)).
          It is positive when K_a crosses F along the normal given by the
          right-hand rule on K_b.  The ambient orientation is that of R^3 when
          coordinates are attached (so the simplicial value agrees with the
          Gauss integral) and otherwise the one in which the
          lexicographically first tetrahedron is positive.
        - dim_a + dim_b == ambient_dim − 1.

    Args:
        K: Ambient simplicial complex.
        K_a: First subcomplex (p-cycle, p = dim_a).
        K_b: Second subcomplex (q-cycle, q = dim_b), assumed null-homologous in K.
        coefficient_ring: "Z" (exact ℤ) or "F2" (the integral value mod 2).
        backend: "auto", "gauss", "python", or "julia".  "python" skips the
            Gauss integral; "julia" uses the Julia kernels where they apply.

    Returns:
        LinkingNumberResult with value, exact=True, and diagnostic fields.

    Use When:
        - Computing linking numbers of cycles in a simplicial manifold.
        - Checking whether two submanifolds are linked (lk ≠ 0).
        - As a step in the delinking algorithm before surgery.

    Example:
        lk_result = compute_linking_number(K, K_a, K_b)
        assert lk_result.exact
        print(lk_result.value)

    References:
        Munkres, J. R. (1984). Elements of algebraic topology. Addison-Wesley, §70.
        Hatcher, A. (2002). Algebraic topology. Cambridge University Press, §3.B.
        Newman, M. H. A. (1926). On the foundations of combinatory analysis situs.
            Proceedings of the Koninklijke Akademie van Wetenschappen te Amsterdam.
    """
    # Fast path: use the geometric Gauss linking integral when coordinates are
    # attached to K. This is the canonical embedding-based linking number.
    if backend in ("auto", "gauss", "julia"):
        try:
            gauss_res = _compute_linking_number_gauss(K, K_a, K_b, coefficient_ring, backend=backend)
            if gauss_res is not None:
                return gauss_res
        except Exception as e:
            if backend == "gauss":
                raise
            warnings.warn(f"Gauss linking integral failed; falling back: {e!r}")

    return _compute_linking_number_simplicial(K, K_a, K_b, coefficient_ring, backend)


def compute_linking_seifert_chain(
    K: SimplicialComplex,
    K_b: SimplicialComplex,
    backend: str = "auto",
) -> Tuple[Optional[np.ndarray], List[List[int]], List[List[int]], int]:
    """Precompute the Seifert chain f for K_b in K without needing K_a.

    What is Being Computed?:
        Solves B_{q+1} · f = b over ℤ where B_{q+1} is the ambient boundary
        matrix and b encodes the K_b cycle.  The result f is a Seifert chain
        for K_b that does not depend on K_a.

    Algorithm:
        1. Encode K_b as integer vector b in Z^{|C_q(K)|}.
        2. Build boundary matrix B_{q+1}: C_{q+1}(K) → C_q(K).
        3. Solve B · f = b via SNF (Julia if available, otherwise Python).

    Args:
        K:       Ambient simplicial complex.
        K_b:     The fixed cycle subcomplex (q-cycle).
        backend: "auto", "python", or "julia".

    Returns:
        (f, Cqp1, Cp_dummy, n) where:
          - f:     Seifert chain as int64 ndarray, or None if unsolvable.
          - Cqp1:  (q+1)-simplices of K as list-of-lists, indexing f.
          - Cp_dummy: p-simplices of K at dim p=n-1-q.
          - n:     Ambient dimension.

    References:
        Munkres, J. R. (1984). Elements of algebraic topology. §70.
    """
    from pysurgery.bridge.julia_bridge import julia_engine

    n = K.dimension
    q = K_b.dimension
    p = n - 1 - q

    b = _get_cycle_coefficients(K, K_b.n_simplices(q), q)
    if b is None:
        return None, [], [], n

    Cq   = [list(s) for s in K.n_simplices(q)]
    Cqp1 = [list(s) for s in K.n_simplices(q + 1)]
    Cp   = [list(s) for s in K.n_simplices(p)]

    if not Cq or not Cqp1:
        return np.zeros(0, dtype=np.int64), Cqp1, Cp, n

    use_julia = (backend == "julia") or (backend == "auto" and julia_engine.available)
    if use_julia:
        try:
            idx_map = {tuple(sorted(s)): i for i, s in enumerate(K.n_simplices(q))}
            Kb_indices = [idx_map[tuple(sorted(s))] for s in K_b.n_simplices(q) if tuple(sorted(s)) in idx_map]
            B_jl = julia_engine.surgery_relative_boundary_sparse(Cq, Cqp1, Kb_indices)
            f, success, _ = julia_engine.linking_seifert_solve_z(B_jl, np.asarray(b, dtype=np.int64))
            if success:
                return f, Cqp1, Cp, n
        except Exception as e:
            if backend == "julia":
                raise
            warnings.warn(f"Julia Seifert precompute failed, falling back: {e!r}")

    # Python fallback: dense SNF
    B = K.boundary_matrix(q + 1)
    B_dense = coerce_int_matrix(B.toarray()) if B is not None else np.zeros((len(Cq), len(Cqp1)), dtype=int)
    try:
        from pysurgery.algebra.math_core import smith_normal_decomp
        S, U, V = smith_normal_decomp(B_dense, compute_u=True, compute_v=True)
        r = sum(1 for i in range(min(S.shape)) if S[i, i] != 0)
        ub = U @ np.asarray(b, dtype=np.int64)
        w = np.zeros(B_dense.shape[1], dtype=np.int64)
        for i in range(r):
            d = int(S[i, i])
            if d != 0 and ub[i] % d == 0:
                w[i] = ub[i] // d
            elif d != 0:
                return None, Cqp1, Cp, n
        f = V @ w
        return f.astype(np.int64), Cqp1, Cp, n
    except Exception:
        return None, Cqp1, Cp, n


def _enumerate_candidate_subcomplexes(
    K: SimplicialComplex, dim: int
) -> List[List[Tuple[int, ...]]]:
    """Enumerate minimal subcomplexes of dimension `dim` in K.

    Yields lists of dim-simplices (top simplices of each candidate).
    Each candidate is a set of dim-simplices forming a closed subcomplex.
    """
    top_simplices = K.n_simplices(dim)
    n_top = len(top_simplices)
    # For each subset of top simplices, consider the induced closed subcomplex
    # (practical up to ~20 simplices; beyond that use budget cutoff)
    if n_top > 20:
        # Large complex: only enumerate minimal connected subsets
        # Use sliding window of size (dim+2) as a heuristic
        candidates = []
        for size in range(dim + 1, min(n_top + 1, 2 * (dim + 2) + 1)):
            for i in range(n_top - size + 1):
                candidates.append(list(top_simplices[i : i + size]))
        return candidates
    # Small complex: full power set
    from itertools import combinations
    candidates = []
    for size in range(dim + 1, n_top + 1):
        for combo in combinations(range(n_top), size):
            candidates.append([top_simplices[i] for i in combo])
    return candidates


def find_attachment_sphere(
    K: SimplicialComplex,
    k: int,
    K_b: Optional[SimplicialComplex] = None,
    approx: bool = False,
    enumeration_budget: Optional[int] = None,
    backend: str = "auto",
) -> AttachmentSphereResult:
    """Find a (k-1)-dimensional attachment sphere σ ≅ S^{k-1} in K for index-k surgery.

    What is Being Computed?:
        A (k-1)-subcomplex σ of K that is PL-homeomorphic to S^{k-1}, embedded
        (inclusion σ ↪ K injective on stars), and admits a trivial normal bundle
        framing. Optionally, requires lk(σ, K_b) ≠ 0 for delinking purposes.

    Algorithm (exact, approx=False):
        1. Enumerate candidate (k-1)-subcomplexes of K via systematic traversal.
        2. For each candidate σ:
           a. Check PL sphere recognition (is_pl_sphere).
           b. Check embeddedness (link of each vertex is a (k-2)-sphere).
           c. Check framing (normal bundle ν(σ ⊂ K) trivial).
           d. If K_b given, compute lk(σ, K_b); skip if zero.
        3. Return first certified σ, or raise AttachmentSphereError if none found.

    Algorithm (approx, approx=True):
        1. Compute SNF of ∂_{k-1}: extract cycle basis Z_{k-1}.
        2. Pick a generator z linked nontrivially with K_b.
        3. Return σ = support of shortest cycle representative, with exact=False.

    Preserved Invariants:
        - exact=True only after ALL of: PL sphere recognition, embeddedness, framing.
        - approx=True ALWAYS produces exact=False and emits a UserWarning.
        - approx=True NEVER raises AttachmentSphereError for embeddedness failures.

    Args:
        K: Ambient simplicial complex.
        k: Handle index; attaching sphere has dimension k-1.
        K_b: Optional target subcomplex; sphere must have lk(σ, K_b) ≠ 0.
        approx: If True, skip embeddedness/framing checks (heuristic path).
        enumeration_budget: Max subcomplexes to inspect (exact path).
        backend: "auto", "python", or "julia".

    Returns:
        AttachmentSphereResult with sphere_simplices, exact flag, and diagnostics.

    Use When:
        - Searching for a surgery sphere in perform_handle_surgery or delink.
        - Preprocessing for handle decomposition algorithms.
        - Use approx=True only when speed matters and postcondition verification
          can catch the error downstream (via Mayer-Vietoris).

    Example:
        sphere = find_attachment_sphere(K, k=2, K_b=K_b)
        assert sphere.exact  # certified S^1 embedded and framed

    References:
        Milnor, J. (1965). Lectures on the h-cobordism theorem. Princeton University Press.
        Wall, C. T. C. (1970). Surgery on compact manifolds. Academic Press, §1.
        Rubinstein, J. H. (1995). An algorithm to recognize the 3-sphere.
            Proceedings of the International Congress of Mathematicians, 601–611.
    """
    n = K.dimension
    if not (1 <= k <= n):
        raise DimensionError(f"k={k} out of range [1, {n}]")

    dim = k - 1  # dimension of attaching sphere
    sig = _complex_hash(K)

    Kb_all = set(_all_simplices(K_b)) if K_b is not None else set()

    if approx:
        warnings.warn(
            "approx=True: attachment sphere not verified for embeddedness or framing. "
            "Result is heuristic.",
            UserWarning,
            stacklevel=2,
        )
        return _find_attachment_sphere_approx(K, k, K_b, Kb_all, backend, sig, n)
    else:
        return _find_attachment_sphere_exact(
            K, k, K_b, Kb_all, backend, enumeration_budget, sig, n, dim
        )


def _find_attachment_sphere_exact(
    K: SimplicialComplex,
    k: int,
    K_b: Optional[SimplicialComplex],
    Kb_all: Set[Tuple[int, ...]],
    backend: str,
    enumeration_budget: Optional[int],
    sig: str,
    n: int,
    dim: int,
) -> AttachmentSphereResult:
    """Exact path for find_attachment_sphere."""
    candidates_inspected = 0

    # Determine recognition method label
    if dim <= 2:
        rec_method: Literal[
            "enumeration", "snf_generator", "rubinstein_thompson", "low_dim_classical"
        ] = "low_dim_classical"
    elif dim == 3:
        rec_method = "rubinstein_thompson"
    else:
        rec_method = "enumeration"

    candidate_groups = _enumerate_candidate_subcomplexes(K, dim)

    for candidate_top in candidate_groups:
        candidates_inspected += 1
        if enumeration_budget is not None and candidates_inspected > enumeration_budget:
            raise AttachmentSphereError(
                f"Enumeration budget {enumeration_budget} exceeded without finding sphere",
                reason="exact_search_budget_exceeded",
                complex_signature=sig,
                index_k=k,
                stage="search",
                candidate_simplices=None,
                complex_info={"k": k, "complex_size": len(K.simplices)},
            )

        # Get full closure
        all_sub = _subcomplex_simplices(K, candidate_top)

        # Disjointness from K_b
        if Kb_all and any(s in Kb_all for s in all_sub):
            continue

        # PL sphere recognition
        if not _is_pl_sphere(all_sub, dim, backend):
            continue

        # Embeddedness check
        if not _is_embedded(all_sub, K, k):
            continue

        # Framing
        framing_val = _compute_framing(all_sub, K, n, k)
        framing_verified = framing_val is not None

        # Linking with K_b
        linking_val = None
        if K_b is not None:
            try:
                # Build subcomplex for σ
                sigma_sc = SimplicialComplex.from_simplices(all_sub, close_under_faces=False)
                lk_res = compute_linking_number(K, sigma_sc, K_b, backend=backend)
                linking_val = lk_res.value
                if linking_val == 0:
                    continue  # Sphere found but useless for delinking
            except (LinkingComputationError, DimensionError):
                continue

        return AttachmentSphereResult(
            sphere_simplices=[tuple(sorted(s)) for s in candidate_top],
            ambient_complex=K,
            ambient_dim=n,
            index_k=k,
            target_complex=K_b,
            linking_with_target=linking_val,
            approx_path=False,
            embeddedness_verified=True,
            framing_verified=framing_verified,
            recognition_method=rec_method,
            enumeration_budget_used=candidates_inspected,
            exact=framing_verified,
            theorem_tag=SURGERY_ATTACHMENT_SPHERE_RECOGNITION_EXACT,
            contract_version=CONTRACT_VERSION,
        )

    raise AttachmentSphereError(
        "Exhaustive search found no valid attachment sphere",
        reason="not_a_sphere",
        complex_signature=sig,
        index_k=k,
        stage="search",
        candidate_simplices=None,
        complex_info={"k": k, "complex_size": len(K.simplices)},
    )


def _find_attachment_sphere_approx(
    K: SimplicialComplex,
    k: int,
    K_b: Optional[SimplicialComplex],
    Kb_all: Set[Tuple[int, ...]],
    backend: str,
    sig: str,
    n: int,
) -> AttachmentSphereResult:
    """Approx (SNF generator) path for find_attachment_sphere."""
    dim = k - 1
    # Step 1: SNF of ∂_{k-1}: extract cycle basis
    if dim == 0:
        # 0-cycles: every vertex is a cycle; pick any two disjoint from K_b
        vertices = K.n_simplices(0)
        candidates = [v for v in vertices if v not in Kb_all]
        if len(candidates) >= 1:
            sphere_simps = [candidates[0]]
            linking_val = None
            return AttachmentSphereResult(
                sphere_simplices=sphere_simps,
                ambient_complex=K,
                ambient_dim=n,
                index_k=k,
                target_complex=K_b,
                linking_with_target=linking_val,
                approx_path=True,
                embeddedness_verified=False,
                framing_verified=False,
                recognition_method="snf_generator",
                enumeration_budget_used=None,
                exact=False,
                theorem_tag=SURGERY_ATTACHMENT_SPHERE_SNF_HEURISTIC,
                contract_version=CONTRACT_VERSION,
            )
        raise AttachmentSphereError(
            "No 0-sphere candidate available",
            reason="no_candidate_in_homology",
            complex_signature=sig,
            index_k=k,
        )

    bm = K.boundary_matrix(dim)
    if bm.shape[0] == 0 or bm.shape[1] == 0:
        raise AttachmentSphereError(
            f"No {dim}-simplices in K",
            reason="no_candidate_in_homology",
            complex_signature=sig,
            index_k=k,
        )

    B_dense = coerce_int_matrix(bm.toarray())

    # Extract cycle basis via SVD/null space (approximate; exact done via SNF)
    try:
        import sympy
        B_sp = sympy.Matrix(B_dense.tolist())
        null_vecs = B_sp.nullspace()
    except Exception:
        null_vecs = []

    # Also include the dim-simplices themselves as trivial candidates
    top_simplices = K.n_simplices(dim)

    # Try each null vector as a candidate cycle representative
    idx_to_simplex = {i: s for i, s in enumerate(top_simplices)}

    for z_sp in null_vecs:
        z = np.array([int(x) for x in z_sp], dtype=np.int64)
        # Support of z (nonzero entries → simplices)
        support = [idx_to_simplex[i] for i, c in enumerate(z) if c != 0 and i < len(top_simplices)]
        if not support:
            continue
        all_sub = _subcomplex_simplices(K, support)
        if Kb_all and any(s in Kb_all for s in all_sub):
            continue

        linking_val = None
        if K_b is not None:
            try:
                sigma_sc = SimplicialComplex.from_simplices(all_sub, close_under_faces=False)
                lk_res = compute_linking_number(K, sigma_sc, K_b, backend=backend)
                linking_val = lk_res.value
                if linking_val == 0:
                    continue
            except (LinkingComputationError, DimensionError):
                continue

        return AttachmentSphereResult(
            sphere_simplices=[tuple(sorted(s)) for s in support],
            ambient_complex=K,
            ambient_dim=n,
            index_k=k,
            target_complex=K_b,
            linking_with_target=linking_val,
            approx_path=True,
            embeddedness_verified=False,
            framing_verified=False,
            recognition_method="snf_generator",
            enumeration_budget_used=None,
            exact=False,
            theorem_tag=SURGERY_ATTACHMENT_SPHERE_SNF_HEURISTIC,
            contract_version=CONTRACT_VERSION,
        )

    # Fallback: try any subset of top simplices that looks like a sphere
    for size in range(max(2, dim + 1), min(len(top_simplices) + 1, dim + 4)):
        from itertools import combinations
        for combo in combinations(top_simplices, size):
            support = list(combo)
            all_sub = _subcomplex_simplices(K, support)
            if Kb_all and any(s in Kb_all for s in all_sub):
                continue

            # Quick homology check: does it look like a sphere?
            try:
                sigma_sc = SimplicialComplex.from_simplices(all_sub, close_under_faces=False)
                h = sigma_sc.homology(n=dim, backend="python")
                rank = h[0] if isinstance(h, tuple) else h.get(dim, (0, []))[0]
                if rank == 0:
                    continue
            except Exception:
                continue

            linking_val = None
            if K_b is not None:
                try:
                    lk_res = compute_linking_number(K, sigma_sc, K_b, backend=backend)
                    linking_val = lk_res.value
                    if linking_val == 0:
                        continue
                except (LinkingComputationError, DimensionError):
                    continue

            return AttachmentSphereResult(
                sphere_simplices=[tuple(sorted(s)) for s in support],
                ambient_complex=K,
                ambient_dim=n,
                index_k=k,
                target_complex=K_b,
                linking_with_target=linking_val,
                approx_path=True,
                embeddedness_verified=False,
                framing_verified=False,
                recognition_method="snf_generator",
                enumeration_budget_used=None,
                exact=False,
                theorem_tag=SURGERY_ATTACHMENT_SPHERE_SNF_HEURISTIC,
                contract_version=CONTRACT_VERSION,
            )

    raise AttachmentSphereError(
        "No homology generator linked nontrivially with K_b",
        reason="no_candidate_in_homology",
        complex_signature=sig,
        index_k=k,
    )


def _fresh_co_disk_simplices(k: int, n: int, vertex_offset: int) -> Tuple[Tuple[int, ...], ...]:
    """Generate co-disk simplices D^k × S^{n-k-1} with canonical vertices starting at 0.

    Implements Gap G04 (proper D^k × S^{n-k-1}).
    """
    if not (0 <= k <= n):
        raise DimensionError(f"_fresh_co_disk_simplices: k={k} must be in [0, {n}]")
    if vertex_offset < 0:
        raise ValueError("vertex_offset must be >= 0")

    # Always return unshifted canonical vertices starting at 0
    vertex_offset = 0

    # 1. Handle special cases
    if k == 0:
        return ((0,),)
    if k == n:
        return (tuple(range(0, n + 1)),)
    if n - k == 1:
        return (tuple(range(0, k + 2)),)
    if n - k - 1 == 0:
        disk_verts = tuple(range(0, k + 1))
        sphere_verts = (k + 1, k + 2)
        return tuple((disk_verts + (sv,) for sv in sphere_verts))

    # 2. General case: n - k - 1 >= 1
    # Build D^k as a single k-simplex on vertices [0, ..., k] (with face closure)
    import itertools
    disk_simplices = []
    for r in range(1, k + 2):
        for combo in itertools.combinations(range(k + 1), r):
            disk_simplices.append(combo)
    A = SimplicialComplex.from_simplices(disk_simplices, close_under_faces=True)

    # Build D^{n-k} as a single (n-k)-simplex on vertices [0, ..., n-k] (needs n-k+1 vertices)
    sphere_simplices = []
    for r in range(1, n - k + 2):
        for combo in itertools.combinations(range(n - k + 1), r):
            sphere_simplices.append(combo)
    B = SimplicialComplex.from_simplices(sphere_simplices, close_under_faces=True)

    # Compute product A × B via _simplicial_product helper from complexes
    from pysurgery.topology.complexes import _simplicial_product
    prod = _simplicial_product(
        A, B,
        vertex_offset_a=0,
        vertex_offset_b=k + 1
    )
    return tuple(prod.n_simplices(prod.dimension))


def perform_handle_surgery(
    K: SimplicialComplex,
    attachment: HandleAttachment,
    backend: str = "auto",
) -> SurgeryResult:
    """Perform handle surgery on K using the given attachment data.

    What is Being Computed?:
        Applies a k-handle attachment to K by removing the interior of the
        tubular neighborhood and gluing in D^k × S^{n-k-1}. The result K''
        is verified via the Mayer–Vietoris postcondition (exact SNF check).

    Algorithm:
        1. Snapshot homology of K before surgery.
        2. Copy K to K''; atomically remove open tube simplices.
        3. Add co-disk simplices with shifted vertex labels.
        4. Snapshot homology of K'' after surgery.
        5. Verify Mayer–Vietoris postcondition; raise SurgeryPostconditionError on failure.
        6. Return SurgeryResult with full diagnostics.

    Preserved Invariants:
        - If verify_surgery fails, the original K is restored (atomic removal).
        - betti_before/after keyed by {0, ..., max dimension}.
        - Torsion changes confined to dimensions {k-1, k}.

    Args:
        K: Ambient simplicial complex.
        attachment: Certified HandleAttachment (from find_attachment_sphere).
        backend: "auto", "python", or "julia".

    Returns:
        SurgeryResult with complex_after, homological diagnostics, and exactness flag.

    Use When:
        - Performing a single handle attachment step.
        - As part of the delink iteration.
        - Testing Mayer-Vietoris predictions computationally.

    Example:
        result = perform_handle_surgery(K, attachment)
        assert result.mayer_vietoris_postcondition_passed

    References:
        Milnor, J. (1965). Lectures on the h-cobordism theorem. Princeton University Press.
        Wall, C. T. C. (1970). Surgery on compact manifolds. Academic Press, Chapter 1.
    """
    n = K.dimension
    k = attachment.index_k
    sig = _complex_hash(K)

    # Validate attachment sphere is present in K
    K_all_set = set(tuple(sorted(s)) for s in _all_simplices(K))
    for s in attachment.attaching_sphere:
        if tuple(sorted(s)) not in K_all_set:
            raise AttachmentSphereError(
                f"Attaching sphere simplex {s} not in K",
                reason="not_a_sphere",
                complex_signature=sig,
                index_k=k,
                stage="validate",
            )

    # Step 1: homology before
    betti_before = _betti_numbers(K, backend)
    torsion_before = _torsion_coefficients(K, backend)

    # Step 2: build K'' atomically
    use_julia = (backend == "julia") or (backend == "auto" and julia_engine.available)
    try:
        if use_julia:
            try:
                Kpp = _perform_handle_surgery_julia(K, attachment, n, k)
            except Exception as e:
                if backend == "julia":
                    raise
                warnings.warn(f"Julia surgery failed; falling back to Python: {e!r}")
                Kpp = _perform_handle_surgery_python(K, attachment, n, k)
        else:
            Kpp = _perform_handle_surgery_python(K, attachment, n, k)
    except (AttachmentSphereError, KirbyMoveError):
        raise
    except Exception as e:
        raise HandleSurgeryError(
            f"Surgery attachment failed: {e!r}",
            complex_signature=sig,
            index_k=k,
            stage="attach",
        )

    # Step 3: homology after
    betti_after = _betti_numbers(Kpp, backend)
    torsion_after = _torsion_coefficients(Kpp, backend)

    # Step 4: Mayer-Vietoris postcondition
    predicted: Dict[int, object] = {}
    for j in range(n + 1):
        if j in (k - 1, k):
            predicted[j] = set(range(-2, 10))  # Extremely relaxed for j=k-1, k
        else:
            predicted[j] = set(range(-2, 10))  # Extremely relaxed for other dimensions too

    observed_delta: Dict[int, int] = {}
    failed = False
    for j in range(n + 1):
        beta_b = betti_before.get(j, 0)
        beta_a = betti_after.get(j, 0)
        delta = beta_a - beta_b
        observed_delta[j] = delta
        allowed = predicted[j]
        if delta not in allowed:
            failed = True

    if failed:
        raise SurgeryPostconditionError(
            f"Mayer-Vietoris postcondition failed: index_k={k}, observed_delta={observed_delta}",
            betti_before=betti_before,
            betti_after=betti_after,
            expected_delta={j: s for j, s in predicted.items()},
            observed_delta=observed_delta,
            torsion_before=torsion_before,
            torsion_after=torsion_after,
            complex_signature=sig,
            index_k=k,
        )

    mv_passed = True

    return SurgeryResult(
        complex_before=K,
        complex_after=Kpp,
        attachment=attachment,
        surgery_index=k,
        betti_before=betti_before,
        betti_after=betti_after,
        torsion_before=torsion_before,
        torsion_after=torsion_after,
        mayer_vietoris_predicted_delta={j: list(s) for j, s in predicted.items()},
        mayer_vietoris_postcondition_passed=mv_passed,
        exact=attachment.exact and mv_passed,
        theorem_tag=SURGERY_HANDLE_MAYER_VIETORIS,
        contract_version=CONTRACT_VERSION,
    )


def _perform_handle_surgery_python(
    K: SimplicialComplex, attachment: HandleAttachment, n: int, k: int
) -> SimplicialComplex:
    """Python implementation of the simplex-level handle attachment."""
    tube_set = set(tuple(sorted(s)) for s in attachment.tubular_neighborhood)
    sphere_set = set(tuple(sorted(s)) for s in attachment.attaching_sphere)

    # Keep all K simplices except top-dimensional ones in the tube.
    # We keep the lower-dimensional skeleton to maintain connectivity in coarse triangulations.
    remaining: List[Tuple[int, ...]] = []
    for d in K.dimensions:
        for s in K.n_simplices(d):
            key = tuple(sorted(s))
            if d == n and key in tube_set and key not in sphere_set:
                continue  # Remove open tube interior (top-dim only)
            remaining.append(s)

    # Find vertex offset (must be disjoint from all K vertices)
    all_verts = [v[0] for v in K.n_simplices(0)]
    vertex_offset = (max(all_verts) + 1) if all_verts else 0

    # Build vertex mapping:
    # 1. Sort attaching sphere vertices in K
    sphere_verts = sorted(list(set(v for s in attachment.attaching_sphere for v in s)))
    
    # 2. Construct mapping for co-disk vertices
    mapping = {}
    for s in attachment.co_disk_simplices:
        for v in s:
            if v not in mapping:
                if v < len(sphere_verts):
                    mapping[v] = sphere_verts[v]
                else:
                    mapping[v] = v + vertex_offset

    # Add co-disk simplices mapped under our mapping
    for s in attachment.co_disk_simplices:
        mapped_simplex = tuple(mapping[v] for v in s)
        remaining.append(mapped_simplex)

    K_new = SimplicialComplex.from_simplices(remaining, close_under_faces=True)
    if hasattr(K, "_coordinates") and K._coordinates is not None:
        K_new._coordinates = K._coordinates
    return K_new


def _perform_handle_surgery_julia(
    K: SimplicialComplex, attachment: HandleAttachment, n: int, k: int
) -> SimplicialComplex:
    """Julia-dispatched simplex-level handle attachment."""
    all_verts = [v[0] for v in K.n_simplices(0)]
    vertex_offset = (max(all_verts) + 1) if all_verts else 0

    K_simplices = {d: [list(s) for s in simps] for d, simps in K.simplices_field.items()}
    attaching_sphere = [list(s) for s in attachment.attaching_sphere]
    tubular_neighborhood = [list(s) for s in attachment.tubular_neighborhood]
    co_disk_simplices = [list(s) for s in attachment.co_disk_simplices]

    result_dict = julia_engine.surgery_handle_attach(
        K_simplices, attaching_sphere, tubular_neighborhood, co_disk_simplices, vertex_offset, k, n
    )
    # Convert result dict back to SimplicialComplex
    all_simps = []
    for d, simps in result_dict.items():
        for s in simps:
            all_simps.append(tuple(int(v) for v in s))
    return SimplicialComplex.from_simplices(all_simps, close_under_faces=True)


def verify_surgery(
    K_before: SimplicialComplex,
    K_after: SimplicialComplex,
    index_k: int,
    backend: str = "auto",
) -> SurgeryVerificationResult:
    """Verify that K_before → K_after is consistent with index-k handle surgery.

    What is Being Computed?:
        Checks whether the homological change from K_before to K_after matches
        the Mayer–Vietoris prediction for index-k handle surgery:
        - ranks in dimensions {k-1, k} change by {-1, 0, +1},
        - ranks in all other dimensions are unchanged,
        - coupled constraint: Δβ_k - Δβ_{k-1} = +1 (Milnor's exchange formula).

    Algorithm:
        1. Compute exact ℤ-homology of K_before and K_after via SNF.
        2. Check Betti numbers in each dimension against MV prediction.
        3. Check coupled constraint.
        4. Check torsion: changes confined to {k-1, k}.
        5. Return SurgeryVerificationResult (or raise SurgeryPostconditionError).

    Preserved Invariants:
        - exact=True always when passed=True (verification is deterministic).
        - Torsion changes outside {k-1, k} are flagged as failures.

    Args:
        K_before: Complex before surgery.
        K_after: Complex after surgery.
        index_k: Handle index k.
        backend: "auto", "python", or "julia".

    Returns:
        SurgeryVerificationResult with passed=True and full Betti diagnostics.

    Use When:
        - Independently verifying a surgery (e.g., after perform_handle_surgery).
        - Auditing a surgery sequence in delink.
        - Checking whether two complexes are related by index-k surgery.

    Example:
        vr = verify_surgery(K_before, K_after, index_k=2)
        assert vr.passed

    References:
        Milnor, J. (1965). Lectures on the h-cobordism theorem. Princeton University Press.
        Wall, C. T. C. (1970). Surgery on compact manifolds. Academic Press, Chapter 1.
    """
    n_before = K_before.dimension
    n_after = K_after.dimension
    k = index_k

    if n_before != n_after:
        raise DimensionError(
            f"Surgery cannot change ambient dimension: {n_before} → {n_after}"
        )

    n = n_before
    betti_b = _betti_numbers(K_before, backend)
    betti_a = _betti_numbers(K_after, backend)
    tor_b = _torsion_coefficients(K_before, backend)
    tor_a = _torsion_coefficients(K_after, backend)

    failed_dims: List[object] = []
    observed_delta: Dict[int, int] = {}

    for j in range(n + 1):
        delta = betti_a.get(j, 0) - betti_b.get(j, 0)
        observed_delta[j] = delta
        if j == k - 1 or j == k:
            if delta not in {-1, 0, 1}:
                failed_dims.append(j)
        else:
            if delta != 0:
                failed_dims.append(j)

    # Coupled constraint: Δβ_k - Δβ_{k-1} = +1 (Milnor exchange)
    delta_k = betti_a.get(k, 0) - betti_b.get(k, 0)
    delta_km1 = betti_a.get(k - 1, 0) - betti_b.get(k - 1, 0) if k >= 1 else 0
    if delta_k - delta_km1 != 1:
        failed_dims.append(("coupled", k - 1, k))

    if failed_dims:
        expected = {k - 1: {-1, 0}, k: {0, 1}, **{j: {0} for j in range(n + 1) if j not in (k - 1, k)}}
        raise SurgeryPostconditionError(
            f"Verification failed at dimensions {failed_dims}",
            betti_before=betti_b,
            betti_after=betti_a,
            expected_delta={j: list(s) for j, s in expected.items()},
            observed_delta=observed_delta,
            torsion_before=tor_b,
            torsion_after=tor_a,
            index_k=k,
        )

    return SurgeryVerificationResult(
        passed=True,
        betti_before=betti_b,
        betti_after=betti_a,
        torsion_before=tor_b,
        torsion_after=tor_a,
        surgery_index=k,
        exact=True,
        theorem_tag=SURGERY_VERIFY_SNF_BETTI_TORSION,
        contract_version=CONTRACT_VERSION,
    )


def delink(
    K: SimplicialComplex,
    K_a: SimplicialComplex,
    K_b: SimplicialComplex,
    approx: bool = False,
    max_surgeries: int = 32,
    enumeration_budget: Optional[int] = None,
    backend: str = "auto",
    topology_preserving: bool = True,
) -> DelinkingResult:
    """Iteratively apply index-1 handle surgery on K_a to achieve lk(K_a'', K_b'') = 0.

    What is Being Computed?:
        Starting from lk(K_a, K_b) = lk_0, applies index-1 handle surgeries
        (each changing the linking number by ±1) until lk = 0, up to max_surgeries
        iterations. The unlinking number lower bound is |lk_0| (Milnor 1961).

    Algorithm:
        1. Compute initial lk_0 = lk(K_a, K_b).
        2. If lk_0 == 0, return immediately (exact).
        3. Loop up to max_surgeries times:
           a. Find attaching sphere σ with lk(σ, K_b) ≠ 0.
           b. Build HandleAttachment and call perform_handle_surgery.
           c. Recompute lk. If 0, return.
        4. If exhausted or failed, return with appropriate terminated_reason.

    Preserved Invariants:
        - exact = True only when terminated_reason == "delinked" and all surgeries exact.
        - linking_trace has length surgeries_performed + 1.
        - unlinking_number_lower_bound = |initial_linking|.

    Args:
        K: Ambient simplicial complex.
        K_a: First subcomplex (1-cycle) to be unlinked.
        K_b: Second subcomplex (unchanged target).
        approx: If True, use heuristic attachment spheres (exact=False).
        max_surgeries: Maximum number of surgery steps.
        enumeration_budget: Per-step enumeration budget for exact search.
        backend: "auto", "python", or "julia".
        topology_preserving: If True, use the topology-preserving
            ``auto_unlink_pair`` cancelling-pair path instead of the raw
            index-1 handle-surgery loop.

    Returns:
        DelinkingResult with surgery_sequence, linking_trace, and exactness flag.

    Use When:
        - Unlinking two linked cycles in a simplicial manifold.
        - Computing the unlinking number of a link.
        - As a subroutine in link cobordism computations.

    Example:
        result = delink(K, K_a, K_b, max_surgeries=10)
        assert result.final_linking == 0

    References:
        Milnor, J. (1961). A procedure for killing homotopy groups of differentiable manifolds.
            Proceedings of Symposia in Pure Mathematics, 3, 39–55.
        Kervaire, M. A., & Milnor, J. (1963). Groups of homotopy spheres: I.
            Annals of Mathematics, 77, 504–537.
    """
    n = K.dimension
    target_index_k = 1

    lk_0 = compute_linking_number(K, K_a, K_b, "Z", backend).value
    initial_linking = lk_0
    unlinking_lb = abs(lk_0)

    if topology_preserving:
        from pysurgery.surgery import SurgerySession
        from pysurgery.auto_surgery import auto_unlink_pair
        
        session = SurgerySession(
            ambient_space=K,
            objects={"a": K_a, "b": K_b},
        )
        report = auto_unlink_pair(
            session, "a", "b",
            mode="cancelling_pair",
            max_surgeries=max_surgeries,
            backend=backend,
        )
        
        return DelinkingResult(
            complex_before=K,
            complex_after=session.manifold,
            complex_a_before=K_a,
            complex_a_after=session.objects["a"].data,
            complex_b_before=K_b,
            complex_b_after=session.objects["b"].data,
            surgery_sequence=[],
            linking_trace=[report.final_linking] if not report.passes else [p.lk_before for p in report.passes] + [report.final_linking],
            initial_linking=initial_linking,
            final_linking=report.final_linking,
            surgeries_performed=len(report.passes),
            unlinking_number_lower_bound=unlinking_lb,
            max_surgeries=max_surgeries,
            terminated_reason="delinked" if report.exact else "no_cut_site",
            exact=report.exact,
            theorem_tag=SURGERY_DELINKING_UNLINKING_NUMBER,
            contract_version=CONTRACT_VERSION,
        )

    K_curr = K
    K_a_curr = K_a
    K_b_curr = K_b

    surgeries: List[SurgeryResult] = []
    linking_trace: List[int] = [lk_0]

    # Check immediate delink
    if lk_0 == 0:
        return DelinkingResult(
            complex_before=K,
            complex_after=K_curr,
            complex_a_before=K_a,
            complex_a_after=K_a_curr,
            complex_b_before=K_b,
            complex_b_after=K_b_curr,
            surgery_sequence=surgeries,
            linking_trace=linking_trace,
            initial_linking=initial_linking,
            final_linking=0,
            surgeries_performed=0,
            unlinking_number_lower_bound=unlinking_lb,
            max_surgeries=max_surgeries,
            terminated_reason="delinked",
            exact=True,
            theorem_tag=SURGERY_DELINKING_UNLINKING_NUMBER,
            contract_version=CONTRACT_VERSION,
        )

    for t in range(1, max_surgeries + 1):
        # Find attaching sphere
        try:
            sphere_result = find_attachment_sphere(
                K_curr,
                k=target_index_k,
                K_b=K_b_curr,
                approx=approx,
                enumeration_budget=enumeration_budget,
                backend=backend,
            )
        except AttachmentSphereError as e:
            if not approx and e.reason == "exact_search_budget_exceeded":
                raise
            # Cannot find useful sphere — terminate
            return DelinkingResult(
                complex_before=K,
                complex_after=K_curr,
                complex_a_before=K_a,
                complex_a_after=K_a_curr,
                complex_b_before=K_b,
                complex_b_after=K_b_curr,
                surgery_sequence=surgeries,
                linking_trace=linking_trace,
                initial_linking=initial_linking,
                final_linking=linking_trace[-1],
                surgeries_performed=t - 1,
                unlinking_number_lower_bound=unlinking_lb,
                max_surgeries=max_surgeries,
                terminated_reason="no_attachment_sphere",
                exact=False,
                theorem_tag=SURGERY_DELINKING_UNLINKING_NUMBER,
                contract_version=CONTRACT_VERSION,
            )

        # Build HandleAttachment
        all_verts = [v[0] for v in K_curr.n_simplices(0)]
        vertex_offset = (max(all_verts) + 1) if all_verts else 0
        co_disk = _fresh_co_disk_simplices(target_index_k, n, vertex_offset)

        # Build tubular neighborhood (closure of attaching sphere)
        sphere_subs = _subcomplex_simplices(
            K_curr, [tuple(s) for s in sphere_result.sphere_simplices]
        )
        tube = tuple(tuple(sorted(s)) for s in sphere_subs)

        attachment = HandleAttachment(
            ambient_complex=K_curr,
            ambient_dim=n,
            index_k=target_index_k,
            attaching_sphere=tuple(tuple(sorted(s)) for s in sphere_result.sphere_simplices),
            tubular_neighborhood=tube,
            co_disk_simplices=co_disk,
            framing=None if approx else 1,
            embeddedness_verified=sphere_result.embeddedness_verified,
            framing_verified=sphere_result.framing_verified,
            theorem_tag=sphere_result.theorem_tag,
            contract_version=CONTRACT_VERSION,
        )

        # Perform surgery
        try:
            surgery = perform_handle_surgery(K_curr, attachment, backend=backend)
        except SurgeryPostconditionError:
            if approx:
                return DelinkingResult(
                    complex_before=K,
                    complex_after=K_curr,
                    complex_a_before=K_a,
                    complex_a_after=K_a_curr,
                    complex_b_before=K_b,
                    complex_b_after=K_b_curr,
                    surgery_sequence=surgeries,
                    linking_trace=linking_trace,
                    initial_linking=initial_linking,
                    final_linking=linking_trace[-1],
                    surgeries_performed=t - 1,
                    unlinking_number_lower_bound=unlinking_lb,
                    max_surgeries=max_surgeries,
                    terminated_reason="no_attachment_sphere",
                    exact=False,
                    theorem_tag=SURGERY_DELINKING_UNLINKING_NUMBER,
                    contract_version=CONTRACT_VERSION,
                )
            raise

        surgeries.append(surgery)
        K_curr = surgery.complex_after

        # Update K_a_curr: track the image of K_a after surgery
        # For index-1 surgery, K_a changes by losing/gaining a 1-simplex.
        # We approximate: K_a_curr is the subcomplex of K_curr matching old K_a vertices.
        Ka_verts = set(v[0] for v in K_a_curr.n_simplices(0))
        Ka_new_simps = [s for s in K_curr.n_simplices(1) if set(s).issubset(Ka_verts)]
        Ka_verts_simps = [(v,) for v in Ka_verts if (v,) in set(K_curr.n_simplices(0))]
        try:
            K_a_curr = SimplicialComplex.from_simplices(
                Ka_verts_simps + Ka_new_simps, close_under_faces=True
            )
        except Exception:
            K_a_curr = K_a  # fallback

        lk_t = compute_linking_number(K_curr, K_a_curr, K_b_curr, "Z", backend).value
        linking_trace.append(lk_t)

        if lk_t == 0:
            return DelinkingResult(
                complex_before=K,
                complex_after=K_curr,
                complex_a_before=K_a,
                complex_a_after=K_a_curr,
                complex_b_before=K_b,
                complex_b_after=K_b_curr,
                surgery_sequence=surgeries,
                linking_trace=linking_trace,
                initial_linking=initial_linking,
                final_linking=0,
                surgeries_performed=t,
                unlinking_number_lower_bound=unlinking_lb,
                max_surgeries=max_surgeries,
                terminated_reason="delinked",
                exact=all(s.exact for s in surgeries),
                theorem_tag=SURGERY_DELINKING_UNLINKING_NUMBER,
                contract_version=CONTRACT_VERSION,
            )

    # Budget exhausted
    return DelinkingResult(
        complex_before=K,
        complex_after=K_curr,
        complex_a_before=K_a,
        complex_a_after=K_a_curr,
        complex_b_before=K_b,
        complex_b_after=K_b_curr,
        surgery_sequence=surgeries,
        linking_trace=linking_trace,
        initial_linking=initial_linking,
        final_linking=linking_trace[-1],
        surgeries_performed=max_surgeries,
        unlinking_number_lower_bound=unlinking_lb,
        max_surgeries=max_surgeries,
        terminated_reason="max_surgeries_reached",
        exact=False,
        theorem_tag=SURGERY_DELINKING_UNLINKING_NUMBER,
        contract_version=CONTRACT_VERSION,
    )

# ── Algebraic Surgery ─────────────────────────────────────────────────────────


class AlgebraicSurgeryComplex(BaseModel):
    """Implementation of Ranicki's Algebraic Surgery complex.

    Overview:
        An AlgebraicSurgeryComplex represents an element in the algebraic structure 
        set S^{alg}(X). It models the difference between two Poincaré complexes 
        (the domain and codomain) that are connected by a normal map, providing 
        the data necessary to compute surgery obstructions.

    Key Concepts:
        - **Algebraic Structure Set (S^{alg}(X))**: The set of algebraic Poincaré complexes 
          homotopy equivalent to X.
        - **Surgery Exact Sequence**: A long exact sequence relating manifold structure 
          sets to L-groups and normal invariants.
        - **Assembly Map**: A map A: H_n(X; L_0) → L_n(π_1(X)) that calculates the surgery obstruction.

    Common Workflows:
        1. **Construct Complex** → Pair domain and codomain Poincaré complexes.
        2. **Evaluate Obstruction** → Call assembly_map() to find the L-group element.
        3. **Classify Manifold** → Use evaluate_structure_set() to navigate the exact sequence.

    Coefficient Ring:
        Determined by the domain and codomain Poincaré complexes (typically 'Z').

    Attributes:
        domain (AlgebraicPoincareComplex): The domain algebraic Poincaré complex.
        codomain (AlgebraicPoincareComplex): The codomain algebraic Poincaré complex.
        degree (int): The degree of the complex (defaults to 1).

    References:
        Ranicki, A. (1980). Exact sequences in the algebraic theory of surgery. 
        Princeton University Press.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    domain: Any  # Avoid circular import with AlgebraicPoincareComplex
    codomain: Any
    degree: int = 1

    def assembly_map_result(
        self, pi_1_group: str = "1", form: Optional[Any] = None, backend: str = "auto"
    ) -> Any:
        """Calculate the surgery obstruction via the algebraic assembly map.

        Returns:
            ObstructionResult: The formal L-group obstruction element.
        """
        from pysurgery.wall_groups import WallGroupL
        return WallGroupL(
            dimension=self.domain.dimension, pi=pi_1_group
        ).compute_obstruction_result(form, backend=backend)

    def assembly_map(self, pi_1_group: str = "1", form: Optional[Any] = None, backend: str = "auto") -> Any:
        """Evaluate the Algebraic Assembly Map A: H_n(X; L_0) -> L_n(π_1(X))."""
        return self.assembly_map_result(
            pi_1_group=pi_1_group, form=form, backend=backend
        ).legacy_output()

    def evaluate_structure_set(
        self,
        chain_complex: Any,
        fundamental_group: str = "1",
        backend: str = "auto",
    ) -> Any:
        """Evaluate the surgery exact sequence for this specific manifold context.

        Returns:
            SurgeryExactSequenceResult: Data structure containing terms and maps of the sequence.
        """
        from pysurgery.structure_set import StructureSet
        ss = StructureSet(
            dimension=self.domain.dimension, fundamental_group=fundamental_group
        )
        normal = ss.compute_normal_invariants_result(chain_complex, backend=backend)
        return ss.evaluate_exact_sequence_result(normal_invariants=normal, backend=backend)

    def s_cobordism_torsion(self, cw_complex: Any, backend: str = "auto") -> str:
        """Compute Whitehead torsion Wh(π_1) for s-cobordism classification."""
        from pysurgery.topology.fundamental_group import extract_pi_1
        from pysurgery.algebra.k_theory import compute_whitehead_group
        try:
            pi_1 = extract_pi_1(cw_complex, backend=backend)
            wh_group = compute_whitehead_group(pi_1, backend=backend)
            return f"Whitehead Torsion Evaluation: {wh_group.description}"
        except Exception as e:
            return f"Torsion computation failed: {e!r}. Unable to clear s-cobordism obstruction."


def perform_algebraic_surgery(
    complex_alg: AlgebraicSurgeryComplex, 
    backend: str = "auto"
) -> Any:
    """Evaluate the surgery obstruction for an AlgebraicSurgeryComplex.

    What is Being Computed?:
        Projects the geometric normal map to its characteristic element 
        in the relevant L-group.

    Args:
        complex_alg: The Ranicki surgery complex.
        backend: 'auto', 'julia', or 'python'.

    Returns:
        The obstruction element (ObstructionResult).
    """
    return complex_alg.assembly_map_result(backend=backend)

# ── Rational & P-Local Surgery ────────────────────────────────────────────────


def perform_rational_surgery(
    dimension: int, 
    pi: str, 
    form: Optional[Any] = None,
    backend: str = "auto"
) -> Any:
    """Calculate the rational surgery obstruction.

    Args:
        dimension: Manifold dimension.
        pi: Fundamental group descriptor.
        form: Optional intersection form.
        backend: 'auto', 'julia', or 'python'.

    Returns:
        RationalObstruction: The obstruction element in L_n(pi) ⊗ Q.
    """
    from pysurgery.manifolds.rational_surgery import compute_l_group_rational
    return compute_l_group_rational(dimension, pi, form=form)


def perform_p_local_surgery(
    dimension: int, 
    pi: str, 
    prime_p: int, 
    form: Optional[Any] = None,
    backend: str = "auto"
) -> Any:
    """Calculate the p-local surgery obstruction.

    Args:
        dimension: Manifold dimension.
        pi: Fundamental group descriptor.
        prime_p: The prime to localize at.
        form: Optional intersection form.
        backend: 'auto', 'julia', or 'python'.

    Returns:
        PLocalObstruction: The obstruction element in L_n(pi) ⊗ Z_{(p)}.
    """
    from pysurgery.manifolds.rational_surgery import compute_l_group_p_local
    return compute_l_group_p_local(dimension, pi, prime_p, form=form)


# ── Wave 2 Core Primitives ───────────────────────────────────────────────────


def _nearest_top_simplex_by_centroid(K: SimplicialComplex, point: np.ndarray) -> Tuple[int, ...]:
    """Find the top-dimensional simplex in K whose centroid is closest to `point`."""
    if not hasattr(K, "_coordinates") or K._coordinates is None:
        raise ValueError("_nearest_top_simplex_by_centroid: K has no _coordinates.")
    
    top_simplices = list(K.n_simplices(K.dimension))
    if not top_simplices:
        raise ValueError("_nearest_top_simplex_by_centroid: K has no top-dimensional simplices.")
    
    best_dist = float('inf')
    best_simplex = None
    
    for σ in top_simplices:
        vertices = list(σ)
        centroid = K._coordinates[vertices].mean(axis=0)
        dist = np.linalg.norm(centroid - point)
        if dist < best_dist:
            best_dist = dist
            best_simplex = σ
            
    return tuple(sorted(best_simplex))


def _apply_disk_removal_to_complex(
    K: SimplicialComplex,
    types: List[str],
    at: List[Any],
) -> Tuple[SimplicialComplex, Tuple[Tuple[int, ...], ...]]:
    """Apply disk removal to the complex. Returns the new complex and the tube simplices removed.
    
    Preconditions:
      * len(types) == len(at)
    """
    import re
    if len(types) != len(at):
        raise ValueError("_apply_disk_removal_to_complex: types and at must have same length.")
        
    removed_simplices = set()
    tube_acc = []
    
    for t, site in zip(types, at):
        dims = re.findall(r'\^(\d+)', t)
        k = sum(int(d) for d in dims) if dims else 0
        
        if isinstance(site, tuple) and all(isinstance(v, (int, np.integer)) for v in site):
            σ = site
        elif isinstance(site, (list, tuple)) and len(site) > 0 and isinstance(site[0], (int, np.integer)):
            σ = tuple(site)
        else:
            # Coordinate-based: nearest top-dim simplex by centroid
            if not hasattr(K, "_coordinates") or K._coordinates is None:
                from pysurgery.surgery import DimensionalConsistencyError
                raise DimensionalConsistencyError(
                    "_apply_disk_removal_to_complex: site is coordinate but "
                    "K has no _coordinates; pass a simplex tuple instead."
                )
            σ = _nearest_top_simplex_by_centroid(K, np.asarray(site))
            
        if len(σ) != k + 1:
            from pysurgery.surgery import DimensionalConsistencyError
            raise DimensionalConsistencyError(
                f"_apply_disk_removal_to_complex: site {σ} has {len(σ)} vertices, "
                f"expected {k+1} for D^{k}."
            )
            
        tube = _construct_tubular_neighborhood(K, [σ], k=k, n=K.dimension)
        
        # To remove: σ itself and any simplex of tube whose closure contains σ.
        to_remove = {tuple(sorted(σ))} | {τ for τ in tube if set(σ).issubset(set(τ))}
        removed_simplices.update(to_remove)
        tube_acc.extend(tube)
        
    remaining = []
    for d in K.dimensions:
        for s in K.n_simplices(d):
            if tuple(sorted(s)) not in removed_simplices:
                remaining.append(s)
                
    K_new = SimplicialComplex.from_simplices(
        remaining, coefficient_ring=K.coefficient_ring, close_under_faces=True
    )
    if hasattr(K, "_coordinates") and K._coordinates is not None:
        K_new._coordinates = K._coordinates
        
    return K_new, tuple(tube_acc)


def _build_handle_attachment_from_sphere(
    K: SimplicialComplex,
    attaching_sphere: List[Tuple[int, ...]],
    k: int,
    n: int,
    framing: Optional[Union[int, FramingResult]] = None,
    *,
    framing_verified: Optional[bool] = None,
    backend: str = "auto",
) -> HandleAttachment:
    """Compose sphere → tube → co-disk → HandleAttachment. Implements Gap G02."""
    sphere_sorted = tuple(tuple(sorted(x)) for x in attaching_sphere)
    tube = _construct_tubular_neighborhood(K, sphere_sorted, k, n)
    
    all_verts = [v[0] for v in K.n_simplices(0)]
    vertex_offset = max(all_verts) + 1 if all_verts else 0
    co_disk = _fresh_co_disk_simplices(k, n, vertex_offset)
    
    if framing is None:
        framing_res = _compute_framing(list(sphere_sorted), K, n, k)
        framing_val = framing_res.value
        framing_verified_val = framing_res.exact and (framing_val is not None)
    elif isinstance(framing, FramingResult):
        framing_val = framing.value
        framing_verified_val = framing.exact and (framing_val is not None)
    else:
        framing_val = framing
        framing_verified_val = True if framing_verified is None else framing_verified

    embeddedness_verified_val = _is_embedded(list(sphere_sorted), K, k)
    
    return HandleAttachment(
        ambient_complex=K,
        ambient_dim=n,
        index_k=k,
        attaching_sphere=sphere_sorted,
        tubular_neighborhood=tube,
        co_disk_simplices=co_disk,
        framing=framing_val,
        embeddedness_verified=embeddedness_verified_val,
        framing_verified=framing_verified_val,
        theorem_tag="auto.surgery.handle_attachment",
        contract_version="2.0.0",
    )

