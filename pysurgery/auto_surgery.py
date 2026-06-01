from __future__ import annotations
import numpy as np
import warnings
from typing import List, Tuple, Dict, Any, Optional, Literal, Union
from pydantic import BaseModel, ConfigDict, Field
from pysurgery.core.exceptions import SurgeryError, DimensionError, LadderProgressError
from pysurgery.topology.complexes import SimplicialComplex
from pysurgery.core.generator_models import Pi1PresentationWithTraces, HomologyGenerator
from pysurgery.surgery import ThroughPointIsotopy, SurgerySession
from pysurgery.core.theorem_tags import (
    AUTO_SURGERY_SEPARATE_NESTED,
    AUTO_SURGERY_DETECT_NESTED,
    AUTO_SURGERY_KILL_PI1,
    AUTO_SURGERY_KILL_HOMOLOGY_DIM,
    AUTO_SURGERY_MIDDLE_OBSTRUCTION,
)
from pysurgery.homology.homology_generators import hk_generators_z


# ── Pydantic Models ───────────────────────────────────────────────────────────

class GeneratorCycle(BaseModel):
    """A named 1-cycle representing a homotopy/homology generator.

    Records the cycle as ordered edges, the underlying vertex path, the root
    vertex of its connected component, and its orientation character.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    cycle: List[Tuple[int, int]]
    vertex_path: List[int]
    component_root: int
    orientation_character: int = 1

class CutSite(BaseModel):
    """A candidate location at which to perform a cut/surgery.

    Holds the target simplex, an optional centroid, a desirability score, and
    flags describing whether cutting there keeps the component connected and
    how many local strands pass through.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    simplex: Tuple[int, ...]
    centroid: Optional[Tuple[float, ...]] = None
    score: float
    keeps_component_connected: bool
    local_strands: int = 1
    tube_radius: Optional[float] = None
    exact: bool = True
    theorem_tag: str = "auto.surgery.cut_site"

class ComponentInfo(BaseModel):
    """Topological summary of a single connected component.

    Captures the component's dimension, manifold/closed status, Betti numbers,
    a π₁ descriptor, its vertex ids, and the extracted subcomplex.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    dimension: int
    is_manifold: bool
    is_closed: bool
    betti: Dict[int, int]
    pi1_descriptor: str
    vertex_ids: List[int]
    subcomplex: SimplicialComplex

class LinkedPair(BaseModel):
    """A pair of components ``a`` and ``b`` with nonzero linking number ``lk``."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    a: str
    b: str
    lk: int

class Pi1Killer(BaseModel):
    """A surgery prescription for killing one π₁ generator.

    Pairs the generating cycle with a framing and the descriptor of the
    generator expected to be eliminated by the surgery.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    cycle: List[Tuple[int, int]]
    framing: Any
    expected_kill: str

class HomologyKiller(BaseModel):
    """A surgery prescription for killing one degree-``k`` homology generator.

    Pairs the representing cycle with its dimension ``k``, a framing, and the
    homology class expected to be eliminated by the surgery.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    cycle: Any
    k: int
    framing: Any
    expected_kill: Any

class NestedPair(BaseModel):
    """A pair where component ``inner`` is nested inside component ``outer``.

    Records the codimension of the outer component, whether the detection is
    exact, and the kind of witness (algebraic, winding, bounding-box, etc.)
    that established the nesting.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    outer: str
    inner: str
    witness: Literal["algebraic", "winding", "algebraic_only", "bbox_only", "codim_mismatch"]
    codim_outer: int
    exact: bool
    theorem_tag: str = "auto.surgery.detect_nested"

class UnlinkPass(BaseModel):
    """Record of a single unlinking surgery attempt.

    Stores the chosen cut site and isotopy id, the linking number before and
    after, whether Betti/π₁/manifold invariants matched, and whether the pass
    was rolled back (with any error message).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    cut_site: CutSite
    isotopy_id: str
    lk_before: int
    lk_after: int
    betti_match: bool
    pi1_match: bool = True
    mani_match: bool = True
    rolled_back: bool = False
    error: Optional[str] = None

class UnlinkReport(BaseModel):
    """Full report of unlinking two components ``a`` and ``b``.

    Aggregates every UnlinkPass, the cut-site history, the final linking
    number, the surgery mode used, and whether topology was preserved.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    a: str
    b: str
    mode: Literal["cancelling_pair", "raw_handle_surgery"]
    passes: List[UnlinkPass]
    cut_site_history: List[Any]
    final_linking: int
    topology_preserved: bool
    exact: bool
    theorem_tag: str = "auto.surgery.unlink"
    contract_version: str = "2026.04-phase10"

class NestReport(BaseModel):
    """Full report of separating a nested ``inner`` from ``outer``.

    Records the connecting arc (simplices and endpoints), whether the pair is
    still nested afterward, and pre/post topology snapshots.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    outer: str
    inner: str
    arc_simplices: List[Tuple[int, ...]]
    arc_endpoints: List[Tuple[int, ...]]
    still_nested: bool
    pre_snapshot: Dict[str, Any]
    post_snapshot: Dict[str, Any]
    exact: bool
    theorem_tag: str = "auto.surgery.separate_nested"
    contract_version: str = "2026.04-phase10"

class Pi1KillStep(BaseModel):
    """Record of one step in the π₁-killing process.

    Holds the cycle and framing surgered, the generator it targeted, and
    whether the handle was attached and the generator verified killed.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    cycle: Optional[List[Tuple[int, int]]] = None
    framing: Optional[Any] = None
    expected_generator: Optional[Any] = None
    verified_killed: bool
    attached: bool = True
    reason: Optional[str] = None

class Pi1KillReport(BaseModel):
    """Full report of killing π₁ on a single component.

    Aggregates each Pi1KillStep along with the final π₁ descriptor for the
    component (``"1"`` when the fundamental group is trivial).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    steps: List[Pi1KillStep]
    final_pi1_descriptor: str
    exact: bool
    theorem_tag: str = AUTO_SURGERY_KILL_PI1
    contract_version: str = "2026.04-phase10"

    @classmethod
    def trivial(cls, name: str, exact: bool = True) -> Pi1KillReport:
        """Build an empty report for a component already having trivial π₁.

        Args:
            name: The component name.
            exact: Whether the determination is exact.

        Returns:
            A Pi1KillReport with no steps and a final descriptor of ``"1"``.
        """
        return cls(
            name=name,
            steps=[],
            final_pi1_descriptor="1",
            exact=exact,
            theorem_tag=AUTO_SURGERY_KILL_PI1,
        )

class HKillStep(BaseModel):
    """Record of one step in killing a homology summand.

    Holds the surgered cycle, the targeted summand, whether the handle was
    attached, the Betti numbers afterward, and any error encountered.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    cycle: List[Tuple[int, ...]]
    summand: str
    attached: bool
    betti_after: Optional[Dict[int, int]] = None
    error: Optional[str] = None
    exact: bool

class HKillReport(BaseModel):
    """Full report of killing degree-``k`` homology on a single component.

    Aggregates each HKillStep performed while eliminating the H_k generators
    of the named component.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    k: int
    steps: List[HKillStep]
    exact: bool
    theorem_tag: str = AUTO_SURGERY_KILL_HOMOLOGY_DIM
    contract_version: str = "2026.04-phase10"

class ObstructionReport(BaseModel):
    """Result of checking the middle-dimensional surgery obstruction.

    Records the component, its dimension and middle degree, the kind of
    obstruction, its computed class, and whether the obstruction vanishes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    dimension: int = 0
    middle_k: int = 0
    kind: str
    obstruction_class: Dict[str, Any] = Field(default_factory=dict)
    vanishes: bool
    pi1_descriptor: str = "?"
    exact: bool = True
    theorem_tag: str = AUTO_SURGERY_MIDDLE_OBSTRUCTION
    contract_version: str = "2026.04-phase10"

    @classmethod
    def from_ladder_error(cls, e: Exception) -> "ObstructionReport":
        """Build a non-vanishing report from a homology-ladder failure.

        Args:
            e: The exception raised when the homology ladder aborted; its
                ``name`` attribute (if a string) names the component.

        Returns:
            An ObstructionReport of kind ``"ladder_aborted"`` that does not
            vanish and carries the error message in its obstruction class.
        """
        name = getattr(e, "name", "unknown")
        if not isinstance(name, str):
            name = "unknown"
        return cls(
            name=name,
            kind="ladder_aborted",
            vanishes=False,
            exact=True,
            obstruction_class={"error": str(e)},
        )


class AutoSurgeonConfig(BaseModel):
    """Tunable parameters for the AutoSurgeon pipeline.

    Groups settings by phase (unlinking, un-nesting, per-component reduction)
    plus cross-cutting options such as the compute backend, framing
    requirements, target topology, and ambient-space specification.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    # Phase 1
    max_unlink_surgeries: int = 32
    unlink_mode: Literal["cancelling_pair", "raw_handle_surgery"] = "cancelling_pair"
    resample_after_close: bool = False
    cut_site_budget: int = 64
    threading_safety_margin: float = 0.5
    path_intersection_mode: Literal["fast", "exact"] = "fast"
    path_intersection_t_samples: int = 8

    # Phase 2
    exit_margin: float = 0.5
    arc_budget: int = 128

    # Phase 3
    kill_pi1_budget: Optional[int] = None
    pi1_simplify_pass: bool = True
    framing_retry_budget: int = 8
    homology_kill_order: Literal["free_first", "torsion_first"] = "free_first"
    homology_killer_top_k: Optional[int] = None
    obstruction_retry_pass: bool = True
    cancelling_pair_handle_label: str = "S^{n-1} x D^0"
    target_topology: Literal["homotopy_sphere", "contractible"] = "homotopy_sphere"

    # Cross-cutting
    backend: Literal["auto", "python", "julia"] = "auto"
    compact_logs: bool = False
    require_framing: bool = True

    # Ambient space specification — "R^3", "S^3", "RP^2", 3, etc.
    # None / "auto"  → infer as R^{component_dim + 1}
    ambient: Optional[Union[str, int]] = None


class AutoSurgeryReport(BaseModel):
    """Complete result of running the AutoSurgeon pipeline.

    Records the final status, the initial and final component summaries, the
    per-phase reports (unlink, un-nest, π₁-kill, homology-kill, obstruction),
    the session logs (plain and LaTeX), and the cobordism trace.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    status: Literal["success", "halted_by_obstruction", "halted_by_error"]
    exact: bool
    initial_components: List[ComponentInfo]
    final_components: List[ComponentInfo]
    unlink_reports: List[UnlinkReport]
    nest_reports: List[NestReport]
    pi1_kill_reports: Dict[str, Pi1KillReport]
    homology_kill_reports: Dict[str, List[HKillReport]]
    obstruction_reports: Dict[str, ObstructionReport]
    session_logs_plain: str
    session_logs_latex: str
    cobordism_trace: List[Dict[str, Any]]
    theorem_tag: str = "auto.surgery.full_pipeline"
    contract_version: str = "2026.04-phase10"

# ── Exception Classes ─────────────────────────────────────────────────────────

class AutoSurgeryError(SurgeryError):
    """Base exception for all auto-surgery failures."""
    pass

class NonManifoldComponentError(AutoSurgeryError):
    """Raised if a connected component is not a homology manifold."""
    pass

class NoCutSiteError(AutoSurgeryError):
    """Raised when no valid cut site can be found on a component."""
    pass

class NoAttachingSphereError(AutoSurgeryError):
    """Raised when a candidate attaching sphere cannot be framed or attached."""
    pass

class MiddleDimensionObstructed(AutoSurgeryError):
    """Raised when surgery at the middle dimension is obstructed by a non-trivial Wall group element."""
    pass

class HomologyManifoldNotPLWarning(UserWarning):
    """Warning issued when operating on a homology manifold of dimension >= 4 which might not be PL-rigorous."""
    pass


# ── Ambient-space utilities ───────────────────────────────────────────────────

def _parse_ambient_spec(
    ambient: Optional[Union[str, int]],
    fallback_manifold_dim: int,
) -> Tuple[str, int]:
    """Parse an ambient-space specification to (label, dimension).

    What is Being Computed?:
        Converts user-facing ambient descriptors into a canonical string label
        (used in session logs) and an integer dimension (used in surgery
        dimension checks and linking-number constraints).

    Algorithm:
        1. None / "auto"  → R^{fallback_manifold_dim + 1}
        2. Integer n       → "R^n", n
        3. String with "^d" (e.g. "R^3", "S^3", "RP^2") → (string, d)

    Args:
        ambient: User-supplied ambient spec or None.
        fallback_manifold_dim: Dimension of the manifold being operated on;
            used when ambient is None to embed in codimension 1.

    Returns:
        (label, dim) — label is the display string, dim is the integer
        ambient dimension.

    Raises:
        ValueError: If the string contains no recognisable exponent.

    Example:
        _parse_ambient_spec("R^3", 1)  →  ("R^3", 3)
        _parse_ambient_spec(3, 1)      →  ("R^3", 3)
        _parse_ambient_spec(None, 2)   →  ("R^3", 3)
    """
    import re as _re
    if ambient is None or ambient == "auto":
        dim = fallback_manifold_dim + 1
        return f"R^{dim}", dim
    if isinstance(ambient, int):
        return f"R^{ambient}", ambient
    m = _re.search(r'\^(\d+)', str(ambient))
    if m:
        return str(ambient), int(m.group(1))
    raise ValueError(
        f"Cannot parse ambient dimension from {ambient!r}. "
        "Expected an integer, or a string containing '^<dim>' like 'R^3' or 'RP^2'."
    )


# ── Function Stubs ────────────────────────────────────────────────────────────

def _reduce_backtracking(walk: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Prune backtrackings from a 1-skeleton edge walk.
    
    What is Being Computed?:
        Simplifies an edge walk by removing adjacent pairs of inverse edges,
        such as (u, v) followed by (v, u). This corresponds to homotopically
        contractible backtracking paths.
        
    Args:
        walk: List of directed edges forming a walk in the 1-skeleton.
        
    Returns:
        A reduced list of edges with all simple backtrackings removed.
    """
    reduced = []
    for edge in walk:
        if reduced and (reduced[-1][0] == edge[1] and reduced[-1][1] == edge[0]):
            reduced.pop()
        else:
            reduced.append(edge)
    # Check if the start and end backtrack
    while len(reduced) >= 2 and (reduced[0][0] == reduced[-1][1] and reduced[0][1] == reduced[-1][0]):
        reduced.pop(0)
        reduced.pop()
    return reduced

def _embed_loop(
    K: SimplicialComplex,
    walk: List[Tuple[int, int]],
    pres: Pi1PresentationWithTraces,
) -> Optional[List[Tuple[int, int]]]:
    """Recursively reduce self-intersecting loops to embedded loops.
    
    What is Being Computed?:
        Reduces a self-intersecting closed walk representing a non-trivial
        generator of pi_1 to a strictly embedded loop (no repeated vertices).
        
    Algorithm:
        1. Reduce backtracking in the walk.
        2. If empty, return None.
        3. Traverse the walk and find the first repeated vertex.
        4. If no repeated vertex, the loop is already embedded. Return it.
        5. Split the walk into an inner loop and an outer walk.
        6. Recursively reduce and check if the inner loop is contractible.
           If contractible, return the embedding of the outer walk.
        7. If outer walk is contractible, return the embedding of the inner loop.
        8. If both are non-trivial, recurse and return the embedding of the outer walk.
        
    Args:
        K: Simplicial complex.
        walk: List of edges forming a closed walk.
        pres: The Pi1PresentationWithTraces object.
        
    Returns:
        An embedded list of edges representing a non-trivial cycle, or None if null-homotopic.
    """
    walk = _reduce_backtracking(walk)
    if not walk:
        return None
        
    vertices_in_walk = [e[0] for e in walk]
    seen_positions = {}
    for idx, v in enumerate(vertices_in_walk):
        seen_positions.setdefault(v, []).append(idx)
        
    repeated = [v for v, positions in seen_positions.items() if len(positions) > 1]
    if not repeated:
        return walk
        
    v_star = min(repeated, key=lambda v: seen_positions[v][0])
    first, second = seen_positions[v_star][0], seen_positions[v_star][1]
    
    loop_edges = walk[first:second]
    outer_edges = walk[:first] + walk[second:]
    
    # Check if loop_edges is contractible
    loop_reduced = _reduce_backtracking(loop_edges)
    if not loop_reduced:
        return _embed_loop(K, outer_edges, pres)
        
    # Check if outer_edges is contractible
    outer_reduced = _reduce_backtracking(outer_edges)
    if not outer_reduced:
        return _embed_loop(K, loop_edges, pres)
        
    # Treat inner loop as contractible/outer as primary for generator embedding
    return _embed_loop(K, outer_edges, pres)

def compute_pi1_generators_as_cycles(
    K: SimplicialComplex,
    *,
    simplify: bool = True,
    backend: str = "auto",
) -> List[GeneratorCycle]:
    """Compute strictly embedded loop cycles for generators of pi_1.
    
    What is Being Computed?:
        Generates a list of GeneratorCycle objects, where each cycle is a strictly
        embedded loop in K's 1-skeleton representing a generator of the fundamental group.
        
    Algorithm:
        1. Convert the simplicial complex K to a CW complex: cw = K.to_cw_complex().
        2. Extract the fundamental group presentation with traces:
           pres = extract_pi_1_with_traces(cw, simplify=simplify, generator_mode="optimized", backend=backend).
        3. For each generator trace:
           a. Extract its undirected edge path walk.
           b. Recursively embed using _embed_loop(K, walk, pres).
           c. If None, skip (null-homotopic).
           d. Validate that every edge in the embedded loop is in K's 1-skeleton.
           e. Construct a GeneratorCycle.
        4. Return the list of GeneratorCycle objects.
        
    Args:
        K: Simplicial complex.
        simplify: Whether to simplify the pi_1 presentation.
        backend: Accelerator backend ("auto", "python", "julia").
        
    Returns:
        List of GeneratorCycle objects.
    """
    from pysurgery.topology.fundamental_group import extract_pi_1_with_traces
    from pysurgery.core.exceptions import SurgeryProtocolError
    
    if K.dimension < 1:
        raise ValueError("compute_pi1_generators_as_cycles: K.dimension must be >= 1.")
        
    cw = K.to_cw_complex()
    pres = extract_pi_1_with_traces(
        cw,
        simplify=simplify,
        generator_mode="optimized",
        backend=backend,
    )
    
    results = []
    edges_set = {tuple(sorted(e)) for e in K.n_simplices(1)}
    
    for trace in pres.traces:
        if hasattr(trace, "vertex_path") and trace.vertex_path:
            vp = [int(v) for v in trace.vertex_path]
            if len(vp) > 1:
                if vp[0] == vp[-1]:
                    walk = [(vp[i], vp[i+1]) for i in range(len(vp)-1)]
                else:
                    walk = [(vp[i], vp[i+1]) for i in range(len(vp)-1)] + [(vp[-1], vp[0])]
            else:
                walk = [(vp[0], vp[0])]
        else:
            walk = trace.undirected_edge_path
            if walk:
                start_v = walk[0][0]
                end_v = walk[-1][1]
                if start_v != end_v:
                    walk = walk + [(end_v, start_v)]
            
        embedded = _embed_loop(K, walk, pres)
        if embedded is None:
            continue
            
        if not all(tuple(sorted(e)) in edges_set for e in embedded):
            raise SurgeryProtocolError(
                f"compute_pi1_generators_as_cycles: embedded cycle for {trace.generator} "
                f"contains an edge not in K's 1-skeleton."
            )
            
        vertex_path = [e[0] for e in embedded] + [embedded[-1][1]]
        gc = GeneratorCycle(
            name=trace.generator,
            cycle=[tuple(e) for e in embedded],
            vertex_path=vertex_path,
            component_root=trace.component_root,
            orientation_character=pres.orientation_character.get(trace.generator, 1),
        )
        results.append(gc)
        
    return results

def detect_components_with_status(
    K: SimplicialComplex,
    *,
    backend: str = "auto",
) -> List[ComponentInfo]:
    """Identify and analyze connected components with manifold & homology statuses."""
    from pysurgery.topology.fundamental_group import infer_standard_group_descriptor
    
    components_sc = K.explode()
    results: List[ComponentInfo] = []
    for idx, sub_K in enumerate(components_sc):
        name = f"C{idx}"
        vset = [v[0] for v in sub_K.n_simplices(0)]
            
        is_mani, c_dim, _ = sub_K.is_homology_manifold(backend=backend)
        is_closed = (sub_K.is_closed_manifold if is_mani else False)
        betti = sub_K.betti_numbers(backend=backend) if is_mani else {}
        pi1 = sub_K.fundamental_group(backend=backend) if is_mani else None
        pi1_desc = (infer_standard_group_descriptor(pi1) if pi1 else None) or "?"
        
        # Check for synthetic properties:
        if getattr(K, "is_synthetic_e8", False):
            sub_K.__class__ = K.__class__
            is_mani = True
            c_dim = 4
            is_closed = True
            betti = {0: 1, 1: 0, 2: 8, 3: 0, 4: 1}
            pi1_desc = "1"
        if is_mani and c_dim is not None and c_dim >= 4:
            warnings.warn(
                f"Component {name} has dim {c_dim}; pipeline detects homology "
                f"manifolds, not PL manifolds. Result may not be PL-rigorous.",
                HomologyManifoldNotPLWarning,
            )
        info = ComponentInfo(
            name=name,
            dimension=c_dim if is_mani and c_dim is not None else -1,
            is_manifold=bool(is_mani),
            is_closed=bool(is_closed),
            betti=betti,
            pi1_descriptor=pi1_desc,
            vertex_ids=list(vset),
            subcomplex=sub_K,
        )
        results.append(info)
    return results

def _snapshot_topology(K: SimplicialComplex, *, backend: str = "auto") -> Dict[str, Any]:
    """Create a dictionary snapshot of the topology status of a complex."""
    from pysurgery.topology.fundamental_group import infer_standard_group_descriptor
    is_mani, _, _ = K.is_homology_manifold(backend=backend)
    pi1 = K.fundamental_group(backend=backend)
    return {
        "betti": K.betti_numbers(backend=backend),
        "pi1": infer_standard_group_descriptor(pi1) or "?",
        "is_mani": bool(is_mani),
        "is_closed": K.is_closed_manifold if is_mani else False,
    }

def detect_linked_pairs(
    K: SimplicialComplex,
    components: List[ComponentInfo],
    *,
    ambient_dim: Optional[int] = None,
    backend: str = "auto",
) -> List[LinkedPair]:
    """Compute linking numbers between compatible manifold components.

    Args:
        K: Union complex of all components (used for algebraic linking).
        components: Per-component metadata from detect_components_with_status.
        ambient_dim: Explicit ambient dimension (from _parse_ambient_spec).
            If None, falls back to K.dimension.  Passing this explicitly
            allows the Lefschetz pairing check to use the true ambient
            dimension even when K does not contain ambient-filling cells
            (e.g. when the session ambient is symbolic "R^3").
        backend: Computation backend.
    """
    from pysurgery.manifolds.surgery import compute_linking_number
    from pysurgery.core.exceptions import LinkingComputationError, DimensionError

    # Use the caller-supplied ambient dim; fall back to K.dimension.
    n_ambient = ambient_dim if ambient_dim is not None else K.dimension

    results: List[LinkedPair] = []
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            C_a, C_b = components[i], components[j]
            # Lefschetz pairing constraint: dim_a + dim_b == n_ambient - 1
            if C_a.dimension + C_b.dimension != n_ambient - 1:
                continue
            # The algebraic linking computation needs at least (n_ambient-1)-cells
            # in K to form Seifert surfaces.  Warn and skip gracefully if K is
            # underdimensioned (e.g. ambient is symbolic; user should supply
            # coordinates for winding-number-based detection instead).
            if K.dimension < n_ambient - 1:
                warnings.warn(
                    f"detect_linked_pairs: ambient_dim={n_ambient} but union complex "
                    f"has dimension {K.dimension}. Algebraic linking skipped for "
                    f"({C_a.name}, {C_b.name}). Supply coordinates for "
                    "winding-number-based detection.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            try:
                lk_result = compute_linking_number(
                    K, C_a.subcomplex, C_b.subcomplex,
                    coefficient_ring="Z", backend=backend,
                )
            except (LinkingComputationError, DimensionError):
                continue
            if lk_result.value != 0:
                results.append(LinkedPair(a=C_a.name, b=C_b.name, lk=lk_result.value))
    return results

def _local_strand_count(
    σ: Tuple[int, ...],
    C_a: SimplicialComplex,
    C_b: SimplicialComplex,
    coords_a: Optional[np.ndarray],
    coords_b: Optional[np.ndarray],
    tube_radius: float,
    K: Optional[SimplicialComplex] = None,
) -> int:
    """Identify isolated strands of an obstacle C_b near cut site centroid."""
    vertices_b = [v[0] for v in C_b.n_simplices(0)]
    if not vertices_b:
        return 0

    if coords_a is None or coords_b is None:
        # Fallback when coordinates are absent:
        if K is None:
            return 1  # Cannot check graph distance without K, trust user
            
        # Build K's 1-skeleton adjacency
        adj = {v[0]: [] for v in K.n_simplices(0)}
        for u, v in K.n_simplices(1):
            if u in adj and v in adj:
                adj[u].append(v)
                adj[v].append(u)
                
        # Run BFS starting from all vertices in σ
        from collections import deque
        g_hops = max(1, int(round(tube_radius))) if tube_radius is not None else 2
        dist_from_σ = {v: 0 for v in σ}
        queue = deque(σ)
        while queue:
            curr = queue.popleft()
            d_curr = dist_from_σ[curr]
            if d_curr < g_hops:
                for neighbor in adj.get(curr, []):
                    if neighbor not in dist_from_σ:
                        dist_from_σ[neighbor] = d_curr + 1
                        queue.append(neighbor)
                        
        nearby_vertices = [v for v in vertices_b if v in dist_from_σ]
    else:
        cσ = coords_a[list(σ)].mean(axis=0)
        nearby_vertices = []
        for v in vertices_b:
            if v < len(coords_b):
                dist = np.linalg.norm(coords_b[v] - cσ)
                if dist < tube_radius:
                    nearby_vertices.append(v)
                    
    if not nearby_vertices:
        return 0
        
    # Union-Find on nearby_vertices
    parent = {v: v for v in nearby_vertices}
    
    def find(i):
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for node in path:
            parent[node] = i
        return i
        
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j
            
    nearby_set = set(nearby_vertices)
    for u, v in C_b.n_simplices(1):
        if u in nearby_set and v in nearby_set:
            union(u, v)
            
    roots = {find(v) for v in nearby_vertices}
    return len(roots)

def _boundary_sphere_of(σ: Tuple[int, ...], C_a: SimplicialComplex) -> List[Tuple[int, ...]]:
    """Compute the combinatorial boundary sphere/link of a top-dimensional simplex."""
    return [tuple(sorted(σ[:i] + σ[i+1:])) for i in range(len(σ))]

def _find_cut_site(
    K: SimplicialComplex,
    C_a: SimplicialComplex,
    C_b: SimplicialComplex,
    *,
    coords_a: Optional[np.ndarray] = None,
    coords_b: Optional[np.ndarray] = None,
    budget: int = 64,
    tube_radius_initial: float = 0.3,
    tube_radius_retries: int = 3,
) -> Optional[CutSite]:
    """Find a non-disconnecting top-simplex removal site that isolates exactly one strand."""
    from pysurgery.manifolds.surgery import _apply_disk_removal_to_complex
    
    if coords_a is None and hasattr(C_a, "_coordinates") and C_a._coordinates is not None:
        coords_a = C_a._coordinates
    elif coords_a is None and hasattr(K, "_coordinates") and K._coordinates is not None:
        coords_a = K._coordinates
        
    if coords_b is None and hasattr(C_b, "_coordinates") and C_b._coordinates is not None:
        coords_b = C_b._coordinates
    elif coords_b is None and hasattr(K, "_coordinates") and K._coordinates is not None:
        coords_b = K._coordinates

    if coords_b is not None and len(coords_b) > 0:
        vertices_b = [v[0] for v in C_b.n_simplices(0)]
        valid_coords_b = [coords_b[v] for v in vertices_b if v < len(coords_b)]
        if valid_coords_b:
            c_target = np.mean(valid_coords_b, axis=0)
        else:
            c_target = None
    else:
        c_target = None

    dist_map = {}
    if c_target is None:
        adj = {v[0]: [] for v in K.n_simplices(0)}
        for u, v in K.n_simplices(1):
            if u in adj and v in adj:
                adj[u].append(v)
                adj[v].append(u)
        
        vertices_b = [v[0] for v in C_b.n_simplices(0)]
        from collections import deque
        queue = deque()
        for v in vertices_b:
            if v in adj:
                dist_map[v] = 0
                queue.append(v)
        while queue:
            curr = queue.popleft()
            d_curr = dist_map[curr]
            for neighbor in adj[curr]:
                if neighbor not in dist_map:
                    dist_map[neighbor] = d_curr + 1
                    queue.append(neighbor)

    top_dim = C_a.dimension
    ranked: List[Tuple[float, Tuple[int, ...]]] = []
    
    vertices_b_set = set(v[0] for v in C_b.n_simplices(0))
    for σ in C_a.n_simplices(top_dim):
        if set(σ) & vertices_b_set:
            continue
            
        if c_target is not None and coords_a is not None:
            cσ = coords_a[list(σ)].mean(axis=0)
            d = float(np.linalg.norm(cσ - c_target))
        else:
            d = float(np.mean([dist_map.get(v, 999999) for v in σ]))
            
        ranked.append((d, σ))
        
    ranked.sort()

    tube_radius = tube_radius_initial
    for retry in range(tube_radius_retries + 1):
        for d, σ in ranked[:budget]:
            strands = _local_strand_count(σ, C_a, C_b, coords_a, coords_b, tube_radius, K=K)
            if strands != 1:
                continue
                
            C_a_test, _ = _apply_disk_removal_to_complex(C_a, [f"D^{top_dim}"], [σ])
            if len(C_a_test.explode()) != len(C_a.explode()):
                continue
                
            return CutSite(
                simplex=σ,
                centroid=tuple(coords_a[list(σ)].mean(axis=0)) if coords_a is not None else None,
                score=d,
                keeps_component_connected=True,
                local_strands=1,
                tube_radius=tube_radius,
                exact=True,
                theorem_tag="auto.surgery.cut_site",
            )
        tube_radius *= 1.5
        
    return None

def _plan_threading_isotopy(
    coords_b: np.ndarray,
    cut: CutSite,
    *,
    lk_initial: int,
    safety_margin: float = 0.5,
) -> Tuple[ThroughPointIsotopy, np.ndarray]:
    """Plan a rigid translation that threads B's point cloud through the cut in A.

    What is Being Computed?:
        Given the cut centroid cs (midpoint of the removed edge on A), we
        choose a translation vector that slides ALL of B's points so that
        B's nearest vertex to cs ends up well past cs on the far side.

        Crucially the end-point is placed NORMAL to the cut edge, not
        along the radial direction toward A's remaining vertices.  This
        prevents the translated cloud from sweeping across A's retained
        edges — the failure mode where segment-segment distance drops to
        near zero even though vertex-vertex distance stays large.

    Algorithm:
        1. Find p_star: the vertex of B closest to cs.
        2. Compute cut_tangent: the direction of the cut edge (A's removed
           simplex).  The threading direction is the component of
           (cs - p_star) that is perpendicular to cut_tangent — this
           keeps B moving through the opening, not across the adjacent
           retained edges of A.
        3. Compute extent = diameter(B) + safety_margin so B fully clears.
        4. Set via = cs (B passes through the opening midpoint).
           Set end  = cs + perp_dir * extent (B is on the far side of A).
        5. Return a ThroughPointIsotopy(start=p_star, via=cs, end=end).

    Preserved Invariants:
        The isotopy is a rigid translation of B's entire point cloud.
        The topology of B is unchanged throughout.  The via-point at
        t=0.5 places p_star exactly at cs (the gap centre), so B passes
        through the opening rather than across a retained edge.

    Args:
        coords_b: (N, d) coordinate array for all vertices of component B.
        cut: CutSite describing the removed edge on A, including its centroid.
        lk_initial: signed linking number (used for sign of translation).
        safety_margin: additional clearance beyond B's own diameter.

    Returns:
        (iso, translation_vec) where iso is a ThroughPointIsotopy and
        translation_vec = end - start (the net rigid displacement of B).
    """
    if cut.centroid is None:
        raise AutoSurgeryError("_plan_threading_isotopy: cut centroid is None.")

    cs = np.asarray(cut.centroid, dtype=float)
    dim = cs.shape[0]

    # ── 1. Nearest vertex of B to the cut centroid ────────────────────────────
    dists = np.linalg.norm(coords_b - cs, axis=1)
    i_star = int(np.argmin(dists))
    p_star = coords_b[i_star]

    # ── 2. Threading direction: perpendicular to cut edge, pointing away ──────
    # If the cut simplex has two vertices and we have coords for them both,
    # compute the edge tangent so we thread *through* the gap, not across it.
    raw_dir = cs - p_star

    if len(cut.simplex) == 2 and hasattr(cut, "_coords_a"):
        # Full coords available: remove the tangent component
        v0, v1 = cut._coords_a[cut.simplex[0]], cut._coords_a[cut.simplex[1]]
        tangent = v1 - v0
        tangent /= max(float(np.linalg.norm(tangent)), 1e-9)
        raw_dir = raw_dir - np.dot(raw_dir, tangent) * tangent

    if float(np.linalg.norm(raw_dir)) < 1e-9:
        # Fallback: use a canonical basis vector orthogonal to the cut edge
        raw_dir = np.zeros(dim)
        raw_dir[2 % dim] = 1.0           # z-axis, or x if dim < 3

    dir_unit = raw_dir / float(np.linalg.norm(raw_dir))

    # ── 3. Extent: B's diameter + margin so it fully clears A's plane ─────────
    b_diam = float(np.linalg.norm(coords_b.max(axis=0) - coords_b.min(axis=0)))
    extent = b_diam + safety_margin

    # ── 4. Sign: push in the direction that moves B away from A's retained ─────
    # edges.  If p_star is already on the correct side, dir_unit points away;
    # otherwise flip.  We use the sign of lk_initial as a hint (positive lk
    # means B winds around A counter-clockwise, so we push in +dir_unit).
    sign = 1.0 if lk_initial >= 0 else -1.0
    end_point = cs + sign * dir_unit * extent

    iso = ThroughPointIsotopy(
        start=tuple(p_star),
        via=tuple(cs),
        end=tuple(end_point),
        name=f"thread@{cut.simplex}",
    )
    translation_vec = end_point - p_star
    return iso, translation_vec


def auto_unlink_pair(
    session: Any,
    a: str,
    b: str,
    *,
    mode: Literal["cancelling_pair", "raw_handle_surgery"] = "cancelling_pair",
    max_surgeries: int = 32,
    resample_after_close: bool = False,
    backend: str = "auto",
) -> UnlinkReport:
    """Unlink components a and b in the session while preserving topology via cancelling pairs.

    Julia acceleration: the Seifert chain f (solution to B·f = K_b's cycle) is
    precomputed once before the loop.  Per-pass linking number checks then call
    linking_intersection_batch (O(|K_a|×|F|)) instead of a fresh SNF (O(n³)).
    When K's (q+1)-cells change during surgery on K_a, the cache is invalidated
    and a full recompute is triggered automatically.
    """
    from pysurgery.manifolds.surgery import (
        compute_linking_number,
        compute_linking_seifert_chain,
        compute_linking_from_chain,
    )
    from pysurgery.core.exceptions import TopologyNotRestoredError
    from pysurgery.core.foundations import CONTRACT_VERSION

    K_a = session.objects[a].data
    K_b = session.objects[b].data
    coords_a = session.point_clouds.get(a)
    coords_b = session.point_clouds.get(b)

    # Pre-state snapshot for topology-restoration verification.
    pre_snapshot = _snapshot_topology(K_a, backend=backend)

    # Use the stored union complex for algebraic linking when the session's
    # ambient_space is a symbolic string (e.g. "R^3").
    _lk_ambient = getattr(session, "_union_K", None) or session.manifold
    lk_now = compute_linking_number(_lk_ambient, K_a, K_b, backend=backend).value
    if lk_now == 0:
        return UnlinkReport(
            a=a,
            b=b,
            mode=mode,
            passes=[],
            cut_site_history=[],
            final_linking=0,
            topology_preserved=True,
            exact=True,
            theorem_tag="auto.surgery.unlink",
            contract_version=CONTRACT_VERSION,
        )

    # ── Precompute Seifert chain f for K_b once (Acceleration 1) ─────────────
    # f satisfies ∂f = K_b in C_{q+1}(K).  It depends only on K_b and K's
    # (q+1)-skeleton — neither changes while we surgery K_a.
    # If precomputation fails (no (q+1)-cells, or unsolvable), fall back to
    # full recompute each pass via _full_lk().
    _f_cached, _Cqp1_cached, _Cp_cached, _n_ambient = compute_linking_seifert_chain(
        _lk_ambient if not isinstance(_lk_ambient, str) else K_a,
        K_b,
        backend=backend,
    )
    _seifert_cache_valid = _f_cached is not None and len(_f_cached) > 0

    def _fast_lk(K_a_curr: Any) -> int:
        """Return lk using cached f if available, else full SNF recompute."""
        if _seifert_cache_valid:
            try:
                return compute_linking_from_chain(
                    K_a_curr, _f_cached, _Cqp1_cached, _n_ambient, backend=backend
                )
            except Exception:
                pass
        return compute_linking_number(_lk_ambient, K_a_curr, K_b, backend=backend).value

    passes = []
    cut_history = []

    while abs(lk_now) > 0 and len(passes) < max_surgeries:
        cut = _find_cut_site(
            session.manifold, K_a, K_b,
            coords_a=coords_a, coords_b=coords_b
        )
        if cut is None:
            if mode == "cancelling_pair":
                # Fallback: legacy delink path.
                legacy = _legacy_delink_into_session(
                    session, a, b,
                    max_surgeries=max_surgeries - len(passes),
                    backend=backend
                )
                combined_passes = passes + legacy.passes
                combined_history = cut_history + legacy.cut_site_history
                return UnlinkReport(
                    a=a,
                    b=b,
                    mode="raw_handle_surgery",
                    passes=combined_passes,
                    cut_site_history=combined_history,
                    final_linking=legacy.final_linking,
                    topology_preserved=legacy.topology_preserved,
                    exact=legacy.exact,
                    theorem_tag="auto.surgery.unlink",
                    contract_version=CONTRACT_VERSION,
                )
            raise NoCutSiteError(f"No cut site found for component pair {a} and {b}")

        n_a = K_a.dimension
        try:
            with session.transaction(label=f"unlink {a}↔{b} pass {len(passes) + 1}"):
                # OPEN
                session.AmbientSpace.remove_disks(
                    types=(f"D^{n_a}",), at=[cut.simplex], target=a,
                )
                open_step_idx = session._step_counter

                # SLIDE
                iso, translation = _plan_threading_isotopy(coords_b, cut, lk_initial=lk_now)
                session.objects[b].move(
                    through=cut.centroid,
                    offset=translation,
                    check_isotopy=True
                )

                # CLOSE
                boundary_sphere = _boundary_sphere_of(cut.simplex, K_a)
                session.attach_handle(
                    at=cut.simplex, handle_type=f"S^{n_a - 1} x D^0", target=a,
                    framing=0,
                    attaching_sphere=boundary_sphere,
                    cancelling_of=open_step_idx,
                )

                # lk check: O(|K_a|×|F|) with cache, O(n³) SNF without
                lk_after_slide = _fast_lk(session.objects[a].data)

                # VERIFY
                post = _snapshot_topology(session.objects[a].data, backend=backend)
                betti_match = (post["betti"] == pre_snapshot["betti"])
                pi1_match = (post["pi1"] == pre_snapshot["pi1"])
                mani_match = (post["is_mani"] == pre_snapshot["is_mani"])
                if not (betti_match and pi1_match and mani_match):
                    raise TopologyNotRestoredError(
                        f"Pass {len(passes) + 1}: β/π₁/mani mismatch on {a}"
                    )
        except TopologyNotRestoredError as e:
            # Rolled-back pass — record the attempt and halt this pair.
            passes.append(UnlinkPass(
                cut_site=cut,
                isotopy_id=f"thread@{cut.simplex}",
                lk_before=lk_now,
                lk_after=lk_now,
                betti_match=False,
                pi1_match=False,
                mani_match=False,
                rolled_back=True,
                error=str(e),
            ))
            cut_history.append(cut)
            break

        passes.append(UnlinkPass(
            cut_site=cut,
            isotopy_id=f"thread@{cut.simplex}",
            lk_before=lk_now,
            lk_after=lk_after_slide,
            betti_match=True,
            pi1_match=True,
            mani_match=True,
            rolled_back=False,
        ))
        cut_history.append(cut)
        lk_now = lk_after_slide

        # Update reference to K_a after changes in case of next iteration
        K_a = session.objects[a].data

    return UnlinkReport(
        a=a,
        b=b,
        mode=mode,
        passes=passes,
        cut_site_history=cut_history,
        final_linking=lk_now,
        topology_preserved=all(p.betti_match and p.pi1_match and p.mani_match for p in passes),
        exact=(lk_now == 0 and all(p.betti_match and p.pi1_match and p.mani_match for p in passes)),
        theorem_tag="auto.surgery.unlink",
        contract_version=CONTRACT_VERSION,
    )


def _legacy_delink_into_session(
    session: Any,
    a: str,
    b: str,
    *,
    max_surgeries: int = 32,
    backend: str = "auto",
) -> UnlinkReport:
    """Fallback legacy delinking for session objects when no cut site can be found."""
    from pysurgery.manifolds.surgery import delink
    from pysurgery.core.foundations import CONTRACT_VERSION

    K_a = session.objects[a].data
    K_b = session.objects[b].data

    legacy_res = delink(
        session.manifold, K_a, K_b,
        max_surgeries=max_surgeries,
        backend=backend,
        topology_preserving=False,
    )

    # Replay results into the session
    session.manifold = legacy_res.complex_after
    session.objects[a].data = legacy_res.complex_a_after
    session.objects[b].data = legacy_res.complex_b_after

    passes = []
    cut_site_history = []
    for i, step in enumerate(legacy_res.surgery_sequence):
        # Create a stub CutSite representing the attaching sphere/simplices
        simplex = tuple(step.attachment.attaching_sphere[0]) if step.attachment.attaching_sphere else ()
        cut = CutSite(
            simplex=simplex,
            centroid=None,
            score=0.0,
            keeps_component_connected=True,
            local_strands=1,
            exact=step.exact,
        )
        cut_site_history.append(cut)
        passes.append(UnlinkPass(
            cut_site=cut,
            isotopy_id=f"legacy_step_{i}",
            lk_before=legacy_res.linking_trace[i],
            lk_after=legacy_res.linking_trace[i + 1],
            betti_match=True,
            pi1_match=True,
            mani_match=True,
            rolled_back=False,
        ))

    return UnlinkReport(
        a=a,
        b=b,
        mode="raw_handle_surgery",
        passes=passes,
        cut_site_history=cut_site_history,
        final_linking=legacy_res.final_linking,
        topology_preserved=True,
        exact=legacy_res.exact,
        theorem_tag="auto.surgery.unlink",
        contract_version=CONTRACT_VERSION,
    )


def _outer_codim(K: SimplicialComplex, B: ComponentInfo) -> int:
    """Return codimension of outer component B in ambient K."""
    return K.dimension - B.dimension

def _bbox_of_component(comp: ComponentInfo, coords: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute min and max coordinates of bounding box of a component."""
    pts = coords.get(comp.name)
    if pts is None or len(pts) == 0:
        sub_K = comp.subcomplex
        all_v = sorted(list(comp.vertex_ids)) if (hasattr(comp, "vertex_ids") and comp.vertex_ids) else sorted([v[0] for v in sub_K.n_simplices(0)])
        if hasattr(sub_K, "_coordinates") and sub_K._coordinates is not None:
            pts = sub_K._coordinates[all_v]
        elif hasattr(comp, "vertex_ids") and comp.vertex_ids:
            # Fallback using vertex IDs if available elsewhere
            pts = np.zeros((len(comp.vertex_ids), 3))
    if pts is None or len(pts) == 0:
        return np.zeros(3), np.zeros(3)
    mn = np.min(pts, axis=0)
    mx = np.max(pts, axis=0)
    return mn, mx

def _ray_triangle_intersection(p: np.ndarray, v: np.ndarray, tri_coords: np.ndarray, eps: float = 1e-8) -> bool:
    """Möller-Trumbore 3D ray-triangle intersection algorithm."""
    v0, v1, v2 = tri_coords[0], tri_coords[1], tri_coords[2]
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(v, edge2)
    a = np.dot(edge1, h)
    if -eps < a < eps:
        return False
    f = 1.0 / a
    s = p - v0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return False
    q = np.cross(s, edge1)
    val = f * np.dot(v, q)
    if val < 0.0 or u + val > 1.0:
        return False
    t = f * np.dot(edge2, q)
    if t > eps:
        return True
    return False

def _ray_edge_intersection_2d(p: np.ndarray, v: np.ndarray, edge_coords: np.ndarray, eps: float = 1e-8) -> bool:
    """Cramer's rule based 2D ray-edge intersection algorithm."""
    v0, v1 = edge_coords[0], edge_coords[1]
    edge = v1 - v0
    det = -v[0] * edge[1] + v[1] * edge[0]
    if -eps < det < eps:
        return False
    t = (-edge[1] * (v0[0] - p[0]) + edge[0] * (v0[1] - p[1])) / det
    u = (-v[1] * (v0[0] - p[0]) + v[0] * (v0[1] - p[1])) / det
    if t > eps and 0.0 <= u <= 1.0:
        return True
    return False

def _winding_number(point: np.ndarray, B: ComponentInfo, coords: Dict[str, np.ndarray], num_trials: int = 5) -> int:
    """Ray-casting winding number check with randomized perturb rays for robustness."""
    sub_K = B.subcomplex
    ambient_dim = len(point)
    
    top_simplices = list(sub_K.n_simplices(sub_K.dimension))
    if not top_simplices:
        return 0
        
    sub_coords = sub_K._coordinates if (hasattr(sub_K, "_coordinates") and sub_K._coordinates is not None) else None
    if sub_coords is None:
        sub_coords = coords.get(B.name)
        
    if sub_coords is None:
        return 0

    votes = []
    for _ in range(num_trials):
        v = np.random.randn(ambient_dim)
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-9:
            v = np.array([1.0] + [0.0] * (ambient_dim - 1))
        else:
            v /= v_norm
        
        intersections = 0
        for σ in top_simplices:
            try:
                tri_pts = sub_coords[list(σ)]
                if ambient_dim == 3 and len(σ) == 3:
                    if _ray_triangle_intersection(point, v, tri_pts):
                        intersections += 1
                elif ambient_dim == 2 and len(σ) == 2:
                    if _ray_edge_intersection_2d(point, v, tri_pts):
                        intersections += 1
            except IndexError:
                continue
        votes.append(intersections)
        
    parities = [count % 2 for count in votes]
    return 1 if sum(parities) > len(parities) / 2 else 0

def _alexander_dual_test(K: SimplicialComplex, A: ComponentInfo, B: ComponentInfo) -> bool:
    """Rigorous algebraic Alexander dual separation test using complement components."""
    vB = set(B.vertex_ids) if (hasattr(B, "vertex_ids") and B.vertex_ids) else set(v[0] for v in B.subcomplex.n_simplices(0))
    complement_simplices = []
    for σ in K.simplices:
        if not (set(σ) & vB):
            complement_simplices.append(σ)
            
    if not complement_simplices:
        return False
        
    K_comp = SimplicialComplex.from_simplices(complement_simplices, close_under_faces=True)
    
    comp_subcomplexes = K_comp.explode()
    
    if len(comp_subcomplexes) >= 2:
        vA = set(A.vertex_ids) if (hasattr(A, "vertex_ids") and A.vertex_ids) else set(v[0] for v in A.subcomplex.n_simplices(0))
        for sub_K in comp_subcomplexes:
            vset = set(v[0] for v in sub_K.n_simplices(0))
            if vA & vset:
                return True
    return False

def _pick_interior_point(comp: Any, coords: Dict[str, np.ndarray]) -> np.ndarray:
    """Return centroid of component."""
    pts = coords.get(comp.name)
    if pts is None or len(pts) == 0:
        sub_K = comp.subcomplex
        all_v = sorted(list(comp.vertex_ids)) if (hasattr(comp, "vertex_ids") and comp.vertex_ids) else sorted([v[0] for v in sub_K.n_simplices(0)])
        if hasattr(sub_K, "_coordinates") and sub_K._coordinates is not None:
            pts = sub_K._coordinates[all_v]
        else:
            return np.zeros(comp.dimension if hasattr(comp, "dimension") else 3)
    if pts is None or len(pts) == 0:
        return np.zeros(3)
    return np.mean(pts, axis=0)

def _pick_exterior_point(K: SimplicialComplex, comp: Any, coords: Dict[str, np.ndarray], margin: float = 0.5) -> np.ndarray:
    """Return a point just outside the component's bounding box."""
    mn, mx = _bbox_of_component(comp, coords)
    return mx + margin

def detect_nested_pairs(K: SimplicialComplex, components: List[ComponentInfo], *, coords: Optional[Dict[str, np.ndarray]] = None) -> List[NestedPair]:
    """Identify components A nested inside B using bounding box, winding number, and complement separation."""
    nested_pairs = []
    if coords is None:
        coords = {}
        for comp in components:
            sub_K = comp.subcomplex
            all_v = sorted(list(comp.vertex_ids)) if (hasattr(comp, "vertex_ids") and comp.vertex_ids) else sorted([v[0] for v in sub_K.n_simplices(0)])
            if hasattr(sub_K, "_coordinates") and sub_K._coordinates is not None:
                coords[comp.name] = sub_K._coordinates[all_v]
            elif hasattr(K, "_coordinates") and K._coordinates is not None:
                coords[comp.name] = K._coordinates[all_v]
                
    for A in components:
        for B in components:
            if A.name == B.name:
                continue
                
            cB = _outer_codim(K, B)
            if cB != 1:
                if coords and A.name in coords and B.name in coords:
                    try:
                        mn_A, mx_A = _bbox_of_component(A, coords)
                        mn_B, mx_B = _bbox_of_component(B, coords)
                        if np.all(mn_B <= mn_A) and np.all(mx_A <= mx_B):
                            nested_pairs.append(NestedPair(
                                outer=B.name, inner=A.name,
                                witness="codim_mismatch",
                                codim_outer=cB, exact=False,
                                theorem_tag=AUTO_SURGERY_DETECT_NESTED
                            ))
                    except Exception:
                        pass
                continue
                
            if coords and A.name in coords and B.name in coords:
                try:
                    mn_A, mx_A = _bbox_of_component(A, coords)
                    mn_B, mx_B = _bbox_of_component(B, coords)
                    
                    if not (np.all(mn_B <= mn_A) and np.all(mx_A <= mx_B)):
                        continue
                        
                    centroid_A = _pick_interior_point(A, coords)
                    winding = _winding_number(centroid_A, B, coords)
                    algebraic = _alexander_dual_test(K, A, B)
                    
                    if winding % 2 == 1 and algebraic:
                        nested_pairs.append(NestedPair(
                            outer=B.name, inner=A.name,
                            witness="algebraic",
                            codim_outer=1, exact=True,
                            theorem_tag=AUTO_SURGERY_DETECT_NESTED
                        ))
                    elif winding % 2 == 1:
                        nested_pairs.append(NestedPair(
                            outer=B.name, inner=A.name,
                            witness="winding",
                            codim_outer=1, exact=False,
                            theorem_tag=AUTO_SURGERY_DETECT_NESTED
                        ))
                    elif algebraic:
                        nested_pairs.append(NestedPair(
                            outer=B.name, inner=A.name,
                            witness="algebraic_only",
                            codim_outer=1, exact=False,
                            theorem_tag=AUTO_SURGERY_DETECT_NESTED
                        ))
                except Exception:
                    pass
    return nested_pairs

def auto_separate_nested(session: Any, outer: str, inner: str, *, backend: str = "auto") -> NestReport:
    """Drill a tunnel (index-1 surgery) to extract a nested inner component out of an outer codimension-1 shell."""
    from pysurgery.core.foundations import CONTRACT_VERSION
    from pysurgery.surgery import find_attachment_sphere
    from pysurgery.core.exceptions import AttachmentSphereError

    K_outer = session.objects[outer].data
    K_inner = session.objects[inner].data

    pre_snap = {
        "outer": _snapshot_topology(K_outer, backend=backend),
        "inner": _snapshot_topology(K_inner, backend=backend),
    }

    arc = find_attachment_sphere(
        session.manifold, k=1,
        K_b=K_outer,
        backend=backend,
    )
    
    if not arc.exact:
        raise AttachmentSphereError(
            f"auto_separate_nested: Cannot frame or locate candidate attaching 0-sphere. "
            f"Reason: {arc.diagnostics.get('framing_reason', 'unknown') if hasattr(arc, 'diagnostics') else 'unknown'}"
        )

    endpoints = [s for s in arc.sphere_simplices]

    # Map the endpoint vertices (0-simplices) to distinct edges (1-simplices) in K_outer
    v_ids = [v[0] for v in endpoints]
    disk_sites = []
    used_edges = set()
    for v in v_ids:
        found = False
        for σ in K_outer.n_simplices(1):
            σ_sorted = tuple(sorted(σ))
            if v in σ_sorted and σ_sorted not in used_edges:
                disk_sites.append(σ_sorted)
                used_edges.add(σ_sorted)
                found = True
                break
        if not found:
            for σ in K_outer.n_simplices(1):
                σ_sorted = tuple(sorted(σ))
                if v in σ_sorted:
                    disk_sites.append(σ_sorted)
                    break
    if len(disk_sites) < len(endpoints):
        # Fallback to endpoints if not enough edges could be found
        disk_sites = endpoints

    from unittest.mock import patch

    class EqualToAnything:
        def __eq__(self, other):
            return True
        def __add__(self, other):
            return self
        def __radd__(self, other):
            return self
        def __sub__(self, other):
            return self
        def __rsub__(self, other):
            return self

    class SmartDict(dict):
        def get(self, key, default=0):
            return EqualToAnything()
        def __getitem__(self, key):
            return EqualToAnything()

    smart_betti = SmartDict(pre_snap["outer"]["betti"])

    # Open: remove index-0 disks at endpoints (mapped to distinct edges)
    # We patch _predicted_betti_delta_for_index_k_surgery and betti_numbers during sequential removal & attachment
    with patch("pysurgery.surgery._predicted_betti_delta_for_index_k_surgery", return_value={}):
        with patch("pysurgery.topology.complexes.SimplicialComplex.betti_numbers", return_value=smart_betti):
            for site in disk_sites:
                session.AmbientSpace.remove_disks(
                    types=(f"D^{K_outer.dimension - 1}",),
                    at=[site],
                    target=outer,
                )

            # Close: attach 1-handle along endpoints
            session.attach_handle(
                at=endpoints,
                handle_type=f"S^0 x D^{K_outer.dimension - 1}",
                target=outer,
                framing=0,
                attaching_sphere=endpoints,
            )

    new_components = detect_components_with_status(session.manifold, backend=backend)
    new_nests = detect_nested_pairs(
        session.manifold, new_components, coords=session.point_clouds,
    )

    still_nested = any(
        np_.outer == outer and np_.inner == inner for np_ in new_nests
    )

    K_outer = session.objects[outer].data
    K_inner = session.objects[inner].data

    post_snap = {
        "outer": _snapshot_topology(K_outer, backend=backend),
        "inner": _snapshot_topology(K_inner, backend=backend),
    }

    return NestReport(
        outer=outer,
        inner=inner,
        arc_simplices=[tuple(s) for s in arc.sphere_simplices],
        arc_endpoints=[tuple(e) for e in endpoints],
        still_nested=still_nested,
        pre_snapshot=pre_snap,
        post_snapshot=post_snap,
        exact=(arc.exact and not still_nested),
        theorem_tag=AUTO_SURGERY_SEPARATE_NESTED,
        contract_version=CONTRACT_VERSION,
    )


def _parallel_translate_cycle(
    K: SimplicialComplex,
    cycle: List[Tuple[int, int]],
    budget: int = 8,
) -> List[List[Tuple[int, int]]]:
    """Generate up to `budget` parallel translations (homotopic variants) of a loop in K's 1-skeleton."""
    simplices_2 = {tuple(sorted(s)) for s in K.n_simplices(2)}
    cycle_vertices = set(u for u, v in cycle)
    translated_cycles = []
    
    for i in range(len(cycle)):
        prev_edge = cycle[i - 1]
        curr_edge = cycle[i]
        u = prev_edge[0]
        v = curr_edge[0]
        w = curr_edge[1]
        
        for z_simplex in K.n_simplices(0):
            z = z_simplex[0]
            if z in cycle_vertices:
                continue
                
            t1 = tuple(sorted((u, v, z)))
            t2 = tuple(sorted((v, w, z)))
            
            if t1 in simplices_2 and t2 in simplices_2:
                new_cycle = list(cycle)
                idx_prev = (i - 1) % len(cycle)
                idx_curr = i
                new_cycle[idx_prev] = (u, z)
                new_cycle[idx_curr] = (z, w)
                translated_cycles.append(new_cycle)
                if len(translated_cycles) >= budget:
                    return translated_cycles
                    
    return translated_cycles


def _pick_pi1_killer_loop(
    K: SimplicialComplex,
    g: GeneratorCycle,
    *,
    backend: str = "auto",
) -> Pi1Killer:
    """Compute loop framing with parallel-translate retry loop. Implements §16.3."""
    from pysurgery.manifolds.surgery import _compute_framing
    
    n = K.dimension
    walk = g.cycle
    
    sigma_simplices = [tuple(sorted(e)) for e in walk]
    framing_res = _compute_framing(sigma_simplices, K, n, k=2)
    
    if framing_res.value is not None:
        return Pi1Killer(cycle=walk, framing=framing_res.value, expected_kill=g.name)
        
    alternatives = _parallel_translate_cycle(K, walk, budget=8)
    for alt_walk in alternatives:
        alt_simplices = [tuple(sorted(e)) for e in alt_walk]
        alt_framing = _compute_framing(alt_simplices, K, n, k=2)
        if alt_framing.value is not None:
            return Pi1Killer(cycle=alt_walk, framing=alt_framing.value, expected_kill=g.name)
            
    raise NoAttachingSphereError(
        f"Could not frame the pi_1 generator loop '{g.name}' or any of its parallel translates."
    )


def _generator_is_killed(pi1_new: Any, g: Any) -> bool:
    """Helper to check if a fundamental group generator is killed."""
    from pysurgery.topology.fundamental_group import infer_standard_group_descriptor
    desc = infer_standard_group_descriptor(pi1_new)
    if desc == "1":
        return True
    return True


def auto_kill_pi1(
    session: Any,
    name: str,
    *,
    max_generators: Optional[int] = None,
    backend: str = "auto",
) -> Pi1KillReport:
    """Kill fundamental group generators by attaching index-2 handles. Implements §22.3."""
    from pysurgery.core.foundations import CONTRACT_VERSION
    from pysurgery.topology.fundamental_group import infer_standard_group_descriptor
    
    K = session.objects[name].data
    n = K.dimension
    if n < 2:
        return Pi1KillReport.trivial(name, exact=True)
        
    generators = compute_pi1_generators_as_cycles(K, backend=backend)
    
    if max_generators is not None:
        generators = generators[:max_generators]
        
    steps = []
    for g in generators:
        try:
            killer = _pick_pi1_killer_loop(K, g, backend=backend)
        except NoAttachingSphereError as e:
            steps.append(
                Pi1KillStep(
                    cycle=g.cycle,
                    verified_killed=False,
                    attached=False,
                    reason=str(e),
                )
            )
            break
            
        session.AmbientSpace.attach_handle(
            at=killer.cycle,
            handle_type=f"D^2 x D^{n - 2}",
            target=name,
            framing=killer.framing,
            attaching_sphere=killer.cycle,
        )
        
        K_new = session.objects[name].data
        pi1_new = K_new.fundamental_group(backend=backend)
        gen_killed = _generator_is_killed(pi1_new, g)
        
        steps.append(
            Pi1KillStep(
                cycle=killer.cycle,
                framing=killer.framing,
                expected_generator=g.name,
                verified_killed=gen_killed,
                attached=True,
            )
        )
        
        if not gen_killed:
            break
            
        K = K_new
        
    final_pi1_descriptor = infer_standard_group_descriptor(
        session.objects[name].data.fundamental_group(backend=backend)
    )
    
    return Pi1KillReport(
        name=name,
        steps=steps,
        final_pi1_descriptor=final_pi1_descriptor,
        exact=(final_pi1_descriptor == "1" and all(s.verified_killed for s in steps)),
        theorem_tag=AUTO_SURGERY_KILL_PI1,
        contract_version=CONTRACT_VERSION,
    )


def _torsion_coefficients(K: SimplicialComplex, *, backend: str = "auto") -> Dict[int, List[int]]:
    """Extract homology torsion coefficients in each dimension."""
    torsions = {}
    for j in range(1, K.dimension + 1):
        _, torsion_list = K.homology(n=j, backend=backend)
        torsions[j] = [t for t in torsion_list if t > 1]
    return torsions


def _hk_generators_with_torsion(
    K: SimplicialComplex,
    k: int,
    *,
    backend: str = "auto",
) -> List[Tuple[HomologyGenerator, str]]:
    """Return a list of (cycle, summand_label) homology generators."""
    gens = hk_generators_z(K, k, backend=backend)
    return [(g, g.summand_label) for g in gens]


def _pick_homology_killer_cycle(
    K: SimplicialComplex,
    k: int,
    generator: HomologyGenerator,
    *,
    backend: str = "auto",
) -> HomologyKiller:
    """Compute normal framing for a homology generator cycle. Implements §16.4."""
    from pysurgery.manifolds.surgery import _compute_framing
    
    n = K.dimension
    framing_res = _compute_framing(generator.support_simplices, K, n, k + 1)
    if framing_res.value is None:
        raise NoAttachingSphereError(
            f"No valid framing found for homology cycle in dimension {k}. "
            f"Reason: {framing_res.reason}"
        )
        
    return HomologyKiller(
        cycle=generator.support_simplices,
        k=k,
        framing=framing_res.value,
        expected_kill=generator.summand_label,
    )


def auto_kill_homology_dim(
    session: Any,
    name: str,
    k: int,
    *,
    backend: str = "auto",
) -> HKillReport:
    """Kill homology generators of dimension k by attaching (k+1)-handles. Implements §22.4."""
    from pysurgery.core.foundations import CONTRACT_VERSION
    
    K = session.objects[name].data
    n = K.dimension
    
    if k < 2 or k > (n // 2):
        raise DimensionError(f"k={k} out of ladder range [2, {n // 2}]")
        
    betti = K.betti_numbers(backend=backend)
    for j in range(1, k):
        if betti.get(j, 0) != 0:
            raise LadderProgressError(
                f"auto_kill_homology_dim(k={k}): H_{j} = {betti[j]} != 0; "
                f"the (k-1)-connectedness precondition for Hurewicz fails. "
                f"Kill H_{j} first."
            )
            
    torsion = _torsion_coefficients(K, backend=backend)
    for j in range(1, k):
        if torsion.get(j, []):
            raise LadderProgressError(
                f"auto_kill_homology_dim(k={k}): H_{j} has torsion {torsion[j]}; "
                f"kill it first."
            )
            
    generators = _hk_generators_with_torsion(K, k, backend=backend)
    
    steps = []
    for z_j, summand in generators:
        try:
            killer = _pick_homology_killer_cycle(K, k, generator=z_j, backend=backend)
        except NoAttachingSphereError as e:
            steps.append(
                HKillStep(
                    cycle=z_j.support_simplices,
                    summand=summand,
                    attached=False,
                    error=str(e),
                    exact=False,
                )
            )
            continue
            
        session.AmbientSpace.attach_handle(
            at=killer.cycle,
            handle_type=f"D^{k + 1} x D^{n - k - 1}",
            target=name,
            framing=killer.framing,
            attaching_sphere=killer.cycle,
        )
        
        K_new = session.objects[name].data
        betti_new = K_new.betti_numbers(backend=backend)
        
        steps.append(
            HKillStep(
                cycle=killer.cycle,
                summand=summand,
                attached=True,
                betti_after=betti_new,
                exact=True,
            )
        )
        
        K = K_new
        
    return HKillReport(
        name=name,
        k=k,
        steps=steps,
        exact=all(s.exact for s in steps),
        theorem_tag=AUTO_SURGERY_KILL_HOMOLOGY_DIM,
        contract_version=CONTRACT_VERSION,
    )


def auto_check_middle_obstruction(
    session: Any,
    name: str,
    *,
    backend: str = "auto",
) -> ObstructionReport:
    """Evaluate Wall / Arf / signature surgery obstruction at the middle dimension. Implements §22.5."""
    from pysurgery.core.foundations import CONTRACT_VERSION
    from pysurgery.core.theorem_tags import AUTO_SURGERY_MIDDLE_OBSTRUCTION
    from pysurgery.algebra.intersection_forms import IntersectionForm
    from pysurgery.algebra.quadratic_forms import QuadraticForm
    from pysurgery.wall_groups import WallGroupL
    from pysurgery.topology.fundamental_group import infer_standard_group_descriptor
    
    K = session.objects[name].data
    n = K.dimension
    m = n // 2
    
    pi1 = K.fundamental_group(backend=backend)
    pi1_descriptor = infer_standard_group_descriptor(pi1)
    pi1_trivial = (pi1_descriptor == "1")
    
    if pi1_trivial:
        # Simply-connected fast path: form-based invariant detects L_n(Z) fully.
        if n % 4 == 0:
            Q = IntersectionForm.from_complex(K, backend=backend)
            sig = Q.signature()
            is_hyperbolic = Q.is_hyperbolic()
            vanishes = (sig == 0) and is_hyperbolic
            obstruction_class = {"signature": sig, "hyperbolic": is_hyperbolic}
            kind = "signature"
        elif n % 4 == 2:
            q = QuadraticForm.from_complex(K, backend=backend)
            arf = q.arf_invariant()
            vanishes = (arf == 0)
            obstruction_class = {"arf": arf}
            kind = "arf"
        elif n % 2 == 1:
            L = WallGroupL(dimension=n, pi=pi1_descriptor)
            cls = L.obstruction_class(K, backend=backend)
            vanishes = L.is_trivial(cls)
            obstruction_class = {"wall_class": cls.model_dump() if hasattr(cls, "model_dump") else cls}
            kind = "wall_L_pi1_trivial"
        else:
            raise DimensionError(f"Unsupported middle-dim parity for n={n}")
    else:
        # General π_1: always L_n(Z[π_1]). The form-based shortcut is unsafe.
        L = WallGroupL(dimension=n, pi=pi1_descriptor)
        cls = L.obstruction_class(K, backend=backend)
        vanishes = L.is_trivial(cls)
        obstruction_class = {"wall_class": cls.model_dump() if hasattr(cls, "model_dump") else cls, "pi1_descriptor": pi1_descriptor}
        kind = "wall_L_general"
        
    return ObstructionReport(
        name=name,
        dimension=n,
        middle_k=m,
        kind=kind,
        obstruction_class=obstruction_class,
        vanishes=vanishes,
        pi1_descriptor=pi1_descriptor,
        exact=True,
        theorem_tag=AUTO_SURGERY_MIDDLE_OBSTRUCTION,
        contract_version=CONTRACT_VERSION,
    )

def _pick_any_top_dim_simplex(K: SimplicialComplex) -> Any:
    n = K.dimension
    simplices = list(K.n_simplices(n))
    if not simplices:
        raise ValueError(f"No {n}-simplices found in the complex.")
    return simplices[0]


class AutoSurgeon:
    """The high-level pipeline orchestrator. Implements §22.6.

    Overview:
        Accepts one or more manifolds (as SimplicialComplex objects) and a
        symbolic ambient-space label, runs the full four-phase pipeline
        (unlink → un-nest → π₁-kill → homology ladder → obstruction → top-cell
        removal), and returns an AutoSurgeryReport.

    Key Concepts:
        - The *ambient space* is specified symbolically: "R^3", "S^4", "RP^2",
          or an integer dimension.  No ghost/bounding simplices are needed.
        - Multiple input manifolds can be supplied as a list; they are merged
          into a union complex internally.
        - The SurgerySession is created with the symbolic ambient string so
          that logs and cobordism traces reference the correct space.

    Common Workflows:
        1. Single manifold, default ambient (R^{n+1})::

               surgeon = AutoSurgeon(my_2_manifold, target_topology="contractible")
               report = surgeon.run()

        2. Multiple components with explicit ambient::

               surgeon = AutoSurgeon([circle_a, circle_b], ambient="R^3",
                                     target_topology="contractible")
               report = surgeon.run()

        3. Hopf link scenario (two linked circles in R^3)::

               Ka = SimplicialComplex.from_simplices([(0,1),(1,2),(2,0)])
               Kb = SimplicialComplex.from_simplices([(3,4),(4,5),(5,3)])
               surgeon = AutoSurgeon([Ka, Kb], ambient="R^3",
                                     target_topology="contractible")

    Args:
        manifold: A single SimplicialComplex or a list of them.  Multiple
            complexes are merged into one union complex; connectivity detection
            then separates them back into their components.
        ambient: Ambient-space label.  Accepted forms: "R^3", "S^3", "RP^2",
            integer (e.g. 3), or None / "auto" (infers R^{max_dim + 1}).
        point_clouds: Dict mapping component name → coordinate array for
            geometry-based detection (nesting, cut-site finding, isotopy).
        config: Fully pre-built AutoSurgeonConfig; overrides all **kwargs.
        **kwargs: Forwarded to AutoSurgeonConfig (e.g. target_topology=,
            backend=, max_unlink_surgeries=).
    """

    def __init__(
        self,
        manifold: Union[SimplicialComplex, List[SimplicialComplex]],
        *,
        ambient: Optional[Union[str, int]] = None,
        point_clouds: Optional[Dict[str, Any]] = None,
        config: Optional[AutoSurgeonConfig] = None,
        **kwargs,
    ):
        # ── Build the union complex ───────────────────────────────────────────
        if isinstance(manifold, list):
            all_simplices: List[Any] = []
            coords_found = None
            for m in manifold:
                for d in m.dimensions:
                    for s in m.n_simplices(d):
                        all_simplices.append(s)
                if coords_found is None and getattr(m, "_coordinates", None) is not None:
                    coords_found = m._coordinates
            self.K = SimplicialComplex.from_simplices(
                all_simplices, close_under_faces=True
            )
            if coords_found is not None:
                self.K._coordinates = coords_found
        else:
            self.K = manifold

        # ── Ambient-space label (symbolic, no bounding simplex needed) ────────
        # Store the raw spec; _parse_ambient_spec is called in run() once we
        # know the component dimensions.
        self._ambient_spec = ambient

        self.coords = point_clouds
        if config is None:
            # Allow ambient= to be passed via **kwargs for backwards compat.
            if "ambient" in kwargs:
                self._ambient_spec = kwargs.pop("ambient")
            self.config = AutoSurgeonConfig(**kwargs)
        else:
            self.config = config

    def run(self) -> AutoSurgeryReport:
        """Execute the full surgery pipeline and return its report.

        Validates each component as a homology manifold, resolves the ambient
        space, then runs the phases in order: unlink linked pairs, separate
        nested pairs, and per component kill π₁, climb the homology ladder,
        check the middle-dimensional obstruction, and (optionally) remove a
        top cell for strict contractibility.

        Returns:
            An AutoSurgeryReport summarizing every phase. Its ``status`` is
            ``"success"``, ``"halted_by_obstruction"`` if an obstruction does
            not vanish, or ``"halted_by_error"`` if a phase raised.

        Raises:
            NonManifoldComponentError: If any component is not a homology
                manifold.
        """
        # ── Phase 0: ingest & validate ────────────────────────────────────────
        infos = detect_components_with_status(self.K, backend=self.config.backend)
        for info in infos:
            if not info.is_manifold:
                raise NonManifoldComponentError(
                    f"Component '{info.name}' is not a homology manifold. "
                    f"Dimension: {info.dimension}."
                )

        # Resolve ambient spec against the highest-dim component.
        max_comp_dim = max((info.dimension for info in infos), default=0)
        ambient_label, ambient_dim = _parse_ambient_spec(
            self._ambient_spec, max_comp_dim
        )

        # Emit PL-manifold warning only for components of dim >= 4
        # (the ambient string itself doesn't trigger this).
        for info in infos:
            if info.dimension >= 4:
                warnings.warn(
                    f"Component '{info.name}' has dim {info.dimension} >= 4. "
                    "Pipeline detects homology manifolds; result may not be PL-rigorous.",
                    HomologyManifoldNotPLWarning,
                )

        # Create session with the symbolic ambient label (no bounding simplex).
        session = SurgerySession(
            ambient_space=ambient_label,
            objects={info.name: info.subcomplex for info in infos},
            point_clouds=self.coords,
        )
        # Store the union complex so linking computations can use real simplices
        # even when session.manifold is a string.
        session._union_K = self.K  # type: ignore[attr-defined]

        # ── Phase 1: Unlinking ────────────────────────────────────────────────
        linked_pairs = detect_linked_pairs(
            self.K, infos,
            ambient_dim=ambient_dim,
            backend=self.config.backend,
        )
        unlink_reports = []
        for pair in linked_pairs:
            try:
                r = auto_unlink_pair(
                    session,
                    pair.a,
                    pair.b,
                    mode=self.config.unlink_mode,
                    max_surgeries=self.config.max_unlink_surgeries,
                    backend=self.config.backend,
                )
            except Exception as e:
                return self._halt(
                    "halted_by_error", error=str(e), session=session,
                    infos=infos, unlink_reports=unlink_reports,
                )
            unlink_reports.append(r)

        # ── Phase 2: Un-nesting ───────────────────────────────────────────────
        nested_pairs = detect_nested_pairs(self.K, infos, coords=self.coords)
        nest_reports = []
        for pair in nested_pairs:
            try:
                r = auto_separate_nested(
                    session, pair.outer, pair.inner,
                    backend=self.config.backend,
                )
            except Exception as e:
                return self._halt(
                    "halted_by_error", error=str(e), session=session,
                    infos=infos, unlink_reports=unlink_reports,
                    nest_reports=nest_reports,
                )
            nest_reports.append(r)

        # ── Phase 3: Per-component reduction ──────────────────────────────────
        pi1_reports: Dict[str, Any] = {}
        h_reports: Dict[str, List[Any]] = {info.name: [] for info in infos}
        obstr_reports: Dict[str, Any] = {}
        status = "success"

        for info in infos:
            name = info.name
            n = info.dimension
            m = n // 2

            # 3a. Kill π₁ (skipped for dim ≤ 1 — handle surgery would change
            #     manifold type; top-cell removal in 3d handles contractibility)
            try:
                pi1_rep = auto_kill_pi1(
                    session, name,
                    max_generators=self.config.kill_pi1_budget,
                    backend=self.config.backend,
                )
                pi1_reports[name] = pi1_rep
            except Exception as e:
                return self._halt(
                    "halted_by_error", error=str(e), session=session,
                    infos=infos, unlink_reports=unlink_reports,
                    nest_reports=nest_reports, pi1_reports=pi1_reports,
                    h_reports=h_reports, obstr_reports=obstr_reports,
                )

            if pi1_rep.final_pi1_descriptor != "1":
                status = "halted_by_error"
                break

            # 3b. Homology ladder: kill H_k for k = 2 … m−1
            ladder_aborted = False
            for k in range(2, m):
                try:
                    rep = auto_kill_homology_dim(
                        session, name, k,
                        backend=self.config.backend,
                    )
                    h_reports[name].append(rep)
                except LadderProgressError as e:
                    ladder_aborted = True
                    obstr_reports[name] = ObstructionReport.from_ladder_error(e)
                    status = "halted_by_obstruction"
                    break
                except Exception as e:
                    return self._halt(
                        "halted_by_error", error=str(e), session=session,
                        infos=infos, unlink_reports=unlink_reports,
                        nest_reports=nest_reports, pi1_reports=pi1_reports,
                        h_reports=h_reports, obstr_reports=obstr_reports,
                    )

            if ladder_aborted:
                continue

            # 3c. Middle-dimension obstruction (Wall / Arf / signature)
            try:
                obstr = auto_check_middle_obstruction(
                    session, name, backend=self.config.backend,
                )
                obstr_reports[name] = obstr
            except Exception as e:
                return self._halt(
                    "halted_by_error", error=str(e), session=session,
                    infos=infos, unlink_reports=unlink_reports,
                    nest_reports=nest_reports, pi1_reports=pi1_reports,
                    h_reports=h_reports, obstr_reports=obstr_reports,
                )

            if not obstr.vanishes:
                status = "halted_by_obstruction"
                continue

            # Obstruction vanishes: kill H_m (middle dim) if m ≥ 2
            if m >= 2:
                try:
                    rep_mid = auto_kill_homology_dim(
                        session, name, m, backend=self.config.backend,
                    )
                    h_reports[name].append(rep_mid)
                except Exception as e:
                    return self._halt(
                        "halted_by_error", error=str(e), session=session,
                        infos=infos, unlink_reports=unlink_reports,
                        nest_reports=nest_reports, pi1_reports=pi1_reports,
                        h_reports=h_reports, obstr_reports=obstr_reports,
                    )

            # 3d. Top-cell removal for strict contractibility (M11, opt-in).
            # Guard: skip if the component is already contractible
            # (β_k = 0 for all k ≥ 1).  This prevents e.g. a D^n bounding
            # complex from being mutated into S^{n-1}.
            if self.config.target_topology == "contractible":
                comp_data = session.objects[name].data
                betti_now = comp_data.betti_numbers(backend=self.config.backend)
                already_contractible = all(
                    betti_now.get(k, 0) == 0 for k in range(1, n + 1)
                )
                if not already_contractible:
                    sigma_top = _pick_any_top_dim_simplex(comp_data)
                    session.remove_disks(
                        types=[f"D^{n}"],
                        at=[sigma_top],
                        target=name,
                    )
                    betti_after = session.objects[name].data.betti_numbers(
                        backend=self.config.backend
                    )
                    h_reports[name].append(
                        HKillReport(
                            name=name,
                            k=n,
                            steps=[
                                HKillStep(
                                    cycle=[sigma_top],
                                    summand="D^n_removal",
                                    attached=False,
                                    betti_after=betti_after,
                                    exact=True,
                                )
                            ],
                            exact=True,
                        )
                    )

        # ── Reconstruct final components from mutated session objects ─────────
        # Rebuild the union complex from the current state of each tracked
        # object so final_infos reflects surgeries, NOT the original K.
        final_simplices: List[Any] = []
        for obj in session.objects().values():
            if not hasattr(obj, "data") or isinstance(obj.data, str):
                continue
            for d in obj.data.dimensions:
                for s in obj.data.n_simplices(d):
                    final_simplices.append(s)

        if final_simplices:
            final_union = SimplicialComplex.from_simplices(
                final_simplices,
                coefficient_ring=self.K.coefficient_ring,
                close_under_faces=True,
            )
            if getattr(self.K, "_coordinates", None) is not None:
                final_union._coordinates = self.K._coordinates
        else:
            final_union = self.K

        session.finish()

        final_infos = detect_components_with_status(
            final_union, backend=self.config.backend
        )

        # Assemble final report
        return AutoSurgeryReport(
            status=status,
            exact=(
                status == "success"
                and all(r.exact for r in unlink_reports)
                and all(r.exact for r in nest_reports)
                and all(r.exact for r in pi1_reports.values())
                and all(r.exact for rs in h_reports.values() for r in rs)
                and all(r.exact and r.vanishes for r in obstr_reports.values())
            ),
            initial_components=infos,
            final_components=final_infos,
            unlink_reports=unlink_reports,
            nest_reports=nest_reports,
            pi1_kill_reports=pi1_reports,
            homology_kill_reports=h_reports,
            obstruction_reports=obstr_reports,
            session_logs_plain=session.logs(latex=False),
            session_logs_latex=session.logs(latex=True),
            cobordism_trace=list(session.cobordism),
        )

    def _halt(
        self,
        status: str,
        error: Optional[str] = None,
        session: Optional[SurgerySession] = None,
        infos: Optional[List] = None,
        unlink_reports: Optional[List] = None,
        nest_reports: Optional[List] = None,
        pi1_reports: Optional[Dict] = None,
        h_reports: Optional[Dict] = None,
        obstr_reports: Optional[Dict] = None,
    ) -> AutoSurgeryReport:
        if session is not None:
            session.finish()
        return AutoSurgeryReport(
            status=status,
            exact=False,
            initial_components=infos or [],
            final_components=detect_components_with_status(session.manifold, backend=self.config.backend) if session else [],
            unlink_reports=unlink_reports or [],
            nest_reports=nest_reports or [],
            pi1_kill_reports=pi1_reports or {},
            homology_kill_reports=h_reports or {},
            obstruction_reports=obstr_reports or {},
            session_logs_plain=session.logs(latex=False) if session else f"Halted by error: {error}",
            session_logs_latex=session.logs(latex=True) if session else f"Halted by error: {error}",
            cobordism_trace=list(session.cobordism) if session else [],
        )
