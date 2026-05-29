"""
pysurgery/knots/analysis.py

High-level analysis: find and classify knots between connected components of a
simplicial complex. This is the main entry point for answering "where are the
knots between my connected components?"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from pysurgery.topology.complexes import SimplicialComplex
from pysurgery.knots.linking import linking_matrix, link_type, LinkType, milnor_triple_invariant
from pysurgery.knots.invariants import (
    alexander_polynomial,
    knot_signature,
    arf_invariant,
    genus_bound,
    knot_determinant,
    classify_knot,
    is_unknot,
)


@dataclass
class ComponentKnotInfo:
    """Invariants of a single knot component K in the ambient complex."""

    component_index: int
    alexander_polynomial: Dict[int, int]
    signature: int
    arf: int
    genus_lower_bound: int
    determinant: int
    knot_type: str
    is_unknotted: bool

    def __str__(self) -> str:
        poly_str = " + ".join(
            f"{c}t^{d}" for d, c in sorted(self.alexander_polynomial.items(), reverse=True)
        )
        return (
            f"Component {self.component_index}: {self.knot_type}\n"
            f"  Δ(t) = {poly_str}\n"
            f"  signature = {self.signature}, Arf = {self.arf}, "
            f"genus ≥ {self.genus_lower_bound}, det = {self.determinant}"
        )


@dataclass
class KnotAnalysisResult:
    """Full knot-theoretic analysis of a simplicial complex with multiple components.

    Attributes:
        components: List of 1-cycle subcomplexes extracted from the input complex.
        linking_matrix: (n × n) integer matrix of pairwise linking numbers.
        link_classification: Overall link type (HOPF, BORROMEAN, UNLINKED, etc.).
        are_linked: True if any linking (pairwise or higher-order) is detected.
        milnor_triple: Milnor μ̄(123) invariant if there are exactly 3 pairwise-unlinked components.
        component_invariants: Per-component knot invariants (Alexander polynomial, signature, etc.).
        linked_pairs: List of (i, j) index pairs where lk(K_i, K_j) ≠ 0.
        undetected_warning: Non-empty string if higher-order linking may be present but undetected.
    """

    components: List[SimplicialComplex] = field(default_factory=list)
    linking_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=int))
    link_classification: LinkType = LinkType.UNLINKED
    are_linked: bool = False
    milnor_triple: Optional[int] = None
    component_invariants: List[ComponentKnotInfo] = field(default_factory=list)
    linked_pairs: List[Tuple[int, int]] = field(default_factory=list)
    undetected_warning: str = ""

    def summary(self) -> str:
        lines = [
            f"Knot Analysis: {len(self.components)} component(s)",
            f"  Link type: {self.link_classification.value}",
            f"  Are linked: {self.are_linked}",
        ]
        if self.milnor_triple is not None:
            lines.append(f"  Milnor μ̄(123) = {self.milnor_triple}")
        if self.linked_pairs:
            lines.append(f"  Linked pairs: {self.linked_pairs}")
        lines.append(f"  Linking matrix:\n{self.linking_matrix}")
        if self.component_invariants:
            lines.append("\nPer-component knot invariants:")
            for info in self.component_invariants:
                lines.append(str(info))
        if self.undetected_warning:
            lines.append(f"\nWarning: {self.undetected_warning}")
        return "\n".join(lines)


def _extract_connected_components(sc: SimplicialComplex) -> List[SimplicialComplex]:
    """Extract the 1-connected-components of a 1-dimensional (or higher) simplicial complex.

    Returns a list of SimplicialComplexes, one per connected component of the
    1-skeleton. Only components that contain at least one 1-simplex (edge) are returned.
    """
    edges = list(sc.n_simplices(1))
    if not edges:
        return []

    # Union-find on vertex set
    vertices = set()
    for e in edges:
        vertices.update(e)
    parent = {v: v for v in vertices}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a != b:
            parent[a] = b

    for e in edges:
        union(e[0], e[1])

    # Group edges by component root
    comp_edges: Dict[int, List] = {}
    for e in edges:
        root = find(e[0])
        comp_edges.setdefault(root, []).append(e)

    components = []
    for root in sorted(comp_edges.keys()):
        comp_sc = SimplicialComplex.from_simplices(comp_edges[root])
        components.append(comp_sc)

    return components


def find_knots_between_components(
    sc: SimplicialComplex,
    components: Optional[List[SimplicialComplex]] = None,
    ambient_complex: Optional[SimplicialComplex] = None,
    compute_per_component_invariants: bool = True,
    backend: str = "auto",
) -> KnotAnalysisResult:
    """Find and classify all knots between the connected components of a complex.

    What is Being Computed?:
        Given a simplicial complex sc (or explicit components), this function:
        1. Extracts the connected 1-cycle components (if not provided).
        2. Computes the full linking matrix (pairwise linking numbers).
        3. Checks for higher-order Milnor linking (Borromean-type, Whitehead-type).
        4. Computes per-component knot invariants (Alexander polynomial, signature, Arf).
        5. Classifies each pair and the overall link type.

    Args:
        sc: Ambient simplicial complex. Used as both the geometric host and the
            ambient S^3 model if ambient_complex is not provided.
        components: Pre-extracted list of 1-cycle subcomplexes. If None, components
            are extracted automatically from the 1-skeleton of sc.
        ambient_complex: The ambient 3-manifold (should triangulate S^3). If None,
            sc itself is used as the ambient complex.
        compute_per_component_invariants: If True, compute Alexander polynomial,
            signature, etc. for each component. Can be slow for large complexes.
        backend: "auto", "julia", or "python".

    Returns:
        KnotAnalysisResult with full linking and knot-theoretic information.
    """
    if ambient_complex is None:
        ambient_complex = sc

    if components is None:
        components = _extract_connected_components(sc)

    result = KnotAnalysisResult(components=components)

    if not components:
        return result

    n = len(components)

    # ── Step 1: Linking matrix ────────────────────────────────────────────────
    L = linking_matrix(ambient_complex, components, backend=backend)
    result.linking_matrix = L

    # ── Step 2: Linked pairs ──────────────────────────────────────────────────
    linked_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if L[i, j] != 0:
                linked_pairs.append((i, j))
    result.linked_pairs = linked_pairs
    result.are_linked = len(linked_pairs) > 0

    # ── Step 3: Higher-order Milnor invariants ────────────────────────────────
    if not result.are_linked and n == 3:
        # All pairwise linking numbers are 0 → check Milnor triple invariant
        try:
            mu = milnor_triple_invariant(
                ambient_complex, components[0], components[1], components[2], backend=backend
            )
            result.milnor_triple = mu
            if mu != 0:
                result.are_linked = True
        except ValueError:
            pass

    if not result.are_linked and n >= 4:
        result.undetected_warning = (
            f"Higher-order Milnor invariants for {n}-component links (n ≥ 4) are not "
            "yet computed. Linkage may be present via μ̄ with multi-index of length ≥ 4."
        )

    # ── Step 4: Link type classification ─────────────────────────────────────
    result.link_classification = link_type(ambient_complex, components, backend=backend)

    # ── Step 5: Per-component knot invariants ─────────────────────────────────
    if compute_per_component_invariants:
        inv_list: List[ComponentKnotInfo] = []
        for idx, comp in enumerate(components):
            try:
                delta = alexander_polynomial(ambient_complex, comp, backend=backend)
                sig = knot_signature(ambient_complex, comp, backend=backend)
                arf = arf_invariant(ambient_complex, comp, backend=backend)
                g = genus_bound(ambient_complex, comp, backend=backend)
                det = knot_determinant(ambient_complex, comp, backend=backend)
                ktype = classify_knot(ambient_complex, comp, backend=backend)
                unknotted = is_unknot(ambient_complex, comp, backend=backend)
            except Exception:
                delta = {0: 1}
                sig = 0
                arf = 0
                g = 0
                det = 1
                ktype = "unknown (computation failed)"
                unknotted = True

            inv_list.append(ComponentKnotInfo(
                component_index=idx,
                alexander_polynomial=delta,
                signature=sig,
                arf=arf,
                genus_lower_bound=g,
                determinant=det,
                knot_type=ktype,
                is_unknotted=unknotted,
            ))
        result.component_invariants = inv_list

    return result


def linking_report(
    ambient_complex: SimplicialComplex,
    components: List[SimplicialComplex],
    backend: str = "auto",
) -> str:
    """Return a human-readable linking report for a list of components.

    Reports:
        - Pairwise linking numbers
        - Milnor triple invariant (if applicable)
        - Overall classification
    """
    result = find_knots_between_components(
        ambient_complex,
        components=components,
        ambient_complex=ambient_complex,
        compute_per_component_invariants=False,
        backend=backend,
    )
    return result.summary()
