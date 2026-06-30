"""Covering spaces, universal covers, and deck transformations.

This module is the single home for pySurgery's covering-space machinery,
unifying two historically separate engines:

  * **Geometric tiling** (``FacePairing``, ``FundamentalPolyhedron``,
    ``construct_fundamental_polyhedron``) — builds the fundamental polyhedron
    of a triangulated manifold and tiles its universal cover via face-pairing
    isometries. Works for manifolds with infinite π₁.

  * **Algebraic finite cover** (``FiniteGroupRing``, ``UniversalCover``) —
    builds the universal cover M̃ → M of a CW complex with *finite* π₁ over ℤ
    via Fox-derivative lifts of the cellular boundary maps, together with the
    deck-transformation action.

On top of those it exposes a unified, dimension-agnostic API:

  * ``Covering`` — the abstract covering object (base, total space, covering
    map, fiber, degree, deck group, path lifting, monodromy).
  * ``DeckTransformationGroup`` — the deck group acting on the cells of a cover.
  * ``Covering.from_permutation_rep`` / ``Covering.from_subgroup`` — general
    (not necessarily universal) covers from a transitive permutation action of
    π₁, generalizing the regular-representation universal cover.
  * Graph covers: ``Graph.universal_cover`` (the unrolled tree) and
    ``cover_graph`` (voltage-graph finite covers) live conceptually here and are
    reached through :mod:`pysurgery.topology.graphs`.

References:
    - Hatcher, A. (2002). Algebraic Topology, §1.3 (covering spaces).
    - Thurston, W. P. (1997). Three-Dimensional Geometry and Topology.
    - Ratcliffe, J. G. (2006). Foundations of Hyperbolic Manifolds.
    - Fox, R. H. (1953). Free differential calculus I. Ann. of Math. 57.
"""

from __future__ import annotations

import os
import itertools
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_tree
from pydantic import BaseModel, ConfigDict, Field

from pysurgery.topology.complexes import (
    CWComplex,
    ChainComplex,
    SimplicialComplex,
    _normalize_simplex,
)
from pysurgery.topology.fundamental_group import (
    FundamentalGroup,
    extract_pi_1_with_traces,
)
from pysurgery.core.exceptions import (
    DimensionError,
    FundamentalGroupError,
    GroupRingError,
)
from pysurgery.core.foundations import CONTRACT_VERSION
from pysurgery.bridge.julia_bridge import julia_engine

__all__ = [
    # Geometric tiling
    "FacePairing",
    "FundamentalPolyhedron",
    "construct_fundamental_polyhedron",
    # Algebraic finite cover
    "FiniteGroupOrderResult",
    "UniversalCoverResult",
    "FiniteGroupRing",
    "UniversalCover",
    # Unified covering API
    "Covering",
    "DeckTransformationGroup",
    "GroupAction",
    "GraphCovering",
    "cover_graph",
    "graph_universal_cover",
]


# ──────────────────────────────────────────────────────────────────────────────
# Geometric tiling: fundamental polyhedron + universal-cover tiling
# ──────────────────────────────────────────────────────────────────────────────


class FacePairing(BaseModel):
    """Represents a gluing identification between two (n-1)-faces of the fundamental polyhedron.
    
    Overview:
        In the construction of a manifold from a fundamental polyhedron, every 
        codimension-1 face on the boundary of the polyhedron must be identified 
        with exactly one other codimension-1 face. This pairing defines a 
        deck transformation in the universal cover and a generator for the 
        fundamental group π₁(M).

    Attributes:
        face_a (Tuple[int, ...]): Sorted vertex indices of the first face.
        face_b (Tuple[int, ...]): Sorted vertex indices of the second face.
        simplex_a_idx (int): Index of the n-simplex in the triangulation containing face_a.
        simplex_b_idx (int): Index of the n-simplex in the triangulation containing face_b.
        permutation (Dict[int, int]): Combinatorial mapping from vertices of face_a 
            to vertices of face_b that defines the gluing.
        generator_symbol (str): The label for this pairing in the group presentation (e.g., 'g1').
        geometric_transform (Optional[Any]): A (d+1, d+1) isometry matrix (Euclidean, 
            Hyperbolic, or Spherical) that moves face_a to face_b in the universal cover.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    face_a: Tuple[int, ...]
    face_b: Tuple[int, ...]
    simplex_a_idx: int
    simplex_b_idx: int
    permutation: Dict[int, int]
    generator_symbol: str
    geometric_transform: Optional[Any] = None


class FundamentalPolyhedron(BaseModel):
    """A topological n-ball representation of a manifold with boundary identifications.
    
    Overview:
        A FundamentalPolyhedron is constructed by taking all n-simplices of a 
        triangulation and "unfolding" them into a single simply-connected block. 
        This is achieved by selecting a spanning tree of the dual graph of the 
        triangulation and gluing along those edges. The remaining unglued 
        (n-1)-faces form the boundary of the polyhedron and are grouped into 
        FacePairings.

    Mathematical Properties:
        - **Simply Connected**: Gluing along a tree ensures the result is a topological n-ball.
        - **Generators**: Each FacePairing corresponds to a generator of π₁(M).
        - **Relations**: Tracing the identifies around (n-2)-dimensional "hinges" 
          yields the relators for the fundamental group presentation.
        - **Universal Cover**: The universal cover is tiled by copies of this 
          polyhedron, indexed by words in the fundamental group.

    Attributes:
        n_simplices (List[Tuple[int, ...]]): The list of top-dimensional simplices.
        dimension (int): The dimension n of the manifold.
        internal_glues (List[Tuple[int, int, Tuple[int, ...]]]): Triplets of 
            (idx1, idx2, shared_face) representing the internal connections 
            (the dual spanning tree).
        face_pairings (List[FacePairing]): The identifications on the boundary.
        relations (List[List[str]]): The relators of the fundamental group 
            extracted from the (n-2)-face hinges.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    n_simplices: List[Tuple[int, ...]]
    dimension: int
    internal_glues: List[Tuple[int, int, Tuple[int, ...]]] = Field(
        description="Dual spanning tree edges: (simplex_a, simplex_b, shared_face)."
    )
    face_pairings: List[FacePairing] = Field(
        description="Identifications between boundary (n-1)-faces."
    )
    relations: List[List[str]] = Field(
        default_factory=list,
        description="Relators for π₁(M) presentation derived from hinges."
    )

    def get_symbolic_atlas(self) -> Tuple[List[str], List[List[str]]]:
        """Return the fundamental group presentation ⟨G | R⟩.
        
        What is Being Computed?:
            The abstract group presentation defined by the polyhedron. 
            Generators G come from face pairings, and relations R come 
            from traversing the cycle of simplices around each (n-2)-face.

        Returns:
            Tuple[List[str], List[List[str]]]: (generators, relations).
        """
        generators = [p.generator_symbol for p in self.face_pairings]
        return generators, self.relations
        
    def get_numerical_atlas(self) -> List[Dict[str, Any]]:
        """Return the numerical gluing data for the manifold's simplified atlas.
        
        What is Being Computed?:
            A list of transition maps for a 1-chart atlas. Each transition 
            map is a combinatorial permutation (and optionally a geometric 
            isometry) between two faces of the fundamental polyhedron.

        Returns:
            List[Dict]: Numerical data for each transition map.
        """
        atlas = []
        for pairing in self.face_pairings:
            atlas.append({
                "generator": pairing.generator_symbol,
                "simplex_a": pairing.simplex_a_idx,
                "simplex_b": pairing.simplex_b_idx,
                "face_a": pairing.face_a,
                "face_b": pairing.face_b,
                "permutation": pairing.permutation,
                "transform": pairing.geometric_transform
            })
        return atlas

    def tile_universal_cover(self, depth: int = 3, initial_transform: Optional[Any] = None) -> List[Dict[str, Any]]:
        """Generate a collection of tiles representing the universal cover M̃.
        
        What is Being Computed?:
            A Breadth-First Search (BFS) traversal of the deck transformation 
            group. Starting from the identity tile (the fundamental polyhedron), 
            it applies face-pairing generators to produce new copies (lifts) 
            of the polyhedron in the universal cover.

        Algorithm:
            1. Initialize a queue with the identity word and optional identity transform.
            2. For each word, apply all available face-pairing generators.
            3. Use free reduction to avoid immediate backtracking.
            4. If geometric transforms are present, compose them to track the 
               absolute position of the tile in the covering space.

        Args:
            depth: The maximum word length (number of hops) to explore.
            initial_transform: Optional starting coordinate transform (e.g., Identity).
            
        Returns:
            List[Dict]: A list of tiles. Each tile is a dict with:
                - 'word_address': String representation of the deck transform.
                - 'depth': Distance from the root tile.
                - 'transform': (Optional) Composed isometry matrix for this tile.
        """
        tiles = []
        # Queue stores: (word_path, current_depth, current_transform)
        queue = deque([([], 0, initial_transform)])
        visited_words = {()}
        
        # Build dictionary for quick transform lookup
        transforms = {}
        for p in self.face_pairings:
            transforms[p.generator_symbol] = p.geometric_transform
            if p.geometric_transform is not None:
                # Naive inverse for NumPy arrays
                try:
                    transforms[p.generator_symbol + "^-1"] = np.linalg.inv(p.geometric_transform)
                except Exception:
                    transforms[p.generator_symbol + "^-1"] = None
            else:
                transforms[p.generator_symbol + "^-1"] = None
        
        while queue:
            word, d, current_transform = queue.popleft()
            
            tile = {
                "word_address": " ".join(word) if word else "id",
                "depth": d
            }
            if current_transform is not None:
                tile["transform"] = current_transform
            tiles.append(tile)
            
            if d < depth:
                for p in self.face_pairings:
                    # Forward
                    next_word_f = tuple(list(word) + [p.generator_symbol])
                    if next_word_f not in visited_words:
                        if not word or word[-1] != p.generator_symbol + "^-1":
                            visited_words.add(next_word_f)
                            next_tf = None
                            if current_transform is not None and transforms[p.generator_symbol] is not None:
                                next_tf = current_transform @ transforms[p.generator_symbol]
                            queue.append((next_word_f, d + 1, next_tf))
                    
                    # Inverse
                    inv_symbol = p.generator_symbol + "^-1"
                    next_word_i = tuple(list(word) + [inv_symbol])
                    if next_word_i not in visited_words:
                        if not word or word[-1] != p.generator_symbol:
                            visited_words.add(next_word_i)
                            next_ti = None
                            if current_transform is not None and transforms[inv_symbol] is not None:
                                next_ti = current_transform @ transforms[inv_symbol]
                            queue.append((next_word_i, d + 1, next_ti))
                            
        return tiles


def construct_fundamental_polyhedron(sc: SimplicialComplex) -> FundamentalPolyhedron:
    """Construct the fundamental polyhedron for a manifold SimplicialComplex.
    
    Args:
        sc: A SimplicialComplex representing a manifold.
        
    Returns:
        A FundamentalPolyhedron instance.
    """
    dim = sc.dimension
    n_simplices = sc.n_simplices(dim)
    if not n_simplices:
        raise ValueError(f"Complex has no {dim}-simplices.")
        
    # 1. Map (n-1)-faces to n-simplices
    face_to_simplices = {}
    for i, simplex in enumerate(n_simplices):
        import itertools
        for face in itertools.combinations(simplex, dim):
            norm_face = _normalize_simplex(face)
            if norm_face not in face_to_simplices:
                face_to_simplices[norm_face] = []
            face_to_simplices[norm_face].append(i)
            
    # 2. Build dual graph
    row, col, data = [], [], []
    internal_face_map = {} # (simplex_i, simplex_j) -> shared_face
    
    for face, indices in face_to_simplices.items():
        if len(indices) == 2:
            u, v = indices
            row.append(u)
            col.append(v)
            data.append(1)
            row.append(v)
            col.append(u)
            data.append(1)
            internal_face_map[tuple(sorted((u, v)))] = face
        elif len(indices) > 2:
            raise ValueError(f"Complex is not a manifold: face {face} is shared by {len(indices)} simplices.")
            
    n = len(n_simplices)
    adj_matrix = csr_matrix((data, (row, col)), shape=(n, n))
    
    # 3. Compute BFS spanning tree
    tree = breadth_first_tree(adj_matrix, 0, directed=False)
    tree_edges = set()
    mst_row, mst_col = tree.nonzero()
    for u, v in zip(mst_row, mst_col):
        tree_edges.add(tuple(sorted((u, v))))
        
    # 4. Partition faces into internal (tree) and external (pairings)
    internal_glues = []
    face_info = {} # face -> (u, v, is_tree, symbol)
    for edge in tree_edges:
        u, v = edge
        face = internal_face_map[edge]
        internal_glues.append((u, v, face))
        face_info[face] = (u, v, True, None)
        
    face_pairings = []
    gen_idx = 1
    for face, indices in face_to_simplices.items():
        if len(indices) == 2:
            u, v = indices
            edge = tuple(sorted((u, v)))
            if edge not in tree_edges:
                symbol = f"g{gen_idx}"
                pairing = FacePairing(
                    face_a=face,
                    face_b=face,
                    simplex_a_idx=u,
                    simplex_b_idx=v,
                    permutation={v_idx: v_idx for v_idx in face},
                    generator_symbol=symbol
                )
                face_pairings.append(pairing)
                face_info[face] = (u, v, False, symbol)
                gen_idx += 1
                
    # 5. Extract Relations from Hinges ((n-2)-faces)
    relations = []
    if dim >= 2:
        hinge_to_n_1_faces = {}
        import itertools
        for face in face_info.keys():
            for hinge in itertools.combinations(face, dim - 1):
                norm_hinge = _normalize_simplex(hinge)
                if norm_hinge not in hinge_to_n_1_faces:
                    hinge_to_n_1_faces[norm_hinge] = []
                hinge_to_n_1_faces[norm_hinge].append(face)
                
        for hinge, n_1_faces in hinge_to_n_1_faces.items():
            adj = {}
            for face in n_1_faces:
                tets = face_to_simplices.get(face)
                if tets and len(tets) == 2:
                    u, v = tets
                    if u not in adj:
                        adj[u] = []
                    if v not in adj:
                        adj[v] = []
                    adj[u].append((v, face))
                    adj[v].append((u, face))
            
            if not adj:
                continue
            
            start_node = list(adj.keys())[0]
            curr_node = start_node
            prev_node = None
            word = []
            
            # Simple cycle traversal around the hinge
            while True:
                neighbors = adj.get(curr_node, [])
                next_node, next_face = None, None
                for n_node, f in neighbors:
                    if n_node != prev_node:
                        next_node, next_face = n_node, f
                        break
                
                if next_node is None:
                    break
                    
                u, v, is_tree, symbol = face_info[next_face]
                if not is_tree:
                    if curr_node == u and next_node == v:
                        word.append(symbol)
                    elif curr_node == v and next_node == u:
                        word.append(symbol + "^-1")
                        
                prev_node = curr_node
                curr_node = next_node
                
                if curr_node == start_node:
                    break
            
            # Basic free reduction
            reduced_word = []
            for w in word:
                if reduced_word and (
                    (reduced_word[-1] == w + "^-1") or 
                    (reduced_word[-1] + "^-1" == w)
                ):
                    reduced_word.pop()
                else:
                    reduced_word.append(w)
            
            if reduced_word and reduced_word not in relations:
                relations.append(reduced_word)

    return FundamentalPolyhedron(
        n_simplices=n_simplices,
        dimension=dim,
        internal_glues=internal_glues,
        face_pairings=face_pairings,
        relations=relations
    )


# ──────────────────────────────────────────────────────────────────────────────
# Algebraic finite cover: group ring, Fox-derivative boundary lifts
# ──────────────────────────────────────────────────────────────────────────────


_MAX_COVER_ORDER_DEFAULT = 200


def _resolve_max_cover_order(explicit: Optional[int]) -> int:
    if explicit is not None:
        return int(explicit)
    env = os.environ.get("PYSURGERY_MAX_COVER_ORDER")
    if env is not None:
        try:
            return int(env)
        except ValueError:
            pass
    return _MAX_COVER_ORDER_DEFAULT


@dataclass(frozen=True)
class FiniteGroupOrderResult:
    """Cayley-table-backed witness that a finitely-presented group is finite.

    Attributes:
        order: |G|.
        cayley: `(|G|, |G|)` int64 array; cayley[i, j] is the (1-based) index
            of the product of group elements i+1 and j+1.
        inverse_indices: 1-based inverse indices.
        identity_index: Cayley index of the identity (always 1).
        element_words: Reduced word representative for each element.
        exact: Always True (Todd-Coxeter is exact when it converges).
        theorem_tag: Stable identifier "controlled_cohomology.finite_group_order".
        contract_version: pySurgery contract version.
    """

    order: int
    cayley: np.ndarray
    inverse_indices: np.ndarray
    identity_index: int
    element_words: List[str]
    exact: bool = True
    theorem_tag: str = "controlled_cohomology.finite_group_order"
    contract_version: str = field(default_factory=lambda: CONTRACT_VERSION)

    def decision_ready(self) -> bool:
        """Return True when the result is exact and the group is non-empty."""
        return self.exact and self.order >= 1


@dataclass
class UniversalCoverResult:
    """Lifted CW data for a finite universal cover.

    Attributes:
        cover_cells: Cells per dimension in M̃ (= |G| × cells of M).
        lifted_attaching: Lifted boundary matrices over ℤ on the cover.
        deck_action: For each group element g, a permutation of cover cells
            in each dimension, returned as a dict `{dim: permutation array}`.
        base_dimensions: Dimensions of the base manifold that were lifted.
        group_order: |π₁|.
        exact: True when all involved boundaries lift exactly.
        theorem_tag: "controlled_cohomology.universal_cover".
        contract_version: pySurgery contract version.
    """

    cover_cells: Dict[int, int]
    lifted_attaching: Dict[int, sp.csr_matrix]
    deck_action: Dict[int, Dict[int, np.ndarray]]
    base_dimensions: List[int]
    group_order: int
    exact: bool = True
    theorem_tag: str = "controlled_cohomology.universal_cover"
    contract_version: str = field(default_factory=lambda: CONTRACT_VERSION)

    def decision_ready(self) -> bool:
        """Return True when the lift is exact and the group is non-empty."""
        return self.exact and self.group_order >= 1


class FiniteGroupRing:
    """Z[G] arithmetic for a finite group G via a Cayley-table convolution.

    Overview:
        Stores the multiplication table of a finite group as a NumPy int64
        array of shape `(|G|, |G|)` and exposes group-ring multiplication via
        a Julia-backed convolution kernel. Refuses to instantiate when π₁ is
        infinite or undecidable within `max_index`.

    Key Concepts:
        - Group elements are integers `1..|G|` matching the Cayley table's
          (1-based) indexing produced by Todd-Coxeter + BFS reduction.
        - Group-ring elements are 1-D `numpy.int64` arrays of length `|G|`.
        - Multiplication: `(a · b)[k] = Σ_{i,j: cayley[i,j] = k} a[i] · b[j]`.

    Coefficient Ring:
        ℤ. (For ℚ or ℂ representation arithmetic, see `TwistedRepresentation`,
        which composes ρ with this Cayley convolution at the matrix level.)
    """

    def __init__(
        self,
        presentation: FundamentalGroup,
        *,
        max_index: int = 10_000,
    ) -> None:
        if not isinstance(presentation, FundamentalGroup):
            raise GroupRingError(
                "FiniteGroupRing requires a FundamentalGroup presentation; "
                f"got {type(presentation).__name__}."
            )
        if not presentation.is_finite(max_index=max_index):
            raise FundamentalGroupError(
                "Cannot build FiniteGroupRing on an infinite group: π₁ has "
                "infinite abelianization or Todd-Coxeter did not converge."
            )
        order_witness = _build_finite_group_order_result(presentation, max_index)
        self._presentation = presentation
        self._order_result = order_witness
        gens = list(presentation.generators)
        # gen_to_group: 1-based index in Cayley of each generator's image.
        # Each generator's element-word is exactly the generator name; locate it.
        gen_to_group = []
        word_to_idx = {w: i + 1 for i, w in enumerate(order_witness.element_words)}
        for g in gens:
            if g not in word_to_idx:
                # Re-resolve via tree-walk: take the column for this generator
                # in the underlying coset table (column index = generator idx).
                # We do not have direct access here; fall back to error.
                raise GroupRingError(
                    f"Generator '{g}' has no canonical word in the Cayley "
                    "BFS reduction. Pass a presentation whose generators "
                    "appear directly in the BFS path tree."
                )
            gen_to_group.append(word_to_idx[g])
        self._generators = gens
        self._gen_to_group = gen_to_group

    @property
    def order(self) -> int:
        """Order |G| of the finite group."""
        return self._order_result.order

    @property
    def cayley(self) -> np.ndarray:
        """Cayley table — read-only view (`(|G|, |G|)` int64, 1-based)."""
        return self._order_result.cayley

    @property
    def inverse_indices(self) -> np.ndarray:
        """Per-element inverse indices (1-based)."""
        return self._order_result.inverse_indices

    @property
    def identity_index(self) -> int:
        """1-based Cayley index of the identity element."""
        return self._order_result.identity_index

    @property
    def element_words(self) -> List[str]:
        """Reduced word representative for each group element."""
        return list(self._order_result.element_words)

    @property
    def generator_indices(self) -> List[int]:
        """1-based Cayley index for each generator name in `presentation.generators`."""
        return list(self._gen_to_group)

    @property
    def generators(self) -> List[str]:
        """Generator names of the underlying presentation."""
        return list(self._generators)

    @property
    def order_result(self) -> FiniteGroupOrderResult:
        """Underlying `FiniteGroupOrderResult` witness for |G|."""
        return self._order_result

    def zero(self) -> np.ndarray:
        """Return the zero element of the group ring."""
        return np.zeros(self.order, dtype=np.int64)

    def one(self) -> np.ndarray:
        """Return the multiplicative identity (1·e)."""
        e = self.zero()
        e[self.identity_index - 1] = 1
        return e

    def basis(self, group_idx_1based: int) -> np.ndarray:
        """Return the basis vector e_g for the 1-based group index g."""
        if not 1 <= group_idx_1based <= self.order:
            raise GroupRingError(
                f"Group index out of range: {group_idx_1based} not in [1, {self.order}]."
            )
        v = self.zero()
        v[group_idx_1based - 1] = 1
        return v

    def multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Multiply two group-ring elements via the Cayley-table convolution.

        Args:
            a: int64 array of length `|G|`.
            b: int64 array of length `|G|`.

        Returns:
            int64 array of length `|G|` with the convolved coefficients.
        """
        a_arr = np.asarray(a, dtype=np.int64).reshape(-1)
        b_arr = np.asarray(b, dtype=np.int64).reshape(-1)
        if a_arr.size != self.order or b_arr.size != self.order:
            raise GroupRingError(
                f"Group-ring elements must have length {self.order}; "
                f"got {a_arr.size} and {b_arr.size}."
            )
        if julia_engine.available:
            return julia_engine.cayley_convolve(a_arr, b_arr, self.cayley)
        return self._multiply_python(a_arr, b_arr)

    def _multiply_python(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        cayley = self.cayley
        n = self.order
        res = np.zeros(n, dtype=np.int64)
        for i in range(n):
            ai = int(a[i])
            if ai == 0:
                continue
            for j in range(n):
                bj = int(b[j])
                if bj == 0:
                    continue
                res[cayley[i, j] - 1] += ai * bj
        return res

    @classmethod
    def from_pi1(
        cls, pi1: FundamentalGroup, *, max_index: int = 10_000
    ) -> "FiniteGroupRing":
        """Construct a `FiniteGroupRing` from a fundamental group.

        Args:
            pi1: The fundamental group presentation, which must be finite.
            max_index: Todd-Coxeter coset-enumeration cap for the finiteness
                check.

        Returns:
            A `FiniteGroupRing` over ℤ[π₁].
        """
        return cls(pi1, max_index=max_index)


def _build_finite_group_order_result(
    presentation: FundamentalGroup, max_index: int
) -> FiniteGroupOrderResult:
    if not presentation.generators:
        return FiniteGroupOrderResult(
            order=1,
            cayley=np.array([[1]], dtype=np.int64),
            inverse_indices=np.array([1], dtype=np.int64),
            identity_index=1,
            element_words=["e"],
        )
    if not julia_engine.available:
        raise FundamentalGroupError(
            "FiniteGroupRing requires the Julia backend for non-trivial "
            "groups (Todd-Coxeter coset enumeration)."
        )
    flat_rels = [" ".join(r) for r in presentation.relations]
    converged, n_cosets, table = julia_engine.todd_coxeter_index(
        list(presentation.generators), flat_rels, int(max_index)
    )
    if not converged:
        raise FundamentalGroupError(
            f"Todd-Coxeter exceeded max_index={max_index}; group is infinite "
            "or undecidable within the bound."
        )
    cayley, inverse, id_idx, words = julia_engine.cayley_table(
        table, list(presentation.generators)
    )
    return FiniteGroupOrderResult(
        order=int(n_cosets),
        cayley=cayley,
        inverse_indices=inverse,
        identity_index=int(id_idx),
        element_words=list(words),
    )


class UniversalCover:
    """Finite universal cover of a CW complex with finite π₁.

    Overview:
        Given a connected CW complex M with one 0-cell and finitely
        presented π₁ certified finite by `is_finite`, builds the universal
        cover M̃ as a CWComplex over ℤ. The cover has |π₁| · n_k(M) cells in
        dimension k for k ∈ {0, 1, 2}; higher-dimensional cells of M are
        currently rejected with `DimensionError`.

    Key Concepts:
        - **Single 0-cell convention**: M is required to have exactly one
          0-cell (the typical post-π₁-extraction shape). Multi-vertex
          complexes can be reduced to this form by collapsing a spanning tree
          of the 1-skeleton.
        - **Edge-generator correspondence**: The 1-cells of M correspond
          one-to-one with the generators of π₁(M), as produced by
          `extract_pi_1_with_traces`.
        - **2-cell relators**: Each 2-cell's attaching word is one of the
          relators of π₁(M); Fox derivatives lift this to a Z[π]-boundary.
    """

    def __init__(
        self,
        base: CWComplex,
        pi1: Optional[FundamentalGroup] = None,
        *,
        max_index: int = 10_000,
        max_order: Optional[int] = None,
    ) -> None:
        if not isinstance(base, CWComplex):
            raise DimensionError(
                "UniversalCover requires a CWComplex base; "
                f"got {type(base).__name__}."
            )
        max_order_val = _resolve_max_cover_order(max_order)
        self._base = base
        # Collect attaching word data via raw extraction so 2-cells map
        # bijectively to relators. Generator names from traces are
        # authoritative — any pi1 passed by the user is reconstructed in
        # those names so that the Cayley table indexes correctly.
        traces = extract_pi_1_with_traces(
            base, simplify=False, generator_mode="raw"
        )
        pi1 = FundamentalGroup(
            generators=list(traces.generators),
            relations=[list(r) for r in traces.relations],
            orientation_character=dict(traces.orientation_character),
        )
        self._pi1 = pi1
        if not pi1.is_finite(max_index=max_index):
            raise FundamentalGroupError(
                "UniversalCover requires finite π₁; this presentation has "
                "infinite abelianization or did not converge under "
                f"Todd-Coxeter (max_index={max_index})."
            )
        self._group_ring = FiniteGroupRing(pi1, max_index=max_index)
        if self._group_ring.order > max_order_val:
            raise FundamentalGroupError(
                f"|π₁| = {self._group_ring.order} exceeds the soft cap "
                f"max_order={max_order_val}. Increase the cap or use Path B "
                "(twisted chains directly on the base) via "
                "`compute_controlled_cohomology(..., path='fox')`."
            )
        self._validate_one_vertex(base)
        self._validate_dimension(base)
        self._cover_cells, self._lifted, self._deck = self._build_cover(
            base, traces.generators, traces.relations, traces.traces
        )

    @staticmethod
    def _validate_one_vertex(base: CWComplex) -> None:
        n0 = int(base.cells.get(0, 0))
        if n0 != 1:
            raise DimensionError(
                f"UniversalCover requires exactly one 0-cell; got {n0}. "
                "Collapse a spanning tree of the 1-skeleton first."
            )

    @staticmethod
    def _validate_dimension(base: CWComplex) -> None:
        for dim, n in base.cells.items():
            if dim >= 3 and n > 0:
                raise DimensionError(
                    f"UniversalCover currently supports CW complexes of "
                    f"dimension ≤ 2; base has {n} cells in dimension {dim}. "
                    "For higher-dimensional manifolds, use Path B "
                    "(`TwistedChainComplex(..., path='fox')`) on the relevant "
                    "skeletal subcomplex."
                )

    @property
    def pi1(self) -> FundamentalGroup:
        """Fundamental group of the base complex (in trace-generator names)."""
        return self._pi1

    @property
    def group_ring(self) -> FiniteGroupRing:
        """Group ring ℤ[π₁] backing the cover construction."""
        return self._group_ring

    @property
    def order(self) -> int:
        """Order |π₁| of the deck transformation group."""
        return self._group_ring.order

    @property
    def cover_cells(self) -> Dict[int, int]:
        """Cell counts per dimension in the cover (= |π₁| × base counts)."""
        return dict(self._cover_cells)

    @property
    def lifted_attaching(self) -> Dict[int, sp.csr_matrix]:
        """Lifted boundary matrices over ℤ on the cover (copies)."""
        return {k: v.copy() for k, v in self._lifted.items()}

    @property
    def deck_action(self) -> Dict[int, Dict[int, np.ndarray]]:
        """Per-element deck permutations of cover cells, keyed by dimension."""
        return {g: {k: v.copy() for k, v in dim_perm.items()}
                for g, dim_perm in self._deck.items()}

    def as_cw_complex(self) -> CWComplex:
        """Return the universal cover as a `CWComplex` over ℤ."""
        return CWComplex(
            cells=self._cover_cells,
            attaching_maps=self._lifted,
            dimensions=sorted(self._cover_cells.keys()),
            coefficient_ring="Z",
        )

    def as_chain_complex(self) -> ChainComplex:
        """Return the cellular chain complex of the universal cover."""
        return self.as_cw_complex().cellular_chain_complex()

    def as_result(self) -> UniversalCoverResult:
        """Package the lifted cover data into a `UniversalCoverResult`."""
        return UniversalCoverResult(
            cover_cells=dict(self._cover_cells),
            lifted_attaching={k: v.copy() for k, v in self._lifted.items()},
            deck_action={g: {k: v.copy() for k, v in dim_perm.items()}
                         for g, dim_perm in self._deck.items()},
            base_dimensions=sorted(self._base.cells.keys()),
            group_order=self._group_ring.order,
        )

    # ── unified Covering facade ──────────────────────────────────────────────

    def _sheet_perms(self) -> Dict[str, np.ndarray]:
        """Right-multiplication permutation of the |G| sheets per generator.

        For the universal cover the sheets are the group elements and a
        generator g_i acts by right multiplication ``h ↦ h·g_i``. Returned
        0-based: ``perm[h-1] = (h·g_i) - 1``.
        """
        cayley = self._group_ring.cayley
        n_g = self._group_ring.order
        gen_to_group = self._group_ring.generator_indices
        out: Dict[str, np.ndarray] = {}
        for name, gid in zip(self._pi1.generators, gen_to_group):
            perm = np.array(
                [int(cayley[h, gid - 1]) - 1 for h in range(n_g)], dtype=np.int64
            )
            out[name] = perm
        return out

    def covering_map(self) -> Dict[int, np.ndarray]:
        """Cellular covering map ``p: M̃ → M`` as ``{dim: array}``.

        Cover cell ``c·|G| + h`` projects to base cell ``c`` in each dimension.
        """
        n_g = self._group_ring.order
        out: Dict[int, np.ndarray] = {}
        for dim, n_cover in self._cover_cells.items():
            n_base = n_cover // n_g
            out[dim] = np.repeat(np.arange(n_base, dtype=np.int64), n_g)
        return out

    def deck_group(self) -> "DeckTransformationGroup":
        """Deck transformation group of M̃ → M (isomorphic to π₁)."""
        labels = self._group_ring.element_words
        cell_perms = {
            g: {k: v.copy() for k, v in dim_perm.items()}
            for g, dim_perm in self._deck.items()
        }
        return DeckTransformationGroup(
            order=self._group_ring.order,
            cell_perms=cell_perms,
            labels={i + 1: labels[i] for i in range(len(labels))},
        )

    def as_covering(self) -> "Covering":
        """Return the unified :class:`Covering` view of this universal cover."""
        return Covering(
            base=self._base,
            total_space=self.as_cw_complex(),
            covering_map=self.covering_map(),
            degree=self._group_ring.order,
            sheet_perms=self._sheet_perms(),
            base_generators=list(self._pi1.generators),
            _deck=self.deck_group(),
        )

    def lift_path(self, word: List[str], start_sheet: int = 0) -> List[int]:
        """Lift a based loop/word to the sequence of sheets it visits."""
        return self.as_covering().lift_path(word, start_sheet=start_sheet)

    def monodromy(self, word: List[str], start_sheet: int = 0) -> int:
        """Monodromy: the sheet a based loop lands on, starting from ``start_sheet``."""
        return self.as_covering().monodromy(word, start_sheet=start_sheet)

    def lift_cellular_path(self, edge_word: List[str], start_sheet: int = 0):
        """Lift a based edge-path to the cover 1-cells it traverses.

        See :meth:`Covering.lift_cellular_path`.
        """
        return self.as_covering().lift_cellular_path(edge_word, start_sheet=start_sheet)

    def monodromy_action(self) -> "GroupAction":
        """The π₁ action on the |π₁| sheets as a :class:`GroupAction`."""
        return self.as_covering().monodromy_action()

    def _build_cover(
        self,
        base: CWComplex,
        raw_generators: List[str],
        raw_relations: List[List[str]],
        raw_traces,
    ):
        n_g = self._group_ring.order
        cayley = self._group_ring.cayley  # 1-based
        gen_to_group = self._group_ring.generator_indices  # list aligned with self._pi1.generators
        gens_pi1 = list(self._pi1.generators)
        gen_name_to_pi1_idx = {g: i for i, g in enumerate(gens_pi1)}

        n_base = {dim: int(base.cells.get(dim, 0)) for dim in base.cells.keys()}
        n_cover = {dim: n_g * n for dim, n in n_base.items()}

        # Map each base 1-cell index → signed Cayley index of its monodromy.
        # `raw_traces` is a list of Pi1GeneratorTrace; each has an
        # `edge_index` (1-cell idx in base) and a `generator` token.
        edge_to_signed_group_idx = self._build_edge_monodromy(
            n_base.get(1, 0), raw_traces, gen_name_to_pi1_idx, gen_to_group
        )

        lifted: Dict[int, sp.csr_matrix] = {}

        # 1-skeleton lift.
        if 1 in base.attaching_maps and n_base.get(1, 0) > 0:
            lifted[1] = self._lift_d1(
                base, n_base, n_g, edge_to_signed_group_idx, cayley
            )

        # 2-skeleton lift via Fox derivatives.
        if 2 in base.attaching_maps and n_base.get(2, 0) > 0:
            lifted[2] = self._lift_d2(
                base, n_base, n_g, raw_relations, gen_name_to_pi1_idx,
                gen_to_group, cayley
            )

        # Deck action: permutations of the cover's cells.
        deck_action: Dict[int, Dict[int, np.ndarray]] = {}
        for g_idx in range(1, n_g + 1):
            dim_perm: Dict[int, np.ndarray] = {}
            for dim, n_b in n_base.items():
                size = n_g * n_b
                perm = np.zeros(size, dtype=np.int64)
                for c_idx in range(n_b):
                    for h_idx in range(1, n_g + 1):
                        # cell (c, h) under deck-action by g goes to (c, g·h)
                        target_h = int(cayley[g_idx - 1, h_idx - 1])
                        src = c_idx * n_g + (h_idx - 1)
                        dst = c_idx * n_g + (target_h - 1)
                        perm[src] = dst
                dim_perm[dim] = perm
            deck_action[g_idx] = dim_perm

        return n_cover, lifted, deck_action

    def _build_edge_monodromy(
        self,
        n_edges: int,
        raw_traces,
        gen_name_to_pi1_idx: Dict[str, int],
        gen_to_group: List[int],
    ) -> Dict[int, int]:
        """Map base 1-cell index → signed 1-based Cayley index of its monodromy.

        Each `Pi1GeneratorTrace` records the 1-cell index it covers and the
        symbolic generator token (positive or `g^-1`). For single-vertex
        complexes every 1-cell receives a trace; we fall back to identity
        monodromy for tree edges (which would only appear in multi-vertex
        complexes — not supported by `UniversalCover`).
        """
        edge_to_signed: Dict[int, int] = {}
        for tr in raw_traces:
            if tr.edge_index is None:
                continue
            tok = tr.generator
            base_name = tok[:-3] if tok.endswith("^-1") else tok
            pi1_idx = gen_name_to_pi1_idx.get(base_name)
            if pi1_idx is None:
                continue
            gid = int(gen_to_group[pi1_idx])
            signed = -gid if tok.endswith("^-1") else gid
            edge_to_signed[int(tr.edge_index)] = signed
        for e in range(n_edges):
            edge_to_signed.setdefault(e, 1)
        return edge_to_signed

    def _lift_d1(
        self,
        base: CWComplex,
        n_base: Dict[int, int],
        n_g: int,
        edge_to_signed_group_idx: Dict[int, int],
        cayley: np.ndarray,
    ) -> sp.csr_matrix:
        """Lift d_1 to the cover for single-vertex base complexes.

        For each base 1-cell e with monodromy g_e (signed Cayley index from
        traces), the cover boundary is `∂(e, g) = (v, g·g_e) − (v, g)`. The
        base d_1 matrix is identically zero in the single-vertex case
        (head minus tail at the same 0-cell cancels), so we drive the lift
        from the edge-monodromy table directly.
        """
        n1 = n_base.get(1, 0)
        n0 = n_base.get(0, 0)
        rows: List[int] = []
        cols: List[int] = []
        vals: List[int] = []
        for e_idx in range(n1):
            signed_g = edge_to_signed_group_idx.get(e_idx, 1)
            if signed_g > 0:
                gid = signed_g
            else:
                # Inverse generator: monodromy is the inverse element.
                base_gid = -signed_g
                gid = int(self._group_ring.inverse_indices[base_gid - 1])
            for g_idx in range(1, n_g + 1):
                target_h = int(cayley[g_idx - 1, gid - 1])
                # Head: (v, g·g_e), coefficient +1.
                rows.append(0 * n_g + (target_h - 1))
                cols.append(e_idx * n_g + (g_idx - 1))
                vals.append(1)
                # Tail: (v, g), coefficient -1.
                rows.append(0 * n_g + (g_idx - 1))
                cols.append(e_idx * n_g + (g_idx - 1))
                vals.append(-1)
        if not rows:
            return sp.csr_matrix((n0 * n_g, n1 * n_g), dtype=np.int64)
        mat = sp.coo_matrix(
            (vals, (rows, cols)),
            shape=(n0 * n_g, n1 * n_g),
            dtype=np.int64,
        ).tocsr()
        mat.sum_duplicates()
        return mat

    def _lift_d2(
        self,
        base: CWComplex,
        n_base: Dict[int, int],
        n_g: int,
        raw_relations: List[List[str]],
        gen_name_to_pi1_idx: Dict[str, int],
        gen_to_group: List[int],
        cayley: np.ndarray,
    ) -> sp.csr_matrix:
        n1 = n_base[1]
        n2 = n_base[2]
        if len(raw_relations) != n2:
            raise DimensionError(
                f"Number of raw π₁ relations ({len(raw_relations)}) does not "
                f"match number of 2-cells ({n2}); cannot lift d_2 unambiguously."
            )

        rows: List[int] = []
        cols: List[int] = []
        vals: List[int] = []

        for j, relator in enumerate(raw_relations):
            # Convert relator to signed-pi1-index list.
            signed = []
            for tok in relator:
                base_name = tok[:-3] if tok.endswith("^-1") else tok
                if base_name not in gen_name_to_pi1_idx:
                    raise GroupRingError(
                        f"Relator token {tok!r} references unknown generator."
                    )
                idx = gen_name_to_pi1_idx[base_name] + 1  # 1-based for Julia
                signed.append(-idx if tok.endswith("^-1") else idx)
            # For each generator g_i (1-based π₁ index), Fox derivative
            # ∂relator/∂g_i is a sum of ±group_elements.
            # We compute it Python-side because here the rep is the
            # regular representation (we don't need ρ matrices, we just
            # want the integer Z[G]-coefficient as a vector).
            for pi1_idx in range(1, len(gen_name_to_pi1_idx) + 1):
                zg_terms = self._fox_derivative_zg(
                    signed, pi1_idx, gen_to_group, cayley,
                    int(self._group_ring.inverse_indices[gen_to_group[pi1_idx - 1] - 1]),
                )
                # zg_terms is a list of (group_idx_1based, coeff).
                # In the cover, for each cover 2-cell index (j, g) we get
                # contributions at cover 1-cell (pi1_idx-1, g·h) with `coeff`
                # for each (h, coeff) in zg_terms.
                edge_base = (pi1_idx - 1) * n_g
                cell_base = j * n_g
                for g_idx in range(1, n_g + 1):
                    for h_idx, coeff in zg_terms:
                        if coeff == 0:
                            continue
                        target_h = int(cayley[g_idx - 1, h_idx - 1])
                        rows.append(edge_base + (target_h - 1))
                        cols.append(cell_base + (g_idx - 1))
                        vals.append(int(coeff))
        # Aggregate duplicate (row, col) entries.
        if not rows:
            return sp.csr_matrix((n1 * n_g, n2 * n_g), dtype=np.int64)
        mat = sp.coo_matrix(
            (vals, (rows, cols)),
            shape=(n1 * n_g, n2 * n_g),
            dtype=np.int64,
        ).tocsr()
        mat.sum_duplicates()
        return mat

    def _fox_derivative_zg(
        self,
        relator_signed: List[int],
        gen_pi1_idx_1based: int,
        gen_to_group: List[int],
        cayley: np.ndarray,
        inverse_of_gen_in_cayley: int,
    ) -> List[Tuple[int, int]]:
        """Compute ∂relator/∂g as a list of (group_idx_1based, coeff) pairs."""
        terms: Dict[int, int] = {}
        prefix = self._group_ring.identity_index  # 1-based
        for sg in relator_signed:
            gen = abs(sg)
            eps = 1 if sg > 0 else -1
            gid = int(gen_to_group[gen - 1])
            inv_gid = int(self._group_ring.inverse_indices[gid - 1])
            new_prefix = (
                int(cayley[prefix - 1, gid - 1]) if eps > 0
                else int(cayley[prefix - 1, inv_gid - 1])
            )
            if gen == gen_pi1_idx_1based:
                if eps > 0:
                    terms[prefix] = terms.get(prefix, 0) + 1
                else:
                    terms[new_prefix] = terms.get(new_prefix, 0) - 1
            prefix = new_prefix
        return [(g, c) for g, c in terms.items() if c != 0]


# ──────────────────────────────────────────────────────────────────────────────
# Unified covering API: Covering + DeckTransformationGroup + general covers
# ──────────────────────────────────────────────────────────────────────────────


def _invert_perm(perm: np.ndarray) -> np.ndarray:
    """Inverse of a 0-based permutation array."""
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm), dtype=perm.dtype)
    return inv


class DeckTransformationGroup:
    """The group of deck transformations of a covering, acting on cover cells.

    Overview:
        A deck transformation of a cover ``p: E → B`` is a self-homeomorphism of
        ``E`` commuting with ``p``. Cellularly it is a permutation of the cells of
        ``E`` in each dimension that covers the identity on ``B``. This object
        stores those per-dimension permutations for each group element and lets
        you act with them.

    Key facts:
        - For the universal cover the deck group is isomorphic to π₁(B).
        - A covering is *regular* (normal) iff the deck group acts transitively on
          each fibre, equivalently ``|deck group| == degree``.
        - A covering deck group always acts *freely* on the cells of the cover.
    """

    def __init__(
        self,
        order: int,
        cell_perms: Dict[Any, Dict[int, np.ndarray]],
        *,
        labels: Optional[Dict[Any, str]] = None,
    ) -> None:
        self._order = int(order)
        self._perms = cell_perms
        self._keys = list(cell_perms.keys())
        self._labels = labels or {k: str(k) for k in self._keys}

    @property
    def order(self) -> int:
        """Number of deck transformations."""
        return self._order

    @property
    def elements(self) -> List[Any]:
        """Group-element keys indexing the deck transformations."""
        return list(self._keys)

    def label(self, g: Any) -> str:
        """Human-readable label for element ``g``."""
        return self._labels.get(g, str(g))

    def permutation(self, g: Any, dim: int) -> np.ndarray:
        """Cell permutation of dimension ``dim`` induced by element ``g``."""
        return self._perms[g][dim]

    def act(self, g: Any, dim: int, cell: int) -> int:
        """Image of ``cell`` (dimension ``dim``) under deck element ``g``."""
        return int(self._perms[g][dim][int(cell)])

    def is_free(self) -> bool:
        """True iff no non-identity element fixes any cover cell."""
        for g, dim_perm in self._perms.items():
            for dim, perm in dim_perm.items():
                fixed = np.flatnonzero(perm == np.arange(len(perm)))
                if len(fixed) == len(perm):
                    continue  # identity on this dimension
                if len(fixed) > 0:
                    # a non-identity element fixing a cell ⇒ not free
                    return False
        return True

    def orbit(self, cell: int, dim: int = 0) -> List[int]:
        """Orbit of a cover ``cell`` (in dimension ``dim``) under the deck group."""
        cell = int(cell)
        seen = set()
        for g in self._keys:
            seen.add(int(self._perms[g][dim][cell]))
        return sorted(seen)

    def stabilizer(self, cell: int, dim: int = 0) -> List[Any]:
        """Deck elements fixing a cover ``cell`` (trivial for a free action)."""
        cell = int(cell)
        return [g for g in self._keys if int(self._perms[g][dim][cell]) == cell]

    def __len__(self) -> int:
        return self._order

    def __iter__(self):
        return iter(self._keys)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"DeckTransformationGroup(order={self._order})"


@dataclass
class Covering:
    """A covering space ``p: E → B`` with its combinatorial bookkeeping.

    Overview:
        Unifies the geometric and algebraic covers behind one interface. Holds
        the base ``B``, the total space ``E`` (as a ``CWComplex`` when realised),
        the cellular covering map ``p`` (cover cell index → base cell index per
        dimension), the ``degree`` (number of sheets), and — for covers arising
        from a π₁ action — the per-generator sheet permutations used for path
        lifting and monodromy.

    Construction:
        - ``UniversalCover.as_covering()`` for the universal cover.
        - :meth:`from_permutation_rep` for an arbitrary finite cover given by a
          transitive (or not) permutation action π₁ → S_n.
        - :meth:`from_subgroup` for the cover associated with a finite-index
          subgroup, via coset enumeration.
    """

    base: Any
    total_space: Optional[CWComplex]
    covering_map: Dict[int, np.ndarray]
    degree: int
    sheet_perms: Optional[Dict[str, np.ndarray]] = None
    base_generators: Optional[List[str]] = None
    _deck: Optional[DeckTransformationGroup] = None
    theorem_tag: str = "coverings.covering"
    contract_version: str = field(default_factory=lambda: CONTRACT_VERSION)

    # ── fibres and regularity ───────────────────────────────────────────────

    def fiber(self, base_cell: int, dim: int = 0) -> List[int]:
        """Cells of ``E`` (dimension ``dim``) lying over ``base_cell``."""
        cmap = self.covering_map.get(dim)
        if cmap is None:
            return []
        return [int(i) for i in np.flatnonzero(cmap == int(base_cell))]

    def deck_group(self) -> Optional[DeckTransformationGroup]:
        """Deck transformation group, or ``None`` if it could not be computed."""
        if self._deck is not None:
            return self._deck
        if self.sheet_perms is None:
            return None
        self._deck = self._compute_deck_group()
        return self._deck

    def is_regular(self) -> bool:
        """True iff the cover is regular (deck group transitive on fibres)."""
        dg = self.deck_group()
        return dg is not None and dg.order == self.degree

    # ── path lifting and monodromy ──────────────────────────────────────────

    def _sheet_perm_for_token(self, tok: str) -> np.ndarray:
        if self.sheet_perms is None:
            raise GroupRingError(
                "This covering has no sheet permutations; path lifting is "
                "unavailable (it was not built from a π₁ action)."
            )
        base_name = tok[:-3] if tok.endswith("^-1") else tok
        if base_name not in self.sheet_perms:
            raise GroupRingError(f"Unknown generator token {tok!r}.")
        perm = self.sheet_perms[base_name]
        return _invert_perm(perm) if tok.endswith("^-1") else perm

    def lift_path(self, word: List[str], start_sheet: int = 0) -> List[int]:
        """Lift a word in the base generators to the sheet sequence it visits.

        Returns ``[s_0, s_1, …, s_L]`` where ``s_0 = start_sheet`` and each step
        applies the next generator's sheet permutation. For a loop (relator) the
        final sheet equals ``start_sheet`` iff the loop lifts to a loop.
        """
        s = int(start_sheet)
        sheets = [s]
        for tok in word:
            perm = self._sheet_perm_for_token(tok)
            s = int(perm[s])
            sheets.append(s)
        return sheets

    def monodromy(self, word: List[str], start_sheet: int = 0) -> int:
        """End sheet of lifting ``word`` from ``start_sheet`` (the monodromy)."""
        return self.lift_path(word, start_sheet=start_sheet)[-1]

    def lift_cellular_path(
        self, edge_word: List[str], start_sheet: int = 0
    ) -> List[Tuple[int, int]]:
        """Lift a based edge-path to the sequence of cover 1-cells it traverses.

        Whereas :meth:`lift_path` returns the visited *sheets*, this returns the
        actual lifted path as cover 1-cells ``[(cell_index, sheet), …]`` — the
        concrete realisation of the lift in the total space. ``edge_word`` is a
        word in the base generators (= base 1-cells); the lift starts on
        ``start_sheet``.
        """
        if self.sheet_perms is None or self.base_generators is None:
            raise GroupRingError(
                "This covering has no sheet permutations / generator data; "
                "cellular path lifting is unavailable."
            )
        gen_to_cell = {g: i for i, g in enumerate(self.base_generators)}
        s = int(start_sheet)
        out: List[Tuple[int, int]] = []
        for tok in edge_word:
            base_name = tok[:-3] if tok.endswith("^-1") else tok
            cell_idx = gen_to_cell[base_name]
            if tok.endswith("^-1"):
                s_new = int(_invert_perm(self.sheet_perms[base_name])[s])
                out.append((cell_idx, s_new))  # traversed backwards from sheet s_new
                s = s_new
            else:
                out.append((cell_idx, s))
                s = int(self.sheet_perms[base_name][s])
        return out

    def monodromy_action(self) -> "GroupAction":
        """The π₁ action on the sheets as a :class:`GroupAction`."""
        if self.sheet_perms is None:
            raise GroupRingError("This covering has no sheet permutations.")
        return GroupAction(self.sheet_perms, self.degree)

    def orbit(self, sheet: int = 0) -> List[int]:
        """Orbit of a sheet under the monodromy action (all sheets iff connected)."""
        return self.monodromy_action().orbit(sheet)

    def stabilizer_generators(self, sheet: int = 0) -> List[List[str]]:
        """Schreier generators of the stabilizer of ``sheet``.

        These generate the subgroup ``H ≤ π₁`` this cover corresponds to.
        """
        return self.monodromy_action().stabilizer_generators(sheet)

    def subgroup_generators(self) -> List[List[str]]:
        """Generators of the defining subgroup ``H = π₁(cover)`` (stabilizer of sheet 0)."""
        return self.stabilizer_generators(0)

    def euler_characteristic_consistent(self) -> bool:
        """Check the multiplicativity χ(E) = degree · χ(B) when ``E`` is realised."""
        if self.total_space is None:
            return True
        chi_e = self.total_space.euler_characteristic()
        chi_b = (
            self.base.euler_characteristic()
            if hasattr(self.base, "euler_characteristic")
            else None
        )
        if chi_b is None:
            return True
        return chi_e == self.degree * chi_b

    # ── deck-group computation for permutation covers ─────────────────────────

    def _compute_deck_group(self) -> DeckTransformationGroup:
        """Deck group of a permutation cover = centraliser of the monodromy.

        A deck transformation is a sheet permutation τ commuting with every
        generator's permutation. For a transitive action τ is determined by the
        image of sheet 0, so we enumerate candidates and verify equivariance.
        """
        n = self.degree
        perms = self.sheet_perms or {}
        gen_perms = list(perms.values())
        # Connected component of sheet 0 under the action (transitive ⇒ all).
        reachable = self._orbit(0, gen_perms)
        taus: List[np.ndarray] = []
        for target in range(n):
            tau = self._equivariant_extension(target, gen_perms, reachable)
            if tau is not None and self._commutes(tau, gen_perms):
                taus.append(tau)
        # Lift each sheet permutation to per-dimension cover-cell permutations.
        cell_perms: Dict[Any, Dict[int, np.ndarray]] = {}
        labels: Dict[Any, str] = {}
        for idx, tau in enumerate(taus):
            dim_perm: Dict[int, np.ndarray] = {}
            for dim, cmap in self.covering_map.items():
                size = len(cmap)
                perm = np.arange(size, dtype=np.int64)
                for cell in range(size):
                    c = int(cmap[cell])
                    s = cell - c * n  # sheet (cover cell = c*n + s)
                    perm[cell] = c * n + int(tau[s])
                dim_perm[dim] = perm
            cell_perms[idx] = dim_perm
            labels[idx] = "id" if np.array_equal(tau, np.arange(n)) else f"τ{idx}"
        return DeckTransformationGroup(
            order=len(taus), cell_perms=cell_perms, labels=labels
        )

    @staticmethod
    def _orbit(start: int, gen_perms: List[np.ndarray]) -> set:
        seen = {start}
        stack = [start]
        while stack:
            s = stack.pop()
            for perm in gen_perms:
                for q in (int(perm[s]), int(_invert_perm(perm)[s])):
                    if q not in seen:
                        seen.add(q)
                        stack.append(q)
        return seen

    def _equivariant_extension(
        self, target: int, gen_perms: List[np.ndarray], reachable: set
    ) -> Optional[np.ndarray]:
        """Build τ with τ(0)=target and τ∘φ=φ∘τ by BFS, or None if inconsistent."""
        n = self.degree
        tau: Dict[int, int] = {0: target}
        queue = deque([0])
        while queue:
            s = queue.popleft()
            for perm in gen_perms:
                inv = _invert_perm(perm)
                for p in (perm, inv):
                    ns = int(p[s])
                    nt = int(p[tau[s]])
                    if ns in tau:
                        if tau[ns] != nt:
                            return None
                    else:
                        tau[ns] = nt
                        queue.append(ns)
        if len(tau) != len(reachable):
            return None  # only defined on the orbit; require full coverage
        arr = np.array([tau.get(i, i) for i in range(n)], dtype=np.int64)
        if len(set(arr.tolist())) != n:
            return None  # not a bijection
        return arr

    @staticmethod
    def _commutes(tau: np.ndarray, gen_perms: List[np.ndarray]) -> bool:
        for perm in gen_perms:
            if not np.array_equal(tau[perm], perm[tau]):
                return False
        return True

    # ── constructors for general covers ──────────────────────────────────────

    @classmethod
    def from_permutation_rep(
        cls,
        base: CWComplex,
        rep: Dict[str, Any],
    ) -> "Covering":
        """Build a finite cover of ``base`` from a permutation action of π₁.

        Args:
            base: A ``CWComplex`` with a single 0-cell and dimension ≤ 2 (the
                shape produced by π₁ extraction). Its 1-cells are the π₁
                generators and 2-cells are the relators.
            rep: ``{generator_name: permutation}`` where each permutation is a
                length-``n`` sequence describing where generator ``g`` sends the
                ``n`` sheets (0-based). The cover is connected iff the action is
                transitive.

        Returns:
            A :class:`Covering` whose ``total_space`` is the lifted ``CWComplex``
            over ℤ. ``degree == n``.
        """
        return _build_permutation_cover(base, rep)

    @classmethod
    def from_subgroup(
        cls,
        pi1: FundamentalGroup,
        subgroup: List[List[str]],
        *,
        base: Optional[CWComplex] = None,
        max_cosets: int = 1000,
    ) -> "Covering":
        r"""Build the cover associated with a finite-index subgroup ``H ≤ π₁``.

        The subgroup is given by a list of generator words. Coset enumeration
        (Todd–Coxeter over ``H``) yields the permutation action of π₁ on the
        cosets ``H\π₁``; that permutation rep then defines the cover.

        Args:
            pi1: The fundamental group.
            subgroup: Generators of ``H`` as relator-style token words.
            base: Optional ``CWComplex`` to realise the cover geometrically. When
                omitted the returned ``Covering`` has ``total_space=None`` but is
                still usable for lifting / monodromy via ``sheet_perms``.
            max_cosets: Coset-enumeration ceiling.

        Returns:
            A :class:`Covering`; ``degree`` equals the index ``[π₁ : H]``.
        """
        rep, _index = _coset_permutation_rep(
            list(pi1.generators),
            [list(r) for r in pi1.relations],
            [list(w) for w in subgroup],
            max_cosets=max_cosets,
        )
        if base is not None:
            return _build_permutation_cover(base, rep)
        degree = len(next(iter(rep.values()))) if rep else 1
        return cls(
            base=pi1,
            total_space=None,
            covering_map={},
            degree=degree,
            sheet_perms={k: np.asarray(v, dtype=np.int64) for k, v in rep.items()},
            base_generators=list(pi1.generators),
        )


def _build_permutation_cover(base: CWComplex, rep: Dict[str, Any]) -> Covering:
    """Realise a degree-n cover of ``base`` from a π₁ permutation rep ``rep``."""
    if not isinstance(base, CWComplex):
        raise DimensionError(
            f"Permutation cover requires a CWComplex base; got {type(base).__name__}."
        )
    n0 = int(base.cells.get(0, 0))
    if n0 != 1:
        raise DimensionError(
            f"Permutation cover requires exactly one 0-cell; got {n0}. "
            "Collapse a spanning tree of the 1-skeleton first."
        )
    for dim, n in base.cells.items():
        if dim >= 3 and n > 0:
            raise DimensionError(
                "Permutation cover currently supports CW complexes of dimension "
                f"≤ 2; base has {n} cells in dimension {dim}."
            )

    traces = extract_pi_1_with_traces(base, simplify=False, generator_mode="raw")
    gens = list(traces.generators)
    if set(rep.keys()) != set(gens):
        raise GroupRingError(
            f"Permutation rep keys {sorted(rep.keys())} must match the base's raw "
            f"π₁ generators {sorted(gens)}."
        )
    perms = {g: np.asarray(rep[g], dtype=np.int64) for g in gens}
    sizes = {len(p) for p in perms.values()}
    if len(sizes) != 1:
        raise GroupRingError("All permutations must act on the same number of sheets.")
    n = sizes.pop()
    for g, p in perms.items():
        if sorted(p.tolist()) != list(range(n)):
            raise GroupRingError(f"rep[{g!r}] is not a permutation of 0..{n - 1}.")

    n1 = int(base.cells.get(1, 0))
    n2 = int(base.cells.get(2, 0))

    gen_to_edge: Dict[str, int] = {}
    for tr in traces.traces:
        if tr.edge_index is not None:
            gen_to_edge[tr.generator] = int(tr.edge_index)
    # Single-vertex complexes: each edge is exactly one generator.
    edge_to_gen = {e: g for g, e in gen_to_edge.items()}

    # d1 lift: cover 1-cell (e, s) goes from vertex (0, s) to (0, perm_e[s]).
    rows: List[int] = []
    cols: List[int] = []
    vals: List[int] = []
    for e_idx in range(n1):
        g = edge_to_gen.get(e_idx)
        perm = perms[g] if g is not None else np.arange(n, dtype=np.int64)
        for s in range(n):
            t = int(perm[s])
            rows.append(0 * n + t)
            cols.append(e_idx * n + s)
            vals.append(1)
            rows.append(0 * n + s)
            cols.append(e_idx * n + s)
            vals.append(-1)
    d1 = sp.coo_matrix(
        (vals, (rows, cols)), shape=(n0 * n, n1 * n), dtype=np.int64
    ).tocsr()
    d1.sum_duplicates()

    lifted: Dict[int, sp.csr_matrix] = {1: d1} if n1 > 0 else {}

    # d2 lift: trace each relator word starting at each sheet.
    if n2 > 0:
        relations = [list(r) for r in traces.relations]
        if len(relations) != n2:
            raise DimensionError(
                f"Number of raw π₁ relations ({len(relations)}) does not match "
                f"the number of 2-cells ({n2})."
            )
        r2: List[int] = []
        c2: List[int] = []
        v2: List[int] = []
        for j, word in enumerate(relations):
            for s0 in range(n):
                current = s0
                for tok in word:
                    base_name = tok[:-3] if tok.endswith("^-1") else tok
                    e = gen_to_edge[base_name]
                    perm = perms[base_name]
                    if tok.endswith("^-1"):
                        new = int(_invert_perm(perm)[current])
                        r2.append(e * n + new)
                        c2.append(j * n + s0)
                        v2.append(-1)
                        current = new
                    else:
                        r2.append(e * n + current)
                        c2.append(j * n + s0)
                        v2.append(1)
                        current = int(perm[current])
        d2 = sp.coo_matrix(
            (v2, (r2, c2)), shape=(n1 * n, n2 * n), dtype=np.int64
        ).tocsr()
        d2.sum_duplicates()
        lifted[2] = d2

    cover_cells = {dim: n * int(cnt) for dim, cnt in base.cells.items()}
    total = CWComplex(
        cells=cover_cells,
        attaching_maps=lifted,
        dimensions=sorted(cover_cells.keys()),
        coefficient_ring="Z",
    )
    cmap = {
        dim: np.repeat(np.arange(int(cnt), dtype=np.int64), n)
        for dim, cnt in base.cells.items()
    }
    return Covering(
        base=base,
        total_space=total,
        covering_map=cmap,
        degree=n,
        sheet_perms=perms,
        base_generators=gens,
    )


def _coset_permutation_rep(
    generators: List[str],
    relations: List[List[str]],
    subgroup: List[List[str]],
    *,
    max_cosets: int = 1000,
) -> Tuple[Dict[str, List[int]], int]:
    r"""Todd–Coxeter coset enumeration of ``H\G`` → permutation rep of ``G``.

    Canonical HLT coset enumeration (definitions + queue-based coincidence
    processing, union-find on coset labels). Coset 0 is ``H`` itself. Returns
    ``({generator: permutation_on_cosets}, index)`` with 0-based coset labels.

    Raises:
        FundamentalGroupError: if the enumeration exceeds ``max_cosets`` (the
            subgroup may have infinite index).
    """
    gen_index = {g: i for i, g in enumerate(generators)}
    n_gen = len(generators)
    width = 2 * n_gen  # column letters: 2*i = g_i, 2*i+1 = g_i^{-1}

    def cols_of(word: List[str]) -> List[int]:
        out: List[int] = []
        for tok in word:
            base = tok[:-3] if tok.endswith("^-1") else tok
            gi = gen_index[base]
            out.append(2 * gi + 1 if tok.endswith("^-1") else 2 * gi)
        return out

    table: List[List[Optional[int]]] = [[None] * width]
    parent = [0]
    coincidence_q: deque = deque()

    def rep(c: int) -> int:
        root = c
        while parent[root] != root:
            root = parent[root]
        while parent[c] != root:
            parent[c], c = root, parent[c]
        return root

    def merge(a: int, b: int) -> None:
        a, b = rep(a), rep(b)
        if a != b:
            lo, hi = (a, b) if a < b else (b, a)
            parent[hi] = lo
            coincidence_q.append(hi)

    def define(c: int, x: int) -> int:
        if len(table) >= max_cosets:
            raise FundamentalGroupError(
                f"Coset enumeration exceeded max_cosets={max_cosets}; "
                "subgroup may have infinite index."
            )
        d = len(table)
        table.append([None] * width)
        parent.append(d)
        table[c][x] = d
        table[d][x ^ 1] = c
        return d

    def process_coincidences() -> None:
        while coincidence_q:
            dead = coincidence_q.popleft()
            for x in range(width):
                delta = table[dead][x]
                if delta is None:
                    continue
                table[dead][x] = None
                # remove the back edge delta --x^1--> dead
                if table[delta][x ^ 1] == dead:
                    table[delta][x ^ 1] = None
                mu = rep(dead)
                nu = rep(delta)
                if table[mu][x] is not None:
                    merge(nu, table[mu][x])
                elif table[nu][x ^ 1] is not None:
                    merge(mu, table[nu][x ^ 1])
                else:
                    table[mu][x] = nu
                    table[nu][x ^ 1] = mu

    def scan_and_fill(start: int, w: List[str]) -> None:
        cols = cols_of(w)
        if not cols:
            return
        while True:
            f = rep(start)
            i = 0
            r = len(cols)
            while i < r and table[f][cols[i]] is not None:
                f = rep(table[f][cols[i]])
                i += 1
            if i == r:  # scanned the whole word
                if f != rep(start):
                    merge(f, start)
                    process_coincidences()
                return
            b = rep(start)
            while r > i and table[b][cols[r - 1] ^ 1] is not None:
                b = rep(table[b][cols[r - 1] ^ 1])
                r -= 1
            if r == i:  # words overlap exactly → coincidence
                merge(f, b)
                process_coincidences()
                return
            if r == i + 1:  # single gap → deduction
                table[f][cols[i]] = b
                table[b][cols[i] ^ 1] = f
                return
            define(f, cols[i])  # gap > 1 → define and re-scan

    rel_cols = relations  # scanned as words below

    # Subgroup generators fix coset 0 (H · h = H).
    for h in subgroup:
        scan_and_fill(0, h)

    c = 0
    while c < len(table):
        if rep(c) != c:
            c += 1
            continue
        for r in rel_cols:
            scan_and_fill(c, r)
            if rep(c) != c:
                break
        if rep(c) != c:
            c += 1
            continue
        for x in range(width):
            if rep(c) != c:
                break
            if table[c][x] is None:
                define(c, x)
        c += 1

    # Compress to live cosets and read off the generator permutations.
    live = [i for i in range(len(table)) if rep(i) == i]
    relabel = {old: new for new, old in enumerate(live)}
    index = len(live)
    perms: Dict[str, List[int]] = {}
    for g, gi in gen_index.items():
        fwd = 2 * gi
        perm = [0] * index
        for old in live:
            tgt = table[old][fwd]
            if tgt is None:
                raise FundamentalGroupError(
                    "Coset table incomplete; increase max_cosets."
                )
            perm[relabel[old]] = relabel[rep(tgt)]
        perms[g] = perm
    return perms, index


# ──────────────────────────────────────────────────────────────────────────────
# Group actions: orbits and stabilizers
# ──────────────────────────────────────────────────────────────────────────────


class GroupAction:
    """A group acting on ``{0, …, n−1}`` via named generator permutations.

    Built from ``{generator_name: permutation}`` (0-based), this exposes the two
    things one most wants from an action: **orbits** and **stabilizers**. The
    stabilizer of a point is returned as a generating set of words (via Schreier's
    lemma) — for a covering's monodromy action these Schreier generators are
    exactly the generators of the subgroup ``H ≤ π₁`` defining the cover.
    """

    def __init__(self, gen_perms: Dict[str, np.ndarray], n_points: int) -> None:
        self.n = int(n_points)
        self.gens = {k: np.asarray(v, dtype=np.int64) for k, v in gen_perms.items()}
        self._inv = {k: _invert_perm(v) for k, v in self.gens.items()}

    def apply_token(self, tok: str, point: int) -> int:
        """Image of ``point`` under a signed generator token (``g`` or ``g^-1``)."""
        base = tok[:-3] if tok.endswith("^-1") else tok
        perm = self._inv[base] if tok.endswith("^-1") else self.gens[base]
        return int(perm[point])

    def apply_word(self, word: List[str], point: int) -> int:
        """Image of ``point`` under a word (applied left to right)."""
        for tok in word:
            point = self.apply_token(tok, point)
        return point

    def orbit(self, point: int) -> List[int]:
        """The orbit of ``point`` (sorted)."""
        seen = {int(point)}
        stack = [int(point)]
        while stack:
            p = stack.pop()
            for name in self.gens:
                for q in (int(self.gens[name][p]), int(self._inv[name][p])):
                    if q not in seen:
                        seen.add(q)
                        stack.append(q)
        return sorted(seen)

    def orbits(self) -> List[List[int]]:
        """Partition of ``{0,…,n−1}`` into orbits."""
        remaining = set(range(self.n))
        out: List[List[int]] = []
        while remaining:
            o = self.orbit(next(iter(remaining)))
            out.append(o)
            remaining -= set(o)
        return out

    def is_transitive(self) -> bool:
        """True iff there is a single orbit covering all points."""
        return self.n > 0 and len(self.orbit(0)) == self.n

    def _schreier_tree(self, base: int) -> Dict[int, List[str]]:
        """Transversal: point → word taking ``base`` to that point (BFS tree)."""
        words: Dict[int, List[str]] = {int(base): []}
        queue = deque([int(base)])
        while queue:
            p = queue.popleft()
            for name in self.gens:
                for tok, perm in ((name, self.gens[name]),
                                  (f"{name}^-1", self._inv[name])):
                    q = int(perm[p])
                    if q not in words:
                        words[q] = words[p] + [tok]
                        queue.append(q)
        return words

    def stabilizer_generators(self, point: int) -> List[List[str]]:
        """Schreier generators (words) for the stabilizer of ``point``.

        These generate ``Stab(point)`` as a subgroup of the acting group; for a
        cover's monodromy action they generate the subgroup ``H`` whose cover
        this is.
        """
        base = int(point)
        transversal = self._schreier_tree(base)
        gens_out: List[List[str]] = []
        seen: set = set()
        for gamma, t_word in transversal.items():
            for name in self.gens:
                for tok in (name, f"{name}^-1"):
                    moved = self.apply_token(tok, gamma)
                    t_moved = transversal.get(moved)
                    if t_moved is None:
                        continue
                    # Schreier generator: t_gamma · s · (t_moved)^-1
                    word = list(t_word) + [tok] + _inverse_token_word(t_moved)
                    word = _free_reduce_tokens(word)
                    if not word:
                        continue
                    key = tuple(word)
                    if key in seen:
                        continue
                    seen.add(key)
                    gens_out.append(word)
        return gens_out

    def stabilizer_index(self, point: int) -> int:
        """Index of the stabilizer = size of the orbit of ``point``."""
        return len(self.orbit(point))


def _inverse_token_word(word: List[str]) -> List[str]:
    out = []
    for tok in reversed(word):
        out.append(tok[:-3] if tok.endswith("^-1") else f"{tok}^-1")
    return out


def _free_reduce_tokens(word: List[str]) -> List[str]:
    stack: List[str] = []
    for tok in word:
        inv = tok[:-3] if tok.endswith("^-1") else f"{tok}^-1"
        if stack and stack[-1] == inv:
            stack.pop()
        else:
            stack.append(tok)
    return stack


# ──────────────────────────────────────────────────────────────────────────────
# Graph covers
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class GraphCovering:
    """A covering ``p: Ẽ → G`` of graphs, with path/walk lifting.

    Holds the base graph, the cover graph, and the covering map on vertices and
    edges. Unknown attributes/methods delegate to the cover graph, so a
    ``GraphCovering`` can be used wherever the cover :class:`Graph` is expected
    (``.betti_number(1)``, ``.is_tree``, ``.euler_characteristic()``, …).
    """

    base: Any
    cover: Any
    vertex_to_base: Dict[int, int]
    edge_to_base: List[int]
    degree: int
    kind: str = "voltage"
    sheet_of: Optional[Dict[int, int]] = None
    _adj_by_base_edge: Optional[Dict[int, List[Tuple[int, int]]]] = None

    def __getattr__(self, name: str):
        if name.startswith("__") or name in ("cover", "base"):
            raise AttributeError(name)
        return getattr(self.cover, name)

    def fiber(self, base_vertex: int) -> List[int]:
        """Cover vertices lying over ``base_vertex``."""
        return sorted(w for w, v in self.vertex_to_base.items() if v == int(base_vertex))

    def covering_map(self, cover_vertex: int) -> int:
        """Base vertex below ``cover_vertex``."""
        return self.vertex_to_base[int(cover_vertex)]

    def _cover_adj(self) -> Dict[int, List[Tuple[int, int]]]:
        if self._adj_by_base_edge is not None:
            return self._adj_by_base_edge
        adj: Dict[int, List[Tuple[int, int]]] = {w: [] for w in self.vertex_to_base}
        for cov_eid, (a, b) in enumerate(self.cover.edges):
            be = self.edge_to_base[cov_eid]
            adj.setdefault(a, []).append((b, be))
            adj.setdefault(b, []).append((a, be))
        object.__setattr__(self, "_adj_by_base_edge", adj)
        return adj

    def lift_walk(self, walk: List[int], start: Optional[int] = None) -> List[int]:
        """Lift a vertex walk in the base to a vertex walk in the cover.

        Args:
            walk: A sequence of base vertices ``[v0, v1, …]`` where consecutive
                vertices are adjacent in the base graph.
            start: A cover vertex lying over ``v0`` (defaults to the first lift in
                ``fiber(v0)``). For the universal cover this is the root.

        Returns:
            The lifted walk ``[w0, w1, …]`` of cover vertices, with
            ``covering_map(wi) == walk[i]``. Raises if a step cannot be lifted
            (e.g. a truncated universal-cover tree ran out of depth).
        """
        if not walk:
            return []
        base_edges = self.base.edges
        cover_adj = self._cover_adj()
        v0 = int(walk[0])
        current = int(start) if start is not None else (self.fiber(v0) or [None])[0]
        if current is None:
            raise ValueError(f"No lift of base vertex {v0} in this cover.")
        if self.vertex_to_base.get(current) != v0:
            raise ValueError(f"start {current} does not lie over base vertex {v0}.")
        lifted = [current]
        for a, b in zip(walk, walk[1:]):
            a, b = int(a), int(b)
            # base edge id realising step a→b
            be = None
            for eid, (x, y) in enumerate(base_edges):
                if {x, y} == {a, b} or (self.base.directed and (x, y) == (a, b)):
                    be = eid
                    break
            if be is None:
                raise ValueError(f"No base edge between {a} and {b}.")
            nxt = None
            for nbr, cov_be in cover_adj.get(current, ()):
                if cov_be == be and self.vertex_to_base.get(nbr) == b:
                    nxt = nbr
                    break
            if nxt is None:
                raise ValueError(
                    f"Cannot lift step {a}->{b} from cover vertex {current} "
                    "(cover may be truncated or the walk is invalid)."
                )
            current = nxt
            lifted.append(current)
        return lifted

    def render(self, root: Optional[int] = None) -> str:
        """ASCII rendering of the cover (a tree for universal covers)."""
        return self.cover.to_ascii_tree(root=root)

    def plot(
        self,
        *,
        depth: Optional[int] = None,
        lifted_paths: Optional[List[List[int]]] = None,
        ax: Optional["Any"] = None,
        with_labels: bool = True,
        node_color: str = "lightgreen",
        edge_color: str = "gray",
        **kwargs,
    ) -> "Any":
        """Draw the cover graph using NetworkX and Matplotlib.

        For universal covers, which are trees, this plots the tree. If ``depth``
        is provided, it plots the subtree up to that depth from the root.
        If ``lifted_paths`` is provided, it highlights these paths on top of the graph.

        Args:
            depth: Maximum depth from the root to plot (for universal covers).
            lifted_paths: A list of node sequences (paths) to highlight.
            ax: Optional matplotlib axes to draw on. If None, the current axes are used.
            with_labels: Whether to draw node labels.
            node_color: Color of the nodes.
            edge_color: Color of the edges.
            **kwargs: Additional arguments passed to ``networkx.draw()``.

        Returns:
            The matplotlib axes used for drawing.
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError as e:
            raise ImportError(
                "The 'plot' method requires 'networkx' and 'matplotlib'."
            ) from e

        nodes_to_plot = set(self.cover.vertices)
        if depth is not None and self.kind == "universal":
            # BFS from root (vertex 0)
            adj = self.cover.adjacency()
            from collections import deque

            queue = deque([(0, 0)])
            visited = {0}
            nodes_to_plot = {0}
            while queue:
                curr, d = queue.popleft()
                if d < depth:
                    for nbr, _eid, _orient in adj.get(curr, ()):
                        if nbr not in visited:
                            visited.add(nbr)
                            nodes_to_plot.add(nbr)
                            queue.append((nbr, d + 1))

        if self.cover._directed:
            G = nx.MultiDiGraph()
        else:
            G = nx.MultiGraph()

        G.add_nodes_from(nodes_to_plot)
        for u, v in self.cover._edge_list:
            if u in nodes_to_plot and v in nodes_to_plot:
                G.add_edge(u, v)

        if ax is None:
            ax = plt.gca()
            
        pos = kwargs.pop("pos", None)
        if pos is None:
            pos = nx.spring_layout(G)

        nx.draw(
            G,
            pos=pos,
            ax=ax,
            with_labels=with_labels,
            node_color=node_color,
            edge_color=edge_color,
            **kwargs,
        )
        
        if lifted_paths:
            cmap = plt.get_cmap("Set1")
            for idx, path in enumerate(lifted_paths):
                if not path:
                    continue
                path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
                # Filter to edges that exist in G
                path_edges = [(u, v) for u, v in path_edges if G.has_node(u) and G.has_node(v)]
                if path_edges:
                    color = cmap(idx % cmap.N)
                    nx.draw_networkx_edges(
                        G,
                        pos=pos,
                        edgelist=path_edges,
                        ax=ax,
                        edge_color=[color],
                        width=3.0,
                        alpha=0.8,
                    )
                    
        return ax


def graph_universal_cover(graph: "SimplicialComplex", depth: int = 3) -> GraphCovering:
    """Unrolled universal cover (tree) of a connected graph, truncated by depth.

    The universal cover of a graph is the tree of reduced (non-backtracking) edge
    paths from a base vertex. For a graph with cycles this tree is infinite, so it
    is generated by breadth-first expansion up to ``depth`` edges from the root.

    Args:
        graph: A :class:`~pysurgery.topology.graphs.Graph`.
        depth: Maximum number of edges from the root vertex.

    Returns:
        A :class:`GraphCovering` whose ``cover`` is a finite tree.
    """
    from pysurgery.topology.graphs import Graph

    verts = graph.vertices
    if not verts:
        empty = Graph.from_edges([], num_vertices=0)
        return GraphCovering(graph, empty, {}, [], 1, kind="universal")
    adjacency = graph.adjacency()
    root_base = verts[0]

    cover_edges: List[Tuple[int, int]] = []
    edge_to_base: List[int] = []
    vertex_to_base: Dict[int, int] = {0: root_base}
    next_id = 1
    # queue entries: (cover_node, base_vertex, incoming_edge_id, dist)
    queue = deque([(0, root_base, None, 0)])
    while queue:
        cnode, bvert, incoming, dist = queue.popleft()
        if dist >= depth:
            continue
        for nbr, eid, _orient in adjacency.get(bvert, ()):  # noqa: B007
            if eid == incoming:
                continue  # do not immediately backtrack along the same edge
            child = next_id
            next_id += 1
            vertex_to_base[child] = nbr
            cover_edges.append((cnode, child))
            edge_to_base.append(eid)
            queue.append((child, nbr, eid, dist + 1))
    cover = Graph.from_edges(
        cover_edges, num_vertices=next_id, coefficient_ring=graph.coefficient_ring
    )
    return GraphCovering(
        graph, cover, vertex_to_base, edge_to_base, degree=1, kind="universal"
    )


def cover_graph(graph: "SimplicialComplex", voltages: Dict[int, Any]) -> GraphCovering:
    """Finite cover of a graph from edge voltages (the derived/voltage graph).

    Args:
        graph: A :class:`~pysurgery.topology.graphs.Graph`.
        voltages: ``{edge_id: permutation}`` mapping an edge's index in
            ``graph.edges`` to a length-``k`` permutation of the ``k`` sheets.
            Edges absent from the dict get the identity. All permutations must
            act on the same number of sheets ``k``.

    Returns:
        A :class:`GraphCovering` whose ``cover`` covers ``graph`` ``k``-fold. For
        a connected base, χ(cover) = k · χ(base).
    """
    from pysurgery.topology.graphs import Graph

    edges = graph.edges
    sizes = {len(np.asarray(p)) for p in voltages.values()}
    if len(sizes) > 1:
        raise ValueError("All voltage permutations must act on the same #sheets.")
    k = sizes.pop() if sizes else 1
    verts = graph.vertices
    vidx = {v: i for i, v in enumerate(verts)}

    def cell(v: int, s: int) -> int:
        return vidx[v] * k + s

    vertex_to_base: Dict[int, int] = {}
    sheet_of: Dict[int, int] = {}
    for v in verts:
        for s in range(k):
            vertex_to_base[cell(v, s)] = v
            sheet_of[cell(v, s)] = s

    cover_edges: List[Tuple[int, int]] = []
    edge_to_base: List[int] = []
    for eid, (u, v) in enumerate(edges):
        perm = np.asarray(voltages.get(eid, list(range(k))), dtype=np.int64)
        if sorted(perm.tolist()) != list(range(k)):
            raise ValueError(f"voltages[{eid}] is not a permutation of 0..{k - 1}.")
        for s in range(k):
            cover_edges.append((cell(u, s), cell(v, int(perm[s]))))
            edge_to_base.append(eid)
    cover = Graph.from_edges(
        cover_edges,
        num_vertices=len(verts) * k,
        coefficient_ring=graph.coefficient_ring,
    )
    return GraphCovering(
        graph, cover, vertex_to_base, edge_to_base, degree=k,
        kind="voltage", sheet_of=sheet_of,
    )
