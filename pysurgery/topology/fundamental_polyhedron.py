"""Fundamental Polyhedron and Universal Cover Tiling.

This module provides tools for constructing the fundamental polyhedron of a 
manifold and exploring its universal cover via tiling.

Key Concepts:
    - **Fundamental Polyhedron**: A single topological n-ball formed by gluing 
      the n-simplices of a triangulation along a dual spanning tree.
    - **Dual Spanning Tree**: A tree in the dual graph (simplices as nodes, 
      shared faces as edges) that ensures the polyhedron is simply connected.
    - **Face Pairing**: Identifications between the boundary faces of the 
      fundamental polyhedron that recover the original manifold.
    - **Universal Cover Tiling**: The process of iteratively applying face 
      pairings to the fundamental polyhedron to tile the universal cover.

References:
    - Thurston, W. P. (1997). Three-Dimensional Geometry and Topology.
    - Ratcliffe, J. G. (2006). Foundations of Hyperbolic Manifolds.
"""

from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from pydantic import BaseModel, ConfigDict, Field
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_tree

from pysurgery.topology.complexes import SimplicialComplex, _normalize_simplex


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
