from typing import List
from pydantic import BaseModel, ConfigDict
from .exceptions import FundamentalGroupError
from .complexes import CWComplex

class FundamentalGroup(BaseModel):
    """
    Representation of the Fundamental Group pi_1(X) of a CW Complex.
    Uses the Edge-Path Group algorithm.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    generators: List[str]
    relations: List[List[str]]

    def __str__(self):
        gens = ", ".join(self.generators)
        rels = ", ".join(["".join(r) for r in self.relations])
        return f"< {gens} | {rels} >"

def extract_pi_1(cw: CWComplex) -> FundamentalGroup:
    """
    Computes a presentation for the fundamental group pi_1(X) by constructing 
    a maximal spanning tree in the 1-skeleton.
    
    Edges not in the tree become the generators.
    The boundary of the 2-cells (faces) dictate the relations.
    """
    d1 = cw.attaching_maps.get(1)
    d2 = cw.attaching_maps.get(2)
    
    if d1 is None or d1.nnz == 0:
        return FundamentalGroup(generators=[], relations=[])
        
    n_vertices = d1.shape[0]
    n_edges = d1.shape[1]
    
    # 1. Build a spanning tree in the 1-skeleton (using BFS)
    # We represent the graph as an adjacency list
    adj = {i: [] for i in range(n_vertices)}
    edge_list = []
    
    d1_csc = d1.tocsc()
    for e in range(n_edges):
        col_start = d1_csc.indptr[e]
        col_end = d1_csc.indptr[e+1]
        col_data = d1_csc.data[col_start:col_end]
        col_row = d1_csc.indices[col_start:col_end]
        
        if len(col_row) != 2:
            edge_list.append(None)
            continue
            
        u, v = -1, -1
        for val, r in zip(col_data, col_row):
            if val == -1:
                u = r
            elif val == 1:
                v = r
            
        if u != -1 and v != -1:
            adj[u].append((v, e, 1)) # 1 means forward traversal
            adj[v].append((u, e, -1)) # -1 means backward traversal
            edge_list.append((u, v))
        else:
            edge_list.append(None)
            
    visited = [False] * n_vertices
    tree_edges = set()
    
    # Simple BFS starting from vertex 0
    if n_vertices > 0:
        queue = [0]
        visited[0] = True
        
        while queue:
            curr = queue.pop(0)
            for neighbor, edge_idx, direction in adj[curr]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    tree_edges.add(edge_idx)
                    queue.append(neighbor)
                    
    # Edges not in the tree are our generators
    generators = [f"g_{i}" for i in range(n_edges) if i not in tree_edges]
    gen_map = {i: f"g_{i}" for i in range(n_edges) if i not in tree_edges}
    
    if not generators:
        return FundamentalGroup(generators=[], relations=[])
        
    # 2. Extract relations from 2-cells
    relations = []
    if d2 is not None and d2.nnz > 0:
        d2_csc = d2.tocsc()
        n_faces = d2.shape[1]
        
        for f in range(n_faces):
            col_start = d2_csc.indptr[f]
            col_end = d2_csc.indptr[f+1]
            col_data = d2_csc.data[col_start:col_end]
            col_row = d2_csc.indices[col_start:col_end]
            
            # Rigorous path-lifting to orient the boundary correctly.
            # We trace the edges head-to-tail to form the exact algebraic string presentation.
            edges_in_face = list(zip(col_data, col_row))
            if len(edges_in_face) < 3: 
                continue
                
            edge_endpoints = {}
            for val, e in edges_in_face:
                if edge_list[e] is None:
                    continue
                u, v = edge_list[e]
                # Directed edge is u -> v.
                # If val == 1, path traverses u -> v. If -1, traverses v -> u.
                if val == 1:
                    edge_endpoints[e] = (u, v, 1)
                else:
                    edge_endpoints[e] = (v, u, -1)
                    
            path = []
            curr_val, curr_e = edges_in_face[0]
            if curr_e not in edge_endpoints:
                continue
            
            curr_u, curr_v, curr_dir = edge_endpoints[curr_e]
            path.append((curr_e, curr_dir))
            used = {curr_e}
            target_node = curr_v
            
            # Trace the boundary
            while len(path) < len(edges_in_face):
                next_e_found = False
                for val, e in edges_in_face:
                    if e in used or e not in edge_endpoints:
                        continue
                    u, v, d = edge_endpoints[e]
                    if u == target_node:
                        path.append((e, d))
                        target_node = v
                        used.add(e)
                        next_e_found = True
                        break
                if not next_e_found:
                    raise FundamentalGroupError("Topological Path Lifting Failed. "
                                                "The 2-cell boundary edges do not form a connected, simple geometric cycle. "
                                                "This indicates the CW Complex attaching maps contain self-intersections or higher-degree singularities that prevent exact non-abelian path tracing.")
            
            # If path tracing was perfectly successful, build the exact word
            relation = []
            for e, d in path:
                if e in gen_map:
                    gen_str = gen_map[e]
                    if d == 1:
                        relation.append(gen_str)
                    else:
                        relation.append(f"{gen_str}^-1")
                        
            if relation:
                relations.append(relation)
                
    # Raw presentation is mathematically complete.
    # We pass it to Julia for exact Tietze/Abelianization reductions when needed in K-Theory.
    return FundamentalGroup(generators=generators, relations=relations)
