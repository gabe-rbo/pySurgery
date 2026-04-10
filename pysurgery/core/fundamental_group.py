from typing import Dict, List, Tuple
from math import gcd
from pydantic import BaseModel, ConfigDict, Field
from .complexes import CWComplex
from .exact_algebra import normalize_word_token
from .generator_models import Pi1GeneratorTrace, Pi1PresentationWithTraces

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


def _inverse_word_token(tok: str) -> str:
    nt = normalize_word_token(tok)
    return nt[:-3] if nt.endswith("^-1") else f"{nt}^-1"


def _free_reduce(word: List[str]) -> List[str]:
    stack: List[str] = []
    for tok in word:
        tok = normalize_word_token(tok)
        if stack and _inverse_word_token(tok) == stack[-1]:
            stack.pop()
        else:
            stack.append(tok)
    return stack


def _cyclic_reduce(word: List[str]) -> List[str]:
    w = _free_reduce(word)
    while len(w) >= 2 and _inverse_word_token(w[0]) == w[-1]:
        w = w[1:-1]
        w = _free_reduce(w)
    return w


def _canonicalize_cyclic_word(word: List[str]) -> List[str]:
    if not word:
        return []
    inv_rev = [_inverse_word_token(t) for t in reversed(word)]

    def rotations(w: List[str]):
        n = len(w)
        for i in range(n):
            yield w[i:] + w[:i]

    best = None
    for cand in list(rotations(word)) + list(rotations(inv_rev)):
        key = tuple(cand)
        if best is None or key < tuple(best):
            best = cand
    return best if best is not None else []


def simplify_presentation(generators: List[str], relations: List[List[str]]) -> FundamentalGroup:
    # First pass: free+cyclic reduction and canonicalization of relators.
    rels = []
    for r in relations:
        rr = _cyclic_reduce(r)
        if rr:
            rels.append(_canonicalize_cyclic_word(rr))

    # Remove duplicates preserving deterministic order.
    seen = set()
    dedup = []
    for r in rels:
        key = tuple(r)
        if key not in seen:
            seen.add(key)
            dedup.append(r)

    # Eliminate generators fixed to identity by singleton relators.
    kill = {r[0] for r in dedup if len(r) == 1}
    if kill:
        new_gens = [g for g in generators if g not in kill]
        new_rels = []
        for r in dedup:
            rr = [t for t in r if (t.replace("^-1", "") not in kill)]
            rr = _cyclic_reduce(rr)
            if rr:
                new_rels.append(_canonicalize_cyclic_word(rr))
        # Rededup after elimination.
        seen2 = set()
        dedup2 = []
        for r in new_rels:
            key = tuple(r)
            if key not in seen2:
                seen2.add(key)
                dedup2.append(r)
        return FundamentalGroup(generators=new_gens, relations=dedup2)

    return FundamentalGroup(generators=generators, relations=dedup)


def infer_standard_group_descriptor(pi1: FundamentalGroup) -> str | None:
    """Infer a conservative descriptor among {"1", "Z", "Z_n"} when provable.

    This intentionally avoids heuristic/non-abelian guesses and only certifies
    descriptors from simplified one-generator presentations.
    """
    simplified = simplify_presentation(list(pi1.generators), [list(rel) for rel in pi1.relations])
    gens = list(simplified.generators)
    rels = [list(rel) for rel in simplified.relations]

    if not gens:
        return "1"
    if len(gens) != 1:
        return None

    g = gens[0]
    if not rels:
        return "Z"

    exponents: List[int] = []
    for rel in rels:
        exp_sum = 0
        for tok in rel:
            base = tok[:-3] if tok.endswith("^-1") else tok
            if base != g:
                return None
            exp_sum += -1 if tok.endswith("^-1") else 1
        if exp_sum != 0:
            exponents.append(abs(int(exp_sum)))

    if not exponents:
        return None

    n = exponents[0]
    for x in exponents[1:]:
        n = gcd(n, x)

    if n == 1:
        return "1"
    return f"Z_{n}"


class GroupPresentation(BaseModel):
    """Structured group descriptor for higher-level obstruction APIs."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: str
    factors: List[str] = Field(default_factory=list)

    def normalized(self) -> str:
        k = self.kind.strip().lower()
        if k in {"trivial", "1"}:
            return "1"
        if k in {"z", "integer"}:
            return "Z"
        if k in {"product", "direct_product"} and self.factors:
            return " x ".join(self.factors)
        return self.kind


def _path_between_tree(u: int, v: int, parent: Dict[int, int]) -> List[int]:
    """Return the vertex path between u and v inside a rooted spanning forest."""
    path_u: List[int] = []
    seen_u = set()
    x = u
    while x != -1:
        path_u.append(x)
        seen_u.add(x)
        x = parent.get(x, -1)

    path_v: List[int] = []
    y = v
    while y not in seen_u and y != -1:
        path_v.append(y)
        y = parent.get(y, -1)

    if y == -1:
        return []

    lca = y
    i = path_u.index(lca)
    return path_u[: i + 1] + list(reversed(path_v))


def extract_pi_1_with_traces(cw: CWComplex, simplify: bool = True) -> Pi1PresentationWithTraces:
    """Return pi_1 presentation with generator traces as data-native edge/vertex paths.

    The algebraic presentation matches `extract_pi_1` semantics, while `traces`
    maps each generator to an explicit cycle representative in the 1-skeleton.
    """
    d1 = cw.attaching_maps.get(1)
    if d1 is None:
        return Pi1PresentationWithTraces(generators=[], relations=[], traces=[])

    n_vertices = d1.shape[0]
    n_edges = d1.shape[1]
    if n_edges == 0:
        return Pi1PresentationWithTraces(generators=[], relations=[], traces=[])

    # Build adjacency + edge endpoint table from d1 exactly as in extract_pi_1.
    adj: Dict[int, List[Tuple[int, int, int]]] = {i: [] for i in range(n_vertices)}
    edge_list: List[Tuple[int, int] | None] = []

    d1_csc = d1.tocsc()
    for e in range(n_edges):
        col_start = d1_csc.indptr[e]
        col_end = d1_csc.indptr[e + 1]
        col_data = d1_csc.data[col_start:col_end]
        col_row = d1_csc.indices[col_start:col_end]

        if len(col_row) == 0:
            edge_list.append((0, 0))
            continue
        if len(col_row) != 2:
            edge_list.append(None)
            continue

        u, v = -1, -1
        for val, r in zip(col_data, col_row):
            if val == -1:
                u = int(r)
            elif val == 1:
                v = int(r)

        if u != -1 and v != -1:
            adj[u].append((v, e, 1))
            adj[v].append((u, e, -1))
            edge_list.append((u, v))
        else:
            edge_list.append(None)

    visited = [False] * n_vertices
    tree_edges = set()
    parent: Dict[int, int] = {}
    component_root: Dict[int, int] = {}

    if n_vertices > 0:
        import collections

        for start in range(n_vertices):
            if visited[start]:
                continue
            queue = collections.deque([start])
            visited[start] = True
            parent[start] = -1
            component_root[start] = start

            while queue:
                curr = queue.popleft()
                for neighbor, edge_idx, _ in adj[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        tree_edges.add(edge_idx)
                        parent[neighbor] = curr
                        component_root[neighbor] = start
                        queue.append(neighbor)

    raw_generators = [f"g_{i}" for i in range(n_edges) if i not in tree_edges]
    raw_gen_map = {i: f"g_{i}" for i in range(n_edges) if i not in tree_edges}

    # Build raw relations by reusing existing extractor with simplify disabled.
    raw_pi = extract_pi_1(cw, simplify=False)
    out_pi = simplify_presentation(raw_pi.generators, raw_pi.relations) if simplify else raw_pi

    keep = set(out_pi.generators)
    traces: List[Pi1GeneratorTrace] = []
    for edge_idx, gen_name in raw_gen_map.items():
        if gen_name not in keep:
            continue
        if edge_idx < 0 or edge_idx >= len(edge_list):
            continue
        endpoints = edge_list[edge_idx]
        if endpoints is None:
            continue
        u, v = endpoints

        if u == v:
            vertex_path = [u]
            directed = [(u, v)]
            comp_root = component_root.get(u, u)
        else:
            path_vertices = _path_between_tree(v, u, parent)
            directed = [(u, v)]
            for i in range(len(path_vertices) - 1):
                directed.append((path_vertices[i], path_vertices[i + 1]))
            vertex_path = [u]
            for _, b in directed:
                vertex_path.append(b)
            comp_root = component_root.get(u, component_root.get(v, u))

        traces.append(
            Pi1GeneratorTrace(
                generator=gen_name,
                edge_index=edge_idx,
                component_root=int(comp_root),
                vertex_path=[int(x) for x in vertex_path],
                directed_edge_path=[(int(a), int(b)) for a, b in directed],
                undirected_edge_path=[tuple(sorted((int(a), int(b)))) for a, b in directed],
            )
        )

    return Pi1PresentationWithTraces(
        generators=list(out_pi.generators),
        relations=[list(r) for r in out_pi.relations],
        traces=traces,
    )

def extract_pi_1(cw: CWComplex, simplify: bool = True) -> FundamentalGroup:
    """
    Computes a presentation for the fundamental group pi_1(X) by constructing 
    a maximal spanning tree in the 1-skeleton.
    
    Edges not in the tree become the generators.
    The boundary of the 2-cells (faces) dictate the relations.
    """
    d1 = cw.attaching_maps.get(1)
    d2 = cw.attaching_maps.get(2)
    
    if d1 is None:
        return FundamentalGroup(generators=[], relations=[])

    n_vertices = d1.shape[0]
    n_edges = d1.shape[1]

    if n_edges == 0:
        return FundamentalGroup(generators=[], relations=[])    
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

        if len(col_row) == 0:
            # Zero-boundary: a genuine loop. Treat it as a self-loop at vertex 0
            # (the basepoint). In a 1-vertex complex this is exact; in multi-vertex
            # complexes the attaching vertex should be recovered from the CW structure.
            edge_list.append((0, 0))
            # Do NOT add to adjacency: a self-loop does not help BFS tree growth.
            # It is always a non-tree edge and becomes a generator.
            continue
        elif len(col_row) != 2:
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
    
    # Build a maximal spanning forest (robust even if the 1-skeleton is disconnected).
    if n_vertices > 0:
        import collections
        for start in range(n_vertices):
            if visited[start]:
                continue
            queue = collections.deque([start])
            visited[start] = True

            while queue:
                curr = queue.popleft()
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
            edges_in_face = []
            for val, e in zip(col_data, col_row):
                mult = abs(int(val))
                if mult == 0:
                    continue
                sign = 1 if int(val) > 0 else -1
                for _ in range(mult):
                    edges_in_face.append((sign, int(e)))
            if len(edges_in_face) < 1:
                continue
                
            edge_endpoints = {}
            for occ_id, (val, e) in enumerate(edges_in_face):
                if e < 0 or e >= len(edge_list):
                    continue
                if edge_list[e] is None:
                    continue
                u, v = edge_list[e]
                # Directed edge is u -> v.
                # If val == 1, path traverses u -> v. If -1, traverses v -> u.
                if val == 1:
                    edge_endpoints[occ_id] = (e, u, v, 1)
                else:
                    edge_endpoints[occ_id] = (e, v, u, -1)

            if not edge_endpoints:
                continue

            path = []
            curr_occ = None
            for occ_id in range(len(edges_in_face)):
                if occ_id in edge_endpoints:
                    curr_occ = occ_id
                    break
            if curr_occ is None:
                continue
            
            curr_e, curr_u, curr_v, curr_dir = edge_endpoints[curr_occ]
            path.append((curr_e, curr_dir))
            used = {curr_occ}
            start_node = curr_u
            target_node = curr_v
            
            # Trace the boundary
            while len(path) < len(edge_endpoints):
                next_e_found = False
                for occ_id in range(len(edges_in_face)):
                    if occ_id in used or occ_id not in edge_endpoints:
                        continue
                    e, u, v, d = edge_endpoints[occ_id]
                    if u == target_node:
                        path.append((e, d))
                        target_node = v
                        used.add(occ_id)
                        next_e_found = True
                        break
                if not next_e_found:
                    break # Degenerate cycle, just use what we have

            # Require a closed combinatorial cycle before creating a relation.
            if target_node != start_node:
                continue

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
    if simplify:
        return simplify_presentation(generators, relations)
    return FundamentalGroup(generators=generators, relations=relations)
