import collections
from typing import Dict, List
from math import gcd
import numpy as np
import sympy as sp
from scipy.sparse import csr_matrix
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .complexes import CWComplex
from .exact_algebra import normalize_word_token
from .generator_models import Pi1GeneratorTrace, Pi1PresentationWithTraces
from ..bridge.julia_bridge import julia_engine


class FundamentalGroup(BaseModel):
    """Presentation of the Fundamental Group π₁(X) = ⟨generators | relations⟩ of a CW complex.
    
    What is the Fundamental Group?:
        The fundamental group π₁(X) measures 1-dimensional "holes" and is the first non-trivial 
        homotopy group. It captures information about loops in the space:
        - π₁(S¹) = ℤ (one independent loop around the circle)
        - π₁(S²) = 1 (trivial; all loops contract on the sphere)
        - π₁(torus) = ℤ ⊕ ℤ (two independent loops)
        - π₁(RP²) = ℤ/2ℤ (nontrivial 2-torsion)
    
    Presentation Format:
        Given as generators g₁, g₂, ... and relations r₁, r₂, ... such that 
        π₁(X) ≅ ⟨g₁, g₂, ... | r₁, r₂, ...⟩.
        Each relation is a word that equals the identity, e.g., a relation "aba^-1b^-1" means 
        the commutator [a,b] = 1 in the group.
    
    Orientation Character (w₁):
        Maps each generator to ±1, encoding non-orientability. A generator with w₁(g) = -1 
        means traversing g reverses orientation (e.g., generators of Klein bottle's π₁).
    
    Attributes:
        generators: List of generator symbols (strings) representing independent elements.
        relations: List of relations, where each is a list of tokens (strings like 'a', 'b^-1').
        orientation_character: Dict[str → {1, -1}] recording orientation-reversing generators.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    generators: List[str]
    relations: List[List[str]]
    orientation_character: Dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _fill_default_w1(self):
        """Ensure all generators have an entry in orientation_character."""
        for g in self.generators:
            if g not in self.orientation_character:
                self.orientation_character[g] = 1
        return self

    def __str__(self):
        """Format the presentation as `<g1, g2 | r1, r2, ...>` with memory efficiency.

        Returns:
            A string representation of the group presentation.
        """
        import io
        buf = io.StringIO()
        buf.write("< ")
        buf.write(", ".join(self.generators))
        buf.write(" | ")
        
        # Stream relators to avoid building one giant string if possible
        # but join is already somewhat optimized. We'll join them once.
        buf.write(", ".join("".join(r) for r in self.relations))
        
        buf.write(" >")
        return buf.getvalue()


def _inverse_word_token(tok: str) -> str:
    """Return the formal inverse of a normalized generator token.

    Args:
        tok: The generator token to invert (e.g., 'a' or 'a^-1').

    Returns:
        The inverted token string.
    """
    nt = normalize_word_token(tok)
    return nt[:-3] if nt.endswith("^-1") else f"{nt}^-1"


def _free_reduce(word: List[str]) -> List[str]:
    """Free-reduce a word by cancelling adjacent inverse token pairs.

    Args:
        word: A list of generator tokens representing a word in the group.

    Returns:
        The free-reduced word as a list of tokens.
    """
    stack: List[str] = []
    for tok in word:
        tok = normalize_word_token(tok)
        if stack and _inverse_word_token(tok) == stack[-1]:
            stack.pop()
        else:
            stack.append(tok)
    return stack


def _cyclic_reduce(word: List[str]) -> List[str]:
    """Cyclically reduce a word after free reduction.

    Args:
        word: A list of generator tokens.

    Returns:
        The cyclically reduced word.
    """
    w = _free_reduce(word)
    while len(w) >= 2 and _inverse_word_token(w[0]) == w[-1]:
        w = w[1:-1]
        w = _free_reduce(w)
    return w


def _canonicalize_cyclic_word(word: List[str]) -> List[str]:
    """Canonicalize a cyclic word up to rotation and inversion using Booth's Algorithm.

    This normalization gives deterministic relator representatives for deduping
    in O(N) time.

    Args:
        word: A list of generator tokens representing a cyclic word.

    Returns:
        The canonicalized word.
    """
    if not word:
        return []

    def least_rotation(s: List[str]) -> List[str]:
        n = len(s)
        ss = s + s
        f = [-1] * (2 * n)
        k = 0
        for j in range(1, 2 * n):
            sj = ss[j]
            i = f[j - k - 1]
            while i != -1 and sj != ss[k + i + 1]:
                if sj < ss[k + i + 1]:
                    k = j - i - 1
                i = f[i]
            if sj != ss[k + i + 1]:
                if sj < ss[k + i + 1]:
                    k = j
                f[j - k] = -1
            else:
                f[j - k] = i + 1
        return ss[k : k + n]

    inv_rev = [_inverse_word_token(t) for t in reversed(word)]
    cand1 = least_rotation(word)
    cand2 = least_rotation(inv_rev)
    return cand1 if tuple(cand1) < tuple(cand2) else cand2


def _normalize_pi1_mode(
    generator_mode: str = "optimized", mode: str | None = None
) -> str:
    """Normalize/validate the user-facing pi1 generator mode selector.

    Args:
        generator_mode: The requested mode ('raw' or 'optimized'). Defaults to 'optimized'.
        mode: Optional override/alias for generator_mode.

    Returns:
        The normalized mode string.

    Raises:
        ValueError: If the chosen mode is not 'raw' or 'optimized'.
    """
    chosen = mode if mode is not None else generator_mode
    chosen = str(chosen).strip().lower()
    if chosen not in {"raw", "optimized"}:
        raise ValueError("generator_mode must be 'raw' or 'optimized'")
    return chosen


def _token_base(tok: str) -> str:
    """Return the unsigned generator name from a word token.

    Args:
        tok: The generator token (e.g., 'g_1' or 'g_1^-1').

    Returns:
        The base generator name (e.g., 'g_1').
    """
    nt = normalize_word_token(tok)
    return nt[:-3] if nt.endswith("^-1") else nt


def _inverse_word(word: List[str]) -> List[str]:
    """Return the group inverse of a word (reverse order + invert each token).

    Args:
        word: A list of generator tokens.

    Returns:
        The inverted word as a list of tokens.
    """
    return [_inverse_word_token(t) for t in reversed(word)]


def _normalize_relations(relations: List[List[str]]) -> List[List[str]]:
    """Apply reduction/canonicalization and deterministic deduplication to relators.

    Args:
        relations: A list of relators, each a list of generator tokens.

    Returns:
        A list of normalized and deduplicated relators.
    """
    rels: List[List[str]] = []
    for r in relations:
        rr = _cyclic_reduce(r)
        if rr:
            rels.append(_canonicalize_cyclic_word(rr))
    seen = set()
    dedup: List[List[str]] = []
    for r in rels:
        key = tuple(r)
        if key not in seen:
            seen.add(key)
            dedup.append(r)
    return dedup


def _kill_singleton_generators(
    generators: List[str], relations: List[List[str]]
) -> tuple[List[str], List[List[str]], set[str]]:
    """Eliminate generators forced to identity by singleton relators.

    Args:
        generators: Current list of generator names.
        relations: Current list of relators.

    Returns:
        A tuple containing:
            - The updated list of generators.
            - The updated list of relators.
            - A set of generator names that were "killed" (removed).
    """
    kill = {_token_base(r[0]) for r in relations if len(r) == 1}
    if not kill:
        return generators, relations, set()
    new_gens = [g for g in generators if g not in kill]
    new_rels = [[t for t in r if _token_base(t) not in kill] for r in relations]
    return new_gens, new_rels, kill


def _solve_generator_from_relator(relator: List[str], idx: int) -> List[str]:
    """Solve a one-occurrence relator for its target generator token.

    Args:
        relator: A relator containing the generator to be solved.
        idx: The index of the generator token in the relator.

    Returns:
        A list of tokens representing the replacement word for the solved generator.
    """
    tok = normalize_word_token(relator[idx])
    rest = relator[idx + 1 :] + relator[:idx]
    return _free_reduce(rest if tok.endswith("^-1") else _inverse_word(rest))


def _substitute_generator_word(
    relator: List[str], target: str, rhs: List[str]
) -> List[str]:
    """Substitute a generator by a solved word in one relator.

    Args:
        relator: The relator to perform substitution in.
        target: The base name of the generator to substitute.
        rhs: The replacement word for the target generator.

    Returns:
        The updated relator after substitution.
    """
    out: List[str] = []
    inv_rhs = _inverse_word(rhs)
    for tok in relator:
        nt = normalize_word_token(tok)
        if _token_base(nt) != target:
            out.append(nt)
            continue
        out.extend(inv_rhs if nt.endswith("^-1") else rhs)
    return out


def _find_substitution_move(
    generators: List[str], relations: List[List[str]]
) -> tuple[str, int, List[str]] | None:
    """Find a length-minimizing one-occurrence substitution candidate.

    Heuristic: Select generator g that minimizes (len(rhs) - 1) * (count(g) - 1)
    to prevent exponential growth in relator lengths during Tietze reduction.

    Args:
        generators: Current list of generator names.
        relations: Current list of relators.

    Returns:
        A tuple `(generator, defining_relator_index, replacement_word)` if a
        candidate is found, otherwise None.
    """
    candidates = []
    g_counts = collections.Counter()
    for rel in relations:
        for tok in rel:
            g_counts[_token_base(tok)] += 1

    for g in sorted(generators): # Deterministic iteration
        count = g_counts[g]
        if count == 0:
            continue
        for rel_idx, rel in enumerate(relations):
            occ = [i for i, tok in enumerate(rel) if _token_base(tok) == g]
            if len(occ) == 1:
                idx = occ[0]
                rhs = _solve_generator_from_relator(rel, idx)
                # Cost is net change in total presentation length
                cost = (len(rhs) - 1) * (count - 1)
                candidates.append((cost, g, rel_idx, rhs))

    if not candidates:
        return None

    # Sort by cost, then generator name for determinism
    candidates.sort(key=lambda x: (x[0], x[1]))
    best = candidates[0]
    return best[1], best[2], best[3]


def simplify_presentation(
    generators: List[str], 
    relations: List[List[str]],
    backend: str = "auto"
) -> FundamentalGroup:
    """Simplify a finitely presented group using a deterministic Tietze-lite loop.

    What is Being Computed?:
        Reduces the complexity of a group presentation ⟨G | R⟩ by iteratively 
        removing redundant generators and simplifying relations while preserving 
        the abstract group isomorphism.

    Algorithm:
        1. Free reduction and cyclic reduction of all relators.
        2. Eliminate "singleton" generators that appear in a relation of length 1 (g = 1).
        3. Identify generators that appear exactly once in a relation and substitute 
           them out (Tietze moves).
        4. Canonicalize and deduplicate relators after each substitution.
        5. Repeat until no further simplifications are possible or limit is reached.

    Preserved Invariants:
        - Group Isomorphism: The resulting presentation represents the same abstract group.
        - All group-theoretic invariants (abelianization, center, etc.) remain unchanged.

    Args:
        generators: List of generator names in the current presentation.
        relations: Relators as lists of tokens (e.g., ['a', 'b', 'a^-1']).
        backend: 'auto', 'julia', or 'python'.

    Returns:
        FundamentalGroup: A simplified presentation of the group.

    Use When:
        - A presentation is too large to interpret manually (e.g., after CW extraction).
        - Preparing for isomorphism testing or abelianization.
        - Reducing noise in π₁ calculations.

    Example:
        pi1_simple = simplify_presentation(['a', 'b'], [['a', 'b', 'a^-1', 'b^-1']])
    """
    gens = [normalize_word_token(g) for g in generators]
    rels = _normalize_relations([list(r) for r in relations])

    for _ in range(256):
        prev = (tuple(gens), tuple(tuple(r) for r in rels))

        gens, rels, killed = _kill_singleton_generators(gens, rels)
        if killed:
            rels = _normalize_relations(rels)
            continue

        move = _find_substitution_move(gens, rels)
        if move is not None:
            target, defining_rel_idx, rhs = move
            substituted: List[List[str]] = []
            for i, rel in enumerate(rels):
                if i == defining_rel_idx:
                    continue
                substituted.append(_substitute_generator_word(rel, target, rhs))
            gens = [g for g in gens if g != target]
            rels = _normalize_relations(substituted)
            continue

        curr = (tuple(gens), tuple(tuple(r) for r in rels))
        if curr == prev:
            break

    return FundamentalGroup(generators=gens, relations=rels)


def infer_standard_group_descriptor(pi1: FundamentalGroup, backend: str = "auto") -> str | None:
    """Infer a conservative descriptor among {"1", "Z", "Z_n"} when provable.

    What is Being Computed?:
        Attempts to identify the abstract isomorphism class of a group from its 
        presentation, focusing on trivial groups, free groups of rank 1, and cyclic groups.

    Algorithm:
        1. Simplify the presentation using Tietze moves.
        2. If 0 generators remain, return "1" (trivial).
        3. If 1 generator remains, check relator exponents to identify Z or Z_n.
        4. If multiple generators remain, compute the Smith Normal Form of the 
           abelianization to check if it's a direct product of cyclic groups.
        5. Return a string like "Z x Z_2" or None if inconclusive.

    Preserved Invariants:
        - Isomorphism class of the group.

    Args:
        pi1: The fundamental group presentation to analyze.
        backend: 'auto', 'julia', or 'python'.

    Returns:
        str | None: A standard group descriptor or None if the group is not easily classifiable.

    Use When:
        - Classifying low-dimensional manifolds via their π₁.
        - Automating topological comparisons.
        - Providing human-readable summaries of fundamental groups.

    Example:
        desc = infer_standard_group_descriptor(pi1)  # "Z x Z" for a torus
    """
    simplified = simplify_presentation(
        list(pi1.generators), [list(rel) for rel in pi1.relations], backend=backend
    )
    gens = list(simplified.generators)
    rels = [list(rel) for rel in simplified.relations]

    if not gens:
        return "1"
    if len(gens) != 1:
        # Check abelianization if it looks like a product of Zs or cyclic groups
        pair_comm = {
            tuple(sorted((gens[i], gens[j]))): _canonicalize_cyclic_word(
                [gens[i], gens[j], f"{gens[i]}^-1", f"{gens[j]}^-1"]
            )
            for i in range(len(gens))
            for j in range(i + 1, len(gens))
        }

        comm_seen: set[tuple[str, str]] = set()
        abelian_rows: list[list[int]] = []
        g_to_idx = {g: i for i, g in enumerate(gens)}

        for rel in rels:
            key_hit = None
            for pair, comm in pair_comm.items():
                if rel == comm:
                    key_hit = pair
                    break
            if key_hit is not None:
                comm_seen.add(key_hit)
                continue

            row = [0] * len(gens)
            has_term = False
            for tok in rel:
                base = _token_base(tok)
                idx = g_to_idx.get(base)
                if idx is None:
                    return None
                row[idx] += -1 if tok.endswith("^-1") else 1
                has_term = True
            if has_term and any(v != 0 for v in row):
                abelian_rows.append(row)

        if set(pair_comm.keys()) != comm_seen:
            return None

        if not abelian_rows:
            return " x ".join(["Z"] * len(gens))

        # --- Surgical SNF Pre-reduction ---
        # 1. First find rank over Z/pZ for large p to identify free rank
        # 2. Reduce the matrix by peeling off obvious pivots to speed up SymPy
        M_np = np.array(abelian_rows, dtype=object)
        m, n = M_np.shape
        
        # Simple structural reduction: peel rows with single +/-1
        diag = []
        keep_rows = [True] * m
        keep_cols = [True] * n
        
        for r in range(m):
            row_nz = np.where(M_np[r, :] != 0)[0]
            if len(row_nz) == 1:
                col = row_nz[0]
                if abs(M_np[r, col]) == 1 and keep_cols[col]:
                    keep_rows[r] = False
                    keep_cols[col] = False
                    # This corresponds to a Z/1Z factor (trivial)
        
        core_rows = [r for r, keep in enumerate(keep_rows) if keep]
        core_cols = [c for c, keep in enumerate(keep_cols) if keep]
        
        if not core_cols:
            factors = ["Z"] * (n - (m - len(core_rows)))
            factors = [f for f in factors if f != "1"]
            return " x ".join(factors) if factors else "1"

        M_core = sp.Matrix(M_np[core_rows, :][:, core_cols])
        S = sp.matrices.normalforms.smith_normal_form(M_core, domain=sp.ZZ)
        
        diag_len = min(S.rows, S.cols)
        diag = [abs(int(S[i, i])) for i in range(diag_len) if int(S[i, i]) != 0]
        
        # Free rank: Total generators - (peeled pivots + core pivots)
        peeled_count = m - len(core_rows)
        free_rank = max(0, n - (peeled_count + len(diag)))
        
        factors = ["Z"] * free_rank + [f"Z_{d}" for d in diag if d > 1]
        if not factors:
            return "1"
        return " x ".join(factors)

    g = gens[0]
    if not rels:
        return "Z"

    exponents: List[int] = []
    for rel in rels:
        exp_sum = 0
        for tok in rel:
            base = _token_base(tok)
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
    return "1" if n == 1 else f"Z_{n}"


class GroupPresentation(BaseModel):
    """Structured group descriptor and classification result.

    Overview:
        GroupPresentation provides a high-level classification of a finitely presented 
        group (e.g., distinguishing between free groups, cyclic groups, or direct 
        products). It translates raw generator/relation data into a human-readable 
        and algebraically meaningful descriptor.

    Key Concepts:
        - **Kind**: The primary classification (e.g., 'Z', 'Z_n', 'Product').
        - **Factors**: Individual group components in a direct product decomposition.
        - **Normalization**: Standardizing the descriptor for consistent comparisons.

    Common Workflows:
        1. **Classification** → Returned by infer_standard_group_descriptor().
        2. **String Representation** → Use normalized() to get a standard group name.

    Attributes:
        kind (str): The type of group (e.g., "1", "Z", "Z_n", "Product").
        factors (List[str]): List of factor descriptors for product groups.
    """

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


def _path_between_tree(u: int, v: int, parent: Dict[int, int], depth: Dict[int, int]) -> List[int]:
    """Return the vertex path between u and v in a rooted spanning forest using O(depth) LCA."""
    path_u: List[int] = []
    path_v: List[int] = []
    
    curr_u, curr_v = u, v
    
    # Bring both nodes to the same depth
    while depth[curr_u] > depth[curr_v]:
        path_u.append(curr_u)
        curr_u = parent[curr_u]
    while depth[curr_v] > depth[curr_u]:
        path_v.append(curr_v)
        curr_v = parent[curr_v]
        
    # Move up until LCA is found
    while curr_u != curr_v:
        path_u.append(curr_u)
        path_v.append(curr_v)
        curr_u = parent[curr_u]
        curr_v = parent[curr_v]
        
    path_u.append(curr_u) # Add the common ancestor
    return path_u + list(reversed(path_v))


def _reconstruct_cycle_from_edges(
    edge_indices: List[int], 
    d1: csr_matrix, 
    raw_gen_map: Dict[int, str]
) -> List[str]:
    """Reconstruct a directed cycle from an unordered set of boundary edges.
    
    Handles non-simple boundaries (e.g., figure-eight) using a robust
    neighbor-traversal matching used edges.
    """
    if not edge_indices:
        return []
    
    local_adj = collections.defaultdict(list)
    
    # Extract endpoint and orientation info from d1 for each edge in the face
    for e in edge_indices:
        col = d1.getcol(e).tocoo()
        verts = col.row
        vals = col.data
        if len(verts) == 2:
            v_idx = verts[np.where(vals == 1)[0][0]] if 1 in vals else verts[1]
            u_idx = verts[np.where(vals == -1)[0][0]] if -1 in vals else verts[0]
            
            # (neighbor, edge_idx, orientation)
            local_adj[u_idx].append((v_idx, e, 1))
            local_adj[v_idx].append((u_idx, e, -1))
        elif len(verts) == 1:
            v = verts[0]
            local_adj[v].append((v, e, 1))
        else:
            # Zero-boundary loop!
            v = 0
            local_adj[v].append((v, e, 1))

    if not local_adj:
        return []

    # Simple robust traversal matching available edge instances
    # For CW boundaries, this effectively traverses the Eulerian circuit
    curr_v = next(iter(local_adj.keys()))
    path = []
    # Track which *instances* of edges in edge_indices have been used
    used_indices = set()
    
    for _ in range(len(edge_indices)):
        found = False
        if curr_v not in local_adj:
            break
            
        for next_v, e_idx, orient in local_adj[curr_v]:
            # Find an unused occurrence of this edge index in the face boundary
            match_idx = -1
            for i, ei in enumerate(edge_indices):
                if ei == e_idx and i not in used_indices:
                    match_idx = i
                    break
            
            if match_idx != -1:
                used_indices.add(match_idx)
                gen = raw_gen_map.get(e_idx)
                if gen:
                    path.append(gen if orient == 1 else f"{gen}^-1")
                curr_v = next_v
                found = True
                break
        
        if not found:
            # Try jumping to another component if disconnected (shouldn't happen for a single cell)
            remaining = set(range(len(edge_indices))) - used_indices
            if remaining:
                # A single 2-cell boundary MUST be connected.
                pass
            break
            
    return path


def _pi1_raw_data_python(cw: CWComplex, backend: str = "auto"):
    """Build raw pi1 data from CW boundary maps with optimized LCA and cycle reconstruction."""
    d1 = cw.attaching_maps.get(1)
    d2 = cw.attaching_maps.get(2)

    if d1 is None:
        return {}, [], [], {"generator_mode": "raw", "backend_used": "python", "orientation_character": {}}

    n_vertices, n_edges = d1.shape
    if n_edges == 0:
        return {}, [], [], {"generator_mode": "raw", "backend_used": "python", "orientation_character": {}}

    # Normalize backend choice
    backend = str(backend).lower().strip()
    use_julia = (backend == "julia") or (backend == "auto" and julia_engine.available)

    # --- Performance Path: Julia ---
    if use_julia:
        try:
            d1_coo = d1.tocoo()
            if d2 is not None and d2.nnz > 0:
                d2_coo = d2.tocoo()
                d2_rows, d2_cols, d2_vals = d2_coo.row, d2_coo.col, d2_coo.data
                n_faces = d2.shape[1]
            else:
                d2_rows, d2_cols, d2_vals = np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)
                n_faces = 0
            
            res = julia_engine.compute_pi1_raw_data(
                d1_coo.row, d1_coo.col, d1_coo.data,
                n_vertices, n_edges,
                d2_rows, d2_cols, d2_vals,
                n_faces
            )
            
            # Reconstruct traces bypassing Pydantic validation for speed
            py_traces = []
            for tr in res["traces"]:
                py_traces.append(Pi1GeneratorTrace.model_construct(
                    generator=tr["generator"],
                    edge_index=tr["edge_index"],
                    component_root=tr["component_root"],
                    vertex_path=tr["vertex_path"],
                    directed_edge_path=[tuple(e) for e in tr["directed_edge_path"]],
                    undirected_edge_path=[tuple(sorted(e)) for e in tr["undirected_edge_path"]]
                ))
            
            w1 = res.get("orientation_character", {g: 1 for g in res["generators"].values()})
            return res["generators"], res["relations"], py_traces, {"generator_mode": "raw", "backend_used": "julia", "orientation_character": w1}
        except Exception as e:
            if backend == "julia":
                raise e
            import warnings
            warnings.warn(f"Julia pi1_raw_data failed ({e!r}). Falling back to optimized Python.")

    visited = [False] * n_vertices
    tree_edges = set()
    parent: Dict[int, int] = {}
    depth: Dict[int, int] = {}
    component_root: Dict[int, int] = {}

    # Build adjacency
    d1_csc = d1.tocsc()
    edge_list = []
    adj = collections.defaultdict(list)
    for e in range(n_edges):
        col_r = d1_csc.indices[d1_csc.indptr[e]:d1_csc.indptr[e+1]]
        col_d = d1_csc.data[d1_csc.indptr[e]:d1_csc.indptr[e+1]]
        if len(col_r) == 2:
            # Check signs for source/target
            try:
                u_idx = col_r[np.where(col_d == -1)[0][0]]
                v_idx = col_r[np.where(col_d == 1)[0][0]]
            except IndexError:
                # Fallback for unsigned edges
                u_idx, v_idx = col_r[0], col_r[1]
            edge_list.append((u_idx, v_idx))
            adj[u_idx].append((v_idx, e, 1))
            adj[v_idx].append((u_idx, e, -1))
        elif len(col_r) == 1:
            v = col_r[0]
            edge_list.append((v, v))
            adj[v].append((v, e, 1))
        else:
            # Zero boundary (e.g. S1 loop with 1 vertex)
            if n_vertices > 0:
                v = 0
                edge_list.append((v, v))
                adj[v].append((v, e, 1))
            else:
                edge_list.append(None)
    
    # Python BFS with depth tracking
    for start in range(n_vertices):
        if visited[start]:
            continue
        queue = collections.deque([(start, 0)])
        visited[start] = True
        parent[start] = -1
        depth[start] = 0
        component_root[start] = start
        while queue:
            curr, d = queue.popleft()
            for neighbor, edge_idx, _ in adj[curr]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    if curr != neighbor:
                        tree_edges.add(edge_idx)
                        parent[neighbor] = curr
                        depth[neighbor] = d + 1
                        component_root[neighbor] = start
                        queue.append((neighbor, d + 1))

    raw_gen_map = {i: f"g_{i}" for i in range(n_edges) if i not in tree_edges and edge_list[i] is not None}
    
    relations = []
    if d2 is not None and d2.nnz > 0:
        d2_csc = d2.tocsc()
        for f in range(d2.shape[1]):
            col_r = d2_csc.indices[d2_csc.indptr[f]:d2_csc.indptr[f+1]]
            col_d = d2_csc.data[d2_csc.indptr[f]:d2_csc.indptr[f+1]]
            
            # Flatten into list of edge indices by multiplicity
            edge_indices = []
            for e, val in zip(col_r, col_d):
                for _ in range(abs(int(val))):
                    edge_indices.append(e)
            
            # Heuristic Cycle Reconstruction for Non-Abelian Relators
            relation = _reconstruct_cycle_from_edges(edge_indices, d1, raw_gen_map)
            if relation:
                relations.append(relation)

    traces: list[Pi1GeneratorTrace] = []
    if julia_engine.available:
        try:
            coo = d1.tocoo()
            raw_trace_dicts = julia_engine.compute_pi1_trace_candidates(
                np.asarray(coo.row, dtype=np.int64),
                np.asarray(coo.col, dtype=np.int64),
                np.asarray(coo.data, dtype=np.int64),
                n_vertices=n_vertices, n_edges=n_edges
            )
            for tr in raw_trace_dicts:
                traces.append(Pi1GeneratorTrace.model_construct(
                    generator=str(tr["generator"]),
                    edge_index=int(tr["edge_index"]),
                    component_root=int(tr["component_root"]),
                    vertex_path=[int(x) for x in tr["vertex_path"]],
                    directed_edge_path=[(int(a), int(b)) for a, b in tr["directed_edge_path"]],
                    undirected_edge_path=[(int(a), int(b)) for a, b in tr["undirected_edge_path"]]
                ))
            return raw_gen_map, relations, traces, {"generator_mode": "raw", "backend_used": "julia"}
        except Exception:
            pass

    for edge_idx, gen_name in raw_gen_map.items():
        u, v = edge_list[edge_idx]
        if u == v:
            path = [u]
            directed = [(u, v)]
        else:
            path_v = _path_between_tree(v, u, parent, depth)
            directed = [(path_v[i], path_v[i+1]) for i in range(len(path_v)-1)]
            path = path_v
        
        traces.append(Pi1GeneratorTrace.model_construct(
            generator=gen_name,
            edge_index=edge_idx,
            component_root=int(component_root.get(u, u)),
            vertex_path=path,
            directed_edge_path=directed,
            undirected_edge_path=[tuple(sorted(e)) for e in directed]
        ))

    w1 = {gen_name: 1 for gen_name in raw_gen_map.values()}
    return raw_gen_map, relations, traces, {"generator_mode": "raw", "backend_used": "python", "orientation_character": w1}


def induced_pi1_map(
    vertex_map: Dict[int, int],
    source_pi1: Pi1PresentationWithTraces,
    target_pi1: Pi1PresentationWithTraces
) -> Dict[str, List[str]]:
    """Compute the induced map f*: pi1(M) -> pi1(X) from a simplicial map f.

    Args:
        vertex_map: Mapping from source vertex IDs to target vertex IDs.
        source_pi1: pi1 presentation of the source complex (M).
        target_pi1: pi1 presentation of the target complex (X).

    Returns:
        A dictionary mapping source generator names to words in the target generators.
    """
    # 1. Build lookup for target generators: (u, v) -> word token
    edge_to_gen = {}
    for tr in target_pi1.traces:
        # A generator trace corresponds to a single non-tree edge loop
        # The first edge in the directed_edge_path is the "defining" edge
        if tr.directed_edge_path:
            u, v = tr.directed_edge_path[0]
            edge_to_gen[(u, v)] = tr.generator
            edge_to_gen[(v, u)] = f"{tr.generator}^-1"

    # 2. Map each source generator
    mapping = {}
    for s_tr in source_pi1.traces:
        # Map vertex path through f
        target_v_path = [vertex_map.get(v) for v in s_tr.vertex_path]
        if None in target_v_path:
            # Incomplete map
            continue
            
        # Convert vertex path to edge word in the target
        word = []
        for i in range(len(target_v_path) - 1):
            u, v = target_v_path[i], target_v_path[i+1]
            if u == v:
                continue # Collapse
            
            # Check if this edge is a generator in target_pi1
            if (u, v) in edge_to_gen:
                word.append(edge_to_gen[(u, v)])
            else:
                # Edge must be in the spanning tree of the target
                # Tree edges contribute identity to the pi1 presentation
                pass
        
        mapping[s_tr.generator] = _free_reduce(word)
        
    return mapping


def extract_pi_1_with_traces(
    cw: CWComplex,
    simplify: bool = True,
    generator_mode: str = "optimized",
    mode: str | None = None,
    backend: str = "auto",
) -> Pi1PresentationWithTraces:
    """Extract π₁ presentation from CW boundary maps, including generator traces (spatial origins).

    What is Being Computed?:
        Extracts π₁(X) = ⟨generators | relations⟩ from the 2-skeleton of a CW complex.
        For a 2-complex, generators correspond to 1-cells and relations to 2-cell attaching maps.
        Also returns Pi1GeneratorTrace objects showing which 1-cells correspond to each generator.

    Algorithm:
        1. From CW boundary matrices d₁ and d₂, read 1-cell labels and 2-cell attachments
        2. Build a "word" for each 2-cell boundary describing which 1-cells appear and with what sign
        3. These words become relators; the 1-cells become generators
        4. Optionally apply Tietze moves (elementary simplifications) to reduce presentation
        5. Compute orientation character w₁ (±1 per generator) if manifold is non-orientable

    Preserved Invariants:
        - π₁(X) is a homotopy invariant (homotopy equivalent complexes have isomorphic π₁)
        - Presentations may differ by Tietze moves, but represent the same group
        - Torsion structure detected: e.g., ℤ/2ℤ generators appear as relations of the form g²

    Args:
        cw: CWComplex to analyze.
        simplify: If True (default), apply Tietze simplification to reduce generator count.
        generator_mode: 'raw' (before simplification) or 'optimized' (after simplification).
        mode: Alias for generator_mode (for backwards compatibility).
        backend: 'auto', 'julia', or 'python' for boundary matrix computation.

    Returns:
        Pi1PresentationWithTraces with fields:
            - generators: List of generator symbols
            - relations: List of relations (words)
            - traces: List of Pi1GeneratorTrace objects linking generators to 1-cells
            - mode_used, generator_mode: Records which mode was used
            - backend_used: Backend name used
            - raw_generator_count, optimized_generator_count: Counts before/after simplification

    Use When:
        - Need spatial location/tracing of generators (e.g., which edges correspond to which generators)
        - Computing homomorphisms from π₁ to other groups
        - Studying group structure in detail
        - Debugging presentations

    Example:
        traces = extract_pi_1_with_traces(cw, simplify=True)
        for trace in traces.traces:
            print(f"Generator {trace.generator} comes from 1-cell {trace.cell_index}")
        π1 = FundamentalGroup(generators=traces.generators, relations=traces.relations)
    """
    actual_mode = _normalize_pi1_mode(generator_mode, mode)
    raw_generators, relations, traces, meta = _pi1_raw_data_python(cw, backend=backend)
    
    if not raw_generators:
        return Pi1PresentationWithTraces(
            generators=[], relations=[], traces=[],
            mode_used=actual_mode, generator_mode=actual_mode,
            backend_used=meta.get("backend_used", "python"),
            orientation_character={}
        )

    if actual_mode == "raw" or not simplify:
        return Pi1PresentationWithTraces(
            generators=list(raw_generators.values()),
            relations=relations,
            orientation_character=meta.get("orientation_character", {}),
            traces=traces,
            mode_used=actual_mode,
            generator_mode="raw" if not simplify else actual_mode,
            backend_used=meta.get("backend_used", "python"),
            raw_generator_count=len(raw_generators),
            optimized_generator_count=len(raw_generators) if not simplify else 0,
            reduced_generator_count=len(raw_generators),
        )

    out_pi = simplify_presentation(list(raw_generators.values()), relations)
    keep = set(out_pi.generators)
    
    # Filter orientation character for kept generators
    raw_w1 = meta.get("orientation_character", {})
    final_w1 = {g: raw_w1.get(g, 1) for g in out_pi.generators}
    
    return Pi1PresentationWithTraces(
        generators=list(out_pi.generators),
        relations=out_pi.relations,
        orientation_character=final_w1,
        traces=[tr for tr in traces if tr.generator in keep],
        mode_used=actual_mode,
        generator_mode="optimized",
        backend_used=meta.get("backend_used", "python"),
        raw_generator_count=len(raw_generators),
        optimized_generator_count=len(out_pi.generators),
        reduced_generator_count=len(out_pi.generators),
    )


def extract_pi_1(
    cw: CWComplex,
    simplify: bool = True,
    generator_mode: str = "optimized",
    mode: str | None = None,
    backend: str = "auto",
) -> FundamentalGroup:
    """Extract π₁ presentation from CW boundary maps as a FundamentalGroup object.

    What is Being Computed?:
        Extracts the fundamental group π₁(X) = ⟨generators | relations⟩ from the 2-skeleton
        of a CW complex in the form of a FundamentalGroup (generators, relations, w₁ character).

    Algorithm:
        1. Call extract_pi_1_with_traces() to get raw presentation and traces
        2. Extract generators, relations, and orientation character from traces
        3. Return FundamentalGroup object (discarding spatial trace information)

    Preserved Invariants:
        - π₁ is a homotopy invariant: homotopy equivalent spaces have isomorphic fundamental groups
        - Presentations may differ, but represent the same abstract group structure
        - Torsion in π₁ (e.g., generators of finite order) is preserved

    Args:
        cw: CWComplex to analyze.
        simplify: If True, apply Tietze simplification to reduce presentation size.
        generator_mode: 'raw' or 'optimized' (controls simplifcation).
        mode: Alias for generator_mode.
        backend: 'auto', 'julia', or 'python'.

    Returns:
        FundamentalGroup: The presentation π₁(X) = ⟨generators | relations⟩ with orientation
                         character (w₁) recording non-orientability info.

    Use When:
        - Need just the group structure, not spatial traces
        - Computing derived invariants (abelianization, homology via Hurewicz)
        - Checking if two spaces are non-homeomorphic via π₁
        - Studying fundamental groups in algebraic topology

    Example:
        π1 = extract_pi_1(cw, simplify=True)
        print(π1)  # Prints: < a, b | aba^-1b^-1 > for a torus
        ab = π1.abelianization()  # ℤ ⊕ ℤ for torus
    """
    traces = extract_pi_1_with_traces(
        cw, simplify=simplify, generator_mode=generator_mode, mode=mode, backend=backend
    )
    return FundamentalGroup(
        generators=traces.generators, 
        relations=traces.relations,
        orientation_character=traces.orientation_character
    )
