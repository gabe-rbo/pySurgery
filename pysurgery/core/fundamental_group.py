import collections
from typing import Dict, List
from math import gcd
import numpy as np
import sympy as sp
from pydantic import BaseModel, ConfigDict, Field
from .complexes import CWComplex
from .exact_algebra import normalize_word_token
from .generator_models import Pi1GeneratorTrace, Pi1PresentationWithTraces
from ..bridge.julia_bridge import julia_engine


class FundamentalGroup(BaseModel):
    """Representation of the Fundamental Group pi_1(X) of a CW Complex.

    Uses the Edge-Path Group algorithm.

    Attributes:
        generators: A list of generator names as strings.
        relations: A list of relators, where each relator is a list of generator tokens.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    generators: List[str]
    relations: List[List[str]]

    def __str__(self):
        """Format the presentation as `<g1, g2 | r1, r2, ...>`.

        Returns:
            A string representation of the group presentation.
        """
        gens = ", ".join(self.generators)
        rels = ", ".join(["".join(r) for r in self.relations])
        return f"< {gens} | {rels} >"


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
    """Canonicalize a cyclic word up to rotation and inversion.

    This normalization gives deterministic relator representatives for deduping.

    Args:
        word: A list of generator tokens representing a cyclic word.

    Returns:
        The canonicalized word.
    """
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
    """Find a deterministic one-occurrence substitution candidate.

    Args:
        generators: Current list of generator names.
        relations: Current list of relators.

    Returns:
        A tuple `(generator, defining_relator_index, replacement_word)` if a
        candidate is found, otherwise None.
    """
    for g in generators:
        for rel_idx, rel in enumerate(relations):
            occ = [i for i, tok in enumerate(rel) if _token_base(tok) == g]
            if len(occ) == 1:
                idx = occ[0]
                rhs = _solve_generator_from_relator(rel, idx)
                return g, rel_idx, rhs
    return None


def simplify_presentation(
    generators: List[str], relations: List[List[str]]
) -> FundamentalGroup:
    """Simplify a finitely presented group using a deterministic Tietze-lite loop.

    Args:
        generators: Generator names in the current presentation.
        relations: Relators as token lists (`g_i`, `g_i^-1`).

    Returns:
        A reduced presentation with canonicalized relators.
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


def infer_standard_group_descriptor(pi1: FundamentalGroup) -> str | None:
    """Infer a conservative descriptor among {"1", "Z", "Z_n"} when provable.

    Args:
        pi1: The fundamental group presentation to analyze.

    Returns:
        A string descriptor if a standard form can be inferred, otherwise None.
    """
    simplified = simplify_presentation(
        list(pi1.generators), [list(rel) for rel in pi1.relations]
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

        M = sp.Matrix(abelian_rows)
        S = sp.matrices.normalforms.smith_normal_form(M, domain=sp.ZZ)
        diag_len = min(S.rows, S.cols)
        diag = [abs(int(S[i, i])) for i in range(diag_len) if int(S[i, i]) != 0]
        free_rank = max(0, len(gens) - len(diag))
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
    """Structured group descriptor.

    Attributes:
        kind: The type of group.
        factors: Factor descriptors for product groups.
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


def _path_between_tree(u: int, v: int, parent: Dict[int, int]) -> List[int]:
    """Return the vertex path between u and v in a rooted spanning forest."""
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


def _pi1_raw_data_python(cw: CWComplex):
    """Build raw pi1 data from CW boundary maps."""
    d1 = cw.attaching_maps.get(1)
    d2 = cw.attaching_maps.get(2)

    if d1 is None:
        return {}, [], [], {"generator_mode": "raw", "backend_used": "python"}

    n_vertices, n_edges = d1.shape
    if n_edges == 0:
        return {}, [], [], {"generator_mode": "raw", "backend_used": "python"}

    # --- Performance Path: Julia ---
    if julia_engine.available:
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
            
            # Reconstruct traces as pydantic models
            py_traces = []
            for tr in res["traces"]:
                py_traces.append(Pi1GeneratorTrace(
                    generator=tr["generator"],
                    edge_index=tr["edge_index"],
                    component_root=tr["component_root"],
                    vertex_path=tr["vertex_path"],
                    directed_edge_path=tr["directed_edge_path"],
                    undirected_edge_path=[tuple(sorted(e)) for e in tr["undirected_edge_path"]]
                ))
            
            return res["generators"], res["relations"], py_traces, {"generator_mode": "raw", "backend_used": "julia"}
        except Exception as e:
            import warnings
            warnings.warn(f"Julia pi1_raw_data failed ({e!r}). Falling back to optimized Python.")

    visited = [False] * n_vertices
    tree_edges = set()
    parent: Dict[int, int] = {}
    component_root: Dict[int, int] = {}

    # Build adjacency
    d1_csc = d1.tocsc()
    edge_list = []
    adj = collections.defaultdict(list)
    for e in range(n_edges):
        col_r = d1_csc.indices[d1_csc.indptr[e]:d1_csc.indptr[e+1]]
        col_d = d1_csc.data[d1_csc.indptr[e]:d1_csc.indptr[e+1]]
        if len(col_r) == 2:
            u, v = col_r[0], col_r[1]
            edge_list.append((u, v))
            adj[u].append((v, e, col_d))
            adj[v].append((u, e, col_d))
        elif len(col_r) == 0:
            # Self-loop! Boundary is v - v = 0.
            # In a CW complex, if d1 column is zero, it's a loop.
            # We must associate it with SOME vertex. If n_vertices > 0, pick 0.
            v = 0
            edge_list.append((v, v))
            adj[v].append((v, e, np.array([0])))
        else:
            edge_list.append(None)
    
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
                    # Self-loops (u==v) are NEVER in a spanning tree
                    if curr != neighbor:
                        tree_edges.add(edge_idx)
                        parent[neighbor] = curr
                        component_root[neighbor] = start
                        queue.append(neighbor)

    raw_gen_map = {i: f"g_{i}" for i in range(n_edges) if i not in tree_edges and edge_list[i] is not None}
    
    relations = []
    if d2 is not None and d2.nnz > 0:
        d2_csc = d2.tocsc()
        for f in range(d2.shape[1]):
            col_r = d2_csc.indices[d2_csc.indptr[f]:d2_csc.indptr[f+1]]
            col_d = d2_csc.data[d2_csc.indptr[f]:d2_csc.indptr[f+1]]
            
            # Simple relator tracing
            relation = []
            for val, e in zip(col_d, col_r):
                sign = 1 if int(val) > 0 else -1
                for _ in range(abs(int(val))):
                    if e in raw_gen_map:
                        gen = raw_gen_map[e]
                        relation.append(gen if sign == 1 else f"{gen}^-1")
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
                traces.append(Pi1GeneratorTrace(
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
            path_v = _path_between_tree(v, u, parent)
            directed = [(u, v)] + [(path_v[i], path_v[i+1]) for i in range(len(path_v)-1)]
            path = [u] + [b for _, b in directed]
        
        traces.append(Pi1GeneratorTrace(
            generator=gen_name,
            edge_index=edge_idx,
            component_root=int(component_root.get(u, u)),
            vertex_path=path,
            directed_edge_path=directed,
            undirected_edge_path=[tuple(sorted(e)) for e in directed]
        ))

    return raw_gen_map, relations, traces, {"generator_mode": "raw", "backend_used": "python"}


def extract_pi_1_with_traces(
    cw: CWComplex,
    simplify: bool = True,
    generator_mode: str = "optimized",
    mode: str | None = None,
) -> Pi1PresentationWithTraces:
    """Return pi_1 presentation with generator traces."""
    actual_mode = _normalize_pi1_mode(generator_mode, mode)
    raw_generators, relations, traces, meta = _pi1_raw_data_python(cw)
    
    if not raw_generators:
        return Pi1PresentationWithTraces(
            generators=[], relations=[], traces=[],
            mode_used=actual_mode, generator_mode=actual_mode,
            backend_used=meta.get("backend_used", "python")
        )

    if actual_mode == "raw" or not simplify:
        return Pi1PresentationWithTraces(
            generators=list(raw_generators.values()),
            relations=relations,
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
    return Pi1PresentationWithTraces(
        generators=list(out_pi.generators),
        relations=out_pi.relations,
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
) -> FundamentalGroup:
    """Compute a pi1 presentation from CW boundary maps."""
    traces = extract_pi_1_with_traces(cw, simplify=simplify, generator_mode=generator_mode, mode=mode)
    return FundamentalGroup(generators=traces.generators, relations=traces.relations)
