from typing import Dict, List, Tuple
from math import gcd
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from .complexes import CWComplex
from .exact_algebra import normalize_word_token
from .generator_models import Pi1GeneratorTrace, Pi1PresentationWithTraces
from ..bridge.julia_bridge import julia_engine

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


def _normalize_pi1_mode(generator_mode: str = "optimized", mode: str | None = None) -> str:
    chosen = mode if mode is not None else generator_mode
    chosen = str(chosen).strip().lower()
    if chosen not in {"raw", "optimized"}:
        raise ValueError("generator_mode must be 'raw' or 'optimized'")
    return chosen


def _token_base(tok: str) -> str:
    nt = normalize_word_token(tok)
    return nt[:-3] if nt.endswith("^-1") else nt


def _inverse_word(word: List[str]) -> List[str]:
    return [_inverse_word_token(t) for t in reversed(word)]


def _normalize_relations(relations: List[List[str]]) -> List[List[str]]:
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


def _kill_singleton_generators(generators: List[str], relations: List[List[str]]) -> tuple[List[str], List[List[str]], set[str]]:
    kill = {_token_base(r[0]) for r in relations if len(r) == 1}
    if not kill:
        return generators, relations, set()
    new_gens = [g for g in generators if g not in kill]
    new_rels = [[t for t in r if _token_base(t) not in kill] for r in relations]
    return new_gens, new_rels, kill


def _solve_generator_from_relator(relator: List[str], idx: int) -> List[str]:
    tok = normalize_word_token(relator[idx])
    rest = relator[idx + 1 :] + relator[:idx]
    # relator = 1 and tok appears once, so we can isolate tok and substitute globally.
    return _free_reduce(rest if tok.endswith("^-1") else _inverse_word(rest))


def _substitute_generator_word(relator: List[str], target: str, rhs: List[str]) -> List[str]:
    out: List[str] = []
    inv_rhs = _inverse_word(rhs)
    for tok in relator:
        nt = normalize_word_token(tok)
        if _token_base(nt) != target:
            out.append(nt)
            continue
        out.extend(inv_rhs if nt.endswith("^-1") else rhs)
    return out


def _find_substitution_move(generators: List[str], relations: List[List[str]]) -> tuple[str, int, List[str]] | None:
    for g in generators:
        for rel_idx, rel in enumerate(relations):
            occ = [i for i, tok in enumerate(rel) if _token_base(tok) == g]
            if len(occ) == 1:
                idx = occ[0]
                rhs = _solve_generator_from_relator(rel, idx)
                return g, rel_idx, rhs
    return None


def simplify_presentation(generators: List[str], relations: List[List[str]]) -> FundamentalGroup:
    gens = [normalize_word_token(g) for g in generators]
    rels = _normalize_relations([list(r) for r in relations])

    # Deterministic Tietze-lite loop: singleton kills first, then single-occurrence substitution.
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


def _pi1_raw_data_python(cw: CWComplex):
    d1 = cw.attaching_maps.get(1)
    d2 = cw.attaching_maps.get(2)

    if d1 is None:
        return [], [], [], {"generator_mode": "raw", "backend_used": "python"}

    n_vertices = d1.shape[0]
    n_edges = d1.shape[1]
    if n_edges == 0:
        return [], [], [], {"generator_mode": "raw", "backend_used": "python"}

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

    raw_gen_map = {i: f"g_{i}" for i in range(n_edges) if i not in tree_edges}

    relations = []
    if d2 is not None and d2.nnz > 0:
        d2_csc = d2.tocsc()
        n_faces = d2.shape[1]

        for f in range(n_faces):
            col_start = d2_csc.indptr[f]
            col_end = d2_csc.indptr[f + 1]
            col_data = d2_csc.data[col_start:col_end]
            col_row = d2_csc.indices[col_start:col_end]

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
                    break

            if target_node != start_node:
                continue

            relation = []
            for e, d in path:
                if e in raw_gen_map:
                    gen_str = raw_gen_map[e]
                    relation.append(gen_str if d == 1 else f"{gen_str}^-1")

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
                n_vertices=n_vertices,
                n_edges=n_edges,
            )
            for tr in raw_trace_dicts:
                traces.append(
                    Pi1GeneratorTrace(
                        generator=str(tr["generator"]),
                        edge_index=int(tr["edge_index"]),
                        component_root=int(tr["component_root"]),
                        vertex_path=[int(x) for x in tr["vertex_path"]],
                        directed_edge_path=[(int(a), int(b)) for a, b in tr["directed_edge_path"]],
                        undirected_edge_path=[(int(a), int(b)) for a, b in tr["undirected_edge_path"]],
                    )
                )
            return raw_gen_map, relations, traces, {"generator_mode": "raw", "backend_used": "julia"}
        except Exception:
            pass

    for edge_idx, gen_name in raw_gen_map.items():
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

    return raw_gen_map, relations, traces, {"generator_mode": "raw", "backend_used": "python"}


def extract_pi_1_with_traces(
    cw: CWComplex,
    simplify: bool = True,
    generator_mode: str = "optimized",
    mode: str | None = None,
) -> Pi1PresentationWithTraces:
    """Return pi_1 presentation with generator traces as data-native edge/vertex paths.

    `generator_mode="raw"` preserves the full spanning-forest generator set.
    `generator_mode="optimized"` simplifies the presentation and filters traces to the surviving generators.
    """
    actual_mode = _normalize_pi1_mode(generator_mode, mode)
    raw_generators, relations, traces, meta = _pi1_raw_data_python(cw)
    if not raw_generators:
        return Pi1PresentationWithTraces(
            generators=[],
            relations=[],
            traces=[],
            mode_used=actual_mode,
            generator_mode=actual_mode,
            backend_used=meta.get("backend_used", "python"),
            raw_generator_count=0,
            optimized_generator_count=0,
            reduced_generator_count=0,
        )

    raw_pi = FundamentalGroup(generators=list(raw_generators.values()), relations=[list(r) for r in relations])
    if actual_mode == "raw" or not simplify:
        return Pi1PresentationWithTraces(
            generators=list(raw_pi.generators),
            relations=[list(r) for r in raw_pi.relations],
            traces=traces,
            mode_used=actual_mode,
            generator_mode="raw" if not simplify else actual_mode,
            backend_used=meta.get("backend_used", "python"),
            raw_generator_count=len(raw_pi.generators),
            optimized_generator_count=0 if not simplify else len(raw_pi.generators),
            reduced_generator_count=len(raw_pi.generators),
        )

    out_pi = simplify_presentation(list(raw_pi.generators), [list(r) for r in raw_pi.relations])
    keep = set(out_pi.generators)
    filtered_traces = [tr for tr in traces if tr.generator in keep]
    return Pi1PresentationWithTraces(
        generators=list(out_pi.generators),
        relations=[list(r) for r in out_pi.relations],
        traces=filtered_traces,
        mode_used=actual_mode,
        generator_mode="optimized",
        backend_used=meta.get("backend_used", "python"),
        raw_generator_count=len(raw_pi.generators),
        optimized_generator_count=len(out_pi.generators),
        reduced_generator_count=len(out_pi.generators),
    )

def extract_pi_1(
    cw: CWComplex,
    simplify: bool = True,
    generator_mode: str = "optimized",
    mode: str | None = None,
) -> FundamentalGroup:
    """
    Computes a presentation for the fundamental group pi_1(X) by constructing
    a maximal spanning tree in the 1-skeleton.

    Edges not in the tree become the generators.
    The boundary of the 2-cells (faces) dictate the relations.
    """
    traces = extract_pi_1_with_traces(cw, simplify=simplify, generator_mode=generator_mode, mode=mode)
    return FundamentalGroup(generators=list(traces.generators), relations=[list(r) for r in traces.relations])

