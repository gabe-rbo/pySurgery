"""Tests for shared manifold-reconstruction repair infrastructure.

Overview:
    Verifies ``is_single_cycle`` (including a cross-check against an independent,
    homology-based oracle built on ``SimplicialComplex.reduced_homology`` -- the same
    engine ``is_homology_manifold`` itself calls internally -- rather than re-deriving
    ``is_single_cycle``'s own graph-walk logic), ``intersect_local_stars``, and the
    ``moser_tardos_repair`` engine's control flow (targeted perturbation, convergence, and
    non-convergence raising ``ReconstructionRepairError``).
"""
import itertools
from collections import Counter

import numpy as np
import pytest

from pysurgery.core.exceptions import ReconstructionRepairError
from pysurgery.geometry.perturbation import (
    intersect_local_stars,
    is_single_cycle,
    is_single_path_or_cycle,
    moser_tardos_repair,
)
from pysurgery.topology.complexes import SimplicialComplex


# ---------------------------------------------------------------------------
# is_single_cycle
# ---------------------------------------------------------------------------

def test_is_single_cycle_hand_built_cases():
    """Verify is_single_cycle on small, unambiguous hand-built graphs."""
    triangle_ok, triangle_order = is_single_cycle([(0, 1), (1, 2), (2, 0)])
    assert triangle_ok and sorted(triangle_order) == [0, 1, 2]

    square_ok, _ = is_single_cycle([(0, 1), (1, 2), (2, 3), (3, 0)])
    assert square_ok

    assert is_single_cycle([(0, 1), (1, 2)]) == (False, None)  # path, degree-1 endpoints
    assert is_single_cycle([(0, 1), (2, 3)]) == (False, None)  # two disjoint edges
    # Two disjoint triangles: every node has degree 2, but not connected/spanning.
    two_triangles = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]
    assert is_single_cycle(two_triangles) == (False, None)
    # A cycle with a pendant edge: node 3 has degree 3 (branching).
    branching = [(0, 1), (1, 2), (2, 0), (0, 3)]
    assert is_single_cycle(branching) == (False, None)
    assert is_single_cycle([]) == (False, None)
    assert is_single_cycle([(0, 0)]) == (False, None)  # self-loop only


def _is_genuine_cycle_via_homology(nodes, edges) -> bool:
    """Independent oracle for "is (nodes, edges) a single simple cycle", built from
    SimplicialComplex.reduced_homology (SNF-based) rather than is_single_cycle's own graph
    walk -- degree-2-everywhere is checked directly (a plain Counter, not a graph walk), and
    connectivity + "exactly one cycle" is checked via the real homology engine (connected =>
    reduced H_0 vanishes; H_1 rank 1 with no torsion => exactly one independent cycle).
    A cycle-plus-pendant-tree graph has the same H_1 rank as a plain cycle, so the explicit
    degree check here is doing real, necessary work, not merely re-deriving is_single_cycle.
    """
    if not edges:
        return False
    degree = Counter()
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
    if any(d != 2 for d in degree.values()):
        return False
    simplices = [(n,) for n in nodes] + [list(e) for e in edges]
    sc = SimplicialComplex.from_simplices(simplices, close_under_faces=True)
    rh = sc.reduced_homology(backend="python")
    non_zero = {k: v for k, v in rh.items() if v[0] > 0 or v[1]}
    if len(non_zero) != 1:
        return False
    (degree_k, (rank, torsion)), = non_zero.items()
    return degree_k == 1 and rank == 1 and not torsion


def test_is_single_cycle_matches_independent_homology_oracle_on_random_graphs():
    """Cross-check: is_single_cycle's graph-walk result must agree with an independently
    implemented, homology-based oracle, for many random graphs -- including ones that are
    NOT restricted to already satisfy the degree<=2 condition, so it exercises cases (e.g. a
    cycle with a pendant branch) where "the graph has the right H_1 rank" alone would
    disagree with "is literally a single simple cycle" (which is exactly why the oracle
    checks degree explicitly, in addition to homology).
    """
    rng = np.random.default_rng(3)
    n_checked = 0
    for _ in range(60):
        n_candidate_nodes = int(rng.integers(3, 8))
        possible_edges = list(itertools.combinations(range(n_candidate_nodes), 2))
        rng.shuffle(possible_edges)
        n_edges = int(rng.integers(1, len(possible_edges) + 1))
        edges = possible_edges[:n_edges]
        nodes_used = sorted({n for e in edges for n in e})
        if len(nodes_used) < 3:
            continue  # a simple cycle needs at least 3 nodes; skip degenerate draws

        predicted_ok, _ = is_single_cycle(edges)
        actual_ok = _is_genuine_cycle_via_homology(nodes_used, edges)
        assert predicted_ok == actual_ok, f"mismatch on edges={edges}: predicted={predicted_ok}, oracle={actual_ok}"
        n_checked += 1
    assert n_checked > 20  # sanity: the random draws actually exercised enough cases


# ---------------------------------------------------------------------------
# is_single_path_or_cycle
# ---------------------------------------------------------------------------

def test_is_single_path_or_cycle_hand_built_cases():
    """Verify is_single_path_or_cycle on small, unambiguous hand-built graphs -- both shapes
    it must accept (cycle, open path) and the defects it must still reject (disconnected,
    branching)."""
    triangle_ok, triangle_order = is_single_path_or_cycle([(0, 1), (1, 2), (2, 0)])
    assert triangle_ok and sorted(triangle_order) == [0, 1, 2]

    path_ok, path_order = is_single_path_or_cycle([(0, 1), (1, 2)])
    assert path_ok and sorted(path_order) == [0, 1, 2]  # open path, unlike is_single_cycle

    single_edge_ok, _ = is_single_path_or_cycle([(0, 1)])
    assert single_edge_ok  # a length-1 path is still a valid (trivial) boundary link

    assert is_single_path_or_cycle([(0, 1), (2, 3)]) == (False, None)  # disconnected
    # Two disjoint triangles: every node degree 2, but not connected/spanning.
    two_triangles = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]
    assert is_single_path_or_cycle(two_triangles) == (False, None)
    # A cycle with a pendant edge: node 3 has degree 3 (branching) -- still a defect.
    branching = [(0, 1), (1, 2), (2, 0), (0, 3)]
    assert is_single_path_or_cycle(branching) == (False, None)
    # A connected path plus a disjoint separate edge: degree counts alone could look like
    # "one path" (exactly two degree-1 nodes among the endpoints of each piece happens not to
    # apply here, but this shape -- a cycle plus a disjoint dangling path -- is the case the
    # oracle below specifically exercises); directly: two disjoint 2-node paths is rejected.
    two_disjoint_paths = [(0, 1), (2, 3)]
    assert is_single_path_or_cycle(two_disjoint_paths) == (False, None)
    assert is_single_path_or_cycle([]) == (False, None)
    assert is_single_path_or_cycle([(0, 0)]) == (False, None)  # self-loop only


def _is_genuine_path_or_cycle_via_homology(nodes, edges) -> bool:
    """Independent oracle for "is (nodes, edges) a single simple path or cycle", extending
    _is_genuine_cycle_via_homology to also accept the open-path case. Degree (every node <=2,
    and either zero or exactly two degree-1 nodes) is checked directly via a plain Counter;
    connectivity and shape are then checked via SimplicialComplex.reduced_homology rather than
    assumed from the degree count alone -- a path must be contractible (every reduced homology
    group vanishes) and a no-endpoint graph must have exactly one nonzero group, rank 1,
    torsion-free H_1 (a single cycle). This catches, e.g., a cycle plus a disjoint dangling
    path, which superficially has "exactly two degree-1 nodes" but is not one connected path.
    """
    if not edges:
        return False
    degree = Counter()
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
    if any(d > 2 for d in degree.values()):
        return False
    n_degree_one = sum(1 for d in degree.values() if d == 1)
    if n_degree_one not in (0, 2):
        return False
    simplices = [(n,) for n in nodes] + [list(e) for e in edges]
    sc = SimplicialComplex.from_simplices(simplices, close_under_faces=True)
    rh = sc.reduced_homology(backend="python")
    non_zero = {k: v for k, v in rh.items() if v[0] > 0 or v[1]}
    if n_degree_one == 2:
        return len(non_zero) == 0  # a path is contractible
    if len(non_zero) != 1:
        return False
    (degree_k, (rank, torsion)), = non_zero.items()
    return degree_k == 1 and rank == 1 and not torsion


def test_is_single_path_or_cycle_matches_independent_homology_oracle_on_random_graphs():
    """Cross-check: is_single_path_or_cycle's graph-walk result must agree with an
    independently implemented, homology-based oracle, for many random graphs -- including
    disconnected combinations (e.g. a cycle plus a disjoint path) that a naive degree-only
    check could mistake for a single path."""
    rng = np.random.default_rng(11)
    n_checked = 0
    for _ in range(80):
        n_candidate_nodes = int(rng.integers(2, 8))
        possible_edges = list(itertools.combinations(range(n_candidate_nodes), 2))
        rng.shuffle(possible_edges)
        n_edges = int(rng.integers(1, len(possible_edges) + 1))
        edges = possible_edges[:n_edges]
        nodes_used = sorted({n for e in edges for n in e})
        if len(nodes_used) < 2:
            continue

        predicted_ok, _ = is_single_path_or_cycle(edges)
        actual_ok = _is_genuine_path_or_cycle_via_homology(nodes_used, edges)
        assert predicted_ok == actual_ok, f"mismatch on edges={edges}: predicted={predicted_ok}, oracle={actual_ok}"
        n_checked += 1
    assert n_checked > 30  # sanity: the random draws actually exercised enough cases


# ---------------------------------------------------------------------------
# intersect_local_stars
# ---------------------------------------------------------------------------

def test_intersect_local_stars_keeps_only_unanimous_simplices():
    """A simplex survives only if every one of its own vertices independently kept it."""
    s_abc = frozenset({0, 1, 2})
    s_abd = frozenset({0, 1, 3})
    per_vertex = {
        0: {s_abc, s_abd},
        1: {s_abc, s_abd},
        2: {s_abc},       # vertex 2 does not have s_abd (it's not even one of its vertices)
        3: set(),         # vertex 3 disagrees with s_abd
    }
    surviving, stats = intersect_local_stars(per_vertex)
    assert surviving == {s_abc}
    assert stats["n_candidates"] == 2
    assert stats["n_inconsistent"] == 1


def test_intersect_local_stars_empty_input():
    surviving, stats = intersect_local_stars({})
    assert surviving == set()
    assert stats == {"n_candidates": 0, "n_inconsistent": 0}


# ---------------------------------------------------------------------------
# moser_tardos_repair
# ---------------------------------------------------------------------------

def test_moser_tardos_repair_converges_and_targets_only_offending_points():
    """Engine mechanics: perturbs only the offending indices, calls rebuild_local each
    round, and stops as soon as detect_conflicts reports none."""
    points = np.zeros((4, 2))
    calls = {"detect": 0, "rebuild_args": []}

    def detect_conflicts(pts):
        calls["detect"] += 1
        if calls["detect"] <= 3:
            return [{"indices": [1, 2]}]
        return []

    def rebuild_local(pts, offending):
        calls["rebuild_args"].append(list(offending))

    result = moser_tardos_repair(
        points, detect_conflicts, rebuild_local, max_rounds=10, perturbation_scale=0.01, seed=0
    )
    assert result.converged
    assert result.rounds_used == 3
    assert calls["rebuild_args"] == [[1, 2], [1, 2], [1, 2]]
    assert np.allclose(result.points[0], [0.0, 0.0])
    assert np.allclose(result.points[3], [0.0, 0.0])
    assert not np.allclose(result.points[1], [0.0, 0.0])
    assert not np.allclose(result.points[2], [0.0, 0.0])


def test_moser_tardos_repair_is_deterministic_given_seed():
    """Same seed -> byte-identical perturbed output."""
    points = np.zeros((3, 2))

    def detect_conflicts(pts):
        return [{"indices": [0]}] if pts[0, 0] == 0.0 else []

    def rebuild_local(pts, offending):
        return None

    r1 = moser_tardos_repair(points, detect_conflicts, rebuild_local, max_rounds=5, seed=7)
    r2 = moser_tardos_repair(points, detect_conflicts, rebuild_local, max_rounds=5, seed=7)
    np.testing.assert_array_equal(r1.points, r2.points)


def test_moser_tardos_repair_raises_after_max_rounds():
    """A perpetually-unresolvable conflict raises ReconstructionRepairError, not an
    infinite loop or a silently-inconsistent result."""
    points = np.zeros((2, 2))

    def detect_conflicts(pts):
        return [{"indices": [0, 1]}]

    def rebuild_local(pts, offending):
        return None

    with pytest.raises(ReconstructionRepairError) as exc_info:
        moser_tardos_repair(points, detect_conflicts, rebuild_local, max_rounds=4, seed=0)
    err = exc_info.value
    assert err.stage == "reconstruction"
    assert err.rounds_attempted == 4
    assert err.max_rounds == 4
    assert err.offending_indices == [0, 1]


def test_moser_tardos_repair_raises_when_conflict_has_no_indices():
    """A conflict that names no offending indices cannot be targeted -> raise immediately."""
    points = np.zeros((2, 2))

    def detect_conflicts(pts):
        return [{"indices": []}]

    def rebuild_local(pts, offending):
        raise AssertionError("rebuild_local should never be called with no offending indices")

    with pytest.raises(ReconstructionRepairError):
        moser_tardos_repair(points, detect_conflicts, rebuild_local, max_rounds=5, seed=0)


def test_moser_tardos_repair_already_clean_does_nothing():
    """No conflicts at all -> zero rounds used, points unchanged."""
    points = np.array([[1.0, 2.0], [3.0, 4.0]])

    def detect_conflicts(pts):
        return []

    def rebuild_local(pts, offending):
        raise AssertionError("rebuild_local should never be called when already clean")

    result = moser_tardos_repair(points, detect_conflicts, rebuild_local)
    assert result.converged
    assert result.rounds_used == 0
    np.testing.assert_array_equal(result.points, points)
