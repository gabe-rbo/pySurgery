"""Tests for the unified covering API (topology.coverings)."""

import numpy as np
import pytest
import scipy.sparse as sp

from pysurgery.topology.complexes import CWComplex
from pysurgery.topology.fundamental_group import (
    FundamentalGroup,
    extract_pi_1_with_traces,
)
from pysurgery.topology.graphs import Graph
from pysurgery.topology.coverings import (
    Covering,
    DeckTransformationGroup,
    UniversalCover,
    cover_graph,
    graph_universal_cover,
    _coset_permutation_rep,
)


def _cw_with_attaching(d2_cols):
    """1-vertex CW with 2-cells attached by the given integer columns."""
    n_e = len(d2_cols[0]) if d2_cols else 0
    n_f = len(d2_cols)
    d2 = np.zeros((n_e, n_f), dtype=np.int64)
    for j, col in enumerate(d2_cols):
        for i, v in enumerate(col):
            d2[i, j] = int(v)
    cells = {0: 1, 1: n_e, 2: n_f}
    maps = {
        1: sp.csr_matrix(np.zeros((1, n_e), dtype=np.int64)),
        2: sp.csr_matrix(d2),
    }
    return CWComplex(cells=cells, attaching_maps=maps, dimensions=[0, 1, 2])


@pytest.fixture
def rp2_min():
    return _cw_with_attaching([[2]])  # π₁ = ℤ/2, relator a²


@pytest.fixture
def wedge2():
    # S¹ ∨ S¹: one vertex, two edges, no 2-cells. π₁ = F₂.
    return CWComplex(
        cells={0: 1, 1: 2, 2: 0},
        attaching_maps={1: sp.csr_matrix(np.zeros((1, 2), dtype=np.int64))},
        dimensions=[0, 1],
    )


class TestUniversalCoverDeck:
    def test_rp2_deck_group_order_two_and_free(self, rp2_min):
        cover = UniversalCover(rp2_min)
        dg = cover.deck_group()
        assert isinstance(dg, DeckTransformationGroup)
        assert dg.order == 2
        assert dg.is_free()

    def test_rp2_cover_is_s2(self, rp2_min):
        cover = UniversalCover(rp2_min)
        cc = cover.as_chain_complex()
        assert cc.homology(0) == (1, [])
        assert cc.homology(1) == (0, [])
        assert cc.homology(2) == (1, [])

    def test_euler_characteristic_multiplicative(self, rp2_min):
        cov = UniversalCover(rp2_min).as_covering()
        assert cov.degree == 2
        assert cov.euler_characteristic_consistent()
        assert cov.total_space.euler_characteristic() == 2 * rp2_min.euler_characteristic()

    def test_lift_path_and_monodromy(self, rp2_min):
        cover = UniversalCover(rp2_min)
        # a² is the relator → lifts to a loop (returns to the start sheet).
        assert cover.monodromy(["g_0", "g_0"], 0) == 0
        # a alone swaps the two sheets.
        assert cover.monodromy(["g_0"], 0) == 1
        assert cover.lift_path(["g_0"], 0) == [0, 1]

    def test_fiber_and_covering_map(self, rp2_min):
        cov = UniversalCover(rp2_min).as_covering()
        assert sorted(cov.fiber(0, 0)) == [0, 1]  # two points over the 0-cell
        assert cov.is_regular()


class TestPermutationCover:
    def test_index_two_cover_of_wedge(self, wedge2):
        gens = extract_pi_1_with_traces(
            wedge2, simplify=False, generator_mode="raw"
        ).generators
        rep = {gens[0]: [1, 0], gens[1]: [0, 1]}
        cov = Covering.from_permutation_rep(wedge2, rep)
        assert cov.degree == 2
        # connected double cover of a wedge of 2 circles: χ = -2 ⇒ b1 = 3.
        assert cov.total_space.euler_characteristic() == 2 * wedge2.euler_characteristic()
        assert cov.total_space.betti_number(1) == 3
        assert cov.is_regular()  # index-2 covers are always normal

    def test_trivial_rep_is_disjoint_copies(self, wedge2):
        gens = extract_pi_1_with_traces(
            wedge2, simplify=False, generator_mode="raw"
        ).generators
        rep = {gens[0]: [0, 1], gens[1]: [0, 1]}  # identity action
        cov = Covering.from_permutation_rep(wedge2, rep)
        # two disjoint copies of S¹∨S¹
        assert cov.total_space.betti_number(0) == 2

    def test_nonregular_cover_has_small_deck_group(self, wedge2):
        gens = extract_pi_1_with_traces(
            wedge2, simplify=False, generator_mode="raw"
        ).generators
        # S₃ acting naturally on 3 points: a=(0 1 2), b=(0 1). Non-normal.
        rep = {gens[0]: [1, 2, 0], gens[1]: [1, 0, 2]}
        cov = Covering.from_permutation_rep(wedge2, rep)
        assert cov.degree == 3
        assert not cov.is_regular()
        assert cov.deck_group().order == 1

    def test_invalid_rep_raises(self, wedge2):
        gens = extract_pi_1_with_traces(
            wedge2, simplify=False, generator_mode="raw"
        ).generators
        with pytest.raises(Exception):
            Covering.from_permutation_rep(wedge2, {gens[0]: [0, 0], gens[1]: [0, 1]})


class TestCosetEnumeration:
    @pytest.mark.parametrize(
        "gens,rels,sub,expected",
        [
            (["a"], [], [["a", "a"]], 2),                 # ⟨a² ⟩ ≤ ℤ
            (["a"], [], [["a", "a", "a"]], 3),            # ⟨a³ ⟩ ≤ ℤ
            (["a"], [["a", "a", "a", "a", "a"]], [], 5),  # 1 ≤ ℤ/5
            (["a", "b"],
             [["a", "a"], ["b", "b"], ["a", "b", "a", "b", "a", "b"]],
             [], 6),                                      # 1 ≤ S₃
        ],
    )
    def test_known_indices(self, gens, rels, sub, expected):
        rep, index = _coset_permutation_rep(gens, rels, sub, max_cosets=2000)
        assert index == expected
        for p in rep.values():
            assert sorted(p) == list(range(index))

    def test_infinite_index_raises(self):
        from pysurgery.core.exceptions import FundamentalGroupError
        # ⟨a⟩ ≤ F₂ has infinite index.
        with pytest.raises(FundamentalGroupError):
            _coset_permutation_rep(["a", "b"], [], [["a"]], max_cosets=200)

    def test_from_subgroup_abstract(self):
        pi = FundamentalGroup(generators=["a"], relations=[])
        cov = Covering.from_subgroup(pi, [["a", "a"]])
        assert cov.degree == 2
        assert cov.total_space is None
        assert cov.monodromy(["a"], 0) == 1


class TestGraphCovers:
    def test_universal_cover_is_a_tree(self):
        tri = Graph.from_edges([(0, 1), (1, 2), (2, 0)])
        tree = graph_universal_cover(tri, depth=4)
        assert tree.is_tree
        assert tree.cyclomatic_number == 0

    def test_universal_cover_via_method(self):
        tri = Graph.from_edges([(0, 1), (1, 2), (2, 0)])
        tree = tri.universal_cover(depth=3)
        assert tree.betti_number(1) == 0

    def test_voltage_cover_euler_multiplicative(self):
        tri = Graph.from_edges([(0, 1), (1, 2), (2, 0)])
        # 2-sheet cover, edge 0 swaps the sheets → connected double cover.
        cg = cover_graph(tri, {0: [1, 0]})
        assert cg.euler_characteristic() == 2 * tri.euler_characteristic()
        assert cg.is_connected()
        assert cg.betti_number(1) == 1

    def test_trivial_voltage_is_disjoint(self):
        tri = Graph.from_edges([(0, 1), (1, 2), (2, 0)])
        cg = cover_graph(tri, {0: [0, 1]})  # identity voltages
        assert cg.num_connected_components() == 2


class TestFundamentalGroupBridge:
    def test_universal_cover_method(self, rp2_min):
        pi = FundamentalGroup(generators=["g_0"], relations=[["g_0", "g_0"]])
        cover = pi.universal_cover(rp2_min)
        assert cover.order == 2
        assert pi.as_deck_group(rp2_min).order == 2


class TestGraphWalkLifting:
    def test_lift_walk_covers_back(self):
        tri = Graph.cycle(3)
        gc = tri.cover({0: [1, 0]})  # connected double cover
        walk = [0, 1, 2, 0, 1, 2, 0]
        lifted = gc.lift_walk(walk, start=gc.fiber(0)[0])
        assert [gc.covering_map(w) for w in lifted] == walk

    def test_double_cover_closes_after_two_loops(self):
        tri = Graph.cycle(3)
        gc = tri.cover({0: [1, 0]})
        start = gc.fiber(0)[0]
        once = gc.lift_walk([0, 1, 2, 0], start=start)
        twice = gc.lift_walk([0, 1, 2, 0, 1, 2, 0], start=start)
        assert once[-1] != start          # one loop does not close
        assert twice[-1] == start         # two loops do

    def test_universal_cover_lift_and_render(self):
        tri = Graph.cycle(3)
        gc = tri.universal_cover(depth=3)
        assert gc.is_tree  # delegated to the cover graph
        lifted = gc.lift_walk([0, 1, 2], start=0)
        assert [gc.covering_map(w) for w in lifted] == [0, 1, 2]
        assert isinstance(gc.render(), str) and "0" in gc.render()

    def test_graphcovering_delegates_graph_methods(self):
        gc = Graph.cycle(3).cover({0: [1, 0]})
        # methods/properties resolve on the underlying cover graph
        assert gc.num_vertices == 6
        assert gc.is_connected()
        assert gc.betti_number(1) == 1


class TestGroupActions:
    def _wedge_and_gens(self):
        wedge = CWComplex(
            cells={0: 1, 1: 2, 2: 0},
            attaching_maps={1: sp.csr_matrix(np.zeros((1, 2), dtype=np.int64))},
            dimensions=[0, 1],
        )
        gens = extract_pi_1_with_traces(
            wedge, simplify=False, generator_mode="raw"
        ).generators
        return wedge, gens

    def test_monodromy_orbit_and_transitivity(self):
        wedge, g = self._wedge_and_gens()
        cov = Covering.from_permutation_rep(wedge, {g[0]: [1, 2, 0], g[1]: [1, 0, 2]})
        act = cov.monodromy_action()
        assert act.orbit(0) == [0, 1, 2]
        assert act.is_transitive()
        assert act.stabilizer_index(0) == 3

    def test_subgroup_generators_stabilize_basepoint(self):
        wedge, g = self._wedge_and_gens()
        cov = Covering.from_permutation_rep(wedge, {g[0]: [1, 2, 0], g[1]: [1, 0, 2]})
        act = cov.monodromy_action()
        for w in cov.subgroup_generators():
            assert act.apply_word(w, 0) == 0  # really fixes sheet 0

    def test_disconnected_action_has_multiple_orbits(self):
        wedge, g = self._wedge_and_gens()
        cov = Covering.from_permutation_rep(wedge, {g[0]: [1, 0, 2], g[1]: [0, 1, 2]})
        act = cov.monodromy_action()
        assert not act.is_transitive()
        assert len(act.orbits()) == 2  # {0,1} and {2}

    def test_deck_orbit_and_free_stabilizer(self, rp2_min):
        dg = UniversalCover(rp2_min).deck_group()
        assert dg.orbit(0, 0) == [0, 1]          # the two lifts of the 0-cell
        assert len(dg.stabilizer(0, 0)) == 1     # free action ⇒ only identity


class TestCellularLifting:
    def test_lift_cellular_path_rp2(self, rp2_min):
        cover = UniversalCover(rp2_min)
        path = cover.lift_cellular_path(["g_0", "g_0"], 0)
        # a² traverses the single 1-cell on sheet 0 then on sheet 1
        assert path == [(0, 0), (0, 1)]
        # and the sheet sequence closes (a² is a relator)
        assert cover.lift_path(["g_0", "g_0"], 0) == [0, 1, 0]
