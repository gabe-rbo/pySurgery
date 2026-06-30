"""Tests for the hybrid ``Graph`` type (topology.graphs)."""

import pytest

from pysurgery.topology.complexes import SimplicialComplex
from pysurgery.topology.graphs import Graph, bfs_spanning_forest, lca_tree_path


class TestGraphConstruction:
    def test_from_edges_basic(self):
        g = Graph.from_edges([(0, 1), (1, 2), (2, 0)])
        assert g.num_vertices == 3
        assert g.num_edges == 3
        assert g.dimension == 1
        assert sorted(g.vertices) == [0, 1, 2]

    def test_isolated_vertices_via_num_vertices(self):
        g = Graph.from_edges([(0, 1)], num_vertices=4)
        assert g.num_vertices == 4
        assert g.num_connected_components() == 3  # {0,1}, {2}, {3}

    def test_from_adjacency(self):
        g = Graph.from_adjacency({0: [1, 2], 1: [2], 2: []})
        assert g.num_vertices == 3
        assert g.num_edges == 3
        assert set(g.neighbors(0)) == {1, 2}

    def test_from_and_to_simplicial_complex_roundtrip(self):
        # 1-skeleton of a filled triangle is the boundary circle (b1 = 1).
        sc = SimplicialComplex.from_simplices([(0, 1, 2)])
        g = Graph.from_simplicial_complex(sc)
        assert g.cyclomatic_number == 1
        sc2 = g.to_simplicial_complex()
        assert sc2.betti_number(1) == 1
        assert sc2.dimension == 1


class TestGraphInvariants:
    def test_cyclomatic_equals_betti1_for_simple_graphs(self):
        # triangle: one independent cycle
        tri = Graph.from_edges([(0, 1), (1, 2), (2, 0)])
        assert tri.cyclomatic_number == 1
        assert tri.betti_number(1) == 1

        # K4: E - V + C = 6 - 4 + 1 = 3
        k4 = Graph.from_edges(
            [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        )
        assert k4.cyclomatic_number == 3
        assert k4.betti_number(1) == 3

    def test_tree_has_no_cycles(self):
        path = Graph.from_edges([(0, 1), (1, 2), (2, 3)])
        assert path.cyclomatic_number == 0
        assert path.is_forest
        assert path.is_tree
        assert path.betti_number(1) == 0

    def test_spanning_forest_size(self):
        # Two components: a triangle and an edge → forest has (3-1)+(2-1)=3 edges.
        g = Graph.from_edges([(0, 1), (1, 2), (2, 0), (3, 4)])
        tree_ids, parent, depth, root = g.spanning_forest()
        assert g.num_connected_components() == 2
        assert len(tree_ids) == g.num_vertices - g.num_connected_components()

    def test_betti0_counts_components(self):
        g = Graph.from_edges([(0, 1), (2, 3)])
        assert g.betti_number(0) == 2


class TestGraphPi1:
    def test_pi1_of_wedge_of_k_circles_is_free_rank_k(self):
        for k in (1, 2, 3, 4):
            # k self-loops at a single vertex = wedge of k circles.
            g = Graph.from_edges([(0, 0)] * k, num_vertices=1)
            assert g.cyclomatic_number == k
            pi = g.fundamental_group()
            assert len(pi.generators) == k
            assert pi.relations == []  # free group

    def test_pi1_of_tree_is_trivial(self):
        g = Graph.from_edges([(0, 1), (1, 2)])
        pi = g.fundamental_group()
        assert pi.generators == []

    def test_fundamental_cycles_count(self):
        tri = Graph.from_edges([(0, 1), (1, 2), (2, 0)])
        cycles = tri.fundamental_cycles()
        assert len(cycles) == 1
        loop = cycles[0]
        # closed loop: ends where it starts
        assert loop[0][0] == loop[-1][1]


class TestMultigraphTolerance:
    def test_parallel_edges_counted_in_cyclomatic(self):
        # Three parallel edges between two vertices: b1 = 3 - 2 + 1 = 2.
        g = Graph.from_edges([(0, 1), (0, 1), (0, 1)])
        assert g.cyclomatic_number == 2
        # The *simple* 1-skeleton collapses them, so inherited b1 differs.
        assert g.betti_number(1) == 0
        assert g.degree(0) == 3

    def test_self_loop_tolerance(self):
        g = Graph.from_edges([(0, 1), (1, 1)])
        assert g.cyclomatic_number == 1
        assert g.degree(1) == 3  # one ordinary edge + a loop (counts twice)
        assert g.fundamental_cycles() == [[(1, 1)]]


class TestStructuralOps:
    def test_subdivide_preserves_topology(self):
        tri = Graph.from_edges([(0, 1), (1, 2), (2, 0)])
        sub = tri.subdivide_edge(0, 1)
        assert sub.num_vertices == 4
        assert sub.cyclomatic_number == 1  # still one cycle

    def test_contract_edge_reduces_vertices(self):
        tri = Graph.from_edges([(0, 1), (1, 2), (2, 0)])
        contracted = tri.contract_edge(0, 1)
        # contracting one edge of a triangle yields a 2-vertex multigraph
        # (a single cycle remains).
        assert contracted.cyclomatic_number == 1

    def test_line_graph_of_path(self):
        # Line graph of a path P3 (edges e0,e1 sharing vertex 1) is a single edge.
        path = Graph.from_edges([(0, 1), (1, 2)])
        lg = path.line_graph()
        assert lg.num_vertices == 2
        assert lg.num_edges == 1


class TestLowLevelRoutines:
    def test_bfs_spanning_forest_direct(self):
        # 0-1-2 path adjacency
        adj = {
            0: [(1, 0, 1)],
            1: [(0, 0, -1), (2, 1, 1)],
            2: [(1, 1, -1)],
        }
        tree_ids, parent, depth, root = bfs_spanning_forest([0, 1, 2], adj)
        assert tree_ids == {0, 1}
        assert depth[0] == 0 and depth[2] == 2
        assert all(root[v] == 0 for v in (0, 1, 2))

    def test_lca_tree_path(self):
        parent = {0: -1, 1: 0, 2: 1}
        depth = {0: 0, 1: 1, 2: 2}
        assert lca_tree_path(2, 0, parent, depth) == [2, 1, 0]


class TestExampleGraphs:
    def test_cycle(self):
        c = Graph.cycle(5)
        assert c.num_vertices == 5 and c.num_edges == 5
        assert c.cyclomatic_number == 1

    def test_complete(self):
        k4 = Graph.complete(4)
        assert k4.num_edges == 6
        assert k4.betti_number(1) == 3

    def test_complete_bipartite_and_star_and_path(self):
        assert Graph.complete_bipartite(2, 3).num_edges == 6
        assert Graph.star(4).num_vertices == 5 and Graph.star(4).is_tree
        assert Graph.path(4).is_tree

    def test_bouquet_is_free(self):
        b = Graph.bouquet(3)
        assert b.cyclomatic_number == 3
        assert len(b.fundamental_group().generators) == 3

    def test_petersen(self):
        p = Graph.petersen()
        assert p.num_vertices == 10 and p.num_edges == 15
        assert all(p.degree(v) == 3 for v in p.vertices)

    def test_grid(self):
        g = Graph.grid(2, 3)
        assert g.num_vertices == 6
        assert g.betti_number(1) == 2  # 2 independent squares


class TestMatricesAndLaplacians:
    def test_adjacency_symmetric(self):
        c = Graph.cycle(4)
        A = c.adjacency_matrix()
        assert (A == A.T).all()
        assert A.sum() == 2 * c.num_edges

    def test_laplacian_zero_row_sums(self):
        g = Graph.complete(4)
        L = g.laplacian()
        assert (L.sum(axis=1) == 0).all()

    def test_hodge_l0_equals_graph_laplacian(self):
        g = Graph.cycle(5)
        import numpy as np
        assert np.array_equal(g.hodge_laplacian(0, sparse=False), g.laplacian())

    def test_hodge_kernels_are_betti(self):
        import numpy as np
        g = Graph.complete(4)
        for k in (0, 1):
            L = g.hodge_laplacian(k, sparse=False).astype(float)
            nullity = int(round((np.abs(np.linalg.eigvalsh(L)) < 1e-9).sum()))
            assert nullity == g.betti_number(k)


class TestZetaFunctions:
    def test_ihara_triangle(self):
        import sympy as sp
        u = sp.symbols("u")
        recip = Graph.cycle(3).ihara_zeta_reciprocal()
        expected = (u - 1) ** 2 * (u**2 + u + 1) ** 2
        assert sp.simplify(recip - expected) == 0

    def test_tree_zeta_is_trivial(self):
        import sympy as sp
        assert sp.simplify(Graph.path(4).ihara_zeta_reciprocal()) == 1

    def test_bartholdi_reduces_to_ihara_at_t0(self):
        import sympy as sp
        g = Graph.complete(4)
        assert sp.simplify(g.bartholdi_zeta_reciprocal(t=0) - g.ihara_zeta_reciprocal()) == 0


class TestPrimeCycles:
    def test_triangle_primes(self):
        c3 = Graph.cycle(3)
        assert len(c3.prime_cycles(3)) == 2                      # both directions
        assert len(c3.prime_cycles(3, up_to_inversion=True)) == 1

    def test_hashimoto_trace_matches_prime_count(self):
        import numpy as np
        from collections import Counter
        g = Graph.complete(4)
        B, _darts = g.hashimoto_matrix()
        cnt = Counter(len(c) for c in g.prime_cycles(5))
        for m in range(1, 6):
            trBm = int(round(np.trace(np.linalg.matrix_power(B.astype(float), m))))
            expected = sum(d * cnt.get(d, 0) for d in range(1, m + 1) if m % d == 0)
            assert trBm == expected, f"m={m}: {trBm} != {expected}"


class TestVisualization:
    def test_to_dot(self):
        dot = Graph.cycle(3).to_dot()
        assert dot.startswith("graph G {")
        assert dot.count("--") == 3

    def test_to_ascii_tree_of_path(self):
        ascii_tree = Graph.path(3).to_ascii_tree()
        assert "0" in ascii_tree and "1" in ascii_tree and "2" in ascii_tree
