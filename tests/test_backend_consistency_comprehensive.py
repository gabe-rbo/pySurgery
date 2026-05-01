import numpy as np
import pytest
from pysurgery.core.complexes import SimplicialComplex
from pysurgery.core.homology_generators import compute_optimal_h1_basis_from_simplices
from pysurgery.bridge.julia_bridge import julia_engine

@pytest.mark.skipif(not julia_engine.available, reason="Julia backend not available")
class TestBackendConsistency:
    def test_expand_consistency(self):
        # Create a simple cycle (square)
        simplices = [(0, 1), (1, 2), (2, 3), (3, 0)]
        sc = SimplicialComplex.from_simplices(simplices, close_under_faces=True)
        
        # Expand up to dimension 1 (should stay the same)
        sc_py = sc.expand(max_dim=1, backend="python")
        sc_jl = sc.expand(max_dim=1, backend="julia")
        
        assert sorted(sc_py.simplices) == sorted(sc_jl.simplices)
        
        # Expand a complete graph K4
        simplices_k4 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        sc_k4 = SimplicialComplex.from_simplices(simplices_k4, close_under_faces=True)
        
        sc_k4_py = sc_k4.expand(max_dim=3, backend="python")
        sc_k4_jl = sc_k4.expand(max_dim=3, backend="julia")
        
        assert sorted(sc_k4_py.simplices) == sorted(sc_k4_jl.simplices)
        assert (0, 1, 2, 3) in sc_k4_py.simplices
        assert (0, 1, 2, 3) in sc_k4_jl.simplices

    def test_simplify_consistency(self):
        # Create a "thick" line that can be simplified to a circle
        simplices = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 1)]
        sc = SimplicialComplex.from_simplices(simplices, close_under_faces=True)
        
        sc_py, _ = sc.simplify(backend="python")
        sc_jl, _ = sc.simplify(backend="julia")
        
        # Both should be homotopy equivalent to S1 (Euler char 0)
        assert sc_py.euler_characteristic() == 0
        assert sc_jl.euler_characteristic() == 0
        # Homology H1 should be rank 1
        assert sc_py.homology(1)[0] == 1
        assert sc_jl.homology(1)[0] == 1

    def test_quick_mapper_consistency(self):
        # Create two disjoint clusters
        simplices = [(0, 1), (1, 2), (2, 0), (10, 11), (11, 12), (12, 10)]
        sc = SimplicialComplex.from_simplices(simplices, close_under_faces=True)
        
        # Higher iterations to ensure modularity converges on this small example
        sc_py, _ = sc.quick_mapper(max_loops=10, backend="python")
        sc_jl, _ = sc.quick_mapper(max_loops=10, backend="julia")
        
        # Check that both backends reduced the number of vertices
        # (Original count was 6)
        assert len(sc_py.n_simplices(0)) < 6
        assert len(sc_jl.n_simplices(0)) < 6

    def test_from_vietoris_rips_consistency(self):
        points = np.array([
            [0, 0], [1, 0], [0, 1], [1, 1],
            [10, 10], [11, 10], [10, 11]
        ])
        
        sc_py = SimplicialComplex.from_vietoris_rips(points, epsilon=1.5, max_dimension=2, backend="python")
        sc_jl = SimplicialComplex.from_vietoris_rips(points, epsilon=1.5, max_dimension=2, backend="julia")
        
        assert sorted(sc_py.simplices) == sorted(sc_jl.simplices)

    def test_from_point_cloud_cknn_consistency(self):
        points = np.random.rand(20, 2)
        
        sc_py = SimplicialComplex.from_point_cloud_cknn(points, k=3, delta=1.0, backend="python")
        sc_jl = SimplicialComplex.from_point_cloud_cknn(points, k=3, delta=1.0, backend="julia")
        
        # CkNN might have slight differences if ties in distances are handled differently,
        # but for random points it should be consistent.
        assert sorted(sc_py.simplices) == sorted(sc_jl.simplices)

    def test_from_crust_algorithm_consistency(self):
        # Points on a circle
        theta = np.linspace(0, 2*np.pi, 10, endpoint=False)
        points = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        
        sc_py = SimplicialComplex.from_crust_algorithm(points, backend="python")
        sc_jl = SimplicialComplex.from_crust_algorithm(points, backend="julia")
        
        assert sorted(sc_py.simplices) == sorted(sc_jl.simplices)

    def test_compute_optimal_h1_basis_consistency(self):
        # S1 x S1 (Torus)
        simplices = [
            (0, 1, 2), (1, 2, 3), # ... simplified torus
        ]
        # Better use a clear 1-skeleton cycle
        simplices = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)] # Two disjoint loops
        
        res_py = compute_optimal_h1_basis_from_simplices(simplices, 6, backend="python")
        res_jl = compute_optimal_h1_basis_from_simplices(simplices, 6, backend="julia")
        
        assert res_py.rank == res_jl.rank == 2
