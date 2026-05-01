"""Comprehensive consistency tests between Python and Julia backends.

Overview:
    This suite ensures that topological operations (expansion, simplification, 
    filtration, and basis computation) yield identical results across the 
    native Python and accelerated Julia backends.

Key Concepts:
    - **Backend Consistency**: Numerical and combinatorial agreement between different engine implementations.
    - **Filtration Methods**: Vietoris-Rips, CkNN, and Crust algorithm consistency.
    - **Reduction Consistency**: Agreement in simplify() and quick_mapper() heuristics.
"""
import numpy as np
import pytest
from pysurgery.core.complexes import SimplicialComplex
from pysurgery.core.homology_generators import compute_optimal_h1_basis_from_simplices
from pysurgery.bridge.julia_bridge import julia_engine

@pytest.mark.skipif(not julia_engine.available, reason="Julia backend not available")
class TestBackendConsistency:
    """Test suite for validating Python vs Julia computational parity."""

    def test_expand_consistency(self):
        """Verify that skeletal expansion is identical across backends.

        What is Being Computed?:
            The k-skeleton expansion of a given set of simplices.

        Algorithm:
            1. Construct a 1-skeleton (square and K₄).
            2. Expand to higher dimensions using both Python and Julia backends.
            3. Compare the resulting simplex sets.

        Preserved Invariants:
            - The expanded complex must contain the same higher-dimensional simplices.
        """
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
        """Verify that simplification heuristics preserve homotopy type consistently.

        What is Being Computed?:
            The simplified (collapsed) complex and its invariants.

        Algorithm:
            1. Define a complex homotopy equivalent to S¹.
            2. Run simplify() on both backends.
            3. Verify that Euler characteristic and H₁ rank match.

        Preserved Invariants:
            - Euler characteristic (χ).
            - Homology groups (Hₙ).
        """
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
        """Verify consistency for the quick_mapper dimensionality reduction tool.

        What is Being Computed?:
            A reduced complex using modularity-based vertex clustering.

        Algorithm:
            1. Define two disjoint clusters.
            2. Run quick_mapper with fixed iterations on both backends.
            3. Check that both achieve vertex reduction.
        """
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
        """Verify Vietoris-Rips complex construction across backends.

        What is Being Computed?:
            A VR complex for a fixed threshold ε.

        Algorithm:
            1. Define 7 points in 2D.
            2. Construct VR complex with ε=1.5 using both backends.
            3. Assert identical simplex sets.
        """
        points = np.array([
            [0, 0], [1, 0], [0, 1], [1, 1],
            [10, 10], [11, 10], [10, 11]
        ])
        
        sc_py = SimplicialComplex.from_vietoris_rips(points, epsilon=1.5, max_dimension=2, backend="python")
        sc_jl = SimplicialComplex.from_vietoris_rips(points, epsilon=1.5, max_dimension=2, backend="julia")
        
        assert sorted(sc_py.simplices) == sorted(sc_jl.simplices)

    def test_from_point_cloud_cknn_consistency(self):
        """Verify Continuous k-Nearest Neighbors (CkNN) construction consistency.

        What is Being Computed?:
            A CkNN complex for fixed k and delta.
        """
        points = np.random.rand(20, 2)
        
        sc_py = SimplicialComplex.from_point_cloud_cknn(points, k=3, delta=1.0, backend="python")
        sc_jl = SimplicialComplex.from_point_cloud_cknn(points, k=3, delta=1.0, backend="julia")
        
        # CkNN might have slight differences if ties in distances are handled differently,
        # but for random points it should be consistent.
        assert sorted(sc_py.simplices) == sorted(sc_jl.simplices)

    def test_from_crust_algorithm_consistency(self):
        """Verify the Crust algorithm for surface reconstruction across backends.

        What is Being Computed?:
            A SimplicialComplex representing the surface reconstruction of a point set.
        """
        # Points on a circle
        theta = np.linspace(0, 2*np.pi, 10, endpoint=False)
        points = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        
        sc_py = SimplicialComplex.from_crust_algorithm(points, backend="python")
        sc_jl = SimplicialComplex.from_crust_algorithm(points, backend="julia")
        
        assert sorted(sc_py.simplices) == sorted(sc_jl.simplices)

    def test_compute_optimal_h1_basis_consistency(self):
        """Verify optimal homology basis computation consistency.

        What is Being Computed?:
            An optimal cycle basis for H₁(X; ℤ).

        Algorithm:
            1. Define two disjoint loops.
            2. Compute the optimal H₁ basis using both backends.
            3. Verify that the basis rank is 2 for both.
        """
        # S1 x S1 (Torus)
        simplices = [
            (0, 1, 2), (1, 2, 3), # ... simplified torus
        ]
        # Better use a clear 1-skeleton cycle
        simplices = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)] # Two disjoint loops
        
        res_py = compute_optimal_h1_basis_from_simplices(simplices, 6, backend="python")
        res_jl = compute_optimal_h1_basis_from_simplices(simplices, 6, backend="julia")
        
        assert res_py.rank == res_jl.rank == 2
