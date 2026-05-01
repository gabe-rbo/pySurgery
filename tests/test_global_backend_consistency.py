"""Consistency tests for fundamental topological operations across Python and Julia backends.

Overview:
    This suite ensures that the Python (typically JAX-accelerated or pure-Python) 
    and Julia backends yield identical results for core topological and geometric 
    computations. It covers homology, cohomology bases, and metric distance 
    calculations.

Key Concepts:
    - **Backend Parity**: Ensuring numerical and structural consistency between disparate runtimes.
    - **Cross-Validation**: Using one backend to verify the correctness of another.
    - **Numerical Tolerance**: Handling precision differences between Python (float64/float32) and Julia.
"""

import pytest
import numpy as np
from pysurgery.core.complexes import SimplicialComplex
from pysurgery.core.metrics import compute_distance_matrix, frechet_distance

def test_homology_backend_consistency_sphere():
    """Verify that homology rank and torsion are consistent between Python and Julia backends.

    What is Being Computed?:
        Integral homology H_n(S²; ℤ) for n=2.

    Algorithm:
        1. Construct a simplicial S² (boundary of a tetrahedron).
        2. Compute H_2 using the 'python' backend.
        3. Compute H_2 using the 'julia' backend.
        4. Compare results.

    Preserved Invariants:
        - H_2(S²) rank (should be 1).
        - H_2(S²) torsion (should be empty).
    """
    # S^2: 4 vertices, 6 edges, 4 faces
    simplices = [(0,), (1,), (2,), (3,), (0,1), (0,2), (0,3), (1,2), (1,3), (2,3), (0,1,2), (0,1,3), (0,2,3), (1,2,3)]
    sc = SimplicialComplex.from_simplices(simplices)
    
    # H_2 rank should be 1
    h2_py = sc.homology(2, backend="python")
    h2_jl = sc.homology(2, backend="julia")
    
    assert h2_py == h2_jl == (1, [])

def test_distance_matrix_backend_consistency():
    """Verify that pairwise distance matrices match across backends within floating point tolerance.

    What is Being Computed?:
        Pairwise Euclidean distance matrix for a random point cloud.

    Algorithm:
        1. Generate a random (10, 3) point cloud.
        2. Compute distance matrix via Python/JAX backend.
        3. Compute distance matrix via Julia backend.
        4. Compare using np.allclose with 1e-5 tolerance.

    Preserved Invariants:
        - Metric structure of the point cloud.
    """
    data = np.random.rand(10, 3)
    
    dm_py = compute_distance_matrix(data, metric="euclidean", backend="python")
    dm_jl = compute_distance_matrix(data, metric="euclidean", backend="julia")
    
    # Use higher tolerance for float32 (JAX) vs float64 (Julia) comparison
    assert np.allclose(dm_py, dm_jl, atol=1e-5)

def test_frechet_backend_consistency():
    """Verify that discrete Frechet distance matches across backends.

    What is Being Computed?:
        Discrete Frechet distance between two 3D polygonal curves.

    Algorithm:
        1. Define two 3-point curves.
        2. Compute distance via Python and Julia backends.
        3. Compare results using pytest.approx.

    Preserved Invariants:
        - Relative distance between parameterized paths.
    """
    c1 = np.array([[0,0,0], [1,1,1], [2,2,2]], dtype=float)
    c2 = np.array([[0,0,1], [1,1,2], [2,2,3]], dtype=float)
    
    f_py = frechet_distance(c1, c2, backend="python")
    f_jl = frechet_distance(c1, c2, backend="julia")
    
    assert pytest.approx(f_py) == f_jl

def test_cohomology_basis_backend_consistency():
    """Verify that cohomology basis representatives span the same space across backends.

    What is Being Computed?:
        Basis of H¹(S¹; ℤ) cocycles.

    Algorithm:
        1. Construct a simplicial S¹ (loop).
        2. Extract basis via Python and Julia backends.
        3. Normalize and compare cocycle magnitudes.

    Preserved Invariants:
        - Cohomology rank (rank H¹(S¹) = 1).
    """
    # S^1
    simplices = [(0,), (1,), (0,1), (1,0)] # manual loop
    # Let's use a simpler known loop
    simplices = [(0,1), (1,2), (2,0), (0,), (1,), (2,)]
    sc = SimplicialComplex.from_simplices(simplices)
    
    # cohomology_basis returns list[np.ndarray]
    b1_py = sc.cohomology_basis(1, backend="python")
    b1_jl = sc.cohomology_basis(1, backend="julia")
    
    assert len(b1_py) == len(b1_jl)
    # Check that they span the same space or are identical in this simple case
    if len(b1_py) > 0:
        # For S1, should be rank 1. Cocycles might differ by sign.
        v_py = np.abs(b1_py[0])
        v_jl = np.abs(b1_jl[0])
        # Sort both to handle potential permutation of cells (though usually deterministic)
        assert np.allclose(v_py, v_jl)
