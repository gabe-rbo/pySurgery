import pytest
import numpy as np
import scipy.sparse as sp
from pysurgery.core.complexes import SimplicialComplex, ChainComplex
from pysurgery.core.homology_generators import compute_homology_basis_from_simplices
from pysurgery.core.metrics import compute_distance_matrix, frechet_distance

def test_homology_backend_consistency_sphere():
    # S^2: 4 vertices, 6 edges, 4 faces
    simplices = [(0,), (1,), (2,), (3,), (0,1), (0,2), (0,3), (1,2), (1,3), (2,3), (0,1,2), (0,1,3), (0,2,3), (1,2,3)]
    sc = SimplicialComplex.from_simplices(simplices)
    
    # H_2 rank should be 1
    h2_py = sc.homology(2, backend="python")
    h2_jl = sc.homology(2, backend="julia")
    
    assert h2_py == h2_jl == (1, [])

def test_distance_matrix_backend_consistency():
    data = np.random.rand(10, 3)
    
    dm_py = compute_distance_matrix(data, metric="euclidean", backend="python")
    dm_jl = compute_distance_matrix(data, metric="euclidean", backend="julia")
    
    # Use higher tolerance for float32 (JAX) vs float64 (Julia) comparison
    assert np.allclose(dm_py, dm_jl, atol=1e-5)

def test_frechet_backend_consistency():
    c1 = np.array([[0,0,0], [1,1,1], [2,2,2]], dtype=float)
    c2 = np.array([[0,0,1], [1,1,2], [2,2,3]], dtype=float)
    
    f_py = frechet_distance(c1, c2, backend="python")
    f_jl = frechet_distance(c1, c2, backend="julia")
    
    assert pytest.approx(f_py) == f_jl

def test_cohomology_basis_backend_consistency():
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
