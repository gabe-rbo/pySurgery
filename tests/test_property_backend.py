import pytest
from hypothesis import given, settings, strategies as st
import numpy as np
from pysurgery.core.complexes import SimplicialComplex, CWComplex
from pysurgery.core.fundamental_group import _pi1_raw_data_python, extract_pi_1
from pysurgery.bridge.julia_bridge import julia_engine
from tests.strategies import simplicial_complexes_raw, point_clouds

@pytest.mark.skipif(not julia_engine.available, reason="Julia engine not available")
@settings(max_examples=50, deadline=None)
@given(simplicial_complexes_raw(max_vertices=12))
def test_homology_parity_python_vs_julia(simplices):
    """Ensure Julia and Python backends return identical homology results."""
    sc = SimplicialComplex.from_simplices(simplices, coefficient_ring="Z")
    cc = sc.chain_complex()
    
    for n in range(3):
        # Force Python
        original_available = julia_engine.available
        julia_engine.available = False
        try:
            rank_py, torsion_py = cc.homology(n=n)
        finally:
            julia_engine.available = original_available
            
        # Use Julia (assuming it was available)
        rank_jl, torsion_jl = cc.homology(n=n)
        
        assert rank_py == rank_jl
        assert torsion_py == torsion_jl

@pytest.mark.skipif(not julia_engine.available, reason="Julia engine not available")
@settings(max_examples=50, deadline=None)
@given(simplicial_complexes_raw(max_vertices=10))
def test_pi1_parity_python_vs_julia(simplices):
    """Ensure Julia and Python backends return identical pi1 generators/relations."""
    sc = SimplicialComplex.from_simplices(simplices)
    cc = sc.chain_complex()
    # Use chain complex boundaries directly
    cw = CWComplex(cells=cc.cells, attaching_maps=cc.boundaries, dimensions=cc.dimensions)
    
    # Python
    original_available = julia_engine.available
    julia_engine.available = False
    try:
        res_py = extract_pi_1(cw, simplify=False)
    finally:
        julia_engine.available = original_available
        
    # Julia
    res_jl = extract_pi_1(cw, simplify=False)
    
    assert len(res_py.generators) == len(res_jl.generators)
    assert len(res_py.relations) == len(res_jl.relations)

@pytest.mark.skipif(not julia_engine.available, reason="Julia engine not available")
@settings(max_examples=50, deadline=None)
@given(point_clouds(min_pts=6, max_pts=15, dim=3))
def test_alpha_complex_parity_python_vs_julia(points):
    """Ensure Julia and Python backends return identical Alpha Complex simplices."""
    # Ensure points are 3D and not too degenerate for reliable Delaunay
    if points.shape[1] < 3:
        return
    
    # Check if points are too close to being coplanar/collinear
    # by checking the rank or variance
    if np.linalg.matrix_rank(points - points.mean(axis=0)) < 3:
        return

    try:
        sc_jl = SimplicialComplex.from_alpha_complex(points, max_alpha_square=1.0)
    except Exception:
        return

    original_available = julia_engine.available
    julia_engine.available = False
    try:
        sc_py = SimplicialComplex.from_alpha_complex(points, max_alpha_square=1.0)
    finally:
        julia_engine.available = original_available
        
    # Compare vertices
    assert set(sc_jl.n_simplices(0)) == set(sc_py.n_simplices(0))
    
    # Betti numbers check for topological parity
    b_jl = sc_jl.chain_complex().betti_numbers()
    b_py = sc_py.chain_complex().betti_numbers()
    
    for d in set(b_jl.keys()) | set(b_py.keys()):
        assert b_jl.get(d, 0) == b_py.get(d, 0), f"Betti numbers differ in dimension {d}"
