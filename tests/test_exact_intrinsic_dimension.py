
from pysurgery.core.complexes import SimplicialComplex
from pysurgery.core.intrinsic_dimension import exact_intrinsic_dimension

def test_s1_intrinsic_dimension():
    # S^1 as a triangle (simplest triangulation)
    # Vertices: 0, 1, 2. Edges: (0,1), (1,2), (0,2)
    s1 = SimplicialComplex.from_simplices([(0,1), (1,2), (0,2)])
    
    # Link of vertex 0 should be {1, 2} as two disjoint points (S^0)
    lk0 = s1.link((0,))
    assert lk0.dimension == 0
    assert len(lk0.n_simplices(0)) == 2
    
    # H_0(S^0) = Z + Z, so reduced H_0(S^0) = Z.
    rh = lk0.reduced_homology()
    assert rh[0] == (1, [])
    
    res = exact_intrinsic_dimension(s1)
    assert res.status == "success"
    assert res.global_dimension == 1.0
    assert res.exact is True

def test_s2_intrinsic_dimension():
    # S^2 as boundary of a tetrahedron
    # Simplices: (0,1,2), (0,1,3), (0,2,3), (1,2,3)
    s2 = SimplicialComplex.from_simplices([(0,1,2), (0,1,3), (0,2,3), (1,2,3)])
    
    # Link of vertex 0 should be the triangle (1,2,3), which is S^1
    lk0 = s2.link((0,))
    assert lk0.dimension == 1
    # Check reduced homology of S^1: H_1=Z, H_0=0
    rh = lk0.reduced_homology()
    assert rh[1] == (1, [])
    assert rh[0] == (0, [])
    
    res = exact_intrinsic_dimension(s2)
    assert res.status == "success"
    assert res.global_dimension == 2.0

def test_manifold_with_boundary():
    # A single triangle (2-disk)
    disk = SimplicialComplex.from_simplices([(0,1,2)])
    
    # All vertices are on the boundary. 
    # Link of 0 is the edge (1,2), which is a 1-disk.
    # Reduced homology of a disk is zero.
    lk0 = disk.link((0,))
    rh = lk0.reduced_homology()
    for k, val in rh.items():
        assert val == (0, [])
        
    res = exact_intrinsic_dimension(disk)
    # It should be detected as a 2-manifold (with boundary)
    assert res.status == "success"
    assert res.global_dimension == 2.0

def test_non_manifold_bowtie():
    # Two triangles meeting at vertex 0
    # (0,1,2) and (0,3,4)
    bowtie = SimplicialComplex.from_simplices([(0,1,2), (0,3,4)])
    
    # Link of 0 is two disjoint edges (1,2) and (3,4)
    # H_0(Lk(0)) = Z + Z (two components), so reduced H_0 = Z.
    # But for a 2-manifold, the link should be S^1 (reduced H_1=Z) or Disk (reduced H=0).
    # Here reduced H_0=1 indicates it's more like a 1-manifold locally, 
    # but the complex has dimension 2.
    
    res = exact_intrinsic_dimension(bowtie)
    assert res.status == "inconclusive"
    assert any("Detected manifold dimension" in d or "multiple non-zero" in d or "non-sphere" in d for d in res.diagnostics)

def test_0_manifold():
    # Points 0, 1, 2
    m0 = SimplicialComplex.from_simplices([(0,), (1,), (2,)])
    res = exact_intrinsic_dimension(m0)
    assert res.status == "success"
    assert res.global_dimension == 0.0
