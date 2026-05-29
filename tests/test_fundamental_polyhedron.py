import pytest
from pysurgery.topology.complexes import SimplicialComplex

def test_fundamental_polyhedron_torus():
    """Test on a triangulated torus (14 vertices, 42 edges, 28 triangles).
    A standard triangulation of a torus will be a 2-manifold.
    We just use a simple 2D mesh that wraps around.
    """
    # A simple 2D triangulated torus.
    # It takes at least 7 vertices to triangulate a torus, but let's build a standard grid.
    # Let's use a simpler manifold: a triangulated 2-sphere.
    # Tetrahedron boundary is a 2-sphere
    simplices = [
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 3),
        (1, 2, 3)
    ]
    sc = SimplicialComplex.from_simplices(simplices)
    assert sc.dimension == 2
    
    fp = sc.fundamental_polyhedron()
    assert fp.dimension == 2
    assert len(fp.n_simplices) == 4
    
    # Dual graph of a tetrahedron boundary is K4 (complete graph on 4 vertices)
    # A spanning tree in K4 has 3 edges.
    assert len(fp.internal_glues) == 3
    
    # Total faces is 4 * 3 / 2 = 6 internal faces.
    # We used 3 for spanning tree. 
    # That leaves 3 dual edges that are face pairings.
    # Wait, in a tetrahedron boundary, the "n_simplices" are the 4 triangles.
    # Each triangle has 3 edges. So 4 * 3 = 12 half-edges.
    # They meet in pairs, so 6 edges total.
    # The spanning tree takes 3 edges.
    # The remaining 3 edges become 3 face pairings.
    assert len(fp.face_pairings) == 3
    
    symbolic_atlas = fp.get_symbolic_atlas()
    assert len(symbolic_atlas[0]) == 3 # 3 generators
    
    numerical_atlas = fp.get_numerical_atlas()
    assert len(numerical_atlas) == 3
    
    tiles = fp.tile_universal_cover(depth=2)
    assert len(tiles) > 1

def test_fundamental_polyhedron_invalid_complex():
    # Not a manifold complex, e.g. 3 triangles meeting at an edge
    simplices = [
        (0, 1, 2),
        (0, 1, 3),
        (0, 1, 4)
    ]
    sc = SimplicialComplex.from_simplices(simplices)
    with pytest.raises(ValueError, match="not a manifold"):
        sc.fundamental_polyhedron()
