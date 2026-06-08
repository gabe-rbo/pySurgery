from pysurgery.topology.complexes import DynamicComplex, SimplicialComplex

def test_dynamic_complex_initialization():
    # Triangle
    sc = SimplicialComplex.from_simplices([(0, 1, 2)], close_under_faces=True)
    dc = DynamicComplex(simplices=sc.simplices_field)
    
    assert dc.betti_number(0) == 1
    assert dc.betti_number(1) == 0
    assert dc.betti_number(2) == 0

def test_dynamic_complex_add_simplex():
    # Start with two disjoint points
    dc = DynamicComplex(simplices={0: [(0,), (1,)]})
    assert dc.betti_number(0) == 2
    
    # Add an edge
    dc.add_simplex((0, 1))
    assert dc.betti_number(0) == 1
    assert dc.consistency_check()

def test_dynamic_complex_remove_simplex():
    # Start with a triangle
    dc = DynamicComplex(simplices={0: [(0,), (1,), (2,)], 1: [(0,1), (1,2), (2,0)], 2: [(0,1,2)]})
    assert dc.betti_number(0) == 1
    assert dc.betti_number(1) == 0
    
    # Remove the 2-simplex (the triangle interior)
    dc.remove_simplex((0, 1, 2))
    # Now it's a circle S^1
    assert dc.betti_number(1) == 1
    assert dc.consistency_check()

def test_dynamic_complex_skeletal_closure_removal():
    # Removing a vertex should remove all edges connected to it
    dc = DynamicComplex(simplices={0: [(0,), (1,)], 1: [(0,1)]})
    assert dc.count_simplices(1) == 1
    
    dc.remove_simplex((0,))
    assert dc.count_simplices(1) == 0
    assert dc.count_simplices(0) == 1
    assert dc.n_simplices(0) == [(1,)]


def test_new_simplicial_complex_methods():
    sc = SimplicialComplex.from_simplices([(0, 1, 2)], close_under_faces=True)
    
    # Test get_subfaces
    subfaces_all = sc.get_subfaces((0, 1, 2))
    assert (0,) in subfaces_all
    assert (0, 1) in subfaces_all
    assert (0, 1, 2) in subfaces_all
    
    subfaces_dim1 = sc.get_subfaces((0, 1, 2), dimension=1)
    assert subfaces_dim1 == {(0, 1), (1, 2), (0, 2)}
    
    # Test get_cofaces
    cofaces_all = sc.get_cofaces((0,))
    assert (0,) in cofaces_all
    assert (0, 1) in cofaces_all
    assert (0, 1, 2) in cofaces_all
    
    cofaces_dim2 = sc.get_cofaces((0,), dimension=2)
    assert cofaces_dim2 == {(0, 1, 2)}
    
    # Test to_dynamic_complex
    dc = sc.to_dynamic_complex()
    assert isinstance(dc, DynamicComplex)
    assert dc.betti_number(0) == 1
    assert dc.betti_number(1) == 0
