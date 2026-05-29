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
