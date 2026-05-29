from pysurgery.topology.complexes import SimplicialComplex
from pysurgery.topology.persistent_homology import compute_barcodes_exact, Barcode

def test_persistence_wrapper_basic():
    # A simple triangle (dim 2)
    sc = SimplicialComplex.from_simplices([(0, 1, 2)], close_under_faces=True)
    
    res = compute_barcodes_exact(sc, dimension=2, field='Z2')
    
    assert res.field == 'Z2'
    assert len(res.barcodes) > 0
    
    for b in res.barcodes:
        assert isinstance(b, Barcode)
        assert b.birth >= 0
        assert b.death >= 0

def test_persistence_wrapper_q():
    sc = SimplicialComplex.from_simplices([(0, 1, 2)], close_under_faces=True)
    res = compute_barcodes_exact(sc, dimension=2, field='Q')
    assert res.field == 'Q'
    assert len(res.barcodes) > 0

def test_persistence_wrapper_tetrahedron():
    sc = SimplicialComplex.from_simplices([(0, 1, 2, 3)], close_under_faces=True)
    res = compute_barcodes_exact(sc, dimension=3, field='Z2')
    
    assert res.field == 'Z2'
    assert len(res.barcodes) > 0
