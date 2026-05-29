from pysurgery.topology.complexes import SimplicialComplex
from pysurgery.topology.persistent_homology import compute_zigzag_persistence

def test_zigzag_persistence_basic():
    # Sequence of complexes
    sc1 = SimplicialComplex.from_simplices([(0, 1)], close_under_faces=True)
    sc2 = SimplicialComplex.from_simplices([(0, 1), (1, 2)], close_under_faces=True)
    sc3 = SimplicialComplex.from_simplices([(1, 2)], close_under_faces=True)
    
    seq = [sc1, sc2, sc3]
    res = compute_zigzag_persistence(seq, field='Z2')
    
    assert res.field == 'Z2'
    assert len(res.barcodes) >= 0

def test_zigzag_persistence_q():
    sc1 = SimplicialComplex.from_simplices([(0, 1, 2)], close_under_faces=True)
    sc2 = SimplicialComplex.from_simplices([(0, 1, 2), (2, 3)], close_under_faces=True)
    
    seq = [sc1, sc2]
    res = compute_zigzag_persistence(seq, field='Q')
    
    assert res.field == 'Q'
    assert len(res.barcodes) > 0