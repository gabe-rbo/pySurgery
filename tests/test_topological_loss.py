import math
from pysurgery.topology.persistent_homology import compute_topological_loss, Barcode

def test_topological_loss_basic():
    # Barcodes
    b1 = [Barcode(birth=0, death=1, dim=0), Barcode(birth=1, death=2, dim=0)]
    b2 = [Barcode(birth=0, death=2, dim=0), Barcode(birth=1, death=3, dim=0)]
    
    loss = compute_topological_loss(b1, b2)
    assert float(loss) >= 0.0
    
def test_topological_loss_empty():
    loss = compute_topological_loss([], [])
    assert math.isclose(float(loss), 0.0)
    
    loss2 = compute_topological_loss([Barcode(birth=0, death=1, dim=0)], [])
    assert float(loss2) > 0.0