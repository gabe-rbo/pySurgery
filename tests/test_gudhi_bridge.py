import pytest
from pysurgery.integrations.gudhi_bridge import signature_landscape

def test_signature_landscape():
    try:
        import gudhi
    except ImportError:
        pytest.skip("GUDHI not available")
        
    st = gudhi.SimplexTree()
    st.insert([0, 1, 2, 3, 4], 1.0)
    
    landscape = signature_landscape(st)
    assert len(landscape) == 1
    assert landscape[0][1] == 0
