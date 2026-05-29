import numpy as np
import pytest
from pysurgery.surgery import SurgerySession

def test_handle_slides_algebraic():
    session = SurgerySession(ambient_space="R^3")
    
    # Attach two 1-handles
    h1 = session.attach_handle(at=(0,0,0), handle_type="S^0xD^2", framing=1)
    h2 = session.attach_handle(at=(1,1,1), handle_type="S^0xD^2", framing=1)
    
    # Slide Handle 1 over Handle 2
    session.slide_handle(slider_id=h1.id, over_id=h2.id, sign=1)
    
    # Check basis change matrix
    # M_attach1 = [[1]]
    # M_attach2 = [[1,0],[0,1]]
    # M_slide = [[1,1],[0,1]]
    # Result = M_slide @ M_attach2 @ M_attach1_padded = [[1,1],[0,1]]
    bc = session._latest_change_of_basis_matrix()
    np.testing.assert_array_equal(bc, [[1, 1], [0, 1]])
    
    # Slide back with sign -1
    session.slide_handle(slider_id=h1.id, over_id=h2.id, sign=-1)
    # New M_slide = [[1, -1], [0, 1]]
    # Result = [[1, -1], [0, 1]] @ [[1, 1], [0, 1]] = Identity
    bc = session._latest_change_of_basis_matrix()
    np.testing.assert_array_equal(bc, np.eye(2, dtype=int))

def test_cancellation_algebraic():
    session = SurgerySession(ambient_space="R^3")
    
    # Attach a 1-handle (index 1)
    h1 = session.attach_handle(at=(0,0,0), handle_type="S^0xD^2")
    
    # Set up a boundary matrix for index 2
    # ∂(h2) = 1 * h1
    from scipy.sparse import csr_matrix
    session.chain_complex.chain_complex.boundaries[2] = csr_matrix(np.array([[1]], dtype=int))
    
    # Attach a 2-handle (index 2)
    h2 = session.attach_handle(at=(0,0,0), handle_type="S^1xD^1")
    
    # Candidate detection
    candidates = session.cancellation_candidates()
    assert len(candidates) == 1
    c = candidates[0]
    assert c.handle_lo.id == h1.id
    assert c.handle_hi.id == h2.id
    assert c.intersection_signed == 1
    assert c.is_geometric is True # Single 1 in row/col
    
    # Execute cancellation
    session.cancel_handles(h_lo_id=h1.id, h_hi_id=h2.id)
    
    # Verify handles removed
    assert len(session.handlebody_state.handles) == 0
    # Verify boundary matrix shrunk
    assert session.chain_complex.chain_complex.boundaries[2].shape == (0, 0)
    # Verify basis change cleared
    assert len(session._cob_basis_change) == 0

def test_cancellation_requirement_geometric():
    session = SurgerySession(ambient_space="R^3")
    h1_a = session.attach_handle(at=(0,0,0), handle_type="S^0xD^2")
    h1_b = session.attach_handle(at=(1,1,1), handle_type="S^0xD^2")
    
    # ∂(h2) = 1*h1_a + 1*h1_b
    from scipy.sparse import csr_matrix
    session.chain_complex.chain_complex.boundaries[2] = csr_matrix(np.array([[1], [1]], dtype=int))
    h2 = session.attach_handle(at=(0,0,0), handle_type="S^1xD^1")
    
    candidates = session.cancellation_candidates()
    # (h1_a, h2) and (h1_b, h2) are algebraic candidates
    assert len(candidates) == 2
    for c in candidates:
        assert c.is_geometric is False # h2 hits two 1-handles
        
    # Should fail with require_geometric=True (default)
    from pysurgery.core.exceptions import KirbyMoveError
    with pytest.raises(KirbyMoveError, match="must be geometric"):
        session.cancel_handles(h_lo_id=h1_a.id, h_hi_id=h2.id)
        
    # Should pass with require_geometric=False
    session.cancel_handles(h_lo_id=h1_a.id, h_hi_id=h2.id, require_geometric=False)
    assert len(session.handlebody_state.handles) == 1
    assert session.handlebody_state.handles[0].id == h1_b.id
