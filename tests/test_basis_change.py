import numpy as np
from pysurgery.surgery import SurgerySession, Framing

def test_basis_change_identities():
    session = SurgerySession(ambient_space="R^3", point_clouds={"T1": np.array([[0,0,0]])})
    
    # move records an identity matrix (0x0 since no handles yet)
    session.move(offset=(1, 0, 0), target="T1")
    bc = session._latest_change_of_basis_matrix()
    assert bc.shape == (0, 0)
    
    # remove_disks records an identity matrix (0x0)
    session.remove_disks(types=["D^3"], at=[(0,0,0)])
    bc = session._latest_change_of_basis_matrix()
    assert bc.shape == (0, 0)

def test_basis_change_with_handles():
    session = SurgerySession(ambient_space="R^3")
    
    # attach handle 1 (integer 1 framing -> diag(1))
    session.attach_handle(at=(0,0,0), handle_type="S^1xD^2", framing=Framing(kind="integer", integer=1))
    bc = session._latest_change_of_basis_matrix()
    assert bc.shape == (1, 1)
    assert bc[0, 0] == 1
    
    # attach handle 2 with framing -5 (should use -1 on diagonal)
    session.attach_handle(at=(1,1,1), handle_type="S^1xD^2", framing=Framing(kind="integer", integer=-5))
    bc = session._latest_change_of_basis_matrix()
    assert bc.shape == (2, 2)
    np.testing.assert_array_equal(bc, [[1, 0], [0, -1]])
    
    # move records 2x2 identity
    session.move(offset=(1,1,1), target="Handle1")
    bc = session._latest_change_of_basis_matrix()
    assert bc.shape == (2, 2)
    np.testing.assert_array_equal(bc, [[1, 0], [0, -1]])

def test_basis_change_slide_placeholder():
    # Since slide_handle is not implemented, I'll manually record a change 
    # to simulate it.
    session = SurgerySession(ambient_space="R^3")
    session.attach_handle(at=(0,0,0), handle_type="S^1xD^2") # 1x1 [[1]]
    session.attach_handle(at=(1,1,1), handle_type="S^1xD^2") # 2x2 [[1,0],[0,1]]
    
    # Simulate a slide: I + E_{0,1} = [[1, 1], [0, 1]]
    slide_matrix = np.array([[1, 1], [0, 1]], dtype=int)
    session._record_basis_change(slide_matrix)
    
    bc = session._latest_change_of_basis_matrix()
    # M_attach2 @ M_attach1 @ M_slide
    # [[1,0],[0,1]] @ [[1,0],[0,1]] @ [[1,1],[0,1]]
    np.testing.assert_array_equal(bc, slide_matrix)
    
    # Another slide: I + E_{2,1} = [[1, 0], [1, 1]]
    slide_matrix2 = np.array([[1, 0], [1, 1]], dtype=int)
    session._record_basis_change(slide_matrix2)
    
    # Product: [[1, 0], [1, 1]] @ [[1, 1], [0, 1]] = [[1, 1], [1, 2]]
    bc = session._latest_change_of_basis_matrix()
    np.testing.assert_array_equal(bc, [[1, 1], [1, 2]])
