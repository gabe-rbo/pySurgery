from pysurgery.surgery import SurgerySession, CobordismComplex
from pysurgery.topology.complexes import SimplicialComplex

def test_cobordism_complex_initialization():
    # Symbolic initial M0
    session = SurgerySession(ambient_space="R^3")
    assert isinstance(session.W, CobordismComplex)
    assert len(session.W.boundary_initial_indices) == 1
    assert session.W.is_collared is True

    # Simplicial initial M0
    m0 = SimplicialComplex.from_simplices([[0, 1, 2]])
    session2 = SurgerySession(ambient_space=m0)
    assert len(session2.W.boundary_initial_indices) == 3
    assert session2.W.euler_characteristic() == 1

def test_cobordism_slab_tracking():
    session = SurgerySession(ambient_space="R^3", point_clouds={"T1": [[0,0,0]]})
    
    # 1. Isotopy
    session.move(offset=(1,0,0), target="T1")
    assert 1 in session.W.slab_for_step
    
    # 2. Disk removal
    session.remove_disks(types=["D^3"], at=[(0,0,0)])
    assert 2 in session.W.slab_for_step
    
    # 3. Handle attachment
    session.attach_handle(at=(0,0,0), handle_type="S^1xD^2")
    assert 3 in session.W.slab_for_step
    assert session.W.is_collared is True # Builders return empty/collared for now

def test_cobordism_logs():
    session = SurgerySession(ambient_space="R^3")
    session.move(offset=(1,0,0))
    
    logs = session.logs()
    assert "V. COBORDISM COMPLEX W" in logs
    assert "Slabs (one per step):    1" in logs


def test_session_initial_W_is_cylinder():
    # Simplicial initial M0 (2D complex)
    m0 = SimplicialComplex.from_simplices([[0, 1, 2]])
    session = SurgerySession(ambient_space=m0)
    assert isinstance(session.W, CobordismComplex)
    
    # 0-simplices in initial boundary = 3 (0, 1, 2)
    # W.underlying should contain the cylinder over m0, which has dimension 3 (since m0 has dim 2)
    # The vertices of W.underlying are 0, 1, 2 at bottom, and 3, 4, 5 at top (since max_v = 2 => offset = 3)
    assert session.W.dimension() == 3
    assert set(session.W.boundary_initial_indices) == {0, 1, 2}
    assert set(session.W.boundary_final_indices) == {3, 4, 5}

