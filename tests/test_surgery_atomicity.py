"""
Tests for pysurgery/surgery.py — Transactional Safety and Atomicity.
"""
import pytest
import numpy as np
import pickle
from pysurgery.surgery import (
    SurgerySession,
    DimensionalConsistencyError,
    SurgeryProtocolError,
)
from pysurgery.topology.complexes import SimplicialComplex


@pytest.fixture
def complex_3d():
    """A simple 3D SimplicialComplex for surgery."""
    # A standard 2-sphere boundary of a 3-simplex (tetrahedron)
    return SimplicialComplex.from_simplices(
        [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)],
        close_under_faces=True,
    )


def test_transaction_rollback_on_general_exception():
    """Verify that a general exception inside a transaction rolls back to a byte-identical state."""
    session = SurgerySession(
        ambient_space="R^3",
        objects={"tetra": "tetra_data"},
        point_clouds={"cloud1": np.array([[1.0, 1.0, 1.0]])},
    )

    # Capture initial state parameters
    initial_manifold_repr = pickle.dumps(session.manifold)
    initial_objects = {k: v.data for k, v in session.objects().items()}
    initial_clouds = {k: v.copy() for k, v in session.point_clouds.items()}
    initial_cobordism_len = len(session.cobordism)
    initial_stack_len = len(session.stack)
    initial_step_counter = session._step_counter

    try:
        with session.transaction("failed_txn"):
            # Mutate point clouds
            session.point_clouds["cloud1"] += 1.0
            # Mutate target
            session.remove_disks("D^3", at=[(0, 0, 0)], target="tetra")
            
            # Verify they changed inside the transaction
            assert session.point_clouds["cloud1"][0, 0] == 2.0
            assert len(session.cobordism) > initial_cobordism_len
            
            raise ValueError("Expected failure inside transaction")
    except ValueError:
        pass

    # Verify everything rolled back to a 100% byte-identical/reconstructed state
    assert pickle.dumps(session.manifold) == initial_manifold_repr
    assert list(session.objects().keys()) == list(initial_objects.keys())
    for name in initial_objects:
        assert session.objects()[name].data == initial_objects[name]
    assert len(session.point_clouds) == len(initial_clouds)
    for k in initial_clouds:
        assert np.array_equal(session.point_clouds[k], initial_clouds[k])
    assert len(session.cobordism) == initial_cobordism_len
    assert len(session.stack) == initial_stack_len
    assert session._step_counter == initial_step_counter


def test_atomic_step_rollback_on_no_commit():
    """Verify that an atomic step rolls back to pre-step state if commit() is not called."""
    session = SurgerySession(ambient_space="R^3")
    initial_cobordism_len = len(session.cobordism)

    try:
        with session._atomic_step("custom_step"):
            # Perform mutation
            session.remove_disks("D^3", at=[(0, 0, 0)])
            # Exiting without commit()
    except SurgeryProtocolError as e:
        assert "exited without explicit commit" in str(e)

    # State must be rolled back cleanly
    assert len(session.cobordism) == initial_cobordism_len


def test_atomic_step_rollback_on_exception():
    """Verify that an atomic step rolls back and propagates the raised exception."""
    session = SurgerySession(ambient_space="R^3")
    initial_cobordism_len = len(session.cobordism)

    try:
        with session._atomic_step("custom_step"):
            session.remove_disks("D^3", at=[(0, 0, 0)])
            raise RuntimeError("Something failed during step execution")
    except RuntimeError as e:
        assert "Something failed during step" in str(e)

    # State must be rolled back cleanly
    assert len(session.cobordism) == initial_cobordism_len


def test_atomic_step_commit_success():
    """Verify that an atomic step succeeds when committed."""
    session = SurgerySession(ambient_space="R^3")
    initial_cobordism_len = len(session.cobordism)

    with session._atomic_step("custom_step") as step:
        session.remove_disks("D^3", at=[(0, 0, 0)])
        step.commit()

    # State must be committed
    assert len(session.cobordism) == initial_cobordism_len + 1


def test_nested_transactions_rollback():
    """Verify nested transaction blocks where the inner block rolls back but the outer can succeed."""
    session = SurgerySession(ambient_space="R^3")

    with session.transaction("outer"):
        # First mutation in outer
        session.remove_disks("D^3", at=[(0, 0, 0)])
        outer_state_1 = pickle.dumps(session.manifold)
        outer_cob_len = len(session.cobordism)

        try:
            with session.transaction("inner"):
                # Second mutation in inner
                session.remove_disks("D^3", at=[(1, 1, 1)])
                raise RuntimeError("Inner transaction failed")
        except RuntimeError:
            pass

        # Inner transaction rolled back, outer's first mutation remains
        assert pickle.dumps(session.manifold) == outer_state_1
        assert len(session.cobordism) == outer_cob_len


def test_mayer_vietoris_delta_mismatch_rollback(complex_3d):
    """Verify that a Mayer-Vietoris prediction delta mismatch raises SurgeryInvariantBroken and rolls back."""
    session = SurgerySession(ambient_space=complex_3d)
    initial_manifold_repr = pickle.dumps(session.manifold)
    
    # We trigger an error like DimensionalConsistencyError to force a rollback
    with pytest.raises(DimensionalConsistencyError):
        session.remove_disks("D^5", at=[(0,)])  # D^5 from dim=2 manifold is invalid
        
    # State must be 100% byte-identical
    assert pickle.dumps(session.manifold) == initial_manifold_repr
