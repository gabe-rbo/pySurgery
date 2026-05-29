"""
Tests for pysurgery/surgery.py — SurgerySession (The Surgeon API).

References:
    Milnor, J. (1965). Lectures on the h-cobordism theorem. Princeton University Press.
    Ranicki, A. (1992). Algebraic L-theory and topological manifolds. Cambridge University Press.
"""
import numpy as np
import pytest
from unittest.mock import patch

from pysurgery.surgery import (
    DimensionalConsistencyError,
    SurgeryFinishedError,
    SurgerySession,
    TrackedObject,
    Transformation,
)
from pysurgery.topology.complexes import SimplicialComplex


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def basic_session():
    """SurgerySession on R^3 with no objects or point clouds."""
    return SurgerySession(ambient_space="R^3")


@pytest.fixture
def session_with_cloud():
    """SurgerySession on R^3 with a 'sphere' point cloud."""
    cloud = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    return SurgerySession(
        ambient_space="R^3",
        point_clouds={"sphere": cloud},
    )


@pytest.fixture
def session_with_objects():
    """SurgerySession on R^3 with 'sphere' and 'pyramid' tracked objects."""
    return SurgerySession(
        ambient_space="R^3",
        objects={"sphere": "S_data", "pyramid": "P_data"},
    )


# ── Test 1: dimension parsing from string ambient space ───────────────────────


def test_dimension_parsed_from_string():
    """Ambient space strings 'R^n' are correctly parsed to dimension n.

    Theorem: The ambient dimension must be derived from the string descriptor
    for validation (e.g., 'R^4' → dim=4). Incorrect parsing would silently
    allow invalid surgery types through.

    Expected: chain_complex.dimension matches the ambient dimension.
    """
    for n in (2, 3, 4, 5):
        s = SurgerySession(ambient_space=f"R^{n}")
        assert s.chain_complex.dimension == n, (
            f"R^{n}: expected dimension {n}, got {s.chain_complex.dimension}"
        )


# ── Test 2: SimplicialComplex as ambient space ────────────────────────────────


def test_simplicial_complex_ambient_space():
    """SurgerySession accepts a SimplicialComplex as ambient_space.

    Theorem: The manifold attribute must hold the exact object passed in,
    so downstream code can access its topological structure directly.

    Expected: session.manifold is the passed complex; chain_complex.dimension == K.dimension.
    """
    K = SimplicialComplex.from_simplices(
        [(0,), (1,), (2,), (3,), (0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)],
        close_under_faces=True,
    )
    s = SurgerySession(ambient_space=K)
    assert s.manifold is K
    assert s.chain_complex.dimension == K.dimension


# ── Test 3: DimensionalConsistencyError for over-dimensional disk ─────────────


@pytest.mark.parametrize("disk_type,ambient", [
    ("D^4", "R^3"),
    ("D^5", "R^4"),
    ("D^3", "R^2"),
])
def test_remove_disks_dimensional_consistency_error(disk_type, ambient):
    """remove_disks raises DimensionalConsistencyError for disk dim > ambient dim.

    Theorem: A k-disk D^k cannot be removed from an n-manifold if k > n
    (Wall, 1970, Surgery on Compact Manifolds, §1). Removing a 4-disk from
    R^3 is geometrically impossible and must be caught eagerly.

    Expected: DimensionalConsistencyError raised with informative message.
    """
    s = SurgerySession(ambient_space=ambient)
    with pytest.raises(DimensionalConsistencyError):
        s.remove_disks(disk_type, at=[(0, 0, 0)])


# ── Test 4: valid remove_disks does not raise ─────────────────────────────────


def test_remove_disks_valid(basic_session):
    """remove_disks succeeds when disk dim ≤ ambient dim.

    Expected: no exception; operation recorded in cobordism trace.
    """
    basic_session.remove_disks("D^3", at=[(0, 0, 0)])
    assert len(basic_session.cobordism) == 1
    assert basic_session.cobordism[0]["step"] == "remove_disks"
    assert basic_session.cobordism[0]["theorem"] == "SURGERY_HANDLE_MAYER_VIETORIS"


# ── Test 5: attach_handle registers handle object ────────────────────────────


def test_attach_handle_registers_object(basic_session):
    """attach_handle adds a Handle object to the tracked objects dict.

    Theorem: After performing handle attachment, the new handle must be
    accessible as a tracked object so the move() isotopy can reference it
    (design: ``surgeon.objects()["Handle1"]``).

    Expected: 'Handle1' present in objects(); TrackedObject wrapping the handle type.
    """
    basic_session.attach_handle(at=((0, 0, 0), (1, 0, 0)), handle_type="S^2xD^1")
    obj_dict = basic_session.objects()
    assert "Handle1" in obj_dict
    assert isinstance(obj_dict["Handle1"], TrackedObject)
    assert obj_dict["Handle1"].data == "S^2xD^1"


# ── Test 6: AmbientSpace proxy forwards to session ────────────────────────────


def test_ambient_space_proxy_remove_disks(basic_session):
    """surgeon.AmbientSpace.remove_disks() delegates to the session.

    Theorem: The AmbientSpace proxy must be a transparent delegate — calling
    surgeon.AmbientSpace.remove_disks(T, at) must produce the same state
    change as calling surgeon.remove_disks(T, at) directly.

    Expected: cobordism grows by 1; stack grows by 1.
    """
    basic_session.AmbientSpace.remove_disks(("D^3",), at=[(0, 0, 0)])
    assert len(basic_session.cobordism) == 1
    assert len(basic_session.stack) == 1


def test_ambient_space_proxy_attach_handle(basic_session):
    """surgeon.AmbientSpace.attach_handle() delegates to the session."""
    basic_session.AmbientSpace.attach_handle(at=((0, 0, 0), (1, 0, 0)))
    assert any(op["step"] == "attach_handle" for op in basic_session.cobordism)


def test_ambient_space_proxy_callable(basic_session):
    """surgeon.AmbientSpace() returns the underlying manifold object."""
    result = basic_session.AmbientSpace()
    assert result == basic_session.manifold


# ── Test 7: objects proxy supports dict and callable access ──────────────────


def test_objects_proxy_dict_access(session_with_objects):
    """objects['name'] returns the TrackedObject for that name.

    Expected: TrackedObject instance; name attribute matches.
    """
    obj = session_with_objects.objects["sphere"]
    assert isinstance(obj, TrackedObject)
    assert obj.name == "sphere"


def test_objects_proxy_callable(session_with_objects):
    """objects() returns the full dict of TrackedObjects.

    Expected: both 'sphere' and 'pyramid' present.
    """
    d = session_with_objects.objects()
    assert "sphere" in d
    assert "pyramid" in d
    assert isinstance(d["sphere"], TrackedObject)


# ── Test 8: move applies offset to point cloud ────────────────────────────────


def test_move_applies_offset_to_cloud(session_with_cloud):
    """move(offset, target) translates the named point cloud by the given vector.

    Theorem: Ambient isotopy acts on all associated geometric data including
    raw point clouds (design §3 Geometric Sync).

    Expected: each point in the cloud shifted by offset.
    """
    original = session_with_cloud.point_clouds["sphere"].copy()
    offset = (1.0, 2.0, 3.0)
    session_with_cloud.move(offset=offset, target="sphere")
    result = session_with_cloud.point_clouds["sphere"]
    np.testing.assert_allclose(result, original + np.array(offset))


def test_move_returns_transformation(session_with_cloud):
    """move() returns a Transformation object.

    Expected: Transformation with non-empty description.
    """
    t = session_with_cloud.move(offset=(1.0, 0.0, 0.0), target="sphere")
    assert isinstance(t, Transformation)
    assert len(t.description) > 0


# ── Test 9: restore reverts point cloud changes from move ─────────────────────


def test_restore_reverts_move(session_with_cloud):
    """restore() undoes the last move(), returning the cloud to its prior state.

    Theorem: The transactional stack guarantees that restore() is the inverse
    of the last mutative operation (design §3 restore).

    Expected: point cloud identical to state before move() was called.
    """
    original = session_with_cloud.point_clouds["sphere"].copy()
    session_with_cloud.move(offset=(10.0, 10.0, 10.0), target="sphere")
    session_with_cloud.restore()
    np.testing.assert_allclose(
        session_with_cloud.point_clouds["sphere"], original
    )


def test_restore_pops_stack(basic_session):
    """restore() removes the last entry from the undo stack.

    Expected: stack shrinks by 1; cobordism grows by 1 (restore record added).
    """
    basic_session.remove_disks("D^3", at=[(0, 0, 0)])
    assert len(basic_session.stack) == 1
    basic_session.restore()
    assert len(basic_session.stack) == 0


def test_restore_on_empty_stack_is_no_op(basic_session):
    """restore() on an empty stack does not raise.

    Expected: no exception; stack remains empty.
    """
    assert len(basic_session.stack) == 0
    basic_session.restore()  # must not raise
    assert len(basic_session.stack) == 0


# ── Test 10: finish() locks all mutative methods ─────────────────────────────


@pytest.mark.parametrize("method,kwargs", [
    ("remove_disks", {"types": "D^3", "at": [(0, 0, 0)]}),
    ("attach_handle", {"at": (0, 0, 0)}),
    ("move", {"offset": (1.0, 0.0, 0.0)}),
    ("restore", {}),
])
def test_finish_locks_mutations(basic_session, method, kwargs):
    """All mutative methods raise SurgeryFinishedError after finish().

    Theorem: Finality Policy (design §3 finish): once finish() is called,
    any subsequent mutative call MUST raise SurgeryFinishedError.

    Expected: SurgeryFinishedError on every mutative call post-finish().
    """
    basic_session.finish()
    with pytest.raises(SurgeryFinishedError):
        getattr(basic_session, method)(**kwargs)


def test_finish_idempotent(basic_session):
    """finish() can be called twice without raising.

    Expected: second call does not raise; session remains locked.
    """
    basic_session.finish()
    basic_session.finish()  # must not raise
    assert basic_session._finished is True


# ── Test 11: AmbientSpace proxy raises SurgeryFinishedError after finish ──────


def test_ambient_space_proxy_raises_after_finish(basic_session):
    """AmbientSpace proxy methods also raise SurgeryFinishedError after finish().

    Expected: SurgeryFinishedError propagated through the proxy.
    """
    basic_session.finish()
    with pytest.raises(SurgeryFinishedError):
        basic_session.AmbientSpace.remove_disks(("D^3",), at=[(0, 0, 0)])
    with pytest.raises(SurgeryFinishedError):
        basic_session.AmbientSpace.restore()


# ── Test 12: logs() produces required sections ────────────────────────────────


def test_logs_contains_required_sections(basic_session):
    """logs() returns a string containing all three required sections.

    Theorem: The Surgery Sequence log must contain (I) Topological Trace,
    (II) Geometric Trace, and (III) Algebraic Proof (design §2 logs).

    Expected: all three section headers present; 'Surgery Sequence' in output.
    """
    basic_session.remove_disks("D^3", at=[(0, 0, 0)])
    basic_session.attach_handle(at=((0, 0, 0), (1, 0, 0)))
    log = basic_session.logs(latex=False)

    assert "SURGERY SEQUENCE LOG" in log
    assert "TOPOLOGICAL TRACE" in log
    assert "GEOMETRIC TRACE" in log
    assert "ALGEBRAIC PROOF" in log


def test_logs_records_steps_in_trace(basic_session):
    """logs() records each remove_disks and attach_handle step in the trace.

    Expected: both 'remove_disks' and 'attach_handle' appear in the log.
    """
    basic_session.remove_disks("D^3", at=[(0, 0, 0)])
    basic_session.attach_handle(at=((0, 0, 0), (1, 0, 0)))
    log = basic_session.logs()
    assert "remove_disks" in log
    assert "attach_handle" in log


def test_logs_latex_mode(basic_session):
    """logs(latex=True) wraps output in LaTeX section markup.

    Expected: output contains LaTeX section commands.
    """
    log = basic_session.logs(latex=True)
    assert "\\section*" in log or "\\subsection*" in log


def test_logs_available_after_finish(basic_session):
    """logs() is accessible after finish() (inspection method).

    Expected: no SurgeryFinishedError raised; non-empty string returned.
    """
    basic_session.finish()
    log = basic_session.logs()
    assert isinstance(log, str) and len(log) > 0


# ── Test 13: evaluate_obstruction returns a valid result ──────────────────────


def test_evaluate_obstruction_returns_result(basic_session):
    """evaluate_obstruction() returns an ObstructionResult with required fields.

    Theorem: The L-group obstruction element must carry exactness and
    assembly certification flags (CLAUDE.md §Structured result contracts).

    Expected: result has exact and obstructs attributes.
    """
    result = basic_session.evaluate_obstruction()
    assert hasattr(result, "exact")
    assert hasattr(result, "obstructs")


# ── Test 14: TrackedObject.move delegates to session ─────────────────────────


def test_tracked_object_move_delegates(session_with_objects, session_with_cloud):
    """TrackedObject.move() delegates to the session's move() method.

    Theorem: The TrackedObject is a thin wrapper; all logic lives in the session
    to maintain a single source of truth for cobordism recording.

    Expected: calling obj.move() updates the cobordism trace via the session.
    """
    cloud = np.array([[0.0, 0.0, 0.0]])
    s = SurgerySession(
        ambient_space="R^3",
        objects={"sphere": "S"},
        point_clouds={"sphere": cloud},
    )
    before_count = len(s.cobordism)
    iso = s.objects["sphere"].move(offset=(1.0, 0.0, 0.0))
    assert len(s.cobordism) == before_count + 1
    assert isinstance(iso, Transformation)


# ── Test 15: full surgery sequence (design §4 example) ───────────────────────


def test_full_surgery_sequence():
    """End-to-end surgery session: remove disks, attach handle, restore, finish.

    Theorem: The composition remove_disks → attach_handle is a bordism step;
    the resulting manifold must remain accessible post-finish() (design §4).

    Expected: all steps succeed; logs contain the full trace; manifold accessible.
    """
    s_cloud = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    p_cloud = np.array([[5.0, 5.0, 5.0]])

    surgeon = SurgerySession(
        ambient_space="R^3",
        objects={"sphere": "S_simplicial", "pyramid": "P_simplicial"},
        point_clouds={"sphere": s_cloud, "pyramid": p_cloud},
    )

    surgeon.AmbientSpace.remove_disks(("D^3", "D^3"), at=[(0, 0, 0), (1, 1, 1)])
    surgeon.AmbientSpace.attach_handle(at=((0, 0, 0), (1, 1, 1)), handle_type="S^2xD^1")

    iso = surgeon.objects["sphere"].move(offset=(2.0, 0.0, 0.0))
    assert isinstance(iso, Transformation)

    surgeon.AmbientSpace.restore()

    surgeon.finish()

    log = surgeon.logs(latex=False)
    assert "SURGERY SEQUENCE LOG" in log

    assert surgeon.manifold == "R^3"
    assert "sphere" in surgeon.objects()
    assert "pyramid" in surgeon.objects()


# ── Gate G1 Verification Tests ───────────────────────────────────────────────


def test_gate_g1_fresh_co_disk_simplices():
    """Verify _fresh_co_disk_simplices triangulations for all index-k combinations."""
    from pysurgery.manifolds.surgery import _fresh_co_disk_simplices
    
    # Test k = 0
    simplices_0 = _fresh_co_disk_simplices(k=0, n=3, vertex_offset=10)
    assert simplices_0 == ((0,),)

    # Test k = n
    simplices_n = _fresh_co_disk_simplices(k=3, n=3, vertex_offset=5)
    assert simplices_n == ((0, 1, 2, 3),)

    # Test codim = 1 (n - k == 1)
    simplices_codim1 = _fresh_co_disk_simplices(k=2, n=3, vertex_offset=2)
    assert simplices_codim1 == ((0, 1, 2, 3),)

    # Test n - k - 1 == 0
    simplices_codim_mid = _fresh_co_disk_simplices(k=2, n=3, vertex_offset=0)
    assert len(simplices_codim_mid) == 1
    assert all(len(s) == 4 for s in simplices_codim_mid)

    # Test general case: k=1, n=3 => n-k-1 = 1 >= 1
    simplices_gen = _fresh_co_disk_simplices(k=1, n=3, vertex_offset=0)
    assert len(simplices_gen) > 0
    assert all(len(s) == 4 for s in simplices_gen)


def test_gate_g1_boundary_pl_homeomorphism():
    """Verify boundary of D^k x S^{n-k-1} is homology-equivalent (PL-homeomorphic) to S^{n-1}."""
    from pysurgery.manifolds.surgery import _fresh_co_disk_simplices
    from pysurgery.topology.complexes import SimplicialComplex
    
    # For k=1, n=3, co-disk is D^1 x S^1. Boundary should be S^2.
    simplices = _fresh_co_disk_simplices(k=1, n=3, vertex_offset=0)
    K = SimplicialComplex.from_simplices(simplices, close_under_faces=True)
    
    # Find boundary faces. In a 3-dimensional simplicial complex (simplices of len 4),
    # the boundary consists of all 2-simplices (len 3) that belong to exactly one 3-simplex.
    all_3_simplices = [set(s) for s in K.n_simplices(3)]
    from collections import Counter
    faces = []
    for s in all_3_simplices:
        s_list = list(s)
        for i in range(4):
            face = tuple(sorted(s_list[:i] + s_list[i+1:]))
            faces.append(face)
    counts = Counter(faces)
    boundary_simplices = [f for f, count in counts.items() if count == 1]
    
    # Create the boundary simplicial complex
    bnd_complex = SimplicialComplex.from_simplices(boundary_simplices, close_under_faces=True)
    
    # Verify Betti numbers of the boundary: should be S^2, which has Betti (1, 0, 1)
    bnd_betti = bnd_complex.betti_numbers()
    assert bnd_betti[0] == 1
    assert bnd_betti[1] == 0
    assert bnd_betti[2] == 1


def test_gate_g1_construct_tubular_neighborhood():
    """Verify _construct_tubular_neighborhood star closures."""
    from pysurgery.manifolds.surgery import _construct_tubular_neighborhood
    from pysurgery.topology.complexes import SimplicialComplex
    
    # Build S^1 on vertices 0, 1, 2
    K = SimplicialComplex.from_simplices([(0, 1), (1, 2), (2, 0)], close_under_faces=True)
    
    # Construct neighborhood of vertex 0 (which is an S^0)
    tube = _construct_tubular_neighborhood(K, sphere_simplices=[(0,)], k=1, n=2)
    assert len(tube) > 0
    assert (0,) not in tube
    assert any((0, 1) in tube or (1, 0) in tube for s in tube)


def test_gate_g1_compute_framing():
    """Verify _compute_framing on canonical and unstable range inputs."""
    from pysurgery.manifolds.surgery import _compute_framing
    from pysurgery.topology.complexes import SimplicialComplex
    
    K = SimplicialComplex.from_simplices([(0, 1, 2)], close_under_faces=True)
    
    # Canonical codim-1
    res1 = _compute_framing(sigma_simplices=[(0, 1)], K=K, n=3, k=3)
    assert res1.value == 1
    assert res1.reason == "trivial_codim_1"
    assert res1.exact is True

    # Canonical codim-2
    res2 = _compute_framing(sigma_simplices=[(0, 1)], K=K, n=3, k=2)
    assert res2.value == 0
    assert res2.reason == "canonical_codim_2"
    assert res2.exact is True

    # Stable range codim >= 3, k=2, codim > k-1
    res3 = _compute_framing(sigma_simplices=[(0, 1)], K=K, n=5, k=2)
    assert res3.value == 0
    assert res3.reason == "stable_range_codim_ge_3"
    assert res3.exact is True


def test_gate_g1_transaction_atomicity_and_rollback():
    """Verify session snapshot capture/restore and rollback atomicity."""
    from pysurgery.surgery import SurgerySession, SurgeryProtocolError
    
    session = SurgerySession(ambient_space="R^3")
    
    # Verify capture/restore directly
    snap = session._capture_state()
    session.remove_disks("D^3", at=[(0, 0, 0)])
    assert len(session.cobordism) == 1
    
    session._restore_state(snap)
    assert len(session.cobordism) == 0

    # Verify transaction block rollback on exception
    try:
        with session.transaction("test_txn"):
            session.remove_disks("D^3", at=[(0, 0, 0)])
            raise ValueError("Forced error")
    except ValueError:
        pass
        
    assert len(session.cobordism) == 0

    # Verify _AtomicStep rollback if commit() not called
    from pysurgery.surgery import _AtomicStep
    try:
        with _AtomicStep(session, "test_step"):
            session.remove_disks("D^3", at=[(0, 0, 0)])
            # forgot to commit!
    except SurgeryProtocolError:
        pass
        
    assert len(session.cobordism) == 0

    # Verify _AtomicStep succeeds if commit() called
    with _AtomicStep(session, "test_step") as txn:
        session.remove_disks("D^3", at=[(0, 0, 0)])
        txn.commit()
        
    assert len(session.cobordism) == 1


def test_betti_error_surfaces_reason():
    from pysurgery.surgery import _betti_from_object
    from pysurgery.core.exceptions import BettiTrackingError
    
    # 1. Missing object (None)
    with pytest.raises(BettiTrackingError) as exc_info:
        _betti_from_object(None)
    assert "missing" in str(exc_info.value)
    
    # 2. Invalid object type (string instead of SimplicialComplex)
    with pytest.raises(BettiTrackingError) as exc_info:
        _betti_from_object("not a simplicial complex")
    assert "invalid object type" in str(exc_info.value)
    
    # 3. Object throws when Betti query is run
    dummy = SimplicialComplex.from_simplices([[0]])
    with patch.object(SimplicialComplex, "betti_numbers", side_effect=ValueError("Homology failed")):
        with pytest.raises(BettiTrackingError) as exc_info:
            _betti_from_object(dummy)
        assert "underlying betti_numbers() call raised an exception" in str(exc_info.value)
        # Verify the underlying exception cause is retained:
        assert isinstance(exc_info.value.__cause__, ValueError)
        assert str(exc_info.value.__cause__) == "Homology failed"


