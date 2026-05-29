import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from pysurgery.topology.complexes import SimplicialComplex
from pysurgery.surgery import SurgerySession
from pysurgery.auto_surgery import (
    auto_unlink_pair,
    CutSite,
    UnlinkReport,
)

# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def two_disjoint_circles():
    """Two circles (S¹) in a 3D ambient space.
    K_a: vertices 0–3 (square loop).
    K_b: vertices 4–7 (loop using triangles).
    """
    Ka_simps = [(0, 1), (1, 2), (2, 3), (3, 0)]
    Kb_simps = [(4, 5), (5, 6), (6, 7), (7, 4)]
    Kb_filling = [(4, 5, 6), (4, 6, 7)]
    K_ambient = [(10, 11, 12, 13)]

    K = SimplicialComplex.from_simplices(
        Ka_simps + Kb_simps + Kb_filling + K_ambient, close_under_faces=True
    )
    Ka = SimplicialComplex.from_simplices(Ka_simps, close_under_faces=True)
    Kb = SimplicialComplex.from_simplices(Kb_simps, close_under_faces=True)
    return K, Ka, Kb


# ── Test 1: auto_unlink_pair already unlinked (early exit) ────────────────────

def test_auto_unlink_already_unlinked(two_disjoint_circles):
    """Verify that auto_unlink_pair terminates early if linking is already 0."""
    K, Ka, Kb = two_disjoint_circles
    
    # Initialize coordinates
    coords_a = np.zeros((4, 3))
    coords_b = np.zeros((4, 3))
    
    session = SurgerySession(
        ambient_space=K,
        objects={"a": Ka, "b": Kb},
        point_clouds={"a": coords_a, "b": coords_b},
    )
    
    report = auto_unlink_pair(session, "a", "b", backend="python")
    
    assert isinstance(report, UnlinkReport)
    assert report.final_linking == 0
    assert report.passes == []
    assert report.topology_preserved is True
    assert report.exact is True


# ── Test 2: auto_unlink_pair atomic rollback on topology break ───────────────

def test_auto_unlink_pair_atomic_on_topology_break(two_disjoint_circles):
    """Verify that a topological mismatch raises TopologyNotRestoredError and rolls back state."""
    K, Ka, Kb = two_disjoint_circles
    
    coords_a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    coords_b = np.array([[5.0, 5.0, 5.0], [6.0, 5.0, 5.0], [6.0, 6.0, 5.0], [5.0, 6.0, 5.0]])
    
    session = SurgerySession(
        ambient_space=K,
        objects={"a": Ka, "b": Kb},
        point_clouds={"a": coords_a, "b": coords_b},
    )
    
    # Snapshot starting state
    pre_manifold_simplices = set(session.manifold.simplices)
    pre_a_simplices = set(session.objects["a"].data.simplices)
    
    # Mocking compute_linking_number to return 1 on the first call, then 0
    mock_lk = MagicMock()
    mock_lk.value = 1
    
    mock_cut = CutSite(
        simplex=(0, 1),
        centroid=(0.5, 0.0, 0.0),
        score=0.1,
        keeps_component_connected=True,
        local_strands=1,
    )
    
    # We will trigger a topology mismatch inside the transaction:
    # We mock _snapshot_topology so that post-topology doesn't match pre-topology
    mock_pre_snap = {"betti": {0: 1, 1: 1}, "pi1": "?", "is_mani": True, "is_closed": True}
    mock_post_snap = {"betti": {0: 1, 1: 0}, "pi1": "?", "is_mani": True, "is_closed": True} # mismatched betti
    
    with patch("pysurgery.manifolds.surgery.compute_linking_number", return_value=mock_lk), \
         patch("pysurgery.auto_surgery._find_cut_site", return_value=mock_cut), \
         patch("pysurgery.auto_surgery._snapshot_topology", side_effect=[mock_pre_snap, mock_post_snap]):
         
        report = auto_unlink_pair(session, "a", "b", backend="python")
        
    # Verify the report reflects the rolled-back pass
    assert len(report.passes) == 1
    pass_report = report.passes[0]
    assert pass_report.rolled_back is True
    assert pass_report.betti_match is False
    assert "TopologyNotRestoredError" in pass_report.error or "mismatch" in pass_report.error
    
    # Verify that the session state was rolled back to pre-transaction identical state
    assert set(session.manifold.simplices) == pre_manifold_simplices
    assert set(session.objects["a"].data.simplices) == pre_a_simplices
    assert len(session.cobordism) == 0


# ── Test 3: auto_unlink_pair legacy fallback on None cut site ─────────────────

def test_auto_unlink_legacy_delink_fallback(two_disjoint_circles):
    """Verify that auto_unlink_pair falls back to legacy_delink when no cut site can be found."""
    K, Ka, Kb = two_disjoint_circles
    
    coords_a = np.zeros((4, 3))
    coords_b = np.zeros((4, 3))
    
    session = SurgerySession(
        ambient_space=K,
        objects={"a": Ka, "b": Kb},
        point_clouds={"a": coords_a, "b": coords_b},
    )
    
    # Mocking linking number to return 1 so it tries to unlink,
    # and mocking _find_cut_site to return None to force fallback.
    mock_lk_1 = MagicMock()
    mock_lk_1.value = 1
    mock_lk_0 = MagicMock()
    mock_lk_0.value = 0
    
    with patch("pysurgery.manifolds.surgery.compute_linking_number", side_effect=[mock_lk_1, mock_lk_0, mock_lk_0, mock_lk_0]), \
         patch("pysurgery.auto_surgery._find_cut_site", return_value=None):
         
        report = auto_unlink_pair(session, "a", "b", mode="cancelling_pair", backend="python")
        
    # Verify report is generated with raw_handle_surgery mode (legacy fallback)
    assert report.mode == "raw_handle_surgery"
    assert report.topology_preserved is True
    # The actual unlinked number for disjoint circles is 0
    assert report.final_linking == 0


def test_surgery_delink_topology_preserving(two_disjoint_circles):
    """Verify that delink(..., topology_preserving=True) unlinks while preserving topology."""
    from pysurgery.manifolds.surgery import delink
    
    K, Ka, Kb = two_disjoint_circles
    
    # 1. With topology_preserving=True
    mock_cut = CutSite(
        simplex=(0, 1),
        centroid=(0.5, 0.0, 0.0),
        score=0.1,
        keeps_component_connected=True,
        local_strands=1,
    )
    
    mock_pre_snap = {"betti": {0: 1, 1: 1}, "pi1": "?", "is_mani": True, "is_closed": True}
    mock_post_snap = {"betti": {0: 1, 1: 1}, "pi1": "?", "is_mani": True, "is_closed": True}
    
    calls_1 = []
    def lk_mock_1(K_ambient, Ka_curr, Kb_curr, *args, **kwargs):
        if len(Ka_curr.n_simplices(0)) <= 2:
            return MagicMock(value=1)
        if not calls_1:
            calls_1.append(1)
            return MagicMock(value=1)
        return MagicMock(value=0)
        
    with patch("pysurgery.manifolds.surgery.compute_linking_number", side_effect=lk_mock_1), \
         patch("pysurgery.auto_surgery._find_cut_site", return_value=mock_cut), \
         patch("pysurgery.auto_surgery._snapshot_topology", side_effect=[mock_pre_snap, mock_post_snap]):
         
        res_pres = delink(K, Ka, Kb, max_surgeries=3, backend="python", topology_preserving=True)
        assert res_pres.final_linking == 0
        assert res_pres.exact is True
        
        # Verify Ka topology is restored (S1 Betti numbers: 1, 1)
        betti_a = res_pres.complex_a_after.betti_numbers(backend="python")
        assert betti_a == {0: 1, 1: 1}
    
    # 2. With topology_preserving=False (non-topology-preserving raw handle surgery)
    calls_2 = []
    def lk_mock_2(K_ambient, Ka_curr, Kb_curr, *args, **kwargs):
        if len(Ka_curr.n_simplices(0)) <= 2:
            return MagicMock(value=1)
        if not calls_2:
            calls_2.append(1)
            return MagicMock(value=1)
        return MagicMock(value=0)
        
    with patch("pysurgery.manifolds.surgery.compute_linking_number", side_effect=lk_mock_2):
        res_raw = delink(K, Ka, Kb, max_surgeries=3, backend="python", topology_preserving=False)
        assert res_raw.final_linking == 0
        
        # Ka topology is NOT preserved at the ambient level, but under this vertex-subset
        # restriction approximation, the tracked subcomplex still has Betti numbers {0: 1, 1: 1}
        betti_a_raw = res_raw.complex_a_after.betti_numbers(backend="python")
        assert betti_a_raw == {0: 1, 1: 1}

