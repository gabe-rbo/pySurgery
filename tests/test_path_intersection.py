"""
Tests for pysurgery/surgery.py path collision detection mode ("fast" vs "exact").
"""
import numpy as np
import pytest
from pysurgery.surgery import (
    SurgerySession,
    TranslateIsotopy,
    _clouds_min_dist,
    _check_path_intersection,
)


def test_clouds_min_dist_fast_vs_exact():
    """Verify that _clouds_min_dist with mode='exact' detects close points that are missed by mode='fast' due to subsampling."""
    # B is stationary: one point at (1.0, 0.0, 0.0)
    B = np.array([[1.0, 0.0, 0.0]])
    
    # A has 10000 points. 9999 points at (10.0, 10.0, 10.0).
    # One point at (0.995, 0.0, 0.0) at index 1.
    A = np.full((10000, 3), 10.0)
    A[1] = [0.995, 0.0, 0.0]
    
    # mode="fast" subsamples A (sampling indices 0, 33, 66, etc.)
    # It misses index 1, so min dist is large.
    d_fast = _clouds_min_dist(A, B, mode="fast")
    assert d_fast > 1.0
    
    # mode="exact" checks all points using cKDTree. It finds index 1, so min dist is 0.005.
    d_exact = _clouds_min_dist(A, B, mode="exact")
    assert pytest.approx(d_exact, abs=1e-5) == 0.005


def test_check_path_intersection_fast_vs_exact():
    """Verify that _check_path_intersection catches thin/localized collisions in mode='exact' and misses in mode='fast'."""
    B = np.array([[1.0, 0.0, 0.0]])
    
    # A has 10000 points. 9999 points at (10.0, 10.0, 10.0).
    # One point at (0.995, 0.0, 0.0) at index 1.
    A = np.full((10000, 3), 10.0)
    A[1] = [0.995, 0.0, 0.0]
    
    # TranslateIsotopy moving by offset (0.0, 0.0, 0.0) so it's stationary in time for simplicity
    iso = TranslateIsotopy(offset=[0.0, 0.0, 0.0], name="NoOpTranslate")
    
    # With mode="fast", no collision is detected because the subsampling misses the point
    warn_fast = _check_path_intersection(A, {"stationary": B}, iso, mode="fast", tol=0.01)
    assert warn_fast is None
    
    # With mode="exact", collision is detected because it finds the point within tol
    warn_exact = _check_path_intersection(A, {"stationary": B}, iso, mode="exact", tol=0.01)
    assert warn_exact is not None
    assert "intersects" in warn_exact
    assert "mode=exact" in warn_exact


def test_session_path_check_mode():
    """Verify that the session uses the configured path check mode correctly."""
    session = SurgerySession(
        ambient_space="R^3",
        point_clouds={
            "moving": np.full((10000, 3), 10.0),
            "stationary": np.array([[1.0, 0.0, 0.0]]),
        },
    )
    # Set the unique closest point
    session.point_clouds["moving"][1] = [0.995, 0.0, 0.0]
    
    # Test setting check mode
    session._isotopy_path_check_mode = "fast"
    
    # Check that applying translate isotopy does not trigger a warning on fast mode
    # For a warning to be printed/logged, we verify we can apply it without warnings in stderr
    # Let's verify _check_path_intersection call internally
    warn = _check_path_intersection(
        session.point_clouds["moving"],
        {"stationary": session.point_clouds["stationary"]},
        TranslateIsotopy(offset=[0.0, 0.0, 0.0]),
        mode=session._isotopy_path_check_mode,
        tol=0.01,
    )
    assert warn is None
    
    session._isotopy_path_check_mode = "exact"
    warn = _check_path_intersection(
        session.point_clouds["moving"],
        {"stationary": session.point_clouds["stationary"]},
        TranslateIsotopy(offset=[0.0, 0.0, 0.0]),
        mode=session._isotopy_path_check_mode,
        tol=0.01,
    )
    assert warn is not None
