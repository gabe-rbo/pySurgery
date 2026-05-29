import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from pysurgery.topology.complexes import SimplicialComplex
from pysurgery.surgery import SurgerySession
from pysurgery.auto_surgery import (
    detect_components_with_status,
    detect_nested_pairs,
    auto_separate_nested,
    NestReport,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def nested_spheres_3d():
    """A small S² nested inside a larger S² in a 3D ambient space."""
    # Standard cube boundary faces (2-sphere)
    faces_cube = [
        (0, 1, 2), (1, 2, 3), (4, 5, 6), (5, 6, 7), (0, 1, 5), (0, 5, 4),
        (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 3, 7), (1, 7, 5)
    ]
    
    # Shift outer sphere vertices by 10
    faces_outer = [tuple(v + 10 for v in face) for face in faces_cube]
    
    # Ambient 3-simplex (tetrahedron) representing ambient 3D space
    faces_ambient = [(20, 21, 22, 23)]
    
    # Combined complex K
    K = SimplicialComplex.from_simplices(
        faces_cube + faces_outer + faces_ambient, close_under_faces=True
    )
    
    # Coordinates mapping
    coords_global = np.zeros((24, 3))
    
    # Inner S²: small cube centered at origin
    coords_global[0:8] = np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    ])
    
    # Outer S²: larger cube centered at origin
    coords_global[10:18] = np.array([
        [-2.0, -2.0, -2.0], [2.0, -2.0, -2.0], [-2.0, 2.0, -2.0], [2.0, 2.0, -2.0],
        [-2.0, -2.0, 2.0], [2.0, -2.0, 2.0], [-2.0, 2.0, 2.0], [2.0, 2.0, 2.0]
    ])
    
    # Ambient tetrahedron vertices (bounding)
    coords_global[20:24] = np.array([
        [-10.0, -10.0, -10.0], [10.0, -10.0, -10.0], [0.0, 10.0, -10.0], [0.0, 0.0, 10.0]
    ])
    
    K._coordinates = coords_global
    
    # Re-extract Ka and Kb
    Ka = SimplicialComplex.from_simplices(faces_cube, close_under_faces=True)
    Ka._coordinates = coords_global
    Kb = SimplicialComplex.from_simplices(faces_outer, close_under_faces=True)
    Kb._coordinates = coords_global
    
    return K, Ka, Kb, coords_global


# ── Test 1: detect_nested_pairs ────────────────────────────────────────────────

def test_detect_nested_sphere_in_sphere(nested_spheres_3d):
    """Verify that detect_nested_pairs correctly identifies nested components."""
    K, Ka, Kb, coords = nested_spheres_3d
    
    # Detect components using detect_components_with_status
    components = detect_components_with_status(K, backend="python")
    
    # Verify we got 3 components: C0 (inner), C1 (outer), C2 (ambient tet)
    assert len(components) == 3
    
    # Get the component names/objects
    c_inner = next(c for c in components if 0 in c.vertex_ids)
    c_outer = next(c for c in components if 10 in c.vertex_ids)
    
    # Create point clouds dict
    point_clouds = {
        c_inner.name: coords[c_inner.vertex_ids],
        c_outer.name: coords[c_outer.vertex_ids],
    }
    
    # Run detect_nested_pairs
    nested = detect_nested_pairs(K, components, coords=point_clouds)
    
    # Verify that inner is nested inside outer
    assert len(nested) == 1
    pair = nested[0]
    assert pair.outer == c_outer.name
    assert pair.inner == c_inner.name
    assert pair.witness == "algebraic" or pair.witness == "winding"
    assert pair.exact is True


# ── Test 2: auto_separate_nested ──────────────────────────────────────────────

def test_auto_separate_nested_sphere_in_sphere(nested_spheres_3d):
    """Verify that auto_separate_nested separates nested spheres."""
    K, Ka, Kb, coords = nested_spheres_3d
    
    # Create SurgerySession
    # components C0 = Ka, C1 = Kb
    session = SurgerySession(
        ambient_space=K,
        objects={"inner": Ka, "outer": Kb},
        point_clouds={
            "inner": coords[sorted([v[0] for v in Ka.n_simplices(0)])],
            "outer": coords[sorted([v[0] for v in Kb.n_simplices(0)])]
        }
    )
    
    # Mock find_attachment_sphere to return a valid 1-arc (0-sphere attaching sphere)
    # in the complement of outer.
    # The endpoints are on outer, e.g. vertices 10 and 11.
    mock_arc = MagicMock()
    mock_arc.exact = True
    mock_arc.sphere_simplices = [(10,), (11,)]
    mock_arc.endpoints = [(10,), (11,)]
    mock_arc.framing = 0
    
    with patch("pysurgery.surgery.find_attachment_sphere", return_value=mock_arc):
        report = auto_separate_nested(session, "outer", "inner", backend="python")
        
    assert isinstance(report, NestReport)
    assert report.outer == "outer"
    assert report.inner == "inner"
    assert report.exact is True
    assert report.still_nested is False
