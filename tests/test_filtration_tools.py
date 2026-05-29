import numpy as np
from pysurgery.topology.filtration_tools import FiltrationReport
from pysurgery.topology.complexes import SimplicialComplex

def test_is_closed_manifold_circle():
    # A simple circle (3 points, though a bit small for manifold link check sometimes, let's use 6)
    theta = np.linspace(0, 2*np.pi, 6, endpoint=False)
    points = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    
    # Epsilon large enough to connect adjacent points but not all
    # Dist between adjacent is approx 2 * sin(pi/6) = 1.0
    sc = SimplicialComplex.from_vietoris_rips(points, epsilon=1.1, max_dimension=1)
    
    # 6 vertices, 6 edges -> Cycle
    assert sc.betti_number(0) == 1
    assert sc.betti_number(1) == 1
    
    is_mani, dim, _ = sc.is_homology_manifold()
    assert is_mani is True
    assert dim == 1
    
    assert sc.is_closed_manifold is True

def test_is_closed_manifold_line():
    # A simple line segment
    points = np.array([[0, 0], [1, 0], [2, 0]])
    sc = SimplicialComplex.from_vietoris_rips(points, epsilon=1.1, max_dimension=1)
    
    # 3 vertices, 2 edges
    assert sc.betti_number(0) == 1
    assert sc.betti_number(1) == 0
    
    is_mani, dim, _ = sc.is_homology_manifold()
    assert is_mani is True
    assert dim == 1
    
    # It has boundary (end points)
    assert sc.is_closed_manifold is False
    assert sc.is_boundary_manifold is True

def test_generate_filtration_report_manual():
    # 10 points on a circle
    theta = np.linspace(0, 2*np.pi, 10, endpoint=False)
    points = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    
    epsilons = [0.1, 0.7, 1.5, 3.0]
    report_obj = FiltrationReport(points, epsilons, max_dimension=1)
    report = str(report_obj)
    
    # Basic checks
    assert "# Betti Numbers Report" in report
    assert "# Manifold Status Report" in report
    assert "Betti \\ Eps" in report
    assert "0.100000" in report
    assert "0.700000" in report

def test_generate_filtration_report_dynamic():
    # 4 points forming a square
    points = np.array([[0,0], [1,0], [1,1], [0,1]])
    
    # Dynamic filtration
    report_obj = FiltrationReport(points, epsilons=None, max_dimension=1)
    report = str(report_obj)
    
    assert "# Betti Numbers Report" in report
    assert "0.000000" in report
    assert "1.000000" in report
    assert "1.414214" in report
    
    # Check that K4 is correctly identified as non-manifold
    assert "No" in report 

def test_generate_filtration_report_with_components():
    # Two disjoint circles
    theta = np.linspace(0, 2*np.pi, 6, endpoint=False)
    circle1 = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    circle2 = circle1 + np.array([5, 0])
    points = np.vstack([circle1, circle2])
    
    # eps=1.1 should give two disjoint cycles
    # eps=6.0 should merge them
    report_obj = FiltrationReport(points, epsilons=[1.1, 6.0], track_connected_components=True)
    report = str(report_obj)
    
    assert "# Connected Components Report" in report
    assert "C_1" in report
    assert "C_2" in report
    assert "M(D:1, Closed)" in report

def test_generate_filtration_report_alpha():
    # 4 points forming a square
    points = np.array([[0,0], [1,0], [1,1], [0,1]])
    
    # Alpha complex mode
    report_obj = FiltrationReport(points, mode="alpha", track_connected_components=True)
    report = str(report_obj)
    
    assert "(Mode: alpha)" in report
    assert "Alpha" in report # Column label

if __name__ == "__main__":
    # If run directly, show an alpha report using class
    points = np.array([[0,0], [1,0], [1,1], [0,1]])
    report = FiltrationReport(points, mode="alpha", track_connected_components=True)
    print(report)
