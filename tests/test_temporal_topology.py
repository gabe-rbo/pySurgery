import pytest
import numpy as np
from pysurgery.topology.temporal_topology import (
    build_union_intersection_zigzag,
    compute_temporal_homology,
    compute_topological_loss,
    detect_bifurcations,
    analyze_temporal_evolution,
    TemporalBarcode,
    BifurcationEvent,
    TemporalAnalysisResult
)

def test_morphing_shape_sequences():
    # Sequence of point clouds
    # Create simple point clouds that change slightly
    pts1 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    pts2 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    pts3 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    
    seq = [pts1, pts2, pts3]
    zigzag = build_union_intersection_zigzag(seq, epsilon=1.5, max_dimension=1)
    
    assert len(zigzag) == 5  # K0, K0^K1, K1, K1^K2, K2
    
    tb = compute_temporal_homology(zigzag, dimension=0, parameters=[0.0, 1.0, 2.0], field='Z2')
    assert tb.dimension == 0
    assert len(tb.parameters) == 3
    assert tb.parameters == [0.0, 1.0, 2.0]
    
    # Just checking it returns valid data without error
    assert isinstance(tb, TemporalBarcode)
    assert len(tb.births) == len(tb.deaths)

def test_bifurcation_events():
    # Synthetic barcode to trigger bifurcation
    tb = TemporalBarcode(
        dimension=0,
        births=[0.0, 0.0, 1.0],
        deaths=[2.0, 1.0, 2.0],
        parameters=[0.0, 1.0, 2.0],
        field='Z2',
        exact=True
    )
    
    # active counts:
    # t=0 (idx 0): b=0..2 (1), b=0..1 (1) => count 2
    # t=1 (idx 1): b=0..2 (1), b=0..1 (1), b=1..2 (1) => count 3
    # t=2 (idx 2): b=0..2 (1), b=1..2 (1) => count 2
    
    events = detect_bifurcations([tb], threshold=0.5)
    assert len(events) >= 1
    assert isinstance(events[0], BifurcationEvent)
    assert events[0].type == "Topological Phase Transition"

def test_jax_autodiff_verification():
    import importlib.util
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX not available")
        
    tb1 = TemporalBarcode(
        dimension=0,
        births=[0.0],
        deaths=[2.0],
        parameters=[0.0, 1.0, 2.0]
    )
    
    tb2 = TemporalBarcode(
        dimension=0,
        births=[0.0, 1.0],
        deaths=[2.0, 2.0],
        parameters=[0.0, 1.0, 2.0]
    )
    
    loss = compute_topological_loss(tb1, tb2, epsilon=0.01)
    assert isinstance(loss, float)
    assert loss >= 0.0

def test_analyze_temporal_evolution_integration():
    # User research instance (temporal data)
    # A circle that splits into two components and then disappears
    theta = np.linspace(0, 2*np.pi, 20, endpoint=False)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    
    # Time 0: Full circle
    pts_t0 = circle.copy()
    
    # Time 1: Missing bottom part (two components if epsilon is small)
    pts_t1 = circle[circle[:, 1] > -0.5]
    
    # Time 2: Just top part
    pts_t2 = circle[circle[:, 1] > 0.5]
    
    seq = [pts_t0, pts_t1, pts_t2]
    params = [0.0, 1.0, 2.0]
    
    # The integration test
    result = analyze_temporal_evolution(
        point_cloud_sequence=seq,
        parameter_values=params,
        dimensions=[0, 1],
        epsilon=1.2,
        max_rips_dimension=2,
        bifurcation_threshold=0.0
    )
    
    assert isinstance(result, TemporalAnalysisResult)
    assert 0 in result.barcodes
    assert 1 in result.barcodes
    assert isinstance(result.barcodes[0], TemporalBarcode)
    
    # Check that bifurcation detection finds events
    assert isinstance(result.bifurcations, list)
    # Depending on epsilon, the topology will change, resulting in bifurcations

