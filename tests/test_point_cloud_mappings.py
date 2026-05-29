import numpy as np
from pysurgery.topology.complexes import SimplicialComplex

def test_point_cloud_mappings_all_constructors():
    # Setup: 6 distinct points in 3D
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0]
    ], dtype=np.float64)

    # 1. Test from_vietoris_rips
    sc_rips = SimplicialComplex.from_vietoris_rips(points, epsilon=1.5, max_dimension=2)
    verify_mappings(sc_rips, points)

    # 2. Test from_point_cloud_cknn
    sc_cknn = SimplicialComplex.from_point_cloud_cknn(points, k=3, delta=1.2, max_dimension=2)
    verify_mappings(sc_cknn, points)

    # 3. Test from_alpha_complex
    sc_alpha = SimplicialComplex.from_alpha_complex(points, alpha=2.0)
    verify_mappings(sc_alpha, points)

    # 4. Test from_crust_algorithm
    sc_crust = SimplicialComplex.from_crust_algorithm(points)
    verify_mappings(sc_crust, points)

    # 5. Test from_witness (using 4 landmarks)
    sc_witness = SimplicialComplex.from_witness(points, n_landmarks=4, alpha=0.5, max_dimension=2)
    # The landmark points are points[landmarks] - the vertices of witness complex are indexed from 0 to 3
    # and map to landmark_points
    verify_mappings(sc_witness, sc_witness._coordinates)


def verify_mappings(sc: SimplicialComplex, point_cloud: np.ndarray):
    """Utility to assert structural correctness of bidirectional mappings."""
    # Ensure point_cloud_to_simplices is generated and complete
    assert hasattr(sc, "point_cloud_to_simplices")
    assert isinstance(sc.point_cloud_to_simplices, dict)
    
    n_pts = len(point_cloud)
    assert len(sc.point_cloud_to_simplices) == n_pts
    for i in range(n_pts):
        assert i in sc.point_cloud_to_simplices
        assert isinstance(sc.point_cloud_to_simplices[i], list)

    # Ensure simplices_to_point_cloud is generated and complete
    assert hasattr(sc, "simplices_to_point_cloud")
    assert isinstance(sc.simplices_to_point_cloud, dict)

    # Check mapping consistency
    all_simplices = sc.simplices
    assert len(sc.simplices_to_point_cloud) == len(all_simplices)

    for simplex in all_simplices:
        assert simplex in sc.simplices_to_point_cloud
        coords = sc.simplices_to_point_cloud[simplex]
        assert isinstance(coords, np.ndarray)
        assert coords.shape == (len(simplex), point_cloud.shape[1])
        
        # Verify coordinates match actual point_cloud coordinates of simplex vertices
        expected_coords = point_cloud[list(simplex)]
        np.testing.assert_allclose(coords, expected_coords)

        # Verify simplex is registered in the list of every vertex it contains
        for vertex in simplex:
            assert simplex in sc.point_cloud_to_simplices[vertex]

    # Verify every registered simplex in point_cloud_to_simplices actually contains the vertex
    for vertex, simplices in sc.point_cloud_to_simplices.items():
        for simplex in simplices:
            assert vertex in simplex
            assert simplex in all_simplices


def test_simplicial_complex_concatenation():
    # Setup two different point clouds
    pts_a = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ], dtype=np.float64)

    pts_b = np.array([
        [10.0, 10.0],
        [11.0, 10.0],
        [10.0, 11.0],
        [11.0, 11.0]
    ], dtype=np.float64)

    # Build two complexes
    sc_a = SimplicialComplex.from_vietoris_rips(pts_a, epsilon=1.5, max_dimension=2)
    sc_b = SimplicialComplex.from_vietoris_rips(pts_b, epsilon=1.5, max_dimension=2)

    # Set some filtrations to verify they are preserved
    sc_a.filtration = {(0,): 0.1, (0, 1): 0.5}
    sc_b.filtration = {(0,): 0.2, (0, 1): 0.6}

    # Concatenate them
    sc_concat = SimplicialComplex.concatenate([sc_a, sc_b])

    # 1. Verify vertex shift
    # Complex A has 3 points, B has 4 points.
    # Total points = 7.
    assert hasattr(sc_concat, "_coordinates")
    assert sc_concat._coordinates.shape == (7, 2)
    
    # Assert coordinates concatenated correctly
    np.testing.assert_allclose(sc_concat._coordinates[:3], pts_a)
    np.testing.assert_allclose(sc_concat._coordinates[3:], pts_b)

    # 2. Verify mappings
    verify_mappings(sc_concat, sc_concat._coordinates)

    # 3. Verify filtration is preserved and shifted correctly
    assert sc_concat.filtration[(0,)] == 0.1
    assert sc_concat.filtration[(0, 1)] == 0.5
    assert sc_concat.filtration[(3,)] == 0.2
    assert sc_concat.filtration[(3, 4)] == 0.6
