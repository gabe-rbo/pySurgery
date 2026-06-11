"""Tests for PointCloud geometric operations."""

import numpy as np
import pytest
from pysurgery.geometry import PointCloud

def test_point_cloud_initialization():
    # Valid initialization
    pts = np.array([
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0]
    ])
    pc = PointCloud(pts)
    assert pc.num_points == 2
    assert pc.dimension == 3
    np.testing.assert_allclose(pc.points, pts)

    # Invalid initialization
    with pytest.raises(ValueError):
        PointCloud([1.0, 2.0, 3.0])  # Not a 2D array

def test_point_cloud_center_of_mass():
    pts = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [1.0, 3.0]
    ])
    pc = PointCloud(pts)
    np.testing.assert_allclose(pc.center_of_mass, [1.0, 1.0])

    pc_empty = PointCloud(np.empty((0, 3)))
    np.testing.assert_allclose(pc_empty.center_of_mass, [0.0, 0.0, 0.0])

def test_get_extreme_points():
    pts = np.array([
        [1.0, 10.0, -2.0],
        [5.0, 2.0, 10.0],
        [-3.0, 4.0, 6.0],
        [2.0, 8.0, 0.0]
    ])
    pc = PointCloud(pts)

    all_extremes = pc.get_extreme_points()
    assert isinstance(all_extremes, dict)
    
    np.testing.assert_allclose(all_extremes["min_indices"], [2, 1, 0])
    np.testing.assert_allclose(all_extremes["min"][0], [-3.0, 4.0, 6.0])
    np.testing.assert_allclose(all_extremes["min"][1], [5.0, 2.0, 10.0])
    np.testing.assert_allclose(all_extremes["min"][2], [1.0, 10.0, -2.0])

    np.testing.assert_allclose(all_extremes["max_indices"], [1, 0, 1])
    np.testing.assert_allclose(all_extremes["max"][0], [5.0, 2.0, 10.0])
    np.testing.assert_allclose(all_extremes["max"][1], [1.0, 10.0, -2.0])
    np.testing.assert_allclose(all_extremes["max"][2], [5.0, 2.0, 10.0])

    min_pt, max_pt = pc.get_extreme_points(axis=1)
    np.testing.assert_allclose(min_pt, [5.0, 2.0, 10.0])
    np.testing.assert_allclose(max_pt, [1.0, 10.0, -2.0])

    pt_min_0 = pc.get_extreme_points(axis=0, extreme="min")
    np.testing.assert_allclose(pt_min_0, [-3.0, 4.0, 6.0])

    pt_max_2 = pc.get_extreme_points(axis=2, extreme="max")
    np.testing.assert_allclose(pt_max_2, [5.0, 2.0, 10.0])

    all_mins = pc.get_extreme_points(extreme="min")
    assert all_mins.shape == (3, 3)
    np.testing.assert_allclose(all_mins[0], [-3.0, 4.0, 6.0])
    np.testing.assert_allclose(all_mins[1], [5.0, 2.0, 10.0])
    np.testing.assert_allclose(all_mins[2], [1.0, 10.0, -2.0])

    with pytest.raises(ValueError):
        pc.get_extreme_points(axis=3)
    with pytest.raises(ValueError):
        pc.get_extreme_points(extreme="invalid")
    
    pc_empty = PointCloud(np.empty((0, 2)))
    with pytest.raises(ValueError):
        pc_empty.get_extreme_points()

def test_get_rotation_lines():
    pts = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])
    pc = PointCloud(pts)
    c = pc.center_of_mass

    lines = pc.get_rotation_lines(plane="xy", angles=4, length=2.0)
    assert lines.shape == (4, 2, 2)

    for i in range(4):
        np.testing.assert_allclose(lines[i, 0], c)

    np.testing.assert_allclose(lines[0, 1], [4.0, 3.0])
    np.testing.assert_allclose(lines[1, 1], [2.0, 5.0])
    np.testing.assert_allclose(lines[2, 1], [0.0, 3.0])
    np.testing.assert_allclose(lines[3, 1], [2.0, 1.0])

    pts_3d = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 4.0, 6.0]
    ])
    pc_3d = PointCloud(pts_3d)
    c_3d = pc_3d.center_of_mass

    angles = np.array([0, np.pi/2])
    lines_rad = pc_3d.get_rotation_lines(plane=(1, 2), angles=angles, length=1.0, use_degrees=False)
    assert lines_rad.shape == (2, 2, 3)
    
    np.testing.assert_allclose(lines_rad[0, 0], c_3d)
    np.testing.assert_allclose(lines_rad[0, 1], [1.0, 3.0, 3.0])

    np.testing.assert_allclose(lines_rad[1, 0], c_3d)
    np.testing.assert_allclose(lines_rad[1, 1], [1.0, 2.0, 4.0])

    with pytest.raises(ValueError):
        pc_3d.get_rotation_lines(plane="invalid")

def test_translate():
    pts = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])
    pc = PointCloud(pts)
    pc_translated = pc.translate([10.0, -5.0])
    
    # Check that original is unchanged
    np.testing.assert_allclose(pc.points, pts)
    # Check translated values
    np.testing.assert_allclose(pc_translated.points, [[11.0, -3.0], [13.0, -1.0]])

    with pytest.raises(ValueError):
        pc.translate([1.0, 2.0, 3.0])  # dimension mismatch

def test_rotate():
    pts = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    pc = PointCloud(pts)
    # Rotate 90 degrees around origin [0, 0]
    pc_rot = pc.rotate(90.0, plane="xy", center=np.zeros(2))
    
    # [1, 0] rotated 90 deg -> [0, 1]
    # [0, 1] rotated 90 deg -> [-1, 0]
    np.testing.assert_allclose(pc_rot.points, [[0.0, 1.0], [-1.0, 0.0]], atol=1e-7)

    # Rotate with "min" anchor point (min X is 0.0 at point [0.0, 1.0])
    pc_rot_min = pc.rotate(180.0, plane="xy", center="min")
    # Rotation center should be [0.0, 1.0]
    # Point [0.0, 1.0] stays static
    # Point [1.0, 0.0] rotates 180 around [0.0, 1.0] -> [-1.0, 2.0]
    np.testing.assert_allclose(pc_rot_min.points, [[-1.0, 2.0], [0.0, 1.0]], atol=1e-7)

def test_shear():
    pts = np.array([
        [0.0, 0.0],
        [0.0, 2.0],
        [2.0, 0.0]
    ])
    pc = PointCloud(pts)
    # Shear X along Y with factor 0.5, anchor="min" (min along Y is Y=0)
    pc_sheared = pc.shear(factor=0.5, axis=0, control_axis=1, anchor="min")
    
    # displacement = 0.5 * (Y - Y_min)
    # point [0, 0] -> Y=0 -> displacement = 0 -> [0, 0]
    # point [0, 2] -> Y=2 -> displacement = 0.5 * (2 - 0) = 1 -> [1, 2]
    # point [2, 0] -> Y=0 -> displacement = 0 -> [2, 0]
    np.testing.assert_allclose(pc_sheared.points, [[0.0, 0.0], [1.0, 2.0], [2.0, 0.0]])

def test_apply_mapping():
    pts = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])
    pc = PointCloud(pts)
    # Custom quadratic mapping
    pc_mapped = pc.apply_mapping(lambda p: p ** 2)
    np.testing.assert_allclose(pc_mapped.points, [[1.0, 4.0], [9.0, 16.0]])

    with pytest.raises(TypeError):
        pc.apply_mapping("not a callable")

def test_scale_to_diameter():
    # Setup point cloud with diameter 10.0 (distance between [-5, 0] and [5, 0])
    pts = np.array([
        [-5.0, 0.0],
        [5.0, 0.0],
        [0.0, 3.0]
    ])
    pc = PointCloud(pts)
    
    # 1. Uniform scaling to diameter 5.0, anchor="center" (which is [0, 1])
    pc_scaled_uniform = pc.scale_to_diameter(5.0, axis=None, anchor="center")
    # Scale factor should be 5.0 / 10.0 = 0.5
    # center is [0.0, 1.0]
    # p' = center + 0.5 * (p - center)
    # [-5, 0] -> [0, 1] + 0.5 * [-5, -1] = [-2.5, 0.5]
    # [5, 0] -> [0, 1] + 0.5 * [5, -1] = [2.5, 0.5]
    # [0, 3] -> [0, 1] + 0.5 * [0, 2] = [0.0, 2.0]
    np.testing.assert_allclose(
        pc_scaled_uniform.points,
        [[-2.5, 0.5], [2.5, 0.5], [0.0, 2.0]]
    )
    
    # 2. Axis specific scaling (axis=0, target_diameter=2.0) relative to "min" (which is [-5, 0])
    pc_scaled_axis = pc.scale_to_diameter(2.0, axis=0, anchor="min")
    # Current diameter along axis 0 is 5.0 - (-5.0) = 10.0
    # Scale factor along axis 0 is 2.0 / 10.0 = 0.2
    # anchor coordinate X is -5.0
    # X' = -5.0 + 0.2 * (X - (-5.0))
    # [-5, 0] -> X' = -5.0 -> [-5.0, 0.0]
    # [5, 0] -> X' = -5.0 + 0.2 * 10 = -3.0 -> [-3.0, 0.0]
    # [0, 3] -> X' = -5.0 + 0.2 * 5 = -4.0 -> [-4.0, 3.0]
    np.testing.assert_allclose(
        pc_scaled_axis.points,
        [[-5.0, 0.0], [-3.0, 0.0], [-4.0, 3.0]]
    )

    # 3. Invalid params
    with pytest.raises(ValueError):
        pc.scale_to_diameter(-1.0)
    with pytest.raises(ValueError):
        pc.scale_to_diameter(5.0, axis=2)  # axis out of bounds

def test_simplicial_complex_integration():
    from pysurgery.topology.complexes import SimplicialComplex

    # Create a simple simplicial complex with coordinates
    pts = np.array([
        [0.0, 0.0],
        [4.0, 0.0],
        [0.0, 3.0]
    ])
    sc = SimplicialComplex.from_vietoris_rips(pts, epsilon=5.0, max_dimension=2)

    # Initial checks
    assert sc.point_cloud is not None
    np.testing.assert_allclose(sc._coordinates, pts)
    np.testing.assert_allclose(sc.point_cloud.points, pts)
    # Check that mappings were created
    assert (0, 1) in sc.simplices_to_point_cloud
    np.testing.assert_allclose(sc.simplices_to_point_cloud[(0, 1)], [[0.0, 0.0], [4.0, 0.0]])

    # Translate the point cloud of the complex
    sc.point_cloud.translate([1.0, 2.0])

    # Coordinates in the complex should be updated directly!
    expected_pts = pts + [1.0, 2.0]
    np.testing.assert_allclose(sc._coordinates, expected_pts)
    np.testing.assert_allclose(sc.point_cloud.points, expected_pts)

    # Dictionaries in the complex must be updated as well
    np.testing.assert_allclose(sc.simplices_to_point_cloud[(0, 1)], [[1.0, 2.0], [5.0, 2.0]])

    # Setting point_cloud directly using setter
    sc.point_cloud = np.array([
        [10.0, 10.0],
        [20.0, 10.0],
        [10.0, 20.0]
    ])
    np.testing.assert_allclose(sc._coordinates, [[10.0, 10.0], [20.0, 10.0], [10.0, 20.0]])
    np.testing.assert_allclose(sc.simplices_to_point_cloud[(0, 1)], [[10.0, 10.0], [20.0, 10.0]])



def test_twist():
    pts = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 2.0]
    ])
    pc = PointCloud(pts)
    
    pc_twisted = pc.twist(rate=45.0, plane="xy", control_axis=2, anchor=np.zeros(3))
    np.testing.assert_allclose(
        pc_twisted.points,
        [[1.0, 0.0, 0.0], [0.0, 1.0, 2.0]],
        atol=1e-7
    )

def test_bend():
    pts = np.array([
        [0.0, 0.0],
        [np.pi, 0.0]
    ])
    pc = PointCloud(pts)
    
    pc_bent = pc.bend(curvature=1.0, axis=0, control_axis=1, anchor="min")
    np.testing.assert_allclose(
        pc_bent.points,
        [[0.0, 0.0], [0.0, 2.0]],
        atol=1e-7
    )

    pc_flat = pc.bend(curvature=0.0, axis=0, control_axis=1, anchor="min")
    np.testing.assert_allclose(pc_flat.points, pts)

def test_taper():
    pts_offset = np.array([
        [2.0, 3.0, 0.0],
        [2.0, 3.0, 2.0]
    ])
    pc_offset = PointCloud(pts_offset)
    
    pc_tapered = pc_offset.taper(factor=1.0, axis=0, control_axis=2, anchor=np.zeros(3))
    np.testing.assert_allclose(
        pc_tapered.points,
        [[2.0, 3.0, 0.0], [6.0, 3.0, 2.0]]
    )

def test_radial_scale():
    pts = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])
    pc = PointCloud(pts)
    pc_scaled = pc.radial_scale(2.0, center=np.array([1.0, 2.0]))
    np.testing.assert_allclose(pc_scaled.points, [[1.0, 2.0], [5.0, 6.0]])

def test_spherize():
    pts = np.array([
        [3.0, 0.0],
        [-3.0, 0.0],
        [0.0, 4.0]
    ])
    pc = PointCloud(pts)
    pc_sph = pc.spherize(factor=1.0, radius=5.0, center=np.zeros(2))
    np.testing.assert_allclose(
        pc_sph.points,
        [[5.0, 0.0], [-5.0, 0.0], [0.0, 5.0]],
        atol=1e-7
    )
    
    pc_sph_def = pc.spherize(factor=1.0, radius=None, center=np.zeros(2))
    avg_rad = 10.0 / 3.0
    np.testing.assert_allclose(
        pc_sph_def.points,
        [[avg_rad, 0.0], [-avg_rad, 0.0], [0.0, avg_rad]],
        atol=1e-7
    )

def test_array_protocols():
    pts = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])
    pc = PointCloud(pts)

    # 1. Test slicing/indexing (__getitem__)
    np.testing.assert_allclose(pc[0], [1.0, 2.0])
    np.testing.assert_allclose(pc[:, 0], [1.0, 3.0])

    # 2. Test direct mutation (__setitem__)
    pc[0, 1] = 99.0
    np.testing.assert_allclose(pc.points[0, 1], 99.0)

    # 3. Test shape, ndim, dtype
    assert pc.shape == (2, 2)
    assert pc.ndim == 2
    assert pc.dtype == np.float64

    # 4. Test len and iteration
    assert len(pc) == 2
    pts_list = list(pc)
    np.testing.assert_allclose(pts_list[0], [1.0, 99.0])

    # 5. Test np.asarray (__array__)
    arr = np.asarray(pc)
    assert isinstance(arr, np.ndarray)
    np.testing.assert_allclose(arr, pc.points)

def test_intrinsic_dimension_and_alignment():
    from pysurgery.geometry import estimate_intrinsic_dimension
    from pysurgery.geometry.metrics import orthogonal_procrustes, compute_distance_matrix

    # Generate a simple point cloud (helix)
    t = np.linspace(0, 4 * np.pi, 50)
    pts = np.column_stack([np.cos(t), np.sin(t), t])
    pc = PointCloud(pts)

    # 1. Test estimate_intrinsic_dimension accepts PointCloud
    res = estimate_intrinsic_dimension(pc, methods=["twonn"])
    assert res is not None

    # 2. Test compute_distance_matrix accepts PointCloud
    D = compute_distance_matrix(pc, metric="euclidean")
    assert D.shape == (50, 50)

    # 3. Test orthogonal_procrustes accepts PointCloud
    # Rotate the helix slightly
    pc_rot = pc.rotate(30.0, plane="xy", center=np.zeros(3))
    
    # Run orthogonal alignment
    R, aligned, disparity = orthogonal_procrustes(pc, pc_rot)
    assert R.shape == (3, 3)
    assert aligned.shape == (50, 3)
    assert disparity < 1e-7


