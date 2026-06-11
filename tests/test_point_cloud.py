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

def test_unbend():
    # 1. Test unbending a bent point cloud
    pts = np.array([
        [0.0, 0.0],
        [np.pi / 2, 0.5],
        [np.pi, -0.2]
    ])
    pc = PointCloud(pts)
    
    # Bend with curvature 0.5
    pc_bent = pc.bend(curvature=0.5, axis=0, control_axis=1, anchor="min")
    
    # Unbend with curvature 0.5
    pc_unbent = pc_bent.unbend(curvature=0.5, axis=0, control_axis=1, anchor="min")
    np.testing.assert_allclose(pc_unbent.points, pts, atol=1e-7)

    # 2. Test negative curvature
    pc_bent_neg = pc.bend(curvature=-0.5, axis=0, control_axis=1, anchor="min")
    pc_unbent_neg = pc_bent_neg.unbend(curvature=-0.5, axis=0, control_axis=1, anchor="min")
    np.testing.assert_allclose(pc_unbent_neg.points, pts, atol=1e-7)

    # 3. Test near-zero curvature (Taylor series)
    pc_bent_zero = pc.bend(curvature=1e-9, axis=0, control_axis=1, anchor="min")
    pc_unbent_zero = pc_bent_zero.unbend(curvature=1e-9, axis=0, control_axis=1, anchor="min")
    np.testing.assert_allclose(pc_unbent_zero.points, pts, atol=1e-7)


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
    
    R, aligned, disparity = orthogonal_procrustes(pc, pc_rot)
    assert R.shape == (3, 3)
    assert aligned.shape == (50, 3)
    assert disparity < 1e-7


def test_point_cloud_undo_revert():
    from pysurgery.topology.complexes import SimplicialComplex

    # Create a point cloud
    pts = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0]
    ])
    sc = SimplicialComplex.from_vietoris_rips(pts, epsilon=5.0, max_dimension=2)
    pc = sc.point_cloud

    # Initial state checks
    assert len(pc.list_transformations()) == 0
    np.testing.assert_allclose(pc.points, pts)

    # 1. Apply translation
    pc2 = pc.translate([1.0, 2.0, 3.0])
    assert len(pc2.list_transformations()) == 1
    assert pc2.list_transformations()[0]["method"] == "translate"
    np.testing.assert_allclose(pc2.points, pts + [1.0, 2.0, 3.0])
    np.testing.assert_allclose(sc._coordinates, pts + [1.0, 2.0, 3.0])

    # 2. Apply rotation
    pc3 = pc2.rotate(90.0, plane="xy", center="center")
    assert len(pc3.list_transformations()) == 2
    assert pc3.list_transformations()[1]["method"] == "rotate"
    
    # 3. Apply shear
    pc4 = pc3.shear(factor=0.5, axis=0, control_axis=1, anchor="min")
    assert len(pc4.list_transformations()) == 3
    assert pc4.list_transformations()[2]["method"] == "shear"

    # 4. Revert all
    pc_reverted = pc4.revert()
    assert len(pc_reverted.list_transformations()) == 0
    np.testing.assert_allclose(pc_reverted.points, pts, atol=1e-7)
    np.testing.assert_allclose(sc._coordinates, pts, atol=1e-7)

    # 5. Undo last (which is shear)
    pc_undone_last = pc4.undo()
    assert len(pc_undone_last.list_transformations()) == 2
    # Should match pc3 coordinates
    np.testing.assert_allclose(pc_undone_last.points, pc3.points, atol=1e-7)
    np.testing.assert_allclose(sc._coordinates, pc3.points, atol=1e-7)

    # 6. Undo multiple/specific indices
    # Let's undo the first operation (translate) from pc4, leaving rotate and shear
    pc_undone_first = pc4.undo(indices=0)
    assert len(pc_undone_first.list_transformations()) == 2
    # Verify the remaining transformations in history
    assert pc_undone_first.list_transformations()[0]["method"] == "rotate"
    assert pc_undone_first.list_transformations()[1]["method"] == "shear"
    
    # Check that we can recreate pc_undone_first by manually applying those on original pts
    pc_expected = pc.rotate(90.0, plane="xy", center="center").shear(factor=0.5, axis=0, control_axis=1, anchor="min")
    np.testing.assert_allclose(pc_undone_first.points, pc_expected.points, atol=1e-7)

    # 7. Check non-invertible transformations
    pc_mapped = pc4.apply_mapping(lambda p: p ** 2)
    
    # revert() should raise ValueError for apply_mapping
    with pytest.raises(ValueError, match="apply_mapping"):
        pc_mapped.revert()

    # undo() should work perfectly even with apply_mapping in history
    pc_undone_mapping = pc_mapped.undo() # undoes the mapping
    np.testing.assert_allclose(pc_undone_mapping.points, pc4.points, atol=1e-7)

    # 8. Spherize invertibility checks
    pc_sph_9 = pc4.spherize(factor=0.9, radius=10.0, center="center")
    pc_sph_reverted = pc_sph_9.revert()
    np.testing.assert_allclose(pc_sph_reverted.points, pts, atol=1e-7)

    # Revert just spherize using undo
    pc_sph_undone = pc_sph_9.undo()
    np.testing.assert_allclose(pc_sph_undone.points, pc4.points, atol=1e-7)

    pc_sph_1 = pc4.spherize(factor=1.0, radius=10.0, center="center")
    with pytest.raises(ValueError, match="Spherization with factor=1.0 is not mathematically invertible"):
        pc_sph_1.revert()

    # 9. Taper, bend, unbend, radial_scale, scale_to_diameter revert round-trip
    pc_t = pc.taper(factor=0.5, axis=0, control_axis=2, anchor="center")
    np.testing.assert_allclose(pc_t.revert().points, pc.points, atol=1e-7)

    pc_b = pc.bend(curvature=0.2, axis=0, control_axis=1, anchor="center")
    np.testing.assert_allclose(pc_b.revert().points, pc.points, atol=1e-7)

    pc_ub = pc.unbend(curvature=0.2, axis=0, control_axis=1, anchor="center")
    np.testing.assert_allclose(pc_ub.revert().points, pc.points, atol=1e-7)

    pc_rs = pc.radial_scale(factor=3.0, center="center")
    np.testing.assert_allclose(pc_rs.revert().points, pc.points, atol=1e-7)

    pc_sd = pc.scale_to_diameter(target_diameter=4.0, axis=0, anchor="center")
    np.testing.assert_allclose(pc_sd.revert().points, pc.points, atol=1e-7)
    
    pc_sd_uniform = pc.scale_to_diameter(target_diameter=4.0, axis=None, anchor="center")
    np.testing.assert_allclose(pc_sd_uniform.revert().points, pc.points, atol=1e-7)

    # 10. Errors on undo out of bounds
    with pytest.raises(IndexError):
        pc4.undo(indices=99)
    with pytest.raises(ValueError, match="No transformations to undo"):
        pc.undo()


def test_point_cloud_space_blocks():
    from pysurgery.geometry import SpaceBlock

    # 1. Test SpaceBlock contains
    sb = SpaceBlock([0.0, 0.0], [5.0, 5.0])
    assert [1.0, 1.0] in sb
    assert [6.0, 1.0] not in sb
    
    pts = np.array([
        [1.0, 1.0],
        [6.0, 2.0],
        [2.0, -1.0],
        [4.0, 4.0]
    ])
    mask = sb.contains(pts)
    np.testing.assert_array_equal(mask, [True, False, False, True])

    # 2. Test block_division
    pc = PointCloud(pts)
    
    # Default: 2^D quadrants
    blocks_def = pc.block_division()
    assert len(blocks_def) == 4
    
    # 3D PointCloud
    pts_3d = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 4.0, 6.0]
    ])
    pc_3d = PointCloud(pts_3d)
    blocks_3d = pc_3d.block_division()
    assert len(blocks_3d) == 8

    # Integer division: along longest axis
    pts_span = np.array([
        [0.0, 0.0],
        [10.0, 2.0]
    ])
    pc_span = PointCloud(pts_span)
    blocks_span = pc_span.block_division(num_blocks=3)
    assert len(blocks_span) == 3
    assert blocks_span[0].min_bounds[0] == 0.0
    np.testing.assert_allclose(blocks_span[0].max_bounds[0], 10.0 / 3.0)
    np.testing.assert_allclose(blocks_span[1].min_bounds[0], 10.0 / 3.0)
    np.testing.assert_allclose(blocks_span[1].max_bounds[0], 20.0 / 3.0)

    # Grid division: list/tuple
    blocks_grid = pc.block_division(num_blocks=[2, 3])
    assert len(blocks_grid) == 6

    # 3. Deformation with movable and static blocks
    pts_def = np.array([
        [-2.0, -2.0],
        [2.0, 2.0]
    ])
    pc_def = PointCloud(pts_def)
    
    # Let's translate only quadrant with X >= 0, Y >= 0 (the top-right quadrant)
    blocks_quad = pc_def.block_division()
    top_right_block = [b for b in blocks_quad if b.min_bounds[0] >= 0 and b.min_bounds[1] >= 0][0]
    
    # Translate top-right quadrant by [10.0, 10.0]
    pc_trans = pc_def.translate([10.0, 10.0], movable_blocks=top_right_block)
    np.testing.assert_allclose(pc_trans.points, [[-2.0, -2.0], [12.0, 12.0]])

    # 4. Revert and Undo
    pc_reverted = pc_trans.revert()
    np.testing.assert_allclose(pc_reverted.points, pts_def)

    pc_undone = pc_trans.undo()
    np.testing.assert_allclose(pc_undone.points, pts_def)

    # Test static_blocks
    pc_static = pc_def.translate([10.0, 10.0], static_blocks=top_right_block)
    np.testing.assert_allclose(pc_static.points, [[8.0, 8.0], [2.0, 2.0]])
    np.testing.assert_allclose(pc_static.revert().points, pts_def)




