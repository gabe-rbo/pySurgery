#!/usr/bin/env python3
"""Test suite for the surface triangulation functionality.

Overview:
    This module provides unit tests for surface triangulation algorithms, specifically
    verifying that point clouds in 3D are correctly converted into 2-dimensional 
    simplicial complexes (triangle meshes). It tests both specific implementations 
    and the public unified API.

Key Concepts:
    - **Surface Triangulation**: Constructing a 2-manifold (with or without boundary) from points.
    - **Skeleton Extraction**: Verifying the presence of vertices, edges, and faces.
    - **Implementation Parity**: Ensuring Python fallbacks match expectation for common surfaces.
"""

import numpy as np
import sys

from pysurgery.integrations.gudhi_bridge import (
    triangulate_surface,
    triangulate_surface_python,
)


def test_triangulate_sphere():
    """Verify triangulation of a sampled sphere.

    What is Being Computed?:
        Checks if the triangulation algorithm correctly reconstructs a sphere-like 
        surface from a stochastic point cloud.

    Algorithm:
        1. Generate 50 points on a unit sphere using uniform spherical sampling.
        2. Run `triangulate_surface_python` on the point set.
        3. Extract the 2-skeleton and count simplices by dimension.
        4. Assert that all vertices are present and both edges/faces are created.
    """
    print("Testing sphere triangulation...")

    # Generate points on a unit sphere
    np.random.seed(42)
    u = np.random.uniform(0, 2 * np.pi, 50)
    v = np.random.uniform(0, np.pi, 50)
    x = np.sin(v) * np.cos(u)
    y = np.sin(v) * np.sin(u)
    z = np.cos(v)

    points = np.column_stack([x, y, z])

    print(f"  Input points: {len(points)}")

    # Test Python implementation
    st = triangulate_surface_python(points)

    print(f"  Vertices: {st.num_vertices()}")
    print(f"  Total simplices: {st.num_simplices()}")
    print(f"  Dimension: {st.dimension()}")

    # Check that we have 0, 1, and 2-simplices
    skeleton = st.get_skeleton(2)
    dim_counts = {}
    for simplex, _ in skeleton:
        d = len(simplex) - 1
        dim_counts[d] = dim_counts.get(d, 0) + 1

    print(f"  Simplices by dimension: {dim_counts}")
    assert dim_counts.get(0, 0) == len(points), "Should have all vertices"
    assert dim_counts.get(2, 0) > 0, "Should have 2-simplices (faces)"
    assert dim_counts.get(1, 0) > 0, "Should have 1-simplices (edges)"

    print("  ✓ Sphere triangulation passed!")

def test_triangulate_torus():
    """Verify triangulation of a sampled torus.

    What is Being Computed?:
        Tests the algorithm's ability to handle surfaces with non-zero genus (torus).

    Algorithm:
        1. Generate 100 points on a torus with major radius 3.0 and minor radius 1.0.
        2. Run `triangulate_surface_python`.
        3. Confirm the presence of 2-dimensional faces in the output complex.
    """
    print("Testing torus triangulation...")

    # Generate points on a torus: (R + r*cos(v)) * (cos(u), sin(u)) in x-y, r*sin(v) in z
    R, r = 3.0, 1.0
    n_points = 100
    u = np.linspace(0, 2 * np.pi, int(np.sqrt(n_points)))
    v = np.linspace(0, 2 * np.pi, int(np.sqrt(n_points)))
    u_grid, v_grid = np.meshgrid(u, v)
    u_flat = u_grid.flatten()
    v_flat = v_grid.flatten()

    x = (R + r * np.cos(v_flat)) * np.cos(u_flat)
    y = (R + r * np.cos(v_flat)) * np.sin(u_flat)
    z = r * np.sin(v_flat)

    points = np.column_stack([x, y, z])

    print(f"  Input points: {len(points)}")

    st = triangulate_surface_python(points)

    print(f"  Vertices: {st.num_vertices()}")
    print(f"  Total simplices: {st.num_simplices()}")

    skeleton = st.get_skeleton(2)
    dim_counts = {}
    for simplex, _ in skeleton:
        d = len(simplex) - 1
        dim_counts[d] = dim_counts.get(d, 0) + 1

    print(f"  Simplices by dimension: {dim_counts}")
    assert dim_counts.get(2, 0) > 0, "Should have 2-simplices"

    print("  ✓ Torus triangulation passed!")

def test_triangulate_plane():
    """Verify triangulation of points on a flat plane.

    What is Being Computed?:
        Ensures the triangulator handles degenerate cases where points are coplanar in 3D.

    Algorithm:
        1. Generate 30 random points on the z=0 plane.
        2. Run `triangulate_surface_python`.
        3. Assert that a 2D complex is produced despite the lack of volumetric span.
    """
    print("Testing planar point cloud triangulation...")

    # Generate points on z=0 plane
    np.random.seed(42)
    n_points = 30
    x = np.random.uniform(-5, 5, n_points)
    y = np.random.uniform(-5, 5, n_points)
    z = np.zeros(n_points)

    points = np.column_stack([x, y, z])

    print(f"  Input points: {len(points)}")

    st = triangulate_surface_python(points)

    print(f"  Vertices: {st.num_vertices()}")
    print(f"  Total simplices: {st.num_simplices()}")
    print(f"  Dimension: {st.dimension()}")

    skeleton = st.get_skeleton(2)
    dim_counts = {}
    for simplex, _ in skeleton:
        d = len(simplex) - 1
        dim_counts[d] = dim_counts.get(d, 0) + 1

    print(f"  Simplices by dimension: {dim_counts}")
    assert dim_counts.get(2, 0) > 0, "Should have 2-simplices"

    print("  ✓ Planar triangulation passed!")

def test_triangulate_public_api():
    """Test the unified public triangulation entry point.

    What is Being Computed?:
        Verifies that the `triangulate_surface` dispatcher correctly routes 
        calls to the available backend.

    Algorithm:
        1. Sample points from a sphere using a grid.
        2. Call the high-level `triangulate_surface` function.
        3. Confirm the resulting SimplicialComplex is non-empty.
    """
    print("Testing public triangulate_surface API...")

    # Simple sphere
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    u_grid, v_grid = np.meshgrid(u, v)
    x = np.sin(v_grid) * np.cos(u_grid)
    y = np.sin(v_grid) * np.sin(u_grid)
    z = np.cos(v_grid)

    points = np.column_stack([x.flatten(), y.flatten(), z.flatten()])

    st = triangulate_surface(points)

    print(f"  Vertices: {st.num_vertices()}")
    print(f"  Total simplices: {st.num_simplices()}")

    assert st.num_vertices() > 0, "Should have vertices"
    assert st.num_simplices() > 0, "Should have simplices"

    print("  ✓ Public API test passed!")


if __name__ == "__main__":
    try:
        test_triangulate_sphere()
        test_triangulate_torus()
        test_triangulate_plane()
        test_triangulate_public_api()

        print("\n✓ All triangulation tests passed!")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
