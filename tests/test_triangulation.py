#!/usr/bin/env python3
"""
Test suite for the surface triangulation functionality.
"""

import numpy as np
import sys

from pysurgery.integrations.gudhi_bridge import (
    triangulate_surface,
    triangulate_surface_python,
)


def test_triangulate_sphere():
    """Test triangulation of a sphere (simplest 2D surface)."""
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
    """Test triangulation of a torus (more complex 2D surface)."""
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
    """Test triangulation of points on a plane (degenerate 2D surface in 3D)."""
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
    """Test the public triangulate_surface API (Python fallback)."""
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
