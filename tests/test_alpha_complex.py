"""Tests for Alpha Complex construction and persistent homology foundations.

Overview:
    This suite verifies the construction of Alpha Complexes from point cloud 
    data using Delaunay triangulations and circumradius filtering. It ensures 
    that the resulting complexes correctly capture the topology of the 
    underlying shapes (S², S¹, etc.).

Key Concepts:
    - **Alpha Complex**: A subcomplex of the Delaunay triangulation consisting 
      of simplices whose circumradius is below a threshold α.
    - **Delaunay Triangulation**: A triangulation such that no point is inside 
      the circumsphere of any simplex.
    - **Nerve Theorem**: Ensures that the Alpha Complex is homotopy equivalent 
      to the union of balls around the points.
"""
import numpy as np
from pysurgery.core.complexes import SimplicialComplex

def test_alpha_complex_tiny_S2():
    """Verify Alpha Complex construction for a 3D octahedron (S²).

    What is Being Computed?:
        The homology of an Alpha Complex built from 6 points on a sphere.

    Algorithm:
        1. Define 6 points representing the vertices of a regular octahedron.
        2. Construct an Alpha Complex with α²=0.9 (captures faces but not the interior).
        3. Compute homology groups H₀, H₁, and H₂.

    Preserved Invariants:
        - Betti numbers for S²: β₀=1, β₁=0, β₂=1.
    """
    # 6 points of a regular octahedron
    points = np.array([
        [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]
    ])
    
    # Build Alpha Complex with a threshold that captures the boundary
    # For this octahedron, edge lengths are sqrt(2) ~ 1.414.
    # Circumradius of faces is sqrt(2/3) ~ 0.816.
    # We use a threshold that allows faces but not the interior.
    sc = SimplicialComplex.from_alpha_complex(points, max_alpha_square=0.9)
    
    cc = sc.chain_complex()
    # H2(S2) = Z, H1(S2) = 0, H0(S2) = Z
    h2_rank, _ = cc.homology(n=2)
    h1_rank, _ = cc.homology(n=1)
    h0_rank, _ = cc.homology(n=0)
    
    # Verify exact Betti numbers for the minimal sphere
    assert h2_rank == 1
    assert h1_rank == 0
    assert h0_rank == 1

def test_alpha_complex_circle():
    """Verify Alpha Complex construction for a 2D circle (S¹).

    What is Being Computed?:
        The homology of an Alpha Complex built from 30 points on a circle.

    Algorithm:
        1. Generate 30 points uniformly distributed on a unit circle.
        2. Construct an Alpha Complex with α²=0.015.
        3. Verify H₀=ℤ, H₁=ℤ.

    Preserved Invariants:
        - Betti numbers for S¹: β₀=1, β₁=1.
    """
    t = np.linspace(0, 2*np.pi, 30, endpoint=False)
    points = np.column_stack([np.cos(t), np.sin(t)])
    
    # Adjacent distance squared ~ 0.043. r2_e = d2/4 ~ 0.011.
    sc = SimplicialComplex.from_alpha_complex(points, max_alpha_square=0.015)
    cc = sc.chain_complex()
    
    h1_rank, _ = cc.homology(n=1)
    h0_rank, _ = cc.homology(n=0)
    
    assert h1_rank == 1
    assert h0_rank == 1

def test_alpha_complex_non_convex_consistency():
    """Verify that Alpha Complexes capture non-convex shapes correctly.

    What is Being Computed?:
        The Euler characteristic of an Alpha Complex for a 'C'-shaped point cloud.

    Algorithm:
        1. Generate points along a partial circle ('C' shape).
        2. Construct an Alpha Complex with a small α².
        3. Verify that the result is contractible (χ=1).

    Preserved Invariants:
        - Contractibility (homotopy type of a point) is preserved for a simple arc.
    """
    # Points in a 'C' shape
    t = np.linspace(0, 1.5*np.pi, 50)
    points = np.column_stack([np.cos(t), np.sin(t)])
    
    # Adjacent distance squared ~ 0.009. r2_e ~ 0.0023.
    sc = SimplicialComplex.from_alpha_complex(points, max_alpha_square=0.005)
    # Should be contractible (a line segment basically)
    assert sc.euler_characteristic() == 1

if __name__ == "__main__":
    test_alpha_complex_tiny_S2()
