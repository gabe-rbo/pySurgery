import numpy as np
from pysurgery.core.complexes import SimplicialComplex

def test_alpha_complex_tiny_S2():
    """
    Test Alpha Complex construction on a minimal 6-point octahedron 
    embedding (homeomorphic to S2). This ensures the native Delaunay 
    circumradius pipeline is verified with minimal computational load.
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

if __name__ == "__main__":
    test_alpha_complex_tiny_S2()
