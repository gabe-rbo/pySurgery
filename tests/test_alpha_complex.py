import numpy as np
from pysurgery.core.complexes import SimplicialComplex

def test_alpha_complex_S2():
    # Points on S2 (2-sphere)
    n = 500
    phi = np.random.uniform(0, 2*np.pi, n)
    theta = np.random.uniform(0, np.pi, n)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    points = np.column_stack([x, y, z])
    
    # Build Alpha Complex
    # Small alpha square to capture the 'skin' of the sphere
    sc = SimplicialComplex.from_alpha_complex(points, max_alpha_square=0.1)
    
    cc = sc.chain_complex()
    # H2(S2) = Z, H1(S2) = 0, H0(S2) = Z
    h2_rank, _ = cc.homology(n=2)
    h1_rank, _ = cc.homology(n=1)
    h0_rank, _ = cc.homology(n=0)
    
    assert h2_rank >= 1
    assert h0_rank >= 1

if __name__ == "__main__":
    test_alpha_complex_S2()
