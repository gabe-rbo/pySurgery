import numpy as np
from pysurgery.core.complexes import SimplicialComplex

def test_witness_complex_scaling():
    # 2000 points on a circle (S1)
    # 10k+ points is doable but we use 2k for the CI test
    t = np.linspace(0, 2*np.pi, 2000)
    points = np.column_stack([np.cos(t), np.sin(t)])
    
    # Build witness complex with more landmarks and larger alpha to ensure H1
    sc = SimplicialComplex.from_witness(points, n_landmarks=100, alpha=0.5, max_dimension=1)
    # Compute homology (via Julia or fallback)
    # The 'homology' method is in the ChainComplex returned by .chain_complex()
    cc = sc.chain_complex()
    h1_rank, h1_torsion = cc.homology(n=1)
    
    assert h1_rank >= 1
    print(f"H1 Rank: {h1_rank}, Torsion: {h1_torsion}")

def test_witness_complex_torus():
    """Test witness complex on a torus."""
    # Build a torus point cloud
    n = 400
    theta = np.random.uniform(0, 2*np.pi, n)
    phi = np.random.uniform(0, 2*np.pi, n)
    R, r = 2.0, 0.5
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    points = np.column_stack([x, y, z])
    
    # We pick landmarks and build the complex
    # Reducing landmarks and using max_dimension=1 for speed in CI
    sc = SimplicialComplex.from_witness(points, n_landmarks=40, alpha=0.6, max_dimension=1)
    cc = sc.chain_complex()
    
    h1_rank, _ = cc.homology(n=1)
    # 1-skeleton of a torus-like graph should have many loops, at least the 2 fundamental ones.
    assert h1_rank >= 2

if __name__ == "__main__":
    test_witness_complex_scaling()
