import numpy as np
from pysurgery.core.complexes import SimplicialComplex

def test_witness_complex_scaling():
    # 2000 points on a circle (S1)
    # 10k+ points is doable but we use 2k for the CI test
    t = np.linspace(0, 2*np.pi, 2000)
    points = np.column_stack([np.cos(t), np.sin(t)])
    
    # Build witness complex with 50 landmarks
    # This should find H1 = Z (1 generator)
    sc = SimplicialComplex.from_witness(points, n_landmarks=50, max_dimension=1)
    
    # Compute homology (via Julia or fallback)
    # The 'homology' method is in the ChainComplex returned by .chain_complex()
    cc = sc.chain_complex()
    h1_rank, h1_torsion = cc.homology(n=1)
    
    assert h1_rank >= 1
    print(f"H1 Rank: {h1_rank}, Torsion: {h1_torsion}")

if __name__ == "__main__":
    test_witness_complex_scaling()
