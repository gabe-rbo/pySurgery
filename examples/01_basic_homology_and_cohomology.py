"""
TUTORIAL 01: BASIC HOMOLOGY AND COHOMOLOGY

In this tutorial, we will learn the foundation of pysurgery:
Constructing a CW Complex and computing its exact Homology and Cohomology
using Smith Normal Form reduction over Z.
"""

import numpy as np
from scipy.sparse import csr_matrix
from pysurgery.core.complexes import CWComplex

def run_tutorial_01():
    print("--- 01: The Real Projective Plane (RP^2) ---")
    print("We will construct RP^2 (1 vertex, 1 edge, 1 face) where the boundary")
    print("of the 2-face wraps around the 1-edge twice (d2 = 2).")
    
    # Define the cellular structure
    cells = {0: 1, 1: 1, 2: 1}
    
    # Boundary operators
    d1 = csr_matrix((1, 1), dtype=np.int64) # d1(e) = v - v = 0
    d2 = csr_matrix([[2]], dtype=np.int64)  # d2(f) = 2e
    
    attaching_maps = {1: d1, 2: d2}
    
    rp2 = CWComplex(cells=cells, attaching_maps=attaching_maps)
    chain = rp2.cellular_chain_complex()
    
    print("\n--- Computing Homology H_n(RP^2; Z) ---")
    for n in [0, 1, 2]:
        rank, torsion = chain.homology(n)
        
        # Format the output string mathematically
        components = []
        if rank > 0:
            components.append(f"Z^{rank}")
        if torsion:
            components.extend([f"Z_{t}" for t in torsion])
            
        group_str = " + ".join(components) if components else "0"
        print(f"H_{n}(RP^2) = {group_str}")

    print("\n--- Computing Cohomology H^n(RP^2; Z) ---")
    print("Notice how the torsion shifts up a dimension via the Universal Coefficient Theorem!")
    for n in [0, 1, 2]:
        rank, torsion = chain.cohomology(n)
        
        components = []
        if rank > 0:
            components.append(f"Z^{rank}")
        if torsion:
            components.extend([f"Z_{t}" for t in torsion])
            
        group_str = " + ".join(components) if components else "0"
        print(f"H^{n}(RP^2) = {group_str}")

if __name__ == "__main__":
    run_tutorial_01()
