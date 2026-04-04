"""
TUTORIAL 05: OMNI-DIMENSIONAL HOMEOMORPHISMS

pysurgery generalizes across dimensions, citing the relevant
major theorems (Classification of Surfaces, Perelman, Freedman, Smale).
"""

import numpy as np
from scipy.sparse import csr_matrix
from pysurgery.core.complexes import ChainComplex
from pysurgery.core.intersection_forms import IntersectionForm
from pysurgery.homeomorphism import (
    analyze_homeomorphism_2d,
    analyze_homeomorphism_3d,
    analyze_homeomorphism_4d,
    analyze_homeomorphism_high_dim
)

def run_tutorial_05():
    print("--- 05: Multi-Dimensional Homeomorphism Analysis ---")
    
    # 2D Example
    print("\n--- 2D Surfaces ---")
    c1 = ChainComplex(boundaries={1: csr_matrix((1, 0)), 2: csr_matrix((0, 1))}, dimensions=[0, 1, 2]) # S^2 mock
    c2 = ChainComplex(boundaries={1: csr_matrix((1, 0)), 2: csr_matrix((0, 1))}, dimensions=[0, 1, 2]) # S^2 mock
    res, report = analyze_homeomorphism_2d(c1, c2)
    print(f"Is Homeomorphic? {res}\\nReport: {report}")
    
    # 3D Example
    print("\n--- 3D Manifolds (The Perelman Warning) ---")
    c3 = ChainComplex(boundaries={1: csr_matrix((1,0)), 2: csr_matrix((0,1)), 3: csr_matrix((0,1))}, dimensions=[0,1,2,3]) # S^3 mock
    res, report = analyze_homeomorphism_3d(c3, c3)
    print(f"Is Homeomorphic? {res}\\nReport: {report}")
    
    # 4D Example
    print("\n--- 4D Manifolds (Freedman's Domain) ---")
    q1 = IntersectionForm(matrix=np.array([[0,1],[1,0]]), dimension=4)
    q2 = IntersectionForm(matrix=np.array([[1,0],[0,-1]]), dimension=4)
    res, report = analyze_homeomorphism_4d(q1, q2)
    print(f"Is Homeomorphic? {res}\\nReport: {report}")
    
    # High-Dim Example
    print("\n--- 5D+ Manifolds (Smale and s-Cobordism) ---")
    c5 = ChainComplex(boundaries={i: csr_matrix((1,0)) for i in range(1,6)}, dimensions=list(range(6)))
    res, report = analyze_homeomorphism_high_dim(c5, c5, dim=5)
    print(f"Is Homeomorphic? {res}\\nReport: {report}")

if __name__ == "__main__":
    run_tutorial_05()
