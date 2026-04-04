"""
TUTORIAL 02: INTERSECTION FORMS AND CLASSIFICATION

In 4D topology, the Intersection Form Q on H_2(M; Z) completely 
dominates the classification of the manifold. Here, we explore
the signature, parity, and classification of famous 4-manifolds.
"""

import numpy as np
from pysurgery.core.intersection_forms import IntersectionForm

def run_tutorial_02():
    print("--- 02: 4-Manifold Intersection Forms ---")
    
    # 1. S^2 x S^2
    print("\n[Manifold A]: S^2 x S^2")
    H = np.array([[0, 1], [1, 0]])
    s2s2 = IntersectionForm(matrix=H, dimension=4)
    print("Matrix:")
    print(s2s2.matrix)
    print("Classification Data:")
    print(s2s2.classify_z_form())
    print("Note: Because 'even' is True, this is a Type II (spin) manifold.")

    # 2. CP^2 # CP^2
    print("\n[Manifold B]: CP^2 # CP^2 (Connected Sum)")
    cp2cp2 = np.array([[1, 0], [0, 1]])
    cc = IntersectionForm(matrix=cp2cp2, dimension=4)
    print("Matrix:")
    print(cc.matrix)
    print("Classification Data:")
    print(cc.classify_z_form())
    print("Note: Because 'even' is False, this is a Type I (non-spin) manifold.")
    
    # 3. E8 Manifold
    print("\n[Manifold C]: The E8 Manifold")
    print("A remarkable simply-connected topological manifold that admits NO smooth structure.")
    E8 = np.array([
        [ 2, -1,  0,  0,  0,  0,  0,  0],
        [-1,  2, -1,  0,  0,  0,  0,  0],
        [ 0, -1,  2, -1,  0,  0,  0, -1],
        [ 0,  0, -1,  2, -1,  0,  0,  0],
        [ 0,  0,  0, -1,  2, -1,  0,  0],
        [ 0,  0,  0,  0, -1,  2, -1,  0],
        [ 0,  0,  0,  0,  0, -1,  2,  0],
        [ 0,  0, -1,  0,  0,  0,  0,  2]
    ])
    e8_man = IntersectionForm(matrix=E8, dimension=4)
    print("Classification Data:")
    print(e8_man.classify_z_form())
    print(f"Determinant: {e8_man.determinant()} (Unimodular!)")
    
if __name__ == "__main__":
    run_tutorial_02()
