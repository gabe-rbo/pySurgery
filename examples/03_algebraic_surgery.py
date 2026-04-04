"""
TUTORIAL 03: ALGEBRAIC SURGERY

This tutorial demonstrates how to use pysurgery to actually modify 
the topology of a manifold using algebraic surgery.
"""

import numpy as np
from pysurgery.core.intersection_forms import IntersectionForm
from pysurgery.core.exceptions import IsotropicError

def run_tutorial_03():
    print("--- 03: Performing Algebraic Surgery ---")
    
    print("\nOur Starting Manifold: S^2 x S^2")
    H = np.array([[0, 1], [1, 0]])
    s2s2 = IntersectionForm(matrix=H, dimension=4)
    print(f"Rank: {s2s2.rank()}, Signature: {s2s2.signature()}")
    
    print("\nSTEP 1: Finding an Isotropic Class")
    print("To do surgery on a 2-sphere inside S^2 x S^2, we must ensure its normal bundle is trivial.")
    print("Algebraically, this means finding a vector x where Q(x, x) = 0.")
    
    x_good = np.array([1, 0])
    self_int = np.dot(x_good, s2s2.matrix @ x_good)
    print(f"Let x = {x_good}. Self-intersection Q(x,x) = {self_int}. It is Isotropic!")
    
    print("\nSTEP 2: The Surgery Operation")
    print("pysurgery will now locate the dual class y, project the lattice onto the orthogonal")
    print("complement {x, y}^perp, and extract a Z-basis using Hermite Normal Form.")
    
    surgered_manifold = s2s2.perform_algebraic_surgery(x_good)
    
    print("\nRESULTING MANIFOLD:")
    print(f"Rank: {surgered_manifold.rank()}")
    print("Because the rank is 0, we have successfully surgered S^2 x S^2 into a homology S^4 sphere!")
    
    print("\nSTEP 3: Surgery Obstructions")
    print("What if we try this on CP^2? Q = [1]")
    cp2 = IntersectionForm(matrix=np.array([[1]]), dimension=4)
    x_bad = np.array([1])
    try:
        cp2.perform_algebraic_surgery(x_bad)
    except IsotropicError as e:
        print(f"TOPOLOGICAL ERROR CAUGHT: {e}")
        print("CP^2 is positive-definite. It has no isotropic vectors, so we CANNOT surger it into a sphere.")

if __name__ == "__main__":
    run_tutorial_03()
