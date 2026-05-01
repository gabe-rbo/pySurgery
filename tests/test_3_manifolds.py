"""Tests for 3-manifold topological invariants and homeomorphism detection.

Overview:
    This suite verifies the computation of homology groups and homeomorphism
    candidates for various 3-manifolds, including Lens spaces, homology spheres,
    and product spaces.

Key Concepts:
    - **Lens Spaces**: L(p,q) spaces defined by cyclic group actions on S³.
    - **Poincaré Homology Sphere**: A manifold with the same homology as S³ but non-trivial π₁.
    - **Homeomorphism Testing**: Heuristic and exact methods for 3-manifold classification.
"""
import numpy as np
import scipy.sparse as sp
from pysurgery.core.complexes import ChainComplex
from pysurgery.homeomorphism import analyze_homeomorphism_3d


def test_lens_space_L31_homology():
    """Verify homology groups for the Lens Space L(3,1).

    What is Being Computed?:
        Homology groups H_n(L(3,1); ℤ) from a CW complex presentation.

    Algorithm:
        1. Construct a ChainComplex with boundaries representing L(3,1).
        2. Compute H_0, H_1, H_2, and H_3 using Smith Normal Form.

    Preserved Invariants:
        - H_1(L(3,1)) = ℤ/3ℤ (torsion).
        - H_3(L(3,1)) = ℤ (orientation).
    """
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    d2 = sp.csr_matrix(np.array([[3]], dtype=np.int64))
    d3 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))

    cc = ChainComplex(
        boundaries={1: d1, 2: d2, 3: d3},
        dimensions=[0, 1, 2, 3],
        cells={0: 1, 1: 1, 2: 1, 3: 1},
    )

    assert cc.homology(0) == (1, [])

    from pysurgery.bridge.julia_bridge import julia_engine

    if julia_engine.available:
        assert cc.homology(1) == (0, [3])
    else:
        assert cc.homology(1)[0] == 0

    assert cc.homology(2) == (0, [])
    assert cc.homology(3) == (1, [])


def test_poincare_homology_sphere():
    """Test homeomorphism detection between Poincaré Homology Sphere and S³.

    What is Being Computed?:
        Homeomorphism candidates using 3D-specific invariants.

    Algorithm:
        1. Define ChainComplexes for PHS and S³ (here simplified to homology spheres).
        2. Call analyze_homeomorphism_3d to determine if they are homeomorphic.

    Preserved Invariants:
        - Homology groups H_*(PHS) ≅ H_*(S³).
        - Fundamental groups π₁(PHS) ≇ π₁(S³).
    """
    d1 = sp.csr_matrix(np.zeros((1, 0), dtype=np.int64))
    d2 = sp.csr_matrix(np.zeros((0, 0), dtype=np.int64))
    d3 = sp.csr_matrix(np.zeros((0, 1), dtype=np.int64))
    cc_phs = ChainComplex(
        boundaries={1: d1, 2: d2, 3: d3},
        dimensions=[0, 1, 2, 3],
        cells={0: 1, 1: 0, 2: 0, 3: 1},
    )
    cc_s3 = ChainComplex(
        boundaries={1: d1, 2: d2, 3: d3},
        dimensions=[0, 1, 2, 3],
        cells={0: 1, 1: 0, 2: 0, 3: 1},
    )

    is_homeo, reason = analyze_homeomorphism_3d(cc_phs, cc_s3)
    assert not is_homeo
    assert "INCONCLUSIVE: Both are homology-sphere candidates" in reason

def test_S2xS1_homology():
    """Verify homology groups for the product space S² × S¹.

    What is Being Computed?:
        Homology groups H_n(S² × S¹; ℤ).

    Algorithm:
        1. Construct a minimal CW structure: one 0-cell, one 1-cell, one 2-cell, and one 3-cell.
        2. Set all boundary operators to zero.
        3. Verify H_0=ℤ, H_1=ℤ, H_2=ℤ, H_3=ℤ.

    Preserved Invariants:
        - Betti numbers: β_0=1, β_1=1, β_2=1, β_3=1.
        - Euler characteristic χ(S² × S¹) = 0.
    """
    # Smallest CW structure: one 0-cell, one 1-cell, one 2-cell, one 3-cell.
    # Boundary operators are all zero.
    cc = ChainComplex(
        boundaries={1: sp.csr_matrix((1, 1)), 2: sp.csr_matrix((1, 1)), 3: sp.csr_matrix((1, 1))},
        dimensions=[0, 1, 2, 3],
        cells={0: 1, 1: 1, 2: 1, 3: 1},
    )
    
    assert cc.homology(0) == (1, [])
    assert cc.homology(1) == (1, [])
    assert cc.homology(2) == (1, [])
    assert cc.homology(3) == (1, [])
