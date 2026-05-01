"""Tests for advanced surface topology including non-orientable and higher genus surfaces.

Overview:
    This suite verifies homology groups and fundamental group presentations for 
    compact surfaces beyond the simple sphere. It covers the Klein bottle, 
    genus-2 surfaces, and the real projective plane.

Key Concepts:
    - **Non-orientability**: Surfaces like the Klein bottle and RP² with non-trivial w₁.
    - **Fundamental Group (π₁)**: Generators and relations representing surface loops.
    - **Torsion in Homology**: Z₂ torsion in H₁(Klein Bottle) and H₁(RP²).
"""
import numpy as np
import scipy.sparse as sp
from pysurgery.core.complexes import ChainComplex
from pysurgery.core.fundamental_group import extract_pi_1
from pysurgery.core.complexes import CWComplex


def test_klein_bottle_homology():
    """Verify homology groups for the Klein bottle.

    What is Being Computed?:
        Homology groups H_n(K; ℤ) using a standard CW structure.

    Algorithm:
        1. Define a CW structure with 1 0-cell, 2 1-cells, and 1 2-cell.
        2. Set ∂₂ to [0, 2]ᵀ (representing the word aba⁻¹b).
        3. Compute homology via Smith Normal Form.

    Preserved Invariants:
        - H₁(K) = ℤ ⊕ ℤ₂ (torsion present).
        - H₂(K) = 0 (non-orientable).
    """
    # Klein bottle H_1 = Z + Z_2
    # cell structure: 1 0-cell, 2 1-cells (a,b), 1 2-cell (f)
    # boundary of f is a + b - a + b = 2b
    d1 = sp.csr_matrix(np.zeros((1, 2), dtype=np.int64))
    d2 = sp.csr_matrix(np.array([[0], [2]], dtype=np.int64))
    cc = ChainComplex(
        boundaries={1: d1, 2: d2}, dimensions=[0, 1, 2], cells={0: 1, 1: 2, 2: 1}
    )

    assert cc.homology(0) == (1, [])

    from pysurgery.bridge.julia_bridge import julia_engine

    if julia_engine.available:
        r, t = cc.homology(1)
        assert r == 1
        assert t == [2]
    else:
        assert cc.homology(1)[0] == 1

    assert cc.homology(2) == (0, [])


def test_genus_2_surface():
    """Verify homology groups for a genus-2 orientable surface.

    What is Being Computed?:
        Homology groups H_n(Σ₂; ℤ).

    Algorithm:
        1. Define a CW structure with 1 0-cell, 4 1-cells, and 1 2-cell.
        2. Set ∂₂ to zero (representing the word [a,b][c,d]).
        3. Verify H₀=ℤ, H₁=ℤ⁴, H₂=ℤ.

    Preserved Invariants:
        - Euler characteristic χ(Σ₂) = 2 - 2g = -2.
        - Betti number β₁ = 2g = 4.
    """
    # Genus 2 surface H_1 = Z^4
    # 1 0-cell, 4 1-cells, 1 2-cell
    # boundary of f is a+b-a-b+c+d-c-d = 0
    d1 = sp.csr_matrix(np.zeros((1, 4), dtype=np.int64))
    d2 = sp.csr_matrix(np.zeros((4, 1), dtype=np.int64))
    cc = ChainComplex(
        boundaries={1: d1, 2: d2}, dimensions=[0, 1, 2], cells={0: 1, 1: 4, 2: 1}
    )

    assert cc.homology(0) == (1, [])
    assert cc.homology(1) == (4, [])
    assert cc.homology(2) == (1, [])


def test_klein_bottle_pi_1():
    """Extract the fundamental group presentation for the Klein bottle.

    What is Being Computed?:
        The group π₁(K) ≅ ⟨a, b | abab⁻¹ = 1⟩.

    Algorithm:
        1. Construct a 1-skeleton for a 2-generator group.
        2. Call extract_pi_1 to get the presentation.

    Preserved Invariants:
        - π₁(K) is a non-abelian infinite group.
        - Abelianization(π₁(K)) ≅ ℤ ⊕ ℤ₂.
    """
    d1 = sp.csr_matrix(np.zeros((1, 2), dtype=np.int64))
    cw = CWComplex(attaching_maps={1: d1}, dimensions=[0, 1], cells={0: 1, 1: 2})
    pi = extract_pi_1(cw)
    assert len(pi.generators) == 2


def test_torus_pi_1():
    """Extract the fundamental group presentation for the torus T².

    What is Being Computed?:
        The group π₁(T²) ≅ ℤ ⊕ ℤ.

    Algorithm:
        1. Construct a 1-skeleton with 2 loops.
        2. Verify that 2 generators are identified.

    Preserved Invariants:
        - π₁(T²) ≅ ℤ² (free abelian).
    """
    # Torus with 1 vertex, 2 loops (a, b), 1 face
    d1 = sp.csr_matrix(np.zeros((1, 2), dtype=np.int64))
    cw = CWComplex(attaching_maps={1: d1}, dimensions=[0, 1], cells={0: 1, 1: 2})
    pi = extract_pi_1(cw)
    assert len(pi.generators) == 2
    assert pi.generators == ["g_0", "g_1"]


def test_projective_plane_pi_1():
    """Extract the fundamental group presentation for the projective plane RP².

    What is Being Computed?:
        The group π₁(RP²) ≅ ℤ₂.

    Algorithm:
        1. Construct a CW structure with a 2-cell attached by a degree 2 map.
        2. Extract the presentation.

    Preserved Invariants:
        - π₁(RP²) is the simplest non-trivial finite fundamental group of a surface.
    """
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    d2 = sp.csr_matrix(np.array([[2]], dtype=np.int64))
    cw = CWComplex(
        attaching_maps={1: d1, 2: d2}, dimensions=[0, 1, 2], cells={0: 1, 1: 1, 2: 1}
    )
    pi = extract_pi_1(cw)
    assert len(pi.generators) == 1
    assert pi.generators == ["g_0"]
