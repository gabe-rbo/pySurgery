"""Tests for K-Theory and Chern Character Computation Engine.

Overview:
    This suite validates algebraic K-theory computations (K_0, K_1) on specific
    group ring structures. It also tests the exact evaluation of Adams operations
    via cup products, and the exact Z/Q arithmetic of the Chern character map.

Key Concepts:
    - **K_0**: Reduced projective class group.
    - **K_1**: Includes the Whitehead group.
    - **Adams Operations (psi^k)**: Symmetric power sums of Chern classes.
    - **Chern Character**: Rational combinations of Chern classes.
"""

import numpy as np
from fractions import Fraction

from discrete_surface_data import build_torus, to_complex
from pysurgery.topology.fundamental_group import FundamentalGroup
from pysurgery.algebra.k_theory import (
    compute_k0_group,
    compute_k1_group,
    adams_operation,
    chern_character,
    atiyah_hirzebruch_k_theory_obstruction,
)


def test_compute_k0_group():
    """Verify algebraic K_0 computations."""
    # Trivial group
    pi1_1 = FundamentalGroup(generators=[], relations=[])
    k0_1 = compute_k0_group(pi1_1)
    assert k0_1.rank == 0
    assert k0_1.exact is True

    # Infinite cyclic group Z
    pi1_z = FundamentalGroup(generators=["a"], relations=[])
    k0_z = compute_k0_group(pi1_z)
    assert k0_z.rank == 0
    assert k0_z.exact is True

    # Finite cyclic group Z_2
    pi1_z2 = FundamentalGroup(generators=["a"], relations=[["a", "a"]])
    k0_z2 = compute_k0_group(pi1_z2)
    assert k0_z2.rank == -1
    assert k0_z2.exact is False
    assert k0_z2.computable is False

    # Free abelian / Free group (Z^2)
    pi1_f2 = FundamentalGroup(generators=["a", "b"], relations=[])
    k0_f2 = compute_k0_group(pi1_f2)
    assert k0_f2.rank == 0
    assert k0_f2.exact is False  # Relies on Farrell-Jones conjecture


def test_compute_k1_group():
    """Verify algebraic K_1 computations."""
    print("Starting test_compute_k1_group")
    # Trivial group -> Wh is 0, abelianization free rank 0 -> K1 rank = 0
    pi1_1 = FundamentalGroup(generators=[], relations=[])
    print("Testing pi1_1")
    k1_1 = compute_k1_group(pi1_1)
    assert k1_1.rank == 0
    assert k1_1.exact is True

    print("Testing pi1_z")
    # Infinite cyclic group Z -> Wh is 0, abelianization free rank 1 -> K1 rank = 1
    pi1_z = FundamentalGroup(generators=["a"], relations=[])
    k1_z = compute_k1_group(pi1_z)
    assert k1_z.rank == 1

    print("Testing pi1_t2")
    # Torus ZxZ -> Wh is 0 (Farrell-Jones), abelianization free rank 2 -> K1 rank = 2
    pi1_t2 = FundamentalGroup(generators=["a", "b"], relations=[["a", "b", "a^-1", "b^-1"]])
    k1_t2 = compute_k1_group(pi1_t2)
    assert k1_t2.rank == 2
    assert k1_t2.exact is False  # Depends on Wh exactness which uses FJ for non-trivial
    print("Done test_compute_k1_group")


def test_adams_operation_and_chern_character():
    """Verify Adams operations and exact Z/Q arithmetic for the Chern character map."""
    sc = to_complex(build_torus())

    # c1 lives in degree 2
    c1 = np.ones(sc.count_simplices(2), dtype=np.int64)
    classes = [c1]

    # Test psi^1 = N_1 = c_1
    nk1 = adams_operation(classes, k=1, base_complex=sc)
    assert np.array_equal(nk1, c1)

    # Test exact Chern character ch_1 = c_1 / 1!
    ch = chern_character(classes, base_complex=sc)
    assert len(ch) >= 1
    assert ch[0][0] == Fraction(1, 1)
    
    # Test k=2 behavior (should gracefully handle missing simplices above manifold dimension)
    nk2 = adams_operation(classes, k=2, base_complex=sc)
    # The degree of N_2 is 2*2=4. The Torus is 2D, so count_simplices(4) == 0
    assert len(nk2) == 0
    assert len(ch) == 1 # Since len(classes) is 1


def test_ahss_obstruction():
    """Verify Atiyah-Hirzebruch spectral sequence d_3 obstruction."""
    sc = to_complex(build_torus())
    
    # For a 2D Torus, we start with a 0-cochain
    alpha = np.ones(sc.count_simplices(0), dtype=np.int64)
    
    # d3: H^0 -> H^3. Since Torus is 2D, result will be empty.
    obs = atiyah_hirzebruch_k_theory_obstruction(alpha, p=0, base_complex=sc)
    
    assert obs.exact is True
    assert "Sq^3" in obs.description
    assert len(obs.d3_obstruction) == 0
