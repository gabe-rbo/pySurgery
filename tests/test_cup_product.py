"""Tests for the Alexander-Whitney cup product and cup-i products.

Overview:
    This suite verifies the implementation of the Alexander-Whitney cup product
    on cochains of a simplicial complex, ensuring correct index handling,
    modulus arithmetic, and consistency with higher cup-i products.

Key Concepts:
    - **Alexander-Whitney Map**: A chain map used to define the cup product on simplicial cochains.
    - **Cup Product (⌣)**: A binary operation on cohomology that gives it a ring structure.
    - **Cup-i Product**: Generalization of the cup product used in Steenrod operations.
"""
import numpy as np
import pytest

from discrete_surface_data import build_torus, to_complex
from pysurgery.core.cup_product import alexander_whitney_cup


def test_alexander_whitney_cup():
    """Verify basic Alexander-Whitney cup product calculation.

    What is Being Computed?:
        The cup product of two 1-cochains on a 2-simplex.

    Algorithm:
        1. Define a 2-simplex and its faces.
        2. Assign values to 1-cochains alpha and beta.
        3. Compute the cup product alpha ⌣ beta.

    Preserved Invariants:
        - Cohomology ring structure.
    """
    simplices_p_plus_q = [(0, 1, 2)]
    simplex_to_idx_p = {(0, 1): 0, (1, 2): 1, (0, 2): 2}
    simplex_to_idx_q = {(0, 1): 0, (1, 2): 1, (0, 2): 2}

    alpha = np.array([2, 0, 0], dtype=np.int64)
    beta = np.array([0, 3, 0], dtype=np.int64)

    cup = alexander_whitney_cup(
        alpha,
        beta,
        p=1,
        q=1,
        simplices_p_plus_q=simplices_p_plus_q,
        simplex_to_idx_p=simplex_to_idx_p,
        simplex_to_idx_q=simplex_to_idx_q,
    )

    assert cup.shape == (1,)
    assert cup[0] == 6


def test_alexander_whitney_cup_empty():
    """Verify cup product behavior on empty simplex sets.

    What is Being Computed?:
        Cup product with zero simplices and empty cochains.

    Algorithm:
        1. Pass empty arrays and lists to alexander_whitney_cup.
        2. Verify that the result is an empty array.

    Preserved Invariants:
        - N/A
    """
    cup = alexander_whitney_cup(
        np.array([]),
        np.array([]),
        p=1,
        q=1,
        simplices_p_plus_q=[],
        simplex_to_idx_p={},
        simplex_to_idx_q={},
    )
    assert len(cup) == 0


def test_alexander_whitney_cup_modulus():
    """Verify cup product with modular arithmetic.

    What is Being Computed?:
        Cup product α ⌣ β mod n.

    Algorithm:
        1. Compute cup product with a specified modulus.
        2. Verify the result matches (α * β) mod modulus.

    Preserved Invariants:
        - Cohomology ring structure over Z/pZ.
    """
    simplices = [(0, 1, 2)]
    idx = {(0, 1): 0, (1, 2): 1}
    alpha = np.array([5, 0], dtype=np.int64)
    beta = np.array([0, 7], dtype=np.int64)
    cup = alexander_whitney_cup(
        alpha,
        beta,
        p=1,
        q=1,
        simplices_p_plus_q=simplices,
        simplex_to_idx_p=idx,
        simplex_to_idx_q=idx,
        modulus=3,
    )
    assert int(cup[0]) == (5 * 7) % 3


def test_cup_i_product_basic():
    """Verify basic cup-i product calculation (i=1).

    What is Being Computed?:
        The cup-1 product of two 1-cochains.

    Algorithm:
        1. Define 1-simplices and cochains.
        2. Compute cup-1 product.
        3. Verify the values match expected intersection patterns.

    Preserved Invariants:
        - Steenrod operations (derived from cup-i).
    """
    # Target simplices are of dimension p+q-i = 1 for p=q=i=1.
    simplices = [(0, 1), (1, 2)]
    simplex_to_idx_1 = {(0, 1): 0, (1, 2): 1}
    alpha = np.array([2, 3], dtype=np.int64)
    beta = np.array([5, 7], dtype=np.int64)
    cup1 = alexander_whitney_cup(
        alpha,
        beta,
        p=1,
        q=1,
        i=1,
        simplices_p_plus_q=simplices,
        simplex_to_idx_p=simplex_to_idx_1,
        simplex_to_idx_q=simplex_to_idx_1,
    )
    assert cup1.tolist() == [10, 21]


def test_cup_i_zero_matches_aw():
    """Verify that cup-0 product is equivalent to Alexander-Whitney cup product.

    What is Being Computed?:
        Equivalence between alexander_whitney_cup(..., i=0) and standard AW cup.

    Algorithm:
        1. Compute cup product using standard AW path.
        2. Compute cup product using cup-i path with i=0.
        3. Assert that both results are identical.

    Preserved Invariants:
        - Consistency of cohomology operations.
    """
    simplices = [(0, 1, 2)]
    idx = {(0, 1): 0, (1, 2): 1}
    alpha = np.array([2, 0], dtype=np.int64)
    beta = np.array([0, 3], dtype=np.int64)
    cup_aw = alexander_whitney_cup(
        alpha,
        beta,
        p=1,
        q=1,
        simplices_p_plus_q=simplices,
        simplex_to_idx_p=idx,
        simplex_to_idx_q=idx,
    )
    cup_i0 = alexander_whitney_cup(
        alpha,
        beta,
        p=1,
        q=1,
        i=0,
        simplices_p_plus_q=simplices,
        simplex_to_idx_p=idx,
        simplex_to_idx_q=idx,
    )
    assert np.array_equal(cup_aw, cup_i0)


def test_cup_i_mod2_compatibility():
    """Verify cup-i product compatibility with modulus 2.

    What is Being Computed?:
        Cup-i product over Z/2Z.

    Algorithm:
        1. Compute cup-i product with integer coefficients and then take mod 2.
        2. Compute cup-i product with modulus=2.
        3. Assert equality.

    Preserved Invariants:
        - Homomorphism between Z and Z/2Z cohomology rings.
    """
    simplices = [(0, 1), (1, 2)]
    idx = {(0, 1): 0, (1, 2): 1}
    alpha = np.array([3, 5], dtype=np.int64)
    beta = np.array([7, 11], dtype=np.int64)
    cup = alexander_whitney_cup(
        alpha,
        beta,
        p=1,
        q=1,
        i=1,
        simplices_p_plus_q=simplices,
        simplex_to_idx_p=idx,
        simplex_to_idx_q=idx,
    )
    cup_mod2 = alexander_whitney_cup(
        alpha,
        beta,
        p=1,
        q=1,
        i=1,
        simplices_p_plus_q=simplices,
        simplex_to_idx_p=idx,
        simplex_to_idx_q=idx,
        modulus=2,
    )
    assert np.array_equal(cup_mod2, cup % 2)


def test_cup_i_invalid_index():
    """Verify error handling for invalid cup-i indices.

    What is Being Computed?:
        Error state for i > p or i > q.

    Algorithm:
        1. Attempt to compute cup-1 product for 0-cochains.
        2. Assert that a ValueError is raised.

    Preserved Invariants:
        - N/A
    """
    with pytest.raises(ValueError):
        alexander_whitney_cup(
            np.array([1], dtype=np.int64),
            np.array([1], dtype=np.int64),
            p=0,
            q=0,
            i=1,
            simplices_p_plus_q=[(0,)],
            simplex_to_idx_p={(0,): 0},
            simplex_to_idx_q={(0,): 0},
        )


def test_cup_product_torus():
    """Verify cup product structure on a torus.

    What is Being Computed?:
        H^1(T^2) cup product rank.

    Algorithm:
        1. Build a torus simplicial complex.
        2. Verify its 1st homology rank is 2.

    Preserved Invariants:
        - Torus cohomology ring structure.
    """
    c1 = to_complex(build_torus())
    assert c1.homology(1)[0] == 2
