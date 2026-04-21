import numpy as np
import pytest

from discrete_surface_data import build_torus, to_complex
from pysurgery.core.cup_product import alexander_whitney_cup


def test_alexander_whitney_cup():
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
    c1 = to_complex(build_torus())
    assert c1.homology(1)[0] == 2
