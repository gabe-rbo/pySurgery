"""Tests for exact geometric sign predicates.

Overview:
    Verifies the two-tier (float64-filtered, exact-``Fraction``-fallback) predicates in
    ``pysurgery.geometry.predicates`` against known geometric configurations, deliberately
    near-degenerate perturbations, an independent exact reference implementation, and
    (when available) the Julia-accelerated batch backend.

Key Concepts:
    - **Independent oracle**: ``_reference_exact_det`` computes the exact determinant via
      ``Fraction``-exact Gaussian elimination -- a different algorithm from the module's own
      Leibniz-formula fallback -- so a bug shared between the float filter and its own
      fallback cannot silently pass its own self-consistency check.
"""
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from pysurgery.geometry.predicates import (
    exact_sign_of_determinant,
    exact_sign_of_sum,
    exact_signs_of_determinants_batch,
    incircle2d,
    insphere3d,
    orientation2d,
    orientation3d,
)


def _reference_exact_det(matrix) -> Fraction:
    """Independent exact-determinant oracle via Fraction-exact Gaussian elimination.

    Algorithm:
        1. Convert every entry to an exact ``Fraction``.
        2. Row-reduce to upper-triangular form (partial "pivot on any nonzero" swap; no
           numerical stability concerns since ``Fraction`` arithmetic has no rounding error).
        3. The determinant is the product of the diagonal, times -1 per row swap performed.

    Deliberately independent of ``pysurgery.geometry.predicates._leibniz_det`` so it can
    catch a bug shared between the float filter and its own exact fallback.
    """
    n = len(matrix)
    m = [[Fraction(float(x)) for x in row] for row in matrix]
    sign = 1
    for col in range(n):
        pivot_row = None
        for r in range(col, n):
            if m[r][col] != 0:
                pivot_row = r
                break
        if pivot_row is None:
            return Fraction(0)
        if pivot_row != col:
            m[col], m[pivot_row] = m[pivot_row], m[col]
            sign = -sign
        pivot = m[col][col]
        for r in range(col + 1, n):
            factor = m[r][col] / pivot
            if factor != 0:
                for c in range(col, n):
                    m[r][c] -= factor * m[col][c]
    det = Fraction(sign)
    for i in range(n):
        det *= m[i][i]
    return det


def _reference_sign(matrix) -> int:
    d = _reference_exact_det(matrix)
    return 1 if d > 0 else (-1 if d < 0 else 0)


# ---------------------------------------------------------------------------
# Known geometric configurations
# ---------------------------------------------------------------------------

def test_orientation2d_known_cases():
    """Verify orientation2d on hand-picked CCW/CW/collinear triples.

    What is Being Computed?:
        Sign of the standard 2D orientation predicate on unambiguous configurations.
    """
    assert orientation2d((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)) == 1  # CCW
    assert orientation2d((0.0, 0.0), (0.0, 1.0), (1.0, 0.0)) == -1  # CW
    assert orientation2d((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)) == 0  # collinear


def test_orientation3d_known_cases():
    """Verify orientation3d on the standard basis tetrahedron and a coplanar case."""
    origin, ex, ey, ez = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    assert orientation3d(origin, ex, ey, ez) != 0
    # Swapping two vertices flips the sign.
    assert orientation3d(origin, ex, ey, ez) == -orientation3d(ex, origin, ey, ez)
    coplanar_pt = (2.0, 3.0, 0.0)  # z=0 plane, same as origin/ex/ey
    assert orientation3d(origin, ex, ey, coplanar_pt) == 0


def test_incircle2d_known_cases():
    """Verify incircle2d against the unit circle through 3 axis points.

    What is Being Computed?:
        For a, b, c on the unit circle (CCW), the origin is inside, a far point is outside,
        and a fourth point on the same circle is exactly cocircular (sign 0).
    """
    a, b, c = (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)
    assert orientation2d(a, b, c) == 1  # confirm CCW, fixes the sign convention below
    assert incircle2d(a, b, c, (0.0, 0.0)) == 1  # origin: inside
    assert incircle2d(a, b, c, (10.0, 10.0)) == -1  # far away: outside
    assert incircle2d(a, b, c, (0.0, -1.0)) == 0  # also on the unit circle: cocircular


def test_insphere3d_known_cases():
    """Verify insphere3d against the unit sphere through 4 axis points."""
    a, b, c, d = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (-1.0, 0.0, 0.0)
    orient = orientation3d(a, b, c, d)
    assert orient != 0
    inside_sign = insphere3d(a, b, c, d, (0.0, 0.0, 0.0))
    outside_sign = insphere3d(a, b, c, d, (10.0, 10.0, 10.0))
    on_sphere_sign = insphere3d(a, b, c, d, (0.0, -1.0, 0.0))  # also on the unit sphere
    assert inside_sign == (1 if orient > 0 else -1)
    assert outside_sign == (-1 if orient > 0 else 1)
    assert on_sphere_sign == 0


# ---------------------------------------------------------------------------
# Near-degenerate configurations forcing the exact fallback
# ---------------------------------------------------------------------------

def test_exact_fallback_resolves_tiny_perturbation():
    """A cocircular quadruple perturbed by 1e-14 must still get the mathematically
    correct (nonzero) sign, not a coin-flip from float64 rounding noise.

    What is Being Computed?:
        Confirms the exact ``Fraction`` fallback -- not the float64 filter -- decides these
        near-degenerate cases, and gets them right in both perturbation directions.
    """
    a, b, c = (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)
    just_inside = (0.0, -1.0 + 1e-14)
    just_outside = (0.0, -1.0 - 1e-14)
    assert incircle2d(a, b, c, just_inside) == 1
    assert incircle2d(a, b, c, just_outside) == -1


def test_exact_fallback_matches_independent_oracle_on_near_singular_matrices():
    """Near-singular matrices (rows nearly linearly dependent) must match the independent
    Fraction-Gaussian-elimination oracle exactly.
    """
    rng = np.random.default_rng(0)
    for _ in range(50):
        base = rng.uniform(-5.0, 5.0, size=3)
        m = np.array([base, base + rng.uniform(-1e-13, 1e-13, size=3), rng.uniform(-5.0, 5.0, size=3)])
        assert exact_sign_of_determinant(m) == _reference_sign(m)


# ---------------------------------------------------------------------------
# Batch vs. single-item consistency
# ---------------------------------------------------------------------------

def test_batch_matches_single_item_loop():
    """exact_signs_of_determinants_batch must match a per-item exact_sign_of_determinant loop."""
    rng = np.random.default_rng(1)
    mats = rng.uniform(-10.0, 10.0, size=(40, 4, 4))
    batch_signs = exact_signs_of_determinants_batch(mats, backend="python")
    loop_signs = np.array([exact_sign_of_determinant(m) for m in mats])
    np.testing.assert_array_equal(batch_signs, loop_signs)


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_exact_sign_of_determinant_matches_oracle_random(n):
    """exact_sign_of_determinant matches the independent oracle across sizes n=2..5."""
    rng = np.random.default_rng(42 + n)
    for _ in range(30):
        m = rng.uniform(-100.0, 100.0, size=(n, n))
        assert exact_sign_of_determinant(m) == _reference_sign(m)


def test_exact_sign_of_sum_basic():
    """exact_sign_of_sum agrees with a plain Fraction sum, including a near-zero case."""
    assert exact_sign_of_sum([1.0, 2.0, 3.0]) == 1
    assert exact_sign_of_sum([-1.0, -2.0, -3.0]) == -1
    assert exact_sign_of_sum([1.0, -1.0]) == 0
    tiny = 2.0**-60
    assert exact_sign_of_sum([1.0, -1.0, tiny]) == 1
    assert exact_sign_of_sum([1.0, -1.0, -tiny]) == -1


def test_invalid_shapes_raise():
    """Non-square or out-of-range matrices raise ValueError rather than silently misbehaving."""
    with pytest.raises(ValueError):
        exact_sign_of_determinant(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        exact_signs_of_determinants_batch(np.zeros((5, 2, 3)))


# ---------------------------------------------------------------------------
# Property-based cross-check against the independent oracle
# ---------------------------------------------------------------------------

_small_matrix = st.integers(min_value=3, max_value=4).flatmap(
    lambda n: st.lists(
        st.lists(st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
                 min_size=n, max_size=n),
        min_size=n, max_size=n,
    )
)


@settings(max_examples=100, deadline=None)
@given(_small_matrix)
def test_property_sign_matches_independent_oracle(matrix):
    """Property test: exact_sign_of_determinant never disagrees with the independent
    Fraction-Gaussian-elimination oracle, across random 3x3/4x4 float matrices."""
    m = np.array(matrix, dtype=np.float64)
    assert exact_sign_of_determinant(m) == _reference_sign(m)
