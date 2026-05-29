"""Tests for Rational Surgery Theory and Localization.

Overview:
    This suite validates rational and p-local L-group computations, 
    verifies exact eigenvalue-based signature logic, and ensures the 
    Chinese Remainder Theorem reconstructs identical SNF invariants.
"""

import pytest
import numpy as np
from pysurgery.algebra.intersection_forms import IntersectionForm
from pysurgery.manifolds.rational_surgery import (
    compute_l_group_rational,
    compute_l_group_p_local,
    prime_local_obstruction_report
)
from pysurgery.bridge.julia_bridge import julia_engine

# Require Julia for the p-local and exact CRT paths
pytestmark = pytest.mark.skipif(
    not julia_engine.available, 
    reason="Julia engine required for p-local tests and CRT reconstruction."
)

def test_compute_l_group_rational():
    """Verify rational tensor reductions and exact signature tracking."""
    # Build a diagonal matrix
    mat = np.diag([2, -2, 3])
    form = IntersectionForm(matrix=mat, dimension=4)
    
    rat = compute_l_group_rational(dimension=4, pi="1", form=form)
    
    assert rat.rank_q == 3
    # Pos = 2 (2, 3), Neg = 1 (-2) -> signature = 1
    assert rat.signature == 1
    assert rat.exact is True


def test_compute_l_group_p_local():
    """Verify p-local bounds and rank drops at specific primes."""
    # Matrix with SNF = diag(1, 2, 6)
    mat = np.diag([2, -2, 3])
    form = IntersectionForm(matrix=mat, dimension=4)
    
    # At p=2
    p2 = compute_l_group_p_local(dimension=4, pi="1", prime_p=2, form=form)
    # over Z/2Z, diag(2, -2, 3) -> diag(0, 0, 1). Rank mod 2 is 1.
    assert p2.rank_mod_p == 1
    # p-adic diagonal at p=2 for SNF(1, 2, 6) is [1, 2, 2]
    assert p2.p_adic_diagonal == [1, 2, 2]
    
    # At p=3
    p3 = compute_l_group_p_local(dimension=4, pi="1", prime_p=3, form=form)
    # over Z/3Z, diag(2, -2, 3) -> diag(2, 1, 0). Rank mod 3 is 2.
    assert p3.rank_mod_p == 2
    # p-adic diagonal at p=3 for SNF(1, 2, 6) is [1, 1, 3]
    assert p3.p_adic_diagonal == [1, 1, 3]


def test_crt_reconstruction():
    """Verify the Chinese Remainder Theorem reconstruction matches standard exact integer SNF output."""
    mat = np.diag([2, -2, 3])
    form = IntersectionForm(matrix=mat, dimension=4)
    
    # Generate the report spanning minimal primes
    report = prime_local_obstruction_report(dimension=4, pi="1", form=form, primes=[2, 3, 5])
    
    # The reconstructed diagonal should precisely match the integral SNF of diag(2, -2, 3)
    assert report.reconstructed_integral_diagonal == [1, 2, 6]
