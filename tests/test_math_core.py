"""Unit tests for core algebraic algorithms, including SNF and GCD.

Overview:
    This suite validates the fundamental exact algebraic operations required for 
    homology computation. It covers the Extended Euclidean Algorithm (GCD) and 
    the Smith Normal Form (SNF) for both dense and sparse matrices.

Key Concepts:
    - **Smith Normal Form (SNF)**: A canonical form for matrices over a PID (like ℤ).
    - **Invariant Factors**: The diagonal elements of the SNF, which reveal the structure of the module.
    - **Extended GCD**: Computing d = gcd(a, b) and coefficients x, y such that ax + by = d.
"""
import numpy as np
import scipy.sparse as sp
import pytest
from pysurgery.core.math_core import (
    get_sparse_snf_diagonal,
    get_snf_diagonal,
    smith_normal_form,
    extended_gcd,
)


def test_extended_gcd():
    """Verify the Extended Euclidean Algorithm for integer pairs.

    What is Being Computed?:
        The GCD d and Bezout coefficients (x, y) such that ax + by = d.

    Algorithm:
        1. Test with (10, 15) -> expect gcd=5.
        2. Test with coprime (17, 13) -> expect gcd=1.
        3. Assert the Bezout identity holds in both cases.
    """
    # GCD of 10 and 15 is 5. 10*x + 15*y = 5.
    g, x, y = extended_gcd(10, 15)
    assert g == 5
    assert 10 * x + 15 * y == 5

    g, x, y = extended_gcd(17, 13)
    assert g == 1
    assert 17 * x + 13 * y == 1


def test_smith_normal_form_exact():
    """Verify the exact SNF computation for small dense matrices.

    What is Being Computed?:
        The canonical Smith Normal Form of a matrix over ℤ.

    Algorithm:
        1. Compute SNF of a 2x2 matrix with determinant -4.
        2. Compute SNF of a singular 3x3 matrix.
        3. Assert the diagonal elements are the correct invariant factors.

    Preserved Invariants:
        - Invariant factors (canonical).
        - Determinant (up to sign).
    """
    A = np.array([[2, 4], [4, 6]])
    # det = -4. Invariant factors should be 2, 2.
    snf = smith_normal_form(A)
    assert abs(snf[0, 0]) == 2
    assert abs(snf[1, 1]) == 2

    B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    snf = smith_normal_form(B)
    diag = [abs(snf[i, i]) for i in range(3)]
    assert diag[0] == 1
    assert diag[1] == 3
    assert diag[2] == 0


def test_get_sparse_snf_diagonal():
    """Verify the invariant factor extraction for sparse matrices.

    What is Being Computed?:
        The non-zero diagonal elements of the SNF of a sparse matrix.

    Algorithm:
        1. Define a 3x3 singular matrix as a CSR sparse matrix.
        2. Extract the SNF diagonal.
        3. Assert the non-zero factors are [1, 3].
    """
    B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    sp_B = sp.csr_matrix(B)
    diag = get_sparse_snf_diagonal(sp_B)
    assert diag.tolist() == [1, 3]


def test_get_sparse_snf_diagonal_exact_mode_warns_exact_python_fallback(monkeypatch):
    """Ensure a warning is issued when the Julia backend fails and fallback to Python/SymPy occurs.

    Algorithm:
        1. Mock the Julia backend to raise an error during SNF computation.
        2. Call get_sparse_snf_diagonal with allow_approx=False.
        3. Assert that a UserWarning about Python fallback is raised.
    """
    from pysurgery.bridge import julia_bridge

    class _FakeJulia:
        available = True

        def compute_sparse_snf(self, *args, **kwargs):
            raise RuntimeError("backend-fail")

    monkeypatch.setattr(julia_bridge, "julia_engine", _FakeJulia())
    A = sp.csr_matrix(np.array([[2]], dtype=np.int64))
    with pytest.warns(UserWarning, match="exact Python/SymPy SNF"):
        diag = get_sparse_snf_diagonal(A, allow_approx=False)
    assert diag.tolist() == [2]


def test_get_snf_diagonal():
    """Verify the high-level API for invariant factor extraction on dense matrices."""
    A = np.array([[3, 0], [0, 3]], dtype=np.int64)
    diag = get_snf_diagonal(A)
    assert sorted(diag.tolist()) == [3, 3]


def test_get_snf_diagonal_trailing_zero_invariants():
    """Ensure that trailing zero invariant factors (corresponding to kernel) are correctly identified and omitted from the diagonal report if needed.

    What is Being Computed?:
        The non-zero invariant factors of a singular matrix.
    """
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)
    diag = get_snf_diagonal(A)
    assert diag.tolist() == [1, 3]
