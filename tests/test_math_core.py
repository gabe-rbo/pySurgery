import numpy as np
import scipy.sparse as sp
import pytest
from pysurgery.core.math_core import get_sparse_snf_diagonal, get_snf_diagonal, smith_normal_form, extended_gcd

def test_extended_gcd():
    # GCD of 10 and 15 is 5. 10*x + 15*y = 5.
    g, x, y = extended_gcd(10, 15)
    assert g == 5
    assert 10*x + 15*y == 5

    g, x, y = extended_gcd(17, 13)
    assert g == 1
    assert 17*x + 13*y == 1

def test_smith_normal_form_exact():
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
    B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    sp_B = sp.csr_matrix(B)
    diag = get_sparse_snf_diagonal(sp_B)
    assert diag.tolist() == [1, 3]


def test_get_sparse_snf_diagonal_exact_mode_warns_exact_python_fallback(monkeypatch):
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
    A = np.array([[3, 0], [0, 3]], dtype=np.int64)
    diag = get_snf_diagonal(A)
    assert sorted(diag.tolist()) == [3, 3]


def test_get_snf_diagonal_trailing_zero_invariants():
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)
    diag = get_snf_diagonal(A)
    assert diag.tolist() == [1, 3]

