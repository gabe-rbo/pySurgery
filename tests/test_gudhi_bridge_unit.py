import numpy as np
import scipy.sparse as sp
import pytest

from pysurgery.integrations import gudhi_bridge


class _FakeSimplexTree:
    def dimension(self):
        return 4


def test_simplex_tree_intersection_form_exact_mode_raises_without_exact_backend(monkeypatch):
    boundaries = {4: sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))}
    cells = {2: 1, 4: 1}
    dim_simplices = {4: [(0, 1, 2, 3, 4)]}
    simplex_to_idx = {2: {(0, 1, 2): 0, (1, 2, 3): 0, (2, 3, 4): 0}}

    class _FakeChainComplex:
        def __init__(self, boundaries, dimensions, coefficient_ring=None):
            self.boundaries = boundaries
            self.dimensions = dimensions

        def cohomology_basis(self, n):
            assert n == 2
            return [np.array([1], dtype=np.int64)]

    monkeypatch.setattr(gudhi_bridge, "extract_complex_data", lambda st: (boundaries, cells, dim_simplices, simplex_to_idx))
    monkeypatch.setattr(gudhi_bridge, "ChainComplex", _FakeChainComplex)
    monkeypatch.setattr(gudhi_bridge.julia_engine, "available", False)
    monkeypatch.setattr(gudhi_bridge.sympy_module, "Matrix", lambda _: (_ for _ in ()).throw(RuntimeError("force svd")))
    monkeypatch.setattr(gudhi_bridge, "alexander_whitney_cup", lambda **kwargs: np.array([1], dtype=np.int64))

    with pytest.raises(Exception):
        gudhi_bridge.simplex_tree_to_intersection_form(_FakeSimplexTree(), allow_approx=False)


def test_simplex_tree_intersection_form_handles_single_4cell_svd_fallback(monkeypatch):
    boundaries = {4: sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))}
    cells = {2: 1, 4: 1}
    dim_simplices = {4: [(0, 1, 2, 3, 4)]}
    simplex_to_idx = {2: {(0, 1, 2): 0, (1, 2, 3): 0, (2, 3, 4): 0}}

    class _FakeChainComplex:
        def __init__(self, boundaries, dimensions, coefficient_ring=None):
            self.boundaries = boundaries
            self.dimensions = dimensions

        def cohomology_basis(self, n):
            assert n == 2
            return [np.array([1], dtype=np.int64)]

    monkeypatch.setattr(gudhi_bridge, "extract_complex_data", lambda st: (boundaries, cells, dim_simplices, simplex_to_idx))
    monkeypatch.setattr(gudhi_bridge, "ChainComplex", _FakeChainComplex)
    monkeypatch.setattr(gudhi_bridge.julia_engine, "available", False)
    monkeypatch.setattr(gudhi_bridge.sympy_module, "Matrix", lambda _: (_ for _ in ()).throw(RuntimeError("force svd")))
    monkeypatch.setattr(gudhi_bridge, "alexander_whitney_cup", lambda **kwargs: np.array([1], dtype=np.int64))

    with pytest.warns(UserWarning) as rec:
        q = gudhi_bridge.simplex_tree_to_intersection_form(_FakeSimplexTree(), allow_approx=True)
    assert q.matrix.shape == (1, 1)
    assert int(q.matrix[0, 0]) == 1
    warning_text = "\n".join(str(w.message) for w in rec)
    assert "force svd" in warning_text
    assert "{e}" not in warning_text

