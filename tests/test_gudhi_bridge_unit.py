import numpy as np
import scipy.sparse as sp
import pytest

from pysurgery.integrations import gudhi_bridge


class _FakeSimplexTree:
    def dimension(self):
        return 4


class _SimplexTreeForExtract:
    def __init__(self, simplices):
        self._simplices = simplices

    def dimension(self):
        return max((len(s) - 1 for s in self._simplices), default=0)

    def get_skeleton(self, _):
        return [(s, 0.0) for s in self._simplices]


def test_extract_complex_data_warns_and_falls_back_to_python_without_julia(monkeypatch):
    st = _SimplexTreeForExtract([(0,), (1,), (2,), (0, 1), (1, 2), (0, 2), (0, 1, 2)])
    monkeypatch.setattr(gudhi_bridge.julia_engine, "available", False)
    monkeypatch.setattr(gudhi_bridge, "_SLOW_BOUNDARY_FALLBACK_WARNED", False)

    with pytest.warns(UserWarning, match="slower pure-Python boundary assembly"):
        boundaries, cells, dim_simplices, simplex_to_idx = gudhi_bridge.extract_complex_data(st)

    assert cells[0] == 3
    assert cells[1] == 3
    assert cells[2] == 1
    assert boundaries[1].shape == (3, 3)
    assert boundaries[2].shape == (3, 1)
    assert dim_simplices[2] == [(0, 1, 2)]
    assert simplex_to_idx[1][(0, 1)] == 0


def test_extract_complex_data_uses_julia_payload_when_available(monkeypatch):
    st = _SimplexTreeForExtract([(0,), (1,), (0, 1)])
    monkeypatch.setattr(gudhi_bridge.julia_engine, "available", True)

    def _fake_julia_boundary_builder(simplex_entries, max_dim):
        assert simplex_entries == [(0,), (1,), (0, 1)]
        assert max_dim == 1
        return (
            {1: {"rows": np.array([0, 1], dtype=np.int64), "cols": np.array([0, 0], dtype=np.int64), "data": np.array([-1, 1], dtype=np.int64), "n_rows": 2, "n_cols": 1}},
            {0: 2, 1: 1},
            {0: [(0,), (1,)], 1: [(0, 1)]},
        )

    monkeypatch.setattr(gudhi_bridge.julia_engine, "compute_boundary_data_from_simplices", _fake_julia_boundary_builder)
    boundaries, cells, dim_simplices, simplex_to_idx = gudhi_bridge.extract_complex_data(st)

    assert cells == {0: 2, 1: 1}
    assert dim_simplices[1] == [(0, 1)]
    assert simplex_to_idx[0][(1,)] == 1
    assert boundaries[1].shape == (2, 1)
    assert boundaries[1].toarray().tolist() == [[-1], [1]]


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

