import pytest
import numpy as np
import scipy.sparse as sp

from pysurgery.bridge.julia_bridge import JuliaBridge, julia_engine


def test_julia_bridge_singleton_identity():
    a = JuliaBridge()
    b = JuliaBridge()
    assert a is b
    assert a is julia_engine


def test_julia_bridge_require_julia_behavior():
    if julia_engine.available:
        # Should not raise when backend is available.
        julia_engine.require_julia()
    else:
        from pysurgery.core.exceptions import SurgeryError

        with pytest.raises(SurgeryError):
            julia_engine.require_julia()


def test_julia_bridge_warmup_unavailable_is_nonfatal(monkeypatch):
    monkeypatch.setattr(julia_engine, "_initialized", True, raising=False)
    monkeypatch.setattr(julia_engine, "_available", False, raising=False)
    monkeypatch.setattr(julia_engine, "error", "missing juliacall", raising=False)

    report = julia_engine.warmup()
    assert report["available"] is False
    assert report["mode"] == "full"
    assert isinstance(report["failed"], dict)


def test_julia_bridge_warmup_full_executes_and_caches(monkeypatch):
    monkeypatch.setattr(julia_engine, "_initialized", True, raising=False)
    monkeypatch.setattr(julia_engine, "_available", True, raising=False)
    monkeypatch.setattr(julia_engine, "jl", object(), raising=False)
    monkeypatch.setattr(julia_engine, "backend", object(), raising=False)
    monkeypatch.setattr(julia_engine, "_warmup_level", 0, raising=False)
    monkeypatch.setattr(julia_engine, "_warmup_report", {}, raising=False)

    calls = {"minimal": 0, "full": 0}

    def _minimal_workloads():
        return [
            ("min_probe", lambda: calls.__setitem__("minimal", calls["minimal"] + 1))
        ]

    def _full_workloads():
        return [("full_probe", lambda: calls.__setitem__("full", calls["full"] + 1))]

    monkeypatch.setattr(
        julia_engine, "_minimal_warmup_workloads", _minimal_workloads, raising=False
    )
    monkeypatch.setattr(
        julia_engine, "_full_warmup_workloads", _full_workloads, raising=False
    )

    report_first = julia_engine.warmup()
    report_second = julia_engine.warmup()

    assert calls["minimal"] == 1
    assert calls["full"] == 1
    assert report_first["available"] is True
    assert report_first["mode"] == "full"
    assert report_first["cached"] is False
    assert report_second["cached"] is True


def test_compute_normal_surface_residual_norms(monkeypatch):
    matrix = sp.csr_matrix(np.array([[1, -1, 0], [0, 1, 1]], dtype=np.int64))
    vectors = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int64)

    class _Backend:
        @staticmethod
        def normal_surface_residual_norms(rows, cols, vals, m, n, coordinate_matrix):
            mm = sp.csr_matrix((vals, (rows, cols)), shape=(m, n), dtype=np.int64)
            return np.linalg.norm((mm @ coordinate_matrix).astype(np.float64), axis=0)

    monkeypatch.setattr(julia_engine, "_initialized", True, raising=False)
    monkeypatch.setattr(julia_engine, "_available", True, raising=False)
    monkeypatch.setattr(julia_engine, "backend", _Backend(), raising=False)

    got = julia_engine.compute_normal_surface_residual_norms(matrix, vectors)
    expected = np.linalg.norm((matrix @ vectors).astype(np.float64), axis=0)
    assert np.allclose(got, expected)


def test_compute_broad_phase_pairs(monkeypatch):
    centroids = np.array([[0.0, 0.0], [0.5, 0.0], [2.0, 0.0]], dtype=np.float64)
    radii = np.array([0.6, 0.6, 0.2], dtype=np.float64)

    class _Backend:
        @staticmethod
        def embedding_broad_phase_pairs(c, r, tol):
            pairs = []
            for i in range(c.shape[0]):
                for j in range(i + 1, c.shape[0]):
                    if np.linalg.norm(c[i] - c[j]) <= (r[i] + r[j] + tol):
                        pairs.append((i, j))
            return np.asarray(pairs, dtype=np.int64)

    monkeypatch.setattr(julia_engine, "_initialized", True, raising=False)
    monkeypatch.setattr(julia_engine, "_available", True, raising=False)
    monkeypatch.setattr(julia_engine, "backend", _Backend(), raising=False)

    got = julia_engine.compute_broad_phase_pairs(centroids, radii, tol=1e-12)
    assert got.shape[1] == 2
    assert set(map(tuple, got.tolist())) == {(0, 1)}


