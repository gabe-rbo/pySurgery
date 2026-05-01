"""Unit tests for the Julia bridge and high-performance backend orchestration.

Overview:
    This suite verifies the JuliaBridge singleton lifecycle, its interaction 
    with the `juliacall` backend, and the correctness of various accelerated 
    computations (Normal Surface residuals, Broad Phase pairing). It heavily 
    utilizes monkeypatching to simulate different backend availability states.

Key Concepts:
    - **Singleton Pattern**: Ensures only one Julia runtime is initialized per process.
    - **Lazy Initialization**: Backend components are loaded only when needed.
    - **Mocked Backends**: Testing high-level logic without requiring a full Julia install.
"""
import pytest
import numpy as np
import scipy.sparse as sp

from pysurgery.bridge.julia_bridge import JuliaBridge, julia_engine


def test_julia_bridge_singleton_identity():
    """Verify that JuliaBridge correctly implements the Singleton pattern.

    What is Being Computed?:
        Object identity of multiple JuliaBridge instantiations.

    Algorithm:
        1. Instantiate two JuliaBridge objects.
        2. Assert they refer to the same memory address.
        3. Assert they match the global julia_engine instance.
    """
    a = JuliaBridge()
    b = JuliaBridge()
    assert a is b
    assert a is julia_engine


def test_julia_bridge_require_julia_behavior():
    """Verify the error handling of require_julia() based on backend availability.

    Algorithm:
        1. Check julia_engine.available.
        2. If available, ensure require_julia() returns silently.
        3. If unavailable, ensure it raises a SurgeryError.
    """
    if julia_engine.available:
        # Should not raise when backend is available.
        julia_engine.require_julia()
    else:
        from pysurgery.core.exceptions import SurgeryError

        with pytest.raises(SurgeryError):
            julia_engine.require_julia()


def test_julia_bridge_warmup_unavailable_is_nonfatal(monkeypatch):
    """Ensure that calling warmup() on an unavailable backend returns a failed report without crashing.

    Algorithm:
        1. Mock julia_engine to simulate a missing Julia installation.
        2. Call warmup().
        3. Verify the report indicates failure but the process remains stable.
    """
    monkeypatch.setattr(julia_engine, "_initialized", True, raising=False)
    monkeypatch.setattr(julia_engine, "_available", False, raising=False)
    monkeypatch.setattr(julia_engine, "error", "missing juliacall", raising=False)

    report = julia_engine.warmup()
    assert report["available"] is False
    assert report["mode"] == "full"
    assert isinstance(report["failed"], dict)


def test_julia_bridge_warmup_full_executes_and_caches(monkeypatch):
    """Verify that full warmup executes workloads exactly once and caches the result.

    Algorithm:
        1. Mock the internal workload runners of julia_engine.
        2. Call warmup() twice.
        3. Assert that workloads were called in the first turn and cached in the second.
    """
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
    """Validate the Julia-accelerated calculation of normal surface residual norms.

    What is Being Computed?:
        The Euclidean norms of the residuals (MxV) for a set of vectors V against a boundary matrix M.

    Algorithm:
        1. Define a sparse boundary matrix and a set of candidate vectors.
        2. Mock the Julia backend to use a NumPy-based reference calculation.
        3. Compare the output of julia_engine.compute_normal_surface_residual_norms with a local reference.
    """
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
    """Validate the broad-phase collision detection logic in the Julia bridge.

    What is Being Computed?:
        A list of indices (i, j) for pairs of spheres that potentially intersect.

    Algorithm:
        1. Define centroids and radii for three spheres.
        2. Mock the Julia backend to use a nested loop distance check.
        3. Assert that only the intersecting pair (0, 1) is returned.
    """
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


