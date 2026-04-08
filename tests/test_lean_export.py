import numpy as np
import pytest

from pysurgery.integrations.lean_export import generate_lean_isomorphism_certificate, run_lean_check


def test_lean_export_uses_decide_for_small_matrices():
    q = np.array([[1, 0], [0, 1]], dtype=np.int64)
    p = np.array([[1, 0], [0, 1]], dtype=np.int64)
    code = generate_lean_isomorphism_certificate(q, q, p, theorem_name="small")
    assert "import Mathlib" in code
    assert "small_valid" in code
    assert "small_unimodular" in code
    assert "  decide" in code
    assert "1.0" not in code


def test_lean_export_uses_native_decide_for_larger_matrices():
    q = np.eye(5, dtype=np.int64)
    p = np.eye(5, dtype=np.int64)
    code = generate_lean_isomorphism_certificate(q, q, p, theorem_name="large")
    assert "native_decide" in code


def test_lean_export_rejects_non_integer_entries():
    q = np.array([[1.5, 0.0], [0.0, 1.0]], dtype=float)
    p = np.eye(2)
    with pytest.raises(ValueError):
        generate_lean_isomorphism_certificate(q, q, p)


def test_lean_export_rejects_shape_mismatch():
    q1 = np.eye(2, dtype=np.int64)
    q2 = np.eye(3, dtype=np.int64)
    p = np.eye(2, dtype=np.int64)
    with pytest.raises(ValueError):
        generate_lean_isomorphism_certificate(q1, q2, p)


def test_lean_export_rejects_invalid_theorem_name():
    q = np.eye(2, dtype=np.int64)
    with pytest.raises(ValueError):
        generate_lean_isomorphism_certificate(q, q, q, theorem_name="bad-name")


def test_run_lean_check_unavailable_executable():
    res = run_lean_check("theorem t : True := by trivial", lean_cmd="lean-does-not-exist")
    assert not res.available
    assert not res.success
    assert res.exit_code == 127


def test_run_lean_check_on_generated_code_returns_structured_result():
    q = np.eye(2, dtype=np.int64)
    code = generate_lean_isomorphism_certificate(q, q, q, theorem_name="small2")
    res = run_lean_check(code, lean_cmd="lean-does-not-exist")
    assert hasattr(res, "available")
    assert hasattr(res, "stderr")


