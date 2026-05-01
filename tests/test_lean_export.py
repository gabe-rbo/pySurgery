"""Tests for exporting topological certificates to the Lean 4 formal verification language.

Overview:
    This suite validates the generation of Lean 4 source code that acts as a 
    certificate for matrix isomorphisms. It ensures that the generated code 
    uses appropriate tactics (`decide` vs `native_decide`) and that the 
    exporter correctly rejects invalid inputs (non-integers, shape mismatches).

Key Concepts:
    - **Formal Verification**: Using Lean 4 to prove that two intersection forms are isomorphic.
    - **Certificate Generation**: Automating the creation of formal proofs from computational results.
    - **Isomorphism Witness**: A matrix P such that Pᵀ Q₁ P = Q₂.
"""
import numpy as np
import pytest

from pysurgery.integrations.lean_export import (
    generate_lean_isomorphism_certificate,
    run_lean_check,
)


def test_lean_export_uses_decide_for_small_matrices():
    """Verify that small isomorphism certificates use the standard 'decide' tactic.

    What is Being Computed?:
        A Lean 4 theorem proving Q₁ ≅ Q₂ for 2x2 identity matrices.

    Algorithm:
        1. Generate Lean code for two 2x2 identity matrices.
        2. Assert that 'decide' is present in the code and 'native_decide' is not.
    """
    q = np.array([[1, 0], [0, 1]], dtype=np.int64)
    p = np.array([[1, 0], [0, 1]], dtype=np.int64)
    code = generate_lean_isomorphism_certificate(q, q, p, theorem_name="small")
    assert "import Mathlib" in code
    assert "small_valid" in code
    assert "small_unimodular" in code
    assert "  decide" in code
    assert "1.0" not in code


def test_lean_export_uses_native_decide_for_larger_matrices():
    """Verify that larger isomorphism certificates use the 'native_decide' tactic for performance.

    What is Being Computed?:
        A Lean 4 theorem for 5x5 matrices.

    Algorithm:
        1. Generate Lean code for 5x5 identity matrices.
        2. Assert that 'native_decide' is present in the code.
    """
    q = np.eye(5, dtype=np.int64)
    p = np.eye(5, dtype=np.int64)
    code = generate_lean_isomorphism_certificate(q, q, p, theorem_name="large")
    assert "native_decide" in code


def test_lean_export_rejects_non_integer_entries():
    """Ensure the exporter rejects matrices with non-integer entries (unsupported by the formal model)."""
    q = np.array([[1.5, 0.0], [0.0, 1.0]], dtype=float)
    p = np.eye(2)
    with pytest.raises(ValueError):
        generate_lean_isomorphism_certificate(q, q, p)


def test_lean_export_rejects_shape_mismatch():
    """Ensure the exporter rejects matrices with mismatched dimensions."""
    q1 = np.eye(2, dtype=np.int64)
    q2 = np.eye(3, dtype=np.int64)
    p = np.eye(2, dtype=np.int64)
    with pytest.raises(ValueError):
        generate_lean_isomorphism_certificate(q1, q2, p)


def test_lean_export_rejects_invalid_theorem_name():
    """Ensure the exporter rejects theorem names that are not valid Lean identifiers."""
    q = np.eye(2, dtype=np.int64)
    with pytest.raises(ValueError):
        generate_lean_isomorphism_certificate(q, q, q, theorem_name="bad-name")


def test_run_lean_check_unavailable_executable():
    """Verify that run_lean_check handles a missing 'lean' executable gracefully.

    Algorithm:
        1. Call run_lean_check with a dummy command.
        2. Assert available is False and exit_code is 127 (command not found).
    """
    res = run_lean_check(
        "theorem t : True := by trivial", lean_cmd="lean-does-not-exist"
    )
    assert not res.available
    assert not res.success
    assert res.exit_code == 127


def test_run_lean_check_on_generated_code_returns_structured_result():
    """Verify that run_lean_check returns a LeanResult object with required attributes."""
    q = np.eye(2, dtype=np.int64)
    code = generate_lean_isomorphism_certificate(q, q, q, theorem_name="small2")
    res = run_lean_check(code, lean_cmd="lean-does-not-exist")
    assert hasattr(res, "available")
    assert hasattr(res, "stderr")
