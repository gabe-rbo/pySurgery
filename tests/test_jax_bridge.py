"""Unit tests for JAX bridge core logic and signature loss construction.

Overview:
    This suite focuses on the robust construction of signature-based loss functions 
    for manifold optimization. It verifies import guards, argument validation, 
    and the correctness of both exact and soft signature modes.

Key Concepts:
    - **Signature Loss**: A differentiable loss function L(A) = (sig(A) - target)².
    - **Import Guarding**: Graceful fallback or error handling when JAX is missing.
    - **Regularization**: Eigengap-based regularization to improve optimization stability.
"""
import numpy as np
import pytest

from pysurgery.integrations import jax_bridge


def test_build_signature_loss_import_guard_or_eval():
    """Verify that signature loss construction handles JAX absence correctly.

    Algorithm:
        1. Check if JAX is available via jax_bridge.HAS_JAX.
        2. If unavailable, assert that calling build_signature_loss_function raises ImportError.
        3. If available, construct a loss function and evaluate it on a diagonal matrix.
    """
    if not jax_bridge.HAS_JAX:
        with pytest.raises(ImportError):
            jax_bridge.build_signature_loss_function(target_signature=0)
        return

    loss_fn = jax_bridge.build_signature_loss_function(target_signature=0, temp=10.0)
    m = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float)
    val = float(loss_fn(m))
    assert val >= 0.0


def test_jax_signature_loss_argument_validation():
    """Ensure that invalid parameters for signature loss construction are rejected.

    Algorithm:
        1. Attempt to construct a loss function with temp=0.0 (invalid).
        2. Attempt to construct a loss function with negative eigengap_weight (invalid).
        3. Assert that ValueError is raised in both cases.
    """
    if not jax_bridge.HAS_JAX:
        with pytest.raises(ImportError):
            jax_bridge.build_signature_loss_function(target_signature=0, temp=0.0)
        return

    with pytest.raises(ValueError):
        jax_bridge.build_signature_loss_function(target_signature=0, temp=0.0)
    with pytest.raises(ValueError):
        jax_bridge.build_signature_loss_function(
            target_signature=0, temp=10.0, eigengap_weight=-1.0
        )


def test_jax_signature_regularizer_keeps_loss_nonnegative():
    """Verify that the eigengap regularizer correctly increases the total loss value.

    What is Being Computed?:
        The difference between a plain signature loss and one with an added penalty 
        for small eigengaps.

    Algorithm:
        1. Build two loss functions: one with eigengap_weight=0.0 and one with 0.1.
        2. Evaluate both on the same matrix.
        3. Assert the regularized loss is greater than or equal to the plain loss.
    """
    if not jax_bridge.HAS_JAX:
        pytest.skip("JAX not installed")

    loss_plain = jax_bridge.build_signature_loss_function(
        target_signature=0, temp=10.0, eigengap_weight=0.0
    )
    loss_reg = jax_bridge.build_signature_loss_function(
        target_signature=0, temp=10.0, eigengap_weight=0.1
    )
    m = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float)
    assert float(loss_reg(m)) >= float(loss_plain(m))


def test_exact_signature_mode_without_jax():
    """Verify that 'exact' mode works correctly using NumPy even if JAX is unavailable.

    What is Being Computed?:
        The exact signature σ(A) = n₊ - n₋ using standard eigenvalue decomposition.

    Algorithm:
        1. Construct a loss function with mode="exact".
        2. Evaluate on a matrix with known signature.
        3. Verify the loss value and the output of exact_signature().
    """
    m = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float)
    loss_fn = jax_bridge.build_signature_loss_function(target_signature=0, mode="exact")
    assert float(loss_fn(m)) == 0.0
    assert jax_bridge.exact_signature(m) == 0


def test_invalid_mode_raises():
    """Ensure that requesting an unsupported computation mode raises an error."""
    with pytest.raises(ValueError):
        jax_bridge.build_signature_loss_function(target_signature=0, mode="unknown")
