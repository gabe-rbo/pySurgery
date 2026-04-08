import numpy as np
import pytest

from pysurgery.integrations import jax_bridge


def test_build_signature_loss_import_guard_or_eval():
    if not jax_bridge.HAS_JAX:
        with pytest.raises(ImportError):
            jax_bridge.build_signature_loss_function(target_signature=0)
        return

    loss_fn = jax_bridge.build_signature_loss_function(target_signature=0, temp=10.0)
    m = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float)
    val = float(loss_fn(m))
    assert val >= 0.0


def test_jax_signature_loss_argument_validation():
    if not jax_bridge.HAS_JAX:
        with pytest.raises(ImportError):
            jax_bridge.build_signature_loss_function(target_signature=0, temp=0.0)
        return

    with pytest.raises(ValueError):
        jax_bridge.build_signature_loss_function(target_signature=0, temp=0.0)
    with pytest.raises(ValueError):
        jax_bridge.build_signature_loss_function(target_signature=0, temp=10.0, eigengap_weight=-1.0)


def test_jax_signature_regularizer_keeps_loss_nonnegative():
    if not jax_bridge.HAS_JAX:
        pytest.skip("JAX not installed")

    loss_plain = jax_bridge.build_signature_loss_function(target_signature=0, temp=10.0, eigengap_weight=0.0)
    loss_reg = jax_bridge.build_signature_loss_function(target_signature=0, temp=10.0, eigengap_weight=0.1)
    m = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float)
    assert float(loss_reg(m)) >= float(loss_plain(m))


def test_exact_signature_mode_without_jax():
    m = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float)
    loss_fn = jax_bridge.build_signature_loss_function(target_signature=0, mode="exact")
    assert float(loss_fn(m)) == 0.0
    assert jax_bridge.exact_signature(m) == 0


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        jax_bridge.build_signature_loss_function(target_signature=0, mode="unknown")


