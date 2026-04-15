import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax.numpy.linalg import eigh

    HAS_JAX = True
except ImportError:
    HAS_JAX = False


def _approximate_signature(matrix: np.ndarray, temp: float = 10.0):
    """
    A differentiable approximation of the signature of a symmetric matrix.
    Uses a soft step function (tanh) on the eigenvalues.

    Parameters
    ----------
    matrix : jnp.ndarray
        The symmetric matrix.
    temp : float
        Temperature for the soft step. Higher is sharper (closer to exact signature).

    Returns
    -------
    float
        The approximated signature.
    """
    if not HAS_JAX:
        raise ImportError(
            "JAX is required for differentiable topology. Install via 'pip install jax jaxlib'."
        )
    if temp <= 0:
        raise ValueError("temp must be positive.")

    # Eigenvalues of a symmetric matrix are real
    eigenvalues, _ = eigh(matrix)

    # Soft count of positive eigenvalues
    # tanh(temp * x) is ~1 for x > 0, ~-1 for x < 0
    # The sum of tanh over eigenvalues gives approximately (Pos - Neg), which is the signature.
    soft_signature = jnp.sum(jnp.tanh(temp * eigenvalues))

    return soft_signature


def exact_signature(matrix: np.ndarray, tol: float | None = None) -> int:
    """Exact (non-differentiable) signature computed with NumPy eigenvalues."""
    mat = np.asarray(matrix, dtype=float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("matrix must be square.")
    vals = np.linalg.eigvalsh((mat + mat.T) / 2.0)
    if tol is None:
        scale = float(max(1.0, np.max(np.abs(vals)))) if len(vals) else 1.0
        tol = mat.shape[0] * np.finfo(float).eps * scale
    pos = int(np.sum(vals > tol))
    neg = int(np.sum(vals < -tol))
    return pos - neg


def build_signature_loss_function_differentiable(
    target_signature: int,
    temp: float = 10.0,
    eigengap_weight: float = 0.0,
):
    """
    Constructs a JAX-jittable loss function that penalizes a neural network
    if its output intersection form deviates from the target Wall obstruction.
    """
    if not HAS_JAX:
        raise ImportError("JAX is required.")
    if temp <= 0:
        raise ValueError("temp must be positive.")
    if eigengap_weight < 0:
        raise ValueError("eigengap_weight must be non-negative.")

    @jax.jit
    def signature_loss(predicted_matrix: jnp.ndarray) -> float:
        # Ensure symmetry
        sym_matrix = (predicted_matrix + predicted_matrix.T) / 2.0

        approx_sig = _approximate_signature(sym_matrix, temp=temp)

        # Encourage eigenvalues away from 0 to reduce ambiguous soft-signature regions.
        eigenvalues, _ = eigh(sym_matrix)
        gap_penalty = jnp.sum(jnp.exp(-temp * jnp.abs(eigenvalues)))

        # Mean Squared Error against the target topological invariant.
        return (approx_sig - target_signature) ** 2 + eigengap_weight * gap_penalty

    return signature_loss


def build_signature_loss_function_exact(target_signature: int):
    """Build a non-differentiable exact signature loss function (NumPy)."""

    def signature_loss(predicted_matrix: np.ndarray) -> float:
        sig = exact_signature(predicted_matrix)
        return float((sig - target_signature) ** 2)

    return signature_loss


def build_signature_loss_function(
    target_signature: int,
    temp: float = 10.0,
    eigengap_weight: float = 0.0,
    mode: str = "differentiable_approx",
):
    """
    Compatibility wrapper for signature losses.

    mode:
      - "differentiable_approx": JAX soft-signature loss.
      - "exact": NumPy exact signature loss.
    """
    mode_n = mode.strip().lower()
    if mode_n == "exact":
        return build_signature_loss_function_exact(target_signature)
    if mode_n == "differentiable_approx":
        return build_signature_loss_function_differentiable(
            target_signature=target_signature,
            temp=temp,
            eigengap_weight=eigengap_weight,
        )
    raise ValueError("mode must be 'differentiable_approx' or 'exact'.")
