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
        raise ImportError("JAX is required for differentiable topology. Install via 'pip install jax jaxlib'.")

    # Eigenvalues of a symmetric matrix are real
    eigenvalues, _ = eigh(matrix)
    
    # Soft count of positive eigenvalues
    # tanh(temp * x) is ~1 for x > 0, ~-1 for x < 0
    # The sum of tanh over eigenvalues gives approximately (Pos - Neg), which is the signature.
    soft_signature = jnp.sum(jnp.tanh(temp * eigenvalues))
    
    return soft_signature

def build_signature_loss_function(target_signature: int, temp: float = 10.0):
    """
    Constructs a JAX-jittable loss function that penalizes a neural network 
    if its output intersection form deviates from the target Wall obstruction.
    """
    if not HAS_JAX:
        raise ImportError("JAX is required.")
        
    @jax.jit
    def signature_loss(predicted_matrix: jnp.ndarray) -> float:
        # Ensure symmetry
        sym_matrix = (predicted_matrix + predicted_matrix.T) / 2.0
        
        approx_sig = _approximate_signature(sym_matrix, temp=temp)
        
        # Mean Squared Error against the target topological invariant
        return (approx_sig - target_signature) ** 2
        
    return signature_loss
