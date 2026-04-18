import numpy as np

try:
    import jax.numpy as jnp
    from jax import jit, vmap, lax

    HAS_JAX = True
except ImportError:
    HAS_JAX = False


def _approximate_signature(matrix: jnp.ndarray, temp: float = 10.0):
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
    eigenvalues = jnp.linalg.eigvalsh(matrix)

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

    @jit
    def signature_loss(predicted_matrix: jnp.ndarray) -> float:
        # Ensure symmetry
        sym_matrix = (predicted_matrix + predicted_matrix.T) / 2.0

        approx_sig = _approximate_signature(sym_matrix, temp=temp)

        # Encourage eigenvalues away from 0 to reduce ambiguous soft-signature regions.
        eigenvalues = jnp.linalg.eigvalsh(sym_matrix)
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


# --- JAX-Accelerated Metrics ---

if HAS_JAX:
    @jit
    def jax_pairwise_distance(data: jnp.ndarray) -> jnp.ndarray:
        """Pairwise Euclidean distance matrix via JAX. Singular implementation for all scales."""
        diff = data[:, None, :] - data[None, :, :]
        # Add a small epsilon to avoid NaN in gradients of sqrt(0)
        return jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-12)

    @jit
    def jax_gromov_wasserstein(
        D_A: jnp.ndarray,
        D_B: jnp.ndarray,
        p: jnp.ndarray,
        q: jnp.ndarray,
        epsilon: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """Entropic Gromov-Wasserstein distance using Sinkhorn iterations in JAX. Singular implementation for all scales."""
        n, m = D_A.shape[0], D_B.shape[0]
        
        # Precompute constants for the L-matrix
        C1 = jnp.dot(D_A**2, p[:, None]) @ jnp.ones((1, m))
        C2 = jnp.ones((n, 1)) @ jnp.dot(q[None, :], (D_B**2).T)
        const_C = C1 + C2

        def body_T(i, T):
            # Compute current cost matrix based on current coupling T
            L = const_C - 2 * jnp.dot(jnp.dot(D_A, T), D_B.T)
            K = jnp.exp(-L / epsilon)
            
            # Sinkhorn projections
            def body_sinkhorn(j, val):
                u, v = val
                u = p / (jnp.dot(K, v) + 1e-12)
                v = q / (jnp.dot(K.T, u) + 1e-12)
                return u, v
            
            u_init = jnp.ones(n) / n
            v_init = jnp.ones(m) / m
            u, v = lax.fori_loop(0, 20, body_sinkhorn, (u_init, v_init))
            return u[:, None] * K * v[None, :]
        
        # Iteratively refine the coupling matrix T
        T_init = jnp.outer(p, q)
        T_final = lax.fori_loop(0, max_iter, body_T, T_init)
        
        # Final GW loss (quadratic)
        L_final = const_C - 2 * jnp.dot(jnp.dot(D_A, T_final), D_B.T)
        gw_sq = jnp.sum(L_final * T_final)
        return jnp.sqrt(jnp.maximum(gw_sq, 0.0) + 1e-12)

    @jit
    def jax_local_pca_dimensions(
        points: jnp.ndarray,
        neighbor_indices: jnp.ndarray,
        variance_threshold: float = 0.9,
    ) -> jnp.ndarray:
        """Vectorized Local PCA to estimate intrinsic dimension at every point."""
        # neighborhood shape: (N, K, D)
        # neighbor_indices has shape (N, K)
        
        def single_point_pca(indices):
            # Extract neighborhood (including the point itself)
            neighborhood = points[indices]
            # Center the neighborhood
            centered = neighborhood - jnp.mean(neighborhood, axis=0)
            # Compute SVD of centered neighborhood
            # (centered.T @ centered) / (K-1) is the covariance matrix
            # We can use SVD on 'centered' directly for better stability
            _, s, _ = jnp.linalg.svd(centered, full_matrices=False)
            eigvals = s**2 / (neighborhood.shape[0] - 1)
            
            total_var = jnp.sum(eigvals)
            explained_var = jnp.cumsum(eigvals) / (total_var + 1e-10)
            
            # Find the first index where explained variance exceeds threshold
            dim = jnp.sum(explained_var < variance_threshold) + 1
            return dim

        return vmap(single_point_pca)(neighbor_indices)

    # Batch support for signatures
    jax_batch_soft_signature = vmap(_approximate_signature, in_axes=(0, None))


def jax_warmup():
    """Warm up JAX JIT compilers with sample workloads."""
    if not HAS_JAX:
        return {"available": False, "status": "JAX not installed"}
    
    try:
        # 1. Warm up soft signature
        m1 = jnp.eye(10)
        _approximate_signature(m1)
        
        # 2. Warm up batch signature
        m_batch = jnp.stack([jnp.eye(5), jnp.ones((5, 5))])
        jax_batch_soft_signature(m_batch, 10.0)
        
        # 3. Warm up metrics
        data = jnp.zeros((10, 3))
        jax_pairwise_distance(data)
        
        # 4. Warm up GW
        p = jnp.ones(5) / 5
        q = jnp.ones(5) / 5
        jax_gromov_wasserstein(jnp.eye(5), jnp.eye(5), p, q)
        
        return {"available": True, "status": "Warmup complete"}
    except Exception as e:
        return {"available": True, "status": f"Warmup failed: {e!r}"}
