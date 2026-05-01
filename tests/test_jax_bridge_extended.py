"""Extended test suite for JAX-accelerated geometric and topological computations.

Overview:
    This suite validates the JAX bridge, focusing on differentiable approximations 
    of topological invariants (like signature) and hardware-accelerated geometric 
    metrics (Gromov-Wasserstein). It ensures parity between JAX-based and pure-Python 
    implementations.

Key Concepts:
    - **Soft Signature**: Differentiable approximation of the signature of a symmetric matrix.
    - **Gromov-Wasserstein (GW)**: Entropic regularization of the GW distance for shape comparison.
    - **Pairwise Distance**: Hardware-accelerated Euclidean distance matrices.
"""
import numpy as np
from pysurgery.integrations.jax_bridge import jax_warmup, _approximate_signature, jax_gromov_wasserstein

def test_jax_warmup():
    """Verify that the JAX backend can be initialized and warmed up.

    Overview:
        Checks the JAX availability report and ensures the warmup JIT compilation succeeds.

    Algorithm:
        1. Call jax_warmup().
        2. Assert 'available' is True and status contains 'Warmup complete'.

    Use When:
        - Debugging JAX installation issues.
    """
    report = jax_warmup()
    assert report["available"] is True
    assert "Warmup complete" in report["status"]

def test_soft_signature():
    """Verify the accuracy of the differentiable 'soft' signature approximation.

    What is Being Computed?:
        The signature of a symmetric matrix using a temperature-scaled softmax of its eigenvalues.

    Algorithm:
        1. Create a diagonal matrix with a known signature (e.g., diag(1, 1, -1) has signature 1).
        2. Compute the soft signature with a high temperature (low temp in our param? wait, temp usually means inverse temp in some contexts, but here it looks like scaling).
        3. Assert the result is close to the exact integer signature.

    Preserved Invariants:
        - Signature (approximated).
    """
    # Symmetric matrix with 2 positive, 1 negative eigenvalue
    # e.g. diag(1, 1, -1)
    matrix = np.diag([1.0, 1.0, -1.0])
    # Exact signature is 2 - 1 = 1
    soft_sig = _approximate_signature(matrix, temp=20.0)
    assert np.isclose(soft_sig, 1.0, atol=0.1)

def test_jax_gw():
    """Validate JAX-accelerated Gromov-Wasserstein distance against the reference Python implementation.

    What is Being Computed?:
        Entropic Gromov-Wasserstein distance between two discrete metric spaces.

    Algorithm:
        1. Define two identical metric spaces (2 points at distance 1).
        2. Compute GW distance using the JAX implementation (jax_gromov_wasserstein).
        3. Compute GW distance using the reference Python implementation.
        4. Assert both results are close to zero and close to each other.

    Preserved Invariants:
        - Gromov-Wasserstein distance (metric on the space of metric spaces).
    """
    # Valid distance matrix (must have zero diagonal)
    # 2 points at distance 1
    D = np.array([[0.0, 1.0], [1.0, 0.0]])
    p = np.array([0.5, 0.5])
    q = np.array([0.5, 0.5])
    
    # JAX version
    # Using smaller epsilon to reduce entropic regularization effect
    dist_jax = jax_gromov_wasserstein(D, D, p, q, epsilon=0.001, max_iter=200)
    
    # Python version
    from pysurgery.core.metrics import gromov_wasserstein_distance
    dist_py = gromov_wasserstein_distance(D, D, p, q, epsilon=0.001, max_iter=200)
    
    # For identical spaces, entropic GW should approach 0 as epsilon -> 0
    # With epsilon=0.001 it should be very small
    assert dist_jax < 0.05
    assert np.isclose(float(dist_jax), dist_py, atol=1e-2)

def test_jax_pairwise_distance_consistency():
    """Verify that JAX pairwise distance computation matches SciPy's reference.

    What is Being Computed?:
        The NxN Euclidean distance matrix for a set of N points in R³.

    Algorithm:
        1. Generate 50 random points in 3D.
        2. Compute distances using jax_pairwise_distance.
        3. Compute distances using scipy.spatial.distance.pdist.
        4. Assert the results match within numerical precision.
    """
    points = np.random.normal(size=(50, 3))
    from pysurgery.integrations.jax_bridge import jax_pairwise_distance
    from scipy.spatial.distance import pdist, squareform
    
    dist_jax = jax_pairwise_distance(points)
    dist_py = squareform(pdist(points, 'euclidean'))
    
    assert np.allclose(np.asarray(dist_jax), dist_py, atol=1e-5)
