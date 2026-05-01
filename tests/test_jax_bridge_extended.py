import numpy as np
from pysurgery.integrations.jax_bridge import jax_warmup, _approximate_signature, jax_gromov_wasserstein

def test_jax_warmup():
    report = jax_warmup()
    assert report["available"] is True
    assert "Warmup complete" in report["status"]

def test_soft_signature():
    # Symmetric matrix with 2 positive, 1 negative eigenvalue
    # e.g. diag(1, 1, -1)
    matrix = np.diag([1.0, 1.0, -1.0])
    # Exact signature is 2 - 1 = 1
    soft_sig = _approximate_signature(matrix, temp=20.0)
    assert np.isclose(soft_sig, 1.0, atol=0.1)

def test_jax_gw():
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
    points = np.random.normal(size=(50, 3))
    from pysurgery.integrations.jax_bridge import jax_pairwise_distance
    from scipy.spatial.distance import pdist, squareform
    
    dist_jax = jax_pairwise_distance(points)
    dist_py = squareform(pdist(points, 'euclidean'))
    
    assert np.allclose(np.asarray(dist_jax), dist_py, atol=1e-5)
