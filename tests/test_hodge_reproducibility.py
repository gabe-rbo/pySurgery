import numpy as np
import pytest
from pysurgery.topology.graphs import Graph

def test_hodge_harmonic_reproducibility():
    """
    Test that the harmonic forms computation is perfectly reproducible across
    multiple calls to the Julia backend, and that the Julia backend produces
    the same harmonic subspace (projector) as the Python/Scipy fallback.
    """
    # Create a figure-8 graph (two triangles sharing a vertex)
    # Betti numbers: b_0 = 1, b_1 = 2
    g = Graph.from_edges([
        (0, 1), (1, 2), (2, 0), # Triangle 1
        (0, 3), (3, 4), (4, 0)  # Triangle 2
    ])

    # 1. Compute harmonic forms with backend="julia" multiple times
    H1_jl = g.harmonic_forms(k=1, backend="julia")
    H2_jl = g.harmonic_forms(k=1, backend="julia")
    
    # Assert exact numerical reproducibility for Julia
    np.testing.assert_allclose(H1_jl, H2_jl, err_msg="Julia backend is non-deterministic!")

    # 2. Compute harmonic forms with backend="python" (scipy fallback)
    H_py = g.harmonic_forms(k=1, backend="python")

    # 3. Assert the subspace projectors are equivalent
    # The basis vectors themselves might differ between solvers (e.g. arbitrary rotations),
    # but the projector H @ H.T onto the harmonic subspace must be identical.
    P_jl = H1_jl @ H1_jl.T
    P_py = H_py @ H_py.T
    
    np.testing.assert_allclose(P_jl, P_py, atol=1e-10, err_msg="Julia and Python backends compute different harmonic subspaces!")
