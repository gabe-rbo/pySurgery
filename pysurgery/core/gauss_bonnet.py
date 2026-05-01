import math
import numpy as np
from typing import Tuple, Dict, Any, Union, Sequence
from .exceptions import MathError
from .uniformization import SurfaceMesh


def verify_gauss_bonnet_2d(
    mesh_data: Union[SurfaceMesh, Tuple[np.ndarray, Sequence[Sequence[int]]]],
    backend: str = "auto",
) -> Dict[str, Any]:
    """Verify the Gauss-Bonnet theorem for a 2D surface mesh.

    What is Being Computed?:
        The total Gaussian curvature of a 2D surface and its consistency with 
        the surface's Euler characteristic. The Gauss-Bonnet theorem states 
        that ∫ K dA = 2πχ(M).

    Algorithm:
        1. Extract or create a SurfaceMesh from the input data.
        2. Compute the Euler characteristic χ via the V - E + F formula.
        3. Compute discrete Gaussian curvature at each vertex (angle deficit).
        4. Sum the vertex curvatures to get the total curvature.
        5. Check if Σ K_v ≈ 2πχ.

    Preserved Invariants:
        - Euler characteristic (χ) — Topological invariant preserved under 
          homeomorphism.
        - Total curvature — Topological invariant for closed surfaces.

    Args:
        mesh_data: Either a SurfaceMesh object or a tuple of (vertices, faces).
        backend: 'auto', 'julia', or 'python'.

    Returns:
        Dict[str, Any]: Results containing 'passed' (bool), 'euler_characteristic', 
                        'total_curvature', and the numerical 'error'.

    Use When:
        - Validating the quality of a surface triangulation.
        - Checking if a discrete metric is consistent with a given topology.

    Example:
        results = verify_gauss_bonnet_2d(sphere_mesh)
        print(f"Passed: {results['passed']}")
    """
    if isinstance(mesh_data, tuple):
        vertices, faces = mesh_data
        mesh = SurfaceMesh.from_vertices_faces(vertices, faces)
    else:
        mesh = mesh_data

    chi = mesh.euler_characteristic
    v_curvatures = mesh.vertex_gaussian_curvature(backend=backend)
    total_curvature = float(np.sum(v_curvatures))

    # sum K_v = 2π χ(M)
    expected = 2.0 * math.pi * chi
    passed = math.isclose(total_curvature, expected, rel_tol=1e-5, abs_tol=1e-8)

    return {
        "passed": passed,
        "euler_characteristic": chi,
        "total_curvature": total_curvature,
        "error": abs(total_curvature - expected),
    }


def chern_gauss_bonnet_integral_expected(dimension: int, chi: int) -> float:
    """Calculate the expected integral of the Euler form.

    What is Being Computed?:
        The theoretical value of the integral of the Euler n-form over a 
        2n-dimensional closed manifold, which is (2π)ⁿ χ(M).

    Algorithm:
        1. Check if the dimension is even.
        2. If odd, verify χ=0 and return 0.0.
        3. If even, compute (2π)^(dim/2) * chi.

    Args:
        dimension: The dimension of the manifold.
        chi: The Euler characteristic.

    Returns:
        float: The expected integral value.

    Raises:
        MathError: If dimension is odd and chi is non-zero.
    """
    if dimension % 2 != 0:
        if chi != 0:
            raise MathError(
                f"Odd-dimensional closed manifolds must have χ=0; received χ={chi}"
            )
        return 0.0

    n = dimension // 2
    return (2.0 * math.pi) ** n * chi


def verify_chern_gauss_bonnet_4d(
    chi: int, weyl_integral: float, q_integral: float, backend: str = "auto"
) -> Dict[str, Any]:
    """Verify the Chern-Gauss-Bonnet theorem for a 4D manifold.

    What is Being Computed?:
        Verification of the 4D Chern-Gauss-Bonnet formula which relates the 
        Euler characteristic to integrals of the Weyl and Q-curvature.
        Formula: χ = 1/(4π²) ∫ [ (1/8)|W|² + Q ] dV.

    Algorithm:
        1. Calculate the theoretical χ from the provided curvature integrals.
        2. Compare the calculated value with the provided integer χ.
        3. Return the pass/fail status and the numerical error.

    Args:
        chi: The expected Euler characteristic.
        weyl_integral: The integral of the Weyl curvature component.
        q_integral: The integral of the Q-curvature component.
        backend: 'auto', 'julia', or 'python'.

    Returns:
        Dict[str, Any]: A dictionary with 'passed', 'calculated_chi', and 'error'.
    """
    # chi = 1/(4pi^2) * (1/8 * weyl_integral + q_integral)
    backend_norm = str(backend).lower().strip()
    if backend_norm == "julia":
        from ..bridge.julia_bridge import julia_engine
        julia_engine.require_julia()

    factor = 1.0 / (4.0 * math.pi**2)
    calculated_chi = factor * (0.125 * weyl_integral + q_integral)
    passed = math.isclose(calculated_chi, float(chi), rel_tol=1e-5, abs_tol=1e-8)

    return {
        "passed": passed,
        "calculated_chi": calculated_chi,
        "error": abs(calculated_chi - chi),
    }
