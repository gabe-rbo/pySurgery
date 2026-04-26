import math
import numpy as np
from typing import Tuple, Dict, Any, Union, Sequence
from .exceptions import MathError
from .uniformization import SurfaceMesh


def verify_gauss_bonnet_2d(
    mesh_data: Union[SurfaceMesh, Tuple[np.ndarray, Sequence[Sequence[int]]]]
) -> Dict[str, Any]:
    """Verify the Gauss-Bonnet theorem for a 2D surface mesh.

    Args:
        mesh_data: Either a SurfaceMesh object or a tuple of (vertices, faces).

    Returns:
        A dictionary with 'passed', 'euler_characteristic', 'total_curvature', and 'error'.
    """
    if isinstance(mesh_data, tuple):
        vertices, faces = mesh_data
        mesh = SurfaceMesh.from_vertices_faces(vertices, faces)
    else:
        mesh = mesh_data

    chi = mesh.euler_characteristic
    v_curvatures = mesh.vertex_gaussian_curvature()
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
    """Calculate the expected integral of the Euler form for a given dimension and Euler characteristic.

    Args:
        dimension: The dimension of the manifold.
        chi: The Euler characteristic.

    Returns:
        The expected integral value.

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
    chi: int, weyl_integral: float, q_integral: float
) -> Dict[str, Any]:
    """Verify the Chern-Gauss-Bonnet theorem for a 4D manifold.

    Args:
        chi: The Euler characteristic.
        weyl_integral: The integral of the Weyl curvature component.
        q_integral: The integral of the Q-curvature component.

    Returns:
        A dictionary with 'passed', 'calculated_chi', and 'error'.
    """
    # chi = 1/(4pi^2) * (1/8 * weyl_integral + q_integral)
    factor = 1.0 / (4.0 * math.pi**2)
    calculated_chi = factor * (0.125 * weyl_integral + q_integral)
    passed = math.isclose(calculated_chi, float(chi), rel_tol=1e-5, abs_tol=1e-8)

    return {
        "passed": passed,
        "calculated_chi": calculated_chi,
        "error": abs(calculated_chi - chi),
    }
