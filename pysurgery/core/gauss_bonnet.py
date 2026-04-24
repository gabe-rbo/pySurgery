"""Gauss-Bonnet and Chern-Gauss-Bonnet theorem implementations.

This module provides tools for verifying the relationship between a manifold's
curvature and its Euler characteristic, spanning from classical 2D surfaces
to higher-dimensional Chern-Gauss-Bonnet generalizations.
"""

import math
import numpy as np
from typing import Optional, Union, Dict, Any

from .uniformization import SurfaceMesh, vertex_gaussian_curvature
from .exceptions import MathError


def verify_gauss_bonnet_2d(mesh: Union[SurfaceMesh, Any]) -> Dict[str, Any]:
    """
    Verifies the classical Gauss-Bonnet theorem for a 2D surface.
    
    The theorem states that for a closed 2D Riemannian manifold M:
        ∫_M K dA = 2 * π * χ(M)
    In the discrete setting (simplicial complexes), the integral is replaced by 
    the sum of angle deficits at the vertices.

    Parameters
    ----------
    mesh : SurfaceMesh or tuple (vertices, faces)
        The triangulated surface to evaluate.

    Returns
    -------
    dict
        Results containing total curvature, Euler characteristic, and verification status.
    """
    if not isinstance(mesh, SurfaceMesh):
        try:
            # Try to build mesh if vertices/faces provided
            if isinstance(mesh, tuple) and len(mesh) == 2:
                mesh = SurfaceMesh.from_vertices_faces(mesh[0], mesh[1])
            else:
                raise ValueError("Input must be a SurfaceMesh or (vertices, faces) tuple.")
        except Exception as e:
            raise MathError(f"Could not construct SurfaceMesh for Gauss-Bonnet verification: {e}")

    # Discrete Gaussian curvature at each vertex (angle deficit)
    k = vertex_gaussian_curvature(mesh)
    total_k = float(np.sum(k))
    chi = int(mesh.euler_characteristic)
    expected_k = 2.0 * math.pi * chi
    
    # Boundary correction: if not closed, GB is ∫ K dA + ∫ k_g ds = 2πχ
    # vertex_gaussian_curvature already handles boundary vertices by subtracting from π instead of 2π.
    
    return {
        "dimension": 2,
        "total_curvature": total_k,
        "euler_characteristic": chi,
        "expected_total_curvature": expected_k,
        "residual": abs(total_k - expected_k),
        "passed": math.isclose(total_k, expected_k, rel_tol=1e-9, abs_tol=1e-11)
    }


def verify_chern_gauss_bonnet_4d(
    chi: int, 
    weyl_norm_sq_integral: float, 
    q_curvature_integral: float
) -> Dict[str, Any]:
    """
    Verifies the 4D Chern-Gauss-Bonnet formula involving Weyl and Q-curvature.
    
    Formula:
        χ(M) = (1 / 4π²) * ∫_M [ (1/8)*|W|^2 + Q_4 ] dV
    where W is the Weyl tensor and Q_4 is the fourth-order Q-curvature.

    Parameters
    ----------
    chi : int
        The topological Euler characteristic of the 4-manifold.
    weyl_norm_sq_integral : float
        The integral of the squared norm of the Weyl tensor over the manifold.
    q_curvature_integral : float
        The integral of the Q-curvature over the manifold.

    Returns
    -------
    dict
        Verification results and the relative contributions of Weyl and Q-curvature.
    """
    # Sum the geometric invariants
    integral_sum = (1.0 / 8.0) * weyl_norm_sq_integral + q_curvature_integral
    calc_chi = integral_sum / (4.0 * math.pi**2)
    
    return {
        "dimension": 4,
        "expected_chi": chi,
        "calculated_chi": calc_chi,
        "weyl_contribution": (1.0 / 8.0) * weyl_norm_sq_integral / (4.0 * math.pi**2),
        "q_contribution": q_curvature_integral / (4.0 * math.pi**2),
        "passed": math.isclose(calc_chi, chi, rel_tol=1e-7)
    }


def chern_gauss_bonnet_integral_expected(dimension: int, chi: int) -> float:
    """
    Returns the expected value of the Euler class integral over a 2n-dimensional manifold.
    
    The Chern-Gauss-Bonnet theorem states:
        χ(M) = ∫_M e(TM)
    where e(TM) is the Euler class. In terms of the curvature form Ω:
        χ(M) = (1 / (2π)^n) * ∫_M Pf(Ω)
    where Pf(Ω) is the Pfaffian of the curvature form.

    Parameters
    ----------
    dimension : int
        The dimension of the manifold (must be even).
    chi : int
        The Euler characteristic of the manifold.

    Returns
    -------
    float
        The expected integral value ∫ Pf(Ω).
    """
    if dimension % 2 != 0:
        if chi != 0:
            raise MathError(f"Odd-dimensional closed manifolds must have χ=0; received χ={chi}")
        return 0.0
        
    n = dimension // 2
    # Integral of the Pfaffian curvature form Pf(Ω)
    return (2.0 * math.pi)**n * chi
