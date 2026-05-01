"""Test suite for discrete surface uniformization and curvature flow.

Overview:
    This module tests algorithms for uniformizing discrete surfaces, including 
    Discrete Ricci Flow and Circle Packing uniformization. it verifies 
    Gaussian curvature computation, Laplace-Beltrami operator (cotangent) construction, 
    and convergence to target curvatures (spherical, flat, hyperbolic).

Key Concepts:
    - **Uniformization**: Conformal deformation of a metric to one with constant curvature.
    - **Gaussian Curvature**: Angle deficit at vertices in a triangle mesh.
    - **Ricci Flow**: Iterative evolution of edge lengths to achieve target curvature.
    - **Circle Packing**: Discrete conformal mapping via circle radii.
"""

import math

import numpy as np

from pysurgery.core.complexes import SimplicialComplex
from pysurgery.core.uniformization import (
    SurfaceMesh,
    circle_packing_uniformization,
    cotangent_laplacian,
    discrete_ricci_flow,
    surface_target_curvature,
    uniformize_surface,
    vertex_gaussian_curvature,
)


TETRA_FACES = np.array(
    [
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
    ],
    dtype=np.int64,
)


def _distorted_tetrahedron_vertices():
    """Generate a non-regular tetrahedron for testing robust curvature logic."""
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.1, 1.4, 0.2],
            [0.2, 0.1, 1.9],
        ],
        dtype=np.float64,
    )


def test_vertex_curvature_and_cotan_laplacian_on_distorted_tetrahedron():
    """Verify basic geometric operators on a distorted tetrahedron.

    What is Being Computed?:
        Computes the vertex-wise Gaussian curvature and the cotangent Laplacian matrix.

    Algorithm:
        1. Create a `SurfaceMesh` from a distorted tetrahedron.
        2. Calculate Gaussian curvature (angle deficit).
        3. Construct the cotangent Laplacian matrix.
        4. Assert Laplacian symmetry, row-sum zero, and Gauss-Bonnet consistency.
    """
    mesh = SurfaceMesh.from_vertices_faces(_distorted_tetrahedron_vertices(), TETRA_FACES)

    curv = vertex_gaussian_curvature(mesh)
    lap = cotangent_laplacian(mesh)

    assert curv.shape == (4,)
    assert lap.shape == (4, 4)
    assert np.allclose(lap.toarray(), lap.toarray().T, atol=1e-10)
    assert np.allclose(lap.toarray().sum(axis=1), 0.0, atol=1e-10)
    assert mesh.target_geometry() == "spherical"
    assert math.isclose(float(surface_target_curvature(mesh).sum()), 4.0 * math.pi, rel_tol=1e-10)
    assert not np.allclose(curv, np.full(4, math.pi))


def test_discrete_ricci_flow_uniformizes_distorted_sphere_like_mesh():
    """Verify that Ricci flow converges to constant curvature on a sphere.

    What is Being Computed?:
        Runs the discrete Ricci flow on a distorted sphere mesh to achieve constant 
        positive Gaussian curvature.

    Algorithm:
        1. Initialize Ricci flow on a distorted tetrahedron.
        2. Iterate until the L2-norm of (K - K_target) is below tolerance.
        3. Assert convergence and that the final curvature is nearly uniform.
    """
    mesh = SurfaceMesh.from_vertices_faces(_distorted_tetrahedron_vertices(), TETRA_FACES)
    initial_residual = np.linalg.norm(
        vertex_gaussian_curvature(mesh) - surface_target_curvature(mesh), ord=2
    )

    result = discrete_ricci_flow(mesh, max_iter=40, tol=1e-8, damping=0.8)

    assert result.target_geometry == "spherical"
    assert result.iterations >= 0
    assert result.residual_norm <= initial_residual
    assert result.converged
    assert np.linalg.norm(result.curvature - result.target_curvature, ord=2) < 1e-5
    assert result.decision_ready()


def test_circle_packing_uniformization_on_combinatorial_tetrahedron():
    """Verify circle packing uniformization on a combinatorial sphere.

    What is Being Computed?:
        Computes circle radii that yield constant curvature for a tetrahedron topology.

    Algorithm:
        1. Create a mesh from a combinatorial complex (unit edge lengths).
        2. Run `circle_packing_uniformization`.
        3. Assert the resulting curvature is uniform and residual is near machine epsilon.
    """
    sc = SimplicialComplex.from_maximal_simplices(TETRA_FACES.tolist())
    mesh = SurfaceMesh.from_simplicial_complex(sc)

    curv = vertex_gaussian_curvature(mesh, method="circle_packing")
    assert np.allclose(curv, np.full(4, math.pi), atol=1e-10)

    result = circle_packing_uniformization(mesh, max_iter=12, tol=1e-10)

    assert result.target_geometry == "spherical"
    assert result.converged
    assert result.residual_norm < 1e-10
    assert np.allclose(result.curvature, result.target_curvature, atol=1e-10)


def test_uniformize_surface_dispatches_by_method():
    """Verify the high-level dispatcher for uniformization algorithms.

    What is Being Computed?:
        Ensures `uniformize_surface` correctly routes requests to the Ricci flow backend.
    """
    sc = SimplicialComplex.from_maximal_simplices(TETRA_FACES.tolist())
    result = uniformize_surface(sc, method="ricci", max_iter=5, tol=1e-10)

    assert result.target_geometry == "spherical"
    assert result.converged
    assert result.mesh.n_vertices == 4

