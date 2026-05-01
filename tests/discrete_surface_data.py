"""Library of discrete surfaces and manifolds for testing topological invariants.

Overview:
    This module provides a collection of hard-coded simplicial complexes representing
    canonical topological spaces (S^1, S^2, RP^2, Torus, etc.). These are used
    across the test suite to verify homology, intersection forms, and surgery obstructions.

Key Concepts:
    - **Canonical Spaces**: Standard models for spheres, projective planes, and tori.
    - **Simplicial Construction**: Defined by explicit lists of maximal simplices.
    - **Manifold Verification**: Used as ground-truth for invariant computation.
"""

from pysurgery.core.complexes import SimplicialComplex

def build_tetrahedron():
    """Construct a tetrahedron as a simplicial complex.

    What is Being Computed?:
        A simplicial complex with 4 vertices and 4 faces, homeomorphic to S^2.

    Returns:
        SimplicialComplex: Boundary of the 3-simplex.
    """
    return SimplicialComplex.from_simplices([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])

def build_octahedron():
    """Construct an octahedron as a simplicial complex.

    What is Being Computed?:
        A simplicial complex with 6 vertices and 8 faces, homeomorphic to S^2.

    Returns:
        SimplicialComplex: The octahedron.
    """
    faces = [
        [0, 2, 4], [0, 2, 5], [0, 3, 4], [0, 3, 5],
        [1, 2, 4], [1, 2, 5], [1, 3, 4], [1, 3, 5],
    ]
    return SimplicialComplex.from_simplices(faces)

def build_icosahedron():
    """Construct an icosahedron as a simplicial complex.

    What is Being Computed?:
        A simplicial complex with 12 vertices and 20 faces, homeomorphic to S^2.

    Returns:
        SimplicialComplex: The icosahedron.
    """
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]
    return SimplicialComplex.from_simplices(faces)

def build_torus():
    """Construct a minimal triangulation of the torus T^2.

    What is Being Computed?:
        A simplicial complex with 9 vertices and 18 faces, representing S^1 x S^1.

    Returns:
        SimplicialComplex: The flat torus.
    """
    faces = [
        [0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4], [2, 0, 3], [2, 3, 5],
        [3, 4, 7], [3, 7, 6], [4, 5, 8], [4, 8, 7], [5, 3, 6], [5, 6, 8],
        [6, 7, 1], [6, 1, 0], [7, 8, 2], [7, 2, 1], [8, 6, 0], [8, 0, 2],
    ]
    return SimplicialComplex.from_simplices(faces)

def build_rp2():
    """Construct a minimal triangulation of the Real Projective Plane RP^2.

    What is Being Computed?:
        A simplicial complex with 6 vertices and 10 faces.

    Returns:
        SimplicialComplex: The projective plane.
    """
    faces = [
        [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 1],
        [1, 2, 4], [2, 3, 5], [3, 4, 1], [4, 5, 2], [5, 1, 3],
    ]
    return SimplicialComplex.from_simplices(faces)

def build_s3():
    """Construct a simplicial complex representing the 3-sphere S^3.

    What is Being Computed?:
        The boundary of a 4-simplex, yielding a 3-dimensional manifold.

    Returns:
        SimplicialComplex: The 3-sphere.
    """
    # S3 is the boundary of the 4-simplex
    simplices = []
    for i in range(5):
        face = [j for j in range(5) if j != i]
        simplices.append(face)
    return SimplicialComplex.from_simplices(simplices)

def to_complex(sc):
    """Identity pass-through for complex-ready objects.

    Returns:
        SimplicialComplex: The input complex.
    """
    return sc

def build_s1():
    """Construct a minimal circle S^1.

    What is Being Computed?:
        A simplicial complex with 3 vertices and 3 edges.

    Returns:
        SimplicialComplex: The circle.
    """
    return SimplicialComplex.from_simplices([[0, 1], [1, 2], [2, 0]])

def build_klein_bottle():
    """Construct a minimal triangulation of the Klein Bottle.

    What is Being Computed?:
        A non-orientable surface with Euler characteristic 0.

    Returns:
        SimplicialComplex: The Klein bottle.
    """
    faces = [
        [0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4], [2, 0, 3], [2, 3, 5],
        [3, 4, 7], [3, 7, 6], [4, 5, 8], [4, 8, 7], [5, 3, 6], [5, 6, 8],
        [6, 7, 2], [6, 2, 1], [7, 8, 0], [7, 0, 2], [8, 6, 1], [8, 1, 0],
    ]
    return SimplicialComplex.from_simplices(faces)

def build_torus_param(nx, ny):
    """Build a torus with configurable grid density.

    What is Being Computed?:
        A torus triangulation based on an nx by ny grid.

    Args:
        nx (int): Number of vertices in x-direction.
        ny (int): Number of vertices in y-direction.

    Returns:
        SimplicialComplex: A parameterized torus.
    """
    simplices = []
    for i in range(nx):
        for j in range(ny):
            v0 = i * ny + j
            v1 = ((i + 1) % nx) * ny + j
            v2 = i * ny + (j + 1) % ny
            v3 = ((i + 1) % nx) * ny + (j + 1) % ny
            simplices.append([v0, v1, v2])
            simplices.append([v1, v2, v3])
    return SimplicialComplex.from_simplices(simplices)

def build_cube():
    """Construct a triangulated 2-sphere as the boundary of a cube.

    What is Being Computed?:
        A 2-sphere with 8 vertices and 12 faces.

    Returns:
        SimplicialComplex: The cube surface.
    """
    faces = [
        [0, 1, 2], [1, 2, 3], [4, 5, 6], [5, 6, 7], [0, 1, 5], [0, 5, 4],
        [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 3, 7], [1, 7, 5],
    ]
    return SimplicialComplex.from_simplices(faces)

def get_surfaces():
    """Retrieve a list of standard surfaces with their expected invariants.

    What is Being Computed?:
        A catalog of surfaces for bulk testing of homology and Betti numbers.

    Returns:
        list[tuple]: List of (name, builder_func, h_ranks, h_torsion, chi).
    """
    surfaces = [
        ("Tetrahedron", build_tetrahedron, {0: 1, 1: 0, 2: 1}, {1: []}, 2),
        ("Octahedron", build_octahedron, {0: 1, 1: 0, 2: 1}, {1: []}, 2),
        ("Icosahedron", build_icosahedron, {0: 1, 1: 0, 2: 1}, {1: []}, 2),
        ("Cube", build_cube, {0: 1, 1: 0, 2: 1}, {1: []}, 2),
        ("RP2", build_rp2, {0: 1, 1: 0, 2: 0}, {1: [2]}, 1),
        ("S1", build_s1, {0: 1, 1: 1}, {}, 0),
    ]
    for nx in [3, 4, 5]:
        for ny in [3, 4, 5]:
            surfaces.append(
                (
                    f"Torus_{nx}x{ny}",
                    lambda nx=nx, ny=ny: build_torus_param(nx, ny),
                    {0: 1, 1: 2, 2: 1},
                    {1: []},
                    0,
                )
            )
    return surfaces

def get_3_manifolds():
    """Retrieve a list of 3-manifolds for testing higher-dimensional invariants.

    Returns:
        list[tuple]: List of 3-manifold data.
    """
    return [
        ("S3", build_s3, {0: 1, 1: 0, 2: 0, 3: 1}, {1: [], 2: []}, 0),
    ]
