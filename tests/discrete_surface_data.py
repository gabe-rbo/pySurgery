import pytest

try:
    import gudhi
except ImportError:
    pytest.skip("GUDHI not available", allow_module_level=True)

from pysurgery.integrations.gudhi_bridge import extract_complex_data
from pysurgery.core.complexes import ChainComplex


def build_tetrahedron():
    st = gudhi.SimplexTree()
    for f in [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]:
        st.insert(f)
    return st


def build_octahedron():
    st = gudhi.SimplexTree()
    faces = [
        [0, 2, 4],
        [0, 2, 5],
        [0, 3, 4],
        [0, 3, 5],
        [1, 2, 4],
        [1, 2, 5],
        [1, 3, 4],
        [1, 3, 5],
    ]
    for f in faces:
        st.insert(f)
    return st


def build_icosahedron():
    st = gudhi.SimplexTree()
    faces = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ]
    for f in faces:
        st.insert(f)
    return st


def build_torus():
    st = gudhi.SimplexTree()
    faces = [
        [0, 1, 4],
        [0, 4, 3],
        [1, 2, 5],
        [1, 5, 4],
        [2, 0, 3],
        [2, 3, 5],
        [3, 4, 7],
        [3, 7, 6],
        [4, 5, 8],
        [4, 8, 7],
        [5, 3, 6],
        [5, 6, 8],
        [6, 7, 1],
        [6, 1, 0],
        [7, 8, 2],
        [7, 2, 1],
        [8, 6, 0],
        [8, 0, 2],
    ]
    for f in faces:
        st.insert(f)
    return st


def build_rp2():
    st = gudhi.SimplexTree()
    faces = [
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 5],
        [0, 5, 1],
        [1, 2, 4],
        [2, 3, 5],
        [3, 4, 1],
        [4, 5, 2],
        [5, 1, 3],
    ]
    for f in faces:
        st.insert(f)
    return st


def build_s3():
    st = gudhi.SimplexTree()
    # S3 is the boundary of the 4-simplex
    for i in range(5):
        face = [j for j in range(5) if j != i]
        st.insert(face)
    return st


def to_complex(st):
    boundaries, cells, _, _ = extract_complex_data(st)
    return ChainComplex(
        boundaries=boundaries, dimensions=list(cells.keys()), cells=cells
    )


def build_s1():
    st = gudhi.SimplexTree()
    st.insert([0, 1])
    st.insert([1, 2])
    st.insert([2, 0])
    return st


def build_klein_bottle():
    # 3x3 grid for KB: 9 vertices, 18 triangles
    st = gudhi.SimplexTree()
    # (x, y) = 3x + y
    # x ~ x+3, y ~ y+3 with twist in y.
    faces = [
        [0, 1, 4],
        [0, 4, 3],
        [1, 2, 5],
        [1, 5, 4],
        [2, 0, 3],
        [2, 3, 5],
        [3, 4, 7],
        [3, 7, 6],
        [4, 5, 8],
        [4, 8, 7],
        [5, 3, 6],
        [5, 6, 8],
        # identify bottom edge with top with reverse!
        [6, 7, 2],
        [6, 2, 1],
        [7, 8, 0],
        [7, 0, 2],
        [8, 6, 1],
        [8, 1, 0],
    ]
    for f in faces:
        st.insert(f)
    return st


def build_torus_param(nx, ny):
    st = gudhi.SimplexTree()
    # grid nx x ny
    for i in range(nx):
        for j in range(ny):
            v0 = i * ny + j
            v1 = ((i + 1) % nx) * ny + j
            v2 = i * ny + (j + 1) % ny
            v3 = ((i + 1) % nx) * ny + (j + 1) % ny
            st.insert([v0, v1, v2])
            st.insert([v1, v2, v3])
    return st


def build_cube():
    st = gudhi.SimplexTree()
    # 8 vertices of cube, 12 triangular faces
    faces = [
        [0, 1, 2],
        [1, 2, 3],  # front
        [4, 5, 6],
        [5, 6, 7],  # back
        [0, 1, 5],
        [0, 5, 4],  # bottom
        [2, 3, 7],
        [2, 7, 6],  # top
        [0, 2, 6],
        [0, 6, 4],  # left
        [1, 3, 7],
        [1, 7, 5],  # right
    ]
    for f in faces:
        st.insert(f)
    return st


def get_surfaces():
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
    return [
        ("S3", build_s3, {0: 1, 1: 0, 2: 0, 3: 1}, {1: [], 2: []}, 0),
    ]
