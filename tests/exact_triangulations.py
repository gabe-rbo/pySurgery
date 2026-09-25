"""Exact triangulations whose topology is known before anything runs.

Ground truth for the local-homology, fundamental-cycle, finite-space and lower-star
Morse tests: a sampled shape's complex depends on a scale, and a wrong answer could
not be blamed on the invariant rather than the sample. These are the objects the
tests lean on, with coordinates where an embedding matters (ported from
TabularTopology's ``topology.complexes.triangulations``).

    boundary_of_simplex(d)   S^(d-1) with d+1 vertices
    octahedron()             S^2, 6 vertices, embedded in R^3
    subdivided_sphere(n)     S^2 at any resolution (6, 18, 66, ... vertices)
    torus(m, n)              T^2 as an m x n periodic grid (m, n >= 3)
    torus_surface(m, n)      the same, embedded in R^3
    klein_bottle(m, n)       the Klein bottle (H_1 = Z + Z/2)
    projective_plane()       RP^2 on 6 vertices (H_1 = Z/2, H_2 = 0)
    mobius_band(m)           the Mobius band (non-orientable, one boundary circle)
    disk(k)                  a fan of k triangles
    pinched_spheres()        two S^2 sharing a vertex
    wedge_of_disks(k)        two disks meeting at their centre only
"""
import itertools

import numpy as np

from pysurgery.topology.complexes import SimplicialComplex


def sc(simplices):
    return SimplicialComplex.from_simplices(list(simplices), close_under_faces=True)


def boundary_of_simplex(d):
    return sc(itertools.combinations(range(d + 1), d))


def octahedron():
    V = np.array([[1., 0, 0], [-1, 0, 0], [0, 1., 0], [0, -1, 0], [0, 0, 1.], [0, 0, -1]])
    F = [(0, 2, 4), (1, 2, 4), (1, 3, 4), (0, 3, 4),
         (0, 2, 5), (1, 2, 5), (1, 3, 5), (0, 3, 5)]
    return sc(F), V


def subdivided_sphere(n_sub=2, radius=1.0, centre=(0., 0., 0.)):
    c = np.asarray(centre, dtype=np.float64)
    V = [np.array(v, dtype=np.float64) for v in
         [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]]
    F = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
         (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    for _ in range(int(n_sub)):
        mid, nf = {}, []

        def midpoint(i, j):
            key = (min(i, j), max(i, j))
            if key not in mid:
                w = V[i] + V[j]
                V.append(w / np.linalg.norm(w))
                mid[key] = len(V) - 1
            return mid[key]

        for (a, b, cc) in F:
            ab, bc, ca = midpoint(a, b), midpoint(b, cc), midpoint(cc, a)
            nf += [(a, ab, ca), (ab, b, bc), (ca, bc, cc), (ab, bc, ca)]
        F = nf
    return sc(F), np.vstack(V) * float(radius) + c


def _grid(m, n, flip):
    def v(i, j):
        if flip and (i // m) % 2:
            j = -j
        return (i % m) * n + (j % n)
    tris = []
    for i in range(m):
        for j in range(n):
            a, b, c, d = v(i, j), v(i + 1, j), v(i, j + 1), v(i + 1, j + 1)
            tris += [(a, b, c), (b, d, c)]
    return sc(tris)


def torus(m=4, n=4):
    return _grid(m, n, flip=False)


def klein_bottle(m=5, n=5):
    return _grid(m, n, flip=True)


def torus_surface(m=12, n=12, R=3.0, r=1.0):
    K = torus(m, n)
    phi = np.arange(m) * 2 * np.pi / m
    th = np.arange(n) * 2 * np.pi / n
    P, T = np.meshgrid(phi, th, indexing="ij")
    V = np.stack([(R + r * np.cos(T)) * np.cos(P),
                  (R + r * np.cos(T)) * np.sin(P),
                  r * np.sin(T)], -1).reshape(m * n, 3)
    return K, V


def projective_plane():
    return sc([(0, 1, 4), (0, 1, 5), (0, 2, 3), (0, 2, 4), (0, 3, 5),
               (1, 2, 3), (1, 2, 5), (1, 3, 4), (2, 4, 5), (3, 4, 5)])


def mobius_band(m=6):
    def v(i, side):
        if (i // m) % 2:
            side = 1 - side
        return (i % m) * 2 + side
    tris = []
    for i in range(m):
        a, b, c, d = v(i, 0), v(i, 1), v(i + 1, 0), v(i + 1, 1)
        tris += [(a, b, c), (b, d, c)]
    return sc(tris)


def disk(k=6):
    return sc([(0, i, i % k + 1) for i in range(1, k + 1)])


def pinched_spheres():
    a = list(itertools.combinations([0, 1, 2, 3], 3))
    b = list(itertools.combinations([0, 4, 5, 6], 3))
    return sc(a + b)


def wedge_of_disks(k=6):
    A = [(0, i, i % k + 1) for i in range(1, k + 1)]
    B = [(0, k + i, k + i % k + 1) for i in range(1, k + 1)]
    return sc(A + B)


def betti(K):
    """Betti numbers as a list, degree 0 upward (python backend, exact)."""
    h = K.homology(backend="python")
    return [h[d][0] for d in sorted(h)]
