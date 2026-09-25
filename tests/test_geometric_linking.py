"""Winding and linking numbers of embedded simplicial cycles, exactly.

Overview:
    An intersection count, not an integral -- in R^2, R^3, R^4 and R^5, with the sign
    conventions pinned: +1 inside a positively oriented simplex boundary and inside a
    counterclockwise circle, lk(B, A) = (-1)^(pq+1) lk(A, B), agreement with the Gauss
    integral in sign, and every refusal named. The Python and Julia backends must give
    the same integers.
"""
import numpy as np
import pytest

import exact_triangulations as T
import synthetic_curves as S
from pysurgery.bridge.julia_bridge import julia_engine
from pysurgery.core.exceptions import NonGenericConfigurationError, UndefinedInvariantError
from pysurgery.knots import geometric_linking as GL
from pysurgery.topology.fundamental_cycles import fundamental_cycle

BACKENDS = ["python"] + (["julia"] if julia_engine.available else [])


def _sphere_cycle(n_sub=1, radius=1.0, embed=None, shift=None):
    """A triangulated 2-sphere as a fundamental 2-cycle, optionally placed in a
    3-dimensional coordinate subspace of a bigger space."""
    K, V = T.subdivided_sphere(n_sub, radius)
    z = fundamental_cycle(K, 2, backend="python").as_pairs()
    if embed is not None:
        W = np.zeros((len(V), embed[0]))
        W[:, list(embed[1])] = V
        V = W
    if shift is not None:
        V = V + np.asarray(shift)
    return z, V


@pytest.mark.parametrize("backend", BACKENDS)
def test_winding_number_sign_convention(backend):
    bd = [((1, 2, 3), 1), ((0, 2, 3), -1), ((0, 1, 3), 1), ((0, 1, 2), -1)]
    X = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    assert GL.winding_number(np.array([0.1, 0.2, 0.15]), bd, X, backend=backend) == 1
    assert GL.winding_number(np.array([1.0, 1.0, 1.0]), bd, X, backend=backend) == 0
    circ = S.circle(50)[:, :2]
    cyc = GL.polygon_cycle(50)
    assert GL.is_cycle(cyc)
    assert GL.winding_number(np.zeros(2), cyc, circ, backend=backend) == 1
    assert GL.winding_number(np.zeros(2), [(s, -c) for s, c in cyc], circ, backend=backend) == -1
    assert GL.winding_number(np.array([3.0, 0.1]), cyc, circ, backend=backend) == 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_winding_numbers_of_a_triangulated_sphere(backend):
    z, V = _sphere_cycle(2)
    pts = np.array([[0, 0, 0], [0.3, -0.2, 0.1], [2, 0, 0], [0, 0, -1.5]])
    w = GL.winding_numbers(pts, z, V, backend=backend)
    assert abs(w[0]) == 1 and w[1] == w[0] and w[2] == 0 and w[3] == 0
    with pytest.raises(UndefinedInvariantError):
        GL.winding_number(V[5], z, V, backend=backend)        # on the cycle


@pytest.mark.parametrize("backend", BACKENDS)
def test_curve_linking_in_r3(backend):
    hopf = S.hopf_link(60)
    assert GL.curve_linking(*hopf, backend=backend) == -1
    assert abs(GL.curve_linking(*S.torus_link(3, 240), backend=backend)) == 3
    assert GL.curve_linking(*S.unlink(2, 40), backend=backend) == 0
    # reversing one component flips the sign
    assert GL.curve_linking(hopf[0][::-1], hopf[1], backend=backend) == 1
    # and the Gauss integral agrees in sign and (approximately) in value
    g = GL.curve_gauss_integral(*S.hopf_link(400))
    assert abs(g - (-1)) < 0.05


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_circle_threading_a_two_sphere_in_r4(backend):
    """p = 1, q = 2, m = 4: the circle crosses the sphere's hyperplane once inside the
    ball and once outside, so |lk| = 1. Swapping gives (-1)^(pq+1) = -1."""
    t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    C = np.stack([1 + np.cos(t), 0 * t, 0 * t, np.sin(t)], 1)
    c = GL.polygon_cycle(48)
    z, V = _sphere_cycle(1, embed=(4, (0, 1, 2)))
    a = GL.simplicial_linking(c, C, z, V, 4, backend=backend)
    b = GL.simplicial_linking(z, V, c, C, 4, backend=backend)
    assert abs(a) == 1 and b == -a
    g = GL.gauss_linking_estimate(c, C, z, V, 4)
    assert np.sign(g) == a and abs(g - a) < 0.3, g            # an estimate, same sign


@pytest.mark.parametrize("backend", BACKENDS)
def test_two_linked_two_spheres_in_r5(backend):
    """p = q = 2, m = 5: S^2 in (x1, x2, x3) and S^2 in (x1, x4, x5) centred at e1.
    Disjoint, |lk| = 1, and lk(B, A) = (-1)^(pq+1) lk(A, B) = -lk(A, B)."""
    zA, VA = _sphere_cycle(1, embed=(5, (0, 1, 2)))
    zB, VB = _sphere_cycle(1, embed=(5, (0, 3, 4)), shift=[1, 0, 0, 0, 0])
    a = GL.simplicial_linking(zA, VA, zB, VB, 5, backend=backend)
    b = GL.simplicial_linking(zB, VB, zA, VA, 5, backend=backend)
    assert abs(a) == 1 and b == -a
    # pulled apart along x2 they are unlinked
    assert GL.simplicial_linking(zA, VA, zB, VB + np.array([0, 5.0, 0, 0, 0]), 5, backend=backend) == 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_refusals(backend):
    z, V = _sphere_cycle(1)
    c = GL.polygon_cycle(20)
    C = S.circle(20)
    with pytest.raises(UndefinedInvariantError):
        GL.simplicial_linking(c, C, c, np.c_[C, np.zeros(20)] + 5, 4, backend=backend)  # p + q != m - 1
    with pytest.raises(ValueError):
        GL.simplicial_linking(c[:-1], C, c, C + 5, 3, backend=backend)           # not a cycle
    with pytest.raises(UndefinedInvariantError):
        GL.winding_number(np.zeros(3), c, C, backend=backend)                   # wrong dimension
    # two circles that intersect have no linking number
    A = S.circle(40)
    B = np.stack([1 + np.cos(np.linspace(0, 2 * np.pi, 40, endpoint=False)),
                  np.sin(np.linspace(0, 2 * np.pi, 40, endpoint=False)), np.zeros(40)], 1)
    with pytest.raises(NonGenericConfigurationError):
        GL.curve_linking(A, B, backend=backend)


def test_definedness_is_decided_before_anything_runs():
    d = GL.definedness([1, 1], 3)
    assert d.linking_ok and d.a2_ok and not d.enclosure_ok
    assert GL.definedness([2, 2], 5).linking_ok
    assert not GL.definedness([2, 2], 7).linking_ok               # identically zero there
    assert GL.definedness([2, 0], 3).enclosure_ok


def test_chain_boundary_respects_vertex_order():
    # the same oriented triangle listed in two vertex orders is the same chain
    assert GL.chain_boundary([((0, 1, 2), 1), ((1, 0, 2), 1)]) == {}
    assert GL.is_cycle([((0, 1), 1), ((1, 2), 1), ((2, 0), 1)])
