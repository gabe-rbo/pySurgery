"""Holonomy of the tangent-frame connection, and the scale window.

Overview:
    Orientability decided over EVERY cycle of the graph (the orientation double cover),
    rotation angles, gauge invariance, and Gauss-Bonnet as the ground truth for the
    angle. The scale window is checked on shapes with a known reach (a circle of radius
    R has reach R) and a known minimum spanning tree.
"""
import numpy as np
import pytest

import synthetic_curves as S
from pysurgery.bridge.julia_bridge import julia_engine
from pysurgery.geometry import holonomy as HO
from pysurgery.geometry import scale_window as SW

BACKENDS = ["python"] + (["julia"] if julia_engine.available else [])


@pytest.mark.parametrize("backend", BACKENDS)
def test_orientability_of_sampled_surfaces(backend):
    rng = np.random.default_rng(0)
    mob = HO.orientation_report(S.mobius_band_sample(1500, rng=rng), 2, backend=backend)
    cyl = HO.orientation_report(S.cylinder_sample(1500, rng=rng), 2, backend=backend)
    kb = HO.orientation_report(S.klein_bottle_sample(2500, rng=rng), 2, backend=backend)
    sph = HO.orientation_report(S.fibonacci_sphere(800), 2, backend=backend)
    assert not mob.orientable and not kb.orientable
    assert cyl.orientable and sph.orientable
    assert cyl.certified and sph.certified                    # no unreliable transport
    assert mob.n_components == cyl.n_components == 1


def test_every_non_tree_edge_is_a_loop():
    X = S.cylinder_sample(400, rng=np.random.default_rng(1))
    E = HO.knn_graph(X, 6)
    loops = HO.fundamental_loops(len(X), E)
    assert len(loops) == len(E) - len(X) + 1                  # one component
    Es = set(E)
    for lp in loops:
        for a, b in zip(lp, lp[1:] + lp[:1]):
            assert (min(a, b), max(a, b)) in Es


def _exact_sphere_frames(P):
    """Orthonormal tangent frames of the unit sphere: (e_phi, e_theta)."""
    x, y, z = P.T
    phi = np.arctan2(y, x)
    th = np.arccos(np.clip(z, -1, 1))
    e_phi = np.stack([-np.sin(phi), np.cos(phi), 0 * phi], 1)
    e_th = np.stack([np.cos(th) * np.cos(phi), np.cos(th) * np.sin(phi), -np.sin(th)], 1)
    return np.stack([e_phi, e_th], 2)                         # (n, 3, 2)


def test_gauss_bonnet_on_a_latitude_circle():
    """Transport around the circle at colatitude theta rotates by the enclosed solid
    angle 2 pi (1 - cos theta)."""
    loop, omega = S.latitude_loop(400, colatitude=0.7)
    F = _exact_sphere_frames(loop)
    H = HO.holonomy_along(F, list(range(len(loop))))
    assert np.isclose(np.linalg.det(H), 1.0)
    assert abs(abs(HO.rotation_angle(H)) - omega) < 1e-3


def test_holonomy_is_gauge_invariant():
    loop, _ = S.latitude_loop(200, colatitude=1.1)
    F = _exact_sphere_frames(loop)
    rng = np.random.default_rng(3)
    G = np.empty_like(F)
    for i in range(len(F)):
        Q, _ = np.linalg.qr(rng.normal(size=(2, 2)))          # any O(2), reflections too
        G[i] = F[i] @ Q
    H1 = HO.holonomy_along(F, list(range(len(loop))))
    H2 = HO.holonomy_along(G, list(range(len(loop))))
    assert np.isclose(np.linalg.det(H1), np.linalg.det(H2))
    assert np.allclose(HO.rotation_angles(H1), HO.rotation_angles(H2))


def test_rotation_angles_of_known_orthogonal_maps():
    assert np.allclose(HO.rotation_angles(np.diag([1.0, -1.0])), [np.pi, 0.0])
    t = 0.8
    R = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    assert np.allclose(HO.rotation_angles(R), [t, 0.0])
    assert np.allclose(HO.rotation_angles(np.diag([1.0, 1.0, -1.0])), [np.pi, 0.0, 0.0])
    assert np.isclose(HO.rotation_angle(R[:2, :2]), t)


@pytest.mark.parametrize("backend", BACKENDS)
def test_holonomy_report_counts_orientation_reversing_loops(backend):
    rng = np.random.default_rng(4)
    cyl = HO.holonomy_report(S.cylinder_sample(800, rng=rng), 2, backend=backend)
    mob = HO.holonomy_report(S.mobius_band_sample(800, rng=rng), 2, backend=backend)
    assert cyl.n_negative == 0 and mob.n_negative > 0
    assert cyl.orientation.orientable and not mob.orientation.orientable


@pytest.mark.parametrize("backend", BACKENDS)
def test_reach_of_a_circle_is_its_radius(backend):
    """The medial axis of a round circle is its centre: reach = R. Federer's formula
    over every pair recovers it up to the tangent estimate."""
    for R in (1.0, 2.5):
        X = S.circle(400, radius=R)
        tau = SW.federer_reach(X, 1, k=6, backend=backend)
        assert abs(tau - R) < 0.02 * R, (tau, R)
    # a flat sample has infinite reach
    line = np.c_[np.linspace(0, 1, 50), np.zeros(50), np.zeros(50)]
    assert SW.federer_reach(line, 1, k=4, backend=backend) == np.inf


def test_scale_window_constraints():
    X = S.circle(200, radius=1.0)
    step = 2 * np.sin(np.pi / 200)                            # MST edge of a regular 200-gon
    w = SW.scale_window(X, dim=1, k_tangent=6)
    assert np.isclose(w.r_min, step / 2)
    assert abs(w.r_max_reach - np.sqrt(3 / 5)) < 0.03 and not w.empty
    grid = w.grid(5)
    assert np.isclose(grid[0], w.r_min) and np.isclose(grid[-1], w.r_max)
    # a second class 0.01 away empties the window
    other = S.circle(200, radius=1.01)
    w2 = SW.scale_window(X, other, dim=1, k_tangent=6)
    assert w2.empty and w2.r_max_gap <= 0.005 + 1e-12
    with pytest.raises(ValueError):
        SW.connectivity_radius(np.vstack([X, X[:1]]))         # duplicates refused
