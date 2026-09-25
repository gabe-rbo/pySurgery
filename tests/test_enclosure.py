"""Enclosure: is a point inside the region a codimension-1 complex bounds?

Overview:
    The degree (winding number against each generator of H_(m-1)) is the verdict; the
    even-odd parity is the independent check, and it is expected to DISAGREE inside
    nested shells -- it counts the region mod 2. The convex hull is a necessary
    condition only (the hole of a torus is in the hull and not enclosed).
"""
import numpy as np

import exact_triangulations as T
from pysurgery.geometry import enclosure as EN


def _nested():
    Ko, Vo = T.subdivided_sphere(2, 2.0)
    Ki, Vi = T.subdivided_sphere(2, 1.0)
    no = len(Vo)
    both = T.sc(list(Ko.n_simplices(2)) + [tuple(v + no for v in s) for s in Ki.n_simplices(2)])
    return np.vstack([Vo, Vi]), Ko, Vo, both


def test_one_sphere_encloses_its_inside_and_nothing_else():
    _X, Ko, Vo, _ = _nested()
    pts = np.array([[0.0, 0.0, 0.0], [1.5, 0.1, -0.2], [3.0, 0.0, 0.0], [0.2, 2.5, 0.0]])
    rep = EN.enclosure_report(pts, Ko, Vo, 3)
    assert rep.defined and rep.n_bounding_classes == 1 and rep.complex_certified
    assert rep.enclosed.tolist() == [True, True, False, False]
    assert np.abs(rep.winding[:, 0]).tolist() == [1, 1, 0, 0]
    assert rep.parity.tolist() == [1, 1, 0, 0] and rep.agreement == 1.0


def test_nested_shells_degree_versus_parity():
    """Inside both shells the winding numbers are (1, 1): enclosed, twice. The parity
    rule counts two crossings and says 'outside' -- the disagreement is reported point
    by point, and the degree stands."""
    X, _Ko, _Vo, both = _nested()
    pts = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.1], [3.0, 0.0, 0.0]])
    rep = EN.enclosure_report(pts, both, X, 3)
    assert rep.n_bounding_classes == 2
    assert rep.enclosed.tolist() == [True, True, False]
    assert (np.abs(rep.winding) > 0).sum(1).tolist() == [2, 1, 0]
    assert rep.parity.tolist() == [0, 1, 0]
    assert rep.disputed.tolist() == [True, False, False]


def test_the_hole_of_a_torus_is_in_the_hull_and_not_enclosed():
    K, V = T.torus_surface(12, 12, R=3.0, r=1.0)
    pts = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, -3.0, 0.2], [6.0, 0.0, 0.0]])
    rep = EN.enclosure_report(pts, K, V, 3)
    assert rep.enclosed.tolist() == [False, True, True, False]
    assert EN.in_convex_hull(V, pts).tolist() == [True, True, True, False]
    assert np.isclose(EN.hull_fraction(V, pts), 0.75)
    assert len(EN.integer_cycle_generators(K, 2)) == 1


def test_non_pseudomanifold_generators_come_from_the_exact_snf_kernel():
    """Three disks glued along their common boundary circle (a 'theta' 2-complex) branch
    along every rim edge, so they are not a pseudomanifold; the exact SNF kernel still
    spans H_2 = Z^2, and a cone over three points above/below/beside the rim puts
    points inside exactly the right bubbles."""
    rim = [(0, 1), (1, 2), (0, 2)]
    K = T.sc([e + (a,) for a in (3, 4, 5) for e in rim])
    cycles = EN.integer_cycle_generators(K, 2)
    assert len(cycles) == 2 and all(len(c) >= 6 for c in cycles)
    X = np.array([[1, 0, 0], [-0.5, 0.87, 0], [-0.5, -0.87, 0],
                  [0, 0, 1.0], [0, 0, -1.0], [0, 0, 0.3]], float)
    rep = EN.enclosure_report(np.array([[0, 0, 0.6], [0, 0, 0.1], [0, 0, -0.5], [3, 0, 0]]), K, X, 3)
    assert rep.n_bounding_classes == 2
    assert rep.enclosed.tolist() == [True, True, True, False]


def test_enclosure_is_undefined_without_a_bounding_cycle_or_in_the_wrong_codimension():
    D = T.disk(6)
    V = np.c_[np.r_[0, np.cos(np.arange(6) * np.pi / 3)], np.r_[0, np.sin(np.arange(6) * np.pi / 3)],
              np.zeros(7)]
    rep = EN.enclosure_report(np.zeros((1, 3)) + 0.1, D, V, 3)
    assert not rep.defined and "beta_2" in rep.reason
    K, Vs = T.octahedron()
    assert not EN.enclosure_report(np.zeros((1, 4)), K, np.c_[Vs, np.zeros(6)], 4).defined


def test_mod2_cycle_detection():
    K, _ = T.octahedron()
    assert EN.is_mod2_cycle(K, 3)
    assert not EN.is_mod2_cycle(T.disk(6), 3)
