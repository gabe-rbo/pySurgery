"""Knot and link invariants of closed polygons from one CERTIFIED generic projection.

Overview:
    Checked against values known from knot tables, against the invariance that makes
    them invariants (every generic projection must give the same integer), against the
    exact cone-intersection linking number of ``knots.geometric_linking`` (a different
    piece of mathematics, same sign), and Python against Julia.
"""
import numpy as np
import pytest

import synthetic_curves as S
from pysurgery.bridge.julia_bridge import julia_engine
from pysurgery.core.exceptions import NonGenericConfigurationError, UndefinedInvariantError
from pysurgery.knots import diagrams as KD
from pysurgery.knots import geometric_linking as GL

BACKENDS = ["python"] + (["julia"] if julia_engine.available else [])


@pytest.mark.parametrize("backend", BACKENDS)
def test_linking_numbers_of_standard_links(backend):
    H = S.hopf_link(120)
    assert KD.diagram_linking_number(*H, backend=backend) == -1
    assert KD.diagram_linking_number(H[1], H[0], backend=backend) == -1   # symmetric for curves
    assert KD.diagram_linking_number(H[0][::-1], H[1], backend=backend) == 1
    mirror = [c * np.array([1, 1, -1]) for c in H]
    assert KD.diagram_linking_number(*mirror, backend=backend) == 1       # mirror image
    for k in (2, 3):
        assert abs(KD.diagram_linking_number(*S.torus_link(k, 300), backend=backend)) == k
    assert KD.diagram_linking_number(*S.unlink(2, 80), backend=backend) == 0
    assert not KD.diagram_linking_matrix(S.borromean_rings(200), backend=backend).any()


def test_three_computations_of_the_linking_number_agree():
    """The crossing count, the intersection with a cone (a Seifert chain), and the
    Gauss integral -- three different pieces of mathematics, one sign."""
    for pair in (S.hopf_link(200), S.torus_link(2, 400)):
        a = KD.diagram_linking_number(*pair)
        b = GL.curve_linking(*pair)
        g = GL.curve_gauss_integral(*pair)
        assert a == b and abs(g - a) < 0.05, (a, b, g)


def test_the_invariant_does_not_depend_on_the_projection():
    H = S.hopf_link(80)
    T = S.trefoil(240)
    for fr in KD.projection_frames(12):
        assert KD.diagram_linking_number(*H, D=KD.knot_diagram(H, frame=fr)) == -1
        assert KD.a2_from_diagram(KD.knot_diagram([T], frame=fr)) == 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_casson_invariant_against_the_knot_table(backend):
    assert KD.casson_a2(S.circle(60), backend=backend) == 0
    assert KD.casson_a2(S.trefoil(300), backend=backend) == 1
    assert KD.casson_a2(S.trefoil(300, mirror=True), backend=backend) == 1    # mirror-blind
    assert KD.casson_a2(S.figure_eight(300), backend=backend) == -1
    for p, q in ((2, 3), (2, 5), (3, 4)):
        assert KD.casson_a2(S.torus_knot(p, q, 500), backend=backend) == (p * p - 1) * (q * q - 1) // 24


def test_writhe_is_a_diagram_quantity_and_the_basepoint_does_not_matter():
    D = KD.knot_diagram([S.figure_eight(300)])
    assert {KD.a2_from_diagram(D, 0, b) for b in (0.5, 71.5, 150.5, 299.5)} == {-1}
    assert isinstance(KD.writhe(D, 0), int)


@pytest.mark.parametrize("backend", BACKENDS)
def test_milnor_triple_linking_of_the_borromean_rings(backend):
    A, B, C = S.borromean_rings(200)
    mu = KD.diagram_milnor_mu123(A, B, C, backend=backend)
    assert abs(mu) == 1
    for fr in KD.projection_frames(6):
        assert KD.diagram_milnor_mu123(A, B, C, D=KD.knot_diagram([A, B, C], frame=fr)) == mu
    assert KD.diagram_milnor_mu123(B, C, A, backend=backend) == mu
    assert KD.diagram_milnor_mu123(C, A, B, backend=backend) == mu
    assert KD.diagram_milnor_mu123(B, A, C, backend=backend) == -mu
    assert KD.diagram_milnor_mu123(A[::-1], B, C, backend=backend) == -mu


def test_milnor_vanishes_on_the_unlink_and_is_refused_when_undefined():
    assert KD.diagram_milnor_mu123(*S.unlink(3, 60)) == 0
    H = S.hopf_link(60)
    with pytest.raises(UndefinedInvariantError):
        KD.diagram_milnor_mu123(H[0], H[1], S.circle(40, centre=(10, 0, 0)))


@pytest.mark.parametrize("backend", BACKENDS)
def test_intersecting_polygons_are_refused(backend):
    A = S.circle(40)
    t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    B = np.stack([1 + np.cos(t), np.zeros(40), 1 + np.sin(t)], 1)  # its vertex 30 IS A's vertex 0
    with pytest.raises(NonGenericConfigurationError):
        KD.knot_diagram([A, B], backend=backend)


@pytest.mark.skipif(not julia_engine.available, reason="Julia backend unavailable")
def test_python_and_julia_diagrams_agree():
    for curves in ([S.trefoil(200)], S.hopf_link(80), S.borromean_rings(120), [S.torus_knot(3, 4, 300)]):
        a = KD.knot_diagram(curves, backend="python")
        b = KD.knot_diagram(curves, backend="julia")
        assert a.tries == b.tries and len(a.crossings) == len(b.crossings)
        assert [(c.over, c.under, c.sign) for c in a.crossings] == \
            [(c.over, c.under, c.sign) for c in b.crossings]
        assert np.allclose([c.under_pos for c in a.crossings], [c.under_pos for c in b.crossings])
