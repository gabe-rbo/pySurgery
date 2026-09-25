"""Fundamental cycles of closed pseudomanifolds -- and the refusals.

Overview:
    A fundamental cycle is summed from coherently signed top simplices; every hypothesis
    that makes that sum THE generator of H_p is checked, and each failure is a named
    refusal, never a cycle of the wrong class. The cycles are cross-checked against
    pySurgery's own boundary operators (sign conventions must agree) and fed to the
    existing Poincare-duality cap-product machinery.
"""
import numpy as np
import pytest

import exact_triangulations as T
from pysurgery.bridge.julia_bridge import julia_engine
from pysurgery.core.exceptions import NoFundamentalClassError
from pysurgery.topology import fundamental_cycles as FC

BACKENDS = ["python"] + (["julia"] if julia_engine.available else [])


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_octahedron_and_the_torus_have_verified_fundamental_cycles(backend):
    for K, n_top in ((T.octahedron()[0], 8), (T.torus(4, 4), 32), (T.subdivided_sphere(2)[0], 128)):
        z = FC.fundamental_cycle(K, 2, backend=backend)
        assert z.verified and len(z.simplices) == n_top
        assert set(np.abs(z.signs).tolist()) == {1}
        chain = z.as_chain(K)
        # the cycle condition in pySurgery's own boundary convention
        assert not np.any(K.boundary_matrix(2) @ chain)
        assert int(np.abs(chain).sum()) == n_top


@pytest.mark.parametrize("backend", BACKENDS)
def test_s3_fundamental_class(backend):
    K = T.boundary_of_simplex(4)
    z = FC.fundamental_cycle(K, 3, backend=backend)
    assert len(z.simplices) == 5 and not np.any(K.boundary_matrix(3) @ z.as_chain(K))


@pytest.mark.parametrize("backend", BACKENDS)
def test_non_orientable_surfaces_have_none_over_z_but_do_over_z2(backend):
    for K in (T.klein_bottle(5, 5), T.projective_plane()):
        with pytest.raises(NoFundamentalClassError, match="non-orientable"):
            FC.fundamental_cycle(K, 2, backend=backend)
        assert not FC.is_orientable_pseudomanifold(K, 2, backend=backend)
        assert FC.top_homology_basis(K, 2, backend=backend) == []   # H_2(K; Z) = 0
        z2 = FC.fundamental_cycle(K, 2, coefficient_ring="Z2", backend=backend)
        assert z2.verified and set(z2.signs) == {1}
    assert FC.is_orientable_pseudomanifold(T.torus(3, 3), 2, backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_each_failed_hypothesis_is_refused_by_name(backend):
    with pytest.raises(NoFundamentalClassError, match="not closed"):
        FC.fundamental_cycle(T.disk(5), 2, backend=backend)
    branch = T.sc([(0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 3), (1, 2, 3)])
    with pytest.raises(NoFundamentalClassError, match="branching"):
        FC.fundamental_cycle(branch, 2, backend=backend)
    solid = T.sc([(0, 1, 2, 3)])
    # its boundary sums to a cycle that BOUNDS in K: refused, not returned
    with pytest.raises(NoFundamentalClassError, match="dimension"):
        FC.fundamental_cycle(solid, 2, backend=backend)
    with pytest.raises(NoFundamentalClassError):
        FC.fundamental_cycle(T.octahedron()[0], 0, backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_several_components_give_one_cycle_each_and_no_single_generator(backend):
    A, _ = T.octahedron()
    two = T.sc(list(A.n_simplices(2)) + [tuple(v + 6 for v in s) for s in A.n_simplices(2)])
    cycles = FC.fundamental_cycles(two, 2, backend=backend)
    assert len(cycles) == 2 and all(c.n_components == 2 and c.verified for c in cycles)
    with pytest.raises(NoFundamentalClassError, match="strong components"):
        FC.fundamental_cycle(two, 2, backend=backend)
    # strongly connected components, not connected components: a pinch point joins two
    # spheres into one space with H_2 = Z^2
    assert len(FC.fundamental_cycles(T.pinched_spheres(), 2, backend=backend)) == 2
    # the basis has the rank of H_2 computed by pySurgery's SNF homology
    assert len(FC.top_homology_basis(T.pinched_spheres(), 2, backend=backend)) == \
        T.pinched_spheres().homology(2, backend="python")[0]


@pytest.mark.skipif(not julia_engine.available, reason="Julia backend unavailable")
def test_python_and_julia_orientations_agree():
    for K in (T.torus(5, 4), T.klein_bottle(5, 5), T.subdivided_sphere(1)[0]):
        a = FC.coherent_orientation(K, 2, backend="python")
        b = FC.coherent_orientation(K, 2, backend="julia")
        assert a.orientable == b.orientable and a.component == b.component
        if a.is_orientable:
            assert a.signs == b.signs


def test_fundamental_class_drives_poincare_duality_cap_product():
    """Capping with the fundamental class of the torus sends the generator of C^2 dual to
    one triangle to a +-1 multiple of a vertex: the degree-0 part of Poincare duality."""
    from pysurgery.homology.poincare_duality_verification import compute_poincare_duality_map

    K = T.torus(4, 4)
    z = FC.fundamental_cycle(K, 2).as_chain(K)
    D = compute_poincare_duality_map(K, 2, z, 2)       # C^2 -> C_0
    assert D.shape == (K.count_simplices(0), K.count_simplices(2))
    assert set(np.abs(D.sum(axis=0)).tolist()) == {1}
