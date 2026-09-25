"""The complement of a link: its fundamental group, and the certificates built on it.

Overview:
    Known groups: the unknot's is Z, the trefoil's has 12 homomorphisms to S_3 (Z has
    6), the Hopf link's is Z^2 (18 commuting pairs in S_3; the free group F_2 has 36).
    Every presentation must pass its Alexander-duality check. The homomorphism counter
    is also exercised on the fundamental groups pySurgery extracts from complexes.
"""
import pytest

import exact_triangulations as T
import synthetic_curves as S
from pysurgery.bridge.julia_bridge import julia_engine
from pysurgery.knots import link_complement as LC
from pysurgery.topology.fundamental_group import FundamentalGroup

BACKENDS = ["python"] + (["julia"] if julia_engine.available else [])


def test_the_unknot_group_is_z():
    G = LC.link_group([S.circle(40)])
    assert len(G.generators) == 1 and not G.relations
    assert LC.alexander_check([S.circle(40)], G)["ok"]


def test_alexander_duality_holds_for_every_presentation():
    for curves in ([S.trefoil(200)], [S.figure_eight(200)], S.hopf_link(60),
                   S.borromean_rings(120), S.torus_link(2, 200)):
        G = LC.link_group(curves)
        assert LC.alexander_check(curves, G)["ok"]
        raw = LC.wirtinger_presentation(curves)
        assert LC.alexander_check(curves, raw)["ok"]      # before Tietze, too


@pytest.mark.parametrize("backend", BACKENDS)
def test_homomorphism_counts_against_hand_computed_values(backend):
    assert LC.count_homomorphisms(LC.link_group([S.trefoil(200)]), 3, backend=backend).count == 12
    assert LC.count_homomorphisms(LC.link_group([S.circle(40)]), 3, backend=backend).count == 6
    assert LC.count_homomorphisms(LC.link_group(S.hopf_link(60)), 3, backend=backend).count == 18
    free2 = FundamentalGroup(generators=["a", "b"], relations=[])
    assert LC.count_homomorphisms(free2, 3, backend=backend).count == LC.free_group_hom_count(3, 2) == 36
    z2 = FundamentalGroup(generators=["a", "b"], relations=[["a", "b", "a^-1", "b^-1"]])
    assert LC.count_homomorphisms(z2, 3, backend=backend).count == 18
    assert LC.count_homomorphisms(z2, 4, backend=backend).count == 24 * 5   # |S_4| x #classes
    capped = LC.count_homomorphisms(LC.link_group([S.trefoil(200)]), 4, budget=5, backend=backend)
    assert not capped.exact


@pytest.mark.skipif(not julia_engine.available, reason="Julia backend unavailable")
def test_python_and_julia_counts_are_identical():
    for G in (LC.wirtinger_presentation([S.trefoil(120)]), LC.link_group(S.borromean_rings(120)),
              FundamentalGroup(generators=["a", "b"], relations=[["a", "a", "b", "b", "b"]])):
        for n in (3, 4):
            a = LC.count_homomorphisms(G, n, backend="python")
            b = LC.count_homomorphisms(G, n, backend="julia")
            assert (a.count, a.exact, a.tried) == (b.count, b.exact, b.tried)
    G = LC.link_group([S.trefoil(200)])
    a = LC.count_homomorphisms(G, 4, budget=7, backend="python")
    b = LC.count_homomorphisms(G, 4, budget=7, backend="julia")
    assert (a.count, a.exact, a.tried) == (b.count, b.exact, b.tried)


def test_hom_counts_on_fundamental_groups_of_complexes():
    """|Hom(pi_1, S_2)| of RP^2 is 2 (pi_1 = Z/2) and of the torus is 4 (Z^2 into Z/2)."""
    rp2 = T.projective_plane().fundamental_group(backend="python")
    assert rp2.count_homomorphisms(2).count == 2
    assert rp2.count_homomorphisms(3).count == 4          # the 3 transpositions and 1
    torus = T.torus(3, 3).fundamental_group(backend="python")
    assert torus.count_homomorphisms(2).count == 4
    sphere = T.octahedron()[0].fundamental_group(backend="python")
    assert sphere.count_homomorphisms(3).count == 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_splitness_certificates(backend):
    assert LC.certify_splitness(S.unlink(2, 40), backend=backend).verdict == "split"
    hopf = LC.certify_splitness(S.hopf_link(60), backend=backend)
    assert hopf.verdict == "non-split" and hopf.detail["whole"] == 18
    borr = LC.certify_splitness(S.borromean_rings(120), backend=backend)
    assert borr.verdict == "non-split"                   # pairwise unlinked, not split


@pytest.mark.parametrize("backend", BACKENDS)
def test_knottedness_certificates(backend):
    assert LC.certify_knottedness(S.circle(40), backend=backend).verdict == "unknotted"
    assert LC.certify_knottedness(S.trefoil(200), backend=backend).verdict == "knotted"
    # the figure-eight is not 3-colourable: whatever S_4 says, "unknotted" would be false
    assert LC.certify_knottedness(S.figure_eight(200), backend=backend).verdict != "unknotted"
