"""Tests for SimplicialComplex.is_homology_manifold and certify_pl_manifold.

Overview:
    ``is_homology_manifold`` checks the link of every simplex, so at dimension <= 3 its
    verdict is already a PL certificate and ``certify_pl_manifold`` coincides with it
    (``exact=True``). Pinned here: a genuine S^2 and S^3; a pinch point; the
    8-vertex/7-tetrahedron complex whose vertex link is S^2 wedged with a disk at a
    point (sphere homology, not a homology manifold -- a vertex-link-only check reported
    it as a manifold); impure complexes whose lower-dimensional pieces have acyclic vertex
    links; and dimension >= 4, where a positive verdict is not a PL proof (``exact=False``
    with a warning) but a negative one is. Every case runs on both backends.
"""
import itertools

import pytest

from pysurgery.topology.complexes import SimplicialComplex
from pysurgery.topology.local_homology import certify_homology_manifold


def _julia_available():
    try:
        from pysurgery.bridge.julia_bridge import julia_engine
        return julia_engine.available
    except Exception:
        return False


BACKENDS = [
    "python",
    pytest.param("julia", marks=pytest.mark.skipif(not _julia_available(),
                                                   reason="Julia backend unavailable")),
]


def _complex(tops):
    return SimplicialComplex.from_simplices(tops, close_under_faces=True)


@pytest.mark.parametrize("backend", BACKENDS)
def test_certify_pl_manifold_matches_is_homology_manifold_at_d_le_2(backend):
    """At dimension <= 2 both methods agree, with exact=True, on S^2 and on a pinch point."""
    sphere = _complex(list(itertools.combinations(range(4), 3)))
    is_mani, dim, diag = sphere.is_homology_manifold(backend=backend)
    cert = sphere.certify_pl_manifold(backend=backend)
    assert cert.is_pl_manifold == is_mani == True
    assert cert.dimension == dim == 2
    assert cert.diagnostics == diag == {}
    assert cert.exact is True

    # Two triangles sharing only vertex 0: the link of 0 is two disjoint edges.
    pinched = _complex([(0, 1, 2), (0, 3, 4)])
    is_mani2, dim2, diag2 = pinched.is_homology_manifold(backend=backend)
    cert2 = pinched.certify_pl_manifold(backend=backend)
    assert cert2.is_pl_manifold == is_mani2 == False
    assert cert2.dimension == dim2 == 2
    assert cert2.diagnostics == diag2
    assert set(diag2) == {0}
    assert cert2.exact is True


@pytest.mark.parametrize("backend", BACKENDS)
def test_certify_pl_manifold_boundary_of_4_simplex_is_exact_s3(backend):
    """The boundary of a 4-simplex is S^3: certified exactly."""
    sc = _complex(list(itertools.combinations(range(5), 4)))
    assert sc.dimension == 3

    cert = sc.certify_pl_manifold(backend=backend)
    assert cert.is_pl_manifold is True
    assert cert.dimension == 3
    assert cert.diagnostics == {}
    assert cert.exact is True


def _wedge_counterexample():
    """Cone a vertex 0 over L = (S^2 on {1,2,3,4}) wedged at vertex 1 with a closed fan on {5,6,7}.

    L has the homology of S^2 (the fan is a disk), so a check of vertex links alone accepts
    vertex 0. But inside L the link of vertex 1 is two disjoint triangles, i.e. in the
    complex the edge (0, 1) has a disconnected link: not a homology manifold.
    """
    return _complex([
        (0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 3, 4), (0, 2, 3, 4),  # v * (S^2 piece)
        (0, 1, 5, 6), (0, 1, 6, 7), (0, 1, 7, 5),                # v * (fan piece)
    ])


@pytest.mark.parametrize("backend", BACKENDS)
def test_is_homology_manifold_catches_wedge_counterexample(backend):
    """The singular edge (0, 1) is found directly, and both methods agree."""
    sc = _wedge_counterexample()
    assert len(sc.n_simplices(0)) == 8
    assert len(sc.n_simplices(3)) == 7

    is_mani, dim, diag = sc.is_homology_manifold(backend=backend)
    assert is_mani is False
    assert dim == 3
    assert set(diag) == {0, 1}
    assert "1-simplex 0-1" in diag[0]
    assert not certify_homology_manifold(sc, backend="python").is_homology_manifold_with_boundary

    cert = sc.certify_pl_manifold(backend=backend)
    assert cert.exact is True
    assert cert.is_pl_manifold is False
    assert cert.diagnostics == diag


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("tops", [
    [(0, 1, 2), (3, 4)],                   # a disk and a disjoint edge
    [(0, 1, 2), (3,)],                     # a disk and an isolated point
    [(0, 1, 2, 3), (4, 5, 6)],             # a ball and a disjoint triangle
])
def test_is_homology_manifold_rejects_impure_complexes(backend, tops):
    """Lower-dimensional pieces with acyclic vertex links are singular (empty links)."""
    sc = _complex(tops)
    is_mani, dim, diag = sc.is_homology_manifold(backend=backend)
    assert is_mani is False
    assert dim == max(len(t) for t in tops) - 1
    assert diag
    assert not certify_homology_manifold(sc, backend="python").is_homology_manifold_with_boundary


@pytest.mark.parametrize("backend", BACKENDS)
def test_certify_pl_manifold_d_ge_4_returns_inexact_with_warning(backend):
    """S^4 (boundary of a 5-simplex): a positive verdict at d >= 4 is not a PL proof."""
    sc = _complex(list(itertools.combinations(range(6), 5)))
    assert sc.dimension == 4

    with pytest.warns(UserWarning, match="cannot exactly certify"):
        cert = sc.certify_pl_manifold(backend=backend)
    assert cert.exact is False
    assert cert.dimension == 4
    assert cert.is_pl_manifold is True  # a homology manifold; PL-ness not proven


@pytest.mark.parametrize("backend", BACKENDS)
def test_certify_pl_manifold_d_ge_4_negative_is_exact(backend):
    """A non-homology-manifold is certainly not a PL manifold, in any dimension."""
    sphere = list(itertools.combinations(range(6), 5))
    sc = _complex(sphere + [(0, 1, 2, 3, 6)])     # a 4-simplex glued along a 3-face: branching
    cert = sc.certify_pl_manifold(backend=backend)
    assert cert.is_pl_manifold is False
    assert cert.exact is True
