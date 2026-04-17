from itertools import combinations
from typing import cast

import numpy as np
import pysurgery.core.geometrization_3d as g3d

from pysurgery.core.complexes import SimplicialComplex
from pysurgery.core.fundamental_group import FundamentalGroup
from pysurgery.core.embedding import analyze_embedding
from pysurgery.core.geometrization_3d import (
    GeometrizationResult,
    Triangulated3Manifold,
    analyze_geometrization,
    normal_surface_candidates,
    normal_surface_matching_matrix,
    prime_decomposition,
    jsj_decomposition,
)
from pysurgery.bridge.julia_bridge import julia_engine
from pysurgery.homeomorphism import analyze_homeomorphism_3d_result
from pysurgery.homeomorphism_witness import build_3d_homeomorphism_witness


def _s3_boundary_of_4_simplex() -> Triangulated3Manifold:
    tetrahedra = list(combinations(range(5), 4))
    return Triangulated3Manifold.from_tetrahedra(tetrahedra, name="s3_boundary")


def _two_tetra_chain() -> Triangulated3Manifold:
    return Triangulated3Manifold.from_tetrahedra(
        [(0, 1, 2, 3), (0, 1, 2, 4)],
        name="two_tetra_chain",
    )


def test_geometrization_builds_canonical_normal_surfaces():
    tri = _s3_boundary_of_4_simplex()
    candidates = normal_surface_candidates(tri)
    matrix = normal_surface_matching_matrix(tri)

    sphere_count = sum(1 for cand in candidates if cand.kind == "sphere")
    torus_count = sum(1 for cand in candidates if cand.kind == "torus")

    assert tri.is_closed
    assert tri.n_tetrahedra == 5
    assert sphere_count >= 5
    assert torus_count >= 10
    assert matrix.shape[1] == 7 * tri.n_tetrahedra
    assert matrix.shape[0] > 0
    assert any(cand.decision_ready() for cand in candidates if cand.kind == "sphere")


def test_geometrization_result_produces_decision_ready_certificate_for_s3_branch():
    tri = _s3_boundary_of_4_simplex()
    result = analyze_geometrization(tri, pi1_descriptor="1")

    assert isinstance(result, GeometrizationResult)
    assert result.status == "success"
    assert result.exact is True
    assert result.validated is True
    assert result.classification == "spherical"
    assert result.decision_ready()
    assert result.to_recognition_certificate().decision_ready()
    assert result.prime_decomposition is not None
    assert result.jsj_decomposition is not None

    c = tri.chain_complex()
    pi = FundamentalGroup(generators=["a", "b"], relations=[])
    homeo = analyze_homeomorphism_3d_result(
        c,
        c,
        pi1_1=pi,
        pi1_2=pi,
        recognition_certificate=result,  # type: ignore[arg-type]
    )
    assert homeo.status == "success"
    assert homeo.is_homeomorphic is True
    assert homeo.theorem == "Geometrization / 3-manifold recognition"
    assert "three_manifold_recognition_certificate" in homeo.certificates


def test_geometrization_witness_accepts_result_object_and_preserves_certificate_payload():
    tri = _s3_boundary_of_4_simplex()
    result = analyze_geometrization(tri, pi1_descriptor="1")
    c = tri.chain_complex()
    pi = FundamentalGroup(generators=["a", "b"], relations=[])

    witness = build_3d_homeomorphism_witness(
        c,
        c,
        pi1_1=pi,
        pi1_2=pi,
        recognition_certificate=result,  # type: ignore[arg-type]
    )

    assert witness.status == "success"
    assert witness.witness is not None
    assert witness.witness.certificates.get("recognition_certificate") is not None
    payload = cast(
        GeometrizationResult, witness.witness.certificates["recognition_certificate"]
    ).to_recognition_certificate()
    assert payload.decision_ready()


def test_prime_and_jsj_decompositions_are_stable_on_the_s3_example():
    tri = _s3_boundary_of_4_simplex()
    prime = prime_decomposition(tri)
    jsj = jsj_decomposition(tri)

    assert prime.kind == "prime"
    assert jsj.kind == "jsj"
    assert len(prime.pieces) >= 1
    assert len(jsj.pieces) >= 1
    assert prime.summary
    assert jsj.summary
    assert any("selection_score=" in note for note in prime.notes)
    assert any("selection_score=" in note for note in jsj.notes)


def test_geometrization_preserves_optional_embedding_certificate():
    tri = _s3_boundary_of_4_simplex()
    surface = SimplicialComplex.from_maximal_simplices(
        [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    )
    coords = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    )
    embedding = analyze_embedding(surface, coords)
    result = analyze_geometrization(
        tri, pi1_descriptor="1", embedding_certificate=embedding
    )

    assert "embedding_certificate" in result.certificates
    assert result.certificates["embedding_certificate"]["embedded"] is True


def test_normal_surface_candidates_are_strictly_coordinate_validated():
    tri = _s3_boundary_of_4_simplex()
    candidates = normal_surface_candidates(tri)

    assert candidates
    for candidate in candidates:
        assert candidate.coordinates.shape == (7 * tri.n_tetrahedra,)
        assert np.all(candidate.coordinates >= 0)
        assert candidate.quadrilateral_ok is True
        support = tuple(
            np.where(np.any(candidate.coordinates.reshape(tri.n_tetrahedra, 7) > 0, axis=1))[0].tolist()
        )
        assert candidate.support_tetrahedra == support


def test_graph_cut_candidate_coordinates_match_support():
    tri = _two_tetra_chain()
    candidates = normal_surface_candidates(tri)
    graph_cut = [c for c in candidates if c.source == "dual_graph_cut"]

    assert graph_cut
    for candidate in graph_cut:
        support = tuple(
            np.where(np.any(candidate.coordinates.reshape(tri.n_tetrahedra, 7) > 0, axis=1))[0].tolist()
        )
        assert support == candidate.support_tetrahedra


def test_geometrization_uses_julia_batch_residuals_when_available(monkeypatch):
    tri = _s3_boundary_of_4_simplex()
    calls = {"count": 0}

    monkeypatch.setattr(g3d, "_JULIA_RESIDUAL_BATCH_THRESHOLD", 1, raising=False)
    monkeypatch.setattr(julia_engine, "available", True, raising=False)

    def _fake_batch(matrix, coordinate_matrix):
        calls["count"] += 1
        return np.linalg.norm((matrix @ coordinate_matrix).astype(np.float64), axis=0)

    monkeypatch.setattr(
        julia_engine,
        "compute_normal_surface_residual_norms",
        _fake_batch,
        raising=False,
    )

    candidates = normal_surface_candidates(tri)
    assert candidates
    assert calls["count"] >= 1


def test_invalid_normal_surface_candidate_is_not_exact_after_validation():
    tri = _s3_boundary_of_4_simplex()
    bad = g3d.NormalSurfaceCandidate(
        kind="sphere",
        surface_type="vertex_link",
        exact=True,
        validated=True,
        coordinates=np.zeros(7 * tri.n_tetrahedra, dtype=np.int64),
        triangle_coordinates=np.zeros((tri.n_tetrahedra, 4), dtype=np.int64),
        quad_coordinates=np.zeros((tri.n_tetrahedra, 3), dtype=np.int64),
        support_tetrahedra=(),
        euler_characteristic=1,
    )
    matrix = normal_surface_matching_matrix(tri)
    validated = g3d._validate_normal_surface_candidate(tri, bad, matching_matrix=matrix)
    assert validated.validated is False
    assert validated.exact is False
    assert any("empty normal support" in note for note in validated.notes)



