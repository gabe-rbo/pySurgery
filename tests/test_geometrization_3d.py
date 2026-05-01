"""Tests for 3-manifold geometrization, normal surfaces, and decomposition heuristics.

Overview:
    This suite verifies the 3-manifold geometrization pipeline, including normal 
    surface generation, coordinate validation, and prime/JSJ decomposition. It 
    ensures that the classification results are stable and compatible with the 
    broader homeomorphism witness infrastructure.

Key Concepts:
    - **Normal Surfaces**: Surfaces in a 3-manifold that intersect tetrahedra in standard discs.
    - **Geometrization Conjecture**: The classification of 3-manifolds into 8 geometric types.
    - **Prime Decomposition**: Breaking a 3-manifold into a connected sum of prime manifolds.
    - **JSJ Decomposition**: Splitting a 3-manifold along essential tori into simple pieces.
"""

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
    """Verify that normal surface generation produce expected sphere and torus candidates.

    What is Being Computed?:
        Normal surface candidates (coordinates and types) for a boundary of a 4-simplex (S³).

    Algorithm:
        1. Generate an S³ triangulation.
        2. Compute normal surface candidates and matching matrix.
        3. Validate counts of sphere and torus candidates.

    Preserved Invariants:
        - Euler characteristic of generated surfaces (sphere=2, torus=0).
    """
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
    """Verify that the geometrization pipeline produces a valid certificate for S³.

    What is Being Computed?:
        A GeometrizationResult and its conversion to a recognition certificate for S³.

    Algorithm:
        1. Perform analyze_geometrization on an S³ triangulation.
        2. Verify status, classification, and certificate readiness.
        3. Cross-validate with 3D homeomorphism analysis.

    Preserved Invariants:
        - Spherical classification for S³.
        - Homeomorphism certificate status.
    """
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
    """Verify that 3D homeomorphism witnesses can embed geometrization certificates.

    What is Being Computed?:
        A homeomorphism witness containing a GeometrizationResult payload.

    Algorithm:
        1. Run analyze_geometrization to get a result.
        2. Pass the result to build_3d_homeomorphism_witness.
        3. Verify the witness contains the original certificate payload.

    Preserved Invariants:
        - Geometric classification data preserved across witness layers.
    """
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
    """Verify that decomposition heuristics are stable on canonical S³ triangulation.

    What is Being Computed?:
        Prime and JSJ decompositions for the 4-simplex boundary.

    Algorithm:
        1. Run prime_decomposition and jsj_decomposition on the triangulation.
        2. Check for presence of decomposition pieces and metadata.

    Preserved Invariants:
        - Topology of prime components (S³ is its own prime factor).
    """
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


def test_triangulated3manifold_homology_accepts_optional_degree():
    """Verify that the homology method supports both all-degree and specific-degree queries.

    What is Being Computed?:
        Integral homology H_n(M; ℤ) for multiple dimensions.

    Algorithm:
        1. Call tri.homology() without arguments to get all groups.
        2. Call tri.homology(n) for specific degrees.
        3. Compare results for consistency.

    Preserved Invariants:
        - Betti numbers and torsion subgroups of the 3-manifold.
    """
    tri = _s3_boundary_of_4_simplex()
    h_all = tri.homology()

    assert isinstance(h_all, dict)
    assert h_all[0] == tri.homology(0)
    assert h_all[1] == tri.homology(1)
    assert h_all[2] == tri.homology(2)
    assert h_all[3] == tri.homology(3)


def test_geometrization_preserves_optional_embedding_certificate():
    """Verify that external embedding certificates are correctly passed through the pipeline.

    What is Being Computed?:
        GeometrizationResult enriched with an embedding certificate.

    Algorithm:
        1. Analyze an embedding of a surface.
        2. Pass the embedding certificate to analyze_geometrization.
        3. Verify the certificate is present in the final result.

    Preserved Invariants:
        - Embedding status of sub-complexes.
    """
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
    """Verify that normal surface candidates have non-negative and correctly shaped coordinates.

    What is Being Computed?:
        Coordinate validation and quadrilateral compatibility for normal surfaces.

    Algorithm:
        1. Generate candidates for an S³ triangulation.
        2. Assert that coordinates are non-negative and match the expected dimension.
        3. Check that the reported support tetrahedra match the non-zero coordinate locations.

    Preserved Invariants:
        - Geometric validity of normal coordinates in each tetrahedron.
    """
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
    """Verify that normal surfaces derived from dual graph cuts have consistent support metadata.

    What is Being Computed?:
        Coordinate consistency for graph-cut-based normal surfaces.

    Algorithm:
        1. Generate candidates for a two-tetrahedron chain.
        2. Filter for dual graph cut candidates.
        3. Verify that coordinates match the reported support tetrahedra.

    Preserved Invariants:
        - Locality of normal surfaces derived from simplicial sub-graphs.
    """
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
    """Verify that normal surface residual computation is delegated to Julia when available.

    What is Being Computed?:
        Execution path for normal surface residual norms.

    Algorithm:
        1. Mock julia_engine to appear available and provide a fake batch calculator.
        2. Set the residual batch threshold to 1.
        3. Trigger candidate generation and verify the mock was called.

    Preserved Invariants:
        - Numerical consistency of residual norms across backends (via mock check).
    """
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
    """Verify that empty or invalid normal surface candidates are rejected during validation.

    What is Being Computed?:
        Validation status of an empty normal surface candidate.

    Algorithm:
        1. Construct a NormalSurfaceCandidate with zero coordinates.
        2. Run _validate_normal_surface_candidate.
        3. Assert that the result is marked as invalid and not exact.

    Preserved Invariants:
        - Geometric admissibility (surfaces must have non-empty support).
    """
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



