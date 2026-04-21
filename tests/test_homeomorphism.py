import numpy as np
import pytest
import scipy.sparse as sp

from discrete_surface_data import (
    build_tetrahedron,
    build_octahedron,
    build_icosahedron,
    build_torus,
    to_complex,
)
from pysurgery.homeomorphism import (
    analyze_homeomorphism_2d,
    HomotopyCompletionCertificate,
    ProductAssemblyCertificate,
    ThreeManifoldRecognitionCertificate,
    analyze_homeomorphism_2d_result,
    analyze_homeomorphism_3d_result,
    analyze_homeomorphism_4d,
    analyze_homeomorphism_4d_result,
    analyze_homeomorphism_high_dim,
    analyze_homeomorphism_high_dim_result,
    surgery_to_remove_impediments,
)
from pysurgery.core.intersection_forms import IntersectionForm
from pysurgery.core.complexes import ChainComplex
from pysurgery.core.fundamental_group import FundamentalGroup
from pysurgery.core.k_theory import WhiteheadGroup
from pysurgery.structure_set import NormalInvariantsResult, SurgeryExactSequenceResult
from pysurgery.wall_groups import ObstructionResult


def test_analyze_homeomorphism_4d_indefinite():
    matrix1 = np.array([[0, 1], [1, 0]])
    form1 = IntersectionForm(matrix=matrix1, dimension=4)

    matrix2 = np.array([[0, 1], [1, 0]])
    form2 = IntersectionForm(matrix=matrix2, dimension=4)

    is_homeo, reason = analyze_homeomorphism_4d(
        form1, form2, ks1=0, ks2=0, simply_connected=True
    )
    assert is_homeo
    assert "SUCCESS" in reason


def test_analyze_homeomorphism_4d_impediment():
    matrix1 = np.array([[0, 1], [1, 0]])
    form1 = IntersectionForm(matrix=matrix1, dimension=4)

    matrix2 = np.array([[1, 0], [0, -1]])
    form2 = IntersectionForm(matrix=matrix2, dimension=4)

    is_homeo, reason = analyze_homeomorphism_4d(
        form1, form2, ks1=0, ks2=0, simply_connected=True
    )
    assert not is_homeo
    assert "Parity mismatch" in reason


def test_analyze_homeomorphism_4d_definite_exact_match_is_homeomorphic():
    form1 = IntersectionForm(matrix=np.array([[1, 0], [0, 1]]), dimension=4)
    form2 = IntersectionForm(matrix=np.array([[1, 0], [0, 1]]), dimension=4)
    is_homeo, reason = analyze_homeomorphism_4d(
        form1, form2, ks1=0, ks2=0, simply_connected=True
    )
    assert is_homeo is True
    assert "SUCCESS" in reason


def test_analyze_homeomorphism_4d_definite_search_finds_larger_isometry_witness():
    form1 = IntersectionForm(matrix=np.array([[1, 0], [0, 1]]), dimension=4)
    u = np.array([[3, 2], [2, 1]])
    form2 = IntersectionForm(matrix=u.T @ np.array([[1, 0], [0, 1]]) @ u, dimension=4)
    res = analyze_homeomorphism_4d_result(
        form1, form2, ks1=0, ks2=0, simply_connected=True
    )
    assert res.status == "success"
    assert res.is_homeomorphic is True
    assert "lattice isomorphism certificate" in res.reasoning


def test_analyze_homeomorphism_4d_definite_search_handles_unimodular_isometry_with_large_entries():
    form1 = IntersectionForm(matrix=np.array([[1, 0], [0, 1]]), dimension=4)
    u = np.array([[5, 2], [2, 1]], dtype=np.int64)
    form2 = IntersectionForm(
        matrix=u.T @ np.array([[1, 0], [0, 1]], dtype=np.int64) @ u, dimension=4
    )
    res = analyze_homeomorphism_4d_result(
        form1, form2, ks1=0, ks2=0, simply_connected=True
    )
    assert res.status == "success"
    assert res.is_homeomorphic is True
    assert "lattice isomorphism certificate" in res.reasoning


def test_analyze_homeomorphism_4d_accepts_decision_ready_definite_certificate_when_search_misses():
    form1 = IntersectionForm(matrix=np.array([[1, 0], [0, 1]]), dimension=4)
    u = np.array([[8, 3], [3, 1]], dtype=np.int64)
    form2 = IntersectionForm(
        matrix=u.T @ np.array([[1, 0], [0, 1]], dtype=np.int64) @ u, dimension=4
    )
    cert = {
        "provided": True,
        "source": "test",
        "exact": True,
        "validated": True,
        "isometry_matrix": u.tolist(),
    }
    res = analyze_homeomorphism_4d_result(
        form1,
        form2,
        ks1=0,
        ks2=0,
        simply_connected=True,
        definite_lattice_isometry_certificate=cert,
    )
    assert res.status == "success"
    assert res.is_homeomorphic is True
    assert res.certificates.get("isometry_search_mode") in {
        "bounded_search",
        "external_certificate",
    }


def test_analyze_homeomorphism_4d_rejects_invalid_decision_ready_certificate():
    form1 = IntersectionForm(matrix=np.array([[1, 0], [0, 1]]), dimension=4)
    u = np.array([[5, 2], [2, 1]], dtype=np.int64)
    form2 = IntersectionForm(
        matrix=u.T @ np.array([[1, 0], [0, 1]], dtype=np.int64) @ u, dimension=4
    )
    bad = {
        "provided": True,
        "source": "test",
        "exact": True,
        "validated": True,
        "isometry_matrix": [[1, 0], [0, 2]],
    }
    res = analyze_homeomorphism_4d_result(
        form1,
        form2,
        ks1=0,
        ks2=0,
        simply_connected=True,
        definite_lattice_isometry_certificate=bad,
    )
    assert res.status in {"inconclusive", "success"}
    if res.status == "inconclusive":
        assert "does not verify" in res.reasoning


def test_analyze_homeomorphism_2d_homology_failure_fallback():
    class BrokenComplex:
        def homology(self, n):
            raise RuntimeError("boom")

    from pysurgery.homeomorphism import analyze_homeomorphism_2d

    is_homeo, reason = analyze_homeomorphism_2d(BrokenComplex(), BrokenComplex())
    assert is_homeo is None
    assert "INCONCLUSIVE" in reason

    with pytest.warns(UserWarning) as rec:
        is_homeo2, reason2 = analyze_homeomorphism_2d(
            BrokenComplex(), BrokenComplex(), allow_approx=True
        )
    assert is_homeo2 is None
    assert "INCONCLUSIVE" in reason2
    warning_text = "\n".join(str(w.message) for w in rec)
    assert "boom" in warning_text
    assert "{e}" not in warning_text


def test_surgery_to_remove_impediments():
    matrix1 = np.array([[1, 0], [0, 1]])  # sig = 2
    form1 = IntersectionForm(matrix=matrix1, dimension=4)

    can_remove, reason = surgery_to_remove_impediments(form1, 10)
    assert can_remove

    can_remove, reason = surgery_to_remove_impediments(form1, 4)
    assert can_remove


def test_analyze_homeomorphism_4d_requires_explicit_hypotheses():
    form1 = IntersectionForm(matrix=np.array([[0, 1], [1, 0]]), dimension=4)
    form2 = IntersectionForm(matrix=np.array([[0, 1], [1, 0]]), dimension=4)
    res = analyze_homeomorphism_4d_result(form1, form2)
    assert res.is_homeomorphic is None
    assert res.status == "inconclusive"
    assert "Simply-connectedness" in res.reasoning


def test_high_dim_reports_surgery_required_for_nonzero_wall_obstruction():
    d1 = sp.csr_matrix(np.zeros((1, 0), dtype=np.int64))
    d5 = sp.csr_matrix(np.zeros((0, 1), dtype=np.int64))
    c = ChainComplex(
        boundaries={1: d1, 5: d5},
        dimensions=[0, 1, 2, 3, 4, 5],
        cells={0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1},
    )

    wh = WhiteheadGroup(rank=0, description="Wh(1)=0", computable=True, exact=True)
    wall = ObstructionResult(
        dimension=5,
        pi="1",
        computable=True,
        exact=True,
        value=1,
        modulus=None,
        message="",
        assumptions=[],
    )
    res = analyze_homeomorphism_high_dim_result(
        c,
        c,
        dim=5,
        pi1=FundamentalGroup(generators=[], relations=[]),
        whitehead_group=wh,
        wall_obstruction=wall,
    )
    assert res.status == "surgery_required"
    assert res.is_homeomorphic is False


def test_high_dim_direct_sum_wall_obstruction_without_zero_certificate_is_inconclusive():
    d1 = sp.csr_matrix(np.zeros((1, 0), dtype=np.int64))
    d5 = sp.csr_matrix(np.zeros((0, 1), dtype=np.int64))
    c = ChainComplex(
        boundaries={1: d1, 5: d5},
        dimensions=[0, 1, 2, 3, 4, 5],
        cells={0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1},
    )

    wh = WhiteheadGroup(rank=0, description="Wh(1)=0", computable=True, exact=True)
    wall = ObstructionResult(
        dimension=5,
        pi="Z x Z",
        computable=True,
        exact=True,
        value=None,
        modulus=None,
        message="Shaneson direct-sum element computable",
        assumptions=[],
        obstructs=None,
        zero_certified=False,
    )
    res = analyze_homeomorphism_high_dim_result(
        c,
        c,
        dim=5,
        pi_group="1",
        whitehead_group=wh,
        wall_obstruction=wall,
    )
    assert res.status == "inconclusive"
    assert res.is_homeomorphic is None
    assert "vanishing is not certified" in res.reasoning


def test_high_dim_direct_sum_wall_obstruction_with_certified_nonzero_is_surgery_required():
    d1 = sp.csr_matrix(np.zeros((1, 0), dtype=np.int64))
    d5 = sp.csr_matrix(np.zeros((0, 1), dtype=np.int64))
    c = ChainComplex(
        boundaries={1: d1, 5: d5},
        dimensions=[0, 1, 2, 3, 4, 5],
        cells={0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1},
    )

    wh = WhiteheadGroup(rank=0, description="Wh(1)=0", computable=True, exact=True)
    wall = ObstructionResult(
        dimension=5,
        pi="Z x Z",
        computable=True,
        exact=True,
        value=None,
        modulus=None,
        message="Shaneson direct-sum includes non-zero summand",
        assumptions=[],
        obstructs=True,
        zero_certified=False,
    )
    res = analyze_homeomorphism_high_dim_result(
        c,
        c,
        dim=5,
        pi_group="1",
        whitehead_group=wh,
        wall_obstruction=wall,
    )
    assert res.status == "surgery_required"
    assert res.is_homeomorphic is False


def test_high_dim_heuristic_whitehead_data_is_not_used_as_a_certified_success():
    d1 = sp.csr_matrix(np.zeros((1, 0), dtype=np.int64))
    d5 = sp.csr_matrix(np.zeros((0, 1), dtype=np.int64))
    c = ChainComplex(
        boundaries={1: d1, 5: d5},
        dimensions=[0, 1, 2, 3, 4, 5],
        cells={0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1},
    )

    wh = WhiteheadGroup(
        rank=0, description="heuristic Wh(pi_1)=0", computable=True, exact=False
    )
    res = analyze_homeomorphism_high_dim_result(
        c,
        c,
        dim=5,
        pi1=FundamentalGroup(generators=[], relations=[]),
        whitehead_group=wh,
    )
    assert res.status == "inconclusive"
    assert (
        "Exact Whitehead torsion certificate" in res.reasoning
        or "heuristic" in res.reasoning
    )


def test_high_dim_accepts_trivialized_pi1_presentation_and_records_certificates():
    d1 = sp.csr_matrix(np.zeros((1, 0), dtype=np.int64))
    d5 = sp.csr_matrix(np.zeros((0, 1), dtype=np.int64))
    c = ChainComplex(
        boundaries={1: d1, 5: d5},
        dimensions=[0, 1, 2, 3, 4, 5],
        cells={0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1},
    )

    pi1 = FundamentalGroup(generators=["a"], relations=[["a"]])
    wh = WhiteheadGroup(rank=0, description="Wh(1)=0", computable=True, exact=True)
    wall = ObstructionResult(
        dimension=5,
        pi="1",
        computable=True,
        exact=True,
        value=0,
        modulus=None,
        message="",
        assumptions=[],
    )
    normal = NormalInvariantsResult(dimension=5, rank_Z=0, rank_Z2=0)
    seq = SurgeryExactSequenceResult(
        dimension=5,
        fundamental_group="1",
        l_n_symbol="0",
        l_n_plus_1_symbol="0",
        computable=True,
        exact=True,
        analysis=["Exact surgery certificate is trivial."],
        normal_invariants=normal,
    )

    res = analyze_homeomorphism_high_dim_result(
        c,
        c,
        dim=5,
        pi1=pi1,
        whitehead_group=wh,
        wall_obstruction=wall,
        normal_invariants_1=normal,
        normal_invariants_2=normal,
        surgery_sequence=seq,
    )
    assert res.status == "success"
    assert res.is_homeomorphic is True
    assert res.certificates["normal_invariants_1"] == normal
    assert res.certificates["surgery_sequence"] == seq


def test_high_dim_auto_sequence_contains_typed_l_n_plus_1_state():
    d1 = sp.csr_matrix(np.zeros((1, 0), dtype=np.int64))
    d5 = sp.csr_matrix(np.zeros((0, 1), dtype=np.int64))
    c = ChainComplex(
        boundaries={1: d1, 5: d5},
        dimensions=[0, 1, 2, 3, 4, 5],
        cells={0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1},
    )
    wh = WhiteheadGroup(rank=0, description="Wh(1)=0", computable=True, exact=True)
    wall = ObstructionResult(
        dimension=5,
        pi="1",
        computable=True,
        exact=True,
        value=0,
        modulus=None,
        assumptions=[],
    )
    res = analyze_homeomorphism_high_dim_result(
        c,
        c,
        dim=5,
        pi1=FundamentalGroup(generators=[], relations=[]),
        whitehead_group=wh,
        wall_obstruction=wall,
    )
    assert res.status == "success"
    seq = res.certificates.get("surgery_sequence")
    assert seq is not None
    assert hasattr(seq, "l_n_state")
    assert hasattr(seq, "l_n_plus_1_state")


def test_high_dim_phase5_decision_dag_is_attached_to_results():
    d1 = sp.csr_matrix(np.zeros((1, 0), dtype=np.int64))
    d5 = sp.csr_matrix(np.zeros((0, 1), dtype=np.int64))
    c = ChainComplex(
        boundaries={1: d1, 5: d5},
        dimensions=[0, 1, 2, 3, 4, 5],
        cells={0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1},
    )
    wh = WhiteheadGroup(rank=0, description="Wh(1)=0", computable=True, exact=True)
    wall = ObstructionResult(
        dimension=5,
        pi="1",
        computable=True,
        exact=True,
        value=0,
        modulus=None,
        assumptions=[],
    )
    res = analyze_homeomorphism_high_dim_result(
        c,
        c,
        dim=5,
        pi1=FundamentalGroup(generators=[], relations=[]),
        whitehead_group=wh,
        wall_obstruction=wall,
    )
    dag = res.certificates.get("decision_dag")
    assert isinstance(dag, dict)
    assert dag.get("dimension") == 5
    assert any(stage.get("id") == "hook_intake" for stage in dag.get("stages", []))


def test_high_dim_phase5_homotopy_hook_passthrough_is_recorded():
    d1 = sp.csr_matrix(np.zeros((1, 0), dtype=np.int64))
    d5 = sp.csr_matrix(np.zeros((0, 1), dtype=np.int64))
    c = ChainComplex(
        boundaries={1: d1, 5: d5},
        dimensions=[0, 1, 2, 3, 4, 5],
        cells={0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1},
    )
    res = analyze_homeomorphism_high_dim_result(
        c,
        c,
        dim=5,
        homotopy_equivalence_witness={"map": "f", "degree": 1},
        homotopy_witness_hook={
            "provided": True,
            "source": "test",
            "exact": False,
            "summary": "manual hook",
        },
    )
    hook = res.certificates.get("homotopy_witness_hook")
    assert isinstance(hook, dict)
    assert hook.get("provided") is True
    assert hook.get("source") == "test"


def test_high_dim_exact_homotopy_hook_completes_nontrivial_pi_classification():
    d1 = sp.csr_matrix(np.zeros((1, 0), dtype=np.int64))
    d5 = sp.csr_matrix(np.zeros((0, 1), dtype=np.int64))
    c = ChainComplex(
        boundaries={1: d1, 5: d5},
        dimensions=[0, 1, 2, 3, 4, 5],
        cells={0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1},
    )
    wh = WhiteheadGroup(rank=0, description="Wh(Z/3)=0", computable=True, exact=True)
    wall = ObstructionResult(
        dimension=5,
        pi="Z/3",
        computable=True,
        exact=True,
        value=0,
        modulus=2,
        assumptions=[],
    )

    res = analyze_homeomorphism_high_dim_result(
        c,
        c,
        dim=5,
        pi_group="Z/3",
        whitehead_group=wh,
        wall_obstruction=wall,
        homotopy_witness_hook={
            "provided": True,
            "source": "test",
            "exact": True,
            "summary": "exact hook",
        },
    )
    assert res.status == "success"
    assert res.is_homeomorphic is True
    dag = res.certificates.get("decision_dag")
    assert any(
        stage.get("id") == "homotopy_completion" and stage.get("outcome") == "passed"
        for stage in dag.get("stages", [])
    )
    assert any(
        stage.get("id") == "final_classification" and stage.get("outcome") == "passed"
        for stage in dag.get("stages", [])
    )


def test_high_dim_nonexact_homotopy_hook_remains_inconclusive():
    d1 = sp.csr_matrix(np.zeros((1, 0), dtype=np.int64))
    d5 = sp.csr_matrix(np.zeros((0, 1), dtype=np.int64))
    c = ChainComplex(
        boundaries={1: d1, 5: d5},
        dimensions=[0, 1, 2, 3, 4, 5],
        cells={0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1},
    )
    wh = WhiteheadGroup(rank=0, description="Wh(Z/3)=0", computable=True, exact=True)
    wall = ObstructionResult(
        dimension=5,
        pi="Z/3",
        computable=True,
        exact=True,
        value=0,
        modulus=2,
        assumptions=[],
    )

    res = analyze_homeomorphism_high_dim_result(
        c,
        c,
        dim=5,
        pi_group="Z/3",
        whitehead_group=wh,
        wall_obstruction=wall,
        homotopy_witness_hook={
            "provided": True,
            "source": "test",
            "exact": False,
            "summary": "heuristic hook",
        },
    )
    assert res.status == "inconclusive"
    assert res.is_homeomorphic is None
    assert "decision-ready" in res.reasoning
    dag = res.certificates.get("decision_dag")
    assert any(
        stage.get("id") == "final_classification"
        and stage.get("outcome") == "inconclusive"
        for stage in dag.get("stages", [])
    )


def test_high_dim_legacy_wrapper_forwards_homotopy_hook_inputs():
    d1 = sp.csr_matrix(np.zeros((1, 0), dtype=np.int64))
    d5 = sp.csr_matrix(np.zeros((0, 1), dtype=np.int64))
    c = ChainComplex(
        boundaries={1: d1, 5: d5},
        dimensions=[0, 1, 2, 3, 4, 5],
        cells={0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1},
    )
    wh = WhiteheadGroup(rank=0, description="Wh(Z/3)=0", computable=True, exact=True)
    wall = ObstructionResult(
        dimension=5,
        pi="Z/3",
        computable=True,
        exact=True,
        value=0,
        modulus=2,
        assumptions=[],
    )

    is_homeo, reason = analyze_homeomorphism_high_dim(
        c,
        c,
        dim=5,
        pi_group="Z/3",
        whitehead_group=wh,
        wall_obstruction=wall,
        homotopy_witness_hook={
            "provided": True,
            "source": "legacy",
            "exact": True,
            "summary": "exact legacy hook",
        },
    )
    assert is_homeo is True
    assert "SUCCESS" in reason


def test_high_dim_typed_completion_certificate_requires_validation():
    d1 = sp.csr_matrix(np.zeros((1, 0), dtype=np.int64))
    d5 = sp.csr_matrix(np.zeros((0, 1), dtype=np.int64))
    c = ChainComplex(
        boundaries={1: d1, 5: d5},
        dimensions=[0, 1, 2, 3, 4, 5],
        cells={0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1},
    )
    wh = WhiteheadGroup(rank=0, description="Wh(Z/3)=0", computable=True, exact=True)
    wall = ObstructionResult(
        dimension=5,
        pi="Z/3",
        computable=True,
        exact=True,
        value=0,
        modulus=2,
        assumptions=[],
    )
    cert = HomotopyCompletionCertificate(
        provided=True,
        source="typed",
        exact=True,
        validated=False,
        equivalence_type="homotopy_equivalence",
        summary="exact but not validated",
    )

    res = analyze_homeomorphism_high_dim_result(
        c,
        c,
        dim=5,
        pi_group="Z/3",
        whitehead_group=wh,
        wall_obstruction=wall,
        homotopy_completion_certificate=cert,
    )
    assert res.status == "inconclusive"
    assert "decision-ready" in res.reasoning


def test_high_dim_legacy_wrapper_forwards_typed_completion_certificate():
    d1 = sp.csr_matrix(np.zeros((1, 0), dtype=np.int64))
    d5 = sp.csr_matrix(np.zeros((0, 1), dtype=np.int64))
    c = ChainComplex(
        boundaries={1: d1, 5: d5},
        dimensions=[0, 1, 2, 3, 4, 5],
        cells={0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1},
    )
    wh = WhiteheadGroup(rank=0, description="Wh(Z/3)=0", computable=True, exact=True)
    wall = ObstructionResult(
        dimension=5,
        pi="Z/3",
        computable=True,
        exact=True,
        value=0,
        modulus=2,
        assumptions=[],
    )

    is_homeo, reason = analyze_homeomorphism_high_dim(
        c,
        c,
        dim=5,
        pi_group="Z/3",
        whitehead_group=wh,
        wall_obstruction=wall,
        homotopy_completion_certificate={
            "provided": True,
            "source": "legacy-wrapper",
            "exact": True,
            "validated": True,
            "equivalence_type": "s_cobordism",
            "summary": "validated certificate",
        },
    )
    assert is_homeo is True
    assert "SUCCESS" in reason


def test_high_dim_product_group_branch_uses_wall_group_ring_theorem_tag_and_assembly_state():
    d1 = sp.csr_matrix(np.zeros((1, 0), dtype=np.int64))
    d5 = sp.csr_matrix(np.zeros((0, 1), dtype=np.int64))
    c = ChainComplex(
        boundaries={1: d1, 5: d5},
        dimensions=[0, 1, 2, 3, 4, 5],
        cells={0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1},
    )
    wh = WhiteheadGroup(
        rank=0, description="Wh(Z_2 x Z_3)=0", computable=True, exact=True
    )
    wall = ObstructionResult(
        dimension=5,
        pi="Z_2 x Z_3",
        computable=True,
        exact=False,
        value=None,
        modulus=None,
        message="factor-wise surrogate",
        assumptions=[],
        decomposition_kind="factor_surrogate",
        assembly_certified=False,
    )
    res = analyze_homeomorphism_high_dim_result(
        c,
        c,
        dim=5,
        pi_group="Z_2 x Z_3",
        whitehead_group=wh,
        wall_obstruction=wall,
    )
    assert res.status == "inconclusive"
    assert res.theorem == "Wall L-theory over group rings"
    assert res.theorem_tag == "highdim.wall.group_ring"
    assert "assembly" in " ".join(res.missing_data).lower()


def test_3d_recognition_certificate_can_complete_non_poincare_branch():
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    d2 = sp.csr_matrix(np.array([[3]], dtype=np.int64))
    d3 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    c = ChainComplex(
        boundaries={1: d1, 2: d2, 3: d3},
        dimensions=[0, 1, 2, 3],
        cells={0: 1, 1: 1, 2: 1, 3: 1},
    )
    pi = FundamentalGroup(generators=["g"], relations=[["g", "g", "g"]])
    cert = ThreeManifoldRecognitionCertificate(
        provided=True,
        source="external",
        exact=True,
        validated=True,
        summary="certified geometrization witness",
    )
    res = analyze_homeomorphism_3d_result(
        c, c, pi1_1=pi, pi1_2=pi, recognition_certificate=cert
    )
    assert res.status == "success"
    assert res.is_homeomorphic is True
    assert res.theorem == "Geometrization / 3-manifold recognition"


def test_3d_nondecision_ready_recognition_certificate_remains_inconclusive():
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    d2 = sp.csr_matrix(np.array([[3]], dtype=np.int64))
    d3 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    c = ChainComplex(
        boundaries={1: d1, 2: d2, 3: d3},
        dimensions=[0, 1, 2, 3],
        cells={0: 1, 1: 1, 2: 1, 3: 1},
    )
    pi = FundamentalGroup(generators=["g"], relations=[["g", "g", "g"]])
    res = analyze_homeomorphism_3d_result(
        c,
        c,
        pi1_1=pi,
        pi1_2=pi,
        recognition_certificate={
            "provided": True,
            "source": "external",
            "exact": True,
            "validated": False,
        },
    )
    assert res.status == "inconclusive"
    assert "decision-ready" in res.reasoning


def test_high_dim_product_group_requires_decision_ready_assembly_certificate_for_success():
    d1 = sp.csr_matrix(np.zeros((1, 0), dtype=np.int64))
    d5 = sp.csr_matrix(np.zeros((0, 1), dtype=np.int64))
    c = ChainComplex(
        boundaries={1: d1, 5: d5},
        dimensions=[0, 1, 2, 3, 4, 5],
        cells={0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1},
    )
    wh = WhiteheadGroup(
        rank=0, description="Wh(Z_2 x Z_3)=0", computable=True, exact=True
    )
    wall = ObstructionResult(
        dimension=5,
        pi="Z_2 x Z_3",
        computable=True,
        exact=True,
        value=0,
        modulus=None,
        assumptions=[],
        decomposition_kind="factor_surrogate",
        assembly_certified=False,
        obstructs=False,
        zero_certified=True,
    )
    comp = HomotopyCompletionCertificate(
        provided=True,
        source="test",
        exact=True,
        validated=True,
        summary="ready",
    )
    assembly = ProductAssemblyCertificate(
        provided=True,
        source="assembly-test",
        exact=True,
        validated=True,
        summary="ready",
    )
    res = analyze_homeomorphism_high_dim_result(
        c,
        c,
        dim=5,
        pi_group="Z_2 x Z_3",
        whitehead_group=wh,
        wall_obstruction=wall,
        homotopy_completion_certificate=comp,
        product_assembly_certificate=assembly,
    )
    assert res.status == "success"
    assert res.theorem == "Wall L-theory over group rings"


def test_high_dim_legacy_wrapper_forwards_obstruction_inputs():
    d1 = sp.csr_matrix(np.zeros((1, 0), dtype=np.int64))
    d5 = sp.csr_matrix(np.zeros((0, 1), dtype=np.int64))
    c = ChainComplex(
        boundaries={1: d1, 5: d5},
        dimensions=[0, 1, 2, 3, 4, 5],
        cells={0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1},
    )
    wh = WhiteheadGroup(rank=0, description="Wh(1)=0", computable=True, exact=True)
    wall = ObstructionResult(
        dimension=5,
        pi="1",
        computable=True,
        exact=True,
        value=1,
        modulus=None,
        assumptions=[],
    )

    is_homeo, reason = analyze_homeomorphism_high_dim(
        c,
        c,
        dim=5,
        pi1=FundamentalGroup(generators=[], relations=[]),
        whitehead_group=wh,
        wall_obstruction=wall,
    )
    assert is_homeo is False
    assert "SURGERY_REQUIRED" in reason


def test_2d_detects_cohomology_mismatch_even_when_homology_matches():
    class FakeSurface:
        coefficient_ring = "Z"

        def __init__(self, h1_torsion, h1co_torsion):
            self._h1_torsion = h1_torsion
            self._h1co_torsion = h1co_torsion

        def homology(self, n):
            if n == 2:
                return 1, []
            if n == 1:
                return 2, list(self._h1_torsion)
            if n == 0:
                return 1, []
            return 0, []

        def cohomology(self, n):
            if n == 2:
                return 1, []
            if n == 1:
                return 2, list(self._h1co_torsion)
            if n == 0:
                return 1, []
            return 0, []

    c1 = FakeSurface(h1_torsion=[], h1co_torsion=[])
    c2 = FakeSurface(h1_torsion=[], h1co_torsion=[2])
    res = analyze_homeomorphism_2d_result(c1, c2)
    assert res.status == "impediment"
    assert res.is_homeomorphic is False
    assert "Cohomology groups differ" in res.reasoning


def test_2d_manual_cohomology_signature_can_replace_failed_extraction():
    class BrokenCohomologySurface:
        coefficient_ring = "Z"

        def homology(self, n):
            if n == 2:
                return 1, []
            if n == 1:
                return 2, []
            if n == 0:
                return 1, []
            return 0, []

        def cohomology(self, n):
            raise RuntimeError("cohomology unavailable")

    signature = {
        "coefficient_ring": "Z",
        "groups": {
            0: {"rank": 1, "torsion": []},
            1: {"rank": 2, "torsion": []},
            2: {"rank": 1, "torsion": []},
        },
    }

    res = analyze_homeomorphism_2d_result(
        BrokenCohomologySurface(),
        BrokenCohomologySurface(),
        cohomology_signature_1=signature,
        cohomology_signature_2=signature,
    )
    assert res.status == "success"
    assert res.is_homeomorphic is True


def test_2d_manual_cohomology_signature_mismatch_is_detected():
    class BrokenCohomologySurface:
        coefficient_ring = "Z"

        def homology(self, n):
            if n == 2:
                return 1, []
            if n == 1:
                return 2, []
            if n == 0:
                return 1, []
            return 0, []

        def cohomology(self, n):
            raise RuntimeError("cohomology unavailable")

    signature_1 = {
        "coefficient_ring": "Z",
        "groups": {0: (1, []), 1: (2, []), 2: (1, [])},
    }
    signature_2 = {
        "coefficient_ring": "Z",
        "groups": {0: (1, []), 1: (3, []), 2: (1, [])},
    }

    res = analyze_homeomorphism_2d_result(
        BrokenCohomologySurface(),
        BrokenCohomologySurface(),
        cohomology_signature_1=signature_1,
        cohomology_signature_2=signature_2,
    )
    assert res.status == "impediment"
    assert "Manual cohomology signatures differ" in res.reasoning


def test_2d_structured_cohomology_ring_witness_detects_basis_mismatch():
    class FakeSurface:
        coefficient_ring = "Z"

        def homology(self, n):
            if n == 2:
                return 1, []
            if n == 1:
                return 2, []
            if n == 0:
                return 1, []
            return 0, []

        def cohomology(self, n):
            if n == 2:
                return 1, []
            if n == 1:
                return 2, []
            if n == 0:
                return 1, []
            return 0, []

    ring_1 = {
        "coefficient_ring": "Z",
        "basis": {0: ["1"], 1: ["a"], 2: ["omega"]},
        "unit": "1",
        "products": {("a", "a"): "omega"},
    }
    ring_2 = {
        "coefficient_ring": "Z",
        "basis": {0: ["1"], 1: ["b"], 2: ["omega"]},
        "unit": "1",
        "products": {("b", "b"): "omega"},
    }

    res = analyze_homeomorphism_2d_result(
        FakeSurface(),
        FakeSurface(),
        cohomology_ring_signature_1=ring_1,
        cohomology_ring_signature_2=ring_2,
    )
    assert res.status == "impediment"
    assert "Ring witness mismatch" in res.evidence


def test_2d_structured_cohomology_ring_witness_can_certify_match():
    class FakeSurface:
        coefficient_ring = "Z"

        def homology(self, n):
            if n == 2:
                return 1, []
            if n == 1:
                return 2, []
            if n == 0:
                return 1, []
            return 0, []

        def cohomology(self, n):
            if n == 2:
                return 1, []
            if n == 1:
                return 2, []
            if n == 0:
                return 1, []
            return 0, []

    ring = {
        "coefficient_ring": "Z",
        "basis": {0: ["1"], 1: ["a"], 2: ["omega"]},
        "unit": "1",
        "products": {("a", "a"): "omega"},
    }

    res = analyze_homeomorphism_2d_result(
        FakeSurface(),
        FakeSurface(),
        cohomology_ring_signature_1=ring,
        cohomology_ring_signature_2=ring,
    )
    assert res.status == "success"
    assert res.is_homeomorphic is True


def test_high_dim_requires_shared_cohomology_coefficients():
    d1 = sp.csr_matrix(np.zeros((1, 0), dtype=np.int64))
    d5 = sp.csr_matrix(np.zeros((0, 1), dtype=np.int64))
    c_z = ChainComplex(
        boundaries={1: d1, 5: d5},
        dimensions=[0, 1, 2, 3, 4, 5],
        cells={0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1},
        coefficient_ring="Z",
    )
    c_q = ChainComplex(
        boundaries={1: d1, 5: d5},
        dimensions=[0, 1, 2, 3, 4, 5],
        cells={0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1},
        coefficient_ring="Q",
    )
    res = analyze_homeomorphism_high_dim_result(c_z, c_q, dim=5)
    assert res.status == "inconclusive"
    assert "shared coefficient ring" in res.reasoning


def test_high_dim_infers_finite_cyclic_descriptor_from_pi1_presentation():
    d1 = sp.csr_matrix(np.zeros((1, 0), dtype=np.int64))
    d5 = sp.csr_matrix(np.zeros((0, 1), dtype=np.int64))
    c = ChainComplex(
        boundaries={1: d1, 5: d5},
        dimensions=[0, 1, 2, 3, 4, 5],
        cells={0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1},
    )
    pi = FundamentalGroup(generators=["g_0"], relations=[["g_0", "g_0", "g_0"]])
    res = analyze_homeomorphism_high_dim_result(c, c, dim=5, pi1=pi)
    assert res.status == "inconclusive"
    assert "pi_1/group-ring descriptor is missing" not in res.reasoning


def test_poincare_homology_sphere_with_explicit_trivial_pi1_is_homeomorphic():
    d1 = sp.csr_matrix(np.zeros((1, 0), dtype=np.int64))
    d2 = sp.csr_matrix(np.zeros((0, 0), dtype=np.int64))
    d3 = sp.csr_matrix(np.zeros((0, 1), dtype=np.int64))
    c = ChainComplex(
        boundaries={1: d1, 2: d2, 3: d3},
        dimensions=[0, 1, 2, 3],
        cells={0: 1, 1: 0, 2: 0, 3: 1},
    )

    pi = FundamentalGroup(generators=[], relations=[])
    res = analyze_homeomorphism_3d_result(c, c, pi1_1=pi, pi1_2=pi)
    assert res.status == "success"
    assert res.is_homeomorphic is True
    assert "Poincaré conjecture" in res.reasoning


def test_2d_detects_cup_product_signature_mismatch():
    class FakeSurface:
        coefficient_ring = "Z"

        def homology(self, n):
            if n == 2:
                return 1, []
            if n == 1:
                return 2, []
            if n == 0:
                return 1, []
            return 0, []

        def cohomology(self, n):
            if n == 2:
                return 1, []
            if n == 1:
                return 2, []
            if n == 0:
                return 1, []
            return 0, []

    c1 = FakeSurface()
    c2 = FakeSurface()
    res = analyze_homeomorphism_2d_result(
        c1,
        c2,
        cup_product_signature_1={"u^2": 1},
        cup_product_signature_2={"u^2": 0},
    )
    assert res.status == "impediment"
    assert "cup-product incompatibility" in res.reasoning


def test_high_dim_cup_product_requires_both_signatures_when_used():
    d1 = sp.csr_matrix(np.zeros((1, 0), dtype=np.int64))
    d5 = sp.csr_matrix(np.zeros((0, 1), dtype=np.int64))
    c = ChainComplex(
        boundaries={1: d1, 5: d5},
        dimensions=[0, 1, 2, 3, 4, 5],
        cells={0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1},
    )
    res = analyze_homeomorphism_high_dim_result(
        c,
        c,
        dim=5,
        cup_product_signature_1={"x": 1},
        cup_product_signature_2=None,
    )
    assert res.status == "inconclusive"
    assert "cup-product signatures" in res.reasoning


def test_s2_models_homeomorphism():
    c1 = to_complex(build_tetrahedron())
    c2 = to_complex(build_octahedron())
    c3 = to_complex(build_icosahedron())

    # They should all be homeomorphic to each other
    is_homeo_1, _ = analyze_homeomorphism_2d(c1, c2)
    is_homeo_2, _ = analyze_homeomorphism_2d(c2, c3)
    assert is_homeo_1
    assert is_homeo_2


def test_s2_vs_torus_homeomorphism():
    c1 = to_complex(build_tetrahedron())
    c2 = to_complex(build_torus())

    is_homeo, _ = analyze_homeomorphism_2d(c1, c2)
    assert not is_homeo
