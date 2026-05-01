"""Tests for the Homeomorphism Witness construction system.

Overview:
    This suite verifies the generation of HomeomorphismWitness objects across 
    different dimensions (3D, 4D, and high-D). It ensures that certificates 
    (isometry matrices, surgery sequence analysis, etc.) are correctly 
    populated and valid.

Key Concepts:
    - **Homeomorphism Witness**: A bundle of mathematical evidence proving two spaces 
      are homeomorphic.
    - **Lattice Isometry**: For 4D manifolds, finding an isometry matrix between 
      intersection forms.
    - **Surgery Sequence**: For high-D manifolds, checking Whitehead torsion and 
      Wall obstructions.
"""
import numpy as np
import scipy.sparse as sp

from pysurgery.core.fundamental_group import FundamentalGroup
from pysurgery.core.intersection_forms import IntersectionForm
from pysurgery.core.k_theory import WhiteheadGroup
from pysurgery.homeomorphism_witness import (
    build_3d_homeomorphism_witness,
    build_4d_homeomorphism_witness,
    build_high_dim_homeomorphism_witness,
    build_homeomorphism_witness,
)
from pysurgery.structure_set import NormalInvariantsResult, SurgeryExactSequenceResult
from pysurgery.wall_groups import ObstructionResult
from pysurgery.core.complexes import ChainComplex


def test_build_4d_definite_homeomorphism_witness_contains_explicit_isometry_matrix():
    """Verify that 4D witness construction returns an explicit isometry matrix.

    What is Being Computed?:
        An isometry matrix $U$ such that $U^T Q_1 U = Q_2$ for two definite 
        intersection forms $Q_1, Q_2$.

    Algorithm:
        1. Define two isomorphic definite quadratic forms.
        2. Call build_4d_homeomorphism_witness.
        3. Assert the returned witness contains the isometry matrix.
        4. Validate the matrix by applying the isometry.

    Preserved Invariants:
        - Intersection form (signature and type) — preserved by isometry.
        - Kirby-Siebenmann invariant (ks) — checked for matching.
    """
    q1 = np.array([[1, 0], [0, 1]], dtype=np.int64)
    u = np.array([[3, 2], [2, 1]], dtype=np.int64)
    q2 = u.T @ q1 @ u

    form1 = IntersectionForm(matrix=q1, dimension=4)
    form2 = IntersectionForm(matrix=np.matmul(np.matmul(u.T, q1), u), dimension=4)

    res = build_4d_homeomorphism_witness(
        form1, form2, ks1=0, ks2=0, simply_connected=True
    )
    assert res.status == "success"
    assert res.witness is not None
    assert res.witness.explicit_map is not None
    witness_u = np.asarray(res.witness.explicit_map, dtype=np.int64)
    assert np.array_equal(np.matmul(np.matmul(witness_u.T, q1), witness_u), q2)


def test_build_4d_definite_homeomorphism_witness_finds_larger_isometry_matrix():
    q1 = np.array([[1, 0], [0, 1]], dtype=np.int64)
    u = np.array([[5, 2], [2, 1]], dtype=np.int64)
    q2 = u.T @ q1 @ u

    form1 = IntersectionForm(matrix=q1, dimension=4)
    form2 = IntersectionForm(matrix=q2, dimension=4)
    res = build_4d_homeomorphism_witness(
        form1, form2, ks1=0, ks2=0, simply_connected=True
    )

    assert res.status == "success"
    assert res.witness is not None
    assert res.witness.explicit_map is not None
    witness_u = np.asarray(res.witness.explicit_map, dtype=np.int64)
    assert np.array_equal(witness_u.T @ q1 @ witness_u, q2)


def test_build_4d_witness_accepts_external_definite_lattice_certificate():
    q1 = np.array([[1, 0], [0, 1]], dtype=np.int64)
    u = np.array([[8, 3], [3, 1]], dtype=np.int64)
    q2 = u.T @ q1 @ u
    form1 = IntersectionForm(matrix=q1, dimension=4)
    form2 = IntersectionForm(matrix=q2, dimension=4)
    res = build_4d_homeomorphism_witness(
        form1,
        form2,
        ks1=0,
        ks2=0,
        simply_connected=True,
        definite_lattice_isometry_certificate={
            "provided": True,
            "source": "external",
            "exact": True,
            "validated": True,
            "isometry_matrix": u.tolist(),
        },
    )
    assert res.status == "success"
    assert res.witness is not None
    assert res.witness.certificates.get("isometry_search_mode") in {
        "bounded_search",
        "external_certificate",
    }


def test_build_3d_trivial_pi1_witness_uses_poincare_branch():
    d1 = sp.csr_matrix(np.zeros((1, 0), dtype=np.int64))
    d2 = sp.csr_matrix(np.zeros((0, 0), dtype=np.int64))
    d3 = sp.csr_matrix(np.zeros((0, 1), dtype=np.int64))
    c = ChainComplex(
        boundaries={1: d1, 2: d2, 3: d3},
        dimensions=[0, 1, 2, 3],
        cells={0: 1, 1: 0, 2: 0, 3: 1},
    )
    pi = FundamentalGroup(generators=[], relations=[])

    res = build_3d_homeomorphism_witness(c, c, pi1_1=pi, pi1_2=pi)
    assert res.status == "success"
    assert res.witness is not None
    assert res.witness.kind == "poincare_conjecture"
    assert "Poincaré conjecture" in res.witness.description


def test_build_3d_witness_accepts_recognition_certificate_for_nontrivial_pi1_case():
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    d2 = sp.csr_matrix(np.array([[3]], dtype=np.int64))
    d3 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    c = ChainComplex(
        boundaries={1: d1, 2: d2, 3: d3},
        dimensions=[0, 1, 2, 3],
        cells={0: 1, 1: 1, 2: 1, 3: 1},
    )
    pi = FundamentalGroup(generators=["g"], relations=[["g", "g", "g"]])
    res = build_3d_homeomorphism_witness(
        c,
        c,
        pi1_1=pi,
        pi1_2=pi,
        recognition_certificate={
            "provided": True,
            "source": "test",
            "exact": True,
            "validated": True,
        },
    )
    assert res.status == "success"
    assert res.witness is not None
    assert res.witness.certificates.get("recognition_certificate") is not None


def test_build_high_dim_witness_returns_certificate_bundle():
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

    res = build_high_dim_homeomorphism_witness(
        c,
        c,
        dim=5,
        pi1=FundamentalGroup(generators=[], relations=[]),
        whitehead_group=wh,
        wall_obstruction=wall,
        normal_invariants_1=normal,
        normal_invariants_2=normal,
        surgery_sequence=seq,
        homotopy_equivalence_witness={"map": "f", "degree": 1},
        homotopy_witness_hook={
            "provided": True,
            "source": "test",
            "exact": True,
            "summary": "manual hook",
        },
        homotopy_completion_certificate={
            "provided": True,
            "source": "typed",
            "exact": True,
            "validated": True,
            "equivalence_type": "s_cobordism",
            "summary": "typed completion",
        },
    )
    assert res.status == "success"
    assert res.witness is not None
    assert res.witness.kind == "s_cobordism_certificate"
    assert res.witness.certificates["whitehead_group"].rank == 0
    assert res.witness.certificates["wall_obstruction"].value == 0
    wall_state = res.witness.certificates["wall_obstruction_state"]
    assert wall_state["available"] is True
    assert wall_state["zero_certified"] is True
    assert wall_state["obstructs"] is False
    assert "surgery_sequence_l_n_state" in res.witness.certificates
    assert "surgery_sequence_l_n_plus_1_state" in res.witness.certificates
    assert "homotopy_witness_hook" in res.witness.certificates
    assert "homotopy_completion_certificate" in res.witness.certificates


def test_build_high_dim_witness_propagates_surgery_required_status():
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

    res = build_high_dim_homeomorphism_witness(
        c,
        c,
        dim=5,
        pi1=FundamentalGroup(generators=[], relations=[]),
        whitehead_group=wh,
        wall_obstruction=wall,
    )
    assert res.status == "surgery_required"
    assert res.witness is None
    assert "SURGERY_REQUIRED" in res.reasoning


def test_build_high_dim_witness_inconclusive_retains_phase5_metadata_on_source_result():
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

    res = build_high_dim_homeomorphism_witness(
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
    assert res.witness is None
    assert res.source_result is not None
    assert isinstance(res.source_result.certificates.get("decision_dag"), dict)
    assert isinstance(res.source_result.certificates.get("homotopy_witness_hook"), dict)


def test_build_high_dim_witness_accepts_product_assembly_certificate_inputs():
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

    res = build_high_dim_homeomorphism_witness(
        c,
        c,
        dim=5,
        pi_group="Z_2 x Z_3",
        whitehead_group=wh,
        wall_obstruction=wall,
        homotopy_completion_certificate={
            "provided": True,
            "source": "test",
            "exact": True,
            "validated": True,
        },
        product_assembly_certificate={
            "provided": True,
            "source": "test",
            "exact": True,
            "validated": True,
        },
    )
    assert res.status == "success"
    assert res.witness is not None
    assert res.witness.certificates.get("product_assembly_certificate") is not None


def test_build_homeomorphism_witness_dim2_dispatch_does_not_forward_3d_only_kwargs():
    d1 = sp.csr_matrix(np.zeros((1, 0), dtype=np.int64))
    d2 = sp.csr_matrix(np.zeros((0, 1), dtype=np.int64))
    c = ChainComplex(
        boundaries={1: d1, 2: d2},
        dimensions=[0, 1, 2],
        cells={0: 1, 1: 0, 2: 1},
    )

    res = build_homeomorphism_witness(
        c1=c, c2=c, dim=2, recognition_certificate={"provided": True}
    )
    assert res.status in {"success", "inconclusive"}


def test_build_homeomorphism_witness_dim3_dispatch_forwards_recognition_certificate():
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    d2 = sp.csr_matrix(np.array([[3]], dtype=np.int64))
    d3 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    c = ChainComplex(
        boundaries={1: d1, 2: d2, 3: d3},
        dimensions=[0, 1, 2, 3],
        cells={0: 1, 1: 1, 2: 1, 3: 1},
    )
    pi = FundamentalGroup(generators=["g"], relations=[["g", "g", "g"]])

    cert = {
        "provided": True,
        "source": "dispatch-test",
        "exact": True,
        "validated": True,
    }
    res = build_homeomorphism_witness(
        c1=c,
        c2=c,
        dim=3,
        pi1_1=pi,
        pi1_2=pi,
        recognition_certificate=cert,
    )

    assert res.status == "success"
    assert res.witness is not None
    assert res.witness.certificates.get("recognition_certificate") is not None
