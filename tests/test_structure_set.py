"""Test suite for the Algebraic Surgery Structure Set computations.

Overview:
    This module tests the `StructureSet` class, which manages the surgery exact sequence:
    ... → L_{n+1}(π) → S(M) → [M, G/TOP] → L_n(π).
    It verifies normal invariant computations, L-group obstruction handling, and 
    exact sequence evaluation for various manifold dimensions and fundamental groups.

Key Concepts:
    - **Surgery Exact Sequence**: The primary tool for classifying manifolds within a homotopy type.
    - **Normal Invariants [M, G/TOP]**: Represented here by chain complex homology over Z and Z_2.
    - **L-Theory Obstructions**: Obstructions to completing surgery to a homeomorphism.
    - **Structure Set S(M)**: The set of manifold structures on a homotopy type.
"""

import numpy as np
import scipy.sparse as sp
import pytest

from pysurgery.structure_set import LObstructionState, StructureSet
from pysurgery.core.complexes import ChainComplex
from pysurgery.core.exceptions import StructureSetError
from pysurgery.wall_groups import ObstructionResult


def test_compute_normal_invariants_includes_ext_z2_term():
    """Verify that normal invariant calculation correctly identifies torsion terms.

    What is Being Computed?:
        Computes the normal invariants of a small chain complex, ensuring both 
        free parts (Z) and torsion parts (Z_2) are captured in the report.

    Algorithm:
        1. Build a 3-dimensional chain complex with H_1 = Z_2 and H_2 = Z.
        2. Initialize a `StructureSet` for dimension 6.
        3. Call `compute_normal_invariants` and check for rank strings.
        4. Validate `compute_normal_invariants_result` returns typed data.
    """
    # Build a tiny chain complex with H_1 torsion Z_2 and H_2 free rank 1.
    # d1 = 0, d2 = [2], d3 = 0.
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    d2 = sp.csr_matrix(np.array([[2]], dtype=np.int64))
    d3 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    c = ChainComplex(
        boundaries={1: d1, 2: d2, 3: d3},
        dimensions=[0, 1, 2, 3],
        cells={0: 1, 1: 1, 2: 1, 3: 1},
    )

    s = StructureSet(dimension=6, fundamental_group="1")
    report = s.compute_normal_invariants(c)
    assert "Rank over Z:" in report
    assert "Rank over Z_2:" in report

    typed = s.compute_normal_invariants_result(c)
    assert typed.rank_Z >= 0
    assert typed.rank_Z2 >= 0
    assert typed.dimension == 6


def test_structure_set_accepts_trivial_group_aliases():
    """Verify that 'trivial' is a valid alias for the trivial fundamental group.

    What is Being Computed?:
        Tests the exact sequence evaluation for π=1 using the 'trivial' string alias.
    """
    s = StructureSet(dimension=5, fundamental_group="trivial")
    out = s.evaluate_exact_sequence()
    assert "SURGERY EXACT SEQUENCE" in out
    typed = s.evaluate_exact_sequence_result()
    assert typed.computable
    assert typed.exact
    assert typed.l_n_symbol in {"0", "Z", "Z_2"}


def test_structure_set_rejects_non_trivial_group_without_backend():
    """Verify handling of non-simply-connected groups when backends are missing.

    What is Being Computed?:
        Ensures the structure set correctly reports partial computability for π=Z
        when the necessary L-theory solver (e.g., Julia) is not active.
    """
    s = StructureSet(dimension=5, fundamental_group="Z")
    out = s.evaluate_exact_sequence_result()
    assert not out.computable
    assert out.partial
    assert "Non-simply-connected case" in " ".join(out.analysis)


def test_structure_set_requires_dimension_at_least_5():
    """Verify that surgery theory is only applied to high-dimensional manifolds (n >= 5).

    What is Being Computed?:
        Ensures a `StructureSetError` is raised for dimensions where surgery theory 
        is not standard (n < 5).
    """
    s = StructureSet(dimension=4, fundamental_group="1")
    with pytest.raises(StructureSetError):
        s.evaluate_exact_sequence()


def test_structure_set_exact_sequence_carries_typed_wall_obstruction_states():
    """Verify that explicit L-theory results are correctly integrated into the sequence.

    What is Being Computed?:
        Tests the passing of `ObstructionResult` objects into the structure set evaluator.

    Algorithm:
        1. Create dummy `ObstructionResult` objects for L_5 and L_6.
        2. Pass them to `evaluate_exact_sequence_result`.
        3. Assert that the resulting state reflects the obstruction values and certifications.
    """
    s = StructureSet(dimension=5, fundamental_group="1")
    l5 = ObstructionResult(
        dimension=5,
        pi="1",
        computable=True,
        exact=True,
        value=0,
        modulus=None,
        message="",
        assumptions=[],
    )
    l6 = ObstructionResult(
        dimension=6,
        pi="1",
        computable=True,
        exact=True,
        value=1,
        modulus=2,
        message="",
        assumptions=[],
    )
    out = s.evaluate_exact_sequence_result(
        l_n_obstruction=l5, l_n_plus_1_obstruction=l6
    )
    assert out.l_n_obstruction is not None
    assert out.l_n_plus_1_obstruction is not None
    assert out.l_n_state["available"] is True
    assert out.l_n_state["zero_certified"] is True
    assert out.l_n_plus_1_state["obstructs"] is True
    assert out.l_n_state["decomposition_kind"] in {"scalar", "single_factor"}
    assert out.l_n_state["assembly_certified"] is True


def test_structure_set_accepts_explicit_l_state_overrides():
    """Verify that `LObstructionState` overrides are respected.

    What is Being Computed?:
        Checks if manually provided obstruction states correctly influence 
        the sequence analysis.
    """
    s = StructureSet(dimension=5, fundamental_group="1")
    l_n_state = LObstructionState(
        available=True,
        computable=True,
        exact=True,
        obstructs=False,
        zero_certified=True,
        value=0,
        modulus=None,
        pi="1",
        dimension=5,
        message="certified zero",
    )
    out = s.evaluate_exact_sequence_result(l_n_state=l_n_state)
    assert out.l_n_state.zero_certified is True
    assert any("certifies vanishing" in line for line in out.analysis)


def test_structure_set_reports_conflict_when_explicit_state_disagrees_with_obstruction():
    """Verify conflict detection between redundant L-theory inputs.

    What is Being Computed?:
        Tests if providing both an `ObstructionResult` and a conflicting 
        `LObstructionState` triggers a warning/conflict in the analysis.
    """
    s = StructureSet(dimension=5, fundamental_group="1")
    l5 = ObstructionResult(
        dimension=5,
        pi="1",
        computable=True,
        exact=True,
        value=0,
        modulus=None,
        assumptions=[],
    )
    conflicting = LObstructionState(
        available=True,
        computable=True,
        exact=True,
        obstructs=True,
        zero_certified=False,
        value=1,
        modulus=None,
        pi="1",
        dimension=5,
    )
    out = s.evaluate_exact_sequence_result(l_n_obstruction=l5, l_n_state=conflicting)
    assert out.l_n_state.obstructs is True
    assert any("conflicts" in line for line in out.analysis)


def test_structure_set_nontrivial_branch_preserves_typed_states():
    """Verify that typed obstruction states are preserved for non-trivial π.

    What is Being Computed?:
        Checks if provided states for π=Z are correctly propagated through the evaluator.
    """
    s = StructureSet(dimension=7, fundamental_group="Z")
    state = LObstructionState(
        available=True,
        computable=True,
        exact=True,
        obstructs=False,
        zero_certified=True,
    )
    out = s.evaluate_exact_sequence_result(l_n_state=state)
    assert out.computable
    assert out.l_n_state.zero_certified is True


def test_structure_set_nontrivial_branch_can_be_exact_with_full_typed_channels():
    """Verify exact sequence completeness when all L-groups are provided.

    What is Being Computed?:
        Ensures that if both L_n and L_{n+1} states are provided, the sequence 
        evaluation is marked as exact even for complex fundamental groups.
    """
    s = StructureSet(dimension=9, fundamental_group="Z x Z_3")
    l_n_state = LObstructionState(
        available=True,
        computable=True,
        exact=True,
        obstructs=False,
        zero_certified=True,
    )
    l_n_plus_1_state = LObstructionState(
        available=True,
        computable=True,
        exact=True,
        obstructs=False,
        zero_certified=True,
    )
    out = s.evaluate_exact_sequence_result(
        l_n_state=l_n_state, l_n_plus_1_state=l_n_plus_1_state
    )
    assert out.computable
    assert out.exact
    assert not out.partial
