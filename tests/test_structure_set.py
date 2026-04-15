import numpy as np
import scipy.sparse as sp
import pytest

from pysurgery.structure_set import LObstructionState, StructureSet
from pysurgery.core.complexes import ChainComplex
from pysurgery.core.exceptions import StructureSetError
from pysurgery.wall_groups import ObstructionResult


def test_compute_normal_invariants_includes_ext_z2_term():
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
    s = StructureSet(dimension=5, fundamental_group="trivial")
    out = s.evaluate_exact_sequence()
    assert "SURGERY EXACT SEQUENCE" in out
    typed = s.evaluate_exact_sequence_result()
    assert typed.computable
    assert typed.exact
    assert typed.l_n_symbol in {"0", "Z", "Z_2"}


def test_structure_set_rejects_non_trivial_group_without_backend():
    s = StructureSet(dimension=5, fundamental_group="Z")
    out = s.evaluate_exact_sequence_result()
    assert not out.computable
    assert out.partial
    assert "Non-simply-connected case" in " ".join(out.analysis)


def test_structure_set_requires_dimension_at_least_5():
    s = StructureSet(dimension=4, fundamental_group="1")
    with pytest.raises(StructureSetError):
        s.evaluate_exact_sequence()


def test_structure_set_exact_sequence_carries_typed_wall_obstruction_states():
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
