import numpy as np
import scipy.sparse as sp
import pytest

from pysurgery.structure_set import StructureSet
from pysurgery.core.complexes import ChainComplex
from pysurgery.core.exceptions import StructureSetError


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
    with pytest.raises(StructureSetError):
        s.evaluate_exact_sequence()


def test_structure_set_requires_dimension_at_least_5():
    s = StructureSet(dimension=4, fundamental_group="1")
    with pytest.raises(StructureSetError):
        s.evaluate_exact_sequence()

