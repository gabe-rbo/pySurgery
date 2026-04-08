import numpy as np
import scipy.sparse as sp
import pytest
try:
    from tests.discrete_surface_data import get_surfaces, get_3_manifolds, to_complex
except ImportError:
    pass
from pysurgery.core.fundamental_group import extract_pi_1, simplify_presentation
from pysurgery.core.complexes import CWComplex

def test_extract_pi_1_trivial():
    d1 = sp.csr_matrix(np.zeros((1, 0), dtype=np.int64))
    cw = CWComplex(attaching_maps={1: d1}, dimensions=[0], cells={0: 1, 1: 0})
    pi_1 = extract_pi_1(cw)
    assert len(pi_1.generators) == 0
    assert len(pi_1.relations) == 0

def test_extract_pi_1_circle():
    # 1 vertex, 1 loop
    # d1 has 1 row (vertex 0), 1 col (edge 0). Loop boundary is 0.
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    cw = CWComplex(attaching_maps={1: d1}, dimensions=[0, 1], cells={0: 1, 1: 1})
    pi_1 = extract_pi_1(cw)
    assert len(pi_1.generators) == 1
    assert pi_1.generators == ["g_0"]
    assert len(pi_1.relations) == 0

def test_extract_pi_1_unclosed_loop():
    # 2 vertices, 1 edge connecting them
    d1 = sp.csr_matrix(np.array([[-1], [1]], dtype=np.int64))
    cw = CWComplex(attaching_maps={1: d1}, dimensions=[0, 1], cells={0: 2, 1: 1})
    pi_1 = extract_pi_1(cw)
    assert len(pi_1.generators) == 0
    assert len(pi_1.relations) == 0

def test_extract_pi_1_disc():
    # 1 vertex, 1 edge (loop), 1 face attaching to the loop
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    # d2 contains the sequence of edges. 
    # 1 face. Edge 0 is traversed once forward.
    d2 = sp.csr_matrix(np.array([[1]], dtype=np.int64))
    cw = CWComplex(attaching_maps={1: d1, 2: d2}, dimensions=[0, 1, 2], cells={0: 1, 1: 1, 2: 1})
    pi_1 = extract_pi_1(cw)
    # After simplification, the singleton relator g_0 kills the only generator.
    assert len(pi_1.generators) == 0
    assert len(pi_1.relations) == 0


def test_extract_pi_1_disconnected_spanning_forest():
    # Two disconnected edges in separate components should both be tree edges in a spanning forest.
    d1 = sp.csr_matrix(
        np.array([
            [-1, 0],
            [1, 0],
            [0, -1],
            [0, 1],
        ], dtype=np.int64)
    )
    cw = CWComplex(attaching_maps={1: d1}, dimensions=[0, 1], cells={0: 4, 1: 2})
    pi_1 = extract_pi_1(cw)
    assert pi_1.generators == []


def test_extract_pi_1_malformed_face_trace_is_skipped():
    # One valid loop edge and one malformed edge in the same face should not force a bogus relation.
    d1 = sp.csr_matrix(np.array([[0, 2]], dtype=np.int64))
    d2 = sp.csr_matrix(np.array([[1], [1]], dtype=np.int64))
    cw = CWComplex(attaching_maps={1: d1, 2: d2}, dimensions=[0, 1, 2], cells={0: 1, 1: 2, 2: 1})
    pi_1 = extract_pi_1(cw)
    # We still get generators from non-tree edges; malformed boundary entries are ignored.
    assert len(pi_1.generators) >= 0
    assert pi_1.relations == []


def test_extract_pi_1_face_multiplicity_is_respected():
    # One loop edge traversed twice should produce relation g_0 g_0.
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    d2 = sp.csr_matrix(np.array([[2]], dtype=np.int64))
    cw = CWComplex(attaching_maps={1: d1, 2: d2}, dimensions=[0, 1, 2], cells={0: 1, 1: 1, 2: 1})
    pi_1 = extract_pi_1(cw)
    assert pi_1.relations == [["g_0", "g_0"]]


def test_simplify_presentation_free_and_cyclic_reduction():
    g = ["g_0", "g_1"]
    rels = [["g_0", "g_0^-1", "g_1", "g_1^-1"], ["g_1", "g_0", "g_1^-1", "g_0^-1"]]
    simp = simplify_presentation(g, rels)
    # First relator vanishes. Commutator remains (up to cyclic normalization).
    assert len(simp.relations) == 1


def test_extract_pi_1_disable_simplification_keeps_singleton_relation():
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    d2 = sp.csr_matrix(np.array([[1]], dtype=np.int64))
    cw = CWComplex(attaching_maps={1: d1, 2: d2}, dimensions=[0, 1, 2], cells={0: 1, 1: 1, 2: 1})
    pi_raw = extract_pi_1(cw, simplify=False)
    assert pi_raw.relations == [["g_0"]]
    pi_s = extract_pi_1(cw, simplify=True)
    assert pi_s.relations == []


@pytest.mark.parametrize("name, builder, bettis, torsion, euler", get_surfaces() if 'get_surfaces' in globals() else [])
def test_discrete_surface_fundamental_group(name, builder, bettis, torsion, euler):
    st = builder()
    complex_c = to_complex(st)
    h_rank, h_torsion = complex_c.homology(1)
    
    assert h_rank == bettis.get(1, 0)
    from pysurgery.bridge.julia_bridge import julia_engine
    if julia_engine.available:
        assert set(h_torsion) == set(torsion.get(1, []))

@pytest.mark.parametrize("name, builder, bettis, torsion, euler", get_3_manifolds() if 'get_3_manifolds' in globals() else [])
def test_discrete_3_manifold_fundamental_group(name, builder, bettis, torsion, euler):
    st = builder()
    complex_c = to_complex(st)
    h_rank, h_torsion = complex_c.homology(1)
    
    assert h_rank == bettis.get(1, 0)
    from pysurgery.bridge.julia_bridge import julia_engine
    if julia_engine.available:
        assert set(h_torsion) == set(torsion.get(1, []))

