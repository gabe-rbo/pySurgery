"""Consistency tests for fundamental group extraction across backends.

Overview:
    This suite verifies that the fundamental group π₁ extraction yields 
    consistent presentations (generators, relations, and traces) across the 
    Python and Julia backends for various CW complexes.

Key Concepts:
    - **Fundamental Group (π₁)**: The group of homotopy classes of loops in a space.
    - **Presentation Consistency**: Agreement on the number of generators and the set of relations.
    - **Spatial Traces**: Detailed paths in the 1-skeleton corresponding to group generators.
"""
import numpy as np
import scipy.sparse as sp
from pysurgery.core.complexes import CWComplex
from pysurgery.core.fundamental_group import extract_pi_1, extract_pi_1_with_traces

def test_pi1_backend_consistency_circle():
    """Verify π₁ consistency for the circle (S¹).

    What is Being Computed?:
        The fundamental group π₁(S¹) ≅ ℤ using both backends.

    Algorithm:
        1. Construct a CW complex for S¹ (1 0-cell, 1 1-cell).
        2. Extract π₁ using Python and Julia backends.
        3. Compare generators, relations, and orientation characters.
    """
    # S^1 loop
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    cw = CWComplex(cells={0: 1, 1: 1}, attaching_maps={1: d1}, dimensions=[0, 1])
    
    pi_py = extract_pi_1(cw, backend="python")
    pi_jl = extract_pi_1(cw, backend="julia")
    
    assert pi_py.generators == pi_jl.generators
    assert pi_py.relations == pi_jl.relations
    assert pi_py.orientation_character == pi_jl.orientation_character

def test_pi1_backend_consistency_rp2():
    """Verify π₁ consistency for the projective plane (RP²).

    What is Being Computed?:
        The fundamental group π₁(RP²) ≅ ℤ₂ using both backends.

    Algorithm:
        1. Construct a CW complex for RP² (1 0-cell, 1 1-cell, 1 2-cell with deg-2 map).
        2. Extract π₁ and compare results.
    """
    # RP^2
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    d2 = sp.csr_matrix(np.array([[2]], dtype=np.int64))
    cw = CWComplex(cells={0: 1, 1: 1, 2: 1}, attaching_maps={1: d1, 2: d2}, dimensions=[0, 1, 2])
    
    pi_py = extract_pi_1(cw, backend="python")
    pi_jl = extract_pi_1(cw, backend="julia")
    
    assert pi_py.generators == pi_jl.generators
    assert pi_py.relations == pi_jl.relations
    # Note: RP2 orientation character is non-trivial for the manifold, 
    # but the generator of pi1(RP2) = Z2 is usually preserved.
    assert pi_py.orientation_character == pi_jl.orientation_character

def test_pi1_backend_consistency_torus():
    """Verify π₁ consistency for the torus (T²).

    What is Being Computed?:
        The fundamental group π₁(T²) ≅ ℤ ⊕ ℤ using both backends.

    Algorithm:
        1. Construct a CW complex for T² (1 vertex, 2 edges).
        2. Extract π₁ and compare generator sets and relations.
    """
    # T^2: 1 vertex, 2 edges (a, b), 1 face (aba^-1b^-1)
    # d1 is 1x2 zero matrix
    d1 = sp.csr_matrix(np.zeros((1, 2), dtype=np.int64))
    # d2 is 2x1 zero matrix (for homology, but for pi1 we need the relator)
    # The current CWComplex.from_simplices or manual construction:
    # Let's use a manual relator via d2=0 and see if they both find zero relators
    # Actually, let's test a case where d2 has non-zero entries.
    d2 = sp.csr_matrix(np.zeros((2, 1), dtype=np.int64))
    cw = CWComplex(cells={0: 1, 1: 2, 2: 1}, attaching_maps={1: d1, 2: d2}, dimensions=[0, 1, 2])
    
    pi_py = extract_pi_1(cw, backend="python")
    pi_jl = extract_pi_1(cw, backend="julia")
    
    assert sorted(pi_py.generators) == sorted(pi_jl.generators)
    assert pi_py.relations == pi_jl.relations

def test_pi1_traces_consistency():
    """Verify consistency of spatial traces for fundamental group generators.

    What is Being Computed?:
        The spatial mapping (vertex/edge paths) of π₁ generators.

    Algorithm:
        1. Construct a minimal S¹ CW complex.
        2. Extract π₁ with traces using both backends.
        3. Compare the resulting generator paths.
    """
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    cw = CWComplex(cells={0: 1, 1: 1}, attaching_maps={1: d1}, dimensions=[0, 1])
    
    tr_py = extract_pi_1_with_traces(cw, backend="python")
    tr_jl = extract_pi_1_with_traces(cw, backend="julia")
    
    assert len(tr_py.traces) == len(tr_jl.traces)
    for t1, t2 in zip(tr_py.traces, tr_jl.traces):
        assert t1.generator == t2.generator
        assert t1.vertex_path == t2.vertex_path
        assert t1.directed_edge_path == t2.directed_edge_path
