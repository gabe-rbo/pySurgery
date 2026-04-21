import numpy as np
import scipy.sparse as sp
import pytest

from discrete_surface_data import get_surfaces, get_3_manifolds, to_complex
from pysurgery.core.fundamental_group import (
    extract_pi_1,
    simplify_presentation,
    infer_standard_group_descriptor,
    FundamentalGroup,
)
from pysurgery.core.complexes import CWComplex


def test_extract_pi_1_trivial():
    d1 = sp.csr_matrix(np.zeros((1, 0), dtype=np.int64))
    cw = CWComplex(cells={0: 1, 1: 0}, attaching_maps={1: d1}, dimensions=[0, 1])
    pi = extract_pi_1(cw)
    assert len(pi.generators) == 0
    assert len(pi.relations) == 0


def test_extract_pi_1_circle():
    # S^1: 1 vertex, 1 edge (loop)
    # d1: Z^1 -> Z^1 is zero
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    cw = CWComplex(cells={0: 1, 1: 1}, attaching_maps={1: d1}, dimensions=[0, 1])
    pi = extract_pi_1(cw)
    assert len(pi.generators) == 1
    assert len(pi.relations) == 0
    assert infer_standard_group_descriptor(pi) == "Z"


def test_extract_pi_1_rp2():
    # RP^2: 1 vertex, 1 edge, 1 face (relator a^2)
    # d1: 0, d2: 2
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    d2 = sp.csr_matrix(np.array([[2]], dtype=np.int64))
    cw = CWComplex(
        cells={0: 1, 1: 1, 2: 1}, attaching_maps={1: d1, 2: d2}, dimensions=[0, 1, 2]
    )
    pi = extract_pi_1(cw)
    assert len(pi.generators) == 1
    assert len(pi.relations) == 1
    # generator 'a', relator 'aa'
    assert infer_standard_group_descriptor(pi) == "Z_2"


def test_simplify_presentation_trivial_loops():
    # Presentation with a trivial generator that doesn't appear in relators
    # or is identified to be trivial.
    # Group G = <a, b | a> -> G = <b | >
    pi = FundamentalGroup(generators=["a", "b"], relations=[["a"]])
    simple = simplify_presentation(pi.generators, pi.relations)
    assert len(simple.generators) == 1
    assert simple.generators == ["b"]
    assert len(simple.relations) == 0


def test_simplify_presentation_cyclic():
    # G = <a | a, a, a> -> G = < | > (trivial)
    pi = FundamentalGroup(generators=["a"], relations=[["a"], ["a"], ["a"]])
    simple = simplify_presentation(pi.generators, pi.relations)
    assert len(simple.generators) == 0
    assert len(simple.relations) == 0


def test_infer_standard_descriptor_finite_cyclic():
    # Z_6 case
    pi = FundamentalGroup(generators=["a"], relations=[["a"] * 6])
    assert infer_standard_group_descriptor(pi) == "Z_6"


@pytest.mark.parametrize(
    "name, builder, bettis, torsion, euler",
    get_surfaces(),
)
def test_discrete_surface_fundamental_group(name, builder, bettis, torsion, euler):
    st = builder()
    complex_c = to_complex(st)
    h_rank, h_torsion = complex_c.homology(1)
    
    pi = extract_pi_1(complex_c)
    # Fundamental group abelianization rank must match H1 rank
    # (Hurewicz theorem for connected spaces)
    # All these discrete surfaces are connected.
    
    # We can't easily compute general group abelianization here 
    # but we can check rank if it's cyclic.
    desc = infer_standard_group_descriptor(pi)
    if h_rank == 1 and not h_torsion:
        assert desc == "Z"
    elif h_rank == 2 and not h_torsion:
        # For a torus, it's Z x Z.
        assert "Z x Z" in desc or "Z^2" in desc or (len(pi.generators) == 2 and len(pi.relations) == 1)
    elif h_rank == 0 and h_torsion == [2]:
        assert desc == "Z_2"


@pytest.mark.parametrize(
    "name, builder, bettis, torsion, euler",
    get_3_manifolds(),
)
def test_discrete_3_manifold_fundamental_group(name, builder, bettis, torsion, euler):
    st = builder()
    complex_c = to_complex(st)
    h_rank, h_torsion = complex_c.homology(1)
    
    pi = extract_pi_1(complex_c)
    desc = infer_standard_group_descriptor(pi)
    
    if name == "S3":
        assert desc == "1"
        assert h_rank == 0
        assert not h_torsion
