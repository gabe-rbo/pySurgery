"""Tests for Poincare Duality Verification & Intersection Pairing Chains."""

import numpy as np
import pytest

from discrete_surface_data import build_torus, build_tetrahedron, to_complex
from pysurgery.topology.complexes import SimplicialComplex
from pysurgery.homology.poincare_duality_verification import (
    is_poincare_duality_complex,
    detect_poincare_dimension,
    simplicial_cap_product,
    compute_poincare_duality_map,
    extract_intersection_pairing_chain
)
from pysurgery.core.exceptions import DimensionError

def test_poincare_duality_verification_torus():
    """Verify that a torus passes Poincare duality checks."""
    sc = to_complex(build_torus())
    cw = sc.to_cw_complex()
    
    cert = detect_poincare_dimension(cw)
    assert cert.is_poincare is True
    assert cert.dimension == 2
    assert cert.betti_numbers[0] == 1
    assert cert.betti_numbers[1] == 2
    assert cert.betti_numbers[2] == 1


def test_poincare_duality_verification_sphere():
    """Verify that a sphere (tetrahedron boundary) passes Poincare duality."""
    sc = to_complex(build_tetrahedron())
    cw = sc.to_cw_complex()
    
    cert = detect_poincare_dimension(cw)
    assert cert.is_poincare is True
    assert cert.dimension == 2
    assert cert.betti_numbers[0] == 1
    assert cert.betti_numbers[1] == 0
    assert cert.betti_numbers[2] == 1


def test_poincare_duality_fails_on_disk():
    """Verify that a disk (sphere with one missing face) fails Poincare duality."""
    # Tetrahedron boundary is a sphere. Remove one face to get a disk.
    sc = SimplicialComplex.from_simplices([[0, 1, 2], [0, 1, 3], [0, 2, 3]])
        
    cw = sc.to_cw_complex()
    detect_poincare_dimension(cw)
    
    # Betti numbers of a disk are H_0 = 1, H_1 = 0, H_2 = 0
    # The max_dim with H_k != 0 is 0. 
    # But wait, max_dim in detect_poincare_dimension is based on the maximum key in bettis.
    # We should just assert that it is NOT a dimension 2 poincare complex.
    assert is_poincare_duality_complex(cw, 2) is False


def test_simplicial_cap_product():
    """Verify simplicial cap product calculation."""
    # Build a single 2-simplex: [0, 1, 2]
    sc = SimplicialComplex.from_simplices([[0, 1, 2]])
    
    # 2-chain: just the simplex [0, 1, 2] with coeff 1
    chain_2 = np.array([1], dtype=np.int64)
    
    # 1-cochain: evaluate on edges. Let's say it's 1 on [0, 1] and 0 elsewhere.
    # Need to know the index of [0, 1].
    idx_1 = sc.simplex_to_index(1)
    cochain_1 = np.zeros(len(idx_1), dtype=np.int64)
    if (0, 1) in idx_1:
        cochain_1[idx_1[(0, 1)]] = 1
        
    # Cap product of 2-chain with 1-cochain -> 1-chain
    # Expected: c([0, 1, 2]) \cap alpha = alpha([0, 1]) * [1, 2] = 1 * [1, 2]
    result = simplicial_cap_product(chain_2, cochain_1, n=2, k=1, sc=sc)
    
    idx_target = sc.simplex_to_index(1) # target is 2-1 = 1-chain
    expected = np.zeros(len(idx_target), dtype=np.int64)
    if (1, 2) in idx_target:
        expected[idx_target[(1, 2)]] = 1
        
    assert np.array_equal(result, expected)


def test_cap_product_dimension_error():
    """Verify that capping with invalid dimensions raises an error."""
    sc = SimplicialComplex.from_simplices([[0, 1]])
    
    chain_1 = np.array([1], dtype=np.int64)
    cochain_2 = np.array([1], dtype=np.int64)
    
    with pytest.raises(DimensionError, match="Cannot cap n-chain"):
        simplicial_cap_product(chain_1, cochain_2, n=1, k=2, sc=sc)


def test_compute_poincare_duality_map():
    """Verify Poincare duality map construction."""
    sc = to_complex(build_tetrahedron())
    
    # Fake fundamental class: all 1s for 2-simplices
    fc = np.ones(sc.count_simplices(2), dtype=np.int64)
    
    # Duality map D: C^1 -> C_1
    D = compute_poincare_duality_map(sc, dimension=2, fundamental_class=fc, k=1)
    
    num_edges = sc.count_simplices(1)
    assert D.shape == (num_edges, num_edges)


def test_extract_intersection_pairing_chain():
    """Verify intersection pairing chain extraction."""
    sc = to_complex(build_tetrahedron())
    
    # Two 1-cocycles (edges)
    num_edges = sc.count_simplices(1)
    alpha = np.zeros(num_edges, dtype=np.int64)
    beta = np.zeros(num_edges, dtype=np.int64)
    
    if num_edges > 1:
        alpha[0] = 1
        beta[1] = 1
        
    # p=1, q=1 -> p+q = 2
    pairing_chain = extract_intersection_pairing_chain(alpha, beta, p=1, q=1, sc=sc)
    
    num_faces = sc.count_simplices(2)
    assert pairing_chain.shape == (num_faces,)
    
