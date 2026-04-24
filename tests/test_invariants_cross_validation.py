import numpy as np
import math
import scipy.sparse as sp
from pysurgery.core.complexes import SimplicialComplex, ChainComplex
from pysurgery.core.gauss_bonnet import verify_gauss_bonnet_2d
from pysurgery.core.intersection_forms import IntersectionForm
from pysurgery.core.characteristic_classes import extract_pontryagin_p1, verify_hirzebruch_signature
from pysurgery.core.fundamental_group import extract_pi_1, simplify_presentation
from pysurgery.core.quadratic_forms import arf_invariant_gf2

def test_euler_characteristic_cross_validation():
    # 1. SimplicialComplex
    faces = [(0,1,2), (0,2,3), (0,3,1), (1,2,3)] # Tetrahedron (S2)
    sc = SimplicialComplex.from_maximal_simplices(faces)
    chi_sc = sc.euler_characteristic()
    
    # 2. ChainComplex
    cc = sc.chain_complex()
    chi_cc = cc.euler_characteristic()
    
    # 3. Geometry (Gauss-Bonnet)
    vertices = np.array([[1,0,0], [0,1,0], [0,0,1], [0,0,0]], dtype=float)
    res_gb = verify_gauss_bonnet_2d((vertices, faces))
    chi_gb = res_gb["euler_characteristic"]
    
    assert chi_sc == 2
    assert chi_cc == 2
    assert chi_gb == 2
    assert math.isclose(res_gb["total_curvature"], 4 * math.pi)

def test_homology_algorithms_consistency():
    # Check that homology(ring='Z') ranks match homology(ring='Q')
    faces = [(0,1,2), (1,2,3), (2,3,4), (3,4,0), (4,0,1)] # Möbius strip-like
    sc_z = SimplicialComplex.from_maximal_simplices(faces, coefficient_ring="Z")
    sc_q = SimplicialComplex.from_maximal_simplices(faces, coefficient_ring="Q")
    
    for n in range(2):
        rank_z, torsion_z = sc_z.chain_complex().homology(n=n)
        rank_q, _ = sc_q.chain_complex().homology(n=n)
        assert rank_z == rank_q

def test_h1_pi1_rank_consistency():
    # For a torus, H1 rank should be 2, pi1 should have 2 generators
    from discrete_surface_data import build_torus
    sc = build_torus()
    
    # H1
    h1_rank, _ = sc.chain_complex().homology(n=1)
    
    # Pi1
    pi1 = extract_pi_1(sc)
    simple_pi1 = simplify_presentation(pi1.generators, pi1.relations)
    
    assert h1_rank == 2
    # Torus pi1 is Z x Z, which has 2 generators after simplification.
    assert len(simple_pi1.generators) == 2

def test_signature_hirzebruch_cross_validation():
    # CP2 intersection form is [[1]]
    Q = np.array([[1]])
    form = IntersectionForm(matrix=Q, dimension=4)
    sig = form.signature()
    p1 = extract_pontryagin_p1(form)
    
    # Signature Theorem: 3 * sig = p1
    assert 3 * sig == p1
    assert verify_hirzebruch_signature(form, p1)

def test_arf_invariant_hyperbolic_pair():
    # A single hyperbolic pair (e, f) with q(e)=1, q(f)=1 and B(e, f)=1 should have Arf = 1*1 = 1
    M = np.array([[0, 1], [1, 0]])
    q = np.array([1, 1])
    assert arf_invariant_gf2(M, q) == 1
    
    # If q(e)=1, q(f)=0, Arf = 1*0 = 0
    q2 = np.array([1, 0])
    assert arf_invariant_gf2(M, q2) == 0

def test_chain_complex_euler_formula():
    """
    Test that sum(-1^i * rank(Ci)) == sum(-1^i * rank(Hi))
    """
    # Create a random complex
    d1 = sp.csr_matrix([[1, 1, 0], [0, 1, 1]]) # 2 rows (1-cells), 3 cols (2-cells) - Wait, Ci is rank(dim i)
    # boundaries[i] maps Ci to Ci-1.
    # If cells = {0: 3, 1: 3, 2: 2}
    # d1: C1 -> C0 (3x3)
    # d2: C2 -> C1 (3x2)
    d1 = sp.csr_matrix([[1, -1, 0], [0, 1, -1], [-1, 0, 1]]) # Triangle boundary
    d2 = sp.csr_matrix(np.zeros((3, 0)))
    cc = ChainComplex(
        boundaries={1: d1, 2: d2},
        dimensions=[0, 1, 2],
        cells={0: 3, 1: 3, 2: 0}
    )
    
    chi_cells = 3 - 3 + 0
    
    h0_r, _ = cc.homology(0)
    h1_r, _ = cc.homology(1)
    h2_r, _ = cc.homology(2)
    chi_homology = h0_r - h1_r + h2_r
    
    assert chi_cells == chi_homology
    assert cc.euler_characteristic() == chi_cells
