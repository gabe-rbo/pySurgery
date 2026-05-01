"""Cross-validation suite for verifying consistency between different topological invariant engines.

Overview:
    This test suite ensures that invariants computed via different pathways (e.g., 
    simplicial homology vs. chain complex homology, or Gauss-Bonnet curvature vs. 
    combinatorial Euler characteristic) yield identical results. It serves as 
    the primary source of truth for engine interoperability.

Key Concepts:
    - **Euler Characteristic Consistency**: χ(K) = χ(C_*(K)) = ∫ K dA / 2π.
    - **Hurewicz Isomorphism**: Consistency between π₁ generators and H₁ rank for surfaces.
    - **Signature Theorem**: Verification of the Hirzebruch Signature Theorem (3σ = p₁).
    - **Arf Invariant**: Correctness of quadratic form invariants on hyperbolic pairs.
"""
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
    """Verify Euler characteristic consistency across simplicial, chain, and geometric pathways.

    What is Being Computed?:
        The Euler characteristic χ of a tetrahedron (S²) using three distinct methods:
        1. Combinatorial sum of simplices in a SimplicialComplex.
        2. Alternating sum of ranks in a ChainComplex.
        3. Integration of Gaussian curvature via the Gauss-Bonnet theorem.

    Algorithm:
        1. Construct a tetrahedron as a SimplicialComplex.
        2. Extract its ChainComplex and compute χ.
        3. Define vertex coordinates and compute total curvature using verify_gauss_bonnet_2d.
        4. Cross-validate that all results equal 2 (for S²).

    Preserved Invariants:
        - Euler characteristic (homotopy invariant).
        - Total curvature (Topological invariant via Gauss-Bonnet).
    """
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
    """Verify that homology rank is independent of the coefficient ring for torsion-free spaces.

    What is Being Computed?:
        The homology ranks of a Möbius strip-like complex using both ℤ and ℚ coefficients.

    Algorithm:
        1. Construct a complex with ℤ coefficients and one with ℚ coefficients.
        2. Compute homology groups for dimensions 0 and 1.
        3. Ensure that the rank (free part) is identical in both cases.

    Preserved Invariants:
        - Homology rank (Betti numbers).
    """
    # Check that homology(ring='Z') ranks match homology(ring='Q')
    faces = [(0,1,2), (1,2,3), (2,3,4), (3,4,0), (4,0,1)] # Möbius strip-like
    sc_z = SimplicialComplex.from_maximal_simplices(faces, coefficient_ring="Z")
    sc_q = SimplicialComplex.from_maximal_simplices(faces, coefficient_ring="Q")
    
    for n in range(2):
        rank_z, torsion_z = sc_z.chain_complex().homology(n=n)
        rank_q, _ = sc_q.chain_complex().homology(n=n)
        assert rank_z == rank_q

def test_h1_pi1_rank_consistency():
    """Validate the Hurewicz isomorphism between π₁ and H₁ for a torus.

    What is Being Computed?:
        The relationship between the number of generators in π₁(T²) and the rank of H₁(T²; ℤ).

    Algorithm:
        1. Build a standard torus triangulation.
        2. Compute H₁ rank via Smith Normal Form.
        3. Extract the fundamental group and simplify its presentation using Tietze moves.
        4. Assert that the number of generators in the simplified π₁ matches the H₁ rank.

    Preserved Invariants:
        - First Betti number (β₁).
        - Fundamental group (π₁).
    """
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
    """Verify the Hirzebruch Signature Theorem for CP².

    What is Being Computed?:
        The relationship 3σ(M) = L₁(p₁, ..., p_k) for a 4-manifold.

    Algorithm:
        1. Initialize the intersection form of CP² (the [1] matrix).
        2. Compute the signature σ of the form.
        3. Extract the first Pontryagin class p₁ using extraction heuristics.
        4. Assert that 3 * σ == p₁ as per the Signature Theorem.

    Preserved Invariants:
        - Signature of the intersection form.
        - Pontryagin classes.
    """
    # CP2 intersection form is [[1]]
    Q = np.array([[1]])
    form = IntersectionForm(matrix=Q, dimension=4)
    sig = form.signature()
    p1 = extract_pontryagin_p1(form)
    
    # Signature Theorem: 3 * sig = p1
    assert 3 * sig == p1
    assert verify_hirzebruch_signature(form, p1)

def test_arf_invariant_hyperbolic_pair():
    """Verify the Arf invariant calculation on hyperbolic quadratic forms over GF(2).

    What is Being Computed?:
        The Arf invariant of a hyperbolic pair (e, f) with varying quadratic assignments.

    Algorithm:
        1. Construct a 2x2 hyperbolic intersection matrix.
        2. Define quadratic values q(e) and q(f).
        3. Compute the Arf invariant as the product q(e) * q(f) in GF(2).

    Preserved Invariants:
        - Arf invariant (cobordism invariant for framed surfaces).
    """
    # A single hyperbolic pair (e, f) with q(e)=1, q(f)=1 and B(e, f)=1 should have Arf = 1*1 = 1
    M = np.array([[0, 1], [1, 0]])
    q = np.array([1, 1])
    assert arf_invariant_gf2(M, q) == 1
    
    # If q(e)=1, q(f)=0, Arf = 1*0 = 0
    q2 = np.array([1, 0])
    assert arf_invariant_gf2(M, q2) == 0

def test_chain_complex_euler_formula():
    """Validate the Euler-Poincaré formula for an arbitrary chain complex.

    What is Being Computed?:
        The equality between the alternating sum of cell ranks and the alternating sum of homology ranks.

    Algorithm:
        1. Construct a manual ChainComplex representing a triangle boundary.
        2. Compute χ_cells = Σ (-1)ⁱ rank(Cᵢ).
        3. Compute χ_homology = Σ (-1)ⁱ rank(Hᵢ).
        4. Assert that χ_cells == χ_homology.

    Preserved Invariants:
        - Euler characteristic (homotopy invariant).
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
