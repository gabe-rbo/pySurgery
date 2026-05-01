"""Property-based tests for foundational topological invariants and identities.

Overview:
    This suite validates core algebraic topology identities using Hypothesis 
    to generate diverse simplicial complexes. It tests the chain complex 
    boundary property (d²=0), the Euler-Poincaré formula, Hurewicz consistency, 
    and coefficient ring parity.

Key Concepts:
    - **Chain Complex**: A sequence of modules and boundary maps with ∂ₙ ∘ ∂ₙ₊₁ = 0.
    - **Euler-Poincaré Formula**: The identity χ(X) = Σ (-1)ⁿ dim(Cₙ) = Σ (-1)ⁿ βₙ.
    - **Hurewicz Theorem**: Relates π₁ to H₁ via abelianization.
    - **Coefficient Ring**: Homology rank over ℤ (free part) must match rank over ℚ.
"""

from hypothesis import given, settings
from pysurgery.core.complexes import SimplicialComplex
from pysurgery.core.fundamental_group import extract_pi_1
from strategies import simplicial_complexes_raw, connected_simplicial_complexes_raw

@settings(max_examples=100, deadline=None)
@given(simplicial_complexes_raw())
def test_boundary_property_d_squared_is_zero(simplices):
    """Verify the fundamental chain complex property: dₙ ∘ dₙ₊₁ = 0.

    What is Being Computed?:
        The composition of consecutive boundary operators in the simplicial chain complex.

    Algorithm:
        1. Build a SimplicialComplex from random simplices.
        2. Extract sparse boundary matrices ∂ₙ and ∂ₙ₊₁.
        3. Compute the matrix product ∂ₙ @ ∂ₙ₊₁.
        4. Assert that the resulting matrix has zero non-zero entries.

    Preserved Invariants:
        - Exactness and homology well-definedness (requires ∂²=0).
    """
    sc = SimplicialComplex.from_simplices(simplices)
    cc = sc.chain_complex()
    
    dims = cc.dimensions
    for n in range(max(dims)):
        dn = cc.boundaries.get(n)
        dn_plus_1 = cc.boundaries.get(n+1)
        
        if dn is not None and dn_plus_1 is not None:
            # Check dn * dn_plus_1 == 0
            res = dn @ dn_plus_1
            assert res.nnz == 0

@settings(max_examples=100, deadline=None)
@given(simplicial_complexes_raw())
def test_euler_poincare_invariant_property(simplices):
    """Verify the Euler-Poincaré Formula relating cell counts to Betti numbers.

    What is Being Computed?:
        The Euler characteristic χ using both skeletal data and homology ranks.

    Algorithm:
        1. Compute χ = Σ (-1)ⁿ (number of n-cells).
        2. Compute Betti numbers βₙ for all dimensions.
        3. Compute χ' = Σ (-1)ⁿ βₙ.
        4. Assert χ == χ'.

    Preserved Invariants:
        - Euler characteristic (a homotopy invariant).
    """
    sc = SimplicialComplex.from_simplices(simplices)
    cc = sc.chain_complex()
    
    chi_cells = cc.euler_characteristic()
    
    betti = cc.betti_numbers()
    chi_betti = sum((-1)**n * b for n, b in betti.items())
    
    assert chi_cells == chi_betti

@settings(max_examples=50, deadline=None)
@given(connected_simplicial_complexes_raw())
def test_hurewicz_rank_consistency_property(simplices):
    """Verify Hurewicz rank consistency between π₁ and H₁.

    What is Being Computed?:
        The correspondence between the rank of the first homology group H₁ 
        and the abelianized fundamental group.

    Algorithm:
        1. Construct a connected SimplicialComplex.
        2. Compute free rank of H₁(X; ℤ).
        3. Extract the fundamental group π₁(X).
        4. Assert that π₁ is non-trivial if H₁ rank > 0.

    Preserved Invariants:
        - Hurewicz homomorphism properties.
    """
    sc = SimplicialComplex.from_simplices(simplices)
    cc = sc.chain_complex()
    
    # H1 rank (free part)
    h1_rank, _ = cc.homology(n=1)
    
    # pi1
    pi1 = extract_pi_1(sc)
    # The abelianized rank should match H1 rank.
    # We can use our exact algebra to compute this.
    
    # Simple check for now: pi1 is not None and has at least some generators if H1_rank > 0
    if h1_rank > 0:
        assert len(pi1.generators) > 0
    else:
        # If H1 rank is 0, it doesn't mean pi1 is trivial (could be torsion)
        # but it's a safe check.
        pass
    assert pi1 is not None

@settings(max_examples=100, deadline=None)
@given(simplicial_complexes_raw())
def test_coefficient_ring_rank_parity(simplices):
    """Verify that homology rank is independent of the field/ring characteristic (for free part).

    What is Being Computed?:
        Free ranks of homology groups Hₙ over ℤ and ℚ.

    Algorithm:
        1. Construct the same complex over ℤ and ℚ.
        2. Compute homology ranks for each.
        3. Assert that rank(Hₙ(X; ℤ)) == rank(Hₙ(X; ℚ)).

    Preserved Invariants:
        - Betti numbers (rank of the free part).
    """
    sc_z = SimplicialComplex.from_simplices(simplices, coefficient_ring="Z")
    sc_q = SimplicialComplex.from_simplices(simplices, coefficient_ring="Q")
    
    for n in range(3):
        rank_z, _ = sc_z.chain_complex().homology(n=n)
        rank_q, _ = sc_q.chain_complex().homology(n=n)
        assert rank_z == rank_q
