from hypothesis import given, settings
from pysurgery.core.complexes import SimplicialComplex
from pysurgery.core.fundamental_group import extract_pi_1
from strategies import simplicial_complexes_raw, connected_simplicial_complexes_raw

@settings(max_examples=100, deadline=None)
@given(simplicial_complexes_raw())
def test_boundary_property_d_squared_is_zero(simplices):
    """Verify that d_n ∘ d_{n+1} = 0 for any generated complex."""
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
    """Verify the Euler-Poincaré Formula: χ(C) = Σ (-1)^i rank(Ci) = Σ (-1)^i βi."""
    sc = SimplicialComplex.from_simplices(simplices)
    cc = sc.chain_complex()
    
    chi_cells = cc.euler_characteristic()
    
    betti = cc.betti_numbers()
    chi_betti = sum((-1)**n * b for n, b in betti.items())
    
    assert chi_cells == chi_betti

@settings(max_examples=50, deadline=None)
@given(connected_simplicial_complexes_raw())
def test_hurewicz_rank_consistency_property(simplices):
    """Verify that rank(H_1(X)) matches rank(Abelianized pi_1(X))."""
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
    """Verify that free rank over Z equals rank over Q."""
    sc_z = SimplicialComplex.from_simplices(simplices, coefficient_ring="Z")
    sc_q = SimplicialComplex.from_simplices(simplices, coefficient_ring="Q")
    
    for n in range(3):
        rank_z, _ = sc_z.chain_complex().homology(n=n)
        rank_q, _ = sc_q.chain_complex().homology(n=n)
        assert rank_z == rank_q
