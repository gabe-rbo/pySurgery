"""Tests for E∞ Resolution, Database, and Upper Bounds.

Overview:
    Validates the iterative resolution logic, local database lookups, 
    and greedy upper-bound reporting for the Adams spectral sequence.

Key Concepts:
    - Database hit: resolver automatically finds d2(h4) in sphere_p2.json.
    - Upper bound: HomotopyGroupApproximation reports potential torsion.
    - Stabilization: resolver loop exits when grid is unchanged.
"""
from pysurgery.homotopy.higher_homotopy_groups import HomotopyGroup
from pysurgery.adams.spectral_sequence import sphere_cohomology_fp

def test_database_lookup_sphere_p2():
    """Verify that differentials are resolved via database when possible."""
    # Using a small window to minimize other ambiguous flags
    hg = HomotopyGroup.from_inputs(sphere_cohomology_fp(0, prime=2), adams_s_max=3, adams_t_max=10)
    
    class MockIO:
        def write(self, msg): pass
        def prompt(self, msg, choices): return "zero" # Fallback if DB fails
        def confirm(self, msg): return True

    result = hg.resolve(7, path="interactive", interactive_kwargs={"cli_io": MockIO()})
    
    # We check if at least some database entries were used
    db_verifications = [v for v in result.supporting_e_infinity.user_verifications if v.user_id == "database"]
    assert len(db_verifications) > 0
    assert result.confidence_score > 0.5

def test_upper_bound_torsion_reporting():
    """Verify that the approximation includes the greedy upper bound."""
    hg = HomotopyGroup.from_inputs(sphere_cohomology_fp(3, prime=2))
    
    # Skip resolution
    result = hg.resolve(3, path="rational_only")
    
    assert result.path_used == "rational_only"
    assert result.upper_bound_torsion is not None
    assert len(result.upper_bound_torsion) >= 1

def test_stabilization_check():
    """Verify that the resolver detects grid stabilization."""
    hg = HomotopyGroup.from_inputs(sphere_cohomology_fp(3, prime=2), adams_s_max=4)
    
    class MockIO:
        def write(self, msg): pass
        def prompt(self, msg, choices): return "zero"
        def confirm(self, msg): return True

    # If all differentials are zero (or resolved), it should stabilize
    result = hg.resolve(3, path="interactive", interactive_kwargs={"cli_io": MockIO()})
    assert result.supporting_e_infinity.status == "success"
    assert result.supporting_e_infinity.convergence_page >= 2

def test_homotopy_group_approximation_metadata():
    """Verify metadata propagation in the approximation contract."""
    hg = HomotopyGroup.from_inputs(sphere_cohomology_fp(3, prime=2), space_label="S3")
    result = hg.resolve(3, path="rational_only")
    
    assert result.space_label == "S3"
    assert result.prime == 2
    assert result.n == 3
