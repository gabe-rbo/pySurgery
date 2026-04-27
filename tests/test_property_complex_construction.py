from hypothesis import given, settings
from pysurgery.core.complexes import SimplicialComplex
from strategies import point_clouds

@settings(max_examples=50, deadline=None)
@given(point_clouds(min_pts=5, max_pts=20))
def test_alpha_complex_properties_property(points):
    """Verify that Alpha Complex is valid and downward closed."""
    # Test across a few alpha values
    for alpha2 in [0.5, 1.0, 2.0]:
        sc = SimplicialComplex.from_alpha_complex(points, max_alpha_square=alpha2)
        validity = sc.verify_structure()
        assert validity["valid"], f"Alpha complex invalid for alpha2={alpha2}: {validity['issues']}"
        assert validity["is_closed"], "Alpha complex not closed under faces"

@settings(max_examples=50, deadline=None)
@given(point_clouds(min_pts=6, max_pts=15))
def test_cknn_complex_properties_property(points):
    """Verify that CkNN complex is valid and downward closed."""
    # Parameters k and delta
    sc = SimplicialComplex.from_point_cloud_cknn(points, k=3, delta=1.5)
    validity = sc.verify_structure()
    assert validity["valid"], f"CkNN complex invalid: {validity['issues']}"
    assert validity["is_closed"], "CkNN complex not closed under faces"

@settings(max_examples=50, deadline=None)
@given(point_clouds(min_pts=4, max_pts=12))
def test_rips_complex_properties_property(points):
    """Verify that Vietoris-Rips complex is valid and downward closed."""
    # Correct method: from_vietoris_rips
    sc = SimplicialComplex.from_vietoris_rips(points, epsilon=2.0, max_dimension=2)
    validity = sc.verify_structure()
    assert validity["valid"], f"Rips complex invalid: {validity['issues']}"
    assert validity["is_closed"], "Rips complex not closed under faces"
