"""Property-based tests for structural integrity of simplicial complex constructions.

Overview:
    This suite validates that complexes generated from point clouds (Alpha, CkNN, 
    and Vietoris-Rips) maintain skeletal closure and structural validity across 
    diverse parameter spaces and input configurations.

Key Concepts:
    - **Skeletal Closure**: Ensuring all faces of a simplex are present in the complex.
    - **Structural Validity**: Absence of orphans or inconsistent boundary mappings.
    - **Construction Heuristics**: Algorithms for building complexes from metric data.
"""

from hypothesis import given, settings
from pysurgery.core.complexes import SimplicialComplex
from strategies import point_clouds

@settings(max_examples=50, deadline=None)
@given(point_clouds(min_pts=5, max_pts=20))
def test_alpha_complex_properties_property(points):
    """Verify structural integrity of Alpha Complexes.

    What is Being Computed?:
        The Alpha Complex for a point cloud at multiple filtration values (α²).

    Algorithm:
        1. Build Alpha Complexes for α² ∈ {0.5, 1.0, 2.0}.
        2. Verify skeletal closure (all faces included).
        3. Validate the underlying sparse boundary structure.

    Preserved Invariants:
        - Downward closure (a fundamental requirement for any simplicial complex).
    """
    # Test across a few alpha values
    for alpha2 in [0.5, 1.0, 2.0]:
        sc = SimplicialComplex.from_alpha_complex(points, max_alpha_square=alpha2)
        validity = sc.verify_structure()
        assert validity["valid"], f"Alpha complex invalid for alpha2={alpha2}: {validity['issues']}"
        assert validity["is_closed"], "Alpha complex not closed under faces"

@settings(max_examples=50, deadline=None)
@given(point_clouds(min_pts=6, max_pts=15))
def test_cknn_complex_properties_property(points):
    """Verify structural integrity of Continuous k-Nearest Neighbor (CkNN) complexes.

    What is Being Computed?:
        CkNN complexes which provide adaptive density-aware connectivity.

    Algorithm:
        1. Construct CkNN complex with k=3 and δ=1.5.
        2. Perform a full structural audit via verify_structure().
        3. Assert validity and closure.

    Preserved Invariants:
        - Skeletal closure and topological consistency.
    """
    # Parameters k and delta
    sc = SimplicialComplex.from_point_cloud_cknn(points, k=3, delta=1.5)
    validity = sc.verify_structure()
    assert validity["valid"], f"CkNN complex invalid: {validity['issues']}"
    assert validity["is_closed"], "CkNN complex not closed under faces"

@settings(max_examples=50, deadline=None)
@given(point_clouds(min_pts=4, max_pts=12))
def test_rips_complex_properties_property(points):
    """Verify structural integrity of Vietoris-Rips complexes.

    What is Being Computed?:
        Vietoris-Rips complexes defined by a global proximity parameter (ε).

    Algorithm:
        1. Construct Rips complex with ε=2.0 up to dimension 2.
        2. Run structural verification suite.
        3. Confirm the complex is downward closed.

    Preserved Invariants:
        - Downward closure.
    """
    # Correct method: from_vietoris_rips
    sc = SimplicialComplex.from_vietoris_rips(points, epsilon=2.0, max_dimension=2)
    validity = sc.verify_structure()
    assert validity["valid"], f"Rips complex invalid: {validity['issues']}"
    assert validity["is_closed"], "Rips complex not closed under faces"
