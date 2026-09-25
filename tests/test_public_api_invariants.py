"""The exact invariants ported from TabularTopology are re-exported at the top level."""
import importlib

import pytest

_REQUIRED = {
    # local homology and manifold certificates
    "certify_homology_manifold", "local_homology", "classify_simplices", "pseudomanifold_report",
    "homology_manifold_boundary", "singular_simplices", "HomologyManifoldCertificate",
    # fundamental cycles
    "fundamental_cycle", "fundamental_cycles", "top_homology_basis", "is_orientable_pseudomanifold",
    # finite spaces and strong collapses
    "FiniteSpace", "McCordCertificate", "strong_collapse", "is_strong_collapsible",
    # lower-star Morse theory
    "GradientField", "lower_star_gradient", "classify_critical_pairs", "lower_star_persistence",
    # embedded linking, enclosure, knots, link complements
    "winding_number", "simplicial_linking", "curve_linking", "linking_definedness",
    "enclosure_report", "in_convex_hull", "knot_diagram", "casson_a2", "writhe",
    "diagram_milnor_mu123", "wirtinger_presentation", "link_group", "count_homomorphisms",
    "certify_splitness", "certify_knottedness",
    # holonomy and scale
    "orientation_report", "holonomy_report", "scale_window", "federer_reach", "connectivity_radius",
    # refusals
    "NoFundamentalClassError", "UndefinedInvariantError", "NonGenericConfigurationError",
    "NotAManifoldError",
}


@pytest.mark.parametrize("name", sorted(_REQUIRED))
def test_invariant_api_re_exported(name):
    pysurgery = importlib.import_module("pysurgery")
    assert hasattr(pysurgery, name), f"pysurgery.{name} is not re-exported"


def test_new_methods_on_existing_classes():
    from pysurgery import FundamentalGroup, SimplicialComplex

    for m in ("local_homology", "certify_homology_manifold", "pseudomanifold_report",
              "face_poset", "strong_collapse", "is_strong_collapsible"):
        assert callable(getattr(SimplicialComplex, m))
    assert callable(FundamentalGroup.count_homomorphisms)
