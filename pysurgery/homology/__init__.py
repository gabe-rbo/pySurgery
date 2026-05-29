def __getattr__(name):
    if name == "AlgebraicPoincareComplex":
        from .algebraic_poincare import AlgebraicPoincareComplex  # noqa: F401
        return AlgebraicPoincareComplex
    if name in [
        "hk_generators_z",
        "compute_homology_basis_from_complex",
        "compute_homology_basis_from_simplices",
        "compute_optimal_h1_basis_from_complex",
        "compute_optimal_h1_basis_from_simplices",
        "greedy_h1_basis",
        "generator_cycles_from_simplices",
        "annot_edge",
    ]:
        from .homology_generators import (  # noqa: F401
            hk_generators_z,
            compute_homology_basis_from_complex,
            compute_homology_basis_from_simplices,
            compute_optimal_h1_basis_from_complex,
            compute_optimal_h1_basis_from_simplices,
            greedy_h1_basis,
            generator_cycles_from_simplices,
            annot_edge,
        )
        return locals()[name]
    if name in [
        "FiniteGroupOrderResult",
        "UniversalCoverResult",
        "TwistedChainResult",
        "ControlledCohomologyResult",
        "TwistedIntersectionFormResult",
        "TwistedObstructionResult",
        "FiniteGroupRing",
        "TwistedRepresentation",
        "UniversalCover",
        "TwistedChainComplex",
        "compute_controlled_cohomology",
        "compute_twisted_intersection_form",
        "compute_twisted_obstruction",
    ]:
        from .controlled_cohomology import (  # noqa: F401
            FiniteGroupOrderResult,
            UniversalCoverResult,
            TwistedChainResult,
            ControlledCohomologyResult,
            TwistedIntersectionFormResult,
            TwistedObstructionResult,
            FiniteGroupRing,
            TwistedRepresentation,
            UniversalCover,
            TwistedChainComplex,
            compute_controlled_cohomology,
            compute_twisted_intersection_form,
            compute_twisted_obstruction,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AlgebraicPoincareComplex",
    "hk_generators_z",
    "compute_homology_basis_from_complex",
    "compute_homology_basis_from_simplices",
    "compute_optimal_h1_basis_from_complex",
    "compute_optimal_h1_basis_from_simplices",
    "greedy_h1_basis",
    "generator_cycles_from_simplices",
    "annot_edge",
    "FiniteGroupOrderResult",
    "UniversalCoverResult",
    "TwistedChainResult",
    "ControlledCohomologyResult",
    "TwistedIntersectionFormResult",
    "TwistedObstructionResult",
    "FiniteGroupRing",
    "TwistedRepresentation",
    "UniversalCover",
    "TwistedChainComplex",
    "compute_controlled_cohomology",
    "compute_twisted_intersection_form",
    "compute_twisted_obstruction",
]
