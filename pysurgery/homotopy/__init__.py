def __getattr__(name):
    if name in [
        "RationalDGA",
        "RationalCohomologyAlgebra",
        "RationalMinimalModelResult",
        "RationalHomotopyGroup",
        "FormalityResult",
        "MasseyProductsResult",
        "sullivan_minimal_model",
        "is_formal_space",
        "extract_massey_products",
        "rational_homotopy_group",
    ]:
        from .rational_homotopy import (  # noqa: F401
            RationalDGA,
            RationalCohomologyAlgebra,
            RationalMinimalModelResult,
            RationalHomotopyGroup,
            FormalityResult,
            MasseyProductsResult,
            sullivan_minimal_model,
            is_formal_space,
            extract_massey_products,
            rational_homotopy_group,
        )
        return locals()[name]
    if name in [
        "HomotopyGroup",
        "HomotopyGroupApproximation",
        "compute_rational_and_adams",
        "synthesize_homotopy_group_with_e_infinity",
    ]:
        from .higher_homotopy_groups import (  # noqa: F401
            HomotopyGroup,
            HomotopyGroupApproximation,
            compute_rational_and_adams,
            synthesize_homotopy_group_with_e_infinity,
        )
        return locals()[name]
    if name in [
        "RationalHomotopyGroupAtDegree",
        "RationalHomotopyProfile",
        "SullivanIntegrationError",
        "sullivan_rational_homotopy",
        "cross_validate_with_serre",
    ]:
        from .sullivan_models import (  # noqa: F401
            RationalHomotopyGroupAtDegree,
            RationalHomotopyProfile,
            SullivanIntegrationError,
            sullivan_rational_homotopy,
            cross_validate_with_serre,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "RationalDGA",
    "RationalCohomologyAlgebra",
    "RationalMinimalModelResult",
    "RationalHomotopyGroup",
    "FormalityResult",
    "MasseyProductsResult",
    "sullivan_minimal_model",
    "is_formal_space",
    "extract_massey_products",
    "rational_homotopy_group",
    "HomotopyGroup",
    "HomotopyGroupApproximation",
    "compute_rational_and_adams",
    "synthesize_homotopy_group_with_e_infinity",
    "RationalHomotopyGroupAtDegree",
    "RationalHomotopyProfile",
    "SullivanIntegrationError",
    "sullivan_rational_homotopy",
    "cross_validate_with_serre",
]
