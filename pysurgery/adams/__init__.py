def __getattr__(name):
    if name in ["SteenrodAlgebra", "AdamsE2Page", "adams_e2_page"]:
        from .spectral_sequence import (  # noqa: F401
            SteenrodAlgebra,
            AdamsE2Page,
            adams_e2_page,
        )
        return locals()[name]
    if name in ["ConvergedAdamsPage", "UserVerifiedDifferential"]:
        from .e_infinity_resolver import (  # noqa: F401
            ConvergedAdamsPage,
            UserVerifiedDifferential,
        )
        return locals()[name]
    if name == "InteractiveAdamsResolver":
        from .interactive_resolver import InteractiveAdamsResolver  # noqa: F401
        return InteractiveAdamsResolver
    if name in ["LeanFormalAdamsResolver", "LeanProofAttempt"]:
        from .lean_resolver import LeanFormalAdamsResolver, LeanProofAttempt  # noqa: F401
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "SteenrodAlgebra",
    "AdamsE2Page",
    "adams_e2_page",
    "ConvergedAdamsPage",
    "UserVerifiedDifferential",
    "InteractiveAdamsResolver",
    "LeanFormalAdamsResolver",
    "LeanProofAttempt",
]
