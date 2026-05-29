def __getattr__(name):
    if name == "KirbyDiagram":
        from .kirby_calculus import KirbyDiagram  # noqa: F401
        return KirbyDiagram
    if name in ["Handle", "HandleDecomposition", "cw_complex_to_handle_decomposition"]:
        from .handle_decompositions import (  # noqa: F401
            Handle,
            HandleDecomposition,
            cw_complex_to_handle_decomposition,
        )
        return locals()[name]
    if name in [
        "RationalObstruction",
        "PLocalObstruction",
        "PrimeLocalReport",
        "compute_l_group_rational",
        "prime_local_obstruction_report",
    ]:
        from .rational_surgery import (  # noqa: F401
            RationalObstruction,
            PLocalObstruction,
            PrimeLocalReport,
            compute_l_group_rational,
            prime_local_obstruction_report,
        )
        return locals()[name]
    if name in [
        "AlgebraicSurgeryComplex",
        "perform_handle_surgery",
        "perform_algebraic_surgery",
        "perform_rational_surgery",
        "perform_p_local_surgery",
    ]:
        from .surgery import (  # noqa: F401
            AlgebraicSurgeryComplex,
            perform_handle_surgery,
            perform_algebraic_surgery,
            perform_rational_surgery,
            perform_p_local_surgery,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "KirbyDiagram",
    "Handle",
    "HandleDecomposition",
    "cw_complex_to_handle_decomposition",
    "RationalObstruction",
    "PLocalObstruction",
    "PrimeLocalReport",
    "compute_l_group_rational",
    "prime_local_obstruction_report",
    "AlgebraicSurgeryComplex",
    "perform_handle_surgery",
    "perform_algebraic_surgery",
    "perform_rational_surgery",
    "perform_p_local_surgery",
]
