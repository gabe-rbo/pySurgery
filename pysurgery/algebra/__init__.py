def __getattr__(name):
    if name == "IntersectionForm":
        from .intersection_forms import IntersectionForm  # noqa: F401
        return IntersectionForm
    if name in ["QuadraticForm", "arf_invariant_gf2"]:
        from .quadratic_forms import QuadraticForm, arf_invariant_gf2  # noqa: F401
        return locals()[name]
    if name == "GroupRingElement":
        from .group_rings import GroupRingElement  # noqa: F401
        return GroupRingElement
    if name in ["coerce_int_matrix", "normalize_word_token", "validate_group_descriptor"]:
        from .exact_algebra import (  # noqa: F401
            coerce_int_matrix,
            normalize_word_token,
            validate_group_descriptor,
        )
        return locals()[name]
    if name in ["WhiteheadGroup", "compute_whitehead_group"]:
        from .k_theory import WhiteheadGroup, compute_whitehead_group  # noqa: F401
        return locals()[name]
    if name in ["Morphism", "ExactSequence", "ShortExactSequence"]:
        from .exact_sequences import Morphism, ExactSequence, ShortExactSequence  # noqa: F401
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "IntersectionForm",
    "QuadraticForm",
    "arf_invariant_gf2",
    "GroupRingElement",
    "coerce_int_matrix",
    "normalize_word_token",
    "validate_group_descriptor",
    "WhiteheadGroup",
    "compute_whitehead_group",
    "Morphism",
    "ExactSequence",
    "ShortExactSequence",
]
