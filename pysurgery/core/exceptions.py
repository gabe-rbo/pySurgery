"""Custom exceptions for surgery theory operations.

Overview:
    Provides a hierarchy of domain-specific exceptions for topological and algebraic
    surgery theory computations. These errors capture specific failure modes such
    as non-unimodularity, dimension mismatches, and surgery obstructions.

Key Concepts:
    - **SurgeryError**: The base class for all library-specific exceptions.
    - **Topological Invariants**: Errors often triggered when expected invariants
      (like unimodularity) are violated.
    - **Algebraic Obstructions**: Errors capturing non-zero elements in Wall groups (L-theory).

Common Workflows:
    1. Catch specific errors (e.g., `IsotropicError`) to handle geometry-specific failures.
    2. Use `SurgeryObstructionError` to identify when surgery cannot be completed.
    3. Monitor `DimensionError` for topologically illegal inputs.

Coefficient Ring:
    Not applicable (Exception hierarchy).

Attributes:
    None (uses standard Exception message).
"""


class SurgeryError(Exception):
    """Base class for all surgery theory related errors.

    Overview:
        The root exception for all errors occurring within the `pysurgery` library.
        Used to distinguish library-specific failures from general Python errors.

    Key Concepts:
        - Domain-specific error handling.
        - Hierarchical categorization of topological failures.

    Common Workflows:
        1. Base class for all specialized exceptions in this module.

    Coefficient Ring:
        N/A.

    Attributes:
        message (str): A human-readable error message.
    """

    pass


class IsotropicError(SurgeryError):
    """Raised when a class is not isotropic (self-intersection is non-zero).

    Overview:
        Indicates that an algebraic homology class does not vanish under the
        intersection form, violating an isotropy requirement for specific embeddings.

    Key Concepts:
        - **Isotropy**: Vanishing of the self-intersection form.

    Common Workflows:
        1. Raised during immersion-to-embedding reduction (Whitney trick pre-checks).

    Coefficient Ring:
        Usually ℤ or ℤ/2ℤ.

    Attributes:
        message (str): A human-readable error message.
    """

    pass


class NonPrimitiveError(SurgeryError):
    """Raised when a homology class is not primitive (not a basis element).

    Overview:
        Indicates that a homology class cannot be completed to a basis,
        meaning it is a multiple of another non-trivial class.

    Key Concepts:
        - **Primitivity**: Element is part of a ℤ-module basis.

    Common Workflows:
        1. Raised during surgery on cycles that are not primitive.

    Coefficient Ring:
        ℤ.

    Attributes:
        message (str): A human-readable error message.
    """

    pass


class UnimodularityError(SurgeryError):
    """Raised when a form is expected to be unimodular but is not.

    Overview:
        Indicates that an intersection or quadratic form has a determinant
        other than ±1 (over ℤ), violating Poincare duality requirements.

    Key Concepts:
        - **Unimodularity**: Map from H_n to H^n is an isomorphism.

    Common Workflows:
        1. Raised during validation of intersection forms for closed manifolds.

    Coefficient Ring:
        ℤ.

    Attributes:
        message (str): A human-readable error message.
    """

    pass


class DimensionError(SurgeryError):
    """Raised when dimensions are topologically incompatible or invalid.

    Overview:
        Indicates a mismatch in dimensions, such as attempting a surgery operation
        on a manifold of the wrong dimension or invalid simplex dimension.

    Key Concepts:
        - **Topological Dimension**: The n in n-manifold.

    Common Workflows:
        1. Raised during manifold construction or surgery obstructions.

    Coefficient Ring:
        N/A.

    Attributes:
        message (str): A human-readable error message.
    """

    pass


class HomologyError(SurgeryError):
    """Raised when homology calculations fail or are inconsistent.

    Overview:
        Indicates an error in the chain complex or Smith Normal Form reduction,
        resulting in inconsistent or invalid homology groups.

    Key Concepts:
        - **Chain Complex**: Sequence of modules and boundary maps.

    Common Workflows:
        1. Raised during `homology()` calls on invalid complexes.

    Coefficient Ring:
        ℤ, ℚ, or ℤ/pℤ.

    Attributes:
        message (str): A human-readable error message.
    """

    pass


class SurgeryObstructionError(SurgeryError):
    """Raised when a non-zero Wall group obstruction prevents surgery.

    Overview:
        Captures a failure in the surgery program: the existence of a non-trivial
        element in L_n(π₁) that prevents the existence of a homotopy equivalence.

    Key Concepts:
        - **Wall Group (L_n)**: Obstruction group for surgery.

    Common Workflows:
        1. Raised when `solve_surgery_problem()` finds a non-zero L-group element.

    Coefficient Ring:
        Group rings ℤ[π₁].

    Attributes:
        message (str): A human-readable error message.
    """

    pass


class NonSymmetricError(SurgeryError):
    """Raised when an intersection form matrix is not symmetric.

    Overview:
        Indicates that an even-dimensional intersection form is not symmetric,
        violating the basic properties of the cup product on H^k.

    Key Concepts:
        - **Symmetry**: A_ij = A_ji in the intersection matrix.

    Common Workflows:
        1. Raised during `IntersectionForm` validation.

    Coefficient Ring:
        ℤ.

    Attributes:
        message (str): A human-readable error message.
    """

    pass


class GroupRingError(SurgeryError):
    """Raised when group ring operations are invalid.

    Overview:
        Indicates failure in algebraic operations within a group ring ℤ[G],
        such as invalid generator multiplication or malformed elements.

    Key Concepts:
        - **Group Ring**: Algebraic structure combining a ring and a group.

    Common Workflows:
        1. Raised during Wall group element construction.

    Coefficient Ring:
        ℤ[G].

    Attributes:
        message (str): A human-readable error message.
    """

    pass


class FundamentalGroupError(SurgeryError):
    """Raised when the fundamental group extraction fails or is ambiguous.

    Overview:
        Indicates failures in extracting π₁ from a 2-skeleton or in applying
        Tietze simplifications to a group presentation.

    Key Concepts:
        - **π₁ (Fundamental Group)**: First homotopy group.

    Common Workflows:
        1. Raised during `extract_pi_1()`.

    Coefficient Ring:
        N/A.

    Attributes:
        message (str): A human-readable error message.
    """

    pass


class KirbyMoveError(SurgeryError):
    """Raised when an invalid Kirby move is attempted (e.g., sliding handles incorrectly).

    Overview:
        Indicates a violation of the rules of Kirby calculus for 3- and 4-manifolds,
        such as an illegal handle slide or blowout.

    Key Concepts:
        - **Kirby Calculus**: Set of moves (slides, stabilizes) on framed links.

    Common Workflows:
        1. Raised during manual or automated link manipulation.

    Coefficient Ring:
        N/A.

    Attributes:
        message (str): A human-readable error message.
    """

    pass


class CharacteristicClassError(SurgeryError):
    """Raised when Wu's formula or Steenrod squares fail on non-mod-2 cohomology.

    Overview:
        Indicates that a characteristic class computation (like Stiefel-Whitney)
        was attempted on an incompatible cohomology theory or ring.

    Key Concepts:
        - **Characteristic Classes**: Topological invariants of vector bundles.

    Common Workflows:
        1. Raised during SW class extraction from non-Z2 complexes.

    Coefficient Ring:
        ℤ/2ℤ (usually).

    Attributes:
        message (str): A human-readable error message.
    """

    pass


class StructureSetError(SurgeryError):
    """Raised when the Surgery Exact Sequence cannot be resolved due to missing L-group data.

    Overview:
        Indicates failure in resolving the structure set S(M), typically when
        the required Wall groups or assembly maps are not computable.

    Key Concepts:
        - **Structure Set**: Set of homotopy equivalent manifolds.

    Common Workflows:
        1. Raised during `StructureSet.resolve()` pipeline.

    Coefficient Ring:
        L-theory modules.

    Attributes:
        message (str): A human-readable error message.
    """

    pass


class MathError(SurgeryError):
    """Raised when general mathematical or geometric consistency checks fail.

    Overview:
        A general-purpose error for mathematical inconsistencies that do not
        fit more specific categories like `HomologyError` or `DimensionError`.

    Key Concepts:
        - Mathematical consistency.

    Common Workflows:
        1. Raised during internal assertions in algebraic surgery engines.

    Coefficient Ring:
        Any.

    Attributes:
        message (str): A human-readable error message.
    """

    pass
