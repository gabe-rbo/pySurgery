"""Custom exceptions for surgery theory operations."""

class SurgeryError(Exception):
    """Base class for all surgery theory related errors.

    Args:
        message: A human-readable error message.
    """

    pass


class IsotropicError(SurgeryError):
    """Raised when a class is not isotropic (self-intersection is non-zero).

    Args:
        message: A human-readable error message.
    """

    pass


class NonPrimitiveError(SurgeryError):
    """Raised when a homology class is not primitive (not a basis element).

    Args:
        message: A human-readable error message.
    """

    pass


class UnimodularityError(SurgeryError):
    """Raised when a form is expected to be unimodular but is not.

    Args:
        message: A human-readable error message.
    """

    pass


class DimensionError(SurgeryError):
    """Raised when dimensions are topologically incompatible or invalid.

    Args:
        message: A human-readable error message.
    """

    pass


class HomologyError(SurgeryError):
    """Raised when homology calculations fail or are inconsistent.

    Args:
        message: A human-readable error message.
    """

    pass


class SurgeryObstructionError(SurgeryError):
    """Raised when a non-zero Wall group obstruction prevents surgery.

    Args:
        message: A human-readable error message.
    """

    pass


class NonSymmetricError(SurgeryError):
    """Raised when an intersection form matrix is not symmetric.

    Args:
        message: A human-readable error message.
    """

    pass


class GroupRingError(SurgeryError):
    """Raised when group ring operations are invalid.

    Args:
        message: A human-readable error message.
    """

    pass


class FundamentalGroupError(SurgeryError):
    """Raised when the fundamental group extraction fails or is ambiguous.

    Args:
        message: A human-readable error message.
    """

    pass


class KirbyMoveError(SurgeryError):
    """Raised when an invalid Kirby move is attempted (e.g., sliding handles incorrectly).

    Args:
        message: A human-readable error message.
    """

    pass


class CharacteristicClassError(SurgeryError):
    """Raised when Wu's formula or Steenrod squares fail on non-mod-2 cohomology.

    Args:
        message: A human-readable error message.
    """

    pass


class StructureSetError(SurgeryError):
    """Raised when the Surgery Exact Sequence cannot be resolved due to missing L-group data.

    Args:
        message: A human-readable error message.
    """

    pass


class MathError(SurgeryError):
    """Raised when general mathematical or geometric consistency checks fail.

    Args:
        message: A human-readable error message.
    """

    pass
