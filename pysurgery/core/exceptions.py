class SurgeryError(Exception):
    """Base class for all surgery theory related errors."""

    pass


class IsotropicError(SurgeryError):
    """Raised when a class is not isotropic (self-intersection is non-zero)."""

    pass


class NonPrimitiveError(SurgeryError):
    """Raised when a homology class is not primitive (not a basis element)."""

    pass


class UnimodularityError(SurgeryError):
    """Raised when a form is expected to be unimodular but is not."""

    pass


class DimensionError(SurgeryError):
    """Raised when dimensions are topologically incompatible or invalid."""

    pass


class HomologyError(SurgeryError):
    """Raised when homology calculations fail or are inconsistent."""

    pass


class SurgeryObstructionError(SurgeryError):
    """Raised when a non-zero Wall group obstruction prevents surgery."""

    pass


class NonSymmetricError(SurgeryError):
    """Raised when an intersection form matrix is not symmetric."""

    pass


class GroupRingError(SurgeryError):
    """Raised when group ring operations are invalid."""

    pass


class FundamentalGroupError(SurgeryError):
    """Raised when the fundamental group extraction fails or is ambiguous."""

    pass


class KirbyMoveError(SurgeryError):
    """Raised when an invalid Kirby move is attempted (e.g., sliding handles incorrectly)."""

    pass


class CharacteristicClassError(SurgeryError):
    """Raised when Wu's formula or Steenrod squares fail on non-mod-2 cohomology."""

    pass


class StructureSetError(SurgeryError):
    """Raised when the Surgery Exact Sequence cannot be resolved due to missing L-group data."""

    pass
