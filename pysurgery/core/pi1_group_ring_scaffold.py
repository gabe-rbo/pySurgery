from __future__ import annotations

from pydantic import BaseModel, Field

from .exact_algebra import validate_group_descriptor
from .fundamental_group import FundamentalGroup, infer_standard_group_descriptor


class Pi1Evidence(BaseModel):
    """Evidence for fundamental group properties.

    Attributes:
        generators (list[str]): List of generator names.
        relation_count (int): Number of relations in the presentation.
        orientation_character (dict[str, int]): Map from generator names to {1, -1}.
        inferred_descriptor (str | None): Inferred standard group descriptor.
    """
    generators: list[str] = Field(default_factory=list)
    relation_count: int = 0
    orientation_character: dict[str, int] = Field(default_factory=dict)
    inferred_descriptor: str | None = None


class GroupRingContext(BaseModel):
    """Context for group ring computations.

    Attributes:
        descriptor (str): The group descriptor string.
        valid_descriptor (bool): Whether the descriptor is valid.
        descriptor_message (str): Status message from descriptor validation.
        family (str): The classified group family.
    """
    descriptor: str
    valid_descriptor: bool
    descriptor_message: str
    family: str


class Phase2Readiness(BaseModel):
    """Readiness status for phase 2 computations.

    Attributes:
        ready (bool): Whether the inputs are ready for phase 2.
        gaps (list[str]): List of missing requirements or gaps.
        pi1_evidence (Pi1Evidence | None): Evidence from the fundamental group.
        group_ring_context (GroupRingContext | None): Group ring context metadata.
    """
    ready: bool
    gaps: list[str] = Field(default_factory=list)
    pi1_evidence: Pi1Evidence | None = None
    group_ring_context: GroupRingContext | None = None


def _descriptor_family(descriptor: str) -> str:
    """Classify descriptor into broad computational family buckets.

    Args:
        descriptor (str): The group descriptor string.

    Returns:
        str: The name of the group family.
    """
    d = descriptor.strip()
    if d == "1":
        return "trivial"
    if d == "Z":
        return "infinite_cyclic"
    if d.startswith("Z_"):
        return "finite_cyclic"
    if "x" in d.lower():
        return "product"
    return "generic"


def build_pi1_evidence(
    pi1: FundamentalGroup | None, backend: str = "auto"
) -> Pi1Evidence | None:
    """Build pi1 evidence payload for readiness diagnostics.

    Args:
        pi1 (FundamentalGroup | None): The fundamental group to analyze.
        backend: 'auto', 'julia', or 'python'.

    Returns:
        Pi1Evidence | None: The evidence payload, or None if pi1 is None.
    """
    if pi1 is None:
        return None
    return Pi1Evidence(
        generators=list(pi1.generators),
        relation_count=len(pi1.relations),
        orientation_character=pi1.orientation_character,
        inferred_descriptor=infer_standard_group_descriptor(pi1),
    )


def build_group_ring_context(descriptor: str | None) -> GroupRingContext | None:
    """Validate descriptor and package group-ring context metadata.

    Args:
        descriptor (str | None): The group descriptor string.

    Returns:
        GroupRingContext | None: The context metadata, or None if descriptor is None.
    """
    if descriptor is None:
        return None
    ok, msg = validate_group_descriptor(descriptor)
    return GroupRingContext(
        descriptor=str(descriptor).strip(),
        valid_descriptor=ok,
        descriptor_message=msg,
        family=_descriptor_family(str(descriptor)),
    )


def evaluate_phase2_readiness(
    pi1: FundamentalGroup | None, descriptor: str | None, backend: str = "auto"
) -> Phase2Readiness:
    """Evaluate whether inputs satisfy current phase-2 group-ring prerequisites.

    Args:
        pi1 (FundamentalGroup | None): The fundamental group.
        descriptor (str | None): The group descriptor string.
        backend: 'auto', 'julia', or 'python'.

    Returns:
        Phase2Readiness: The readiness evaluation result.
    """
    pi1_evidence = build_pi1_evidence(pi1, backend=backend)
    context = build_group_ring_context(descriptor)

    gaps: list[str] = []
    if context is None:
        gaps.append("Missing pi_1 group descriptor")
    elif not context.valid_descriptor:
        gaps.append(
            f"Descriptor not supported by current grammar ({context.descriptor_message})"
        )

    if (
        pi1 is not None
        and pi1_evidence is not None
        and pi1_evidence.inferred_descriptor is None
        and descriptor is None
    ):
        gaps.append(
            "Could not infer a conservative standard descriptor from pi_1 presentation"
        )

    if context is not None and context.family not in {
        "trivial",
        "infinite_cyclic",
        "finite_cyclic",
        "product",
    }:
        gaps.append("Descriptor family lacks phase-2 computational support")

    return Phase2Readiness(
        ready=(len(gaps) == 0),
        gaps=gaps,
        pi1_evidence=pi1_evidence,
        group_ring_context=context,
    )
