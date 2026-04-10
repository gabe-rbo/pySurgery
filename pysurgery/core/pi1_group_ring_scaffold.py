from __future__ import annotations

from pydantic import BaseModel, Field

from .exact_algebra import validate_group_descriptor
from .fundamental_group import FundamentalGroup, infer_standard_group_descriptor


class Pi1Evidence(BaseModel):
    generators: list[str] = Field(default_factory=list)
    relation_count: int = 0
    inferred_descriptor: str | None = None


class GroupRingContext(BaseModel):
    descriptor: str
    valid_descriptor: bool
    descriptor_message: str
    family: str


class Phase2Readiness(BaseModel):
    ready: bool
    gaps: list[str] = Field(default_factory=list)
    pi1_evidence: Pi1Evidence | None = None
    group_ring_context: GroupRingContext | None = None


def _descriptor_family(descriptor: str) -> str:
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


def build_pi1_evidence(pi1: FundamentalGroup | None) -> Pi1Evidence | None:
    if pi1 is None:
        return None
    return Pi1Evidence(
        generators=list(pi1.generators),
        relation_count=len(pi1.relations),
        inferred_descriptor=infer_standard_group_descriptor(pi1),
    )


def build_group_ring_context(descriptor: str | None) -> GroupRingContext | None:
    if descriptor is None:
        return None
    ok, msg = validate_group_descriptor(descriptor)
    return GroupRingContext(
        descriptor=str(descriptor).strip(),
        valid_descriptor=ok,
        descriptor_message=msg,
        family=_descriptor_family(str(descriptor)),
    )


def evaluate_phase2_readiness(pi1: FundamentalGroup | None, descriptor: str | None) -> Phase2Readiness:
    pi1_evidence = build_pi1_evidence(pi1)
    context = build_group_ring_context(descriptor)

    gaps: list[str] = []
    if context is None:
        gaps.append("Missing pi_1 group descriptor")
    elif not context.valid_descriptor:
        gaps.append(f"Descriptor not supported by current grammar ({context.descriptor_message})")

    if pi1 is not None and pi1_evidence is not None and pi1_evidence.inferred_descriptor is None and descriptor is None:
        gaps.append("Could not infer a conservative standard descriptor from pi_1 presentation")

    if context is not None and context.family not in {"trivial", "infinite_cyclic", "finite_cyclic", "product"}:
        gaps.append("Descriptor family lacks phase-2 computational support")

    return Phase2Readiness(
        ready=(len(gaps) == 0),
        gaps=gaps,
        pi1_evidence=pi1_evidence,
        group_ring_context=context,
    )

