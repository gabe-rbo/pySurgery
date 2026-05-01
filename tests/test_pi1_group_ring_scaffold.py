"""Tests for π₁ group ring scaffold and phase 2 readiness evaluation.

Overview:
    This suite tests the logic for determining if a fundamental group and its
    associated descriptor are ready for group ring operations (Phase 2 surgery).
    It validates family identification (trivial, product, etc.) and gap reporting.

Key Concepts:
    - **Phase 2 Readiness**: Criteria for advancing to group ring-based surgery obstructions.
    - **Group Ring Context**: Metadata describing the structure of ℤ[π₁].
    - **Descriptor**: A string representation (e.g., 'Z x Z_3') of the group family.
"""

from pysurgery.core.fundamental_group import FundamentalGroup
from pysurgery.core.pi1_group_ring_scaffold import evaluate_phase2_readiness


def test_phase2_readiness_reports_ready_for_trivial_descriptor():
    """Verify readiness for a trivial fundamental group.

    What is Being Computed?:
        Readiness state and group ring context for π₁ = {1}.

    Algorithm:
        1. Initialize trivial FundamentalGroup.
        2. Evaluate readiness with descriptor "1".
        3. Assert ready=True and family='trivial'.
    """
    pi1 = FundamentalGroup(generators=[], relations=[])
    readiness = evaluate_phase2_readiness(pi1=pi1, descriptor="1")
    assert readiness.ready
    assert readiness.group_ring_context is not None
    assert readiness.group_ring_context.family == "trivial"


def test_phase2_readiness_reports_gaps_for_missing_descriptor():
    """Ensure gaps are reported when a descriptor is missing for a non-trivial group.

    What is Being Computed?:
        Gap analysis for incomplete group ring configuration.

    Algorithm:
        1. Initialize non-trivial FundamentalGroup.
        2. Evaluate readiness with descriptor=None.
        3. Assert ready=False and check for 'descriptor' in gaps.
    """
    pi1 = FundamentalGroup(generators=["a", "b"], relations=[])
    readiness = evaluate_phase2_readiness(pi1=pi1, descriptor=None)
    assert not readiness.ready
    assert any("descriptor" in g.lower() for g in readiness.gaps)


def test_phase2_readiness_accepts_supported_product_descriptor():
    """Verify readiness for supported product group descriptors.

    What is Being Computed?:
        Readiness state for product group families (e.g., ℤ × ℤ₃).

    Algorithm:
        1. Initialize FundamentalGroup with one generator.
        2. Evaluate readiness with descriptor "Z x Z_3".
        3. Assert ready=True and family='product'.
    """
    pi1 = FundamentalGroup(generators=["a"], relations=[])
    readiness = evaluate_phase2_readiness(pi1=pi1, descriptor="Z x Z_3")
    assert readiness.ready
    assert readiness.group_ring_context is not None
    assert readiness.group_ring_context.family == "product"
