from pysurgery.core.fundamental_group import FundamentalGroup
from pysurgery.core.pi1_group_ring_scaffold import evaluate_phase2_readiness


def test_phase2_readiness_reports_ready_for_trivial_descriptor():
    pi1 = FundamentalGroup(generators=[], relations=[])
    readiness = evaluate_phase2_readiness(pi1=pi1, descriptor="1")
    assert readiness.ready
    assert readiness.group_ring_context is not None
    assert readiness.group_ring_context.family == "trivial"


def test_phase2_readiness_reports_gaps_for_missing_descriptor():
    pi1 = FundamentalGroup(generators=["a", "b"], relations=[])
    readiness = evaluate_phase2_readiness(pi1=pi1, descriptor=None)
    assert not readiness.ready
    assert any("descriptor" in g.lower() for g in readiness.gaps)


def test_phase2_readiness_accepts_supported_product_descriptor():
    pi1 = FundamentalGroup(generators=["a"], relations=[])
    readiness = evaluate_phase2_readiness(pi1=pi1, descriptor="Z x Z_3")
    assert readiness.ready
    assert readiness.group_ring_context is not None
    assert readiness.group_ring_context.family == "product"


