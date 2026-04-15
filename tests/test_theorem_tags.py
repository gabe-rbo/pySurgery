from pysurgery.homeomorphism import HomeomorphismResult
from pysurgery.core.theorem_tags import infer_theorem_tag


def test_infer_theorem_tag_known_label():
    tag = infer_theorem_tag("Freedman classification")
    assert tag == "4d.freedman.simply_connected"


def test_homeomorphism_result_auto_sets_theorem_tag():
    out = HomeomorphismResult(
        status="inconclusive",
        is_homeomorphic=None,
        reasoning="INCONCLUSIVE",
        theorem="s-Cobordism / surgery classification",
    )
    assert out.theorem_tag == "highdim.scobordism.surgery"
    assert out.contract_version.startswith("2026.04")


def test_infer_theorem_tag_wall_group_ring_label():
    tag = infer_theorem_tag("Wall L-theory over group rings")
    assert tag == "highdim.wall.group_ring"
