"""Test suite for the Theorem Tagging and Classification system.

Overview:
    This module verifies the correctness of the `infer_theorem_tag` logic,
    which maps human-readable mathematical descriptions (e.g., "Freedman classification")
    to internal machine-parsable tags used for result certification and metadata tracking.

Key Concepts:
    - **Theorem Tags**: Semantic identifiers for major topological results.
    - **Tag Inference**: Heuristic mapping from strings to hierarchical tags.
    - **Homeomorphism Metadata**: Automatic tagging of results based on the theorem used.
"""

from pysurgery.homeomorphism import HomeomorphismResult
from pysurgery.core.theorem_tags import infer_theorem_tag


def test_infer_theorem_tag_known_label():
    """Verify inference for a standard 4D manifold classification label.

    What is Being Computed?:
        Maps "Freedman classification" to its canonical tag.
    """
    tag = infer_theorem_tag("Freedman classification")
    assert tag == "4d.freedman.simply_connected"


def test_homeomorphism_result_auto_sets_theorem_tag():
    """Verify that `HomeomorphismResult` automatically resolves theorem tags.

    What is Being Computed?:
        Checks the automatic resolution of `theorem_tag` when a `HomeomorphismResult` 
        is initialized with a known theorem name.

    Algorithm:
        1. Initialize `HomeomorphismResult` with "s-Cobordism / surgery classification".
        2. Assert the `theorem_tag` is correctly set to "highdim.scobordism.surgery".
        3. Verify the contract version is present.
    """
    out = HomeomorphismResult(
        status="inconclusive",
        is_homeomorphic=None,
        reasoning="INCONCLUSIVE",
        theorem="s-Cobordism / surgery classification",
    )
    assert out.theorem_tag == "highdim.scobordism.surgery"
    assert out.contract_version.startswith("2026.04")


def test_infer_theorem_tag_wall_group_ring_label():
    """Verify inference for high-dimensional L-theory over group rings.

    What is Being Computed?:
        Maps "Wall L-theory over group rings" to "highdim.wall.group_ring".
    """
    tag = infer_theorem_tag("Wall L-theory over group rings")
    assert tag == "highdim.wall.group_ring"
