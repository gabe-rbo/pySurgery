"""Test suite for the Tutorial Coverage and Snippet Validation system.

Overview:
    This module ensures that the documentation's "Tutorial Coverage Matrix" 
    stays in sync with the codebase. it verifies that all required code 
    snippets mentioned in `tutorial_coverage.json` actually exist in the 
    project and that the validator correctly flags missing snippets.

Key Concepts:
    - **Tutorial Coverage**: Mapping between mathematical features and documentation tutorials.
    - **Snippet Validation**: Automated check for the existence of tagged code sections.
"""

import json
from pathlib import Path

from pysurgery.utils.tutorial_coverage_validator import validate_tutorial_coverage


def test_tutorial_coverage_contract_is_valid_for_current_workspace():
    """Verify that the master tutorial coverage file is internally consistent.

    What is Being Computed?:
        Checks `docs/tutorial_coverage.json` against the current project state.

    Algorithm:
        1. Locate the project root and the coverage JSON file.
        2. Run `validate_tutorial_coverage`.
        3. Assert that no missing snippets or invalid paths are found.
    """
    root = Path(__file__).resolve().parents[1]
    coverage = root / "docs" / "tutorial_coverage.json"
    errors = validate_tutorial_coverage(coverage, root)
    assert errors == []


def test_tutorial_coverage_validator_detects_missing_snippet(tmp_path: Path):
    """Verify that the validator correctly identifies and reports missing snippets.

    What is Being Computed?:
        Tests the error-detection capability of the `validate_tutorial_coverage` tool.

    Algorithm:
        1. Load the valid coverage JSON and inject a non-existent snippet name.
        2. Save this modified "bad" coverage file to a temporary directory.
        3. Run the validator on the bad file.
        4. Assert that the validator returns an error message mentioning "missing snippet".
    """
    root = Path(__file__).resolve().parents[1]
    source = root / "docs" / "tutorial_coverage.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    data["items"][0]["required_snippets"] = ["__SNIPPET_THAT_DOES_NOT_EXIST__"]
    bad = tmp_path / "bad_coverage.json"
    bad.write_text(json.dumps(data), encoding="utf-8")

    errors = validate_tutorial_coverage(bad, root)
    assert any("missing snippet" in err.lower() for err in errors)
