import json
from pathlib import Path

from pysurgery.utils.tutorial_coverage_validator import validate_tutorial_coverage


def test_tutorial_coverage_contract_is_valid_for_current_workspace():
    root = Path(__file__).resolve().parents[1]
    coverage = root / "docs" / "tutorial_coverage.json"
    errors = validate_tutorial_coverage(coverage, root)
    assert errors == []


def test_tutorial_coverage_validator_detects_missing_snippet(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    source = root / "docs" / "tutorial_coverage.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    data["items"][0]["required_snippets"] = ["__SNIPPET_THAT_DOES_NOT_EXIST__"]
    bad = tmp_path / "bad_coverage.json"
    bad.write_text(json.dumps(data), encoding="utf-8")

    errors = validate_tutorial_coverage(bad, root)
    assert any("missing snippet" in err.lower() for err in errors)

