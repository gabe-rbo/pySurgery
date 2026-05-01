"""Validation logic for tutorial documentation coverage.

Overview:
    This module provides tools to verify that the library's theoretical results 
    (theorems, lemmas, etc.) are properly documented with functional code examples 
    in Jupyter notebooks. It cross-references a coverage JSON file against 
    the actual contents of the examples directory.

Key Concepts:
    - **Coverage Item**: A mapping between a theorem ID and its notebook location.
    - **Snippet Verification**: Ensuring specific code patterns exist in the target notebooks.
    - **Contract Version**: Ensuring the metadata format matches the library version.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pysurgery.core.foundations import CONTRACT_VERSION, COVERAGE_MATRIX


REQUIRED_ITEM_KEYS = {"id", "kind", "target", "notebooks", "required_snippets"}


def _read_text(path: Path) -> str:
    """Read and return the content of a text file.

    Algorithm:
        Reads file content using UTF-8 encoding, ignoring decoding errors.

    Args:
        path: Path to the file to be read.

    Returns:
        str: The full text content of the file.
    """
    return path.read_text(encoding="utf-8", errors="ignore")


def validate_tutorial_coverage(
    coverage_path: str | Path, workspace_root: str | Path
) -> list[str]:
    """Validate that the tutorial coverage JSON file correctly maps theorem tags to snippets.

    What is Being Computed?:
        Validates the integrity and completeness of the tutorial coverage metadata, ensuring that
        every documented theorem tag corresponds to actual code snippets within the tutorial notebooks.

    Algorithm:
        1. Load and parse the coverage JSON file.
        2. Verify `contract_version` against the current library version.
        3. For each coverage item:
           a. Validate schema (required keys like 'id', 'kind', 'target', etc.).
           b. Verify that referenced Jupyter notebooks exist on disk.
           c. Scan notebook content to ensure all 'required_snippets' are present.
        4. Cross-reference the identified theorem tags against the global `COVERAGE_MATRIX`
           to identify any uncovered theoretical results.

    Preserved Invariants:
        - None (This is a validation utility and does not modify topological data).

    Args:
        coverage_path: Path to the tutorial_coverage.json file.
        workspace_root: Path to the root of the repository containing the 'examples' directory.

    Returns:
        list[str]: A list of descriptive error messages. If empty, validation passed.

    Use When:
        - Running CI/CD pipelines to ensure documentation does not drift from implementation.
        - Auditing tutorial coverage after adding new theoretical features or theorem tags.
        - Verifying that notebook refactoring hasn't broken the documentation links.

    Example:
        errors = validate_tutorial_coverage("docs/tutorial_coverage.json", ".")
        if not errors:
            print("Documentation is synchronized with examples.")
    """
    coverage_file = Path(coverage_path)
    root = Path(workspace_root)
    errors: list[str] = []

    if not coverage_file.exists():
        return [f"Coverage file not found: {coverage_file}"]

    try:
        data: dict[str, Any] = json.loads(coverage_file.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"Failed to parse coverage file '{coverage_file}': {exc}"]

    if data.get("contract_version") != CONTRACT_VERSION:
        errors.append(
            "Contract version mismatch: coverage file has "
            f"{data.get('contract_version')!r}, code has {CONTRACT_VERSION!r}."
        )

    items = data.get("items")
    if not isinstance(items, list) or not items:
        return errors + ["Coverage file must define a non-empty 'items' list."]

    seen_ids: set[str] = set()
    theorem_targets: set[str] = set()

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(f"Item at index {idx} is not an object.")
            continue
        missing = REQUIRED_ITEM_KEYS - set(item.keys())
        if missing:
            errors.append(f"Item at index {idx} is missing keys: {sorted(missing)}")
            continue

        item_id = str(item["id"])
        if item_id in seen_ids:
            errors.append(f"Duplicate coverage item id: {item_id}")
        seen_ids.add(item_id)

        kind = str(item["kind"])
        if kind == "theorem_tag":
            theorem_targets.add(str(item["target"]))

        notebooks = item.get("notebooks")
        snippets = item.get("required_snippets")
        if not isinstance(notebooks, list) or not notebooks:
            errors.append(f"Item '{item_id}' must define a non-empty notebooks list.")
            continue
        if not isinstance(snippets, list) or not snippets:
            errors.append(
                f"Item '{item_id}' must define a non-empty required_snippets list."
            )
            continue

        notebook_texts: list[str] = []
        for nb_rel in notebooks:
            nb_path = root / str(nb_rel)
            if not nb_path.exists():
                errors.append(f"Item '{item_id}' references missing notebook: {nb_rel}")
                continue
            notebook_texts.append(_read_text(nb_path))

        if not notebook_texts:
            continue

        for snippet in snippets:
            s = str(snippet)
            if not any(s in t for t in notebook_texts):
                errors.append(
                    f"Item '{item_id}' missing snippet '{s}' in referenced notebooks."
                )

    required_theorem_tags = {entry.theorem_tag for entry in COVERAGE_MATRIX}
    uncovered_theorems = sorted(required_theorem_tags - theorem_targets)
    if uncovered_theorems:
        errors.append(
            "Tutorial coverage is missing theorem-tag entries for: "
            + ", ".join(uncovered_theorems)
        )

    return errors


def main() -> int:
    """Execute the tutorial coverage validation suite.

    Overview:
        Locates the project's tutorial coverage metadata and repository root, then executes
        the validation logic to ensure all theorem tags are properly documented in notebooks.

    Algorithm:
        1. Resolve the workspace root relative to this script's location.
        2. Define the expected path for 'docs/tutorial_coverage.json'.
        3. Invoke `validate_tutorial_coverage`.
        4. Print any discovered errors to stdout.
        5. Return a non-zero exit code if validation fails.

    Returns:
        int: 0 if validation passes, 1 if any errors or mismatches are found.
    """
    root = Path(__file__).resolve().parents[2]
    coverage = root / "docs" / "tutorial_coverage.json"
    errors = validate_tutorial_coverage(coverage, root)
    if errors:
        for err in errors:
            print(f"ERROR: {err}")
        return 1
    print("Tutorial coverage validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
