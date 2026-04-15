from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pysurgery.core.foundations import CONTRACT_VERSION, COVERAGE_MATRIX


REQUIRED_ITEM_KEYS = {"id", "kind", "target", "notebooks", "required_snippets"}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def validate_tutorial_coverage(
    coverage_path: str | Path, workspace_root: str | Path
) -> list[str]:
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
