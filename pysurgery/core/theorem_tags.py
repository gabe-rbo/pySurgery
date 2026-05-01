from __future__ import annotations


THEOREM_TAGS: dict[str, str] = {
    "Classification of Closed Surfaces": "surface.classification.closed",
    "Geometrization / 3-manifold recognition": "3d.geometrization.recognition",
    "Poincare Conjecture / Geometrization": "3d.poincare.geometrization",
    "Freedman classification": "4d.freedman.simply_connected",
    "s-Cobordism / surgery classification": "highdim.scobordism.surgery",
    "s-Cobordism / generalized Poincare": "highdim.scobordism.generalized_poincare",
    "Wall L-theory over group rings": "highdim.wall.group_ring",
}


def infer_theorem_tag(theorem: str | None) -> str | None:
    """Infer a stable theorem tag from user-facing theorem labels.

    What is Being Computed?:
        Maps human-readable theorem names or partial descriptions to canonical,
        stable tag strings used for classification and indexing.

    Algorithm:
        1. Clean and normalize input string (strip, handle accents).
        2. Check for exact matches in the THEOREM_TAGS dictionary.
        3. Fall back to keyword-based heuristic matching for common topological terms.
        4. Return 'unscoped.unknown' if no match is found.

    Preserved Invariants:
        None (this is a string mapping utility).

    Args:
        theorem: The user-facing theorem label to infer a tag for.

    Returns:
        str | None: The inferred theorem tag, or None if the input is None.

    Use When:
        - Mapping user input to internal classification tags
        - Indexing results by major topological theorems
        - Normalizing theorem references in reports

    Example:
        tag = infer_theorem_tag("Poincare Conjecture 3D")
        # returns "3d.poincare.geometrization"
    """
    if theorem is None:
        return None
    t = str(theorem).strip()
    t = t.replace("é", "e")
    if not t:
        return None
    direct = THEOREM_TAGS.get(t)
    if direct is not None:
        return direct

    low = t.lower()
    if "surface" in low:
        return "surface.classification.closed"
    if "freedman" in low:
        return "4d.freedman.simply_connected"
    if "poincare" in low and "3" in low:
        return "3d.poincare.geometrization"
    if "geometrization" in low:
        return "3d.geometrization.recognition"
    if "wall" in low and "group" in low:
        return "highdim.wall.group_ring"
    if "s-cobord" in low or "surgery" in low:
        return "highdim.scobordism.surgery"
    return "unscoped.unknown"
