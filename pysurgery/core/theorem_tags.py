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
    """Infer a stable theorem tag from user-facing theorem labels."""
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


