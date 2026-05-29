from __future__ import annotations


THEOREM_TAGS: dict[str, str] = {
    "Classification of Closed Surfaces": "surface.classification.closed",
    "Geometrization / 3-manifold recognition": "3d.geometrization.recognition",
    "Poincare Conjecture / Geometrization": "3d.poincare.geometrization",
    "Freedman classification": "4d.freedman.simply_connected",
    "s-Cobordism / surgery classification": "highdim.scobordism.surgery",
    "s-Cobordism / generalized Poincare": "highdim.scobordism.generalized_poincare",
    "Wall L-theory over group rings": "highdim.wall.group_ring",
    "Adams E2 page": "adams.e2.ext_steenrod",
    "Adams forced differential vanishing": "adams.differential.forced_vanishing",
}


# Stable, importable theorem-tag constants.
RATIONAL_QUILLEN_SULLIVAN: str = "rational.quillen_sullivan.minimal_model"
RATIONAL_FORMALITY_DGMS: str = "rational.formality.deligne_griffiths_morgan_sullivan"
RATIONAL_MASSEY: str = "rational.massey_products.sullivan_model"
ADAMS_E2_EXT_STEENROD: str = "adams.e2.ext_steenrod"
ADAMS_DIFFERENTIAL_FORCED: str = "adams.differential.forced_vanishing"

# E∞ resolver tags (see RFC-einf-v2)
ADAMS_EINF_INTERACTIVE: str = "adams.einf.interactive_v1"
ADAMS_EINF_LEAN_FORMAL: str = "adams.einf.lean4_formal_v1"
ADAMS_EINF_HYBRID: str = "adams.einf.hybrid_v1"
ADAMS_EINF_PARTIAL: str = "adams.einf.partial_v1"
HOMOTOPY_SYNTHESIS: str = "homotopy.synthesis.einf_v1"


# ── Handle Surgery theorem tags (Phase 10) ───────────────────────────────────

# Milnor 1965 — index-k handle produces β_{k−1}(M'') = β_{k−1}(M) ± 1, β_k(M'') = β_k(M) ± 1
SURGERY_HANDLE_MAYER_VIETORIS: str = "surgery.handle.mayer_vietoris"

# Exact ℤ linking via Seifert chain F (∂F = K_b) back-solved by SNF of ∂_{q+1}
SURGERY_LINKING_RELATIVE_SNF_Z: str = "surgery.linking.relative_snf_z"

# Equivalent linking via Lefschetz duality H_q(K, K \ K_b) ≅ H^{n−q}(K_b)
SURGERY_LINKING_LEFSCHETZ_PAIRING: str = "surgery.linking.lefschetz_pairing_z"

# Mod-2 linking; exact over F₂ but loses ℤ-torsion information
SURGERY_LINKING_F2_HEURISTIC: str = "surgery.linking.f2_torsion_blind"

# σ ⊂ K certified PL-homeomorphic to S^{k−1}, embedded, and framed
SURGERY_ATTACHMENT_SPHERE_RECOGNITION_EXACT: str = "surgery.attachment.sphere_recognition_exact"

# SNF cycle representative — embeddedness and framing NOT verified; exact=False
SURGERY_ATTACHMENT_SPHERE_SNF_HEURISTIC: str = "surgery.attachment.sphere_snf_heuristic"

# Iterated index-1 surgery achieves lk = 0 in |lk_0| steps (Milnor 1961)
SURGERY_DELINKING_UNLINKING_NUMBER: str = "surgery.delinking.unlinking_number"

# Post-surgery verification via SNF over ℤ; certifies MV postcondition and torsion
SURGERY_VERIFY_SNF_BETTI_TORSION: str = "surgery.verify.snf_betti_torsion"

# Normal bundle ν(σ ⊂ M) trivial with chosen trivialization (Wall 1970, §1)
SURGERY_HANDLE_FRAMING_TRIVIAL: str = "surgery.handle.framing_trivial_normal_bundle"

# ── Auto Surgery theorem tags (Phase 2.0.0) ──────────────────────────────────
AUTO_SURGERY_FULL_PIPELINE: str = "auto.surgery.full_pipeline"
AUTO_SURGERY_UNLINK: str = "auto.surgery.unlink"
AUTO_SURGERY_SEPARATE_NESTED: str = "auto.surgery.separate_nested"
AUTO_SURGERY_DETECT_NESTED: str = "auto.surgery.detect_nested"
AUTO_SURGERY_KILL_PI1: str = "auto.surgery.kill_pi1"
AUTO_SURGERY_KILL_HOMOLOGY_DIM: str = "auto.surgery.kill_homology_dim"
AUTO_SURGERY_MIDDLE_OBSTRUCTION: str = "auto.surgery.middle_obstruction"
AUTO_SURGERY_CUT_SITE: str = "auto.surgery.cut_site"
SURGERY_CANCELLING_PAIR: str = "surgery.cancelling_pair"  # for Gap G16

THEOREM_TAGS.update(
    {
        "Handle Surgery Mayer-Vietoris": SURGERY_HANDLE_MAYER_VIETORIS,
        "Linking Relative SNF Z": SURGERY_LINKING_RELATIVE_SNF_Z,
        "Linking Lefschetz Pairing": SURGERY_LINKING_LEFSCHETZ_PAIRING,
        "Linking F2 Heuristic": SURGERY_LINKING_F2_HEURISTIC,
        "Attachment Sphere Recognition Exact": SURGERY_ATTACHMENT_SPHERE_RECOGNITION_EXACT,
        "Attachment Sphere SNF Heuristic": SURGERY_ATTACHMENT_SPHERE_SNF_HEURISTIC,
        "Delinking Unlinking Number": SURGERY_DELINKING_UNLINKING_NUMBER,
        "Verify SNF Betti Torsion": SURGERY_VERIFY_SNF_BETTI_TORSION,
        "Handle Framing Trivial": SURGERY_HANDLE_FRAMING_TRIVIAL,
        "Auto Surgery Full Pipeline": AUTO_SURGERY_FULL_PIPELINE,
        "Auto Surgery Unlink": AUTO_SURGERY_UNLINK,
        "Auto Surgery Separate Nested": AUTO_SURGERY_SEPARATE_NESTED,
        "Auto Surgery Detect Nested": AUTO_SURGERY_DETECT_NESTED,
        "Auto Surgery Kill Pi1": AUTO_SURGERY_KILL_PI1,
        "Auto Surgery Kill Homology Dim": AUTO_SURGERY_KILL_HOMOLOGY_DIM,
        "Auto Surgery Middle Obstruction": AUTO_SURGERY_MIDDLE_OBSTRUCTION,
        "Auto Surgery Cut Site": AUTO_SURGERY_CUT_SITE,
        "Surgery Cancelling Pair": SURGERY_CANCELLING_PAIR,
    }
)


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
    if "linking" in low or "seifert" in low:
        return SURGERY_LINKING_RELATIVE_SNF_Z
    if "delink" in low:
        return SURGERY_DELINKING_UNLINKING_NUMBER
    if "handle" in low and "mayer" in low:
        return SURGERY_HANDLE_MAYER_VIETORIS
    if "handle" in low or "attachment" in low:
        return SURGERY_ATTACHMENT_SPHERE_RECOGNITION_EXACT
    return "unscoped.unknown"
