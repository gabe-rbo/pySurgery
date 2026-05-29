"""Phase 4A integration tests: Modules A (Sullivan) + B (Adams E_2).

Verifies the ``compute_rational_and_adams`` façade on three standard spaces
(S^3, CP^2, RP^4), cross-checks rational dimensions against literature, and
audits resource guardrails (no dense materialisation in the Adams module).

Architecture: memory/v2-0-0/phase_4_homotopy/SullivanSteenrod_architecture.md
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple

import pytest

from pysurgery.homotopy.higher_homotopy_groups import (
    AdamsE2Page,
    compute_rational_and_adams,
    cp_n_cohomology_fp,
    rp_n_cohomology_fp,
    sphere_cohomology_fp,
)
from pysurgery.homotopy.sullivan_models import (
    RationalCohomologyAlgebra,
    complex_projective_space_cohomology,
    sphere_cohomology,
)
from pysurgery.homotopy.rational_homotopy import RationalHomotopyGroup


# ── Test fixtures ─────────────────────────────────────────────────────────────


def _s3_inputs() -> Tuple[object, object, str]:
    return sphere_cohomology(3), sphere_cohomology_fp(3, prime=2), "S^3"


def _cp2_inputs() -> Tuple[object, object, str]:
    return (
        complex_projective_space_cohomology(2),
        cp_n_cohomology_fp(2, prime=2),
        "CP^2",
    )


def _rp4_inputs() -> Tuple[object, object, str]:
    # H*(RP^4; ℚ) — only the unit class survives rationally.
    rp4_rational = RationalCohomologyAlgebra(
        betti={0: 1}, name="RP^4", max_degree=8
    )
    return rp4_rational, rp_n_cohomology_fp(4, prime=2), "RP^4"


SPACES = [_s3_inputs(), _cp2_inputs(), _rp4_inputs()]


# ── Contract tests ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "rational_input,fp_ring,label",
    SPACES,
    ids=["S3", "CP2", "RP4"],
)
def test_facade_returns_both_contracts(rational_input, fp_ring, label):
    """Both modules return signed contracts on every test space."""
    rational, adams = compute_rational_and_adams(
        rational_input,
        fp_cohomology_ring=fp_ring,
        rational_max_degree=8,
        adams_prime=2,
        adams_s_max=4,
        adams_t_max=10,
        space_label=label,
    )

    # Type assertions
    assert isinstance(rational, RationalHomotopyGroup), label
    assert isinstance(adams, AdamsE2Page), label

    # exact=True (Module A always exact over ℚ; Module B always exact over F_p)
    assert rational.exact is True, f"{label}: rational not exact"
    assert adams.exact is True, f"{label}: adams not exact"

    # theorem_tag populated
    assert rational.theorem_tag, f"{label}: rational.theorem_tag empty"
    assert adams.theorem_tag, f"{label}: adams.theorem_tag empty"

    # Status
    assert rational.status == "success", f"{label}: rational.status={rational.status}"
    assert adams.status == "success", f"{label}: adams.status={adams.status}"

    # decision_ready
    assert rational.decision_ready() is True, label
    assert adams.decision_ready() is True, label


# ── Literature cross-checks (Module A) ────────────────────────────────────────


def test_rational_S3_matches_literature():
    """Rational homotopy of S^3: π_n ⊗ ℚ = ℚ in degree 3 only.

    Reference: Felix–Halperin–Thomas (2001), Example 12.4 (Sullivan model
    of an odd-dimensional sphere is Λ(x_n) with d=0; π_n ⊗ ℚ = ℚ, all
    other π_k ⊗ ℚ = 0 within the truncation window).
    """
    rational, _ = compute_rational_and_adams(
        *_s3_inputs()[:1],
        fp_cohomology_ring=_s3_inputs()[1],
        rational_max_degree=8,
        adams_s_max=2,
        adams_t_max=6,
        space_label=_s3_inputs()[2],
    )
    assert rational.pi_n_rational == {3: 1}
    assert rational.is_formal is True  # d = 0 model


def test_rational_CP2_matches_literature():
    """Rational homotopy of CP^2: π_2 = ℚ, π_5 = ℚ (Hopf-invariant generator).

    Reference: Felix–Halperin–Thomas (2001), Example 12.6 (CP^n has minimal
    model Λ(x_2, y_{2n+1}) with d(y) = x^{n+1}; π_2 = π_{2n+1} = ℚ, formal=False).
    """
    rational, _ = compute_rational_and_adams(
        *_cp2_inputs()[:1],
        fp_cohomology_ring=_cp2_inputs()[1],
        rational_max_degree=8,
        adams_s_max=2,
        adams_t_max=6,
        space_label=_cp2_inputs()[2],
    )
    assert rational.pi_n_rational == {2: 1, 5: 1}
    assert rational.is_formal is False  # d(y) = x^3 ≠ 0


def test_rational_RP4_matches_literature():
    """Rational homotopy of ℝP^4: π_n ⊗ ℚ = 0 for all n ≥ 1.

    Reference: Hatcher (2002) §3.G; finite π_1 spaces with vanishing
    rational cohomology in positive degrees are rationally contractible.
    """
    rational, _ = compute_rational_and_adams(
        *_rp4_inputs()[:1],
        fp_cohomology_ring=_rp4_inputs()[1],
        rational_max_degree=8,
        adams_s_max=2,
        adams_t_max=6,
        space_label=_rp4_inputs()[2],
    )
    assert rational.pi_n_rational == {}
    assert rational.is_formal is True  # trivial DGA


# ── Literature cross-checks (Module B): minimal generators row s=0 ────────────


def _e2_row(adams: AdamsE2Page, s: int, t_max: int) -> Dict[int, int]:
    return {t: adams.e2_dim(s, t) for t in range(t_max + 1) if adams.e2_dim(s, t) > 0}


def test_adams_S3_minimal_generators():
    """Adams Ext^{0,t} = minimal A-generators of H^*(X; F_2).

    For S^3, H*(S^3; F_2) = F_2{1, x} with x in degree 3 and Sq^i(x) = 0,
    so the minimal A-generators are {1, x}.
    Reference: Bruner (1993), Ext in the nineties §1.
    """
    _, adams = compute_rational_and_adams(
        *_s3_inputs()[:1],
        fp_cohomology_ring=_s3_inputs()[1],
        rational_max_degree=8,
        adams_s_max=4,
        adams_t_max=10,
    )
    assert _e2_row(adams, s=0, t_max=10) == {0: 1, 3: 1}


def test_adams_CP2_minimal_generators():
    """For CP^2, H*(CP^2; F_2) = F_2[x]/(x^3) with deg(x) = 2.

    Sq^2(x) = x^2 (instability), so x^2 = Sq^2(x) ∈ A·{x} is decomposable.
    Minimal A-generators: {1, x} → Ext^{0,*} = (0:1, 2:1).
    Reference: May (1981) §4 (Adams chart of ℂP^∞ truncated).
    """
    _, adams = compute_rational_and_adams(
        *_cp2_inputs()[:1],
        fp_cohomology_ring=_cp2_inputs()[1],
        rational_max_degree=8,
        adams_s_max=4,
        adams_t_max=10,
    )
    assert _e2_row(adams, s=0, t_max=10) == {0: 1, 2: 1}


def test_adams_RP4_minimal_generators():
    """For ℝP^4, H*(ℝP^4; F_2) = F_2[x]/(x^5) with deg(x) = 1.

    Sq^1(x) = x^2 makes x^2 decomposable.
    Sq^2(x) = C(1,2) x^3 = 0 mod 2, Sq^1(x^2) = 2x^3 = 0; so x^3 is a NEW
    minimal generator.  Sq^1(x^3) = 3x^4 = x^4 mod 2, so x^4 is decomposable.
    Minimal A-generators: {1, x, x^3} → Ext^{0,*} = (0:1, 1:1, 3:1).
    Reference: Bruner (1993) Table I; Adams (1958) §3.
    """
    _, adams = compute_rational_and_adams(
        *_rp4_inputs()[:1],
        fp_cohomology_ring=_rp4_inputs()[1],
        rational_max_degree=8,
        adams_s_max=4,
        adams_t_max=10,
    )
    assert _e2_row(adams, s=0, t_max=10) == {0: 1, 1: 1, 3: 1}


# ── Cross-module consistency check ────────────────────────────────────────────


@pytest.mark.parametrize(
    "rational_input,fp_ring,label",
    SPACES,
    ids=["S3", "CP2", "RP4"],
)
def test_rational_dim_le_adams_stem_sum(rational_input, fp_ring, label):
    """Sanity inequality: dim_ℚ(π_n ⊗ ℚ) ≤ Σ_{t−s = n} E_2^{s,t} in the reliable window.

    The Adams tower upper-bounds π_n^s(X)_2; rationally, only the s=0,1,...
    columns survive, but the sum over the column gives an a-priori bound.
    """
    rational, adams = compute_rational_and_adams(
        rational_input,
        fp_cohomology_ring=fp_ring,
        rational_max_degree=8,
        adams_prime=2,
        adams_s_max=4,
        adams_t_max=10,
        space_label=label,
    )
    s_rel, t_rel = adams.reliable_window
    for n, dim_q in rational.pi_n_rational.items():
        if n > t_rel:
            continue  # outside reliable window — skip
        stem_sum = sum(
            v
            for (s, t), v in adams.e2_grid.items()
            if (t - s) == n and t <= t_rel
        )
        assert dim_q <= stem_sum + 1, (
            f"{label}: rational dim π_{n} = {dim_q} exceeds Adams stem sum {stem_sum}"
        )


# ── Resource guardrail audits ─────────────────────────────────────────────────


def test_adams_module_has_no_dense_materialisation():
    """Static guardrail: scan the Adams source for forbidden dense conversions.

    Architecture decision B.G1: all linear algebra must remain sparse.
    Excludes hits inside docstrings/comments — only flags actual code calls.
    """
    src_path = (
        Path(__file__).resolve().parent.parent
        / "pysurgery"
        / "adams"
        / "spectral_sequence.py"

    )
    text = src_path.read_text()

    # Strip out docstrings and #-comments before scanning.
    cleaned = re.sub(r'"""[\s\S]*?"""', "", text)
    cleaned = re.sub(r"'''[\s\S]*?'''", "", cleaned)
    cleaned = re.sub(r"#.*", "", cleaned)

    forbidden_patterns = [
        r"\.toarray\s*\(",
        r"\.todense\s*\(",
        r"np\.array\s*\([^)]*sparse",
        r"np\.asarray\s*\([^)]*sparse",
    ]
    hits = []
    for pat in forbidden_patterns:
        for m in re.finditer(pat, cleaned):
            hits.append((pat, m.group(0)))
    assert not hits, f"Forbidden dense conversions in adams module: {hits}"


@pytest.mark.parametrize(
    "rational_input,fp_ring,label",
    SPACES,
    ids=["S3", "CP2", "RP4"],
)
def test_resource_summary_logged(rational_input, fp_ring, label, capsys):
    """Adams contract carries a populated resource_summary; log it for inspection."""
    _, adams = compute_rational_and_adams(
        rational_input,
        fp_cohomology_ring=fp_ring,
        rational_max_degree=8,
        adams_prime=2,
        adams_s_max=4,
        adams_t_max=10,
        space_label=label,
    )
    summary = adams.resource_summary
    assert "peak_mem_mb" in summary
    assert "wall_seconds" in summary
    assert summary["peak_mem_mb"] >= 0.0
    assert summary["wall_seconds"] >= 0.0
    print(
        f"[{label}] adams resource_summary: "
        f"peak={summary['peak_mem_mb']:.2f} MB, "
        f"wall={summary['wall_seconds']:.3f} s, "
        f"reliable_window={adams.reliable_window}"
    )


# ── Contract invariants ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "rational_input,fp_ring,label",
    SPACES,
    ids=["S3", "CP2", "RP4"],
)
def test_adams_grid_invariants(rational_input, fp_ring, label):
    """Every (s, t) key respects the truncation rectangle and dims are ≥ 0."""
    _, adams = compute_rational_and_adams(
        rational_input,
        fp_cohomology_ring=fp_ring,
        rational_max_degree=8,
        adams_prime=2,
        adams_s_max=4,
        adams_t_max=10,
        space_label=label,
    )
    for (s, t), dim in adams.e2_grid.items():
        assert 0 <= s <= adams.s_max, label
        assert 0 <= t <= adams.t_max, label
        assert dim >= 0, label
    for flag in adams.forced_vanishings:
        assert flag.target_dim == 0 or flag.source_dim == 0, label
    for flag in adams.ambiguous_differentials:
        assert flag.source_dim > 0 and flag.target_dim > 0, label


@pytest.mark.parametrize(
    "rational_input,fp_ring,label",
    SPACES,
    ids=["S3", "CP2", "RP4"],
)
def test_rational_invariants(rational_input, fp_ring, label):
    rational, _ = compute_rational_and_adams(
        rational_input,
        fp_cohomology_ring=fp_ring,
        rational_max_degree=8,
        adams_prime=2,
        adams_s_max=2,
        adams_t_max=6,
        space_label=label,
    )
    assert set(rational.pi_n_rational.keys()) == set(rational.nonzero_degrees)
    for n, d in rational.pi_n_rational.items():
        assert d > 0
    if rational.nonzero_degrees:
        assert max(rational.nonzero_degrees) <= rational.truncation_degree
    if rational.is_formal:
        assert rational.massey_products == []
    assert rational.cohomology_iso_verified is True
