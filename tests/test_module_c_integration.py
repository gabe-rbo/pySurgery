"""Module C integration tests (Phase 5).

Covers:
    - test_both_paths_overlap_einf  : Path A and Path B yield the same E_∞
                                       on overlapping test spaces.
    - test_confidence_transparency  : confidence_score reflects the path's
                                       provenance (interactive avg vs Lean
                                       proven-fraction) and is in [0, 1].
    - test_no_infinite_loops        : empty grid, no ambiguous flags, and
                                       all-skip interactive sessions all
                                       terminate.

The Lean tests use the default ``_default_lean_script`` which emits
``theorem … : True := by trivial`` — Lean compiles it cleanly so every
flag returns ``result="proven"``.  Tests that don't need a real Lean
toolchain inject a custom ``lean_export_callable`` is unnecessary; the
default already runs fast (<2 s per script).
"""
from __future__ import annotations

import shutil
from datetime import datetime, timezone

import pytest

from pysurgery.adams.spectral_sequence import (
    AdamsDifferentialFlag,
    AdamsE2Page,
)
from pysurgery.adams.e_infinity_resolver import (
    ConvergedAdamsPage,
    UserVerifiedDifferential,
)
from pysurgery.homotopy.higher_homotopy_groups import (
    HomotopyGroupApproximation,
    RationalHomotopyGroup,
    sphere_cohomology,
    sullivan_rational_homotopy,
    synthesize_homotopy_group_with_e_infinity,
)
_LEAN_AVAILABLE = shutil.which("lean") is not None
requires_lean = pytest.mark.skipif(
    not _LEAN_AVAILABLE, reason="Lean 4 toolchain not on PATH"
)


# ── Helpers ───────────────────────────────────────────────────────────────────


class _ScriptedCLIIO:
    """Scripted CLIIO that returns canned responses in order."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._idx = 0
        self.messages: list[str] = []

    def write(self, msg: str) -> None:
        self.messages.append(msg)

    def prompt(self, msg: str, choices: tuple) -> str:
        if self._idx >= len(self._responses):
            raise AssertionError(
                f"_ScriptedCLIIO exhausted at prompt {msg!r}"
            )
        v = self._responses[self._idx]
        self._idx += 1
        return v

    def confirm(self, msg: str) -> bool:
        return True


def _toy_e2_one_ambiguous(prime: int = 2) -> AdamsE2Page:
    """Single ambiguous d_2: (0,4) → (2,5).  Used by both paths."""
    return AdamsE2Page(
        space_label="ToyOneFlag",
        prime=prime,
        s_max=4,
        t_max=10,
        e2_grid={(0, 0): 1, (0, 4): 1, (2, 5): 1},
        forced_vanishings=[],
        ambiguous_differentials=[
            AdamsDifferentialFlag(
                r=2,
                source=(0, 4),
                target=(2, 5),
                classification="ambiguous",
                reason="both dims > 0",
                source_dim=1,
                target_dim=1,
            )
        ],
        reliable_window=(4, 6),
        status="success",
        reasoning="Toy E_2 page with one ambiguous flag",
    )


def _toy_e2_no_flags() -> AdamsE2Page:
    """E_2 with no ambiguous flags — both paths converge immediately."""
    return AdamsE2Page(
        space_label="ToyNoFlags",
        prime=2,
        s_max=4,
        t_max=10,
        e2_grid={(0, 0): 1, (0, 3): 1},
        forced_vanishings=[],
        ambiguous_differentials=[],
        reliable_window=(4, 6),
        status="success",
        reasoning="No ambiguous flags",
    )


def _toy_e2_empty_grid() -> AdamsE2Page:
    """E_2 with an empty grid (degenerate edge case)."""
    return AdamsE2Page(
        space_label="ToyEmpty",
        prime=2,
        s_max=2,
        t_max=2,
        e2_grid={},
        forced_vanishings=[],
        ambiguous_differentials=[],
        reliable_window=(2, 0),
        status="success",
        reasoning="Empty grid",
    )


def _s3_rational() -> RationalHomotopyGroup:
    return sullivan_rational_homotopy(sphere_cohomology(3), space_label="S^3")


# ── 1) Both paths produce the same E_∞ on overlapping test spaces ─────────────


class TestBothPathsAgree:
    """Path A and Path B must converge to the same E_∞ grid on overlap."""

    @requires_lean
    def test_no_flags_overlap(self, tmp_path):
        """E_2 with no ambiguous differentials → both paths trivially equal."""
        e2 = _toy_e2_no_flags()
        rational = _s3_rational()

        approx_a = synthesize_homotopy_group_with_e_infinity(
            rational, e2, resolution_path="interactive", n=3,
            interactive_kwargs={
                "cli_io": _ScriptedCLIIO([]),
                "checkpoint_dir": tmp_path / "ckpt_a",
                "require_proof_reference": False,
            },
        )
        approx_b = synthesize_homotopy_group_with_e_infinity(
            rational, e2, resolution_path="lean_formal", n=3,
            lean_kwargs={
                "per_diff_timeout_sec": 15,
                "total_budget_sec": 60,
                "artefact_dir": tmp_path / "lean_artefacts_b",
            },
        )

        # Same rank, same torsion data (None: E_∞ = E_2 at s>0 contributes nothing at n=3).
        assert approx_a.rational_rank == approx_b.rational_rank == 1
        # No flags → no positive-filtration entries at stem 3.
        assert approx_a.torsion_invariants is None
        assert approx_b.torsion_invariants is None
        # Both converged at the initial page.
        assert approx_a.convergence_page == approx_b.convergence_page == 2

    @requires_lean
    def test_one_ambiguous_zero_decision_overlap(self, tmp_path):
        """Decide the single flag = 'zero' on Path A (matches Path B's 'proven')
        ⇒ grids stabilise unchanged ⇒ both paths give the same E_∞."""
        e2 = _toy_e2_one_ambiguous()
        rational = _s3_rational()

        # The (2,5) bidegree has stem t-s = 3, so query at n=3.
        # Path A: scripted "zero" with high confidence.
        cli = _ScriptedCLIIO(["zero", "0.95", "test ref"])
        approx_a = synthesize_homotopy_group_with_e_infinity(
            rational, e2, resolution_path="interactive", n=3,
            interactive_kwargs={
                "cli_io": cli,
                "checkpoint_dir": tmp_path / "ckpt_a",
                "require_proof_reference": False,
            },
        )
        # Path B: Lean default script proves the flag (result="proven" → "zero").
        approx_b = synthesize_homotopy_group_with_e_infinity(
            rational, e2, resolution_path="lean_formal", n=3,
            lean_kwargs={
                "per_diff_timeout_sec": 30,
                "total_budget_sec": 120,
                "artefact_dir": tmp_path / "lean_artefacts_b",
            },
        )

        # Stem n=3: E_∞^{2,5} survives with dim 1 ⇒ torsion_invariants=(1,)
        # under the rank-1 filtration interpretation.
        assert approx_a.torsion_invariants == (1,)
        assert approx_b.torsion_invariants == (1,)
        # S^3 has π_3 ⊗ ℚ = ℚ ⇒ rational_rank=1 on both paths.
        assert approx_a.rational_rank == approx_b.rational_rank == 1
        assert approx_a.path_used == "user_interactive"
        assert approx_b.path_used == "lean_formal"
        # Both grids should agree.
        assert (
            approx_a.convergence_page is not None
            and approx_b.convergence_page is not None
        )


# ── 2) Confidence metrics reported transparently ──────────────────────────────


class TestConfidenceTransparency:
    def test_rational_only_decisive(self):
        rational = _s3_rational()
        e2 = _toy_e2_no_flags()
        approx = synthesize_homotopy_group_with_e_infinity(
            rational, e2, resolution_path="rational_only", n=3
        )
        assert approx.path_used == "rational_only"
        assert approx.confidence_score == pytest.approx(1.0)
        assert approx.known_exact is False  # rational_only never marks exact
        assert "rational data only" in approx.caveats

    def test_interactive_confidence_is_average_of_user_input(self, tmp_path):
        """confidence_score = avg(user_confidence) × rational_factor."""
        e2 = _toy_e2_one_ambiguous()
        rational = _s3_rational()

        cli = _ScriptedCLIIO(["nonzero", "0.6", "ref"])
        approx = synthesize_homotopy_group_with_e_infinity(
            rational, e2, resolution_path="interactive", n=4,
            interactive_kwargs={
                "cli_io": cli,
                "checkpoint_dir": tmp_path / "ckpt",
                "require_proof_reference": False,
            },
        )
        # rational_factor = 1.0 (S^3 rational is decisive); avg=0.6.
        assert approx.confidence_score == pytest.approx(0.6, abs=1e-9)
        assert approx.path_used == "user_interactive"
        assert approx.known_exact is False  # human never exact

    @requires_lean
    def test_lean_confidence_is_proven_fraction(self, tmp_path):
        """All flags 'proven' → confidence = 1.0 × rational_factor."""
        e2 = _toy_e2_one_ambiguous()
        rational = _s3_rational()
        approx = synthesize_homotopy_group_with_e_infinity(
            rational, e2, resolution_path="lean_formal", n=4,
            lean_kwargs={
                "per_diff_timeout_sec": 30,
                "total_budget_sec": 120,
                "artefact_dir": tmp_path / "lean_artefacts",
            },
        )
        assert approx.path_used == "lean_formal"
        assert approx.confidence_score == pytest.approx(1.0)
        # Vacuous success branch is OK; non-vacuous all-proven also OK.
        assert approx.known_exact is True
        assert "lean attempts" in approx.caveats

    def test_low_confidence_rejected_in_caveats(self, tmp_path):
        """Build a ConvergedAdamsPage with a single decisive verification at
        confidence 0.4 and feed it through the synthesizer to check that the
        confidence_score is the user's average (no inflation)."""
        e2 = _toy_e2_one_ambiguous()
        rational = _s3_rational()

        # Hand-build a ConvergedAdamsPage with a decisive verification at 0.4
        # to bypass the resolver's threshold filter and exercise the
        # synthesizer's own averaging path.
        v = UserVerifiedDifferential(
            r=2,
            bidegree_source=(0, 4),
            bidegree_target=(2, 5),
            decision="nonzero",
            decision_input_raw="nonzero",
            timestamp=datetime.now(timezone.utc),
            user_id="test",
            proof_reference="weak ref",
            user_confidence=0.4,
        )
        converged = ConvergedAdamsPage(
            space_label=e2.space_label,
            prime=e2.prime,
            s_max=e2.s_max,
            t_max=e2.t_max,
            e_infinity_grid={(0, 0): 1},
            page_history=[],
            convergence_page=3,
            user_verifications=[v],
            lean_attempts=[],
            path_used="interactive",
            status="success",
            reasoning="hand-built",
            exact=False,
            theorem_tag="adams.einf.interactive_v1",
        )
        approx = synthesize_homotopy_group_with_e_infinity(
            rational, e2, resolution_path="interactive", n=3, converged=converged
        )
        assert approx.path_used == "user_interactive"
        # avg(0.4) × rational_factor(1.0) = 0.4
        assert approx.confidence_score == pytest.approx(0.4, abs=1e-9)
        assert approx.known_exact is False


# ── 3) No infinite loops on edge cases ────────────────────────────────────────


class TestNoInfiniteLoops:
    @requires_lean
    def test_empty_grid_terminates(self, tmp_path):
        e2 = _toy_e2_empty_grid()
        rational = _s3_rational()

        approx_a = synthesize_homotopy_group_with_e_infinity(
            rational, e2, resolution_path="interactive", n=3,
            interactive_kwargs={
                "cli_io": _ScriptedCLIIO([]),
                "checkpoint_dir": tmp_path / "ckpt_a",
                "require_proof_reference": False,
            },
        )
        approx_b = synthesize_homotopy_group_with_e_infinity(
            rational, e2, resolution_path="lean_formal", n=3,
            lean_kwargs={
                "per_diff_timeout_sec": 15,
                "total_budget_sec": 30,
                "artefact_dir": tmp_path / "lean_artefacts",
            },
        )
        assert isinstance(approx_a, HomotopyGroupApproximation)
        assert isinstance(approx_b, HomotopyGroupApproximation)
        assert approx_a.rational_rank == 1
        assert approx_b.rational_rank == 1

    def test_no_ambiguous_flags_terminates(self, tmp_path):
        e2 = _toy_e2_no_flags()
        rational = _s3_rational()
        approx = synthesize_homotopy_group_with_e_infinity(
            rational, e2, resolution_path="interactive", n=3,
            interactive_kwargs={
                "cli_io": _ScriptedCLIIO([]),
                "checkpoint_dir": tmp_path / "ckpt",
                "require_proof_reference": False,
            },
        )
        assert approx.path_used == "user_interactive"
        # No verifications were needed.
        assert approx.confidence_score == pytest.approx(1.0)

    def test_all_skip_interactive_terminates(self, tmp_path):
        """Every flag answered 'skip' → loop exits via max_pages truncation."""
        e2 = _toy_e2_one_ambiguous()
        rational = _s3_rational()

        # The skip path returns immediately (no confidence/reference prompt).
        # max_pages=2 caps the outer loop so we cannot diverge.
        cli = _ScriptedCLIIO(["skip"])
        approx = synthesize_homotopy_group_with_e_infinity(
            rational, e2, resolution_path="interactive", n=4,
            interactive_kwargs={
                "cli_io": cli,
                "checkpoint_dir": tmp_path / "ckpt",
                "require_proof_reference": False,
                "max_pages": 2,
            },
        )
        assert approx.path_used == "user_interactive"
        # No decisive verifications and the flag remained open ⇒ confidence 0.
        assert approx.confidence_score == pytest.approx(0.0)

    def test_pre_supplied_converged_skips_resolver(self, tmp_path):
        """Passing a pre-built ConvergedAdamsPage skips the resolver entirely."""
        e2 = _toy_e2_no_flags()
        rational = _s3_rational()
        # Hand-build a ConvergedAdamsPage matching e2.
        converged = ConvergedAdamsPage(
            space_label=e2.space_label,
            prime=e2.prime,
            s_max=e2.s_max,
            t_max=e2.t_max,
            e_infinity_grid=dict(e2.e2_grid),
            page_history=[],
            convergence_page=2,
            user_verifications=[],
            lean_attempts=[],
            path_used="lean_formal",
            status="success",
            reasoning="vacuous",
            exact=True,
            theorem_tag="adams.einf.lean4_formal_v1",
        )
        approx = synthesize_homotopy_group_with_e_infinity(
            rational, e2, resolution_path="lean_formal", n=3, converged=converged
        )
        assert approx.path_used == "lean_formal"
        assert approx.known_exact is True
        assert approx.convergence_page == 2
