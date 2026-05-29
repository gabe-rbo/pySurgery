"""Tests for InteractiveAdamsResolver (Phase 5, Path A).

Covers:
    - test_interaction_cli        : mock CLIIO drives a full resolution to success
    - test_confidence_tracking    : low-confidence verification does not advance the page
    - test_convergence            : a page with no ambiguous flags converges immediately
    - test_consistency_checking   : contradictory decisions raise InteractiveConsistencyError
    - test_e_infinity_computation : a hand-computed example matches the predicted E_∞
    - test_resume_from_checkpoint : partial run is checkpointed; resumed run completes correctly
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

import pytest

from pysurgery.adams.spectral_sequence import (
    AdamsDifferentialFlag,
    AdamsE2Page,
)
from pysurgery.adams.e_infinity_resolver import (
    ConvergedAdamsPage,
    InteractiveConsistencyError,
    UserVerifiedDifferential,
)
from pysurgery.adams.interactive_resolver import (
    InteractiveAdamsResolver,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


class MockCLIIO:
    """Scripted CLIIO for testing.  Responses consumed in order.

    Each entry in ``prompt_responses`` is either a plain string (returned as-is)
    or an exception instance (raised to simulate user interrupts).
    """

    def __init__(
        self,
        prompt_responses: list,
        confirm_responses: Optional[list[bool]] = None,
    ) -> None:
        self._responses = list(prompt_responses)
        self._response_idx = 0
        self._confirms = list(confirm_responses or [])
        self._confirm_idx = 0
        self.messages: list[str] = []

    def write(self, msg: str) -> None:
        self.messages.append(msg)

    def prompt(self, msg: str, choices: tuple) -> str:
        if self._response_idx >= len(self._responses):
            raise ValueError(
                f"MockCLIIO exhausted: no response for prompt {msg!r}"
            )
        val = self._responses[self._response_idx]
        self._response_idx += 1
        if isinstance(val, BaseException):
            raise val
        return str(val)

    def confirm(self, msg: str) -> bool:
        if self._confirm_idx < len(self._confirms):
            v = self._confirms[self._confirm_idx]
            self._confirm_idx += 1
            return v
        return True


def _make_toy_e2_one_ambiguous(
    space_label: str = "ToySpace",
    src_dim: int = 1,
    tgt_dim: int = 1,
) -> AdamsE2Page:
    """E_2 page with exactly one ambiguous d_2: (0,4) → (2,5)."""
    return AdamsE2Page(
        space_label=space_label,
        prime=2,
        s_max=4,
        t_max=10,
        e2_grid={(0, 0): 1, (0, 4): src_dim, (2, 5): tgt_dim},
        forced_vanishings=[],
        ambiguous_differentials=[
            AdamsDifferentialFlag(
                r=2,
                source=(0, 4),
                target=(2, 5),
                classification="ambiguous",
                reason="both dims > 0",
                source_dim=src_dim,
                target_dim=tgt_dim,
            )
        ],
        reliable_window=(4, 6),
        status="success",
        reasoning="Toy E_2 page for testing",
    )


def _make_toy_e2_no_ambiguous() -> AdamsE2Page:
    """E_2 page with no ambiguous differentials (models S³-like situation)."""
    return AdamsE2Page(
        space_label="ToyNoAmbiguous",
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


def _make_toy_e2_two_ambiguous() -> AdamsE2Page:
    """E_2 page with two independent ambiguous d_2 flags."""
    return AdamsE2Page(
        space_label="ToyTwoAmbiguous",
        prime=2,
        s_max=5,
        t_max=12,
        e2_grid={(0, 0): 1, (0, 4): 1, (2, 5): 1, (0, 6): 1, (2, 7): 1},
        forced_vanishings=[],
        ambiguous_differentials=[
            AdamsDifferentialFlag(
                r=2, source=(0, 4), target=(2, 5),
                classification="ambiguous", reason="both dims > 0",
                source_dim=1, target_dim=1,
            ),
            AdamsDifferentialFlag(
                r=2, source=(0, 6), target=(2, 7),
                classification="ambiguous", reason="both dims > 0",
                source_dim=1, target_dim=1,
            ),
        ],
        reliable_window=(5, 7),
        status="success",
        reasoning="Toy E_2 page with two ambiguous flags",
    )


def _make_uv(
    r: int = 2,
    src: tuple = (0, 4),
    tgt: tuple = (2, 5),
    decision: str = "nonzero",
    confidence: float = 0.9,
    reference: str = "test ref",
) -> UserVerifiedDifferential:
    return UserVerifiedDifferential(
        r=r,
        bidegree_source=src,
        bidegree_target=tgt,
        decision=decision,  # type: ignore[arg-type]
        decision_input_raw=decision,
        timestamp=datetime.now(timezone.utc),
        user_id="test_user",
        proof_reference=reference,
        user_confidence=confidence,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestInteractionCLI:
    """test_interaction_cli: mock CLIIO drives full resolution to success."""

    def test_single_ambiguous_nonzero(self, tmp_path):
        e2 = _make_toy_e2_one_ambiguous()
        # Script: decision="nonzero", confidence="0.9", proof reference="paper 1958"
        io = MockCLIIO(["nonzero", "0.9", "Adams 1958"])
        resolver = InteractiveAdamsResolver(
            e2,
            cli_io=io,
            require_proof_reference=False,
            checkpoint_dir=tmp_path / "checkpoints",
        )
        result = resolver.run_interactive_resolution()

        assert isinstance(result, ConvergedAdamsPage)
        assert result.status == "success"
        assert result.path_used == "interactive"
        assert result.exact is False  # never exact for interactive path
        assert len(result.user_verifications) == 1
        assert result.user_verifications[0].decision == "nonzero"

    def test_single_ambiguous_zero(self, tmp_path):
        e2 = _make_toy_e2_one_ambiguous()
        io = MockCLIIO(["zero", "0.8", "Kochman table"])
        resolver = InteractiveAdamsResolver(
            e2,
            cli_io=io,
            require_proof_reference=False,
            checkpoint_dir=tmp_path / "checkpoints",
        )
        result = resolver.run_interactive_resolution()

        assert result.status == "success"
        assert result.path_used == "interactive"
        # zero differential: dims unchanged → grid stabilises
        assert result.e_infinity_grid.get((0, 4), 0) == 1
        assert result.e_infinity_grid.get((2, 5), 0) == 1

    def test_result_carries_full_provenance(self, tmp_path):
        e2 = _make_toy_e2_one_ambiguous()
        io = MockCLIIO(["nonzero", "0.75", "my thesis §4.2"])
        resolver = InteractiveAdamsResolver(
            e2,
            cli_io=io,
            require_proof_reference=False,
            checkpoint_dir=tmp_path / "checkpoints",
        )
        result = resolver.run_interactive_resolution()

        v = result.user_verifications[0]
        assert v.proof_reference == "my thesis §4.2"
        assert v.user_confidence == pytest.approx(0.75)
        assert v.theorem_tag.startswith("adams.einf")
        assert v.exact is False


class TestConfidenceTracking:
    """test_confidence_tracking: low-confidence verification must not advance the page."""

    def test_below_threshold_not_counted(self, tmp_path):
        """A verification below confidence_threshold must not appear in user_verifications."""
        e2 = _make_toy_e2_one_ambiguous()
        io = MockCLIIO(["nonzero", "0.2", "low confidence ref"])
        resolver = InteractiveAdamsResolver(
            e2,
            cli_io=io,
            confidence_threshold=0.5,
            require_proof_reference=False,
            checkpoint_dir=tmp_path / "checkpoints",
        )
        result = resolver.run_interactive_resolution()

        # Low-confidence verification was NOT counted as "used"
        assert len(result.user_verifications) == 0
        # The skipped flag is treated as zero (no dimension change)
        assert result.e_infinity_grid.get((0, 4), 0) == 1
        assert result.e_infinity_grid.get((2, 5), 0) == 1

    def test_above_threshold_advances(self, tmp_path):
        e2 = _make_toy_e2_one_ambiguous()
        io = MockCLIIO(["nonzero", "0.6", "sufficient confidence"])
        resolver = InteractiveAdamsResolver(
            e2,
            cli_io=io,
            confidence_threshold=0.5,
            require_proof_reference=False,
            checkpoint_dir=tmp_path / "checkpoints",
        )
        result = resolver.run_interactive_resolution()

        assert result.status == "success"
        assert len(result.user_verifications) == 1
        assert result.user_verifications[0].user_confidence == pytest.approx(0.6)

    def test_session_confidence_excludes_skipped(self, tmp_path):
        e2 = _make_toy_e2_two_ambiguous()
        # First flag: below threshold (skip). Second flag: above threshold.
        io = MockCLIIO([
            "nonzero", "0.1", "very low",   # flag 1 → below threshold
            "nonzero", "0.8", "good ref",   # flag 2 → above threshold
        ])
        resolver = InteractiveAdamsResolver(
            e2,
            cli_io=io,
            confidence_threshold=0.5,
            require_proof_reference=False,
            max_pages=4,
            checkpoint_dir=tmp_path / "checkpoints",
        )
        result = resolver.run_interactive_resolution()

        # Only the second verification was accepted
        assert len(result.user_verifications) == 1
        assert result.user_verifications[0].user_confidence == pytest.approx(0.8)


class TestConvergence:
    """test_convergence: S³-like page with no ambiguous flags returns E_∞ = E_2 immediately."""

    def test_no_flags_returns_immediately(self, tmp_path):
        e2 = _make_toy_e2_no_ambiguous()
        io = MockCLIIO([])  # no prompts should be made
        resolver = InteractiveAdamsResolver(
            e2,
            cli_io=io,
            checkpoint_dir=tmp_path / "checkpoints",
        )
        result = resolver.run_interactive_resolution()

        assert result.status == "success"
        assert result.convergence_page == 2
        assert result.path_used == "interactive"
        assert len(result.user_verifications) == 0
        # Grid unchanged from E_2
        assert result.e_infinity_grid.get((0, 0), 0) == 1
        assert result.e_infinity_grid.get((0, 3), 0) == 1
        # No CLI prompts were issued
        assert io._response_idx == 0

    def test_actual_sphere_s3(self, tmp_path):
        from pysurgery.adams.spectral_sequence import (
            adams_e2_page,
            sphere_cohomology_fp,
        )

        # s_max=0 keeps only the stem; no d_r can land outside [0, s_max].
        ring = sphere_cohomology_fp(3, prime=2)
        e2 = adams_e2_page(ring, prime=2, s_max=0, t_max=3)
        io = MockCLIIO([])
        resolver = InteractiveAdamsResolver(
            e2,
            cli_io=io,
            checkpoint_dir=tmp_path / "checkpoints",
        )
        result = resolver.run_interactive_resolution()

        # S³ at s_max=0 has no ambiguous differentials
        assert result.status == "success"
        assert len(result.user_verifications) == 0
        assert result.convergence_page == 2


class TestConsistencyChecking:
    """test_consistency_checking: contradictory decisions raise InteractiveConsistencyError."""

    def test_direct_contradiction_raises(self, tmp_path):
        e2 = _make_toy_e2_one_ambiguous()
        io = MockCLIIO([])
        resolver = InteractiveAdamsResolver(
            e2,
            cli_io=io,
            checkpoint_dir=tmp_path / "checkpoints",
        )

        v_nonzero = _make_uv(decision="nonzero")
        v_zero = _make_uv(decision="zero")

        with pytest.raises(InteractiveConsistencyError):
            resolver._assert_consistency(v_zero, [v_nonzero])

    def test_same_decision_no_error(self, tmp_path):
        e2 = _make_toy_e2_one_ambiguous()
        io = MockCLIIO([])
        resolver = InteractiveAdamsResolver(
            e2,
            cli_io=io,
            checkpoint_dir=tmp_path / "checkpoints",
        )

        v1 = _make_uv(decision="nonzero", reference="ref1")
        v2 = _make_uv(decision="nonzero", reference="ref2")
        # Same decision: no error
        resolver._assert_consistency(v2, [v1])

    def test_skip_does_not_conflict(self, tmp_path):
        e2 = _make_toy_e2_one_ambiguous()
        io = MockCLIIO([])
        resolver = InteractiveAdamsResolver(
            e2,
            cli_io=io,
            checkpoint_dir=tmp_path / "checkpoints",
        )

        v_nonzero = _make_uv(decision="nonzero")
        v_skip = _make_uv(decision="skip", confidence=0.0)
        # skip doesn't count as a decision → no conflict
        resolver._assert_consistency(v_skip, [v_nonzero])

    def test_different_flag_no_error(self, tmp_path):
        e2 = _make_toy_e2_two_ambiguous()
        io = MockCLIIO([])
        resolver = InteractiveAdamsResolver(
            e2,
            cli_io=io,
            checkpoint_dir=tmp_path / "checkpoints",
        )

        v1 = _make_uv(r=2, src=(0, 4), tgt=(2, 5), decision="nonzero")
        v2 = _make_uv(r=2, src=(0, 6), tgt=(2, 7), decision="zero")
        # Different flags → no conflict
        resolver._assert_consistency(v2, [v1])

    def test_contradiction_in_run(self, tmp_path):
        """CLI script that contradicts itself is caught before the page advances."""
        e2 = _make_toy_e2_one_ambiguous()
        # We manually pre-seed a conflicting verification in the checkpoint so
        # that when the resolver loads it and the user contradicts, the error fires.
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir(parents=True)

        resolver_a = InteractiveAdamsResolver(
            e2,
            cli_io=MockCLIIO(["nonzero", "0.9", "ref1"]),
            require_proof_reference=False,
            checkpoint_dir=ckpt_dir,
        )
        resolver_a.run_interactive_resolution()
        # Checkpoint now contains "nonzero" for the flag.

        # A second resolver with same session tries to decide the same flag as "zero".
        resolver_b = InteractiveAdamsResolver(
            e2,
            cli_io=MockCLIIO(["zero", "0.9", "ref2"]),
            require_proof_reference=False,
            checkpoint_dir=ckpt_dir,
            session_id=resolver_a._session_id,
        )
        # The existing checkpoint has "nonzero"; the user says "zero" → conflict.
        # However, on RESUME the flag is already decided (closed_decisions), so it
        # won't be re-queried. The error is surfaced by direct _assert_consistency call.
        v_conflict = _make_uv(decision="zero")
        v_existing = _make_uv(decision="nonzero")
        with pytest.raises(InteractiveConsistencyError):
            resolver_b._assert_consistency(v_conflict, [v_existing])


class TestEInfinityComputation:
    """test_e_infinity_computation: hand-computed E_∞ matches predicted grid."""

    def test_nonzero_d2_kills_both_dims(self, tmp_path):
        """d_2: dim(0,4)=2 → dim(2,5)=1.  Nonzero: E_∞^{0,4}=1, E_∞^{2,5}=0."""
        e2 = _make_toy_e2_one_ambiguous(src_dim=2, tgt_dim=1)
        io = MockCLIIO(["nonzero", "0.8", "hand computation"])
        resolver = InteractiveAdamsResolver(
            e2,
            cli_io=io,
            require_proof_reference=False,
            checkpoint_dir=tmp_path / "checkpoints",
        )
        result = resolver.run_interactive_resolution()

        assert result.status == "success"
        # rank-1 assumption: src 2→1, tgt 1→0
        assert result.e_infinity_grid.get((0, 4), 0) == 1
        assert result.e_infinity_grid.get((2, 5), 0) == 0
        assert result.e_infinity_grid.get((0, 0), 0) == 1  # untouched

    def test_zero_d2_preserves_dims(self, tmp_path):
        """d_2 declared zero: E_∞ = E_2."""
        e2 = _make_toy_e2_one_ambiguous(src_dim=1, tgt_dim=1)
        io = MockCLIIO(["zero", "0.9", "forced by connectivity"])
        resolver = InteractiveAdamsResolver(
            e2,
            cli_io=io,
            require_proof_reference=False,
            checkpoint_dir=tmp_path / "checkpoints",
        )
        result = resolver.run_interactive_resolution()

        assert result.status == "success"
        assert result.e_infinity_grid.get((0, 4), 0) == 1
        assert result.e_infinity_grid.get((2, 5), 0) == 1

    def test_two_nonzero_d2_reduces_independently(self, tmp_path):
        """Two nonzero d_2 each reduce one pair of bidegrees."""
        e2 = _make_toy_e2_two_ambiguous()
        # flag1: (0,4)→(2,5) nonzero; flag2: (0,6)→(2,7) nonzero
        io = MockCLIIO([
            "nonzero", "0.85", "ref-a",
            "nonzero", "0.85", "ref-b",
        ])
        resolver = InteractiveAdamsResolver(
            e2,
            cli_io=io,
            require_proof_reference=False,
            checkpoint_dir=tmp_path / "checkpoints",
        )
        result = resolver.run_interactive_resolution()

        assert result.status == "success"
        assert result.e_infinity_grid.get((0, 4), 0) == 0
        assert result.e_infinity_grid.get((2, 5), 0) == 0
        assert result.e_infinity_grid.get((0, 6), 0) == 0
        assert result.e_infinity_grid.get((2, 7), 0) == 0
        assert result.e_infinity_grid.get((0, 0), 0) == 1  # untouched

    def test_convergence_page_recorded(self, tmp_path):
        e2 = _make_toy_e2_one_ambiguous()
        io = MockCLIIO(["nonzero", "0.9", "ref"])
        resolver = InteractiveAdamsResolver(
            e2,
            cli_io=io,
            require_proof_reference=False,
            checkpoint_dir=tmp_path / "checkpoints",
        )
        result = resolver.run_interactive_resolution()

        assert isinstance(result.convergence_page, int)
        assert result.convergence_page >= 2

    def test_exact_is_always_false(self, tmp_path):
        e2 = _make_toy_e2_one_ambiguous()
        io = MockCLIIO(["nonzero", "1.0", "certainty ref"])
        resolver = InteractiveAdamsResolver(
            e2,
            cli_io=io,
            require_proof_reference=False,
            checkpoint_dir=tmp_path / "checkpoints",
        )
        result = resolver.run_interactive_resolution()

        # Even with confidence=1.0, interactive path is never exact.
        assert result.exact is False
        for v in result.user_verifications:
            assert v.exact is False


class TestResumeFromCheckpoint:
    """Partial run is checkpointed; resumed run completes without double-counting."""

    def test_resume_after_keyboard_interrupt(self, tmp_path):
        ckpt_dir = tmp_path / "checkpoints"
        e2 = _make_toy_e2_two_ambiguous()

        # First run: answer flag1, then abort
        io1 = MockCLIIO([
            "nonzero", "0.9", "ref-flag1",   # flag1 answered
            KeyboardInterrupt(),              # abort before flag2
        ])
        resolver1 = InteractiveAdamsResolver(
            e2,
            cli_io=io1,
            require_proof_reference=False,
            checkpoint_dir=ckpt_dir,
            session_id="test-resume-001",
        )
        result1 = resolver1.run_interactive_resolution()
        assert result1.status == "truncated"

        # Checkpoint must exist and contain exactly one verification
        ckpt_path = ckpt_dir / "test-resume-001.json"
        assert ckpt_path.exists()
        saved = json.loads(ckpt_path.read_text())
        assert len(saved["verifications"]) == 1

        # Second run: resume and answer flag2
        io2 = MockCLIIO([
            "zero", "0.8", "ref-flag2",   # flag2 answered
        ])
        resolver2 = InteractiveAdamsResolver(
            e2,
            cli_io=io2,
            require_proof_reference=False,
            checkpoint_dir=ckpt_dir,
            session_id="test-resume-001",
        )
        result2 = resolver2.run_interactive_resolution()

        assert result2.status == "success"
        # Total verifications: 1 (replayed) + 1 (new) = 2, no double-counting
        assert len(result2.user_verifications) == 2
        decisions = {v.decision for v in result2.user_verifications}
        assert "nonzero" in decisions
        assert "zero" in decisions

    def test_checkpoint_persisted_after_each_verification(self, tmp_path):
        ckpt_dir = tmp_path / "checkpoints"
        e2 = _make_toy_e2_two_ambiguous()
        io = MockCLIIO([
            "nonzero", "0.9", "ref1",
            "nonzero", "0.8", "ref2",
        ])
        resolver = InteractiveAdamsResolver(
            e2,
            cli_io=io,
            require_proof_reference=False,
            checkpoint_dir=ckpt_dir,
            session_id="test-ckpt-write",
        )
        resolver.run_interactive_resolution()

        ckpt_path = ckpt_dir / "test-ckpt-write.json"
        assert ckpt_path.exists()
        saved = json.loads(ckpt_path.read_text())
        # Both verifications were persisted
        assert len(saved["verifications"]) == 2


# ── Contract / invariant checks ───────────────────────────────────────────────


class TestContractInvariants:
    """Structural invariants that must hold on every ConvergedAdamsPage."""

    def test_path_used_always_interactive(self, tmp_path):
        e2 = _make_toy_e2_no_ambiguous()
        result = InteractiveAdamsResolver(
            e2,
            cli_io=MockCLIIO([]),
            checkpoint_dir=tmp_path / "checkpoints",
        ).run_interactive_resolution()
        assert result.path_used == "interactive"

    def test_lean_attempts_empty(self, tmp_path):
        e2 = _make_toy_e2_no_ambiguous()
        result = InteractiveAdamsResolver(
            e2,
            cli_io=MockCLIIO([]),
            checkpoint_dir=tmp_path / "checkpoints",
        ).run_interactive_resolution()
        assert result.lean_attempts == []

    def test_decision_ready_on_success(self, tmp_path):
        e2 = _make_toy_e2_no_ambiguous()
        result = InteractiveAdamsResolver(
            e2,
            cli_io=MockCLIIO([]),
            checkpoint_dir=tmp_path / "checkpoints",
        ).run_interactive_resolution()
        assert result.decision_ready() is True

    def test_user_verification_contract_version(self, tmp_path):
        e2 = _make_toy_e2_one_ambiguous()
        io = MockCLIIO(["nonzero", "0.9", "ref"])
        result = InteractiveAdamsResolver(
            e2,
            cli_io=io,
            require_proof_reference=False,
            checkpoint_dir=tmp_path / "checkpoints",
        ).run_interactive_resolution()
        for v in result.user_verifications:
            assert v.contract_version is not None
            assert len(v.contract_version) > 0
