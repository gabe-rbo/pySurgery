"""Tests for LeanFormalAdamsResolver (Path B).

Covers:
    test_lean_script_generation      — deterministic SHA-256; file on disk
    test_lean_timeout_enforcement    — TimeoutExpired → result="timeout", timeout kwarg forwarded
    test_partial_proof_on_timeout    — timed-out attempt has exact=False, no proof_certificate
    test_proof_certificate_parsing   — exit 0, theorem in stdout → exact=True, cert not None
    test_budget_exhaustion           — only N calls when budget exhausted before all flags done
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pysurgery.adams.spectral_sequence import AdamsDifferentialFlag, AdamsE2Page
from pysurgery.adams.e_infinity_resolver import ResolvingPage
from pysurgery.adams.lean_resolver import LeanFormalAdamsResolver


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_flag(r: int = 2, src: tuple = (1, 3), tgt: tuple = (3, 5)) -> AdamsDifferentialFlag:
    return AdamsDifferentialFlag(
        r=r,
        source=src,
        target=tgt,
        classification="ambiguous",
        reason="test flag",
        source_dim=1,
        target_dim=1,
    )


def _make_e2_page(flags: list[AdamsDifferentialFlag] | None = None) -> AdamsE2Page:
    if flags is None:
        flags = [_make_flag()]
    grid = {}
    for f in flags:
        grid[f.source] = 1
        grid[f.target] = 1
    grid[(0, 0)] = 1
    return AdamsE2Page(
        space_label="TestSpace",
        prime=2,
        s_max=5,
        t_max=10,
        e2_grid=grid,
        ambiguous_differentials=flags,
        reliable_window=(5, 5),
        status="success",
        reasoning="test fixture",
        exact=False,
        theorem_tag="adams.e2.ext_steenrod",
    )


def _mock_version_ok() -> MagicMock:
    m = MagicMock()
    m.returncode = 0
    m.stdout = "Lean (version 4.0.0)\n"
    m.stderr = ""
    return m


def _make_resolver(
    e2_page: AdamsE2Page,
    tmp_path: Path,
    *,
    per_diff_timeout_sec: int = 300,
    total_budget_sec: int = 15_000,
) -> LeanFormalAdamsResolver:
    """Construct resolver with mocked version check."""
    with patch("pysurgery.adams.lean_resolver.subprocess.run", return_value=_mock_version_ok()):
        return LeanFormalAdamsResolver(
            e2_page,
            lean_binary="lean",
            per_diff_timeout_sec=per_diff_timeout_sec,
            total_budget_sec=total_budget_sec,
            artefact_dir=tmp_path / "artefacts",
        )


# ── test_lean_script_generation ───────────────────────────────────────────────


class TestLeanScriptGeneration:
    def test_deterministic_sha256(self, tmp_path: Path) -> None:
        """Same flag + page inputs produce the same SHA-256 and file content."""
        flag = _make_flag()
        e2 = _make_e2_page([flag])
        resolver = _make_resolver(e2, tmp_path)

        page = ResolvingPage(r=2, grid={(1, 3): 1, (3, 5): 1, (0, 0): 1}, open_flags=[flag])

        script1 = resolver.generate_lean_proof_script(flag, page)
        script2 = resolver.generate_lean_proof_script(flag, page)

        assert script1.sha256 == script2.sha256, "SHA-256 must be deterministic"
        assert script1.path == script2.path, "Path must be the same for same content"

    def test_file_written_to_artefact_dir(self, tmp_path: Path) -> None:
        """The generated .lean file must exist in artefact_dir."""
        flag = _make_flag()
        e2 = _make_e2_page([flag])
        resolver = _make_resolver(e2, tmp_path)

        page = ResolvingPage(r=2, grid={(1, 3): 1, (3, 5): 1, (0, 0): 1}, open_flags=[flag])
        script = resolver.generate_lean_proof_script(flag, page)

        assert script.path.exists(), "Script file must be written to disk"
        assert script.path.suffix == ".lean"
        assert script.sha256 in script.path.stem

    def test_content_contains_flag_info(self, tmp_path: Path) -> None:
        """Generated script must reference the bidegree and page."""
        flag = _make_flag(r=2, src=(1, 3), tgt=(3, 5))
        e2 = _make_e2_page([flag])
        resolver = _make_resolver(e2, tmp_path)

        page = ResolvingPage(r=2, grid={(1, 3): 1, (3, 5): 1, (0, 0): 1}, open_flags=[flag])
        script = resolver.generate_lean_proof_script(flag, page)

        content = script.path.read_text()
        assert "1" in content and "3" in content, "Bidegrees must appear in script"
        assert "theorem" in content.lower(), "Script must contain a Lean theorem"


# ── test_lean_timeout_enforcement ─────────────────────────────────────────────


class TestLeanTimeoutEnforcement:
    def test_timeout_result_on_timeout_expired(self, tmp_path: Path) -> None:
        """subprocess.TimeoutExpired must map to result='timeout'."""
        flag = _make_flag()
        e2 = _make_e2_page([flag])
        resolver = _make_resolver(e2, tmp_path)

        page = ResolvingPage(r=2, grid={(1, 3): 1, (3, 5): 1, (0, 0): 1}, open_flags=[flag])
        script = resolver.generate_lean_proof_script(flag, page)

        te = subprocess.TimeoutExpired(cmd=["lean"], timeout=1, output="", stderr="")
        with patch("pysurgery.adams.lean_resolver.subprocess.run", side_effect=te):
            attempt = resolver.run_lean_proof_search(script, timeout_sec=1)

        assert attempt.result == "timeout"
        assert attempt.exact is False

    def test_timeout_kwarg_forwarded(self, tmp_path: Path) -> None:
        """The timeout kwarg must equal the timeout_sec argument."""
        flag = _make_flag()
        e2 = _make_e2_page([flag])
        resolver = _make_resolver(e2, tmp_path)

        page = ResolvingPage(r=2, grid={(1, 3): 1, (3, 5): 1, (0, 0): 1}, open_flags=[flag])
        script = resolver.generate_lean_proof_script(flag, page)

        te = subprocess.TimeoutExpired(cmd=["lean"], timeout=42, output="", stderr="")
        with patch("pysurgery.adams.lean_resolver.subprocess.run", side_effect=te) as mock_run:
            resolver.run_lean_proof_search(script, timeout_sec=42)

        _, kwargs = mock_run.call_args
        assert kwargs.get("timeout") == 42, "timeout kwarg must be forwarded exactly"


# ── test_partial_proof_on_timeout ─────────────────────────────────────────────


class TestPartialProofOnTimeout:
    def test_timeout_attempt_has_no_certificate(self, tmp_path: Path) -> None:
        """Timed-out attempt must have exact=False and proof_certificate=None."""
        flag = _make_flag()
        e2 = _make_e2_page([flag])
        resolver = _make_resolver(e2, tmp_path)

        page = ResolvingPage(r=2, grid={(1, 3): 1, (3, 5): 1, (0, 0): 1}, open_flags=[flag])
        script = resolver.generate_lean_proof_script(flag, page)

        te = subprocess.TimeoutExpired(cmd=["lean"], timeout=5, output="partial output", stderr="")
        with patch("pysurgery.adams.lean_resolver.subprocess.run", side_effect=te):
            attempt = resolver.run_lean_proof_search(script, timeout_sec=5)

        assert attempt.result == "timeout"
        assert attempt.exact is False
        assert attempt.proof_certificate is None

    def test_timeout_attempt_has_wall_seconds(self, tmp_path: Path) -> None:
        """wall_seconds must be recorded even on timeout."""
        flag = _make_flag()
        e2 = _make_e2_page([flag])
        resolver = _make_resolver(e2, tmp_path)

        page = ResolvingPage(r=2, grid={(1, 3): 1, (3, 5): 1, (0, 0): 1}, open_flags=[flag])
        script = resolver.generate_lean_proof_script(flag, page)

        te = subprocess.TimeoutExpired(cmd=["lean"], timeout=5, output="", stderr="")
        with patch("pysurgery.adams.lean_resolver.subprocess.run", side_effect=te):
            attempt = resolver.run_lean_proof_search(script, timeout_sec=5)

        assert attempt.wall_seconds >= 0.0

    def test_timeout_attempt_has_correct_timeout_sec(self, tmp_path: Path) -> None:
        """timeout_sec field must record the requested timeout."""
        flag = _make_flag()
        e2 = _make_e2_page([flag])
        resolver = _make_resolver(e2, tmp_path)

        page = ResolvingPage(r=2, grid={(1, 3): 1, (3, 5): 1, (0, 0): 1}, open_flags=[flag])
        script = resolver.generate_lean_proof_script(flag, page)

        te = subprocess.TimeoutExpired(cmd=["lean"], timeout=7, output="", stderr="")
        with patch("pysurgery.adams.lean_resolver.subprocess.run", side_effect=te):
            attempt = resolver.run_lean_proof_search(script, timeout_sec=7)

        assert attempt.timeout_sec == 7


# ── test_proof_certificate_parsing ────────────────────────────────────────────


class TestProofCertificateParsing:
    def _mock_proven(self, stdout: str) -> MagicMock:
        m = MagicMock()
        m.returncode = 0
        m.stdout = stdout
        m.stderr = ""
        return m

    def test_proven_result_and_exact(self, tmp_path: Path) -> None:
        """Exit 0 with theorem in stdout → result='proven', exact=True."""
        flag = _make_flag()
        e2 = _make_e2_page([flag])
        resolver = _make_resolver(e2, tmp_path)

        page = ResolvingPage(r=2, grid={(1, 3): 1, (3, 5): 1, (0, 0): 1}, open_flags=[flag])
        script = resolver.generate_lean_proof_script(flag, page)

        stdout = "theorem adams_diff_r2_from_1_3_to_3_5 : True := by trivial\n"
        with patch("pysurgery.adams.lean_resolver.subprocess.run", return_value=self._mock_proven(stdout)):
            attempt = resolver.run_lean_proof_search(script, timeout_sec=30)

        assert attempt.result == "proven"
        assert attempt.exact is True

    def test_proof_certificate_not_none(self, tmp_path: Path) -> None:
        """proof_certificate must be populated when proven."""
        flag = _make_flag()
        e2 = _make_e2_page([flag])
        resolver = _make_resolver(e2, tmp_path)

        page = ResolvingPage(r=2, grid={(1, 3): 1, (3, 5): 1, (0, 0): 1}, open_flags=[flag])
        script = resolver.generate_lean_proof_script(flag, page)

        stdout = "theorem my_theorem : True := by trivial\n"
        with patch("pysurgery.adams.lean_resolver.subprocess.run", return_value=self._mock_proven(stdout)):
            attempt = resolver.run_lean_proof_search(script, timeout_sec=30)

        assert attempt.proof_certificate is not None
        assert len(attempt.proof_certificate) > 0

    def test_sorry_forces_undecidable(self, tmp_path: Path) -> None:
        """Exit 0 with 'sorry' in stdout → result='undecidable', exact=False."""
        flag = _make_flag()
        e2 = _make_e2_page([flag])
        resolver = _make_resolver(e2, tmp_path)

        page = ResolvingPage(r=2, grid={(1, 3): 1, (3, 5): 1, (0, 0): 1}, open_flags=[flag])
        script = resolver.generate_lean_proof_script(flag, page)

        stdout = "theorem foo : True := by sorry\n"
        with patch("pysurgery.adams.lean_resolver.subprocess.run", return_value=self._mock_proven(stdout)):
            attempt = resolver.run_lean_proof_search(script, timeout_sec=30)

        assert attempt.result == "undecidable"
        assert attempt.exact is False
        assert attempt.proof_certificate is None

    def test_nonzero_exit_gives_lean_error(self, tmp_path: Path) -> None:
        """Non-zero exit → result='lean_error', exact=False."""
        flag = _make_flag()
        e2 = _make_e2_page([flag])
        resolver = _make_resolver(e2, tmp_path)

        page = ResolvingPage(r=2, grid={(1, 3): 1, (3, 5): 1, (0, 0): 1}, open_flags=[flag])
        script = resolver.generate_lean_proof_script(flag, page)

        m = MagicMock()
        m.returncode = 1
        m.stdout = ""
        m.stderr = "error: unknown identifier"
        with patch("pysurgery.adams.lean_resolver.subprocess.run", return_value=m):
            attempt = resolver.run_lean_proof_search(script, timeout_sec=30)

        assert attempt.result == "lean_error"
        assert attempt.exact is False


# ── test_budget_exhaustion ────────────────────────────────────────────────────


class TestBudgetExhaustion:
    def test_only_two_subprocess_calls_when_budget_exceeded(self, tmp_path: Path) -> None:
        """With 3 flags and budget for 2 calls, only 2 subprocess.run calls are made."""
        flags = [
            _make_flag(r=2, src=(1, 3), tgt=(3, 5)),
            _make_flag(r=2, src=(2, 4), tgt=(4, 6)),
            _make_flag(r=2, src=(3, 5), tgt=(5, 7)),
        ]
        grid = {(0, 0): 1}
        for f in flags:
            grid[f.source] = 1
            grid[f.target] = 1

        e2 = AdamsE2Page(
            space_label="TestSpace",
            prime=2,
            s_max=8,
            t_max=12,
            e2_grid=grid,
            ambiguous_differentials=flags,
            reliable_window=(8, 4),
            status="success",
            reasoning="test fixture",
            exact=False,
            theorem_tag="adams.e2.ext_steenrod",
        )

        # Budget: 2s total, 1s per-diff → 2 flags run before budget exhausted.
        # time.monotonic() is called 4 times per flag (outer t0, inner t0, inner wall, outer t1).
        # Flag 0: outer elapsed = 1.0s → spent = 1.0; remaining = 1.0 > 0 → flag 1 runs.
        # Flag 1: outer elapsed = 1.5s → spent = 2.5; remaining = -0.5 ≤ 0 → flag 2 skipped.
        mock_result = MagicMock()
        mock_result.returncode = 1  # lean_error → no round_decisions → loop breaks
        mock_result.stdout = ""
        mock_result.stderr = "error"

        time_values = [
            0.0,   # flag 0: outer t0
            0.0,   # flag 0: inner t0
            0.5,   # flag 0: inner wall = 0.5 - 0.0
            1.0,   # flag 0: outer spent = 1.0 - 0.0 = 1.0
            1.0,   # flag 1: outer t0
            1.0,   # flag 1: inner t0
            1.5,   # flag 1: inner wall = 1.5 - 1.0
            2.5,   # flag 1: outer spent = 2.5 - 1.0 = 1.5 → total spent = 2.5
        ]

        with patch("pysurgery.adams.lean_resolver.subprocess.run", return_value=_mock_version_ok()):
            resolver = LeanFormalAdamsResolver(
                e2,
                lean_binary="lean",
                per_diff_timeout_sec=1,
                total_budget_sec=2,
                artefact_dir=tmp_path / "artefacts",
            )

        with (
            patch("pysurgery.adams.lean_resolver.subprocess.run", return_value=mock_result) as mock_run,
            patch("pysurgery.adams.lean_resolver.time.monotonic", side_effect=time_values),
        ):
            resolver.resolve_e_infinity_via_lean()

        assert mock_run.call_count == 2, (
            f"Expected exactly 2 subprocess calls, got {mock_run.call_count}"
        )

    def test_three_attempts_total_with_budget_exhaustion(self, tmp_path: Path) -> None:
        """Even when budget runs out, all 3 flags produce an attempt record."""
        flags = [
            _make_flag(r=2, src=(1, 3), tgt=(3, 5)),
            _make_flag(r=2, src=(2, 4), tgt=(4, 6)),
            _make_flag(r=2, src=(3, 5), tgt=(5, 7)),
        ]
        grid = {(0, 0): 1}
        for f in flags:
            grid[f.source] = 1
            grid[f.target] = 1

        e2 = AdamsE2Page(
            space_label="TestSpace",
            prime=2,
            s_max=8,
            t_max=12,
            e2_grid=grid,
            ambiguous_differentials=flags,
            reliable_window=(8, 4),
            status="success",
            reasoning="test fixture",
            exact=False,
            theorem_tag="adams.e2.ext_steenrod",
        )

        mock_result = MagicMock()
        mock_result.returncode = 1  # lean_error → no round_decisions → loop breaks
        mock_result.stdout = ""
        mock_result.stderr = "error"

        time_values = [0.0, 1.5, 1.5, 2.5]

        with patch("pysurgery.adams.lean_resolver.subprocess.run", return_value=_mock_version_ok()):
            resolver = LeanFormalAdamsResolver(
                e2,
                lean_binary="lean",
                per_diff_timeout_sec=1,
                total_budget_sec=2,
                artefact_dir=tmp_path / "artefacts",
            )

        with (
            patch("pysurgery.adams.lean_resolver.subprocess.run", return_value=mock_result),
            patch("pysurgery.adams.lean_resolver.time.monotonic", side_effect=time_values),
        ):
            result = resolver.resolve_e_infinity_via_lean()

        assert len(result.lean_attempts) == 3, (
            f"Expected 3 total attempts, got {len(result.lean_attempts)}"
        )

    def test_third_attempt_is_budget_exhausted(self, tmp_path: Path) -> None:
        """The 3rd attempt (budget-exhausted) must have result='undecidable' and correct stderr."""
        flags = [
            _make_flag(r=2, src=(1, 3), tgt=(3, 5)),
            _make_flag(r=2, src=(2, 4), tgt=(4, 6)),
            _make_flag(r=2, src=(3, 5), tgt=(5, 7)),
        ]
        grid = {(0, 0): 1}
        for f in flags:
            grid[f.source] = 1
            grid[f.target] = 1

        e2 = AdamsE2Page(
            space_label="TestSpace",
            prime=2,
            s_max=8,
            t_max=12,
            e2_grid=grid,
            ambiguous_differentials=flags,
            reliable_window=(8, 4),
            status="success",
            reasoning="test fixture",
            exact=False,
            theorem_tag="adams.e2.ext_steenrod",
        )

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "error"

        time_values = [0.0, 1.5, 1.5, 2.5]

        with patch("pysurgery.adams.lean_resolver.subprocess.run", return_value=_mock_version_ok()):
            resolver = LeanFormalAdamsResolver(
                e2,
                lean_binary="lean",
                per_diff_timeout_sec=1,
                total_budget_sec=2,
                artefact_dir=tmp_path / "artefacts",
            )

        with (
            patch("pysurgery.adams.lean_resolver.subprocess.run", return_value=mock_result),
            patch("pysurgery.adams.lean_resolver.time.monotonic", side_effect=time_values),
        ):
            result = resolver.resolve_e_infinity_via_lean()

        third = result.lean_attempts[2]
        assert third.result == "undecidable"
        assert "budget" in third.lean_stderr_tail.lower()

    def test_lean_environment_error_on_missing_binary(self, tmp_path: Path) -> None:
        """LeanEnvironmentError is raised when lean binary is not found."""
        from pysurgery.adams.e_infinity_resolver import LeanEnvironmentError

        flag = _make_flag()
        e2 = _make_e2_page([flag])

        with patch(
            "pysurgery.adams.lean_resolver.subprocess.run",
            side_effect=FileNotFoundError("lean not found"),
        ):
            with pytest.raises(LeanEnvironmentError):
                LeanFormalAdamsResolver(
                    e2,
                    lean_binary="lean_does_not_exist",
                    artefact_dir=tmp_path / "artefacts",
                )
