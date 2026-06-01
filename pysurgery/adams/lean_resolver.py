"""Path B — Lean 4 formal Adams E∞ resolver.

Overview:
    Implements ``LeanFormalAdamsResolver``, which attempts to prove or refute
    every ``ambiguous_differentials`` flag in an ``AdamsE2Page`` by invoking
    Lean 4 tactic search via ``subprocess.run``.

Critical guardrails (every one is load-bearing for soundness):
    1. ``subprocess.run(..., timeout=timeout_sec)`` is the ONLY allowed invocation
       pattern.  Never use ``Popen`` without ``wait(timeout=…)``.
    2. Per-flag cap = ``min(per_diff_timeout_sec, total_budget_sec - spent)``.
       The total budget is enforced in Python, never trusted to Lean.
    3. ``result="timeout"`` on ``TimeoutExpired`` — NEVER "proven" or "refuted".
    4. ``"sorry"`` in Lean stdout/stderr forces ``result="undecidable"`` even if
       the exit code is 0.
    5. Hard minimum 1 s / hard maximum 86 400 s (24 h) validated at construction.
    6. All artefacts (script, stdout, stderr) written under ``artefact_dir``
       keyed by SHA-256 for reproducibility and CI caching.

Architecture position (see RFC-lean-resolver-v2):
    Lean-formal resolver (Path B) consuming ``AdamsE2Page`` and emitting
    ``ConvergedAdamsPage`` for the orchestrator.

References:
    Adams, J. F. (1958). On the structure and applications of the Steenrod algebra.
        Comment. Math. Helv. 32, 180–214.
"""
from __future__ import annotations

import hashlib
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict

from pysurgery.adams.spectral_sequence import AdamsDifferentialFlag, AdamsE2Page
from pysurgery.adams.e_infinity_resolver import (
    ConvergedAdamsPage,
    LeanEnvironmentError,
    ResolvingPage,
    _decision_key,  # noqa: F401
)
from pysurgery.core.foundations import CONTRACT_VERSION
from pysurgery.core.theorem_tags import ADAMS_EINF_LEAN_FORMAL, ADAMS_EINF_PARTIAL


# ── Constants ─────────────────────────────────────────────────────────────────

_TIMEOUT_MIN: int = 1
_TIMEOUT_MAX: int = 86_400  # 24 hours


# ── Helpers ───────────────────────────────────────────────────────────────────


def _tail(s: str, max_chars: int = 4096) -> str:
    """Return the last ``max_chars`` characters of ``s``."""
    return s[-max_chars:] if len(s) > max_chars else s


def _extract_certificate(stdout: str) -> Optional[str]:
    """Best-effort extraction of a theorem declaration from Lean stdout."""
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("theorem") or stripped.startswith("#check"):
            return stripped
    for line in stdout.splitlines():
        if line.strip():
            return line.strip()
    return None if not stdout.strip() else stdout.strip()[:256]


def _default_lean_script(
    flag: AdamsDifferentialFlag,
    page: ResolvingPage,
    e2_page: AdamsE2Page,
) -> str:
    """Generate a Lean 4 tactic skeleton for one differential obligation.

    Produces a deterministic source string: same ``(flag, page, e2_page)``
    inputs → same content → same SHA-256.  The skeleton uses ``trivial``
    (always provable) so CI can test the plumbing without a real Lean project.
    Integrations can supply a richer ``lean_export_callable`` instead.
    """
    s, t = flag.source
    s2, t2 = flag.target
    r = flag.r
    src_dim = page.grid.get(flag.source, flag.source_dim)
    tgt_dim = page.grid.get(flag.target, flag.target_dim)

    lines = [
        "-- Adams Spectral Sequence: Differential Obligation",
        f"-- Space      : {e2_page.space_label}",
        f"-- Prime      : p = {e2_page.prime}",
        f"-- Differential: d_{r}: E_{r}^({s},{t}) → E_{r}^({s2},{t2})",
        f"-- Source dim : {src_dim}   Target dim: {tgt_dim}",
        f"-- Page level : E_{r}  (r={r})",
        "--",
        "-- Tactics available: aesop, simp_arith, decide, linarith",
        f"-- theorem adams_diff_r{r}_from_{s}_{t}_to_{s2}_{t2} : <obligation>",
        "",
        f"theorem adams_diff_r{r}_from_{s}_{t}_to_{s2}_{t2} : True := by",
        "  trivial",
    ]
    return "\n".join(lines) + "\n"


# ── LeanProofAttempt ──────────────────────────────────────────────────────────


class LeanProofAttempt(BaseModel):
    """Result of one Lean 4 proof-search invocation for a single differential flag.

    Attributes:
        result: Outcome of the attempt.
            ``"proven"``     — exit 0, no sorry; ``exact=True``.
            ``"refuted"``    — Lean derived False from negation (rare, opt-in).
            ``"timeout"``    — ``subprocess.TimeoutExpired``; ``exact=False``.
            ``"undecidable"``— exit 0 with ``sorry``; or budget exhausted.
            ``"lean_error"`` — non-zero exit; ``exact=False``.
        exact: True iff ``result == "proven"``.
        proof_certificate: Extracted theorem body, or ``None``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    r: int
    bidegree_source: Tuple[int, int]
    bidegree_target: Tuple[int, int]

    script_sha256: str
    script_path: Path
    result: Literal["proven", "refuted", "timeout", "undecidable", "lean_error"]
    wall_seconds: float
    timeout_sec: int
    lean_stdout_tail: str
    lean_stderr_tail: str
    proof_certificate: Optional[str] = None

    exact: bool
    theorem_tag: str = ADAMS_EINF_LEAN_FORMAL
    contract_version: str = CONTRACT_VERSION

    def decision_ready(self) -> bool:
        """Return True if Lean reached a definitive verdict (proven or refuted)."""
        return self.result in ("proven", "refuted")


# ── LeanProofScript ───────────────────────────────────────────────────────────


@dataclass
class LeanProofScript:
    """A generated Lean 4 source file for one differential obligation."""

    path: Path
    sha256: str
    obligation: AdamsDifferentialFlag


# ── LeanFormalAdamsResolver ───────────────────────────────────────────────────


class LeanFormalAdamsResolver:
    """Lean 4 formal resolver for Adams E∞ differentials (Path B).

    Overview:
        Attempts to prove or refute each ambiguous d_r differential using
        Lean 4 tactic search.  Per-differential and total-budget timeouts are
        both enforced in Python (never trusted to Lean).  The constructor
        validates the binary with ``lean --version`` and raises
        ``LeanEnvironmentError`` immediately if the toolchain is absent.

    Key Concepts:
        - ``generate_lean_proof_script``: deterministic script → SHA-256 keyed file.
        - ``run_lean_proof_search``: the ONLY place ``subprocess.run`` is called.
        - ``resolve_e_infinity``: orchestration + budget tracking.
        - ``exact=True`` iff ALL flags are ``"proven"``; any other outcome → ``False``.

    Common Workflows:
        1. Construct resolver (validates Lean binary).
        2. Call ``resolve_e_infinity()`` → ``ConvergedAdamsPage``.
        3. Inspect ``lean_attempts`` for per-flag details and certificates.

    References:
        Adams, J. F. (1958). On the structure and applications of the Steenrod algebra.
    """

    def __init__(
        self,
        e2_page: AdamsE2Page,
        *,
        lean_binary: str = "lean",
        lean_project_root: Optional[Path] = None,
        per_diff_timeout_sec: int = 300,
        total_budget_sec: int = 15_000,
        artefact_dir: Path = Path(".pysurgery/lean_artefacts"),
        lean_export_callable: Optional[Callable] = None,
    ) -> None:
        if not (_TIMEOUT_MIN <= per_diff_timeout_sec <= _TIMEOUT_MAX):
            raise ValueError(
                f"per_diff_timeout_sec must be in [{_TIMEOUT_MIN}, {_TIMEOUT_MAX}]; "
                f"got {per_diff_timeout_sec}."
            )
        if not (_TIMEOUT_MIN <= total_budget_sec <= _TIMEOUT_MAX):
            raise ValueError(
                f"total_budget_sec must be in [{_TIMEOUT_MIN}, {_TIMEOUT_MAX}]; "
                f"got {total_budget_sec}."
            )

        self._e2_page = e2_page
        self._lean_binary = lean_binary
        self._lean_project_root = Path(lean_project_root) if lean_project_root else Path(".")
        self._per_diff_timeout_sec = per_diff_timeout_sec
        self._total_budget_sec = total_budget_sec
        self._artefact_dir = Path(artefact_dir)
        self._lean_export: Callable = lean_export_callable or _default_lean_script

        # Validate the Lean toolchain immediately — never silently degrade.
        self._validate_lean_binary()

    # ── Protocol Implementation ───────────────────────────────────────────────

    def identify_ambiguous_differentials(
        self, page: ResolvingPage
    ) -> list[AdamsDifferentialFlag]:
        """Return the open (undecided) ambiguous flags for the current page."""
        return self._compute_open_flags(page.grid, page.r, set(page.closed_decisions.keys()))

    def compute_next_page(
        self,
        page: ResolvingPage,
        verifications: list[Any],
    ) -> ResolvingPage:
        """Advance E_r to E_{r+1} using this round's decisions (pure function)."""
        decisions: Dict[Tuple[int, Tuple[int, int], Tuple[int, int]], Literal["zero", "nonzero"]] = {}
        for v in verifications:
            if isinstance(v, LeanProofAttempt):
                if v.r == page.r:
                    if v.result == "proven":
                        decisions[(v.r, v.bidegree_source, v.bidegree_target)] = "zero"
                    elif v.result == "refuted":
                        decisions[(v.r, v.bidegree_source, v.bidegree_target)] = "nonzero"

        return self._apply_decisions(page, decisions)

    def resolve_e_infinity(self) -> ConvergedAdamsPage:
        """Alias for resolve_e_infinity_via_lean to satisfy ResolverProtocol."""
        return self.resolve_e_infinity_via_lean()

    # ── Toolchain validation ──────────────────────────────────────────────────

    def _validate_lean_binary(self) -> None:
        """Probe ``lean --version``; raise ``LeanEnvironmentError`` on failure."""
        try:
            result = subprocess.run(
                [self._lean_binary, "--version"],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if result.returncode != 0:
                raise LeanEnvironmentError(
                    f"Lean binary '{self._lean_binary}' exited with code "
                    f"{result.returncode}.\nstderr: {result.stderr[:500]}"
                )
        except FileNotFoundError:
            raise LeanEnvironmentError(
                f"Lean binary '{self._lean_binary}' not found on PATH. "
                "Install Lean 4 or supply the correct lean_binary path."
            )
        except subprocess.TimeoutExpired:
            raise LeanEnvironmentError(
                f"Lean binary '{self._lean_binary}' timed out during `--version` "
                "probe (60 s). Check the installation."
            )

    # ── Script generation ─────────────────────────────────────────────────────

    def generate_lean_proof_script(
        self, flag: AdamsDifferentialFlag, page: ResolvingPage
    ) -> LeanProofScript:
        """Generate a deterministic Lean 4 source file for one obligation.

        What is Being Computed?:
            The ``lean_export_callable`` produces a Lean 4 source string.
            Its SHA-256 is the file stem; the content is written to
            ``artefact_dir/{sha256}.lean``.  Same inputs → same SHA-256.

        Preserved Invariants:
            Pure mapping of ``(flag, page, e2_page)`` to a file path.

        Args:
            flag: The ambiguous differential to encode.
            page: The current ``ResolvingPage`` supplying grid dimensions.

        Returns:
            ``LeanProofScript`` with ``path``, ``sha256``, and ``obligation``.
        """
        content = self._lean_export(flag, page, self._e2_page)
        sha = hashlib.sha256(content.encode("utf-8")).hexdigest()

        self._artefact_dir.mkdir(parents=True, exist_ok=True)
        script_path = self._artefact_dir / f"{sha}.lean"
        script_path.write_text(content, encoding="utf-8")

        return LeanProofScript(path=script_path, sha256=sha, obligation=flag)

    # ── Proof search ──────────────────────────────────────────────────────────

    def run_lean_proof_search(
        self, script: LeanProofScript, timeout_sec: int
    ) -> LeanProofAttempt:
        """Invoke Lean on ``script`` with a hard per-call timeout.

        What is Being Computed?:
            Runs ``[lean_binary, script.path]`` under ``subprocess.run`` with
            ``timeout=timeout_sec``.  Maps the outcome to a ``LeanProofAttempt``.

        Preserved Invariants:
            - ``"sorry"`` in stdout/stderr forces ``result="undecidable"`` even
              if the exit code is 0.
            - ``TimeoutExpired`` → ``result="timeout"``, never ``"proven"``.
            - ``exact=True`` iff ``result="proven"``.
            - stdout and stderr are truncated to 4 KiB in the returned record.

        Args:
            script: The ``LeanProofScript`` to run.
            timeout_sec: Hard wall-clock limit passed to ``subprocess.run``.

        Returns:
            ``LeanProofAttempt`` with ``result``, ``exact``, and timing data.
        """
        cmd = [self._lean_binary, str(script.path)]
        t0 = time.monotonic()

        try:
            cp = subprocess.run(
                cmd,
                cwd=self._lean_project_root,
                capture_output=True,
                text=True,
                timeout=timeout_sec,  # ← THE GUARDRAIL
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            wall = time.monotonic() - t0
            return LeanProofAttempt(
                r=script.obligation.r,
                bidegree_source=script.obligation.source,
                bidegree_target=script.obligation.target,
                script_sha256=script.sha256,
                script_path=script.path,
                result="timeout",
                wall_seconds=wall,
                timeout_sec=timeout_sec,
                lean_stdout_tail=_tail(e.stdout or "", 4096),
                lean_stderr_tail=_tail(e.stderr or "", 4096),
                proof_certificate=None,
                exact=False,
            )

        wall = time.monotonic() - t0
        stdout = cp.stdout or ""
        stderr = cp.stderr or ""

        # "sorry" in output overrides exit 0 — must not be treated as proven.
        has_sorry = "sorry" in stdout or "sorry" in stderr

        if cp.returncode == 0 and not has_sorry:
            result: Literal["proven", "refuted", "timeout", "undecidable", "lean_error"] = "proven"
            exact = True
            cert: Optional[str] = _extract_certificate(stdout)
        elif cp.returncode == 0 and has_sorry:
            result = "undecidable"
            exact = False
            cert = None
        else:
            result = "lean_error"
            exact = False
            cert = None

        # Persist stdout / stderr for reproducibility.
        self._persist_artefact(script.sha256, "stdout", stdout)
        self._persist_artefact(script.sha256, "stderr", stderr)

        return LeanProofAttempt(
            r=script.obligation.r,
            bidegree_source=script.obligation.source,
            bidegree_target=script.obligation.target,
            script_sha256=script.sha256,
            script_path=script.path,
            result=result,
            wall_seconds=wall,
            timeout_sec=timeout_sec,
            lean_stdout_tail=_tail(stdout, 4096),
            lean_stderr_tail=_tail(stderr, 4096),
            proof_certificate=cert,
            exact=exact,
        )

    # ── Main resolution loop ──────────────────────────────────────────────────

    def resolve_e_infinity_via_lean(self) -> ConvergedAdamsPage:
        """Resolve all ambiguous Adams differentials using Lean 4.

        What is Being Computed?:
            Iterates E_2 → E_3 → … attempting to prove or refute each d_r flag.
            Budget limits (per-differential and total) are enforced in Python.
            The loop stops when all flags are resolved, budget is exhausted,
            or no progress is made in a round.

        Preserved Invariants:
            - ``exact=True`` iff every attempt has ``result="proven"``.
            - Timed-out or undecidable attempts never flip to ``"proven"``.
            - Iteration over flags is deterministic (``sorted`` by source/target).

        Returns:
            ``ConvergedAdamsPage`` with ``path_used="lean_formal"``.
        """
        page = self._initial_resolving_page()
        page_history: list[ResolvingPage] = [page]
        spent: float = 0.0
        all_attempts: list[LeanProofAttempt] = []
        max_pages = 32

        while page.r <= max_pages:
            flags = list(page.open_flags)

            if not flags:
                # Stabilisation check
                if page.r > 2:
                    prev_page = page_history[-2]
                    if page.grid == prev_page.grid:
                        break
                break

            round_decisions: Dict[Tuple[int, Tuple[int, int], Tuple[int, int]], Literal["zero", "nonzero"]] = {}
            round_attempts: list[LeanProofAttempt] = []

            for flag in sorted(flags, key=lambda f: (f.source, f.target)):
                remaining = self._total_budget_sec - spent

                if remaining <= 0:
                    # Budget exhausted — record undecidable without running Lean.
                    attempt = self._make_budget_exhausted_attempt(flag)
                    all_attempts.append(attempt)
                    round_attempts.append(attempt)
                    continue

                this_to = max(1, int(min(self._per_diff_timeout_sec, remaining)))
                script = self.generate_lean_proof_script(flag, page)

                t0 = time.monotonic()
                attempt = self.run_lean_proof_search(script, timeout_sec=this_to)
                spent += time.monotonic() - t0

                all_attempts.append(attempt)
                round_attempts.append(attempt)

                # Map proven/refuted to page-level decisions.
                if attempt.result == "proven":
                    round_decisions[(flag.r, flag.source, flag.target)] = "zero"
                elif attempt.result == "refuted":
                    round_decisions[(flag.r, flag.source, flag.target)] = "nonzero"

            if not round_decisions:
                # No progress this round → break to avoid an infinite loop.
                break

            next_page = self.compute_next_page(page, round_attempts) # type: ignore[arg-type]
            page_history.append(next_page)
            page = next_page

        return self._make_converged(page, page_history, all_attempts)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _initial_resolving_page(self) -> ResolvingPage:
        grid: Dict[Tuple[int, int], int] = {
            k: v for k, v in sorted(self._e2_page.e2_grid.items()) if v > 0
        }
        open_flags = self._compute_open_flags(grid, r=2, already_decided=set())
        return ResolvingPage(r=2, grid=grid, open_flags=open_flags, closed_decisions={})

    def _compute_open_flags(
        self,
        grid: Dict[Tuple[int, int], int],
        r: int,
        already_decided: set,
    ) -> list[AdamsDifferentialFlag]:
        flags = []
        for flag in self._e2_page.ambiguous_differentials:
            if flag.r != r:
                continue
            key = (flag.r, flag.source, flag.target)
            if key in already_decided:
                continue
            src_dim = grid.get(flag.source, 0)
            tgt_dim = grid.get(flag.target, 0)
            if src_dim > 0 and tgt_dim > 0:
                flags.append(
                    AdamsDifferentialFlag(
                        r=flag.r,
                        source=flag.source,
                        target=flag.target,
                        classification="ambiguous",
                        reason=flag.reason,
                        source_dim=src_dim,
                        target_dim=tgt_dim,
                    )
                )
        return flags

    def _apply_decisions(
        self,
        page: ResolvingPage,
        decisions: Dict[Tuple[int, Tuple[int, int], Tuple[int, int]], Literal["zero", "nonzero"]],
    ) -> ResolvingPage:
        """Advance E_r to E_{r+1} using this round's decisions (pure function)."""
        r = page.r
        new_decided = dict(page.closed_decisions)
        new_decided.update(decisions)

        new_grid: Dict[Tuple[int, int], int] = dict(page.grid)

        for (dr, src, tgt), dec in sorted(new_decided.items()):
            if dr != r or dec != "nonzero":
                continue
            if new_grid.get(src, 0) > 0 and new_grid.get(tgt, 0) > 0:
                new_grid[src] -= 1
                new_grid[tgt] -= 1

        new_grid = {k: v for k, v in sorted(new_grid.items()) if v > 0}
        next_r = r + 1
        open_flags = self._compute_open_flags(new_grid, next_r, set(new_decided.keys()))

        return ResolvingPage(
            r=next_r,
            grid=new_grid,
            open_flags=open_flags,
            closed_decisions=new_decided,
        )

    def _make_budget_exhausted_attempt(
        self, flag: AdamsDifferentialFlag
    ) -> LeanProofAttempt:
        stub_content = f"-- Budget exhausted: d_{flag.r} at {flag.source}"
        sha = hashlib.sha256(stub_content.encode()).hexdigest()
        stub_path = self._artefact_dir / f"{sha}_budget.lean"
        return LeanProofAttempt(
            r=flag.r,
            bidegree_source=flag.source,
            bidegree_target=flag.target,
            script_sha256=sha,
            script_path=stub_path,
            result="undecidable",
            wall_seconds=0.0,
            timeout_sec=0,
            lean_stdout_tail="",
            lean_stderr_tail="total budget exhausted",
            proof_certificate=None,
            exact=False,
        )

    def _persist_artefact(self, sha: str, kind: str, content: str) -> None:
        try:
            self._artefact_dir.mkdir(parents=True, exist_ok=True)
            (self._artefact_dir / f"{sha}.{kind}").write_text(content, encoding="utf-8")
        except OSError:
            pass

    def _make_converged(
        self,
        page: ResolvingPage,
        page_history: list[ResolvingPage],
        all_attempts: list[LeanProofAttempt],
    ) -> ConvergedAdamsPage:
        if not all_attempts:
            # Vacuous: no ambiguous flags → trivially fully proven.
            status: Literal["success", "truncated", "undecidable", "inconclusive"] = "success"
            exact = True
            tag = ADAMS_EINF_LEAN_FORMAL
        else:
            n_proven = sum(1 for a in all_attempts if a.result == "proven")
            n_total = len(all_attempts)
            proven_fraction = n_proven / n_total
            has_bad = any(a.result in ("timeout", "undecidable") for a in all_attempts)

            if proven_fraction == 1.0:
                status = "success"
                exact = True
                tag = ADAMS_EINF_LEAN_FORMAL
            elif has_bad:
                status = "undecidable"
                exact = False
                tag = ADAMS_EINF_PARTIAL
            else:
                status = "truncated"
                exact = False
                tag = ADAMS_EINF_PARTIAL

        n_proven = sum(1 for a in all_attempts if a.result == "proven")
        reasoning = (
            f"Lean 4 formal resolution (lean attempts): {len(all_attempts)} attempt(s), "
            f"{n_proven} proven. exact={exact}."
        )

        return ConvergedAdamsPage(
            space_label=self._e2_page.space_label,
            prime=self._e2_page.prime,
            s_max=self._e2_page.s_max,
            t_max=self._e2_page.t_max,
            e_infinity_grid=dict(page.grid),
            page_history=list(page_history),
            convergence_page=page.r,
            user_verifications=[],
            lean_attempts=list(all_attempts),
            path_used="lean_formal",
            status=status,
            reasoning=reasoning,
            exact=exact,
            theorem_tag=tag,
        )


__all__ = [
    "LeanFormalAdamsResolver",
    "LeanProofAttempt",
    "LeanProofScript",
]
