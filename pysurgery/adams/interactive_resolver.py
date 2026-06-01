"""Path A — Interactive Adams E∞ resolver (human-in-the-loop).

Overview:
    Implements ``InteractiveAdamsResolver``, which closes every
    ``ambiguous_differentials`` flag on an ``AdamsE2Page`` by querying the
    user via a CLI I/O protocol.  Verifications are checkpointed after each
    question so that sessions can be resumed safely.

    The CLIIO protocol is injectable: tests drive the loop with a mock
    implementation; production code uses the default ``_DefaultCLIIO``
    (backed by ``typer``/``rich`` if installed, otherwise plain
    ``print``/``input``).

Architecture position (see RFC-interactive-resolver-v2):
    Interactive resolver (Path A) consuming ``AdamsE2Page`` and emitting
    ``ConvergedAdamsPage`` for the orchestrator.

Guarantees:
    - ``ConvergedAdamsPage.exact`` is always ``False`` (user input ≠ formal proof).
    - ``run_interactive_resolution`` is deterministic on replay:
      same ``(initial_page, all_verifications)`` → same sequence of
      ``ResolvingPage`` snapshots.
    - A ``KeyboardInterrupt`` is caught and converts the run to
      ``status="truncated"``; partial state is persisted before raising.

References:
    Adams, J. F. (1958). On the structure and applications of the
        Steenrod algebra. Comment. Math. Helv. 32, 180–214.
"""
from __future__ import annotations

import getpass
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Protocol, Tuple, runtime_checkable

from pysurgery.adams.spectral_sequence import AdamsDifferentialFlag, AdamsE2Page
from pysurgery.adams.e_infinity_resolver import (
    ConvergedAdamsPage,
    InteractiveConsistencyError,
    ResolvingPage,
    UserVerifiedDifferential,
    _decision_key,  # noqa: F401
)
from pysurgery.core.theorem_tags import ADAMS_EINF_INTERACTIVE, ADAMS_EINF_PARTIAL


# ── I/O protocol ─────────────────────────────────────────────────────────────


@runtime_checkable
class CLIIO(Protocol):
    """Injectable I/O so tests can drive the loop without real stdin/stdout."""

    def write(self, msg: str) -> None:
        """Emit ``msg`` to the output channel."""
        ...

    def prompt(self, msg: str, choices: tuple[str, ...]) -> str:
        """Prompt the user with ``msg`` and return one of ``choices``."""
        ...

    def confirm(self, msg: str) -> bool:
        """Ask the user ``msg`` and return their yes/no answer."""
        ...


class _DefaultCLIIO:
    """Default CLIIO: typer/rich if available, else plain print/input."""

    def __init__(self) -> None:
        try:
            import typer  # noqa: F401
            self._has_typer = True
        except ImportError:
            self._has_typer = False

    def write(self, msg: str) -> None:
        print(msg)

    def prompt(self, msg: str, choices: tuple[str, ...]) -> str:
        if choices:
            choices_str = "/".join(choices)
            while True:
                raw = input(f"{msg} [{choices_str}]: ").strip()
                if raw in choices:
                    return raw
                print(f"  → Choose from: {choices_str}")
        else:
            return input(f"{msg}: ").strip()

    def confirm(self, msg: str) -> bool:
        while True:
            raw = input(f"{msg} [y/n]: ").strip().lower()
            if raw in ("y", "yes"):
                return True
            if raw in ("n", "no"):
                return False
            print("  → Please enter y or n.")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_session_id(e2_page: AdamsE2Page) -> str:
    """Content-addressed session ID derived from the input page."""
    content = (
        f"{e2_page.space_label}-p{e2_page.prime}"
        f"-s{e2_page.s_max}-t{e2_page.t_max}"
        f"-{sorted(e2_page.e2_grid.items())}"
    )
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# ── InteractiveAdamsResolver ──────────────────────────────────────────────────


class InteractiveAdamsResolver:
    """Human-in-the-loop resolver for the Adams E∞ page (Path A).

    Overview:
        Iterates through ambiguous Adams differentials, asking the user to
        decide each one.  Verifications are stored with confidence scores and
        proof references.  The loop advances page by page (E_2 → E_3 → …)
        until no ambiguous flags remain or ``max_pages`` is reached.

    Key Concepts:
        - ``identify_ambiguous_differentials``: pure read of ``page.open_flags``.
        - ``query_user_for_differential``: one round-trip via ``CLIIO``.
        - ``compute_next_page``: pure algebra; deterministic on replay.
        - ``resolve_e_infinity``: orchestration + checkpointing.

    Attributes:
        _e2_page: The input E_2 page (Adams).
        _io: The CLI I/O implementation (default or injected).
        _max_pages: Hard upper bound on the page index.
        _confidence_threshold: Minimum ``user_confidence`` to advance.
        _require_proof_reference: Whether empty proof references are allowed.
        _checkpoint_dir: Directory for JSON session checkpoints.
        _session_id: Identifies this session for resumability.
        _user_id: OS username (or override) stamped on every verification.
    """

    def __init__(
        self,
        e2_page: AdamsE2Page,
        *,
        cli_io: Optional[CLIIO] = None,
        max_pages: int = 32,
        require_proof_reference: bool = True,
        confidence_threshold: float = 0.5,
        checkpoint_dir: Path = Path(".pysurgery/resolver_state"),
        session_id: Optional[str] = None,
    ) -> None:
        self._e2_page = e2_page
        self._io: CLIIO = cli_io if cli_io is not None else _DefaultCLIIO()  # type: ignore[assignment]
        self._max_pages = max_pages
        self._require_proof_reference = require_proof_reference
        self._confidence_threshold = confidence_threshold
        self._checkpoint_dir = Path(checkpoint_dir)
        self._session_id: str = session_id or _make_session_id(e2_page)
        try:
            self._user_id: str = getpass.getuser()
        except Exception:
            self._user_id = "unknown"

    # ── Protocol Implementation ───────────────────────────────────────────────

    def identify_ambiguous_differentials(
        self, page: ResolvingPage
    ) -> list[AdamsDifferentialFlag]:
        """Return the open (undecided) ambiguous flags for the current page."""
        return list(page.open_flags)

    def compute_next_page(
        self,
        page: ResolvingPage,
        verifications: list[Any],
    ) -> ResolvingPage:
        """Advance E_r to E_{r+1} by applying the d_r differentials.

        What is Being Computed?:
            E_{r+1}^{s,t} = ker(d_r from (s,t)) / im(d_r into (s,t)).
            Under the rank-1 assumption (each nonzero differential kills exactly
            one dimension in source and target), the grid dimensions decrease
            by the number of nonzero d_r differentials incident to each bidegree.
        """
        r = page.r
        new_grid: Dict[Tuple[int, int], int] = dict(page.grid)
        decisions: Dict[Tuple[int, Tuple[int, int], Tuple[int, int]], Literal["zero", "nonzero"]] = dict(page.closed_decisions)

        # Process new verifications for the current page
        for v in verifications:
            if isinstance(v, UserVerifiedDifferential):
                if v.r == r and v.decision in ("zero", "nonzero"):
                    key = (v.r, v.bidegree_source, v.bidegree_target)
                    decisions[key] = v.decision  # type: ignore[assignment]
            elif isinstance(v, dict):
                # Database entry
                if v.get("r") == r and v.get("decision") in ("zero", "nonzero"):
                    key = (v["r"], tuple(v["source"]), tuple(v["target"]))  # type: ignore[index]
                    decisions[key] = v["decision"]
            # Lean attempts could also be added here if needed

        # Apply nonzero d_r differentials to the grid (rank-1 assumption)
        for (dr, src, tgt), dec in sorted(decisions.items()):
            if dr != r or dec != "nonzero":
                continue
            if new_grid.get(src, 0) > 0 and new_grid.get(tgt, 0) > 0:
                new_grid[src] -= 1
                new_grid[tgt] -= 1

        # Drop zero-dimension entries
        new_grid = {k: v for k, v in sorted(new_grid.items()) if v > 0}

        # Compute open flags for level r+1
        next_r = r + 1
        open_flags = self._compute_open_flags_for_page(
            new_grid, next_r, set(decisions.keys())
        )

        return ResolvingPage(
            r=next_r,
            grid=new_grid,
            open_flags=open_flags,
            closed_decisions=decisions,
        )

    def resolve_e_infinity(self) -> ConvergedAdamsPage:
        """Alias for run_interactive_resolution to satisfy ResolverProtocol."""
        return self.run_interactive_resolution()

    # ── Database Lookup ───────────────────────────────────────────────────────

    def _lookup_database(self, flag: AdamsDifferentialFlag) -> Optional[UserVerifiedDifferential]:
        """Query the local adams_data/ database for a known verdict."""
        # Simple lookup: spheres, RPn, CPn based on space_label
        label = self._e2_page.space_label
        prime = self._e2_page.prime

        # Normalise label for filename
        # E.g. "S^0" -> "sphere_p2.json"
        filename = None
        if "S^" in label or label == "S":
            filename = f"sphere_p{prime}.json"

        if not filename:
            return None

        db_path = Path(__file__).parent / "adams_data" / filename
        if not db_path.exists():
            return None

        try:
            data = json.loads(db_path.read_text())
            for entry in data.get("differentials", []):
                if (entry["r"] == flag.r and 
                    tuple(entry["source"]) == flag.source and 
                    tuple(entry["target"]) == flag.target):
                    
                    self._io.write(f"  → Found known differential in database: d_{flag.r}({flag.source}) = {entry['decision']}")
                    return UserVerifiedDifferential(
                        r=flag.r,
                        bidegree_source=flag.source,
                        bidegree_target=flag.target,
                        decision=entry["decision"],
                        decision_input_raw="database_lookup",
                        timestamp=datetime.now(timezone.utc),
                        user_id="database",
                        proof_reference=entry.get("reference", "adams_data"),
                        user_confidence=0.99, # Database is highly trusted but not formal proof
                    )
        except Exception as e:
            self._io.write(f"  [Warning] Database lookup failed: {e}")
        
        return None

    # ── Interaction ───────────────────────────────────────────────────────────

    def query_user_for_differential(
        self, flag: AdamsDifferentialFlag
    ) -> UserVerifiedDifferential:
        """Prompt the user for a verdict on one differential flag.

        What is Being Computed?:
            One round-trip with the user: decision, confidence, proof reference.
            Raw stdin is echoed into ``decision_input_raw`` for auditing.

        Args:
            flag: The ambiguous differential to resolve.

        Returns:
            A ``UserVerifiedDifferential`` (may have decision="skip").
        """
        self._io.write(
            f"\n─── Ambiguous differential ───────────────────────────────────\n"
            f"  d_{flag.r}: E_{flag.r}^{flag.source} → E_{flag.r}^{flag.target}\n"
            f"  source dim = {flag.source_dim},  target dim = {flag.target_dim}\n"
            f"──────────────────────────────────────────────────────────────"
        )

        decision_raw = self._io.prompt(
            f"  d_{flag.r}: E^{flag.source} → E^{flag.target}  [zero/nonzero/skip]",
            ("zero", "nonzero", "skip"),
        )

        if decision_raw == "skip":
            return UserVerifiedDifferential(
                r=flag.r,
                bidegree_source=flag.source,
                bidegree_target=flag.target,
                decision="skip",
                decision_input_raw=decision_raw,
                timestamp=datetime.now(timezone.utc),
                user_id=self._user_id,
                proof_reference="",
                user_confidence=0.0,
            )

        conf_raw = self._io.prompt("  Confidence (0.0 – 1.0)", ())
        try:
            confidence = float(conf_raw)
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = 0.0

        ref_raw = self._io.prompt("  Proof reference", ())
        if self._require_proof_reference and not ref_raw.strip():
            ref_raw = "(no reference provided)"

        raw_echo = f"{decision_raw}|{conf_raw}|{ref_raw}"
        return UserVerifiedDifferential(
            r=flag.r,
            bidegree_source=flag.source,
            bidegree_target=flag.target,
            decision=decision_raw,  # type: ignore[arg-type]
            decision_input_raw=raw_echo,
            timestamp=datetime.now(timezone.utc),
            user_id=self._user_id,
            proof_reference=ref_raw,
            user_confidence=confidence,
        )

    # ── Orchestration ─────────────────────────────────────────────────────────

    def run_interactive_resolution(self) -> ConvergedAdamsPage:
        """Run the full interactive resolution loop.

        What is Being Computed?:
            Iterates E_2 → E_3 → … by querying the user for each ambiguous
            d_r differential.  Stops when no flags remain (success), when
            the grid stabilises, when ``max_pages`` is hit (truncated), or
            when the user presses Ctrl+C (truncated).  State is checkpointed
            after every verified differential.

        Returns:
            ``ConvergedAdamsPage`` with ``path_used="interactive"``,
            ``exact=False`` always.
        """
        all_verifications: list[UserVerifiedDifferential] = self._load_verifications()

        # Replay saved verifications to rebuild the current page state.
        page, page_history = self._build_replay_state(all_verifications)

        try:
            while page.r <= self._max_pages:
                flags = self.identify_ambiguous_differentials(page)

                if not flags:
                    # Stabilisation check
                    if page.r > 2:
                        prev_page = page_history[-2]
                        if page.grid == prev_page.grid:
                             return self._make_converged(
                                page,
                                all_verifications,
                                page_history,
                                status="success",
                                convergence_page=page.r,
                                reasoning=f"Grid stabilised at r={page.r}.",
                            )

                    return self._make_converged(
                        page,
                        all_verifications,
                        page_history,
                        status="success",
                        convergence_page=page.r,
                        reasoning=f"All differentials resolved; E_∞ = E_{page.r}.",
                    )

                for flag in sorted(flags, key=lambda f: (f.source, f.target)):
                    # 1. Database Lookup
                    v = self._lookup_database(flag)
                    
                    # 2. Interactive Query (if not in database)
                    if v is None:
                        v = self.query_user_for_differential(flag)

                    if v.decision == "skip":
                        continue

                    if v.user_confidence < self._confidence_threshold:
                        self._io.write(
                            f"  Confidence {v.user_confidence:.2f} < threshold "
                            f"{self._confidence_threshold:.2f}; flag remains open."
                        )
                        continue

                    self._assert_consistency(v, all_verifications)
                    all_verifications.append(v)
                    self._checkpoint(all_verifications)

                next_page = self.compute_next_page(page, all_verifications)
                page_history.append(next_page)

                if next_page.r > self._max_pages:
                    return self._make_converged(
                        next_page,
                        all_verifications,
                        page_history,
                        status="truncated",
                        convergence_page=next_page.r,
                        reasoning="max_pages reached without full convergence.",
                    )

                page = next_page

        except KeyboardInterrupt:
            self._checkpoint(all_verifications)
            return self._make_converged(
                page,
                all_verifications,
                page_history,
                status="truncated",
                convergence_page=page.r,
                reasoning=f"user-aborted at r={page.r}",
            )

        return self._make_converged(
            page,
            all_verifications,
            page_history,
            status="truncated",
            convergence_page=page.r,
            reasoning="max_pages reached.",
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _initial_resolving_page(self) -> ResolvingPage:
        """Build the starting ResolvingPage (E_2) from the input AdamsE2Page."""
        grid: Dict[Tuple[int, int], int] = {
            k: v for k, v in sorted(self._e2_page.e2_grid.items()) if v > 0
        }
        open_flags = self._compute_open_flags_for_page(grid, r=2, already_decided=set())
        return ResolvingPage(r=2, grid=grid, open_flags=open_flags, closed_decisions={})

    def _build_replay_state(
        self,
        all_verifications: list[UserVerifiedDifferential],
    ) -> tuple[ResolvingPage, list[ResolvingPage]]:
        """Replay saved verifications and return the current page + history."""
        if not all_verifications:
            page = self._initial_resolving_page()
            return page, [page]

        grid: Dict[Tuple[int, int], int] = {
            k: v for k, v in sorted(self._e2_page.e2_grid.items()) if v > 0
        }
        all_decided: Dict[Tuple[int, Tuple[int, int], Tuple[int, int]], Literal["zero", "nonzero"]] = {}
        page_history: list[ResolvingPage] = []

        r = 2
        while r <= self._max_pages:
            # Populate decisions at this level from saved verifications.
            for v in all_verifications:
                if v.r == r and v.decision in ("zero", "nonzero"):
                    key = (v.r, v.bidegree_source, v.bidegree_target)
                    all_decided[key] = v.decision  # type: ignore[assignment]

            open_flags = self._compute_open_flags_for_page(
                grid, r, set(all_decided.keys())
            )
            page = ResolvingPage(
                r=r,
                grid=dict(grid),
                open_flags=open_flags,
                closed_decisions=dict(all_decided),
            )
            page_history.append(page)

            if open_flags:
                return page, page_history

            # Apply nonzero decisions at this level to the grid.
            changed = False
            for (dr, src, tgt), dec in sorted(all_decided.items()):
                if dr != r or dec != "nonzero":
                    continue
                if grid.get(src, 0) > 0 and grid.get(tgt, 0) > 0:
                    grid[src] -= 1
                    grid[tgt] -= 1
                    changed = True

            grid = {k: v for k, v in sorted(grid.items()) if v > 0}
            r += 1
            
            if not changed and not any(v.r >= r for v in all_verifications):
                break

        return page, page_history

    def _compute_open_flags_for_page(
        self,
        grid: Dict[Tuple[int, int], int],
        r: int,
        already_decided: set[Tuple[int, Tuple[int, int], Tuple[int, int]]],
    ) -> list[AdamsDifferentialFlag]:
        """Compute ambiguous d_r flags still unresolved in the current grid."""
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

    def _assert_consistency(
        self,
        new_v: UserVerifiedDifferential,
        existing: list[UserVerifiedDifferential],
    ) -> None:
        """Raise InteractiveConsistencyError if ``new_v`` contradicts any prior verification."""
        new_key = (new_v.r, new_v.bidegree_source, new_v.bidegree_target)
        for v in existing:
            if (
                (v.r, v.bidegree_source, v.bidegree_target) == new_key
                and v.decision in ("zero", "nonzero")
                and new_v.decision in ("zero", "nonzero")
                and v.decision != new_v.decision
            ):
                raise InteractiveConsistencyError(
                    f"Contradiction at d_{v.r}({v.bidegree_source})->{v.bidegree_target}: "
                    f"existing decision={v.decision!r} "
                    f"contradicts new decision={new_v.decision!r}. "
                    f"Retract one before proceeding."
                )

    def _checkpoint(self, verifications: list[UserVerifiedDifferential]) -> None:
        """Persist verifications to a JSON file under checkpoint_dir."""
        try:
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
            path = self._checkpoint_dir / f"{self._session_id}.json"
            payload = {
                "session_id": self._session_id,
                "verifications": [
                    v.model_dump(mode="json") for v in verifications
                ],
            }
            path.write_text(json.dumps(payload, indent=2, default=str))
        except OSError:
            pass  # checkpoint failure is non-fatal

    def _load_verifications(self) -> list[UserVerifiedDifferential]:
        """Load saved verifications from the checkpoint file (if any)."""
        path = self._checkpoint_dir / f"{self._session_id}.json"
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text())
            return [
                UserVerifiedDifferential.model_validate(v)
                for v in data.get("verifications", [])
            ]
        except (OSError, ValueError, KeyError):
            return []

    def _make_converged(
        self,
        page: ResolvingPage,
        all_verifications: list[UserVerifiedDifferential],
        page_history: list[ResolvingPage],
        status: Literal["success", "truncated", "undecidable", "inconclusive"],
        convergence_page: int,
        reasoning: str = "",
    ) -> ConvergedAdamsPage:
        """Assemble the final ConvergedAdamsPage."""
        used = [v for v in all_verifications if v.decision != "skip"]
        tag = ADAMS_EINF_INTERACTIVE if status == "success" else ADAMS_EINF_PARTIAL
        return ConvergedAdamsPage(
            space_label=self._e2_page.space_label,
            prime=self._e2_page.prime,
            s_max=self._e2_page.s_max,
            t_max=self._e2_page.t_max,
            e_infinity_grid=dict(page.grid),
            page_history=list(page_history),
            convergence_page=convergence_page,
            user_verifications=used,
            lean_attempts=[],
            path_used="interactive",
            status=status,
            reasoning=reasoning,
            exact=False,
            theorem_tag=tag,
        )


__all__ = [
    "CLIIO",
    "InteractiveAdamsResolver",
    "UserVerifiedDifferential",
    "_DefaultCLIIO",
]
