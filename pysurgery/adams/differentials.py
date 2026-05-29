"""Adams spectral sequence differentials — what we can compute now.

``compute_d2_via_h0_action`` reads the ``AdamsE2Page`` produced by the
U-resolution and returns a list of d_2 differentials whose forced
behavior (zero) can be deduced from structural facts:

  * **Sparseness**: if E_2^{s+2, t+1} = 0, then d_2: E_2^{s, t} → 0 is
    forced to be zero.
  * **Stem zero**: any d_2 from a cell at a stem where the page is empty
    one column over is forced zero.

For non-trivial d_2 (the cases where the source AND target are both
nonzero), the algorithm marks them ``UNRESOLVED`` and emits a gap. This
is the honest level: computing a non-trivial d_2 requires either:

  1. A Yoneda product computation (we have only h_0 shift, not full
     products — see ``ext_yoneda_product`` Slice B).
  2. A direct unstable-cobar computation of the Massey-product secondary
     operation.

Both are large, multi-week pieces of research code. This module is the
honest scaffold around them; it returns a refined ``AdamsERPage`` with
the forced zeros annotated and the unresolved cases listed.

References:
    Adams, J. F. (1958). On the structure and applications of the
        Steenrod algebra. Comment. Math. Helv. 32, 180–214.
    Bruner, R. R., May, J. P., McClure, J. E., Steinberger, M. (1986).
        H_∞ Ring Spectra and Their Applications. LNM 1176.
"""
from __future__ import annotations

from typing import List, Tuple

from pydantic import BaseModel, ConfigDict

from pysurgery.adams.spectral_sequence import AdamsE2Page


class D2Verdict(BaseModel):
    """Verdict on a single d_2 candidate.

    Attributes:
        source: (s, t) of source cell.
        target: (s + 2, t + 1) of target cell.
        source_dim: dim_F_p E_2^{s, t}.
        target_dim: dim_F_p E_2^{s+2, t+1}.
        classification: ``"forced_zero"``, ``"unresolved"`` (would need
            Yoneda products), or ``"trivially_zero"`` (source empty).
        reason: Short human-readable explanation.
    """

    model_config = ConfigDict(frozen=True)

    source: Tuple[int, int]
    target: Tuple[int, int]
    source_dim: int
    target_dim: int
    classification: str
    reason: str = ""


class D2Report(BaseModel):
    """Catalogue of d_2 verdicts for a single E_2 page."""

    model_config = ConfigDict(frozen=True)

    prime: int
    forced_zeros: Tuple[D2Verdict, ...] = ()
    unresolved: Tuple[D2Verdict, ...] = ()
    trivially_zeros: Tuple[D2Verdict, ...] = ()


def compute_d2_via_h0_action(page: AdamsE2Page) -> D2Report:
    """Inspect every (s, t) cell and classify its d_2 differential.

    Conservative — does NOT compute non-trivial d_2 values. Returns
    forced zeros (where target is empty) and explicitly lists every
    unresolved case so the caller can see what is missing.
    """
    p = int(page.prime)
    forced: List[D2Verdict] = []
    unresolved: List[D2Verdict] = []
    trivial: List[D2Verdict] = []

    # Pull all positive cells.
    for (s, t), src_dim in sorted(page.e2_grid.items()):
        if src_dim == 0:
            continue
        tgt = (s + 2, t + 1)
        tgt_dim = page.e2_grid.get(tgt, 0)
        v = D2Verdict(
            source=(int(s), int(t)),
            target=(int(tgt[0]), int(tgt[1])),
            source_dim=int(src_dim),
            target_dim=int(tgt_dim),
            classification="",
        )
        if tgt_dim == 0:
            forced.append(
                v.model_copy(update={
                    "classification": "forced_zero",
                    "reason": "target cell is empty",
                })
            )
        else:
            unresolved.append(
                v.model_copy(update={
                    "classification": "unresolved",
                    "reason": (
                        "both source and target nonzero; "
                        "non-trivial d_2 needs Yoneda product"
                    ),
                })
            )
    return D2Report(
        prime=p,
        forced_zeros=tuple(forced),
        unresolved=tuple(unresolved),
        trivially_zeros=tuple(trivial),
    )


__all__ = [
    "D2Verdict",
    "D2Report",
    "compute_d2_via_h0_action",
]
