"""Solve the Adams-filtration extension problem via h_0-towers.

Given an E_∞ (or, at this slice level, an E_2 upper bound) page with
per-stem dimensions ``E^{s, n+s}``, the homotopy group π_n^s(X)_p is
filtered:

    π_n_(p) ⊃ F_1 ⊃ F_2 ⊃ ... ⊃ F_k ⊃ 0,
    F_s / F_{s+1} ≅ E_∞^{s, n+s}    (as F_p-vector spaces).

The extension problem is: which abelian group does this filtration
describe?  The h_0-tower picture answers it column-by-column:

  * A **maximal h_0-tower** is a maximal set of consecutive cells
    (s_0, n+s_0), (s_0+1, n+s_0+1), …, (s_0+L-1, n+s_0+L-1) such that
    h_0-multiplication is an iso (or surjection / injection of the
    right rank) between each adjacent pair.
  * Each h_0-tower of length L contributes (at the order level) a
    factor of p^L to the p-primary order of π_n.  Whether the tower
    extends to **one** Z/p^L or to **L copies of Z/p** depends on
    higher Massey products / multiplication-by-p in π_n.

Without explicit Yoneda-product data, we report **both** views:

  * ``most_collapsed_invariant_factors`` — assume every tower extends
    maximally to a single Z/p^L summand. This typically matches Toda
    for spheres at small stems (the η-tower in π_n(S^2)).
  * ``most_split_invariant_factors`` — assume no tower extends; each
    cell is a separate Z/p. This is the conservative case.

The TRUE answer is between these two. Both bound the order at p^N.

References:
    Adams, J. F. (1966). On the groups J(X). IV. Topology 5, 21-71.
    Ravenel, D. C. (1986). Complex Cobordism and Stable Homotopy Groups
        of Spheres. Academic Press. §3.1.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict

from pysurgery.adams.spectral_sequence import AdamsE2Page


# ── Schema ────────────────────────────────────────────────────────────────────


class H0Tower(BaseModel):
    """A maximal column of consecutive cells at a fixed stem.

    Attributes:
        stem: t - s value (the stem).
        s_start: filtration at the bottom of the tower (smallest s with
            a cell at this stem, possibly s=0 for the generator).
        length: number of consecutive filtrations occupied.
        prime: coefficient prime.
        cell_dims: per-cell dimension (usually all 1 for a "thin" tower).
    """

    model_config = ConfigDict(frozen=True)

    stem: int
    s_start: int
    length: int
    prime: int
    cell_dims: Tuple[int, ...]


class StemExtensionHypothesis(BaseModel):
    """Algorithmic extension hypothesis for a single stem at a single prime.

    Attributes:
        stem: The homotopy stem n.
        prime: Coefficient prime p.
        towers: Detected h_0-towers at filtration ≥ 1 (the "torsion towers").
        free_tower: Optional length of the s=0..L-1 column when present at
            filtration 0; this is the candidate Z (free) summand. None when
            no filtration-0 cell exists at this stem.
        most_collapsed_invariant_factors: Each filtration-≥1 tower extended
            into a single Z/p^L summand. Sorted descending.
        most_split_invariant_factors: Each cell treated as Z/p. Sorted
            descending.
        p_order_upper_bound: p^(total cell count at filtration ≥ 1).
    """

    model_config = ConfigDict(frozen=True)

    stem: int
    prime: int
    towers: Tuple[H0Tower, ...]
    free_tower: Optional[H0Tower] = None
    most_collapsed_invariant_factors: Tuple[int, ...] = ()
    most_split_invariant_factors: Tuple[int, ...] = ()
    p_order_upper_bound: int = 1


# ── Tower detection ──────────────────────────────────────────────────────────


def _stem_filtration_cells(
    page: AdamsE2Page, n: int
) -> List[Tuple[int, int]]:
    """Sorted ``[(s, dim_E2^{s, n+s})]`` for s ≥ 0 at the given stem."""
    cells: List[Tuple[int, int]] = []
    for (s, t), d in page.e2_grid.items():
        if t - s != n or d <= 0:
            continue
        cells.append((int(s), int(d)))
    cells.sort()
    return cells


def _absorb_free_rank(
    cells: List[Tuple[int, int]], free_rank: int
) -> List[Tuple[int, int]]:
    """Subtract `free_rank` from each cell of the bottom-most connected tower.

    Math (the rational-vs-torsion classifier): when π_n(X) ⊗ ℚ = ℚ^r (r ≥ 1),
    the Adams chart contains r independent **infinite** h_0-towers. Each
    tower starts at some filtration s_0 (the lowest filtration where the
    class is first detected) and extends upward indefinitely under
    h_0-multiplication. These towers ARE the Z^r summand (its p-adic
    completion Z_p^r) — they are not separate torsion.

    The "bottom" of the rational tower is NOT always s=0. For S^2 at stem 2
    the bottom is s=0 (the generator x). For S^2 at stem 3 the bottom is
    s=1 (the η class lives at filtration 1 in the Adams chart, not 0). So
    we find the lowest filtration with cells and absorb free_rank h_0-shifts
    from each consecutive filtration starting there.

    After absorption, what remains is genuinely torsion (towers disconnected
    from the rational one OR cell-dimension excess beyond the rational
    multiplicity).
    """
    if free_rank <= 0 or not cells:
        return list(cells)
    by_s = {int(s): int(d) for s, d in cells if int(d) > 0}
    if not by_s:
        return []
    s_min = min(by_s.keys())
    out_by_s = dict(by_s)
    s = s_min
    while s in out_by_s and out_by_s[s] > 0:
        out_by_s[s] = max(0, out_by_s[s] - free_rank)
        s += 1
    return [(s, d) for s, d in sorted(out_by_s.items()) if d > 0]


def detect_h0_towers(
    page: AdamsE2Page, n: int, *, free_rank: int = 0
) -> Tuple[Optional[H0Tower], List[H0Tower]]:
    """Detect maximal consecutive-filtration runs at stem n.

    Returns ``(free_tower, torsion_towers)`` where:
      * ``free_tower`` is the run starting at s = 0 (if any), interpreted
        as the candidate for a Z summand (extends to infinity in true
        π_n_(p)).
      * ``torsion_towers`` are runs at filtration ≥ 1.

    When ``free_rank > 0``, ``free_rank`` h_0-shifts are absorbed from
    each consecutive filtration starting at s=0 (because each rational Z
    summand contributes one cell per filtration to the Adams chart). Only
    the remaining cells become torsion towers.
    """
    prime = int(page.prime)
    raw_cells = _stem_filtration_cells(page, n)
    cells = _absorb_free_rank(raw_cells, int(free_rank))
    if not cells:
        return None, []
    free_tower: Optional[H0Tower] = None
    torsion: List[H0Tower] = []
    # Pack cells into runs of consecutive s, treating s=0 as its own run
    # boundary (filtration 0 is "rational/free", filtration ≥ 1 is torsion;
    # they must not be merged even if both are nonempty at this stem).
    i = 0
    while i < len(cells):
        run_start = cells[i][0]
        run_dims = [cells[i][1]]
        j = i + 1
        while j < len(cells) and cells[j][0] == cells[j - 1][0] + 1:
            # Boundary: split off s=0 into its own tower.
            if run_start == 0 and cells[j][0] == 1:
                break
            run_dims.append(cells[j][1])
            j += 1
        tower = H0Tower(
            stem=n,
            s_start=run_start,
            length=len(run_dims),
            prime=prime,
            cell_dims=tuple(run_dims),
        )
        if run_start == 0:
            free_tower = tower
        else:
            torsion.append(tower)
        i = j
    return free_tower, torsion


# ── Tower → invariant factors ────────────────────────────────────────────────


def _tower_collapsed_factors(tower: H0Tower) -> List[int]:
    """One Z/p^L summand per F_p-cell of the *thickest* row, length L."""
    p = tower.prime
    # If cell_dims has multiplicity k at each step, treat the tower as k
    # parallel towers of equal length (the conservative collapsed reading).
    width = min(tower.cell_dims) if tower.cell_dims else 0
    return [p ** tower.length] * width + sum(
        ([p ** (tower.length)] * (max(d, 0) - width) for d in tower.cell_dims),
        [],
    )[:0]  # additional excess width — not modeled here


def _tower_split_factors(tower: H0Tower) -> List[int]:
    """Each cell as Z/p (no extension)."""
    return [tower.prime] * sum(tower.cell_dims)


def solve_stem(
    page: AdamsE2Page, n: int, *, free_rank: int = 0
) -> StemExtensionHypothesis:
    """Build the extension hypothesis for a single stem at one prime.

    ``free_rank``: the rank of π_n(X) ⊗ ℚ from the Sullivan minimal model.
    When > 0, ``free_rank`` h_0-shifts at every filtration starting at s=0
    are absorbed into the free part and excluded from torsion accounting.
    """
    prime = int(page.prime)
    free_tower, torsion_towers = detect_h0_towers(page, n, free_rank=free_rank)

    collapsed: List[int] = []
    split: List[int] = []
    p_order = 1
    for tw in torsion_towers:
        collapsed.extend(_tower_collapsed_factors(tw))
        split.extend(_tower_split_factors(tw))
        for d in tw.cell_dims:
            p_order *= prime ** int(d)
    collapsed.sort(reverse=True)
    split.sort(reverse=True)

    return StemExtensionHypothesis(
        stem=int(n),
        prime=prime,
        towers=tuple(torsion_towers),
        free_tower=free_tower,
        most_collapsed_invariant_factors=tuple(collapsed),
        most_split_invariant_factors=tuple(split),
        p_order_upper_bound=int(p_order),
    )


def solve_extensions(
    pages_by_prime: Dict[int, AdamsE2Page], n: int
) -> Dict[int, StemExtensionHypothesis]:
    """Solve the per-prime extension problem at stem n.

    ``pages_by_prime[p]`` is the E_2 (or refined) page at prime ``p``.
    """
    return {int(p): solve_stem(page, n) for p, page in pages_by_prime.items()}


__all__ = [
    "H0Tower",
    "StemExtensionHypothesis",
    "detect_h0_towers",
    "solve_stem",
    "solve_extensions",
]
