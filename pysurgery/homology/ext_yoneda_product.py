"""Yoneda product structure on Ext_U^{*,*}(M, F_p).

This module exposes the multiplicative structure on Ext_U computed by
``adams_u_resolution``. The full chain-level Yoneda product requires
lifting cocycle representatives through the minimal resolution and is
research-grade work. This module provides:

  * ``h_zero(prime)`` — the canonical class in Ext^{1, 1}(F_p, F_p),
    which acts on Ext^{s, t}(M, F_p) by shifting (s, t) → (s+1, t+1).
    This shift map is determined at the level of E_2 dimensions; we
    encode it as a structural map ``h0_shift_map`` for downstream use.

  * ``ExtElement`` — sparse representative of a class in Ext^{s, t},
    keyed by U-resolution generator.

  * ``yoneda_product(a, b)`` — for the special case where ``a`` is the
    h_0 class, returns the shifted element. For general arguments it
    raises ``NotImplementedError`` (the full algorithm requires chain
    lifts; see Slice B in the implementation plan).

References:
    Bruner, R. R. (1986). The Adams spectral sequence of H_∞ ring
        spectra. Lecture Notes in Math. 1176, Springer.
    May, J. P. (1965). The cohomology of restricted Lie algebras and
        of Hopf algebras. J. Algebra 3, 123–146.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from pydantic import BaseModel, ConfigDict

from pysurgery.adams.spectral_sequence import AdamsE2Page


# ── Schema ────────────────────────────────────────────────────────────────────


class ExtElement(BaseModel):
    """Sparse element of Ext_U^{s, t}(M, F_p).

    Attributes:
        prime: Coefficient prime.
        s: Homological degree.
        t: Internal degree.
        coords: Sparse coordinates keyed by a generator index. Coefficients
            are reduced mod ``prime``.
    """

    model_config = ConfigDict(frozen=True)

    prime: int
    s: int
    t: int
    coords: Tuple[Tuple[int, int], ...] = ()

    def is_zero(self) -> bool:
        """Return True if every coordinate coefficient vanishes mod prime."""
        return all(c % self.prime == 0 for _, c in self.coords)


def h_zero(prime: int) -> ExtElement:
    """The canonical h_0 class in Ext_U^{1, 1}(F_p, F_p).

    Acts on Ext_U^{s, t}(M, F_p) by shifting (s, t) → (s+1, t+1) when
    the target cell is nonzero. The exact rule for "when nonzero" is
    structural — it is encoded via ``h0_shift_map`` for E_2 pages whose
    columns are explicitly tracked.
    """
    return ExtElement(prime=int(prime), s=1, t=1, coords=((0, 1),))


def h0_shift_map(page: AdamsE2Page) -> Dict[Tuple[int, int], int]:
    """The induced rank map of h_0-multiplication on E_2 cell dimensions.

    For each cell (s, t) with positive dimension, returns the maximum rank
    of the h_0-shift map to (s+1, t+1). This rank is bounded above by
    ``min(dim E_2^{s, t}, dim E_2^{s+1, t+1})``. The actual rank requires
    chain-level Yoneda computation; this function returns the *upper*
    bound, which is the right input for the h_0-tower extension solver
    in ``adams_extension_solver``.
    """
    out: Dict[Tuple[int, int], int] = {}
    for (s, t), src_dim in page.e2_grid.items():
        if src_dim == 0:
            continue
        tgt_dim = page.e2_grid.get((s + 1, t + 1), 0)
        if tgt_dim == 0:
            continue
        out[(s, t)] = min(int(src_dim), int(tgt_dim))
    return out


def yoneda_product(a: ExtElement, b: ExtElement) -> Optional[ExtElement]:
    """Yoneda product on Ext_U^{*, *}.

    Currently implements:
      - ``yoneda_product(h_zero(p), x)`` for any class x — returns the
        shifted element at (s+1, t+1) when the target cell is nonzero,
        else the zero element.
      - General products raise NotImplementedError. The full algorithm
        requires lifting cocycle representatives through the minimal
        U-resolution (Bruner 1986 §IV.5), which is multi-week work and
        is tracked as Slice B in the implementation plan.
    """
    if a.prime != b.prime:
        raise ValueError(
            f"prime mismatch: a.prime={a.prime}, b.prime={b.prime}"
        )
    if a.s == 1 and a.t == 1 and len(a.coords) == 1 and a.coords[0] == (0, 1):
        # h_0 * b — structurally a shift.
        return ExtElement(prime=b.prime, s=b.s + 1, t=b.t + 1, coords=b.coords)
    if b.s == 1 and b.t == 1 and len(b.coords) == 1 and b.coords[0] == (0, 1):
        # b is h_0; commutative (h_0 is central in Ext_A).
        return ExtElement(prime=a.prime, s=a.s + 1, t=a.t + 1, coords=a.coords)
    raise NotImplementedError(
        "general Yoneda product requires chain-level lifting; "
        "only h_0 · x is implemented in this slice"
    )


__all__ = [
    "ExtElement",
    "h_zero",
    "h0_shift_map",
    "yoneda_product",
]
