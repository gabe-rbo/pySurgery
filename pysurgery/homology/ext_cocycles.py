"""Cocycle representatives for Ext_U^{*,*}(M, F_p) classes.

This module exposes the **explicit cocycle data** underlying a class in
Ext_U^{s, t}(M, F_p) computed by ``adams_u_resolution.UnstableResolution``.
In a minimal free unstable A_p-resolution F_* → M, the canonical
Ext-basis at bidegree (s, t) consists of the F_s generators of degree t
themselves; their dual indicator functions are automatically cocycles
(because d_{s+1}(F_{s+1}) ⊂ aug·F_s vanishes on the basis generators).

Use cases:
  - Foundation for the chain-map lifting algorithm (Slice Y2).
  - Input to the Yoneda product (Slice Y3).
  - Audit / debugging of Ext_U^{*,*} computations.

This module is **read-only on the resolution**. It does not mutate any
``UnstableResolution`` state.

References:
    Cartan-Eilenberg (1956). *Homological Algebra*, Chapter V.
    Bruner, R. R. (1986). The Adams spectral sequence of H_∞ ring
        spectra. LNM 1176, §IV.5.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from pydantic import BaseModel, ConfigDict, Field

from pysurgery.adams.u_resolution import UGenerator, UnstableResolution


class ExtCocycle(BaseModel):
    """A cocycle representative of a class in Ext_U^{s, t}(M, F_p).

    Attributes:
        prime: Coefficient prime.
        s: Homological degree (filtration).
        t: Internal degree.
        coefs: Sparse map ``{gid_of_F_s_generator: F_p_coef}``. The cocycle
            is the linear functional on F_s sending each generator γ to its
            coefficient (or 0 if absent) and extending by zero on
            Sq-propagation.
    """

    model_config = ConfigDict(frozen=True)

    prime: int
    s: int
    t: int
    coefs: Dict[int, int] = Field(default_factory=dict)

    def is_zero(self) -> bool:
        """Return True if every coefficient vanishes mod prime."""
        return all((c % self.prime) == 0 for c in self.coefs.values())

    def normalized(self) -> "ExtCocycle":
        """Drop zero coefficients and reduce mod prime."""
        cleaned = {
            gid: int(c) % int(self.prime)
            for gid, c in self.coefs.items()
            if (int(c) % int(self.prime)) != 0
        }
        return ExtCocycle(
            prime=int(self.prime), s=int(self.s), t=int(self.t), coefs=cleaned
        )


def _generators_at(resolution: UnstableResolution, s: int, t: int) -> List[UGenerator]:
    if s < 0 or s >= len(resolution.F):
        return []
    return [g for g in resolution.F[s] if g.degree == t]


def basis_cocycles(
    resolution: UnstableResolution, s: int, t: int
) -> List[ExtCocycle]:
    """Return the canonical basis of Ext_U^{s, t}(M, F_p).

    In a **minimal** resolution every F_s-generator at internal degree t
    contributes exactly one Ext class — its indicator functional. These
    indicators are automatically cocycles because the minimality condition
    makes ``d_{s+1}(F_{s+1})`` land inside the augmentation ideal, which
    vanishes on the dual basis.

    Returns one ``ExtCocycle`` per generator, sorted by ``gid``.
    """
    gens = _generators_at(resolution, s, t)
    return [
        ExtCocycle(
            prime=int(resolution.prime),
            s=int(s),
            t=int(t),
            coefs={int(g.gid): 1},
        )
        for g in sorted(gens, key=lambda g: g.gid)
    ]


def evaluate(
    resolution: UnstableResolution,
    alpha: ExtCocycle,
    element: Dict[Tuple[int, "AdmissibleSeq"], int],  # noqa: F821 - forward ref
) -> int:
    """Evaluate a cocycle α on a chain-level element of F_s.

    The ``element`` is keyed by ``(gid, admissible_op)`` pairs as used by
    ``UnstableResolution._d_on``. Sq^I-actions are zero on the dual basis
    (because α is the indicator of a generator), so only the identity
    admissible ``()`` contributes.
    """
    if alpha.is_zero():
        return 0
    total = 0
    for (gid, op), c in element.items():
        if not op:  # identity admissible
            coef = alpha.coefs.get(int(gid), 0)
            total = (total + int(c) * int(coef)) % int(alpha.prime)
    return total % int(alpha.prime)


def is_cocycle(
    resolution: UnstableResolution, alpha: ExtCocycle, *, t_max_window: int = 30
) -> bool:
    """Verify that α annihilates ``d_{s+1}(F_{s+1}[t])`` (basic sanity check).

    For a generator-indicator cocycle this is automatic by minimality;
    this function exists for auditing custom-constructed cocycles.
    """
    s = int(alpha.s)
    t = int(alpha.t)
    if s + 1 >= len(resolution.F):
        return True
    for g in resolution.F[s + 1]:
        if g.degree != t:
            continue
        # d(γ) lives in F_s[t]. Evaluate α on it.
        image = resolution._d_on(s + 1, g.gid, ())
        if evaluate(resolution, alpha, image) != 0:
            return False
    return True


__all__ = [
    "ExtCocycle",
    "basis_cocycles",
    "evaluate",
    "is_cocycle",
]
