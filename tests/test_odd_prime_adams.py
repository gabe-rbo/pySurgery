"""Tests for the odd-prime Adams E_2 page via cobar of A_p^*.

Module under test: pysurgery/core/adams_odd_prime_cobar.py
Dispatched from: pysurgery/core/adams_spectral_sequence.py::adams_e2_page
"""
from __future__ import annotations

from typing import Dict

import pytest

from pysurgery.adams.odd_prime_cobar import (
    DualSteenrodAlgebra,
    ext_cobar,
    is_trivial_ap_module,
)
from pysurgery.adams.spectral_sequence import (
    SteenrodAlgebra,
    adams_e2_page,
    sphere_cohomology_fp,
)


# ── A_p^* basis dimensions match the admissible-basis enumeration ────────────


@pytest.mark.parametrize("p", [3, 5])
def test_dual_basis_dim_matches_admissibles(p: int) -> None:
    """dim A_p^*[t] = dim A_p[t] (duality), both equal to len(admissible_basis(t))."""
    A_star = DualSteenrodAlgebra(p, 14 if p == 3 else 16)
    A = SteenrodAlgebra(prime=p, max_t=A_star.t_max)
    for t in range(A_star.t_max + 1):
        n_star = len(A_star.basis(t))
        n_alg = len(A.admissible_basis(t))
        assert n_star == n_alg, (
            f"p={p}, t={t}: dim A_p^*[t]={n_star} != dim A_p[t]={n_alg}"
        )


# ── d^2 = 0 on the cobar complex ─────────────────────────────────────────────


def _apply_cobar_d(A: DualSteenrodAlgebra, state: Dict[tuple, int]) -> Dict[tuple, int]:
    """Apply one cobar differential step in the convention used by ext_cobar.

    Sign: (-1)^{i+1} with i 0-based; no extra Koszul accumulation.
    """
    p = A.p
    out: Dict[tuple, int] = {}
    for src, coef_src in state.items():
        if coef_src == 0:
            continue
        s = len(src)
        if s == 0:
            continue
        for i in range(s):
            a = src[i]
            sgn = ((-1) ** (i + 1)) % p
            for L, R, c in A.coproduct_reduced(a):
                new_t = src[:i] + (L, R) + src[i + 1:]
                final = (coef_src * c * sgn) % p
                if final == 0:
                    continue
                out[new_t] = (out.get(new_t, 0) + final) % p
    return {k: v for k, v in out.items() if v != 0}


@pytest.mark.parametrize("p,t_max", [(3, 12), (5, 14)])
def test_d_squared_is_zero_on_s1(p: int, t_max: int) -> None:
    """d^2 = 0 on every basis element of (A_+)^1 in degree <= t_max."""
    A = DualSteenrodAlgebra(p, t_max)
    for t in range(1, t_max + 1):
        for m in A.basis(t):
            state = {(m,): 1}
            d1 = _apply_cobar_d(A, state)
            d2 = _apply_cobar_d(A, d1)
            assert not d2, (
                f"d^2 != 0 at m={m} (deg {t}, p={p}): d^2(m)={d2}"
            )


def test_d_squared_is_zero_on_s2_p3() -> None:
    """d^2 = 0 on every basis element of (A_+)^2 in low degree at p=3."""
    A = DualSteenrodAlgebra(3, 8)
    for t in range(2, 9):
        # build (A_+)^2 at degree t
        for a_deg in range(1, t):
            b_deg = t - a_deg
            for a in A.basis(a_deg):
                for b in A.basis(b_deg):
                    state = {(a, b): 1}
                    d1 = _apply_cobar_d(A, state)
                    d2 = _apply_cobar_d(A, d1)
                    assert not d2, (
                        f"d^2 != 0 at (a={a}, b={b}) (deg {t}, p=3): d^2={d2}"
                    )


# ── Ext^{0,0} = F_p ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("p", [3, 5])
def test_ext_00_is_one(p: int) -> None:
    g = ext_cobar(p, s_max=2, t_max=4)
    assert g.get((0, 0), 0) == 1, f"Ext^{{0,0}}(F_{p}, F_{p}) should be 1; got {g.get((0,0),0)}"


# ── Ext^{1, *} matches primitives of A_p^* (the indecomposables of A_p) ──────


def test_ext_1_line_p3_primitives() -> None:
    """At p=3, Ext^{1,t}(F_3, F_3) = dim of primitives of A_3^* in degree t.

    Primitives of A_3^* in low degree are tau_0 (1), xi_1 (4), xi_1^3 (12).
    Note: tau_1 has Δ̄(tau_1) = xi_1 ⊗ tau_0 != 0 -- NOT primitive.
    Note: xi_2 has Δ̄(xi_2) = xi_1^3 ⊗ xi_1 != 0 -- NOT primitive.
    """
    g = ext_cobar(3, s_max=1, t_max=13)
    expected_nonzero = {1, 4, 12}
    for t in range(0, 14):
        d = g.get((1, t), 0)
        if t in expected_nonzero:
            assert d == 1, f"Ext^{{1,{t}}} should be 1 (primitive); got {d}"
        else:
            assert d == 0, f"Ext^{{1,{t}}} should be 0; got {d}"


# ── adams_e2_page dispatch round-trip ────────────────────────────────────────


@pytest.mark.parametrize("p", [3, 5])
def test_adams_e2_page_dispatches_for_S2(p: int) -> None:
    """adams_e2_page on S^2 at odd p returns a 'success' page with E_2 cells."""
    ring = sphere_cohomology_fp(2, prime=p)
    assert is_trivial_ap_module(ring), (
        f"sphere_cohomology_fp(2, p={p}) should be a trivial A_p-module"
    )
    page = adams_e2_page(ring, prime=p, s_max=4, t_max=12)
    assert page.status == "success", (
        f"odd-prime Adams should succeed on trivial-module input; got {page.status}: {page.reasoning}"
    )
    nonzero = {k: v for k, v in page.e2_grid.items() if v}
    assert (0, 0) in nonzero and nonzero[(0, 0)] == 1, "E_2^{0,0} = 1 for the bottom cell"
    assert (0, 2) in nonzero and nonzero[(0, 2)] == 1, "E_2^{0,2} = 1 for the top cell"


def test_adams_e2_page_p2_still_works() -> None:
    """Regression: the p=2 path is unaffected by the odd-prime dispatch."""
    ring = sphere_cohomology_fp(2, prime=2)
    page = adams_e2_page(ring, prime=2, s_max=3, t_max=8)
    assert page.status == "success"
    assert page.e2_grid.get((0, 0), 0) == 1
    assert page.e2_grid.get((0, 2), 0) == 1
    assert page.e2_grid.get((1, 1), 0) == 1
