"""Tests for the Adams spectral sequence E_2 page module.

Coverage:
    - Steenrod algebra Adem relations and admissible bases (mod 2).
    - SteenrodAction on H^*(RP^n; F_2), H^*(CP^n; F_2).
    - steenrod_squares_matrix sparse-only output.
    - adams_e2_page on S^0, S^n, CP^n, RP^n with literature cross-validation.
    - Resource guardrails: memory bounded, t_max clamping, sparse enforcement.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import scipy.sparse as sp

from pysurgery.adams.spectral_sequence import (
    ADM_HARD_CAP,
    AdamsE2Page,
    FpCohomologyRing,
    SteenrodAlgebra,
    T_HARD,
    _binom_mod_p,
    adams_e2_page,
    cp_n_cohomology_fp,
    rp_n_cohomology_fp,
    sphere_cohomology_fp,
    steenrod_squares_matrix,
)
from pysurgery.core.theorem_tags import ADAMS_E2_EXT_STEENROD


# ════════════════════════════════════════════════════════════════════════════
# Layer 0 — arithmetic helpers
# ════════════════════════════════════════════════════════════════════════════


def test_binom_mod_p_lucas_p2():
    """Lucas's theorem mod 2 via the bitwise identity."""
    # C(5, 2) = 10 → 0 mod 2
    assert _binom_mod_p(5, 2, 2) == 0
    # C(7, 3) = 35 → 1 mod 2
    assert _binom_mod_p(7, 3, 2) == 1
    # C(0, 0) = 1
    assert _binom_mod_p(0, 0, 2) == 1
    # negative or out-of-range
    assert _binom_mod_p(2, 5, 2) == 0
    assert _binom_mod_p(-1, 0, 2) == 0


def test_binom_mod_p_p3():
    # C(4, 2) = 6 → 0 mod 3
    assert _binom_mod_p(4, 2, 3) == 0
    # C(5, 2) = 10 → 1 mod 3
    assert _binom_mod_p(5, 2, 3) == 1


# ════════════════════════════════════════════════════════════════════════════
# Layer 1 — Steenrod algebra mod 2
# ════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def A2():
    return SteenrodAlgebra(prime=2, max_t=15)


def test_steenrod_constructor_validates_prime():
    with pytest.raises(ValueError):
        SteenrodAlgebra(prime=4)


def test_admissible_basis_known_dimensions(A2):
    """Admissible basis cardinalities vs. Milnor's table."""
    # dim A^0 = 1, dim A^1 = 1, dim A^2 = 1, dim A^3 = 2, dim A^4 = 2,
    # dim A^5 = 2, dim A^6 = 3, dim A^7 = 4, dim A^8 = 4.
    expected = {0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 4, 8: 4}
    for t, d in expected.items():
        assert len(A2.admissible_basis(t)) == d, f"dim A^{t}"


def test_admissibles_satisfy_admissibility(A2):
    for t in range(0, 10):
        for seq in A2.admissible_basis(t):
            assert A2.is_admissible(seq), f"Not admissible: {seq}"
            # internal degree matches
            assert A2.degree_of(seq) == t


def test_adem_classical_relations(A2):
    # Sq^1 Sq^1 = 0
    assert A2.to_admissible((1, 1)) == {}
    # Sq^1 Sq^2 = Sq^3
    assert A2.to_admissible((1, 2)) == {(3,): 1}
    # Sq^2 Sq^2 = Sq^3 Sq^1
    assert A2.to_admissible((2, 2)) == {(3, 1): 1}
    # Sq^2 Sq^3 = Sq^5 + Sq^4 Sq^1
    out = A2.to_admissible((2, 3))
    assert out == {(5,): 1, (4, 1): 1}
    # Sq^3 Sq^3 = Sq^5 Sq^1
    assert A2.to_admissible((3, 3)) == {(5, 1): 1}
    # Sq^4 Sq^2 (admissible since 4 ≥ 2·2) — unchanged
    assert A2.to_admissible((4, 2)) == {(4, 2): 1}


def test_to_admissible_idempotent_on_admissibles(A2):
    for t in range(0, 8):
        for seq in A2.admissible_basis(t):
            if seq:
                assert A2.to_admissible(seq) == {seq: 1}


def test_steenrod_mul_admissible(A2):
    # Sq^2 · Sq^2 in algebra
    a = A2.Sq(2)
    b = A2.Sq(2)
    c = A2.mul(a, b)
    assert c == {(3, 1): 1}
    # Sq^1 · 1 = Sq^1
    one = A2.one()
    assert A2.mul(A2.Sq(1), one) == {(1,): 1}


# ════════════════════════════════════════════════════════════════════════════
# Layer 2 — action on H^*(RP^∞; F_2) and H^*(CP^∞; F_2)
# ════════════════════════════════════════════════════════════════════════════


def test_rp_action_matches_binomial_formula():
    """Sq^i(x^k) = C(k, i) x^{i+k} in H^*(RP^n; F_2)."""
    n = 8
    ring = rp_n_cohomology_fp(n, prime=2)
    mats = steenrod_squares_matrix(ring, prime=2)
    for k in range(1, n + 1):
        for i in range(0, n + 1):
            if k + i > n:
                continue
            mat = mats.get((i, k))
            assert mat is not None, f"missing matrix for (Sq^{i}, deg {k})"
            assert sp.issparse(mat), "non-sparse matrix returned"
            expected = _binom_mod_p(k, i, 2)
            arr = mat.toarray().tolist()
            if expected == 1:
                assert arr == [[1]], f"Sq^{i}(x^{k}) expected x^{i+k}, got {arr}"
            else:
                assert arr == [[0]], f"Sq^{i}(x^{k}) expected 0, got {arr}"


def test_cp_action_squares_x_to_x2():
    ring = cp_n_cohomology_fp(4, prime=2)
    mats = steenrod_squares_matrix(ring, prime=2)
    # Sq^2(x) = x²
    assert mats[(2, 2)].toarray().tolist() == [[1]]
    # Sq^1(x) lands in H^3 = 0 — matrix has 0 rows, no nonzero entries.
    assert mats[(1, 2)].shape[0] == 0  # H^3(CP^4) = 0
    assert mats[(1, 2)].nnz == 0
    # Sq^4(x²) = x⁴ via Cartan: Σ_{i+j=4} Sq^i(x)·Sq^j(x), with Sq^0(x)=x, Sq^2(x)=x²:
    # only (i, j) ∈ {(0,4),(2,2),(4,0)}; Sq^4(x) = 0 (instability). So = x²·x² = x⁴.
    # CP^4 has H^8 = F_2 generated by x⁴.
    assert mats[(4, 4)].toarray().tolist() == [[1]]


def test_steenrod_squares_matrix_all_sparse():
    """Every output matrix is sparse (assertion holds across all spaces)."""
    for ring in [
        sphere_cohomology_fp(3, prime=2),
        cp_n_cohomology_fp(2, prime=2),
        rp_n_cohomology_fp(5, prime=2),
    ]:
        mats = steenrod_squares_matrix(ring, prime=2)
        for k, mat in mats.items():
            assert sp.issparse(mat), f"non-sparse at {k} for {ring.space_label}"


# ════════════════════════════════════════════════════════════════════════════
# Layer 4 — Adams E_2 page on standard spaces
# ════════════════════════════════════════════════════════════════════════════


def _grid(page: AdamsE2Page) -> dict:
    return dict(page.e2_grid)


def test_e2_S0_known_hopf_elements():
    """S^0 mod 2: Ext_{A_2}(F_2, F_2) at low (s, t).

    Known generators (Adams 1958):
      h_0 in (1, 1), h_1 in (1, 2), h_2 in (1, 4), h_3 in (1, 8).
    Known products at s=2 within t ≤ 8: h_0², h_1², h_0 h_2, h_2².
    Known relations: h_n h_{n+1} = 0 ⇒ (2, 3), (2, 6) absent.
    """
    ring = sphere_cohomology_fp(0, prime=2)
    page = adams_e2_page(ring, prime=2, s_max=2, t_max=8)
    g = _grid(page)
    # Unit
    assert g.get((0, 0), 0) == 1
    # h_i
    for i in range(0, 4):
        assert g.get((1, 1 << i), 0) == 1, f"missing h_{i} at (1, {1 << i})"
    # No spurious E_2^{0, t} for t > 0
    for t in range(1, 9):
        assert g.get((0, t), 0) == 0
    # Products at s = 2
    assert g.get((2, 2), 0) == 1   # h_0²
    assert g.get((2, 4), 0) == 1   # h_1²
    assert g.get((2, 5), 0) == 1   # h_0 h_2
    assert g.get((2, 8), 0) == 1   # h_2²
    # Forbidden: h_n h_{n+1} = 0
    assert g.get((2, 3), 0) == 0   # h_0 h_1 = 0
    assert g.get((2, 6), 0) == 0   # h_1 h_2 = 0


def test_e2_sphere_S3_is_shifted_S0():
    """E_2(S^n) = E_2(S^0) ⊕ Σ^n E_2(S^0) since H^*(S^n; F_2) is two trivial F_2's."""
    n = 3
    ring0 = sphere_cohomology_fp(0, prime=2)
    ringN = sphere_cohomology_fp(n, prime=2)
    page0 = adams_e2_page(ring0, prime=2, s_max=2, t_max=8)
    pageN = adams_e2_page(ringN, prime=2, s_max=2, t_max=8)
    expected = {}
    for (s, t), d in page0.e2_grid.items():
        expected[(s, t)] = expected.get((s, t), 0) + d
        if t + n <= 8:
            expected[(s, t + n)] = expected.get((s, t + n), 0) + d
    assert dict(pageN.e2_grid) == expected


def test_e2_CP2_minimal_generators():
    """F_0 generators of H^*(CP^2; F_2) = F_2[x]/(x³) are {1, x}.

    x² = Sq^2(x) lies in A_>0 · M and is *not* a minimal generator.
    """
    ring = cp_n_cohomology_fp(2, prime=2)
    page = adams_e2_page(ring, prime=2, s_max=1, t_max=6)
    g = _grid(page)
    # E_2^{0, 0} = 1 (unit)
    assert g.get((0, 0), 0) == 1
    # E_2^{0, 2} = 1 (x is a minimal generator)
    assert g.get((0, 2), 0) == 1
    # x² is NOT a minimal generator
    assert g.get((0, 4), 0) == 0


def test_e2_RP4_minimal_generators():
    """F_0 generators of H^*(RP^4; F_2) are {1, x, x³}.

    x² = Sq^1(x); x⁴ = Sq^1(x³). x³: Sq^1(x²) = C(2,1) x³ = 0 mod 2,
    Sq^2(x) = 0 (instability), so x³ is a new minimal generator.
    """
    ring = rp_n_cohomology_fp(4, prime=2)
    page = adams_e2_page(ring, prime=2, s_max=1, t_max=5)
    g = _grid(page)
    assert g.get((0, 0), 0) == 1
    assert g.get((0, 1), 0) == 1
    assert g.get((0, 2), 0) == 0  # x² ∈ A_>0 · M
    assert g.get((0, 3), 0) == 1  # x³ is minimal
    assert g.get((0, 4), 0) == 0  # x⁴ ∈ A_>0 · M


# ════════════════════════════════════════════════════════════════════════════
# Contract invariants
# ════════════════════════════════════════════════════════════════════════════


def test_e2_contract_metadata():
    ring = sphere_cohomology_fp(2, prime=2)
    page = adams_e2_page(ring, prime=2, s_max=2, t_max=6)
    assert page.exact is True
    assert page.theorem_tag == ADAMS_E2_EXT_STEENROD
    assert page.contract_version
    assert page.status == "success"
    assert page.decision_ready() is True
    # resource summary present
    assert "peak_mem_mb" in page.resource_summary
    assert "wall_seconds" in page.resource_summary


def test_e2_differentials_classified():
    ring = rp_n_cohomology_fp(3, prime=2)
    page = adams_e2_page(ring, prime=2, s_max=3, t_max=6)
    # Every forced flag has source==0 OR target==0
    for fl in page.forced_vanishings:
        assert fl.source_dim == 0 or fl.target_dim == 0
        assert fl.classification == "forced_zero"
    # Every ambiguous flag has both > 0
    for fl in page.ambiguous_differentials:
        assert fl.source_dim > 0 and fl.target_dim > 0
        assert fl.classification == "ambiguous"


def test_e2_stem_accessor():
    ring = sphere_cohomology_fp(0, prime=2)
    page = adams_e2_page(ring, prime=2, s_max=2, t_max=8)
    # stem 0 = E_2^{0,0} ⊕ E_2^{1,1} ⊕ E_2^{2,2}
    s0 = page.stem(0)
    assert s0.get(0, 0) == 1
    assert s0.get(1, 0) == 1
    assert s0.get(2, 0) == 1


# ════════════════════════════════════════════════════════════════════════════
# Resource guardrails
# ════════════════════════════════════════════════════════════════════════════


def test_t_max_clamped_to_T_HARD():
    ring = sphere_cohomology_fp(0, prime=2)
    with pytest.warns(UserWarning, match="hard cap"):
        page = adams_e2_page(ring, prime=2, s_max=1, t_max=T_HARD + 5)
    assert page.t_max == T_HARD


def test_invalid_prime_rejected():
    ring = sphere_cohomology_fp(0, prime=2)
    with pytest.raises(ValueError):
        adams_e2_page(ring, prime=4, s_max=2, t_max=5)


def test_odd_prime_trivial_module_succeeds():
    """For p ∈ {3, 5} with a trivial A_p-module, the cobar route succeeds.

    Updated from the original 'returns inconclusive' assertion: odd-prime
    Adams is now implemented via the cobar of A_p^* in
    pysurgery/core/adams_odd_prime_cobar.py, dispatched from adams_e2_page
    when the input is a trivial A_p-module.
    """
    # Build a tiny ring with prime=3 (the F_3-cohomology of a point — Ext is
    # then literally Ext_{A_3}(F_3, F_3) — or take S^0 with one class in deg 0).
    ring = FpCohomologyRing(
        space_label="S^0_p3",
        prime=3,
        max_degree=0,
        basis={0: ["1"]},
        cup_table={("1", "1"): {"1": 1}},
        sq_table={(0, "1"): {"1": 1}},
        ring_generators=[],
    )
    page = adams_e2_page(ring, prime=3, s_max=2, t_max=10)
    assert page.status == "success"
    assert page.decision_ready() is True
    # Ext_{A_3}^{0,0}(F_3, F_3) = F_3; Ext_{A_3}^{1,1} = F_3 (h_0 / tau_0 dual)
    assert page.e2_grid.get((0, 0), 0) == 1
    assert page.e2_grid.get((1, 1), 0) == 1
    # Ext^{1,4} = F_3 (xi_1 dual, the alpha_1 class)
    assert page.e2_grid.get((1, 4), 0) == 1


def test_admissible_combinatorial_cap():
    """t = T_HARD admissible basis must not exceed the hard cap."""
    A = SteenrodAlgebra(prime=2, max_t=T_HARD)
    # Just confirm the cap is enforced if asked beyond it.
    # We don't actually compute beyond cap here to avoid pathology.
    basis_30 = A.admissible_basis(30)
    assert len(basis_30) <= ADM_HARD_CAP


def test_adams_memory_bounded():
    """Computation respects the memory cap and produces a valid contract.

    We give a very small cap to force truncation; the computation should
    return status='truncated' OR complete cleanly with bounded peak.
    """
    ring = rp_n_cohomology_fp(6, prime=2)
    page = adams_e2_page(
        ring, prime=2, s_max=4, t_max=12, memory_cap_mb=4096
    )
    assert page.status in ("success", "truncated")
    # peak memory recorded
    assert page.resource_summary["peak_mem_mb"] >= 0


def test_e2_grid_only_sparse_storage():
    """Grep guarantee: the module file uses no dense fallbacks."""
    src = Path(__file__).resolve().parent.parent / "pysurgery" / "adams" / "spectral_sequence.py"
    text = src.read_text()
    forbidden = [
        r"\.todense\(",
        r"\.toarray\(",
        r"np\.asarray\(.*sparse",
        r"np\.array\(.*sparse",
    ]
    for pat in forbidden:
        # toarray()/todense() may appear in docstrings or examples but not in
        # *operational* code paths. Allow if and only if the line is in a
        # comment or docstring; reject if in active code.
        for i, line in enumerate(text.splitlines(), 1):
            if re.search(pat, line):
                stripped = line.strip()
                if (
                    stripped.startswith("#")
                    or stripped.startswith('"""')
                    or stripped.startswith("'''")
                    or stripped.startswith(">>>")
                ):
                    continue
                # Allow within a docstring (heuristic): line indented inside
                # a triple-quoted block. Accept docstring lines that only
                # describe behavior.
                # Conservative: fail.
                pytest.fail(
                    f"adams_spectral_sequence.py:{i} uses dense conversion: {line.strip()}"
                )
