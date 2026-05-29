"""Tests for compute_pi_n and verify_against_known.

Key invariant: the algorithm always runs, even when a table entry exists.
The verifier only annotates the comparison; it never substitutes the table
value into the public output.
"""
from __future__ import annotations


from pysurgery.adams.spectral_sequence import (
    cp_n_cohomology_fp,
    rp_n_cohomology_fp,
    sphere_cohomology_fp,
)
from pysurgery.homotopy.higher_homotopy_groups import (
    HomotopyGroup,
    sphere_cohomology,
)
from pysurgery.homotopy.homotopy_verifier import (
    ComputedHomotopyGroup,
    VerificationResult,
    detect_space_family,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _hg_sphere(k: int, prime: int = 2, t_max: int = 12) -> HomotopyGroup:
    return HomotopyGroup.from_inputs(
        sphere_cohomology(k),
        fp_cohomology_ring=sphere_cohomology_fp(k, prime=prime),
        adams_prime=prime,
        adams_s_max=4,
        adams_t_max=t_max,
        rational_max_degree=max(k + 2, 6),
        space_label=f"S^{k}",
    )


# ── detect_space_family ───────────────────────────────────────────────────────


def test_detect_sphere_s3_at_p2():
    ring = sphere_cohomology_fp(3, prime=2)
    family, parameter = detect_space_family(ring)
    assert family == "S"
    assert parameter == 3


def test_detect_cp2_at_p2():
    ring = cp_n_cohomology_fp(2, prime=2)
    family, parameter = detect_space_family(ring)
    assert family == "CP"
    assert parameter == 2


def test_detect_cp3_at_p3():
    ring = cp_n_cohomology_fp(3, prime=3)
    family, parameter = detect_space_family(ring)
    assert family == "CP"
    assert parameter == 3


def test_detect_rp2_at_p2():
    ring = rp_n_cohomology_fp(2, prime=2)
    family, parameter = detect_space_family(ring)
    assert family == "RP"
    assert parameter == 2


def test_detect_unknown_returns_none():
    family, parameter = detect_space_family(None)
    assert family is None and parameter is None


# ── compute_pi_n: algorithm always runs ───────────────────────────────────────


def test_compute_pi_n_returns_computed_object_on_s3():
    hg = _hg_sphere(3, prime=2, t_max=10)
    result = hg.compute_pi_n(3)
    assert isinstance(result, ComputedHomotopyGroup)
    assert result.n == 3
    # π_3(S^3) ⊗ ℚ = ℚ
    assert result.free_rank == 1


def test_compute_pi_n_rank_zero_on_high_stem_s3():
    """π_n(S^3) ⊗ ℚ = 0 for n != 3."""
    hg = _hg_sphere(3, prime=2, t_max=10)
    for n in (2, 4, 5, 6, 7):
        result = hg.compute_pi_n(n)
        assert result.free_rank == 0


def test_compute_pi_n_marks_e2_only_as_upper_bound():
    hg = _hg_sphere(2, prime=2, t_max=12)
    result = hg.compute_pi_n(4)
    assert result.is_upper_bound is True
    # v1: every covered prime must emit at least one gap describing the
    # status of d_2 (forced zero or unresolved).
    assert any(("d_2" in g or "differential" in g) for g in result.gaps)


# ── verify_against_known: comparison only, no substitution ────────────────────


def test_verify_returns_a_verification_result_with_table():
    hg = _hg_sphere(3, prime=2, t_max=14)
    result = hg.verify_against_known(3)
    assert isinstance(result, VerificationResult)
    # The verifier always exposes the computed object separately.
    assert isinstance(result.computed, ComputedHomotopyGroup)
    assert result.table is not None
    # Both should agree on rational rank for π_3(S^3) = Z.
    assert result.computed.free_rank == 1
    assert result.table.free_rank == 1


def test_verify_status_match_or_bound_on_rationally_pure_stem():
    """π_3(S^3) = Z is "exact at the rational level".

    At the E_2-only stage the algorithm sees the h_0-tower of the Z-summand
    and conservatively reports it as additional torsion bound. That's an
    honest over-estimate: BOUND_CONTAINS is the correct verdict (the
    published Z is contained in our Z ⊕ Z/2^k bound). When differentials
    and the h_0-tower extension solver land, this row tightens to
    EXACT_MATCH.
    """
    hg = _hg_sphere(3, prime=2, t_max=14)
    result = hg.verify_against_known(3)
    assert result.match_status in {"EXACT_MATCH", "BOUND_CONTAINS"}
    assert result.computed.free_rank == result.table.free_rank == 1


def test_verify_never_violates_table_on_published_stems_s3():
    """For every published stem of S^3 in our window, the algorithm's upper
    bound must contain the table value (or match it exactly).
    BOUND_VIOLATES_TABLE indicates a bug."""
    hg = _hg_sphere(3, prime=2, t_max=14)
    for n in range(2, 8):
        result = hg.verify_against_known(n)
        assert result.match_status in {
            "EXACT_MATCH",
            "BOUND_CONTAINS",
            "NO_TABLE",
        }, f"S^3, n={n}: {result.match_status} - {result.explanation}"


def test_verify_no_table_for_unrecognized_space():
    """An empty/anonymous fp-ring → NO_TABLE."""
    hg = _hg_sphere(3, prime=2, t_max=10)
    # Manually clear the ring to simulate an unrecognized space.
    object.__setattr__(hg, "fp_cohomology_ring", None)
    object.__setattr__(hg, "adams", None)
    result = hg.verify_against_known(3)
    assert result.match_status == "NO_TABLE"


# ── The non-substitution invariant ────────────────────────────────────────────


def test_compute_pi_n_never_uses_known_tables_module():
    """Sanity check: the homotopy_verifier.compute_pi_n function body does
    not import or call into known_homotopy_tables for its output."""
    import pysurgery.homotopy.homotopy_verifier as hv
    src = open(hv.__file__).read()
    # The verifier imports `lookup as table_lookup` and `KnownHomotopyEntry`,
    # but the compute_pi_n function body must not use either.
    # Coarse but effective check: search the compute_pi_n function body.
    start = src.index("def compute_pi_n(")
    end = src.index("\ndef ", start + 1)
    body = src[start:end]
    assert "table_lookup" not in body
    assert "KnownHomotopyEntry" not in body
    assert "KNOWN_PI_N" not in body
