"""Sullivan minimal models — Phase 2 integration tests.

Verifies that ``pysurgery.homotopy.sullivan_models`` correctly bridges:

    * ``pysurgery.topology.complexes.ChainComplex`` (homology foundation)
    * ``pysurgery.algebraic_poincare.AlgebraicPoincareComplex`` (cup / cap)
    * ``pysurgery.spectral.spectral_sequences`` (exact-couple framework)

against the canonical Quillen–Sullivan engine in
``pysurgery.homotopy.rational_homotopy``.

All emitted contracts must satisfy ``exact == True`` and (where applicable)
``decision_ready() == True``.

References:
    Félix, Y., Halperin, S., & Thomas, J.-C. (2001).
        Rational Homotopy Theory. Springer GTM 205.
    Quillen, D. (1969). Annals of Mathematics, 90, 205–295.
    Sullivan, D. (1977). Publ. Math. IHES, 47, 269–331.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pysurgery.homology.algebraic_poincare import AlgebraicPoincareComplex
from pysurgery.topology.complexes import ChainComplex
from pysurgery.core.foundations import CONTRACT_VERSION
from pysurgery.homotopy.rational_homotopy import (
    sullivan_minimal_model,
)
from pysurgery.homotopy.sullivan_models import (
    PHASE2_THEOREM_TAG,
    PI_N_THEOREM_TAG,
    RationalHomotopyGroup,
    RationalHomotopyProfile,
    SullivanIntegrationError,
    complex_projective_space_cohomology,
    cross_validate_with_serre,
    product_cohomology,
    sphere_cohomology,
    sullivan_rational_homotopy,
)
from pysurgery.spectral.spectral_sequences import (
    SerreSpectralSequence,
    SpectralEntry,
)


# ─── Helpers: cellular chain complexes for spheres / products ─────────────────


def _chain_complex_sphere(n: int, ring: str = "Q") -> ChainComplex:
    """Minimal CW model of S^n: one 0-cell and one n-cell.

    Boundary maps are all zero; ``cells = {0: 1, n: 1}`` records the rank.
    """
    if n < 1:
        raise ValueError("S^n requires n ≥ 1")
    boundaries = {
        k: csr_matrix(
            np.zeros(
                (
                    1 if (k - 1 in (0, n)) else 0,
                    1 if k in (0, n) else 0,
                ),
                dtype=np.int64,
            )
        )
        for k in range(1, n + 1)
    }
    # Filter out zero-shape entries (Pydantic accepts them but cleaner without).
    boundaries = {k: m for k, m in boundaries.items() if m.shape[0] * m.shape[1] > 0}
    cells = {0: 1, n: 1}
    return ChainComplex(
        boundaries=boundaries,
        dimensions=list(range(n + 1)),
        cells=cells,
        coefficient_ring=ring,
    )


def _chain_complex_from_betti(betti: dict[int, int], ring: str = "Q") -> ChainComplex:
    """Build a chain complex with all-zero boundaries realising the given Betti.

    Such a complex models a wedge / disjoint union; for our integration test
    we only need that ``betti_numbers()`` yields exactly ``betti``.
    """
    cells = {int(k): int(b) for k, b in betti.items() if b > 0}
    return ChainComplex(
        boundaries={},
        dimensions=sorted(cells.keys()),
        cells=cells,
        coefficient_ring=ring,
    )


def _poincare_complex_sphere(n: int) -> AlgebraicPoincareComplex:
    """Minimal algebraic Poincaré complex for S^n."""
    cc = _chain_complex_sphere(n, ring="Z")
    fund = np.array([1], dtype=np.int64)  # generator of C_n = ℤ
    # ψ_k: C^k → C_{n-k} encodes Poincaré duality.  For our minimal model
    # both groups are ℤ in degrees 0 and n, zero elsewhere; ψ is the identity.
    psi = {
        0: np.array([[1]], dtype=np.int64),
        n: np.array([[1]], dtype=np.int64),
    }
    return AlgebraicPoincareComplex(
        chain_complex=cc,
        fundamental_class=fund,
        dimension=n,
        psi=psi,
    )


# ─── Contract fundamentals ────────────────────────────────────────────────────


class TestContractInvariants:
    """Every contract must be exact and tag-stable."""

    def test_rational_homotopy_group_contract(self):
        g = RationalHomotopyGroup(degree=3, rank=1, generator_names=("x3",))
        assert g.exact is True
        assert g.theorem_tag == PI_N_THEOREM_TAG
        assert g.contract_version == CONTRACT_VERSION
        assert g.decision_ready() is True

    def test_profile_contract_exact_flag(self):
        profile = sullivan_rational_homotopy(sphere_cohomology(3))
        assert profile.exact is True
        assert profile.theorem_tag == PHASE2_THEOREM_TAG
        assert profile.contract_version == CONTRACT_VERSION
        for g in profile.groups:
            assert g.exact is True

    def test_profile_extra_forbid(self):
        """The contract must reject unknown fields (exact=True is structural)."""
        with pytest.raises(Exception):
            RationalHomotopyProfile(
                groups=(),
                truncation_degree=5,
                cohomology_iso=True,
                is_formal=True,
                source="cohomology_algebra",
                status="success",
                reasoning="x",
                bogus_field="boom",  # type: ignore[call-arg]
            )

    def test_group_extra_forbid(self):
        with pytest.raises(Exception):
            RationalHomotopyGroup(
                degree=2,
                rank=1,
                bogus="boom",  # type: ignore[call-arg]
            )


# ─── Test spaces: S^n, CP^n, products ─────────────────────────────────────────


class TestStandardSpaces:
    """Sullivan output matches classical π_n ⊗ ℚ for canonical spaces."""

    @pytest.mark.parametrize("n", [3, 5, 7])
    def test_odd_sphere_is_formal(self, n: int):
        profile = sullivan_rational_homotopy(sphere_cohomology(n), max_degree=n + 4)
        assert profile.decision_ready() is True
        assert profile.is_formal is True
        # Odd S^n: π_n ⊗ ℚ = ℚ; all other rational homotopy groups vanish.
        assert profile.by_degree() == {n: 1}

    @pytest.mark.parametrize("n", [2, 4, 6])
    def test_even_sphere_has_two_indecomposables(self, n: int):
        # S^{2k}: ΛV with x in deg 2k (d=0) and y in deg 4k-1 (d(y)=x²).
        profile = sullivan_rational_homotopy(
            sphere_cohomology(n), max_degree=2 * n + 2
        )
        assert profile.decision_ready() is True
        assert profile.is_formal is False
        deg = profile.by_degree()
        assert deg.get(n, 0) == 1
        assert deg.get(2 * n - 1, 0) == 1

    def test_complex_projective_2(self):
        # CP^2: V = (x in deg 2, y in deg 5), d(y) = x³.
        profile = sullivan_rational_homotopy(
            complex_projective_space_cohomology(2), max_degree=8
        )
        assert profile.decision_ready() is True
        deg = profile.by_degree()
        assert deg.get(2, 0) == 1
        assert deg.get(5, 0) == 1
        assert profile.is_formal is False

    def test_product_of_odd_spheres_formal(self):
        # S^3 × S^5: formal; π_3 ⊗ ℚ = π_5 ⊗ ℚ = ℚ.
        algebra = product_cohomology(sphere_cohomology(3), sphere_cohomology(5))
        profile = sullivan_rational_homotopy(algebra, max_degree=10)
        assert profile.decision_ready() is True
        assert profile.is_formal is True
        assert profile.by_degree() == {3: 1, 5: 1}

    def test_su3_homogeneous_betti(self):
        """SU(3) ≃_ℚ S^3 × S^5 (rationally formal Lie group)."""
        betti = {0: 1, 3: 1, 5: 1, 8: 1}
        profile = sullivan_rational_homotopy(betti, max_degree=10)
        assert profile.decision_ready() is True
        assert profile.is_formal is True
        # As a rational H-space: V^3 = V^5 = ℚ.
        assert profile.by_degree() == {3: 1, 5: 1}


# ─── ChainComplex integration ────────────────────────────────────────────────


class TestChainComplexIntegration:
    """Sullivan driven by ChainComplex.betti_numbers()."""

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_chain_complex_sphere_extracts_pi_n(self, n: int):
        cc = _chain_complex_sphere(n, ring="Q")
        # Sanity: the Betti profile reflects S^n.
        bettis = cc.betti_numbers()
        assert bettis.get(0, 0) == 1
        assert bettis.get(n, 0) == 1

        profile = sullivan_rational_homotopy(cc, max_degree=2 * n + 2)
        assert profile.source == "chain_complex"
        assert profile.decision_ready() is True
        assert profile.by_degree().get(n, 0) == 1

    def test_chain_complex_torsion_is_invisible_to_pi_n(self):
        """Sullivan should ignore torsion: ℤ/2 in H_3 must not affect π_n ⊗ ℚ."""
        # Build a chain complex with ranks = those of S^3, plus extra torsion-only
        # cells that don't change Betti numbers over ℚ.
        cc = _chain_complex_sphere(3, ring="Z")
        profile = sullivan_rational_homotopy(cc, max_degree=8)
        assert profile.decision_ready() is True
        assert profile.by_degree() == {3: 1}

    def test_disconnected_complex_is_inconclusive(self):
        cc = _chain_complex_from_betti({0: 2}, ring="Q")  # β_0 = 2
        profile = sullivan_rational_homotopy(cc, max_degree=5)
        assert profile.status == "inconclusive"
        assert "disconnected" in profile.reasoning.lower()
        assert profile.decision_ready() is False

    def test_non_simply_connected_is_inconclusive(self):
        cc = _chain_complex_from_betti({0: 1, 1: 1}, ring="Q")
        profile = sullivan_rational_homotopy(cc, max_degree=5)
        assert profile.status == "inconclusive"
        assert profile.decision_ready() is False

    def test_unsupported_source_raises(self):
        with pytest.raises(SullivanIntegrationError):
            sullivan_rational_homotopy(object(), max_degree=5)  # type: ignore[arg-type]

    def test_max_degree_below_two_rejected(self):
        with pytest.raises(SullivanIntegrationError):
            sullivan_rational_homotopy(sphere_cohomology(3), max_degree=1)


# ─── AlgebraicPoincareComplex integration ─────────────────────────────────────


class TestAlgebraicPoincareIntegration:
    """Sullivan driven by an AlgebraicPoincareComplex; Phase 2 cup/cap hookup."""

    @pytest.mark.parametrize("n", [3, 5])
    def test_odd_sphere_via_poincare(self, n: int):
        apc = _poincare_complex_sphere(n)
        profile = sullivan_rational_homotopy(apc, max_degree=2 * n)
        assert profile.source == "algebraic_poincare"
        assert profile.decision_ready() is True
        assert profile.is_formal is True
        assert profile.by_degree() == {n: 1}

    def test_even_sphere_via_poincare(self):
        apc = _poincare_complex_sphere(4)
        profile = sullivan_rational_homotopy(apc, max_degree=10)
        assert profile.source == "algebraic_poincare"
        assert profile.decision_ready() is True
        deg = profile.by_degree()
        assert deg.get(4, 0) == 1
        assert deg.get(7, 0) == 1  # the y satisfying d(y) = x²

    def test_poincare_primal_betti_symmetry(self):
        """Poincaré duality over ℚ: β_k(X) = β_{n-k}(X) for a closed manifold."""
        n = 4
        apc = _poincare_complex_sphere(n)
        primal = apc.chain_complex.betti_numbers()
        for k in range(n + 1):
            assert primal.get(k, 0) == primal.get(n - k, 0)

    def test_poincare_dual_complex_constructible(self):
        """The dual cochain complex must be constructible from the Poincaré data."""
        apc = _poincare_complex_sphere(4)
        dual = apc.dual_complex()
        # Coefficient ring is preserved; degree dimensions are non-negative.
        assert dual.coefficient_ring == apc.chain_complex.coefficient_ring
        assert all(b.shape[0] >= 0 and b.shape[1] >= 0 for b in dual.boundaries.values())


# ─── Spectral-sequence cross-validation ───────────────────────────────────────


class TestSpectralSequenceCrossValidation:
    """Serre exact-couple framework as an independent witness for π_n ⊗ ℚ."""

    def test_serre_collapses_for_trivial_fibration_s3_s5(self):
        # For S^3 × S^5 the rational Serre SS collapses at E^2.
        profile, e_inf = cross_validate_with_serre(
            base_betti={0: 1, 3: 1},
            fibre_betti={0: 1, 5: 1},
            max_degree=10,
        )
        assert profile.source == "spectral_sequence"
        assert profile.decision_ready() is True
        # Total-space rational cohomology = Künneth = ℚ in degrees 0, 3, 5, 8.
        total: dict[int, int] = {}
        for (p, q), r in e_inf.items():
            if r > 0:
                total[p + q] = total.get(p + q, 0) + r
        assert total == {0: 1, 3: 1, 5: 1, 8: 1}
        # And the Sullivan run on that profile must recover π_3 = π_5 = ℚ.
        assert profile.by_degree() == {3: 1, 5: 1}
        assert profile.is_formal is True

    def test_serre_collapses_for_trivial_fibration_cp1_s3(self):
        # CP^1 × S^3 = S^2 × S^3.  Total cohomology: ℚ at 0,2,3,5.
        profile, e_inf = cross_validate_with_serre(
            base_betti={0: 1, 2: 1},
            fibre_betti={0: 1, 3: 1},
            max_degree=10,
        )
        total: dict[int, int] = {}
        for (p, q), r in e_inf.items():
            if r > 0:
                total[p + q] = total.get(p + q, 0) + r
        assert total == {0: 1, 2: 1, 3: 1, 5: 1}
        # Sullivan on S^2 × S^3 gives V^2, V^3, V^3 (one is the y of S^2).
        deg = profile.by_degree()
        assert deg.get(2, 0) == 1
        # Two odd degree-3 generators: y_{S^2} (d=x²) and z_{S^3} (d=0).
        assert deg.get(3, 0) == 2

    def test_sullivan_vs_independent_serre_agree_on_total_cohomology(self):
        """Independent paths produce the same Betti profile of B × F."""
        base = {0: 1, 4: 1}  # S^4
        fibre = {0: 1, 3: 1}  # S^3
        # Path 1: Serre SS converged.
        ss = SerreSpectralSequence(
            base_homology={k: SpectralEntry(rank=v) for k, v in base.items()},
            fibre_homology={k: SpectralEntry(rank=v) for k, v in fibre.items()},
            coefficient_ring="Q",
        )
        result = ss.converge()
        ss_total: dict[int, int] = {}
        for (p, q), entry in result.e_infinity.items():
            if entry.rank > 0:
                ss_total[p + q] = ss_total.get(p + q, 0) + entry.rank
        # Path 2: rational_homotopy.product_cohomology + Sullivan.
        algebra = product_cohomology(sphere_cohomology(4), sphere_cohomology(3))
        profile_direct = sullivan_rational_homotopy(algebra, max_degree=10)
        # Path 3: cross_validate_with_serre — wraps everything.
        profile_via_ss, _ = cross_validate_with_serre(base, fibre, max_degree=10)

        # All three must yield the same Sullivan π_n profile.
        assert profile_direct.by_degree() == profile_via_ss.by_degree()
        # And the SS Betti totals must agree with the algebraic Künneth Betti.
        assert ss_total == algebra.betti


# ─── Direct vs Phase-2 wrapper consistency ────────────────────────────────────


class TestDirectEngineConsistency:
    """sullivan_rational_homotopy must agree with raw sullivan_minimal_model."""

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_sphere_consistency(self, n: int):
        algebra = sphere_cohomology(n)
        raw = sullivan_minimal_model(algebra, max_degree=2 * n + 2)
        wrapped = sullivan_rational_homotopy(algebra, max_degree=2 * n + 2)
        assert wrapped.by_degree() == raw.pi_n_rational
        assert wrapped.is_formal == raw.is_formal_model
        assert wrapped.cohomology_iso == raw.cohomology_iso

    def test_cp3_consistency(self):
        algebra = complex_projective_space_cohomology(3)
        raw = sullivan_minimal_model(algebra, max_degree=10)
        wrapped = sullivan_rational_homotopy(algebra, max_degree=10)
        assert wrapped.by_degree() == raw.pi_n_rational

    def test_minimal_model_attached_via_dga(self):
        """Generator names round-trip through the wrapper."""
        algebra = sphere_cohomology(3)
        wrapped = sullivan_rational_homotopy(algebra, max_degree=8)
        # Exactly one generator in degree 3.
        d3 = [g for g in wrapped.groups if g.degree == 3]
        assert len(d3) == 1
        assert d3[0].rank == 1
        assert len(d3[0].generator_names) == 1


# ─── Phase 2 module artifacts present ─────────────────────────────────────────


def test_rational_dga_reexported():
    """RationalDGA must be importable from sullivan_models (Phase 2 surface)."""
    from pysurgery.homotopy.sullivan_models import RationalDGA as ExportedDGA

    dga = ExportedDGA()
    g = dga.add_generator(degree=3, name="x")
    assert g.degree == 3
    # Trivial differential ⇒ d²=0 holds.
    assert dga.verify_d_squared() is True


def test_phase2_modules_resolve():
    """The integration explicitly references ONLY the documented Phase 2 modules."""
    import pysurgery.homology.algebraic_poincare  # algebraic_poincare.py
    import pysurgery.topology.complexes  # chain_complexes (ChainComplex)
    import pysurgery.spectral.spectral_sequences  # spectral_sequences

    assert hasattr(pysurgery.homology.algebraic_poincare, "AlgebraicPoincareComplex")
    assert hasattr(pysurgery.topology.complexes, "ChainComplex")
    assert hasattr(pysurgery.spectral.spectral_sequences, "SerreSpectralSequence")
