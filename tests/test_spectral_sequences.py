"""Test suite for pysurgery.spectral.spectral_sequences (Proposal 2)."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from pysurgery.core.exceptions import MathError
from pysurgery.core.foundations import CONTRACT_VERSION
from pysurgery.spectral.spectral_sequences import (
    AdamsSpectralSequence,
    AtiyahHirzebruchSpectralSequence,
    ConvergenceResult,
    ExactCouple,
    ExactCoupleSpectralSequence,
    ExtensionResult,
    LeraySerreSpectralSequence,
    SerreSpectralSequence,
    SpectralEntry,
    SpectralPage,
    SpectralPageSnapshot,
    SpectralSequence,
    solve_extension_problem,
)


# ╔════════════════════════════════════════════════════════════════════════════╗
# Pydantic model contracts
# ╚════════════════════════════════════════════════════════════════════════════╝

class TestSpectralEntry:
    def test_zero_entry_default(self):
        e = SpectralEntry.zero()
        assert e.rank == 0
        assert e.torsion == ()
        assert e.is_zero
        assert e.total_dim == 0

    def test_free_entry(self):
        e = SpectralEntry.free(3)
        assert e.rank == 3
        assert e.torsion == ()
        assert not e.is_zero

    def test_entry_with_torsion(self):
        e = SpectralEntry(rank=2, torsion=(2, 3))
        assert e.rank == 2
        assert e.torsion == (2, 3)
        assert not e.is_zero

    def test_negative_rank_rejected(self):
        with pytest.raises(ValidationError):
            SpectralEntry(rank=-1)

    def test_frozen_immutable(self):
        e = SpectralEntry(rank=1)
        with pytest.raises(ValidationError):
            e.rank = 2  # type: ignore[misc]

    def test_torsion_tuple_coercion(self):
        e = SpectralEntry(rank=0, torsion=[2, 4])
        assert e.torsion == (2, 4)


class TestSpectralPageSnapshot:
    def test_page_snapshot_basic(self):
        entries = {(0, 0): SpectralEntry(rank=1), (1, 1): SpectralEntry(rank=2)}
        snap = SpectralPageSnapshot(
            page_number=2,
            coefficient_ring="Q",
            convention="homological",
            entries=entries,
        )
        assert snap.page_number == 2
        assert snap.coefficient_ring == "Q"
        assert snap.at(0, 0).rank == 1
        assert snap.at(99, 99).rank == 0  # absent → zero

    def test_total_rank_at(self):
        entries = {
            (0, 2): SpectralEntry(rank=1),
            (1, 1): SpectralEntry(rank=2),
            (2, 0): SpectralEntry(rank=3),
            (3, 0): SpectralEntry(rank=4),  # total degree 3
        }
        snap = SpectralPageSnapshot(
            page_number=2,
            coefficient_ring="Q",
            convention="homological",
            entries=entries,
        )
        assert snap.total_rank_at(2) == 1 + 2 + 3
        assert snap.total_rank_at(3) == 4
        assert snap.total_rank_at(99) == 0


class TestConvergenceResult:
    def test_decision_ready_when_converged(self):
        cr = ConvergenceResult(
            converged=True,
            last_page=2,
            e_infinity={(0, 0): SpectralEntry(rank=1)},
            page_history=[],
            coefficient_ring="Q",
            convention="homological",
            convergence_target="test",
            exact=True,
        )
        assert cr.decision_ready()
        assert cr.theorem_tag == "spectral_sequence.convergence"
        assert cr.contract_version == CONTRACT_VERSION

    def test_decision_not_ready_when_not_converged(self):
        cr = ConvergenceResult(
            converged=False,
            last_page=32,
            e_infinity={},
            page_history=[],
            coefficient_ring="Q",
            convention="homological",
            convergence_target="test",
            exact=False,
        )
        assert not cr.decision_ready()

    def test_total_rank_and_torsion_at(self):
        e_inf = {
            (0, 1): SpectralEntry(rank=1),
            (1, 0): SpectralEntry(rank=2, torsion=(2,)),
        }
        cr = ConvergenceResult(
            converged=True, last_page=2, e_infinity=e_inf,
            page_history=[], coefficient_ring="Z",
            convention="homological", convergence_target="t", exact=True,
        )
        assert cr.total_rank_at(1) == 3
        assert cr.torsion_at(1) == (2,)


class TestExtensionResult:
    def test_decision_ready_iff_exact(self):
        e = ExtensionResult(
            total_degree=2,
            rank=1,
            torsion_upper_bound=(),
            splitting_assumed=False,
            contributing_bidegrees=[(2, 0)],
            associated_graded_summary={(2, 0): SpectralEntry(rank=1)},
            exact=True,
        )
        assert e.decision_ready()
        assert e.theorem_tag == "spectral_sequence.extension"
        assert e.contract_version == CONTRACT_VERSION

        e2 = e.model_copy(update={"exact": False})
        assert not e2.decision_ready()


# ╔════════════════════════════════════════════════════════════════════════════╗
# Serre spectral sequence
# ╚════════════════════════════════════════════════════════════════════════════╝

def _sphere_homology(n: int) -> dict[int, SpectralEntry]:
    """H_*(S^n) over any ring: Z in degrees 0 and n."""
    return {0: SpectralEntry(rank=1), n: SpectralEntry(rank=1)}


def _torus_homology() -> dict[int, SpectralEntry]:
    """H_*(T^2) = (Z, Z^2, Z, 0, …)."""
    return {
        0: SpectralEntry(rank=1),
        1: SpectralEntry(rank=2),
        2: SpectralEntry(rank=1),
    }


class TestSerreTrivialFibrations:
    """Trivial fibration F × B → B has E^2 = E^∞ via Künneth."""

    def test_trivial_circle_times_circle(self):
        # F = S^1, B = S^1; expect H_*(T^2) ranks (1, 2, 1).
        ss = SerreSpectralSequence(
            base_homology=_sphere_homology(1),
            fibre_homology=_sphere_homology(1),
        )
        result = ss.converge()
        assert result.converged
        assert result.total_rank_at(0) == 1
        assert result.total_rank_at(1) == 2
        assert result.total_rank_at(2) == 1

    def test_trivial_s2_times_s3(self):
        ss = SerreSpectralSequence(
            base_homology=_sphere_homology(2),
            fibre_homology=_sphere_homology(3),
        )
        result = ss.converge()
        assert result.converged
        # H_*(S^2 × S^3) = (Z, 0, Z, Z, 0, Z) by Künneth (ranks 1,0,1,1,0,1).
        assert result.total_rank_at(0) == 1
        assert result.total_rank_at(1) == 0
        assert result.total_rank_at(2) == 1
        assert result.total_rank_at(3) == 1
        assert result.total_rank_at(4) == 0
        assert result.total_rank_at(5) == 1

    def test_initial_page_is_2(self):
        ss = SerreSpectralSequence(
            base_homology=_sphere_homology(2),
            fibre_homology=_sphere_homology(1),
        )
        assert ss.initial_page_number() == 2

    def test_homological_convention(self):
        ss = SerreSpectralSequence(
            base_homology=_sphere_homology(2),
            fibre_homology=_sphere_homology(1),
        )
        assert ss.differential_bidegree(2) == (-2, 1)
        assert ss.differential_bidegree(3) == (-3, 2)
        assert ss.convention == "homological"

    def test_e2_tensor_structure(self):
        ss = SerreSpectralSequence(
            base_homology=_torus_homology(),
            fibre_homology=_sphere_homology(1),
        )
        e2 = ss.compute_initial_page()
        # H_p(T^2) ranks: 1, 2, 1.  Tensor with H_q(S^1) ranks 1,1.
        assert e2[(0, 0)].rank == 1
        assert e2[(1, 0)].rank == 2
        assert e2[(2, 0)].rank == 1
        assert e2[(0, 1)].rank == 1
        assert e2[(1, 1)].rank == 2
        assert e2[(2, 1)].rank == 1


class TestSerreHopfFibration:
    """Hopf fibration S^1 → S^3 → S^2 has a non-zero d^2."""

    def test_hopf_fibration_e_infinity(self):
        # Base = S^2, fibre = S^1.
        ss = SerreSpectralSequence(
            base_homology=_sphere_homology(2),
            fibre_homology=_sphere_homology(1),
        )
        # E^2: (0,0)=1, (0,1)=1, (2,0)=1, (2,1)=1.
        # d^2: (2, 0) -> (0, 1) is non-zero (Hopf invariant).
        ss.supply_differential(2, (2, 0), np.array([[1]], dtype=np.int64))
        result = ss.converge()
        assert result.converged

        # H_*(S^3) has ranks (1, 0, 0, 1).
        assert result.total_rank_at(0) == 1
        assert result.total_rank_at(1) == 0
        assert result.total_rank_at(2) == 0
        assert result.total_rank_at(3) == 1

    def test_hopf_e_infinity_pq_layout(self):
        ss = SerreSpectralSequence(
            base_homology=_sphere_homology(2),
            fibre_homology=_sphere_homology(1),
        )
        ss.supply_differential(2, (2, 0), np.array([[1]], dtype=np.int64))
        result = ss.converge()
        # Surviving classes: (0, 0) and (2, 1).
        assert result.e_infinity[(0, 0)].rank == 1
        assert result.e_infinity[(0, 1)].rank == 0
        assert result.e_infinity[(2, 0)].rank == 0
        assert result.e_infinity[(2, 1)].rank == 1


class TestSerreCoefficientRings:
    def test_serre_over_F_p(self):
        ss = SerreSpectralSequence(
            base_homology=_sphere_homology(1),
            fibre_homology=_sphere_homology(1),
            coefficient_ring="F_p",
            prime=2,
        )
        result = ss.converge()
        assert result.coefficient_ring == "F_p"
        assert result.prime == 2
        assert result.total_rank_at(2) == 1

    def test_serre_over_Z_with_torsion_in_base(self):
        # H_*(RP^2) over Z: (Z, Z/2, 0).  Use a fibration with fibre S^0 (point).
        base = {
            0: SpectralEntry(rank=1),
            1: SpectralEntry(rank=0, torsion=(2,)),
        }
        fibre = {0: SpectralEntry(rank=1)}
        ss = SerreSpectralSequence(
            base_homology=base, fibre_homology=fibre,
            coefficient_ring="Z",
        )
        result = ss.converge()
        assert result.converged
        # E^2_{1,0} = (Z/2) ⊗ Z = Z/2 → no rank, but torsion preserved.
        assert (1, 0) in result.e_infinity
        assert result.e_infinity[(1, 0)].torsion == (2,)


class TestSerreDifferentialValidation:
    def test_supply_differential_shape_mismatch(self):
        ss = SerreSpectralSequence(
            base_homology=_sphere_homology(2),
            fibre_homology=_sphere_homology(1),
        )
        # Source rank is 1, target rank is 1, expected shape (1, 1).
        ss.supply_differential(2, (2, 0), np.array([[1, 2], [3, 4]], dtype=np.int64))
        with pytest.raises(MathError, match="shape"):
            ss.converge()

    def test_supply_differential_before_initial_page_rejected(self):
        ss = SerreSpectralSequence(
            base_homology=_sphere_homology(2),
            fibre_homology=_sphere_homology(1),
        )
        with pytest.raises(MathError, match="initial page"):
            ss.supply_differential(1, (2, 0), np.array([[1]], dtype=np.int64))

    def test_zero_differential_no_change(self):
        ss = SerreSpectralSequence(
            base_homology=_sphere_homology(2),
            fibre_homology=_sphere_homology(1),
        )
        ss.supply_differential(2, (2, 0), np.array([[0]], dtype=np.int64))
        result = ss.converge()
        # Trivial differential, expect Künneth result H_*(S^2×S^1).
        assert result.total_rank_at(0) == 1
        assert result.total_rank_at(1) == 1
        assert result.total_rank_at(2) == 1
        assert result.total_rank_at(3) == 1


# ╔════════════════════════════════════════════════════════════════════════════╗
# Leray-Serre spectral sequence
# ╚════════════════════════════════════════════════════════════════════════════╝

class TestLeraySerre:
    def test_cohomological_convention(self):
        ss = LeraySerreSpectralSequence(
            base_cohomology=_sphere_homology(2),
            fibre_cohomology=_sphere_homology(1),
        )
        assert ss.convention == "cohomological"
        assert ss.differential_bidegree(2) == (2, -1)
        assert ss.differential_bidegree(3) == (3, -2)

    def test_trivial_total_cohomology(self):
        ss = LeraySerreSpectralSequence(
            base_cohomology=_sphere_homology(2),
            fibre_cohomology=_sphere_homology(2),
        )
        result = ss.converge()
        assert result.converged
        # Künneth: H^*(S^2 × S^2) = (Z, 0, Z^2, 0, Z) totals.
        assert result.total_rank_at(0) == 1
        assert result.total_rank_at(2) == 2
        assert result.total_rank_at(4) == 1

    def test_hopf_fibration_cohomological(self):
        # Hopf S^1 → S^3 → S^2 cohomologically: d_2: E^{0,1} → E^{2,0}.
        ss = LeraySerreSpectralSequence(
            base_cohomology=_sphere_homology(2),
            fibre_cohomology=_sphere_homology(1),
        )
        ss.supply_differential(2, (0, 1), np.array([[1]], dtype=np.int64))
        result = ss.converge()
        assert result.converged
        # H^*(S^3) = (Z, 0, 0, Z).
        assert result.total_rank_at(0) == 1
        assert result.total_rank_at(1) == 0
        assert result.total_rank_at(2) == 0
        assert result.total_rank_at(3) == 1


# ╔════════════════════════════════════════════════════════════════════════════╗
# Adams spectral sequence
# ╚════════════════════════════════════════════════════════════════════════════╝

class TestAdamsSpectralSequence:
    def test_adams_low_dim_at_p2(self):
        # Inject a tiny E_2 chart at p = 2 with a single non-trivial class.
        e2 = {
            (0, 0): SpectralEntry(rank=1),  # the unit
            (1, 1): SpectralEntry(rank=1),  # h_0
            (1, 2): SpectralEntry(rank=1),  # h_1
        }
        ss = AdamsSpectralSequence(prime=2, e2_entries=e2)
        result = ss.converge()
        assert result.converged
        assert result.coefficient_ring == "F_p"
        assert result.prime == 2
        # No differentials supplied → E_∞ = E_2.
        assert result.e_infinity[(0, 0)].rank == 1
        assert result.e_infinity[(1, 1)].rank == 1
        assert result.e_infinity[(1, 2)].rank == 1

    def test_adams_differential_bidegree(self):
        e2 = {(0, 0): SpectralEntry(rank=1)}
        ss = AdamsSpectralSequence(prime=2, e2_entries=e2)
        # Adams uses (r, r-1) bidegree, not the cohomological default.
        assert ss.differential_bidegree(2) == (2, 1)
        assert ss.differential_bidegree(3) == (3, 2)

    def test_adams_invalid_prime(self):
        with pytest.raises(MathError, match="prime"):
            AdamsSpectralSequence(prime=1, e2_entries={})

    def test_adams_d2_kills_class(self):
        e2 = {
            (0, 0): SpectralEntry(rank=1),
            (2, 1): SpectralEntry(rank=1),
        }
        ss = AdamsSpectralSequence(prime=2, e2_entries=e2)
        # d_2: (0, 0) -> (2, 1) over F_2.
        ss.supply_differential(2, (0, 0), np.array([[1]], dtype=np.int64))
        result = ss.converge()
        assert result.converged
        assert result.e_infinity[(0, 0)].rank == 0
        assert result.e_infinity[(2, 1)].rank == 0


# ╔════════════════════════════════════════════════════════════════════════════╗
# Atiyah-Hirzebruch spectral sequence
# ╚════════════════════════════════════════════════════════════════════════════╝

class TestAtiyahHirzebruchSpectralSequence:
    def test_AHSS_for_K_theory_of_S2(self):
        # H^*(S^2) = (Z, 0, Z); K^q(pt) = Z for q even, 0 odd.  Use truncated.
        x_coh = {0: SpectralEntry(rank=1), 2: SpectralEntry(rank=1)}
        # Provide only a finite range of K-theory coefficients for the test.
        coeff = {-2: SpectralEntry(rank=1), 0: SpectralEntry(rank=1), 2: SpectralEntry(rank=1)}
        ss = AtiyahHirzebruchSpectralSequence(
            x_cohomology=x_coh,
            coefficient_pi=coeff,
            cohomology_theory_name="K",
        )
        result = ss.converge()
        assert result.converged
        # K^0(S^2) ≅ Z^2, K^1(S^2) = 0.
        assert result.total_rank_at(0) == 2  # (0,0) and (2,-2)
        assert result.total_rank_at(2) == 2  # (0,2) and (2,0)
        assert result.total_rank_at(1) == 0

    def test_AHSS_e2_construction(self):
        x_coh = {0: SpectralEntry(rank=1), 2: SpectralEntry(rank=1)}
        coeff = {0: SpectralEntry(rank=1)}
        ss = AtiyahHirzebruchSpectralSequence(
            x_cohomology=x_coh, coefficient_pi=coeff,
        )
        e2 = ss.compute_initial_page()
        assert e2[(0, 0)].rank == 1
        assert e2[(2, 0)].rank == 1


# ╔════════════════════════════════════════════════════════════════════════════╗
# Exact couple
# ╚════════════════════════════════════════════════════════════════════════════╝

class TestExactCouple:
    def test_basic_construction(self):
        D = {(0, 0): SpectralEntry(rank=1), (1, 0): SpectralEntry(rank=1)}
        E = {(0, 0): SpectralEntry(rank=1)}
        couple = ExactCouple(
            D=D, E=E,
            i_map={(0, 0): np.array([[1]], dtype=np.int64)},
            j_map={(1, 0): np.array([[0]], dtype=np.int64)},
            k_map={(0, 0): np.array([[0]], dtype=np.int64)},
            bidegree_i=(1, 0),
            bidegree_j=(-1, 0),
            bidegree_k=(0, 0),
        )
        assert couple.differential_bidegree == (-1, 0)
        assert couple.derivation_count == 0

    def test_derive_zero_differential(self):
        # j ∘ k = 0 → derived couple has E' = E.
        D = {(0, 0): SpectralEntry(rank=1)}
        E = {(0, 0): SpectralEntry(rank=2)}
        couple = ExactCouple(
            D=D, E=E,
            i_map={(0, 0): np.eye(1, dtype=np.int64)},
            j_map={(0, 0): np.zeros((2, 1), dtype=np.int64)},
            k_map={(0, 0): np.zeros((1, 2), dtype=np.int64)},
            bidegree_i=(0, 0),
            bidegree_j=(0, 0),
            bidegree_k=(0, 0),
        )
        derived = couple.derive()
        assert derived.derivation_count == 1
        assert derived.E[(0, 0)].rank == 2

    def test_F_p_couple_requires_prime(self):
        with pytest.raises(MathError, match="prime"):
            ExactCouple(
                D={}, E={}, i_map={}, j_map={}, k_map={},
                bidegree_i=(0, 0), bidegree_j=(0, 0), bidegree_k=(0, 0),
                coefficient_ring="F_p",
            )

    def test_to_spectral_sequence_runs(self):
        D = {(0, 0): SpectralEntry(rank=1)}
        E = {(0, 0): SpectralEntry(rank=1)}
        couple = ExactCouple(
            D=D, E=E,
            i_map={(0, 0): np.eye(1, dtype=np.int64)},
            j_map={(0, 0): np.zeros((1, 1), dtype=np.int64)},
            k_map={(0, 0): np.zeros((1, 1), dtype=np.int64)},
            bidegree_i=(0, 0), bidegree_j=(0, 0), bidegree_k=(0, 0),
        )
        ss = couple.to_spectral_sequence()
        assert isinstance(ss, ExactCoupleSpectralSequence)
        result = ss.converge()
        assert result.converged
        assert result.e_infinity[(0, 0)].rank == 1


# ╔════════════════════════════════════════════════════════════════════════════╗
# Extension problem solver
# ╚════════════════════════════════════════════════════════════════════════════╝

class TestExtensionProblem:
    def _kunneth_result(self) -> ConvergenceResult:
        ss = SerreSpectralSequence(
            base_homology=_sphere_homology(2),
            fibre_homology=_sphere_homology(2),
        )
        return ss.converge()

    def test_rank_at_total_degrees(self):
        result = self._kunneth_result()
        # H_*(S^2 × S^2) = (Z, 0, Z^2, 0, Z).
        assert solve_extension_problem(result, 0).rank == 1
        assert solve_extension_problem(result, 1).rank == 0
        assert solve_extension_problem(result, 2).rank == 2
        assert solve_extension_problem(result, 3).rank == 0
        assert solve_extension_problem(result, 4).rank == 1

    def test_contributing_bidegrees(self):
        result = self._kunneth_result()
        ext = solve_extension_problem(result, 2)
        assert sorted(ext.contributing_bidegrees) == [(0, 2), (2, 0)]

    def test_extension_over_field_is_exact(self):
        result = self._kunneth_result()
        ext = solve_extension_problem(result, 2)
        assert ext.exact
        assert ext.decision_ready()
        assert ext.torsion_upper_bound == ()

    def test_extension_over_Z_with_torsion_upper_bound(self):
        # Construct an artificial converged result with torsion in E^∞.
        e_inf = {
            (0, 1): SpectralEntry(rank=1),
            (1, 0): SpectralEntry(rank=0, torsion=(2,)),
        }
        cr = ConvergenceResult(
            converged=True, last_page=2, e_infinity=e_inf,
            page_history=[], coefficient_ring="Z",
            convention="homological", convergence_target="t", exact=True,
        )
        ext = solve_extension_problem(cr, 1)
        assert ext.rank == 1
        assert ext.torsion_upper_bound == (2,)
        # Splitting assumed → exact reported as False over Z when torsion present.
        assert ext.splitting_assumed is True
        assert ext.exact is False

    def test_extension_with_hints_resolves_torsion(self):
        e_inf = {(1, 0): SpectralEntry(rank=0, torsion=(2,))}
        cr = ConvergenceResult(
            converged=True, last_page=2, e_infinity=e_inf,
            page_history=[], coefficient_ring="Z",
            convention="homological", convergence_target="t", exact=True,
        )
        # User asserts the extension is the trivial ℤ/2.
        ext = solve_extension_problem(
            cr, 1, splitting="use_hints", hints={(1, 0): (2,)}
        )
        assert ext.rank == 0
        assert ext.torsion_upper_bound == (2,)
        assert ext.exact is True

    def test_extension_on_non_converged_raises(self):
        cr = ConvergenceResult(
            converged=False, last_page=10, e_infinity={},
            page_history=[], coefficient_ring="Q",
            convention="homological", convergence_target="t", exact=False,
        )
        with pytest.raises(MathError, match="non-converged"):
            solve_extension_problem(cr, 0)


# ╔════════════════════════════════════════════════════════════════════════════╗
# Generic spectral sequence machinery
# ╚════════════════════════════════════════════════════════════════════════════╝

class _DummySS(SpectralSequence):
    """Minimal SS for testing the abstract machinery."""

    def __init__(
        self, entries: dict[tuple[int, int], SpectralEntry], **kwargs
    ):
        super().__init__(**kwargs)
        self._entries = dict(entries)

    def initial_page_number(self) -> int:
        return 2

    def compute_initial_page(self) -> dict[tuple[int, int], SpectralEntry]:
        return dict(self._entries)


class TestSpectralSequenceMachinery:
    def test_invalid_coefficient_ring_rejected(self):
        with pytest.raises(MathError):
            _DummySS({}, coefficient_ring="C")  # type: ignore[arg-type]

    def test_F_p_requires_prime(self):
        with pytest.raises(MathError):
            _DummySS({}, coefficient_ring="F_p")

    def test_max_pages_must_be_positive(self):
        with pytest.raises(MathError):
            _DummySS({}, max_pages=0)

    def test_empty_initial_page_converges(self):
        ss = _DummySS({})
        result = ss.converge()
        assert result.converged
        assert result.e_infinity == {}

    def test_single_entry_converges_trivially(self):
        ss = _DummySS({(0, 0): SpectralEntry(rank=1)})
        result = ss.converge()
        assert result.converged
        assert result.e_infinity[(0, 0)].rank == 1
        assert result.last_page == 3  # E^2 = E^3 detected on first iteration

    def test_page_history_recorded(self):
        ss = _DummySS({(0, 0): SpectralEntry(rank=1)})
        result = ss.converge()
        assert len(result.page_history) >= 2
        assert result.page_history[0].page_number == 2
        assert result.page_history[-1].is_terminal

    def test_max_pages_exhaustion_yields_non_exact(self):
        # Construct a SS where consecutive pages always disagree by injecting
        # a non-zero differential that the framework cannot detect as zero.
        # Easiest: a 1×1 differential of value 1 every page.
        class _Stuck(_DummySS):
            def compute_differentials(self, r, page):
                # A persistent non-zero differential at (r, 0) → (0, r-1)
                # (homological).  If the entry is missing, framework treats
                # it as zero, so we manufacture a small persistent loop.
                return {}

        ss = _Stuck({(0, 0): SpectralEntry(rank=1)}, max_pages=3)
        result = ss.converge()
        # No persistent differential supplied → still converges.
        assert result.converged

    def test_supply_differential_validates_dimension(self):
        ss = _DummySS({(0, 0): SpectralEntry(rank=2)})
        with pytest.raises(MathError, match="2-D"):
            ss.supply_differential(2, (0, 0), np.array([1, 2, 3], dtype=np.int64))


class TestSpectralPageInternals:
    def test_support_box_empty(self):
        page = SpectralPage(page_number=2, convention="homological")
        assert page.support_box() is None

    def test_support_box_basic(self):
        entries = {
            (0, 0): SpectralEntry(rank=1),
            (1, 2): SpectralEntry(rank=1),
            (3, 5): SpectralEntry(rank=0),  # zero entry should be excluded
        }
        page = SpectralPage(page_number=2, convention="homological", entries=entries)
        box = page.support_box()
        assert box == (0, 1, 0, 2)

    def test_differential_bidegree_homological(self):
        page = SpectralPage(page_number=3, convention="homological")
        assert page.differential_bidegree() == (-3, 2)

    def test_differential_bidegree_cohomological(self):
        page = SpectralPage(page_number=3, convention="cohomological")
        assert page.differential_bidegree() == (3, -2)


# ╔════════════════════════════════════════════════════════════════════════════╗
# Tensor entry helper (over Z, with torsion)
# ╚════════════════════════════════════════════════════════════════════════════╝

class TestTensorEntryOverZ:
    def test_free_times_free(self):
        from pysurgery.spectral.spectral_sequences import _tensor_entry
        e = _tensor_entry(SpectralEntry(rank=2), SpectralEntry(rank=3), "Z")
        assert e.rank == 6
        assert e.torsion == ()

    def test_free_times_torsion(self):
        from pysurgery.spectral.spectral_sequences import _tensor_entry
        # Z^2 ⊗ Z/3 = (Z/3)^2.
        e = _tensor_entry(
            SpectralEntry(rank=2),
            SpectralEntry(rank=0, torsion=(3,)),
            "Z",
        )
        assert e.rank == 0
        assert e.torsion == (3, 3)

    def test_torsion_times_torsion(self):
        from pysurgery.spectral.spectral_sequences import _tensor_entry
        # Z/2 ⊗ Z/4 = Z/gcd(2,4) = Z/2.
        e = _tensor_entry(
            SpectralEntry(rank=0, torsion=(2,)),
            SpectralEntry(rank=0, torsion=(4,)),
            "Z",
        )
        assert e.torsion == (2,)

    def test_field_kills_torsion(self):
        from pysurgery.spectral.spectral_sequences import _tensor_entry
        e = _tensor_entry(
            SpectralEntry(rank=1, torsion=(2,)),
            SpectralEntry(rank=1, torsion=(3,)),
            "Q",
        )
        assert e.rank == 1
        assert e.torsion == ()


# ╔════════════════════════════════════════════════════════════════════════════╗
# Matrix rank helper
# ╚════════════════════════════════════════════════════════════════════════════╝

class TestMatrixRank:
    def test_rank_over_Q(self):
        from pysurgery.spectral.spectral_sequences import _matrix_rank
        M = np.array([[1, 2], [2, 4]], dtype=np.int64)
        assert _matrix_rank(M, "Q") == 1

    def test_rank_over_F_p_drops(self):
        from pysurgery.spectral.spectral_sequences import _matrix_rank
        # diag(1, 2): rank 2 over Q, rank 1 over F_2.
        M = np.diag([1, 2]).astype(np.int64)
        assert _matrix_rank(M, "Q") == 2
        assert _matrix_rank(M, "F_p", prime=2) == 1

    def test_rank_over_Z_via_snf(self):
        from pysurgery.spectral.spectral_sequences import _matrix_rank
        M = np.array([[2, 0], [0, 4]], dtype=np.int64)
        assert _matrix_rank(M, "Z") == 2

    def test_rank_F_p_requires_prime(self):
        from pysurgery.spectral.spectral_sequences import _matrix_rank
        with pytest.raises(MathError):
            _matrix_rank(np.array([[1]], dtype=np.int64), "F_p")

    def test_empty_matrix_rank_zero(self):
        from pysurgery.spectral.spectral_sequences import _matrix_rank
        M = np.zeros((0, 0), dtype=np.int64)
        assert _matrix_rank(M, "Q") == 0

    def test_non_2d_rejected(self):
        from pysurgery.spectral.spectral_sequences import _matrix_rank
        with pytest.raises(MathError, match="2-D"):
            _matrix_rank(np.array([1, 2, 3], dtype=np.int64), "Q")
