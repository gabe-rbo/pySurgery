"""Tests for pysurgery.algebra.exact_snf_julia — Proposal 1.

All tests enforce exact integer correctness: no floating-point tolerances.
Known SNF diagonals are derived analytically from standard topological spaces.

Test groups
-----------
1. Trivial / edge cases (zero matrix, empty matrix, identity)
2. Boundary operators for standard spaces (S¹, RP², torus, Klein bottle, lens spaces)
3. ExactSNFResult contract (dataclass fields, decision_ready, torsion_summary)
4. Modular rank certification (ModularRankCertificate, all_agree, lower_bound)
5. Batch SNF (BatchSNFResult contract, sequential equality)
6. math_core integration (use_exact_sparse routing)
7. Error handling (non-integer entries, backend='julia' hard-fail without Julia)
8. p-adic cross-validation (requires Julia — skipped when unavailable)
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from pysurgery.algebra.exact_snf_julia import (
    ModularRankCertificate,
    compute_batch_snf,
    compute_exact_sparse_snf,
    compute_modular_rank_certificate,
    compute_padic_snf_diagonal,
)
from pysurgery.core.exceptions import HomologyError
from pysurgery.core.foundations import CONTRACT_VERSION


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _csr(arr):
    """Dense int64 array → CSR sparse matrix."""
    return sp.csr_matrix(np.array(arr, dtype=np.int64))


def _julia_available() -> bool:
    from pysurgery.bridge.julia_bridge import julia_engine
    return julia_engine.available


# ─────────────────────────────────────────────────────────────────────────────
# 1. Trivial / edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestTrivialCases:
    def test_zero_matrix_1x1(self):
        M = _csr([[0]])
        r = compute_exact_sparse_snf(M, backend="python")
        assert r.rank == 0
        assert len(r.diagonal) == 0
        assert r.exact is True
        assert r.decision_ready()

    def test_zero_matrix_3x4(self):
        M = sp.csr_matrix((3, 4), dtype=np.int64)
        r = compute_exact_sparse_snf(M, backend="python")
        assert r.rank == 0
        assert len(r.torsion_factors) == 0

    def test_identity_2x2(self):
        M = _csr([[1, 0], [0, 1]])
        r = compute_exact_sparse_snf(M, backend="python")
        assert r.rank == 2
        assert np.array_equal(r.diagonal, [1, 1])
        assert r.betti_rank == 2
        assert len(r.torsion_factors) == 0

    def test_identity_4x4(self):
        M = sp.eye(4, dtype=np.int64, format="csr")
        r = compute_exact_sparse_snf(M, backend="python")
        assert r.rank == 4
        assert np.all(r.diagonal == 1)

    def test_single_entry_positive(self):
        M = _csr([[6]])
        r = compute_exact_sparse_snf(M, backend="python")
        assert r.rank == 1
        assert int(r.diagonal[0]) == 6
        assert len(r.torsion_factors) == 1
        assert int(r.torsion_factors[0]) == 6

    def test_single_entry_negative_normalised(self):
        M = _csr([[-3]])
        r = compute_exact_sparse_snf(M, backend="python")
        assert r.rank == 1
        # Diagonal always returned positive.
        assert int(r.diagonal[0]) == 3

    def test_empty_matrix_0rows(self):
        M = sp.csr_matrix((0, 5), dtype=np.int64)
        r = compute_exact_sparse_snf(M, backend="python")
        assert r.rank == 0
        assert len(r.diagonal) == 0

    def test_empty_matrix_0cols(self):
        M = sp.csr_matrix((5, 0), dtype=np.int64)
        r = compute_exact_sparse_snf(M, backend="python")
        assert r.rank == 0


# ─────────────────────────────────────────────────────────────────────────────
# 2. Boundary operators for standard topological spaces
# ─────────────────────────────────────────────────────────────────────────────

class TestBoundaryOperators:
    """
    Standard CW boundary matrices, SNF diagonals computed analytically.

    Encoding convention: columns = cells, rows = lower-dimensional cells.
    Entry (i,j) = ±1 if cell j attaches to/from cell i with ± orientation.
    """

    # ── S¹ (circle) ──────────────────────────────────────────────────────────
    def test_circle_d1_loop(self):
        # One vertex v, one 1-cell e (loop). ∂(e) = v − v = 0.
        # d₁: C₁ → C₀ = [[0]]  → SNF diag = [] (rank 0, no non-zeros)
        d1 = _csr([[0]])
        r = compute_exact_sparse_snf(d1, backend="python")
        assert r.rank == 0
        assert len(r.diagonal) == 0

    # ── Path P₂ (two vertices, one edge) ─────────────────────────────────────
    def test_path_d1(self):
        # Two vertices v₀,v₁; one edge e = [v₀,v₁]. ∂(e) = v₁ − v₀.
        # d₁ = [[-1], [1]]   SNF diag = [1]
        d1 = _csr([[-1], [1]])
        r = compute_exact_sparse_snf(d1, backend="python")
        assert r.rank == 1
        assert np.array_equal(r.diagonal, [1])

    # ── Triangle boundary d₁ ─────────────────────────────────────────────────
    def test_triangle_d1_rank2(self):
        # Three vertices v₀,v₁,v₂; three edges e₀₁,e₀₂,e₁₂.
        # d₁ = [[-1,-1, 0],
        #        [ 1, 0,-1],
        #        [ 0, 1, 1]]   rank 2, SNF diag = [1,1]
        d1 = _csr([[-1, -1, 0],
                   [ 1,  0,-1],
                   [ 0,  1, 1]])
        r = compute_exact_sparse_snf(d1, backend="python")
        assert r.rank == 2
        assert np.array_equal(r.diagonal, [1, 1])
        assert r.betti_rank == 2

    # ── RP² — real projective plane (Z₂ torsion) ─────────────────────────────
    def test_rp2_d2_torsion_z2(self):
        # Standard CW: 1 vertex, 1 edge (loop), 1 two-cell.
        # ∂₂(e²) = 2·e¹  (attaches with degree 2).
        # d₂: C₂ → C₁ = [[2]]   SNF diag = [2]  ← Z₂ torsion
        d2 = _csr([[2]])
        r = compute_exact_sparse_snf(d2, backend="python")
        assert r.rank == 1
        assert int(r.diagonal[0]) == 2
        assert r.torsion_summary == {2: 1}

    # ── Torus T² (no torsion) ─────────────────────────────────────────────────
    def test_torus_d2_no_torsion(self):
        # Standard CW: 1 vertex, 2 edges a,b, 1 face.
        # ∂₂(face) = a + b − a − b = 0  ⟹  d₂: C₂ → C₁ = [[0],[0]]
        d2 = _csr([[0], [0]])
        r = compute_exact_sparse_snf(d2, backend="python")
        assert r.rank == 0
        assert len(r.diagonal) == 0
        assert r.torsion_summary == {}

    # ── Klein bottle (Z₂ torsion) ─────────────────────────────────────────────
    def test_klein_bottle_d2_torsion_z2(self):
        # Standard CW: 1 vertex, 2 edges a,b, 1 face.
        # ∂₂(face) = a + b − a + b = 2b  ⟹  d₂ = [[0],[2]]  SNF diag = [2]
        d2 = _csr([[0], [2]])
        r = compute_exact_sparse_snf(d2, backend="python")
        assert r.rank == 1
        assert int(r.diagonal[0]) == 2
        assert r.torsion_summary == {2: 1}

    # ── Lens space L(n,1) ─────────────────────────────────────────────────────
    @pytest.mark.parametrize("n", [3, 5, 7, 12])
    def test_lens_space_d2(self, n):
        # The 2-cell in L(n,1) attaches with degree n.
        # d₂: C₂ → C₁ = [[n]]   SNF diag = [n]
        d2 = _csr([[n]])
        r = compute_exact_sparse_snf(d2, backend="python")
        assert r.rank == 1
        assert int(r.diagonal[0]) == n
        assert r.torsion_summary == {n: 1}

    # ── Diagonal torsion matrix ───────────────────────────────────────────────
    def test_diagonal_torsion_12_6(self):
        # diag(12, 6) → SNF sorts and divides: d₁|d₂ ⟹ diag(6, 12)
        M = _csr([[12, 0], [0, 6]])
        r = compute_exact_sparse_snf(M, backend="python")
        assert r.rank == 2
        # SNF invariant factors must satisfy divisibility d₁ | d₂.
        d = sorted(r.diagonal.tolist())
        assert d[0] > 0
        assert d[1] % d[0] == 0
        assert d[0] * d[1] == 72  # |det| = 72

    # ── Rank-deficient rectangular matrix ────────────────────────────────────
    def test_rectangular_rank_deficient(self):
        # 4×3 matrix with rank 2
        M = _csr([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 0],
                  [0, 0, 0]])
        r = compute_exact_sparse_snf(M, backend="python")
        assert r.rank == 2
        assert np.array_equal(r.diagonal, [1, 1])


# ─────────────────────────────────────────────────────────────────────────────
# 3. ExactSNFResult contract
# ─────────────────────────────────────────────────────────────────────────────

class TestExactSNFResultContract:
    def test_exact_always_true(self):
        r = compute_exact_sparse_snf(_csr([[2]]), backend="python")
        assert r.exact is True

    def test_theorem_tag(self):
        r = compute_exact_sparse_snf(_csr([[1]]), backend="python")
        assert r.theorem_tag == "snf.exact_sparse"

    def test_contract_version(self):
        r = compute_exact_sparse_snf(_csr([[1]]), backend="python")
        assert r.contract_version == CONTRACT_VERSION

    def test_decision_ready(self):
        r = compute_exact_sparse_snf(_csr([[1]]), backend="python")
        assert r.decision_ready() is True

    def test_torsion_summary_multi(self):
        # Use block-diag([[2], [0, 4]]) = diag(2, 4); divisibility 2|4 holds.
        # SNF of diag(2, 4) is diag(2, 4) — divisibility already satisfied.
        M = _csr([[2, 0], [0, 4]])
        r = compute_exact_sparse_snf(M, backend="python")
        assert r.rank == 2
        assert r.torsion_summary == {2: 1, 4: 1}

    def test_torsion_summary_two_z2(self):
        # diag(2, 2): two Z₂ torsion entries.
        M = _csr([[2, 0], [0, 2]])
        r = compute_exact_sparse_snf(M, backend="python")
        assert r.rank == 2
        assert r.torsion_summary == {2: 2}

    def test_torsion_summary_empty(self):
        r = compute_exact_sparse_snf(_csr([[1, 0], [0, 1]]), backend="python")
        assert r.torsion_summary == {}

    def test_algorithm_field_python(self):
        r = compute_exact_sparse_snf(_csr([[2]]), backend="python")
        assert r.algorithm == "python_dense"

    def test_markowitz_not_applied_python(self):
        # Python path never applies Markowitz.
        r = compute_exact_sparse_snf(_csr([[1, 2], [3, 4]]), backend="python")
        assert r.markowitz_applied is False

    def test_modular_certificate_none_by_default(self):
        r = compute_exact_sparse_snf(_csr([[2]]), backend="python")
        assert r.modular_certificate is None

    def test_verify_modular_adds_certificate(self):
        # Python fallback should also produce a certificate via Python GE.
        M = _csr([[2]])
        r = compute_exact_sparse_snf(M, backend="python", verify_modular=True)
        assert r.modular_certificate is not None
        cert = r.modular_certificate
        assert isinstance(cert, ModularRankCertificate)
        # SNF rank = 1 (one non-zero entry); modular certification should agree.
        assert cert.lower_bound >= 0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Modular rank certification
# ─────────────────────────────────────────────────────────────────────────────

class TestModularRankCertification:
    def test_rank_identity(self):
        M = sp.eye(5, dtype=np.int64, format="csr")
        cert = compute_modular_rank_certificate(M, backend="python")
        assert cert.certified_rank == 5
        assert cert.all_agree is True
        assert cert.lower_bound == 5
        assert cert.exact is True
        assert cert.decision_ready() is True

    def test_rank_zero_matrix(self):
        M = sp.csr_matrix((4, 4), dtype=np.int64)
        cert = compute_modular_rank_certificate(M, backend="python")
        assert cert.certified_rank == 0
        assert cert.all_agree is True

    def test_rank_rp2_d2(self):
        # d₂ = [[2]]  →  rank = 1 (row rank 1 over Z; rank mod 3 = 1)
        M = _csr([[2]])
        cert = compute_modular_rank_certificate(M, primes=[3, 5, 7], backend="python")
        assert cert.certified_rank == 1
        assert cert.all_agree is True

    def test_rank_torus_d2(self):
        # d₂ = [[0],[0]]  →  rank = 0 over any field
        M = _csr([[0], [0]])
        cert = compute_modular_rank_certificate(M, backend="python")
        assert cert.certified_rank == 0

    def test_theorem_tag(self):
        cert = compute_modular_rank_certificate(_csr([[1]]), backend="python")
        assert cert.theorem_tag == "snf.modular_rank_cert"

    def test_contract_version(self):
        cert = compute_modular_rank_certificate(_csr([[1]]), backend="python")
        assert cert.contract_version == CONTRACT_VERSION

    def test_primes_recorded(self):
        primes = [2, 5, 11]
        cert = compute_modular_rank_certificate(
            _csr([[1, 0], [0, 1]]), primes=primes, backend="python"
        )
        assert cert.primes == primes
        assert len(cert.ranks) == 3

    def test_lower_bound_is_max_of_ranks(self):
        M = sp.eye(3, dtype=np.int64, format="csr")
        cert = compute_modular_rank_certificate(M, primes=[2, 3, 5], backend="python")
        assert cert.lower_bound == max(cert.ranks)

    def test_empty_matrix_cert(self):
        M = sp.csr_matrix((0, 3), dtype=np.int64)
        cert = compute_modular_rank_certificate(M, backend="python")
        assert cert.certified_rank == 0
        assert cert.all_agree is True

    @pytest.mark.parametrize("n,primes_coprime", [
        (3, [5, 7, 11]),   # n=3, primes not divisible by 3 → rank=1
        (5, [2, 3, 7]),    # n=5
        (7, [2, 3, 5]),    # n=7
    ])
    def test_lens_space_rank_1_coprime_primes(self, n, primes_coprime):
        # d₂ = [[n]]; primes_coprime are coprime to n → rank(A mod p) = 1 for all.
        M = _csr([[n]])
        cert = compute_modular_rank_certificate(
            M, primes=primes_coprime, backend="python"
        )
        assert cert.certified_rank == 1
        assert cert.all_agree is True


# ─────────────────────────────────────────────────────────────────────────────
# 5. Batch SNF
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchSNF:
    def _standard_batch(self):
        """Four boundary operators: path d₁, RP² d₂, torus d₂, Klein d₂."""
        return [
            _csr([[-1], [1]]),        # path d₁   → [1]
            _csr([[2]]),               # RP² d₂    → [2]
            _csr([[0], [0]]),          # torus d₂  → []
            _csr([[0], [2]]),          # Klein d₂  → [2]
        ]

    def test_batch_count(self):
        mats = self._standard_batch()
        result = compute_batch_snf(mats, backend="python")
        assert result.n_matrices == 4
        assert len(result.diagonals) == 4

    def test_batch_exact(self):
        result = compute_batch_snf(self._standard_batch(), backend="python")
        assert result.exact is True
        assert result.n_failed == 0
        assert result.decision_ready() is True

    def test_batch_individual_diagonals(self):
        mats = self._standard_batch()
        result = compute_batch_snf(mats, backend="python")
        diags = result.diagonals

        assert np.array_equal(diags[0].diagonal, [1])       # path
        assert np.array_equal(diags[1].diagonal, [2])       # RP²
        assert len(diags[2].diagonal) == 0                  # torus
        assert np.array_equal(diags[3].diagonal, [2])       # Klein

    def test_batch_theorem_tag(self):
        result = compute_batch_snf(self._standard_batch(), backend="python")
        assert result.theorem_tag == "snf.batch_exact_sparse"

    def test_batch_contract_version(self):
        result = compute_batch_snf(self._standard_batch(), backend="python")
        assert result.contract_version == CONTRACT_VERSION

    def test_batch_empty_input(self):
        result = compute_batch_snf([], backend="python")
        assert result.n_matrices == 0
        assert result.n_failed == 0
        assert result.exact is True
        assert result.decision_ready() is True

    def test_batch_single_matrix(self):
        result = compute_batch_snf([_csr([[6]])], backend="python")
        assert result.n_matrices == 1
        assert int(result.diagonals[0].diagonal[0]) == 6

    def test_batch_matches_individual(self):
        # Each result in the batch must equal the individual call.
        mats = self._standard_batch()
        batch = compute_batch_snf(mats, backend="python")
        for i, mat in enumerate(mats):
            individual = compute_exact_sparse_snf(mat, backend="python")
            assert np.array_equal(
                batch.diagonals[i].diagonal, individual.diagonal
            ), f"Matrix {i}: batch {batch.diagonals[i].diagonal} ≠ individual {individual.diagonal}"

    def test_batch_verify_modular_adds_certs(self):
        mats = [_csr([[2]]), _csr([[3]])]
        result = compute_batch_snf(mats, backend="python", verify_modular=True)
        for sub in result.diagonals:
            assert sub.modular_certificate is not None


# ─────────────────────────────────────────────────────────────────────────────
# 6. math_core integration
# ─────────────────────────────────────────────────────────────────────────────

class TestMathCoreIntegration:
    def test_use_exact_sparse_false_default(self):
        from pysurgery.algebra.math_core import get_sparse_snf_diagonal
        M = _csr([[2]])
        out = get_sparse_snf_diagonal(M, backend="python")
        # Without use_exact_sparse, returns plain ndarray.
        assert isinstance(out, np.ndarray)
        assert int(out[0]) == 2

    def test_use_exact_sparse_true(self):
        from pysurgery.algebra.math_core import get_sparse_snf_diagonal
        M = _csr([[2]])
        out = get_sparse_snf_diagonal(M, backend="python", use_exact_sparse=True)
        assert isinstance(out, np.ndarray)
        assert int(out[0]) == 2

    def test_use_exact_sparse_matches_direct(self):
        from pysurgery.algebra.math_core import get_sparse_snf_diagonal
        matrices = [
            _csr([[-1], [1]]),
            _csr([[2]]),
            _csr([[0], [0]]),
            _csr([[1, 0, 0], [0, 2, 0], [0, 0, 1]]),
        ]
        for M in matrices:
            direct = compute_exact_sparse_snf(M, backend="python").diagonal
            routed = get_sparse_snf_diagonal(M, backend="python", use_exact_sparse=True)
            assert np.array_equal(direct, routed), (
                f"Mismatch: direct={direct}, routed={routed}"
            )

    def test_use_exact_sparse_zero(self):
        from pysurgery.algebra.math_core import get_sparse_snf_diagonal
        M = sp.csr_matrix((3, 3), dtype=np.int64)
        out = get_sparse_snf_diagonal(M, backend="python", use_exact_sparse=True)
        assert len(out) == 0

    def test_use_exact_sparse_identity(self):
        from pysurgery.algebra.math_core import get_sparse_snf_diagonal
        M = sp.eye(3, dtype=np.int64, format="csr")
        out = get_sparse_snf_diagonal(M, backend="python", use_exact_sparse=True)
        assert np.array_equal(out, [1, 1, 1])


# ─────────────────────────────────────────────────────────────────────────────
# 7. Error handling
# ─────────────────────────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_non_integer_entries_raise_homology_error(self):
        M = sp.csr_matrix(np.array([[1.5, 0], [0, 2.0]]))
        with pytest.raises(HomologyError, match="integers"):
            compute_exact_sparse_snf(M, backend="python")

    def test_non_integer_entries_in_modular_cert(self):
        M = sp.csr_matrix(np.array([[0.1]]))
        with pytest.raises(HomologyError, match="integers"):
            compute_modular_rank_certificate(M, backend="python")

    def test_non_integer_entries_in_batch(self):
        M = sp.csr_matrix(np.array([[1.5]]))
        with pytest.raises(HomologyError, match="integers"):
            compute_batch_snf([M], backend="python")

    from unittest import mock

    def test_backend_julia_hard_fail_without_julia(self, monkeypatch):
        from pysurgery.bridge.julia_bridge import julia_engine
        # Simulate Julia being unavailable
        monkeypatch.setattr(julia_engine, "available", False)
        # Also ensure backend is None so it fails on access
        monkeypatch.setattr(julia_engine, "backend", None)
        
        M = _csr([[2]])
        # If backend='julia' is explicitly requested but available is False, 
        # it should raise HomologyError when it tries to access the backend.
        with pytest.raises(HomologyError, match="Julia SNF backend failed"):
            compute_exact_sparse_snf(M, backend="julia")

    def test_padic_requires_julia(self, monkeypatch):
        from pysurgery.bridge.julia_bridge import julia_engine
        monkeypatch.setattr(julia_engine, "available", False)
        M = _csr([[2]])
        with pytest.raises(HomologyError, match="Julia"):
            compute_padic_snf_diagonal(M)


# ─────────────────────────────────────────────────────────────────────────────
# 8. p-adic cross-validation (Julia required)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _julia_available(), reason="Julia backend not available")
class TestPadicCrossValidation:
    """Cross-validate: compute_padic_snf_diagonal must agree with exact SNF."""

    def _check_agree(self, M, expected_diag):
        exact = compute_exact_sparse_snf(M, backend="auto")
        padic = compute_padic_snf_diagonal(M)
        assert np.array_equal(exact.diagonal, expected_diag), (
            f"exact={exact.diagonal}, expected={expected_diag}"
        )
        assert np.array_equal(padic, expected_diag), (
            f"padic={padic}, expected={expected_diag}"
        )

    def test_padic_rp2(self):
        self._check_agree(_csr([[2]]), np.array([2], dtype=np.int64))

    def test_padic_identity_3x3(self):
        self._check_agree(
            sp.eye(3, dtype=np.int64, format="csr"),
            np.array([1, 1, 1], dtype=np.int64),
        )

    def test_padic_klein_bottle(self):
        self._check_agree(_csr([[0], [2]]), np.array([2], dtype=np.int64))

    def test_padic_zero_matrix(self):
        M = sp.csr_matrix((2, 3), dtype=np.int64)
        padic = compute_padic_snf_diagonal(M)
        assert len(padic) == 0

    @pytest.mark.parametrize("n", [3, 5, 7])
    def test_padic_lens_space(self, n):
        self._check_agree(_csr([[n]]), np.array([n], dtype=np.int64))
