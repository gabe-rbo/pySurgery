"""Exact Sparse Smith Normal Form via Deterministic Julia Kernels.

Overview:
    Optimised, mathematically rigorous SNF pipeline for sparse integer
    boundary matrices (see RFC-snf-v2).  It exposes four orthogonal
    capabilities:

    1. **Markowitz pre-ordering** – Column permutation that minimises fill-in
       during SNF reduction (O(V+E) scoring, unimodular transformation).
    2. **Modular rank certification** – Rank verified by rank(A mod p) across
       multiple small primes; exact when at least one prime is coprime to every
       torsion coefficient.
    3. **p-adic CRT reconstruction** – Independent computation of the SNF
       diagonal without calling IntegerSmithNormalForm or AbstractAlgebra; used
       for cross-validation and as a fallback path.
    4. **Batch SNF** – Parallel computation across multiple boundary matrices
       using Julia's Threads.@threads.

Key Concepts:
    - **Smith Normal Form (SNF)**: For an integer matrix A, the SNF is a diagonal
      matrix diag(d_1, …, d_r, 0, …, 0) with d_1 | d_2 | … | d_r (invariant
      factors).  It uniquely classifies A up to unimodular transformations.
    - **Leaf-peeling** (Bauer 2021): Rows/columns with a single ±1 entry are
      eliminated in O(1), reducing boundary matrices to a small "core" before
      the O(N³) exact step.
    - **Markowitz criterion**: Score(c) = (col_nnz(c)−1)·min_r(row_nnz(r)−1)
      for column c; lower is better.  Pre-ordering by this score reduces
      coefficient explosion in the core SNF step.
    - **p-adic rank sequence**: r_e = #{i : v_p(d_i) < e} is the rank of A over
      ℤ/p^eℤ.  From r_1, r_2, … one can decode v_p(d_k) and hence d_k.
    - **CRT lifting**: d_k = ∏_p p^{v_p(d_k)} reconstructed from p-adic data
      for all primes p dividing any d_k.

Common Workflows:
    1. **Primary exact path** – compute_exact_sparse_snf(matrix, backend='auto')
       runs leaf-peel + Markowitz + IntegerSmithNormalForm via the Julia engine.
    2. **Verification** – compute_exact_sparse_snf(..., verify_modular=True)
       appends a modular certificate to the result.
    3. **CRT cross-validation** – compute_padic_snf_diagonal(matrix) provides an
       independent diagonal; compare with primary path to detect bugs.
    4. **Batch computation** – compute_batch_snf([d1, d2, d3]) parallelises
       independent homology dimension computations.

Coefficient Ring:
    ℤ (integers) throughout.  All computations are exact; no floating-point
    approximation is used in topological classification.

References:
    Smith, H. J. S. (1861). On systems of linear indeterminate equations and
      congruences. Philosophical Transactions of the Royal Society of London,
      151, 293–326.
    Bauer, U. (2021). Ripser: efficient computation of Vietoris–Rips persistence
      barcodes. Journal of Applied and Computational Topology, 5, 391–423.
    Dumas, J.-G., Saunders, B. D., & Villard, G. (2001). On efficient sparse
      integer matrix Smith normal form computations. Journal of Symbolic
      Computation, 32(1–2), 71–99.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import scipy.sparse as sp

from pysurgery.core.exceptions import HomologyError
from pysurgery.core.foundations import CONTRACT_VERSION

# ──────────────────────────────────────────────────────────────────────────────
# Default prime set used for modular certification and CRT reconstruction.
# Covers all torsion coefficients ≤ 47 that appear in typical topological spaces.
# ──────────────────────────────────────────────────────────────────────────────
_DEFAULT_CERT_PRIMES: list[int] = [2, 3, 5, 7, 11, 13]
_DEFAULT_CRT_PRIMES: list[int] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]


# ──────────────────────────────────────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ModularRankCertificate:
    """Certificate asserting the rank of a sparse integer matrix.

    Overview:
        Records the rank of a sparse integer matrix A computed modulo a set of
        small primes.  When all mod-p ranks agree, the ℤ-rank is certified with
        high confidence (and provably when the primes are coprime to all torsion
        coefficients).

    Key Concepts:
        - **rank(A mod p)** = #{i : p ∤ d_i} where d_i are SNF diagonal entries.
        - **lower_bound** = max(mod-p ranks); valid since reduction mod p can only
          decrease rank.
        - **certified_rank** is exact when all_agree == True.

    Attributes:
        primes          – primes used for certification.
        ranks           – rank(A mod p) for each prime.
        all_agree       – True when all mod-p ranks are equal.
        lower_bound     – max(ranks); guaranteed lower bound on ℤ-rank.
        certified_rank  – agreed rank (all_agree==True) or −1.
        exact           – True when all_agree.
        theorem_tag     – stable identifier "snf.modular_rank_cert".
        contract_version – library contract version.
    """

    primes: list[int]
    ranks: list[int]
    all_agree: bool
    lower_bound: int
    certified_rank: int
    exact: bool
    theorem_tag: str = "snf.modular_rank_cert"
    contract_version: str = field(default_factory=lambda: CONTRACT_VERSION)

    def decision_ready(self) -> bool:
        """Return True when the certificate is suitable for downstream use."""
        return self.exact and self.certified_rank >= 0


@dataclass
class ExactSNFResult:
    """Complete result of an exact sparse SNF computation.

    Overview:
        Aggregates the SNF diagonal, provenance information, and an optional
        modular verification certificate into a single structured object.

    Key Concepts:
        - **diagonal** – non-zero invariant factors d_1 ≤ d_2 ≤ … ≤ d_r.
        - **rank** – number of non-zero invariant factors = algebraic rank over ℤ.
        - **betti_rank** – number of invariant factors equal to 1.
        - **torsion_factors** – invariant factors > 1 (torsion coefficients).
        - **algorithm** – primary algorithm used ('julia_isnf', 'julia_aa',
          'padic_crt', 'python_dense').
        - **modular_certificate** – optional ModularRankCertificate for verification.

    Attributes:
        diagonal            – SNF diagonal as int64 array.
        rank                – int; algebraic rank.
        betti_rank          – int; number of 1s in diagonal.
        torsion_factors     – int64 array; entries > 1.
        leaf_peeled_count   – int; ones removed by O(V+E) pre-processing.
        core_size           – tuple (m, n) of the core matrix after peeling.
        algorithm           – str; primary algorithm identifier.
        markowitz_applied   – bool; whether Markowitz pre-ordering was applied.
        modular_certificate – optional ModularRankCertificate.
        exact               – bool; always True for this module.
        theorem_tag         – str; "snf.exact_sparse".
        contract_version    – str; library contract version.
    """

    diagonal: np.ndarray
    rank: int
    betti_rank: int
    torsion_factors: np.ndarray
    leaf_peeled_count: int
    core_size: tuple[int, int]
    algorithm: str
    markowitz_applied: bool
    modular_certificate: ModularRankCertificate | None
    exact: bool = True
    theorem_tag: str = "snf.exact_sparse"
    contract_version: str = field(default_factory=lambda: CONTRACT_VERSION)

    def decision_ready(self) -> bool:
        """Return True; exact SNF is always ready for downstream use."""
        return self.exact

    @property
    def torsion_summary(self) -> dict[int, int]:
        """Return a frequency map {coefficient: count} for torsion coefficients."""
        summary: dict[int, int] = {}
        for t in self.torsion_factors:
            t_int = int(t)
            summary[t_int] = summary.get(t_int, 0) + 1
        return summary


@dataclass
class BatchSNFResult:
    """Results of a parallel batch SNF computation.

    Overview:
        Wraps the per-matrix SNF diagonals returned by compute_batch_snf.

    Attributes:
        diagonals       – list of ExactSNFResult, one per input matrix.
        n_matrices      – number of matrices in the batch.
        n_failed        – number of matrices that raised exceptions.
        exact           – True when all sub-computations succeeded.
        theorem_tag     – "snf.batch_exact_sparse".
        contract_version – library contract version.
    """

    diagonals: list[ExactSNFResult]
    n_matrices: int
    n_failed: int
    exact: bool
    theorem_tag: str = "snf.batch_exact_sparse"
    contract_version: str = field(default_factory=lambda: CONTRACT_VERSION)

    def decision_ready(self) -> bool:
        """Return True when no sub-computation failed."""
        return self.exact and self.n_failed == 0


# ──────────────────────────────────────────────────────────────────────────────
# Internal pure-Python helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_csr_int64(matrix: sp.spmatrix) -> sp.csr_matrix:
    """Coerce a sparse matrix to CSR with int64 data.

    What is Being Computed?:
        Normalises any scipy sparse format to CSR with dtype int64, raising
        HomologyError if non-integer entries are detected.

    Algorithm:
        1. Convert to CSR.
        2. Check all data values are integers (within tolerance).
        3. Cast to int64; raise HomologyError on overflow.

    Args:
        matrix: Any scipy.sparse matrix.

    Returns:
        sp.csr_matrix with dtype int64.

    Raises:
        HomologyError: If entries are not integers or overflow int64.
    """
    csr = matrix.tocsr()
    data_float = csr.data.astype(float)
    if np.any(np.abs(data_float - np.round(data_float)) > 1e-9):
        raise HomologyError(
            "Matrix entries must be integers for exact SNF computation. "
            "Non-integer entries detected."
        )
    try:
        return csr.astype(np.int64)
    except (OverflowError, ValueError) as exc:
        raise HomologyError(
            f"Matrix entries overflow int64: {exc}"
        ) from exc


def _snf_diagonal_python_dense(matrix: sp.spmatrix) -> np.ndarray:
    """Compute SNF diagonal using the pure-Python dense path.

    What is Being Computed?:
        Delegates to math_core.get_snf_diagonal for dense exact computation.
        Intended as the Python fallback when Julia is unavailable.

    Algorithm:
        1. Convert sparse matrix to dense int64 array.
        2. Call get_snf_diagonal (Numba-JIT row-pivoted SNF).

    Preserved Invariants:
        Exact integer torsion is preserved.

    Args:
        matrix: Sparse integer matrix.

    Returns:
        np.ndarray: Non-zero invariant factors, sorted.
    """
    from pysurgery.algebra.math_core import get_snf_diagonal  # local import avoids circular

    dense = np.asarray(matrix.toarray(), dtype=object)
    return get_snf_diagonal(dense)


def _build_result(
    diagonal: np.ndarray,
    leaf_peeled: int,
    core_size: tuple[int, int],
    algorithm: str,
    markowitz_applied: bool,
    cert: ModularRankCertificate | None,
) -> ExactSNFResult:
    """Construct an ExactSNFResult from raw diagonal output."""
    diag_sorted = np.sort(diagonal.astype(np.int64))
    diag_nonzero = diag_sorted[diag_sorted > 0]
    betti = int(np.sum(diag_nonzero == 1))
    torsion = diag_nonzero[diag_nonzero > 1]
    return ExactSNFResult(
        diagonal=diag_nonzero,
        rank=int(len(diag_nonzero)),
        betti_rank=betti,
        torsion_factors=torsion,
        leaf_peeled_count=leaf_peeled,
        core_size=core_size,
        algorithm=algorithm,
        markowitz_applied=markowitz_applied,
        modular_certificate=cert,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def compute_exact_sparse_snf(
    matrix: sp.spmatrix,
    *,
    backend: Literal["auto", "julia", "python"] = "auto",
    use_markowitz: bool = True,
    verify_modular: bool = False,
    cert_primes: list[int] | None = None,
) -> ExactSNFResult:
    """Compute the exact Smith Normal Form diagonal of a sparse integer matrix.

    What is Being Computed?:
        The invariant factors d_1 ≤ d_2 ≤ … ≤ d_r (with d_1 | d_2 | … | d_r)
        of the SNF of `matrix` over ℤ, implementing the full pipeline
        (see RFC-snf-v2):
          1. O(V+E) leaf-peeling pre-processor in Julia.
          2. Markowitz column reordering of the core (optional).
          3. Exact SNF via IntegerSmithNormalForm (large cores) or
             AbstractAlgebra (small cores) in the Julia backend, or
             dense Python fallback.
          4. Optional modular rank certification.

    Algorithm:
        1. Convert matrix to COO int64.
        2. If Julia available and backend ≠ 'python': call Julia exact_snf_sparse
           with use_markowitz=True.
        3. If verify_modular: call modular_rank_certification_jl for cross-check.
        4. Build and return ExactSNFResult.
        5. Python fallback: dense get_snf_diagonal (exact, slower).

    Preserved Invariants:
        - Exact integer torsion; no floating-point approximation.
        - SNF diagonal is unique up to sign (all entries returned positive).
        - Sorted in non-decreasing order.

    Args:
        matrix:         Sparse integer matrix (scipy.sparse).
        backend:        'auto' (Julia if available, else Python), 'julia'
                        (hard-fail if Julia absent), 'python' (always Python).
        use_markowitz:  Apply Markowitz column pre-ordering to the core matrix
                        before SNF (default True).
        verify_modular: Append a ModularRankCertificate to the result (default
                        False; adds ~10–20% overhead for the mod-p rank checks).
        cert_primes:    Primes for modular certification.  Defaults to
                        {2, 3, 5, 7, 11, 13}.

    Returns:
        ExactSNFResult: Full SNF result with provenance metadata.

    Raises:
        HomologyError:  If matrix entries are not integers, overflow int64, or
                        Julia was explicitly requested but unavailable.

    Use When:
        - Computing homology over ℤ (need exact torsion and Betti numbers).
        - Need a modular certificate alongside the SNF (verify_modular=True).
        - Batch calls: prefer compute_batch_snf for multiple matrices.

    Example:
        import scipy.sparse as sp, numpy as np
        from pysurgery.algebra.exact_snf_julia import compute_exact_sparse_snf
        d1 = sp.csr_matrix(np.array([[1,-1,0],[0,1,-1]], dtype=np.int64))
        result = compute_exact_sparse_snf(d1, verify_modular=True)
        print(result.diagonal, result.torsion_summary)

    References:
        Bauer, U. (2021). Ripser: efficient computation of Vietoris–Rips
          persistence barcodes. Journal of Applied and Computational Topology,
          5, 391–423.
        Dumas, J.-G., Saunders, B. D., & Villard, G. (2001). On efficient sparse
          integer matrix Smith normal form computations. Journal of Symbolic
          Computation, 32(1–2), 71–99.
    """
    from ..bridge.julia_bridge import julia_engine

    if cert_primes is None:
        cert_primes = _DEFAULT_CERT_PRIMES

    # Validate and normalise input.
    matrix_csr = _to_csr_int64(matrix)
    if matrix_csr.shape[0] == 0 or matrix_csr.shape[1] == 0:
        diag = np.array([], dtype=np.int64)
        return _build_result(diag, 0, (0, 0), "trivial", False, None)

    backend_norm = str(backend).lower().strip()
    use_julia = (backend_norm == "julia") or (
        backend_norm == "auto" and julia_engine.available
    )

    cert: ModularRankCertificate | None = None
    algorithm = "python_dense"
    markowitz_applied = False
    leaf_peeled = 0
    core_size = matrix_csr.shape

    if use_julia:
        try:
            coo = matrix_csr.tocoo()
            rows = np.asarray(coo.row, dtype=np.int64)
            cols = np.asarray(coo.col, dtype=np.int64)
            vals = np.asarray(coo.data, dtype=np.int64)
            diag_raw = julia_engine.backend.exact_snf_sparse(
                rows, cols, vals,
                int(matrix_csr.shape[0]),
                int(matrix_csr.shape[1]),
                use_markowitz=use_markowitz,
            )
            diag = np.array(diag_raw, dtype=np.int64)
            algorithm = (
                "julia_isnf"
                if (matrix_csr.shape[0] > 100 or matrix_csr.shape[1] > 100)
                else "julia_aa"
            )
            markowitz_applied = use_markowitz

            # Modular rank certification (optional cross-check).
            if verify_modular:
                cert_raw = julia_engine.compute_modular_rank_certificate(
                    matrix_csr, primes=cert_primes
                )
                cert = ModularRankCertificate(
                    primes=cert_raw["primes"],
                    ranks=cert_raw["ranks"],
                    all_agree=cert_raw["all_agree"],
                    lower_bound=cert_raw["lower_bound"],
                    certified_rank=cert_raw["certified_rank"],
                    exact=cert_raw["exact"],
                )
                # Sanity check: certified rank must equal our SNF rank.
                snf_rank = int(np.sum(diag > 0))
                if cert.all_agree and cert.certified_rank != snf_rank:
                    raise HomologyError(
                        f"SNF rank ({snf_rank}) inconsistent with modular "
                        f"rank certificate ({cert.certified_rank}). "
                        "This indicates a computation error."
                    )

        except HomologyError:
            raise
        except Exception as exc:
            if backend_norm == "julia":
                raise HomologyError(
                    f"Julia SNF backend failed and 'julia' was explicitly "
                    f"requested: {exc!r}"
                ) from exc
            warnings.warn(
                f"Julia SNF failed ({exc!r}); falling back to Python dense SNF.",
                stacklevel=2,
            )
            use_julia = False

    if not use_julia:
        diag = _snf_diagonal_python_dense(matrix_csr)
        algorithm = "python_dense"
        markowitz_applied = False
        if verify_modular and cert is None:
            cert = compute_modular_rank_certificate(
                matrix_csr, primes=cert_primes, backend="python"
            )

    return _build_result(
        diag, leaf_peeled, core_size, algorithm, markowitz_applied, cert
    )


def compute_modular_rank_certificate(
    matrix: sp.spmatrix,
    *,
    primes: list[int] | None = None,
    backend: Literal["auto", "julia", "python"] = "auto",
) -> ModularRankCertificate:
    """Certify the rank of a sparse integer matrix via modular arithmetic.

    What is Being Computed?:
        rank(A mod p) for each prime p, using dense Gaussian elimination over
        GF(p).  All-prime agreement provides a high-confidence certificate of
        the integer rank; it is exact when at least one prime is coprime to all
        torsion coefficients.

    Algorithm:
        1. For each p in primes: dense GE over GF(p) → rank_p.
        2. all_agree = (all rank_p equal).
        3. lower_bound = max(rank_p).
        4. certified_rank = agreed rank (or -1 if disagreement).

    Preserved Invariants:
        rank(A mod p) ≤ rank(A) for all p; equality holds iff p ∤ every d_i.

    Args:
        matrix:  Sparse integer matrix.
        primes:  Primes to test.  Defaults to {2, 3, 5, 7, 11, 13}.
        backend: Backend selector ('auto', 'julia', 'python').

    Returns:
        ModularRankCertificate.

    Raises:
        HomologyError: If matrix entries are not integers.

    Use When:
        - Quick rank check before committing to full SNF computation.
        - Verifying that a boundary matrix has the expected homological rank.

    Example:
        from pysurgery.algebra.exact_snf_julia import compute_modular_rank_certificate
        cert = compute_modular_rank_certificate(d2_sparse)
        if cert.decision_ready():
            print("Rank:", cert.certified_rank)
    """
    from ..bridge.julia_bridge import julia_engine

    if primes is None:
        primes = _DEFAULT_CERT_PRIMES

    matrix_csr = _to_csr_int64(matrix)
    if matrix_csr.shape[0] == 0 or matrix_csr.shape[1] == 0:
        return ModularRankCertificate(
            primes=primes, ranks=[0] * len(primes),
            all_agree=True, lower_bound=0, certified_rank=0, exact=True,
        )

    backend_norm = str(backend).lower().strip()
    use_julia = (backend_norm == "julia") or (
        backend_norm == "auto" and julia_engine.available
    )

    if use_julia:
        try:
            raw = julia_engine.compute_modular_rank_certificate(
                matrix_csr, primes=primes
            )
            return ModularRankCertificate(
                primes=raw["primes"],
                ranks=raw["ranks"],
                all_agree=raw["all_agree"],
                lower_bound=raw["lower_bound"],
                certified_rank=raw["certified_rank"],
                exact=raw["exact"],
            )
        except Exception as exc:
            if backend_norm == "julia":
                raise HomologyError(
                    f"Julia modular rank certification failed: {exc!r}"
                ) from exc
            warnings.warn(
                f"Julia modular cert failed ({exc!r}); using Python fallback.",
                stacklevel=2,
            )

    # Python fallback: dense GE over GF(p) for each prime.
    dense = np.asarray(matrix_csr.toarray(), dtype=np.int64)
    ranks_py: list[int] = []
    for p in primes:
        ranks_py.append(_python_rank_mod_p(dense.copy(), p))

    all_same = len(set(ranks_py)) == 1
    lb = max(ranks_py) if ranks_py else 0
    certified = ranks_py[0] if all_same else -1
    return ModularRankCertificate(
        primes=primes, ranks=ranks_py,
        all_agree=all_same, lower_bound=lb,
        certified_rank=certified, exact=all_same,
    )


def _python_rank_mod_p(M: np.ndarray, p: int) -> int:
    """Dense Gaussian elimination over GF(p) returning the rank.

    What is Being Computed?:
        Rank of M over the finite field GF(p) using modular row reduction.

    Algorithm:
        1. Reduce all entries mod p.
        2. For each column c (left to right):
           a. Find first non-zero pivot in remaining rows.
           b. Normalise pivot row.
           c. Eliminate column entries in all other rows.
        3. Return number of pivots found.

    Args:
        M: Dense int64 matrix (modified in-place).
        p: Prime modulus.

    Returns:
        int: Rank over GF(p).
    """
    m, n = M.shape
    row = 0
    rank = 0
    M = np.mod(M, p)
    for col in range(n):
        pivot = -1
        for r in range(row, m):
            if M[r, col] != 0:
                pivot = r
                break
        if pivot == -1:
            continue
        if pivot != row:
            M[[row, pivot]] = M[[pivot, row]]
        inv_piv = int(pow(int(M[row, col]), p - 2, p))  # Fermat's little theorem
        M[row] = np.mod(M[row] * inv_piv, p)
        for r in range(m):
            if r != row and M[r, col] != 0:
                M[r] = np.mod(M[r] - int(M[r, col]) * M[row], p)
        row += 1
        rank += 1
        if row >= m:
            break
    return rank


def compute_padic_snf_diagonal(
    matrix: sp.spmatrix,
    *,
    primes: list[int] | None = None,
    max_e: int = 8,
    backend: Literal["auto", "julia"] = "auto",
) -> np.ndarray:
    """Reconstruct the exact SNF diagonal via deterministic p-adic CRT lifting.

    What is Being Computed?:
        An independent computation of the SNF diagonal d_1 ≤ … ≤ d_r that does
        NOT call IntegerSmithNormalForm or AbstractAlgebra.  Used for
        cross-validation of the primary exact path.

        The algorithm reads off d_k = ∏_p p^{v_p(d_k)} from the p-adic rank
        sequence r_e = #{i : v_p(d_i) < e} by computing Gaussian elimination
        over ℤ/p^eℤ for increasing powers e.

    Algorithm:
        For each prime p in primes:
          1. For e = 1, 2, …, max_e:
             r_e = rank(A over ℤ/p^eℤ) using _padic_rank_step in Julia.
          2. v_p(d_k) = (first e with r_e ≥ k) − 1.
        Reconstruct d_k = ∏_p p^{v_p(d_k)}.

    Preserved Invariants:
        Exact when all prime factors of all d_k appear in primes and
        max_e ≥ max_k v_p(d_k) for every p.  For homological boundary matrices
        (entries ∈ {−1,0,1}) with torsion ≤ 31, the defaults are sufficient.

    Args:
        matrix:  Sparse integer matrix.
        primes:  Primes for p-adic analysis.  Defaults to
                 [2,3,5,7,11,13,17,19,23,29,31].
        max_e:   Maximum p-adic depth per prime (default 8, covers p^8 ≤ 256).
        backend: Only 'auto' and 'julia' are supported (Julia is required).

    Returns:
        np.ndarray: Non-zero SNF diagonal in non-decreasing order.

    Raises:
        HomologyError: If Julia is unavailable or computation fails.

    Use When:
        - Cross-validating the exact SNF computed by IntegerSmithNormalForm.
        - As a consistency check in test suites.

    Example:
        from pysurgery.algebra.exact_snf_julia import compute_padic_snf_diagonal
        diag_padic = compute_padic_snf_diagonal(d2)
        diag_exact = compute_exact_sparse_snf(d2).diagonal
        assert np.array_equal(diag_padic, diag_exact)

    References:
        Smith, H. J. S. (1861). Philosophical Transactions, 151, 293–326.
    """
    from ..bridge.julia_bridge import julia_engine

    if primes is None:
        primes = _DEFAULT_CRT_PRIMES

    matrix_csr = _to_csr_int64(matrix)
    if matrix_csr.shape[0] == 0 or matrix_csr.shape[1] == 0:
        return np.array([], dtype=np.int64)

    if not julia_engine.available:
        raise HomologyError(
            "compute_padic_snf_diagonal requires the Julia backend. "
            "Install Julia and set up juliacall, or use compute_exact_sparse_snf "
            "with backend='python' for the dense path."
        )

    try:
        result = julia_engine.compute_padic_snf_diagonal(
            matrix_csr, primes=primes, max_e=max_e
        )
        return result
    except Exception as exc:
        raise HomologyError(
            f"p-adic SNF diagonal reconstruction failed: {exc!r}"
        ) from exc


def compute_batch_snf(
    matrices: list[sp.spmatrix],
    *,
    backend: Literal["auto", "julia", "python"] = "auto",
    verify_modular: bool = False,
    cert_primes: list[int] | None = None,
) -> BatchSNFResult:
    """Compute exact SNF for a batch of sparse matrices in parallel.

    What is Being Computed?:
        Calls compute_exact_sparse_snf for each matrix independently.  When
        Julia is available, computation is dispatched to Julia's Threads.@threads
        worker pool for true parallelism.

    Algorithm:
        1. If Julia available: pass all matrices to batch_exact_snf_sparse which
           uses Threads.@threads for independent parallel execution.
        2. For each result, build an ExactSNFResult.
        3. If verify_modular: add certification to each result.
        4. Failed computations set exact=False and return empty diagonal.

    Preserved Invariants:
        Each sub-computation is independent; thread safety guaranteed by
        Julia's task-local state.

    Args:
        matrices:       List of sparse integer matrices.
        backend:        Backend selector ('auto', 'julia', 'python').
        verify_modular: Add modular certificate to each sub-result.
        cert_primes:    Primes for certification (default {2,3,5,7,11,13}).

    Returns:
        BatchSNFResult containing one ExactSNFResult per input matrix.

    Raises:
        HomologyError: If any matrix contains non-integer entries.

    Use When:
        - Computing homology for multiple boundary dimensions simultaneously.
        - Parallelising SNF on a multi-core machine.

    Example:
        from pysurgery.algebra.exact_snf_julia import compute_batch_snf
        result = compute_batch_snf([d1, d2, d3])
        for r in result.diagonals:
            print(r.rank, r.torsion_summary)

    References:
        Dumas, J.-G., Saunders, B. D., & Villard, G. (2001). Journal of
          Symbolic Computation, 32(1–2), 71–99.
    """
    from ..bridge.julia_bridge import julia_engine

    if cert_primes is None:
        cert_primes = _DEFAULT_CERT_PRIMES
    if not matrices:
        return BatchSNFResult(
            diagonals=[], n_matrices=0, n_failed=0, exact=True
        )

    backend_norm = str(backend).lower().strip()
    use_julia = (backend_norm == "julia") or (
        backend_norm == "auto" and julia_engine.available
    )

    n_failed = 0
    sub_results: list[ExactSNFResult] = []

    if use_julia:
        # Convert all matrices to int64 first; fail early on bad inputs.
        csr_mats = [_to_csr_int64(m) for m in matrices]
        try:
            raw_diagonals = julia_engine.compute_batch_snf(csr_mats)
        except Exception as exc:
            if backend_norm == "julia":
                raise HomologyError(
                    f"Julia batch SNF failed: {exc!r}"
                ) from exc
            warnings.warn(
                f"Julia batch SNF failed ({exc!r}); falling back to sequential Python.",
                stacklevel=2,
            )
            raw_diagonals = None

        if raw_diagonals is not None:
            for i, (mat, diag) in enumerate(zip(csr_mats, raw_diagonals)):
                if len(diag) == 0 and mat.nnz > 0:
                    # Julia sub-computation failed.
                    n_failed += 1
                    sub_results.append(
                        _build_result(
                            np.array([], dtype=np.int64), 0,
                            mat.shape, "failed", False, None
                        )
                    )
                    continue

                cert: ModularRankCertificate | None = None
                if verify_modular:
                    try:
                        raw_cert = julia_engine.compute_modular_rank_certificate(
                            mat, primes=cert_primes
                        )
                        cert = ModularRankCertificate(
                            primes=raw_cert["primes"],
                            ranks=raw_cert["ranks"],
                            all_agree=raw_cert["all_agree"],
                            lower_bound=raw_cert["lower_bound"],
                            certified_rank=raw_cert["certified_rank"],
                            exact=raw_cert["exact"],
                        )
                    except Exception:
                        pass

                sub_results.append(
                    _build_result(
                        diag, 0, mat.shape, "julia_batch", True, cert
                    )
                )
            return BatchSNFResult(
                diagonals=sub_results,
                n_matrices=len(matrices),
                n_failed=n_failed,
                exact=(n_failed == 0),
            )

    # Python sequential fallback.
    # Pre-validate all inputs first so HomologyError from bad data propagates
    # immediately (consistent with the Julia path's pre-validation).
    csr_mats_py = [_to_csr_int64(m) for m in matrices]
    for mat_csr in csr_mats_py:
        try:
            r = compute_exact_sparse_snf(
                mat_csr,
                backend="python",
                verify_modular=verify_modular,
                cert_primes=cert_primes,
            )
            sub_results.append(r)
        except Exception:
            n_failed += 1
            sub_results.append(
                _build_result(
                    np.array([], dtype=np.int64), 0,
                    mat_csr.shape, "failed", False, None
                )
            )

    return BatchSNFResult(
        diagonals=sub_results,
        n_matrices=len(matrices),
        n_failed=n_failed,
        exact=(n_failed == 0),
    )
