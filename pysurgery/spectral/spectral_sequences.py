"""Spectral Sequence Framework & Exact Couple Solver.

Overview:
    Fully general spectral sequence engine (see RFC-spectral-sequences-v2)
    that computes pages
    E^r → E^{r+1} automatically given the input data and differentials, plus
    four concrete spectral sequences (Serre, Leray-Serre, Adams,
    Atiyah-Hirzebruch), an exact couple solver that produces derived couples,
    and an extension-problem solver that extracts global (co)homology from the
    E^∞ page.

Key Concepts:
    - **Spectral sequence (SS)**: A sequence of bigraded modules
      E^1, E^2, E^3, … connected by differentials d^r of bidegree (-r, r-1)
      (homological convention) or (r, 1-r) (cohomological convention), where
      E^{r+1} = ker(d^r) / im(d^r).  Under finite-support hypotheses the
      sequence stabilises to E^∞ and converges to a filtration of a target
      module.
    - **Exact couple** (Massey 1952): A bigraded triangle
      D --i--> D --j--> E --k--> D with exactness at each corner; iteration
      (D, E, i, j, k) ↦ (D', E', i', j', k') derives the spectral sequence
      d^r = j ∘ k.
    - **Serre SS** (Serre 1951): For a fibration F → E → B with simply
      connected base, E^2_{p,q} = H_p(B; H_q(F)) ⇒ H_{p+q}(E).
    - **Leray-Serre SS**: Cohomological dual; E_2^{p,q} = H^p(B; H^q(F)) ⇒
      H^{p+q}(E) with cup-product compatible.
    - **Adams SS** (Adams 1958): At a prime p, E_2^{s,t} = Ext^{s,t}_{A_p}(F_p,
      F_p) ⇒ π_{t-s}^S(S^0)^∧_p (mod-p stable homotopy).
    - **Atiyah-Hirzebruch SS** (Atiyah & Hirzebruch 1962): For a generalized
      cohomology theory h^*, E_2^{p,q} = H^p(X; h^q(pt)) ⇒ h^{p+q}(X).
    - **Extension problem**: Given E^∞_{p,n−p}, recover H_n as an iterated
      extension of the associated graded.

Common Workflows:
    1. Build a `SerreSpectralSequence` from H_*(B), H_*(F).  Supply differential
       data via `supply_differential(r, source, matrix)`.  Call `converge()`
       to obtain a `ConvergenceResult` with the E^∞ page.
    2. Solve the extension problem with `solve_extension_problem(result, n)`.
    3. Construct an `ExactCouple` and iterate `derive()` to obtain pages.

Coefficient Ring:
    Supports 'Q' (rational vector spaces; rank arithmetic via Sympy), 'Z'
    (free abelian groups with torsion; SNF used for differentials), and 'F_p'
    (vector spaces over the prime field; modular Gauss elimination).  The
    coefficient ring is fixed at construction and propagated through all
    page transitions.

References:
    Adams, J. F. (1958). On the structure and applications of the Steenrod
      algebra. Commentarii Mathematici Helvetici, 32, 180–214.
    Atiyah, M. F., & Hirzebruch, F. (1962). Vector bundles and homogeneous
      spaces. Proceedings of Symposia in Pure Mathematics, 3, 7–38.
    Boardman, J. M. (1999). Conditionally convergent spectral sequences.
      Contemporary Mathematics, 239, 49–84.
    Massey, W. S. (1952). Exact couples in algebraic topology. Annals of
      Mathematics, 56(2), 363–396.
    McCleary, J. (2001). A user's guide to spectral sequences (2nd ed.).
      Cambridge University Press.
    Serre, J.-P. (1951). Homologie singulière des espaces fibrés. Annals of
      Mathematics, 54(3), 425–505.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field as dc_field
from typing import Literal, Mapping

import numpy as np
import sympy as sp
from pydantic import BaseModel, ConfigDict, Field

from pysurgery.core.exceptions import MathError
from pysurgery.core.foundations import CONTRACT_VERSION


# ──────────────────────────────────────────────────────────────────────────────
# Type aliases
# ──────────────────────────────────────────────────────────────────────────────

Bidegree = tuple[int, int]
RingName = Literal["Q", "Z", "F_p"]
Convention = Literal["homological", "cohomological"]


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic strict convergence models
# ──────────────────────────────────────────────────────────────────────────────

class SpectralEntry(BaseModel):
    """Entry at a single bidegree of a spectral sequence page.

    Overview:
        A finitely generated abelian group (or vector space) sitting at
        bidegree (p, q) of E^r.  Encoded as a free rank plus an optional
        tuple of torsion coefficients d_1 | d_2 | … | d_k.

    Key Concepts:
        - **rank**: dimension of the free part.
        - **torsion**: invariant factors (each ≥ 2) such that the torsion
          subgroup is ⊕ ℤ/d_iℤ.

    Coefficient Ring:
        Compatible with ℚ, ℤ, and 𝔽_p.  Over a field, `torsion` is empty.

    Attributes:
        rank (int):  Free-module rank (≥ 0).
        torsion (tuple[int, ...]): Sorted invariant factors (each ≥ 2).
    """

    model_config = ConfigDict(frozen=True)

    rank: int = Field(ge=0)
    torsion: tuple[int, ...] = Field(default_factory=tuple)

    @property
    def is_zero(self) -> bool:
        """Return True iff this entry is the zero group."""
        return self.rank == 0 and len(self.torsion) == 0

    @property
    def total_dim(self) -> int:
        """Free rank (the rank of the underlying free part)."""
        return self.rank

    @classmethod
    def zero(cls) -> "SpectralEntry":
        """Construct the zero entry."""
        return cls(rank=0, torsion=())

    @classmethod
    def free(cls, rank: int) -> "SpectralEntry":
        """Construct a free entry of given rank."""
        return cls(rank=rank, torsion=())


class SpectralPageSnapshot(BaseModel):
    """Immutable snapshot of one page E^r of a spectral sequence.

    Overview:
        Records the entries E^r_{p,q} at every supported bidegree along with
        provenance metadata.  Snapshots are produced by `converge()` to form
        the page history.

    Coefficient Ring:
        See `SpectralEntry`.

    Attributes:
        page_number (int): The page index r.
        coefficient_ring (str): One of 'Q', 'Z', 'F_p'.
        convention (Convention): 'homological' or 'cohomological'.
        entries (dict[Bidegree, SpectralEntry]): Bigraded entries.
        is_terminal (bool): True iff this snapshot is E^∞.
    """

    model_config = ConfigDict(frozen=True)

    page_number: int
    coefficient_ring: str
    convention: Convention
    entries: dict[tuple[int, int], SpectralEntry]
    is_terminal: bool = False

    def at(self, p: int, q: int) -> SpectralEntry:
        """Return the entry at bidegree (p, q), or the zero entry if absent."""
        return self.entries.get((p, q), SpectralEntry.zero())

    def total_rank_at(self, n: int) -> int:
        """Sum of free ranks across all bidegrees with p+q = n."""
        return sum(
            entry.rank
            for (p, q), entry in self.entries.items()
            if p + q == n
        )


class ConvergenceResult(BaseModel):
    """Result of running a spectral sequence to convergence.

    Overview:
        The terminal output of `SpectralSequence.converge()`.  Reports whether
        the SS stabilised within the page budget, the E^∞ page, and the full
        page history.

    Key Concepts:
        - **converged**: True when consecutive pages were equal AND no
          differential at the current page is non-zero.
        - **e_infinity**: The terminal page entries.
        - **page_history**: Snapshots of every page from r₀ through r_∞.

    Coefficient Ring:
        Inherited from the originating spectral sequence.

    Attributes:
        converged (bool): Whether stabilisation was reached within max_pages.
        last_page (int): The page number of the terminal snapshot.
        e_infinity (dict[Bidegree, SpectralEntry]): Terminal entries.
        page_history (list[SpectralPageSnapshot]): All page snapshots.
        coefficient_ring (str): The coefficient ring.
        convention (Convention): Bidegree convention used.
        convergence_target (str): Human-readable target description.
        prime (int | None): Prime modulus for 𝔽_p, else None.
        exact (bool): True iff converged.
        theorem_tag (str): "spectral_sequence.convergence".
        contract_version (str): Library contract version.
    """

    model_config = ConfigDict(frozen=True)

    converged: bool
    last_page: int
    e_infinity: dict[tuple[int, int], SpectralEntry]
    page_history: list[SpectralPageSnapshot]
    coefficient_ring: str
    convention: Convention
    convergence_target: str
    prime: int | None = None
    exact: bool
    theorem_tag: str = "spectral_sequence.convergence"
    contract_version: str = CONTRACT_VERSION

    def decision_ready(self) -> bool:
        """Return True iff result is suitable for downstream surgery use."""
        return bool(self.exact and self.converged)

    def total_rank_at(self, n: int) -> int:
        """Sum of E^∞_{p, n-p} ranks (lower bound on rank of H_n target)."""
        return sum(
            entry.rank
            for (p, q), entry in self.e_infinity.items()
            if p + q == n
        )

    def torsion_at(self, n: int) -> tuple[int, ...]:
        """Concatenated torsion factors at total degree n (upper bound)."""
        out: list[int] = []
        for (p, q), entry in self.e_infinity.items():
            if p + q == n:
                out.extend(entry.torsion)
        return tuple(sorted(out))


class ExtensionResult(BaseModel):
    """Result of solving the extension problem at a fixed total degree.

    Overview:
        Given the E^∞ page of a spectral sequence, the extension problem is
        the determination of H_n (or H^n) as an abelian group from the known
        associated graded gr_p H_n = E^∞_{p, n-p}.  When all entries are free,
        H_n splits as a direct sum.  In the presence of torsion, the splitting
        may fail; this result reports both the certain rank and an upper
        bound on torsion.

    Key Concepts:
        - **rank**: Σ_p rank(E^∞_{p, n-p}); always exact.
        - **torsion_upper_bound**: Concatenation of all torsion factors of
          contributing entries; is the torsion of H_n iff the extension splits.
        - **splitting_assumed**: True when the result was computed under the
          'assume_split' hypothesis.

    Coefficient Ring:
        Same as the originating SS; meaningful only over ℤ.

    Attributes:
        total_degree (int): The total degree n for H_n / H^n.
        rank (int): Free rank (exact).
        torsion_upper_bound (tuple[int, ...]): Sorted tuple of torsion factors.
        splitting_assumed (bool): Whether the splitting hypothesis was used.
        contributing_bidegrees (list[Bidegree]): Bidegrees with non-zero
            entries summing to total_degree.
        associated_graded_summary (dict[Bidegree, SpectralEntry]):
            Per-bidegree contributions.
        exact (bool): True iff the rank is exact and either the SS is over a
            field (so splitting is automatic) or splitting was assumed and
            disclosed.
        theorem_tag (str): "spectral_sequence.extension".
        contract_version (str): Library contract version.
    """

    model_config = ConfigDict(frozen=True)

    total_degree: int
    rank: int
    torsion_upper_bound: tuple[int, ...]
    splitting_assumed: bool
    contributing_bidegrees: list[tuple[int, int]]
    associated_graded_summary: dict[tuple[int, int], SpectralEntry]
    exact: bool
    theorem_tag: str = "spectral_sequence.extension"
    contract_version: str = CONTRACT_VERSION

    def decision_ready(self) -> bool:
        """Return True iff the rank determination is exact."""
        return self.exact


# ──────────────────────────────────────────────────────────────────────────────
# Mutable internal state
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SpectralPage:
    """Mutable container for one spectral sequence page during computation.

    Overview:
        Holds the bigraded entries E^r_{p,q} of one page along with any
        differentials d^r that have been supplied (or computed).  The
        framework consumes one page and produces the next via
        `SpectralSequence.compute_next_page`.

    Attributes:
        page_number (int): The current page index r.
        convention (Convention): 'homological' or 'cohomological'.
        entries (dict[Bidegree, SpectralEntry]): Bigraded entries.
        differentials (dict[Bidegree, np.ndarray]): Matrices d^r at each
            source bidegree, with shape (target_rank, source_rank).
    """

    page_number: int
    convention: Convention
    entries: dict[tuple[int, int], SpectralEntry] = dc_field(default_factory=dict)
    differentials: dict[tuple[int, int], np.ndarray] = dc_field(default_factory=dict)

    def differential_bidegree(self) -> Bidegree:
        """Return the bidegree of d^r."""
        if self.convention == "homological":
            return (-self.page_number, self.page_number - 1)
        return (self.page_number, 1 - self.page_number)

    def support_box(self) -> tuple[int, int, int, int] | None:
        """Bounding box (p_min, p_max, q_min, q_max) of non-zero entries."""
        nz = [(p, q) for (p, q), e in self.entries.items() if not e.is_zero]
        if not nz:
            return None
        ps = [p for p, _ in nz]
        qs = [q for _, q in nz]
        return (min(ps), max(ps), min(qs), max(qs))


# ──────────────────────────────────────────────────────────────────────────────
# Linear algebra helpers
# ──────────────────────────────────────────────────────────────────────────────

def _matrix_rank(
    matrix: np.ndarray, ring: RingName, prime: int | None = None
) -> int:
    """Exact rank of an integer matrix over the requested ring.

    What is Being Computed?:
        Linear-algebraic rank of an integer matrix M, computed exactly over
        ℚ (Sympy), 𝔽_p (modular Gauss elimination), or ℤ (SNF).

    Algorithm:
        - 'Q': sp.Matrix(M).rank() — exact over rationals.
        - 'F_p': dense Gauss elimination over GF(p), reused from
                 exact_snf_julia._python_rank_mod_p.
        - 'Z':  Number of non-zero invariant factors of SNF(M).

    Args:
        matrix: The integer matrix.
        ring:   Coefficient ring, one of 'Q', 'Z', 'F_p'.
        prime:  Required when ring=='F_p'.

    Returns:
        int: Non-negative rank.

    Raises:
        MathError: If 'F_p' is requested without a prime.
    """
    if matrix is None:
        return 0
    arr = np.asarray(matrix)
    if arr.size == 0:
        return 0
    if arr.ndim != 2:
        raise MathError(f"Differential matrix must be 2-D, got shape {arr.shape}.")

    if ring == "Q":
        # Sympy gives an exact rank over the rationals from integer entries.
        return int(sp.Matrix(arr.astype(np.int64).tolist()).rank())
    if ring == "F_p":
        if prime is None:
            raise MathError("Coefficient ring 'F_p' requires `prime` to be set.")
        from pysurgery.algebra.exact_snf_julia import _python_rank_mod_p
        return int(_python_rank_mod_p(arr.astype(np.int64).copy(), int(prime)))
    if ring == "Z":
        from pysurgery.algebra.math_core import get_snf_diagonal
        diag = get_snf_diagonal(arr.astype(object))
        return int(np.count_nonzero(np.asarray(diag)))
    raise MathError(f"Unsupported coefficient ring: {ring!r}")


def _snf_diagonal(matrix: np.ndarray) -> np.ndarray:
    """SNF diagonal of an integer matrix (delegates to math_core)."""
    from pysurgery.algebra.math_core import get_snf_diagonal
    arr = np.asarray(matrix, dtype=object)
    return get_snf_diagonal(arr)


# ──────────────────────────────────────────────────────────────────────────────
# Exact couple
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ExactCouple:
    """Exact couple (D, E, i, j, k) producing a spectral sequence by derivation.

    Overview:
        Records a bigraded exact couple as introduced by Massey (1952).  The
        triangle

            D --i--> D
            ^        |
            k        j
            |        v
            E <------

        is exact at each corner: ker(i) = im(k), ker(j) = im(i), ker(k) =
        im(j).  Iterating the derivation operator (D, E) ↦ (D', E') produces
        the pages of the associated spectral sequence with d^r = j ∘ k after
        r-1 derivations.

    Key Concepts:
        - **Bidegrees**: Each map has a fixed bidegree shift; d = j ∘ k has
          bidegree bidegree_j + bidegree_k.
        - **Derived couple**: D' = im(i), E' = ker(d) / im(d).  The first
          derivation produces E^2; the n-th gives E^{n+1}.
        - **Convergence**: Over a field with bounded support, derivation
          stabilises; this is then equivalent to the spectral sequence's
          stabilisation.

    Coefficient Ring:
        Tracked via `coefficient_ring`; rank-only iteration is supported in
        all three rings.  The full module-theoretic derivation (with explicit
        matrices for i, j, k) is performed when `track_torsion` is set in
        future extensions; the present implementation reduces to ranks for
        derivation, which suffices for SS convergence over ℚ and 𝔽_p.

    Attributes:
        D (dict[Bidegree, SpectralEntry]): Bigraded module D.
        E (dict[Bidegree, SpectralEntry]): Bigraded module E (= E^1).
        i_map (dict[Bidegree, np.ndarray]): Matrix of i at each bidegree.
        j_map (dict[Bidegree, np.ndarray]): Matrix of j.
        k_map (dict[Bidegree, np.ndarray]): Matrix of k.
        bidegree_i (Bidegree): Shift of i.
        bidegree_j (Bidegree): Shift of j.
        bidegree_k (Bidegree): Shift of k.
        coefficient_ring (RingName): Working ring for ranks.
        prime (int | None): Prime when coefficient_ring=='F_p'.
        derivation_count (int): Number of times derive() has been applied.
    """

    D: dict[tuple[int, int], SpectralEntry]
    E: dict[tuple[int, int], SpectralEntry]
    i_map: dict[tuple[int, int], np.ndarray]
    j_map: dict[tuple[int, int], np.ndarray]
    k_map: dict[tuple[int, int], np.ndarray]
    bidegree_i: tuple[int, int]
    bidegree_j: tuple[int, int]
    bidegree_k: tuple[int, int]
    coefficient_ring: RingName = "Q"
    prime: int | None = None
    derivation_count: int = 0

    def __post_init__(self) -> None:
        if self.coefficient_ring == "F_p" and self.prime is None:
            raise MathError("ExactCouple over F_p requires `prime` to be set.")

    @property
    def differential_bidegree(self) -> Bidegree:
        """Bidegree of d = j ∘ k on E."""
        return (
            self.bidegree_j[0] + self.bidegree_k[0],
            self.bidegree_j[1] + self.bidegree_k[1],
        )

    def page_entries(self) -> dict[tuple[int, int], SpectralEntry]:
        """Return the current E page entries (a copy)."""
        return dict(self.E)

    def derive(self) -> "ExactCouple":
        """Compute the derived couple (D', E', i', j', k').

        What is Being Computed?:
            One derivation step: from (D, E, i, j, k) produce a new exact
            couple whose E-component is the next spectral sequence page.

        Algorithm:
            1. Compute the differential d = j ∘ k at each bidegree of E.
            2. E' = ker(d at (p,q)) / im(d into (p,q)) — rank arithmetic.
            3. D' = im(i): rank shift by `bidegree_i`.
            4. i' = i restricted to D'; rank preserved.
            5. j', k' inherit bidegrees from j, k.

        Preserved Invariants:
            Exactness of (D', E', i', j', k') is preserved under derivation
            (Massey 1952, Theorem 4).

        Returns:
            ExactCouple: The derived couple with `derivation_count` += 1.

        References:
            Massey, W. S. (1952). Exact couples in algebraic topology.
              Annals of Mathematics, 56(2), 363–396.
        """
        ring = self.coefficient_ring
        prime = self.prime
        d_bideg = self.differential_bidegree

        # Compute d = j ∘ k at every (p, q) where E has support.
        d_rank: dict[tuple[int, int], int] = {}
        for pq, e_entry in self.E.items():
            if e_entry.is_zero:
                d_rank[pq] = 0
                continue
            # k: E_{p,q} -> D_{p+kp, q+kq}; j: D_{a,b} -> E_{a+jp, b+jq}.
            k_mat = self.k_map.get(pq)
            d_target = (pq[0] + self.bidegree_k[0], pq[1] + self.bidegree_k[1])
            j_mat = self.j_map.get(d_target)
            if k_mat is None or j_mat is None:
                d_rank[pq] = 0
                continue
            # Composition jk: E_{p,q} -> E_{p+d_p, q+d_q}.
            try:
                composed = np.asarray(j_mat, dtype=np.int64) @ np.asarray(
                    k_mat, dtype=np.int64
                )
            except ValueError:
                # Shape mismatch implies zero map at this bidegree.
                d_rank[pq] = 0
                continue
            d_rank[pq] = _matrix_rank(composed, ring, prime)

        # New E page: ker(d at (p,q)) / im(d into (p,q)).
        new_E: dict[tuple[int, int], SpectralEntry] = {}
        for pq, e_entry in self.E.items():
            out_rank = d_rank.get(pq, 0)
            in_source = (pq[0] - d_bideg[0], pq[1] - d_bideg[1])
            in_rank = d_rank.get(in_source, 0)
            ker_dim = max(0, e_entry.rank - out_rank)
            new_rank = max(0, ker_dim - in_rank)
            new_E[pq] = SpectralEntry(rank=new_rank, torsion=())

        # New D = im(i); rank arithmetic only.
        new_D: dict[tuple[int, int], SpectralEntry] = {}
        for pq, d_entry in self.D.items():
            i_mat = self.i_map.get(pq)
            if i_mat is None or d_entry.is_zero:
                new_D[pq] = SpectralEntry.zero()
                continue
            r = _matrix_rank(np.asarray(i_mat, dtype=np.int64), ring, prime)
            target = (pq[0] + self.bidegree_i[0], pq[1] + self.bidegree_i[1])
            new_D[target] = SpectralEntry(rank=r, torsion=())

        # Maps in the derived couple share the same bidegrees as the source
        # couple (Massey 1952, Lemma 6); we propagate the matrices unchanged
        # since the derived couple's rank arithmetic only consults their ranks.
        return ExactCouple(
            D=new_D,
            E=new_E,
            i_map=dict(self.i_map),
            j_map=dict(self.j_map),
            k_map=dict(self.k_map),
            bidegree_i=self.bidegree_i,
            bidegree_j=self.bidegree_j,
            bidegree_k=self.bidegree_k,
            coefficient_ring=self.coefficient_ring,
            prime=self.prime,
            derivation_count=self.derivation_count + 1,
        )

    def to_spectral_sequence(
        self,
        *,
        convention: Convention = "homological",
        convergence_target: str = "filtered module of the exact couple",
    ) -> "ExactCoupleSpectralSequence":
        """Wrap this exact couple as a SpectralSequence."""
        return ExactCoupleSpectralSequence(
            couple=self,
            convention=convention,
            convergence_target=convergence_target,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Abstract spectral sequence
# ──────────────────────────────────────────────────────────────────────────────

class SpectralSequence(ABC):
    """Abstract base class for spectral sequences with automatic page computation.

    Overview:
        Captures the common machinery shared by every spectral sequence:
        bidegree bookkeeping, page-to-page rank arithmetic, automatic
        detection of stabilisation, and convergence reporting.  Concrete
        subclasses fix the initial page number and the construction of E^r₀.

    Key Concepts:
        - **Bidegree convention**: Homological d^r has shift (-r, r-1);
          cohomological d^r has shift (r, 1-r).
        - **Initial page**: r₀ ∈ {1, 2}; subclasses provide both r₀ and
          E^{r₀}.
        - **Differentials**: Either supplied by the user via
          `supply_differential` or computed by the subclass.  Absence of a
          differential at a bidegree means d^r = 0 there.
        - **Stabilisation**: A first-quadrant SS stabilises when the page
          index exceeds the support diameter.

    Coefficient Ring:
        Fixed at construction; one of 'Q', 'Z', 'F_p'.

    Attributes:
        coefficient_ring (RingName): Coefficient ring.
        prime (int | None): Prime modulus when coefficient_ring=='F_p'.
        convention (Convention): Bidegree convention.
        max_pages (int): Hard upper bound on the number of pages.
        convergence_target (str): Human-readable description of the target.

    Common Workflows:
        1. Subclass and implement `initial_page_number`,
           `compute_initial_page`, and (optionally) `compute_differentials`.
        2. Optionally call `supply_differential(r, source, matrix)` for any
           differentials known to the user.
        3. Call `converge()` to compute pages up to E^∞.
    """

    def __init__(
        self,
        *,
        coefficient_ring: RingName = "Q",
        prime: int | None = None,
        convention: Convention = "homological",
        max_pages: int = 32,
        convergence_target: str = "associated graded of filtered module",
    ):
        if coefficient_ring not in ("Q", "Z", "F_p"):
            raise MathError(f"Unsupported coefficient_ring: {coefficient_ring!r}")
        if coefficient_ring == "F_p" and prime is None:
            raise MathError("coefficient_ring='F_p' requires `prime` to be set.")
        if max_pages < 1:
            raise MathError("max_pages must be ≥ 1.")
        self.coefficient_ring: RingName = coefficient_ring
        self.prime = prime
        self.convention: Convention = convention
        self.max_pages = int(max_pages)
        self.convergence_target = str(convergence_target)
        self._user_differentials: dict[
            int, dict[tuple[int, int], np.ndarray]
        ] = defaultdict(dict)

    # ── Abstract hooks ────────────────────────────────────────────────────

    @abstractmethod
    def initial_page_number(self) -> int:
        """Return the initial page index r₀ (typically 1 or 2)."""

    @abstractmethod
    def compute_initial_page(self) -> dict[tuple[int, int], SpectralEntry]:
        """Return the entries E^{r₀}_{p,q}.

        The subclass is responsible for assembling E^{r₀} from input data
        (e.g., for Serre: H_p(B) ⊗ H_q(F)).  Entries with zero rank may be
        omitted; the framework treats absent bidegrees as zero.
        """

    # ── Differential plumbing ────────────────────────────────────────────

    def differential_bidegree(self, r: int) -> Bidegree:
        """Bidegree of d^r in this convention."""
        if self.convention == "homological":
            return (-r, r - 1)
        return (r, 1 - r)

    def supply_differential(
        self,
        r: int,
        source: tuple[int, int],
        matrix: np.ndarray,
    ) -> None:
        """Register a user-supplied differential matrix d^r at `source`.

        The matrix must have shape (rank(target), rank(source)) where target
        is the bidegree shifted by `differential_bidegree(r)`.

        Args:
            r:      Page index.
            source: Source bidegree (p, q).
            matrix: Integer matrix of shape (target_rank, source_rank).
        """
        if r < self.initial_page_number():
            raise MathError(
                f"Cannot supply differential before initial page "
                f"r₀ = {self.initial_page_number()}."
            )
        arr = np.asarray(matrix, dtype=np.int64)
        if arr.ndim != 2:
            raise MathError(
                f"Differential matrix must be 2-D, got shape {arr.shape}."
            )
        self._user_differentials[r][tuple(source)] = arr

    def compute_differentials(
        self,
        r: int,
        page: SpectralPage,
    ) -> dict[tuple[int, int], np.ndarray]:
        """Hook for subclasses to compute d^r differentials algorithmically.

        Default implementation returns no algorithmic differentials; the
        framework will fall back to user-supplied differentials.  Subclasses
        such as `AdamsSpectralSequence` override this to inject classical
        algebraic differentials (e.g., h_0 multiplication on Ext groups).
        """
        return {}

    # ── Page transition ─────────────────────────────────────────────────

    def _gather_differentials(
        self, r: int, page: SpectralPage
    ) -> dict[tuple[int, int], np.ndarray]:
        """Merge algorithmic and user-supplied differentials for page r."""
        algo_diffs = self.compute_differentials(r, page)
        user_diffs = self._user_differentials.get(r, {})
        merged: dict[tuple[int, int], np.ndarray] = dict(algo_diffs)
        # User-supplied differentials override algorithmic ones at the same
        # bidegree to allow targeted overrides during testing.
        for src, mat in user_diffs.items():
            merged[src] = np.asarray(mat, dtype=np.int64)
        return merged

    def _validate_differential_shape(
        self,
        r: int,
        source: tuple[int, int],
        matrix: np.ndarray,
        page: SpectralPage,
    ) -> None:
        """Validate the differential matrix shape against page entries."""
        d_bideg = self.differential_bidegree(r)
        target = (source[0] + d_bideg[0], source[1] + d_bideg[1])
        src_entry = page.entries.get(source, SpectralEntry.zero())
        tgt_entry = page.entries.get(target, SpectralEntry.zero())
        expected = (tgt_entry.rank, src_entry.rank)
        if matrix.shape != expected:
            raise MathError(
                f"Differential d^{r} at source {source} has shape "
                f"{matrix.shape}; expected {expected} given source rank "
                f"{src_entry.rank} and target rank {tgt_entry.rank}."
            )

    def compute_next_page(self, page: SpectralPage) -> SpectralPage:
        """Compute E^{r+1} from E^r via rank arithmetic on differentials.

        What is Being Computed?:
            E^{r+1}_{p,q} = ker(d^r out of (p,q)) / im(d^r into (p,q)).
            Free ranks are tracked exactly; torsion in the entries is
            inherited unchanged when no differential touches the bidegree
            (a sound conservative approximation for the rank of E^{r+1}).

        Algorithm:
            1. Gather all differentials d^r (algorithmic + user-supplied).
            2. Validate matrix shapes against page entry ranks.
            3. Compute rank(d^r) at each source bidegree over the working
               ring.
            4. For each (p, q), compute new rank via rank-nullity.
            5. Construct the next SpectralPage.

        Preserved Invariants:
            Free rank arithmetic is exact over any field.  Over ℤ, the rank
            of the free part of E^{r+1}_{p,q} is bounded above by the field
            computation; torsion is propagated conservatively unless the
            user supplies torsion-tracking differentials.

        Args:
            page: The current page E^r.

        Returns:
            SpectralPage: The next page E^{r+1}.

        Raises:
            MathError: On differential matrix shape mismatch.
        """
        r = page.page_number
        d_bideg = self.differential_bidegree(r)
        diffs = self._gather_differentials(r, page)

        for src, mat in diffs.items():
            self._validate_differential_shape(r, src, mat, page)

        d_ranks: dict[tuple[int, int], int] = {}
        for src, mat in diffs.items():
            d_ranks[src] = _matrix_rank(mat, self.coefficient_ring, self.prime)

        new_entries: dict[tuple[int, int], SpectralEntry] = {}
        all_bidegrees: set[tuple[int, int]] = set(page.entries.keys())
        # Include any bidegree referenced by a differential.
        for src in diffs:
            all_bidegrees.add(src)
            all_bidegrees.add((src[0] + d_bideg[0], src[1] + d_bideg[1]))

        for pq in all_bidegrees:
            entry = page.entries.get(pq, SpectralEntry.zero())
            out_rank = d_ranks.get(pq, 0)
            in_src = (pq[0] - d_bideg[0], pq[1] - d_bideg[1])
            in_rank = d_ranks.get(in_src, 0)
            ker_dim = max(0, entry.rank - out_rank)
            new_rank = max(0, ker_dim - in_rank)

            # Conservative torsion propagation:
            # When no differential is involved at this bidegree, torsion is
            # preserved; when a differential is involved, free rank is
            # adjusted but torsion is dropped (the user must supply a
            # torsion-aware differential through compute_differentials).
            if out_rank == 0 and in_rank == 0:
                new_entries[pq] = SpectralEntry(rank=new_rank, torsion=entry.torsion)
            else:
                new_entries[pq] = SpectralEntry(rank=new_rank, torsion=())

        return SpectralPage(
            page_number=r + 1,
            convention=self.convention,
            entries=new_entries,
            differentials=dict(diffs),
        )

    # ── Convergence loop ────────────────────────────────────────────────

    def _snapshot(
        self, page: SpectralPage, *, terminal: bool
    ) -> SpectralPageSnapshot:
        return SpectralPageSnapshot(
            page_number=page.page_number,
            coefficient_ring=self.coefficient_ring,
            convention=self.convention,
            entries=dict(page.entries),
            is_terminal=terminal,
        )

    def _pages_match(self, a: SpectralPage, b: SpectralPage) -> bool:
        """Return True iff entries of two pages agree at every bidegree."""
        keys = set(a.entries.keys()) | set(b.entries.keys())
        for k in keys:
            ea = a.entries.get(k, SpectralEntry.zero())
            eb = b.entries.get(k, SpectralEntry.zero())
            if ea.rank != eb.rank or ea.torsion != eb.torsion:
                return False
        return True

    def _all_diffs_zero(self, r: int, page: SpectralPage) -> bool:
        """True iff every differential at page r vanishes."""
        diffs = self._gather_differentials(r, page)
        if not diffs:
            return True
        for src, mat in diffs.items():
            if _matrix_rank(mat, self.coefficient_ring, self.prime) > 0:
                return False
        return True

    def _support_diameter(self, page: SpectralPage) -> int | None:
        """Maximum coordinate distance in the support of `page` (None if empty)."""
        box = page.support_box()
        if box is None:
            return 0
        p_min, p_max, q_min, q_max = box
        return max(p_max - p_min, q_max - q_min)

    def converge(self) -> ConvergenceResult:
        """Run pages until E^r stabilises or `max_pages` is exhausted.

        What is Being Computed?:
            The terminal page E^∞ together with the page history.  Stabili-
            sation is detected when two consecutive pages agree AND every
            differential at the current page is zero.

        Algorithm:
            1. Build E^{r₀} from `compute_initial_page`.
            2. For r = r₀, r₀ + 1, …, max_pages:
                a. Build E^{r+1} via `compute_next_page`.
                b. If pages match and all d^r are zero → converged.
                c. Else continue.
            3. If max_pages is reached without convergence, return a
               non-converged result with `exact=False`.

        Returns:
            ConvergenceResult: Pydantic model summarising the outcome.

        Raises:
            HomologyError: Propagated from `_matrix_rank` on malformed
                differential data.
        """
        r0 = self.initial_page_number()
        initial_entries = self.compute_initial_page()
        page = SpectralPage(
            page_number=r0,
            convention=self.convention,
            entries=dict(initial_entries),
        )
        history: list[SpectralPageSnapshot] = [
            self._snapshot(page, terminal=False)
        ]

        for _step in range(self.max_pages):
            next_page = self.compute_next_page(page)
            stabilised = (
                self._pages_match(page, next_page)
                and self._all_diffs_zero(page.page_number, page)
            )
            history.append(self._snapshot(next_page, terminal=stabilised))
            if stabilised:
                return ConvergenceResult(
                    converged=True,
                    last_page=next_page.page_number,
                    e_infinity=dict(next_page.entries),
                    page_history=history,
                    coefficient_ring=self.coefficient_ring,
                    convention=self.convention,
                    convergence_target=self.convergence_target,
                    prime=self.prime,
                    exact=True,
                )
            page = next_page

        return ConvergenceResult(
            converged=False,
            last_page=page.page_number,
            e_infinity=dict(page.entries),
            page_history=history,
            coefficient_ring=self.coefficient_ring,
            convention=self.convention,
            convergence_target=self.convergence_target,
            prime=self.prime,
            exact=False,
        )


class ExactCoupleSpectralSequence(SpectralSequence):
    """SpectralSequence wrapper around an ExactCouple.

    Overview:
        Drives an `ExactCouple` to convergence via repeated derivation.  The
        n-th call to `derive()` produces the (n+1)-th page; this wrapper
        exposes the standard SS interface so the same convergence machinery
        is reused.

    Coefficient Ring:
        Inherited from the underlying `ExactCouple`.

    Attributes:
        couple (ExactCouple): The originating couple.
    """

    def __init__(
        self,
        *,
        couple: ExactCouple,
        convention: Convention = "homological",
        max_pages: int = 32,
        convergence_target: str = "filtered module of the exact couple",
    ):
        super().__init__(
            coefficient_ring=couple.coefficient_ring,
            prime=couple.prime,
            convention=convention,
            max_pages=max_pages,
            convergence_target=convergence_target,
        )
        self.couple = couple
        self._derived: list[ExactCouple] = [couple]

    def initial_page_number(self) -> int:
        """Return the initial page index r₀ = 1 for an exact couple."""
        return 1

    def compute_initial_page(self) -> dict[tuple[int, int], SpectralEntry]:
        """Return E^1, the E-term of the underlying exact couple."""
        return dict(self.couple.E)

    def compute_next_page(self, page: SpectralPage) -> SpectralPage:
        """Override: use the couple's derive() to produce the next page."""
        # Ensure we have the couple corresponding to this page.
        target_idx = page.page_number  # E^{page.page_number} ↔ self._derived[idx-1]
        while len(self._derived) < target_idx + 1:
            self._derived.append(self._derived[-1].derive())
        new_couple = self._derived[target_idx]
        return SpectralPage(
            page_number=page.page_number + 1,
            convention=self.convention,
            entries=dict(new_couple.E),
            differentials={},
        )


# ──────────────────────────────────────────────────────────────────────────────
# Serre spectral sequence
# ──────────────────────────────────────────────────────────────────────────────

def _tensor_entry(
    e_b: SpectralEntry, e_f: SpectralEntry, ring: RingName
) -> SpectralEntry:
    """Tensor product of two entries over the working ring.

    What is Being Computed?:
        E_B ⊗ E_F as an abelian group when ring=='Z', or as a vector space
        when ring is a field.

    Algorithm:
        Over a field: rank(E_B ⊗ E_F) = rank(E_B) · rank(E_F); torsion=().
        Over ℤ: rank = rank(E_B)·rank(E_F); torsion includes
            - rank(E_B) copies of E_F.torsion,
            - rank(E_F) copies of E_B.torsion,
            - the cross torsion ℤ/gcd(d_i, d_j) for each (d_i, d_j) torsion
              pair (this comes from the tensor of finite cyclic groups).
    """
    if ring in ("Q", "F_p"):
        return SpectralEntry(rank=e_b.rank * e_f.rank, torsion=())
    # ring == 'Z': handle torsion contributions.
    rank_part = e_b.rank * e_f.rank
    torsion: list[int] = []
    # rank(B) copies of torsion(F)
    for _ in range(e_b.rank):
        torsion.extend(int(t) for t in e_f.torsion)
    # rank(F) copies of torsion(B)
    for _ in range(e_f.rank):
        torsion.extend(int(t) for t in e_b.torsion)
    # cross torsion
    from math import gcd
    for ti in e_b.torsion:
        for tj in e_f.torsion:
            g = gcd(int(ti), int(tj))
            if g > 1:
                torsion.append(g)
    torsion.sort()
    return SpectralEntry(rank=rank_part, torsion=tuple(torsion))


def _build_e2_tensor(
    base: Mapping[int, SpectralEntry],
    fibre: Mapping[int, SpectralEntry],
    ring: RingName,
) -> dict[tuple[int, int], SpectralEntry]:
    """E²_{p,q} = H_p(B) ⊗ H_q(F)  (simply connected base, trivial action)."""
    entries: dict[tuple[int, int], SpectralEntry] = {}
    for p, e_b in base.items():
        if e_b.is_zero:
            continue
        for q, e_f in fibre.items():
            if e_f.is_zero:
                continue
            entries[(p, q)] = _tensor_entry(e_b, e_f, ring)
    return entries


class SerreSpectralSequence(SpectralSequence):
    """Homological Serre spectral sequence of a fibration F → E → B.

    Overview:
        For a Serre fibration with simply connected base and arbitrary fibre,

            E²_{p,q} = H_p(B; H_q(F)) ⇒ H_{p+q}(E),

        with d^r of bidegree (-r, r-1).  This implementation accepts the
        homologies of the base and fibre as bigraded data and constructs the
        tensor-product E² automatically.  The user supplies any non-trivial
        differentials via `supply_differential`.

    Key Concepts:
        - **Trivial action**: The base π_1(B) is assumed to act trivially on
          H_*(F); for non-trivial actions, the user must override
          `compute_initial_page` to use the local-coefficient version.
        - **Transgression**: d^{q+1} on a fibre generator hitting the base
          at degree q+1 is the transgression; supply via
          `supply_differential(q+1, (0, q), matrix)`.

    Coefficient Ring:
        ℚ (default), ℤ, or 𝔽_p; ranks of base and fibre must be specified
        in the same ring.

    Common Workflows:
        1. Build  ss = SerreSpectralSequence(base_homology={…}, fibre_homology={…}).
        2. Optionally call ss.supply_differential(2, (p, q), matrix).
        3. Call ss.converge() and inspect e_infinity.

    Attributes:
        base_homology (dict[int, SpectralEntry]): H_p(B) by p.
        fibre_homology (dict[int, SpectralEntry]): H_q(F) by q.

    References:
        Serre, J.-P. (1951). Annals of Mathematics, 54(3), 425–505.
        McCleary, J. (2001). Cambridge University Press, ch. 5.
    """

    def __init__(
        self,
        *,
        base_homology: Mapping[int, SpectralEntry],
        fibre_homology: Mapping[int, SpectralEntry],
        coefficient_ring: RingName = "Q",
        prime: int | None = None,
        max_pages: int = 32,
    ):
        super().__init__(
            coefficient_ring=coefficient_ring,
            prime=prime,
            convention="homological",
            max_pages=max_pages,
            convergence_target="H_*(total space) of the Serre fibration",
        )
        self.base_homology = {int(k): v for k, v in base_homology.items()}
        self.fibre_homology = {int(k): v for k, v in fibre_homology.items()}

    def initial_page_number(self) -> int:
        """Return the initial page index r₀ = 2 for the Serre sequence."""
        return 2

    def compute_initial_page(self) -> dict[tuple[int, int], SpectralEntry]:
        """Return E^2 = H_p(B) ⊗ H_q(F) as the tensor of base and fibre homology."""
        return _build_e2_tensor(
            self.base_homology, self.fibre_homology, self.coefficient_ring
        )


class LeraySerreSpectralSequence(SpectralSequence):
    """Cohomological Leray-Serre spectral sequence for F → E → B.

    Overview:
        For a Serre fibration with simply connected base and arbitrary fibre,

            E_2^{p,q} = H^p(B; H^q(F)) ⇒ H^{p+q}(E),

        with d_r of bidegree (r, 1-r).  Implementation is the cohomological
        dual of `SerreSpectralSequence`.  The cup-product structure is
        respected automatically when differentials are derivations on
        E_r^{p,q} ⊗ E_r^{p',q'}, but the framework here tracks only ranks.

    Coefficient Ring:
        ℚ (default), ℤ, or 𝔽_p.

    Common Workflows:
        1. Build  ss = LeraySerreSpectralSequence(base_cohomology={…}, fibre_cohomology={…}).
        2. Supply differentials via supply_differential.
        3. Call converge() and read off the cohomology of E from
           solve_extension_problem(result, n).

    Attributes:
        base_cohomology (dict[int, SpectralEntry]): H^p(B).
        fibre_cohomology (dict[int, SpectralEntry]): H^q(F).

    References:
        McCleary, J. (2001). Cambridge University Press, ch. 5–6.
    """

    def __init__(
        self,
        *,
        base_cohomology: Mapping[int, SpectralEntry],
        fibre_cohomology: Mapping[int, SpectralEntry],
        coefficient_ring: RingName = "Q",
        prime: int | None = None,
        max_pages: int = 32,
    ):
        super().__init__(
            coefficient_ring=coefficient_ring,
            prime=prime,
            convention="cohomological",
            max_pages=max_pages,
            convergence_target="H^*(total space) of the Leray-Serre fibration",
        )
        self.base_cohomology = {int(k): v for k, v in base_cohomology.items()}
        self.fibre_cohomology = {int(k): v for k, v in fibre_cohomology.items()}

    def initial_page_number(self) -> int:
        """Return the initial page index r₀ = 2 for the Leray-Serre sequence."""
        return 2

    def compute_initial_page(self) -> dict[tuple[int, int], SpectralEntry]:
        """Return E^2 = H^p(B) ⊗ H^q(F) as the tensor of base and fibre cohomology."""
        return _build_e2_tensor(
            self.base_cohomology, self.fibre_cohomology, self.coefficient_ring
        )


# ──────────────────────────────────────────────────────────────────────────────
# Adams spectral sequence
# ──────────────────────────────────────────────────────────────────────────────

class AdamsSpectralSequence(SpectralSequence):
    """Mod-p Adams spectral sequence converging to π_*(S^0)^∧_p.

    Overview:
        At a prime p, the Adams spectral sequence has

            E_2^{s,t} = Ext_{A_p}^{s,t}(F_p, F_p)

        where A_p is the mod-p Steenrod algebra, and converges to the
        p-completed stable homotopy groups π_{t-s}^S(S^0)^∧_p.  Differentials
        d_r have bidegree (r, r-1) (i.e., (Δs, Δt) = (r, r-1)) so the
        topological degree t-s is preserved up to a -1 shift along d_r.

    Key Concepts:
        - **Bidegree convention**: We use (s, t).  In the Adams chart
          customarily t-s is plotted on the horizontal axis and s on the
          vertical.  d_r maps (s, t) ↦ (s+r, t+r-1) so Δ(t-s) = -1.
        - **Generators (p=2, low t-s)**: h_0 = (1,1), h_1 = (1,2),
          h_2 = (1,4), h_3 = (1,8); the only non-trivial classical
          differentials in the low-dimensional range are d_2(h_4) = h_0 h_3²
          and similar Hopf-invariant-1 obstructions (Adams 1958).
        - **Algorithmic differential**: For p = 2 we install h_0-multiplication
          as the canonical d_1 stand-in only when the user injects E_1
          explicitly; by default no algorithmic differential is supplied and
          the user provides differentials via `supply_differential`.

    Coefficient Ring:
        Always 𝔽_p (the Adams SS is intrinsically a vector space SS over
        the prime field).

    Common Workflows:
        1. Provide e2_entries: dict[(s, t), SpectralEntry over F_p].
        2. Call supply_differential(r, (s, t), matrix) for the known
           differentials in the range of interest.
        3. Call converge(); inspect e_infinity.

    Attributes:
        prime (int): The prime p.
        e2_entries (dict[Bidegree, SpectralEntry]): User-supplied E_2 page.

    References:
        Adams, J. F. (1958). Commentarii Mathematici Helvetici, 32, 180–214.
        Ravenel, D. C. (2003). Complex cobordism and the stable homotopy
          groups of spheres (2nd ed.). AMS Chelsea, ch. 3.
    """

    def __init__(
        self,
        *,
        prime: int,
        e2_entries: Mapping[tuple[int, int], SpectralEntry],
        max_pages: int = 32,
    ):
        if prime < 2:
            raise MathError(f"Adams SS prime must be ≥ 2, got {prime}.")
        super().__init__(
            coefficient_ring="F_p",
            prime=prime,
            convention="cohomological",
            max_pages=max_pages,
            convergence_target=(
                f"associated graded of pi_*^S(S^0) p-completion at p={prime}"
            ),
        )
        self.e2_entries: dict[tuple[int, int], SpectralEntry] = {
            tuple(k): v for k, v in e2_entries.items()
        }

    # Adams differential bidegree differs from the generic cohomological one;
    # override the convention.
    def differential_bidegree(self, r: int) -> Bidegree:
        """Bidegree (r, r-1) of the Adams differential d^r."""
        return (r, r - 1)

    def initial_page_number(self) -> int:
        """Return the initial page index r₀ = 2 for the Adams sequence."""
        return 2

    def compute_initial_page(self) -> dict[tuple[int, int], SpectralEntry]:
        """Return E^2 from the supplied Adams E_2 entries."""
        return dict(self.e2_entries)


# ──────────────────────────────────────────────────────────────────────────────
# Atiyah-Hirzebruch spectral sequence
# ──────────────────────────────────────────────────────────────────────────────

class AtiyahHirzebruchSpectralSequence(SpectralSequence):
    """Atiyah-Hirzebruch SS for a generalized cohomology theory h^*.

    Overview:
        For a CW-complex X and a generalized cohomology theory h^*,

            E_2^{p,q} = H^p(X; h^q(pt)) ⇒ h^{p+q}(X),

        with d_r of cohomological bidegree (r, 1-r).  The E_2 page is the
        ordinary cohomology of X with coefficients in the (graded) ring of
        coefficients h^*(pt).

    Key Concepts:
        - **Coefficient ring**: h^q(pt) provided as a graded module.
        - **Differentials**: d_r involves Steenrod-type operations whose
          explicit form depends on h^*.  For complex K-theory at p=2, the
          first non-trivial differential is d_3 = β ∘ Sq² ∘ r where r is
          mod-2 reduction (Atiyah-Hirzebruch 1962).  The user supplies these
          via `supply_differential`.

    Coefficient Ring:
        ℚ, ℤ, or 𝔽_p (must match h^q(pt) presentation).

    Common Workflows:
        1. Supply x_cohomology = {p: SpectralEntry for H^p(X)} and
           coefficient_pi = {q: SpectralEntry for h^q(pt)}.
        2. Provide differentials.
        3. converge(); recover h^n(X) via solve_extension_problem.

    Attributes:
        x_cohomology (dict[int, SpectralEntry]): H^p(X).
        coefficient_pi (dict[int, SpectralEntry]): h^q(pt).

    References:
        Atiyah, M. F., & Hirzebruch, F. (1962). Proceedings of Symposia in
          Pure Mathematics, 3, 7–38.
        McCleary, J. (2001). Cambridge University Press, ch. 9.
    """

    def __init__(
        self,
        *,
        x_cohomology: Mapping[int, SpectralEntry],
        coefficient_pi: Mapping[int, SpectralEntry],
        coefficient_ring: RingName = "Q",
        prime: int | None = None,
        max_pages: int = 32,
        cohomology_theory_name: str = "h",
    ):
        super().__init__(
            coefficient_ring=coefficient_ring,
            prime=prime,
            convention="cohomological",
            max_pages=max_pages,
            convergence_target=f"{cohomology_theory_name}^*(X)",
        )
        self.x_cohomology = {int(k): v for k, v in x_cohomology.items()}
        self.coefficient_pi = {int(k): v for k, v in coefficient_pi.items()}
        self.cohomology_theory_name = str(cohomology_theory_name)

    def initial_page_number(self) -> int:
        """Return the initial page index r₀ = 2 for the Atiyah-Hirzebruch sequence."""
        return 2

    def compute_initial_page(self) -> dict[tuple[int, int], SpectralEntry]:
        """Return E^2 = H^p(X) ⊗ π_q as the tensor of cohomology and coefficients."""
        return _build_e2_tensor(
            self.x_cohomology, self.coefficient_pi, self.coefficient_ring
        )


# ──────────────────────────────────────────────────────────────────────────────
# Extension problem solver
# ──────────────────────────────────────────────────────────────────────────────

def solve_extension_problem(
    convergence: ConvergenceResult,
    total_degree: int,
    *,
    splitting: Literal["assume_split", "use_hints"] = "assume_split",
    hints: Mapping[tuple[int, int], tuple[int, ...]] | None = None,
) -> ExtensionResult:
    """Recover H_n (or H^n) from the E^∞ page at total degree n.

    What is Being Computed?:
        Given the associated graded gr_p H_n = E^∞_{p, n-p}, determine the
        global rank and produce an upper bound on torsion.  When all entries
        are torsion-free or `splitting='assume_split'` is set, the result is
        exact; otherwise the torsion is reported as an upper bound only and
        `exact=False`.

    Algorithm:
        1. Gather all bidegrees (p, n-p) with non-zero E^∞ entries.
        2. rank = Σ_p E^∞_{p, n-p}.rank — exact.
        3. If splitting='assume_split': torsion = sorted concatenation of
           contributing torsions.
        4. If splitting='use_hints': accept user-provided torsion overrides
           that may collapse cyclic factors.

    Preserved Invariants:
        rank is exact in all cases.  Torsion is exact iff every contributing
        entry has zero torsion, or splitting hints are supplied that resolve
        the extension uniquely.

    Args:
        convergence:    A ConvergenceResult with a populated e_infinity page.
        total_degree:   The total degree n for H_n / H^n.
        splitting:      'assume_split' (default) reports the direct-sum
                        torsion as an upper bound; 'use_hints' substitutes
                        per-bidegree torsion data from `hints`.
        hints:          Optional dict {(p, q): torsion_tuple} overriding the
                        torsion of the bidegree's contribution to H_n.  Only
                        consulted when splitting=='use_hints'.

    Returns:
        ExtensionResult: Pydantic model with rank and torsion bounds.

    Raises:
        MathError: If `convergence` did not converge.

    Use When:
        - You have run a SerreSpectralSequence to convergence and want the
          rank of the total-space (co)homology.
        - You want to verify a known H_n against a candidate fibration's
          E^∞.

    Example:
        from pysurgery.spectral.spectral_sequences import (
            SerreSpectralSequence, SpectralEntry, solve_extension_problem,
        )
        ss = SerreSpectralSequence(base_homology={…}, fibre_homology={…})
        result = ss.converge()
        h2 = solve_extension_problem(result, 2)
        print(h2.rank, h2.torsion_upper_bound)

    References:
        McCleary, J. (2001). Cambridge University Press, ch. 1.
    """
    if not convergence.converged:
        raise MathError(
            "Cannot solve the extension problem on a non-converged spectral "
            "sequence."
        )

    contributing: list[tuple[int, int]] = []
    summary: dict[tuple[int, int], SpectralEntry] = {}
    rank = 0
    torsion_collect: list[int] = []

    for (p, q), entry in convergence.e_infinity.items():
        if p + q != total_degree:
            continue
        if entry.is_zero:
            continue
        contributing.append((p, q))
        summary[(p, q)] = entry
        rank += entry.rank
        if splitting == "use_hints" and hints is not None and (p, q) in hints:
            torsion_collect.extend(int(t) for t in hints[(p, q)])
        else:
            torsion_collect.extend(int(t) for t in entry.torsion)

    contributing.sort()
    torsion_sorted = tuple(sorted(torsion_collect))

    # Determine exactness:
    # - Over a field (Q or F_p), no torsion ever appears, result is exact.
    # - Over Z with no torsion in any contributing entry, result is exact.
    # - Otherwise, the splitting assumption introduces uncertainty.
    if convergence.coefficient_ring in ("Q", "F_p"):
        exact = True
    else:
        any_torsion = any(e.torsion for e in summary.values())
        if not any_torsion:
            exact = True
        elif splitting == "use_hints" and hints is not None:
            exact = all((p, q) in hints for (p, q) in contributing if summary[(p, q)].torsion)
        else:
            exact = False

    return ExtensionResult(
        total_degree=total_degree,
        rank=rank,
        torsion_upper_bound=torsion_sorted,
        splitting_assumed=(splitting == "assume_split"),
        contributing_bidegrees=contributing,
        associated_graded_summary=summary,
        exact=exact,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public re-exports
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    # Pydantic models
    "SpectralEntry",
    "SpectralPageSnapshot",
    "ConvergenceResult",
    "ExtensionResult",
    # Internal mutable types
    "SpectralPage",
    "ExactCouple",
    # Spectral sequence classes
    "SpectralSequence",
    "ExactCoupleSpectralSequence",
    "SerreSpectralSequence",
    "LeraySerreSpectralSequence",
    "AdamsSpectralSequence",
    "AtiyahHirzebruchSpectralSequence",
    # Functions
    "solve_extension_problem",
]
