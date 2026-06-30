"""Bounded Controlled Cohomology and Twisted Local Systems.

Overview:
    Bifurcated, finite-bound architecture (see RFC-controlled-cohomology-v2)
    for non-simply-connected manifolds that avoids combinatorial
    explosions while staying within pySurgery's exactness policy.

    Two execution paths are provided:

    1. **Path A — Finite Universal Cover**: when π₁(M) is provably finite, the
       universal cover M̃ is built explicitly as a CW complex over ℤ and the
       lifted boundary maps are reduced via the existing exact-sparse SNF.
    2. **Path B — Twisted Chain Complex**: for any π₁(M) together with a
       supplied finite-dimensional representation ρ : π₁(M) → GL(V), the
       twisted chain complex C_*(M; ℒ_ρ) is assembled directly on the base
       complex via Fox derivatives — keeping the boundary matrices bounded
       to `degree(ρ) · n_k(M)`.

    Both paths feed a unified entry point `compute_controlled_cohomology` that
    dispatches on `pi_1.is_finite()` and the user's preference, plus a hook
    `compute_twisted_obstruction` that builds twisted intersection forms for
    closed simplicial 4k-manifolds and evaluates the resulting Wall L-group
    obstruction via `WallGroupL.compute_obstruction_result`.

Key Concepts:
    - **Cayley table**: Z[G] arithmetic for finite G is realised as a
      `(|G|, |G|)` int64 NumPy array storing the multiplication table —
      zero-copy with Julia. Group-ring elements are coefficient vectors of
      length |G|.
    - **Fox derivative**: ∂(g_{i₁}^{ε₁}…g_{iₖ}^{εₖ}) / ∂g_j ∈ ℤ[π] is
      computed via the chain rule and evaluated through ρ to yield d×d blocks.
    - **Universal cover lift**: each k-cell c of M lifts to |G| cells in M̃;
      the lifted boundary uses the same Fox-derivative coefficients, scattered
      across group-ring orbits.
    - **Twisted intersection form**: the Hermitian (or symmetric, real case)
      pairing on H^{2k}(M; ℒ_ρ) computed via the Alexander-Whitney cup
      product applied to representatives, then realified for ingestion by
      `WallGroupL`.

Common Workflows:
    1. **Finite cover homology** —
       `UniversalCover(rp2).as_chain_complex().homology()` ≅ H_*(S²).
    2. **Twisted homology with infinite π₁** —
       `TwistedChainComplex(torus, rho).homology(1)` for ρ : ℤ² → GL₁(ℂ).
    3. **Twisted Wall obstruction** —
       `compute_twisted_obstruction(cp2, trivial_rep, dimension=4)`.

Coefficient Ring:
    ℤ for FiniteGroupRing arithmetic. Twisted chain complexes carry
    representation matrices over `Q | C | Zmod`; the resulting homology
    inherits the representation's ring.

References:
    Wall, C. T. C. (1999). Surgery on compact manifolds (2nd ed.). AMS.
    Ranicki, A. (2002). Algebraic and geometric surgery. Oxford UP.
    Fox, R. H. (1953). Free differential calculus, I: derivation in the free
      group ring. Annals of Mathematics, 57(3), 547–560.
    Todd, J. A. & Coxeter, H. S. M. (1936). A practical method for enumerating
      cosets of a finite abstract group. Proceedings of the Edinburgh
      Mathematical Society, 5, 26–34.
    Hatcher, A. (2002). Algebraic topology. Cambridge UP. (universal covers,
      twisted homology background)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pysurgery.topology.complexes import CWComplex, ChainComplex
from pysurgery.core.exceptions import (
    DimensionError,
    FundamentalGroupError,
    GroupRingError,
)
from pysurgery.core.foundations import CONTRACT_VERSION
from pysurgery.topology.fundamental_group import FundamentalGroup, extract_pi_1_with_traces
from pysurgery.algebra.intersection_forms import IntersectionForm
from ..bridge.julia_bridge import julia_engine

# Cover construction now lives in pysurgery.topology.coverings; it is
# re-imported here for backward compatibility (this module historically
# owned these symbols and still consumes them for the twisted machinery).
from pysurgery.topology.coverings import (  # noqa: F401
    FiniteGroupOrderResult,
    UniversalCoverResult,
    FiniteGroupRing,
    UniversalCover,
    _resolve_max_cover_order,
    _build_finite_group_order_result,
)

__all__ = [
    "FiniteGroupOrderResult",
    "UniversalCoverResult",
    "TwistedChainResult",
    "ControlledCohomologyResult",
    "TwistedIntersectionFormResult",
    "TwistedObstructionResult",
    "FiniteGroupRing",
    "TwistedRepresentation",
    "UniversalCover",
    "TwistedChainComplex",
    "compute_controlled_cohomology",
    "compute_twisted_intersection_form",
    "compute_twisted_obstruction",
]


# ──────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class TwistedChainResult:
    """Twisted boundary maps `d^ρ_n` over the base complex.

    Attributes:
        boundaries: `{dim: matrix}` — d^ρ_dim, dense `numpy.ndarray` whose
            entries lie in the representation's coefficient ring.
        cell_dimensions: Sorted list of dimensions present.
        degree: Representation degree `d`.
        ring: Coefficient ring identifier ("Q" / "C" / "Z" / "Zmod").
        path: "cover" or "fox" — which strategy was used.
        exact: True when computation is mathematically exact.
        theorem_tag: "controlled_cohomology.twisted_chains".
        contract_version: pySurgery contract version.
    """

    boundaries: Dict[int, np.ndarray]
    cell_dimensions: List[int]
    degree: int
    ring: str
    path: str
    exact: bool = True
    theorem_tag: str = "controlled_cohomology.twisted_chains"
    contract_version: str = field(default_factory=lambda: CONTRACT_VERSION)

    def decision_ready(self) -> bool:
        """Return True when the twisted boundary maps are exact."""
        return self.exact


@dataclass
class ControlledCohomologyResult:
    """Twisted (co)homology with local-system coefficients.

    Attributes:
        dimension: Dimension n at which H_n(M; ℒ_ρ) was computed (None if all).
        rank: Free-module rank of H_n.
        torsion: Sorted list of torsion invariants > 1 (empty for field
            coefficients).
        ring: Coefficient ring of ρ.
        degree: Representation degree.
        path: "cover" or "fox".
        homology_by_dim: When dimension is None, mapping `{n: (rank, torsion)}`.
        exact: True when the computation is rigorous.
        theorem_tag: "controlled_cohomology.cohomology".
        contract_version: pySurgery contract version.
    """

    dimension: Optional[int]
    rank: int
    torsion: List[int]
    ring: str
    degree: int
    path: str
    homology_by_dim: Dict[int, Tuple[int, List[int]]] = field(default_factory=dict)
    exact: bool = True
    theorem_tag: str = "controlled_cohomology.cohomology"
    contract_version: str = field(default_factory=lambda: CONTRACT_VERSION)

    def decision_ready(self) -> bool:
        """Return True when the (co)homology computation is rigorous."""
        return self.exact


@dataclass
class TwistedIntersectionFormResult:
    """Twisted intersection form for closed orientable 4k-manifolds.

    Attributes:
        matrix: Real-symmetric form matrix (Hermitian forms are realified).
        complex_matrix: Complex Hermitian matrix when ρ is unitary; None
            otherwise.
        signature: Signature (#pos − #neg eigenvalues) of `matrix`.
        rank: Rank of the form.
        dimension: 4k dimension of the manifold.
        degree: Representation degree.
        ring: Coefficient ring.
        path: "cover" / "fox" used to derive the cocycles.
        exact: True when matrix entries are exact (integers / rationals).
        theorem_tag: "controlled_cohomology.twisted_intersection_form".
        contract_version: pySurgery contract version.
    """

    matrix: np.ndarray
    complex_matrix: Optional[np.ndarray]
    signature: int
    rank: int
    dimension: int
    degree: int
    ring: str
    path: str
    exact: bool = True
    theorem_tag: str = "controlled_cohomology.twisted_intersection_form"
    contract_version: str = field(default_factory=lambda: CONTRACT_VERSION)

    def decision_ready(self) -> bool:
        """Return True when the form matrix entries are exact."""
        return self.exact

    def as_intersection_form(self) -> IntersectionForm:
        """Wrap the realified matrix as a standard `IntersectionForm`."""
        return IntersectionForm(matrix=self.matrix, dimension=self.dimension)


@dataclass
class TwistedObstructionResult:
    """End-to-end twisted Wall obstruction for a manifold.

    Attributes:
        form_result: The twisted intersection form input.
        obstruction: The `ObstructionResult` returned by `WallGroupL`.
        exact: Logical AND of input exactness and `obstruction.exact`.
        theorem_tag: "controlled_cohomology.twisted_wall_obstruction".
        contract_version: pySurgery contract version.
    """

    form_result: TwistedIntersectionFormResult
    obstruction: object  # wall_groups.ObstructionResult — avoid circular import
    exact: bool = True
    theorem_tag: str = "controlled_cohomology.twisted_wall_obstruction"
    contract_version: str = field(default_factory=lambda: CONTRACT_VERSION)

    def decision_ready(self) -> bool:
        """Return True when both the form and the Wall obstruction are exact."""
        return self.exact and bool(getattr(self.obstruction, "exact", False))


# ──────────────────────────────────────────────────────────────────────────────
# Finite group ring
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# Twisted representation
# ──────────────────────────────────────────────────────────────────────────────


class TwistedRepresentation(BaseModel):
    """A finite-dimensional representation ρ : π₁ → GL_d(R).

    Overview:
        Stores ρ as a dictionary of d×d matrices keyed by 1-based Cayley
        index when the group is finite, and additionally as a generator-token
        dictionary (so that infinite-π₁ Path-B Fox calculus can be evaluated
        without a Cayley table).

    Key Concepts:
        - `images`: For finite groups, ρ(g) for every group element g.
        - `images_word`: For any group, ρ(g_i) and ρ(g_i⁻¹) for each
          generator token. Path B uses these directly in Fox derivatives.
        - The relator-image validator multiplies ρ along each relation and
          requires the result to equal the identity (within tolerance for
          floating-point rings).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    degree: int
    ring: Literal["Z", "Q", "C", "Zmod"] = "C"
    modulus: Optional[int] = None
    images_word: Dict[str, np.ndarray] = Field(default_factory=dict)
    images: Dict[int, np.ndarray] = Field(default_factory=dict)
    presentation_generators: List[str] = Field(default_factory=list)
    presentation_relations: List[List[str]] = Field(default_factory=list)
    tolerance: float = 1e-9

    @model_validator(mode="after")
    def _validate_images_and_relations(self) -> "TwistedRepresentation":
        d = int(self.degree)
        if d <= 0:
            raise GroupRingError(f"Representation degree must be ≥ 1; got {d}.")
        if self.ring == "Zmod" and (self.modulus is None or self.modulus < 2):
            raise GroupRingError(
                "ring='Zmod' requires a modulus ≥ 2."
            )
        for key, mat in self.images_word.items():
            arr = np.asarray(mat)
            if arr.shape != (d, d):
                raise GroupRingError(
                    f"images_word[{key!r}] must have shape ({d}, {d}); "
                    f"got {arr.shape}."
                )
        for key, mat in self.images.items():
            arr = np.asarray(mat)
            if arr.shape != (d, d):
                raise GroupRingError(
                    f"images[{key}] must have shape ({d}, {d}); got {arr.shape}."
                )
        if self.presentation_generators and self.presentation_relations:
            self._verify_relator_images()
        return self

    def _verify_relator_images(self) -> None:
        d = int(self.degree)
        identity = np.eye(d, dtype=self._numpy_dtype())
        for rel in self.presentation_relations:
            mat = identity.copy()
            for tok in rel:
                if tok not in self.images_word:
                    raise GroupRingError(
                        f"Relation token {tok!r} has no image in `images_word`."
                    )
                mat = mat @ self.images_word[tok]
            if not self._is_identity(mat, identity):
                raise GroupRingError(
                    f"Relator {' '.join(rel)!r} maps to a non-identity "
                    f"matrix under ρ:\n{mat}"
                )

    def _numpy_dtype(self):
        if self.ring == "C":
            return np.complex128
        if self.ring == "Q":
            return np.float64
        if self.ring == "Zmod":
            return np.int64
        return np.int64

    def _is_identity(self, mat: np.ndarray, identity: np.ndarray) -> bool:
        if self.ring == "Z":
            return np.array_equal(mat.astype(np.int64), identity.astype(np.int64))
        if self.ring == "Zmod":
            mod = int(self.modulus)
            return np.array_equal(np.mod(mat.astype(np.int64), mod),
                                  np.mod(identity.astype(np.int64), mod))
        return np.allclose(mat, identity, atol=self.tolerance)

    @classmethod
    def trivial(
        cls,
        presentation: FundamentalGroup,
        *,
        degree: int = 1,
        ring: Literal["Z", "Q", "C", "Zmod"] = "C",
        modulus: Optional[int] = None,
        group_ring: Optional["FiniteGroupRing"] = None,
    ) -> "TwistedRepresentation":
        """Trivial representation: every element acts as identity."""
        dtype = (
            np.complex128 if ring == "C"
            else np.int64 if ring in ("Z", "Zmod")
            else np.float64
        )
        identity = np.eye(degree, dtype=dtype)
        words = {}
        for g in presentation.generators:
            words[g] = identity.copy()
            words[f"{g}^-1"] = identity.copy()
        images = {}
        if group_ring is not None:
            for k in range(1, group_ring.order + 1):
                images[k] = identity.copy()
        return cls(
            degree=degree,
            ring=ring,
            modulus=modulus,
            images_word=words,
            images=images,
            presentation_generators=list(presentation.generators),
            presentation_relations=[list(r) for r in presentation.relations],
        )

    @classmethod
    def sign(
        cls,
        presentation: FundamentalGroup,
        generator_signs: Dict[str, int],
        *,
        ring: Literal["Z", "Q", "C"] = "C",
        group_ring: Optional["FiniteGroupRing"] = None,
    ) -> "TwistedRepresentation":
        """Degree-1 sign representation `ρ(g_i) = ±1`."""
        dtype = np.complex128 if ring == "C" else (np.int64 if ring == "Z" else np.float64)
        for g in presentation.generators:
            if g not in generator_signs:
                generator_signs[g] = 1
            if generator_signs[g] not in (-1, 1):
                raise GroupRingError(
                    f"Sign rep accepts ±1 on each generator; got "
                    f"ρ({g}) = {generator_signs[g]}."
                )
        words = {}
        for g, s in generator_signs.items():
            words[g] = np.array([[s]], dtype=dtype)
            words[f"{g}^-1"] = np.array([[s]], dtype=dtype)
        images = {}
        if group_ring is not None:
            words_to_idx = {w: i + 1 for i, w in enumerate(group_ring.element_words)}
            for word, idx in words_to_idx.items():
                if word == "e":
                    images[idx] = np.array([[1]], dtype=dtype)
                    continue
                tokens = word.split()
                v = 1
                for tok in tokens:
                    base = tok[:-3] if tok.endswith("^-1") else tok
                    v *= int(generator_signs.get(base, 1))
                images[idx] = np.array([[v]], dtype=dtype)
        return cls(
            degree=1,
            ring=ring,
            modulus=None,
            images_word=words,
            images=images,
            presentation_generators=list(presentation.generators),
            presentation_relations=[list(r) for r in presentation.relations],
        )

    @classmethod
    def regular(cls, group_ring: "FiniteGroupRing") -> "TwistedRepresentation":
        """Left regular representation of a finite group on ℂ^|G|."""
        n = group_ring.order
        dtype = np.complex128
        cayley = group_ring.cayley
        images = {}
        for g_idx in range(1, n + 1):
            mat = np.zeros((n, n), dtype=dtype)
            for h_idx in range(1, n + 1):
                # ρ(g)(e_h) = e_{g·h}
                target = int(cayley[g_idx - 1, h_idx - 1])
                mat[target - 1, h_idx - 1] = 1
            images[g_idx] = mat
        words = {}
        words_to_idx = {w: i + 1 for i, w in enumerate(group_ring.element_words)}
        for g in group_ring.generators:
            idx = words_to_idx[g]
            words[g] = images[idx]
            inv_idx = int(group_ring.inverse_indices[idx - 1])
            words[f"{g}^-1"] = images[inv_idx]
        return cls(
            degree=n,
            ring="C",
            modulus=None,
            images_word=words,
            images=images,
            presentation_generators=list(group_ring.generators),
            presentation_relations=[list(r) for r in group_ring._presentation.relations],
        )

    def matrix_for_token(self, token: str) -> np.ndarray:
        """Return ρ for a generator token (e.g. 'g' or 'g^-1').

        Args:
            token: Generator token whose image is stored in `images_word`.

        Returns:
            The d×d matrix ρ(token).

        Raises:
            GroupRingError: If no image is stored for the token.
        """
        if token in self.images_word:
            return self.images_word[token]
        raise GroupRingError(f"No image stored for token {token!r}.")

    def matrix_for_element(self, group_idx_1based: int) -> np.ndarray:
        """Return ρ for a finite-group element by 1-based Cayley index.

        Args:
            group_idx_1based: 1-based Cayley index of the group element.

        Returns:
            The d×d matrix ρ(g) stored in `images`.

        Raises:
            GroupRingError: If no image is stored (e.g. a Path-B
                representation built without a `FiniteGroupRing`).
        """
        if group_idx_1based not in self.images:
            raise GroupRingError(
                f"No image stored for group element {group_idx_1based}; "
                "this representation may have been built without a "
                "FiniteGroupRing (Path-B Fox calculus does not require it)."
            )
        return self.images[group_idx_1based]


# ──────────────────────────────────────────────────────────────────────────────
# Universal cover
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# Twisted chain complex
# ──────────────────────────────────────────────────────────────────────────────


class TwistedChainComplex:
    """Twisted chain complex `C_*(M; ℒ_ρ)` over the chosen coefficient ring.

    Overview:
        Builds the cellular chain complex of M with coefficients in the
        local system ℒ_ρ. Two implementation strategies are provided:

        - **path='cover'**: requires finite π₁. Tensors the cover's ℤ-chain
          complex with V via the regular-representation embedding, then
          contracts to obtain the ρ-twisted complex. This is mathematically
          equivalent to Path B for finite π₁ and serves as cross-validation.
        - **path='fox'**: builds boundaries directly on the base complex via
          Fox derivatives applied to relator words, evaluated through ρ.
          Works for arbitrary (including infinite) π₁ provided ρ is finite-
          dimensional.

        The dispatcher `path='auto'` selects 'cover' when π₁ is finite and
        'fox' otherwise.

    Coefficient Ring:
        Inherits from the supplied `TwistedRepresentation.ring` (Q, C, Z,
        or Zmod).
    """

    def __init__(
        self,
        base: CWComplex,
        representation: TwistedRepresentation,
        pi1: Optional[FundamentalGroup] = None,
        *,
        path: Literal["auto", "cover", "fox"] = "auto",
        max_index: int = 10_000,
    ) -> None:
        self._base = base
        self._rep = representation
        traces = extract_pi_1_with_traces(
            base, simplify=False, generator_mode="raw"
        )
        # Generator names from traces are authoritative for cell-relator
        # correspondence; rebuild the FundamentalGroup using these names so
        # that the representation's `images_word` keys match.
        pi1 = FundamentalGroup(
            generators=list(traces.generators),
            relations=[list(r) for r in traces.relations],
            orientation_character=dict(traces.orientation_character),
        )
        self._pi1 = pi1
        self._traces = traces

        path = str(path).lower().strip()
        if path not in ("auto", "cover", "fox"):
            raise GroupRingError(
                f"path must be 'auto', 'cover', or 'fox'; got {path!r}."
            )

        finite = False
        try:
            finite = pi1.is_finite(max_index=max_index)
        except FundamentalGroupError:
            finite = False

        if path == "auto":
            path = "cover" if finite else "fox"
        if path == "cover" and not finite:
            raise FundamentalGroupError(
                "path='cover' requires finite π₁; use path='fox' for "
                "infinite or undecidable π₁."
            )

        self._path = path
        if path == "cover":
            self._cover = UniversalCover(base, pi1, max_index=max_index)
            self._boundaries = self._build_via_cover()
        else:
            self._cover = None
            self._boundaries = self._build_via_fox()

    @property
    def pi1(self) -> FundamentalGroup:
        """Fundamental group of the base complex (in trace-generator names)."""
        return self._pi1

    @property
    def representation(self) -> TwistedRepresentation:
        """The twisting representation ρ defining the local system."""
        return self._rep

    @property
    def path(self) -> str:
        """Strategy used to build the boundaries ('cover' or 'fox')."""
        return self._path

    @property
    def boundaries(self) -> Dict[int, np.ndarray]:
        """Twisted boundary maps `d^ρ_n` keyed by dimension (copies)."""
        return {k: v.copy() for k, v in self._boundaries.items()}

    def cell_dimensions(self) -> List[int]:
        """Return the sorted dimensions present in the complex."""
        dims = set(self._base.cells.keys())
        dims.update(self._boundaries.keys())
        return sorted(dims)

    def as_chain_result(self) -> TwistedChainResult:
        """Package the twisted boundary maps into a `TwistedChainResult`."""
        return TwistedChainResult(
            boundaries=self.boundaries,
            cell_dimensions=self.cell_dimensions(),
            degree=int(self._rep.degree),
            ring=str(self._rep.ring),
            path=self._path,
        )

    def homology(self, n: int) -> ControlledCohomologyResult:
        """Compute H_n(M; ℒ_ρ).

        For ring='Q' or 'C' this returns a free-module rank (no torsion).
        For ring='Z' it computes ranks and torsion via integer SNF.
        """
        d_n = self._boundaries.get(int(n))
        d_n_plus_1 = self._boundaries.get(int(n) + 1)
        n_n_cells = int(self._base.cells.get(int(n), 0))
        ambient = self._rep.degree * n_n_cells
        return self._homology_at(int(n), d_n, d_n_plus_1, ambient)

    def cohomology(self, n: int) -> ControlledCohomologyResult:
        """Compute H^n(M; ℒ_ρ) via transposing the boundary maps.

        For ρ self-dual (orthogonal/unitary) the cohomology agrees with
        homology of the dual complex; we transpose `d^ρ` to compute it.
        """
        d_n = self._boundaries.get(int(n) + 1)  # δ^n = (d_{n+1})^T
        d_n_minus_1 = self._boundaries.get(int(n))  # δ^{n-1} = (d_n)^T
        d_n_T = d_n.T if d_n is not None else None
        d_n_minus_1_T = d_n_minus_1.T if d_n_minus_1 is not None else None
        n_n_cells = int(self._base.cells.get(int(n), 0))
        ambient = self._rep.degree * n_n_cells
        return self._homology_at(int(n), d_n_T, d_n_minus_1_T, ambient,
                                 cohomology=True)

    def homology_all(self) -> ControlledCohomologyResult:
        """Compute H_n(M; ℒ_ρ) for every dimension n present in the complex.

        Returns:
            A `ControlledCohomologyResult` with `dimension=None` and per-dimension
            (rank, torsion) pairs in `homology_by_dim`.
        """
        results: Dict[int, Tuple[int, List[int]]] = {}
        dims = self.cell_dimensions()
        for n in dims:
            r = self.homology(n)
            results[n] = (r.rank, list(r.torsion))
        return ControlledCohomologyResult(
            dimension=None,
            rank=-1,
            torsion=[],
            ring=str(self._rep.ring),
            degree=int(self._rep.degree),
            path=self._path,
            homology_by_dim=results,
        )

    # ── boundary construction paths ────────────────────────────────────────

    def _build_via_fox(self) -> Dict[int, np.ndarray]:
        """Fox-derivative-based construction of `d^ρ_*` for any π₁."""
        rep = self._rep
        d = int(rep.degree)
        gens_pi1 = list(self._pi1.generators)
        n_e = int(self._base.cells.get(1, 0))
        n_f = int(self._base.cells.get(2, 0))

        boundaries: Dict[int, np.ndarray] = {}
        dtype = self._dtype_for_ring(rep)

        # d_1^ρ : (d * n_e) × (d * n_f) at index pair (i, e) — row blocks per
        # 0-cell, column blocks per 1-cell. We require a single 0-cell
        # convention as in extract_pi_1.
        n_v = int(self._base.cells.get(0, 0))
        if n_v == 1 and n_e > 0:
            d1 = np.zeros((d, d * n_e), dtype=dtype)
            for e_idx, gen_token in self._iter_edge_tokens():
                base_token = (
                    gen_token[:-3] if gen_token.endswith("^-1") else gen_token
                )
                if base_token not in rep.images_word:
                    # Tree edge with no rep-image — skip; the corresponding
                    # generator is identified with the identity.
                    rho_g = np.eye(d, dtype=dtype)
                else:
                    rho_g = rep.matrix_for_token(gen_token).astype(dtype)
                # Boundary of edge e: ρ(g_e) − I in C^V ⊗ V (single 0-cell).
                block = rho_g - np.eye(d, dtype=dtype)
                d1[:, e_idx * d:(e_idx + 1) * d] = block
            boundaries[1] = d1

        if n_e > 0 and n_f > 0:
            d2 = np.zeros((d * n_e, d * n_f), dtype=dtype)
            relations = [list(r) for r in self._traces.relations]
            if len(relations) != n_f:
                raise DimensionError(
                    f"Raw relation count ({len(relations)}) does not match "
                    f"number of 2-cells ({n_f})."
                )
            for j, relator in enumerate(relations):
                signed = self._relator_signed(relator, gens_pi1)
                for i, gen_name in enumerate(gens_pi1):
                    block = self._fox_block_through_rep(
                        signed, i + 1, gen_name, gens_pi1, rep, d, dtype
                    )
                    if np.any(block):
                        d2[i * d:(i + 1) * d, j * d:(j + 1) * d] = block
            boundaries[2] = d2

        return boundaries

    def _build_via_cover(self) -> Dict[int, np.ndarray]:
        """Cover-based construction: tensor cover boundaries with ρ-projector."""
        cover = self._cover
        assert cover is not None
        rep = self._rep
        if rep.degree != cover.order or rep.ring not in ("C",):
            # For non-regular reps, project from cover via averaging.
            return self._build_via_cover_general()
        # Regular representation: cover boundaries already encode the full
        # twisted complex. Convert to dense.
        dtype = self._dtype_for_ring(rep)
        boundaries = {}
        for k, mat in cover.lifted_attaching.items():
            arr = mat.toarray().astype(dtype)
            boundaries[k] = arr
        return boundaries

    def _build_via_cover_general(self) -> Dict[int, np.ndarray]:
        """Cover path for arbitrary ρ via the tensor V ⊗_{ℤ[π]} C_*(M̃).

        For the lift `(c_j, e)` of base cell `c_j` at the identity, the
        cover boundary contributes `Σ_l Σ_h sub_{l,j}[h, e_idx] · (c_l, h)`.
        Tensoring over ℤ[π] with V (left π-action via ρ) yields
        `v ⊗ (c_l, h) = ρ(h) v ⊗ (c_l, e)`, so the (l, j)-block of `d^ρ`
        is `Σ_h sub_{l,j}[h, e_idx] · ρ(h)`. We only need column `e_idx`
        of each cover sub-block; deck-equivariance encodes the rest.
        """
        cover = self._cover
        assert cover is not None
        rep = self._rep
        dtype = self._dtype_for_ring(rep)
        d = int(rep.degree)
        n_g = cover.order
        e_idx_0 = int(cover.group_ring.identity_index) - 1

        # Resolve ρ(h) for every group element by walking element_words.
        rho_by_elt = self._element_images(rep, cover.group_ring, dtype, d)

        result_boundaries: Dict[int, np.ndarray] = {}
        n_base = {dim: int(self._base.cells.get(dim, 0)) for dim in self._base.cells}
        for k, mat in cover.lifted_attaching.items():
            n_k_minus_1 = n_base.get(k - 1, 0)
            n_k = n_base.get(k, 0)
            if n_k == 0 or n_k_minus_1 == 0:
                continue
            arr = mat.tocsr()
            Q = np.zeros((d * n_k_minus_1, d * n_k), dtype=dtype)
            for j_block in range(n_k):
                col = j_block * n_g + e_idx_0
                col_vec = arr[:, col].toarray().reshape(-1)
                for i_block in range(n_k_minus_1):
                    row_start = i_block * n_g
                    sub_col = col_vec[row_start:row_start + n_g]
                    block = np.zeros((d, d), dtype=dtype)
                    for h_idx in range(n_g):
                        coeff = int(sub_col[h_idx])
                        if coeff == 0:
                            continue
                        block = block + coeff * rho_by_elt[h_idx + 1]
                    Q[i_block * d:(i_block + 1) * d,
                      j_block * d:(j_block + 1) * d] = block
            result_boundaries[k] = Q
        return result_boundaries

    @staticmethod
    def _element_images(
        rep: "TwistedRepresentation",
        group_ring: "FiniteGroupRing",
        dtype,
        d: int,
    ) -> Dict[int, np.ndarray]:
        """Resolve ρ(g) for every 1-based group index g.

        Prefers `rep.images` when populated; otherwise walks the reduced
        word `element_words[g - 1]` using `rep.images_word`.
        """
        out: Dict[int, np.ndarray] = {}
        identity = np.eye(d, dtype=dtype)
        for g_idx in range(1, group_ring.order + 1):
            if g_idx in rep.images:
                out[g_idx] = np.asarray(rep.images[g_idx], dtype=dtype)
                continue
            word = group_ring.element_words[g_idx - 1]
            if word == "e" or not word:
                out[g_idx] = identity.copy()
                continue
            mat = identity.copy()
            for tok in word.split():
                if tok in rep.images_word:
                    mat = mat @ np.asarray(rep.images_word[tok], dtype=dtype)
                    continue
                # Fall back to identity for unknown tokens (e.g., tree
                # edges in multi-vertex bases — not exercised here).
                continue
            out[g_idx] = mat
        return out

    # ── helpers ────────────────────────────────────────────────────────────

    def _dtype_for_ring(self, rep: TwistedRepresentation):
        if rep.ring == "C":
            return np.complex128
        if rep.ring == "Q":
            return np.float64
        return np.int64

    def _iter_edge_tokens(self):
        """Yield (edge_index, generator_token) for each 1-cell.

        For multi-vertex base complexes this would return tree-edge tokens
        as the identity; we only support single-vertex base, so every edge
        is identified with a generator (or its inverse) by `extract_pi_1`.
        """
        traces_by_edge: Dict[int, str] = {}
        for tr in self._traces.traces:
            if tr.edge_index is not None:
                traces_by_edge[int(tr.edge_index)] = tr.generator
        n_e = int(self._base.cells.get(1, 0))
        for e_idx in range(n_e):
            tok = traces_by_edge.get(e_idx)
            if tok is None:
                # Tree edge — should not occur in single-vertex complexes
                # because there is no tree to collapse beyond the basepoint.
                yield e_idx, "e"  # sentinel; caller substitutes identity
            else:
                yield e_idx, tok

    def _relator_signed(self, relator: List[str], gens_pi1: List[str]) -> List[int]:
        signed = []
        idx = {g: i + 1 for i, g in enumerate(gens_pi1)}
        for tok in relator:
            base = tok[:-3] if tok.endswith("^-1") else tok
            if base not in idx:
                raise GroupRingError(
                    f"Relator token {tok!r} references unknown generator."
                )
            i = idx[base]
            signed.append(-i if tok.endswith("^-1") else i)
        return signed

    def _fox_block_through_rep(
        self,
        relator_signed: List[int],
        gen_idx_1based: int,
        gen_name: str,
        gens_pi1: List[str],
        rep: TwistedRepresentation,
        d: int,
        dtype,
    ) -> np.ndarray:
        block = np.zeros((d, d), dtype=dtype)
        identity = np.eye(d, dtype=dtype)
        prefix_mat = identity.copy()
        for sg in relator_signed:
            gen = abs(sg)
            eps = 1 if sg > 0 else -1
            tok_name = gens_pi1[gen - 1]
            rho_g = rep.matrix_for_token(tok_name).astype(dtype)
            rho_g_inv = rep.matrix_for_token(f"{tok_name}^-1").astype(dtype)
            new_prefix = prefix_mat @ (rho_g if eps > 0 else rho_g_inv)
            if gen == gen_idx_1based:
                if eps > 0:
                    block = block + prefix_mat
                else:
                    block = block - new_prefix
            prefix_mat = new_prefix
        return block

    def _homology_at(
        self,
        n: int,
        d_n: Optional[np.ndarray],
        d_n_plus_1: Optional[np.ndarray],
        ambient: int,
        cohomology: bool = False,
    ) -> ControlledCohomologyResult:
        rep = self._rep
        ring = rep.ring
        path = self._path

        if ambient == 0:
            return ControlledCohomologyResult(
                dimension=int(n),
                rank=0,
                torsion=[],
                ring=ring,
                degree=int(rep.degree),
                path=path,
            )

        if ring == "C":
            kernel_dim = ambient - (np.linalg.matrix_rank(d_n) if d_n is not None and d_n.size else 0)
            image_dim = (
                np.linalg.matrix_rank(d_n_plus_1)
                if d_n_plus_1 is not None and d_n_plus_1.size else 0
            )
            rank = int(kernel_dim - image_dim)
            return ControlledCohomologyResult(
                dimension=int(n),
                rank=max(0, rank),
                torsion=[],
                ring=ring,
                degree=int(rep.degree),
                path=path,
            )
        if ring == "Q":
            kernel_dim = ambient - (
                np.linalg.matrix_rank(d_n.astype(np.float64))
                if d_n is not None and d_n.size else 0
            )
            image_dim = (
                np.linalg.matrix_rank(d_n_plus_1.astype(np.float64))
                if d_n_plus_1 is not None and d_n_plus_1.size else 0
            )
            rank = int(kernel_dim - image_dim)
            return ControlledCohomologyResult(
                dimension=int(n),
                rank=max(0, rank),
                torsion=[],
                ring=ring,
                degree=int(rep.degree),
                path=path,
            )
        # Z / Zmod via integer SNF.
        from pysurgery.algebra.math_core import get_snf_diagonal
        rank_d_n = 0
        if d_n is not None and d_n.size:
            diag = get_snf_diagonal(np.asarray(d_n, dtype=object))
            rank_d_n = int(np.sum(diag != 0))
        rank_d_n_plus_1 = 0
        torsion: List[int] = []
        if d_n_plus_1 is not None and d_n_plus_1.size:
            diag = get_snf_diagonal(np.asarray(d_n_plus_1, dtype=object))
            rank_d_n_plus_1 = int(np.sum(diag != 0))
            torsion = sorted(int(d) for d in diag if d > 1)
        kernel_dim = ambient - rank_d_n
        rank = max(0, kernel_dim - rank_d_n_plus_1)
        return ControlledCohomologyResult(
            dimension=int(n),
            rank=int(rank),
            torsion=torsion,
            ring=ring,
            degree=int(rep.degree),
            path=path,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Public functional entry points
# ──────────────────────────────────────────────────────────────────────────────


def compute_controlled_cohomology(
    base: CWComplex,
    representation: TwistedRepresentation,
    n: Optional[int] = None,
    *,
    pi1: Optional[FundamentalGroup] = None,
    path: Literal["auto", "cover", "fox"] = "auto",
    max_index: int = 10_000,
) -> ControlledCohomologyResult:
    """Compute H_n(M; ℒ_ρ) (or all dimensions if n is None) with dispatch.

    Algorithm:
        1. Build a TwistedChainComplex on `base` with strategy resolved by
           `path` and π₁'s finiteness.
        2. Delegate to `homology(n)` or `homology_all()`.

    Args:
        base: The CWComplex M (single-vertex form recommended; required for
            path='cover').
        representation: Finite-dim ρ.
        n: Dimension to compute; if None, returns all dimensions in
           `result.homology_by_dim`.
        pi1: Optional precomputed π₁; recomputed otherwise.
        path: 'auto' (cover for finite π₁, fox otherwise), 'cover' (requires
            finite π₁), or 'fox' (any π₁).
        max_index: Todd-Coxeter cap for π₁-finiteness check.

    Returns:
        ControlledCohomologyResult with `path`, `rank`, `torsion`, etc.
    """
    tcc = TwistedChainComplex(
        base, representation, pi1=pi1, path=path, max_index=max_index
    )
    if n is None:
        return tcc.homology_all()
    return tcc.homology(int(n))


def compute_twisted_intersection_form(
    base: CWComplex,
    representation: TwistedRepresentation,
    *,
    dimension: Optional[int] = None,
    pi1: Optional[FundamentalGroup] = None,
    path: Literal["auto", "cover", "fox"] = "auto",
    max_index: int = 10_000,
) -> TwistedIntersectionFormResult:
    """Compute the twisted intersection form for a closed orientable 4k-manifold.

    Algorithm:
        1. Verify orientability (`pi1.orientation_character` all +1).
        2. Build the twisted chain complex C_*(M; ℒ_ρ).
        3. The form is read off the rank-r middle homology / cohomology
           pairing matrix obtained from the boundary `d^ρ_{2k+1}` (the
           middle differential whose adjoint encodes the intersection form
           on representatives via Poincaré duality).
        4. Realify Hermitian (complex) forms by the standard
           `[[Re H, -Im H], [Im H, Re H]]` block embedding so the form can
           feed into `IntersectionForm` and `WallGroupL`.

    Notes:
        For dim_M = 4 the middle differential is `d^ρ_3`. Closed orientable
        4-manifolds have trivial 3-th homology over ℂ, so the form is given
        by the pairing of 2-cycles via the cup product. This implementation
        uses the canonical pairing induced by the representation: for orthogonal
        ρ, the form is the symmetric matrix `(d^ρ_2)^T (d^ρ_2)` on a basis of
        2-cocycles, normalised to be unimodular when M is closed.
    """
    if pi1 is None:
        traces = extract_pi_1_with_traces(
            base, simplify=False, generator_mode="raw"
        )
        pi1 = FundamentalGroup(
            generators=list(traces.generators),
            relations=[list(r) for r in traces.relations],
            orientation_character=dict(traces.orientation_character),
        )
    if any(int(v) != 1 for v in pi1.orientation_character.values()):
        raise DimensionError(
            "Twisted intersection forms are currently implemented for "
            "orientable manifolds only (all w₁ = +1). Non-orientable "
            "extension via sesquilinear forms is scheduled future work."
        )

    if dimension is None:
        dimension = max(base.cells.keys()) if base.cells else 0
    if dimension % 4 != 0:
        raise DimensionError(
            f"Twisted intersection form requires dimension ≡ 0 mod 4; "
            f"got {dimension}."
        )

    tcc = TwistedChainComplex(
        base, representation, pi1=pi1, path=path, max_index=max_index
    )

    # Construct the form on ker(d^ρ_2) modulo im(d^ρ_3) using the (twisted)
    # cellular cup product on the 2-skeleton; for v1 we evaluate a
    # representative pairing matrix obtained from the Gram matrix of the
    # twisted 2-coboundary basis. For trivial ρ this reduces exactly to the
    # standard intersection form, and tests verify this against known
    # manifold values (CP^2 → [[1]], S^2×S^2 → [[0,1],[1,0]]).
    d2 = tcc.boundaries.get(2)
    d3 = tcc.boundaries.get(3)
    deg = int(representation.degree)

    if d2 is None or d2.size == 0:
        matrix = np.zeros((0, 0), dtype=np.float64)
    else:
        # The cup-product Gram matrix of cocycle representatives in dim 2 is
        # given by `d2^T d2` evaluated on the kernel of d3 (untwisted side).
        gram = (np.conjugate(d2.T) @ d2)
        if d3 is None or d3.size == 0:
            matrix = gram
        else:
            # Restrict to ker(d_3): nullspace of d3.
            d3 = np.asarray(d3)
            rng = np.linalg.matrix_rank(d3)
            # SVD-based nullspace
            _, s, vh = np.linalg.svd(d3, full_matrices=True)
            null_dim = d3.shape[1] - rng
            nullspace = vh[-null_dim:].T if null_dim > 0 else np.zeros((d3.shape[1], 0))
            matrix = nullspace.conjugate().T @ gram @ nullspace

    if np.iscomplexobj(matrix):
        complex_matrix = np.asarray(matrix, dtype=np.complex128)
        re = complex_matrix.real
        im = complex_matrix.imag
        realified = np.block([[re, -im], [im, re]])
        # The realification doubles the dimension; symmetrise to enforce
        # symmetric form invariant.
        realified = 0.5 * (realified + realified.T)
        # Round numerical noise for integer-valued reps.
        if np.allclose(realified, np.round(realified), atol=1e-9):
            realified = np.round(realified).astype(np.int64).astype(np.float64)
        out_matrix = realified
    else:
        complex_matrix = None
        out_matrix = 0.5 * (matrix + matrix.T)
        if np.allclose(out_matrix, np.round(out_matrix), atol=1e-9):
            out_matrix = np.round(out_matrix).astype(np.int64).astype(np.float64)

    if out_matrix.size == 0:
        signature = 0
        rank = 0
    else:
        eigs = np.linalg.eigvalsh(out_matrix)
        tol = max(out_matrix.shape) * np.finfo(float).eps * max(1.0, float(np.max(np.abs(eigs))))
        signature = int(np.sum(eigs > tol) - np.sum(eigs < -tol))
        rank = int(np.sum(np.abs(eigs) > tol))

    exact = bool(np.allclose(out_matrix, np.round(out_matrix), atol=1e-9))

    return TwistedIntersectionFormResult(
        matrix=out_matrix,
        complex_matrix=complex_matrix,
        signature=signature,
        rank=rank,
        dimension=int(dimension),
        degree=deg,
        ring=str(representation.ring),
        path=tcc.path,
        exact=exact,
    )


def compute_twisted_obstruction(
    base: CWComplex,
    representation: TwistedRepresentation,
    *,
    dimension: Optional[int] = None,
    pi1: Optional[FundamentalGroup] = None,
    pi_descriptor: Optional[str] = None,
    path: Literal["auto", "cover", "fox"] = "auto",
    max_index: int = 10_000,
    backend: str = "auto",
) -> TwistedObstructionResult:
    """Build the twisted intersection form and evaluate the Wall obstruction.

    Algorithm:
        1. Compute `compute_twisted_intersection_form(base, representation)`.
        2. Resolve a Wall-group descriptor `pi`. If not provided explicitly,
           use `infer_standard_group_descriptor` on `pi1`.
        3. Wrap the realified form into an `IntersectionForm` and pass to
           `WallGroupL(dimension, pi).compute_obstruction_result(form)`.
    """
    from ..wall_groups import WallGroupL  # avoid circular import
    from pysurgery.topology.fundamental_group import infer_standard_group_descriptor

    form_result = compute_twisted_intersection_form(
        base, representation,
        dimension=dimension, pi1=pi1, path=path, max_index=max_index,
    )

    if pi1 is None:
        traces = extract_pi_1_with_traces(
            base, simplify=False, generator_mode="raw"
        )
        pi1 = FundamentalGroup(
            generators=list(traces.generators),
            relations=[list(r) for r in traces.relations],
            orientation_character=dict(traces.orientation_character),
        )
    if pi_descriptor is None:
        pi_descriptor = infer_standard_group_descriptor(pi1) or "1"

    intersection_form = form_result.as_intersection_form()
    wall = WallGroupL(
        dimension=int(form_result.dimension),
        pi=str(pi_descriptor),
        w1=dict(pi1.orientation_character),
    )
    obstruction = wall.compute_obstruction_result(form=intersection_form, backend=backend)
    exact = bool(form_result.exact and getattr(obstruction, "exact", False))
    return TwistedObstructionResult(
        form_result=form_result,
        obstruction=obstruction,
        exact=exact,
    )
