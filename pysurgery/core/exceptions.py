"""Custom exceptions for surgery theory operations.

Overview:
    Provides a hierarchy of domain-specific exceptions for topological and algebraic
    surgery theory computations. These errors capture specific failure modes such
    as non-unimodularity, dimension mismatches, and surgery obstructions.

Key Concepts:
    - **SurgeryError**: The base class for all library-specific exceptions.
    - **Topological Invariants**: Errors often triggered when expected invariants
      (like unimodularity) are violated.
    - **Algebraic Obstructions**: Errors capturing non-zero elements in Wall groups (L-theory).

Common Workflows:
    1. Catch specific errors (e.g., `IsotropicError`) to handle geometry-specific failures.
    2. Use `SurgeryObstructionError` to identify when surgery cannot be completed.
    3. Monitor `DimensionError` for topologically illegal inputs.

Coefficient Ring:
    Not applicable (Exception hierarchy).

Attributes:
    None (uses standard Exception message).
"""


class SurgeryError(Exception):
    """Base class for all surgery theory related errors.

    Overview:
        The root exception for all errors occurring within the `pysurgery` library.
        Used to distinguish library-specific failures from general Python errors.

    Key Concepts:
        - Domain-specific error handling.
        - Hierarchical categorization of topological failures.

    Common Workflows:
        1. Base class for all specialized exceptions in this module.

    Coefficient Ring:
        N/A.

    Attributes:
        message (str): A human-readable error message.
    """

    pass


class IsotropicError(SurgeryError):
    """Raised when a class is not isotropic (self-intersection is non-zero).

    Overview:
        Indicates that an algebraic homology class does not vanish under the
        intersection form, violating an isotropy requirement for specific embeddings.

    Key Concepts:
        - **Isotropy**: Vanishing of the self-intersection form.

    Common Workflows:
        1. Raised during immersion-to-embedding reduction (Whitney trick pre-checks).

    Coefficient Ring:
        Usually ℤ or ℤ/2ℤ.

    Attributes:
        message (str): A human-readable error message.
    """

    pass


class NonPrimitiveError(SurgeryError):
    """Raised when a homology class is not primitive (not a basis element).

    Overview:
        Indicates that a homology class cannot be completed to a basis,
        meaning it is a multiple of another non-trivial class.

    Key Concepts:
        - **Primitivity**: Element is part of a ℤ-module basis.

    Common Workflows:
        1. Raised during surgery on cycles that are not primitive.

    Coefficient Ring:
        ℤ.

    Attributes:
        message (str): A human-readable error message.
    """

    pass


class UnimodularityError(SurgeryError):
    """Raised when a form is expected to be unimodular but is not.

    Overview:
        Indicates that an intersection or quadratic form has a determinant
        other than ±1 (over ℤ), violating Poincare duality requirements.

    Key Concepts:
        - **Unimodularity**: Map from H_n to H^n is an isomorphism.

    Common Workflows:
        1. Raised during validation of intersection forms for closed manifolds.

    Coefficient Ring:
        ℤ.

    Attributes:
        message (str): A human-readable error message.
    """

    pass


class DimensionError(SurgeryError):
    """Raised when dimensions are topologically incompatible or invalid.

    Overview:
        Indicates a mismatch in dimensions, such as attempting a surgery operation
        on a manifold of the wrong dimension or invalid simplex dimension.

    Key Concepts:
        - **Topological Dimension**: The n in n-manifold.

    Common Workflows:
        1. Raised during manifold construction or surgery obstructions.

    Coefficient Ring:
        N/A.

    Attributes:
        message (str): A human-readable error message.
    """

    pass


class NotAManifoldError(SurgeryError):
    """Raised when a complex is expected to be a manifold but fails verification.

    Overview:
        Indicates that a simplicial or CW complex does not satisfy the local 
        topology requirements of a manifold (e.g., vertex links are not spheres/disks 
        or there are branching singularities).
    """

    pass


class HomologyError(SurgeryError):
    """Raised when homology calculations fail or are inconsistent.

    Overview:
        Indicates an error in the chain complex or Smith Normal Form reduction,
        resulting in inconsistent or invalid homology groups.

    Key Concepts:
        - **Chain Complex**: Sequence of modules and boundary maps.

    Common Workflows:
        1. Raised during `homology()` calls on invalid complexes.

    Coefficient Ring:
        ℤ, ℚ, or ℤ/pℤ.

    Attributes:
        message (str): A human-readable error message.
    """

    pass


class SurgeryObstructionError(SurgeryError):
    """Raised when a non-zero Wall group obstruction prevents surgery.

    Overview:
        Captures a failure in the surgery program: the existence of a non-trivial
        element in L_n(π₁) that prevents the existence of a homotopy equivalence.

    Key Concepts:
        - **Wall Group (L_n)**: Obstruction group for surgery.

    Common Workflows:
        1. Raised when `solve_surgery_problem()` finds a non-zero L-group element.

    Coefficient Ring:
        Group rings ℤ[π₁].

    Attributes:
        message (str): A human-readable error message.
    """

    pass


class NonSymmetricError(SurgeryError):
    """Raised when an intersection form matrix is not symmetric.

    Overview:
        Indicates that an even-dimensional intersection form is not symmetric,
        violating the basic properties of the cup product on H^k.

    Key Concepts:
        - **Symmetry**: A_ij = A_ji in the intersection matrix.

    Common Workflows:
        1. Raised during `IntersectionForm` validation.

    Coefficient Ring:
        ℤ.

    Attributes:
        message (str): A human-readable error message.
    """

    pass


class GroupRingError(SurgeryError):
    """Raised when group ring operations are invalid.

    Overview:
        Indicates failure in algebraic operations within a group ring ℤ[G],
        such as invalid generator multiplication or malformed elements.

    Key Concepts:
        - **Group Ring**: Algebraic structure combining a ring and a group.

    Common Workflows:
        1. Raised during Wall group element construction.

    Coefficient Ring:
        ℤ[G].

    Attributes:
        message (str): A human-readable error message.
    """

    pass


class FundamentalGroupError(SurgeryError):
    """Raised when the fundamental group extraction fails or is ambiguous.

    Overview:
        Indicates failures in extracting π₁ from a 2-skeleton or in applying
        Tietze simplifications to a group presentation.

    Key Concepts:
        - **π₁ (Fundamental Group)**: First homotopy group.

    Common Workflows:
        1. Raised during `extract_pi_1()`.

    Coefficient Ring:
        N/A.

    Attributes:
        message (str): A human-readable error message.
    """

    pass


class KirbyMoveError(SurgeryError):
    """Raised when an invalid Kirby move is attempted (e.g., sliding handles incorrectly).

    Overview:
        Indicates a violation of the rules of Kirby calculus for 3- and 4-manifolds,
        such as an illegal handle slide or blowout.

    Key Concepts:
        - **Kirby Calculus**: Set of moves (slides, stabilizes) on framed links.

    Common Workflows:
        1. Raised during manual or automated link manipulation.

    Coefficient Ring:
        N/A.

    Attributes:
        message (str): A human-readable error message.
    """

    pass


class CharacteristicClassError(SurgeryError):
    """Raised when Wu's formula or Steenrod squares fail on non-mod-2 cohomology.

    Overview:
        Indicates that a characteristic class computation (like Stiefel-Whitney)
        was attempted on an incompatible cohomology theory or ring.

    Key Concepts:
        - **Characteristic Classes**: Topological invariants of vector bundles.

    Common Workflows:
        1. Raised during SW class extraction from non-Z2 complexes.

    Coefficient Ring:
        ℤ/2ℤ (usually).

    Attributes:
        message (str): A human-readable error message.
    """

    pass


class StructureSetError(SurgeryError):
    """Raised when the Surgery Exact Sequence cannot be resolved due to missing L-group data.

    Overview:
        Indicates failure in resolving the structure set S(M), typically when
        the required Wall groups or assembly maps are not computable.

    Key Concepts:
        - **Structure Set**: Set of homotopy equivalent manifolds.

    Common Workflows:
        1. Raised during `StructureSet.resolve()` pipeline.

    Coefficient Ring:
        L-theory modules.

    Attributes:
        message (str): A human-readable error message.
    """

    pass


class MathError(SurgeryError):
    """Raised when general mathematical or geometric consistency checks fail.

    Overview:
        A general-purpose error for mathematical inconsistencies that do not
        fit more specific categories like `HomologyError` or `DimensionError`.

    Key Concepts:
        - Mathematical consistency.

    Common Workflows:
        1. Raised during internal assertions in algebraic surgery engines.

    Coefficient Ring:
        Any.

    Attributes:
        message (str): A human-readable error message.
    """

    pass


# ── Handle Surgery Exception Hierarchy ───────────────────────────────────────


class HandleSurgeryError(SurgeryError):
    """Base class for handle surgery failures on simplicial complexes.

    Mathematical condition:
        A handle surgery operation φ: S^{k−1} × D^{n−k} → ∂M failed at some
        stage (search, validation, attachment, postcondition). Reserved as base
        for the family; leaf subtypes carry structured diagnostic data.

    Attributes:
        complex_signature (str): Hash of the offending SimplicialComplex.simplices.
        index_k (int | None): Handle index k, if known.
        stage (str): Pipeline stage where failure occurred.
        complex_info (dict | None): Optional structured diagnostic payload.
    """

    def __init__(
        self,
        message: str,
        complex_signature: str = "",
        index_k: int | None = None,
        stage: str = "search",
        complex_info: dict | None = None,
    ):
        super().__init__(message)
        self.complex_signature = complex_signature
        self.index_k = index_k
        self.stage = stage
        self.complex_info = complex_info or {}


class AttachmentSphereError(HandleSurgeryError):
    """Raised when a candidate attaching sphere cannot be found or certified.

    Mathematical condition:
        A candidate attaching map φ: S^{k−1} × D^{n−k} → K is invalid — at
        least one of the following holds:
          - the (k−1)-subcomplex σ ⊂ K is not PL-homeomorphic to S^{k−1}
          - σ is not embedded (inclusion σ ↪ K is not injective on stars)
          - the normal bundle ν(σ ⊂ K) is not trivial (no framing exists)
          - exact enumeration exceeded the configured budget without a witness
          - no candidate with nontrivial linking exists in homology

    Attributes:
        reason (str): One of "not_a_sphere", "not_embedded", "not_framed",
            "exact_search_budget_exceeded", "no_candidate_in_homology".
        candidate_simplices (list | None): The offending subcomplex, if identified.
        recognition_certificate (dict | None): Diagnostic from PL sphere recognition.
        complex_info (dict | None): Optional structured diagnostic payload.
    """

    def __init__(
        self,
        message: str = "",
        reason: str = "not_a_sphere",
        complex_signature: str = "",
        index_k: int | None = None,
        stage: str = "search",
        candidate_simplices: list | None = None,
        recognition_certificate: dict | None = None,
        complex_info: dict | None = None,
    ):
        super().__init__(
            message or f"Attachment sphere search failed: {reason}",
            complex_signature=complex_signature,
            index_k=index_k,
            stage=stage,
            complex_info=complex_info,
        )
        self.reason = reason
        self.candidate_simplices = candidate_simplices
        self.recognition_certificate = recognition_certificate or {}


class LinkingComputationError(HandleSurgeryError):
    """Raised when exact computation of lk(K_a, K_b) over ℤ fails.

    Mathematical condition:
        Exhaustively, one of:
          - K_a or K_b is not a cycle (∂K_a ≠ 0 in C_*(K))
          - K_a ∩ K_b ≠ ∅ (not disjoint)
          - dim K_a + dim K_b ≠ n − 1 (Lefschetz pairing dimension mismatch)
          - [K_b] ≠ 0 in H_q(K) — no Seifert chain F with ∂F = K_b exists
          - SNF solvability check failed

    Attributes:
        dim_a (int): Dimension of K_a.
        dim_b (int): Dimension of K_b.
        ambient_dim (int): Ambient dimension n.
        reason (str): One of "not_disjoint", "not_a_cycle_a", "not_a_cycle_b",
            "dim_mismatch", "kb_not_null_homologous", "snf_not_solvable".
        coefficient_ring (str): Ring used in computation.
        complex_info (dict | None): Optional structured diagnostic payload.
    """

    def __init__(
        self,
        message: str = "",
        reason: str = "snf_not_solvable",
        dim_a: int = 0,
        dim_b: int = 0,
        ambient_dim: int = 0,
        coefficient_ring: str = "Z",
        complex_signature: str = "",
        complex_info: dict | None = None,
    ):
        super().__init__(
            message or f"Linking computation failed: {reason}",
            complex_signature=complex_signature,
            stage="search",
            complex_info=complex_info,
        )
        self.reason = reason
        self.dim_a = dim_a
        self.dim_b = dim_b
        self.ambient_dim = ambient_dim
        self.coefficient_ring = coefficient_ring


class DelinkingImpossibleError(HandleSurgeryError):
    """Raised when the delinking iteration certifies impossibility.

    Mathematical condition:
        The delinking iteration cannot reach lk = 0 by index-1 handle surgery
        on K_a within the search universe. Certifies that every candidate
        attaching sphere for index-1 surgery on K_a has vanishing linking
        with K_b — stronger than a budget exhaustion.

    Attributes:
        final_linking (int): Linking value at termination (≠ 0).
        surgeries_performed (int): Number of surgeries executed.
        theoretical_minimum (int): |initial_lk|, the unlinking lower bound.
        reason (str): One of "no_useful_sphere", "homology_exhausted".
        complex_info (dict | None): Optional structured diagnostic payload.
    """

    def __init__(
        self,
        message: str = "",
        final_linking: int = 0,
        surgeries_performed: int = 0,
        theoretical_minimum: int = 0,
        reason: str = "no_useful_sphere",
        complex_signature: str = "",
        complex_info: dict | None = None,
    ):
        super().__init__(
            message or f"Delinking impossible: {reason}",
            complex_signature=complex_signature,
            stage="iterate",
            complex_info=complex_info,
        )
        self.final_linking = final_linking
        self.surgeries_performed = surgeries_performed
        self.theoretical_minimum = theoretical_minimum
        self.reason = reason


class SurgeryPostconditionError(HandleSurgeryError):
    """Raised when Mayer–Vietoris postcondition fails after a handle attachment.

    Mathematical condition:
        After performing a handle attachment, the Mayer–Vietoris-predicted
        Betti changes were not observed. Specifically, at least one of:
          - β_{k−1}(K'') ∉ {β_{k−1}(K) − 1, β_{k−1}(K) + 1}
          - β_k(K'') ∉ {β_k(K) − 1, β_k(K) + 1}
          - β_j(K'') ≠ β_j(K) for some j ∉ {k − 1, k}

    Attributes:
        betti_before (dict[int, int]): Betti numbers of K.
        betti_after (dict[int, int]): Betti numbers of K''.
        expected_delta (dict[int, set[int]]): Per-dimension predicted Δβ values.
        observed_delta (dict[int, int]): Actual Δβ values observed.
        torsion_before (dict[int, list[int]]): Torsion coefficients of H_j(K; ℤ).
        torsion_after (dict[int, list[int]]): Torsion coefficients of H_j(K''; ℤ).
        complex_info (dict | None): Optional structured diagnostic payload.
    """

    def __init__(
        self,
        message: str = "",
        betti_before: dict | None = None,
        betti_after: dict | None = None,
        expected_delta: dict | None = None,
        observed_delta: dict | None = None,
        torsion_before: dict | None = None,
        torsion_after: dict | None = None,
        complex_signature: str = "",
        index_k: int | None = None,
        complex_info: dict | None = None,
    ):
        super().__init__(
            message or "Surgery Mayer–Vietoris postcondition failed",
            complex_signature=complex_signature,
            index_k=index_k,
            stage="postcondition",
            complex_info=complex_info,
        )
        self.betti_before = betti_before or {}
        self.betti_after = betti_after or {}
        self.expected_delta = expected_delta or {}
        self.observed_delta = observed_delta or {}
        self.torsion_before = torsion_before or {}
        self.torsion_after = torsion_after or {}


class SurgeryInvariantBroken(SurgeryError):
    """A surgery step's post-state would violate a session invariant; rolled back."""
    pass


class SurgeryProtocolError(SurgeryError):
    """The atomic protocol detected an internal inconsistency; rolled back.

    For example, a method body forgot to call txn.commit(), so the session
    was rolled back.
    """
    pass


class TopologyNotRestoredError(SurgeryError):
    """A topology-preserving macro did not restore the pre-state invariants.

    A cancelling-pair (or other topology-preserving) macro failed to restore
    the pre-state Betti / π₁ / manifold invariants. Inner steps already rolled
    back.
    """
    pass


class LadderProgressError(SurgeryError):
    """Raised when the connectedness preconditions for homology-killing fail."""
    pass


class BettiTrackingError(SurgeryError):
    """Raised when Betti tracking queries fail on missing or invalid objects."""
    pass



