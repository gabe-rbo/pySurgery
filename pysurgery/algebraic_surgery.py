from pydantic import BaseModel, ConfigDict
from .algebraic_poincare import AlgebraicPoincareComplex
from .wall_groups import WallGroupL, ObstructionResult
from .structure_set import StructureSet, SurgeryExactSequenceResult
from .core.k_theory import compute_whitehead_group
from .core.fundamental_group import extract_pi_1
from typing import Optional, Any


class AlgebraicSurgeryComplex(BaseModel):
    """Implementation of Ranicki's Algebraic Surgery complex.

    Overview:
        An AlgebraicSurgeryComplex represents an element in the algebraic structure 
        set S^{alg}(X). It models the difference between two Poincaré complexes 
        (the domain and codomain) that are connected by a normal map, providing 
        the data necessary to compute surgery obstructions.

    Key Concepts:
        - **Algebraic Structure Set (S^{alg}(X))**: The set of algebraic Poincaré complexes 
          homotopy equivalent to X.
        - **Surgery Exact Sequence**: A long exact sequence relating manifold structure 
          sets to L-groups and normal invariants.
        - **Assembly Map**: A map A: H_n(X; L_0) → L_n(π_1(X)) that calculates the surgery obstruction.

    Common Workflows:
        1. **Construct Complex** → Pair domain and codomain Poincaré complexes.
        2. **Evaluate Obstruction** → Call assembly_map() to find the L-group element.
        3. **Classify Manifold** → Use evaluate_structure_set() to navigate the exact sequence.

    Coefficient Ring:
        Determined by the domain and codomain Poincaré complexes (typically 'Z').

    Attributes:
        domain (AlgebraicPoincareComplex): The domain algebraic Poincaré complex.
        codomain (AlgebraicPoincareComplex): The codomain algebraic Poincaré complex.
        degree (int): The degree of the complex (defaults to 1).

    References:
        Ranicki, A. (1980). Exact sequences in the algebraic theory of surgery. 
        Princeton University Press.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    domain: AlgebraicPoincareComplex
    codomain: AlgebraicPoincareComplex
    degree: int = 1

    def assembly_map_result(
        self, pi_1_group: str = "1", form: Optional[Any] = None, backend: str = "auto"
    ) -> ObstructionResult:
        """Calculate the surgery obstruction via the algebraic assembly map.

        What is Being Computed?:
            Computes the element in the L-group L_n(π_1(X)) that represents the 
            obstruction to making the underlying normal map a homotopy equivalence.

        Algorithm:
            1. Instantiate WallGroupL for the given dimension and fundamental group.
            2. Compute the obstruction result (signature, quadratic form, etc.).
            3. Return an ObstructionResult object.

        Preserved Invariants:
            - Surgery Obstruction: A homotopy invariant of the normal map.
            - Vanishing Obstruction: If the result is 0, the map is cobordant to a homotopy equivalence.

        Args:
            pi_1_group: Descriptor for the fundamental group π_1(X).
            form: Optional intersection form for direct evaluation.
            backend: 'auto', 'julia', or 'python'.

        Returns:
            ObstructionResult: The formal L-group obstruction element.
        """
        return WallGroupL(
            dimension=self.domain.dimension, pi=pi_1_group
        ).compute_obstruction_result(form, backend=backend)

    def assembly_map(self, pi_1_group: str = "1", form: Optional[Any] = None, backend: str = "auto") -> Any:
        """Evaluate the Algebraic Assembly Map A: H_n(X; L_0) -> L_n(π_1(X)).

        What is Being Computed?:
            Evaluates the surgery obstruction of the normal map underlying this complex 
            by projecting local geometric data (intersection forms) to the global 
            fundamental group ring ℤ[π_1].

        Algorithm:
            1. Call assembly_map_result() to get the full obstruction data.
            2. Extract the legacy output (integer signature or descriptor).
            3. Return the obstruction value.

        Args:
            pi_1_group: The fundamental group descriptor.
            form: Optional intersection form for evaluation.
            backend: 'auto', 'julia', or 'python'.

        Returns:
            Any: The obstruction value (int) or a diagnostic message (str).

        Use When:
            - Checking if a manifold can be surgery-transformed into another.
            - Calculating Wall's L-group obstructions for manifold classification.
        """
        return self.assembly_map_result(
            pi_1_group=pi_1_group, form=form, backend=backend
        ).legacy_output()

    def evaluate_structure_set(
        self,
        chain_complex: Any,
        fundamental_group: str = "1",
        backend: str = "auto",
    ) -> SurgeryExactSequenceResult:
        """Evaluate the surgery exact sequence for this specific manifold context.

        What is Being Computed?:
            Calculates the terms of the surgery exact sequence:
            ... → L_{n+1}(π_1) → S(X) → [X, G/Top] → L_n(π_1) → ...

        Algorithm:
            1. Instantiate a StructureSet for the given dimension and group.
            2. Compute normal invariants [X, G/Top] from the chain complex.
            3. Evaluate the maps to the L-group and compute the structure set image.

        Args:
            chain_complex: The chain complex of the manifold.
            fundamental_group: The fundamental group descriptor.
            backend: 'auto', 'julia', or 'python'.

        Returns:
            SurgeryExactSequenceResult: Data structure containing terms and maps of the sequence.
        """
        ss = StructureSet(
            dimension=self.domain.dimension, fundamental_group=fundamental_group
        )
        normal = ss.compute_normal_invariants_result(chain_complex, backend=backend)
        return ss.evaluate_exact_sequence_result(normal_invariants=normal, backend=backend)

    def s_cobordism_torsion(self, cw_complex: Any, backend: str = "auto") -> str:
        """Compute Whitehead torsion Wh(π_1) for s-cobordism classification.

        What is Being Computed?:
            Calculates the Whitehead torsion obstruction τ(f) for a homotopy 
            equivalence f: M → N. If τ(f) = 0, the cobordism is an s-cobordism.

        Algorithm:
            1. Extract the fundamental group π_1 from the CW complex.
            2. Compute the Whitehead group Wh(π_1).
            3. Evaluate the torsion of the underlying chain map.

        Preserved Invariants:
            - s-Cobordism: Vanishing torsion implies homeomorphism (in high dimensions).

        Args:
            cw_complex: The CW complex representing the manifold.
            backend: 'auto', 'julia', or 'python'.

        Returns:
            str: A description of the Whitehead torsion evaluation result.

        Use When:
            - Classifying manifolds in dimensions n ≥ 5.
            - Verifying if a homotopy equivalence is a simple homotopy equivalence.
        """
        try:
            pi_1 = extract_pi_1(cw_complex, backend=backend)
            wh_group = compute_whitehead_group(pi_1, backend=backend)
            return f"Whitehead Torsion Evaluation: {wh_group.description}"
        except Exception as e:
            return f"Torsion computation failed: {e!r}. Unable to clear s-cobordism obstruction."
