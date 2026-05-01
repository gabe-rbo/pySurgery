from pydantic import BaseModel, ConfigDict
from .algebraic_poincare import AlgebraicPoincareComplex
from .wall_groups import WallGroupL, ObstructionResult
from .structure_set import StructureSet, SurgeryExactSequenceResult
from .core.k_theory import compute_whitehead_group
from .core.fundamental_group import extract_pi_1
from typing import Optional, Any


class AlgebraicSurgeryComplex(BaseModel):
    """Implementation of Ranicki's Algebraic Surgery complex.

    Represents an element in the algebraic structure set S^{alg}(X).

    References:
        Ranicki, A. (1980). Exact sequences in the algebraic theory of surgery. 
        Princeton University Press.

    Attributes:
        domain: The domain AlgebraicPoincareComplex.
        codomain: The codomain AlgebraicPoincareComplex.
        degree: The degree of the complex. Defaults to 1.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    domain: AlgebraicPoincareComplex
    codomain: AlgebraicPoincareComplex
    degree: int = 1

    def assembly_map_result(
        self, pi_1_group: str = "1", form: Optional[Any] = None, backend: str = "auto"
    ) -> ObstructionResult:
        """Typed assembly-map obstruction result.

        Args:
            pi_1_group: The fundamental group descriptor. Defaults to "1".
            form: Optional intersection form for evaluation.
            backend: 'auto', 'julia', or 'python'.

        Returns:
            ObstructionResult: An ObstructionResult instance.
        """
        return WallGroupL(
            dimension=self.domain.dimension, pi=pi_1_group
        ).compute_obstruction_result(form, backend=backend)

    def assembly_map(self, pi_1_group: str = "1", form: Optional[Any] = None, backend: str = "auto") -> Any:
        """Evaluates the Algebraic Assembly Map A: H_n(X; L_0) -> L_n(pi_1(X)).

        This evaluates the surgery obstruction of the normal map underlying this complex
        by projecting local geometric data (the intersection forms of the sub-manifolds)
        directly to the global fundamental group ring Z[pi_1].

        Args:
            pi_1_group: The fundamental group descriptor. Defaults to "1".
            form: Optional intersection form for evaluation.
            backend: 'auto', 'julia', or 'python'.

        Returns:
            Any: The obstruction value (int) or a diagnostic message (str).
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
        """Typed structure-set exact-sequence evaluation for this surgery context.

        Args:
            chain_complex: The chain complex of the manifold.
            fundamental_group: The fundamental group descriptor. Defaults to "1".
            backend: 'auto', 'julia', or 'python'.

        Returns:
            SurgeryExactSequenceResult: A SurgeryExactSequenceResult instance.
        """
        ss = StructureSet(
            dimension=self.domain.dimension, fundamental_group=fundamental_group
        )
        normal = ss.compute_normal_invariants_result(chain_complex, backend=backend)
        return ss.evaluate_exact_sequence_result(normal_invariants=normal, backend=backend)

    def s_cobordism_torsion(self, cw_complex: Any, backend: str = "auto") -> str:
        """Computes the Whitehead torsion Wh(pi_1) for the s-cobordism theorem classification.

        This bridges directly to the K-Theory engine and fundamental group extractor.

        Args:
            cw_complex: The CW complex to extract the fundamental group from.
            backend: 'auto', 'julia', or 'python'.

        Returns:
            str: A string describing the Whitehead torsion evaluation.
        """
        try:
            pi_1 = extract_pi_1(cw_complex, backend=backend)
            wh_group = compute_whitehead_group(pi_1, backend=backend)
            return f"Whitehead Torsion Evaluation: {wh_group.description}"
        except Exception as e:
            return f"Torsion computation failed: {e!r}. Unable to clear s-cobordism obstruction."
