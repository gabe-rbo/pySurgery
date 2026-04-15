from pydantic import BaseModel, ConfigDict
from .algebraic_poincare import AlgebraicPoincareComplex
from .wall_groups import WallGroupL, ObstructionResult
from .structure_set import StructureSet, SurgeryExactSequenceResult
from .core.k_theory import compute_whitehead_group
from .core.fundamental_group import extract_pi_1


class AlgebraicSurgeryComplex(BaseModel):
    """
    Implementation of Ranicki's Algebraic Surgery complex.
    Represents an element in the algebraic structure set S^{alg}(X).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    domain: AlgebraicPoincareComplex
    codomain: AlgebraicPoincareComplex
    degree: int = 1

    def assembly_map_result(
        self, pi_1_group: str = "1", form=None
    ) -> ObstructionResult:
        """Typed assembly-map obstruction result."""
        return WallGroupL(
            dimension=self.domain.dimension, pi=pi_1_group
        ).compute_obstruction_result(form)

    def assembly_map(self, pi_1_group: str = "1", form=None):
        """
        Evaluates the Algebraic Assembly Map A: H_n(X; L_0) -> L_n(pi_1(X)).
        This evaluates the surgery obstruction of the normal map underlying this complex
        by projecting local geometric data (the intersection forms of the sub-manifolds)
        directly to the global fundamental group ring Z[pi_1].
        """
        return self.assembly_map_result(
            pi_1_group=pi_1_group, form=form
        ).legacy_output()

    def evaluate_structure_set(
        self,
        chain_complex,
        fundamental_group: str = "1",
    ) -> SurgeryExactSequenceResult:
        """Typed structure-set exact-sequence evaluation for this surgery context."""
        ss = StructureSet(
            dimension=self.domain.dimension, fundamental_group=fundamental_group
        )
        normal = ss.compute_normal_invariants_result(chain_complex)
        return ss.evaluate_exact_sequence_result(normal_invariants=normal)

    def s_cobordism_torsion(self, cw_complex) -> str:
        """
        Computes the Whitehead torsion Wh(pi_1) for the s-cobordism theorem classification
        by bridging directly to the K-Theory engine and fundamental group extractor.
        """
        try:
            pi_1 = extract_pi_1(cw_complex)
            wh_group = compute_whitehead_group(pi_1)
            return f"Whitehead Torsion Evaluation: {wh_group.description}"
        except Exception as e:
            return f"Torsion computation failed: {e!r}. Unable to clear s-cobordism obstruction."
