from typing import List, Optional, Union
from pydantic import BaseModel, Field
from .core.intersection_forms import IntersectionForm
from .core.quadratic_forms import QuadraticForm
from .core.fundamental_group import GroupPresentation
from .core.exceptions import SurgeryObstructionError
from .bridge.julia_bridge import julia_engine


class ObstructionResult(BaseModel):
    """Typed result for Wall obstruction evaluations."""

    dimension: int
    pi: str
    computable: bool
    exact: bool
    value: Optional[int] = None
    modulus: Optional[int] = None
    message: str = ""
    assumptions: List[str] = Field(default_factory=list)

    def legacy_output(self) -> Union[int, str]:
        if self.computable and self.value is not None:
            return int(self.value)
        return self.message

class WallGroupL(BaseModel):
    """
    Interface for computing Wall's surgery obstruction groups L_n(pi).
    Extends beyond the simply-connected case into finite groups and Z.
    """
    
    dimension: int
    pi: Union[str, GroupPresentation] = "1"

    @staticmethod
    def _normalize_pi(pi: Union[str, GroupPresentation]) -> str:
        if isinstance(pi, GroupPresentation):
            return pi.normalized()
        return str(pi)

    @staticmethod
    def _signature_over_8(form: IntersectionForm, context: str) -> int:
        sig = form.signature()
        if sig % 8 != 0:
            raise SurgeryObstructionError(
                f"{context}: signature {sig} is not divisible by 8, so the obstruction is not integral in this model."
            )
        return sig // 8

    def compute_obstruction(self, form: Optional[IntersectionForm] = None) -> Union[int, str]:
        """
        Backward-compatible output (int or diagnostic string).
        """
        return self.compute_obstruction_result(form).legacy_output()

    def _compute_for_single_factor(self, n: int, pi: str, form: Optional[IntersectionForm]) -> ObstructionResult:
        assumptions: List[str] = []
        if pi == "1":
            assumptions.append("Classical simply-connected Wall L-group model")
            if n % 4 == 0:
                if form is None:
                    raise SurgeryObstructionError("Intersection form required to compute Wall group L_{4k}(1) obstruction. "
                                                  "Hint: Provide an IntersectionForm instance derived from the manifold's H_{2k} basis.")
                return ObstructionResult(
                    dimension=n,
                    pi=pi,
                    computable=True,
                    exact=True,
                    value=self._signature_over_8(form, "L_{4k}(1)"),
                    modulus=None,
                    assumptions=assumptions,
                )
            if n % 4 == 2:
                if form is None or not isinstance(form, QuadraticForm):
                    raise SurgeryObstructionError("L_{4k+2}(1) obstruction requires the Arf Invariant. "
                                                  "Hint: Symmetric bilinear forms are insufficient. Provide a QuadraticForm with explicit Z_2 quadratic refinement q: H -> Z_2.")
                return ObstructionResult(
                    dimension=n,
                    pi=pi,
                    computable=True,
                    exact=True,
                    value=form.arf_invariant(),
                    modulus=2,
                    assumptions=assumptions,
                )
            return ObstructionResult(
                dimension=n,
                pi=pi,
                computable=True,
                exact=True,
                value=0,
                modulus=None,
                assumptions=assumptions,
            )

        if pi == "Z":
            assumptions.append("Shaneson splitting model for L_n(Z)")
            if n % 4 == 0:
                return ObstructionResult(
                    dimension=n,
                    pi=pi,
                    computable=True,
                    exact=True,
                    value=self._signature_over_8(form, "L_{4k}(Z)") if form else 0,
                    assumptions=assumptions,
                )
            if n % 4 == 1:
                return ObstructionResult(
                    dimension=n,
                    pi=pi,
                    computable=True,
                    exact=True,
                    value=self._signature_over_8(form, "L_{4k+1}(Z)") if form else 0,
                    assumptions=assumptions,
                )
            if n % 4 == 2:
                if form and isinstance(form, QuadraticForm):
                    return ObstructionResult(
                        dimension=n,
                        pi=pi,
                        computable=True,
                        exact=True,
                        value=form.arf_invariant(),
                        modulus=2,
                        assumptions=assumptions,
                    )
                return ObstructionResult(
                    dimension=n,
                    pi=pi,
                    computable=False,
                    exact=False,
                    message="Requires Arf invariant form input",
                    assumptions=assumptions,
                )
            if n % 4 == 3:
                if form and isinstance(form, QuadraticForm):
                    return ObstructionResult(
                        dimension=n,
                        pi=pi,
                        computable=True,
                        exact=True,
                        value=form.arf_invariant(),
                        modulus=2,
                        assumptions=assumptions,
                    )
                return ObstructionResult(
                    dimension=n,
                    pi=pi,
                    computable=False,
                    exact=False,
                    message="Requires Arf invariant form input",
                    assumptions=assumptions,
                )

        if pi.startswith("Z_"):
            p = int(pi.split("_")[1])
            assumptions.append("Finite cyclic group ring obstruction via multisignature")
            if n % 4 != 0:
                return ObstructionResult(
                    dimension=n,
                    pi=pi,
                    computable=False,
                    exact=False,
                    message=f"L_{n}(Z_{p}) needs representation-theoretic data beyond current API",
                    assumptions=assumptions,
                )
            if form is None:
                raise SurgeryObstructionError(f"Intersection form required to compute Wall group L_{{{n}}}(Z_{p}) multisignature obstruction.")
            if julia_engine.available:
                return ObstructionResult(
                    dimension=n,
                    pi=pi,
                    computable=True,
                    exact=True,
                    value=int(julia_engine.compute_multisignature(form.matrix, p)),
                    assumptions=assumptions,
                )
            return ObstructionResult(
                dimension=n,
                pi=pi,
                computable=False,
                exact=False,
                message="JuliaBridge backend unavailable for exact multisignature",
                assumptions=assumptions,
            )

        return ObstructionResult(
            dimension=n,
            pi=pi,
            computable=False,
            exact=False,
            message=f"Unsupported group descriptor '{pi}' for current Wall obstruction API",
            assumptions=["Group-ring representation data not fully specified"],
        )

    def compute_obstruction_result(self, form: Optional[IntersectionForm] = None) -> ObstructionResult:
        """Compute surgery obstruction as a typed result with exactness metadata."""
        n = self.dimension
        pi = self._normalize_pi(self.pi)
        if "x" not in pi.lower():
            return self._compute_for_single_factor(n, pi, form)

        factors = [f.strip() for f in pi.split("x") if f.strip()]
        nontrivial = [f for f in factors if f != "1"]
        if len(nontrivial) == 0:
            return self._compute_for_single_factor(n, "1", form)
        if len(nontrivial) == 1:
            return self._compute_for_single_factor(n, nontrivial[0], form)
        return ObstructionResult(
            dimension=n,
            pi=pi,
            computable=False,
            exact=False,
            message=(
                f"Product group '{pi}' needs a full factor-wise L-theory decomposition; "
                "current API only reduces products with at most one nontrivial factor."
            ),
            assumptions=["No complete product-group assembly map implemented yet"],
        )

def l_group_symbol(n: int, pi: Union[str, GroupPresentation] = "1") -> str:
    """
    Returns the mathematical symbol/structure of L_n(pi).
    """
    if isinstance(pi, GroupPresentation):
        pi = pi.normalized()

    if pi == "1":
        if n % 4 == 0:
            return "Z"
        if n % 4 == 2:
            return "Z_2"
        return "0"
    elif pi == "Z":
        # By Shaneson splitting: L_n(Z) = L_n(1) + L_{n-1}(1)
        if n % 4 == 0:
            return "Z"
        if n % 4 == 1:
            return "Z"
        if n % 4 == 2:
            return "Z_2"
        if n % 4 == 3:
            return "Z_2"
    if "x" in str(pi).lower():
        return f"L_{n}({pi}) (product-group decomposition required)"
    return f"L_{n}({pi})"
