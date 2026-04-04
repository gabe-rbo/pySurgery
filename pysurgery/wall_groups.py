import numpy as np
from typing import Dict, List, Optional, Union
from pydantic import BaseModel
from .core.intersection_forms import IntersectionForm
from .core.quadratic_forms import QuadraticForm
from .core.exceptions import SurgeryObstructionError, DimensionError

class WallGroupL(BaseModel):
    """
    Interface for computing Wall's surgery obstruction groups L_n(pi).
    Extends beyond the simply-connected case into finite groups and Z.
    """
    
    dimension: int
    pi: str = "1"

    def compute_obstruction(self, form: Optional[IntersectionForm] = None) -> Union[int, str]:
        """
        Compute the surgery obstruction in L_n(pi).
        """
        n = self.dimension
        if self.pi == "1":
            if n % 4 == 0:
                if form is None:
                    raise SurgeryObstructionError("Intersection form required to compute Wall group L_{4k}(1) obstruction. "
                                                  "Hint: Provide an IntersectionForm instance derived from the manifold's H_{2k} basis.")
                # L_{4k}(1) = Z, given by Signature / 8
                return form.signature() // 8
            elif n % 4 == 2:
                if form is None or not isinstance(form, QuadraticForm):
                    raise SurgeryObstructionError("L_{4k+2}(1) obstruction requires the Arf Invariant. "
                                                  "Hint: Symmetric bilinear forms are insufficient. Provide a QuadraticForm with explicit Z_2 quadratic refinement q: H -> Z_2.")
                # L_{4k+2}(1) = Z_2, given by Arf invariant
                return form.arf_invariant()
            else:
                return 0
        elif self.pi == "Z":
            # For pi = Z, we need the Alexander polynomial or knot signature.
            return "Obstruction over Z[Z] (Requires Hermitian form over Laurent polynomials)"
        elif self.pi == "Z_2":
            return "Obstruction over Z[Z_2] (Requires multisignature evaluation)"
        else:
            return f"Unimplemented for group {self.pi}"

def l_group_symbol(n: int, pi: str = "1") -> str:
    """
    Returns the mathematical symbol/structure of L_n(pi).
    """
    if pi == "1":
        if n % 4 == 0: return "Z"
        if n % 4 == 2: return "Z_2"
        return "0"
    elif pi == "Z":
        # By Shaneson splitting: L_n(Z) = L_n(1) + L_{n-1}(1)
        if n % 4 == 0: return "Z"
        if n % 4 == 1: return "Z + Z"
        if n % 4 == 2: return "Z_2"
        if n % 4 == 3: return "Z_2"
    return f"L_{n}({pi})"
