import numpy as np
from pydantic import BaseModel, ConfigDict
from typing import Optional, List
from .intersection_forms import IntersectionForm
from .exceptions import DimensionError

class QuadraticForm(IntersectionForm):
    """
    A quadratic form on an abelian group, which is a refinement of a symmetric bilinear form.
    Specifically, this models the Z/2Z refinements required for L_{4k+2} surgery obstructions 
    and the computation of the Arf invariant.
    
    Attributes
    ----------
    q_refinement : List[int]
        The quadratic mapping q: H -> Z_2 evaluated on the basis elements.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    q_refinement: List[int]

    def arf_invariant(self) -> int:
        """
        Compute the Arf invariant of the quadratic form.
        For a symplectic basis (e_i, f_i) where q(e_i)=a_i and q(f_i)=b_i,
        Arf(q) = sum(a_i * b_i) mod 2.
        
        This assumes the underlying intersection form matrix represents a symplectic basis.
        """
        n = self.rank()
        if n % 2 != 0:
            raise DimensionError(f"The Arf invariant requires a symplectic basis (e_i, f_i), implying an even rank. "
                                 f"The provided quadratic form has odd rank {n}.")
        
        # In a real implementation, we would first algorithmically construct a symplectic basis.
        # For simplicity, we assume the basis is already (e1, f1, e2, f2, ...).
        arf = 0
        for i in range(0, n, 2):
            arf += self.q_refinement[i] * self.q_refinement[i+1]
        
        return arf % 2
