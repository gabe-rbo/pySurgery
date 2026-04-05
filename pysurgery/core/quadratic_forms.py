import numpy as np
from pydantic import ConfigDict
from typing import List
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
        
        # Algorithmic Symplectic Gram-Schmidt over GF(2)
        M = self.matrix % 2
        q_vals = np.array(self.q_refinement) % 2
        
        basis = np.eye(n, dtype=int)
        active_indices = list(range(n))
        arf = 0
        
        while len(active_indices) >= 2:
            # Find a hyperbolic pair
            found = False
            for i_idx, i in enumerate(active_indices):
                for j_idx, j in enumerate(active_indices[i_idx+1:], start=i_idx+1):
                    val = (basis[i] @ M @ basis[j]) % 2
                    if val == 1:
                        e_idx, f_idx = i, j
                        found = True
                        break
                if found:
                    break
            
            if not found:
                break # Radical is non-empty
                
            e = basis[e_idx]
            f = basis[f_idx]
            
            # Evaluate q on the basis vectors
            def eval_q(vec):
                lin = np.sum(vec * q_vals)
                cross = 0
                for k in range(n):
                    for m_idx in range(k+1, n):
                        cross += vec[k] * vec[m_idx] * M[k, m_idx]
                return (lin + cross) % 2
                
            qe = eval_q(e)
            qf = eval_q(f)
            arf = (arf + qe * qf) % 2
            
            # Orthogonalize remaining basis
            new_active = []
            for k in active_indices:
                if k == e_idx or k == f_idx:
                    continue
                v = basis[k]
                v_dot_f = (v @ M @ f) % 2
                v_dot_e = (v @ M @ e) % 2
                basis[k] = (v - v_dot_f * e - v_dot_e * f) % 2
                new_active.append(k)
            active_indices = new_active
            
        return int(arf)
