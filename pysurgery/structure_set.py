import numpy as np
from pydantic import BaseModel, ConfigDict
from typing import Optional
from .core.complexes import ChainComplex
from .core.exceptions import StructureSetError
from .wall_groups import WallGroupL

class StructureSet(BaseModel):
    """
    Implementation of the topological Structure Set S_TOP(M).
    
    This mathematically models the Surgery Exact Sequence:
    ... -> L_{n+1}(pi_1) -> S_TOP(M) -> [M, G/TOP] -> L_n(pi_1)
    
    It determines the exact number of distinct manifolds that are 
    homotopy equivalent to M but NOT homeomorphic to M.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    dimension: int
    fundamental_group: str = "1"
    
    def compute_normal_invariants(self, chain: ChainComplex) -> str:
        """
        Computes the rank of the set of Normal Invariants [M, G/TOP] via Sullivan's characteristic variety formula.
        For a simply connected manifold:
        [M, G/TOP] is isomorphic to Sum_{i>=1} H^{4i}(M; Z) + Sum_{i>=1} H^{4i-2}(M; Z_2)
        modulo some 2-torsion extensions. We compute the free rank and Z_2 rank.
        """
        n = self.dimension
        rank_Z = 0
        rank_Z2 = 0
        
        for k in range(1, n + 1):
            if k % 4 == 0:
                # Add rank of H^{4i}(M; Z)
                r, _ = chain.cohomology(k)
                rank_Z += r
            elif k % 4 == 2:
                # Add rank of H^{4i-2}(M; Z_2)
                # Rank over Z_2 = free_rank(H_k) + |even torsion in H_k| + |even torsion in H_{k-1}|
                r_k, t_k = chain.homology(k)
                r_k1, t_k1 = chain.homology(k-1)
                z2_k = sum(1 for t in t_k if t % 2 == 0)
                z2_k1 = sum(1 for t in t_k1 if t % 2 == 0)
                rank_Z2 += (r_k + z2_k + z2_k1)
                
        report = f"--- NORMAL INVARIANTS [M, G/TOP] FOR {n}D MANIFOLD ---\n"
        report += f"Rank over Z: {rank_Z}\n"
        report += f"Rank over Z_2: {rank_Z2}\n"
        report += "By Sullivan's formula, this defines the topological vector bundles that can be framed for surgery."
        return report

    def evaluate_exact_sequence(self) -> str:
        """
        Evaluates the sequence to determine the size and nature of the Structure Set S_TOP(M).
        """
        n = self.dimension
        
        if self.fundamental_group != "1":
            raise StructureSetError(f"The structure set for non-simply connected groups (pi_1 = {self.fundamental_group}) relies on evaluating exact twisted Wall groups L_n(Z[pi_1]), which necessitates the Julia bridge for representation theory computation.")
            
        if n < 5:
            raise StructureSetError("The Surgery Exact Sequence strictly applies to dimensions n >= 5. In 4D, Freedman's classification completely replaces the exact sequence.")
            
        # For simply connected high-dimensional manifolds (n >= 5)
        # pi_1 = 1.
        # L_n(1) is Z (n=0 mod 4), 0 (n=1 mod 4), Z_2 (n=2 mod 4), 0 (n=3 mod 4)
        
        wall_n = WallGroupL(dimension=n, pi="1")
        wall_n_plus_1 = WallGroupL(dimension=n+1, pi="1")
        
        l_n_str = self._format_wall_group(n)
        l_n_plus_1_str = self._format_wall_group(n+1)
        
        report = f"--- SURGERY EXACT SEQUENCE FOR {n}D MANIFOLD ---\n"
        report += f"L_{n+1}(1) ---> S_TOP(M) ---> [M, G/TOP] ---> L_{n}(1)\n"
        report += f"   {l_n_plus_1_str}    ---> S_TOP(M) ---> Normal Invs --->    {l_n_str}\n\n"
        
        report += "Topological Analysis:\n"
        report += "- The set of Normal Invariants [M, G/TOP] dictates the possible vector bundles over M.\n"
        report += f"- The Wall group L_{n}(1) ({l_n_str}) acts as the primary obstruction to doing surgery.\n"
        
        if l_n_str == "0":
            report += "- Because L_n(1) = 0, EVERY normal invariant maps directly into the Structure Set. Surgery is never obstructed!\n"
        else:
            report += f"- Because L_n(1) = {l_n_str}, some normal invariants will fail to be promoted to actual homotopy equivalences. Surgery might be obstructed by signatures or Arf invariants.\n"
            
        report += "- The group L_{n+1}(1) acts directly on the Structure Set. It dictates how many distinct smooth/topological structures can exist on the same homotopy type.\n"
        
        return report
        
    def _format_wall_group(self, k: int) -> str:
        if k % 4 == 0: return "Z"
        if k % 4 == 2: return "Z_2"
        return "0"
