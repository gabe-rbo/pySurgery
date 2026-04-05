from pydantic import BaseModel, ConfigDict
from .fundamental_group import FundamentalGroup
from ..bridge.julia_bridge import julia_engine

class WhiteheadGroup(BaseModel):
    """
    Representation of the Whitehead group Wh(pi_1) = K_1(Z[pi_1]) / (+- pi_1).
    Used as the obstruction to the s-Cobordism theorem.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    rank: int
    description: str

def compute_whitehead_group(pi1: FundamentalGroup) -> WhiteheadGroup:
    """
    Computes or approximates the Whitehead group for the given fundamental group.
    Uses Julia for exact abelianization and rank extractions via Bass-Heller-Swan.
    """
    if not pi1.generators:
        return WhiteheadGroup(rank=0, description="Wh(1) = 0. The s-Cobordism theorem has no obstruction.")
        
    if not pi1.relations:
        return WhiteheadGroup(rank=0, description=f"Wh(Free({len(pi1.generators)})) = 0 by Farrell-Jones. No s-Cobordism obstruction.")
        
    # Send to Julia for exact Abelianization H_1 = Z^r x Z_t1 x Z_t2...
    if not julia_engine.available:
        return WhiteheadGroup(rank=-1, description="Wh(pi_1) computation for non-trivial groups requires Julia bridge. Potential s-Cobordism obstruction.")
        
    try:
        free_rank, torsions = julia_engine.abelianize_and_bhs_rank(pi1.generators, pi1.relations)
        
        # Bass-Heller-Swan and general K-theory tells us Wh(Z^r) = 0.
        # So if there is no torsion in the abelianization, Wh is 0 (assuming torsion-free group, satisfying Farrell-Jones).
        if len(torsions) == 0:
            return WhiteheadGroup(rank=0, description=f"Abelianization is free Z^{free_rank}. Assuming Farrell-Jones, Wh(pi_1) = 0. No s-Cobordism obstruction.")
            
        # If there is torsion, it's a product of cyclic groups. We evaluate Wh(C_n).
        total_rank = 0
        for n in torsions:
            if n > 1:
                divisors = sum(1 for i in range(1, n + 1) if n % i == 0)
                rank_n = (n // 2) + 1 - divisors
                total_rank += max(0, rank_n)
                
        if total_rank == 0:
            return WhiteheadGroup(rank=0, description="Wh(pi_1) evaluates to rank 0. No free s-Cobordism obstruction.")
        else:
            return WhiteheadGroup(rank=total_rank, description=f"Wh(pi_1) contains free abelian parts of rank >= {total_rank}. Torsion obstruction definitively exists for s-Cobordism.")
            
    except Exception as e:
        return WhiteheadGroup(rank=-1, description=f"Wh(pi_1) computation failed: {e}. Potential s-Cobordism obstruction.")
