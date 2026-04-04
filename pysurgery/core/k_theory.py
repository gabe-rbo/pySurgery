from pydantic import BaseModel, ConfigDict
from .fundamental_group import FundamentalGroup
import math

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
    """
    if not pi1.generators:
        return WhiteheadGroup(rank=0, description="Wh(1) = 0. The s-Cobordism theorem has no obstruction.")
        
    if not pi1.relations:
        return WhiteheadGroup(rank=0, description=f"Wh(Free({len(pi1.generators)})) = 0 by Farrell-Jones. No s-Cobordism obstruction.")
        
    if len(pi1.generators) == 1 and len(pi1.relations) == 1:
        n = len(pi1.relations[0])
        if n > 0:
            divisors = sum(1 for i in range(1, n + 1) if n % i == 0)
            rank = math.floor(n / 2.0) + 1 - divisors
            if rank < 0: rank = 0
            if rank == 0:
                return WhiteheadGroup(rank=0, description=f"Wh(C_{n}) = 0. No s-Cobordism obstruction.")
            else:
                return WhiteheadGroup(rank=rank, description=f"Wh(C_{n}) is free abelian of rank {rank}. Torsion obstruction may exist for s-Cobordism.")
                
    return WhiteheadGroup(rank=-1, description="Wh(pi_1) computation requires advanced K-theory. Potential s-Cobordism obstruction.")
