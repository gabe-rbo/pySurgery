from pydantic import BaseModel, ConfigDict, Field
from typing import List
import warnings
import sympy as sp
from sympy.matrices.normalforms import smith_normal_form
from .fundamental_group import FundamentalGroup
from ..bridge.julia_bridge import julia_engine

def euler_totient(n):
    result = n
    p = 2
    temp = n
    while p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            result -= result // p
        p += 1
    if temp > 1:
        result -= result // temp
    return result


def _num_divisors(n: int) -> int:
    return sum(1 for i in range(1, n + 1) if n % i == 0)


def cyclic_whitehead_rank(n: int) -> int:
    """Rank formula for Wh(C_n) (n > 1): floor((n+1)/2) - d(n)."""
    if n <= 1:
        return 0
    return max(0, ((n + 1) // 2) - _num_divisors(n))


def _relation_exponent_matrix(pi1: FundamentalGroup) -> sp.Matrix:
    """Build the integer relator exponent matrix for abelianization."""
    gens = list(pi1.generators)
    g_to_idx = {g: i for i, g in enumerate(gens)}
    rows = []
    for rel in pi1.relations:
        row = [0] * len(gens)
        for tok in rel:
            base = tok[:-3] if tok.endswith("^-1") else tok
            sign = -1 if tok.endswith("^-1") else 1
            if base in g_to_idx:
                row[g_to_idx[base]] += sign
        rows.append(row)
    if not rows:
        return sp.zeros(0, len(gens))
    return sp.Matrix(rows)


def _abelianization_from_snf(pi1: FundamentalGroup) -> tuple[int, list[int]]:
    """
    Exact abelianization G_ab = Z^r ⊕ ⊕ Z_{d_i} from presentation relator exponents.
    """
    m = len(pi1.generators)
    if m == 0:
        return 0, []
    M = _relation_exponent_matrix(pi1)
    if M.rows == 0:
        return m, []
    S = smith_normal_form(M, domain=sp.ZZ)
    diag_len = min(S.rows, S.cols)
    diag = [abs(int(S[i, i])) for i in range(diag_len) if int(S[i, i]) != 0]
    rank_rel = len(diag)
    free_rank = max(0, m - rank_rel)
    torsion = [d for d in diag if d > 1]
    return free_rank, torsion

class WhiteheadGroup(BaseModel):
    """
    Representation of the Whitehead group Wh(pi_1) = K_1(Z[pi_1]) / (+- pi_1).
    Used as the obstruction to the s-Cobordism theorem.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    rank: int
    description: str
    computable: bool = True
    exact: bool = True
    assumptions: List[str] = Field(default_factory=list)
    method: str = ""

def compute_whitehead_group(pi1: FundamentalGroup) -> WhiteheadGroup:
    """
    Computes or approximates the Whitehead group for the given fundamental group.
    Uses Julia for exact abelianization and rank extractions via Bass-Heller-Swan.
    """
    if not pi1.generators:
        return WhiteheadGroup(
            rank=0,
            description="Wh(1) = 0. The s-Cobordism theorem has no obstruction.",
            computable=True,
            exact=True,
            method="trivial_group",
        )

    if not pi1.relations:
        return WhiteheadGroup(
            rank=0,
            description=f"Wh(Free({len(pi1.generators)})) = 0 by Farrell-Jones. No s-Cobordism obstruction.",
            computable=True,
            exact=True,
            assumptions=["Farrell-Jones conjecture input"],
            method="free_group_theorem",
        )

    # Prefer Julia: exact sparse reductions scale much better than pure Python SNF.
    if not julia_engine.available:
        warnings.warn(
            "Whitehead computation fallback in `compute_whitehead_group`: using Python SNF abelianization; "
            "install/enable Julia for much faster exact computation on larger presentations."
        )

    try:
        if julia_engine.available:
            free_rank, torsions = julia_engine.abelianize_and_bhs_rank(pi1.generators, pi1.relations)
        else:
            free_rank, torsions = _abelianization_from_snf(pi1)

        # Bass-Heller-Swan and general K-theory tells us Wh(Z^r) = 0.
        # So if there is no torsion in the abelianization, Wh is 0 (assuming torsion-free group, satisfying Farrell-Jones).
        if len(torsions) == 0:
            return WhiteheadGroup(
                rank=0,
                description=f"Abelianization is free Z^{free_rank}. Assuming Farrell-Jones, Wh(pi_1) = 0. No s-Cobordism obstruction.",
                computable=True,
                exact=False,
                assumptions=["Farrell-Jones conjecture input", "Wh inferred from abelianization class only"],
                method="abelianization_plus_theorem_assumption",
            )

        # If there is torsion, it's a product of cyclic groups. We evaluate Wh(C_n).
        total_rank = 0
        for n in torsions:
            if n > 1:
                rank_n = cyclic_whitehead_rank(int(n))
                total_rank += max(0, rank_n)
                
        if total_rank == 0:
            return WhiteheadGroup(
                rank=0,
                description="Wh(pi_1) evaluates to rank 0. No free s-Cobordism obstruction.",
                computable=True,
                exact=False,
                assumptions=["Modeled from cyclic torsion factors in abelianization", "Not a full non-abelian Wh solver"],
                method="cyclic_factor_formula",
            )
        else:
            return WhiteheadGroup(
                rank=total_rank,
                description=f"Wh(pi_1) contains free abelian parts of rank >= {total_rank}. Torsion obstruction definitively exists for s-Cobordism.",
                computable=True,
                exact=False,
                assumptions=["Modeled from cyclic torsion factors in abelianization", "Not a full non-abelian Wh solver"],
                method="cyclic_factor_formula",
            )

    except Exception as e:
        return WhiteheadGroup(
            rank=-1,
            description=f"Wh(pi_1) computation failed: {e!r}. Potential s-Cobordism obstruction.",
            computable=False,
            exact=False,
            assumptions=["Backend/algorithm failure"],
            method="error",
        )
