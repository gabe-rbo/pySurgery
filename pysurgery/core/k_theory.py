from pydantic import BaseModel, ConfigDict, Field
from typing import List
import warnings
import sympy as sp
from sympy.matrices.normalforms import smith_normal_form
from .fundamental_group import FundamentalGroup, infer_standard_group_descriptor
from ..bridge.julia_bridge import julia_engine


def euler_totient(n: int) -> int:
    """Compute Euler's totient function φ(n) by prime-factor stripping.

    Args:
        n (int): The integer to compute the totient for.

    Returns:
        int: The value of φ(n).
    """
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
    """Return the divisor-count function d(n) in O(sqrt(n)).

    Args:
        n (int): The integer to count divisors for.

    Returns:
        int: The number of divisors of n.
    """
    if n <= 0:
        return 0
    count = 0
    limit = int(n**0.5)
    for i in range(1, limit + 1):
        if n % i == 0:
            count += 1
            if i != n // i:
                count += 1
    return count


def cyclic_whitehead_rank(n: int) -> int:
    """Rank formula for Wh(C_n) (n > 1): floor(n/2) + 1 - d(n).

    Args:
        n (int): The order of the cyclic group.

    Returns:
        int: The rank of the Whitehead group Wh(C_n).
    """
    if n <= 1:
        return 0
    return max(0, (n // 2) + 1 - _num_divisors(n))


def _relation_exponent_matrix(pi1: FundamentalGroup) -> sp.Matrix:
    """Build the integer relator exponent matrix for abelianization.

    Args:
        pi1 (FundamentalGroup): The fundamental group.

    Returns:
        sp.Matrix: The exponent matrix.
    """
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
    """Exact abelianization G_ab = Z^r ⊕ ⊕ Z_{d_i} from presentation relator exponents.

    Args:
        pi1 (FundamentalGroup): The fundamental group.

    Returns:
        tuple[int, list[int]]: A tuple containing the free rank and a list of
            torsion coefficients.
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
    """Representation of the Whitehead group Wh(pi_1) = K_1(Z[pi_1]) / (+- pi_1).

    Used as the obstruction to the s-Cobordism theorem.

    References:
        Milnor, J. W. (1966). Whitehead torsion. 
        Bulletin of the American Mathematical Society, 72(3), 358-426.

    Attributes:
        rank (int): The free rank of the Whitehead group.
        description (str): A human-readable description of the group.
        computable (bool): Whether the group was computable. Defaults to True.
        exact (bool): Whether the computation was exact. Defaults to True.
        assumptions (List[str]): List of assumptions made during computation.
        method (str): The method used for computation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rank: int
    description: str
    computable: bool = True
    exact: bool = True
    assumptions: List[str] = Field(default_factory=list)
    method: str = ""


def compute_whitehead_group(pi1: FundamentalGroup, backend: str = "auto") -> WhiteheadGroup:
    """Computes or approximates the Whitehead group for the given fundamental group.

    References:
        Bass, H., Heller, A., & Swan, R. G. (1964). The Whitehead group of a polynomial extension. 
        Publications Mathématiques de l'IHÉS, 22, 61-79.
        
        Farrell, F. T., & Jones, L. E. (1993). Isomorphism conjectures in algebraic K-theory. 
        Journal of the American Mathematical Society, 6(2), 249-297.

    Args:
        pi1 (FundamentalGroup): The fundamental group.
        backend (str): 'auto', 'julia', or 'python'.

    Returns:
        WhiteheadGroup: The computed Whitehead group representation.
    """
    descriptor = infer_standard_group_descriptor(pi1, backend=backend)

    if descriptor == "1":
        return WhiteheadGroup(
            rank=0,
            description="Wh(1) = 0. The s-Cobordism theorem has no obstruction.",
            computable=True,
            exact=True,
            method="trivial_group",
        )

    if descriptor == "Z":
        return WhiteheadGroup(
            rank=0,
            description="Wh(Z) = 0 by Bass-Heller-Swan. No s-Cobordism obstruction.",
            computable=True,
            exact=True,
            method="infinite_cyclic_theorem",
        )

    if descriptor is not None and descriptor.startswith("Z_"):
        n = int(descriptor.split("_", 1)[1])
        rank = cyclic_whitehead_rank(n)
        if rank == 0:
            return WhiteheadGroup(
                rank=0,
                description=f"Wh(C_{n}) has rank 0 by the finite cyclic Whitehead rank formula.",
                computable=True,
                exact=True,
                method="finite_cyclic_theorem",
            )
        return WhiteheadGroup(
            rank=rank,
            description=f"Wh(C_{n}) has free abelian rank {rank} by the finite cyclic Whitehead rank formula.",
            computable=True,
            exact=True,
            method="finite_cyclic_theorem",
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

    # Normalize backend
    backend_norm = str(backend).lower().strip()
    use_julia = (backend_norm == "julia") or (backend_norm == "auto" and julia_engine.available)

    if not use_julia and backend_norm != "python":
        warnings.warn(
            "Whitehead computation fallback in `compute_whitehead_group`: using Python SNF abelianization; "
            "install/enable Julia for much faster exact computation on larger presentations."
        )

    try:
        if use_julia:
            free_rank, torsions = julia_engine.abelianize_and_bhs_rank(
                pi1.generators, pi1.relations
            )
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
                assumptions=[
                    "Farrell-Jones conjecture input",
                    "Wh inferred from abelianization class only",
                ],
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
                assumptions=[
                    "Modeled from cyclic torsion factors in abelianization",
                    "Not a full non-abelian Wh solver",
                ],
                method="cyclic_factor_formula",
            )
        else:
            return WhiteheadGroup(
                rank=total_rank,
                description=f"Wh(pi_1) contains free abelian parts of rank >= {total_rank}. Torsion obstruction definitively exists for s-Cobordism.",
                computable=True,
                exact=False,
                assumptions=[
                    "Modeled from cyclic torsion factors in abelianization",
                    "Not a full non-abelian Wh solver",
                ],
                method="cyclic_factor_formula",
            )

    except Exception as e:
        if backend_norm == "julia":
            raise e
        return WhiteheadGroup(
            rank=-1,
            description=f"Wh(pi_1) computation failed: {e!r}. Potential s-Cobordism obstruction.",
            computable=False,
            exact=False,
            assumptions=["Backend/algorithm failure"],
            method="error",
        )
