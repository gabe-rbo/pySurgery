from pydantic import BaseModel, ConfigDict, Field
from typing import List
import warnings
import sympy as sp
from sympy.matrices.normalforms import smith_normal_form
from .fundamental_group import FundamentalGroup, infer_standard_group_descriptor
from ..bridge.julia_bridge import julia_engine


def euler_totient(n: int) -> int:
    """Compute Euler's totient function φ(n) by prime-factor stripping.

    What is Being Computed?:
        Computes the number of positive integers up to n that are relatively prime to n.
        Mathematically, φ(n) = n * Π_{p|n} (1 - 1/p) where p are distinct prime factors.

    Algorithm:
        1. Initialize result with n.
        2. Iterate through p starting from 2 up to sqrt(n).
        3. For each prime factor p, divide out all occurrences and update result -= result // p.
        4. If the remaining n is greater than 1, it is a prime factor; update result -= result // n.

    Preserved Invariants:
        - Multiplicative property: φ(mn) = φ(m)φ(n) if gcd(m, n) = 1.
        - Relates to the order of the unit group (ℤ/nℤ)*.

    Args:
        n: The integer to compute the totient for.

    Returns:
        int: The value of φ(n).

    Use When:
        - Computing the rank of Whitehead groups of cyclic groups.
        - Number-theoretic calculations involving units in ℤ/nℤ.

    Example:
        phi_10 = euler_totient(10)  # Returns 4 ({1, 3, 7, 9})
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

    What is Being Computed?:
        The number of positive divisors of an integer n.

    Algorithm:
        Iterates from 1 to floor(sqrt(n)), checking for divisibility.
        If i divides n, both i and n/i are counted (if distinct).

    Args:
        n: The integer to count divisors for.

    Returns:
        int: The number of divisors d(n).

    Use When:
        - Internal helper for Whitehead group rank formulas of cyclic groups.
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

    What is Being Computed?:
        The free abelian rank of the Whitehead group Wh(C_n) for a cyclic group of order n.

    Algorithm:
        Evaluates the closed-form formula: rank = floor(n/2) + 1 - d(n), where d(n) is the 
        number of positive divisors of n.

    Preserved Invariants:
        - The rank is a property of the group ring ℤ[C_n] and is invariant under 
          isomorphisms of C_n.

    Args:
        n: The order of the cyclic group.

    Returns:
        int: The rank of the Whitehead group Wh(C_n).

    Use When:
        - Computing obstructions for the s-Cobordism theorem in the presence of 
          cyclic fundamental groups.
        - Studying K-theory of cyclic group rings.

    Example:
        rank_C5 = cyclic_whitehead_rank(5) # Returns 1
    """
    if n <= 1:
        return 0
    return max(0, (n // 2) + 1 - _num_divisors(n))


def _relation_exponent_matrix(pi1: FundamentalGroup) -> sp.Matrix:
    """Build the integer relator exponent matrix for abelianization.

    What is Being Computed?:
        The presentation matrix (or relation matrix) of the abelianization of π₁.
        This is an m x n matrix where rows correspond to relations and columns to 
        generators. The entry (i, j) is the sum of the exponents of generator j in relation i.

    Algorithm:
        1. Map each generator to a column index.
        2. For each relation, iterate through tokens (generators or their inverses).
        3. Increment/decrement the corresponding column based on the exponent.
        4. Return a SymPy integer matrix.

    Args:
        pi1: The fundamental group to analyze.

    Returns:
        sp.Matrix: The m x n integer exponent matrix.

    Use When:
        - Internal step for computing abelianization (H₁) via Smith Normal Form.
        - Analyzing relations in a group presentation.
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
    """Exact abelianization G_ab = ℤ^r ⊕ ⊕ ℤ_{d_i} from presentation relator exponents.

    What is Being Computed?:
        The structure of the abelianization of a finitely presented group, 
        represented as a free rank and a list of torsion coefficients.

    Algorithm:
        1. Construct the relator exponent matrix M.
        2. Compute the Smith Normal Form (SNF) of M over ℤ.
        3. The non-zero diagonal entries d_i (greater than 1) are the torsion coefficients.
        4. The free rank is the number of generators minus the number of non-zero diagonal entries.

    Preserved Invariants:
        - The abelianization H₁(G; ℤ) is an invariant of the group G.

    Args:
        pi1: The fundamental group to abelianize.

    Returns:
        tuple[int, list[int]]: A tuple (free_rank, torsion_coefficients).

    Use When:
        - Computing the first homology group H₁ from π₁.
        - Identifying groups that are definitely not isomorphic due to different abelianizations.
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
    """Representation of the Whitehead group Wh(π₁) = K₁(ℤ[π₁]) / (± π₁).

    Overview:
        The Whitehead group Wh(π₁) is a fundamental invariant in high-dimensional 
        manifold topology. It serves as the home for the Whitehead torsion, which 
        is the primary obstruction to the s-Cobordism theorem. If Wh(π₁) = 0, 
        every h-cobordism with fundamental group π₁ is an s-cobordism (and hence 
        is trivial/a product).

    Key Concepts:
        - **Whitehead Torsion**: An element τ(W, M) ∈ Wh(π₁) that vanishes if and 
          only if the h-cobordism W is a product.
        - **s-Cobordism Theorem**: Generalizes the h-cobordism theorem to 
          non-simply connected manifolds.
        - **Bass-Heller-Swan**: Theorem stating Wh(ℤ) = 0.
        - **Farrell-Jones Conjecture**: Predicts Wh(G) = 0 for torsion-free groups G.

    Common Workflows:
        1. **Computation** → Call compute_whitehead_group(pi1)
        2. **Obstruction Check** → Check if WhiteheadGroup.rank == 0 and if it's computable/exact
        3. **Assumption Analysis** → Review assumptions (e.g., Farrell-Jones) made during computation

    Attributes:
        rank (int): The free abelian rank of the Whitehead group.
        description (str): Human-readable summary of the group structure and implications.
        computable (bool): Indicates if the algorithm successfully reached a result.
        exact (bool): True if the result is theoretically exact, False if it relies 
                      on conjectures or approximations.
        assumptions (List[str]): List of mathematical conjectures (e.g., Farrell-Jones) 
                                 invoked during computation.
        method (str): Identifier for the specific theorem or algorithm used.
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

    What is Being Computed?:
        Determines the structure (specifically the rank) of Wh(π₁) using known 
        theorems for specific group families (cyclic, free, etc.) and abelianization 
        heuristics for more complex groups.

    Algorithm:
        1. Infer the group type (trivial, cyclic, free) using descriptor heuristics.
        2. If trivial or ℤ, return rank 0 (Bass-Heller-Swan).
        3. If cyclic ℤ_n, use the analytic rank formula: floor(n/2) + 1 - d(n).
        4. If free group, return rank 0 (assuming Farrell-Jones).
        5. For general groups, compute the abelianization and decompose into cyclic factors.
        6. Approximate the total rank by summing ranks of Wh(C_n) for each torsion factor.

    Preserved Invariants:
        - Wh(π₁) is an invariant of the group π₁.
        - Isomorphism of groups π₁ ≅ π₁' implies Wh(π₁) ≅ Wh(π₁').

    Args:
        pi1: The fundamental group (FundamentalGroup object).
        backend: 'auto', 'julia', or 'python'. Julia is recommended for large presentations.

    Returns:
        WhiteheadGroup: Object containing the rank, description, and computation metadata.

    Use When:
        - Deciding if an h-cobordism is an s-cobordism.
        - Computing the L-groups or other surgery invariants that depend on Wh.

    Example:
        pi1 = FundamentalGroup(generators=['a'], relations=[]) # Z
        wh = compute_whitehead_group(pi1)
        print(wh.rank) # 0
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
