import warnings
from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Dict
import numpy as np

from pysurgery.algebra.intersection_forms import IntersectionForm
from pysurgery.core.exceptions import SurgeryError

class RationalObstruction(BaseModel):
    """Rational Surgery Obstruction over Z (x) Q."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    dimension: int
    pi: str
    signature: Optional[int] = None
    rank_q: Optional[int] = None
    exact: bool = True

class PLocalObstruction(BaseModel):
    """p-Local Surgery Obstruction over Z_(p)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    dimension: int
    pi: str
    prime: int
    rank_mod_p: Optional[int] = None
    p_adic_diagonal: Optional[List[int]] = None
    exact: bool = True

class PrimeLocalReport(BaseModel):
    """Prime-Local Obstruction Report combining Rational and p-Local sequences."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    dimension: int
    pi: str
    rational_obstruction: RationalObstruction
    p_local_obstructions: Dict[int, PLocalObstruction]
    reconstructed_integral_diagonal: Optional[List[int]] = None
    exact: bool = True


def compute_l_group_rational(dimension: int, pi: str, form: Optional[IntersectionForm] = None) -> RationalObstruction:
    """Deprecated: use ``IntersectionForm.l_group_rational(dimension, pi)`` instead.

    When ``form`` is ``None`` no signature is computable, so a stub
    :class:`RationalObstruction` is returned (with ``exact=True`` since
    no float arithmetic was used).
    """
    warnings.warn(
        "compute_l_group_rational is deprecated; use IntersectionForm.l_group_rational(dim, pi)",
        DeprecationWarning,
        stacklevel=2,
    )
    if form is None or form.matrix is None:
        return RationalObstruction(
            dimension=dimension,
            pi=pi,
            signature=None,
            rank_q=None,
            exact=True,
        )
    return form.l_group_rational(dimension, pi)


def compute_l_group_p_local(dimension: int, pi: str, prime_p: int, form: Optional[IntersectionForm] = None) -> PLocalObstruction:
    """Computes the p-local L-group obstruction for a specific prime p.
    
    What is Being Computed?:
        The image of the surgery obstruction under the localization map L*(Z) -> L*(Z_(p)).
        Computes rank over GF(p) and the p-adic CRT SNF diagonal factors.
    """
    rank_mod_p = None
    p_adic_diagonal = None
    
    if form is not None and form.matrix is not None:
        from ..bridge.julia_bridge import julia_engine
        import scipy.sparse as sp
        
        if julia_engine.available:
            matrix_sparse = sp.csr_matrix(form.matrix)
            rank_mod_p = julia_engine.compute_sparse_rank_mod_p(matrix_sparse, prime_p)
            p_adic_diagonal = julia_engine.compute_padic_snf_diagonal(matrix_sparse, primes=[prime_p]).tolist()
            
    return PLocalObstruction(
        dimension=dimension,
        pi=pi,
        prime=prime_p,
        rank_mod_p=rank_mod_p,
        p_adic_diagonal=p_adic_diagonal
    )


def reconstruct_integral_obstruction(rational_part: RationalObstruction, p_local_parts: Dict[int, PLocalObstruction]) -> List[int]:
    """Reconstructs the exact integral SNF diagonal using the Chinese Remainder Theorem.
    
    What is Being Computed?:
        The integer Smith Normal Form diagonal using pure prime-local arithmetic.
        
    Algorithm:
        1. Uses rank_q from the rational part to establish the exact number of non-zero entries.
        2. Multiplies the p-adic factors from each prime.
        3. Since julia_engine returns ordered factors, element-wise multiplication naturally
           satisfies the divisibility chain condition d_i | d_{i+1}.
    """
    if rational_part.rank_q is None:
        raise SurgeryError("Rational rank is required to reconstruct integral SNF diagonal.")
        
    rank = rational_part.rank_q
    integral_diagonal = np.ones(rank, dtype=object)
    
    for p, p_local in p_local_parts.items():
        if p_local.p_adic_diagonal is None:
            continue
            
        diag = p_local.p_adic_diagonal
        # Filter out 0s if any exist, as we only construct the non-zero rank portion
        non_zero_diag = [x for x in diag if x != 0]
        
        # Pad or truncate to match rank. Typically length is <= rank.
        padded_diag = np.ones(rank, dtype=object)
        n = min(rank, len(non_zero_diag))
        padded_diag[:n] = non_zero_diag[:n]
        
        integral_diagonal *= padded_diag
        
    return integral_diagonal.tolist()


def prime_local_obstruction_report(
    dimension: int, 
    pi: str, 
    form: Optional[IntersectionForm] = None, 
    primes: List[int] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
) -> PrimeLocalReport:
    """Generates the full prime-local obstruction report merging rational and p-local views.
    
    What is Being Computed?:
        The complete breakdown of the manifold classification obstruction into its
        rational and p-local components, along with the CRT-reconstructed integral invariant.
    """
    if form is None or form.matrix is None:
        rat = RationalObstruction(
            dimension=dimension, pi=pi, signature=None, rank_q=None, exact=True,
        )
    else:
        rat = form.l_group_rational(dimension, pi)
    
    p_locals = {}
    for p in primes:
        p_local = compute_l_group_p_local(dimension, pi, p, form)
        p_locals[p] = p_local
        
    recon = None
    if rat.rank_q is not None and len(p_locals) > 0:
        # Check if we successfully computed p_local bounds
        if any(v.p_adic_diagonal is not None for v in p_locals.values()):
            recon = reconstruct_integral_obstruction(rat, p_locals)
        
    exact_all = rat.exact and all(p.exact for p in p_locals.values())
        
    return PrimeLocalReport(
        dimension=dimension,
        pi=pi,
        rational_obstruction=rat,
        p_local_obstructions=p_locals,
        reconstructed_integral_diagonal=recon,
        exact=exact_all
    )
