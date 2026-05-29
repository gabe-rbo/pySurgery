from pydantic import BaseModel, ConfigDict
from typing import Dict
import numpy as np

from pysurgery.topology.complexes import CWComplex, SimplicialComplex
from pysurgery.core.exceptions import DimensionError

class PoincareDualityCertificate(BaseModel):
    """Rigorous manifold certificate based on Poincare Duality."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    is_poincare: bool
    dimension: int
    betti_numbers: Dict[int, int]
    exact: bool = True
    message: str = ""

def is_poincare_duality_complex(cw: CWComplex, dimension: int) -> bool:
    """Algorithmically verify that a complex satisfies Poincare Duality up to Betti numbers.
    
    What is Being Computed?:
        Verifies if rank(H_k) == rank(H_{d-k}) for all k.
        
    Algorithm:
        1. Computes Betti numbers of the CW Complex.
        2. Checks symmetry around the dimension midpoint.
    """
    bettis = cw.betti_numbers()
    for k in range(dimension + 1):
        if bettis.get(k, 0) != bettis.get(dimension - k, 0):
            return False
    return True

def detect_poincare_dimension(cw: CWComplex) -> PoincareDualityCertificate:
    """Algorithmically detects if a space is a Poincare Duality complex and its dimension.
    
    What is Being Computed?:
        Finds the maximum homological dimension n where H_n != 0.
        Checks if H_n is rank 1 (connected orientable top class).
        Verifies symmetry of Betti numbers.
    """
    bettis = cw.betti_numbers()
    if not bettis:
        return PoincareDualityCertificate(is_poincare=False, dimension=-1, betti_numbers=bettis, message="Empty complex")
        
    max_dim = max(bettis.keys())
    if bettis.get(max_dim, 0) != 1:
        return PoincareDualityCertificate(
            is_poincare=False, 
            dimension=max_dim, 
            betti_numbers=bettis, 
            message=f"Top homology H_{max_dim} is not rank 1 (rank is {bettis.get(max_dim, 0)})"
        )
        
    is_pd = is_poincare_duality_complex(cw, max_dim)
    return PoincareDualityCertificate(
        is_poincare=is_pd, 
        dimension=max_dim if is_pd else -1, 
        betti_numbers=bettis, 
        message="Poincare duality holds (Betti symmetry verified)" if is_pd else "Betti numbers asymmetric"
    )

def simplicial_cap_product(
    chain_n: np.ndarray, 
    cochain_k: np.ndarray, 
    n: int, 
    k: int, 
    sc: SimplicialComplex
) -> np.ndarray:
    """Computes the exact simplicial cap product chain_n \cap cochain_k.
    
    Algorithm:
        For each simplex [v_0, ..., v_n] in chain_n:
        result += cochain_k([v_0, ..., v_k]) * [v_k, ..., v_n]
    """
    if n < k:
        raise DimensionError(f"Cannot cap n-chain (n={n}) with k-cochain (k={k}) where n < k.")
        
    target_dim = n - k
    simplices_n = list(sc.n_simplices(n))
    
    if len(chain_n) != len(simplices_n):
        raise DimensionError(f"Chain length {len(chain_n)} does not match number of {n}-simplices ({len(simplices_n)})")
        
    # We need indices for k-simplices and (n-k)-simplices
    idx_k = sc.simplex_to_index(k)
    idx_target = sc.simplex_to_index(target_dim)
    
    result = np.zeros(sc.count_simplices(target_dim), dtype=np.int64)
    
    for i, sn in enumerate(simplices_n):
        c_val = chain_n[i]
        if c_val == 0:
            continue
            
        front_face = tuple(sn[:k+1])
        back_face = tuple(sn[k:])
        
        # Check if faces exist in the complex (they should, by closure)
        if front_face in idx_k and back_face in idx_target:
            front_idx = idx_k[front_face]
            back_idx = idx_target[back_face]
            
            alpha_val = cochain_k[front_idx]
            result[back_idx] += c_val * alpha_val
            
    return result

def compute_poincare_duality_map(
    sc: SimplicialComplex, 
    dimension: int, 
    fundamental_class: np.ndarray,
    k: int
) -> np.ndarray:
    """Computes the Poincare Duality map D: C^k -> C_{n-k} via cap product with fundamental class.
    
    What is Being Computed?:
        The isomorphism given by capping with the fundamental class [X].
        Returns the matrix representing this map.
    """
    num_k_simplices = sc.count_simplices(k)
    target_dim = dimension - k
    num_target_simplices = sc.count_simplices(target_dim)
    
    if num_k_simplices == 0 or num_target_simplices == 0:
        return np.zeros((num_target_simplices, num_k_simplices), dtype=np.int64)
        
    D = np.zeros((num_target_simplices, num_k_simplices), dtype=np.int64)
    
    # Apply to basis elements of C^k
    for i in range(num_k_simplices):
        e_i = np.zeros(num_k_simplices, dtype=np.int64)
        e_i[i] = 1
        
        mapped_chain = simplicial_cap_product(fundamental_class, e_i, dimension, k, sc)
        D[:, i] = mapped_chain
        
    return D

def extract_intersection_pairing_chain(
    alpha: np.ndarray, 
    beta: np.ndarray, 
    p: int, 
    q: int, 
    sc: SimplicialComplex
) -> np.ndarray:
    """Constructs intersection pairing chains (cellular cocycles encoding the intersection form).
    
    What is Being Computed?:
        The cup product chain alpha U beta. If p+q = dim(X), this evaluates on the
        fundamental class to give the intersection pairing.
    """
    from pysurgery.homology.cup_product import alexander_whitney_cup
    
    simplices_target = list(sc.n_simplices(p + q))
    idx_p = sc.simplex_to_index(p)
    idx_q = sc.simplex_to_index(q)
    
    return alexander_whitney_cup(
        alpha, beta, p, q, simplices_target, idx_p, idx_q
    )
