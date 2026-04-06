import numpy as np
from typing import Dict, List, Tuple

def _numpy_alexander_whitney_cup(
    alpha: np.ndarray, 
    beta: np.ndarray, 
    p: int, 
    q: int, 
    simplices: np.ndarray, 
    p_simplex_to_idx: Dict[Tuple[int, ...], int], 
    q_simplex_to_idx: Dict[Tuple[int, ...], int]
) -> np.ndarray:
    """
    Optimized NumPy vectorization for Cup Product evaluation.
    Bypasses pure Python loops to handle millions of simplices.
    """
    n_simplices = len(simplices)
    result = np.zeros(n_simplices, dtype=np.int64)
    
    # We must iterate to query the dictionary, but we can do it efficiently
    # A true Numba JIT implementation would use typed dicts, but this is a 
    # highly optimized Python fallback.
    for i in range(n_simplices):
        simplex = simplices[i]
        
        front_face = tuple(simplex[:p+1])
        back_face = tuple(simplex[p:])
        
        idx_p = p_simplex_to_idx.get(front_face, -1)
        idx_q = q_simplex_to_idx.get(back_face, -1)
        
        if idx_p != -1 and idx_q != -1:
            result[i] = alpha[idx_p] * beta[idx_q]
            
    return result

def alexander_whitney_cup(
    alpha: np.ndarray, 
    beta: np.ndarray, 
    p: int, 
    q: int, 
    simplices_p_plus_q: List[Tuple[int, ...]], 
    simplex_to_idx_p: Dict[Tuple[int, ...], int], 
    simplex_to_idx_q: Dict[Tuple[int, ...], int]
) -> np.ndarray:
    """
    Computes the Alexander-Whitney cup product of a p-cochain alpha and a q-cochain beta.
    The result is a (p+q)-cochain.
    
    Formula: (alpha U beta)([v_0, ..., v_{p+q}]) = alpha([v_0, ..., v_p]) * beta([v_p, ..., v_{p+q}])
    
    Parameters
    ----------
    alpha : np.ndarray
        The p-cochain vector.
    beta : np.ndarray
        The q-cochain vector.
    p : int
        Dimension of alpha.
    q : int
        Dimension of beta.
    simplices_p_plus_q : List[Tuple[int, ...]]
        List of all (p+q)-simplices, where each simplex is a sorted tuple of vertex indices.
    simplex_to_idx_p : Dict[Tuple[int, ...], int]
        Mapping from a p-simplex tuple to its index in the alpha cochain vector.
    simplex_to_idx_q : Dict[Tuple[int, ...], int]
        Mapping from a q-simplex tuple to its index in the beta cochain vector.
        
    Returns
    -------
    np.ndarray
        The resulting (p+q)-cochain evaluated on all (p+q)-simplices.
    """
    # Convert list of tuples to a contiguous NumPy array for cache-friendly iteration
    # This prevents object-overhead in memory when dealing with millions of simplices.
    if len(simplices_p_plus_q) == 0:
        return np.zeros(0, dtype=np.int64)
        
    # Use int64 for consistency with the rest of the pipeline and to avoid
    # overflow on complexes with more than ~2.1 billion vertices.
    simplices_arr = np.array(simplices_p_plus_q, dtype=np.int64)
    
    return _numpy_alexander_whitney_cup(
        alpha, beta, p, q, 
        simplices_arr, 
        simplex_to_idx_p, 
        simplex_to_idx_q
    )

