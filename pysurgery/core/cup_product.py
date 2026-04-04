import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Tuple

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
    n_simplices = len(simplices_p_plus_q)
    result = np.zeros(n_simplices, dtype=np.int64)
    
    for i, simplex in enumerate(simplices_p_plus_q):
        # Front p-face: first p+1 vertices
        front_face = tuple(simplex[:p+1])
        # Back q-face: last q+1 vertices
        back_face = tuple(simplex[p:])
        
        idx_p = simplex_to_idx_p.get(front_face, -1)
        idx_q = simplex_to_idx_q.get(back_face, -1)
        
        if idx_p != -1 and idx_q != -1:
            result[i] = alpha[idx_p] * beta[idx_q]
            
    return result
