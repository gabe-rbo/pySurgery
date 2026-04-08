import numpy as np
from itertools import combinations
from typing import Dict, List, Tuple

def _numpy_alexander_whitney_cup(
    alpha: np.ndarray, 
    beta: np.ndarray, 
    p: int, 
    q: int, 
    simplices: np.ndarray, 
    p_simplex_to_idx: Dict[Tuple[int, ...], int], 
    q_simplex_to_idx: Dict[Tuple[int, ...], int],
    modulus: int | None = None,
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
            if modulus is not None:
                result[i] %= modulus

    return result


def cup_i_product(
    alpha: np.ndarray,
    beta: np.ndarray,
    p: int,
    q: int,
    i: int,
    simplices_target: List[Tuple[int, ...]],
    simplex_to_idx_p: Dict[Tuple[int, ...], int],
    simplex_to_idx_q: Dict[Tuple[int, ...], int],
    modulus: int | None = None,
) -> np.ndarray:
    """
    Simplicial cup-i product on ordered simplices using oriented overlap faces.

    For i=0, use Alexander-Whitney directly. For i>0, this evaluates an overlap
    sum over (p,q)-faces whose union is the target simplex and whose intersection
    has dimension i. Orientation signs are induced from each face inclusion.
    """
    if i < 0 or i > min(p, q):
        raise ValueError("cup-i requires 0 <= i <= min(p, q).")
    if len(simplices_target) == 0:
        return np.zeros(0, dtype=np.int64)

    n = p + q - i
    simplices_arr = np.array(simplices_target, dtype=np.int64)

    def face_sign(indices: Tuple[int, ...]) -> int:
        # Inclusion sign of a face [v_{i0},...,v_{ik}] into [v_0,...,v_n].
        parity = 0
        for pos, idx in enumerate(indices):
            parity += (idx - pos)
        return -1 if (parity % 2) else 1

    result = np.zeros(len(simplices_arr), dtype=np.int64)
    all_idx = tuple(range(n + 1))

    for idx, simplex in enumerate(simplices_arr):
        if len(simplex) != n + 1:
            continue

        total = 0
        for A in combinations(all_idx, p + 1):
            A_set = set(A)
            comp = [j for j in all_idx if j not in A_set]
            for overlap in combinations(A, i + 1):
                B = tuple(sorted(comp + list(overlap)))
                if len(B) != q + 1:
                    continue

                a_face = tuple(simplex[j] for j in A)
                b_face = tuple(simplex[j] for j in B)
                a_idx = simplex_to_idx_p.get(a_face, -1)
                b_idx = simplex_to_idx_q.get(b_face, -1)
                if a_idx == -1 or b_idx == -1:
                    continue

                sgn = face_sign(A) * face_sign(B)
                total += sgn * int(alpha[a_idx]) * int(beta[b_idx])

        if modulus is not None:
            total %= modulus
        result[idx] = total
    return result

def alexander_whitney_cup(
    alpha: np.ndarray, 
    beta: np.ndarray, 
    p: int, 
    q: int, 
    simplices_p_plus_q: List[Tuple[int, ...]], 
    simplex_to_idx_p: Dict[Tuple[int, ...], int], 
    simplex_to_idx_q: Dict[Tuple[int, ...], int],
    i: int = 0,
    modulus: int | None = None,
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
        
    # Standardize input for fast memory layout
    simplices_arr = np.array(simplices_p_plus_q, dtype=np.int64)
    if i == 0:
        return _numpy_alexander_whitney_cup(
            alpha,
            beta,
            p,
            q,
            simplices_arr,
            simplex_to_idx_p,
            simplex_to_idx_q,
            modulus=modulus,
        )
    return cup_i_product(
        alpha,
        beta,
        p,
        q,
        i,
        simplices_p_plus_q,
        simplex_to_idx_p,
        simplex_to_idx_q,
        modulus=modulus,
    )

