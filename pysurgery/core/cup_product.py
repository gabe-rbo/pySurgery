import numpy as np
import warnings
from typing import Dict, List, Tuple
from ..bridge.julia_bridge import julia_engine


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
    """Optimized map-based evaluation for Cup Product.

    What is Being Computed?:
        Vectorized evaluation of the standard cup product (α ∪ β) at the cochain level.

    Algorithm:
        1. Slices the target (p+q)-simplices into front p-faces and back q-faces.
        2. Maps faces to their respective cochain indices using provided dictionaries.
        3. Identifies valid face pairs and performs vectorized multiplication.
        4. Applies modulus if provided.

    Preserved Invariants:
        - Homotopy invariance of the cup product.

    Args:
        alpha: The p-cochain vector.
        beta: The q-cochain vector.
        p: Dimension of alpha.
        q: Dimension of beta.
        simplices: (n, p+q+1) array of (p+q)-simplices.
        p_simplex_to_idx: Mapping from p-simplex tuple to cochain index.
        q_simplex_to_idx: Mapping from q-simplex tuple to cochain index.
        modulus: Optional modulus for arithmetic.

    Returns:
        np.ndarray: The resulting (p+q)-cochain evaluation.

    Use When:
        - Internal Python-backend evaluation of standard cup products.
        - Julia backend is unavailable or not preferred.
    """
    if len(simplices) == 0:
        return np.zeros(0, dtype=np.int64)

    # Use map/list comprehensions which are executed at C-speed in the CPython VM
    front_faces = (tuple(s) for s in simplices[:, : p + 1])
    back_faces = (tuple(s) for s in simplices[:, p:])

    # Extract indices, defaulting to -1
    idx_p_arr = np.fromiter((p_simplex_to_idx.get(f, -1) for f in front_faces), dtype=np.int64, count=len(simplices))
    idx_q_arr = np.fromiter((q_simplex_to_idx.get(f, -1) for f in back_faces), dtype=np.int64, count=len(simplices))

    # Identify valid overlapping faces
    valid_mask = (idx_p_arr != -1) & (idx_q_arr != -1)
    
    result = np.zeros(len(simplices), dtype=np.int64)
    # Perform actual NumPy vectorization on the arithmetic
    result[valid_mask] = alpha[idx_p_arr[valid_mask]] * beta[idx_q_arr[valid_mask]]

    if modulus is not None:
        result %= modulus

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
    """Simplicial cup-i product on ordered simplices using exact Steenrod interleaved intervals.

    What is Being Computed?:
        Evaluates the cup-i product α ∪_i β of cochains. This is a higher-order 
        topological operation that measures the failure of the cup product 
        to be commutative at the cochain level.

    Algorithm:
        1. Generate all valid Steenrod interleaving sequences for a (p+q-i)-simplex.
        2. For each sequence, extract the front and back face indices.
        3. XOR-evaluate the cochains on these faces and sum the results with 
           the appropriate sign determined by the sequence parity.

    Preserved Invariants:
        - Homotopy invariance of Steenrod squares.
        - Provides the foundation for secondary cohomology operations.

    Args:
        alpha: The p-cochain vector.
        beta: The q-cochain vector.
        p: Dimension of alpha.
        q: Dimension of beta.
        i: Interleaving degree (cup-i).
        simplices_target: List of (p+q-i)-simplices.
        simplex_to_idx_p: Mapping from p-simplex to cochain index.
        simplex_to_idx_q: Mapping from q-simplex to cochain index.
        modulus: Optional modulus.

    Returns:
        np.ndarray: The resulting (p+q-i)-cochain evaluated on target simplices.

    Use When:
        - Computing Steenrod squares Sq^k.
        - Studying higher-order homotopy invariants.
        - Working with mod-2 cohomology where cup-i products are fundamental.

    Example:
        sq2 = cup_i_product(alpha, alpha, p, p, p-2, simplices_target, idx_p, idx_p, modulus=2)

    References:
        Steenrod, N. E. (1947). Products of cocycles and extensions of mappings. 
        Annals of Mathematics, 48(2), 290-320.
    """
    if i < 0 or i > min(p, q):
        raise ValueError("cup-i requires 0 <= i <= min(p, q).")
    if len(simplices_target) == 0:
        return np.zeros(0, dtype=np.int64)

    n = p + q - i
    simplices_arr = np.array(simplices_target, dtype=np.int64)
    result = np.zeros(len(simplices_arr), dtype=np.int64)

    import itertools
    valid_seqs = []
    for k in itertools.combinations_with_replacement(range(n + 1), i + 1):
        A_set = set()
        B_set = set()
        for m in range(i + 1):
            if m % 2 == 0:
                start = 0 if m == 0 else k[m-1]
                end = k[m]
                A_set.update(range(start, end + 1))
                B_set.add(end)
            else:
                start = k[m-1]
                end = k[m]
                B_set.update(range(start, end + 1))
                A_set.add(end)
                
        start = k[-1]
        end = n
        if i % 2 == 0:
            B_set.update(range(start, end + 1))
        else:
            A_set.update(range(start, end + 1))
            
        A_idx = sorted(list(A_set))
        B_idx = sorted(list(B_set))
        
        if len(A_idx) == p + 1 and len(B_idx) == q + 1:
            eps = sum(k[j] - j for j in range(1, i + 1))
            sgn = -1 if eps % 2 != 0 else 1
            valid_seqs.append((A_idx, B_idx, sgn))

    # Optimized evaluation across all simplices
    # Pre-calculate face keys for each valid sequence to allow vectorized lookup
    for A_idx, B_idx, sgn in valid_seqs:
        # simplices_arr[:, A_idx] is a N x (p+1) array
        # We need to map each row to its index in simplex_to_idx_p
        # We use fromiter for efficient mapping from the pre-sliced array
        a_faces = (tuple(row) for row in simplices_arr[:, A_idx])
        b_faces = (tuple(row) for row in simplices_arr[:, B_idx])
        
        idx_p = np.fromiter((simplex_to_idx_p.get(f, -1) for f in a_faces), dtype=np.int64, count=len(simplices_arr))
        idx_q = np.fromiter((simplex_to_idx_q.get(f, -1) for f in b_faces), dtype=np.int64, count=len(simplices_arr))
        
        mask = (idx_p != -1) & (idx_q != -1)
        if np.any(mask):
            term = (alpha[idx_p[mask]] * beta[idx_q[mask]]).astype(np.int64)
            if sgn == -1:
                result[mask] -= term
            else:
                result[mask] += term
                
    if modulus is not None:
        result %= modulus
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
    backend: str = "auto",
) -> np.ndarray:
    """Computes the Alexander-Whitney cup product of a p-cochain alpha and a q-cochain beta.

    What is Being Computed?:
        Evaluates the cup product (α ∪ β) or cup-i product (α ∪_i β) of cochains. 
        For i=0, this is the standard Alexander-Whitney cup product which induces 
        the ring structure on cohomology.

    Algorithm:
        1. If i=0 and Julia is available, delegate to the Julia backend.
        2. If i=0 and using Python, use `_numpy_alexander_whitney_cup` for vectorized 
           front-face/back-face evaluation.
        3. If i > 0, use `cup_i_product` for the interleaved Steenrod definition.

    Preserved Invariants:
        - Cup product structure is a homotopy invariant.
        - Preserves the graded-commutative algebra structure of H*(X; R).

    Args:
        alpha: The p-cochain vector.
        beta: The q-cochain vector.
        p: Dimension of alpha.
        q: Dimension of beta.
        simplices_p_plus_q: List of all (p+q-i)-simplices, each as a sorted tuple.
        simplex_to_idx_p: Mapping from p-simplex tuple to cochain index.
        simplex_to_idx_q: Mapping from q-simplex tuple to cochain index.
        i: Interleaving degree (defaults to 0 for standard cup product).
        modulus: Optional modulus for arithmetic (e.g., 2).
        backend: 'auto', 'julia', or 'python'.

    Returns:
        np.ndarray: The resulting (p+q-i)-cochain evaluated on target simplices.

    Use When:
        - Computing the cohomology ring H*(X).
        - Identifying non-homeomorphic spaces with identical Betti numbers.
        - Computing characteristic classes or Steenrod squares.

    Example:
        cup = alexander_whitney_cup(alpha, beta, p, q, simplices_pq, p_idx, q_idx)
    """
    # Normalize backend
    backend_norm = str(backend).lower().strip()
    use_julia = (backend_norm == "julia") or (backend_norm == "auto" and julia_engine.available)

    # Convert list of tuples to a contiguous NumPy array for cache-friendly iteration
    # This prevents object-overhead in memory when dealing with millions of simplices.
    if len(simplices_p_plus_q) == 0:
        return np.zeros(0, dtype=np.int64)

    if i == 0 and use_julia:
        try:
            return julia_engine.compute_alexander_whitney_cup(
                alpha,
                beta,
                p,
                q,
                simplices_p_plus_q,
                simplex_to_idx_p,
                simplex_to_idx_q,
                modulus=modulus,
            )
        except Exception as e:
            if backend_norm == "julia":
                raise e
            warnings.warn(
                f"Topological Hint: Julia cup product acceleration failed ({e!r}). "
                "Falling back to slower pure-Python evaluation."
            )

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


def steenrod_square(
    alpha: np.ndarray,
    p: int,
    k: int,
    simplices_target: List[Tuple[int, ...]],
    simplex_to_idx_p: Dict[Tuple[int, ...], int],
    modulus: int = 2,
) -> np.ndarray:
    """Computes the k-th Steenrod square Sq^k(alpha) for a p-cochain alpha.

    What is Being Computed?:
        Evaluates the k-th Steenrod square Sq^k: H^p(X; Z/2) -> H^{p+k}(X; Z/2).

    Algorithm:
        1. Relates Sq^k(α) to the cup-(p-k) product: Sq^k(α) = α ∪_{p-k} α.
        2. Calls `cup_i_product` with i = p - k and modulus = 2.

    Preserved Invariants:
        - Steenrod squares are stable cohomology operations.
        - Homotopy invariant under continuous maps.
        - Natural with respect to induced maps of spaces.

    Args:
        alpha: The p-cochain vector.
        p: Dimension of alpha.
        k: The degree of the square.
        simplices_target: List of (p+k)-simplices.
        simplex_to_idx_p: Mapping from p-simplex to cochain index.
        modulus: Arithmetic modulus (must be 2 for Steenrod squares).

    Returns:
        np.ndarray: The resulting (p+k)-cochain evaluated on target simplices.

    Use When:
        - Computing stable cohomology operations.
        - Distinguishing homotopy types when cup products are insufficient.
        - Working with characteristic classes (Wu classes, Stiefel-Whitney).

    Example:
        sq1_alpha = steenrod_square(alpha, p, 1, targets, idx_p)
    """
    if k < 0 or k > p:
        return np.zeros(len(simplices_target), dtype=np.int64)
    
    # Sq^k(alpha) lives in H^{p+k}.
    # cup_i: H^p x H^q -> H^{p+q-i}.
    # Here q=p, i=p-k. Result dim = p + p - (p - k) = p + k.
    return cup_i_product(
        alpha, alpha, p, p, p - k,
        simplices_target,
        simplex_to_idx_p,
        simplex_to_idx_p,
        modulus=modulus
    )
