import hashlib
import itertools
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import numba
import warnings
import sympy as sp
from scipy.sparse import csr_matrix
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, cast, Literal, Set
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
from .math_core import get_sparse_snf_diagonal
from ..bridge.julia_bridge import julia_engine


def _parse_coefficient_ring(ring: str) -> tuple[str, int | None]:
    """Parse user ring labels into internal `(kind, modulus)` form.

    What is Being Computed?:
        Translates human-readable ring specifications (e.g., 'Z', 'Q', 'Z/2Z') into a 
        standardized internal format consisting of a kind ('Z', 'Q', 'ZMOD') and 
        an optional modulus.

    Algorithm:
        1. Strip whitespace and convert input to uppercase.
        2. If 'Z', return ('Z', None).
        3. If 'Q', return ('Q', None).
        4. If 'Z/pZ' format, extract p, validate p > 1, and return ('ZMOD', p).
        5. Otherwise, raise a ValueError for unsupported formats.

    Preserved Invariants:
        - None (purely a parsing utility).

    Args:
        ring: User ring label (e.g., 'Z', 'Q', 'Z/2Z').

    Returns:
        tuple[str, int | None]: A tuple (kind, modulus) where kind is one of 'Z', 'Q', or 'ZMOD'.

    Use When:
        - Initializing a ChainComplex or SimplicialComplex with a user-provided ring string.
        - Validating ring inputs before performing homological computations.

    Example:
        kind, p = _parse_coefficient_ring("Z/2Z")  # ('ZMOD', 2)
    """
    rs = ring.strip().upper()
    if rs == "Z":
        return "Z", None
    if rs == "Q":
        return "Q", None
    if rs.startswith("Z/") and rs.endswith("Z"):
        p_str = rs[2:-1]
        p = int(p_str)
        if p <= 1:
            raise ValueError("Z/pZ requires p > 1.")
        return "ZMOD", p
    raise ValueError(f"Unsupported coefficient ring '{ring}'. Use 'Z', 'Q', or 'Z/pZ'.")


def _coerce_csr_matrix(matrix: csr_matrix | np.ndarray | list | tuple) -> csr_matrix:
    """Coerce sparse/dense matrix-like data to CSR with integer entries.

    What is Being Computed?:
        Converts various matrix-like inputs (dense arrays, lists, tuples, or existing sparse matrices)
        into a consistent SciPy CSR (Compressed Sparse Row) format with 64-bit integer entries.

    Algorithm:
        1. If input is already a CSR matrix, copy it and cast to int64.
        2. Otherwise, convert input to a NumPy array of int64, then to a CSR matrix.

    Preserved Invariants:
        - The linear map represented by the matrix is preserved (up to integer truncation if input is float).

    Args:
        matrix: Matrix-like data to coerce (csr_matrix, ndarray, list, or tuple).

    Returns:
        csr_matrix: The matrix in CSR format with int64 entries.

    Use When:
        - Standardizing input for boundary matrices in ChainComplex.
        - Preparing data for sparse algebraic operations (e.g., SNF, rank).

    Example:
        sparse_d = _coerce_csr_matrix([[1, -1, 0], [0, 1, -1]])
    """
    if isinstance(matrix, csr_matrix):
        return matrix.copy().astype(np.int64)
    return csr_matrix(np.asarray(matrix, dtype=np.int64), dtype=np.int64)


def _normalize_simplex(simplex: Iterable[int]) -> tuple[int, ...]:
    """Return a canonical, sorted simplex tuple with distinct integer vertices.

    What is Being Computed?:
        A canonical representation of a simplex. Since a simplex is defined by its 
        set of vertices, this function ensures that any representation of the 
        same simplex results in the same sorted tuple.

    Algorithm:
        1. Convert all vertices to integers.
        2. Remove duplicates using a set.
        3. Sort the unique vertices.
        4. Return as a tuple.
        5. Raise ValueError if the resulting simplex is empty.

    Preserved Invariants:
        - Simplex identity: Two sets of vertices representing the same simplex will 
          have the same normalized form.

    Args:
        simplex: Iterable of vertex indices.

    Returns:
        tuple[int, ...]: A sorted tuple of unique vertex indices.

    Use When:
        - Adding simplices to a SimplicialComplex.
        - Using simplices as keys in a dictionary or items in a set.

    Example:
        s = _normalize_simplex([3, 1, 2, 1])  # (1, 2, 3)
    """
    vertices = tuple(sorted(set(int(v) for v in simplex)))
    if len(vertices) == 0:
        raise ValueError("Simplices must be non-empty.")
    return vertices


def _canonicalize_simplices_by_dim(
    raw_grouped: dict[int, list[tuple[int, ...]]]
) -> dict[int, list[tuple[int, ...]]]:
    """Sort and deduplicate simplex lists across dimensions.

    What is Being Computed?:
        A cleaned dictionary where each dimension maps to a sorted, unique list of 
        normalized simplices.

    Algorithm:
        1. Iterate through each dimension and its list of simplices.
        2. Deduplicate the list using a dictionary (preserving order if possible, though sorted later).
        3. Sort the unique simplices for each dimension.
        4. Return the new dictionary.

    Preserved Invariants:
        - The set of simplices in each dimension remains the same.
        - Overall complex structure is preserved.

    Args:
        raw_grouped: Dictionary mapping dimension (int) to list of simplex tuples.

    Returns:
        dict[int, list[tuple[int, ...]]]: A dictionary with sorted and deduplicated simplex lists.

    Use When:
        - Finalizing the internal representation of a SimplicialComplex after adding many simplices.
        - Preparing simplex lists for boundary matrix construction.

    Example:
        raw = {0: [(1,), (0,), (1,)], 1: [(0, 1), (0, 1)]}
        clean = _canonicalize_simplices_by_dim(raw)  # {0: [(0,), (1,)], 1: [(0, 1)]}
    """
    out = {}
    for d, simplices in raw_grouped.items():
        out[d] = sorted(list(dict.fromkeys(simplices)))
    return out


def _close_single_generator(simplex_raw):
    t = _normalize_simplex(simplex_raw)
    local_faces = defaultdict(set)
    for r in range(1, len(t) + 1):
        for face in itertools.combinations(t, r):
            local_faces[r-1].add(tuple(face))
    return local_faces

def _simplicial_closure_from_generators(
    simplices: Iterable[Iterable[int]],
) -> dict[int, list[tuple[int, ...]]]:
    """Generate all faces of the given simplices and group by dimension.

    What is Being Computed?:
        The downward skeletal closure of a set of simplices. For every simplex 
        provided, all its sub-simplices (faces) are generated and included in 
        the result.

    Algorithm:
        1. Convert input to a list of simplices.
        2. If the number of simplices is large (>5000), use a ThreadPoolExecutor 
           to parallelize the face generation.
        3. For each simplex, find all non-empty subsets of its vertices.
        4. Accumulate unique faces in a dictionary grouped by dimension.
        5. Return the dictionary with sorted simplex lists.

    Preserved Invariants:
        - Ensures the result is a valid simplicial complex (closed under taking faces).
        - Preserves the homotopy type defined by the input generators.

    Args:
        simplices: Iterable of simplices (generators).

    Returns:
        dict[int, list[tuple[int, ...]]]: A dictionary mapping dimension to sorted lists of unique simplices.

    Use When:
        - Constructing a SimplicialComplex from a set of maximal simplices (facets).
        - Ensuring that a list of simplices forms a closed complex.

    Example:
        closure = _simplicial_closure_from_generators([(0, 1, 2)])
        # {0: [(0,), (1,), (2,)], 1: [(0, 1), (0, 2), (1, 2)], 2: [(0, 1, 2)]}
    """
    final_grouped = defaultdict(set)
    # Avoid nested threading overhead for small complexes (like vertex links)
    # by only using ThreadPool if input size is significant.
    simplex_list_for_closure = list(simplices)
    if len(simplex_list_for_closure) > 5000:
        with ThreadPoolExecutor() as executor:
            results = executor.map(_close_single_generator, simplex_list_for_closure)
            for local_faces in results:
                for dim, faces in local_faces.items():
                    final_grouped[dim].update(faces)
    else:
        for s in simplex_list_for_closure:
            local_faces = _close_single_generator(s)
            for dim, faces in local_faces.items():
                final_grouped[dim].update(faces)
                
    return {d: sorted(list(s)) for d, s in final_grouped.items()}


def _boundary_matrix_from_simplices_with_maps(
    simplices_n: List[Tuple[int, ...]],
    nm1_map: Dict[Tuple[int, ...], int],
) -> csr_matrix:
    """Construct an oriented boundary matrix using a pre-computed face map.

    What is Being Computed?:
        The n-th boundary operator ∂_n: C_n → C_{n-1} as a sparse matrix. The matrix 
        entry at (i, j) is (-1)^k if the i-th (n-1)-simplex is the k-th face of the 
        j-th n-simplex, and 0 otherwise.

    Algorithm:
        1. Handle the base case of an empty list of n-simplices.
        2. Define a chunked computation function to populate sparse matrix indices.
        3. For each n-simplex, iterate through its (n-1)-dimensional faces.
        4. Lookup each face in the pre-computed `nm1_map` to find its row index.
        5. Assign the alternating sign (-1)^i to the entry.
        6. Parallelize the computation using ThreadPoolExecutor for large inputs (>5000).
        7. Assemble and return a SciPy CSR matrix.

    Preserved Invariants:
        - Homology property: ∂_n ∘ ∂_{n+1} = 0 is satisfied if the complex is valid.
        - Orientation consistency: Respects the canonical ordering of vertices.

    Args:
        simplices_n: List of n-simplices (sorted tuples).
        nm1_map: Pre-computed mapping from (n-1)-simplex tuple to its index.

    Returns:
        csr_matrix: The sparse boundary matrix d_n in CSR format.

    Use When:
        - Building a ChainComplex from a SimplicialComplex.
        - Computing boundary maps for specific dimensions of a complex.

    Example:
        s1 = [(0,), (1,), (2,)]
        s2 = [(0, 1), (1, 2)]
        map1 = {v: i for i, v in enumerate(s1)}
        d1 = _boundary_matrix_from_simplices_with_maps(s2, map1)
    """
    if not simplices_n:
        return csr_matrix((len(nm1_map), 0), dtype=np.int64)

    rows, cols, data = [], [], []

    def _compute_cols_for_chunk(chunk_data):
        chunk_simplices, start_j = chunk_data
        l_rows, l_cols, l_data = [], [], []
        for offset, simplex in enumerate(chunk_simplices):
            j = start_j + offset
            for i in range(len(simplex)):
                face = simplex[:i] + simplex[i + 1 :]
                if face in nm1_map:
                    l_rows.append(nm1_map[face])
                    l_cols.append(j)
                    l_data.append((-1) ** i)
        return l_rows, l_cols, l_data

    # Avoid nested threading overhead for small complexes
    if len(simplices_n) > 5000:
        n_workers = 8
        chunk_size = max(1, len(simplices_n) // (n_workers * 4))
        chunks = [(simplices_n[i : i + chunk_size], i) for i in range(0, len(simplices_n), chunk_size)]

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for l_rows, l_cols, l_data in executor.map(_compute_cols_for_chunk, chunks):
                rows.extend(l_rows)
                cols.extend(l_cols)
                data.extend(l_data)
    else:
        l_rows, l_cols, l_data = _compute_cols_for_chunk((simplices_n, 0))
        rows, cols, data = l_rows, l_cols, l_data

    return csr_matrix(
        (data, (rows, cols)),
        shape=(len(nm1_map), len(simplices_n)),
        dtype=np.int64,
    )


def _csr_matrix_signature(m: csr_matrix) -> tuple[int, int, int, str]:
    """Return a content-based signature for a sparse matrix to detect changes.

    What is Being Computed?:
        A unique fingerprint of a sparse matrix's contents, including its shape, 
        sparsity pattern (nonzero indices), and data values.

    Algorithm:
        1. Extract nonzero row and column indices.
        2. Feed the shape, row indices, column indices, and data values into a 
           SHA-256 hash function.
        3. Return a tuple containing (number of rows, number of columns, 
           number of non-zero elements, and the hex digest of the hash).

    Preserved Invariants:
        - Deterministic: Identical matrices always produce the same signature.
        - Sensitivity: Any change in matrix entries or structure results in a 
          different signature.

    Args:
        m: The sparse CSR matrix to hash.

    Returns:
        tuple[int, int, int, str]: A signature tuple (rows, cols, nnz, hash_str).

    Use When:
        - Implementing caching mechanisms where the result depends on matrix data.
        - Detecting if a complex has been modified to invalidate cached invariants.

    Example:
        sig = _csr_matrix_signature(boundary_matrix)
    """
    rows, cols = m.nonzero()
    data = m.data
    h = hashlib.sha256()
    h.update(np.asarray(m.shape, dtype=np.int64).tobytes())
    h.update(rows.tobytes())
    h.update(cols.tobytes())
    h.update(data.tobytes())
    return int(m.shape[0]), int(m.shape[1]), int(m.nnz), h.hexdigest()


def _clone_cache_value(v: Any) -> Any:
    """Return a shallow copy of large objects to prevent accidental cache mutation.

    What is Being Computed?:
        A copy of a value being retrieved from or stored in the cache. This 
        prevents external modifications to the original object from affecting 
        the cached copy (and vice versa).

    Algorithm:
        1. If the value is an immutable primitive (int, float, str, bool, tuple, None), return it as is.
        2. If it's a list, return a new list with copies of its elements.
        3. If it's a dictionary, recursively clone its values.
        4. If the object has a `.copy()` method, use it.
        5. Otherwise, return the object as is (fallback).

    Preserved Invariants:
        - Value equivalence: The cloned object represents the same mathematical data.

    Args:
        v: The value to clone (any type).

    Returns:
        Any: A copy of the value.

    Use When:
        - Getting/setting values in the internal cache of ChainComplex or SimplicialComplex.
        - Ensuring that mutable objects (like matrices or lists) aren't shared across different parts of the system accidentally.

    Example:
        safe_copy = _clone_cache_value(some_mutable_list)
    """
    if isinstance(v, (int, float, str, bool, tuple)) or v is None:
        return v
    if isinstance(v, list):
        return [x.copy() if hasattr(x, "copy") else x for x in v]
    if isinstance(v, dict):
        return {k: _clone_cache_value(val) for k, val in v.items()}
    if hasattr(v, "copy"):
        return v.copy()
    return v


def _is_prime(n: int) -> bool:
    """Check if n is prime (heuristic for small moduli).

    What is Being Computed?:
        Primality test for an integer n.

    Algorithm:
        1. If n < 2, return False.
        2. If n = 2, return True.
        3. If n is even, return False.
        4. Check odd divisors from 3 up to sqrt(n).
        5. If any divisor is found, return False. Otherwise, return True.

    Preserved Invariants:
        - None.

    Args:
        n: The integer to check.

    Returns:
        bool: True if n is prime, False otherwise.

    Use When:
        - Validating if a coefficient ring Z/pZ is over a field (p prime).
        - Choosing algorithms that require prime moduli (e.g., specific rank computations).

    Example:
        is_p = _is_prime(7)  # True
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


@numba.njit(cache=True)
def _gcd_extended_numba(a: int, b: int):
    x0, x1, y0, y1 = 1, 0, 0, 1
    while b != 0:
        q = a // b
        a, b = b, a % b
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    return a, x0, y0

def _gcd_extended(a: int, b: int) -> tuple[int, int, int]:
    """Extended Euclidean algorithm.

    What is Being Computed?:
        The greatest common divisor (GCD) g of integers a and b, along with 
        coefficients x and y such that ax + by = g.

    Algorithm:
        Delegates to `_gcd_extended_numba`, which implements the standard 
        iterative extended Euclidean algorithm.

    Preserved Invariants:
        - Bézout's identity: ax + by = gcd(a, b).

    Args:
        a: First integer.
        b: Second integer.

    Returns:
        tuple[int, int, int]: (g, x, y) where g is the GCD and x, y are the coefficients.

    Use When:
        - Computing modular inverses (e.g., in Z/pZ arithmetic).
        - Performing exact linear algebra over integers or finite rings.

    Example:
        g, x, y = _gcd_extended(10, 6)  # (2, -1, 2) since 10*(-1) + 6*2 = 2
    """
    return _gcd_extended_numba(a, b)

@numba.njit(cache=True)
def _rank_mod_p_numba(M: np.ndarray, p: int) -> int:
    m, n = M.shape
    row = 0
    rank = 0
    for col in range(n):
        pivot = -1
        for r in range(row, m):
            if M[r, col] % p != 0:
                pivot = r
                break
        if pivot == -1:
            continue
        if pivot != row:
            for j in range(col, n):
                M[row, j], M[pivot, j] = M[pivot, j], M[row, j]

        # Euclidean reduction
        for r in range(row + 1, m):
            while M[r, col] % p != 0:
                q = M[row, col] // M[r, col]
                for j in range(col, n):
                    M[row, j] = (M[row, j] - q * M[r, j]) % p
                    M[row, j], M[r, j] = M[r, j], M[row, j]

        # Check invertibility
        g, x, y = _gcd_extended_numba(M[row, col], p)
        if g == 1:
            inv = x % p
            for j in range(col, n):
                M[row, j] = (M[row, j] * inv) % p
            for r in range(m):
                if r != row and M[r, col] % p != 0:
                    factor = M[r, col]
                    for j in range(col, n):
                        M[r, j] = (M[r, j] - factor * M[row, j]) % p
        else:
            for r in range(m):
                if r != row and M[r, col] % p != 0:
                    g2, x2, y2 = _gcd_extended_numba(M[row, col], M[r, col])
                    a, b = M[row, col], M[r, col]
                    for j in range(col, n):
                        row_i = (x2 * M[row, j] + y2 * M[r, j]) % p
                        row_j = ((-b // g2) * M[row, j] + (a // g2) * M[r, j]) % p
                        M[row, j] = row_i
                        M[r, j] = row_j

        row += 1
        rank += 1
        if row == m:
            break
    return rank

def _rank_mod_p(A: np.ndarray, p: int) -> int:
    """Compute matrix rank over `Z/pZ` via Euclidean row reduction (handles composite p).

    What is Being Computed?:
        The rank of a matrix over the ring Z/pZ. This measures the number of 
        linearly independent rows/columns in the finite field (if p is prime) 
        or ring (if p is composite).

    Algorithm:
        1. Copy the input array and take values modulo p.
        2. Delegate to `_rank_mod_p_numba`, which performs Gaussian-like 
           elimination using Euclidean reduction to handle possibly non-prime p.
        3. Count the number of pivots found.

    Preserved Invariants:
        - Rank is invariant under elementary row and column operations over Z/pZ.

    Args:
        A: The input dense matrix as a NumPy array.
        p: The modulus.

    Returns:
        int: The rank of the matrix over Z/pZ.

    Use When:
        - Computing Betti numbers or homology with Z/pZ coefficients.
        - Checking linear independence of chains over a finite field.

    Example:
        rank = _rank_mod_p(np.array([[1, 0], [0, 2]]), 2)  # 1 (since 2 mod 2 is 0)
    """
    M = (A.astype(np.int64) % p).copy()
    return _rank_mod_p_numba(M, p)

@numba.njit(cache=True)
def _rref_mod_p_numba(M: np.ndarray, p: int):
    m, n = M.shape
    row = 0
    pivots = []
    for col in range(n):
        pivot = -1
        for r in range(row, m):
            if M[r, col] % p != 0:
                pivot = r
                break
        if pivot == -1:
            continue
        if pivot != row:
            for j in range(col, n):
                M[row, j], M[pivot, j] = M[pivot, j], M[row, j]

        # Euclidean reduction
        for r in range(row + 1, m):
            while M[r, col] % p != 0:
                q = M[row, col] // M[r, col]
                for j in range(col, n):
                    M[row, j] = (M[row, j] - q * M[r, j]) % p
                    M[row, j], M[r, j] = M[r, j], M[row, j]

        g, x, y = _gcd_extended_numba(M[row, col], p)
        if g == 1:
            inv = x % p
            for j in range(col, n):
                M[row, j] = (M[row, j] * inv) % p
            for r in range(m):
                if r != row and M[r, col] % p != 0:
                    factor = M[r, col]
                    for j in range(col, n):
                        M[r, j] = (M[r, j] - factor * M[row, j]) % p
        else:
            for r in range(m):
                if r != row and M[r, col] % p != 0:
                    g2, x2, y2 = _gcd_extended_numba(M[row, col], M[r, col])
                    a, b = M[row, col], M[r, col]
                    for j in range(col, n):
                        row_i = (x2 * M[row, j] + y2 * M[r, j]) % p
                        row_j = ((-b // g2) * M[row, j] + (a // g2) * M[r, j]) % p
                        M[row, j] = row_i
                        M[r, j] = row_j

        pivots.append(col)
        row += 1
        if row == m:
            break
    
    # Numba handles returning lists, but typed lists are better. We just convert to array.
    return M, pivots

def _rref_mod_p(A: np.ndarray, p: int) -> tuple[np.ndarray, list[int]]:
    """Compute row-reduced echelon form over `Z/pZ` via Euclidean reduction.

    What is Being Computed?:
        The row-reduced echelon form (RREF) of a matrix over Z/pZ, and the 
        indices of the pivot columns.

    Algorithm:
        1. Copy the input matrix and take values modulo p.
        2. Perform row reduction using the Euclidean algorithm for pivots to 
           handle composite p.
        3. Eliminate entries above and below pivots to achieve reduced form.
        4. Return the reduced matrix and the list of pivot column indices.

    Preserved Invariants:
        - Row space of the matrix is preserved.
        - Rank (number of pivots) is preserved.

    Args:
        A: The input dense matrix.
        p: The modulus.

    Returns:
        tuple[np.ndarray, list[int]]: A tuple (reduced_matrix, pivot_columns).

    Use When:
        - Computing nullspaces or basis vectors over finite fields.
        - Solving linear systems over Z/pZ.

    Example:
        rref, pivots = _rref_mod_p(np.array([[2, 1], [1, 1]]), 3)
    """
    M = (A.astype(np.int64) % p).copy()
    M_out, pivots = _rref_mod_p_numba(M, p)
    return M_out, list(pivots)



def _nullspace_basis_mod_p(A: np.ndarray, p: int) -> list[np.ndarray]:
    """Return a basis of `ker(A)` over `Z/pZ`.

    What is Being Computed?:
        A basis for the kernel (nullspace) of a linear map A over the ring Z/pZ.

    Algorithm:
        1. Compute the row-reduced echelon form (RREF) of A over Z/pZ using `_rref_mod_p`.
        2. Identify free columns (columns without pivots).
        3. For each free column, construct a basis vector by setting the free 
           variable to 1 and solving for pivot variables.
        4. Return the list of basis vectors.

    Preserved Invariants:
        - The span of the returned vectors is exactly the nullspace of A.

    Args:
        A: The input matrix as a dense NumPy array.
        p: The modulus.

    Returns:
        list[np.ndarray]: A list of basis vectors for the nullspace.

    Use When:
        - Computing homology or cohomology basis over finite fields.
        - Solving for cycles that are not boundaries.

    Example:
        basis = _nullspace_basis_mod_p(np.array([[1, 1]]), 2)  # [array([1, 1])]
    """
    m, n = A.shape
    rref, pivots = _rref_mod_p(A, p)
    pivot_set = set(pivots)
    free_cols = [j for j in range(n) if j not in pivot_set]
    if not free_cols:
        return []
    basis: list[np.ndarray] = []
    for free in free_cols:
        v = np.zeros(n, dtype=np.int64)
        v[free] = 1
        for i, col in enumerate(pivots):
            v[col] = (-rref[i, free]) % p
        basis.append(v)
    return basis


@numba.njit(cache=True)
def _is_independent_wrt_mod_p_kernel(work, pivot_indices, pivot_rows, n_pivots, p):
    for i in range(len(work)):
        if work[i] == 0:
            continue
        
        found_pivot = -1
        for idx in range(n_pivots):
            if pivot_indices[idx] == i:
                found_pivot = idx
                break
        
        if found_pivot == -1:
            # Found a new pivot dimension
            return i, work
        
        # Euclidean reduction step
        pivot_vec = pivot_rows[found_pivot]
        while work[i] != 0:
            q = work[i] // pivot_vec[i]
            if q != 0:
                for j in range(i, len(work)):
                    work[j] = (work[j] - q * pivot_vec[j]) % p
            
            if work[i] != 0:
                # Swap and continue Euclidean step
                for j in range(i, len(work)):
                    tmp = work[j]
                    work[j] = pivot_vec[j]
                    pivot_vec[j] = tmp
    return -1, work

def _is_independent_wrt_optimized(
    v: np.ndarray, 
    pivot_matrix: np.ndarray, 
    pivot_indices: np.ndarray, 
    n_pivots: int, 
    p: Optional[int] = None
) -> tuple[bool, int]:
    """Check independence using a pre-allocated pivot matrix for zero allocation.

    What is Being Computed?:
        Checks if a vector `v` is linearly independent of a set of existing 
        pivot vectors. If it is independent, it is reduced and added to the 
        pivot set.

    Algorithm:
        1. Copy the input vector into a work array.
        2. If a modulus `p` is provided (Z/pZ case):
           a. Perform Euclidean reduction of `work` using existing pivot rows.
           b. If a new pivot dimension is found, normalize the vector (if possible) 
              and add it to `pivot_matrix`.
        3. If no modulus is provided (Q/R case):
           a. Use standard field reduction (Gaussian elimination with floats).
           b. Check if the remaining vector is non-zero (above a threshold).
           c. If non-zero, normalize and add to `pivot_matrix`.
        4. Return whether it was independent and the updated pivot count.

    Preserved Invariants:
        - The span of the pivot matrix is maintained and optionally extended.

    Args:
        v: The vector to test for independence.
        pivot_matrix: Pre-allocated 2D array storing current pivot vectors.
        pivot_indices: Array storing the column index of the first non-zero entry for each pivot.
        n_pivots: Current number of pivots in the matrix.
        p: Optional modulus for Z/pZ.

    Returns:
        tuple[bool, int]: (is_independent, updated_n_pivots).

    Use When:
        - Building a basis incrementally from a set of candidate vectors.
        - High-performance basis reduction where minimizing allocations is critical.

    Example:
        is_indep, n = _is_independent_wrt_optimized(v, pivots, indices, n)
    """
    work = np.asarray(v, dtype=np.int64).copy()
    if p is not None:
        p = int(p)
        work %= p
        
        new_pivot_idx, final_work = _is_independent_wrt_mod_p_kernel(
            work, pivot_indices, pivot_matrix, n_pivots, p
        )
            
        if new_pivot_idx != -1:
            # Normalize if possible
            try:
                inv = pow(int(final_work[new_pivot_idx]), -1, p)
                final_work = (final_work * inv) % p
            except ValueError:
                pass
            pivot_matrix[n_pivots] = final_work
            pivot_indices[n_pivots] = new_pivot_idx
            return True, n_pivots + 1
        return False, n_pivots
    else:
        # Field reduction over Q/R (Float)
        work = work.astype(float)
        # For float field, we use a simpler loop or similar logic
        for i in range(n_pivots):
            idx = int(pivot_indices[i])
            factor = work[idx] / pivot_matrix[i, idx]
            work -= factor * pivot_matrix[i]
            
        first_nz = np.where(np.abs(work) > 1e-14)[0]
        if len(first_nz) > 0:
            idx = int(first_nz[0])
            work /= work[idx]
            pivot_matrix[n_pivots] = work
            pivot_indices[n_pivots] = idx
            return True, n_pivots + 1
        return False, n_pivots

def _is_independent_wrt(
    v: np.ndarray, pivots: dict[int, np.ndarray], p: Optional[int] = None
) -> bool:
    """Legacy wrapper for _is_independent_wrt (avoid re-packing if possible in callers).

    What is Being Computed?:
        Checks if vector `v` is independent of a set of pivot vectors stored 
        in a dictionary.

    Algorithm:
        1. Copy `v` and take modulo `p` if provided.
        2. If `pivots` dictionary is empty, find the first non-zero entry, 
           normalize `v`, store it in `pivots`, and return True.
        3. Convert the `pivots` dictionary to arrays.
        4. Delegate to `_is_independent_wrt_mod_p_kernel` or perform field 
           reduction if `p` is None.
        5. Update the `pivots` dictionary with any changes and return independence result.

    Preserved Invariants:
        - The span of the vectors in the `pivots` dictionary is maintained.

    Args:
        v: The vector to check.
        pivots: Dictionary mapping pivot column index to pivot vector.
        p: Optional modulus.

    Returns:
        bool: True if the vector was independent and added to the dictionary.

    Use When:
        - Legacy code where basis is stored as a dictionary of indices.
        - Incremental basis building where zero-allocation optimization is not needed.

    Example:
        is_new = _is_independent_wrt(vec, my_pivots, p=5)
    """
    # This remains for backward compatibility but callers should migrate to _is_independent_wrt_optimized
    work = np.asarray(v, dtype=np.int64).copy()
    if p is not None:
        p = int(p)
        work %= p
        
        if not pivots:
            first_nz = np.where(work != 0)[0]
            if len(first_nz) > 0:
                idx = int(first_nz[0])
                try:
                    inv = pow(int(work[idx]), -1, p)
                    work = (work * inv) % p
                except ValueError:
                    pass
                pivots[idx] = work
                return True
            return False

        indices = np.array(sorted(pivots.keys()), dtype=np.int64)
        rows = np.array([pivots[i] for i in indices], dtype=np.int64)
        
        new_pivot_idx, final_work = _is_independent_wrt_mod_p_kernel(work, indices, rows, len(indices), p)
        
        for i, idx in enumerate(indices):
            pivots[int(idx)] = rows[i]
            
        if new_pivot_idx != -1:
            try:
                inv = pow(int(final_work[new_pivot_idx]), -1, p)
                final_work = (final_work * inv) % p
            except ValueError:
                pass
            pivots[int(new_pivot_idx)] = final_work
            return True
        return False
    else:
        # Field reduction
        work = work.astype(float)
        for i in range(len(work)):
            if abs(work[i]) < 1e-14:
                continue
            if i not in pivots:
                work /= work[i]
                pivots[i] = work
                return True
            factor = work[i] / pivots[i][i]
            work -= factor * pivots[i]
        return False


def _matrix_rank_for_ring(
    matrix: csr_matrix, ring_kind: str, p: int | None = None, backend: str = "auto"
) -> int:
    """Compute matrix rank in the requested coefficient field with backend selection.

    What is Being Computed?:
        The rank of a sparse matrix over a specified ring or field (Q or Z/pZ).

    Algorithm:
        1. If the matrix is empty or has no non-zeros, return 0.
        2. Normalize backend selection (auto, julia, or python).
        3. If `ring_kind` is 'Q':
           a. Try using Julia's sparse rank computation over Q if available.
           b. Fallback to NumPy's dense rank on the float-converted matrix.
        4. If `ring_kind` is 'ZMOD':
           a. Validate modulus `p`.
           b. Try using Julia's sparse rank computation over Z/pZ if available.
           c. Fallback to Python's Euclidean elimination on the dense matrix.
        5. Warn if performing large dense rank computations.

    Preserved Invariants:
        - Rank is a fundamental invariant of the linear map represented by the matrix.

    Args:
        matrix: The sparse CSR matrix.
        ring_kind: One of 'Q' (Rationals) or 'ZMOD' (Finite ring/field).
        p: Modulus for 'ZMOD'; ignored for 'Q'.
        backend: Backend selector ('auto', 'julia', or 'python').

    Returns:
        int: The computed rank.

    Use When:
        - Computing Betti numbers in homology.
        - Calculating ranks of boundary operators over different coefficient rings.

    Example:
        r = _matrix_rank_for_ring(d2, "ZMOD", p=2)
    """
    if matrix is None or matrix.nnz == 0:
        return 0

    # Normalize backend
    backend_norm = str(backend).lower().strip()
    use_julia = (backend_norm == "julia") or (backend_norm == "auto" and julia_engine.available)

    if ring_kind == "Q":
        if use_julia:
            try:
                return int(julia_engine.compute_sparse_rank_q(matrix))
            except Exception as exc:
                if backend_norm == "julia":
                    raise exc
                warnings.warn(
                    "Topological Hint: Julia rank over Q failed in `ChainComplex.homology`; "
                    f"falling back to NumPy dense rank ({exc!r})."
                )
        
        if matrix.shape[0] * matrix.shape[1] > 1000000:
             warnings.warn(f"Topological Warning: Dense rank computation over Q for matrix {matrix.shape}. Enable Julia for sparse support.")
        return int(np.linalg.matrix_rank(matrix.toarray().astype(float)))

    if ring_kind == "ZMOD":
        if p is None:
            raise ValueError("Prime modulus p is required for Z/pZ rank computation.")
        if use_julia:
            try:
                return int(julia_engine.compute_sparse_rank_mod_p(matrix, int(p)))
            except Exception as exc:
                if backend_norm == "julia":
                    raise exc
                warnings.warn(
                    "Topological Hint: Julia rank over Z/pZ failed in `ChainComplex.homology`; "
                    f"falling back to Python elimination ({exc!r})."
                )
        
        if matrix.shape[0] * matrix.shape[1] > 1000000:
             warnings.warn(f"Topological Warning: Dense rank computation over Z/{p}Z for matrix {matrix.shape}. Enable Julia for sparse support.")
        return _rank_mod_p(matrix.toarray(), int(p))

    raise ValueError(f"Unsupported rank ring kind '{ring_kind}'.")


def _composite_mod_uct_decomposition(
    free_rank: int,
    torsion_n: List[int],
    torsion_nm1: List[int],
    modulus: int,
) -> Tuple[int, List[int]]:
    """Compute Z/n decomposition from integral data via UCT tensor/Tor terms.

    What is Being Computed?:
        The structure of the homology group H_n(X; Z/mZ) given the integral 
        homology H_n(X; Z) and H_{n-1}(X; Z).

    Algorithm:
        1. Start with the free rank of H_n(X; Z).
        2. Apply the Universal Coefficient Theorem (UCT):
           H_n(X; Z/mZ) ≅ (H_n(X; Z) ⊗ Z/mZ) ⊕ Tor(H_{n-1}(X; Z), Z/mZ).
        3. For each torsion coefficient t in H_n(Z), compute g = gcd(t, m). If g = m, 
           it contributes to the free part of H_n(Z/mZ); if 1 < g < m, it 
           contributes a Z/gZ factor.
        4. Repeat the same process for torsion coefficients in H_{n-1}(Z) to 
           account for the Tor term.
        5. Return the total rank and sorted list of torsion coefficients.

    Preserved Invariants:
        - Homological structure under coefficient change.

    Args:
        free_rank: Free rank of H_n(Z).
        torsion_n: Torsion coefficients of H_n(Z).
        torsion_nm1: Torsion coefficients of H_{n-1}(Z).
        modulus: The modulus m for Z/mZ coefficients.

    Returns:
        tuple[int, list[int]]: (rank_mod, torsion_mod) representing H_n(Z/mZ).

    Use When:
        - Computing homology with composite Z/mZ coefficients without direct reduction.
        - Deriving modular homology from previously computed integral homology.

    Example:
        rank, tors = _composite_mod_uct_decomposition(1, [2], [2], 2)
        # H_n(Z)=Z+Z/2, H_{n-1}(Z)=Z/2, modulus=2 => (1+1+1, []) = (3, [])
    """
    rank_mod = int(free_rank)
    torsion_mod: List[int] = []

    # Tensor terms from H_n(Z): Z_t ⊗ Z_n ≅ Z_gcd(t, n)
    for t in torsion_n:
        g = int(np.gcd(int(t), modulus))
        if g <= 1:
            continue
        if g == modulus:
            rank_mod += 1
        else:
            torsion_mod.append(g)

    # Tor terms from H_{n-1}(Z): Tor(Z_t, Z_n) ≅ Z_gcd(t, n)
    for t in torsion_nm1:
        g = int(np.gcd(int(t), modulus))
        if g <= 1:
            continue
        if g == modulus:
            rank_mod += 1
        else:
            torsion_mod.append(g)

    return rank_mod, sorted(torsion_mod)


class ChainComplex(BaseModel):
    """An abstract Chain Complex C_* over Z (or other rings).

    Overview:
        A ChainComplex represents a sequence of modules (chain groups) and linear 
        maps (boundary operators) between them, such that the composition of any 
        two consecutive maps is zero (∂_n ∘ ∂_{n+1} = 0). It serves as the 
        fundamental algebraic object for computing homology and cohomology 
        groups of topological spaces.

    Key Concepts:
        - **Chain Groups (C_n)**: Free Z-modules generated by cells of dimension n.
        - **Boundary Maps (∂_n)**: Linear maps C_n → C_{n-1} represented as sparse matrices.
        - **Cycles (Z_n)**: Elements in the kernel of ∂_n (chains with no boundary).
        - **Boundaries (B_n)**: Elements in the image of ∂_{n+1}.
        - **Homology (H_n)**: The quotient group Z_n / B_n, measuring "holes".
        - **Smith Normal Form (SNF)**: The matrix reduction algorithm used to compute homology over Z.

    Common Workflows:
        1. **Initialization** → Directly from boundary matrices and cell counts.
        2. **Homology computation** → Use homology() to get rank and torsion.
        3. **Cohomology computation** → Use cohomology() and cohomology_basis().
        4. **Invariants summary** → Use topological_invariants() for a comprehensive report.

    Coefficient Ring:
        - 'Z' (default): Integer coefficients (exact computation with torsion).
        - 'Q': Rational coefficients (only ranks/Betti numbers).
        - 'Z/pZ': Modular coefficients (for prime p or composite moduli).

    Attributes:
        boundaries (Dict[int, csr_matrix]): Mapping from dimension n to the boundary map ∂_n.
        dimensions (List[int]): Sorted list of dimensions present in the complex.
        cells (Dict[int, int]): Mapping from dimension n to the rank of the chain group C_n.
        coefficient_ring (str): The ring or field over which computations are performed.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    boundaries: Dict[int, csr_matrix]
    dimensions: List[int]
    cells: Dict[int, int] = Field(default_factory=dict)
    coefficient_ring: str = "Z"

    _cache_enabled: bool = PrivateAttr(default=True)
    _cache: dict[tuple[object, ...], object] = PrivateAttr(default_factory=dict)
    _cache_hits: int = PrivateAttr(default=0)
    _cache_misses: int = PrivateAttr(default=0)
    _cache_signature: tuple[object, ...] | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _normalize_model(self):
        """Validate and normalize the chain complex model data."""
        object.__setattr__(
            self,
            "boundaries",
            {int(dim): _coerce_csr_matrix(mat) for dim, mat in self.boundaries.items()},
        )
        object.__setattr__(
            self, "dimensions", sorted({int(dim) for dim in self.dimensions})
        )
        object.__setattr__(
            self, "cells", {int(dim): int(count) for dim, count in self.cells.items()}
        )
        object.__setattr__(self, "coefficient_ring", str(self.coefficient_ring))
        return self

    def _structure_signature(self) -> tuple[object, ...]:
        """Return a signature representing the structural state of the complex.

        Returns:
            A tuple of structural signatures.
        """
        boundary_sig = tuple(
            (int(dim), _csr_matrix_signature(mat))
            for dim, mat in sorted(self.boundaries.items())
        )
        cells_sig = tuple(
            (int(dim), int(count)) for dim, count in sorted(self.cells.items())
        )
        return (
            boundary_sig,
            tuple(self.dimensions),
            cells_sig,
            str(self.coefficient_ring),
        )

    def _ensure_cache_valid(self) -> None:
        """Clear the cache if the complex structure has changed."""
        current = self._structure_signature()
        if self._cache_signature != current:
            self._cache.clear()
            self._cache_signature = current

    def _cache_get(self, key: tuple[object, ...]) -> object | None:
        """Retrieve a value from the cache.

        Args:
            key: The cache key.

        Returns:
            The cached value or None if not found.
        """
        self._ensure_cache_valid()
        if not self._cache_enabled:
            return None
        if key in self._cache:
            self._cache_hits += 1
            return _clone_cache_value(self._cache[key])
        self._cache_misses += 1
        return None

    def _cache_set(self, key: tuple[object, ...], value: object) -> None:
        """Store a value in the cache.

        Args:
            key: The cache key.
            value: The value to store.
        """
        self._ensure_cache_valid()
        if not self._cache_enabled:
            return
        self._cache[key] = _clone_cache_value(value)

    def clear_cache(self, namespace: str | None = None) -> None:
        """Clear the cache, optionally filtered by namespace.

        Args:
            namespace: Optional namespace prefix to clear.
        """
        if namespace is None:
            self._cache.clear()
            return
        prefix = (str(namespace),)
        keys = [k for k in self._cache if k[:1] == prefix]
        for key in keys:
            self._cache.pop(key, None)

    def cache_info(self) -> dict[str, object]:
        """Return information about the cache state.

        Returns:
            A dictionary containing cache statistics.
        """
        self._ensure_cache_valid()
        return {
            "enabled": bool(self._cache_enabled),
            "size": int(len(self._cache)),
            "hits": int(self._cache_hits),
            "misses": int(self._cache_misses),
            "keys": [
                list(map(str, key)) for key in sorted(self._cache.keys(), key=repr)
            ],
        }

    def set_cache_enabled(
        self, enabled: bool, *, clear_when_disabled: bool = True
    ) -> None:
        """Enable or disable caching.

        Args:
            enabled: Whether to enable caching.
            clear_when_disabled: Whether to clear the cache when disabling.
        """
        self._cache_enabled = bool(enabled)
        if not self._cache_enabled and clear_when_disabled:
            self._cache.clear()

    def _homological_dimensions(self) -> List[int]:
        """Return sorted degrees that have meaningful chain data.

        Returns:
            Sorted list of homological dimensions.
        """
        key = ("chain", "homological_dimensions")
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        dims: set[int] = set()
        dims.update(int(dim) for dim in self.dimensions if int(dim) >= 0)
        dims.update(int(dim) for dim in self.cells.keys() if int(dim) >= 0)
        for dim in self.boundaries.keys():
            dim_int = int(dim)
            if dim_int >= 0:
                dims.add(dim_int)
            if dim_int - 1 >= 0:
                dims.add(dim_int - 1)

        out = sorted(dims)
        self._cache_set(key, out)
        return out

    def _homology_over_z(self, n: int, backend: str = "auto") -> Tuple[int, List[int]]:
        """Exact integral homology helper used by coefficient-change formulas.

        Args:
            n: Homological degree.
            backend: 'auto', 'julia', or 'python'.

        Returns:
            A tuple (rank, torsion) for H_n(C; Z).
        """
        n = int(n)
        key = ("chain", "homology_over_z", n, backend)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        dn = self.boundaries.get(n)
        dn_plus_1 = self.boundaries.get(n + 1)

        if (
            n not in self.dimensions
            and n not in self.cells
            and dn is None
            and dn_plus_1 is None
        ):
            out = (0, [])
            self._cache_set(key, out)
            return out

        if n in self.cells:
            c_n_size = self.cells[n]
        elif dn is not None:
            c_n_size = dn.shape[1]
        elif dn_plus_1 is not None:
            c_n_size = dn_plus_1.shape[0]
        else:
            out = (0, [])
            self._cache_set(key, out)
            return out

        if dn is not None and dn.nnz > 0:
            snf_n = get_sparse_snf_diagonal(dn, backend=backend)
            rank_n = np.count_nonzero(snf_n)
        else:
            rank_n = 0
        dim_ker_n = c_n_size - rank_n

        if dn_plus_1 is not None and dn_plus_1.nnz > 0:
            snf_n_plus_1 = get_sparse_snf_diagonal(dn_plus_1, backend=backend)
            rank_im_n_plus_1 = np.count_nonzero(snf_n_plus_1)
            torsion = [int(x) for x in snf_n_plus_1 if x > 1]
            if not torsion and any(x == 1 for x in snf_n_plus_1):
                warnings.warn(
                    "Integral homology fallback in `ChainComplex.homology`: torsion may be underestimated without exact Julia sparse SNF; "
                    "install/enable Julia for faster and more reliable exact torsion extraction."
                )
        else:
            rank_im_n_plus_1 = 0
            torsion = []
        betti_n = max(0, dim_ker_n - rank_im_n_plus_1)
        out = (int(betti_n), torsion)
        self._cache_set(key, out)
        return out

    def homology(
        self, n: int | None = None, backend: str = "auto"
    ) -> Tuple[int, List[int]] | Dict[int, Tuple[int, List[int]]]:
        """Compute homology groups H_n(C) = ker(∂_n) / im(∂_{n+1}), measuring n-dimensional "holes".
        
        What is Homology?:
            Homology groups measure the topological features of a space (holes, voids, etc.) in each 
            dimension. H_0 counts connected components, H_1 counts 1-dimensional holes (loops), H_2 
            counts 2-dimensional voids, etc. Each H_n is expressed as (rank, torsion): the free part 
            (rank over ℤ) and the torsion coefficients.
        
        Algorithm:
            1. Extract the boundary matrices ∂_n and ∂_{n+1} from the chain complex
            2. Compute the image of ∂_{n+1} (cycles that are boundaries of (n+1)-chains)
            3. Compute the kernel of ∂_n (n-chains with no boundary)
            4. Form H_n = ker(∂_n) / im(∂_{n+1}) using Smith Normal Form (Julia or Python backend)
            5. Return (free_rank, torsion_list) where torsion_list contains torsion coefficients
        
        Preserved Invariants:
            - Homotopy equivalent complexes have isomorphic homology groups
            - Betti numbers (free ranks) are topological invariants
            - Torsion structure detects higher-order phenomena (e.g., fundamental group imperfections)
        
        Args:
            n: Homological degree (int). If None, computes homology for all positive degrees.
            backend: 'auto' (tries Julia, falls back to Python), 'julia', or 'python'.
        
        Returns:
            If n is int: Tuple (rank, torsion) where rank is the free rank and torsion is a 
                         list of torsion coefficients (e.g., [2, 2] means ℤ/2ℤ ⊕ ℤ/2ℤ).
            If n is None: Dict[int → (rank, torsion)] for all degrees in the complex.
        
        Use When:
            - You need to classify the topology of a space (e.g., sphere vs. torus)
            - Computing Betti numbers, Euler characteristic, or persistence diagrams
            - Checking homotopy equivalence: identical homology is necessary (but not sufficient)
            - Julia backend recommended for complexes with >1000 simplices
        
        Example:
            h0 = sc.homology(0)  # (1, []) means 1 connected component, no torsion
            h1 = sc.homology(1)  # (1, []) means 1 "hole" (fundamental group has rank 1)
            all_homology = sc.homology()  # {0: (1, []), 1: (1, []), ...}
        """
        if n is None:
            key_all = ("chain", "homology", "all", str(self.coefficient_ring), backend)
            cached_all = self._cache_get(key_all)
            if cached_all is not None:
                return cached_all
            out_all = {dim: self.homology(dim, backend=backend) for dim in self._homological_dimensions()}
            self._cache_set(key_all, out_all)
            return out_all

        n = int(n)
        key = ("chain", "homology", n, str(self.coefficient_ring), backend)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        ring_kind, p = _parse_coefficient_ring(self.coefficient_ring)

        if ring_kind == "ZMOD" and p is not None and not _is_prime(int(p)):
            r_n, t_n = self._homology_over_z(n, backend=backend)
            _, t_nm1 = self._homology_over_z(n - 1, backend=backend)
            modulus = int(p)

            rank_mod, torsion_mod = _composite_mod_uct_decomposition(
                r_n, t_n, t_nm1, modulus
            )
            out = (int(rank_mod), torsion_mod)
            self._cache_set(key, out)
            return out

        dn = self.boundaries.get(n)
        dn_plus_1 = self.boundaries.get(n + 1)

        if (
            n not in self.dimensions
            and n not in self.cells
            and dn is None
            and dn_plus_1 is None
        ):
            out = (0, [])
            self._cache_set(key, out)
            return out

        if n in self.cells:
            c_n_size = self.cells[n]
        elif dn is not None:
            c_n_size = dn.shape[1]
        elif dn_plus_1 is not None:
            c_n_size = dn_plus_1.shape[0]
        else:
            out = (0, [])
            self._cache_set(key, out)
            return out

        if dn is not None and dn.nnz > 0:
            if ring_kind == "Z":
                snf_n = get_sparse_snf_diagonal(dn, backend=backend)
                rank_n = np.count_nonzero(snf_n)
            elif ring_kind == "Q":
                rank_n = _matrix_rank_for_ring(dn, "Q", backend=backend)
            else:
                rank_n = _matrix_rank_for_ring(dn, "ZMOD", int(p), backend=backend)
        else:
            rank_n = 0

        dim_ker_n = c_n_size - rank_n

        if dn_plus_1 is not None and dn_plus_1.nnz > 0:
            if ring_kind == "Z":
                snf_n_plus_1 = get_sparse_snf_diagonal(dn_plus_1, backend=backend)
                rank_im_n_plus_1 = np.count_nonzero(snf_n_plus_1)
                torsion = [int(x) for x in snf_n_plus_1 if x > 1]
                if not torsion and any(x == 1 for x in snf_n_plus_1):
                    warnings.warn(
                        "Torsion certification may be incomplete for this complex; the sparse integer reduction returned"
                        " only unit factors, so torsion could not be fully resolved."
                    )
            elif ring_kind == "Q":
                rank_im_n_plus_1 = _matrix_rank_for_ring(dn_plus_1, "Q", backend=backend)
                torsion = []
            else:
                rank_im_n_plus_1 = _matrix_rank_for_ring(dn_plus_1, "ZMOD", int(p), backend=backend)
                torsion = []
        else:
            rank_im_n_plus_1 = 0
            torsion = []
        betti_n = max(0, dim_ker_n - rank_im_n_plus_1)
        out = (int(betti_n), torsion)
        self._cache_set(key, out)
        return out

    def cohomology(
        self, n: int | None = None, backend: str = "auto"
    ) -> Tuple[int, List[int]] | Dict[int, Tuple[int, List[int]]]:
        """Compute cohomology groups H^n(C), the dual of homology with cup product structure.
        
        What is Cohomology?:
            Cohomology is the dual notion to homology, living in H^n(C) = Hom(H_n(C), ℤ). For CW 
            complexes over ℤ, H^n ≅ H_n, BUT cohomology carries additional structure: the cup product 
            (∪), which encodes higher-order intersection information. This matters for manifolds, 
            characteristic classes, and surgery theory.
        
        Algorithm:
            1. Compute homology H_n(C) and H_{n-1}(C) to obtain rank and torsion
            2. By the Universal Coefficient Theorem over ℤ:
               H^n(C; ℤ) ≅ Hom(H_n(C), ℤ) ⊕ Ext¹(H_{n-1}(C), ℤ)
               which simplifies to: H^n has rank = rank(H_n), torsion = torsion(H_{n-1})
            3. Cache result; subsequent queries use cached homology data
        
        Preserved Invariants:
            - Cohomology groups are topological invariants (homotopy equivalence preserves them)
            - Cup product ring structure is preserved under homotopy equivalence
            - Characteristic classes derived from cohomology are stable topological invariants
        
        Args:
            n: Cohomological degree. If None, computes for all degrees.
            backend: 'auto', 'julia', or 'python' (used when computing underlying homology).
        
        Returns:
            If n is int: Tuple (rank, torsion) representing H^n(C).
            If n is None: Dict[int → (rank, torsion)] for all degrees.
        
        Use When:
            - You need cup product structure or characteristic classes
            - Working with manifolds (where cup product gives intersection form)
            - Computing Stiefel-Whitney or Pontryagin classes
            - Stuifying Poincaré duality in manifold surgery
        
        Example:
            rank, torsion = sc.cohomology(1)  # H^1: copies of ℤ (plus torsion)
            # On a closed surface: H^1 has rank = 2g (g = genus), H^2 has rank = 1
        """
        if n is None:
            key_all = ("chain", "cohomology", "all", str(self.coefficient_ring), backend)
            cached_all = self._cache_get(key_all)
            if cached_all is not None:
                return cached_all
            out_all = {
                dim: self.cohomology(dim, backend=backend) for dim in self._homological_dimensions()
            }
            self._cache_set(key_all, out_all)
            return out_all

        n = int(n)
        key = ("chain", "cohomology", n, str(self.coefficient_ring), backend)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        ring_kind, _ = _parse_coefficient_ring(self.coefficient_ring)
        if ring_kind == "ZMOD":
            _, p = _parse_coefficient_ring(self.coefficient_ring)
            if p is not None and not _is_prime(int(p)):
                r_n, t_n = self._homology_over_z(n, backend=backend)
                _, t_nm1 = self._homology_over_z(n - 1, backend=backend)
                modulus = int(p)
                rank_mod, torsion_mod = _composite_mod_uct_decomposition(
                    r_n, t_n, t_nm1, modulus
                )
                out = (int(rank_mod), torsion_mod)
                self._cache_set(key, out)
                return out
        free_rank, _ = self.homology(n, backend=backend)
        if ring_kind == "Z":
            _, prev_torsion = self.homology(n - 1)
            out = (free_rank, prev_torsion)
            self._cache_set(key, out)
            return out
        out = (free_rank, [])
        self._cache_set(key, out)
        return out

    def _chain_group_rank_for_degree(self, n: int) -> int:
        """Return the rank of the chain group C_n.

        Args:
            n: Homological degree.

        Returns:
            Rank of C_n.
        """
        n = int(n)
        if n in self.cells:
            return int(self.cells[n])

        dn = self.boundaries.get(n)
        if dn is not None:
            return int(dn.shape[1])

        dn_plus_1 = self.boundaries.get(n + 1)
        if dn_plus_1 is not None:
            return int(dn_plus_1.shape[0])

        return 0

    def rank(self, n: int | None = None) -> int | Dict[int, int]:
        """Return the rank of the n-th chain group C_n.

        Args:
            n: Optional degree.

        Returns:
            Rank or dictionary of ranks.
        """
        if n is None:
            key_all = ("chain", "rank", "all", str(self.coefficient_ring))
            cached_all = self._cache_get(key_all)
            if cached_all is not None:
                return cached_all
            out_all = {
                dim: self._chain_group_rank_for_degree(dim)
                for dim in self._homological_dimensions()
            }
            self._cache_set(key_all, out_all)
            return out_all

        n = int(n)
        key = ("chain", "rank", n, str(self.coefficient_ring))
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        out = int(self._chain_group_rank_for_degree(n))
        self._cache_set(key, out)
        return out

    def betti_number(self, n: int | None = None, backend: str = "auto") -> int | Dict[int, int]:
        """Return the n-th Betti number β_n, the free rank of H_n(C).
        
        What is the Betti Number?:
            The n-th Betti number β_n = rank(H_n(C)) is the number of independent n-dimensional 
            "holes" in the space. For example:
            - β₀ = number of connected components
            - β₁ = 2g on a closed surface of genus g (counts loops)
            - β₂ = number of 2-dimensional voids
            - Euler characteristic χ = Σ (-1)^n β_n
        
        Algorithm:
            1. Compute homology H_n(C) as (rank, torsion)
            2. Extract and return the free rank component
            3. Cache result for efficiency
        
        Preserved Invariants:
            - Betti numbers are homotopy invariants (homotopy equivalent spaces have identical Betti numbers)
            - They are intrinsic properties that don't depend on triangulation
            - Provide complete topological classification for simply-connected spaces (e.g., spheres)
        
        Args:
            n: Homological degree. If None, returns Betti numbers for all degrees.
            backend: 'auto', 'julia', or 'python'.
        
        Returns:
            If n is int: The Betti number β_n (a non-negative integer).
            If n is None: Dict[int → β_n] for all degrees.
        
        Use When:
            - Quick summary of topology without full torsion info
            - Euler characteristic: χ = Σ (-1)^n β_n
            - Classification: e.g., closed surfaces differ by Betti number
            - Need a single topological "fingerprint" per dimension
        
        Example:
            β1 = sc.betti_number(1)  # Rank of the 1st homology group
            all_bettis = sc.betti_numbers()  # {0: 1, 1: 2, 2: 1} for a torus
        """
        if n is None:
            key_all = ("chain", "betti_number", "all", str(self.coefficient_ring), backend)
            cached_all = self._cache_get(key_all)
            if cached_all is not None:
                return cached_all
            hom_all = self.homology(backend=backend)
            out_all = {dim: rank for dim, (rank, _) in hom_all.items()}
            self._cache_set(key_all, out_all)
            return out_all

        n = int(n)
        key = ("chain", "betti_number", n, str(self.coefficient_ring), backend)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        rank, _ = self.homology(n, backend=backend)
        out = int(rank)
        self._cache_set(key, out)
        return out

    def betti_numbers(self, backend: str = "auto") -> Dict[int, int]:
        """Return all Betti numbers β₀, β₁, β₂, ... as a dictionary.
        
        Algorithm:
            Calls betti_number(n=None) to obtain {degree: β_n} for all dimensions in the complex.
        
        Use When:
            - Need a complete topological summary
            - Computing Euler characteristic χ = Σ (-1)^n β_n
            - Quick classification or comparison of spaces
        
        Args:
            backend: 'auto', 'julia', or 'python'.
        
        Returns:
            Dict[int → β_n] with Betti numbers for all degrees.
        
        Example:
            bettis = sc.betti_numbers()  # {0: 1, 1: 1, 2: 0}
            χ = sum((-1)**n * β for n, β in bettis.items())
        """
        out = self.betti_number(backend=backend)
        return out

    def cohomology_basis(self, n: int, backend: str = "auto") -> list[np.ndarray]:
        """Computes a basis for the free part of the n-th cohomology group H^n(C).

        Args:
            n: Degree.
            backend: 'auto', 'julia', or 'python'.

        Returns:
            List of cochain vectors forming a basis.
        """
        n = int(n)
        key = ("chain", "cohomology_basis", n, str(self.coefficient_ring), backend)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        dn_plus_1 = self.boundaries.get(n + 1)
        dn = self.boundaries.get(n)

        if n in self.cells:
            cn_size = self.cells[n]
        elif dn is not None:
            cn_size = dn.shape[1]
        elif dn_plus_1 is not None:
            cn_size = dn_plus_1.shape[0]
        else:
            out: List[np.ndarray] = []
            self._cache_set(key, out)
            return out

        ring_kind, p = _parse_coefficient_ring(self.coefficient_ring)

        # Normalize backend
        backend_norm = str(backend).lower().strip()
        use_julia = (backend_norm == "julia") or (backend_norm == "auto" and julia_engine.available)

        if use_julia:
            try:
                # Delegate to Julia for exact/fast sparse SNF-based basis
                if ring_kind == "Z":
                    d_np1_coo = dn_plus_1.tocoo() if dn_plus_1 is not None and dn_plus_1.nnz > 0 else None
                    
                    if d_np1_coo is not None:
                        # Julia expects: d_np1, d_n, cn_size
                        # We pass boundaries[n+1] as the first map
                        basis_raw = julia_engine.compute_sparse_cohomology_basis(
                            d_np1_coo.row, d_np1_coo.col, d_np1_coo.data,
                            cn_size=cn_size
                        )
                        out = [np.asarray(v) for v in basis_raw]
                        self._cache_set(key, out)
                        return out
                elif ring_kind == "ZMOD":
                    # Julia mod-p implementation hook
                    pass
            except Exception as e:
                if backend_norm == "julia":
                    raise e

        if ring_kind == "ZMOD" and p is not None and not _is_prime(int(p)):
            integral_complex = ChainComplex(
                boundaries=self.boundaries,
                dimensions=self.dimensions,
                cells=self.cells,
                coefficient_ring="Z",
            )
            basis_z = integral_complex.cohomology_basis(n, backend=backend)
            out = []
            modulus = int(p)
            target_rank, _ = self.cohomology(n, backend=backend)
            for v in basis_z[:target_rank]:
                out.append(np.asarray(v, dtype=np.int64) % modulus)
            while len(out) < target_rank and cn_size > 0:
                e = np.zeros(cn_size, dtype=np.int64)
                e[len(out) % cn_size] = 1
                out.append(e)
            self._cache_set(key, out)
            return out

        if ring_kind in {"Q", "ZMOD"}:
            if julia_engine.available:
                try:
                    if ring_kind == "Q":
                        out = julia_engine.compute_sparse_cohomology_basis(
                            dn_plus_1, dn, cn_size=cn_size
                        )
                    else:
                        out = julia_engine.compute_sparse_cohomology_basis_mod_p(
                            dn_plus_1, dn, int(p), cn_size=cn_size
                        )
                    self._cache_set(key, out)
                    return out
                except Exception as exc:
                    warnings.warn(
                        f"Topological Hint: Julia field cohomology basis backend failed; "
                        f"falling back to Python dense implementation ({exc!r})."
                    )

            if cn_size > 10000:
                warnings.warn(
                    f"Topological Warning: Performing dense cohomology reduction for size {cn_size}. "
                    "This may exceed memory limits. Enable Julia for sparse support."
                )

            if dn_plus_1 is None or dn_plus_1.nnz == 0:
                z_basis = [np.eye(cn_size, dtype=np.int64)[j] for j in range(cn_size)]
            else:
                if ring_kind == "Q":
                    from scipy.linalg import null_space
                    # Use SVD-based nullspace for stability over Q
                    ns = null_space(dn_plus_1.T.toarray().astype(float))
                    z_basis = [ns[:, j] for j in range(ns.shape[1])]
                else:
                    z_basis = _nullspace_basis_mod_p(dn_plus_1.T.toarray(), int(p))

            mod_p = int(p) if ring_kind == "ZMOD" else None
            target_rank, _ = self.cohomology(n)

            # Optimized incremental reduction using pre-allocated matrix
            max_pivots = cn_size
            pivot_matrix = np.zeros((max_pivots, cn_size), dtype=np.int64 if mod_p else float)
            pivot_indices = np.zeros(max_pivots, dtype=np.int64)
            n_pivots = 0

            if dn is not None:
                dn_T_arr = dn.T.toarray()
                for j in range(dn_T_arr.shape[1]):
                    _, n_pivots = _is_independent_wrt_optimized(
                        dn_T_arr[:, j], pivot_matrix, pivot_indices, n_pivots, p=mod_p
                    )

            reps = []
            for z in z_basis:
                if len(reps) >= target_rank:
                    break
                is_indep, n_pivots = _is_independent_wrt_optimized(
                    z, pivot_matrix, pivot_indices, n_pivots, p=mod_p
                )
                if is_indep:
                    reps.append(z)

            out = []
            for v in reps:
                arr = np.array(v, dtype=np.int64).flatten()
                if ring_kind == "ZMOD":
                    arr = arr % int(p)
                out.append(arr)
            self._cache_set(key, out)
            return out

        if julia_engine.available:
            try:
                out = julia_engine.compute_sparse_cohomology_basis(
                    dn_plus_1, dn, cn_size=cn_size
                )
                self._cache_set(key, out)
                return out
            except Exception as e:
                warnings.warn(f"Topological Hint: Julia bridge failed ({e!r}). Falling back to pure Python.")

        if cn_size > 5000:
            warnings.warn(f"Topological Warning: Large integral SNF reduction for C^{n} (size {cn_size}). Expect slowdowns.")

        # Robust integral fallback using Smith Normal Form on dn_plus_1^T to get Z^n
        # then independent check against im(dn^T)
        if dn_plus_1 is not None and dn_plus_1.nnz > 0:
            from .math_core import smith_normal_decomp
            dnp1_T = sp.SparseMatrix(dn_plus_1.shape[1], dn_plus_1.shape[0], dict(dn_plus_1.T.todok().items()))
            S, U, V = smith_normal_decomp(dnp1_T)
            # Z^n = ker(dn_plus_1^T). Columns of V corresponding to zero diagonal in S.
            z_basis = []
            for j in range(V.shape[1]):
                if j >= S.shape[0] or S[j, j] == 0:
                    z_basis.append(np.array(V[:, j], dtype=np.int64).flatten())
        else:
            z_basis = [np.eye(cn_size, dtype=np.int64)[j] for j in range(cn_size)]
        # Optimized integral incremental reduction
        max_pivots = cn_size
        pivot_matrix = np.zeros((max_pivots, cn_size), dtype=np.int64)
        pivot_indices = np.zeros(max_pivots, dtype=np.int64)
        n_pivots = 0

        if dn is not None and dn.nnz > 0:
            dn_T_arr = dn.T.toarray()
            for j in range(dn_T_arr.shape[1]):
                _is_independent_wrt_optimized(dn_T_arr[:, j], pivot_matrix, pivot_indices, n_pivots)

        target_rank, _ = self.cohomology(n)
        reps = []
        for z in z_basis:
            if len(reps) >= target_rank:
                break
            is_indep, n_pivots = _is_independent_wrt_optimized(z, pivot_matrix, pivot_indices, n_pivots)
            if is_indep:
                reps.append(z)
        
        self._cache_set(key, reps)
        return reps

    def euler_characteristic(self) -> int:
        """Compute the Euler characteristic of the chain complex.

        chi(C) = sum_{n} (-1)^n * rank(C_n).

        Returns:
            Euler characteristic.
        """
        key = ("chain", "euler_characteristic")
        cached = self._cache_get(key)
        if cached is not None:
            return int(cast(int, cached))

        chi = 0
        for dim in self._homological_dimensions():
            rank = self._chain_group_rank_for_degree(dim)
            if dim % 2 == 0:
                chi += rank
            else:
                chi -= rank

        self._cache_set(key, int(chi))
        return int(chi)

    def topological_invariants(self, backend: str = "auto") -> Dict[str, Any]:
        """Compute all key topological invariants at once.

        Args:
            backend: 'auto', 'julia', or 'python'.

        Returns:
            A dictionary containing homology, cohomology, Betti numbers,
            and the Euler characteristic.
        """
        key = ("chain", "topological_invariants", str(self.coefficient_ring), backend)
        cached = self._cache_get(key)
        if cached is not None:
            return cast(Dict[str, Any], cached)

        homology = self.homology(backend=backend)
        cohomology = self.cohomology(backend=backend)
        betti = self.betti_numbers(backend=backend)
        chi = self.euler_characteristic()

        out = {
            "homology": homology,
            "cohomology": cohomology,
            "betti_numbers": betti,
            "euler_characteristic": chi,
            "coefficient_ring": self.coefficient_ring,
        }

        self._cache_set(key, out)
        return out


class CWComplex(BaseModel):
    """Representation of a Finite CW Complex X.

    Overview:
        A CWComplex models a topological space built inductively by attaching n-cells 
        to the (n-1)-skeleton. It is a highly flexible representation that generalizes 
        simplicial complexes, allowing for more efficient descriptions of spaces (e.g., 
        a torus as 1 vertex, 2 edges, and 1 face). This class maintains the cellular 
        structure and provides tools to compute its cellular homology.

    Key Concepts:
        - **n-Cells**: The basic building blocks (disk-like components of dimension n).
        - **Attaching Maps**: The boundary operators that describe how n-cells are glued to the (n-1)-skeleton.
        - **Cellular Chain Complex**: The algebraic structure C_*(X) derived from the cell decomposition.
        - **2-Skeleton**: Sufficient for computing the fundamental group π₁(X).

    Common Workflows:
        1. **From Simplicial** → CWComplex.from_simplicial_complex(sc).
        2. **Cellular Homology** → Compute homology() or betti_numbers() directly.
        3. **Algebraic Lifting** → Obtain the cellular_chain_complex() for advanced homological algebra.
        4. **Fundamental Group** → Pass to extract_pi_1() to compute π₁(X).

    Coefficient Ring:
        - 'Z' (default), 'Q', 'Z/pZ'.

    Attributes:
        cells (Dict[int, int]): Mapping from dimension n to the number of n-cells.
        attaching_maps (Dict[int, csr_matrix]): Mapping from n to the boundary map ∂_n.
        dimensions (List[int]): Dimensions present in the complex.
        coefficient_ring (str): Default coefficient ring for computations.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cells: Dict[int, int]
    attaching_maps: Dict[int, csr_matrix]
    dimensions: List[int] = Field(default_factory=list)
    coefficient_ring: str = "Z"

    _cache: dict[tuple[object, ...], object] = PrivateAttr(default_factory=dict)
    _cache_hits: int = PrivateAttr(default=0)
    _cache_misses: int = PrivateAttr(default=0)
    _cache_signature: tuple[object, ...] | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _normalize_cw(self):
        """Validate and normalize CW complex model data."""
        if not self.dimensions:
             dims = set(self.cells.keys())
             dims.update(self.attaching_maps.keys())
             for d in self.attaching_maps.keys():
                 dims.add(d-1)
             object.__setattr__(self, "dimensions", sorted([d for d in dims if d >= 0]))
        else:
             object.__setattr__(self, "dimensions", sorted({int(dim) for dim in self.dimensions}))
        return self

    @classmethod
    def from_simplicial_complex(cls, sc: "SimplicialComplex") -> "CWComplex":
        """Converts a SimplicialComplex to a CWComplex.

        Each n-simplex is treated as an n-cell.

        Args:
            sc: The input simplicial complex.

        Returns:
            A CWComplex representation.
        """
        # Check if Julia pre-calculated the boundary operators
        if hasattr(sc, "_boundaries_cache") and hasattr(sc, "_cells_cache") and sc._boundaries_cache and sc._cells_cache:
            boundaries = sc._boundaries_cache
            cells = sc._cells_cache
        else:
            boundaries = sc.boundary_matrices()
            cells = {dim: len(simplices) for dim, simplices in sc.simplices_field.items()}
            
        return cls(
            cells=cells,
            attaching_maps=boundaries,
            dimensions=sc.dimensions,
            coefficient_ring=sc.coefficient_ring
        )

    def _structure_signature(self) -> tuple[object, ...]:
        """Return a signature for the structural state of the CW complex.

        Returns:
            A structural signature tuple.
        """
        map_sig = tuple(
            (int(dim), _csr_matrix_signature(mat))
            for dim, mat in sorted(self.attaching_maps.items())
        )
        return map_sig, tuple(self.dimensions), str(self.coefficient_ring)

    def _ensure_cache_valid(self) -> None:
        """Clear the cache if the CW complex structure has changed."""
        current = self._structure_signature()
        if self._cache_signature != current:
            self._cache.clear()
            self._cache_signature = current

    def _cache_get(self, key: tuple[object, ...]) -> object | None:
        """Retrieve a value from the cache.

        Args:
            key: The cache key.

        Returns:
            The cached value or None.
        """
        self._ensure_cache_valid()
        if key in self._cache:
            self._cache_hits += 1
            return _clone_cache_value(self._cache[key])
        self._cache_misses += 1
        return None

    def _cache_set(self, key: tuple[object, ...], value: object) -> None:
        """Store a value in the cache.

        Args:
            key: The cache key.
            value: The value to store.
        """
        self._ensure_cache_valid()
        self._cache[key] = _clone_cache_value(value)

    def cache_info(self) -> dict[str, object]:
        """Return information about the cache state.

        Returns:
            Dictionary with cache stats.
        """
        self._ensure_cache_valid()
        return {
            "size": int(len(self._cache)),
            "hits": int(self._cache_hits),
            "misses": int(self._cache_misses),
        }

    def boundary_matrix(self, d: int) -> csr_matrix:
        """Return the attaching matrix for dimension d.

        Args:
            d: Dimension.

        Returns:
            Attaching matrix in CSR format.
        """
        return self.attaching_maps.get(int(d), csr_matrix((self.cells.get(d-1, 0), self.cells.get(d, 0)), dtype=np.int64))

    def boundary_matrices(self) -> Dict[int, csr_matrix]:
        """Return all attaching matrices.

        Returns:
            Dictionary mapping dimension to attaching matrix.
        """
        return self.attaching_maps

    def homology(
        self, n: int | None = None, backend: str = "auto"
    ) -> Union[Tuple[int, List[int]], Dict[int, Tuple[int, List[int]]]]:
        """Compute the n-th cellular homology.

        Args:
            n: Optional degree.
            backend: 'auto', 'julia', or 'python'.

        Returns:
            Rank/torsion or dictionary of results.
        """
        return self.cellular_chain_complex().homology(n, backend=backend)

    def cohomology(
        self, n: int | None = None, backend: str = "auto"
    ) -> Union[Tuple[int, List[int]], Dict[int, Tuple[int, List[int]]]]:
        """Compute the n-th cellular cohomology.

        Args:
            n: Optional degree.
            backend: 'auto', 'julia', or 'python'.

        Returns:
            Rank/torsion or dictionary of results.
        """
        return self.cellular_chain_complex().cohomology(n, backend=backend)

    def cohomology_basis(self, n: int, backend: str = "auto") -> list[np.ndarray]:
        """Compute a basis for cellular cohomology.

        Args:
            n: Degree.
            backend: 'auto', 'julia', or 'python'.

        Returns:
            List of basis vectors.
        """
        return self.cellular_chain_complex().cohomology_basis(n, backend=backend)

    def euler_characteristic(self) -> int:
        """Compute the Euler characteristic.

        Returns:
            Euler characteristic.
        """
        return self.cellular_chain_complex().euler_characteristic()

    def betti_number(self, n: int | None = None, backend: str = "auto") -> int | Dict[int, int]:
        """Return the n-th Betti number.

        Args:
            n: Optional degree.
            backend: 'auto', 'julia', or 'python'.

        Returns:
            Betti number or dictionary of Betti numbers.
        """
        return self.cellular_chain_complex().betti_number(n, backend=backend)

    def betti_numbers(self, backend: str = "auto") -> Dict[int, int]:
        """Return all Betti numbers.

        Returns:
            Dictionary mapping degree to Betti number.
        """
        return self.cellular_chain_complex().betti_numbers(backend=backend)

    def topological_invariants(self, backend: str = "auto") -> Dict[str, Any]:
        """Compute all key topological invariants.

        Args:
            backend: 'auto', 'julia', or 'python'.

        Returns:
            Dictionary of invariants.
        """
        return self.cellular_chain_complex().topological_invariants(backend=backend)

    def cellular_chain_complex(
        self, *, coefficient_ring: str | None = None
    ) -> ChainComplex:
        """Return the cellular chain complex.

        Args:
            coefficient_ring: Optional coefficient ring override.

        Returns:
            A ChainComplex instance.
        """
        ring = coefficient_ring if coefficient_ring is not None else self.coefficient_ring
        key = ("cellular", "chain_complex", ring)
        cached = self._cache_get(key)
        if cached is not None:
            return cast(ChainComplex, cached)

        out = ChainComplex(
            boundaries=self.attaching_maps,
            dimensions=self.dimensions,
            cells=self.cells,
            coefficient_ring=ring,
        )
        self._cache_set(key, out)
        return out


class SimplicialComplex(BaseModel):
    """A finite simplicial complex with sparse boundary matrices and topological computation tools.
    
    Overview:
        A SimplicialComplex represents a finite CW complex built from simplices (points, edges, 
        triangles, tetrahedra, etc.). It automatically maintains skeletal closure, computes boundary 
        and coboundary operators via sparse matrices, and provides access to fundamental topological 
        invariants: homology groups, fundamental groups, Betti numbers, Euler characteristic, and more.
    
    Key Concepts:
        - **Simplices**: Vertices (0-simplices), edges (1-simplices), triangles (2-simplices), etc.
        - **Skeletal Closure**: The complex automatically includes all faces of any simplex added.
        - **Boundary Operator (∂_n)**: Maps n-dimensional chains to (n-1)-dimensional chains.
        - **Chain Complex**: C_0 ←d₁ C_1 ←d₂ C_2 ← ... where C_n is the ℤ-module of n-chains.
        - **Homology H_n(C)**: The quotient ker(d_n) / im(d_{n+1}) (roughly: "holes" of dimension n).
    
    Common Workflows:
        1. **From geometry** → SimplicialComplex.from_simplices() or from_mesh()
        2. **Reduction** → simplify() or collapse() to get homotopy-equivalent smaller complex
        3. **Invariant computation** → homology(), fundamental_group(), betti_numbers()
        4. **Lifting results** → Use simplify()'s simplex_map to lift invariants back to original
    
    Coefficient Ring:
        - 'Z' (default): Integer coefficients; returns (rank, torsion) pairs
        - 'Q': Rational coefficients; returns ranks only
        - 'Z/pZ': Mod p coefficients; for prime p returns mod p structure
    
    Attributes:
        coefficient_ring (str): The coefficient ring for homology computations ('Z', 'Q', 'Z/pZ').
        filtration (dict): Optional Rips/Vietoris filtration (simplex → filtration_value).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _simplices_table: Dict[int, List[Tuple[int, ...]]] = PrivateAttr(default_factory=dict)
    coefficient_ring: str = "Z"
    filtration: Dict[Tuple[int, ...], float] = Field(default_factory=dict)

    _cache_enabled: bool = PrivateAttr(default=True)
    _cache: dict[tuple[object, ...], object] = PrivateAttr(default_factory=dict)
    _cache_hits: int = PrivateAttr(default=0)
    _cache_misses: int = PrivateAttr(default=0)
    _cache_signature: tuple[object, ...] | None = PrivateAttr(default=None)

    _boundaries_cache: Dict[int, csr_matrix] = PrivateAttr(default_factory=dict)
    _cells_cache: Dict[int, int] = PrivateAttr(default_factory=dict)

    def __init__(self, **data):
        """Initialize the simplicial complex.

        Args:
            **data: Pydantic model data, including 'simplices' (optional).
        """
        if "simplices" in data and not isinstance(data["simplices"], dict):
            # Handle possible list-of-lists input if any legacy code did that
            pass
        super().__init__(**data)
        if "simplices" in data:
            self._simplices_table = data["simplices"]

    def _structure_signature(self) -> tuple[object, ...]:
        """Return a signature for the structural state of the simplicial complex.

        Returns:
            A structural signature tuple.
        """
        simplex_sig = tuple(
            (int(d), len(s)) for d, s in sorted(self._simplices_table.items())
        )
        # For efficiency, we ignore the exact floating-point filtration in signature,
        # but track if it's empty or not.
        filtration_sig = bool(self.filtration)
        return simplex_sig, str(self.coefficient_ring), filtration_sig

    def _ensure_cache_valid(self) -> None:
        """Clear the cache if the complex structure has changed."""
        current = self._structure_signature()
        if self._cache_signature != current:
            self._cache.clear()
            self._cache_signature = current

    def _cache_get(self, key: tuple[object, ...]) -> object | None:
        """Retrieve a value from the cache.

        Args:
            key: The cache key.

        Returns:
            The cached value or None.
        """
        self._ensure_cache_valid()
        if not self._cache_enabled:
            return None
        if key in self._cache:
            self._cache_hits += 1
            return _clone_cache_value(self._cache[key])
        self._cache_misses += 1
        return None

    def _cache_set(self, key: tuple[object, ...], value: object) -> None:
        """Store a value in the cache.

        Args:
            key: The cache key.
            value: The value to store.
        """
        self._ensure_cache_valid()
        if not self._cache_enabled:
            return
        self._cache[key] = _clone_cache_value(value)

    def cache_info(self) -> dict[str, object]:
        """Return information about the cache state.

        Returns:
            Dictionary with cache stats.
        """
        self._ensure_cache_valid()
        return {
            "size": int(len(self._cache)),
            "hits": int(self._cache_hits),
            "misses": int(self._cache_misses),
        }

    def clear_cache(self, namespace: str | None = None) -> None:
        """Clear the cache, optionally filtered by namespace.

        Args:
            namespace: Optional namespace prefix to clear.
        """
        if namespace is None:
            self._cache.clear()
            return
        prefix = (str(namespace),)
        keys = [k for k in self._cache if k[:1] == prefix]
        for key in keys:
            self._cache.pop(key, None)

    @classmethod
    def from_vietoris_rips(
        cls,
        points: np.ndarray,
        epsilon: float,
        max_dimension: int,
        coefficient_ring: str = "Z",
        backend: str = "auto",
    ) -> "SimplicialComplex":
        """Generate a Vietoris-Rips complex from a point cloud.

        Uses a bottom-up Flag Complex (Clique Complex) construction.
        The 1-skeleton is built using a distance threshold, and higher-dimensional
        simplices are found by identifying cliques in the edge graph.

        Args:
            points: (N, D) array of point coordinates.
            epsilon: Distance threshold for edges.
            max_dimension: Maximum simplex dimension to include (e.g., 2 for triangles).
            coefficient_ring: Coefficient ring for the complex.

        Returns:
            A SimplicialComplex instance with geometric coordinates attached.
        """
        from ..bridge.julia_bridge import julia_engine

        points = np.asarray(points, dtype=np.float64)
        n_pts = points.shape[0]

        if julia_engine.available:
            try:
                simplices = julia_engine.compute_vietoris_rips(points, epsilon, max_dimension)
                sc = cls.from_simplices(
                    simplices, coefficient_ring=coefficient_ring, close_under_faces=True
                )
                sc._coordinates = points
                return sc
            except Exception as e:
                warnings.warn(f"Julia Vietoris-Rips failed: {e!r}. Falling back to Python.")

        # --- Optimized Python Fallback ---
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        pairs = tree.query_pairs(epsilon)
        valid_simplices = [(i,) for i in range(n_pts)] + list(pairs)

        sc = cls.from_simplices(
            valid_simplices, coefficient_ring=coefficient_ring, close_under_faces=True
        )
        sc._coordinates = points
        
        if max_dimension > 1:
            sc = sc.expand(max_dimension)
        
        return sc

    @classmethod
    def from_simplices(
        cls,
        simplices: Iterable[Iterable[int]],
        coefficient_ring: str = "Z",
        *,
        close_under_faces: bool = True,
    ) -> "SimplicialComplex":
        """Create a simplicial complex from generators, optionally taking the full closure.

        Args:
            simplices: Iterable of simplices (generators).
            coefficient_ring: Coefficient ring label.
            close_under_faces: Whether to automatically add all faces.

        Returns:
            A SimplicialComplex instance.
        """
        from ..bridge.julia_bridge import julia_engine

        # Ensure we can iterate multiple times if it's a generator
        simplex_list = [tuple(int(v) for v in s) for s in simplices]

        if close_under_faces:
            if julia_engine.available:
                # Accelerate full skeletal closure and boundary assembly in Julia
                max_dim = max((len(s) - 1 for s in simplex_list), default=-1)

                if max_dim >= 0:
                    try:
                        (
                            b_data,
                            cells,
                            dim_simplices,
                            _,
                        ) = julia_engine.compute_boundary_data_from_simplices(
                            simplex_list, max_dim
                        )

                        # Verify that Julia returned all dimensions (basic sanity check)
                        if all(d in dim_simplices for d in range(max_dim + 1)):
                            obj = cls(
                                simplices=dim_simplices,
                                coefficient_ring=coefficient_ring,
                            )

                            # Populate private caches to accelerate subsequent .chain_complex() calls
                            from scipy.sparse import csr_matrix

                            boundaries = {}
                            for dim, payload in b_data.items():
                                boundaries[dim] = csr_matrix(
                                    (payload["data"], (payload["rows"], payload["cols"])),
                                    shape=(payload["n_rows"], payload["n_cols"]),
                                    dtype=np.int64,
                                )

                            object.__setattr__(obj, "_boundaries_cache", boundaries)
                            object.__setattr__(obj, "_cells_cache", cells)
                            return obj
                    except Exception:
                        # Fallback to Python if Julia compute fails
                        pass

            simplex_table = _simplicial_closure_from_generators(simplex_list)
        else:
            grouped: dict[int, list[tuple[int, ...]]] = {}
            for simplex in simplex_list:
                t = _normalize_simplex(simplex)
                grouped.setdefault(len(t) - 1, []).append(t)
            simplex_table = _canonicalize_simplices_by_dim(grouped)

        return cls(simplices=simplex_table, coefficient_ring=coefficient_ring)

    @classmethod
    def from_maximal_simplices(
        cls,
        maximal_simplices: Iterable[Iterable[int]],
        coefficient_ring: str = "Z",
    ) -> "SimplicialComplex":
        """Build the full simplicial closure from a list of maximal simplices.

        Args:
            maximal_simplices: Iterable of maximal simplices.
            coefficient_ring: Coefficient ring label.

        Returns:
            A SimplicialComplex instance.
        """
        # This now benefits directly from the Julia acceleration in from_simplices
        return cls.from_simplices(
            maximal_simplices,
            coefficient_ring=coefficient_ring,
            close_under_faces=True,
        )

    @classmethod
    def from_point_cloud_cknn(
        cls,
        points: np.ndarray,
        k: int = 5,
        delta: float = 1.0,
        max_dimension: int = 2,
        *,
        coefficient_ring: str = "Z",
        backend: str = "auto",
    ) -> "SimplicialComplex":
        """Construct a Continuous k-Nearest Neighbors (CkNN) complex.

        CkNN is more robust to varying density than epsilon-graphs.

        Args:
            points: Array of point coordinates.
            k: Number of neighbors.
            delta: Scaling parameter.
            max_dimension: Maximum dimension of simplices.
            coefficient_ring: Coefficient ring label.

        Returns:
            A SimplicialComplex instance.
        """
        from ..bridge.julia_bridge import julia_engine
        pts = np.asarray(points, dtype=np.float64)
        n = len(pts)

        if n == 0:
            return cls.from_simplices([], coefficient_ring=coefficient_ring)
        if k >= n:
            k = n - 1
        if k < 1:
            return cls.from_simplices([(i,) for i in range(n)], coefficient_ring=coefficient_ring)

        simplices = [(i,) for i in range(n)]
        
        # We always compute rho in Python using cKDTree as it is extremely fast and efficient
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=k+1)
        rho = dists[:, k]
        
        # Try Julia acceleration for the large edge list filtering
        julia_success = False
        if julia_engine.available:
            try:
                pairs = julia_engine.compute_cknn_graph_accelerated(pts, rho, delta)
                for i, j in pairs:
                    simplices.append((int(i), int(j)))
                julia_success = True
            except Exception as e:
                import warnings
                warnings.warn(f"Julia CkNN failed ({e!r}). Falling back to SciPy query_pairs.")

        if not julia_success:
            # Vectorized density check
            max_search_radius = delta * np.max(rho)
            pairs_arr = np.array(list(tree.query_pairs(max_search_radius)), dtype=np.int64)
            if len(pairs_arr) > 0:
                i_idx = pairs_arr[:, 0]
                j_idx = pairs_arr[:, 1]
                
                # Check CkNN condition: dist(i,j)^2 < delta^2 * rho[i] * rho[j]
                dists_sq = np.sum((pts[i_idx] - pts[j_idx])**2, axis=1)
                thresholds_sq = (delta ** 2) * rho[i_idx] * rho[j_idx]
                
                valid_mask = dists_sq < thresholds_sq
                valid_pairs = pairs_arr[valid_mask]
                
                for p_idx in range(len(valid_pairs)):
                    simplices.append((int(valid_pairs[p_idx, 0]), int(valid_pairs[p_idx, 1])))

        sc = cls.from_simplices(simplices, coefficient_ring=coefficient_ring, close_under_faces=True)
        if max_dimension > 1:
            return sc.expand(max_dimension)
        return sc
    @classmethod
    def from_alpha_complex(
        cls,
        points: np.ndarray,
        alpha: float | None = None,
        *,
        max_alpha_square: Optional[float] = None,
        coefficient_ring: str = "Z",
        backend: str = "auto",
    ) -> "SimplicialComplex":
        """Compute an Alpha complex for the given points and distance threshold.

        Utilizes high-performance Delaunay filtration.

        Args:
            points: Array of point coordinates.
            alpha: Distance threshold (circumradius).
            max_alpha_square: Squared distance threshold override.
            coefficient_ring: Coefficient ring label.
            backend: 'auto', 'julia', or 'python'.

        Returns:
            A SimplicialComplex instance.
        """
        from scipy.spatial import Delaunay
        import itertools

        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2:
            raise ValueError("points must be a 2D array of coordinates.")
        n_pts, dim = pts.shape
        if n_pts < dim + 1:
            return cls.from_simplices([[i] for i in range(n_pts)], coefficient_ring=coefficient_ring)

        dt = Delaunay(pts, qhull_options="QJ")
        simplices_d = dt.simplices

        # Normalize backend
        backend_norm = str(backend).lower().strip()
        use_julia = (backend_norm == "julia") or (backend_norm == "auto" and julia_engine.available)

        # Handle lack of alpha (EMST Heuristic)
        if alpha is None and max_alpha_square is None:
            import warnings
            warnings.warn("No alpha provided. Defaulting to EMST maximum edge length to ensure connectivity.")

            alpha2 = None
            if use_julia:
                try:
                    alpha2 = julia_engine.compute_alpha_threshold_emst(pts, simplices_d)
                    warnings.warn(f"[Alpha Complex] Alpha value found through EMST Heuristic {alpha2}")
                except Exception as e:
                    if backend_norm == "julia":
                        raise e

            if alpha2 is None:
                # Python fallback - Vectorized edge extraction
                if len(simplices_d) > 0:
                    # All unique edges
                    all_edges = []
                    for i, j in itertools.combinations(range(simplices_d.shape[1]), 2):
                        all_edges.append(np.sort(simplices_d[:, [i, j]], axis=1))

                    edges_arr = np.unique(np.concatenate(all_edges, axis=0), axis=0)
                    u_idx, v_idx = edges_arr[:, 0], edges_arr[:, 1]

                    from scipy.sparse import csr_matrix
                    from scipy.sparse.csgraph import minimum_spanning_tree
                    dists = np.sqrt(np.sum((pts[u_idx] - pts[v_idx])**2, axis=1))
                    adj = csr_matrix((dists, (u_idx, v_idx)), shape=(n_pts, n_pts))
                    mst = minimum_spanning_tree(adj)
                    alpha2 = (mst.data.max() / 2.0)**2 if mst.nnz > 0 else 0.0
                else:
                    alpha2 = 0.0
        elif max_alpha_square is not None:
            alpha2 = float(max_alpha_square)
        else:
            alpha2 = float(alpha**2)

        if use_julia:
            try:
                valid_simplices_list = julia_engine.compute_alpha_complex_simplices(
                    pts, simplices_d, alpha2, dim
                )
                return cls.from_simplices(valid_simplices_list, coefficient_ring=coefficient_ring, close_under_faces=True)
            except Exception as e:
                if backend_norm == "julia":
                    raise e
                import warnings
                warnings.warn(f"Julia Alpha Complex failed ({e!r}). Falling back to pure Python.")

        # Python fallback - Robust Alpha Complex (Gabriel condition)
        # A simplex is included if its circumradius is <= alpha.
        def get_all_faces(simplices, d):
            faces = set()
            for s in simplices:
                for combo in itertools.combinations(s, d+1):
                    faces.add(tuple(sorted(int(v) for v in combo)))
            if not faces:
                return np.zeros((0, d + 1), dtype=np.int64)
            return np.array(list(faces), dtype=np.int64)

        _r2_cache_py = {}

        def get_r2_py(s_indices):
            s_key = tuple(sorted(s_indices))
            if s_key in _r2_cache_py:
                return _r2_cache_py[s_key]
            
            k = len(s_key)
            if k == 1:
                return 0.0
            
            pts_s = pts[list(s_key)]
            if k == 2:
                val = np.sum((pts_s[0] - pts_s[1])**2) / 4.0
                _r2_cache_py[s_key] = val
                return val
            
            if k == 3:
                # Triangle
                p0, p1, p2 = pts_s[0], pts_s[1], pts_s[2]
                v1, v2 = p1 - p0, p2 - p0
                area2 = 0.25 * np.sum(np.cross(v1, v2)**2)
                a2, b2, c2 = np.sum((p1-p2)**2), np.sum((p0-p2)**2), np.sum((p0-p1)**2)
                r2_acute = (a2 * b2 * c2) / (16.0 * area2 + 1e-30)
                is_obtuse = (a2 + b2 < c2) | (a2 + c2 < b2) | (b2 + c2 < a2)
                val = max(a2, b2, c2) / 4.0 if is_obtuse else r2_acute
                _r2_cache_py[s_key] = val
                return val

            # Generic N-dimensional
            p0 = pts_s[0]
            A_mat = pts_s[1:] - p0
            b_vec = 0.5 * np.sum((pts_s[1:] - p0)**2, axis=1)
            try:
                # Check for degeneracy in 3D (k=4)
                if k-1 == dim and abs(np.linalg.det(A_mat)) < 1e-15:
                    r2_max = 0.0
                    for face in itertools.combinations(s_key, k-1):
                        r2_max = max(r2_max, get_r2_py(face))
                    _r2_cache_py[s_key] = r2_max
                    return r2_max

                c_vec, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                val = np.sum(c_vec**2)
                _r2_cache_py[s_key] = val
                return val
            except Exception:
                _r2_cache_py[s_key] = np.inf
                return np.inf

        valid_simplices_final = set()
        for i in range(n_pts):
            valid_simplices_final.add((i,))

        # Evaluate all faces of Delaunay triangulation
        for d in range(1, dim + 1):
            d_simplices = get_all_faces(simplices_d, d)
            for s in d_simplices:
                if get_r2_py(s) <= alpha2:
                    valid_simplices_final.add(tuple(s))
        
        return cls.from_simplices(list(valid_simplices_final), coefficient_ring=coefficient_ring, close_under_faces=True)

    @classmethod
    def from_crust_algorithm(
        cls,
        points: np.ndarray,
        coefficient_ring: str = "Z",
        backend: str = "auto",
    ) -> "SimplicialComplex":
        """Reconstruct a surface from a point cloud using the Crust algorithm (Amenta et al., 1998).

        This is a parameter-free algorithm that uses Voronoi poles to adapt to variable 
        sampling density.

        Args:
            points: An (N, D) array of point coordinates.
            coefficient_ring: Coefficient ring label.

        Returns:
            A SimplicialComplex representing the reconstructed manifold.
        """
        from scipy.spatial import Delaunay, Voronoi
        
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2:
            raise ValueError("points must be a 2D array of coordinates.")
        n_pts, dim = pts.shape
        if n_pts < dim + 1:
            return cls.from_simplices([[i] for i in range(n_pts)], coefficient_ring=coefficient_ring)
        
        # 1. Compute Voronoi diagram to get poles (Voronoi vertices)
        vor = Voronoi(pts, qhull_options="QJ")
        vor_vertices = vor.vertices
        
        # 2. Combine original points and Voronoi vertices
        combined_pts = np.vstack([pts, vor_vertices])
        
        # 3. Compute Delaunay triangulation of combined set
        dt_combined = Delaunay(combined_pts, qhull_options="QJ")
        
        # 4. Extract simplices of dimension (dim-1) where all vertices are from the original point set
        # This is the "Crust".
        target_dim = dim - 1
        valid_simplices = set()
        for s in dt_combined.simplices:
            for face in itertools.combinations(s, target_dim + 1):
                if np.all(np.array(face) < n_pts):
                    valid_simplices.add(tuple(sorted(int(v) for v in face)))
        
        # Ensure vertices are included
        for i in range(n_pts):
            valid_simplices.add((i,))
            
        return cls.from_simplices(valid_simplices, coefficient_ring=coefficient_ring, close_under_faces=True)

    @classmethod
    def from_witness(
        cls,
        points: np.ndarray,
        n_landmarks: int,
        alpha: float | None = None,
        *,
        max_dimension: int = 2,
        coefficient_ring: str = "Z",
    ) -> "SimplicialComplex":
        """Construct a Witness complex from a point cloud.

        Used as a sparse approximation for large-scale TDA.

        Args:
            points: Array of point coordinates.
            n_landmarks: Number of landmark points to choose.
            alpha: Relaxation parameter.
            max_dimension: Maximum dimension.
            coefficient_ring: Coefficient ring label.

        Returns:
            A SimplicialComplex instance.
        """
        from scipy.spatial.distance import cdist

        n_pts = len(points)
        landmarks_idx = np.random.choice(n_pts, n_landmarks, replace=False)
        from ..bridge.julia_bridge import julia_engine
        simplices = None
        if julia_engine.available:
            try:
                alpha_val = float(alpha) if alpha is not None else 0.0
                simplices = julia_engine.compute_witness_complex_simplices(points, landmarks_idx, alpha_val, max_dimension)
            except Exception:
                pass
                
        if simplices is None:
            # Python Fallback
            distances = cdist(points, points[landmarks_idx])
            m_dist = np.min(distances, axis=1)
            alpha_val = float(alpha) if alpha is not None else 0.0
            valid_witnesses = (distances <= (m_dist[:, None] + alpha_val)).astype(np.int8)
            shared = valid_witnesses.T @ valid_witnesses
            rows, cols = np.nonzero(np.triu(shared, k=1))
            simplices = [(i,) for i in range(n_landmarks)] + [
                (int(r), int(c)) for r, c in zip(rows, cols)
            ]

        sc = cls.from_simplices(simplices, coefficient_ring=coefficient_ring, close_under_faces=True)
        if max_dimension > 1:
            return sc.expand(max_dimension)
        return sc

    @property
    def simplices(self) -> List[Tuple[int, ...]]:
        """Return all simplices in the complex as a flat list."""
        all_s = []
        for d in self.dimensions:
            all_s.extend(self.n_simplices(d))
        return all_s

    @property
    def simplices_field(self) -> Dict[int, List[Tuple[int, ...]]]:
        """Return the dictionary of simplices grouped by dimension.

        Returns:
            Dictionary mapping dimension to list of simplices.
        """
        return self._simplices_table

    @property
    def simplices_dict(self) -> Dict[int, List[Tuple[int, ...]]]:
        """Alias for simplices_field used in some legacy tests.

        Returns:
            Dictionary mapping dimension to list of simplices.
        """
        return self._simplices_table

    @property
    def dimensions(self) -> List[int]:
        """Return the list of dimensions present in the complex.

        Returns:
            Sorted list of dimensions.
        """
        return sorted(list(self._simplices_table.keys()))

    @property
    def dimension(self) -> int:
        """Return the maximum dimension of the complex.

        Returns:
            Maximum dimension or -1 if empty.
        """
        return max(self.dimensions) if self.dimensions else -1

    def count_simplices(self, d: int) -> int:
        """Return the number of simplices in dimension d.

        Args:
            d: Dimension.

        Returns:
            Number of d-simplices.
        """
        return len(self._simplices_table.get(int(d), []))

    @property
    def cells(self) -> Dict[int, int]:
        """Return a dictionary mapping dimension to simplex count.

        Returns:
            Dictionary of simplex counts.
        """
        return {d: self.count_simplices(d) for d in self.dimensions}

    @property
    def attaching_maps(self) -> Dict[int, csr_matrix]:
        """Return all boundary matrices for the complex.

        Returns:
            Dictionary mapping dimension to boundary matrix.
        """
        return self.boundary_matrices()

    def n_simplices(self, d: int) -> List[Tuple[int, ...]]:
        """Return the list of all simplices of dimension d (d-simplices).
        
        What are n-Simplices?:
            An n-simplex is the n-dimensional generalization of a triangle:
            - 0-simplex: A single vertex {v}
            - 1-simplex: An edge {u, v}
            - 2-simplex: A triangle {u, v, w}
            - 3-simplex: A tetrahedron {u, v, w, x}
            - n-simplex: The convex hull of n+1 affinely independent points
        
        Algorithm:
            Retrieve the pre-computed list of n-dimensional simplices from the internal 
            _simplices_table dictionary, which is built and maintained during complex construction.
        
        Args:
            d: Dimension (int). d=0 returns vertices, d=1 returns edges, etc.
        
        Returns:
            List[Tuple[int, ...]]: All d-simplices as sorted tuples of vertex indices.
                                  Empty list if no simplices of dimension d exist.
        
        Use When:
            - Iterating over simplices of a specific dimension
            - Building custom algorithms (e.g., computing link, star, or specific cofaces)
            - Filtering by dimension for targeted computations
            - Exporting the complex to other formats
        
        Example:
            vertices = sc.n_simplices(0)  # List of vertex labels
            edges = sc.n_simplices(1)  # Edges (1-simplices)
            triangles = sc.n_simplices(2)  # Triangles (2-simplices)
            for edge in sc.n_simplices(1):
                u, v = edge
                # process edge
        """
        return self._simplices_table.get(int(d), [])

    def _get_coface_map(self) -> Dict[int, Set[Tuple[int, ...]]]:
        """Return a mapping from vertex to all simplices containing it.

        Returns:
            Dict[int, Set[Tuple[int, ...]]]: Vertex-to-simplices map.
        """
        key = ("simplicial", "coface_map")
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        
        coface_map = defaultdict(set)
        for d in self.dimensions:
            for simplex in self.n_simplices(d):
                for v in simplex:
                    coface_map[v].add(simplex)
        
        self._cache_set(key, dict(coface_map))
        return dict(coface_map)

    def star(self, simplex: Iterable[int]) -> Set[Tuple[int, ...]]:
        """Compute the star of a simplex: all simplices in the complex that contain it.

        The star St(σ) is the set of all simplices τ such that σ ⊆ τ.
        Uses an optimized coface map for O(Local Star) complexity.

        Args:
            simplex: The simplex σ as an iterable of vertices.

        Returns:
            A set of simplices τ in the star.
        """
        sigma = _normalize_simplex(simplex)
        sigma_set = set(sigma)
        if not sigma:
            return set()
            
        coface_map = self._get_coface_map()
        # The star of sigma must be a subset of the intersection of cofaces of its vertices
        potential_tau = None
        for v in sigma:
            v_cofaces = coface_map.get(v, set())
            if potential_tau is None:
                potential_tau = v_cofaces.copy()
            else:
                potential_tau &= v_cofaces
            if not potential_tau:
                break
                
        if not potential_tau:
            return set()
            
        star = {tau for tau in potential_tau if sigma_set.issubset(tau)}
        return star

    def closed_star(self, simplex: Iterable[int]) -> "SimplicialComplex":
        """Compute the closed star of a simplex: the smallest subcomplex containing its star.

        Args:
            simplex: The simplex as an iterable of vertices.

        Returns:
            A SimplicialComplex representing the closed star.
        """
        star_simplices = self.star(simplex)
        return SimplicialComplex.from_simplices(star_simplices, coefficient_ring=self.coefficient_ring, close_under_faces=True)

    def link(self, simplex: Iterable[int]) -> "SimplicialComplex":
        """Compute the link of a simplex: Lk(σ) = {τ ∈ Cl(St(σ)) : τ ∩ σ = ∅}.

        The link is the "boundary" of the star, consisting of all faces of simplices
        in the star that do not intersect the given simplex.

        Args:
            simplex: The simplex σ as an iterable of vertices.

        Returns:
            A SimplicialComplex representing the link.
        """
        sigma = _normalize_simplex(simplex)
        sigma_set = set(sigma)
        
        # Collect all simplices in the star
        star_simplices = self.star(sigma)
        
        # The link consists of all faces of simplices in the star that are disjoint from sigma
        link_simplices = set()
        for tau in star_simplices:
            # For each simplex tau in the star, sigma is a subset.
            # Any face of tau that is disjoint from sigma must be a subset of tau \ sigma.
            face_candidates = tuple(sorted(set(tau) - sigma_set))
            if face_candidates:
                link_simplices.add(face_candidates)
        
        # link_simplices contains the maximal simplices of the link. 
        # Since the parent complex is already closed, the link is also closed
        # if we include all faces. from_simplices will organize them.
        return SimplicialComplex.from_simplices(link_simplices, coefficient_ring=self.coefficient_ring, close_under_faces=True)

    def simplex_to_index(self, d: int) -> Dict[Tuple[int, ...], int]:
        """Map each d-simplex to its index in the ordered simplex list.

        Args:
            d: Simplex dimension.

        Returns:
            Dictionary mapping simplex tuple to its integer index.
        """
        key = ("simplicial", "simplex_to_index", int(d))
        cached = self._cache_get(key)
        if cached is not None:
            return cached
            
        mapping = {s: i for i, s in enumerate(self.n_simplices(d))}
        self._cache_set(key, mapping)
        return mapping

    def f_vector(self) -> dict[int, int]:
        """Return the f-vector of the complex.

        Returns:
            dict[int, int]: Dictionary mapping dimension to simplex count.
        """
        return {d: self.count_simplices(d) for d in self.dimensions}

    def euler_characteristic(self) -> int:
        """Return the Euler characteristic of the complex.

        Returns:
            int: Euler characteristic chi = sum (-1)^i f_i.
        """
        chi = 0
        for d in self.dimensions:
            count = self.count_simplices(d)
            if d % 2 == 0:
                chi += count
            else:
                chi -= count
        return chi

    def is_closed_under_faces(self) -> bool:
        """Check whether the complex is closed under taking faces.

        Returns:
            bool: True if all codimension-1 faces of every simplex are present, False otherwise.
        """
        key = ("simplicial", "is_closed_under_faces")
        cached = self._cache_get(key)
        if cached is not None:
            return bool(cached)
        for dim, simplices in self.simplices_field.items():
            if dim <= 0:
                continue
            target = set(self.simplices_field.get(dim - 1, []))
            for simplex in simplices:
                for i in range(len(simplex)):
                    if simplex[:i] + simplex[i + 1 :] not in target:
                        self._cache_set(key, False)
                        return False
        self._cache_set(key, True)
        return True

    def verify_structure(self) -> Dict[str, Any]:
        """Comprehensive mathematical validity check for the simplicial complex.
        
        Verifies:
        1. Downward Closure: Every face of every simplex is in the complex.
        2. Orientation Consistency: All simplices are stored in canonical (sorted) order.
        3. Boundary of Boundary: Composition of consecutive boundary operators is zero (d^2 = 0).
        4. No Gap Dimensions: No intermediate dimensions are missing (e.g., has 0 and 2 but no 1).

        Returns:
            Dict[str, Any]: A dictionary containing validity status and detailed issues.
        """
        issues = []
        dims = sorted(self.dimensions)
        
        # 1. & 2. Closure and Orientation
        is_closed = self.is_closed_under_faces()
        if not is_closed:
            issues.append("Downward closure failure: some faces are missing from the complex.")
            
        for dim, simplices in self.simplices_field.items():
            for s in simplices:
                if list(s) != sorted(set(s)):
                    issues.append(f"Orientation inconsistency: simplex {s} is not in canonical sorted order or has duplicates.")
                    break
        
        # 3. d^2 = 0
        for d in dims:
            if d <= 1:
                continue
            mat_d = self.boundary_matrix(d)
            mat_dm1 = self.boundary_matrix(d - 1)
            
            if mat_d.nnz > 0 and mat_dm1.nnz > 0:
                prod = mat_dm1 @ mat_d
                if prod.nnz > 0:
                    if np.any(prod.data != 0):
                        issues.append(f"Boundary consistency failure: d_{d-1} * d_{d} != 0 at dimension {d}.")

        # 4. Gap Dimensions
        if dims:
            for d in range(dims[0], dims[-1] + 1):
                if d not in self.simplices_field:
                    issues.append(f"Gap dimension failure: dimension {d} is missing.")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "is_closed": is_closed,
            "is_canonical": all(list(s) == sorted(set(s)) for simps in self.simplices_field.values() for s in simps),
            "has_gaps": any("Gap dimension" in issue for issue in issues)
        }

    def hasse_edges(self) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        """Return the codimension-one relations of the Hasse diagram.

        Returns:
            list[tuple[tuple[int, ...], tuple[int, ...]]]: List of (face, simplex) tuples representing Hasse edges.
        """
        key = ("simplicial", "hasse_edges")
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        edges = []
        for d in range(1, self.dimension + 1):
            lower = set(self.n_simplices(d - 1))
            for simplex in self.n_simplices(d):
                for i in range(len(simplex)):
                    face = simplex[:i] + simplex[i + 1 :]
                    if face in lower:
                        edges.append((face, simplex))
        self._cache_set(key, edges)
        return edges

    def boundary_matrix(self, d: int) -> csr_matrix:
        """Return the boundary operator matrix ∂_d: C_d → C_{d-1}.
        
        What is the Boundary Operator?:
            The boundary of an n-simplex {v₀, v₁, ..., v_n} is the formal sum of its (n-1)-faces
            with alternating signs: ∂{v₀, ..., v_n} = Σ (-1)^i {v₀, ..., v̂ᵢ, ..., v_n}.
            The boundary matrix ∂_d encodes this operation as a sparse integer matrix with rows 
            indexed by (d-1)-simplices and columns by d-simplices.
        
        Algorithm:
            1. Extract all d-simplices from the complex
            2. For each d-simplex σ, compute its (d-1) faces with signs
            3. Assemble a CSR sparse matrix where entry (i, j) = ±1 if face_i is a face of simplex_j
            4. Cache result for fast re-access
        
        Preserved Invariants:
            - ∂_{d-1} ∘ ∂_d = 0 (boundary of a boundary is zero; ker(∂_d) ⊇ im(∂_{d+1}))
            - Rank and null-space determine homology groups
            - Matrix is sparse and efficient to work with
        
        Args:
            d: Dimension. Returns ∂_d: C_d → C_{d-1}.
        
        Returns:
            csr_matrix: Sparse matrix of shape (|C_{d-1}|, |C_d|) with ±1 entries.
        
        Use When:
            - Direct computation of homology via Smith Normal Form
            - Studying the chain complex structure
            - Computing with alternative backends (e.g., Julia)
            - Custom homological algebra computations
        
        Example:
            ∂2 = sc.boundary_matrix(2)  # Boundary operator for triangles → edges
            # ∂2.shape = (num_edges, num_triangles)
            # Use for homology: H_1 = ker(∂1) / im(∂2)
        """
        key = ("simplicial", "boundary_matrix", int(d))
        cached = self._cache_get(key)
        if cached is not None:
            return cast(csr_matrix, cached)

        # Check if we have a pre-calculated boundary from Julia or previous calls
        if hasattr(self, "_boundaries_cache") and d in self._boundaries_cache:
            return self._boundaries_cache[d]

        mat = _boundary_matrix_from_simplices_with_maps(self.n_simplices(d), self.simplex_to_index(d - 1))
        self._cache_set(key, mat)
        return mat

    def homology(
        self, n: int | None = None, backend: str = "auto"
    ) -> Union[Tuple[int, List[int]], Dict[int, Tuple[int, List[int]]]]:
        """Compute the homology of the simplicial complex.

        Args:
            n: Optional homological degree to compute. If None, computes for all degrees.
            backend: 'auto', 'julia', or 'python'.

        Returns:
            If n is provided: A tuple (rank, torsion).
            If n is None: A dictionary mapping degree to (rank, torsion).
        """
        return self.cellular_chain_complex().homology(n, backend=backend)

    def cohomology(
        self, n: int | None = None, backend: str = "auto"
    ) -> Union[Tuple[int, List[int]], Dict[int, Tuple[int, List[int]]]]:
        """Compute the cohomology of the simplicial complex.

        Args:
            n: Optional homological degree to compute. If None, computes for all degrees.
            backend: 'auto', 'julia', or 'python'.

        Returns:
            If n is provided: A tuple (rank, torsion).
            If n is None: A dictionary mapping degree to (rank, torsion).
        """
        return self.cellular_chain_complex().cohomology(n, backend=backend)

    def cohomology_basis(self, n: int, backend: str = "auto") -> list[np.ndarray]:
        """Compute a basis for the n-th cohomology group.

        Args:
            n: Degree.
            backend: 'auto', 'julia', or 'python'.

        Returns:
            list[np.ndarray]: List of cochain vectors forming a basis.
        """
        return self.cellular_chain_complex().cohomology_basis(n, backend=backend)

    def reduced_homology(
        self, n: int | None = None, backend: str = "auto"
    ) -> Union[Tuple[int, List[int]], Dict[int, Tuple[int, List[int]]]]:
        r"""Compute the reduced homology groups \tilde{H}_n(K).
        
        \tilde{H}_n(K) is isomorphic to H_n(K) for n > 0.
        For n = 0, \tilde{H}_0(K) is the free group of rank (Betti_0 - 1).
        If the complex is empty, \tilde{H}_{-1}(\emptyset) = Z.

        Args:
            n: Optional homological degree to compute. If None, returns all degrees.
            backend: 'auto', 'julia', or 'python'.

        Returns:
            If n is provided: A tuple (rank, torsion).
            If n is None: A dictionary mapping degree to (rank, torsion).
        """
        h = self.homology(backend=backend)
        if not isinstance(h, dict):
            h = {i: self.homology(i, backend=backend) for i in self._homological_dimensions()}
            
        rh = {}
        for k, (rank, torsion) in h.items():
            if k > 0:
                rh[k] = (rank, torsion)
            elif k == 0:
                rh[0] = (max(0, rank - 1), torsion)
        
        if n is not None:
            n = int(n)
            if n < -1:
                return (0, [])
            if n == -1:
                # Standard convention: reduced homology of empty complex is Z at degree -1
                return (1, []) if not self.dimensions else (0, [])
            return rh.get(n, (0, []))
        
        # If the complex is empty, explicitly include H_{-1} = Z
        if not self.dimensions:
            rh[-1] = (1, [])
            
        return rh

    def is_homology_manifold(self, backend: str = "auto") -> tuple[bool, int | None, dict[int, str]]:
        r"""Check if the simplicial complex is a homology manifold (potentially with boundary).

        A complex is a d-dimensional homology manifold if for every vertex v:
        - \tilde{H}_*(Lk(v)) \cong \tilde{H}_*(S^{d-1}) (interior vertex)
        - \tilde{H}_*(Lk(v)) \cong \tilde{H}_*(D^{d-1}) \cong 0 (boundary vertex)

        Args:
            backend: 'auto', 'julia', or 'python'.

        Returns:
            tuple: A tuple containing:
                - is_manifold (bool): True if it's a homology manifold.
                - dimension (int | None): The detected intrinsic dimension.
                - diagnostics (dict[int, str]): Mapping vertex ID to failure reason.
        """
        # Normalize backend
        backend_norm = str(backend).lower().strip()
        use_julia = (backend_norm == "julia") or (backend_norm == "auto" and julia_engine.available)

        if use_julia:
            try:
                # Accelerate heavy vertex link homology loop in Julia
                all_simplices = []
                for d in self.dimensions:
                    all_simplices.extend(self.n_simplices(d))
                return julia_engine.is_homology_manifold_jl(all_simplices, self.dimension)
            except Exception as e:
                if backend_norm == "julia":
                    raise e
                import warnings
                warnings.warn(f"Julia is_homology_manifold failed ({e!r}). Falling back to pure Python.")
        vertices = [v[0] for v in self.n_simplices(0)]
        if not vertices:
            return True, -1, {}
            
        local_dims = {}
        diagnostics = {}
        
        def _check_vertex(v):
            lk = self.link((v,))
            rh = lk.reduced_homology()
            # Filter non-zero reduced homology groups
            non_zero = {k: val for k, val in rh.items() if val[0] > 0 or val[1]}
            
            if not non_zero:
                return v, None, None
            elif len(non_zero) == 1:
                k = list(non_zero.keys())[0]
                rank, torsion = non_zero[k]
                if rank == 1 and not torsion:
                    return v, k + 1, None
                else:
                    return v, None, f"Link has non-sphere homology at degree {k}: rank={rank}, torsion={torsion}"
            else:
                return v, None, f"Link has multiple non-zero homology groups: {list(non_zero.keys())}"

        with ThreadPoolExecutor() as executor:
            for v, dim_val, diag in executor.map(_check_vertex, vertices):
                if diag:
                    diagnostics[v] = diag
                local_dims[v] = dim_val
                
        # Determine global dimension from non-None local estimates
        detected_dims = {d for d in local_dims.values() if d is not None}
        
        if not detected_dims:
            # All links were acyclic but non-empty? 
            # This shouldn't happen for a pure manifold unless it's just a contractible blob.
            if any(v is None for v in local_dims.values()):
                # If everything is acyclic, it might be a disk-like thing.
                # We can't easily guess the dimension from acyclic links only.
                # But we can look at the complex dimension.
                d = self.dimension
                return True, d, {}
            return False, None, {"global": "Could not determine dimension from links."}
            
        if len(detected_dims) > 1:
            return False, None, {"global": f"Inconsistent local dimensions: {detected_dims}"}
            
        d = list(detected_dims)[0]
        
        # Pure manifold condition: detected dimension must match top-level simplex dimension
        if d != self.dimension:
            return False, d, {"global": f"Detected manifold dimension {d} does not match complex dimension {self.dimension}"}

        # Check if all acyclic links are consistent with this dimension
        # (Actually, a d-manifold vertex link can be acyclic if it's on the boundary)
        if diagnostics:
            return False, d, diagnostics
            
        return True, d, {}

    def boundary_matrices(self) -> Dict[int, csr_matrix]:
        """Return all boundary matrices for the complex.

        Returns:
            Dict[int, csr_matrix]: Dictionary mapping dimension to boundary matrix.
        """
        return {d: self.boundary_matrix(d) for d in range(1, self.dimension + 1)}

    def chain_complex(self, coefficient_ring: str | None = None) -> ChainComplex:
        """Return the chain complex C_*(X) over the specified ring.

        Args:
            coefficient_ring: Optional coefficient ring label.

        Returns:
            ChainComplex: The resulting chain complex instance.
        """
        ring = coefficient_ring if coefficient_ring is not None else self.coefficient_ring
        _parse_coefficient_ring(ring)
        key = ("simplicial", "chain_complex", ring)
        cached = self._cache_get(key)
        if cached is not None:
            return cast(ChainComplex, cached)

        # Check if Julia pre-calculated the boundary operators
        if hasattr(self, "_boundaries_cache") and hasattr(self, "_cells_cache") and self._boundaries_cache and self._cells_cache:
            boundaries = self._boundaries_cache
            cells = self._cells_cache
        else:
            boundaries = self.boundary_matrices()
            cells = {dim: len(simplices) for dim, simplices in self.simplices_field.items()}
            
        out = ChainComplex(
            boundaries=boundaries,
            dimensions=self.dimensions,
            cells=cells,
            coefficient_ring=ring,
        )
        self._cache_set(key, out)
        return out

    def cellular_chain_complex(self, *, coefficient_ring: str | None = None) -> ChainComplex:
        """Alias for `chain_complex` for compatibility with cellular workflows.

        Args:
            coefficient_ring: Optional coefficient ring label.

        Returns:
            ChainComplex: The resulting chain complex instance.
        """
        return self.chain_complex(coefficient_ring=coefficient_ring)

    def to_cw_complex(self) -> "CWComplex":
        """Converts the simplicial complex to a CWComplex.

        Returns:
            CWComplex: The resulting CW complex.
        """
        return CWComplex.from_simplicial_complex(self)

    def expand(self, max_dim: int | None = None, backend: str = "auto") -> "SimplicialComplex":
        """Expands the simplicial complex into a Flag Complex (Clique Complex).
        
        This adds all cliques of size up to max_dim + 1 as simplices.
        Commonly used after skeleton-only algorithms like quick_mapper.
        
        Args:
            max_dim: The maximum dimension of simplices to include.
                     If None, defaults to the maximum dimension of the space 
                     (number of vertices - 1).

        Returns:
            SimplicialComplex: The expanded flag complex.
        """
        from ..bridge.julia_bridge import julia_engine
        import scipy.sparse as sp
        
        # 1. Get 1-skeleton (vertices and edges)
        edges = self.n_simplices(1)
        vertices = [v[0] for v in self.n_simplices(0)]
        if not vertices:
             return self.model_copy()
             
        n_v = max(vertices) + 1 if vertices else 0
        
        # Default max_dim to the "maximum dimension of the space"
        if max_dim is None:
            max_dim = max(0, n_v - 1)
            if max_dim > 10:
                warnings.warn(
                    f"Auto-expanding up to max_dim={max_dim} may be extremely slow. "
                    "Consider citing an explicit max_dim for performance."
                )

        # 2. Try Julia acceleration (much faster for large clique finding)
        if julia_engine.available:
            try:
                # Build adjacency in CSR format for Julia
                row_indices = []
                col_indices = []
                for u, v in edges:
                    row_indices.extend([u, v])
                    col_indices.extend([v, u])
                
                adj = sp.csr_matrix(
                    (np.ones(len(row_indices), dtype=np.int64), (row_indices, col_indices)),
                    shape=(n_v, n_v)
                )
                
                # indptr is rowptr, indices is colval
                cliques = julia_engine.enumerate_cliques_sparse(
                    np.asarray(adj.indptr + 1, dtype=np.int64), # 1-based for Julia
                    np.asarray(adj.indices + 1, dtype=np.int64),
                    int(n_v),
                    int(max_dim)
                )
                
                # Convert 1-based Julia results back to 0-based
                new_simplices = [tuple(sorted(int(x) - 1 for x in c)) for c in cliques]
                return self.__class__.from_simplices(
                    new_simplices, 
                    coefficient_ring=self.coefficient_ring,
                    close_under_faces=True
                )
            except Exception as e:
                warnings.warn(f"Julia clique expansion failed: {e}. Falling back to Python.")

        # 3. Python Fallback (Bron-Kerbosch with pivoting)
        adj_dict = {v: set() for v in vertices}
        for u, v in edges:
            adj_dict[u].add(v)
            adj_dict[v].add(u)
            
        all_cliques = []
        
        def bron_kerbosch_pivot(r, p, x):
            if not p and not x:
                if r:
                    all_cliques.append(tuple(sorted(r)))
                return
            if not p:
                return
            
            # Choose pivot as vertex in P U X with max degree in P
            u = max(p | x, key=lambda v: len(adj_dict[v] & p))
            for v in list(p - adj_dict[u]):
                if len(r) < max_dim:
                    bron_kerbosch_pivot(r | {v}, p & adj_dict[v], x & adj_dict[v])
                else:
                    # If we reached max_dim, r|{v} is a maximal-allowed clique
                    all_cliques.append(tuple(sorted(r | {v})))
                p.remove(v)
                x.add(v)

        bron_kerbosch_pivot(set(), set(vertices), set())
            
        return self.__class__.from_simplices(
            all_cliques,
            coefficient_ring=self.coefficient_ring,
            close_under_faces=True
        )

    def simplify(self, backend: str = "auto") -> tuple["SimplicialComplex", dict[tuple, list[tuple]]]:
        """Simplify the simplicial complex into a smaller homotopy equivalent one using Link Condition edge contractions.
        
        **Algorithm:** Iteratively contracts edges (u, v) that satisfy the Link Condition:
        Lk(u) ∩ Lk(v) == Lk(uv), where Lk denotes the link of a simplex.
        This is a topologically rigorous reduction that guarantees homotopy equivalence.
        
        **Preserved Invariants (Homotopy Equivalence):**
        - Fundamental group (π₁) — same generators and relations
        - Homology groups (H_n) — identical Betti numbers
        - Cohomology rings — Cup product structure preserved
        - All derived invariants (Euler characteristic, etc.)
        
        **Returns:**
            tuple: (simplified_complex, simplex_map) where:
                - simplified_complex: A homotopy-equivalent complex with fewer simplices
                - simplex_map: dict[new_simplex] → [original_simplices]
                  Maps each simplex in the simplified complex back to all original simplices
                  that were contracted/merged into it. Use this to lift invariants back to
                  the original geometry.
        
        **Use When:**
        - You need the mapping back to original simplices for lifting invariants
        - Working with large complexes (1000+ simplices) with Julia acceleration
        - You require rigorous topological guarantees via Link Condition
        
        **Example:**
            sc_reduced, simplex_map = sc.simplify()
            # Use simplex_map to lift homology generators back to original points
        """
        from collections import defaultdict
        import numpy as np
        from ..bridge.julia_bridge import julia_engine

        if julia_engine.available:
            try:
                # 1. Pass raw simplices to Julia and let it handle skeletal closure efficiently
                raw_simplices = []
                for d in self.dimensions:
                    raw_simplices.extend(self.n_simplices(d))
                    
                out_simplices, v_map_orig, s_map_raw = julia_engine.simplify_jl(raw_simplices)
                
                new_complex = SimplicialComplex.from_simplices(
                    out_simplices, 
                    close_under_faces=True,
                    coefficient_ring=self.coefficient_ring
                )
                
                if hasattr(self, "_coordinates") and self._coordinates is not None:
                     old_coords = self._coordinates
                     n_v_new = len(v_map_orig)
                     new_coords = np.zeros((n_v_new, old_coords.shape[1]))
                     for new_v, old_vs in v_map_orig.items():
                         new_coords[new_v] = np.mean(old_coords[old_vs], axis=0)
                     new_complex._coordinates = new_coords
                     
                return new_complex, s_map_raw
            except Exception as e:
                import warnings
                warnings.warn(f"Julia Simplify failed ({e!r}). Falling back to pure Python.")
        
        # --- Python Fallback (Link Condition Contraction) ---
        # 1. Prepare skeletal closure in Python
        all_simplices_set = set()
        for d in self.dimensions:
            for simp in self.n_simplices(d):
                s = tuple(sorted(simp))
                for i in range(1, 1 << len(s)):
                    subface = tuple(sorted(s[j] for j in range(len(s)) if (i >> j) & 1))
                    all_simplices_set.add(subface)
        
        all_simplices = list(all_simplices_set)

        V_orig = sorted([s[0] for s in all_simplices if len(s) == 1])
        parent = {v: v for v in V_orig}

        def find(i):
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]

        def union(i, j):
            root_i, root_j = find(i), find(j)
            if root_i != root_j:
                parent[root_j] = root_i
                return True
            return False

        current_simplices = set(all_simplices)

        any_change = True
        while any_change:
            any_change = False

            # Build coface map for fast link lookups
            coface_map = defaultdict(list)
            for s in current_simplices:
                for v in s:
                    coface_map[v].append(s)

            def get_link_fast(sigma):
                sigma_set = set(sigma)
                candidate_simplices = None
                for v in sigma:
                    if v not in coface_map:
                        return set()
                    if candidate_simplices is None:
                        candidate_simplices = set(coface_map[v])
                    else:
                        candidate_simplices.intersection_update(coface_map[v])
                
                if not candidate_simplices:
                    return set()

                link = set()
                for s in candidate_simplices:
                    tau = tuple(sorted(set(s) - sigma_set))
                    if tau: 
                        link.add(tau)
                    else:
                        # Empty set is technically in the link, 
                        # but for intersection purposes we can use an explicit marker if needed.
                        # However, Link(u) \cap Link(v) == Link(uv) usually concerns non-empty simplices.
                        # If we want to be rigorous, we can include the empty set.
                        link.add(())
                return link

            edges = [s for s in current_simplices if len(s) == 2]
            edges.sort()

            for u, v in edges:
                # Check Link Condition for homotopy equivalence
                lk_u = get_link_fast((u,))
                lk_v = get_link_fast((v,))
                lk_uv = get_link_fast((u, v))

                if (lk_u & lk_v) == lk_uv:
                    union(u, v)
                    new_simplices = set()
                    for s in current_simplices:
                        new_s = tuple(sorted(set(u if x == v else x for x in s)))
                        if new_s:
                            new_simplices.add(new_s)
                    current_simplices = new_simplices
                    any_change = True
                    break

        final_roots = sorted(list(set(find(v) for v in V_orig)))
        root_to_new = {root: i for i, root in enumerate(final_roots)}
        def map_vertex(v):
            return root_to_new[find(v)]

        final_simplices, final_s_map = set(), defaultdict(list)
        for orig_s in all_simplices:
            mapped_s = tuple(sorted(set(map_vertex(v) for v in orig_s)))
            final_simplices.add(mapped_s)
            final_s_map[mapped_s].append(orig_s)

        sc_qm = self.__class__.from_simplices(
            list(final_simplices), 
            close_under_faces=True, 
            coefficient_ring=self.coefficient_ring
        )

        if hasattr(self, "_coordinates") and self._coordinates is not None:
            old_coords = self._coordinates
            new_coords = np.zeros((len(final_roots), old_coords.shape[1]))
            new_v_to_orig_vs = defaultdict(list)
            for v_orig in V_orig:
                new_v_to_orig_vs[map_vertex(v_orig)].append(v_orig)
            for new_v, orig_vs in new_v_to_orig_vs.items():
                new_coords[new_v] = np.mean(old_coords[orig_vs], axis=0)
            sc_qm._coordinates = new_coords

        return sc_qm, dict(final_s_map)

    def quick_mapper(
        self,
        max_loops: int = 1,
        min_modularity_gain: float = 1e-6,
        backend: str = "auto",
    ) -> tuple["SimplicialComplex", dict[tuple, list[tuple]]]:
        """Simplify the complex using modularity-based vertex merging (Liu-Xie-Yi 2012).
        
        This method uses a fast label propagation algorithm to group vertices into 
        communities and merge them. Note that this process does NOT preserve topology 
        and is intended for high-performance structural summarization of large data.

        Args:
            max_loops: Maximum number of label propagation iterations.
            min_modularity_gain: Minimum gain required to continue iterations.

        Returns:
            tuple: (simplified_complex, simplex_map)
        """
        from collections import defaultdict
        import random
        import numpy as np
        from ..bridge.julia_bridge import julia_engine

        V = [v[0] for v in self.n_simplices(0)]
        E = self.n_simplices(1)
        
        if not V:
            return self, {}
            
        if julia_engine.available:
            try:
                G_raw = {"V": V, "E": E}
                G_simple, L_dict = julia_engine.quick_mapper_jl(G_raw, max_loops, min_modularity_gain)
                
                # Julia already returned normalized L_dict (old_v -> 0..N-1)
                L_remapped = L_dict
                n_v_new = len(set(L_dict.values()))
                
                s_map = defaultdict(list)
                new_simplices = set()
                
                for d in self.dimensions:
                    for simp in self.n_simplices(d):
                        new_simp = tuple(sorted(set(L_remapped[v] for v in simp)))
                        if new_simp:
                            new_simplices.add(new_simp)
                            s_map[new_simp].append(simp)
                            
                new_complex = SimplicialComplex.from_simplices(
                    list(new_simplices), 
                    close_under_faces=True,
                    coefficient_ring=self.coefficient_ring
                )
                
                if hasattr(self, "_coordinates") and self._coordinates is not None:
                     old_coords = self._coordinates
                     new_coords = np.zeros((n_v_new, old_coords.shape[1]))
                     v_groups = defaultdict(list)
                     for old_v, new_v in L_remapped.items():
                         v_groups[new_v].append(old_v)
                     for new_v, old_vs in v_groups.items():
                         new_coords[new_v] = np.mean(old_coords[old_vs], axis=0)
                     new_complex._coordinates = new_coords
                     
                return new_complex, dict(s_map)
            except Exception as e:
                import warnings
                warnings.warn(f"Julia QuickMapper failed ({e!r}). Falling back to pure Python.")

        # --- Python Fallback (Liu-Xie-Yi 2012) ---
        adj = {v: [] for v in V}
        for u, v in E:
            adj[u].append(v)
            adj[v].append(u)
            
        m = len(E)
        L = {v: v for v in V}
        if m == 0:
            return self, {tuple([v]): [tuple([v])] for v in V}
             
        degree = {v: len(adj[v]) for v in V}
        num_of_loops = 0
        modularity_gain = 1000.0
        two_m = 2.0 * m
        
        while modularity_gain > min_modularity_gain and num_of_loops < max_loops:
            vertex_order = list(V)
            random.shuffle(vertex_order)
            
            for vertex in vertex_order:
                neighbors = adj[vertex]
                if not neighbors:
                    continue
                    
                nbr_labels = set([L[vertex]])
                for nbr in neighbors:
                    nbr_labels.add(L[nbr])
                    
                best_labels = []
                max_contribution = -float('inf')
                deg_v = degree[vertex]
                
                for label in nbr_labels:
                    contribution = 0.0
                    for j in neighbors:
                        if L[j] == label:
                            contribution += (1.0 - (deg_v * degree[j]) / two_m)
                            
                    if contribution > max_contribution:
                        max_contribution = contribution
                        best_labels = [label]
                    elif contribution == max_contribution:
                        best_labels.append(label)
                L[vertex] = random.choice(best_labels)
            modularity_gain = 0.0 
            num_of_loops += 1

        unique_labels = sorted(list(set(L.values())))
        label_to_new = {old: new for new, old in enumerate(unique_labels)}
        
        final_s_map = defaultdict(list)
        final_simplices = set()
        
        for d in self.dimensions:
            for simp in self.n_simplices(d):
                new_simp = tuple(sorted(set(label_to_new[L[v]] for v in simp)))
                if new_simp:
                    final_simplices.add(new_simp)
                    final_s_map[new_simp].append(simp)
        
        sc_qm = self.__class__.from_simplices(list(final_simplices), close_under_faces=True, coefficient_ring=self.coefficient_ring)
        if hasattr(self, "_coordinates") and self._coordinates is not None:
             old_coords = self._coordinates
             new_coords = np.zeros((len(unique_labels), old_coords.shape[1]))
             v_groups = defaultdict(list)
             for old_v, label in L.items():
                 v_groups[label_to_new[label]].append(old_v)
             for new_v, old_vs in v_groups.items():
                 new_coords[new_v] = np.mean(old_coords[old_vs], axis=0)
             sc_qm._coordinates = new_coords
             
        return sc_qm, dict(final_s_map)

    def to_gudhi_simplex_tree(self, *, use_filtration: bool = True):
        """Convert to a GUDHI SimplexTree for advanced TDA operations.

        Args:
            use_filtration: Whether to include filtration values.

        Returns:
            gudhi.SimplexTree: The GUDHI simplex tree object.
        """
        try:
            import gudhi
        except ImportError:
            raise ImportError("gudhi is required for to_gudhi_simplex_tree()")
        st = gudhi.SimplexTree()
        for d in self.dimensions:
            for simplex in self.n_simplices(d):
                if use_filtration:
                    st.insert(simplex, filtration=self.filtration.get(simplex, 0.0))
                else:
                    st.insert(simplex)
        return st

    @classmethod
    def from_gudhi_simplex_tree(cls, st, *, include_filtration: bool = True) -> "SimplicialComplex":
        """Convert a GUDHI SimplexTree to a SimplicialComplex.

        Args:
            st: The GUDHI simplex tree object.
            include_filtration: Whether to extract and include filtration values.

        Returns:
            SimplicialComplex: The resulting simplicial complex.
        """
        simplices = {d: [] for d in range(st.dimension() + 1)}
        filtration = {}
        for simplex, filt in st.get_filtration():
            d = len(simplex) - 1
            s_tuple = tuple(sorted(simplex))
            simplices[d].append(s_tuple)
            if include_filtration:
                filtration[s_tuple] = filt
        return cls(simplices=simplices, filtration=filtration)

    def to_trimesh(self):
        """Convert to a trimesh object.

        Returns:
            trimesh.Trimesh: The resulting trimesh object.
        """
        try:
            import trimesh
        except ImportError:
            raise ImportError("trimesh is required for to_trimesh()")
        
        faces = self.n_simplices(2)
        return trimesh.Trimesh(faces=faces)

    @classmethod
    def from_trimesh(cls, mesh) -> "SimplicialComplex":
        """Convert a trimesh object to a SimplicialComplex.

        Args:
            mesh: The trimesh object.

        Returns:
            SimplicialComplex: The resulting simplicial complex.
        """
        return cls.from_maximal_simplices(mesh.faces.tolist())

    def to_pyg_data(self):
        """Convert to a PyTorch Geometric Data object.

        Returns:
            torch_geometric.data.Data: The resulting PyG Data object.
        """
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError:
            raise ImportError("torch and torch_geometric are required for to_pyg_data()")
            
        edges = self.n_simplices(1)
        edge_index = torch.tensor(list(zip(*edges)), dtype=torch.long)
        return Data(edge_index=edge_index)

    @classmethod
    def from_pyg_data(cls, data) -> "SimplicialComplex":
        """Convert a PyTorch Geometric Data object to a SimplicialComplex.

        Args:
            data: The PyG Data object.

        Returns:
            SimplicialComplex: The resulting simplicial complex.
        """
        edges = data.edge_index.t().tolist()
        return cls.from_simplices(edges)

    @property
    def cells_count(self) -> Dict[int, int]:
        """Return a dictionary mapping dimension to the count of simplices.

        Returns:
            Dict[int, int]: Dictionary mapping dimension to simplex count.
        """
        return {d: len(list(self.n_simplices(d))) for d in self.dimensions}

    def betti_number(self, n: int | None = None, backend: str = "auto") -> int | dict[int, int]:
        """Return the n-th Betti number of the simplicial complex."""
        return self.cellular_chain_complex().betti_number(n, backend=backend)

    def betti_numbers(self, backend: str = "auto") -> dict[int, int]:
        """Return all Betti numbers of the simplicial complex."""
        return self.cellular_chain_complex().betti_numbers(backend=backend)

    def fundamental_group(self, simplify: bool = True, backend: str = "auto"):
        """Compute the fundamental group of the simplicial complex."""
        from .fundamental_group import extract_pi_1
        return extract_pi_1(self.to_cw_complex(), simplify=simplify, backend=backend)

    def collapse(self) -> "SimplicialComplex":
        """Reduce the complex by removing all free faces (simplicial collapses).
        
        Algorithm:
            Iteratively identifies "free faces" (faces with exactly one coface)
            and removes both the free face and its unique coface. This reduces the complex
            size while preserving homotopy equivalence.
        
        Preserved Invariants (Homotopy Equivalence):
            - Fundamental group (π₁) — unchanged
            - Homology groups (H_n) — Betti numbers identical
            - Cohomology rings — Cup product structure preserved
            - All derived invariants (Euler characteristic, etc.)
        
        Returns:
            SimplicialComplex: A homotopy-equivalent complex with fewer simplices.
            Note: No mapping information is provided. Use simplify() if you need
            to track which original simplices were removed.
        
        Use When:
            - You want quick, aggressive size reduction without tracking the mapping
            - Working on small-to-medium complexes (<5000 simplices)
            - You don't need to lift invariants back to original geometry
            - Speed is critical and Julia acceleration is unavailable
        
        Example:
            sc_minimal = sc.collapse()  # Fast reduction, typical 90%+ size reduction
            # Just the minimal homotopy equivalent; no original mapping available
        """
        current_simplices = set()
        for d in self.dimensions:
            for s in self.n_simplices(d):
                current_simplices.add(tuple(sorted(s)))
        
        any_change = True
        while any_change:
            any_change = False
            from collections import defaultdict
            coface_count = defaultdict(int)
            coface_target = {}
            for s in current_simplices:
                n = len(s)
                if n < 2:
                    continue
                for i in range(n):
                    face = s[:i] + s[i+1:]
                    coface_count[face] += 1
                    coface_target[face] = s
            
            free_faces = [f for f, count in coface_count.items() if count == 1 and f in current_simplices]
            if free_faces:
                free_faces.sort(key=len, reverse=True)
                for f in free_faces:
                    if f in current_simplices:
                        tau = coface_target[f]
                        if tau in current_simplices:
                            current_simplices.remove(f)
                            current_simplices.remove(tau)
                            any_change = True
                            break
        
        return SimplicialComplex.from_simplices(list(current_simplices), coefficient_ring=self.coefficient_ring)

    def discrete_morse_gradient(self, backend: str = "auto") -> dict[tuple[int, ...], tuple[int, ...]]:
        """Compute a discrete Morse gradient vector field."""
        from ..bridge.julia_bridge import julia_engine
        import warnings
        backend_norm = str(backend).lower().strip()
        use_julia = (backend_norm == "julia") or (backend_norm == "auto" and julia_engine.available)

        if use_julia:
            try:
                raw_simplices = [list(s) for d in self.dimensions for s in self.n_simplices(d)]
                matching_raw = julia_engine.compute_discrete_morse_gradient_jl(raw_simplices)
                return {tuple(sorted(pair[0])): tuple(sorted(pair[1])) for pair in matching_raw}
            except Exception as e:
                if backend_norm == "julia":
                    raise e
                warnings.warn(f"Julia Morse gradient failed: {e!r}. Falling back to Python.")

        from collections import defaultdict
        matching = {}
        matched = set()
        all_simplices = []
        for d in self.dimensions:
            for s in self.n_simplices(d):
                all_simplices.append(tuple(sorted(s)))
        all_simplices.sort(key=len)
        
        any_change = True
        while any_change:
            any_change = False
            unmatched = [s for s in all_simplices if s not in matched]
            if not unmatched:
                break
            
            unmatched_set = set(unmatched)
            coface_count = defaultdict(int)
            coface_target = {}
            for s in unmatched:
                n = len(s)
                if n < 2:
                    continue
                for i in range(n):
                    face = s[:i] + s[i+1:]
                    if face in unmatched_set:
                        coface_count[face] += 1
                        coface_target[face] = s
            
            free_faces = [f for f, count in coface_count.items() if count == 1]
            if free_faces:
                free_faces.sort(key=len, reverse=True)
                for f in free_faces:
                    if f not in matched:
                        tau = coface_target[f]
                        if tau not in matched:
                            matching[f] = tau
                            matched.add(f)
                            matched.add(tau)
                            any_change = True
                if any_change:
                    continue
            
            for s in unmatched:
                if s not in matched:
                    matched.add(s)
                    any_change = True
                    break
        return matching

    def morse_complex(self, backend: str = "auto") -> "ChainComplex":
        """Construct the minimal Morse chain complex."""
        from collections import defaultdict
        from scipy.sparse import csr_matrix
        import numpy as np

        gradient = self.discrete_morse_gradient(backend=backend)
        all_simplices = sorted([tuple(sorted(s)) for d in self.dimensions for s in self.n_simplices(d)], key=len)
        matched_sigma = set(gradient.keys())
        matched_tau = set(gradient.values())
        critical = [s for s in all_simplices if s not in matched_sigma and s not in matched_tau]
        
        critical_by_dim = defaultdict(list)
        for s in critical:
            critical_by_dim[len(s) - 1].append(s)
            
        morse_boundaries = {}
        max_dim = max(critical_by_dim.keys()) if critical_by_dim else 0
        
        for d in range(1, max_dim + 1):
            source_crit = critical_by_dim[d]
            target_crit = critical_by_dim[d-1]
            if not source_crit or not target_crit:
                morse_boundaries[d] = csr_matrix((len(target_crit), len(source_crit)), dtype=np.int64)
                continue
            
            matrix = np.zeros((len(target_crit), len(source_crit)), dtype=np.int64)
            target_to_idx = {s: i for i, s in enumerate(target_crit)}
            
            for j, tau_crit in enumerate(source_crit):
                counts = defaultdict(int)
                for i in range(len(tau_crit)):
                    sigma = tau_crit[:i] + tau_crit[i+1:]
                    sgn = 1 if i % 2 == 0 else -1
                    counts[sigma] += sgn
                
                active = True
                while active:
                    active = False
                    next_counts = defaultdict(int)
                    for sigma, weight in counts.items():
                        if weight == 0:
                            continue
                        if sigma in target_to_idx:
                            matrix[target_to_idx[sigma], j] += weight
                        elif sigma in gradient:
                            tau_next = gradient[sigma]
                            sgn_sigma = 0
                            for i in range(len(tau_next)):
                                if tau_next[:i] + tau_next[i+1:] == sigma:
                                    sgn_sigma = 1 if i % 2 == 0 else -1
                                    break
                            
                            w_step = -sgn_sigma
                            for i in range(len(tau_next)):
                                s_next = tau_next[:i] + tau_next[i+1:]
                                if s_next != sigma:
                                    sgn_s_next = 1 if i % 2 == 0 else -1
                                    next_counts[s_next] += weight * w_step * sgn_s_next
                                    active = True
                    counts = next_counts
            morse_boundaries[d] = csr_matrix(matrix)
            
        cells = {d: len(crit) for d, crit in critical_by_dim.items()}
        return ChainComplex(
            dimensions=list(range(max_dim + 1)),
            boundaries=morse_boundaries,
            cells=cells,
            coefficient_ring=self.coefficient_ring
        )

    def is_homology_isomorphic(self, other: "SimplicialComplex", backend: str = "auto") -> bool:
        """Check if two simplicial complexes are homology isomorphic."""
        max_d = max(self.dimension, other.dimension)
        for d in range(max_d + 1):
            h1 = self.homology(d, backend=backend)
            h2 = other.homology(d, backend=backend)
            if h1 != h2:
                return False
        return True

    def compute_homology_basis(
        self,
        dimension: int,
        point_cloud: Optional[np.ndarray] = None,
        *,
        mode: Literal["valid", "optimal"] = "valid",
        max_roots: Optional[int] = None,
        root_stride: int = 1,
        max_cycles: Optional[int] = None,
    ):
        """Compute H_k generators directly from this complex.

        Args:
            dimension: Degree k of homology to compute.
            point_cloud: Optional coordinate data for geometric optimization.
            mode: Either "valid" (standard) or "optimal" (shortest-cycle heuristic).
            max_roots: Optional cap on the number of root simplices to probe.
            root_stride: Sampling stride for root simplices.
            max_cycles: Optional cap on the total number of cycles to return.

        Returns:
            The homology basis result.
        """
        from ..core.homology_generators import compute_homology_basis_from_complex

        return compute_homology_basis_from_complex(
            self,
            dimension=dimension,
            point_cloud=point_cloud,
            mode=mode,
            max_roots=max_roots,
            root_stride=root_stride,
            max_cycles=max_cycles,
        )

    def compute_optimal_h1_basis(
        self,
        point_cloud: Optional[np.ndarray] = None,
        *,
        max_roots: Optional[int] = None,
        root_stride: int = 1,
        max_cycles: Optional[int] = None,
    ):
        """Compute an optimal H1 basis directly from this complex.

        Args:
            point_cloud: Optional coordinate data for geometric optimization.
            max_roots: Optional cap on the number of root simplices to probe.
            root_stride: Sampling stride for root simplices.
            max_cycles: Optional cap on the total number of cycles to return.

        Returns:
            The optimal H1 basis result.
        """
        from ..core.homology_generators import compute_optimal_h1_basis_from_complex

        return compute_optimal_h1_basis_from_complex(
            self,
            point_cloud=point_cloud,
            max_roots=max_roots,
            root_stride=root_stride,
            max_cycles=max_cycles,
        )


__all__ = ["SimplicialComplex", "ChainComplex", "CWComplex"]
