import copy
import warnings
import numpy as np
import scipy.sparse as sp
from typing import List, Tuple
import sympy as sympy_module
import scipy.sparse.linalg as spla

from pysurgery.core.intersection_forms import IntersectionForm
from pysurgery.core.cup_product import alexander_whitney_cup
from pysurgery.core.complexes import ChainComplex
from pysurgery.core.exceptions import HomologyError
from pysurgery.bridge.julia_bridge import julia_engine

try:
    import gudhi
    HAS_GUDHI = True
except ImportError:
    HAS_GUDHI = False

def extract_complex_data(simplex_tree):
    """
    Extracts boundary matrices, cells, and simplex mappings from a GUDHI SimplexTree.
    """
    boundaries = {}
    simplices = list(simplex_tree.get_skeleton(simplex_tree.dimension()))
    
    dim_simplices = {}
    for s, _ in simplices:
        d = len(s) - 1
        if d not in dim_simplices:
            dim_simplices[d] = []
        dim_simplices[d].append(tuple(sorted(s)))
        
    for k in dim_simplices:
        dim_simplices[k].sort()
        
    simplex_to_idx = {k: {s: i for i, s in enumerate(dim_simplices[k])} for k in dim_simplices}
    
    for k in range(1, simplex_tree.dimension() + 1):
        if k not in dim_simplices or k-1 not in dim_simplices:
            continue
        
        rows, cols, data = [], [], []
        prev_dim_map = simplex_to_idx[k-1]
        
        for j, simplex in enumerate(dim_simplices[k]):
            for i in range(k + 1):
                face = tuple(simplex[:i] + simplex[i+1:])
                if face in prev_dim_map:
                    rows.append(prev_dim_map[face])
                    cols.append(j)
                    data.append((-1)**i)
                    
        n_rows = len(dim_simplices[k-1])
        n_cols = len(dim_simplices[k])
        boundaries[k] = sp.csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols), dtype=np.int64)
        
    cells = {d: len(dim_simplices[d]) for d in dim_simplices}
    return boundaries, cells, dim_simplices, simplex_to_idx

def simplex_tree_to_intersection_form(simplex_tree) -> IntersectionForm:
    """
    Automatically derives the rigorous Intersection Form Q for a 4D manifold
    directly from its GUDHI SimplexTree filtration.
    
    It extracts the 2-cohomology basis, evaluates the Alexander-Whitney Cup Product,
    and applies it to the fundamental class [M].
    """
    boundaries, cells, dim_simplices, simplex_to_idx = extract_complex_data(simplex_tree)
    complex_c = ChainComplex(boundaries=boundaries, dimensions=list(cells.keys()))
    
    basis_2 = complex_c.cohomology_basis(2)
    
    if not basis_2 or 4 not in cells:
        return IntersectionForm(matrix=np.zeros((0,0), dtype=int), dimension=4)
        
    # Find the fundamental class [M] in H_4(X)
    if 4 in boundaries:
        fund_class_found = False
        
        if julia_engine.available:
            try:
                # We need the nullspace of d4. compute_sparse_cohomology_basis computes nullspace of d_np1.T
                # So we pass d4.T to find ker(d4).
                basis_4 = julia_engine.compute_sparse_cohomology_basis(boundaries[4].T, None)
                if len(basis_4) > 0:
                    fund_class = basis_4[0].flatten()
                    fund_class_found = True
            except Exception as e:
                warnings.warn(f"Topological Hint: Julia bridge failed to extract [M] ({e}). Falling back to SymPy exact nullspace.")
                
        if not fund_class_found:
            # Attempt SymPy exact nullspace for small/medium matrices
            try:
                sym_d4 = sympy_module.Matrix(boundaries[4].toarray())
                null_4 = sym_d4.nullspace()
                
                if null_4:
                    fund_class = np.array(null_4[0]).astype(float).flatten()
                    denoms = [sympy_module.fraction(x)[1] for x in null_4[0]]
                    lcm = np.lcm.reduce([int(d) for d in denoms])
                    fund_class = np.array([int(x * lcm) for x in fund_class], dtype=np.int64)
                    gcd = np.gcd.reduce(fund_class)
                    if gcd != 0:
                        fund_class = fund_class // gcd
                    fund_class_found = True
            except Exception as e:
                warnings.warn(f"Topological Hint: SymPy exact nullspace failed ({e}). This dataset is too massive for exact integer algebra. Falling back to floating-point SVD to approximate [M] over R.")
                
        if not fund_class_found:
            # Fallback to Sparse SVD for massive datasets when Julia is unavailable
            d4_sparse = boundaries[4].astype(float)
            try:
                u, s, vt = spla.svds(d4_sparse, k=min(cells[4]-1, 5), which='SM')
                tol = cells[4] * np.finfo(float).eps * max(s) if len(s) > 0 else 1e-10
                null_idx = np.where(s <= tol)[0]
                if len(null_idx) > 0:
                    fund_class = vt[null_idx[0], :].flatten()
                    fund_class_found = True
            except Exception as e:
                warnings.warn(f"Topological Hint: Sparse SVD failed to converge for [M] ({e}). The simplicial complex may be too topologically degenerate.")

        if not fund_class_found:
            raise HomologyError("No fundamental class [M] found (H_4 is empty or computation failed). "
                                "Topological translation: The simplicial complex does not represent a closed, orientable 4-manifold. The Cup Product cannot be evaluated without [M].")
    else:
        # d4 is not in the boundary dict even though 4-simplices exist.
        # This means the chain complex is incomplete; we cannot recover [M] safely.
        if cells[4] == 1:
            # Special case: exactly one 4-simplex → the all-ones vector is correct.
            fund_class = np.ones(1, dtype=np.int64)
            fund_class_found = True
        else:
            raise HomologyError(
                f"Cannot determine fundamental class [M]: {cells[4]} 4-simplices found but "
                "the boundary map d4 is absent from the chain complex. "
                "Provide a complete chain complex with all boundary operators."
            )
        
    r = len(basis_2)
    
    # Determine type of Q based on basis
    is_float = False
    if r > 0 and basis_2[0].dtype.kind == 'f':
        is_float = True
        
    dtype = np.float64 if is_float else np.int64
    Q = np.zeros((r, r), dtype=dtype)
    
    simplices_4 = dim_simplices.get(4, [])
    idx_2 = simplex_to_idx.get(2, {})
    
    for i in range(r):
        for j in range(r):
            cup_ij = alexander_whitney_cup(
                alpha=basis_2[i],
                beta=basis_2[j],
                p=2, q=2,
                simplices_p_plus_q=simplices_4,
                simplex_to_idx_p=idx_2,
                simplex_to_idx_q=idx_2
            )
            Q[i, j] = np.sum(cup_ij * fund_class)
            
    # Due to chain level artifacts, enforce perfect symmetry on the cohomology level matrix
    if not np.array_equal(Q, Q.T):
        warnings.warn("Topological Hint: Cup product matrix is not strictly symmetric. Forcing symmetry may corrupt the intersection form.")
        
    if is_float:
        Q_sym = (Q + Q.T) / 2.0
    else:
        Q_sym = np.round((Q + Q.T) / 2.0).astype(np.int64)
    
    return IntersectionForm(matrix=Q_sym, dimension=4)

def extract_persistence_to_surgery(simplex_tree, min_persistence=0.5):
    """
    Analyzes a GUDHI SimplexTree's persistence diagram.
    Identifies 'long-lived' cycles and attempts to construct an algebraic surgery plan 
    for them, essentially applying surgery theory to denoise the manifold's topology.
    
    Parameters
    ----------
    simplex_tree : gudhi.SimplexTree
        A filtered simplicial complex.
    min_persistence : float
        The threshold for a cycle to be considered 'real' topological data rather than noise.
        
    Returns
    -------
    list
        A list of dimensions where surgery might be required.
    """
    if not HAS_GUDHI:
        raise ImportError("GUDHI is required. Install via 'pip install gudhi'.")
        
    simplex_tree.compute_persistence()
    persistence = simplex_tree.persistence()
    
    surgery_targets = []
    
    for dim, (birth, death) in persistence:
        # If a feature dies, it has a finite lifetime
        if death != float('inf'):
            lifetime = death - birth
            if lifetime > min_persistence:
                # This is a significant feature. If we assume the underlying space 
                # should be a sphere, this is an obstruction!
                surgery_targets.append({
                    "dimension": dim,
                    "birth": birth,
                    "death": death,
                    "recommendation": f"Perform surgery on {dim}-cycle to remove feature."
                })
                
    return surgery_targets

def signature_landscape(simplex_tree) -> List[Tuple[float, int]]:
    """
    A novel TDA invariant: The Signature Landscape.
    Instead of Betti numbers, we track the evolution of the intersection form's signature 
    across a sequence of filtered simplicial complexes (representing a filtration).

    Note: complexity is O(N·C) where N is the number of unique filtration values and C is
    the cost of one ``simplex_tree_to_intersection_form`` call.  For large filtrations, this
    can be expensive.  Only 4-dimensional subcomplexes yield non-zero signatures; a warning
    is emitted if the complex has no 4-simplices.

    Parameters
    ----------
    simplex_tree : gudhi.SimplexTree
        A filtered GUDHI SimplexTree.
        
    Returns
    -------
    List[Tuple[float, int]]
        The sequence of (filtration_value, signature) over the filtration.
    """
    if not HAS_GUDHI:
        raise ImportError("GUDHI is required. Install via 'pip install gudhi'.")

    max_dim = simplex_tree.dimension()
    if max_dim < 4:
        warnings.warn(
            f"Topological Hint: signature_landscape computes intersection forms on 4-simplices. "
            f"This complex has dimension {max_dim}. All signature values will be 0."
        )

    signatures = []
    # Get all unique filtration values where new simplices appear.
    filtration_values = sorted(set(f_val for _, f_val in simplex_tree.get_filtration()))

    for val in filtration_values:
        # Build the subcomplex up to this filtration value using deepcopy + prune,
        # avoiding a full quadratic reconstruction from scratch.
        st_sub = copy.deepcopy(simplex_tree)
        st_sub.prune_above_filtration(val)

        try:
            q_form = simplex_tree_to_intersection_form(st_sub)
            signatures.append((val, q_form.signature()))
        except Exception:
            # At early filtration steps the subcomplex may not form a closed 4-manifold,
            # so the fundamental class or intersection form may not exist.
            signatures.append((val, 0))

    return signatures
