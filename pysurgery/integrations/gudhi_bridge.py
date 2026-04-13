import warnings
from functools import reduce
from math import gcd, lcm
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

try:
    HAS_SCIPY_SPATIAL = True
except ImportError:
    HAS_SCIPY_SPATIAL = False

_SLOW_BOUNDARY_FALLBACK_WARNED = False


def _warn_slow_boundary_fallback(reason: str) -> None:
    global _SLOW_BOUNDARY_FALLBACK_WARNED
    if _SLOW_BOUNDARY_FALLBACK_WARNED:
        return
    warnings.warn(
        f"Topological Hint: {reason} Using slower pure-Python boundary assembly for chain-complex extraction. "
        "Install/enable Julia for large simplicial complexes.",
    )
    _SLOW_BOUNDARY_FALLBACK_WARNED = True


def _extract_complex_data_python(simplex_tree, simplices=None, max_dim=None):
    boundaries = {}
    if max_dim is None:
        max_dim = simplex_tree.dimension()
    if simplices is None:
        simplices = list(simplex_tree.get_skeleton(max_dim))

    dim_simplices = {}
    for s, _ in simplices:
        d = len(s) - 1
        if d not in dim_simplices:
            dim_simplices[d] = []
        dim_simplices[d].append(tuple(s))

    for k in dim_simplices:
        dim_simplices[k].sort()

    simplex_to_idx = {k: {s: i for i, s in enumerate(dim_simplices[k])} for k in dim_simplices}

    for k in range(1, max_dim + 1):
        if k not in dim_simplices or k - 1 not in dim_simplices:
            continue

        rows, cols, data = [], [], []
        prev_dim_map = simplex_to_idx[k - 1]

        for j, simplex in enumerate(dim_simplices[k]):
            for i in range(k + 1):
                face = tuple(simplex[:i] + simplex[i + 1:])
                if face in prev_dim_map:
                    rows.append(prev_dim_map[face])
                    cols.append(j)
                    data.append((-1) ** i)

        n_rows = len(dim_simplices[k - 1])
        n_cols = len(dim_simplices[k])
        boundaries[k] = sp.csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols), dtype=np.int64)

    cells = {d: len(dim_simplices[d]) for d in dim_simplices}
    return boundaries, cells, dim_simplices, simplex_to_idx

def extract_complex_data(simplex_tree, *, include_metadata: bool = True):
    """
    Extracts boundary matrices, cells, and simplex mappings from a GUDHI SimplexTree.
    """
    simplices = list(simplex_tree.get_skeleton(simplex_tree.dimension()))
    max_dim = simplex_tree.dimension()

    if julia_engine.available:
        try:
            simplex_entries = [s for s, _ in simplices]
            if include_metadata:
                boundary_payload, cells, dim_simplices, simplex_to_idx = julia_engine.compute_boundary_data_from_simplices(
                    simplex_entries,
                    max_dim,
                )
            else:
                boundary_payload, cells = julia_engine.compute_boundary_payload_from_simplices(
                    simplex_entries,
                    max_dim,
                    include_metadata=False,
                )
                cells = {int(k): int(v) for k, v in dict(cells).items()}
                dim_simplices, simplex_to_idx = {}, {}
            boundaries = {}
            for k, payload in boundary_payload.items():
                boundaries[k] = sp.csr_matrix(
                    (payload["data"], (payload["rows"], payload["cols"])),
                    shape=(payload["n_rows"], payload["n_cols"]),
                    dtype=np.int64,
                )
            return boundaries, cells, dim_simplices, simplex_to_idx
        except Exception as e:
            _warn_slow_boundary_fallback(
                f"Julia boundary assembly failed ({e!r})."
            )
            return _extract_complex_data_python(simplex_tree, simplices=simplices, max_dim=max_dim)

    _warn_slow_boundary_fallback("Julia backend unavailable.")
    return _extract_complex_data_python(simplex_tree, simplices=simplices, max_dim=max_dim)


def extract_boundary_chain_data(simplex_tree):
    """Lightweight extraction for callers that only need boundary operators and cell counts."""
    boundaries, cells, _, _ = extract_complex_data(simplex_tree, include_metadata=False)
    return boundaries, cells

def simplex_tree_to_intersection_form(simplex_tree, allow_approx: bool = False) -> IntersectionForm:
    """
    Automatically derives the rigorous Intersection Form Q for a 4D manifold
    directly from its GUDHI SimplexTree filtration.
    
    It extracts the 2-cohomology basis, evaluates the Alexander-Whitney Cup Product,
    and applies it to the fundamental class [M].
    """
    boundaries, cells, dim_simplices, simplex_to_idx = extract_complex_data(simplex_tree)
    complex_c = ChainComplex(boundaries=boundaries, dimensions=list(cells.keys()), coefficient_ring="Z")

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
                basis_4 = julia_engine.compute_sparse_cohomology_basis(boundaries[4].T, None, cn_size=cells[4])
                if len(basis_4) > 0:
                    fund_class = basis_4[0].flatten()
                    fund_class_found = True
            except Exception as e:
                msg = f"Topological Hint: Julia bridge failed to extract [M] ({e!r}). Falling back to SymPy exact nullspace."
                warnings.warn(msg)
                
        if not fund_class_found:
            # Attempt SymPy exact nullspace for small/medium matrices
            try:
                sym_d4 = sympy_module.Matrix(boundaries[4].toarray())
                null_4 = sym_d4.nullspace()
                
                if null_4:
                    null_vec = null_4[0]
                    denoms = [sympy_module.fraction(x)[1] for x in null_vec]
                    common_lcm = reduce(lcm, (int(d) for d in denoms), 1)
                    fund_class = np.array(
                        [int(sympy_module.Integer(x * common_lcm)) for x in null_vec],
                        dtype=np.int64,
                    )
                    gcd_val = reduce(gcd, (abs(int(v)) for v in fund_class.tolist()), 0)
                    if gcd_val > 1:
                        fund_class = fund_class // gcd_val
                    fund_class_found = True
            except Exception as e:
                if allow_approx:
                    msg = (
                        f"Topological Hint: SymPy exact nullspace failed ({e!r}). This dataset is too massive for exact integer algebra. "
                        "Falling back to floating-point SVD to approximate [M] over R."
                    )
                    warnings.warn(msg)
                else:
                    msg = (
                        f"Topological Hint: SymPy exact nullspace failed ({e!r}). "
                        "Exact fallback to SVD is disabled unless allow_approx=True."
                    )
                    warnings.warn(msg)

        if not fund_class_found:
            if not allow_approx:
                raise HomologyError(
                    "Exact fundamental class extraction failed without Julia. "
                    "Install Julia for fast exact sparse algebra or rerun with allow_approx=True."
                )

            warnings.warn(
                "APPROXIMATION FALLBACK: Using floating-point SVD to estimate [M]. "
                "This may lose exact integer cycle information."
            )

            # Explicitly opt-in approximate fallback via Sparse SVD.
            d4_sparse = boundaries[4].astype(float)
            try:
                if cells[4] == 1:
                    col = d4_sparse.toarray()[:, 0]
                    if np.allclose(col, 0.0, atol=1e-12):
                        fund_class = np.array([1], dtype=np.int64)
                        fund_class_found = True
                else:
                    k_svd = min(cells[4] - 1, 5)
                    u, s, vt = spla.svds(d4_sparse, k=k_svd, which='SM')
                    tol = cells[4] * np.finfo(float).eps * max(s) if len(s) > 0 else 1e-10
                    null_idx = np.where(s <= tol)[0]
                    if len(null_idx) > 0:
                        approx = vt[null_idx[0], :].flatten()
                        fund_candidate = np.round(approx).astype(np.int64)
                        if np.linalg.norm(d4_sparse @ fund_candidate) <= 1e-6:
                            fund_class = fund_candidate
                            fund_class_found = True
                        else:
                            warnings.warn(
                                "SVD-derived integer candidate is not a numerical cycle (d4*[M] != 0 within tolerance)."
                            )
            except Exception as e:
                msg = (
                    f"Topological Hint: Sparse SVD failed to converge for [M] ({e!r}). "
                    "The simplicial complex may be too topologically degenerate."
                )
                warnings.warn(msg)

        if not fund_class_found:
            raise HomologyError("No fundamental class [M] found (H_4 is empty or computation failed). "
                                "Topological translation: The simplicial complex does not represent a closed, orientable 4-manifold. The Cup Product cannot be evaluated without [M].")
    else:
        raise HomologyError("Fundamental class requires topological boundary data. Cannot default to vector of ones.")
        
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
    if not np.allclose(Q, Q.T, atol=1e-8):
        raise HomologyError("Cup product matrix is not symmetric — indicates a basis computation failure. Forcing symmetry may corrupt the intersection form.")

    if is_float:
        Q_sym = (Q + Q.T) / 2.0
    else:
        if not np.array_equal(Q, Q.T):
            raise HomologyError("Cup product matrix is not exactly symmetric over Z.")
        Q_sym = Q.astype(np.int64)

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

def signature_landscape(simplex_tree, allow_approx: bool = False) -> List[Tuple[float, int]]:
    """
    A novel TDA invariant: The Signature Landscape.
    Instead of Betti numbers, we track the evolution of the intersection form's signature 
    across a sequence of filtered simplicial complexes (representing a filtration).

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

    signatures = []
    if simplex_tree.dimension() != 4:
        import warnings
        warnings.warn("signature_landscape currently hard-codes dimension 4. Other dimensions will return signature 0.")

    st_sub = gudhi.SimplexTree()
    filtration = sorted(simplex_tree.get_filtration(), key=lambda x: x[1])
    filtration_values = sorted(list(set([s[1] for s in filtration])))

    idx = 0
    for val in filtration_values:
        while idx < len(filtration) and filtration[idx][1] <= val:
            st_sub.insert(filtration[idx][0], filtration[idx][1])
            idx += 1

        try:
            q_form = simplex_tree_to_intersection_form(st_sub, allow_approx=allow_approx)
            signatures.append((val, q_form.signature()))
        except Exception:
            if allow_approx:
                signatures.append((val, 0))
            else:
                raise

    return signatures

def triangulate_surface_python(points: np.ndarray, tolerance: float = 1e-10) -> gudhi.SimplexTree:
    """
    Pure Python implementation: Triangulates a 2D surface from a point cloud.
    
    Geometric approach:
    1. Centers the point cloud at origin
    2. Projects points onto a local tangent plane via PCA
    3. Performs Delaunay triangulation in 2D
    4. Returns GUDHI SimplexTree with edges and faces
    
    Parameters
    ----------
    points : np.ndarray
        Point cloud of shape (n_points, 3) embedded in 3D space
    tolerance : float
        Tolerance for detecting degenerate cases
        
    Returns
    -------
    gudhi.SimplexTree
        Simplex tree with 0-cells (vertices), 1-cells (edges), 2-cells (faces)
    """
    if not HAS_GUDHI:
        raise ImportError("GUDHI is required. Install via 'pip install gudhi'.")
    
    points = np.asarray(points, dtype=np.float64)
    if points.shape[1] != 3:
        raise ValueError("Points must be 3D coordinates (shape: n_points × 3)")
    if points.shape[0] < 3:
        raise ValueError("At least 3 points required for triangulation")
    
    # Center the point cloud
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    
    # Perform PCA to find the surface plane
    U, S, Vt = np.linalg.svd(centered, full_matrices=True)
    
    # The first two principal components define the surface plane
    # The third (smallest) singular value represents normal direction
    if S[2] > tolerance:
        warnings.warn(
            f"Topological Hint: Point cloud has significant variance in normal direction ({S[2]:.2e}). "
            "Surface may not be truly 2D. Proceeding with best-fit plane."
        )
    
    # Project points onto 2D tangent plane
    v1 = Vt[0, :]  # First principal direction
    v2 = Vt[1, :]  # Second principal direction
    
    projected_2d = np.column_stack([
        centered @ v1,
        centered @ v2
    ])
    
    # Perform Delaunay triangulation in 2D
    try:
        from scipy.spatial import Delaunay
        delaunay = Delaunay(projected_2d)
        faces = delaunay.simplices
    except Exception as e:
        raise HomologyError(f"Delaunay triangulation failed: {e}")
    
    # Build SimplexTree with vertices, edges, and faces
    st = gudhi.SimplexTree()
    
    # Insert all vertices (0-simplices)
    for i in range(len(points)):
        st.insert([i])
    
    # Insert all edges (1-simplices) from faces
    edges_set = set()
    for face in faces:
        v0, v1, v2 = face
        edges_set.add((min(v0, v1), max(v0, v1)))
        edges_set.add((min(v1, v2), max(v1, v2)))
        edges_set.add((min(v0, v2), max(v0, v2)))
    
    for edge in edges_set:
        st.insert(list(edge))
    
    # Insert all faces (2-simplices)
    for face in faces:
        st.insert(sorted(face))
    
    return st


def triangulate_surface(points: np.ndarray, tolerance: float = 1e-10) -> gudhi.SimplexTree:
    """
    Triangulates a 2D surface from a point cloud using Julia acceleration when available.
    
    Automatically detects surface geometry without requiring parameter tuning.
    Works for a single connected component.
    
    Geometric approach:
    1. Centers the point cloud at origin
    2. Projects points onto a local tangent plane via PCA
    3. Performs Delaunay triangulation in 2D
    4. Returns GUDHI SimplexTree with edges and faces
    
    Parameters
    ----------
    points : np.ndarray
        Point cloud of shape (n_points, 3) embedded in 3D space.
        Should represent a single connected 2D surface.
    tolerance : float
        Tolerance for detecting degenerate cases (default: 1e-10)
        
    Returns
    -------
    gudhi.SimplexTree
        Simplex tree with:
        - 0-cells (vertices) for each input point
        - 1-cells (edges) connecting vertices
        - 2-cells (faces) forming the triangulation
        
    Raises
    ------
    ImportError
        If GUDHI is not installed
    ValueError
        If points are not 3D or fewer than 3 points provided
    HomologyError
        If Delaunay triangulation fails
        
    Notes
    -----
    - Automatically determines surface orientation via PCA
    - No parameter tuning required (unlike GUDHI's max_edge_length)
    - Julia acceleration is always preferred when available
    
    Examples
    --------
    >>> points = np.random.randn(100, 3) * 5
    >>> points = points / np.linalg.norm(points, axis=1, keepdims=True)  # Project to sphere
    >>> st = triangulate_surface(points)
    >>> print(f"Vertices: {st.num_vertices()}, Faces: {st.num_simplices()}")
    """
    points = np.asarray(points, dtype=np.float64)
    
    # Prefer Julia acceleration whenever available.
    if julia_engine.available:
        try:
            triangles = julia_engine.triangulate_surface_delaunay(points, tolerance)
            
            # Build SimplexTree from triangles
            st = gudhi.SimplexTree()
            
            # Insert vertices
            for i in range(len(points)):
                st.insert([i])
            
            # Insert edges
            edges_set = set()
            for tri in triangles:
                v0, v1, v2 = tri
                edges_set.add((min(v0, v1), max(v0, v1)))
                edges_set.add((min(v1, v2), max(v1, v2)))
                edges_set.add((min(v0, v2), max(v0, v2)))
            
            for edge in edges_set:
                st.insert(list(edge))
            
            # Insert faces
            for tri in triangles:
                st.insert(sorted(tri))
            
            return st
        except Exception as e:
            warnings.warn(
                f"Topological Hint: Julia surface triangulation failed ({e!r}). "
                "Falling back to slower pure-Python Delaunay triangulation."
            )
    
    # Python fallback
    return triangulate_surface_python(points, tolerance)


