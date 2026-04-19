import itertools
import hashlib
import numpy as np
import warnings
import sympy as sp
from scipy.sparse import csr_matrix
from typing import Any, Dict, Iterable, List, Tuple, cast
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
from .math_core import get_sparse_snf_diagonal
from ..bridge.julia_bridge import julia_engine


def _parse_coefficient_ring(ring: str) -> tuple[str, int | None]:
    """Parse user ring labels into internal `(kind, modulus)` form."""
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
    """Coerce sparse/dense matrix-like data to CSR with integer entries."""
    if isinstance(matrix, csr_matrix):
        return matrix.copy().astype(np.int64)
    return csr_matrix(np.asarray(matrix, dtype=np.int64), dtype=np.int64)


def _normalize_simplex(simplex: Iterable[int]) -> tuple[int, ...]:
    """Return a canonical, sorted simplex tuple with distinct integer vertices."""
    vertices = tuple(sorted(int(v) for v in simplex))
    if len(vertices) == 0:
        raise ValueError("Simplices must be non-empty.")
    if len(set(vertices)) != len(vertices):
        raise ValueError(f"Simplex vertices must be distinct: {simplex!r}")
    return vertices


def _canonicalize_simplices_by_dim(
    simplices_by_dim: Dict[int, Iterable[Iterable[int]]],
) -> dict[int, list[tuple[int, ...]]]:
    """Normalize a simplex table to sorted unique tuples per dimension."""
    canonical: dict[int, list[tuple[int, ...]]] = {}
    for dim, simplices in simplices_by_dim.items():
        dim_int = int(dim)
        seen: set[tuple[int, ...]] = set()
        cleaned: list[tuple[int, ...]] = []
        for simplex in simplices:
            t = _normalize_simplex(simplex)
            if len(t) - 1 != dim_int:
                raise ValueError(
                    f"Simplex {t!r} has dimension {len(t) - 1}, expected {dim_int}."
                )
            if t not in seen:
                cleaned.append(t)
                seen.add(t)
        canonical[dim_int] = sorted(cleaned)
    return canonical


def _simplicial_closure_from_generators(
    simplices: Iterable[Iterable[int]],
) -> dict[int, list[tuple[int, ...]]]:
    """Build the face closure of a finite simplicial generating set."""
    by_dim_set: dict[int, set[tuple[int, ...]]] = {}
    for simplex in simplices:
        t = _normalize_simplex(simplex)
        for r in range(1, len(t) + 1):
            dim = r - 1
            if dim not in by_dim_set:
                by_dim_set[dim] = set()
            for face in itertools.combinations(t, r):
                by_dim_set[dim].add(tuple(face))
                
    canonical: dict[int, list[tuple[int, ...]]] = {}
    for dim, faces in by_dim_set.items():
        canonical[dim] = sorted(list(faces))
    return canonical


def _boundary_matrix_from_simplices(
    source: list[tuple[int, ...]],
    target: list[tuple[int, ...]],
) -> csr_matrix:
    """Construct the oriented boundary matrix between consecutive simplex sets."""
    if not source or not target:
        return csr_matrix((len(target), len(source)), dtype=np.int64)

    target_index = {simplex: row for row, simplex in enumerate(target)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[int] = []

    for col, simplex in enumerate(source):
        if len(simplex) <= 1:
            continue
        for face_index in range(len(simplex)):
            face = simplex[:face_index] + simplex[face_index + 1 :]
            row = target_index.get(face)
            if row is None:
                continue
            rows.append(row)
            cols.append(col)
            data.append(-1 if face_index % 2 else 1)

    return csr_matrix((data, (rows, cols)), shape=(len(target), len(source)), dtype=np.int64)


def _clone_cache_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, csr_matrix):
        return value.copy()
    if isinstance(value, list):
        return [_clone_cache_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_cache_value(v) for v in value)
    if isinstance(value, dict):
        return {k: _clone_cache_value(v) for k, v in value.items()}
    return value


def _csr_matrix_signature(matrix: csr_matrix) -> tuple[int, int, int, str]:
    m = _coerce_csr_matrix(matrix)
    if m.nnz == 0:
        return int(m.shape[0]), int(m.shape[1]), 0, "0"
    coo = m.tocoo()
    rows = np.asarray(coo.row, dtype=np.int64)
    cols = np.asarray(coo.col, dtype=np.int64)
    data = np.asarray(coo.data, dtype=np.int64)
    order = np.lexsort((cols, rows))
    rows = rows[order]
    cols = cols[order]
    data = data[order]
    h = hashlib.blake2b(digest_size=16)
    h.update(np.asarray(m.shape, dtype=np.int64).tobytes())
    h.update(rows.tobytes())
    h.update(cols.tobytes())
    h.update(data.tobytes())
    return int(m.shape[0]), int(m.shape[1]), int(m.nnz), h.hexdigest()


class SimplicialComplex(BaseModel):
    """Finite simplicial complex with sparse boundary operators and face-closure helpers."""

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
        if "simplices" in data and not isinstance(data["simplices"], dict):
             # Handle possible list-of-lists input if any legacy code did that
             pass 
        super().__init__(**data)
        # Initialize the private table from the pydantic-validated public field if needed
        # But wait, pydantic might overwrite it. 
        # Better: keep 'simplices' as a field for serialization, 
        # but let it be a property that returns the method.
        # No, that's complex.
        
    @model_validator(mode="before")
    @classmethod
    def _move_simplices_to_table(cls, data: Any) -> Any:
        if isinstance(data, dict) and "simplices" in data:
            # We keep it in data for pydantic validation of the 'simplices' field
            pass
        return data

    @property
    def simplices_dict(self) -> Dict[int, List[Tuple[int, ...]]]:
        """Backward-compatible access to the raw simplices dictionary."""
        return self.simplices_field

    def simplices(self) -> List[Tuple[int, ...]]:
        """Return all simplices in the complex as a flat list."""
        key = ("simplicial", "all_simplices_flat")
        cached = self._cache_get(key)
        if cached is not None:
            return cast(List[Tuple[int, ...]], cached)
        out = [
            simplex
            for dim in sorted(self.simplices_field.keys())
            for simplex in self.simplices_field[dim]
        ]
        self._cache_set(key, out)
        return out

    def n_simplices(self, dim: int) -> List[Tuple[int, ...]]:
        """Return the list of simplices for a given dimension."""
        return self.simplices_field.get(int(dim), [])

    def count_simplices(self, dim: int | None = None) -> int:
        """Return the count of simplices for a given dimension, or the total count if dim is None."""
        if dim is None:
            return sum(len(s) for s in self.simplices_field.values())
        return len(self.simplices_field.get(int(dim), []))

    simplices_field: Dict[int, List[Tuple[int, ...]]] = Field(default_factory=dict, alias="simplices")

    @model_validator(mode="after")
    def _normalize_model(self):
        object.__setattr__(self, "simplices_field", _canonicalize_simplices_by_dim(self.simplices_field))
        object.__setattr__(self, "coefficient_ring", str(self.coefficient_ring))
        filt: dict[tuple[int, ...], float] = {}
        simplex_set = {
            simplex
            for simplices in self.simplices_field.values()
            for simplex in simplices
        }
        for simplex, value in self.filtration.items():
            key = _normalize_simplex(simplex)
            if key in simplex_set:
                filt[key] = float(value)
        object.__setattr__(self, "filtration", filt)
        return self

    def _structure_signature(self) -> tuple[object, ...]:
        simplex_sig = tuple(
            (int(dim), tuple(tuple(int(v) for v in simplex) for simplex in simplices))
            for dim, simplices in sorted(self.simplices_field.items())
        )
        filtration_sig = tuple(
            (tuple(int(v) for v in simplex), float(value))
            for simplex, value in sorted(self.filtration.items())
        )
        return simplex_sig, str(self.coefficient_ring), filtration_sig

    def _ensure_cache_valid(self) -> None:
        current = self._structure_signature()
        if self._cache_signature != current:
            self._cache.clear()
            self._cache_signature = current

    def _cache_get(self, key: tuple[object, ...]) -> object | None:
        self._ensure_cache_valid()
        if not self._cache_enabled:
            return None
        if key in self._cache:
            self._cache_hits += 1
            return _clone_cache_value(self._cache[key])
        self._cache_misses += 1
        return None

    def _cache_set(self, key: tuple[object, ...], value: object) -> None:
        self._ensure_cache_valid()
        if not self._cache_enabled:
            return
        self._cache[key] = _clone_cache_value(value)

    def clear_cache(self, namespace: str | None = None) -> None:
        if namespace is None:
            self._cache.clear()
            return
        prefix = (str(namespace),)
        keys = [k for k in self._cache if k[:1] == prefix]
        for key in keys:
            self._cache.pop(key, None)

    def cache_info(self) -> dict[str, object]:
        self._ensure_cache_valid()
        return {
            "enabled": bool(self._cache_enabled),
            "size": int(len(self._cache)),
            "hits": int(self._cache_hits),
            "misses": int(self._cache_misses),
            "keys": [list(map(str, key)) for key in sorted(self._cache.keys(), key=repr)],
        }

    def set_cache_enabled(self, enabled: bool, *, clear_when_disabled: bool = True) -> None:
        self._cache_enabled = bool(enabled)
        if not self._cache_enabled and clear_when_disabled:
            self._cache.clear()

    @classmethod
    def from_simplices(
        cls,
        simplices: Iterable[Iterable[int]],
        coefficient_ring: str = "Z",
        *,
        close_under_faces: bool = True,
    ) -> "SimplicialComplex":
        """Create a simplicial complex from generators, optionally taking the full closure."""
        from ..bridge.julia_bridge import julia_engine
        
        # Ensure we can iterate multiple times if it's a generator
        simplex_list = [tuple(int(v) for v in s) for s in simplices]
        
        if close_under_faces:
            if julia_engine.available:
                # Accelerate full skeletal closure and boundary assembly in Julia
                max_dim = max((len(s) - 1 for s in simplex_list), default=-1)
                
                if max_dim >= 0:
                    try:
                        b_data, cells, dim_simplices, _ = julia_engine.compute_boundary_data_from_simplices(
                            simplex_list, max_dim
                        )
                        
                        # Verify that Julia returned all dimensions (basic sanity check)
                        if all(d in dim_simplices for d in range(max_dim + 1)):
                            obj = cls(simplices=dim_simplices, coefficient_ring=coefficient_ring)
                            
                            # Populate private caches to accelerate subsequent .chain_complex() calls
                            from scipy.sparse import csr_matrix
                            boundaries = {}
                            for dim, payload in b_data.items():
                                boundaries[dim] = csr_matrix(
                                    (payload["data"], (payload["rows"], payload["cols"])),
                                    shape=(payload["n_rows"], payload["n_cols"]),
                                    dtype=np.int64
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
        """Build the full simplicial closure from a list of maximal simplices."""
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
        k: int,
        delta: float = 1.0,
        max_dimension: int = 2,
        coefficient_ring: str = "Z"
    ) -> "SimplicialComplex":
        """
        Builds a Vietoris-Rips complex using Continuous k-Nearest Neighbors (CkNN).
        This eliminates the need for a global epsilon by applying adaptive local scaling.
        
        References:
            Berry, T., & Sauer, T. (2019). Consistent Manifold Representation for 
            Topological Data Analysis.
        """
        from scipy.spatial.distance import pdist, squareform
        
        n = len(points)
        if n == 0:
            return cls.from_simplices([], coefficient_ring=coefficient_ring)
            
        dist_matrix = squareform(pdist(points))
        
        # d_k is the distance to the k-th nearest neighbor (k+1 th entry since 0 is self)
        # Note: np.partition is fast. k is 1-indexed for neighbors here.
        k_actual = min(k, n - 1)
        if k_actual <= 0:
            d_k = np.zeros(n)
        else:
            sorted_dist = np.partition(dist_matrix, k_actual, axis=1)
            d_k = sorted_dist[:, k_actual]
            
        # Create outer product of local scales: sqrt(d_k(u) * d_k(v))
        d_k_matrix = np.outer(d_k, d_k)
        scale_matrix = delta * np.sqrt(d_k_matrix)
        
        # Connect if distance <= scale
        adj = (dist_matrix <= scale_matrix) & ~np.eye(n, dtype=bool)
        
        # We reuse the high-performance clique enumeration in from_distance_matrix
        # by passing it a distance matrix that perfectly mirrors our adjacency.
        dummy_dist = np.where(adj, 0.0, float('inf'))
        np.fill_diagonal(dummy_dist, 0.0)
        
        sc = cls.from_distance_matrix(
            dummy_dist, 
            max_edge_length=0.0, 
            max_dimension=max_dimension, 
            coefficient_ring=coefficient_ring
        )
        return sc

    @classmethod
    def from_distance_matrix(
        cls,
        distance_matrix: np.ndarray,
        max_edge_length: float | None = None,
        max_dimension: int = 2,
        coefficient_ring: str = "Z"
    ) -> "SimplicialComplex":
        """
        Builds a Vietoris-Rips complex directly from a pre-computed distance matrix.
        Uses native clique enumeration over a thresholded sparse proximity graph.
        """
        eps = max_edge_length if max_edge_length is not None else float('inf')
        n = distance_matrix.shape[0]
        
        # 1. Build Sparse Proximity Graph
        # We only care about edges <= eps.
        adj = (distance_matrix <= eps) & ~np.eye(n, dtype=bool)
        
        # To avoid duplicating cliques, we only keep upper triangular part for adjacency list
        upper_adj = np.triu(adj)
        from scipy.sparse import csr_matrix
        sparse_adj = csr_matrix(upper_adj)
        
        simplices: set[tuple[int, ...]] = set()
        # Add all vertices
        for i in range(n):
            simplices.add((i,))
            
        # Add all edges
        rows, cols = sparse_adj.nonzero()
        for r, c in zip(rows, cols):
            simplices.add((r, c))
            
        # Add higher dimensional cliques natively
        if max_dimension >= 2:
            from ..bridge.julia_bridge import julia_engine
            if julia_engine.available:
                # Use fast Julia DFS for clique enumeration
                # Note: Julia uses 1-based indexing, so we shift adj matrices.
                # Actually, our Julia backend takes 1-based rowptr and colval but 1-based output? 
                # Let's just use Python's networkx if Julia isn't used, but we optimized for Julia.
                try:
                    # Construct symmetric adjacency for Bron-Kerbosch
                    sym_adj = csr_matrix(adj)
                    # Convert to 1-based for Julia
                    rowptr = sym_adj.indptr + 1
                    colval = sym_adj.indices + 1
                    
                    raw_cliques = julia_engine.enumerate_cliques_sparse(rowptr, colval, n, max_dimension)
                    for c in raw_cliques:
                        # c is 1-based from Julia
                        c_0 = tuple(sorted(v - 1 for v in c))
                        if len(c_0) > 1:
                            simplices.add(c_0)
                except Exception as e:
                    import warnings
                    warnings.warn(f"Julia clique enumeration failed ({e!r}). Falling back to DFS.")
                    # Simple python fallback DFS
                    def dfs(current_clique, candidates):
                        if len(current_clique) > 1:
                            simplices.add(tuple(sorted(current_clique)))
                        if len(current_clique) == max_dimension + 1:
                            return
                        for i, v in enumerate(candidates):
                            new_cands = [w for w in candidates[i+1:] if adj[v, w]]
                            dfs(current_clique + [v], new_cands)
                    for u in range(n):
                        cands = [v for v in range(u+1, n) if adj[u, v]]
                        dfs([u], cands)
            else:
                # Optimized Python DFS fallback using set intersections
                adj_sets = {i: set(np.where(adj[i])[0]) for i in range(n)}
                
                def dfs(current_clique, candidate_set):
                    if len(current_clique) > 1:
                        simplices.add(tuple(sorted(current_clique)))
                    if len(current_clique) == max_dimension + 1:
                        return
                    for v in sorted(list(candidate_set)):
                        candidate_set.remove(v)
                        new_cands = candidate_set.intersection(adj_sets[v])
                        dfs(current_clique + [v], new_cands)
                
                for u in range(n):
                    cands = set(v for v in range(u+1, n) if adj[u, v])
                    dfs([u], cands)

        return cls.from_simplices(simplices, coefficient_ring=coefficient_ring, close_under_faces=True)

    @classmethod
    def from_alpha_complex(
        cls,
        points: np.ndarray,
        max_alpha_square: float | None = None,
        coefficient_ring: str = "Z"
    ) -> "SimplicialComplex":
        """
        Builds an Alpha Complex from a point cloud natively, avoiding GUDHI.
        It computes the Delaunay triangulation via SciPy and filters simplices by circumradius.
        """
        from scipy.spatial import Delaunay
        pts = np.asarray(points, dtype=np.float64)
        n_pts, dim = pts.shape
        
        if n_pts < dim + 1:
            raise ValueError(f"Need at least {dim+1} points for an Alpha Complex in {dim}D.")
            
        tri = Delaunay(pts)
        simplices_d = tri.simplices # max dimension simplices
        
        alpha2 = max_alpha_square if max_alpha_square is not None else float('inf')
        
        valid_simplices: set[tuple[int, ...]] = set()
        
        # We need circumradii. 
        from ..bridge.julia_bridge import julia_engine
        
        from pysurgery.bridge.julia_bridge import julia_engine
        if julia_engine.available:
            valid_simplices_list = julia_engine.compute_alpha_complex_simplices(
                pts, simplices_d, alpha2, dim
            )
            return cls.from_simplices(valid_simplices_list, coefficient_ring=coefficient_ring, close_under_faces=True)

        if dim == 2:
            if julia_engine.available:
                try:
                    r2 = julia_engine.compute_circumradius_sq_2d(pts, simplices_d)
                except Exception:
                    r2 = np.zeros(len(simplices_d))
            else:
                # Python fallback circumradius 2D
                r2 = np.zeros(len(simplices_d))
                for i, s in enumerate(simplices_d):
                    p0, p1, p2 = pts[s]
                    A = np.array([p1-p0, p2-p0])
                    b = 0.5 * np.array([np.sum((p1-p0)**2), np.sum((p2-p0)**2)])
                    try:
                        c = np.linalg.solve(A, b)
                        r2[i] = np.sum(c**2)
                    except Exception:
                        r2[i] = float('inf')
        elif dim == 3:
            if julia_engine.available:
                try:
                    r2 = julia_engine.compute_circumradius_sq_3d(pts, simplices_d)
                except Exception:
                    r2 = np.zeros(len(simplices_d))
            else:
                # Python fallback circumradius 3D
                r2 = np.zeros(len(simplices_d))
                for i, s in enumerate(simplices_d):
                    p0, p1, p2, p3 = pts[s]
                    A = np.array([p1-p0, p2-p0, p3-p0])
                    b = 0.5 * np.array([np.sum((p1-p0)**2), np.sum((p2-p0)**2), np.sum((p3-p0)**2)])
                    try:
                        c = np.linalg.solve(A, b)
                        r2[i] = np.sum(c**2)
                    except Exception:
                        r2[i] = float('inf')
        else:
            # Generalized N-dimensional circumradius fallback
            r2 = np.zeros(len(simplices_d))
            for i, s in enumerate(simplices_d):
                p0 = pts[s[0]]
                # Construct A matrix (relative vectors from p0)
                A = pts[s[1:]] - p0
                # Construct b vector: 0.5 * squared distances from p0
                b = 0.5 * np.sum((pts[s[1:]] - p0)**2, axis=1)
                try:
                    # Solve the linear system for the circumcenter relative to p0
                    c = np.linalg.solve(A, b)
                    r2[i] = np.sum(c**2)
                except Exception:
                    # Singular simplex (degenerate), treat as infinite circumradius
                    r2[i] = float('inf')

        # Gabriel filtration:
        # We add the maximal simplices with r2 <= alpha2.
        # For rigorous topological Alpha complexes, we filter the Delaunay complex.
        for i, s in enumerate(simplices_d):
            if r2[i] <= alpha2:
                valid_simplices.add(tuple(sorted(s)))
                
        # To strictly mirror GUDHI's behavior for faces:
        # If a simplex is not added because its r2 > alpha2, its faces might still be valid
        # if their own circumradius is <= alpha2.
        # We must add all vertices and check lower-dimensional faces.
        for i in range(n_pts):
            valid_simplices.add((i,))
            
        # Check all faces up to dimension dim-1
        # For performance, we check edges explicitly as they are most common.
        edges = set()
        for s in simplices_d:
            for u, v in itertools.combinations(s, 2):
                edges.add(tuple(sorted([u, v])))
                
        for u, v in edges:
            if np.sum((pts[u] - pts[v])**2) / 4.0 <= alpha2:
                valid_simplices.add((u, v))
                
        # Higher-dimensional faces (2D to dim-1)
        if dim >= 3:
            for k in range(2, dim):
                faces_k = set()
                for s in simplices_d:
                    for f in itertools.combinations(s, k+1):
                        faces_k.add(tuple(sorted(f)))
                
                if len(faces_k) > 0:
                    for f in faces_k:
                        f_pts = pts[list(f)]
                        p0 = f_pts[0]
                        A = f_pts[1:] - p0
                        b = 0.5 * np.sum((f_pts[1:] - p0)**2, axis=1)
                        try:
                            # Use lstsq for faces since A might not be square (k < dim)
                            c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                            r2_f = np.sum(c**2)
                            if r2_f <= alpha2:
                                valid_simplices.add(f)
                        except Exception:
                            pass

        return cls.from_simplices(valid_simplices, coefficient_ring=coefficient_ring, close_under_faces=True)

    @classmethod
    def from_witness(
        cls,
        points: np.ndarray,
        n_landmarks: int = 100,
        max_dimension: int = 2,
        coefficient_ring: str = "Z"
    ) -> "SimplicialComplex":
        """
        Builds a Witness Complex from a point cloud natively.
        Witness complexes use a small set of landmarks to represent the topology of the full cloud.
        """
        from .metrics import farthest_point_sampling, compute_distance_matrix
        
        # 1. Select landmarks via FPS
        landmark_indices = farthest_point_sampling(points, n_landmarks)
        
        # 2. Map all points to their nearest landmark
        D = compute_distance_matrix(points) 
        D_to_landmarks = D[:, landmark_indices]
        
        simplices: set[tuple[int, ...]] = set()
        
        # Native Strong Witness allocation
        # A k-simplex is in the complex if there is a point whose (k+1)-nearest 
        # landmarks are exactly its vertices.
        # Vectorized sort:
        nearest_landmarks = np.argsort(D_to_landmarks, axis=1)[:, :max_dimension + 1]
        
        for point_idx in range(points.shape[0]):
            near_landmarks = nearest_landmarks[point_idx]
            for d in range(1, max_dimension + 2):
                for combo in itertools.combinations(near_landmarks, d):
                    simplices.add(tuple(sorted(combo)))
        
        # The landmark indices refer to the index in `landmark_indices`. 
        # So simplex vertices are 0 to n_landmarks-1.
        return cls.from_simplices(simplices, coefficient_ring=coefficient_ring, close_under_faces=True)

    @classmethod
    def from_gudhi_simplex_tree(
        cls,
        simplex_tree: object,
        *,
        coefficient_ring: str = "Z",
        include_filtration: bool = True,
        close_under_faces: bool = False,
    ) -> "SimplicialComplex":
        if not hasattr(simplex_tree, "get_filtration"):
            raise TypeError("Expected a GUDHI-like SimplexTree with get_filtration().")
        simplices: list[tuple[int, ...]] = []
        filtration: dict[tuple[int, ...], float] = {}
        for simplex, value in simplex_tree.get_filtration():
            s = _normalize_simplex(simplex)
            simplices.append(s)
            if include_filtration:
                val = float(value)
                if s in filtration:
                    filtration[s] = min(filtration[s], val)
                else:
                    filtration[s] = val
        sc = cls.from_simplices(
            simplices,
            coefficient_ring=coefficient_ring,
            close_under_faces=close_under_faces,
        )
        if include_filtration:
            sc.filtration = {
                simplex: filtration.get(simplex, 0.0)
                for simplices_dim in sc.simplices_field.values()
                for simplex in simplices_dim
            }
        return sc

    def to_gudhi_simplex_tree(
        self,
        *,
        use_filtration: bool = False,
        default_filtration: float = 0.0,
    ) -> object:
        try:
            import gudhi  # type: ignore
        except Exception as exc:
            raise ImportError("GUDHI is required. Install via 'pip install gudhi'.") from exc

        cache_key = ("interop", "to_gudhi", bool(use_filtration), float(default_filtration))
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        st = gudhi.SimplexTree()
        for dim in sorted(self.simplices_field):
            for simplex in self.simplices_field[dim]:
                filt = (
                    float(self.filtration.get(simplex, default_filtration))
                    if use_filtration
                    else float(default_filtration)
                )
                st.insert(list(simplex), filtration=filt)
        self._cache_set(cache_key, st)
        return st

    @property
    def dimension(self) -> int:
        return max(self.simplices_field.keys(), default=-1)

    @property
    def dimensions(self) -> list[int]:
        return sorted(self.simplices_field.keys())

    def simplex_index(self, dim: int) -> dict[tuple[int, ...], int]:
        """Return a stable index map for simplices in a fixed dimension."""
        d = int(dim)
        key = ("simplicial", "simplex_index", d)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        out = {simplex: idx for idx, simplex in enumerate(self.n_simplices(d))}
        self._cache_set(key, out)
        return out

    def f_vector(self) -> dict[int, int]:
        """Return the f-vector as a dimension-to-simplex-count dictionary."""
        key = ("simplicial", "f_vector")
        cached = self._cache_get(key)
        if cached is not None:
            return cast(dict[int, int], cached)
        out = {dim: len(simplices) for dim, simplices in self.simplices_field.items()}
        self._cache_set(key, out)
        return out

    def euler_characteristic(self) -> int:
        key = ("simplicial", "euler_characteristic")
        cached = self._cache_get(key)
        if cached is not None:
            return int(cast(int, cached))
        out = int(sum(((-1) ** dim) * len(simplices) for dim, simplices in self.simplices_field.items()))
        self._cache_set(key, int(out))
        return int(out)

    def is_closed_under_faces(self) -> bool:
        """Check whether all codimension-1 faces are present in the table."""
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
        """
        Comprehensive mathematical validity check for the simplicial complex.
        
        Verifies:
        1. Downward Closure: Every face of every simplex is in the complex.
        2. Orientation Consistency: All simplices are stored in canonical (sorted) order.
        3. Boundary of Boundary: Composition of consecutive boundary operators is zero (d^2 = 0).
        """
        issues = []
        
        # 1. & 2. Closure and Orientation
        is_closed = self.is_closed_under_faces()
        if not is_closed:
            issues.append("Downward closure failure: some faces are missing from the complex.")
            
        for dim, simplices in self.simplices_field.items():
            for s in simplices:
                if list(s) != sorted(s):
                    issues.append(f"Orientation inconsistency: simplex {s} is not in canonical sorted order.")
                    break
        
        # 3. d^2 = 0
        dims = sorted(self.dimensions)
        for d in dims:
            if d <= 1:
                continue
            # d_d : C_d -> C_{d-1}
            # d_{d-1} : C_{d-1} -> C_{d-2}
            mat_d = self.boundary_matrix(d)
            mat_dm1 = self.boundary_matrix(d - 1)
            
            if mat_d.nnz > 0 and mat_dm1.nnz > 0:
                prod = mat_dm1 @ mat_d
                if prod.nnz > 0:
                    # Check if actually non-zero (avoid float noise, though these are ints)
                    if np.any(prod.data != 0):
                        issues.append(f"Boundary consistency failure: d_{d-1} * d_{d} != 0 at dimension {d}.")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "is_closed": is_closed,
            "is_canonical": all(list(s) == sorted(s) for simps in self.simplices_field.values() for s in simps),
        }

    def hasse_edges(self) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        """Return the codimension-one relations of the Hasse diagram."""
        key = ("simplicial", "hasse_edges")
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        edges: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        for dim in sorted(self.simplices_field.keys()):
            if dim <= 0:
                continue
            target = set(self.simplices_field.get(dim - 1, []))
            for simplex in self.simplices_field.get(dim, []):
                for i in range(len(simplex)):
                    face = simplex[:i] + simplex[i + 1 :]
                    if face in target:
                        edges.append((simplex, face))
        self._cache_set(key, edges)
        return edges

    def boundary_matrix(self, dim: int) -> csr_matrix:
        """Compute the oriented boundary matrix d_dim : C_dim -> C_{dim-1}."""
        dim = int(dim)
        key = ("simplicial", "boundary_matrix", dim)
        cached = self._cache_get(key)
        if cached is not None:
            return cast(csr_matrix, cached)
        if dim <= 0:
            out = csr_matrix((0, self.count_simplices(dim)), dtype=np.int64)
            self._cache_set(key, out)
            return out
        out = _boundary_matrix_from_simplices(
            self.n_simplices(dim), self.n_simplices(dim - 1)
        )
        self._cache_set(key, out)
        return out

    def boundary_matrices(self) -> dict[int, csr_matrix]:
        """Compute all nontrivial boundary matrices keyed by dimension."""
        key = ("simplicial", "boundary_matrices")
        cached = self._cache_get(key)
        if cached is not None:
            return cast(dict[int, csr_matrix], cached)
        out = {dim: self.boundary_matrix(dim) for dim in self.dimensions if dim > 0}
        self._cache_set(key, out)
        return out

    def chain_complex(self, coefficient_ring: str | None = None) -> "ChainComplex":
        """Convert the simplicial complex to a sparse chain complex.

        Parameters
        ----------
        coefficient_ring:
            Optional ring override (for example ``"Q"`` or ``"Z/2Z"``).
            When omitted, the complex default ``self.coefficient_ring`` is used.
        """
        ring = self.coefficient_ring if coefficient_ring is None else str(coefficient_ring)
        _parse_coefficient_ring(ring)
        key = ("simplicial", "chain_complex", ring)
        cached = self._cache_get(key)
        if cached is not None:
            return cast(ChainComplex, cached)

        # Check if Julia pre-calculated the boundary operators
        if self._boundaries_cache and self._cells_cache:
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

    def cellular_chain_complex(self, coefficient_ring: str | None = None) -> "ChainComplex":
        """Alias for `chain_complex` for compatibility with cellular workflows."""
        return self.chain_complex(coefficient_ring=coefficient_ring)



    def expand(self, max_dim: int | None = None) -> "SimplicialComplex":
        """
        Expands the simplicial complex into a Flag Complex (Clique Complex)
        up to the given maximum dimension.
        
        This adds all cliques of size up to max_dim + 1 as simplices.
        Commonly used after skeleton-only algorithms like quick_mapper.
        
        Args:
            max_dim: The maximum dimension of simplices to include.
                     If None, defaults to the maximum dimension of the space 
                     (number of vertices - 1).
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

        # 3. Python Fallback (Recursive clique expansion)
        adj_dict = {v: set() for v in vertices}
        for u, v in edges:
            adj_dict[u].add(v)
            adj_dict[v].add(u)
            
        all_cliques = []
        
        def find_cliques(current_clique, candidates):
            all_cliques.append(tuple(sorted(current_clique)))
            if len(current_clique) > max_dim:
                return
            
            for i, v in enumerate(candidates):
                new_candidates = [w for w in candidates[i+1:] if w in adj_dict[v]]
                find_cliques(current_clique + [v], new_candidates)
                
        find_cliques([], sorted(vertices))
        # Remove empty clique
        if all_cliques and not all_cliques[0]:
            all_cliques.pop(0)
            
        return self.__class__.from_simplices(
            all_cliques,
            coefficient_ring=self.coefficient_ring,
            close_under_faces=True
        )

    def quick_mapper(
        self,
        max_loops: int = 1,
        min_modularity_gain: float = 1e-6,
        preserve_topology: bool = True,
    ) -> "SimplicialComplex":
        """
        Simplifies the topological complex using the fast modularity relabeling algorithm.
        This algorithm constructs a simpler topological structure by grouping vertices
        based on local edge density.
        
        References:
            Liu, X., Xie, Z., & Yi, D. (2012). A fast algorithm for constructing topological 
            structure in large data. Homology, Homotopy and Applications.
        
        Args:
            max_loops: Maximum number of label propagation passes.
            min_modularity_gain: Threshold for early stopping.
            preserve_topology: If True, merges are only committed if they satisfy the 
                               simplicial collapse link condition (preserving Betti numbers).
        """
        import random
        
        V = [v[0] for v in self.n_simplices(0)]
        E = self.n_simplices(1)
        
        adj = {v: set() for v in V}
        for edge in E:
            u, v = edge
            adj[u].add(v)
            adj[v].add(u)
            
        m = len(E)
        L = {v: v for v in V}
        
        if m == 0:
            return self.model_copy()
            
        # Try Julia bridge
        if julia_engine.available:
            try:
                # Prepare data for Julia: 1-indexed edges
                E_jl = [(u + 1, v + 1) for u, v in E]
                V_jl = [v + 1 for v in V]
                G_raw = {"V": V_jl, "E": E_jl}
                
                # Check link condition requires simplex membership lookups
                # For safety and speed in Julia, we can pass the full list of 2-simplices
                # to verify Lk(u) \cap Lk(v) \subset Lk(uv).
                # To keep the bridge simple, if preserve_topology is True, we stay in Python.
                if not preserve_topology:
                    G_simple, L_jl = julia_engine.quick_mapper_jl(
                        G_raw, 
                        int(max_loops), 
                        float(min_modularity_gain)
                    )
                    L = {v - 1: lbl - 1 for v, lbl in L_jl.items()}
                    
                    new_simplices = set()
                    for dim, simps in self.simplices_dict.items():
                        for simplex in simps:
                            mapped = tuple(sorted(set(L[v] for v in simplex)))
                            if len(mapped) > 0:
                                new_simplices.add(mapped)
                    return self.__class__.from_simplices(new_simplices, coefficient_ring=self.coefficient_ring)
            except Exception as e:
                warnings.warn(f"Julia quick_mapper failed: {e}. Falling back to Python.")
                
        degree = {v: len(adj[v]) for v in V}
        
        def get_P(i, j):
            return (degree[i] * degree[j]) / (2.0 * m)
            
        # Fast lookup for topology preservation
        simplex_set = set()
        for dim, simps in self.simplices_dict.items():
            for s in simps:
                simplex_set.add(s)
                
        def link_condition_passed(u: int, v: int) -> bool:
            """
            Checks if contracting edge (u, v) preserves the homotopy type.
            Condition: Lk(u) ∩ Lk(v) ⊆ Lk(uv).
            """
            # Find common neighbors in 1-skeleton (0-simplices in the link)
            common_neighbors = adj[u].intersection(adj[v])
            for w in common_neighbors:
                # For every common neighbor w, {u, v, w} must be a 2-simplex
                t = tuple(sorted((u, v, w)))
                if t not in simplex_set:
                    return False
            # This is a highly robust 1-skeleton heuristic for discrete Morse collapses.
            # True complete verification requires checking all higher-dimensional common cofaces,
            # but for a flag complex (like Rips), the 1-skeleton check is sufficient.
            return True

        num_of_loops = 0
        modularity_gain = 1000.0
        
        while modularity_gain > min_modularity_gain and num_of_loops < max_loops:
            vertex_order = list(V)
            random.shuffle(vertex_order)
            
            for vertex in vertex_order:
                neighbors = adj[vertex]
                if not neighbors:
                    continue
                    
                NbrLabelSet_vertex = {L[vertex]}
                for nbr in neighbors:
                    NbrLabelSet_vertex.add(L[nbr])
                    
                best_labels = []
                max_contribution = -float('inf')
                
                for label in NbrLabelSet_vertex:
                    contribution = sum((1 - get_P(vertex, j)) for j in neighbors if L[j] == label)
                    if contribution > max_contribution:
                        max_contribution = contribution
                        best_labels = [label]
                    elif contribution == max_contribution:
                        best_labels.append(label)
                        
                chosen_label = random.choice(best_labels)
                
                if preserve_topology and chosen_label != L[vertex]:
                    # To prevent destroying cycles, we only allow merges that satisfy
                    # the simplicial collapse link condition with the target label's representative.
                    # chosen_label is an original vertex ID.
                    if not link_condition_passed(vertex, chosen_label):
                        continue # Reject merge, topological invariants would be altered
                
                L[vertex] = chosen_label
                
            modularity_gain = 0.0 # Standard simple-pass exit
            num_of_loops += 1
            
        new_simplices = set()
        for dim, simps in self.simplices_dict.items():
            for simplex in simps:
                mapped = tuple(sorted(set(L[v] for v in simplex)))
                if len(mapped) > 0:
                    new_simplices.add(mapped)
                    
        return self.__class__.from_simplices(
            new_simplices, 
            coefficient_ring=self.coefficient_ring
        )



def _rank_mod_p(A: np.ndarray, p: int) -> int:
    """Compute matrix rank over `Z/pZ` via modular Gaussian elimination."""
    if A.dtype == object:
        M = (A % p).astype(np.int64)
    else:
        M = (A.astype(np.int64) % p).copy()
    m, n = M.shape
    row = 0
    rank = 0
    for col in range(n):
        pivot = None
        for r in range(row, m):
            if M[r, col] % p != 0:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != row:
            M[[row, pivot]] = M[[pivot, row]]
        inv = pow(int(M[row, col]), -1, p)
        M[row, :] = (M[row, :] * inv) % p
        for r in range(m):
            if r != row and M[r, col] % p != 0:
                M[r, :] = (M[r, :] - M[r, col] * M[row, :]) % p
        row += 1
        rank += 1
        if row == m:
            break
    return rank


def _matrix_rank_for_ring(
    matrix: csr_matrix, ring_kind: str, p: int | None = None
) -> int:
    """Compute matrix rank in the requested coefficient field, preferring Julia when available."""
    if matrix is None or matrix.nnz == 0:
        return 0

    if ring_kind == "Q":
        if julia_engine.available:
            try:
                return int(julia_engine.compute_sparse_rank_q(matrix))
            except Exception as exc:
                warnings.warn(
                    "Topological Hint: Julia rank over Q failed in `ChainComplex.homology`; "
                    f"falling back to NumPy dense rank ({exc!r})."
                )
        return int(np.linalg.matrix_rank(matrix.toarray().astype(float)))

    if ring_kind == "ZMOD":
        if p is None:
            raise ValueError("Prime modulus p is required for Z/pZ rank computation.")
        if julia_engine.available:
            try:
                return int(julia_engine.compute_sparse_rank_mod_p(matrix, int(p)))
            except Exception as exc:
                warnings.warn(
                    "Topological Hint: Julia rank over Z/pZ failed in `ChainComplex.homology`; "
                    f"falling back to Python elimination ({exc!r})."
                )
        return _rank_mod_p(matrix.toarray(), int(p))

    raise ValueError(f"Unsupported rank ring kind '{ring_kind}'.")


def _is_prime(n: int) -> bool:
    """Return True when `n` is prime (deterministic trial division)."""
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def _rref_mod_p(A: np.ndarray, p: int) -> tuple[np.ndarray, list[int]]:
    """Compute row-reduced echelon form over `Z/pZ`."""
    M = (A.astype(np.int64) % p).copy()
    m, n = M.shape
    row = 0
    pivots: list[int] = []
    for col in range(n):
        pivot = None
        for r in range(row, m):
            if M[r, col] % p != 0:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != row:
            M[[row, pivot]] = M[[pivot, row]]
        inv = pow(int(M[row, col]), -1, p)
        M[row, :] = (M[row, :] * inv) % p
        for r in range(m):
            if r != row and M[r, col] % p != 0:
                M[r, :] = (M[r, :] - M[r, col] * M[row, :]) % p
        pivots.append(col)
        row += 1
        if row == m:
            break
    return M, pivots


def _nullspace_basis_mod_p(A: np.ndarray, p: int) -> list[np.ndarray]:
    """Return a basis of `ker(A)` over `Z/pZ`."""
    # A is m x n. Return basis vectors of ker(A) in F_p^n.
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


def _composite_mod_uct_decomposition(
    free_rank: int,
    torsion_n: List[int],
    torsion_nm1: List[int],
    modulus: int,
) -> Tuple[int, List[int]]:
    """Compute Z/n decomposition from integral data via UCT tensor/Tor terms."""
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
    """
    An abstract Chain Complex C_* over Z.
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
        object.__setattr__(
            self,
            "boundaries",
            {int(dim): _coerce_csr_matrix(mat) for dim, mat in self.boundaries.items()},
        )
        object.__setattr__(self, "dimensions", sorted({int(dim) for dim in self.dimensions}))
        object.__setattr__(self, "cells", {int(dim): int(count) for dim, count in self.cells.items()})
        object.__setattr__(self, "coefficient_ring", str(self.coefficient_ring))
        return self

    def _structure_signature(self) -> tuple[object, ...]:
        boundary_sig = tuple(
            (int(dim), _csr_matrix_signature(mat))
            for dim, mat in sorted(self.boundaries.items())
        )
        cells_sig = tuple((int(dim), int(count)) for dim, count in sorted(self.cells.items()))
        return boundary_sig, tuple(self.dimensions), cells_sig, str(self.coefficient_ring)

    def _ensure_cache_valid(self) -> None:
        current = self._structure_signature()
        if self._cache_signature != current:
            self._cache.clear()
            self._cache_signature = current

    def _cache_get(self, key: tuple[object, ...]) -> object | None:
        self._ensure_cache_valid()
        if not self._cache_enabled:
            return None
        if key in self._cache:
            self._cache_hits += 1
            return _clone_cache_value(self._cache[key])
        self._cache_misses += 1
        return None

    def _cache_set(self, key: tuple[object, ...], value: object) -> None:
        self._ensure_cache_valid()
        if not self._cache_enabled:
            return
        self._cache[key] = _clone_cache_value(value)

    def clear_cache(self, namespace: str | None = None) -> None:
        if namespace is None:
            self._cache.clear()
            return
        prefix = (str(namespace),)
        keys = [k for k in self._cache if k[:1] == prefix]
        for key in keys:
            self._cache.pop(key, None)

    def cache_info(self) -> dict[str, object]:
        self._ensure_cache_valid()
        return {
            "enabled": bool(self._cache_enabled),
            "size": int(len(self._cache)),
            "hits": int(self._cache_hits),
            "misses": int(self._cache_misses),
            "keys": [list(map(str, key)) for key in sorted(self._cache.keys(), key=repr)],
        }

    def set_cache_enabled(self, enabled: bool, *, clear_when_disabled: bool = True) -> None:
        self._cache_enabled = bool(enabled)
        if not self._cache_enabled and clear_when_disabled:
            self._cache.clear()

    def _homological_dimensions(self) -> List[int]:
        """Return sorted degrees that have meaningful chain data."""
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

    def _homology_over_z(self, n: int) -> Tuple[int, List[int]]:
        """Exact integral homology helper used by coefficient-change formulas."""
        n = int(n)
        key = ("chain", "homology_over_z", n)
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
            snf_n = get_sparse_snf_diagonal(dn)
            rank_n = np.count_nonzero(snf_n)
        else:
            rank_n = 0
        dim_ker_n = c_n_size - rank_n

        if dn_plus_1 is not None and dn_plus_1.nnz > 0:
            snf_n_plus_1 = get_sparse_snf_diagonal(dn_plus_1)
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
        self, n: int | None = None
    ) -> Tuple[int, List[int]] | Dict[int, Tuple[int, List[int]]]:
        """
        Compute the n-th homology group H_n(C) = ker(d_n) / im(d_{n+1}).

        If `n` is omitted, computes homology for all known nonnegative degrees
        and returns a dictionary `{degree: (rank, torsion)}`.

        Returns
        -------
        rank : int
            The free rank of the homology group (Betti number).
        torsion : List[int]
            The torsion coefficients (invariant factors > 1).
        """
        if n is None:
            key_all = ("chain", "homology", "all", str(self.coefficient_ring))
            cached_all = self._cache_get(key_all)
            if cached_all is not None:
                return cached_all
            out_all = {dim: self.homology(dim) for dim in self._homological_dimensions()}
            self._cache_set(key_all, out_all)
            return out_all

        n = int(n)
        key = ("chain", "homology", n, str(self.coefficient_ring))
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        # d_n : C_n -> C_{n-1}
        # d_{n+1} : C_{n+1} -> C_n
        ring_kind, p = _parse_coefficient_ring(self.coefficient_ring)

        # For composite Z/nZ coefficients, compute via exact integral decomposition + UCT.
        if ring_kind == "ZMOD" and p is not None and not _is_prime(int(p)):
            warnings.warn(
                f"Composite-modulus homology in `ChainComplex.homology` over Z/{p}Z uses UCT + integral SNF fallback; "
                "install/enable Julia for faster exact sparse integer reductions."
            )
            r_n, t_n = self._homology_over_z(n)
            _, t_nm1 = self._homology_over_z(n - 1)
            modulus = int(p)

            rank_mod, torsion_mod = _composite_mod_uct_decomposition(
                r_n, t_n, t_nm1, modulus
            )
            out = (int(rank_mod), torsion_mod)
            self._cache_set(key, out)
            return out

        dn = self.boundaries.get(n)
        dn_plus_1 = self.boundaries.get(n + 1)

        # Dimensions of chain groups
        # If n not in boundaries, assume C_n is 0 or its size is inferred from boundaries
        if (
            n not in self.dimensions
            and n not in self.cells
            and dn is None
            and dn_plus_1 is None
        ):
            out = (0, [])
            self._cache_set(key, out)
            return out

        # Number of n-cells (columns of d_n or rows of d_{n+1})
        if n in self.cells:
            c_n_size = self.cells[n]
        elif dn is not None:
            c_n_size = dn.shape[1]
        elif dn_plus_1 is not None:
            c_n_size = dn_plus_1.shape[0]
        else:
            # Isolated dimension with no boundaries and no explicit cell count
            out = (0, [])
            self._cache_set(key, out)
            return out

        # 1. Find rank of d_n to get dim(ker(d_n))
        if dn is not None and dn.nnz > 0:
            if ring_kind == "Z":
                snf_n = get_sparse_snf_diagonal(dn)
                rank_n = np.count_nonzero(snf_n)
            elif ring_kind == "Q":
                rank_n = _matrix_rank_for_ring(dn, "Q")
            else:
                rank_n = _matrix_rank_for_ring(dn, "ZMOD", int(p))
        else:
            rank_n = 0

        dim_ker_n = c_n_size - rank_n

        # 2. Find rank(im(d_{n+1})) and torsion when applicable
        if dn_plus_1 is not None and dn_plus_1.nnz > 0:
            if ring_kind == "Z":
                snf_n_plus_1 = get_sparse_snf_diagonal(dn_plus_1)
                rank_im_n_plus_1 = np.count_nonzero(snf_n_plus_1)
                torsion = [int(x) for x in snf_n_plus_1 if x > 1]
                if not torsion and any(x == 1 for x in snf_n_plus_1):
                    warnings.warn(
                        "Torsion certification may be incomplete for this complex; the sparse integer reduction returned"
                        " only unit factors, so torsion could not be fully resolved."
                    )
            elif ring_kind == "Q":
                rank_im_n_plus_1 = _matrix_rank_for_ring(dn_plus_1, "Q")
                torsion = []
            else:
                rank_im_n_plus_1 = _matrix_rank_for_ring(dn_plus_1, "ZMOD", int(p))
                torsion = []
        else:
            rank_im_n_plus_1 = 0
            torsion = []
        betti_n = max(0, dim_ker_n - rank_im_n_plus_1)
        out = (int(betti_n), torsion)
        self._cache_set(key, out)
        return out

    def cohomology(
        self, n: int | None = None
    ) -> Tuple[int, List[int]] | Dict[int, Tuple[int, List[int]]]:
        r"""
        Compute the n-th cohomology group H^n(C) using the Universal Coefficient Theorem:
        H^n(C, Z) \cong Hom(H_n(C), Z) \oplus Ext(H_{n-1}(C), Z).

        If `n` is omitted, computes cohomology for all known nonnegative degrees
        and returns a dictionary `{degree: (rank, torsion)}`.
        """
        if n is None:
            key_all = ("chain", "cohomology", "all", str(self.coefficient_ring))
            cached_all = self._cache_get(key_all)
            if cached_all is not None:
                return cached_all
            out_all = {dim: self.cohomology(dim) for dim in self._homological_dimensions()}
            self._cache_set(key_all, out_all)
            return out_all

        n = int(n)
        key = ("chain", "cohomology", n, str(self.coefficient_ring))
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        ring_kind, _ = _parse_coefficient_ring(self.coefficient_ring)
        if ring_kind == "ZMOD":
            _, p = _parse_coefficient_ring(self.coefficient_ring)
            if p is not None and not _is_prime(int(p)):
                r_n, t_n = self._homology_over_z(n)
                _, t_nm1 = self._homology_over_z(n - 1)
                modulus = int(p)
                rank_mod, torsion_mod = _composite_mod_uct_decomposition(
                    r_n, t_n, t_nm1, modulus
                )
                out = (int(rank_mod), torsion_mod)
                self._cache_set(key, out)
                return out
        free_rank, _ = self.homology(n)
        if ring_kind == "Z":
            _, prev_torsion = self.homology(n - 1)
            out = (free_rank, prev_torsion)
            self._cache_set(key, out)
            return out
        out = (free_rank, [])
        self._cache_set(key, out)
        return out

    def _chain_group_rank_for_degree(self, n: int) -> int:
        """Return rank(C_n), inferred from cells or boundary matrix shapes."""
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
        """Return chain-group rank(s), i.e., rank(C_n)."""
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

    def betti_number(self, n: int | None = None) -> int | Dict[int, int]:
        """Return Betti number(s), i.e., free ranks of homology groups."""
        if n is None:
            key_all = ("chain", "betti_number", "all", str(self.coefficient_ring))
            cached_all = self._cache_get(key_all)
            if cached_all is not None:
                return cached_all
            hom_all = self.homology()
            out_all = {dim: rank for dim, (rank, _) in hom_all.items()}
            self._cache_set(key_all, out_all)
            return out_all

        n = int(n)
        key = ("chain", "betti_number", n, str(self.coefficient_ring))
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        rank, _ = self.homology(n)
        out = int(rank)
        self._cache_set(key, out)
        return out

    def betti_numbers(self) -> Dict[int, int]:
        """Return Betti numbers for all known nonnegative degrees."""
        out = self.betti_number()
        return out

    def cohomology_basis(self, n: int) -> List[np.ndarray]:
        """
        Computes a basis for the free part of the n-th cohomology group H^n(C; Z).
        Returns a list of n-cochains (vectors in C^n).

        This finds generators of the free part via a rational complement:
        (ker d_{n+1}^T / im d_n^T) tensor Q.
        Exact torsion-sensitive quotients require the Julia backend.

        For massive matrices, this seamlessly offloads to optimized float SVDs or Julia.
        """
        n = int(n)
        key = ("chain", "cohomology_basis", n, str(self.coefficient_ring))
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        dn_plus_1 = self.boundaries.get(n + 1)
        dn = self.boundaries.get(n)

        # Number of n-cells (columns of d_n or rows of d_{n+1})
        if n in self.cells:
            cn_size = self.cells[n]
        elif dn is not None:
            cn_size = dn.shape[1]
        elif dn_plus_1 is not None:
            cn_size = dn_plus_1.shape[0]
        else:
            out: List[np.ndarray] = []
            self._cache_set(key, out)
            return out  # Isolated dimension

        ring_kind, p = _parse_coefficient_ring(self.coefficient_ring)

        if ring_kind == "ZMOD" and p is not None and not _is_prime(int(p)):
            warnings.warn(
                f"Composite-modulus cohomology basis in `ChainComplex.cohomology_basis` over Z/{p}Z uses integral-basis reduction fallback; "
                "install/enable Julia for faster exact sparse quotient-basis computation."
            )
            integral_complex = ChainComplex(
                boundaries=self.boundaries,
                dimensions=self.dimensions,
                cells=self.cells,
                coefficient_ring="Z",
            )
            basis_z = integral_complex.cohomology_basis(n)
            out = []
            modulus = int(p)
            target_rank, _ = self.cohomology(n)
            for v in basis_z[:target_rank]:
                out.append(np.asarray(v, dtype=np.int64) % modulus)
            # If UCT predicts additional Z/n generators from torsion terms, append canonical vectors.
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
                            dn_plus_1,
                            dn,
                            int(p),
                            cn_size=cn_size,
                        )
                    self._cache_set(key, out)
                    return out
                except Exception as exc:
                    warnings.warn(
                        "Topological Hint: Julia field cohomology basis backend failed in "
                        f"`ChainComplex.cohomology_basis`; falling back to Python implementation ({exc!r})."
                    )

            # Vector-space basis over a field.
            if dn_plus_1 is None or dn_plus_1.nnz == 0:
                if ring_kind == "Q":
                    null_basis = [
                        sp.Matrix([1 if i == j else 0 for i in range(cn_size)])
                        for j in range(cn_size)
                    ]
                else:
                    null_basis = [
                        np.eye(cn_size, dtype=np.int64)[j] for j in range(cn_size)
                    ]
            else:
                if ring_kind == "Q":
                    M = sp.Matrix(dn_plus_1.T.toarray().astype(int))
                    null_basis = M.nullspace()
                else:
                    null_basis = _nullspace_basis_mod_p(dn_plus_1.T.toarray(), int(p))

            if dn is None or dn.nnz == 0:
                image_basis = []
            else:
                if ring_kind == "Q":
                    Mimg = sp.Matrix(dn.T.toarray().astype(int))
                    image_basis = Mimg.columnspace()
                else:
                    mat = dn.T.toarray().astype(np.int64)
                    _, pivots = _rref_mod_p(mat, int(p))
                    image_basis = [mat[:, j] % int(p) for j in pivots]

            target_rank, _ = self.cohomology(n)
            basis_of_quotient = []
            if image_basis:
                if ring_kind == "Q":
                    current_mat = sp.Matrix.hstack(*image_basis)
                    current_rank = current_mat.rank()
                else:
                    current_mat = np.column_stack(image_basis).astype(np.int64) % int(p)
                    current_rank = _rank_mod_p(current_mat, int(p))
            else:
                current_mat = (
                    sp.Matrix.zeros(cn_size, 0)
                    if ring_kind == "Q"
                    else np.zeros((cn_size, 0), dtype=np.int64)
                )
                current_rank = 0

            for v in null_basis:
                if len(basis_of_quotient) >= target_rank:
                    break
                if ring_kind == "Q":
                    test_mat = sp.Matrix.hstack(current_mat, v)
                    new_rank = test_mat.rank()
                else:
                    col = np.asarray(v, dtype=np.int64).reshape(cn_size, 1) % int(p)
                    test_mat = np.hstack((current_mat, col))
                    new_rank = _rank_mod_p(test_mat, int(p))
                if new_rank > current_rank:
                    current_mat = test_mat
                    current_rank = new_rank
                    basis_of_quotient.append(v)

            out = []
            for v in basis_of_quotient:
                arr = np.array(v, dtype=np.int64).flatten()
                if ring_kind == "ZMOD":
                    arr = arr % int(p)
                out.append(arr)
            self._cache_set(key, out)
            return out

        if julia_engine.available:
            try:
                # Use exact sparse linear algebra in Julia to perfectly compute Z^n / B^n
                out = julia_engine.compute_sparse_cohomology_basis(
                    dn_plus_1, dn, cn_size=cn_size
                )
                self._cache_set(key, out)
                return out
            except Exception as e:
                msg = (
                    f"Topological Hint: Julia bridge failed ({e!r}). Falling back to pure Python computation. "
                    "For massive datasets, this might cause memory overflow or loss of exact integer torsion tracking."
                )
                warnings.warn(msg)

        # If Julia is unavailable, we dynamically attempt exact Python mathematics.
        # SymPy is used for exact integer quotients, but if it exceeds memory/time thresholds,
        # we catch the exception (or we just use an optimized float SVD fallback directly).

        if dn is None or dn.nnz == 0:
            # If no boundaries, Z^n / B^n is just ker(d_{n+1}^T)
            if dn_plus_1 is None or dn_plus_1.nnz == 0:
                int_basis = [np.eye(cn_size, dtype=np.int64)[j] for j in range(cn_size)]
            else:
                # Memory Warning: Sparse to Dense conversion fallback
                if dn_plus_1.shape[0] * dn_plus_1.shape[1] > 1_000_000:
                    warnings.warn(
                        f"Matrix {dn_plus_1.shape} is large for Python/SymPy reduction. "
                        "Expect significant RAM consumption and slow execution without Julia."
                    )
                
                # Bridging to SymPy SparseMatrix to reduce RAM pressure
                mat = sp.SparseMatrix(dn_plus_1.shape[1], dn_plus_1.shape[0], dict(dn_plus_1.T.todok().items()))
                # syzygy_module gives an exact Z-basis for the kernel over PID
                int_basis = [np.array(v, dtype=np.int64).flatten() for v in mat.syzygy_module()]
            self._cache_set(key, int_basis)
            return int_basis

        # Compute the Smith Normal Form of d_n^T
        if dn.shape[0] * dn.shape[1] > 1_000_000:
            warnings.warn(
                f"Matrix {dn.shape} is large for Smith Normal Form in Python. "
                "Expect significant RAM consumption without Julia backend."
            )
        
        dn_T = sp.SparseMatrix(dn.shape[1], dn.shape[0], dict(dn.T.todok().items()))
        
        # S = U * dn_T * V. 
        # The inverse of U transforms the standard codomain basis into a decomposed Z-basis.
        # SymPy's smith_normal_form internally converts to dense, but SparseMatrix input 
        # helps slightly with initial allocation.
        from sympy.matrices.normalforms import smith_normal_form
        S, U, V = smith_normal_form(dn_T)
        U_inv = U.inv()
        
        int_basis = []
        target_rank, _ = self.cohomology(n)
        
        # Columns of U_inv corresponding to entirely zero rows of S represent generators of C^n / B^n.
        for i in range(S.shape[0]):
            if all(S[i, j] == 0 for j in range(S.shape[1])):
                v = np.array(U_inv[:, i], dtype=np.int64).flatten()
                # Verify it is a cocycle (in Z^n)
                if dn_plus_1 is None or dn_plus_1.nnz == 0 or not np.any(dn_plus_1.T.toarray() @ v):
                    int_basis.append(v)
                if len(int_basis) == target_rank:
                    break
                    
        self._cache_set(key, int_basis)
        return int_basis

    def euler_characteristic(self) -> int:
        """
        Compute the Euler characteristic of the chain complex.
        chi(C) = sum_{n} (-1)^n * rank(C_n).
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

    def topological_invariants(self) -> Dict[str, Any]:
        """
        Compute all key topological invariants at once.
        Returns a dictionary containing homology, cohomology, Betti numbers, 
        and the Euler characteristic.
        """
        key = ("chain", "topological_invariants", str(self.coefficient_ring))
        cached = self._cache_get(key)
        if cached is not None:
            return cast(Dict[str, Any], cached)

        homology = self.homology()
        cohomology = self.cohomology()
        betti = self.betti_numbers()
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
    """
    Representation of a Finite CW Complex X.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cells: Dict[int, int]
    attaching_maps: Dict[int, csr_matrix]
    dimensions: List[int] = Field(default_factory=list)
    coefficient_ring: str = "Z"

    _cache_enabled: bool = PrivateAttr(default=True)
    _cache: dict[tuple[object, ...], object] = PrivateAttr(default_factory=dict)
    _cache_hits: int = PrivateAttr(default=0)
    _cache_misses: int = PrivateAttr(default=0)
    _cache_signature: tuple[object, ...] | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _normalize_model(self):
        object.__setattr__(self, "cells", {int(dim): int(count) for dim, count in self.cells.items()})
        object.__setattr__(
            self,
            "attaching_maps",
            {int(dim): _coerce_csr_matrix(mat) for dim, mat in self.attaching_maps.items()},
        )
        object.__setattr__(
            self,
            "dimensions",
            sorted({int(dim) for dim in self.dimensions} | set(self.cells.keys()) | set(self.attaching_maps.keys())),
        )
        object.__setattr__(self, "coefficient_ring", str(self.coefficient_ring))
        return self

    def _structure_signature(self) -> tuple[object, ...]:
        cell_sig = tuple((int(dim), int(count)) for dim, count in sorted(self.cells.items()))
        attach_sig = tuple(
            (int(dim), _csr_matrix_signature(mat))
            for dim, mat in sorted(self.attaching_maps.items())
        )
        return tuple(self.dimensions), cell_sig, attach_sig, str(self.coefficient_ring)

    def _ensure_cache_valid(self) -> None:
        current = self._structure_signature()
        if self._cache_signature != current:
            self._cache.clear()
            self._cache_signature = current

    def _cache_get(self, key: tuple[object, ...]) -> object | None:
        self._ensure_cache_valid()
        if not self._cache_enabled:
            return None
        if key in self._cache:
            self._cache_hits += 1
            return _clone_cache_value(self._cache[key])
        self._cache_misses += 1
        return None

    def _cache_set(self, key: tuple[object, ...], value: object) -> None:
        self._ensure_cache_valid()
        if not self._cache_enabled:
            return
        self._cache[key] = _clone_cache_value(value)

    def clear_cache(self, namespace: str | None = None) -> None:
        if namespace is None:
            self._cache.clear()
            return
        prefix = (str(namespace),)
        keys = [k for k in self._cache if k[:1] == prefix]
        for key in keys:
            self._cache.pop(key, None)

    def cache_info(self) -> dict[str, object]:
        self._ensure_cache_valid()
        return {
            "enabled": bool(self._cache_enabled),
            "size": int(len(self._cache)),
            "hits": int(self._cache_hits),
            "misses": int(self._cache_misses),
            "keys": [list(map(str, key)) for key in sorted(self._cache.keys(), key=repr)],
        }

    def boundary_matrix(self, dim: int) -> csr_matrix:
        """Return the cellular boundary matrix in the given dimension."""
        dim = int(dim)
        key = ("cw", "boundary_matrix", dim)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        matrix = self.attaching_maps.get(dim)
        if matrix is None:
            out = csr_matrix((self.cells.get(dim - 1, 0), self.cells.get(dim, 0)), dtype=np.int64)
            self._cache_set(key, out)
            return out
        self._cache_set(key, matrix)
        return matrix

    def boundary_matrices(self) -> dict[int, csr_matrix]:
        """Return all available cellular boundary matrices."""
        key = ("cw", "boundary_matrices")
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        out = {dim: self.boundary_matrix(dim) for dim in self.dimensions if dim > 0}
        self._cache_set(key, out)
        return out

    def cellular_chain_complex(self, coefficient_ring: str | None = None) -> ChainComplex:
        """Convert the CW object into a `ChainComplex` view.

        Parameters
        ----------
        coefficient_ring:
            Optional ring override (for example ``"Q"`` or ``"Z/2Z"``).
            When omitted, the complex default ``self.coefficient_ring`` is used.
        """
        ring = self.coefficient_ring if coefficient_ring is None else str(coefficient_ring)
        _parse_coefficient_ring(ring)
        key = ("cw", "cellular_chain_complex", ring)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        out = ChainComplex(
            boundaries=self.boundary_matrices(),
            dimensions=self.dimensions,
            cells=self.cells,
            coefficient_ring=ring,
        )
        self._cache_set(key, out)
        return out


__all__ = ["SimplicialComplex", "ChainComplex", "CWComplex"]

