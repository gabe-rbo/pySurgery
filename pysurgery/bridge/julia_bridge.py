import os
import threading
import importlib.util
from collections import OrderedDict
import numpy as np

HAS_JULIACALL = importlib.util.find_spec("juliacall") is not None

class JuliaBridge:
    """
    Zero-Copy Bridge to execute high-performance Julia algebraic topology operations.
    Replaces subprocess mocks with native memory sharing via `juliacall`.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(JuliaBridge, cls).__new__(cls)
                    cls._instance._initialized = False
                    cls._instance._available = False
                    cls._instance.error = None
                    cls._instance.jl = None
                    cls._instance.backend = None
                    cls._instance._coo_cache = OrderedDict()
        return cls._instance

    def _initialize(self):
        if self._initialized:
            return

        self._available = False
        if not HAS_JULIACALL:
            self.error = "juliacall is not installed. Install via `pip install juliacall`."
            self._initialized = True
            return
            
        try:
            from juliacall import Main as jl_main
            self.jl = jl_main
            backend_script = os.path.join(os.path.dirname(__file__), "surgery_backend.jl")
            self.jl.include(backend_script)
            self.backend = self.jl.SurgeryBackend
            self._available = True
            self._warm_up_compilers()
        except Exception as e:
            self.error = f"Failed to initialize Julia backend: {e!r}"
            self._available = False
        finally:
            self._initialized = True

    def _warm_up_compilers(self) -> None:
        """Best-effort, silent precompile warmup for first-call latency."""
        if not self._available or self.jl is None or self.backend is None:
            return
        try:
            # Note: Julia side now handles warming up via @compile_workload
            pass
        except Exception:
            return

    def _ensure_initialized(self):
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            self._initialize()

    @property
    def available(self) -> bool:
        self._ensure_initialized()
        return self._available

    @available.setter
    def available(self, value: bool):
        # Allow tests to monkeypatch availability while preserving singleton API.
        self._initialized = True
        self._available = bool(value)

    def require_julia(self):
        from pysurgery.core.exceptions import SurgeryError
        if not self.available:
            raise SurgeryError(f"High-performance exact algebra requires Julia: {self.error}")

    def _coo_cache_key(self, matrix) -> tuple:
        data_ptr = None
        try:
            data_ptr = int(matrix.data.__array_interface__["data"][0])
        except Exception:
            data_ptr = None
        return (id(matrix), int(matrix.shape[0]), int(matrix.shape[1]), int(matrix.nnz), data_ptr)

    def _coo_triplets_cached(self, matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = self._coo_cache_key(matrix)
        cached = self._coo_cache.get(key)
        if cached is not None:
            self._coo_cache.move_to_end(key)
            return cached

        coo = matrix.tocoo()
        triplets = (
            np.asarray(coo.row, dtype=np.int64),
            np.asarray(coo.col, dtype=np.int64),
            np.asarray(coo.data, dtype=np.int64),
        )
        self._coo_cache[key] = triplets
        if len(self._coo_cache) > 24:
            self._coo_cache.popitem(last=False)
        return triplets

    def _flatten_simplices(self, simplices: list) -> tuple[np.ndarray, np.ndarray]:
        offsets = np.zeros(len(simplices) + 1, dtype=np.int64)
        total = 0
        for i, simplex in enumerate(simplices, start=1):
            total += len(simplex)
            offsets[i] = total
        flat = np.empty(total, dtype=np.int64)
        cursor = 0
        for simplex in simplices:
            slen = len(simplex)
            if slen:
                flat[cursor : cursor + slen] = np.asarray(simplex, dtype=np.int64)
                cursor += slen
        return flat, offsets

    def compute_hermitian_signature(self, matrix_array: np.ndarray) -> int:
        self.require_julia()
        # Direct zero-copy passing to Julia via PyArray
        return int(self.backend.hermitian_signature(np.asarray(matrix_array, dtype=np.float64)))

    def compute_sparse_snf(self, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, shape: tuple) -> np.ndarray:
        """Executes the highly optimized Julia Sparse SNF backend."""
        self.require_julia()
        # Direct zero-copy passing of NumPy arrays
        factors = self.backend.exact_snf_sparse(
            np.asarray(rows, dtype=np.int64),
            np.asarray(cols, dtype=np.int64),
            np.asarray(vals, dtype=np.int64),
            int(shape[0]),
            int(shape[1])
        )
        return np.array(factors, dtype=np.int64)

    def compute_sparse_rank_q(self, matrix) -> int:
        """Compute matrix rank over Q using Julia backend from sparse COO data."""
        self.require_julia()
        if matrix is None or matrix.nnz == 0:
            return 0
        rows, cols, vals = self._coo_triplets_cached(matrix)
        rank = self.backend.rank_q_sparse(
            rows,
            cols,
            vals,
            int(matrix.shape[0]),
            int(matrix.shape[1]),
        )
        return int(rank)

    def compute_sparse_rank_mod_p(self, matrix, p: int) -> int:
        """Compute matrix rank over Z/pZ using Julia backend from sparse COO data."""
        self.require_julia()
        if matrix is None or matrix.nnz == 0:
            return 0
        rows, cols, vals = self._coo_triplets_cached(matrix)
        rank = self.backend.rank_mod_p_sparse(
            rows,
            cols,
            vals,
            int(matrix.shape[0]),
            int(matrix.shape[1]),
            int(p),
        )
        return int(rank)

    def compute_sparse_cohomology_basis(self, d_np1, d_n, cn_size: int | None = None) -> list:
        """Executes the exact Julia sparse cohomology basis extraction Z^n / B^n."""
        self.require_julia()
        
        if d_np1 is None or d_np1.nnz == 0:
            d_np1_rows = np.array([], dtype=np.int64)
            d_np1_cols = np.array([], dtype=np.int64)
            d_np1_vals = np.array([], dtype=np.int64)
            d_np1_m, d_np1_n = (cn_size, 0) if cn_size is not None else (0, 0)
        else:
            d_np1_rows, d_np1_cols, d_np1_vals = self._coo_triplets_cached(d_np1)
            d_np1_m, d_np1_n = d_np1.shape
            
        if d_n is None or d_n.nnz == 0:
            d_n_rows = np.array([], dtype=np.int64)
            d_n_cols = np.array([], dtype=np.int64)
            d_n_vals = np.array([], dtype=np.int64)
            d_n_m, d_n_n = (0, cn_size) if cn_size is not None else (0, d_np1_m)
        else:
            d_n_rows, d_n_cols, d_n_vals = self._coo_triplets_cached(d_n)
            d_n_m, d_n_n = d_n.shape
            
        # Julia now returns a flat Matrix{Int64}
        basis_mat = self.backend.exact_sparse_cohomology_basis(
            d_np1_rows, d_np1_cols, d_np1_vals, int(d_np1_m), int(d_np1_n),
            d_n_rows, d_n_cols, d_n_vals, int(d_n_m), int(d_n_n)
        )
        
        # Convert columns to list of vectors
        basis_py = []
        for j in range(basis_mat.shape[1]):
            basis_py.append(np.array(basis_mat[:, j], dtype=np.int64))
        return basis_py

    def compute_sparse_cohomology_basis_mod_p(self, d_np1, d_n, p: int, cn_size: int | None = None) -> list:
        """Executes Julia sparse cohomology basis extraction over Z/pZ for prime p."""
        self.require_julia()

        if d_np1 is None or d_np1.nnz == 0:
            d_np1_rows = np.array([], dtype=np.int64)
            d_np1_cols = np.array([], dtype=np.int64)
            d_np1_vals = np.array([], dtype=np.int64)
            d_np1_m, d_np1_n = (cn_size, 0) if cn_size is not None else (0, 0)
        else:
            d_np1_rows, d_np1_cols, d_np1_vals = self._coo_triplets_cached(d_np1)
            d_np1_m, d_np1_n = d_np1.shape

        if d_n is None or d_n.nnz == 0:
            d_n_rows = np.array([], dtype=np.int64)
            d_n_cols = np.array([], dtype=np.int64)
            d_n_vals = np.array([], dtype=np.int64)
            d_n_m, d_n_n = (0, cn_size) if cn_size is not None else (0, d_np1_m)
        else:
            d_n_rows, d_n_cols, d_n_vals = self._coo_triplets_cached(d_n)
            d_n_m, d_n_n = d_n.shape

        basis_mat = self.backend.sparse_cohomology_basis_mod_p(
            d_np1_rows, d_np1_cols, d_np1_vals, int(d_np1_m), int(d_np1_n),
            d_n_rows, d_n_cols, d_n_vals, int(d_n_m), int(d_n_n),
            int(p),
        )

        basis_py = []
        for j in range(basis_mat.shape[1]):
            basis_py.append(np.array(basis_mat[:, j], dtype=np.int64))
        return basis_py

    def compute_boundary_payload_from_simplices(
        self,
        simplex_entries: list,
        max_dim: int,
        *,
        include_metadata: bool = True,
    ) -> tuple:
        """Build boundary payloads through Julia with optional metadata suppression."""
        self.require_julia()
        flat_vertices, simplex_offsets = self._flatten_simplices(simplex_entries)
        return self.backend.compute_boundary_payload_from_flat_simplices(
            flat_vertices,
            simplex_offsets,
            int(max_dim),
            bool(include_metadata),
        )

    def group_ring_multiply(self, coeffs1: dict, coeffs2: dict, group_order: int) -> dict:
        self.require_julia()
        # Direct passing of dictionaries; Julia side now uses pyconvert for speed.
        res_keys, res_vals = self.backend.group_ring_multiply(coeffs1, coeffs2, int(group_order))
        return {str(k): int(v) for k, v in zip(res_keys, res_vals)}
        
    def compute_multisignature(self, matrix: np.ndarray, p: int) -> int:
        """Evaluates L_{4k}(Z_p) obstruction by computing multisignature."""
        self.require_julia()
        return int(self.backend.multisignature(np.asarray(matrix, dtype=np.float64), int(p)))

    def integral_lattice_isometry(self, matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray | None:
        """Find U in GL_n(Z) with U^T * matrix1 * U = matrix2 for definite forms."""
        self.require_julia()
        candidate = self.backend.integral_lattice_isometry(
            np.asarray(matrix1, dtype=np.int64),
            np.asarray(matrix2, dtype=np.int64)
        )
        if candidate is None:
            return None
        return np.array(candidate, dtype=np.int64)

    def abelianize_and_bhs_rank(self, generators: list, relations: list) -> tuple:
        """
        Takes raw string generators and relations, computes the abelianization,
        and extracts the free and torsion ranks for the Bass-Heller-Swan K-theory formula.
        """
        self.require_julia()
        flat_rels = [" ".join(r) for r in relations]
        free_rank, torsions = self.backend.abelianize_group(generators, flat_rels)
        return int(free_rank), list(torsions)

    def compute_optimal_h1_basis_from_simplices(
        self,
        simplices: list,
        num_vertices: int,
        *,
        point_cloud: np.ndarray | None = None,
        max_roots: int | None = None,
        root_stride: int = 1,
        max_cycles: int | None = None,
    ) -> list:
        """Compute an optimal H1 basis via Julia backend (Algorithms 8+7 composition)."""
        self.require_julia()
        pts = np.asarray(point_cloud, dtype=np.float64) if point_cloud is not None else self.jl.nothing
        mr = int(max_roots) if max_roots is not None else self.jl.nothing
        mc = int(max_cycles) if max_cycles is not None else self.jl.nothing
        out = self.backend.optgen_from_simplices(simplices, int(num_vertices), pts, mr, int(root_stride), mc)
        basis_py = []
        for cyc in out:
            basis_py.append([tuple((int(e[0]), int(e[1]))) for e in cyc])
        return basis_py

    def compute_homology_basis_from_simplices(
        self,
        simplices: list,
        num_vertices: int,
        dimension: int,
        *,
        mode: str = "valid",
        point_cloud: np.ndarray | None = None,
        max_roots: int | None = None,
        root_stride: int = 1,
        max_cycles: int | None = None,
    ) -> list[dict]:
        """Compute H_k generator representatives from simplices via Julia backend over Z/2."""
        self.require_julia()
        pts = np.asarray(point_cloud, dtype=np.float64) if point_cloud is not None else self.jl.nothing
        mr = int(max_roots) if max_roots is not None else self.jl.nothing
        mc = int(max_cycles) if max_cycles is not None else self.jl.nothing
        out = self.backend.homology_generators_from_simplices(
            simplices,
            int(num_vertices),
            int(dimension),
            str(mode),
            pts,
            mr,
            int(root_stride),
            mc,
        )
        parsed: list[dict] = []
        for g in out:
            support_simplices = [tuple(int(x) for x in simplex) for simplex in g["support_simplices"]]
            support_edges = [tuple((int(e[0]), int(e[1]))) for e in g["support_edges"]]
            parsed.append(
                {
                    "dimension": int(g["dimension"]),
                    "support_simplices": support_simplices,
                    "support_edges": support_edges,
                    "weight": float(g["weight"]),
                    "certified_cycle": bool(g["certified_cycle"]),
                }
            )
        return parsed

    def compute_boundary_data_from_simplices(self, simplex_entries: list, max_dim: int) -> tuple[dict, dict, dict, dict]:
        """Build boundary COO payloads and simplex tables through Julia for large simplicial workloads."""
        result = self.compute_boundary_payload_from_simplices(
            simplex_entries,
            max_dim,
            include_metadata=True,
        )

        boundaries_jl, cells_jl, dim_simplices_jl, simplex_to_idx_jl = result

        boundaries_py: dict[int, dict[str, object]] = {}
        for k, payload in dict(boundaries_jl).items():
            kk = int(k)
            boundaries_py[kk] = {
                "rows": np.asarray(payload["rows"], dtype=np.int64),
                "cols": np.asarray(payload["cols"], dtype=np.int64),
                "data": np.asarray(payload["data"], dtype=np.int64),
                "n_rows": int(payload["n_rows"]),
                "n_cols": int(payload["n_cols"]),
            }

        cells_py = {int(k): int(v) for k, v in dict(cells_jl).items()}
        
        # dim_simplices_jl is now returning matrices for each dimension
        dim_simplices_py = {}
        for k, matrix in dict(dim_simplices_jl).items():
            kk = int(k)
            # matrix is (d+1, N)
            dim_simplices_py[kk] = [tuple(int(x) for x in matrix[:, j]) for j in range(matrix.shape[1])]

        simplex_to_idx_py = {
            int(k): {
                tuple(int(x) for x in simplex): int(idx)
                for simplex, idx in dict(idx_map).items()
            }
            for k, idx_map in dict(simplex_to_idx_jl).items()
            if len(dict(idx_map)) > 0
        }
        return boundaries_py, cells_py, dim_simplices_py, simplex_to_idx_py

    def compute_boundary_mod2_matrix(self, source_simplices: list, target_simplices: list) -> dict:
        """Compute mod-2 boundary matrix through Julia for fast homology generator extraction."""
        self.require_julia()
        payload = self.backend.compute_boundary_mod2_matrix(source_simplices, target_simplices)
        return {
            "rows": np.asarray(payload["rows"], dtype=np.int64),
            "cols": np.asarray(payload["cols"], dtype=np.int64),
            "data": np.asarray(payload["data"], dtype=np.int64),
            "m": int(payload["m"]),
            "n": int(payload["n"]),
        }

    def compute_alexander_whitney_cup(
        self,
        alpha: np.ndarray,
        beta: np.ndarray,
        p: int,
        q: int,
        simplices_p_plus_q: list,
        simplex_to_idx_p: dict,
        simplex_to_idx_q: dict,
        modulus: int | None = None,
    ) -> np.ndarray:
        """Compute Alexander-Whitney cup product through Julia for fast intersection form extraction."""
        self.require_julia()
        result = self.backend.compute_alexander_whitney_cup(
            np.asarray(alpha, dtype=np.int64),
            np.asarray(beta, dtype=np.int64),
            int(p),
            int(q),
            simplices_p_plus_q,
            simplex_to_idx_p,
            simplex_to_idx_q,
            int(modulus) if modulus is not None else self.jl.nothing,
        )
        return np.asarray(result, dtype=np.int64)

    def compute_trimesh_boundary_data(self, faces: list, n_vertices: int) -> dict:
        """Compute trimesh boundary operators (d1, d2) through Julia."""
        self.require_julia()
        if isinstance(faces, np.ndarray) and faces.ndim == 2:
            faces_arr = np.asarray(faces, dtype=np.int64)
            flat = np.ascontiguousarray(faces_arr.reshape(-1), dtype=np.int64)
            offsets = np.arange(0, flat.size + 1, faces_arr.shape[1], dtype=np.int64)
            payload = self.backend.compute_trimesh_boundary_data_flat(
                flat,
                offsets,
                int(n_vertices),
            )
        else:
            flat, offsets = self._flatten_simplices(list(faces))
            payload = self.backend.compute_trimesh_boundary_data_flat(
                flat,
                offsets,
                int(n_vertices),
            )
        return {
            "d1_rows": np.asarray(payload["d1_rows"], dtype=np.int64),
            "d1_cols": np.asarray(payload["d1_cols"], dtype=np.int64),
            "d1_data": np.asarray(payload["d1_data"], dtype=np.int64),
            "n_vertices": int(payload["n_vertices"]),
            "n_edges": int(payload["n_edges"]),
            "d2_rows": np.asarray(payload["d2_rows"], dtype=np.int64),
            "d2_cols": np.asarray(payload["d2_cols"], dtype=np.int64),
            "d2_data": np.asarray(payload["d2_data"], dtype=np.int64),
            "n_faces": int(payload["n_faces"]),
        }

    def triangulate_surface_delaunay(self, points: np.ndarray, tolerance: float = 1e-10) -> list:
        """Triangulate a 2D surface from a point cloud using Delaunay triangulation."""
        self.require_julia()
        triangles = self.backend.triangulate_surface_delaunay(
            np.asarray(points, dtype=np.float64),
            float(tolerance)
        )
        return [list(tri) for tri in triangles]


julia_engine = JuliaBridge()
