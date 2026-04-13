import os
import threading
import importlib.util
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
        except Exception as e:
            self.error = f"Failed to initialize Julia backend: {e!r}"
            self._available = False
        finally:
            self._initialized = True

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

    def compute_hermitian_signature(self, matrix_array: np.ndarray) -> int:
        self.require_julia()
        # Direct zero-copy passing to Julia
        return int(self.backend.hermitian_signature(matrix_array))

    def compute_sparse_snf(self, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, shape: tuple) -> np.ndarray:
        """Executes the highly optimized Julia Sparse SNF backend."""
        self.require_julia()
        # Keep zero-based indices on Python side; backend lifts to Julia indexing.
        jl_rows = self.jl.Array(np.asarray(rows, dtype=np.int64))
        jl_cols = self.jl.Array(np.asarray(cols, dtype=np.int64))
        jl_vals = self.jl.Array(np.asarray(vals, dtype=np.int64))
        factors = self.backend.exact_snf_sparse(jl_rows, jl_cols, jl_vals, shape[0], shape[1])
        return np.array(factors, dtype=np.int64)

    def compute_sparse_rank_q(self, matrix) -> int:
        """Compute matrix rank over Q using Julia backend from sparse COO data."""
        self.require_julia()
        if matrix is None or matrix.nnz == 0:
            return 0
        coo = matrix.tocoo()
        rows = np.asarray(coo.row, dtype=np.int64)
        cols = np.asarray(coo.col, dtype=np.int64)
        vals = np.asarray(coo.data, dtype=np.int64)
        rank = self.backend.rank_q_sparse(
            self.jl.Array(rows),
            self.jl.Array(cols),
            self.jl.Array(vals),
            int(matrix.shape[0]),
            int(matrix.shape[1]),
        )
        return int(rank)

    def compute_sparse_rank_mod_p(self, matrix, p: int) -> int:
        """Compute matrix rank over Z/pZ using Julia backend from sparse COO data."""
        self.require_julia()
        if matrix is None or matrix.nnz == 0:
            return 0
        coo = matrix.tocoo()
        rows = np.asarray(coo.row, dtype=np.int64)
        cols = np.asarray(coo.col, dtype=np.int64)
        vals = np.asarray(coo.data, dtype=np.int64)
        rank = self.backend.rank_mod_p_sparse(
            self.jl.Array(rows),
            self.jl.Array(cols),
            self.jl.Array(vals),
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
            if cn_size is not None:
                d_np1_m, d_np1_n = (cn_size, 0)
            else:
                d_np1_m, d_np1_n = (d_n.shape[1], 0) if d_n is not None else (0, 0)
        else:
            d_np1_coo = d_np1.tocoo()
            d_np1_rows, d_np1_cols, d_np1_vals = d_np1_coo.row, d_np1_coo.col, d_np1_coo.data
            d_np1_m, d_np1_n = d_np1.shape
            
        if d_n is None or d_n.nnz == 0:
            d_n_rows = np.array([], dtype=np.int64)
            d_n_cols = np.array([], dtype=np.int64)
            d_n_vals = np.array([], dtype=np.int64)
            if cn_size is not None:
                d_n_m, d_n_n = (0, cn_size)
            else:
                d_n_m, d_n_n = (0, d_np1_m) if d_np1 is not None else (0, 0)
        else:
            d_n_coo = d_n.tocoo()
            d_n_rows, d_n_cols, d_n_vals = d_n_coo.row, d_n_coo.col, d_n_coo.data
            d_n_m, d_n_n = d_n.shape
            
        # Keep zero-based indices on Python side; backend lifts to Julia indexing.
        d_np1_rows = np.asarray(d_np1_rows, dtype=np.int64)
        d_np1_cols = np.asarray(d_np1_cols, dtype=np.int64)
        d_np1_vals = np.asarray(d_np1_vals, dtype=np.int64)
        d_n_rows = np.asarray(d_n_rows, dtype=np.int64)
        d_n_cols = np.asarray(d_n_cols, dtype=np.int64)
        d_n_vals = np.asarray(d_n_vals, dtype=np.int64)
        basis_jl = self.backend.exact_sparse_cohomology_basis(
            self.jl.Array(d_np1_rows), self.jl.Array(d_np1_cols), self.jl.Array(d_np1_vals), d_np1_m, d_np1_n,
            self.jl.Array(d_n_rows), self.jl.Array(d_n_cols), self.jl.Array(d_n_vals), d_n_m, d_n_n
        )
        
        # Parse output
        basis_py = []
        for vec in basis_jl:
            basis_py.append(np.array(vec, dtype=np.int64))
        return basis_py

    def compute_sparse_cohomology_basis_mod_p(self, d_np1, d_n, p: int, cn_size: int | None = None) -> list:
        """Executes Julia sparse cohomology basis extraction over Z/pZ for prime p."""
        self.require_julia()

        if d_np1 is None or d_np1.nnz == 0:
            d_np1_rows = np.array([], dtype=np.int64)
            d_np1_cols = np.array([], dtype=np.int64)
            d_np1_vals = np.array([], dtype=np.int64)
            if cn_size is not None:
                d_np1_m, d_np1_n = (cn_size, 0)
            else:
                d_np1_m, d_np1_n = (d_n.shape[1], 0) if d_n is not None else (0, 0)
        else:
            d_np1_coo = d_np1.tocoo()
            d_np1_rows, d_np1_cols, d_np1_vals = d_np1_coo.row, d_np1_coo.col, d_np1_coo.data
            d_np1_m, d_np1_n = d_np1.shape

        if d_n is None or d_n.nnz == 0:
            d_n_rows = np.array([], dtype=np.int64)
            d_n_cols = np.array([], dtype=np.int64)
            d_n_vals = np.array([], dtype=np.int64)
            if cn_size is not None:
                d_n_m, d_n_n = (0, cn_size)
            else:
                d_n_m, d_n_n = (0, d_np1_m) if d_np1 is not None else (0, 0)
        else:
            d_n_coo = d_n.tocoo()
            d_n_rows, d_n_cols, d_n_vals = d_n_coo.row, d_n_coo.col, d_n_coo.data
            d_n_m, d_n_n = d_n.shape

        d_np1_rows = np.asarray(d_np1_rows, dtype=np.int64)
        d_np1_cols = np.asarray(d_np1_cols, dtype=np.int64)
        d_np1_vals = np.asarray(d_np1_vals, dtype=np.int64)
        d_n_rows = np.asarray(d_n_rows, dtype=np.int64)
        d_n_cols = np.asarray(d_n_cols, dtype=np.int64)
        d_n_vals = np.asarray(d_n_vals, dtype=np.int64)
        basis_jl = self.backend.sparse_cohomology_basis_mod_p(
            self.jl.Array(d_np1_rows), self.jl.Array(d_np1_cols), self.jl.Array(d_np1_vals), d_np1_m, d_np1_n,
            self.jl.Array(d_n_rows), self.jl.Array(d_n_cols), self.jl.Array(d_n_vals), d_n_m, d_n_n,
            int(p),
        )

        basis_py = []
        for vec in basis_jl:
            basis_py.append(np.array(vec, dtype=np.int64) % int(p))
        return basis_py
        
    def group_ring_multiply(self, coeffs1: dict, coeffs2: dict, group_order: int) -> dict:
        self.require_julia()
        # Normalize Python values before crossing language boundary.
        k1 = [str(k) for k in coeffs1.keys()]
        v1 = [int(v) for v in coeffs1.values()]
        k2 = [str(k) for k in coeffs2.keys()]
        v2 = [int(v) for v in coeffs2.values()]
        res_keys, res_vals = self.backend.group_ring_multiply(k1, v1, k2, v2, group_order)
        return {str(k): int(v) for k, v in zip(res_keys, res_vals)}
        
    def compute_multisignature(self, matrix: np.ndarray, p: int) -> int:
        """Evaluates L_{4k}(Z_p) obstruction by computing multisignature."""
        self.require_julia()
        return int(self.backend.multisignature(matrix, p))

    def integral_lattice_isometry(self, matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray | None:
        """Find U in GL_n(Z) with U^T * matrix1 * U = matrix2 for definite forms."""
        self.require_julia()
        candidate = self.backend.integral_lattice_isometry(matrix1, matrix2)
        if candidate is None:
            return None
        return np.array(candidate, dtype=np.int64)

    def abelianize_and_bhs_rank(self, generators: list, relations: list) -> tuple:
        """
        Takes raw string generators and relations, computes the abelianization,
        and extracts the free and torsion ranks for the Bass-Heller-Swan K-theory formula.
        Returns (free_rank, torsion_orders_list)
        """
        self.require_julia()
        # Flat relations for passing
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
        pts = point_cloud if point_cloud is not None else self.jl.nothing
        mr = max_roots if max_roots is not None else self.jl.nothing
        mc = max_cycles if max_cycles is not None else self.jl.nothing
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
        pts = point_cloud if point_cloud is not None else self.jl.nothing
        mr = max_roots if max_roots is not None else self.jl.nothing
        mc = max_cycles if max_cycles is not None else self.jl.nothing
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
        self.require_julia()
        result = self.backend.compute_boundary_data_from_simplices(
            simplex_entries,
            int(max_dim),
        )

        if len(result) == 4:
            boundaries_jl, cells_jl, dim_simplices_jl, simplex_to_idx_jl = result
        elif len(result) == 3:
            boundaries_jl, cells_jl, dim_simplices_jl = result
            simplex_to_idx_jl = {
                k: {simplex: i for i, simplex in enumerate(simplices)}
                for k, simplices in dict(dim_simplices_jl).items()
            }
        else:
            raise ValueError(
                "Julia boundary builder returned an unexpected number of values: "
                f"{len(result)}"
            )

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
        dim_simplices_py = {
            int(k): [tuple(int(x) for x in simplex) for simplex in simplices]
            for k, simplices in dict(dim_simplices_jl).items()
            if len(simplices) > 0
        }
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
            self.jl.Array(np.asarray(alpha, dtype=np.int64)),
            self.jl.Array(np.asarray(beta, dtype=np.int64)),
            int(p),
            int(q),
            simplices_p_plus_q,
            simplex_to_idx_p,
            simplex_to_idx_q,
            modulus if modulus is not None else self.jl.nothing,
        )
        return np.asarray(result, dtype=np.int64)

    def compute_trimesh_boundary_data(self, faces: list, n_vertices: int) -> dict:
        """Compute trimesh boundary operators (d1, d2) through Julia."""
        self.require_julia()
        payload = self.backend.compute_trimesh_boundary_data(faces, int(n_vertices))
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
        """
        Triangulate a 2D surface from a point cloud using Delaunay triangulation.

        Parameters
        ----------
        points : np.ndarray
            Point cloud of shape (n_points, 3)
        tolerance : float
            Tolerance for detecting degenerate cases

        Returns
        -------
        list
            List of triangles, where each triangle is a sorted list of 3 vertex indices
        """
        self.require_julia()
        points = np.asarray(points, dtype=np.float64)
        triangles = self.backend.triangulate_surface_delaunay(points, float(tolerance))
        # Convert Julia arrays to Python lists
        return [list(tri) for tri in triangles]


julia_engine = JuliaBridge()
