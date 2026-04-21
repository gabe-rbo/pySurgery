import os
import threading
import importlib.util
import warnings
from collections import OrderedDict
import numpy as np
import scipy.sparse as sp

HAS_JULIACALL = importlib.util.find_spec("juliacall") is not None


class JuliaBridge:
    """
    Zero-Copy Bridge to execute high-performance Julia algebraic topology operations.
    Replaces subprocess mocks with native memory sharing via `juliacall`.
    """

    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        """Return a process-wide singleton bridge instance."""
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
                    cls._instance._warmup_level = 0
                    cls._instance._warmup_report = {}
        return cls._instance

    def _initialize(self):
        """Initialize Julia runtime and load backend module lazily."""
        if self._initialized:
            return

        self._available = False
        if not HAS_JULIACALL:
            self.error = (
                "juliacall is not installed. Install via `pip install juliacall`."
            )
            self._initialized = True
            return

        try:
            from juliacall import Main as jl_main

            self.jl = jl_main
            backend_script = os.path.join(
                os.path.dirname(__file__), "surgery_backend.jl"
            )
            self.jl.include(backend_script)
            self.backend = self.jl.SurgeryBackend
            self._available = True
            # Mark initialized before warm-up workloads so require_julia() checks
            # do not recursively re-enter _initialize().
            self._initialized = True
            self._warm_up_compilers()
        except Exception as e:
            self.error = f"Failed to initialize Julia backend: {e!r}"
            self._available = False
            self._initialized = True
        finally:
            self._initialized = True

    def _warm_up_compilers(self) -> None:
        """Best-effort automatic warm-up on first Julia initialization."""
        mode = os.getenv("PYSURGERY_JULIA_WARMUP_MODE", "minimal").strip().lower()
        if mode in {"", "off", "0", "false", "none"}:
            return
        if mode not in {"minimal", "full"}:
            mode = "minimal"
        self._run_warmup(mode)

    def _minimal_warmup_workloads(self) -> list[tuple[str, callable]]:
        """Return a small workload set that compiles common topology paths."""
        square_simplices = [(0, 1), (1, 2), (2, 3), (3, 0)]
        square_pts = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        tetra_faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        tetra_pts = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.9, 0.0], [0.5, 0.3, 0.8]],
            dtype=np.float64,
        )
        payload_simplices = [
            (0,),
            (1,),
            (2,),
            (3,),
            (0, 1),
            (1, 2),
            (2, 3),
            (0, 3),
            (0, 1, 2),
            (0, 2, 3),
        ]

        return [
            (
                "h1_opt_square",
                lambda: self.compute_optimal_h1_basis_from_simplices(
                    square_simplices,
                    4,
                    point_cloud=square_pts,
                    max_cycles=4,
                ),
            ),
            (
                "h1_valid_square",
                lambda: self.compute_homology_basis_from_simplices(
                    square_simplices,
                    4,
                    1,
                    point_cloud=square_pts,
                    mode="valid",
                    max_cycles=4,
                ),
            ),
            (
                "h2_tetra_boundary",
                lambda: self.compute_homology_basis_from_simplices(
                    tetra_faces,
                    4,
                    2,
                    point_cloud=tetra_pts,
                    mode="valid",
                    max_cycles=2,
                ),
            ),
            (
                "boundary_payload",
                lambda: self.compute_boundary_payload_from_simplices(
                    payload_simplices,
                    2,
                    include_metadata=False,
                ),
            ),
            (
                "boundary_mod2",
                lambda: self.compute_boundary_mod2_matrix(
                    [(0, 1, 2), (0, 2, 3)],
                    [(0, 1), (1, 2), (0, 2), (2, 3), (0, 3)],
                ),
            ),
            (
                "pi1_raw_traces",
                lambda: self.compute_pi1_trace_candidates(
                    np.array([0, 1], dtype=np.int64),
                    np.array([0, 0], dtype=np.int64),
                    np.array([-1, 1], dtype=np.int64),
                    n_vertices=2,
                    n_edges=1,
                ),
            ),
            (
                "metrics_warmup",
                lambda: (
                    self.orthogonal_procrustes(np.eye(2), np.eye(2)),
                    self.pairwise_distance_matrix(np.array([[0., 0.], [1., 1.]]), "euclidean"),
                    self.frechet_distance(np.array([[0., 0.], [1., 1.]]), np.array([[0., 1.], [1., 0.]])),
                    self.gromov_wasserstein_distance(
                        np.array([[0., 1.], [1., 0.]]), np.array([[0., 1.], [1., 0.]]),
                        np.array([0.5, 0.5]), np.array([0.5, 0.5]), 0.01, 2
                    ),
                    self.quick_mapper_jl({"V": [0, 1], "E": [(0, 1)]}, 1, -1.0),
                    self.compute_cknn_graph(np.array([[0., 0.], [1., 1.]]), 1, 1.0)
                ),
            ),
        ]

    def _full_warmup_workloads(self) -> list[tuple[str, callable]]:
        """Return the extended warm-up workload set for all heavy kernels."""

        def _sparse_rank_q_workload():
            import scipy.sparse as sp

            m = sp.csr_matrix(np.array([[1, 0], [0, 1]], dtype=np.int64))
            return self.compute_sparse_rank_q(m)

        def _sparse_rank_mod_p_workload():
            import scipy.sparse as sp

            m = sp.csr_matrix(np.array([[1, 1], [0, 1]], dtype=np.int64))
            return self.compute_sparse_rank_mod_p(m, 2)

        def _sparse_cohomology_workload():
            import scipy.sparse as sp

            d_np1 = sp.csr_matrix(np.zeros((1, 0), dtype=np.int64))
            d_n = sp.csr_matrix(np.array([[1], [0]], dtype=np.int64))
            return self.compute_sparse_cohomology_basis(d_np1, d_n, cn_size=1)

        def _normal_surface_residual_workload():
            import scipy.sparse as sp

            m = sp.csr_matrix(np.array([[1, -1], [0, 1]], dtype=np.int64))
            v = np.array([[1, 0], [0, 1]], dtype=np.int64)
            return self.compute_normal_surface_residual_norms(m, v)

        return [
            (
                "sparse_snf",
                lambda: self.compute_sparse_snf(
                    np.array([0, 1], dtype=np.int64),
                    np.array([0, 1], dtype=np.int64),
                    np.array([1, 1], dtype=np.int64),
                    (2, 2),
                ),
            ),
            ("sparse_rank_q", _sparse_rank_q_workload),
            ("sparse_rank_mod_p", _sparse_rank_mod_p_workload),
            ("sparse_cohomology_basis", _sparse_cohomology_workload),
            (
                "sparse_cohomology_mod_p",
                lambda: self.compute_sparse_cohomology_basis_mod_p(
                    sp.csr_matrix(np.zeros((1, 0), dtype=np.int64)),
                    sp.csr_matrix(np.array([[1], [0]], dtype=np.int64)),
                    2,
                    cn_size=1
                ),
            ),
            (
                "alexander_whitney_cup",
                lambda: self.compute_alexander_whitney_cup(
                    np.array([1, 0, 1], dtype=np.int64),
                    np.array([1, 1, 0], dtype=np.int64),
                    1,
                    1,
                    [(0, 1, 2)],
                    {(0, 1): 0, (1, 2): 1, (0, 2): 2},
                    {(0, 1): 0, (1, 2): 1, (0, 2): 2},
                    modulus=2,
                ),
            ),
            (
                "group_ring_multiply",
                lambda: self.group_ring_multiply({(0,): 1}, {(0,): 1}, 2),
            ),
            (
                "pi1_abelianization",
                lambda: self.abelianize_and_bhs_rank(["a"], [["a", "a"]]),
            ),
            (
                "integral_lattice_isometry",
                lambda: self.integral_lattice_isometry(
                    np.array([[1, 0], [0, 1]], dtype=np.int64),
                    np.array([[1, 0], [0, 1]], dtype=np.int64),
                ),
            ),
            (
                "normal_surface_residual_norms",
                _normal_surface_residual_workload,
            ),
            (
                "manifold_certification",
                lambda: self.is_homology_manifold_jl([(0, 1), (1, 2), (0, 2)], 1),
            ),
            (
                "broad_phase_pairs",
                lambda: self.compute_broad_phase_pairs(
                    np.array([[0., 0.], [1., 1.]]), np.array([0.1, 0.1]), tol=0.1
                ),
            ),
            (
                "hermitian_signature",
                lambda: self.compute_hermitian_signature(np.eye(2)),
            ),
            (
                "multisignature",
                lambda: self.compute_multisignature(np.eye(2), 2),
            ),
            (
                "boundary_data_assembly",
                lambda: (
                    self.compute_boundary_data_from_simplices([(0, 1, 2)], 2),
                    self.compute_boundary_payload_from_flat_simplices([(0, 1, 2)], 2),
                    self.compute_trimesh_boundary_data([(0, 1, 2)], 3)
                ),
            ),
            (
                "geometric_kernels",
                lambda: (
                    self.triangulate_surface_delaunay(np.array([[0,0,0], [1,0,0], [0,1,0], [1,1,0]], dtype=float)),
                    self.enumerate_cliques_sparse(np.array([1, 2, 3], dtype=np.int64), np.array([1, 0], dtype=np.int64), 2, 2),
                    self.compute_circumradius_sq_2d(np.array([[0,0], [1,0], [0,1]], dtype=float), np.array([[0, 1, 2]], dtype=np.int64)),
                    self.compute_circumradius_sq_3d(np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=float), np.array([[0, 1, 2, 3]], dtype=np.int64)),
                    self.compute_cknn_graph_accelerated(np.array([[0,0], [1,1]], dtype=float), np.array([1.0, 1.0]), 1.0),
                    self.quick_mapper_topology_jl([(0, 1), (1, 2), (0, 2)], 1, 0.001)
                ),
            ),
        ]

    def compute_normal_surface_residual_norms(
        self,
        matrix,
        coordinate_matrix: np.ndarray,
    ) -> np.ndarray:
        """Compute `||A * x_i||_2` in batch for normal-surface coordinate columns."""
        self.require_julia()
        coords = np.asarray(coordinate_matrix, dtype=np.int64)
        if coords.ndim != 2:
            raise ValueError("coordinate_matrix must be a 2D array")
        if matrix is None:
            raise ValueError("matrix is required")
        if int(matrix.shape[1]) != int(coords.shape[0]):
            raise ValueError(
                "coordinate_matrix row count must match matrix column count"
            )
        if coords.shape[1] == 0:
            return np.zeros(0, dtype=np.float64)
        if int(matrix.shape[0]) == 0:
            return np.zeros(coords.shape[1], dtype=np.float64)

        if matrix.nnz == 0:
            rows = np.array([], dtype=np.int64)
            cols = np.array([], dtype=np.int64)
            vals = np.array([], dtype=np.int64)
        else:
            rows, cols, vals = self._coo_triplets_cached(matrix)

        norms = self.backend.normal_surface_residual_norms(
            rows,
            cols,
            vals,
            int(matrix.shape[0]),
            int(matrix.shape[1]),
            coords,
        )
        return np.asarray(norms, dtype=np.float64)

    def compute_broad_phase_pairs(
        self,
        centroids: np.ndarray,
        radii: np.ndarray,
        *,
        tol: float,
    ) -> np.ndarray:
        """Compute candidate simplex index pairs `(i, j)` with `i < j` via Julia."""
        self.require_julia()
        ctr = np.asarray(centroids, dtype=np.float64)
        rad = np.asarray(radii, dtype=np.float64).reshape(-1)
        if ctr.ndim != 2:
            raise ValueError("centroids must be a 2D array")
        if rad.ndim != 1:
            raise ValueError("radii must be a 1D array")
        if ctr.shape[0] != rad.shape[0]:
            raise ValueError("centroids and radii must have the same number of rows")
        if ctr.shape[0] <= 1:
            return np.zeros((0, 2), dtype=np.int64)
        pairs = self.backend.embedding_broad_phase_pairs(ctr, rad, float(tol))
        out = np.asarray(pairs, dtype=np.int64)
        if out.size == 0:
            return np.zeros((0, 2), dtype=np.int64)
        return out.reshape(-1, 2)

    def _run_warmup(self, mode: str) -> dict:
        """Execute warm-up workloads and cache the resulting status report."""
        target_level = 2 if mode == "full" else 1
        with self._lock:
            if self._warmup_level >= target_level:
                report = dict(self._warmup_report)
                report["cached"] = True
                return report

            if not self._available or self.jl is None or self.backend is None:
                report = {
                    "mode": mode,
                    "available": False,
                    "completed": [],
                    "failed": {},
                    "cached": False,
                }
                self._warmup_report = report
                return report

            workloads = list(self._minimal_warmup_workloads())
            if mode == "full":
                workloads.extend(self._full_warmup_workloads())

            report = {
                "mode": mode,
                "available": True,
                "completed": [],
                "failed": {},
                "cached": False,
            }
            for name, workload in workloads:
                try:
                    workload()
                    report["completed"].append(name)
                except Exception as exc:
                    report["failed"][name] = repr(exc)

            # Explicitly delete trash variables generated during warmup to free memory
            workloads.clear()
            del workloads
            import gc
            gc.collect()

            self._warmup_level = max(self._warmup_level, target_level)
            self._warmup_report = report
            if report["failed"]:
                warnings.warn(
                    "Julia warm-up completed with partial failures; regular execution will still proceed. "
                    f"Failed workloads: {sorted(report['failed'].keys())}",
                    stacklevel=2,
                )
            return report

    def warmup(self) -> dict:
        """Fully warm up Julia-backed bridge paths (best-effort, non-fatal)."""
        if not self.available:
            return {
                "mode": "full",
                "available": False,
                "completed": [],
                "failed": {"initialize": self.error or "Julia unavailable"},
                "cached": False,
            }
        return self._run_warmup("full")

    def _ensure_initialized(self):
        """Initialize the bridge on first use."""
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            self._initialize()

    @property
    def available(self) -> bool:
        """Whether Julia backend is initialized and available for compute calls."""
        self._ensure_initialized()
        return self._available

    @available.setter
    def available(self, value: bool):
        """Test hook to override availability without reinitializing Julia."""
        # Allow tests to monkeypatch availability while preserving singleton API.
        self._initialized = True
        self._available = bool(value)

    def require_julia(self):
        """Raise a structured error when Julia backend is unavailable."""
        from pysurgery.core.exceptions import SurgeryError

        if not self.available:
            raise SurgeryError(
                f"High-performance exact algebra requires Julia: {self.error}"
            )

    def _coo_cache_key(self, matrix) -> tuple:
        """Build a stable cache key for sparse COO triplet conversion."""
        data_ptr = None
        try:
            data_ptr = int(matrix.data.__array_interface__["data"][0])
        except Exception:
            data_ptr = None
        return (
            id(matrix),
            int(matrix.shape[0]),
            int(matrix.shape[1]),
            int(matrix.nnz),
            data_ptr,
        )

    def _coo_triplets_cached(self, matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return cached COO triplets `(rows, cols, vals)` for a sparse matrix."""
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
        """Flatten ragged simplex lists into `(flat_vertices, offsets)` arrays."""
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
        """Compute the signature of a symmetric real matrix via Julia backend."""
        self.require_julia()
        # Direct zero-copy passing to Julia via PyArray
        return int(
            self.backend.hermitian_signature(np.asarray(matrix_array, dtype=np.float64))
        )

    def compute_sparse_snf(
        self, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, shape: tuple
    ) -> np.ndarray:
        """Executes the highly optimized Julia Sparse SNF backend."""
        self.require_julia()
        # Direct zero-copy passing of NumPy arrays
        factors = self.backend.exact_snf_sparse(
            np.asarray(rows, dtype=np.int64),
            np.asarray(cols, dtype=np.int64),
            np.asarray(vals, dtype=np.int64),
            int(shape[0]),
            int(shape[1]),
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

    def compute_sparse_cohomology_basis(
        self, d_np1, d_n, cn_size: int | None = None
    ) -> list:
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
            d_np1_rows,
            d_np1_cols,
            d_np1_vals,
            int(d_np1_m),
            int(d_np1_n),
            d_n_rows,
            d_n_cols,
            d_n_vals,
            int(d_n_m),
            int(d_n_n),
        )

        # Convert columns to list of vectors
        basis_py = []
        for j in range(basis_mat.shape[1]):
            basis_py.append(np.array(basis_mat[:, j], dtype=np.int64))
        return basis_py

    def compute_sparse_cohomology_basis_mod_p(
        self, d_np1, d_n, p: int, cn_size: int | None = None
    ) -> list:
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
            d_np1_rows,
            d_np1_cols,
            d_np1_vals,
            int(d_np1_m),
            int(d_np1_n),
            d_n_rows,
            d_n_cols,
            d_n_vals,
            int(d_n_m),
            int(d_n_n),
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

    def group_ring_multiply(
        self, coeffs1: dict, coeffs2: dict, group_order: int
    ) -> dict:
        """Multiply sparse group-ring coefficient dictionaries in the Julia backend."""
        self.require_julia()
        # Direct passing of dictionaries; Julia side now uses pyconvert for speed.
        res_keys, res_vals = self.backend.group_ring_multiply(
            coeffs1, coeffs2, int(group_order)
        )
        return {str(k): int(v) for k, v in zip(res_keys, res_vals)}

    def compute_multisignature(self, matrix: np.ndarray, p: int) -> int:
        """Evaluates L_{4k}(Z_p) obstruction by computing multisignature."""
        self.require_julia()
        return int(
            self.backend.multisignature(np.asarray(matrix, dtype=np.float64), int(p))
        )

    def integral_lattice_isometry(
        self, matrix1: np.ndarray, matrix2: np.ndarray
    ) -> np.ndarray | None:
        """Find U in GL_n(Z) with U^T * matrix1 * U = matrix2 for definite forms."""
        self.require_julia()
        candidate = self.backend.integral_lattice_isometry(
            np.asarray(matrix1, dtype=np.int64), np.asarray(matrix2, dtype=np.int64)
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
        pts = (
            np.asarray(point_cloud, dtype=np.float64)
            if point_cloud is not None
            else self.jl.nothing
        )
        mr = int(max_roots) if max_roots is not None else self.jl.nothing
        mc = int(max_cycles) if max_cycles is not None else self.jl.nothing
        out = self.backend.optgen_from_simplices(
            simplices, int(num_vertices), pts, mr, int(root_stride), mc
        )
        basis_py = []
        for g in out:
            # g is a dict: {"dimension", "support_simplices", "support_edges", "weight", "certified_cycle"}
            basis_py.append([tuple((int(e[0]), int(e[1]))) for e in g["support_edges"]])
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
        pts = (
            np.asarray(point_cloud, dtype=np.float64)
            if point_cloud is not None
            else self.jl.nothing
        )
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
            support_simplices = [
                tuple(int(x) for x in simplex) for simplex in g["support_simplices"]
            ]
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

    def compute_pi1_trace_candidates(
        self,
        d1_rows: np.ndarray,
        d1_cols: np.ndarray,
        d1_vals: np.ndarray,
        *,
        n_vertices: int,
        n_edges: int,
    ) -> list[dict]:
        """Compute raw pi1 generator trace candidates from d1 COO data via Julia."""
        self.require_julia()
        out = self.backend.pi1_trace_candidates_from_d1(
            np.asarray(d1_rows, dtype=np.int64),
            np.asarray(d1_cols, dtype=np.int64),
            np.asarray(d1_vals, dtype=np.int64),
            int(n_vertices),
            int(n_edges),
        )
        parsed: list[dict] = []
        for tr in out:
            parsed.append(
                {
                    "generator": str(tr["generator"]),
                    "edge_index": int(tr["edge_index"]),
                    "component_root": int(tr["component_root"]),
                    "vertex_path": [int(x) for x in tr["vertex_path"]],
                    "directed_edge_path": [
                        (int(e[0]), int(e[1])) for e in tr["directed_edge_path"]
                    ],
                    "undirected_edge_path": [
                        (int(e[0]), int(e[1])) for e in tr["undirected_edge_path"]
                    ],
                }
            )
        return parsed

    def compute_boundary_data_from_simplices(
        self, simplex_entries: list, max_dim: int
    ) -> tuple[dict, dict, dict, dict]:
        """Build boundary COO payloads and simplex tables through Julia for large simplicial workloads."""
        self.require_julia()
        result = self.backend.compute_boundary_data_from_simplices_jl(
            simplex_entries,
            int(max_dim),
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

        # dim_simplices_jl can return matrices (d+1, N) or lists of tuples
        dim_simplices_py = {}
        for k, val in dict(dim_simplices_jl).items():
            kk = int(k)
            if hasattr(val, "shape") and len(val.shape) == 2:
                # val is (d+1, N)
                dim_simplices_py[kk] = [
                    tuple(int(x) for x in val[:, j]) for j in range(val.shape[1])
                ]
            else:
                # val is a sequence of tuples/vectors
                dim_simplices_py[kk] = [
                    tuple(int(x) for x in s) for s in val
                ]

        simplex_to_idx_py = {
            int(k): {
                tuple(int(x) for x in simplex): int(idx)
                for simplex, idx in dict(idx_map).items()
            }
            for k, idx_map in dict(simplex_to_idx_jl).items()
            if len(dict(idx_map)) > 0
        }
        return boundaries_py, cells_py, dim_simplices_py, simplex_to_idx_py

    def compute_boundary_mod2_matrix(
        self, source_simplices: list, target_simplices: list
    ) -> dict:
        """Compute mod-2 boundary matrix through Julia for fast homology generator extraction."""
        self.require_julia()
        payload = self.backend.compute_boundary_mod2_matrix(
            source_simplices, target_simplices
        )
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

    def triangulate_surface_delaunay(
        self, points: np.ndarray, tolerance: float = 1e-10
    ) -> list:
        """Triangulate a 2D surface from a point cloud using Delaunay triangulation."""
        self.require_julia()
        triangles = self.backend.triangulate_surface_delaunay(
            np.asarray(points, dtype=np.float64), float(tolerance)
        )
        return [list(tri) for tri in triangles]

    def orthogonal_procrustes(self, A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        self.require_julia()
        res = self.backend.orthogonal_procrustes(
            np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)
        )
        # res is a Tuple: (R, B_aligned, disparity)
        return np.asarray(res[0]), np.asarray(res[1]), float(res[2])

    def pairwise_distance_matrix(self, data: np.ndarray, metric: str = "euclidean") -> np.ndarray:
        self.require_julia()
        res = self.backend.pairwise_distance_matrix(
            np.asarray(data, dtype=np.float64), str(metric)
        )
        return np.asarray(res)

    def frechet_distance(self, curve_a: np.ndarray, curve_b: np.ndarray) -> float:
        self.require_julia()
        res = self.backend.frechet_distance(
            np.asarray(curve_a, dtype=np.float64), np.asarray(curve_b, dtype=np.float64)
        )
        return float(res)

    def gromov_wasserstein_distance(
        self, D_A: np.ndarray, D_B: np.ndarray, p: np.ndarray, q: np.ndarray, epsilon: float, max_iter: int
    ) -> float:
        self.require_julia()
        res = self.backend.gromov_wasserstein_distance(
            np.asarray(D_A, dtype=np.float64),
            np.asarray(D_B, dtype=np.float64),
            np.asarray(p, dtype=np.float64),
            np.asarray(q, dtype=np.float64),
            float(epsilon),
            int(max_iter)
        )
        return float(res)


    def enumerate_cliques_sparse(self, rowptr: np.ndarray, colval: np.ndarray, n_vertices: int, max_dim: int) -> list:
        self.require_julia()
        try:
            res = self.backend.enumerate_cliques_sparse(
                rowptr,
                colval,
                int(n_vertices),
                int(max_dim)
            )
            return [[int(x) for x in c] for c in res]
        except Exception as e:
            raise RuntimeError(f"enumerate_cliques_sparse failed: {e!r}")

    def compute_circumradius_sq_3d(self, points: np.ndarray, simplices: np.ndarray) -> np.ndarray:
        if not self.available:
            raise RuntimeError("Julia backend unavailable.")
        try:
            res = self.backend.compute_circumradius_sq_3d(
                np.asarray(points, dtype=np.float64),
                np.asarray(simplices, dtype=np.int64) + 1 # 1-based for Julia
            )
            return np.array(res, dtype=np.float64)
        except Exception as e:
            raise RuntimeError(f"compute_circumradius_sq_3d failed: {e!r}")

    def compute_circumradius_sq_2d(self, points: np.ndarray, simplices: np.ndarray) -> np.ndarray:
        if not self.available:
            raise RuntimeError("Julia backend unavailable.")
        try:
            res = self.backend.compute_circumradius_sq_2d(
                np.asarray(points, dtype=np.float64),
                np.asarray(simplices, dtype=np.int64) + 1 # 1-based for Julia
            )
            return np.array(res, dtype=np.float64)
        except Exception as e:
            raise RuntimeError(f"compute_circumradius_sq_2d failed: {e!r}")

    def compute_alpha_complex_simplices(
        self,
        points: np.ndarray,
        max_simplices: np.ndarray,
        alpha2: float,
        max_dim: int,
    ) -> list[tuple[int, ...]]:
        """Compute filtered Alpha Complex simplices via Julia backend."""
        self.require_julia()
        res = self.backend.compute_alpha_complex_simplices_jl(
            np.asarray(points, dtype=np.float64),
            np.asarray(max_simplices, dtype=np.int64),
            float(alpha2),
            int(max_dim),
        )
        # res is Vector{Vector{Int64}} from Julia
        return [tuple(int(x) for x in s) for s in res]

    def compute_cknn_graph(self, pts: np.ndarray, k: int, delta: float) -> np.ndarray:
        """Compute the Continuous k-Nearest Neighbors graph using Julia."""
        self.require_julia()
        pts_view = np.asarray(pts, dtype=np.float64, order="C")
        try:
            with self._lock:
                out = self.backend.cknn_graph_jl(pts_view, int(k), float(delta))
                pairs = np.array(out, dtype=np.int64)
                if pairs.size == 0:
                    return np.zeros((0, 2), dtype=np.int64)
                return pairs.reshape(-1, 2)
        except Exception as e:
            raise RuntimeError(f"Julia cknn_graph failed: {e!r}")

    
    def compute_cknn_graph_accelerated(self, pts: np.ndarray, rho: np.ndarray, delta: float) -> np.ndarray:
        """Compute the CkNN graph using pre-computed rho values for speed."""
        self.require_julia()
        pts_view = np.asarray(pts, dtype=np.float64, order="C")
        rho_view = np.asarray(rho, dtype=np.float64)
        try:
            with self._lock:
                out = self.backend.cknn_graph_accelerated_jl(pts_view, rho_view, float(delta))
                pairs = np.array(out, dtype=np.int64)
                if pairs.size == 0:
                    return np.zeros((0, 2), dtype=np.int64)
                return pairs.reshape(-1, 2)
        except Exception as e:
            raise RuntimeError(f"Julia cknn_graph_accelerated failed: {e!r}")

    def quick_mapper_jl(self, G_raw: dict, max_loops: int = 1, min_modularity_gain: float = 1e-6) -> tuple[dict, dict]:
        """
        Executes the high-performance QuickMapper algorithm in Julia.
        G_raw must be a dict with keys "V" (list of ints) and "E" (list of tuples of ints).
        Returns a simplified graph dict and a mapping dictionary L.
        """
        self.require_julia()
        try:
            G_simple, L_jl = self.backend.quick_mapper_jl(
                G_raw,
                int(max_loops),
                float(min_modularity_gain)
            )
            L_py = dict(L_jl)
            G_simple_py = dict(G_simple)
            if "E" in G_simple_py:
                G_simple_py["E"] = [tuple(e) for e in G_simple_py["E"]]
            return G_simple_py, L_py
        except Exception as e:
            raise RuntimeError(f"quick_mapper_jl failed: {e!r}")

    def quick_mapper_topology_jl(self, simplices: list[tuple], max_loops: int = 1, min_modularity_gain: float = 1e-6) -> tuple[list[tuple], dict]:
        """
        Executes the high-performance topology-preserving QuickMapper algorithm in Julia.
        simplices must be a list of tuples representing all simplices in the complex.
        Returns a list of remaining simplices and a mapping dictionary L.
        """
        self.require_julia()
        try:
            simplices_jl, L_jl = self.backend.quick_mapper_topology_jl(
                simplices,
                int(max_loops),
                float(min_modularity_gain)
            )
            L_py = dict(L_jl)
            simplices_py = [tuple(s) for s in simplices_jl]
            return simplices_py, L_py
        except Exception as e:
            raise RuntimeError(f"quick_mapper_topology_jl failed: {e!r}")

    def is_homology_manifold_jl(self, simplices: list[list[int]], max_dim: int) -> tuple[bool, int, dict[int, str]]:
        """
        Accelerated manifold certification in Julia.
        Returns (is_manifold, dimension, diagnostics).
        """
        self.require_julia()
        try:
            is_manifold, dimension, diagnostics_jl = self.backend.is_homology_manifold_jl(
                simplices,
                int(max_dim)
            )
            # diagnostics_jl is a Dict{Int, String} from Julia
            diagnostics = {int(k): str(v) for k, v in dict(diagnostics_jl).items()}
            return bool(is_manifold), int(dimension), diagnostics
        except Exception as e:
            raise RuntimeError(f"is_homology_manifold_jl failed: {e!r}")

    # Singleton instance
julia_engine = JuliaBridge()
