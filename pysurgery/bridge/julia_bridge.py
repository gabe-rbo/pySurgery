import os
import threading
import importlib.util
import warnings
from collections import OrderedDict
import numpy as np
import scipy.sparse as sp

HAS_JULIACALL = importlib.util.find_spec("juliacall") is not None


class JuliaBridge:
    """Zero-Copy Bridge to execute high-performance Julia algebraic topology operations.

    Replaces subprocess mocks with native memory sharing via `juliacall`.

    Attributes:
        _instance: The singleton instance of JuliaBridge.
        _lock: Reentrant lock for thread-safe initialization.
        _initialized: Whether the bridge has attempted initialization.
        _available: Whether the Julia backend is successfully loaded.
        error: Error message if initialization failed.
        jl: The Julia Main module.
        backend: The SurgeryBackend Julia module.
        _coo_cache: Cache for sparse COO triplets.
        _warmup_level: Level of warm-up performed (0: none, 1: minimal, 2: full).
        _warmup_report: Report of warm-up workload results.
    """

    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        """Return a process-wide singleton bridge instance.

        Returns:
            The JuliaBridge singleton instance.
        """
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
        """Initialize Julia runtime and load backend module lazily.

        Returns:
            None
        """
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
            # CRITICAL: Set JULIA_NUM_THREADS before juliacall is imported/initialized
            # to ensure multi-threaded Julia loops actually run in parallel.
            if "JULIA_NUM_THREADS" not in os.environ:
                os.environ["JULIA_NUM_THREADS"] = str(os.cpu_count() or 1)

            # CRITICAL FIX for Segmentation Faults:
            # When JULIA_NUM_THREADS > 1, juliacall requires explicit signal handling 
            # orchestration to prevent hard SIGSEGV crashes on multi-threaded executions.
            if "PYTHON_JULIACALL_HANDLE_SIGNALS" not in os.environ:
                os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

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
        """Best-effort automatic warm-up on first Julia initialization.

        Returns:
            None
        """
        mode = os.getenv("PYSURGERY_JULIA_WARMUP_MODE", "minimal").strip().lower()
        if mode in {"", "off", "0", "false", "none"}:
            return
        if mode not in {"minimal", "full"}:
            mode = "minimal"
        self._run_warmup(mode)

    def _minimal_warmup_workloads(self) -> list[tuple[str, callable]]:
        """Return a small workload set that compiles common topology paths.

        Returns:
            A list of (name, workload_callable) tuples for minimal warm-up.
        """
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
                    self.simplify_jl([(0, 1), (1, 2), (0, 2)]),
                    self.compute_cknn_graph(np.array([[0., 0.], [1., 1.]]), 1, 1.0)
                ),
            ),
        ]

    def _full_warmup_workloads(self) -> list[tuple[str, callable]]:
        """Return the extended warm-up workload set for all heavy kernels.

        Returns:
            A list of (name, workload_callable) tuples for full warm-up.
        """

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
                    self.compute_boundary_payload_from_simplices([(0, 1, 2)], 2),
                    self.compute_trimesh_boundary_data([(0, 1, 2)], 3)
                ),
            ),
            (
                "pi1_data_assembly",
                lambda: self.compute_pi1_raw_data(
                    np.array([0, 1], dtype=np.int64), np.array([0, 0], dtype=np.int64), np.array([-1, 1], dtype=np.int64),
                    2, 1,
                    np.array([0], dtype=np.int64), np.array([0], dtype=np.int64), np.array([1], dtype=np.int64),
                    1
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
                    self.simplify_jl([(0, 1), (1, 2), (0, 2)])
                ),
            ),
        ]

    def compute_normal_surface_residual_norms(
        self,
        matrix,
        coordinate_matrix: np.ndarray,
    ) -> np.ndarray:
        """Compute `||A * x_i||_2` in batch for normal-surface coordinate columns.

        Args:
            matrix: The sparse matrix A.
            coordinate_matrix: A 2D array where each column is a coordinate vector x_i.

        Returns:
            An array of norms for each column.

        Raises:
            ValueError: If inputs are invalid or incompatible.
        """
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
        """Compute candidate simplex index pairs `(i, j)` with `i < j` via Julia.

        Args:
            centroids: An (N, D) array of simplex centroids.
            radii: An (N,) array of simplex radii.
            tol: Tolerance for overlap detection.

        Returns:
            An (M, 2) array of candidate pairs.

        Raises:
            ValueError: If inputs are invalid or incompatible.
        """
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
        """Execute warm-up workloads and cache the resulting status report.

        Args:
            mode: Warm-up mode ('minimal' or 'full').

        Returns:
            A dictionary containing the warm-up status report.
        """
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
        """Fully warm up Julia-backed bridge paths (best-effort, non-fatal).

        Returns:
            A dictionary containing the warm-up status report.
        """
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
        """Initialize the bridge on first use.

        Returns:
            None
        """
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            self._initialize()

    @property
    def available(self) -> bool:
        """Whether Julia backend is initialized and available for compute calls.

        Returns:
            True if available, False otherwise.
        """
        self._ensure_initialized()
        return self._available

    @available.setter
    def available(self, value: bool):
        """Test hook to override availability without reinitializing Julia.

        Args:
            value: The availability status to set.
        """
        # Allow tests to monkeypatch availability while preserving singleton API.
        self._initialized = True
        self._available = bool(value)

    def require_julia(self):
        """Raise a structured error when Julia backend is unavailable.

        Raises:
            SurgeryError: If the Julia backend is unavailable.
        """
        from pysurgery.core.exceptions import SurgeryError

        if not self.available:
            raise SurgeryError(
                f"High-performance exact algebra requires Julia: {self.error}"
            )

    def _coo_cache_key(self, matrix) -> tuple:
        """Build a stable cache key for sparse COO triplet conversion.

        Args:
            matrix: The sparse matrix to build a key for.

        Returns:
            A tuple representing the cache key.
        """
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
        """Return cached COO triplets `(rows, cols, vals)` for a sparse matrix.

        Args:
            matrix: The sparse matrix to get triplets for.

        Returns:
            A tuple of (rows, cols, vals) arrays.
        """
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
        """Flatten ragged simplex lists into `(flat_vertices, offsets)` arrays.

        Args:
            simplices: A list of simplex tuples/lists.

        Returns:
            A tuple of (flat_vertices, offsets) arrays.
        """
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
        """Compute the signature of a symmetric real matrix via Julia backend.

        Args:
            matrix_array: The symmetric real matrix.

        Returns:
            The topological signature of the matrix.
        """
        self.require_julia()
        # Direct zero-copy passing to Julia via PyArray
        return int(
            self.backend.hermitian_signature(np.asarray(matrix_array, dtype=np.float64))
        )

    def compute_sparse_snf(
        self, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, shape: tuple
    ) -> np.ndarray:
        """Executes the highly optimized Julia Sparse SNF backend.

        Args:
            rows: Row indices of the sparse matrix.
            cols: Column indices of the sparse matrix.
            vals: Values of the sparse matrix.
            shape: Shape of the sparse matrix.

        Returns:
            An array containing the diagonal factors of the SNF.
        """
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
        """Compute matrix rank over Q using Julia backend from sparse COO data.

        Args:
            matrix: The sparse matrix.

        Returns:
            The rank of the matrix over the rationals.
        """
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
        """Compute matrix rank over Z/pZ using Julia backend from sparse COO data.

        Args:
            matrix: The sparse matrix.
            p: The prime modulus.

        Returns:
            The rank of the matrix over Z/pZ.
        """
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
        """Executes the exact Julia sparse cohomology basis extraction Z^n / B^n.

        Args:
            d_np1: The boundary matrix d_{n+1}.
            d_n: The boundary matrix d_n.
            cn_size: The size of the n-chain group.

        Returns:
            A list of vectors forming a basis for the cohomology group.
        """
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
        """Executes Julia sparse cohomology basis extraction over Z/pZ for prime p.

        Args:
            d_np1: The boundary matrix d_{n+1}.
            d_n: The boundary matrix d_n.
            p: The prime modulus.
            cn_size: The size of the n-chain group.

        Returns:
            A list of vectors forming a basis for the cohomology group over Z/pZ.
        """
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
        """Build boundary payloads through Julia with optional metadata suppression.

        Args:
            simplex_entries: A list of simplex tuples/lists.
            max_dim: The maximum dimension of simplices to consider.
            include_metadata: Whether to include metadata in the result.

        Returns:
            A tuple containing boundary matrices and related data.
        """
        self.require_julia()
        flat_vertices, simplex_offsets = self._flatten_simplices(simplex_entries)
        return self.backend.compute_boundary_payload_from_flat_simplices(
            flat_vertices,
            simplex_offsets,
            int(max_dim),
            bool(include_metadata),
        )

    def compute_boundary_payload_from_flat_simplices(
        self,
        flat_vertices: np.ndarray,
        simplex_offsets: np.ndarray,
        max_dim: int,
        *,
        include_metadata: bool = True,
    ) -> tuple:
        """Build boundary payloads through Julia using pre-flattened data.

        Args:
            flat_vertices: Flattened array of simplex vertices.
            simplex_offsets: Offsets into the flat_vertices array for each simplex.
            max_dim: The maximum dimension of simplices to consider.
            include_metadata: Whether to include metadata in the result.

        Returns:
            A tuple containing boundary matrices and related data.
        """
        self.require_julia()
        return self.backend.compute_boundary_payload_from_flat_simplices(
            np.asarray(flat_vertices, dtype=np.int64),
            np.asarray(simplex_offsets, dtype=np.int64),
            int(max_dim),
            bool(include_metadata),
        )

    def group_ring_multiply(
        self, coeffs1: dict, coeffs2: dict, group_order: int
    ) -> dict:
        """Multiply sparse group-ring coefficient dictionaries in the Julia backend.

        Args:
            coeffs1: The first coefficient dictionary.
            coeffs2: The second coefficient dictionary.
            group_order: The order of the group.

        Returns:
            The resulting coefficient dictionary after multiplication.
        """
        self.require_julia()
        # Direct passing of dictionaries; Julia side now uses pyconvert for speed.
        res_keys, res_vals = self.backend.group_ring_multiply(
            coeffs1, coeffs2, int(group_order)
        )
        return {("1" if int(k) == 0 else f"g_{int(k)}"): int(v) for k, v in zip(res_keys, res_vals)}

    def compute_multisignature(self, matrix: np.ndarray, p: int) -> int:
        """Evaluates L_{4k}(Z_p) obstruction by computing multisignature.

        Args:
            matrix: The symmetric real matrix.
            p: The prime modulus.

        Returns:
            The multisignature value.
        """
        self.require_julia()
        return int(
            self.backend.multisignature(np.asarray(matrix, dtype=np.float64), int(p))
        )

    def integral_lattice_isometry(
        self, matrix1: np.ndarray, matrix2: np.ndarray
    ) -> np.ndarray | None:
        """Find U in GL_n(Z) with U^T * matrix1 * U = matrix2 for definite forms.

        Args:
            matrix1: The first integral matrix.
            matrix2: The second integral matrix.

        Returns:
            The isometry matrix U if found, None otherwise.
        """
        self.require_julia()
        candidate = self.backend.integral_lattice_isometry(
            np.asarray(matrix1, dtype=np.int64), np.asarray(matrix2, dtype=np.int64)
        )
        if candidate is None:
            return None
        return np.array(candidate, dtype=np.int64)

    def abelianize_and_bhs_rank(self, generators: list, relations: list) -> tuple:
        """Computes the abelianization and extracts free and torsion ranks.

        Takes raw string generators and relations, computes the abelianization,
        and extracts the free and torsion ranks for the Bass-Heller-Swan K-theory formula.

        Args:
            generators: List of generator names.
            relations: List of relations.

        Returns:
            A tuple of (free_rank, torsion_ranks).
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
        """Compute an optimal H1 basis via Julia backend (Algorithms 8+7 composition).

        Args:
            simplices: A list of simplex tuples/lists.
            num_vertices: Number of vertices in the complex.
            point_cloud: Optional (N, D) array of vertex coordinates.
            max_roots: Maximum number of root vertices for cycle search.
            root_stride: Stride for selecting root vertices.
            max_cycles: Maximum number of cycles to extract.

        Returns:
            A list of representative cycles as lists of edges.
        """
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
        """Compute H_k generator representatives from simplices via Julia backend over Z/2.

        Args:
            simplices: A list of simplex tuples/lists.
            num_vertices: Number of vertices in the complex.
            dimension: The dimension k of the homology group H_k.
            mode: Search mode for generators ('valid', 'optimized', etc.).
            point_cloud: Optional (N, D) array of vertex coordinates.
            max_roots: Maximum number of root vertices for cycle search.
            root_stride: Stride for selecting root vertices.
            max_cycles: Maximum number of cycles to extract.

        Returns:
            A list of dictionaries representing homology generators.
        """
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
            mc
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

    def compute_pi1_raw_data(
        self,
        d1_rows: np.ndarray,
        d1_cols: np.ndarray,
        d1_vals: np.ndarray,
        n_vertices: int,
        n_edges: int,
        d2_rows: np.ndarray,
        d2_cols: np.ndarray,
        d2_vals: np.ndarray,
        n_faces: int,
    ) -> dict:
        """Compute full raw pi1 data (generators, relations, traces) via Julia.

        Args:
            d1_rows: Row indices of the d1 boundary matrix.
            d1_cols: Column indices of the d1 boundary matrix.
            d1_vals: Values of the d1 boundary matrix.
            n_vertices: Number of vertices.
            n_edges: Number of edges.
            d2_rows: Row indices of the d2 boundary matrix.
            d2_cols: Column indices of the d2 boundary matrix.
            d2_vals: Values of the d2 boundary matrix.
            n_faces: Number of faces (2-cells).

        Returns:
            A dictionary with 'generators', 'relations', and 'traces'.
        """
        self.require_julia()
        res = self.backend.extract_pi1_raw_data_jl(
            np.asarray(d1_rows, dtype=np.int64),
            np.asarray(d1_cols, dtype=np.int64),
            np.asarray(d1_vals, dtype=np.int64),
            int(n_vertices),
            int(n_edges),
            np.asarray(d2_rows, dtype=np.int64),
            np.asarray(d2_cols, dtype=np.int64),
            np.asarray(d2_vals, dtype=np.int64),
            int(n_faces),
        )
        
        # Parse traces
        parsed_traces = []
        for tr in res["traces"]:
            parsed_traces.append({
                "generator": str(tr["generator"]),
                "edge_index": int(tr["edge_index"]),
                "component_root": int(tr["component_root"]),
                "vertex_path": [int(x) for x in tr["vertex_path"]],
                "directed_edge_path": [(int(e[0]), int(e[1])) for e in tr["directed_edge_path"]],
                "undirected_edge_path": [(int(e[0]), int(e[1])) for e in tr["undirected_edge_path"]],
            })
            
        return {
            "generators": {int(k): str(v) for k, v in dict(res["generators"]).items()},
            "relations": [[str(t) for t in r] for r in res["relations"]],
            "traces": parsed_traces
        }

    def compute_pi1_trace_candidates(
        self,
        d1_rows: np.ndarray,
        d1_cols: np.ndarray,
        d1_vals: np.ndarray,
        *,
        n_vertices: int,
        n_edges: int,
    ) -> list[dict]:
        """Compute raw pi1 generator trace candidates from d1 COO data via Julia.

        Args:
            d1_rows: Row indices of the d1 boundary matrix.
            d1_cols: Column indices of the d1 boundary matrix.
            d1_vals: Values of the d1 boundary matrix.
            n_vertices: Number of vertices.
            n_edges: Number of edges.

        Returns:
            A list of dictionaries representing pi1 generator trace candidates.
        """
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
        """Build boundary COO payloads and simplex tables through Julia for large simplicial workloads.

        Args:
            simplex_entries: A list of simplex tuples/lists.
            max_dim: The maximum dimension of simplices to consider.

        Returns:
            A tuple of (boundaries, cells, dim_simplices, simplex_to_idx) dictionaries.
        """
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
        """Compute mod-2 boundary matrix through Julia for fast homology generator extraction.

        Args:
            source_simplices: List of source simplices.
            target_simplices: List of target simplices.

        Returns:
            A dictionary containing the sparse matrix payload.
        """
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
        """Compute Alexander-Whitney cup product through Julia for fast intersection form extraction.

        Args:
            alpha: Cochain coefficients for the first class.
            beta: Cochain coefficients for the second class.
            p: Dimension of the first class.
            q: Dimension of the second class.
            simplices_p_plus_q: List of (p+q)-dimensional simplices.
            simplex_to_idx_p: Mapping from p-simplices to their indices.
            simplex_to_idx_q: Mapping from q-simplices to their indices.
            modulus: Optional modulus for coefficients.

        Returns:
            An array containing the cup product coefficients.
        """
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
        """Compute trimesh boundary operators (d1, d2) through Julia.

        Args:
            faces: List of triangular faces.
            n_vertices: Number of vertices.

        Returns:
            A dictionary containing d1 and d2 sparse matrix payloads.
        """
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
        """Triangulate a 2D surface from a point cloud using Delaunay triangulation.

        Args:
            points: An (N, 2) array of point coordinates.
            tolerance: Geometric tolerance for triangulation.

        Returns:
            A list of triangle faces as vertex index lists.
        """
        self.require_julia()
        triangles = self.backend.triangulate_surface_delaunay(
            np.asarray(points, dtype=np.float64), float(tolerance)
        )
        return [list(tri) for tri in triangles]

    def orthogonal_procrustes(self, A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Solves the orthogonal Procrustes problem using Julia.

        Args:
            A: The first point cloud matrix.
            B: The second point cloud matrix.

        Returns:
            A tuple of (Rotation matrix R, aligned matrix B_aligned, disparity).
        """
        self.require_julia()
        res = self.backend.orthogonal_procrustes(
            np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)
        )
        # res is a Tuple: (R, B_aligned, disparity)
        return np.asarray(res[0]), np.asarray(res[1]), float(res[2])

    def pairwise_distance_matrix(self, data: np.ndarray, metric: str = "euclidean") -> np.ndarray:
        """Computes a pairwise distance matrix using Julia.

        Args:
            data: An (N, D) array of points.
            metric: The distance metric to use.

        Returns:
            An (N, N) pairwise distance matrix.
        """
        self.require_julia()
        res = self.backend.pairwise_distance_matrix(
            np.asarray(data, dtype=np.float64), str(metric)
        )
        return np.asarray(res)

    def frechet_distance(self, curve_a: np.ndarray, curve_b: np.ndarray) -> float:
        """Computes the discrete Frechet distance between two curves using Julia.

        Args:
            curve_a: An (N, D) array representing the first curve.
            curve_b: An (M, D) array representing the second curve.

        Returns:
            The discrete Frechet distance.
        """
        self.require_julia()
        res = self.backend.frechet_distance(
            np.asarray(curve_a, dtype=np.float64), np.asarray(curve_b, dtype=np.float64)
        )
        return float(res)

    def gromov_wasserstein_distance(
        self, D_A: np.ndarray, D_B: np.ndarray, p: np.ndarray, q: np.ndarray, epsilon: float, max_iter: int
    ) -> float:
        """Computes the entropic Gromov-Wasserstein distance using Julia.

        Args:
            D_A: Distance matrix of the first space.
            D_B: Distance matrix of the second space.
            p: Probability distribution over points in the first space.
            q: Probability distribution over points in the second space.
            epsilon: Regularization parameter.
            max_iter: Maximum number of Sinkhorn iterations.

        Returns:
            The entropic Gromov-Wasserstein distance.
        """
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
        """Enumerates cliques in a sparse graph using Julia.

        Args:
            rowptr: Row pointers of the sparse adjacency matrix.
            colval: Column values of the sparse adjacency matrix.
            n_vertices: Number of vertices in the graph.
            max_dim: Maximum dimension of cliques to enumerate.

        Returns:
            A list of cliques, where each clique is a list of vertex indices.

        Raises:
            RuntimeError: If the Julia call fails.
        """
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
        """Computes squared circumradii for 3D simplices using Julia.

        Args:
            points: An (N, 3) array of point coordinates.
            simplices: An (M, 4) array of tetrahedra vertex indices.

        Returns:
            An (M,) array of squared circumradii.

        Raises:
            RuntimeError: If the Julia backend is unavailable or the call fails.
        """
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
        """Computes squared circumradii for 2D simplices using Julia.

        Args:
            points: An (N, 2) array of point coordinates.
            simplices: An (M, 3) array of triangle vertex indices.

        Returns:
            An (M,) array of squared circumradii.

        Raises:
            RuntimeError: If the Julia backend is unavailable or the call fails.
        """
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
        """Compute filtered Alpha Complex simplices via Julia backend.

        Args:
            points: An (N, D) array of point coordinates.
            max_simplices: Array of maximal simplices (e.g., from Delaunay).
            alpha2: Squared alpha parameter.
            max_dim: Maximum dimension of simplices to return.

        Returns:
            A list of simplex tuples.
        """
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
        """Compute the Continuous k-Nearest Neighbors graph using Julia.

        Args:
            pts: An (N, D) array of point coordinates.
            k: Number of neighbors.
            delta: The delta parameter for CkNN.

        Returns:
            An (M, 2) array of edge index pairs.

        Raises:
            RuntimeError: If the Julia call fails.
        """
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
        """Compute the CkNN graph using pre-computed rho values for speed.

        Args:
            pts: An (N, D) array of point coordinates.
            rho: An (N,) array of pre-computed neighbor distances.
            delta: The delta parameter for CkNN.

        Returns:
            An (M, 2) array of edge index pairs.

        Raises:
            RuntimeError: If the Julia call fails.
        """
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
        """Executes the high-performance QuickMapper algorithm in Julia.

        Args:
            G_raw: A dict with keys "V" (list of ints) and "E" (list of tuples of ints).
            max_loops: Maximum number of simplification loops.
            min_modularity_gain: Minimum modularity gain to continue.

        Returns:
            A tuple of (simplified_graph_dict, mapping_dict_L).

        Raises:
            RuntimeError: If the Julia call fails.
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

    def simplify_jl(self, simplices: list[tuple]) -> tuple[list[tuple], dict[int, list[int]], dict[tuple, list[tuple]]]:
        """Executes high-performance topology-preserving simplification in Julia.

        Args:
            simplices: A list of tuples representing all simplices in the complex.

        Returns:
            A tuple of (remaining_simplices_list, new_to_original_vertex_map, new_to_original_simplex_map).
        """
        self.require_julia()
        try:
            simplices_jl, v_map_jl, s_map_jl = self.backend.simplify_jl(simplices)
            
            # v_map_jl is Dict{Int, Vector{Int}}
            v_map_py = {int(k): [int(v) for v in val] for k, val in dict(v_map_jl).items()}
            
            # s_map_jl is Dict{Tuple, Vector{Tuple}}
            s_map_py = {tuple(sorted(int(v) for v in k)): [tuple(sorted(int(v) for v in o)) for o in o_list]
                        for k, o_list in dict(s_map_jl).items()}
            
            return list(simplices_jl), v_map_py, s_map_py
        except Exception as e:
            raise RuntimeError(f"simplify_jl failed: {e!r}")
    def compute_alpha_threshold_emst(self, points: np.ndarray, simplices: np.ndarray) -> float:
        """Compute the EMST-based alpha threshold for a point cloud.

        Args:
            points: (N, D) array of point coordinates.
            simplices: (M, D+1) array of Delaunay simplices.

        Returns:
            The squared alpha threshold.
        """
        self.require_julia()
        try:
            return float(self.backend.compute_alpha_threshold_emst_jl(
                np.asarray(points, dtype=np.float64),
                np.asarray(simplices, dtype=np.int64)
            ))
        except Exception as e:
            raise RuntimeError(f"compute_alpha_threshold_emst failed: {e!r}")

    def compute_crust_simplices(self, points: np.ndarray, combined_simplices: np.ndarray, n_pts_orig: int) -> list[tuple[int, ...]]:
        """Extract Crust simplices via Julia backend.

        Args:
            points: Combined array of points and Voronoi vertices.
            combined_simplices: Delaunay simplices of combined set.
            n_pts_orig: Number of original points.

        Returns:
            A list of simplex tuples.
        """
        self.require_julia()
        try:
            res = self.backend.compute_crust_simplices_jl(
                np.asarray(points, dtype=np.float64),
                np.asarray(combined_simplices, dtype=np.int64),
                int(n_pts_orig)
            )
            return [tuple(sorted(int(v) for v in s)) for s in res]
        except Exception as e:
            raise RuntimeError(f"compute_crust_simplices failed: {e!r}")

    def compute_witness_complex_simplices(self, points: np.ndarray, landmarks_idx: np.ndarray, alpha: float, max_dim: int) -> list[tuple[int, ...]]:
        """Compute Witness Complex 1-skeleton via Julia backend.

        Args:
            points: (N, D) array of point coordinates.
            landmarks_idx: Indices of landmark points.
            alpha: Relaxation parameter.
            max_dim: Maximum dimension.

        Returns:
            A list of simplex tuples.
        """
        self.require_julia()
        try:
            res = self.backend.compute_witness_complex_simplices_jl(
                np.asarray(points, dtype=np.float64),
                np.asarray(landmarks_idx, dtype=np.int64),
                float(alpha),
                int(max_dim)
            )
            return [tuple(sorted(int(v) for v in s)) for s in res]
        except Exception as e:
            raise RuntimeError(f"compute_witness_complex_simplices failed: {e!r}")

    def compute_vietoris_rips(self, points: np.ndarray, epsilon: float, max_dim: int) -> list[tuple[int, ...]]:
        """Compute Vietoris-Rips complex via Julia backend.

        Args:
            points: (N, D) array of point coordinates.
            epsilon: Distance threshold.
            max_dim: Maximum dimension.

        Returns:
            A list of simplex tuples.
        """
        self.require_julia()
        try:
            res = self.backend.compute_vietoris_rips(
                np.asarray(points, dtype=np.float64),
                float(epsilon),
                int(max_dim)
            )
            return [tuple(sorted(int(v) for v in s)) for s in res]
        except Exception as e:
            raise RuntimeError(f"compute_vietoris_rips failed: {e!r}")

    def is_homology_manifold_jl(self, simplices: list[list[int]], max_dim: int) -> tuple[bool, int, dict[int, str]]:
        """Accelerated manifold certification in Julia.

        Args:
            simplices: A list of simplex vertex index lists.
            max_dim: The maximum dimension of simplices to consider.

        Returns:
            A tuple of (is_manifold, dimension, diagnostics_dict).

        Raises:
            RuntimeError: If the Julia call fails.
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

    def compute_discrete_morse_gradient_jl(self, simplices: list[list[int]]) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        """Accelerated Discrete Morse matching in Julia.

        Args:
            simplices: A list of simplex vertex index lists.

        Returns:
            A list of pairs ((sigma), (tau)) representing the gradient matching.
        """
        self.require_julia()
        try:
            res = self.backend.compute_discrete_morse_gradient_jl(simplices)
            # res is Vector{Vector{Vector{Int64}}} from Julia
            return [(tuple(sorted(int(x) for x in pair[0])), tuple(sorted(int(x) for x in pair[1]))) for pair in res]
        except Exception as e:
            raise RuntimeError(f"compute_discrete_morse_gradient_jl failed: {e!r}")

    # Singleton instance
julia_engine = JuliaBridge()
