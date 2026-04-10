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

julia_engine = JuliaBridge()
