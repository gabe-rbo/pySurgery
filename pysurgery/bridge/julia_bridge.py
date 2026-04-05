import os
import numpy as np

try:
    from juliacall import Main as jl
    HAS_JULIACALL = True
except ImportError:
    HAS_JULIACALL = False

class JuliaBridge:
    """
    Zero-Copy Bridge to execute high-performance Julia algebraic topology operations.
    Replaces subprocess mocks with native memory sharing via `juliacall`.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(JuliaBridge, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.available = False
        if not HAS_JULIACALL:
            self.error = "juliacall is not installed. Install via `pip install juliacall`."
            return
            
        try:
            self.jl = jl
            backend_script = os.path.join(os.path.dirname(__file__), "surgery_backend.jl")
            self.jl.include(backend_script)
            self.backend = self.jl.SurgeryBackend
            self.available = True
        except Exception as e:
            self.error = f"Failed to initialize Julia backend: {e}"

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
        # Julia uses 1-based indexing for sparse matrices
        jl_rows = self.jl.Array(rows + 1)
        jl_cols = self.jl.Array(cols + 1)
        jl_vals = self.jl.Array(vals)
        factors = self.backend.exact_snf_sparse(jl_rows, jl_cols, jl_vals, shape[0], shape[1])
        return np.array(factors, dtype=np.int64)

    def compute_sparse_cohomology_basis(self, d_np1, d_n) -> list:
        """Executes the exact Julia sparse cohomology basis extraction Z^n / B^n."""
        self.require_julia()
        
        if d_np1 is None or d_np1.nnz == 0:
            d_np1_rows, d_np1_cols, d_np1_vals = np.array([]), np.array([]), np.array([])
            d_np1_m, d_np1_n = (d_n.shape[1], 0) if d_n is not None else (0, 0)
        else:
            d_np1_coo = d_np1.tocoo()
            d_np1_rows, d_np1_cols, d_np1_vals = d_np1_coo.row, d_np1_coo.col, d_np1_coo.data
            d_np1_m, d_np1_n = d_np1.shape
            
        if d_n is None or d_n.nnz == 0:
            d_n_rows, d_n_cols, d_n_vals = np.array([]), np.array([]), np.array([])
            d_n_m, d_n_n = (0, d_np1_m) if d_np1 is not None else (0, 0)
        else:
            d_n_coo = d_n.tocoo()
            d_n_rows, d_n_cols, d_n_vals = d_n_coo.row, d_n_coo.col, d_n_coo.data
            d_n_m, d_n_n = d_n.shape
            
        # 1-based indexing
        basis_jl = self.backend.exact_sparse_cohomology_basis(
            self.jl.Array(d_np1_rows + 1), self.jl.Array(d_np1_cols + 1), self.jl.Array(d_np1_vals), d_np1_m, d_np1_n,
            self.jl.Array(d_n_rows + 1), self.jl.Array(d_n_cols + 1), self.jl.Array(d_n_vals), d_n_m, d_n_n
        )
        
        # Parse output
        basis_py = []
        for vec in basis_jl:
            basis_py.append(np.array(vec, dtype=np.int64))
        return basis_py
        
    def group_ring_multiply(self, coeffs1: dict, coeffs2: dict, group_order: int) -> dict:
        self.require_julia()
        # We convert dicts to arrays to pass to Julia
        k1, v1 = list(coeffs1.keys()), list(coeffs1.values())
        k2, v2 = list(coeffs2.keys()), list(coeffs2.values())
        res_keys, res_vals = self.backend.group_ring_multiply(k1, v1, k2, v2, group_order)
        return {str(k): int(v) for k, v in zip(res_keys, res_vals)}
        
    def compute_multisignature(self, matrix: np.ndarray, p: int) -> int:
        """Evaluates L_{4k}(Z_p) obstruction by computing multisignature."""
        self.require_julia()
        return int(self.backend.multisignature(matrix, p))

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

julia_engine = JuliaBridge()
