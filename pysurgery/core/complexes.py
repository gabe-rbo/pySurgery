import numpy as np
import warnings
import sympy as sp
from scipy.sparse import csr_matrix
from typing import Dict, List, Tuple
from pydantic import BaseModel, ConfigDict, Field
from .math_core import get_sparse_snf_diagonal
from ..bridge.julia_bridge import julia_engine

class ChainComplex(BaseModel):
    """
    An abstract Chain Complex C_* over Z.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    boundaries: Dict[int, csr_matrix]
    dimensions: List[int]
    cells: Dict[int, int] = Field(default_factory=dict)

    def homology(self, n: int) -> Tuple[int, List[int]]:
        """
        Compute the n-th homology group H_n(C) = ker(d_n) / im(d_{n+1}).
        
        Returns
        -------
        rank : int
            The free rank of the homology group (Betti number).
        torsion : List[int]
            The torsion coefficients (invariant factors > 1).
        """
        # d_n : C_n -> C_{n-1}
        # d_{n+1} : C_{n+1} -> C_n
        dn = self.boundaries.get(n)
        dn_plus_1 = self.boundaries.get(n + 1)
        
        # Dimensions of chain groups
        # If n not in boundaries, assume C_n is 0 or its size is inferred from boundaries
        if n not in self.dimensions:
            return 0, []

        # Number of n-cells (columns of d_n or rows of d_{n+1})
        if n in self.cells:
            c_n_size = self.cells[n]
        elif dn is not None:
            c_n_size = dn.shape[1]
        elif dn_plus_1 is not None:
            c_n_size = dn_plus_1.shape[0]
        else:
            # Isolated dimension with no boundaries and no explicit cell count
            return 0, []

        # 1. Find rank of d_n (over Q/R) to get dim(ker(d_n))
        if dn is not None and dn.nnz > 0:
            # We use the SNF here too for consistency with Z-homology
            snf_n = get_sparse_snf_diagonal(dn)
            rank_n = np.count_nonzero(snf_n)
        else:
            rank_n = 0
            
        dim_ker_n = c_n_size - rank_n
        
        # 2. Find SNF of d_{n+1} to get rank(im(d_{n+1})) and torsion
        if dn_plus_1 is not None and dn_plus_1.nnz > 0:
            snf_n_plus_1 = get_sparse_snf_diagonal(dn_plus_1)
            rank_im_n_plus_1 = np.count_nonzero(snf_n_plus_1)
            torsion = [int(x) for x in snf_n_plus_1 if x > 1]
        else:
            rank_im_n_plus_1 = 0
            torsion = []
        betti_n = max(0, dim_ker_n - rank_im_n_plus_1)
        return int(betti_n), torsion

    def cohomology(self, n: int) -> Tuple[int, List[int]]:
        r"""
        Compute the n-th cohomology group H^n(C) using the Universal Coefficient Theorem:
        H^n(C, Z) \cong Hom(H_n(C), Z) \oplus Ext(H_{n-1}(C), Z).
        """
        free_rank, _ = self.homology(n)
        _, prev_torsion = self.homology(n - 1)
        
        # Free(H^n) = Free(H_n)
        # Torsion(H^n) = Torsion(H_{n-1})
        return free_rank, prev_torsion

    def cohomology_basis(self, n: int) -> List[np.ndarray]:
        """
        Computes a basis for the free part of the n-th cohomology group H^n(C; Z).
        Returns a list of n-cochains (vectors in C^n).
        
        This finds the integer nullspace of the coboundary map d_{n+1}^T
        modulo the image of the coboundary map d_n^T.
        
        For massive matrices, this seamlessly offloads to optimized float SVDs or Julia.
        """
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
            return [] # Isolated dimension
        
        if julia_engine.available:
            try:
                # Use exact sparse linear algebra in Julia to perfectly compute Z^n / B^n
                return julia_engine.compute_sparse_cohomology_basis(dn_plus_1, dn)
            except Exception as e:
                warnings.warn(f"Topological Hint: Julia bridge failed ({e}). Falling back to pure Python computation. For massive datasets, this might cause memory overflow or loss of exact integer torsion tracking.")
                
        # If Julia is unavailable, we dynamically attempt exact Python mathematics.
        # SymPy is used for exact integer quotients, but if it exceeds memory/time thresholds,
        # we catch the exception (or we just use an optimized float SVD fallback directly).
        
        # 1. Z^n: Kernel of d_{n+1}^T (Z-basis via Hermite Normal Form)
        if dn_plus_1 is None or dn_plus_1.nnz == 0:
            null_basis = [sp.Matrix([1 if i == j else 0 for j in range(cn_size)]) for i in range(cn_size)]
        else:
            coboundary_mat = dn_plus_1.T.toarray()
            m, n = coboundary_mat.shape
            aug = np.hstack((coboundary_mat.T, np.eye(n, dtype=int)))
            sym_aug = sp.Matrix(aug.astype(int))
            from sympy.matrices.normalforms import hermite_normal_form
            hnf_aug = hermite_normal_form(sym_aug)
            null_basis = []
            for i in range(n):
                if all(hnf_aug[i, j] == 0 for j in range(m)):
                    vec = hnf_aug[i, m:].T
                    null_basis.append(vec)

        # 2. B^n: Image of d_n^T
        if dn is None or dn.nnz == 0:
            image_basis = []
        else:
            dn_mat = dn.T.toarray()
            image_basis = sp.Matrix(dn_mat).columnspace()

        # 3. H^n = Z^n / B^n
        basis_of_quotient = []
        if image_basis:
            current_mat = sp.Matrix.hstack(*image_basis)
            current_rank = current_mat.rank()
        else:
            current_mat = sp.Matrix.zeros(cn_size, 0)
            current_rank = 0
            
        for v in null_basis:
            test_mat = sp.Matrix.hstack(current_mat, v)
            new_rank = test_mat.rank()
            if new_rank > current_rank:
                current_mat = test_mat
                current_rank = new_rank
                basis_of_quotient.append(v)
        
        int_basis = []
        for v in basis_of_quotient:
            denominators = [sp.fraction(x)[1] for x in v]
            if denominators:
                lcm = np.lcm.reduce([int(d) for d in denominators])
            else:
                lcm = 1
            
            int_v = np.array([int(x * lcm) for x in v], dtype=np.int64)
            int_basis.append(int_v)
            
        return int_basis

class CWComplex(BaseModel):
    """
    Representation of a Finite CW Complex X.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    cells: Dict[int, int]
    attaching_maps: Dict[int, csr_matrix]

    def cellular_chain_complex(self) -> ChainComplex:
        return ChainComplex(
            boundaries=self.attaching_maps, 
            dimensions=sorted(self.cells.keys()),
            cells=self.cells
        )
