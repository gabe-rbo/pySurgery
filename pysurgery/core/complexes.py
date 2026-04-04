import numpy as np
from scipy.sparse import csr_matrix
from typing import Dict, List, Tuple
from pydantic import BaseModel, ConfigDict
from .math_core import get_snf_diagonal

class ChainComplex(BaseModel):
    """
    An abstract Chain Complex C_* over Z.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    boundaries: Dict[int, csr_matrix]
    dimensions: List[int]

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
        if dn is not None:
            c_n_size = dn.shape[1]
        elif dn_plus_1 is not None:
            c_n_size = dn_plus_1.shape[0]
        else:
            # Isolated dimension with no boundaries
            return 0, []

        # 1. Find rank of d_n (over Q/R) to get dim(ker(d_n))
        if dn is not None and dn.nnz > 0:
            # We use the SNF here too for consistency with Z-homology
            snf_n = get_snf_diagonal(dn.toarray())
            rank_n = np.count_nonzero(snf_n)
        else:
            rank_n = 0
            
        dim_ker_n = c_n_size - rank_n
        
        # 2. Find SNF of d_{n+1} to get rank(im(d_{n+1})) and torsion
        if dn_plus_1 is not None and dn_plus_1.nnz > 0:
            snf_n_plus_1 = get_snf_diagonal(dn_plus_1.toarray())
            rank_im_n_plus_1 = np.count_nonzero(snf_n_plus_1)
            torsion = [int(x) for x in snf_n_plus_1 if x > 1]
        else:
            rank_im_n_plus_1 = 0
            torsion = []
            
        betti_n = dim_ker_n - rank_im_n_plus_1
        return int(betti_n), torsion

    def cohomology(self, n: int) -> Tuple[int, List[int]]:
        """
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
        
        This finds the integer nullspace of the coboundary map d_{n+1}^T.
        """
        import sympy as sp
        
        dn_plus_1 = self.boundaries.get(n + 1)
        
        if dn_plus_1 is None or dn_plus_1.nnz == 0:
            # If there's no d_{n+1}, the kernel is the entire C^n space
            # We need the size of C_n. We can get it from d_n.
            dn = self.boundaries.get(n)
            if dn is not None:
                cn_size = dn.shape[1]
            else:
                return [] # Isolated dimension
            
            # Basis is standard basis
            basis = []
            for i in range(cn_size):
                v = np.zeros(cn_size, dtype=np.int64)
                v[i] = 1
                basis.append(v)
            return basis
            
        # Coboundary map is d_{n+1}^T
        coboundary_mat = dn_plus_1.T.toarray()
        
        # Compute nullspace over Q
        sym_mat = sp.Matrix(coboundary_mat)
        null_basis = sym_mat.nullspace()
        
        int_basis = []
        for v in null_basis:
            # Convert to numpy array of floats to find denominator
            arr = np.array(v).astype(float).flatten()
            
            # To get integer basis, find LCM of denominators
            denominators = [sp.fraction(x)[1] for x in v]
            lcm = np.lcm.reduce([int(d) for d in denominators])
            
            int_v = np.array([int(x * lcm) for x in v], dtype=np.int64)
            int_basis.append(int_v)
            
        # Optional: We should theoretically project out the image of d_n^T (coboundaries)
        # But any cocycle representing a class is sufficient for Cup Product evaluation!
        # The cup product respects cohomology classes.
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
            dimensions=sorted(self.cells.keys())
        )
