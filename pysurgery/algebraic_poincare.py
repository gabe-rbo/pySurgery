import numpy as np
from typing import Dict, Optional, Any, List, Tuple
from pydantic import BaseModel, ConfigDict, Field
from .core.complexes import ChainComplex
from .core.exceptions import DimensionError


class AlgebraicPoincareComplex(BaseModel):
    """Representation of an Algebraic Poincare complex (C_*, psi).

    References:
        Ranicki, A. (1980). Exact sequences in the algebraic theory of surgery. 
        Princeton University Press.

    Attributes:
        chain_complex: The underlying chain complex C_*.
        fundamental_class: The fundamental class [X] in H_n(C).
        dimension: The dimension of the complex.
        psi: The higher-order diagonal map components representing the chain homotopy
            equivalence between C^* and C_{n-*}.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chain_complex: ChainComplex
    fundamental_class: np.ndarray
    dimension: int
    psi: Dict[int, np.ndarray] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Validate Poincare data after initialization.

        Args:
            __context: Initialization context.
        """
        self._validate_poincare_data()

    def _chain_group_size(self, k: int) -> Optional[int]:
        """Get the size of the chain group C_k.

        Args:
            k: The degree.

        Returns:
            Optional[int]: The size of C_k, or None if not determinable.
        """
        if k < 0:
            return 0
        if k in self.chain_complex.cells:
            return int(self.chain_complex.cells[k])
        d_k = self.chain_complex.boundaries.get(k)
        if d_k is not None:
            return int(d_k.shape[1])
        d_kp1 = self.chain_complex.boundaries.get(k + 1)
        if d_kp1 is not None:
            return int(d_kp1.shape[0])
        return None

    def _validate_poincare_data(self) -> None:
        """Internal validation for Poincare complex data.

        Raises:
            DimensionError: If data dimensions are inconsistent.
        """
        fund = np.asarray(self.fundamental_class).flatten()
        if fund.ndim != 1:
            raise DimensionError("fundamental_class must be a 1D chain vector.")

        top_dim = self.dimension
        c_top = self._chain_group_size(top_dim)
        if c_top is not None and len(fund) != c_top:
            raise DimensionError(
                f"fundamental_class length {len(fund)} must match C_{top_dim} size {c_top}."
            )

        d_top = self.chain_complex.boundaries.get(top_dim)
        if d_top is not None and len(fund) == d_top.shape[1]:
            boundary = d_top @ fund
            if np.any(boundary != 0):
                raise DimensionError("fundamental_class must be a cycle: d_n([X]) = 0.")

        for k, psi_k in self.psi.items():
            psi_arr = np.asarray(psi_k)
            if psi_arr.ndim != 2:
                raise DimensionError(
                    f"psi_{k} must be a 2D matrix; got shape {psi_arr.shape}."
                )
            c_k = self._chain_group_size(k)
            c_nk = self._chain_group_size(self.dimension - k)
            if c_k is not None and psi_arr.shape[1] != c_k:
                raise DimensionError(
                    f"psi_{k} domain mismatch: expected C^{k} size {c_k}, got {psi_arr.shape[1]} columns."
                )
            if c_nk is not None and psi_arr.shape[0] != c_nk:
                raise DimensionError(
                    f"psi_{k} codomain mismatch: expected C_{{n-k}} size {c_nk}, got {psi_arr.shape[0]} rows."
                )

    def dual_complex(self) -> ChainComplex:
        """Compute the dual chain complex C^* = Hom(C, Z).

        Returns:
            ChainComplex: The dual ChainComplex instance.
        """
        # Transpose the boundary operators to get coboundary operators.
        # Store δ^n at key n+1 so that boundaries[k] means "map going into degree k-1"
        coboundaries = {
            n: self.chain_complex.boundaries[n].T.tocsr()
            for n in self.chain_complex.boundaries.keys()
        }
        return ChainComplex(
            boundaries=coboundaries,
            dimensions=self.chain_complex.dimensions,
            coefficient_ring=self.chain_complex.coefficient_ring,
        )

    def cap_product(
        self,
        cohomology_class: np.ndarray,
        k: int,
        simplices: Optional[Dict[int, List[Tuple[int, ...]]]] = None,
    ) -> np.ndarray:
        r"""Compute the cap product [X] \cap \alpha.

        If self.psi[k] is available, applies it directly. 
        Otherwise, if 'simplices' (the simplicial structure) is provided,
        it evaluates the cap product using the Alexander-Whitney diagonal:
        (sigma \cap alpha) = alpha(front k-face of sigma) * (back (n-k)-face of sigma).

        Args:
            cohomology_class: The cohomology class alpha to cap with.
            k: The degree of the cohomology class.
            simplices: Optional simplicial structure for AW diagonal evaluation.

        Returns:
            np.ndarray: The resulting chain in C_{n-k}.

        Raises:
            DimensionError: If dimensions are inconsistent or required data is missing.
        """
        n = self.dimension
        alpha = np.asarray(cohomology_class)
        
        # 1. Preferred path: supplies psi mapping
        if k in self.psi:
            psi_k = np.asarray(self.psi[k])
            if psi_k.shape[1] != alpha.shape[0]:
                 raise DimensionError(
                    f"Dimension mismatch in cap product: psi_{k} has {psi_k.shape[1]} columns but cohomology_class has length {alpha.shape[0]}."
                )
            return psi_k @ alpha

        # 2. Geometric path: Alexander-Whitney diagonal on simplices
        if simplices is not None:
            # We assume [X] is given as a chain in C_n
            # result is a chain in C_{n-k}
            target_dim = n - k
            if target_dim < 0:
                raise DimensionError(f"Cannot compute cap product for cohomology degree {k} in dimension {n}.")
            
            c_nk_size = self._chain_group_size(target_dim)
            if c_nk_size is None:
                raise DimensionError(f"Chain group C_{target_dim} not well-defined in underlying complex.")
            
            result = np.zeros(c_nk_size, dtype=self.chain_complex.boundaries[1].dtype if self.chain_complex.boundaries else np.int64)
            
            n_simplices = simplices.get(n, [])
            nk_simplices = simplices.get(target_dim, [])
            k_simplices = simplices.get(k, [])
            
            if not n_simplices or not k_simplices or not nk_simplices:
                return result
                
            k_map = {s: idx for idx, s in enumerate(k_simplices)}
            nk_map = {s: idx for idx, s in enumerate(nk_simplices)}
            
            # evaluate alpha on [X]
            for s_idx, coeff in enumerate(self.fundamental_class):
                if coeff == 0:
                    continue
                sigma = n_simplices[s_idx]
                
                # AW: front k-face, back (n-k)-face
                front = sigma[:k+1]
                back = sigma[k:]
                
                if front in k_map and back in nk_map:
                    val = alpha[k_map[front]]
                    result[nk_map[back]] += int(coeff) * int(val)
            
            return result

        raise DimensionError(
            f"Diagonal map psi_{k} not defined and no simplicial structure provided to evaluate Alexander-Whitney diagonal."
        )
