import numpy as np
from typing import Dict
from pydantic import BaseModel, ConfigDict, Field
from .core.complexes import ChainComplex
from .core.exceptions import DimensionError

class AlgebraicPoincareComplex(BaseModel):
    """
    Representation of an Algebraic Poincare complex (C_*, psi).
    
    Attributes
    ----------
    chain_complex : ChainComplex
        The underlying chain complex C_*.
    fundamental_class : np.ndarray
        The fundamental class [X] in H_n(C).
    psi : Dict[int, np.ndarray]
        The higher-order diagonal map components representing the chain homotopy
        equivalence between C^* and C_{n-*}.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    chain_complex: ChainComplex
    fundamental_class: np.ndarray
    dimension: int
    psi: Dict[int, np.ndarray] = Field(default_factory=dict)

    def model_post_init(self, __context):
        self._validate_poincare_data()

    def _chain_group_size(self, k: int):
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
                raise DimensionError(f"psi_{k} must be a 2D matrix; got shape {psi_arr.shape}.")
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
        """
        Compute the dual chain complex C^* = Hom(C, Z).
        """
        # Transpose the boundary operators to get coboundary operators.
        # Store δ^n at key n+1 so that boundaries[k] means "map going into degree k-1"
        coboundaries = {n + 1: self.chain_complex.boundaries[n + 1].T.tocsr()
                        for n in self.chain_complex.dimensions
                        if (n + 1) in self.chain_complex.boundaries}
        return ChainComplex(
            boundaries=coboundaries,
            dimensions=self.chain_complex.dimensions,
            coefficient_ring=self.chain_complex.coefficient_ring,
        )

    def cap_product(self, cohomology_class: np.ndarray, k: int) -> np.ndarray:
        r"""
        Compute the cap product [X] \cap \alpha, which defines the map H^k(X) -> H_{n-k}(X).
        
        For a chain complex, this is implemented via the evaluation of the 
        higher-order diagonal map psi on the fundamental class and the cohomology class.
        
        Parameters
        ----------
        cohomology_class : np.ndarray
            The cohomology class alpha in H^k(X).
        k : int
            The dimension of the cohomology class.
            
        Returns
        -------
        homology_class : np.ndarray
            The homology class [X] \cap \alpha in H_{n-k}(X).
        """
        # In a concrete Algebraic Poincare Complex, psi_0: C^k -> C_{n-k} 
        # is the chain map inducing the Poincare duality isomorphism.
        if k not in self.psi:
            raise DimensionError(f"Diagonal map psi_{k} not defined for dimension {k}. "
                                 "Topological translation: The Algebraic Poincaré Complex lacks the higher-order chain map psi_{k}: C^k -> C_{n-k} needed to induce Poincaré Duality.")

        psi_k = np.asarray(self.psi[k])
        alpha = np.asarray(cohomology_class)
        if psi_k.ndim != 2:
            raise DimensionError(f"psi_{k} must be a 2D matrix; got shape {psi_k.shape}.")
        if alpha.ndim != 1:
            raise DimensionError(f"cohomology_class must be a 1D vector; got shape {alpha.shape}.")
        if psi_k.shape[1] != alpha.shape[0]:
            raise DimensionError(
                f"Dimension mismatch in cap product: psi_{k} has {psi_k.shape[1]} columns but cohomology_class has length {alpha.shape[0]}."
            )

        # Apply the chain map psi_k to the cohomology class.
        return psi_k @ alpha
