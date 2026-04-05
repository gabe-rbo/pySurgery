import numpy as np
from pydantic import BaseModel, ConfigDict
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

    def dual_complex(self) -> ChainComplex:
        """
        Compute the dual chain complex C^* = Hom(C, Z).
        """
        # Transpose the boundary operators to get coboundary operators.
        coboundaries = {n: self.chain_complex.boundaries[n].T for n in self.chain_complex.dimensions}
        return ChainComplex(boundaries=coboundaries, dimensions=self.chain_complex.dimensions)

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
        
        # Apply the chain map psi_k to the cohomology class.
        return self.psi[k] @ cohomology_class
