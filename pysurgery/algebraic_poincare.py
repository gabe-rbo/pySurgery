import numpy as np
from typing import Dict, Optional, Any, List, Tuple
from pydantic import BaseModel, ConfigDict, Field
from .core.complexes import ChainComplex
from .core.exceptions import DimensionError


class AlgebraicPoincareComplex(BaseModel):
    """An algebraic representation of a Poincaré complex (C_*, psi).

    Overview:
        An AlgebraicPoincareComplex represents a chain complex equipped with a chain 
        homotopy equivalence between its chains and cochains, modeling the duality 
        properties of a compact manifold without a choice of triangulation. It 
        encapsulates the data (C_*, psi) where psi represents the higher diagonal 
        maps involved in Poincaré duality.

    Key Concepts:
        - **Chain Complex (C_*)**: The underlying sequence of modules and boundaries.
        - **Fundamental Class ([X])**: A cycle in the top-dimensional homology H_n(C).
        - **Poincaré Duality**: The isomorphism H^k(X) ≅ H_{n-k}(X) induced by capping with [X].
        - **Psi (ψ)**: A collection of maps ψ_k: C^k → C_{n-k} that induce the duality isomorphism.

    Common Workflows:
        1. **Initialization** → Provide ChainComplex, fundamental_class, and psi maps.
        2. **Duality Analysis** → Use dual_complex() to get the cochain complex.
        3. **Product Computation** → Use cap_product() to evaluate duality on specific classes.

    Coefficient Ring:
        Inherited from the underlying ChainComplex (typically 'Z' or 'Q').

    Attributes:
        chain_complex (ChainComplex): The underlying chain complex C_*.
        fundamental_class (np.ndarray): The fundamental class [X] in H_n(C).
        dimension (int): The topological dimension n.
        psi (Dict[int, np.ndarray]): The higher-order diagonal map components.

    References:
        Ranicki, A. (1980). Exact sequences in the algebraic theory of surgery. 
        Princeton University Press.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chain_complex: ChainComplex
    fundamental_class: np.ndarray
    dimension: int
    psi: Dict[int, np.ndarray] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Perform post-initialization validation of Poincaré duality data.

        What is Being Computed?:
            Validates that the fundamental class is a cycle and that all psi matrices 
            have dimensions consistent with the chain groups.

        Algorithm:
            1. Flatten and validate fundamental_class dimensions.
            2. Verify [X] is a cycle: d_n([X]) = 0.
            3. Check each psi_k matrix for correct domain (C^k) and codomain (C_{n-k}).

        Args:
            __context: Pydantic initialization context.

        Raises:
            DimensionError: If any consistency check fails.
        """
        self._validate_poincare_data()

    def _chain_group_size(self, k: int) -> Optional[int]:
        """Determine the rank/dimension of the k-th chain group C_k.

        What is Being Computed?:
            Extracts the size of the module C_k from boundaries or explicit cell counts.

        Algorithm:
            1. Return 0 if k < 0.
            2. Check self.chain_complex.cells for explicit count.
            3. Infer from boundaries[k] (columns) or boundaries[k+1] (rows).

        Args:
            k: The dimension degree.

        Returns:
            Optional[int]: The number of generators in C_k, or None if unknown.
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
        """Internal mathematical consistency check for Poincaré data.

        What is Being Computed?:
            Verifies cycle condition and matrix compatibility.

        Algorithm:
            1. Ensure fundamental_class is 1D.
            2. Check d_n(fund) == 0.
            3. Verify psi_k dimensions against chain group sizes C_k and C_{n-k}.

        Preserved Invariants:
            - Cycle condition: ensures [X] defines a valid homology class.

        Raises:
            DimensionError: If invariants or dimensions are violated.
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
        """Construct the dual cochain complex C^* = Hom(C, Z).

        What is Being Computed?:
            Generates a new ChainComplex where boundary operators are the 
            transposes of the original, representing coboundaries.

        Algorithm:
            1. Iterate over all boundary matrices in the original complex.
            2. Transpose each matrix to obtain the coboundary operator.
            3. Return a new ChainComplex instance with these operators.

        Preserved Invariants:
            - Duality relationship: (C^*)^* is isomorphic to C.
            - Cohomology H^k(C) is computed as homology of the dual complex.

        Returns:
            ChainComplex: The dual complex representing the cochain structure.

        Example:
            cc_dual = poincare_complex.dual_complex()
            print(cc_dual.homology(1))  # Computes H^1(X)
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
        r"""Compute the cap product [X] ∩ α.

        What is Being Computed?:
            The cap product map - ∩ [X]: H^k(X) → H_{n-k}(X). This is the 
            fundamental operation of Poincaré duality.

        Algorithm:
            1. If self.psi[k] is defined, return psi[k] @ cohomology_class (Algebraic path).
            2. Otherwise, if 'simplices' is provided, evaluate via Alexander-Whitney diagonal (Geometric path):
               (σ ∩ α) = α(front k-face of σ) * (back (n-k)-face of σ).
            3. Sum contributions over the fundamental class chain.

        Preserved Invariants:
            - Poincaré Duality: For a valid Poincaré complex, this map induces an isomorphism on (co)homology.
            - Naturality: The cap product is natural with respect to maps of Poincaré complexes.

        Args:
            cohomology_class: The cohomology class α (as a vector) to cap with.
            k: The dimension degree of the cohomology class.
            simplices: Optional simplicial structure (needed if psi is missing).

        Returns:
            np.ndarray: The resulting chain in C_{n-k}.

        Use When:
            - Implementing Poincaré duality isomorphisms.
            - Computing intersection forms (via cap and cup product relationship).
            - Analyzing surgery obstructions.

        Example:
            alpha = np.array([1, 0, 0])  # H^1 generator
            chain = poincare.cap_product(alpha, k=1)
            # 'chain' is the dual cycle in C_{n-1}

        Raises:
            DimensionError: If neither psi nor simplicial data is sufficient.
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
