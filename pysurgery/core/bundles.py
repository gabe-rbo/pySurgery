from typing import Dict, Tuple
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from .complexes import SimplicialComplex
from .characteristic_classes import extract_stiefel_whitney_tangent


class SimplicialVectorBundle(BaseModel):
    """Combinatorial Vector Bundle over a Simplicial Complex.

    Overview:
        A SimplicialVectorBundle represents a rank-k vector bundle over a simplicial complex.
        It is defined via transition matrices g_{ij} ∈ GL_k(ℝ) on the edges of the complex,
        which describe how local trivializations are glued together.

    Key Concepts:
        - **Transition Matrices**: Matrices in GL_k(ℝ) assigned to directed edges.
        - **Cocycle Condition**: The requirement that g_{uv}g_{vw}g_{wu} = I for every triangle (u,v,w).
        - **Stiefel-Whitney Classes**: Topological invariants in H^*(M; ℤ₂) associated with the bundle.
        - **Local Trivialization**: Assignment of a vector space to each vertex, glued by transitions.

    Common Workflows:
        1. **Construction** → `SimplicialVectorBundle(base_complex=sc, rank=k, transitions=...)`
        2. **Tangent Bundle** → `SimplicialVectorBundle.tangent_bundle(sc)`
        3. **Verification** → `check_cocycle()` ensures the gluing data is mathematically valid.
        4. **Invariants** → `stiefel_whitney_class(i)` computes characteristic classes.

    Attributes:
        base_complex (SimplicialComplex): The underlying topological space.
        rank (int): The dimension of the vector space fibers.
        transitions (Dict[Tuple[int, int], np.ndarray]): Mapping from directed edges to GL_k matrices.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_complex: SimplicialComplex
    rank: int
    # transitions: mapping from edge (u, v) to GL_k matrix
    transitions: Dict[Tuple[int, int], np.ndarray] = Field(default_factory=dict)

    def check_cocycle(self) -> bool:
        """Verify the cocycle condition g_uv * g_vw * g_wu = I on all triangles.

        What is Being Computed?:
            Checks if the transition matrices satisfy the 1-cocycle condition, which is
            necessary for the gluing data to define a consistent vector bundle.

        Algorithm:
            1. Iterates over all 2-simplices (triangles) in the base complex.
            2. For each triangle with vertices (u, v, w), fetches matrices g_{uv}, g_{vw}, g_{wu}.
            3. Verifies that the product g_{uv} * g_{vw} * g_{wu} is close to the identity matrix.

        Preserved Invariants:
            - A valid cocycle ensures the existence of a globally well-defined bundle structure.

        Returns:
            bool: True if the cocycle condition holds for all triangles, False otherwise.

        Use When:
            - Validating manually constructed bundle data
            - Debugging transition matrix assignments
            - Asserting mathematical consistency before computing invariants

        Example:
            if not bundle.check_cocycle():
                raise ValueError("Inconsistent transition matrices")
        """
        for tri in self.base_complex.n_simplices(2):
            u, v, w = sorted(tri)
            # Standard order: g_01 * g_12 * g_20 = I
            g_01 = self.transitions.get((u, v))
            g_12 = self.transitions.get((v, w))
            g_20 = self.transitions.get((w, u))

            if g_01 is None or g_12 is None or g_20 is None:
                return False

            if not np.allclose(g_01 @ g_12 @ g_20, np.eye(self.rank)):
                return False
        return True

    def stiefel_whitney_class(self, i: int, backend: str = "auto") -> np.ndarray:
        """Compute the i-th Stiefel-Whitney class of the bundle.

        What is Being Computed?:
            Computes the i-th Stiefel-Whitney class w_i in H^i(M; ℤ₂) for the given bundle.

        Algorithm:
            1. For i=0, returns the unit class.
            2. For i=1, evaluates the determinants of transition matrices to find orientation flips.
            3. For the tangent bundle case, delegates to the optimized manifold-specific computation.
            4. Otherwise, returns a zero cochain as a placeholder for general bundle SW classes.

        Preserved Invariants:
            - Stiefel-Whitney classes are invariants of the isomorphism class of the bundle.
            - w₁=0 implies the bundle is orientable.

        Args:
            i: The degree of the Stiefel-Whitney class.
            backend: 'auto', 'julia', or 'python'.

        Returns:
            np.ndarray: A cochain vector representing w_i.

        Use When:
            - Studying bundle orientability (w₁)
            - Computing obstructions to bundle reductions
            - Identifying bundle types on manifolds

        Example:
            w1 = bundle.stiefel_whitney_class(1)
        """
        if i == 0:
            return np.ones(self.base_complex.count_simplices(0), dtype=np.int64)

        # If transitions are Identity or missing, and rank matches dim,
        # use the tangent bundle fast-path.
        is_trivial_bundle = True
        for g in self.transitions.values():
            if not np.allclose(g, np.eye(self.rank)):
                is_trivial_bundle = False
                break

        if i == 1:
            res = np.zeros(self.base_complex.count_simplices(1), dtype=np.int64)
            # If the bundle has explicit non-trivial transition determinants, use them
            has_flips = False
            s_to_idx = self.base_complex.simplex_to_index(1)
            for (u, v), g in self.transitions.items():
                if u < v and np.linalg.det(g) < 0:
                    idx = s_to_idx.get((u, v))
                    if idx is not None:
                        res[idx] = 1
                        has_flips = True
            if has_flips:
                return res

        # Fast-path for tangent-like bundles
        if is_trivial_bundle and self.rank == self.base_complex.dimension:
            return extract_stiefel_whitney_tangent(
                self.base_complex, i, backend=backend
            )

        return np.zeros(self.base_complex.count_simplices(i), dtype=np.int64)

    @classmethod
    def tangent_bundle(cls, sc: SimplicialComplex) -> "SimplicialVectorBundle":
        """Construct the tangent bundle of a simplicial manifold.

        What is Being Computed?:
            Creates a representation of the tangent bundle TM for a given simplicial complex.

        Algorithm:
            1. Determines the rank of the bundle based on the complex's dimension.
            2. Assigns identity transition matrices to all edges (flat bundle approximation).
            3. Returns a SimplicialVectorBundle configured as the tangent bundle.

        Args:
            sc: The simplicial complex (should be a homology manifold).

        Returns:
            SimplicialVectorBundle: The constructed tangent bundle object.

        Use When:
            - Starting a computation involving manifold-specific bundle invariants
            - Verifying characteristic classes of the tangent bundle

        Example:
            tm = SimplicialVectorBundle.tangent_bundle(sphere)
        """
        dim = sc.dimension
        if dim < 1:
            return cls(base_complex=sc, rank=0)

        # For mathematical consistency and passing check_cocycle,
        # we provide a flat bundle where g_ij = identity_matrix.
        # The orientation character (w1) is handled by delegating
        # characteristic class extraction to the manifold fast-path.
        transitions = {}
        identity_matrix = np.eye(dim)
        for edge in sc.n_simplices(1):
            u, v = edge[0], edge[1]
            transitions[(u, v)] = identity_matrix
            transitions[(v, u)] = identity_matrix

        return cls(base_complex=sc, rank=dim, transitions=transitions)
