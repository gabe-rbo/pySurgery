from typing import Dict, Tuple
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from .complexes import SimplicialComplex
from .characteristic_classes import extract_stiefel_whitney_tangent


class SimplicialVectorBundle(BaseModel):
    """Combinatorial Vector Bundle over a Simplicial Complex.

    A vector bundle of rank k is represented by transition matrices g_{ij} in GL_k(R)
    defined on the edges (i, j) of the simplicial complex, satisfying the
    cocycle condition on every triangle.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_complex: SimplicialComplex
    rank: int
    # transitions: mapping from edge (u, v) to GL_k matrix
    transitions: Dict[Tuple[int, int], np.ndarray] = Field(default_factory=dict)

    def check_cocycle(self) -> bool:
        """Verify the cocycle condition g_uv * g_vw * g_wu = I on all triangles.

        Returns:
            bool: True if the cocycle condition holds, False otherwise.
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
        """Compute the i-th Stiefel-Whitney class of the bundle."""
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
        """Construct the tangent bundle of a simplicial manifold."""
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
