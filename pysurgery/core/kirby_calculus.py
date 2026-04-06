import numpy as np
from pydantic import BaseModel, ConfigDict
from .intersection_forms import IntersectionForm
from .exceptions import KirbyMoveError

class KirbyDiagram(BaseModel):
    """
    Represents a 3D/4D manifold as a framed link in S^3 via Kirby Calculus.
    Each component of the link represents the attachment of a 2-handle.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    framings: np.ndarray # Diagonal of the linking matrix
    linking_matrix: np.ndarray # The symmetric linking numbers Lk(K_i, K_j)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray):
        """Creates a Kirby Diagram directly from an integer matrix."""
        mat = np.array(matrix, dtype=int)
        if not np.allclose(mat, mat.T):
            raise KirbyMoveError("A Kirby linking matrix must be symmetric.")
        return cls(framings=np.diag(mat).copy(), linking_matrix=mat.copy())

    def blow_up(self, sign: int = 1) -> 'KirbyDiagram':
        """
        Performs a +1 or -1 blow-up.
        This corresponds geometrically to taking the connected sum with CP^2 or -CP^2.
        It adds an isolated +1 or -1 framed unknot to the diagram.
        """
        if sign not in (1, -1):
            raise KirbyMoveError("Blow-up sign must be strictly +1 or -1.")
            
        n = self.linking_matrix.shape[0]
        new_mat = np.zeros((n+1, n+1), dtype=int)
        new_mat[:n, :n] = self.linking_matrix
        new_mat[n, n] = sign
        
        return KirbyDiagram.from_matrix(new_mat)

    def handle_slide(self, source_idx: int, target_idx: int) -> 'KirbyDiagram':
        """
        Performs a handle slide of K_{source} over K_{target}.
        This adds the target column (and row) to the source column (and row).
        
        Geometrically, the framing of the source knot changes by:
        f_s -> f_s + f_t + 2 * lk(K_s, K_t).
        """
        n = self.linking_matrix.shape[0]
        if not (0 <= source_idx < n) or not (0 <= target_idx < n):
            raise KirbyMoveError(f"Handle indices out of bounds. Valid indices: 0 to {n-1}.")
            
        if source_idx == target_idx:
            raise KirbyMoveError("Cannot slide a handle over itself.")
            
        # The algebraic realization of a handle slide is a change of basis matrix P.
        # If we slide handle source_idx over handle target_idx, we map the basis 
        # vector e_source to e_source + e_target.
        # The change of basis matrix is P = I + E_{target, source}.
        P = np.eye(n, dtype=int)
        P[target_idx, source_idx] = 1
        
        new_mat = P.T @ self.linking_matrix @ P
        
        return KirbyDiagram.from_matrix(new_mat)

    def extract_intersection_form(self) -> IntersectionForm:
        """
        The symmetric linking matrix of a Kirby diagram defines the 
        intersection form of the resulting 4-manifold exactly.
        """
        return IntersectionForm(matrix=self.linking_matrix, dimension=4)
