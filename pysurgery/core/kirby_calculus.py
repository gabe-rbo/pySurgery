import numpy as np
from pydantic import BaseModel, ConfigDict
from .intersection_forms import IntersectionForm
from .exceptions import KirbyMoveError


class KirbyDiagram(BaseModel):
    """Represents a 3D/4D manifold as a framed link in S^3 via Kirby Calculus.

    Overview:
        A KirbyDiagram encodes the instructions for building a 4-dimensional 
        manifold (and its 3-dimensional boundary) by attaching 2-handles to 
        a 4-ball. The diagram consists of a framed link in S^3, where each 
        component represents the core of a handle attachment.

    Key Concepts:
        - **Framed Link**: A collection of knots with integer weights (framings).
        - **Handle Attachment**: Gluing a D^2 x D^2 along S^1 x D^2 to the boundary.
        - **Kirby Moves**: Moves (blow-up, handle slide) that change the diagram 
          but preserve the homeomorphism type of the resulting 3-manifold.
        - **Linking Matrix**: The matrix recording self-linking (framing) and 
          mutual linking numbers.

    Common Workflows:
        1. **Simplification** → Use `handle_slide()` to diagonalize the linking matrix.
        2. **Stabilization** → Use `blow_up()` to add hyperbolic or CP^2 factors.
        3. **Invariants** → Use `extract_intersection_form()` for 4-manifold classification.

    Attributes:
        framings (np.ndarray): Diagonal of the linking matrix.
        linking_matrix (np.ndarray): The symmetric linking numbers Lk(K_i, K_j).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    framings: np.ndarray  # Diagonal of the linking matrix
    linking_matrix: np.ndarray  # The symmetric linking numbers Lk(K_i, K_j)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray):
        """Creates a Kirby Diagram directly from an integer matrix.

        Args:
            matrix: An integer matrix representing the linking numbers and framings.

        Returns:
            A KirbyDiagram instance.

        Raises:
            KirbyMoveError: If the matrix is not symmetric.
        """
        mat = np.array(matrix, dtype=int)
        if not np.allclose(mat, mat.T):
            raise KirbyMoveError("A Kirby linking matrix must be symmetric.")
        return cls(framings=np.diag(mat).copy(), linking_matrix=mat.copy())

    def blow_up(self, sign: int = 1) -> "KirbyDiagram":
        """Performs a +1 or -1 blow-up on the Kirby diagram.

        What is Being Computed?:
            The Kirby diagram of the manifold after taking the connected sum 
            with CP^2 (sign=+1) or -CP^2 (sign=-1).

        Algorithm:
            1. Validate the sign is ±1.
            2. Append a new isolated row and column to the linking matrix.
            3. Set the new diagonal entry to the specified sign.

        Preserved Invariants:
            - Preserves the homotopy type of the resulting 3-manifold (boundary).
            - Changes the signature of the 4-manifold by exactly the sign.

        Args:
            sign: The sign of the blow-up, must be 1 or -1. Defaults to 1.

        Returns:
            KirbyDiagram: A new diagram instance with the added component.

        Raises:
            KirbyMoveError: If sign is not 1 or -1.

        Use When:
            - You need to stabilize the intersection form.
            - Changing the parity or signature of a 4-manifold diagram.

        Example:
            # Add a -1 framed unknot to the diagram
            blown_up = diagram.blow_up(sign=-1)
        """
        if sign not in (1, -1):
            raise KirbyMoveError("Blow-up sign must be strictly +1 or -1.")

        n = self.linking_matrix.shape[0]
        new_mat = np.zeros((n + 1, n + 1), dtype=int)
        new_mat[:n, :n] = self.linking_matrix
        new_mat[n, n] = sign

        return KirbyDiagram.from_matrix(new_mat)

    def handle_slide(self, source_idx: int, target_idx: int) -> "KirbyDiagram":
        """Performs a handle slide of one link component over another.

        What is Being Computed?:
            The updated linking matrix after sliding handle K_{source} over K_{target}. 
            Algebraically, this corresponds to a basis change in the second 
            homology of the 4-manifold.

        Algorithm:
            1. Construct an elementary basis change matrix P where P[s, t] = 1.
            2. Compute the new linking matrix L' = P * L * P^T.
            3. Geometrically, this updates the source framing: 
               f_s -> f_s + f_t + 2 * lk(K_s, K_t).

        Preserved Invariants:
            - Homeomorphism type of the 4-manifold — Unchanged.
            - Homeomorphism type of the 3-manifold boundary — Unchanged.
            - Intersection form (up to isomorphism) — Unchanged.

        Args:
            source_idx: Index of the knot being slid.
            target_idx: Index of the knot being slid over.

        Returns:
            KirbyDiagram: The updated diagram after the move.

        Raises:
            KirbyMoveError: If indices are out of bounds or source_idx equals target_idx.

        Use When:
            - Diagonalizing the linking matrix for homology computation.
            - Simplifying the diagram towards a standard form.

        Example:
            # Slide handle 0 over handle 1
            diagram_after = diagram.handle_slide(0, 1)
        """
        n = self.linking_matrix.shape[0]
        if not (0 <= source_idx < n) or not (0 <= target_idx < n):
            raise KirbyMoveError(
                f"Handle indices out of bounds. Valid indices: 0 to {n - 1}."
            )

        if source_idx == target_idx:
            raise KirbyMoveError("Cannot slide a handle over itself.")

        # The algebraic realization of a handle slide is a change of basis matrix P
        # with e_source -> e_source + e_target.
        P = np.eye(n, dtype=int)
        P[source_idx, target_idx] = 1

        new_mat = P @ self.linking_matrix @ P.T

        return KirbyDiagram.from_matrix(new_mat)

    def extract_intersection_form(self) -> IntersectionForm:
        """Extracts the intersection form of the resulting 4-manifold.

        The symmetric linking matrix of a Kirby diagram defines the
        intersection form exactly.

        Returns:
            The IntersectionForm of the manifold.
        """
        return IntersectionForm(matrix=self.linking_matrix, dimension=4)
