"""Discrete surface uniformization tools.

This module implements a practical, exact-topology / numeric-geometry bridge for
closed or bordered triangulated surfaces.

Core ideas:
- represent a surface as a triangulated piecewise-flat metric,
- compute vertex Gaussian curvature via angle deficits,
- build the cotangent Laplacian from current triangle geometry,
- solve a discrete Ricci-flow / discrete conformal system by damped Newton steps,
- provide a circle-packing style variant as a combinatorial fallback.

The implementation is intentionally self-contained and uses only NumPy + SciPy
sparse linear algebra.

References
----------
- Chow & Luo, ``Combinatorial Ricci Flows on Surfaces``.
- Thurston, ``The Geometry and Topology of Three-Manifolds`` (circle packing ideas).
- Springborn, Schröder, Pinkall, discrete conformal equivalence / variational methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional, Sequence

import numpy as np
from scipy.sparse import csr_matrix

from .complexes import SimplicialComplex

_TWO_PI = 2.0 * math.pi


@dataclass
class SurfaceMesh:
    """Triangulated surface with a sparse combinatorial and metric representation.

    Attributes:
        faces: (m, 3) array of vertex indices for each triangle.
        n_vertices: Total number of vertices.
        base_edge_lengths: Initial lengths for each edge.
        edges: (e, 2) array of vertex pairs for each edge.
        edge_to_index: Mapping from sorted vertex pair to edge index.
        edge_faces: List of face indices incident to each edge.
        vertex_faces: List of face indices incident to each vertex.
        vertex_neighbors: List of sets of neighbor vertices for each vertex.
        boundary_vertices: Indices of vertices on the boundary.
        coordinates: Optional (n, d) array of vertex coordinates.
        simplicial_complex: Optional SimplicialComplex representation.
    """

    faces: np.ndarray
    n_vertices: int
    base_edge_lengths: np.ndarray
    edges: np.ndarray
    edge_to_index: dict[tuple[int, int], int]
    edge_faces: list[list[int]]
    vertex_faces: list[list[int]]
    vertex_neighbors: list[set[int]]
    boundary_vertices: np.ndarray
    coordinates: Optional[np.ndarray] = None
    simplicial_complex: Optional[SimplicialComplex] = None

    @classmethod
    def from_vertices_faces(
        cls,
        vertices: np.ndarray,
        faces: Sequence[Sequence[int]],
        *,
        validate: bool = True,
    ) -> "SurfaceMesh":
        """Create a SurfaceMesh from vertex coordinates and face indices.

        Args:
            vertices: (n, d) array of vertex coordinates.
            faces: Sequence of 3-tuples of vertex indices.
            validate: Whether to perform manifold validation.

        Returns:
            A new SurfaceMesh instance.

        Raises:
            ValueError: If inputs are malformed or invalid.
        """
        vertices_arr = np.asarray(vertices, dtype=np.float64)
        faces_arr = np.asarray(faces, dtype=np.int64)
        if faces_arr.ndim != 2 or faces_arr.shape[1] != 3:
            raise ValueError("faces must be an array of shape (m, 3)")
        if vertices_arr.ndim != 2 or vertices_arr.shape[0] == 0:
            raise ValueError("vertices must be a non-empty array of shape (n, d)")
        n_vertices = int(vertices_arr.shape[0])
        if np.min(faces_arr) < 0 or np.max(faces_arr) >= n_vertices:
            raise ValueError("faces reference out-of-range vertex indices")

        edges, edge_to_index, face_edges = _build_edges_from_faces(faces_arr)
        edge_faces, vertex_faces, vertex_neighbors, boundary_vertices = _build_incidence(
            n_vertices, faces_arr, edges, edge_to_index, face_edges
        )
        # Fix: build mesh instance first to pass it to validation (needed for link checks)
        sc = SimplicialComplex.from_maximal_simplices(map(tuple, faces_arr.tolist()))
        mesh = cls(
            faces=faces_arr,
            n_vertices=n_vertices,
            base_edge_lengths=_compute_base_edge_lengths(vertices_arr, edges),
            edges=edges,
            edge_to_index=edge_to_index,
            edge_faces=edge_faces,
            vertex_faces=vertex_faces,
            vertex_neighbors=vertex_neighbors,
            boundary_vertices=boundary_vertices,
            coordinates=vertices_arr,
            simplicial_complex=sc,
        )
        if validate:
            _validate_surface(mesh, faces_arr, edges, edge_faces)
        return mesh

    @classmethod
    def from_simplicial_complex(
        cls,
        complex_: SimplicialComplex,
        coordinates: Optional[np.ndarray] = None,
        *,
        validate: bool = True,
    ) -> "SurfaceMesh":
        """Create a SurfaceMesh from a SimplicialComplex.

        Args:
            complex_: The input simplicial complex.
            coordinates: Optional vertex coordinates.
            validate: Whether to perform manifold validation.

        Returns:
            A new SurfaceMesh instance.

        Raises:
            ValueError: If the complex is not a surface or coordinates are mismatched.
        """
        faces = np.asarray(complex_.n_simplices(2), dtype=np.int64)
        if faces.size == 0:
            raise ValueError("A surface triangulation requires at least one 2-simplex.")
        vertex_labels = [int(v[0]) for v in complex_.n_simplices(0)]
        if len(vertex_labels) == 0:
            raise ValueError("The simplicial complex must contain vertices.")
        # `SimplicialComplex` stores abstract vertex labels; reindex them densely.
        label_to_index = {label: idx for idx, label in enumerate(vertex_labels)}
        faces = np.array(
            [[label_to_index[int(v)] for v in face] for face in faces.tolist()], dtype=np.int64
        )
        if coordinates is not None:
            coords = np.asarray(coordinates, dtype=np.float64)
            if coords.shape[0] != len(vertex_labels):
                raise ValueError("coordinates must have one row per vertex in the simplicial complex")
        else:
            coords = None
        if coords is None:
            # Purely combinatorial metric: start with unit edge lengths.
            return cls._from_faces_only(faces, len(vertex_labels), validate=validate, complex_=complex_)
        return cls.from_vertices_faces(coords, faces, validate=validate)

    @classmethod
    def _from_faces_only(
        cls,
        faces: np.ndarray,
        n_vertices: int,
        *,
        validate: bool = True,
        complex_: Optional[SimplicialComplex] = None,
    ) -> "SurfaceMesh":
        """Create a SurfaceMesh from faces only (unit metric).

        Args:
            faces: (m, 3) array of vertex indices.
            n_vertices: Total number of vertices.
            validate: Whether to perform manifold validation.
            complex_: Optional SimplicialComplex.

        Returns:
            A new SurfaceMesh instance with unit edge lengths.
        """
        faces_arr = np.asarray(faces, dtype=np.int64)
        edges, edge_to_index, face_edges = _build_edges_from_faces(faces_arr)
        edge_faces, vertex_faces, vertex_neighbors, boundary_vertices = _build_incidence(
            n_vertices, faces_arr, edges, edge_to_index, face_edges
        )
        mesh = cls(
            faces=faces_arr,
            n_vertices=int(n_vertices),
            base_edge_lengths=np.ones(len(edges), dtype=np.float64),
            edges=edges,
            edge_to_index=edge_to_index,
            edge_faces=edge_faces,
            vertex_faces=vertex_faces,
            vertex_neighbors=vertex_neighbors,
            boundary_vertices=boundary_vertices,
            coordinates=None,
            simplicial_complex=complex_,
        )
        if validate:
            _validate_surface(mesh, faces_arr, edges, edge_faces)
        return mesh

    @property
    def num_faces(self) -> int:
        """Total number of faces."""
        return int(self.faces.shape[0])

    @property
    def num_edges(self) -> int:
        """Total number of edges."""
        return int(self.edges.shape[0])

    @property
    def euler_characteristic(self) -> int:
        """Euler characteristic of the surface (V - E + F)."""
        return int(self.n_vertices - self.num_edges + self.num_faces)

    @property
    def is_closed(self) -> bool:
        """True if the surface has no boundary."""
        return int(len(self.boundary_vertices)) == 0

    def target_geometry(self) -> str:
        """Determine the target geometry based on Euler characteristic.

        Returns:
            One of 'spherical', 'euclidean', or 'hyperbolic'.
        """
        chi = self.euler_characteristic
        if chi > 0:
            return "spherical"
        if chi == 0:
            return "euclidean"
        return "hyperbolic"

    def conformal_edge_lengths(self, u: np.ndarray, method: str = "ricci") -> np.ndarray:
        """Compute edge lengths after a conformal scaling.

        Args:
            u: (n_vertices,) array of log-conformal factors.
            method: Method for scaling ('ricci' or 'circle_packing').

        Returns:
            Array of scaled edge lengths.

        Raises:
            ValueError: If u has wrong shape or method is unknown.
        """
        u = np.asarray(u, dtype=np.float64)
        if u.shape != (self.n_vertices,):
            raise ValueError("u must have shape (n_vertices,)")
        a = self.edges[:, 0]
        b = self.edges[:, 1]
        base = self.base_edge_lengths
        if method == "circle_packing":
            scale = float(np.mean(base)) if len(base) else 1.0
            return scale * 0.5 * (np.exp(u[a]) + np.exp(u[b]))
        if method == "ricci":
            return base * np.exp(0.5 * (u[a] + u[b]))
        raise ValueError("Unknown conformal method: {!r}".format(method))

    def triangle_edge_lengths(
        self, u: Optional[np.ndarray] = None, method: str = "ricci"
    ) -> np.ndarray:
        """Get the edge lengths for each triangle in the mesh.

        Args:
            u: Optional log-conformal factors.
            method: Conformal scaling method.

        Returns:
            (num_faces, 3) array of edge lengths.
        """
        if u is None:
            u = np.zeros(self.n_vertices, dtype=np.float64)
        lengths = self.conformal_edge_lengths(u, method=method)
        tri_lengths = np.empty((self.num_faces, 3), dtype=np.float64)
        for f_idx, face in enumerate(self.faces):
            tri_lengths[f_idx, 0] = lengths[self.edge_to_index[_sorted_edge(face[1], face[2])]]
            tri_lengths[f_idx, 1] = lengths[self.edge_to_index[_sorted_edge(face[2], face[0])]]
            tri_lengths[f_idx, 2] = lengths[self.edge_to_index[_sorted_edge(face[0], face[1])]]
        return tri_lengths

    def triangle_angles(
        self, u: Optional[np.ndarray] = None, method: str = "ricci"
    ) -> np.ndarray:
        """Get the internal angles for each triangle in the mesh.

        Args:
            u: Optional log-conformal factors.
            method: Conformal scaling method.

        Returns:
            (num_faces, 3) array of triangle angles.
        """
        tri_lengths = self.triangle_edge_lengths(u=u, method=method)
        angles = np.empty((self.num_faces, 3), dtype=np.float64)
        for f_idx, (a, b, c) in enumerate(tri_lengths):
            angles[f_idx] = _triangle_angles_from_lengths(a, b, c)
        return angles

    def vertex_gaussian_curvature(
        self,
        u: Optional[np.ndarray] = None,
        method: str = "ricci",
    ) -> np.ndarray:
        """Compute the Gaussian curvature at each vertex.

        Args:
            u: Optional log-conformal factors.
            method: Conformal scaling method.

        Returns:
            (n_vertices,) array of vertex curvatures.
        """
        angles = self.triangle_angles(u=u, method=method)
        curvature = np.full(self.n_vertices, _TWO_PI, dtype=np.float64)
        if len(self.boundary_vertices) > 0:
            curvature[self.boundary_vertices] = math.pi
        for f_idx, face in enumerate(self.faces):
            curvature[face[0]] -= angles[f_idx, 0]
            curvature[face[1]] -= angles[f_idx, 1]
            curvature[face[2]] -= angles[f_idx, 2]
        return curvature

    def cotangent_laplacian(
        self,
        u: Optional[np.ndarray] = None,
        method: str = "ricci",
    ) -> csr_matrix:
        """Computes the cotangent Laplacian matrix for the current metric state.

        Uses vectorized COO triplet assembly for high performance.

        Args:
            u: Optional log-conformal factors.
            method: Conformal scaling method.

        Returns:
            A sparse CSR matrix representing the cotangent Laplacian.
        """
        angles = self.triangle_angles(u=u, method=method)
        n_faces = len(self.faces)
        
        # Precompute all cotangents for all faces (vectorized)
        cots = np.zeros((n_faces, 3), dtype=np.float64)
        for f_idx in range(n_faces):
            cots[f_idx] = _cotangents_from_angles(angles[f_idx])
        
        # Assemble COO triplets for edges
        i_idx = self.faces[:, 0]
        j_idx = self.faces[:, 1]
        k_idx = self.faces[:, 2]
        
        # Flattened row/col/data for sparse matrix construction
        rows = np.concatenate([i_idx, j_idx, j_idx, k_idx, k_idx, i_idx])
        cols = np.concatenate([j_idx, i_idx, k_idx, j_idx, i_idx, k_idx])
        
        # 0.5 * cotangent weights
        w_k = 0.5 * cots[:, 2]
        w_i = 0.5 * cots[:, 0]
        w_j = 0.5 * cots[:, 1]
        
        data = -np.concatenate([w_k, w_k, w_i, w_i, w_j, w_j])
        
        # Filter non-finite values
        mask = np.isfinite(data)
        rows = rows[mask]
        cols = cols[mask]
        data = data[mask]
        
        # The off-diagonal part of the Laplacian
        L_off = csr_matrix((data, (rows, cols)), shape=(self.n_vertices, self.n_vertices))
        
        # Diagonal part (sum of row off-diagonals)
        diag_data = -np.array(L_off.sum(axis=1)).flatten()
        L_diag = csr_matrix((diag_data, (np.arange(self.n_vertices), np.arange(self.n_vertices))), 
                           shape=(self.n_vertices, self.n_vertices))
        
        return L_off + L_diag


@dataclass
class SurfaceUniformizationResult:
    """Result object for a numerical uniformization solve.

    Attributes:
        method: Method used ('ricci' or 'circle_packing').
        target_geometry: Type of target geometry ('spherical', etc.).
        converged: Whether the solver reached the target tolerance.
        iterations: Number of iterations performed.
        residual_norm: Final norm of the curvature residual.
        conformal_factors: Final log-conformal factors.
        curvature: Final vertex Gaussian curvature.
        target_curvature: Prescribed target curvature.
        edge_lengths: Final edge lengths.
        mesh: The SurfaceMesh object.
        history: Residual norm history.
        notes: Diagnostic messages.
    """

    method: str
    target_geometry: str
    converged: bool
    iterations: int
    residual_norm: float
    conformal_factors: np.ndarray
    curvature: np.ndarray
    target_curvature: np.ndarray
    edge_lengths: np.ndarray
    mesh: SurfaceMesh
    history: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def decision_ready(self) -> bool:
        """Check if the result is conclusive and high-accuracy.

        Returns:
            True if converged and residual is below a threshold.
        """
        return self.converged and self.residual_norm < 1e-8


def _sorted_edge(i: int, j: int) -> tuple[int, int]:
    """Return a canonical sorted tuple for an edge.

    Args:
        i: First vertex index.
        j: Second vertex index.

    Returns:
        Sorted tuple (min(i, j), max(i, j)).
    """
    a = int(i)
    b = int(j)
    if a <= b:
        return a, b
    return b, a


def _build_edges_from_faces(
    faces: np.ndarray,
) -> tuple[np.ndarray, dict[tuple[int, int], int], list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]]:
    """Build edge list and mapping from face indices.

    Args:
        faces: (m, 3) array of faces.

    Returns:
        A tuple (edges, edge_to_index, face_edges).
    """
    edge_set = set()
    face_edges = []
    for face in faces:
        i, j, k = [int(x) for x in face]
        e0 = _sorted_edge(j, k)
        e1 = _sorted_edge(k, i)
        e2 = _sorted_edge(i, j)
        face_edges.append((e0, e1, e2))
        edge_set.update([e0, e1, e2])
    edges = np.array(sorted(edge_set), dtype=np.int64)
    edge_to_index = {tuple(edge): idx for idx, edge in enumerate(edges.tolist())}
    return edges, edge_to_index, face_edges


def _build_incidence(
    n_vertices: int,
    faces: np.ndarray,
    edges: np.ndarray,
    edge_to_index: dict[tuple[int, int], int],
    face_edges: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]],
) -> tuple[list[list[int]], list[list[int]], list[set[int]], np.ndarray]:
    """Build incidence structures for the mesh.

    Args:
        n_vertices: Total number of vertices.
        faces: (m, 3) array of faces.
        edges: (e, 2) array of edges.
        edge_to_index: Edge index mapping.
        face_edges: List of edges per face.

    Returns:
        A tuple (edge_faces, vertex_faces, vertex_neighbors, boundary_vertices).
    """
    edge_faces: list[list[int]] = [[] for _ in range(len(edges))]
    vertex_faces: list[list[int]] = [[] for _ in range(int(n_vertices))]
    vertex_neighbors: list[set[int]] = [set() for _ in range(int(n_vertices))]
    for f_idx, face in enumerate(faces):
        i, j, k = [int(x) for x in face]
        vertex_faces[i].append(f_idx)
        vertex_faces[j].append(f_idx)
        vertex_faces[k].append(f_idx)
        vertex_neighbors[i].update([j, k])
        vertex_neighbors[j].update([i, k])
        vertex_neighbors[k].update([i, j])
        for edge in face_edges[f_idx]:
            edge_faces[edge_to_index[edge]].append(f_idx)
    boundary_vertices = sorted(
        {
            v
            for edge_idx, incident in enumerate(edge_faces)
            if len(incident) == 1
            for v in edges[edge_idx]
        }
    )
    return edge_faces, vertex_faces, vertex_neighbors, np.asarray(boundary_vertices, dtype=np.int64)


def _validate_surface(
    mesh: "SurfaceMesh",
    faces: np.ndarray,
    edges: np.ndarray,
    edge_faces: list[list[int]],
) -> None:
    """Validate that the triangulation represents a 2-manifold.

    Args:
        mesh: The SurfaceMesh instance.
        faces: (m, 3) array of faces.
        edges: (e, 2) array of edges.
        edge_faces: Incident faces per edge.

    Raises:
        ValueError: If the mesh is not a manifold or has degenerate triangles.
    """
    for idx, incident in enumerate(edge_faces):
        if len(incident) > 2:
            raise ValueError(
                "Edge {!r} is incident to more than two faces; the input is not a 2-manifold.".format(
                    tuple(edges[idx])
                )
            )
    for face in faces:
        if len(set(int(v) for v in face)) != 3:
            raise ValueError("Degenerate triangle with repeated vertex indices detected.")

    # Vertex link validation (Detect pinched vertices)
    for v in range(mesh.n_vertices):
        # Extract the link of vertex v
        link_edges = []
        for f_idx in mesh.vertex_faces[v]:
            face = list(mesh.faces[f_idx])
            v_idx = face.index(v)
            # The edge in the link is opposite to v in the triangle
            others = [face[(v_idx + 1) % 3], face[(v_idx + 2) % 3]]
            link_edges.append(tuple(sorted(others)))
        
        if not link_edges:
            continue
            
        link_graph = {}
        for u, w in link_edges:
            link_graph.setdefault(u, set()).add(w)
            link_graph.setdefault(w, set()).add(u)
        
        degrees = [len(nbrs) for nbrs in link_graph.values()]
        is_boundary = v in mesh.boundary_vertices
        
        if is_boundary:
            if degrees.count(1) != 2 or any(d > 2 for d in degrees):
                raise ValueError(f"Vertex {v} has non-manifold link (pinched boundary).")
        else:
            if any(d != 2 for d in degrees):
                raise ValueError(f"Vertex {v} has non-manifold link (pinched internal point).")
                
        # Link must be connected
        start_node = next(iter(link_graph.keys()))
        visited = {start_node}
        stack = [start_node]
        while stack:
            curr = stack.pop()
            for nbr in link_graph[curr]:
                if nbr not in visited:
                    visited.add(nbr)
                    stack.append(nbr)
        if len(visited) != len(link_graph):
            raise ValueError(f"Vertex {v} is a topological singularity (disconnected link).")


def _compute_base_edge_lengths(vertices: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Compute Euclidean lengths for a set of edges.

    Args:
        vertices: (n, d) array of coordinates.
        edges: (e, 2) array of vertex indices.

    Returns:
        Array of edge lengths.

    Raises:
        ValueError: If any edge has zero length.
    """
    base = np.empty(len(edges), dtype=np.float64)
    for idx, (i, j) in enumerate(edges):
        diff = vertices[int(i)] - vertices[int(j)]
        base[idx] = float(np.linalg.norm(diff))
    # Avoid zero lengths from coincident vertices.
    if np.any(base <= 0):
        raise ValueError("All edges must have positive length.")
    return base


def _triangle_angles_from_lengths(a: float, b: float, c: float) -> np.ndarray:
    """Return the three angles opposite sides a, b, c.

    Args:
        a: Length of first side.
        b: Length of second side.
        c: Length of third side.

    Returns:
        (3,) array of angles in radians.

    Raises:
        ValueError: If lengths are non-positive or violate triangle inequality.
    """
    if min(a, b, c) <= 0:
        raise ValueError("Triangle edge lengths must be strictly positive.")
    if not (a + b > c and a + c > b and b + c > a):
        raise ValueError("Triangle inequality violated by current metric.")
    cos_A = np.clip((b * b + c * c - a * a) / (2.0 * b * c), -1.0, 1.0)
    cos_B = np.clip((a * a + c * c - b * b) / (2.0 * a * c), -1.0, 1.0)
    cos_C = np.clip((a * a + b * b - c * c) / (2.0 * a * b), -1.0, 1.0)
    return np.array([
        math.acos(cos_A),
        math.acos(cos_B),
        math.acos(cos_C),
    ], dtype=np.float64)


def _cotangents_from_angles(angles: np.ndarray) -> np.ndarray:
    """Compute cotangents for a set of angles with stabilization.

    Args:
        angles: (3,) array of angles.

    Returns:
        (3,) array of cotangents.
    """
    s = np.sin(angles)
    c = np.cos(angles)
    out = np.empty_like(angles)
    eps = 1e-14
    for idx in range(3):
        if abs(s[idx]) < eps:
            out[idx] = math.copysign(1e14, c[idx])
        else:
            out[idx] = c[idx] / s[idx]
    return out


def _accumulate_edge_weight(
    weights: dict[tuple[int, int], float],
    i: int,
    j: int,
    value: float,
) -> None:
    """Helper to accumulate weight into an edge dictionary.

    Args:
        weights: Dictionary mapping sorted edge tuples to floats.
        i: First vertex.
        j: Second vertex.
        value: Weight to add.
    """
    edge = _sorted_edge(i, j)
    weights[edge] = weights.get(edge, 0.0) + float(value)


def _surface_target_curvature(mesh: SurfaceMesh, target: Optional[np.ndarray]) -> np.ndarray:
    """Compute target curvature distribution based on Gauss-Bonnet.

    Args:
        mesh: The SurfaceMesh instance.
        target: Optional explicitly prescribed curvature.

    Returns:
        (n_vertices,) array of target curvatures.

    Raises:
        ValueError: If prescribed target sum violates Gauss-Bonnet.
    """
    total_goal = _TWO_PI * float(mesh.euler_characteristic)
    if target is not None:
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (mesh.n_vertices,):
            raise ValueError("target_curvature must have shape (n_vertices,)")
        # Sum must strictly match 2*pi*chi
        if not np.isclose(np.sum(target), total_goal, atol=1e-8):
            raise ValueError(f"Prescribed curvature sum {np.sum(target)} must be {total_goal}.")
        return target

    target_curv = np.zeros(mesh.n_vertices, dtype=np.float64)
    boundary = mesh.boundary_vertices
    internal = np.setdiff1d(np.arange(mesh.n_vertices), boundary)

    # Default: Geodesic boundary (k=0) and constant internal Gaussian curvature.
    if len(boundary) > 0:
        target_curv[boundary] = 0.0
        if len(internal) > 0:
            target_curv[internal] = total_goal / len(internal)
    else:
        target_curv[:] = total_goal / mesh.n_vertices
    return target_curv



def _solve_pinned_linear_system(
    matrix: csr_matrix,
    rhs: np.ndarray,
    pin_vertex: int,
) -> np.ndarray:
    """Stable singular system solver using LSQR and rigid pinning.

    Args:
        matrix: Sparse Laplacian matrix.
        rhs: Right-hand side (curvature residual).
        pin_vertex: Index of the vertex to pin to zero.

    Returns:
        Minimal-norm solution vector.
    """
    from scipy.sparse.linalg import lsqr
    
    # LSQR handles the rank-1 deficiency of the Laplacian across all components
    # and provides a minimal-norm solution if A is singular.
    res = lsqr(matrix, rhs, atol=1e-10, btol=1e-10)
    u = res[0]
    
    # Apply the pinning constraint as a rigid shift to the solution.
    # Conformal factors u_i are defined up to a global constant in the closed case.
    return u - u[int(pin_vertex)]


def _metric_is_valid(mesh: SurfaceMesh, u: np.ndarray, method: str) -> bool:
    """Check if the metric remains valid (positive lengths, triangle inequality).

    Args:
        mesh: The SurfaceMesh instance.
        u: Current log-conformal factors.
        method: Conformal scaling method.

    Returns:
        True if the metric is valid.
    """
    try:
        tri_lengths = mesh.triangle_edge_lengths(u=u, method=method)
    except Exception:
        return False
    for a, b, c in tri_lengths:
        if min(a, b, c) <= 0:
            return False
        if not (a + b > c and a + c > b and b + c > a):
            return False
    return True


def _solve_uniformization(
    mesh: SurfaceMesh,
    *,
    method: str,
    target_curvature: Optional[np.ndarray],
    max_iter: int,
    tol: float,
    pin_vertex: int,
    damping: float,
    line_search: bool,
) -> SurfaceUniformizationResult:
    """Internal solver loop for Ricci flow and circle packing.

    Args:
        mesh: SurfaceMesh to uniformize.
        method: Scaling method.
        target_curvature: Optional prescribed target.
        max_iter: Maximum iterations.
        tol: Convergence tolerance.
        pin_vertex: Index of pinned vertex.
        damping: Initial step damping.
        line_search: Whether to use backtracking line search.

    Returns:
        A SurfaceUniformizationResult instance.
    """
    target = _surface_target_curvature(mesh, target_curvature)
    u = np.zeros(mesh.n_vertices, dtype=np.float64)
    history: list[float] = []
    notes: list[str] = []
    converged = False

    for it in range(int(max_iter)):
        curvature = mesh.vertex_gaussian_curvature(u=u, method=method)
        residual = curvature - target
        residual_norm = float(np.linalg.norm(residual, ord=2))
        history.append(residual_norm)
        if residual_norm <= tol:
            converged = True
            break

        lap = mesh.cotangent_laplacian(u=u, method=method)
        try:
            step = _solve_pinned_linear_system(lap, residual, pin_vertex)
        except Exception as exc:
            notes.append("linear solve failed: {!r}".format(exc))
            break

        step *= float(damping)
        step[pin_vertex] = 0.0

        alpha = 1.0
        accepted = False
        candidate = u - step
        if line_search:
            while alpha >= 1.0 / 128.0:
                candidate = u - alpha * step
                candidate[pin_vertex] = 0.0
                if not _metric_is_valid(mesh, candidate, method):
                    alpha *= 0.5
                    continue
                candidate_curv = mesh.vertex_gaussian_curvature(u=candidate, method=method)
                cand_residual = candidate_curv - target
                cand_norm = float(np.linalg.norm(cand_residual, ord=2))
                if cand_norm < residual_norm:
                    accepted = True
                    u = candidate
                    break
                alpha *= 0.5
        else:
            candidate[pin_vertex] = 0.0
            if _metric_is_valid(mesh, candidate, method):
                u = candidate
                accepted = True

        if not accepted:
            # Fall back to a mild damped update if line search stalls.
            candidate = u - 0.25 * step
            candidate[pin_vertex] = 0.0
            if _metric_is_valid(mesh, candidate, method):
                u = candidate
                accepted = True
            else:
                notes.append("line search stalled at iteration {!r}".format(it))
                break

    curvature = mesh.vertex_gaussian_curvature(u=u, method=method)
    residual = curvature - target
    residual_norm = float(np.linalg.norm(residual, ord=2))
    edge_lengths = mesh.conformal_edge_lengths(u, method=method)
    if residual_norm <= tol:
        converged = True

    return SurfaceUniformizationResult(
        method=method,
        target_geometry=mesh.target_geometry(),
        converged=converged,
        iterations=len(history),
        residual_norm=residual_norm,
        conformal_factors=u,
        curvature=curvature,
        target_curvature=target,
        edge_lengths=edge_lengths,
        mesh=mesh,
        history=history,
        notes=notes,
    )


def discrete_ricci_flow(
    surface: SurfaceMesh | SimplicialComplex | tuple[np.ndarray, Sequence[Sequence[int]]],
    *,
    coordinates: Optional[np.ndarray] = None,
    target_curvature: Optional[np.ndarray] = None,
    max_iter: int = 50,
    tol: float = 1e-8,
    pin_vertex: int = 0,
    damping: float = 1.0,
    line_search: bool = True,
    validate: bool = True,
) -> SurfaceUniformizationResult:
    """Uniformize a triangulated surface using discrete Ricci flow.

    Args:
        surface: Input surface (Mesh, Complex, or (V, F) pair).
        coordinates: Optional explicit coordinates.
        target_curvature: Optional target curvature vector.
        max_iter: Maximum iterations.
        tol: Convergence tolerance.
        pin_vertex: Index of pinned vertex.
        damping: Damping factor for Newton steps.
        line_search: Whether to use backtracking line search.
        validate: Whether to validate the mesh.

    Returns:
        A SurfaceUniformizationResult instance.
    """
    mesh = _coerce_surface_mesh(surface, coordinates=coordinates, validate=validate)
    return _solve_uniformization(
        mesh,
        method="ricci",
        target_curvature=target_curvature,
        max_iter=max_iter,
        tol=tol,
        pin_vertex=pin_vertex,
        damping=damping,
        line_search=line_search,
    )


def circle_packing_uniformization(
    surface: SurfaceMesh | SimplicialComplex | tuple[np.ndarray, Sequence[Sequence[int]]],
    *,
    coordinates: Optional[np.ndarray] = None,
    target_curvature: Optional[np.ndarray] = None,
    max_iter: int = 50,
    tol: float = 1e-8,
    pin_vertex: int = 0,
    damping: float = 0.9,
    line_search: bool = True,
    validate: bool = True,
) -> SurfaceUniformizationResult:
    """Uniformize a triangulated surface via a circle-packing style metric ansatz.

    Args:
        surface: Input surface.
        coordinates: Optional coordinates.
        target_curvature: Optional target curvature.
        max_iter: Maximum iterations.
        tol: Tolerance.
        pin_vertex: Pinned vertex index.
        damping: Damping factor.
        line_search: Whether to use line search.
        validate: Whether to validate.

    Returns:
        A SurfaceUniformizationResult instance.
    """
    mesh = _coerce_surface_mesh(surface, coordinates=coordinates, validate=validate)
    return _solve_uniformization(
        mesh,
        method="circle_packing",
        target_curvature=target_curvature,
        max_iter=max_iter,
        tol=tol,
        pin_vertex=pin_vertex,
        damping=damping,
        line_search=line_search,
    )


def uniformize_surface(
    surface: SurfaceMesh | SimplicialComplex | tuple[np.ndarray, Sequence[Sequence[int]]],
    *,
    method: str = "ricci",
    coordinates: Optional[np.ndarray] = None,
    target_curvature: Optional[np.ndarray] = None,
    max_iter: int = 50,
    tol: float = 1e-8,
    pin_vertex: int = 0,
    damping: float = 1.0,
    line_search: bool = True,
    validate: bool = True,
) -> SurfaceUniformizationResult:
    """User-facing uniformization entry point.

    Args:
        surface: Input surface.
        method: Solver method ('ricci' or 'circle_packing').
        coordinates: Optional coordinates.
        target_curvature: Optional target curvature.
        max_iter: Maximum iterations.
        tol: Tolerance.
        pin_vertex: Pinned vertex index.
        damping: Damping factor.
        line_search: Whether to use line search.
        validate: Whether to validate.

    Returns:
        A SurfaceUniformizationResult instance.

    Raises:
        ValueError: If the method is unknown.
    """
    if method == "ricci":
        return discrete_ricci_flow(
            surface,
            coordinates=coordinates,
            target_curvature=target_curvature,
            max_iter=max_iter,
            tol=tol,
            pin_vertex=pin_vertex,
            damping=damping,
            line_search=line_search,
            validate=validate,
        )
    if method == "circle_packing":
        return circle_packing_uniformization(
            surface,
            coordinates=coordinates,
            target_curvature=target_curvature,
            max_iter=max_iter,
            tol=tol,
            pin_vertex=pin_vertex,
            damping=damping,
            line_search=line_search,
            validate=validate,
        )
    raise ValueError("Unsupported uniformization method: {!r}".format(method))


def _coerce_surface_mesh(
    surface: SurfaceMesh | SimplicialComplex | tuple[np.ndarray, Sequence[Sequence[int]]],
    *,
    coordinates: Optional[np.ndarray] = None,
    validate: bool = True,
) -> SurfaceMesh:
    """Coerce various inputs into a SurfaceMesh instance.

    Args:
        surface: The input surface data.
        coordinates: Optional coordinates for complexes.
        validate: Whether to validate the mesh.

    Returns:
        A SurfaceMesh instance.

    Raises:
        TypeError: If the input type is not supported.
    """
    if isinstance(surface, SurfaceMesh):
        return surface
    if isinstance(surface, SimplicialComplex):
        return SurfaceMesh.from_simplicial_complex(surface, coordinates=coordinates, validate=validate)
    if isinstance(surface, tuple) and len(surface) == 2:
        vertices, faces = surface
        return SurfaceMesh.from_vertices_faces(vertices, faces, validate=validate)
    raise TypeError(
        "surface must be a SurfaceMesh, a SimplicialComplex, or a (vertices, faces) pair"
    )


__all__ = [
    "SurfaceMesh",
    "SurfaceUniformizationResult",
    "circle_packing_uniformization",
    "discrete_ricci_flow",
    "uniformize_surface",
    "vertex_gaussian_curvature",
    "cotangent_laplacian",
    "surface_target_curvature",
]


# Convenience wrappers around the mesh methods for a functional API.

def vertex_gaussian_curvature(
    surface: SurfaceMesh | SimplicialComplex | tuple[np.ndarray, Sequence[Sequence[int]]],
    u: Optional[np.ndarray] = None,
    *,
    method: str = "ricci",
    coordinates: Optional[np.ndarray] = None,
    validate: bool = True,
) -> np.ndarray:
    """Functional wrapper for vertex Gaussian curvature.

    Args:
        surface: Input surface.
        u: Optional log-conformal factors.
        method: Conformal scaling method.
        coordinates: Optional coordinates.
        validate: Whether to validate.

    Returns:
        (n_vertices,) array of curvatures.
    """
    mesh = _coerce_surface_mesh(surface, coordinates=coordinates, validate=validate)
    return mesh.vertex_gaussian_curvature(u=u, method=method)


def cotangent_laplacian(
    surface: SurfaceMesh | SimplicialComplex | tuple[np.ndarray, Sequence[Sequence[int]]],
    u: Optional[np.ndarray] = None,
    *,
    method: str = "ricci",
    coordinates: Optional[np.ndarray] = None,
    validate: bool = True,
) -> csr_matrix:
    """Functional wrapper for the cotangent Laplacian.

    Args:
        surface: Input surface.
        u: Optional log-conformal factors.
        method: Conformal scaling method.
        coordinates: Optional coordinates.
        validate: Whether to validate.

    Returns:
        Sparse Laplacian matrix.
    """
    mesh = _coerce_surface_mesh(surface, coordinates=coordinates, validate=validate)
    return mesh.cotangent_laplacian(u=u, method=method)


def surface_target_curvature(
    surface: SurfaceMesh | SimplicialComplex | tuple[np.ndarray, Sequence[Sequence[int]]],
    *,
    coordinates: Optional[np.ndarray] = None,
    validate: bool = True,
) -> np.ndarray:
    """Functional wrapper for target curvature distribution.

    Args:
        surface: Input surface.
        coordinates: Optional coordinates.
        validate: Whether to validate.

    Returns:
        (n_vertices,) array of target curvatures.
    """
    mesh = _coerce_surface_mesh(surface, coordinates=coordinates, validate=validate)
    return _surface_target_curvature(mesh, None)
