"""Computational 3-manifold geometrization scaffolding.

This module implements a practical, conservative combinatorial workflow for
triangulated 3-manifolds.

Goals
-----
- build a native triangulated-manifold wrapper from simplicial data,
- generate canonical normal-surface candidates (vertex links and edge links),
- assemble sparse matching constraints and validate candidates,
- perform tractable prime/JSJ decomposition heuristics via graph cuts,
- emit a recognition result that can be converted into the existing 3D
  homeomorphism certificate path.

The implementation is intentionally conservative: it prefers exact positive
certificates when the input data are strong enough and otherwise returns an
inconclusive or heuristic result rather than overclaiming.

References
----------
- Haken, W. (1961). Theorie der Normalflächen. Acta Mathematica, 105(3-4), 245-375.
- Jaco, W., & Rubinstein, J. H. (2003). 0-efficient triangulations of 3-manifolds. 
  Journal of Differential Geometry, 65(1), 61-168.
- Matveev, S. (2003). Algorithmic topology and classification of 3-manifolds. 
  Springer Science & Business Media.
- Thurston, W. P. (1982). Three-dimensional manifolds, Kleinian groups and hyperbolic geometry. 
  Bulletin of the American Mathematical Society, 6(3), 357-381.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import combinations
from typing import Iterable, Optional, Sequence

import numpy as np
from scipy.sparse import csr_matrix

from .complexes import ChainComplex, SimplicialComplex
from .theorem_tags import infer_theorem_tag
from ..bridge.julia_bridge import julia_engine

def _freeze_value(value: object) -> object:
    """Recursively freeze mutable structures into immutable hashes.

    Args:
        value (object): The value to freeze.

    Returns:
        object: The frozen immutable version of the value.
    """
    if isinstance(value, dict):
        return tuple(
            (str(k), _freeze_value(v))
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted((_freeze_value(v) for v in value), key=repr))
    if isinstance(value, np.ndarray):
        return _freeze_value(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    return value


def _sorted_face(face: Iterable[int]) -> tuple[int, int, int]:
    """Normalize a triangle face to a sorted tuple.

    Args:
        face (Iterable[int]): The vertices of the face.

    Returns:
        tuple[int, int, int]: The sorted vertices.

    Raises:
        ValueError: If the face does not have exactly 3 vertices.
    """
    t = tuple(sorted(int(v) for v in face))
    if len(t) != 3:
        raise ValueError(f"Expected a triangle face, got {face!r}")
    return t


def _sorted_tetra(tetra: Iterable[int]) -> tuple[int, int, int, int]:
    """Normalize a tetrahedron to a sorted tuple.

    Args:
        tetra (Iterable[int]): The vertices of the tetrahedron.

    Returns:
        tuple[int, int, int, int]: The sorted vertices.

    Raises:
        ValueError: If the tetrahedron does not have exactly 4 distinct vertices.
    """
    t = tuple(sorted(int(v) for v in tetra))
    if len(t) != 4:
        raise ValueError(f"Expected a tetrahedron, got {tetra!r}")
    if len(set(t)) != 4:
        raise ValueError(f"Tetrahedron vertices must be distinct: {tetra!r}")
    return t


_QUAD_EDGE_PAIRS = {
    frozenset({0, 1}): 0,
    frozenset({2, 3}): 0,
    frozenset({0, 2}): 1,
    frozenset({1, 3}): 1,
    frozenset({0, 3}): 2,
    frozenset({1, 2}): 2,
}

_MATCHING_RESIDUAL_TOL = 1e-12
_JULIA_RESIDUAL_BATCH_THRESHOLD = 32


@dataclass
class Triangulated3Manifold:
    """A finite triangulated 3-manifold encoded by tetrahedra and a simplicial closure.

    Overview:
        A Triangulated3Manifold represents a 3-dimensional manifold via a 
        collection of tetrahedra (3-simplices) and their gluing information. 
        It provides high-level topological tools specific to 3-manifold theory, 
        including normal surface candidate generation and prime/JSJ 
        decomposition heuristics.

    Key Concepts:
        - **Tetrahedra**: The fundamental 3-dimensional building blocks.
        - **Normal Surfaces**: Surfaces embedded in the manifold that intersect 
          each tetrahedron in a collection of elementary discs (triangles or quads).
        - **Prime Decomposition**: Breaking a 3-manifold into pieces that 
          cannot be further decomposed via connected sums.
        - **JSJ Decomposition**: A canonical decomposition of irreducible 
          3-manifolds along incompressible tori.

    Common Workflows:
        1. **Creation** → `from_tetrahedra()` or `from_simplicial_complex()`.
        2. **Analysis** → `homology()`, `euler_characteristic()`.
        3. **Geometrization** → Pass to `analyze_geometrization()` for recognition.

    Attributes:
        tetrahedra (tuple[tuple[int, int, int, int], ...]): List of tetrahedra.
        simplicial_complex (SimplicialComplex): The underlying simplicial complex.
        face_to_tetrahedra (dict[tuple[int, int, int], tuple[int, ...]]): Mapping from
            faces to indices of tetrahedra containing them.
        boundary_faces (tuple[tuple[int, int, int], ...]): List of boundary faces.
        name (str): The name of the manifold.
    """

    tetrahedra: tuple[tuple[int, int, int, int], ...]
    simplicial_complex: SimplicialComplex
    face_to_tetrahedra: dict[tuple[int, int, int], tuple[int, ...]]
    boundary_faces: tuple[tuple[int, int, int], ...] = field(default_factory=tuple)
    name: str = "triangulated_3_manifold"

    @classmethod
    def from_tetrahedra(
        cls,
        tetrahedra: Sequence[Sequence[int]],
        *,
        name: str = "triangulated_3_manifold",
    ) -> "Triangulated3Manifold":
        """Create a manifold from a list of tetrahedra.

        Args:
            tetrahedra (Sequence[Sequence[int]]): The tetrahedra.
            name (str): The name of the manifold. Defaults to "triangulated_3_manifold".

        Returns:
            Triangulated3Manifold: The created manifold.

        Raises:
            ValueError: If no tetrahedra are provided.
        """
        tetra_list = tuple(_sorted_tetra(t) for t in tetrahedra)
        if not tetra_list:
            raise ValueError("At least one tetrahedron is required.")
        simplex = SimplicialComplex.from_maximal_simplices(tetra_list)
        face_to_tets = _build_face_to_tetrahedra(tetra_list)
        boundary_faces = tuple(sorted(face for face, tets in face_to_tets.items() if len(tets) == 1))
        return cls(
            tetrahedra=tetra_list,
            simplicial_complex=simplex,
            face_to_tetrahedra=face_to_tets,
            boundary_faces=boundary_faces,
            name=name,
        )

    @classmethod
    def from_simplicial_complex(
        cls,
        complex_: SimplicialComplex,
        *,
        name: str = "triangulated_3_manifold",
    ) -> "Triangulated3Manifold":
        """Create a manifold from a SimplicialComplex.

        Args:
            complex_ (SimplicialComplex): The simplicial complex.
            name (str): The name of the manifold. Defaults to "triangulated_3_manifold".

        Returns:
            Triangulated3Manifold: The created manifold.

        Raises:
            ValueError: If the complex contains no 3-simplices.
        """
        tetrahedra = complex_.n_simplices(3)
        if not tetrahedra:
            raise ValueError("A 3-manifold triangulation requires 3-simplices.")
        return cls.from_tetrahedra(tetrahedra, name=name)

    @property
    def n_tetrahedra(self) -> int:
        """Return the number of tetrahedra in the triangulation."""
        return len(self.tetrahedra)

    @property
    def n_vertices(self) -> int:
        """Return the number of vertices in the triangulation."""
        return len(self.simplicial_complex.n_simplices(0))

    @property
    def is_closed(self) -> bool:
        """Return True if the manifold is closed (no boundary faces)."""
        return len(self.boundary_faces) == 0

    @property
    def euler_characteristic(self) -> int:
        """Return the Euler characteristic of the manifold."""
        return self.simplicial_complex.euler_characteristic()

    def chain_complex(self) -> ChainComplex:
        """Return the chain complex of the manifold."""
        return self.simplicial_complex.chain_complex()

    def homology(
        self, n: int | None = None, backend: str = "auto"
    ) -> tuple[int, list[int]] | dict[int, tuple[int, list[int]]]:
        """Return homology in degree ``n`` or all degrees when ``n`` is omitted.

        Args:
            n (int | None): The degree of homology. Defaults to None.
            backend: 'auto', 'julia', or 'python'.

        Returns:
            tuple[int, list[int]] | dict[int, tuple[int, list[int]]]: Homology group
                or dictionary of homology groups.
        """
        return self.simplicial_complex.homology(n, backend=backend)

    def dual_graph(self) -> dict[int, set[int]]:
        """Compute the dual graph of the triangulation.

        Returns:
            dict[int, set[int]]: Adjacency list for the dual graph.
        """
        graph = {i: set() for i in range(self.n_tetrahedra)}
        for tets in self.face_to_tetrahedra.values():
            if len(tets) == 2:
                a, b = tets
                graph[a].add(b)
                graph[b].add(a)
        return graph

    def edge_set(self) -> list[tuple[int, int]]:
        """Return the set of edges in the triangulation.

        Returns:
            list[tuple[int, int]]: List of edges as sorted pairs of vertex indices.
        """
        return [tuple(edge) for edge in self.simplicial_complex.n_simplices(1)]

    def submanifold(self, tetra_indices: Iterable[int], *, name: Optional[str] = None) -> "Triangulated3Manifold":
        """Extract a sub-manifold defined by a subset of tetrahedra.

        Args:
            tetra_indices (Iterable[int]): Indices of tetrahedra to include.
            name (Optional[str]): Name of the submanifold.

        Returns:
            Triangulated3Manifold: The extracted submanifold.
        """
        idx = sorted({int(i) for i in tetra_indices if 0 <= int(i) < self.n_tetrahedra})
        if not idx:
            return self
        tetrahedra = [self.tetrahedra[i] for i in idx]
        return Triangulated3Manifold.from_tetrahedra(
            tetrahedra,
            name=name or (self.name + "_subpiece"),
        )

    def to_legacy_dict(self) -> dict[str, object]:
        """Return a dictionary representation of the manifold for serialization.

        Returns:
            dict[str, object]: Dictionary with manifold data.
        """
        return {
            "name": self.name,
            "tetrahedra": [list(t) for t in self.tetrahedra],
            "boundary_faces": [list(f) for f in self.boundary_faces],
            "n_tetrahedra": self.n_tetrahedra,
            "n_vertices": self.n_vertices,
            "euler_characteristic": self.euler_characteristic,
        }


@dataclass
class NormalSurfaceCandidate:
    """A canonical normal-surface candidate with exact combinatorial bookkeeping.

    References:
        Haken, W. (1961). Theorie der Normalflächen. Acta Mathematica, 105(3-4), 245-375.

    Attributes:
        kind (str): Surface kind (e.g., "sphere", "torus").
        surface_type (str): Type of surface (e.g., "vertex_link", "edge_link").
        exact (bool): Whether the candidate is exact.
        validated (bool): Whether the candidate has been validated.
        coordinates (np.ndarray): Normal coordinates.
        triangle_coordinates (np.ndarray): Triangle coordinates.
        quad_coordinates (np.ndarray): Quadrilateral coordinates.
        support_tetrahedra (tuple[int, ...]): Indices of tetrahedra in the support.
        euler_characteristic (int): Euler characteristic of the surface.
        matching_residual (float): Residual of the matching constraints.
        quadrilateral_ok (bool): Whether quadrilateral constraints are satisfied.
        source (str): Source of the candidate.
        notes (list[str]): Additional notes.
    """

    kind: str
    surface_type: str
    exact: bool
    validated: bool
    coordinates: np.ndarray
    triangle_coordinates: np.ndarray
    quad_coordinates: np.ndarray
    support_tetrahedra: tuple[int, ...]
    euler_characteristic: int
    matching_residual: float = 0.0
    quadrilateral_ok: bool = True
    source: str = "canonical_link"
    notes: list[str] = field(default_factory=list)

    def decision_ready(self) -> bool:
        """Check if the candidate is ready for geometric decision making.

        Returns:
            bool: True if exact, validated, and satisfying all constraints.
        """
        return bool(
            self.exact
            and self.validated
            and self.quadrilateral_ok
            and self.matching_residual <= _MATCHING_RESIDUAL_TOL
        )

    def to_legacy_dict(self) -> dict[str, object]:
        """Return a dictionary representation for serialization.

        Returns:
            dict[str, object]: Dictionary representation.
        """
        return {
            "kind": self.kind,
            "surface_type": self.surface_type,
            "exact": self.exact,
            "validated": self.validated,
            "coordinates": self.coordinates.tolist(),
            "triangle_coordinates": self.triangle_coordinates.tolist(),
            "quad_coordinates": self.quad_coordinates.tolist(),
            "support_tetrahedra": list(self.support_tetrahedra),
            "euler_characteristic": self.euler_characteristic,
            "matching_residual": float(self.matching_residual),
            "quadrilateral_ok": bool(self.quadrilateral_ok),
            "source": self.source,
            "notes": list(self.notes),
            "decision_ready": self.decision_ready(),
        }

    @classmethod
    def vertex_link(
        cls,
        manifold: Triangulated3Manifold,
        vertex: int,
        *,
        matching_matrix: Optional[csr_matrix] = None,
    ) -> "NormalSurfaceCandidate":
        """Create a normal-surface candidate from a vertex link.

        Args:
            manifold (Triangulated3Manifold): The 3-manifold.
            vertex (int): The vertex index.
            matching_matrix (Optional[csr_matrix]): Optional pre-computed matching matrix.

        Returns:
            NormalSurfaceCandidate: The vertex-link candidate.
        """
        n_tet = manifold.n_tetrahedra
        triangle_coords = np.zeros((n_tet, 4), dtype=np.int64)
        support = []
        for t_idx, tet in enumerate(manifold.tetrahedra):
            if int(vertex) not in tet:
                continue
            local = tet.index(int(vertex))
            triangle_coords[t_idx, local] = 1
            support.append(t_idx)
        coords = np.hstack((triangle_coords, np.zeros((n_tet, 3), dtype=np.int64))).reshape(-1)
        residual = _normal_surface_matching_residual(
            manifold,
            coords,
            matching_matrix=matching_matrix,
        )
        return cls(
            kind="sphere",
            surface_type="vertex_link",
            exact=True,
            validated=True,
            coordinates=coords,
            triangle_coordinates=triangle_coords,
            quad_coordinates=np.zeros((n_tet, 3), dtype=np.int64),
            support_tetrahedra=tuple(sorted(set(support))),
            euler_characteristic=2,
            matching_residual=residual,
            quadrilateral_ok=True,
            source="vertex_link",
            notes=["Canonical vertex-link sphere"],
        )

    @classmethod
    def edge_link(
        cls,
        manifold: Triangulated3Manifold,
        edge: tuple[int, int],
        *,
        matching_matrix: Optional[csr_matrix] = None,
    ) -> "NormalSurfaceCandidate":
        """Create a normal-surface candidate from an edge link.

        Args:
            manifold (Triangulated3Manifold): The 3-manifold.
            edge (tuple[int, int]): The edge vertex indices.
            matching_matrix (Optional[csr_matrix]): Optional pre-computed matching matrix.

        Returns:
            NormalSurfaceCandidate: The edge-link candidate.
        """
        n_tet = manifold.n_tetrahedra
        triangle_coords = np.zeros((n_tet, 4), dtype=np.int64)
        quad_coords = np.zeros((n_tet, 3), dtype=np.int64)
        support = []
        edge = tuple(sorted(int(v) for v in edge))
        for t_idx, tet in enumerate(manifold.tetrahedra):
            if edge[0] not in tet or edge[1] not in tet:
                continue
            local = (tet.index(edge[0]), tet.index(edge[1]))
            quad_type = _quad_type_for_local_edge(local)
            quad_coords[t_idx, quad_type] = 1
            support.append(t_idx)
        coords = np.hstack((triangle_coords, quad_coords)).reshape(-1)
        residual = _normal_surface_matching_residual(
            manifold,
            coords,
            matching_matrix=matching_matrix,
        )
        return cls(
            kind="torus",
            surface_type="edge_link",
            exact=True,
            validated=True,
            coordinates=coords,
            triangle_coordinates=triangle_coords,
            quad_coordinates=quad_coords,
            support_tetrahedra=tuple(sorted(set(support))),
            euler_characteristic=0,
            matching_residual=residual,
            quadrilateral_ok=True,
            source="edge_link",
            notes=["Canonical edge-link torus"],
        )

    @classmethod
    def graph_cut(
        cls,
        manifold: Triangulated3Manifold,
        support_tetrahedra: Iterable[int],
        *,
        kind: str,
        surface_type: str,
        source: str,
        matching_matrix: Optional[csr_matrix] = None,
    ) -> "NormalSurfaceCandidate":
        """Create a normal-surface candidate from a dual-graph cut.

        Args:
            manifold (Triangulated3Manifold): The 3-manifold.
            support_tetrahedra (Iterable[int]): Tetrahedra in the cut support.
            kind (str): Surface kind.
            surface_type (str): Surface type.
            source (str): Source description.
            matching_matrix (Optional[csr_matrix]): Optional matching matrix.

        Returns:
            NormalSurfaceCandidate: The heuristic cut candidate.
        """
        support = tuple(sorted({int(i) for i in support_tetrahedra}))
        n_tet = manifold.n_tetrahedra
        coords = np.zeros(7 * n_tet, dtype=np.int64)
        triangle_coords = np.zeros((n_tet, 4), dtype=np.int64)
        quad_coords = np.zeros((n_tet, 3), dtype=np.int64)
        for t_idx in support:
            if t_idx >= n_tet:
                continue
            triangle_coords[t_idx, 0] = 1
        coords = np.hstack((triangle_coords, quad_coords)).reshape(-1)
        residual = _normal_surface_matching_residual(
            manifold,
            coords,
            matching_matrix=matching_matrix,
        )
        return cls(
            kind=kind,
            surface_type=surface_type,
            exact=False,
            validated=False,
            coordinates=coords,
            triangle_coordinates=triangle_coords,
            quad_coordinates=quad_coords,
            support_tetrahedra=support,
            euler_characteristic=0 if kind == "torus" else 2,
            matching_residual=residual,
            quadrilateral_ok=True,
            source=source,
            notes=["Graph-cut heuristic surface"],
        )


@dataclass
class PieceDecomposition:
    """A coarse decomposition returned by the prime/JSJ/crushing heuristics.

    Attributes:
        kind (str): Type of decomposition (e.g., "prime", "jsj").
        exact (bool): Whether the decomposition is exact.
        validated (bool): Whether the decomposition has been validated.
        pieces (list[Triangulated3Manifold]): List of manifolds resulting from the cut.
        cut_surfaces (list[NormalSurfaceCandidate]): Surfaces used for the cut.
        adjacency (list[tuple[int, int]]): Adjacency in the decomposition graph.
        support_tetrahedra (list[list[int]]): Tetrahedra in the support of the cuts.
        summary (str): Brief summary of the decomposition.
        notes (list[str]): Additional notes.
    """

    kind: str
    exact: bool
    validated: bool
    pieces: list[Triangulated3Manifold] = field(default_factory=list)
    cut_surfaces: list[NormalSurfaceCandidate] = field(default_factory=list)
    adjacency: list[tuple[int, int]] = field(default_factory=list)
    support_tetrahedra: list[list[int]] = field(default_factory=list)
    summary: str = ""
    notes: list[str] = field(default_factory=list)

    def to_legacy_dict(self) -> dict[str, object]:
        """Return a dictionary representation for serialization.

        Returns:
            dict[str, object]: Dictionary representation.
        """
        return {
            "kind": self.kind,
            "exact": self.exact,
            "validated": self.validated,
            "pieces": [piece.to_legacy_dict() for piece in self.pieces],
            "cut_surfaces": [surface.to_legacy_dict() for surface in self.cut_surfaces],
            "adjacency": [list(edge) for edge in self.adjacency],
            "support_tetrahedra": [list(s) for s in self.support_tetrahedra],
            "summary": self.summary,
            "notes": list(self.notes),
        }


@dataclass
class GeometrizationResult:
    """Geometrization-oriented recognition result with an exportable certificate payload.

    Attributes:
        status (str): Classification status ("success" or "inconclusive").
        classification (str): Geometric classification.
        theorem (str): Theorem name.
        theorem_tag (str): Theorem unique tag.
        exact (bool): Whether the result is exact.
        validated (bool): Whether the result has been validated.
        manifold (Triangulated3Manifold): The analyzed manifold.
        homology (dict[int, tuple[int, list[int]]]): Homology groups.
        candidates (list[NormalSurfaceCandidate]): Evaluated normal surfaces.
        prime_decomposition (Optional[PieceDecomposition]): Prime decomposition.
        jsj_decomposition (Optional[PieceDecomposition]): JSJ decomposition.
        pi1_descriptor (Optional[str]): Fundamental group descriptor.
        assumptions (list[str]): List of assumptions.
        evidence (list[str]): List of evidentiary findings.
        missing_data (list[str]): List of missing data required for exactness.
        certificates (dict[str, object]): Dictionary of supporting certificates.
        summary (str): Human-readable summary.
    """

    status: str
    classification: str
    theorem: str
    theorem_tag: str
    exact: bool
    validated: bool
    manifold: Triangulated3Manifold
    homology: dict[int, tuple[int, list[int]]]
    candidates: list[NormalSurfaceCandidate] = field(default_factory=list)
    prime_decomposition: Optional[PieceDecomposition] = None
    jsj_decomposition: Optional[PieceDecomposition] = None
    pi1_descriptor: Optional[str] = None
    assumptions: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    missing_data: list[str] = field(default_factory=list)
    certificates: dict[str, object] = field(default_factory=dict)
    summary: str = ""

    def decision_ready(self) -> bool:
        """Check if the result is ready for downstream geometric logic.

        Returns:
            bool: True if status is success, exact, and validated.
        """
        return bool(self.status == "success" and self.exact and self.validated)

    def to_recognition_certificate(self):
        """Convert the result into a 3-manifold recognition certificate.

        Returns:
            ThreeManifoldRecognitionCertificate: The generated certificate.
        """
        from ..homeomorphism import ThreeManifoldRecognitionCertificate

        return ThreeManifoldRecognitionCertificate(
            provided=True,
            source="geometrization_3d",
            exact=self.exact,
            validated=self.validated,
            method=self.theorem_tag,
            summary=self.summary or self.classification,
            assumptions=list(self.assumptions),
            payload=self.to_legacy_dict(),
        )

    def to_legacy_dict(self) -> dict[str, object]:
        """Return a dictionary representation for serialization.

        Returns:
            dict[str, object]: Dictionary representation.
        """
        return {
            "status": self.status,
            "classification": self.classification,
            "theorem": self.theorem,
            "theorem_tag": self.theorem_tag,
            "exact": self.exact,
            "validated": self.validated,
            "manifold": self.manifold.to_legacy_dict(),
            "homology": {
                int(dim): [int(rank), [int(x) for x in torsion]]
                for dim, (rank, torsion) in self.homology.items()
            },
            "candidates": [candidate.to_legacy_dict() for candidate in self.candidates],
            "prime_decomposition": self.prime_decomposition.to_legacy_dict()
            if self.prime_decomposition is not None
            else None,
            "jsj_decomposition": self.jsj_decomposition.to_legacy_dict()
            if self.jsj_decomposition is not None
            else None,
            "pi1_descriptor": self.pi1_descriptor,
            "assumptions": list(self.assumptions),
            "evidence": list(self.evidence),
            "missing_data": list(self.missing_data),
            "certificates": _freeze_value(self.certificates),
            "summary": self.summary,
            "decision_ready": self.decision_ready(),
        }


def _build_face_to_tetrahedra(
    tetrahedra: Sequence[tuple[int, int, int, int]],
) -> dict[tuple[int, int, int], tuple[int, ...]]:
    """Build a mapping from faces to indices of tetrahedra containing them.

    Args:
        tetrahedra (Sequence[tuple[int, int, int, int]]): List of tetrahedra.

    Returns:
        dict[tuple[int, int, int], tuple[int, ...]]: Face to tetrahedron indices.
    """
    face_map: dict[tuple[int, int, int], list[int]] = {}
    for tet_idx, tet in enumerate(tetrahedra):
        for face in combinations(tet, 3):
            face_map.setdefault(_sorted_face(face), []).append(tet_idx)
    return {face: tuple(indices) for face, indices in face_map.items()}


def _quad_type_for_local_edge(local_edge: tuple[int, int]) -> int:
    """Return the quadrilateral type index for a local edge pair.

    Args:
        local_edge (tuple[int, int]): Indices of local vertices in a tetrahedron.

    Returns:
        int: The quad type index (0, 1, or 2).

    Raises:
        ValueError: If the edge pair is invalid.
    """
    key = frozenset(local_edge)
    if key not in _QUAD_EDGE_PAIRS:
        raise ValueError(f"Invalid local edge for tetrahedron: {local_edge!r}")
    return _QUAD_EDGE_PAIRS[key]


def _piece_intersects_face(local_vertex_count: np.ndarray, local_quad_count: np.ndarray, face_local_vertices: tuple[int, int, int]) -> np.ndarray:
    """Determine which normal pieces in a tetrahedron intersect a given face.

    Args:
        local_vertex_count (np.ndarray): Dummy counts.
        local_quad_count (np.ndarray): Dummy counts.
        face_local_vertices (tuple[int, int, int]): Indices of vertices forming the face.

    Returns:
        np.ndarray: A binary mask of size 7 indicating intersection.
    """
    face_mask = np.zeros(7, dtype=np.int64)
    for idx in face_local_vertices:
        face_mask[idx] = 1
    face_mask[4:] = 1
    return face_mask


def _normal_surface_matching_matrix(manifold: Triangulated3Manifold) -> csr_matrix:
    """Build a sparse face-matching matrix for the canonical 7-coefficient normal coordinates.

    This matrix encodes a conservative combinatorial compatibility check: the
    count of normal pieces intersecting each paired face must agree on both
    sides of the face identification.

    Args:
        manifold (Triangulated3Manifold): The 3-manifold.

    Returns:
        csr_matrix: The face-matching matrix.
    """

    rows: list[int] = []
    cols: list[int] = []
    data: list[int] = []
    row = 0
    for face, tets in sorted(manifold.face_to_tetrahedra.items()):
        face_set = set(face)
        if len(tets) == 1:
            tet_idx = tets[0]
            tet = manifold.tetrahedra[tet_idx]
            local_face = tuple(i for i, v in enumerate(tet) if v in face_set)
            mask = _piece_intersects_face(np.zeros(4, dtype=np.int64), np.zeros(3, dtype=np.int64), local_face)
            for j, coeff in enumerate(mask):
                if coeff:
                    rows.append(row)
                    cols.append(7 * tet_idx + j)
                    data.append(int(coeff))
            row += 1
            continue
        if len(tets) != 2:
            # Non-manifold or degenerate face; skip but leave a note in validation.
            continue
        a, b = tets
        for tet_idx, sign in ((a, 1), (b, -1)):
            tet = manifold.tetrahedra[tet_idx]
            local_face = tuple(i for i, v in enumerate(tet) if v in face_set)
            mask = _piece_intersects_face(np.zeros(4, dtype=np.int64), np.zeros(3, dtype=np.int64), local_face)
            for j, coeff in enumerate(mask):
                if coeff:
                    rows.append(row)
                    cols.append(7 * tet_idx + j)
                    data.append(sign * int(coeff))
        row += 1
    shape = (max(row, 0), 7 * manifold.n_tetrahedra)
    return csr_matrix((data, (rows, cols)), shape=shape, dtype=np.int64)


def _normal_surface_matching_residual(
    manifold: Triangulated3Manifold,
    coords: np.ndarray,
    *,
    matching_matrix: Optional[csr_matrix] = None,
) -> float:
    """Compute the residual of the matching constraints for given coordinates.

    Args:
        manifold (Triangulated3Manifold): The manifold.
        coords (np.ndarray): Normal coordinates.
        matching_matrix (Optional[csr_matrix]): Optional pre-computed matching matrix.

    Returns:
        float: The L2 norm of the matching constraint violation.
    """
    matrix = matching_matrix if matching_matrix is not None else _normal_surface_matching_matrix(manifold)
    return _normal_surface_matching_residual_with_matrix(matrix, coords)


def _normal_surface_matching_residual_with_matrix(matrix: csr_matrix, coords: np.ndarray) -> float:
    """Compute matching residual given a matching matrix.

    Args:
        matrix (csr_matrix): The matching matrix.
        coords (np.ndarray): Normal coordinates.

    Returns:
        float: The L2 norm of the residual.
    """
    if matrix.shape[0] == 0:
        return 0.0
    resid = matrix @ np.asarray(coords, dtype=np.int64)
    return float(np.linalg.norm(np.asarray(resid, dtype=np.float64), ord=2))


def _normal_surface_quadrilateral_ok(coords: np.ndarray, n_tetrahedra: int) -> bool:
    """Check if the quadrilateral constraints are satisfied (at most one quad type per tet).

    Args:
        coords (np.ndarray): Normal coordinates.
        n_tetrahedra (int): Number of tetrahedra.

    Returns:
        bool: True if constraints are satisfied, False otherwise.
    """
    coords = np.asarray(coords, dtype=np.int64).reshape(n_tetrahedra, 7)
    quads = coords[:, 4:]
    for row in quads:
        if np.count_nonzero(row) > 1:
            return False
    return True


def normal_surface_matching_matrix(manifold: Triangulated3Manifold) -> csr_matrix:
    """Public API for building the normal-surface matching matrix.

    Args:
        manifold (Triangulated3Manifold): The manifold.

    Returns:
        csr_matrix: The matching matrix.
    """
    return _normal_surface_matching_matrix(manifold)


def normal_surface_candidates(
    manifold: Triangulated3Manifold, backend: str = "auto"
) -> list[NormalSurfaceCandidate]:
    """Generate a list of canonical normal-surface candidates.

    Includes vertex links, edge links, and surfaces derived from dual-graph cuts.

    Args:
        manifold (Triangulated3Manifold): The manifold.
        backend: 'auto', 'julia', or 'python'.

    Returns:
        list[NormalSurfaceCandidate]: List of validated candidates.
    """
    candidates: list[NormalSurfaceCandidate] = []
    seen: set[tuple[str, tuple[int, ...], str]] = set()
    matching_matrix = _normal_surface_matching_matrix(manifold)

    for vertex in manifold.simplicial_complex.n_simplices(0):
        cand = NormalSurfaceCandidate.vertex_link(
            manifold,
            int(vertex[0]),
            matching_matrix=matching_matrix,
        )
        key = (cand.kind, cand.support_tetrahedra, cand.surface_type)
        if key not in seen:
            candidates.append(cand)
            seen.add(key)

    for edge in manifold.edge_set():
        if len(edge) != 2:
            continue
        try:
            cand = NormalSurfaceCandidate.edge_link(
                manifold,
                edge,
                matching_matrix=matching_matrix,
            )
        except ValueError:
            continue
        key = (cand.kind, cand.support_tetrahedra, cand.surface_type)
        if key not in seen:
            candidates.append(cand)
            seen.add(key)

    for tet_idx in _dual_graph_articulation_points(manifold.dual_graph()):
        cand = NormalSurfaceCandidate.graph_cut(
            manifold,
            support_tetrahedra=[tet_idx],
            kind="sphere",
            surface_type="dual_graph_articulation",
            source="dual_graph_cut",
            matching_matrix=matching_matrix,
        )
        key = (cand.kind, cand.support_tetrahedra, cand.surface_type)
        if key not in seen:
            candidates.append(cand)
            seen.add(key)

    for a, b in _dual_graph_bridges(manifold.dual_graph()):
        cand = NormalSurfaceCandidate.graph_cut(
            manifold,
            support_tetrahedra=[a, b],
            kind="torus",
            surface_type="dual_graph_bridge",
            source="dual_graph_cut",
            matching_matrix=matching_matrix,
        )
        key = (cand.kind, cand.support_tetrahedra, cand.surface_type)
        if key not in seen:
            candidates.append(cand)
            seen.add(key)

    residuals = _batch_matching_residuals(
        matching_matrix,
        [candidate.coordinates for candidate in candidates],
        backend=backend,
    )
    validated: list[NormalSurfaceCandidate] = []
    for candidate, residual in zip(candidates, residuals):
        validated.append(
            _validate_normal_surface_candidate(
                manifold,
                candidate,
                matching_matrix=matching_matrix,
                residual=float(residual),
            )
        )
    return validated


def _batch_matching_residuals(
    matrix: csr_matrix,
    coordinates: Sequence[np.ndarray],
    backend: str = "auto",
) -> np.ndarray:
    """Compute matching residuals for a batch of coordinate vectors.

    Args:
        matrix (csr_matrix): Matching matrix.
        coordinates (Sequence[np.ndarray]): List of coordinate vectors.
        backend: 'auto', 'julia', or 'python'.

    Returns:
        np.ndarray: Vector of residual norms.
    """
    if not coordinates:
        return np.zeros(0, dtype=np.float64)
    if matrix.shape[0] == 0:
        return np.zeros(len(coordinates), dtype=np.float64)

    backend_norm = str(backend).lower().strip()
    use_julia = (backend_norm == "julia") or (backend_norm == "auto" and julia_engine.available)

    coord_matrix = np.column_stack([np.asarray(c, dtype=np.int64).reshape(-1) for c in coordinates])

    if use_julia:
        try:
            values = julia_engine.compute_normal_surface_residual_norms(matrix, coord_matrix)
            return np.asarray(values, dtype=np.float64).reshape(-1)
        except Exception as e:
            if backend_norm == "julia":
                raise e
            import warnings
            warnings.warn(f"Julia residual batch failed, falling back to Python: {e!r}")

    resid = matrix @ coord_matrix
    return np.linalg.norm(np.asarray(resid, dtype=np.float64), axis=0)


def _validate_normal_surface_candidate(
    manifold: Triangulated3Manifold,
    candidate: NormalSurfaceCandidate,
    *,
    matching_matrix: csr_matrix,
    residual: Optional[float] = None,
    tol: float = _MATCHING_RESIDUAL_TOL,
) -> NormalSurfaceCandidate:
    """Validate a normal-surface candidate against coordinate and matching constraints.

    Args:
        manifold (Triangulated3Manifold): The manifold.
        candidate (NormalSurfaceCandidate): The candidate to validate.
        matching_matrix (csr_matrix): Matching matrix.
        residual (Optional[float]): Optional pre-computed residual.
        tol (float): Tolerance for matching residuals. Defaults to 1e-12.

    Returns:
        NormalSurfaceCandidate: The validated (and possibly adjusted) candidate.
    """
    n_tet = manifold.n_tetrahedra
    notes = list(candidate.notes)
    exact = bool(candidate.exact)
    validated = True

    coords = np.asarray(candidate.coordinates, dtype=np.int64).reshape(-1)
    expected_size = 7 * n_tet
    if coords.size != expected_size:
        validated = False
        exact = False
        notes.append(f"Coordinate vector has size {coords.size}, expected {expected_size}.")

    tri = np.asarray(candidate.triangle_coordinates, dtype=np.int64)
    if tri.shape != (n_tet, 4):
        validated = False
        exact = False
        notes.append("Triangle coordinates have invalid shape.")
        tri = np.zeros((n_tet, 4), dtype=np.int64)

    quad = np.asarray(candidate.quad_coordinates, dtype=np.int64)
    if quad.shape != (n_tet, 3):
        validated = False
        exact = False
        notes.append("Quadrilateral coordinates have invalid shape.")
        quad = np.zeros((n_tet, 3), dtype=np.int64)

    if np.any(coords < 0) or np.any(tri < 0) or np.any(quad < 0):
        validated = False
        notes.append("Normal-surface coordinates must be nonnegative.")

    rebuilt = np.hstack((tri, quad)).reshape(-1)
    if coords.size == rebuilt.size and not np.array_equal(coords, rebuilt):
        validated = False
        notes.append("Flattened coordinates do not match triangle/quad blocks.")
        coords = rebuilt

    if coords.size != expected_size:
        pad = np.zeros(expected_size, dtype=np.int64)
        keep = min(expected_size, coords.size)
        if keep > 0:
            pad[:keep] = coords[:keep]
        coords = pad
        validated = False

    quadrilateral_ok = _normal_surface_quadrilateral_ok(coords, n_tet)
    if not quadrilateral_ok:
        validated = False
        notes.append("Quadrilateral constraints violated in at least one tetrahedron.")

    residual_value = (
        float(residual)
        if residual is not None
        else _normal_surface_matching_residual_with_matrix(matching_matrix, coords)
    )
    if residual_value > tol:
        validated = False
        notes.append(f"Matching residual {residual_value:.3e} exceeds tolerance {tol:.1e}.")

    support_from_coords = tuple(
        int(i)
        for i in np.where(np.any(coords.reshape(n_tet, 7) > 0, axis=1))[0].tolist()
    )
    support = tuple(
        sorted(
            int(i)
            for i in set(candidate.support_tetrahedra)
            if 0 <= int(i) < n_tet
        )
    )
    if support != support_from_coords:
        validated = False
        notes.append("Support tetrahedra adjusted to match nonzero coordinates.")

    if not support_from_coords:
        validated = False
        notes.append("Candidate has empty normal support.")

    if candidate.surface_type == "vertex_link" and candidate.kind == "sphere" and candidate.euler_characteristic != 2:
        validated = False
        notes.append("Vertex-link sphere must have Euler characteristic 2.")
    if candidate.surface_type == "edge_link" and candidate.kind == "torus" and candidate.euler_characteristic != 0:
        validated = False
        notes.append("Edge-link torus must have Euler characteristic 0.")

    exact = bool(exact and validated)

    return replace(
        candidate,
        exact=exact,
        validated=validated,
        coordinates=coords,
        triangle_coordinates=coords.reshape(n_tet, 7)[:, :4].copy(),
        quad_coordinates=coords.reshape(n_tet, 7)[:, 4:].copy(),
        support_tetrahedra=support_from_coords,
        matching_residual=float(residual_value),
        quadrilateral_ok=quadrilateral_ok,
        notes=notes,
    )


def _dual_graph_articulation_points(graph: dict[int, set[int]]) -> list[int]:
    """Find articulation points in the dual graph of the manifold.

    Args:
        graph (dict[int, set[int]]): Adjacency list of the dual graph.

    Returns:
        list[int]: Indices of tetrahedra which are articulation points.
    """
    index = 0
    indices: dict[int, int] = {}
    lowlink: dict[int, int] = {}
    stack: list[int] = []
    on_stack: set[int] = set()
    articulation: set[int] = set()

    def dfs(v: int, parent: Optional[int]) -> None:
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)
        children = 0
        for w in graph.get(v, set()):
            if w not in indices:
                children += 1
                dfs(w, v)
                lowlink[v] = min(lowlink[v], lowlink[w])
                if parent is not None and lowlink[w] >= indices[v]:
                    articulation.add(v)
            elif w in on_stack and w != parent:
                lowlink[v] = min(lowlink[v], indices[w])
        if parent is None and children > 1:
            articulation.add(v)
        stack.pop()
        on_stack.remove(v)

    for v in graph:
        if v not in indices:
            dfs(v, None)
    return sorted(articulation)


def _dual_graph_bridges(graph: dict[int, set[int]]) -> list[tuple[int, int]]:
    """Find bridges in the dual graph of the manifold.

    Args:
        graph (dict[int, set[int]]): Adjacency list of the dual graph.

    Returns:
        list[tuple[int, int]]: Pairs of tetrahedra indices forming bridges.
    """
    index = 0
    indices: dict[int, int] = {}
    lowlink: dict[int, int] = {}
    bridges: list[tuple[int, int]] = []

    def dfs(v: int, parent: Optional[int]) -> None:
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        for w in graph.get(v, set()):
            if w not in indices:
                dfs(w, v)
                lowlink[v] = min(lowlink[v], lowlink[w])
                if lowlink[w] > indices[v]:
                    bridges.append(tuple(sorted((v, w))))
            elif w != parent:
                lowlink[v] = min(lowlink[v], indices[w])

    for v in graph:
        if v not in indices:
            dfs(v, None)
    return sorted(set(bridges))


def crush_normal_surface(
    manifold: Triangulated3Manifold,
    surface: NormalSurfaceCandidate,
) -> PieceDecomposition:
    """Crush a manifold along a normal surface and decompose into components.

    References:
        Jaco, W., & Rubinstein, J. H. (2003). 0-efficient triangulations of 3-manifolds. 
        Journal of Differential Geometry, 65(1), 61-168.

    Args:
        manifold (Triangulated3Manifold): The manifold.
        surface (NormalSurfaceCandidate): The normal surface to cut along.

    Returns:
        PieceDecomposition: The resulting decomposition.
    """
    support = set(surface.support_tetrahedra)
    remaining = [i for i in range(manifold.n_tetrahedra) if i not in support]
    if not remaining:
        return PieceDecomposition(
            kind="crushed",
            exact=surface.exact,
            validated=surface.validated,
            pieces=[manifold],
            cut_surfaces=[surface],
            adjacency=[],
            support_tetrahedra=[list(surface.support_tetrahedra)],
            summary="Crushing removed the entire support; returning the original manifold as a fallback.",
            notes=["No residual tetrahedra remained after crushing."],
        )
    residual = manifold.submanifold(remaining, name=manifold.name + "_crushed")
    graph = residual.dual_graph()
    components = _connected_components(graph)
    pieces = [residual.submanifold(component, name=f"{residual.name}_piece_{i}") for i, component in enumerate(components)]
    adjacency = _graph_adjacency_from_components(graph)
    return PieceDecomposition(
        kind="crushed",
        exact=surface.exact,
        validated=surface.validated,
        pieces=pieces,
        cut_surfaces=[surface],
        adjacency=adjacency,
        support_tetrahedra=[list(surface.support_tetrahedra)],
        summary="Crushed along a candidate surface and decomposed into dual-graph components.",
        notes=["Graph-cut crushing heuristic"],
    )


def _connected_components(graph: dict[int, set[int]]) -> list[list[int]]:
    """Find connected components in an adjacency list graph.

    Args:
        graph (dict[int, set[int]]): The graph.

    Returns:
        list[list[int]]: List of component vertex sets.
    """
    seen: set[int] = set()
    components: list[list[int]] = []
    for start in graph:
        if start in seen:
            continue
        stack = [start]
        comp = []
        seen.add(start)
        while stack:
            v = stack.pop()
            comp.append(v)
            for w in graph.get(v, set()):
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        components.append(sorted(comp))
    return components


def _graph_adjacency_from_components(graph: dict[int, set[int]]) -> list[tuple[int, int]]:
    """Return all edges in the graph as sorted pairs.

    Args:
        graph (dict[int, set[int]]): The graph.

    Returns:
        list[tuple[int, int]]: List of edges.
    """
    edges = set()
    for a, neighbors in graph.items():
        for b in neighbors:
            if a != b:
                edges.add(tuple(sorted((a, b))))
    return sorted(edges)


def _choose_best_decomposition(
    manifold: Triangulated3Manifold,
    candidates: Sequence[NormalSurfaceCandidate],
    kind: str,
) -> PieceDecomposition:
    """Evaluate and choose the best decomposition among candidates.

    Args:
        manifold (Triangulated3Manifold): The manifold.
        candidates (Sequence[NormalSurfaceCandidate]): Candidate surfaces.
        kind (str): Target decomposition kind ("prime" or "jsj").

    Returns:
        PieceDecomposition: The selected best decomposition.
    """
    best: Optional[PieceDecomposition] = None
    best_score = -np.inf
    best_notes: list[str] = []
    for surface in candidates:
        if kind == "prime" and surface.kind != "sphere":
            continue
        if kind == "jsj" and surface.kind != "torus":
            continue
        decomposed = crush_normal_surface(manifold, surface)
        score, score_notes = _decomposition_score(manifold, surface, decomposed)
        current_tiebreak = (
            bool(surface.decision_ready()),
            bool(surface.validated),
            bool(surface.exact),
            int(len(decomposed.pieces)),
            float(-surface.matching_residual),
            int(-len(surface.support_tetrahedra)),
            tuple(int(x) for x in surface.support_tetrahedra),
        )
        best_tiebreak = None
        if best is not None and best.cut_surfaces:
            best_surface = best.cut_surfaces[0]
            best_tiebreak = (
                bool(best_surface.decision_ready()),
                bool(best_surface.validated),
                bool(best_surface.exact),
                int(len(best.pieces)),
                float(-best_surface.matching_residual),
                int(-len(best_surface.support_tetrahedra)),
                tuple(int(x) for x in best_surface.support_tetrahedra),
            )

        if (
            best is None
            or score > best_score
            or (np.isclose(score, best_score) and best_tiebreak is not None and current_tiebreak > best_tiebreak)
        ):
            best = decomposed
            best_score = float(score)
            best_notes = list(score_notes)
    if best is not None:
        best.kind = kind
        best.notes = list(best.notes) + [f"selection_score={best_score:.3f}"] + best_notes
        return best
    return PieceDecomposition(
        kind=kind,
        exact=True,
        validated=False,
        pieces=[manifold],
        cut_surfaces=[],
        adjacency=[],
        support_tetrahedra=[],
        summary="No combinatorial cut surface was found; returning the manifold as a single piece.",
        notes=["No split detected by the current heuristics."],
    )


def _decomposition_score(
    manifold: Triangulated3Manifold,
    surface: NormalSurfaceCandidate,
    decomposition: PieceDecomposition,
) -> tuple[float, list[str]]:
    """Compute a heuristic score for a decomposition.

    Args:
        manifold (Triangulated3Manifold): The original manifold.
        surface (NormalSurfaceCandidate): The surface used for cutting.
        decomposition (PieceDecomposition): The resulting decomposition.

    Returns:
        tuple[float, list[str]]: The score and a list of scoring notes.
    """
    n_pieces = len(decomposition.pieces)
    split_gain = 8.0 * max(0, n_pieces - 1)
    validation_bonus = 4.0 if surface.validated else -6.0
    exact_bonus = 2.0 if surface.exact else -1.5
    decision_bonus = 2.5 if surface.decision_ready() else -1.0
    residual_penalty = 8.0 * min(max(surface.matching_residual, 0.0), 1.0)

    total_tet = max(1, manifold.n_tetrahedra)
    support_ratio = float(len(surface.support_tetrahedra)) / float(total_tet)
    support_penalty = 2.0 * support_ratio

    balance_bonus = 0.0
    if n_pieces > 1:
        sizes = [max(1, piece.n_tetrahedra) for piece in decomposition.pieces]
        imbalance = (max(sizes) - min(sizes)) / max(sizes)
        balance_bonus = 3.0 * (1.0 - imbalance)

    score = (
        split_gain
        + validation_bonus
        + exact_bonus
        + decision_bonus
        + balance_bonus
        - residual_penalty
        - support_penalty
    )
    notes = [
        f"candidate_kind={surface.kind}",
        f"candidate_validated={surface.validated}",
        f"candidate_decision_ready={surface.decision_ready()}",
        f"pieces={n_pieces}",
        f"matching_residual={surface.matching_residual:.3e}",
    ]
    return float(score), notes


def prime_decomposition(manifold: Triangulated3Manifold, backend: str = "auto") -> PieceDecomposition:
    """Heuristically compute the prime decomposition of a 3-manifold.

    Args:
        manifold (Triangulated3Manifold): The manifold.
        backend: 'auto', 'julia', or 'python'.

    Returns:
        PieceDecomposition: The selected best prime decomposition.
    """
    return _choose_best_decomposition(
        manifold, 
        normal_surface_candidates(manifold, backend=backend), 
        kind="prime"
    )


def jsj_decomposition(manifold: Triangulated3Manifold, backend: str = "auto") -> PieceDecomposition:
    """Heuristically compute the JSJ decomposition of a 3-manifold.

    Args:
        manifold (Triangulated3Manifold): The manifold.
        backend: 'auto', 'julia', or 'python'.

    Returns:
        PieceDecomposition: The selected best JSJ decomposition.
    """
    return _choose_best_decomposition(
        manifold, 
        normal_surface_candidates(manifold, backend=backend), 
        kind="jsj"
    )


def _homology_sphere_like(manifold: Triangulated3Manifold) -> bool:
    """Check if the manifold has the homology of a 3-sphere.

    Args:
        manifold (Triangulated3Manifold): The manifold.

    Returns:
        bool: True if homology matches S^3, False otherwise.
    """
    h = manifold.homology()
    # A 3-manifold is homology-sphere-like if H_*(M; Z) ≅ H_*(S^3; Z)
    # This requires rank(H0)=1, rank(H3)=1, and H1=H2=0 (including NO torsion)
    is_h0 = (h[0][0] == 1 and not h[0][1])
    is_h1 = (h[1][0] == 0 and not h[1][1])
    is_h2 = (h[2][0] == 0 and not h[2][1])
    is_h3 = (h[3][0] == 1 and not h[3][1])
    return is_h0 and is_h1 and is_h2 and is_h3


def analyze_geometrization(
    manifold: Triangulated3Manifold | SimplicialComplex | Sequence[Sequence[int]],
    *,
    pi1_descriptor: Optional[str] = None,
    embedding_certificate: Optional[object] = None,
    allow_approx: bool = False,
    name: str = "triangulated_3_manifold",
    backend: str = "auto",
) -> GeometrizationResult:
    """Analyze a triangulated 3-manifold using conservative combinatorial heuristics.

    What is Being Computed?:
        An automated classification of the 3-manifold's geometry (e.g., spherical, 
        JSJ-decomposed, hyperbolic candidate) by evaluating homology, 
        fundamental group cues, and normal surface candidates.

    Algorithm:
        1. Coerce input into a Triangulated3Manifold.
        2. Compute homology groups across all degrees (0 to 3).
        3. Generate and validate canonical normal surface candidates (vertex/edge links).
        4. Perform heuristic prime and JSJ decompositions using graph-cut crushing.
        5. Infer geometric status based on homology and detected surfaces.
        6. Assemble results into a GeometrizationResult.

    Preserved Invariants:
        - Homology groups — Computed exactly or via Julia backend.
        - Euler characteristic — Fundamental 3-manifold invariant (usually 0).
        - Geometric type — Infers the Thurston geometry class if evidence is conclusive.

    Args:
        manifold: The manifold data (tetrahedra, complex, or object).
        pi1_descriptor: Known fundamental group info (e.g., "1" for trivial).
        embedding_certificate: Optional evidence of embedding.
        allow_approx: If True, allow success status even for heuristic decompositions.
        name: Name for the manifold.
        backend: Computation backend ('auto', 'julia', 'python').

    Returns:
        GeometrizationResult: Comprehensive analysis containing classification, 
                             evidence, and decomposition pieces.

    Use When:
        - You have a 3D triangulation and want to identify its topological type.
        - Checking if a manifold is a homology sphere or has an S^3 branch.
        - Preparing data for a 3-manifold homeomorphism certificate.

    Example:
        # Analyze a simple sphere triangulation
        result = analyze_geometrization(tetrahedra_list, pi1_descriptor="1")
        if result.status == "success":
            print(f"Manifold classified as: {result.classification}")
    """

    tri = _coerce_manifold(manifold, name=name)
    homology = {n: tri.homology(n, backend=backend) for n in range(4)}
    candidates = normal_surface_candidates(tri, backend=backend)
    prime = prime_decomposition(tri, backend=backend)
    jsj = jsj_decomposition(tri, backend=backend)
    theorem = "Geometrization / 3-manifold recognition"
    theorem_tag = infer_theorem_tag(theorem)

    evidence = [
        f"triangulation with {tri.n_tetrahedra} tetrahedra",
        f"{len(candidates)} canonical normal-surface candidates",
    ]
    missing_data: list[str] = []
    assumptions: list[str] = ["Closed connected 3-manifold hypotheses should be supplied externally."]
    classification = "inconclusive"
    status = "inconclusive"
    exact = False
    validated = False

    sphere_like = _homology_sphere_like(tri)
    has_torus = any(c.kind == "torus" and c.decision_ready() for c in candidates)
    has_sphere = any(c.kind == "sphere" and c.decision_ready() for c in candidates)
    validated_candidates = sum(1 for c in candidates if c.validated)
    evidence.append(f"{validated_candidates} candidates passed strict coordinate/matching validation")
    if sphere_like:
        evidence.append("Homology sphere checks passed")
    if has_sphere:
        evidence.append("Canonical sphere candidate detected")
    if has_torus:
        evidence.append("Canonical torus candidate detected")

    if sphere_like and pi1_descriptor == "1":
        classification = "spherical"
        status = "success"
        exact = True
        validated = True
        evidence.append("Trivial pi_1 descriptor supplied")
        summary = "Exact combinatorial sphere recognition: the manifold is consistent with a spherical 3-manifold / Poincaré branch."
    elif len(prime.pieces) > 1:
        classification = "prime_decomposed"
        status = "success" if allow_approx else "inconclusive"
        exact = False
        validated = False
        summary = "Prime decomposition detected by a graph-cut crushing heuristic."
        evidence.append(f"Prime cut produced {len(prime.pieces)} pieces")
    elif has_torus:
        classification = "jsj_decomposed"
        status = "success" if allow_approx else "inconclusive"
        exact = False
        validated = False
        summary = "JSJ-style torus candidates were found; a geometric classification is not certified."
        evidence.append("JSJ candidate surfaces present")
    else:
        if sphere_like:
            classification = "hyperbolic_candidate"
            summary = "Homology-sphere evidence is compatible with geometrization, but a geometric certificate is missing."
            missing_data.append("Decision-ready geometrization certificate")
        elif tri.euler_characteristic == 0:
            classification = "seifert_fibered_candidate"
            summary = "Torus-like combinatorics suggest a Seifert/graph-manifold candidate, but no certificate is provided."
            missing_data.append("Geometric decomposition certificate")
        else:
            classification = "hyperbolic_candidate"
            summary = "No decisive combinatorial cut surfaces were found; hyperbolic/irreducible status remains heuristic."
            missing_data.append("Geometric recognition certificate")
        status = "success" if allow_approx else "inconclusive"
        exact = False
        validated = False

    certificates = {
        "triangulated_manifold": tri.to_legacy_dict(),
        "normal_surface_candidates": [cand.to_legacy_dict() for cand in candidates],
        "matching_matrix_shape": list(normal_surface_matching_matrix(tri).shape),
        "prime_decomposition": prime.to_legacy_dict(),
        "jsj_decomposition": jsj.to_legacy_dict(),
        "homology": {
            int(dim): [int(rank), [int(x) for x in torsion]] for dim, (rank, torsion) in homology.items()
        },
    }
    if embedding_certificate is not None:
        certificates["embedding_certificate"] = (
            embedding_certificate.to_legacy_dict()
            if hasattr(embedding_certificate, "to_legacy_dict")
            else _freeze_value(embedding_certificate)
        )

    return GeometrizationResult(
        status=status,
        classification=classification,
        theorem=theorem,
        theorem_tag=theorem_tag,
        exact=exact,
        validated=validated,
        manifold=tri,
        homology=homology,
        candidates=candidates,
        prime_decomposition=prime,
        jsj_decomposition=jsj,
        pi1_descriptor=pi1_descriptor,
        assumptions=assumptions,
        evidence=evidence,
        missing_data=missing_data,
        certificates=certificates,
        summary=summary,
    )


def _coerce_manifold(
    manifold: Triangulated3Manifold | SimplicialComplex | Sequence[Sequence[int]],
    *,
    name: str = "triangulated_3_manifold",
) -> Triangulated3Manifold:
    """Coerce various input formats into a Triangulated3Manifold.

    Args:
        manifold (Triangulated3Manifold | SimplicialComplex | Sequence[Sequence[int]]):
            Input data.
        name (str): Name for the manifold. Defaults to "triangulated_3_manifold".

    Returns:
        Triangulated3Manifold: The coerced manifold object.
    """
    if isinstance(manifold, Triangulated3Manifold):
        return manifold
    if isinstance(manifold, SimplicialComplex):
        return Triangulated3Manifold.from_simplicial_complex(manifold, name=name)
    return Triangulated3Manifold.from_tetrahedra(manifold, name=name)


__all__ = [
    "GeometrizationResult",
    "NormalSurfaceCandidate",
    "PieceDecomposition",
    "Triangulated3Manifold",
    "analyze_geometrization",
    "crush_normal_surface",
    "jsj_decomposition",
    "normal_surface_candidates",
    "normal_surface_matching_matrix",
    "prime_decomposition",
]
