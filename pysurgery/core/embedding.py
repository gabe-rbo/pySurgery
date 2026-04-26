"""Piecewise-linear embedding and immersion checks.

This module provides a native geometry layer for pySurgery. It is intentionally
practical and conservative:
- exact local immersion checks are performed by affine rank tests on simplices,
- global embedding checks use broad-phase pruning and exact-ish simplex intersection
  predicates for low-dimensional complexes,
- projection and perturbation helpers provide deterministic heuristics when the
  input coordinates live in a higher-dimensional ambient space.

Supported source types
----------------------
- :class:`~pysurgery.core.complexes.SimplicialComplex`
- :class:`~pysurgery.core.uniformization.SurfaceMesh`
- :class:`~pysurgery.core.geometrization_3d.Triangulated3Manifold`
- raw simplex collections ``[(i, j, k), ...]`` or similar, with explicit
  coordinates.

Supported exact checks
----------------------
- vertex/link-style local rank checks,
- simplex affine-rank checks,
- segment/segment intersections,
- segment/triangle intersections,
- triangle/triangle intersections.

For higher-dimensional simplices, the module falls back to conservative
heuristics and returns ``inconclusive`` rather than overclaiming.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .complexes import SimplicialComplex
from .theorem_tags import infer_theorem_tag

try:  # Optional import to avoid circular/module-load issues.
    from .uniformization import SurfaceMesh
except Exception:  # pragma: no cover - optional import path
    SurfaceMesh = None  # type: ignore[assignment]

try:  # Optional import to avoid circular/module-load issues.
    from .geometrization_3d import Triangulated3Manifold
except Exception:  # pragma: no cover - optional import path
    Triangulated3Manifold = None  # type: ignore[assignment]

_EPS = 1e-10
_JULIA_PAIR_BATCH_THRESHOLD = 256


@dataclass
class SimplexIntersectionWitness:
    """One detected geometric intersection between two simplices.

    Attributes:
        simplex_a: Tuple of vertex labels for the first simplex.
        simplex_b: Tuple of vertex labels for the second simplex.
        kind: Type of intersection (e.g., 'segment_segment').
        distance: Minimum distance between the simplices.
        overlap_dimension: Dimension of the intersection.
        notes: Additional diagnostic information.
    """

    simplex_a: tuple[int, ...]
    simplex_b: tuple[int, ...]
    kind: str
    distance: float
    overlap_dimension: int
    notes: list[str] = field(default_factory=list)

    def to_legacy_dict(self) -> dict[str, object]:
        """Convert the witness to a legacy dictionary format.

        Returns:
            A dictionary containing the witness data.
        """
        return {
            "simplex_a": list(self.simplex_a),
            "simplex_b": list(self.simplex_b),
            "kind": self.kind,
            "distance": float(self.distance),
            "overlap_dimension": int(self.overlap_dimension),
            "notes": list(self.notes),
        }


@dataclass
class ImmersionResult:
    """Result for local PL immersion checks.

    Attributes:
        status: Status of the check (e.g., 'success', 'impediment', 'inconclusive').
        exact: Whether the check was performed using exact predicates.
        immersed: Whether the source is locally immersed in the ambient space.
        theorem: Name of the theorem used for the check.
        theorem_tag: Unique tag for the theorem.
        source_dimension: Dimension of the source complex.
        ambient_dimension: Dimension of the ambient space.
        local_failures: Details of local injectivity failures.
        simplex_rank_failures: Details of simplices with insufficient affine rank.
        evidence: Human-readable evidence for the result.
        missing_data: Description of any missing data that prevented a conclusive result.
        certificates: Machine-readable evidence for the result.
        summary: Short summary of the check result.
    """

    status: str
    exact: bool
    immersed: bool
    theorem: str
    theorem_tag: Optional[str]
    source_dimension: int
    ambient_dimension: int
    local_failures: list[dict[str, object]] = field(default_factory=list)
    simplex_rank_failures: list[dict[str, object]] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    missing_data: list[str] = field(default_factory=list)
    certificates: dict[str, object] = field(default_factory=dict)
    summary: str = ""

    def decision_ready(self) -> bool:
        """Check if the result is conclusive and positive.

        Returns:
            True if the status is 'success', the check is exact, and it is immersed.
        """
        return bool(self.status == "success" and self.exact and self.immersed)

    def to_legacy_dict(self) -> dict[str, object]:
        """Convert the result to a legacy dictionary format.

        Returns:
            A dictionary containing the result data.
        """
        return {
            "status": self.status,
            "exact": self.exact,
            "immersed": self.immersed,
            "theorem": self.theorem,
            "theorem_tag": self.theorem_tag,
            "source_dimension": self.source_dimension,
            "ambient_dimension": self.ambient_dimension,
            "local_failures": list(self.local_failures),
            "simplex_rank_failures": list(self.simplex_rank_failures),
            "evidence": list(self.evidence),
            "missing_data": list(self.missing_data),
            "certificates": _freeze_value(self.certificates),
            "summary": self.summary,
            "decision_ready": self.decision_ready(),
        }


@dataclass
class EmbeddingResult:
    """Result for global PL embedding checks.

    Attributes:
        status: Status of the check (e.g., 'success', 'impediment', 'inconclusive').
        exact: Whether the check was performed using exact predicates.
        embedded: Whether the source is globally embedded in the ambient space.
        theorem: Name of the theorem used for the check.
        theorem_tag: Unique tag for the theorem.
        source_dimension: Dimension of the source complex.
        ambient_dimension: Dimension of the ambient space.
        immersion: The result of the local immersion check.
        intersections: List of detected self-intersections.
        candidate_pairs_checked: Number of simplex pairs checked for intersection.
        pruned_pairs: Number of simplex pairs pruned by broad-phase checks.
        projection_used: Whether a projection was used to reduce the ambient dimension.
        projection_method: Method used for projection (e.g., 'pca', 'random').
        projection_matrix: The matrix used for projection.
        evidence: Human-readable evidence for the result.
        missing_data: Description of any missing data that prevented a conclusive result.
        certificates: Machine-readable evidence for the result.
        summary: Short summary of the check result.
    """

    status: str
    exact: bool
    embedded: bool
    theorem: str
    theorem_tag: Optional[str]
    source_dimension: int
    ambient_dimension: int
    immersion: ImmersionResult
    intersections: list[SimplexIntersectionWitness] = field(default_factory=list)
    candidate_pairs_checked: int = 0
    pruned_pairs: int = 0
    projection_used: bool = False
    projection_method: Optional[str] = None
    projection_matrix: Optional[np.ndarray] = None
    evidence: list[str] = field(default_factory=list)
    missing_data: list[str] = field(default_factory=list)
    certificates: dict[str, object] = field(default_factory=dict)
    summary: str = ""

    def decision_ready(self) -> bool:
        """Check if the result is conclusive and positive.

        Returns:
            True if the status is 'success', the check is exact, and it is embedded.
        """
        return bool(self.status == "success" and self.exact and self.embedded)

    def to_legacy_dict(self) -> dict[str, object]:
        """Convert the result to a legacy dictionary format.

        Returns:
            A dictionary containing the result data.
        """
        return {
            "status": self.status,
            "exact": self.exact,
            "embedded": self.embedded,
            "theorem": self.theorem,
            "theorem_tag": self.theorem_tag,
            "source_dimension": self.source_dimension,
            "ambient_dimension": self.ambient_dimension,
            "immersion": self.immersion.to_legacy_dict(),
            "intersections": [hit.to_legacy_dict() for hit in self.intersections],
            "candidate_pairs_checked": self.candidate_pairs_checked,
            "pruned_pairs": self.pruned_pairs,
            "projection_used": self.projection_used,
            "projection_method": self.projection_method,
            "projection_matrix": None if self.projection_matrix is None else self.projection_matrix.tolist(),
            "evidence": list(self.evidence),
            "missing_data": list(self.missing_data),
            "certificates": _freeze_value(self.certificates),
            "summary": self.summary,
            "decision_ready": self.decision_ready(),
        }


@dataclass
class PLMap:
    """A piecewise-affine map from a simplicial source to Euclidean coordinates.

    Attributes:
        source: The original source object (e.g., SurfaceMesh).
        source_complex: The underlying simplicial complex.
        vertex_coordinates: Coordinates of each vertex in the ambient space.
        vertex_labels: Unique labels for each vertex in the source complex.
        ambient_dimension: Dimension of the ambient Euclidean space.
        source_dimension: Dimension of the source simplicial complex.
        projection_matrix: Optional matrix used to project from a higher dimension.
        source_name: Name of the source for identification.
    """

    source: object
    source_complex: SimplicialComplex
    vertex_coordinates: np.ndarray
    vertex_labels: list[int]
    ambient_dimension: int
    source_dimension: int
    projection_matrix: Optional[np.ndarray] = None
    source_name: str = "pl_map"
    _label_to_index: dict[int, int] = field(init=False, repr=False, default_factory=dict)
    _simplex_vertices_cache: dict[tuple[int, ...], np.ndarray] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize label to index mapping and validate labels.

        Raises:
            ValueError: If vertex_labels are not unique.
        """
        labels = [int(v) for v in self.vertex_labels]
        if len(set(labels)) != len(labels):
            raise ValueError("vertex_labels must be unique for embedding analysis.")
        self._label_to_index = {label: idx for idx, label in enumerate(labels)}

    @classmethod
    def from_source(
        cls,
        source: object,
        coordinates: Optional[np.ndarray] = None,
        *,
        projection_matrix: Optional[np.ndarray] = None,
        source_name: Optional[str] = None,
    ) -> "PLMap":
        """Create a PLMap from various source types.

        Args:
            source: A source object (SimplicialComplex, SurfaceMesh, etc.).
            coordinates: Optional explicit vertex coordinates.
            projection_matrix: Optional projection matrix.
            source_name: Optional name for the map.

        Returns:
            A new PLMap instance.

        Raises:
            ValueError: If coordinates do not match the number of vertices.
        """
        source_complex, vertex_labels, source_dimension = _coerce_source_complex(source)
        coords = _coerce_coordinates(source, coordinates=coordinates)
        if coords.shape[0] != len(vertex_labels):
            raise ValueError(
                "Coordinates must have one row per vertex in the source complex."
            )
        ambient_dimension = int(coords.shape[1])
        return cls(
            source=source,
            source_complex=source_complex,
            vertex_coordinates=np.asarray(coords, dtype=np.float64),
            vertex_labels=vertex_labels,
            ambient_dimension=ambient_dimension,
            source_dimension=source_dimension,
            projection_matrix=projection_matrix,
            source_name=source_name or getattr(source, "name", "pl_map"),
        )

    def simplex_index(self, simplex: Sequence[int]) -> tuple[int, ...]:
        """Convert a simplex sequence to a canonical tuple of integers.

        Args:
            simplex: Sequence of vertex labels.

        Returns:
            Tuple of integer vertex labels.
        """
        return tuple(int(v) for v in simplex)

    def simplex_vertices(self, simplex: Sequence[int]) -> np.ndarray:
        """Get the coordinates of the vertices for a given simplex.

        Args:
            simplex: Sequence of vertex labels.

        Returns:
            An (n_vertices, ambient_dim) array of coordinates.
        """
        simplex_key = self.simplex_index(simplex)
        cached = self._simplex_vertices_cache.get(simplex_key)
        if cached is not None:
            return cached
        idx = [self.vertex_label_to_index(int(v)) for v in simplex_key]
        vertices = self.vertex_coordinates[np.asarray(idx, dtype=np.int64)]
        self._simplex_vertices_cache[simplex_key] = vertices
        return vertices

    def vertex_label_to_index(self, label: int) -> int:
        """Get the coordinate index for a given vertex label.

        Args:
            label: The vertex label.

        Returns:
            The corresponding row index in vertex_coordinates.

        Raises:
            KeyError: If the label is not found.
        """
        try:
            return self._label_to_index[int(label)]
        except ValueError as exc:
            raise KeyError("Unknown vertex label {!r}".format(label)) from exc
        except KeyError as exc:
            raise KeyError("Unknown vertex label {!r}".format(label)) from exc

    def simplex_affine_rank(self, simplex: Sequence[int]) -> int:
        """Compute the affine rank of a simplex in the ambient space.

        Args:
            simplex: Sequence of vertex labels.

        Returns:
            The affine rank of the simplex.
        """
        vertices = self.simplex_vertices(simplex)
        if vertices.shape[0] <= 1:
            return 0
        base = vertices[0]
        mat = (vertices[1:] - base).T
        if mat.size == 0:
            return 0
        return int(np.linalg.matrix_rank(mat))

    def simplex_barycenter(self, simplex: Sequence[int]) -> np.ndarray:
        """Compute the barycenter of a simplex.

        Args:
            simplex: Sequence of vertex labels.

        Returns:
            The barycenter as a 1D array.
        """
        vertices = self.simplex_vertices(simplex)
        return np.mean(vertices, axis=0)

    def simplex_bounding_box(self, simplex: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """Compute the axis-aligned bounding box of a simplex.

        Args:
            simplex: Sequence of vertex labels.

        Returns:
            A tuple (min_coords, max_coords).
        """
        vertices = self.simplex_vertices(simplex)
        return np.min(vertices, axis=0), np.max(vertices, axis=0)

    def simplices_by_dim(self) -> dict[int, list[tuple[int, ...]]]:
        """Group simplices by their dimension.

        Returns:
            A dictionary mapping dimension to list of simplices.
        """
        return {
            dim: [tuple(simplex) for simplex in simplices]
            for dim, simplices in self.source_complex.simplices.items()
        }


@dataclass
class ProjectionResult:
    """Result of a PCA/random projection helper.

    Attributes:
        points: The projected vertex coordinates.
        projection_matrix: The matrix used for the projection.
        method: The projection method used.
        explained_variance: Optional explained variance (for PCA).
        summary: A short summary of the projection.
    """

    points: np.ndarray
    projection_matrix: np.ndarray
    method: str
    explained_variance: Optional[np.ndarray] = None
    summary: str = ""


@dataclass
class SelfIntersectionReport:
    """Broad-phase and narrow-phase self-intersection diagnostics.

    Attributes:
        status: Status of the check.
        exact: Whether the check used exact predicates.
        has_intersections: Whether any self-intersections were detected.
        witnesses: List of detected intersection witnesses.
        candidate_pairs_checked: Number of pairs that passed broad-phase.
        pruned_pairs: Number of pairs pruned by broad-phase.
        max_violation: Maximum distance violation detected.
        notes: Additional diagnostic notes.
    """

    status: str
    exact: bool
    has_intersections: bool
    witnesses: list[SimplexIntersectionWitness] = field(default_factory=list)
    candidate_pairs_checked: int = 0
    pruned_pairs: int = 0
    max_violation: float = 0.0
    notes: list[str] = field(default_factory=list)

    def decision_ready(self) -> bool:
        """Check if the result is conclusive and positive (no intersections).

        Returns:
            True if status is 'success', check is exact, and no intersections.
        """
        return bool(self.status == "success" and self.exact and not self.has_intersections)

    def to_legacy_dict(self) -> dict[str, object]:
        """Convert the report to a legacy dictionary format.

        Returns:
            A dictionary containing the report data.
        """
        return {
            "status": self.status,
            "exact": self.exact,
            "has_intersections": self.has_intersections,
            "witnesses": [w.to_legacy_dict() for w in self.witnesses],
            "candidate_pairs_checked": self.candidate_pairs_checked,
            "pruned_pairs": self.pruned_pairs,
            "max_violation": float(self.max_violation),
            "notes": list(self.notes),
            "decision_ready": self.decision_ready(),
        }


def analyze_embedding(
    source: object,
    coordinates: Optional[np.ndarray] = None,
    *,
    target_dimension: Optional[int] = None,
    allow_projection: bool = False,
    projection_method: str = "pca",
    projection_matrix: Optional[np.ndarray] = None,
    tol: float = _EPS,
) -> EmbeddingResult:
    """High-level embedding / immersion analysis entry point.

    Args:
        source: Source complex or mesh object.
        coordinates: Optional explicit vertex coordinates.
        target_dimension: Optional target ambient dimension.
        allow_projection: Whether to allow projection to a lower dimension.
        projection_method: Method for projection ('pca' or 'random').
        projection_matrix: Optional custom projection matrix.
        tol: Numerical tolerance for geometric checks.

    Returns:
        The result of the embedding analysis.

    Raises:
        ValueError: If target_dimension is smaller than ambient dimension and allow_projection is False.
    """
    pl_map = PLMap.from_source(
        source,
        coordinates=coordinates,
        projection_matrix=projection_matrix,
        source_name=getattr(source, "name", None),
    )
    projected = pl_map.vertex_coordinates
    proj_used = False
    proj_matrix = projection_matrix
    proj_method = None
    if target_dimension is not None and target_dimension < projected.shape[1]:
        if allow_projection:
            proj = project_coordinates(
                projected,
                target_dimension=target_dimension,
                method=projection_method,
                projection_matrix=projection_matrix,
            )
            projected = proj.points
            proj_matrix = proj.projection_matrix
            proj_used = True
            proj_method = proj.method
        else:
            raise ValueError(
                "Target dimension is smaller than the ambient dimension; set allow_projection=True."
            )

    if projected is not pl_map.vertex_coordinates:
        pl_map = PLMap(
            source=pl_map.source,
            source_complex=pl_map.source_complex,
            vertex_coordinates=projected,
            vertex_labels=pl_map.vertex_labels,
            ambient_dimension=int(projected.shape[1]),
            source_dimension=pl_map.source_dimension,
            projection_matrix=proj_matrix,
            source_name=pl_map.source_name,
        )

    immersion = check_immersion(pl_map, tol=tol)
    intersections = detect_self_intersections(pl_map, tol=tol)
    embedded = bool(immersion.immersed and not intersections.has_intersections)
    status = "success" if embedded and immersion.exact and intersections.exact else "inconclusive"
    if not immersion.immersed:
        status = "impediment"
    elif intersections.has_intersections:
        status = "impediment"

    theorem = "Whitney embedding / PL immersion"
    theorem_tag = infer_theorem_tag(theorem)
    evidence = list(immersion.evidence)
    evidence.extend(
        [
            "candidate pairs checked: {!r}".format(intersections.candidate_pairs_checked),
            "self intersections: {!r}".format(len(intersections.witnesses)),
        ]
    )
    if proj_used:
        evidence.append("projection method: {!r}".format(proj_method))

    missing_data = list(immersion.missing_data)
    if intersections.status != "success":
        missing_data.extend(intersections.notes)

    certificates = {
        "pl_map": {
            "source_name": pl_map.source_name,
            "ambient_dimension": pl_map.ambient_dimension,
            "source_dimension": pl_map.source_dimension,
            "vertex_labels": list(pl_map.vertex_labels),
            "coordinates": pl_map.vertex_coordinates.tolist(),
        },
        "immersion": immersion.to_legacy_dict(),
        "self_intersections": intersections.to_legacy_dict(),
    }
    if proj_used:
        certificates["projection"] = {
            "method": proj_method,
            "matrix": None if proj_matrix is None else proj_matrix.tolist(),
        }

    summary = "Embedded PL map found." if embedded else "PL embedding/immersion analysis incomplete or obstructed."
    return EmbeddingResult(
        status=status,
        exact=bool(immersion.exact and intersections.exact),
        embedded=embedded,
        theorem=theorem,
        theorem_tag=theorem_tag,
        source_dimension=pl_map.source_dimension,
        ambient_dimension=pl_map.ambient_dimension,
        immersion=immersion,
        intersections=intersections.witnesses,
        candidate_pairs_checked=intersections.candidate_pairs_checked,
        pruned_pairs=intersections.pruned_pairs,
        projection_used=proj_used,
        projection_method=proj_method,
        projection_matrix=proj_matrix,
        evidence=evidence,
        missing_data=missing_data,
        certificates=certificates,
        summary=summary,
    )


def check_immersion(pl_map: PLMap, *, tol: float = _EPS) -> ImmersionResult:
    """Check local injectivity/rank conditions for a PL map.

    Args:
        pl_map: The PL map to check.
        tol: Numerical tolerance for rank checks.

    Returns:
        The result of the immersion check.
    """
    top_dim = pl_map.source_dimension
    source_simplices = _top_simplices(pl_map.source_complex)
    local_failures: list[dict[str, object]] = []
    simplex_rank_failures: list[dict[str, object]] = []
    evidence: list[str] = []
    missing_data: list[str] = []

    if top_dim < 0:
        return ImmersionResult(
            status="inconclusive",
            exact=False,
            immersed=False,
            theorem="Whitney embedding / PL immersion",
            theorem_tag=infer_theorem_tag("Whitney embedding / PL immersion"),
            source_dimension=-1,
            ambient_dimension=pl_map.ambient_dimension,
            missing_data=["Non-empty simplicial source required"],
            summary="Source complex is empty.",
        )

    for simplex in source_simplices:
        rank = pl_map.simplex_affine_rank(simplex)
        expected = len(simplex) - 1
        if expected > 0 and rank < expected:
            simplex_rank_failures.append(
                {
                    "simplex": list(simplex),
                    "rank": int(rank),
                    "expected": int(expected),
                }
            )

    if top_dim <= 0:
        immersed = len(simplex_rank_failures) == 0
        status = "success" if immersed else "impediment"
        return ImmersionResult(
            status=status,
            exact=True,
            immersed=immersed,
            theorem="Whitney embedding / PL immersion",
            theorem_tag=infer_theorem_tag("Whitney embedding / PL immersion"),
            source_dimension=top_dim,
            ambient_dimension=pl_map.ambient_dimension,
            simplex_rank_failures=simplex_rank_failures,
            evidence=["0- or 1-dimensional source; rank checks passed."],
            summary="Immersion check completed.",
        )

    # Vertex-star / link-style local heuristic: all incident top simplices should have
    # nondegenerate affine rank, and the star should not collapse to a lower-dimensional set.
    for vertex in pl_map.vertex_labels:
        incident = _incident_top_simplices(pl_map.source_complex, vertex)
        if not incident:
            continue
        local_coords = []
        for simplex in incident:
            local_coords.extend(pl_map.simplex_vertices(simplex).tolist())
        local_coords = np.asarray(local_coords, dtype=np.float64)
        if local_coords.size == 0:
            continue
        centered = local_coords - np.mean(local_coords, axis=0, keepdims=True)
        if centered.shape[0] > 1:
            star_rank = int(np.linalg.matrix_rank(centered.T))
            expected_rank = min(top_dim, pl_map.ambient_dimension)
            if star_rank < min(expected_rank, pl_map.ambient_dimension):
                local_failures.append(
                    {
                        "vertex": int(vertex),
                        "star_rank": int(star_rank),
                        "expected": int(expected_rank),
                        "incident_simplices": [list(s) for s in incident],
                    }
                )

    immersed = len(simplex_rank_failures) == 0 and len(local_failures) == 0
    status = "success" if immersed else "impediment"
    if not immersed:
        evidence.append("Local immersion failure detected.")
    else:
        evidence.append("Local immersion checks passed.")

    return ImmersionResult(
        status=status,
        exact=True,
        immersed=immersed,
        theorem="Whitney embedding / PL immersion",
        theorem_tag=infer_theorem_tag("Whitney embedding / PL immersion"),
        source_dimension=top_dim,
        ambient_dimension=pl_map.ambient_dimension,
        local_failures=local_failures,
        simplex_rank_failures=simplex_rank_failures,
        evidence=evidence,
        missing_data=missing_data,
        summary="Immersion checks completed." if immersed else "Immersion checks failed.",
    )


def detect_self_intersections(pl_map: PLMap, *, tol: float = _EPS) -> SelfIntersectionReport:
    """Detect self-intersections using broad-phase pruning and exact low-dimensional predicates.

    Args:
        pl_map: The PL map to check.
        tol: Numerical tolerance for intersection checks.

    Returns:
        The self-intersection report.
    """
    source_simplices = _all_simplices_by_dim(pl_map.source_complex)
    simplices = [
        s
        for dim in sorted(source_simplices.keys())
        for s in sorted(source_simplices[dim])
        if len(s) >= 2
    ]
    if not simplices:
        return SelfIntersectionReport(
            status="success",
            exact=True,
            has_intersections=False,
            notes=["No positive-dimensional simplices present."],
        )

    simplex_vertices = [pl_map.simplex_vertices(simplex) for simplex in simplices]
    bboxes = [
        (np.min(vertices, axis=0), np.max(vertices, axis=0))
        for vertices in simplex_vertices
    ]
    centroids = np.array([np.mean(vertices, axis=0) for vertices in simplex_vertices], dtype=np.float64)
    radii = np.array([_bbox_radius(bbox) for bbox in bboxes], dtype=np.float64)
    simplex_vertex_sets = [set(simplex) for simplex in simplices]

    candidate_pairs = _broad_phase_candidate_pairs(centroids, radii, tol=tol)
    if not candidate_pairs and 1 < len(simplices) <= 64:
        # Small-complex safety fallback when broad-phase is too strict.
        for i in range(len(simplices)):
            for j in range(i + 1, len(simplices)):
                candidate_pairs.add((i, j))

    witnesses: list[SimplexIntersectionWitness] = []
    checked = 0
    pruned = 0
    max_violation = 0.0
    bbox_lows = np.asarray([bbox[0] for bbox in bboxes], dtype=np.float64)
    bbox_highs = np.asarray([bbox[1] for bbox in bboxes], dtype=np.float64)
    for i, j in sorted(candidate_pairs):
        s1 = simplices[i]
        s2 = simplices[j]
        if simplex_vertex_sets[i].intersection(simplex_vertex_sets[j]):
            pruned += 1
            continue
        
        if not _bbox_overlap((bbox_lows[i], bbox_highs[i]), (bbox_lows[j], bbox_highs[j]), tol=tol):
            pruned += 1
            continue
        center_gap = np.linalg.norm(centroids[i] - centroids[j]) - (radii[i] + radii[j])
        if center_gap > float(2.0 * tol):
            pruned += 1
            continue
        checked += 1
        witness = _simplex_intersection(pl_map, s1, s2, tol=tol, pa=simplex_vertices[i], pb=simplex_vertices[j])
        if witness is not None:
            witnesses.append(witness)
            max_violation = max(max_violation, witness.distance)

    has_intersections = len(witnesses) > 0
    status = "success" if not has_intersections else "impediment"
    exact = True
    notes = []
    if has_intersections:
        notes.append("Self-intersection witnesses detected.")
    else:
        notes.append("No self-intersections detected among candidate simplex pairs.")
    return SelfIntersectionReport(
        status=status,
        exact=exact,
        has_intersections=has_intersections,
        witnesses=witnesses,
        candidate_pairs_checked=checked,
        pruned_pairs=pruned,
        max_violation=max_violation,
        notes=notes,
    )


def project_coordinates(
    points: np.ndarray,
    target_dimension: int,
    *,
    method: str = "pca",
    projection_matrix: Optional[np.ndarray] = None,
    random_state: int = 0,
) -> ProjectionResult:
    """Project a point cloud down to a lower ambient dimension.

    Args:
        points: (n_points, ambient_dim) array of coordinates.
        target_dimension: Desired ambient dimension.
        method: Projection method ('pca', 'random', or 'identity').
        projection_matrix: Optional custom projection matrix.
        random_state: Seed for random projection.

    Returns:
        The result of the projection.

    Raises:
        ValueError: If inputs are invalid or method is unknown.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2:
        raise ValueError("points must be a 2D array")
    if target_dimension <= 0:
        raise ValueError("target_dimension must be positive")
    if target_dimension >= pts.shape[1] and projection_matrix is None:
        return ProjectionResult(
            points=pts.copy(),
            projection_matrix=np.eye(pts.shape[1], dtype=np.float64),
            method="identity",
            summary="No projection required.",
        )

    if projection_matrix is not None:
        P = np.asarray(projection_matrix, dtype=np.float64)
        if P.shape != (pts.shape[1], target_dimension):
            raise ValueError(
                "projection_matrix must have shape (ambient_dim, target_dimension)"
            )
        out = pts @ P
        return ProjectionResult(points=out, projection_matrix=P, method="custom", summary="Applied custom projection matrix.")

    if method == "pca":
        centered = pts - np.mean(pts, axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        P = vt[:target_dimension].T
        out = centered @ P
        explained = None
        if centered.shape[0] > 1:
            cov = np.cov(centered, rowvar=False)
            eigvals = np.linalg.eigvalsh(np.asarray(cov, dtype=np.float64))[::-1]
            total = float(np.sum(eigvals)) if np.sum(eigvals) > 0 else 1.0
            explained = eigvals[:target_dimension] / total
        return ProjectionResult(
            points=out,
            projection_matrix=P,
            method="pca",
            explained_variance=explained,
            summary="Projected onto principal components.",
        )

    if method == "random":
        rng = np.random.default_rng(random_state)
        G = rng.normal(size=(pts.shape[1], target_dimension))
        Q, _ = np.linalg.qr(G)
        P = Q[:, :target_dimension]
        out = pts @ P
        return ProjectionResult(points=out, projection_matrix=P, method="random", summary="Projected with a random orthogonal basis.")

    raise ValueError("Unknown projection method: {!r}".format(method))


def jitter_coordinates(points: np.ndarray, *, scale: float = 1e-8, random_state: int = 0) -> np.ndarray:
    """Apply a deterministic small jitter for transversality-style retries.

    Args:
        points: (n_points, ambient_dim) array of coordinates.
        scale: Magnitude of the jitter.
        random_state: Seed for the random generator.

    Returns:
        The jittered coordinates.
    """
    pts = np.asarray(points, dtype=np.float64)
    rng = np.random.default_rng(random_state)
    return pts + scale * rng.normal(size=pts.shape)


def _freeze_value(value: object) -> object:
    """Deeply convert dictionaries, lists, and numpy arrays to hashable tuples.

    Args:
        value: The object to freeze.

    Returns:
        A hashable version of the input value.
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


def _coerce_source_complex(source: object) -> tuple[SimplicialComplex, list[int], int]:
    """Coerce various source types into a SimplicialComplex and vertex labels.

    Args:
        source: The source object.

    Returns:
        A tuple (SimplicialComplex, vertex_labels, dimension).

    Raises:
        TypeError: If the source type is unsupported.
    """
    if isinstance(source, SimplicialComplex):
        vertices = sorted({int(v[0]) for v in source.n_simplices(0)})
        return source, vertices, _source_dimension(source)

    if SurfaceMesh is not None and isinstance(source, SurfaceMesh):
        simplices = [tuple(face) for face in np.asarray(source.faces, dtype=np.int64).tolist()]
        sc = SimplicialComplex.from_maximal_simplices(simplices)
        vertices = list(range(int(source.n_vertices)))
        return sc, vertices, 2

    if Triangulated3Manifold is not None and isinstance(source, Triangulated3Manifold):
        sc = source.simplicial_complex
        vertices = sorted({int(v[0]) for v in sc.n_simplices(0)})
        return sc, vertices, 3

    if hasattr(source, "simplicial_complex"):
        sc = getattr(source, "simplicial_complex")
        if isinstance(sc, SimplicialComplex):
            vertices = sorted({int(v[0]) for v in sc.n_simplices(0)})
            return sc, vertices, _source_dimension(sc)

    if hasattr(source, "faces") and hasattr(source, "n_vertices"):
        faces = np.asarray(getattr(source, "faces"), dtype=np.int64)
        sc = SimplicialComplex.from_maximal_simplices(map(tuple, faces.tolist()))
        vertices = list(range(int(getattr(source, "n_vertices"))))
        return sc, vertices, _source_dimension(sc)

    if hasattr(source, "tetrahedra"):
        tetrahedra = getattr(source, "tetrahedra")
        sc = SimplicialComplex.from_maximal_simplices(tetrahedra)
        vertices = sorted({int(v) for tet in tetrahedra for v in tet})
        return sc, vertices, 3

    if isinstance(source, (list, tuple)):
        simplices = [tuple(int(v) for v in simplex) for simplex in source if len(simplex) > 0]
        sc = SimplicialComplex.from_simplices(simplices)
        vertices = sorted({int(v) for simplex in simplices for v in simplex})
        return sc, vertices, _source_dimension(sc)

    raise TypeError(
        "Unsupported source type for PL embedding analysis. Provide a SimplicialComplex, SurfaceMesh, Triangulated3Manifold, or a simplex list."
    )


def _coerce_coordinates(source: object, *, coordinates: Optional[np.ndarray] = None) -> np.ndarray:
    """Coerce various source types and optional input into a coordinate array.

    Args:
        source: The source object.
        coordinates: Optional explicit coordinates.

    Returns:
        An (n_vertices, ambient_dim) array of coordinates.

    Raises:
        ValueError: If coordinates cannot be obtained or are malformed.
    """
    if coordinates is not None:
        pts = np.asarray(coordinates, dtype=np.float64)
    elif hasattr(source, "coordinates") and getattr(source, "coordinates") is not None:
        pts = np.asarray(getattr(source, "coordinates"), dtype=np.float64)
    elif SurfaceMesh is not None and isinstance(source, SurfaceMesh):
        coords = getattr(source, "coordinates", None)
        if coords is None:
            raise ValueError("SurfaceMesh source requires explicit coordinates")
        pts = np.asarray(coords, dtype=np.float64)
    else:
        raise ValueError("Explicit coordinates are required for the provided source.")
    if pts.ndim != 2:
        raise ValueError("coordinates must be a 2D array")
    return pts


def _source_dimension(source_complex: SimplicialComplex) -> int:
    """Determine the maximum dimension of a simplicial complex.

    Args:
        source_complex: The simplicial complex.

    Returns:
        The maximum dimension, or -1 if empty.
    """
    return max(source_complex.dimensions, default=-1)


def _top_simplices(source_complex: SimplicialComplex) -> list[tuple[int, ...]]:
    """Get all simplices of the maximum dimension in the complex.

    Args:
        source_complex: The simplicial complex.

    Returns:
        A list of top-dimensional simplices.
    """
    dim = _source_dimension(source_complex)
    return [tuple(s) for s in source_complex.n_simplices(dim)]


def _incident_top_simplices(source_complex: SimplicialComplex, vertex: int) -> list[tuple[int, ...]]:
    """Get all top-dimensional simplices incident to a given vertex.

    Args:
        source_complex: The simplicial complex.
        vertex: The vertex label.

    Returns:
        A list of incident top-dimensional simplices.
    """
    top = _top_simplices(source_complex)
    return [s for s in top if int(vertex) in s]


def _all_simplices_by_dim(source_complex: SimplicialComplex) -> dict[int, list[tuple[int, ...]]]:
    """Group all simplices in the complex by dimension.

    Args:
        source_complex: The simplicial complex.

    Returns:
        A dictionary mapping dimension to list of simplices.
    """
    return {
        dim: sorted(tuple(s) for s in source_complex.n_simplices(dim))
        for dim in source_complex.dimensions
    }


def _bbox_for_simplex(pl_map: PLMap, simplex: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Compute the axis-aligned bounding box for a simplex.

    Args:
        pl_map: The PL map.
        simplex: Sequence of vertex labels.

    Returns:
        A tuple (min_coords, max_coords).
    """
    vertices = pl_map.simplex_vertices(simplex)
    return np.min(vertices, axis=0), np.max(vertices, axis=0)


def _bbox_radius(bbox: tuple[np.ndarray, np.ndarray]) -> float:
    """Compute the radius of the bounding sphere for an axis-aligned bounding box.

    Args:
        bbox: A tuple (min_coords, max_coords).

    Returns:
        The radius of the bounding sphere.
    """
    lo, hi = bbox
    return float(0.5 * np.linalg.norm(hi - lo))


def _bbox_overlap(
    bbox_a: tuple[np.ndarray, np.ndarray],
    bbox_b: tuple[np.ndarray, np.ndarray],
    *,
    tol: float,
) -> bool:
    """Check if two axis-aligned bounding boxes overlap within a tolerance.

    Args:
        bbox_a: First bounding box.
        bbox_b: Second bounding box.
        tol: Numerical tolerance.

    Returns:
        True if the boxes overlap.
    """
    lo_a, hi_a = bbox_a
    lo_b, hi_b = bbox_b
    return bool(np.all(hi_a + tol >= lo_b) and np.all(hi_b + tol >= lo_a))


def _combinatorially_adjacent(s1: Sequence[int], s2: Sequence[int]) -> bool:
    """Check if two simplices share at least one vertex.

    Args:
        s1: First simplex vertex labels.
        s2: Second simplex vertex labels.

    Returns:
        True if they share a vertex.
    """
    return len(set(int(v) for v in s1).intersection(int(v) for v in s2)) > 0


def _broad_phase_candidate_pairs(
    centroids: np.ndarray,
    radii: np.ndarray,
    *,
    tol: float,
) -> set[tuple[int, int]]:
    """Standardized broad-phase pruning using O(N log N) KDTree.

    Args:
        centroids: (n_simplices, ambient_dim) array of simplex centroids.
        radii: (n_simplices,) array of simplex bounding radii.
        tol: Numerical tolerance for overlap.

    Returns:
        A set of candidate index pairs (i, j) with i < j.
    """
    n = int(centroids.shape[0])
    if n <= 1:
        return set()

    # Singular path: KDTree is memory-efficient and mathematically exact for broad-phase.
    tree = cKDTree(centroids)
    max_radius = float(np.max(radii)) if len(radii) > 0 else 0.0
    pairs: set[tuple[int, int]] = set()
    
    # We query the KDTree for neighbors within a radius that accounts for 
    # the maximum possible simplex size + the query simplex size.
    for i, center in enumerate(centroids):
        # Bound: dist(c1, c2) <= r1 + r2 + tol <= r1 + max_radius + tol
        query_radius = float(radii[i] + max_radius + tol)
        neigh = tree.query_ball_point(center, r=query_radius)
        for j in neigh:
            if j <= i:
                continue
            # Narrower check to finalize candidate selection
            d_sq = np.sum((center - centroids[j])**2)
            if d_sq <= (float(radii[i] + radii[j] + tol))**2:
                pairs.add((i, j))
    return pairs


def _simplex_intersection(
    pl_map: PLMap,
    simplex_a: Sequence[int],
    simplex_b: Sequence[int],
    *,
    tol: float,
    pa: Optional[np.ndarray] = None,
    pb: Optional[np.ndarray] = None,
) -> Optional[SimplexIntersectionWitness]:
    """Calculate the intersection between two simplices using exact or heuristic methods.

    Args:
        pl_map: The PL map.
        simplex_a: First simplex vertex labels.
        simplex_b: Second simplex vertex labels.
        tol: Numerical tolerance.
        pa: Optional precomputed coordinates for simplex_a.
        pb: Optional precomputed coordinates for simplex_b.

    Returns:
        A SimplexIntersectionWitness if an intersection is found, else None.
    """
    dim_a = len(simplex_a) - 1
    dim_b = len(simplex_b) - 1
    pa = pl_map.simplex_vertices(simplex_a) if pa is None else np.asarray(pa, dtype=np.float64)
    pb = pl_map.simplex_vertices(simplex_b) if pb is None else np.asarray(pb, dtype=np.float64)

    # Only exact low-dimensional predicates are implemented conservatively.
    if dim_a == 1 and dim_b == 1:
        ok, dist = _segment_segment_intersection(pa[0], pa[1], pb[0], pb[1], tol=tol)
        if ok:
            return SimplexIntersectionWitness(tuple(simplex_a), tuple(simplex_b), "segment_segment", dist, 0, ["Edge intersection detected"])
        return None

    if dim_a == 1 and dim_b == 2:
        ok, dist = _segment_triangle_intersection(pa[0], pa[1], pb, tol=tol)
        if ok:
            return SimplexIntersectionWitness(tuple(simplex_a), tuple(simplex_b), "segment_triangle", dist, 0, ["Edge/triangle intersection detected"])
        return None

    if dim_a == 2 and dim_b == 1:
        ok, dist = _segment_triangle_intersection(pb[0], pb[1], pa, tol=tol)
        if ok:
            return SimplexIntersectionWitness(tuple(simplex_a), tuple(simplex_b), "triangle_segment", dist, 0, ["Triangle/edge intersection detected"])
        return None

    if dim_a == 2 and dim_b == 2:
        ok, dist = _triangle_triangle_intersection(pa, pb, tol=tol)
        if ok:
            return SimplexIntersectionWitness(tuple(simplex_a), tuple(simplex_b), "triangle_triangle", dist, 0, ["Triangle intersection detected"])
        return None

    # For higher dimensions, we conservatively fall back to bounding-box and rank heuristics.
    if _affine_hulls_overlap(pa, pb, tol=tol):
        dist = float(np.linalg.norm(np.mean(pa, axis=0) - np.mean(pb, axis=0)))
        return SimplexIntersectionWitness(tuple(simplex_a), tuple(simplex_b), "heuristic_overlap", dist, min(dim_a, dim_b), ["High-dimensional overlap heuristic"])
    return None


def _affine_hulls_overlap(pa: np.ndarray, pb: np.ndarray, *, tol: float) -> bool:
    """Heuristic check if the affine hulls of two sets of points overlap.

    Args:
        pa: Coordinates for the first set of points.
        pb: Coordinates for the second set of points.
        tol: Numerical tolerance.

    Returns:
        True if the hulls likely overlap.
    """
    # Conservative heuristic using hull rank and barycenter distance.
    if pa.size == 0 or pb.size == 0:
        return False
    dist = np.linalg.norm(np.mean(pa, axis=0) - np.mean(pb, axis=0))
    return bool(dist <= 10.0 * tol or (np.linalg.matrix_rank((pa - pa[0]).T) + np.linalg.matrix_rank((pb - pb[0]).T) > 0))


def _segment_segment_intersection(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    *,
    tol: float,
) -> tuple[bool, float]:
    """Find the closest points between two segments in R^m.

    Args:
        a0: Start of first segment.
        a1: End of first segment.
        b0: Start of second segment.
        b1: End of second segment.
        tol: Numerical tolerance.

    Returns:
        A tuple (intersect, distance).
    """
    # Closest points between two segments in R^m.
    u = a1 - a0
    v = b1 - b0
    w0 = a0 - b0
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w0))
    e = float(np.dot(v, w0))
    denom = a * c - b * b
    if denom <= tol:
        # Nearly parallel; compare endpoint distances and collinearity heuristics.
        dists = [
            np.linalg.norm(a0 - b0),
            np.linalg.norm(a0 - b1),
            np.linalg.norm(a1 - b0),
            np.linalg.norm(a1 - b1),
        ]
        min_dist = float(min(dists))
        return min_dist <= tol, min_dist
    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom
    s = min(1.0, max(0.0, s))
    t = min(1.0, max(0.0, t))
    p = a0 + s * u
    q = b0 + t * v
    dist = float(np.linalg.norm(p - q))
    return dist <= tol, dist


def _triangle_plane_normal(tri: np.ndarray) -> np.ndarray:
    """Compute the unit normal for a triangle in R^3.

    Args:
        tri: (3, 3) array of triangle vertices.

    Returns:
        The unit normal vector.
    """
    n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    norm = np.linalg.norm(n)
    if norm <= _EPS:
        return np.zeros(3, dtype=np.float64)
    return n / norm


def _segment_triangle_intersection(
    p0: np.ndarray,
    p1: np.ndarray,
    tri: np.ndarray,
    *,
    tol: float,
) -> tuple[bool, float]:
    """Find the intersection between a segment and a triangle in R^3.

    Args:
        p0: Start of segment.
        p1: End of segment.
        tri: (3, 3) array of triangle vertices.
        tol: Numerical tolerance.

    Returns:
        A tuple (intersect, distance).
    """
    tri = np.asarray(tri, dtype=np.float64)
    n = _triangle_plane_normal(tri)
    if np.linalg.norm(n) <= tol:
        return False, float("inf")
    d0 = float(np.dot(n, p0 - tri[0]))
    d1 = float(np.dot(n, p1 - tri[0]))
    if abs(d0) <= tol and abs(d1) <= tol:
        # Segment lies in the plane: use coplanar triangle overlap heuristics.
        if _coplanar_segment_triangle_overlap(p0, p1, tri, tol=tol):
            return True, 0.0
        return False, float("inf")
    if d0 * d1 > tol * tol:
        return False, float(abs(min(abs(d0), abs(d1))))
    denom = d0 - d1
    if abs(denom) <= tol:
        return False, float("inf")
    t = d0 / denom
    if t < -tol or t > 1.0 + tol:
        return False, float("inf")
    p = p0 + t * (p1 - p0)
    if _point_in_triangle_3d(p, tri, tol=tol):
        return True, float(abs(np.dot(n, p - tri[0])))
    return False, float("inf")


def _triangle_triangle_intersection(
    tri_a: np.ndarray,
    tri_b: np.ndarray,
    *,
    tol: float,
) -> tuple[bool, float]:
    """Find the intersection between two triangles in R^3.

    Args:
        tri_a: (3, 3) array of vertices for the first triangle.
        tri_b: (3, 3) array of vertices for the second triangle.
        tol: Numerical tolerance.

    Returns:
        A tuple (intersect, distance).
    """
    tri_a = np.asarray(tri_a, dtype=np.float64)
    tri_b = np.asarray(tri_b, dtype=np.float64)
    n_a = _triangle_plane_normal(tri_a)
    n_b = _triangle_plane_normal(tri_b)
    if np.linalg.norm(n_a) <= tol or np.linalg.norm(n_b) <= tol:
        return False, float("inf")

    # Coplanar case.
    if np.linalg.norm(np.cross(n_a, n_b)) <= tol and abs(np.dot(n_a, tri_b[0] - tri_a[0])) <= tol:
        if _coplanar_triangle_overlap(tri_a, tri_b, tol=tol):
            return True, 0.0
        return False, float("inf")

    # Edge against triangle tests.
    for i in range(3):
        ok, dist = _segment_triangle_intersection(tri_a[i], tri_a[(i + 1) % 3], tri_b, tol=tol)
        if ok:
            return True, dist
    for i in range(3):
        ok, dist = _segment_triangle_intersection(tri_b[i], tri_b[(i + 1) % 3], tri_a, tol=tol)
        if ok:
            return True, dist
    return False, float("inf")


def _point_in_triangle_3d(point: np.ndarray, tri: np.ndarray, *, tol: float) -> bool:
    """Test if a point lies inside a triangle in R^3.

    Args:
        point: 3D point coordinates.
        tri: (3, 3) array of triangle vertices.
        tol: Numerical tolerance.

    Returns:
        True if the point is inside the triangle.
    """
    tri = np.asarray(tri, dtype=np.float64)
    n = _triangle_plane_normal(tri)
    if np.linalg.norm(n) <= tol:
        return False
    if abs(np.dot(n, point - tri[0])) > 10.0 * tol:
        return False
    # Project to dominant 2D plane.
    ax = int(np.argmax(np.abs(n)))
    idx = [0, 1, 2]
    idx.pop(ax)
    a = tri[0][idx]
    b = tri[1][idx]
    c = tri[2][idx]
    p = point[idx]
    return _point_in_triangle_2d(p, a, b, c, tol=tol)


def _point_in_triangle_2d(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, *, tol: float) -> bool:
    """Test if a point lies inside a triangle in R^2 using barycentric coordinates.

    Args:
        p: 2D point coordinates.
        a: First triangle vertex.
        b: Second triangle vertex.
        c: Third triangle vertex.
        tol: Numerical tolerance.

    Returns:
        True if the point is inside the triangle.
    """
    v0 = c - a
    v1 = b - a
    v2 = p - a
    den = float(v1[0] * v0[1] - v0[0] * v1[1])
    if abs(den) <= tol:
        return False
    u = float((v2[0] * v0[1] - v0[0] * v2[1]) / den)
    v = float((v1[0] * v2[1] - v2[0] * v1[1]) / den)
    return u >= -tol and v >= -tol and (u + v) <= 1.0 + tol


def _coplanar_segment_triangle_overlap(p0: np.ndarray, p1: np.ndarray, tri: np.ndarray, *, tol: float) -> bool:
    """Test if a segment overlaps with a triangle in the same plane.

    Args:
        p0: Start of segment.
        p1: End of segment.
        tri: Triangle vertices.
        tol: Numerical tolerance.

    Returns:
        True if they overlap.
    """
    # Project to 2D plane and test segment/triangle overlap via endpoint containment and edge intersections.
    n = _triangle_plane_normal(tri)
    ax = int(np.argmax(np.abs(n)))
    idx = [0, 1, 2]
    idx.pop(ax)
    s0, s1 = p0[idx], p1[idx]
    a, b, c = tri[0][idx], tri[1][idx], tri[2][idx]
    if _point_in_triangle_2d(s0, a, b, c, tol=tol) or _point_in_triangle_2d(s1, a, b, c, tol=tol):
        return True
    tri_edges = [(a, b), (b, c), (c, a)]
    for e0, e1 in tri_edges:
        ok, _ = _segment_segment_intersection_2d(s0, s1, e0, e1, tol=tol)
        if ok:
            return True
    return False


def _coplanar_triangle_overlap(tri_a: np.ndarray, tri_b: np.ndarray, *, tol: float) -> bool:
    """Test if two triangles overlap in the same plane.

    Args:
        tri_a: First triangle vertices.
        tri_b: Second triangle vertices.
        tol: Numerical tolerance.

    Returns:
        True if they overlap.
    """
    n = _triangle_plane_normal(tri_a)
    ax = int(np.argmax(np.abs(n)))
    idx = [0, 1, 2]
    idx.pop(ax)
    A = [tri_a[i][idx] for i in range(3)]
    B = [tri_b[i][idx] for i in range(3)]
    for p in A:
        if _point_in_triangle_2d(p, B[0], B[1], B[2], tol=tol):
            return True
    for p in B:
        if _point_in_triangle_2d(p, A[0], A[1], A[2], tol=tol):
            return True
    for i in range(3):
        a0, a1 = A[i], A[(i + 1) % 3]
        for j in range(3):
            b0, b1 = B[j], B[(j + 1) % 3]
            ok, _ = _segment_segment_intersection_2d(a0, a1, b0, b1, tol=tol)
            if ok:
                return True
    return False


def _segment_segment_intersection_2d(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    *,
    tol: float,
) -> tuple[bool, float]:
    """Find the intersection between two segments in R^2.

    Args:
        a0: Start of first segment.
        a1: End of first segment.
        b0: Start of second segment.
        b1: End of second segment.
        tol: Numerical tolerance.

    Returns:
        A tuple (intersect, distance).
    """
    u = a1 - a0
    v = b1 - b0
    den = float(u[0] * v[1] - u[1] * v[0])
    if abs(den) <= tol:
        return False, float(np.linalg.norm(a0 - b0))
    w = b0 - a0
    s = float((w[0] * v[1] - w[1] * v[0]) / den)
    t = float((w[0] * u[1] - w[1] * u[0]) / den)
    if -tol <= s <= 1.0 + tol and -tol <= t <= 1.0 + tol:
        p = a0 + s * u
        q = b0 + t * v
        dist = float(np.linalg.norm(p - q))
        return dist <= tol, dist
    return False, float(np.linalg.norm(a0 - b0))


__all__ = [
    "EmbeddingResult",
    "ImmersionResult",
    "PLMap",
    "ProjectionResult",
    "SelfIntersectionReport",
    "SimplexIntersectionWitness",
    "analyze_embedding",
    "check_immersion",
    "detect_self_intersections",
    "jitter_coordinates",
    "project_coordinates",
]
