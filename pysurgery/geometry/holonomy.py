r"""Holonomy of sampled manifolds: discrete Levi-Civita transport and orientability.

Overview:
    No learned parameters. The connection is the one the data carries: a tangent frame
    at every point by local PCA (``intrinsic_dimension.local_pca_tangent_basis``), and
    transport along an edge by the orthogonal map closest to the change of basis -- the
    polar factor of ``F_j^T F_i`` (the discrete Levi-Civita connection of Singer-Wu
    vector diffusion maps). Going around a closed loop gives an element of O(d): the
    HOLONOMY.

Key Concepts:
    - **Gauge invariance.** Local PCA frames have an arbitrary sign and rotation inside
      the tangent space. Replacing ``F_j`` by ``F_j A`` (A in O(d)) sends
      ``O_ij -> A^T O_ij`` and ``O_jk -> O_jk A``, so around a loop every interior A
      cancels and the holonomy changes by conjugation at the base point,
      ``H -> A^T H A``. Its determinant and rotation angles are exactly invariant.
    - **Orientability, exactly given the transports.** The determinant of a transport is
      +-1, and the determinant of the holonomy is a homomorphism ``H_1(graph) -> Z/2``:
      the first Stiefel-Whitney class. It is trivial iff the sampled manifold is
      orientable. That is decided by the ORIENTATION DOUBLE COVER -- signs propagated
      along a spanning forest, then EVERY remaining edge checked -- so every cycle of the
      graph is examined at once. What is not exact is the input: the frames are
      estimates, and an edge whose two tangent spaces are far apart (a small singular
      value of ``F_j^T F_i``, i.e. a principal angle near 90 degrees) has an unreliable
      transport sign. Such edges are counted, and a verdict with any of them is reported
      as NOT certified.
    - **Gauss-Bonnet as ground truth.** On the unit sphere, transport around a latitude
      circle at colatitude theta rotates by the enclosed solid angle
      ``2 pi (1 - cos theta)``.

    For a complex that is a closed pseudomanifold, orientability is also decidable purely
    combinatorially, with no frames at all:
    ``pysurgery.topology.fundamental_cycles.is_orientable_pseudomanifold``.

Common Workflows:
    1. **Is this point cloud a sample of an orientable surface?** ->
       ``orientation_report(X, 2)``.
    2. **Holonomy fingerprint** -> ``holonomy_report(X, d)``.
    3. **Transport around one loop** -> ``holonomy_along(F, loop)``.

Coefficient Ring:
    Real (O(d)); the orientation verdict is a Z/2 class.
"""

from __future__ import annotations

import warnings
from collections import deque
from typing import List, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..bridge.julia_bridge import julia_engine

__all__ = [
    "tangent_frames",
    "transport",
    "knn_graph",
    "OrientationReport",
    "orientation_report",
    "holonomy_along",
    "rotation_angles",
    "rotation_angle",
    "fundamental_loops",
    "HolonomyReport",
    "holonomy_report",
]


def tangent_frames(X: np.ndarray, d: int, k: int = 12) -> np.ndarray:
    """Local-PCA tangent frames at EVERY point: shape ``(n, ambient, d)``.

    Args:
        X: ``(n, ambient)`` point cloud.
        d: Intrinsic dimension.
        k: Neighbours per point (the point itself is added).

    Returns:
        Orthonormal frames, one per point.
    """
    from .intrinsic_dimension import local_pca_tangent_basis

    X = np.asarray(X, dtype=np.float64)
    return np.asarray(local_pca_tangent_basis(X, int(d), neighborhood_size=int(k)).bases)


def transport(F_i: np.ndarray, F_j: np.ndarray) -> np.ndarray:
    """The orthogonal d x d map from the frame at i to the frame at j.

    The polar factor of ``F_j^T F_i`` (the orthogonal Procrustes solution).

    Args:
        F_i: ``(ambient, d)`` frame at i.
        F_j: ``(ambient, d)`` frame at j.

    Returns:
        ``O_ij`` in O(d).
    """
    u, _s, vt = np.linalg.svd(F_j.T @ F_i)
    return u @ vt


def knn_graph(X: np.ndarray, k: int = 8) -> List[Tuple[int, int]]:
    """Undirected k-nearest-neighbour edge list (exact neighbours).

    Args:
        X: Point cloud.
        k: Neighbours per point.

    Returns:
        Sorted ``(i, j)`` edges with ``i < j``.
    """
    from scipy.spatial import cKDTree

    X = np.asarray(X, dtype=np.float64)
    kk = min(int(k), len(X) - 1)
    _d, idx = cKDTree(X).query(X, k=kk + 1)
    E = set()
    for i in range(len(idx)):
        for j in np.atleast_1d(idx[i])[1:]:
            E.add((min(i, int(j)), max(i, int(j))))
    return sorted(E)


def _spanning_forest(n: int, edges) -> tuple:
    adj: dict = {i: [] for i in range(n)}
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    parent = [-1] * n
    root_of = [-1] * n
    seen = [False] * n
    tree = set()
    order = []
    for s in range(n):
        if seen[s]:
            continue
        seen[s] = True
        root_of[s] = s
        q = deque([s])
        while q:
            u = q.popleft()
            order.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    parent[v] = u
                    root_of[v] = s
                    tree.add((min(u, v), max(u, v)))
                    q.append(v)
    return parent, tree, root_of, order


def _edge_data(F: np.ndarray, pairs: np.ndarray, backend: str) -> Tuple[np.ndarray, np.ndarray]:
    """For every directed pair (i, j): sign(det(F_j^T F_i)) and its smallest singular value."""
    if len(pairs) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    b = str(backend).lower().strip()
    if b == "julia" or (b == "auto" and julia_engine.available):
        try:
            return julia_engine.edge_transport_data(F, pairs)
        except Exception as e:  # pragma: no cover - depends on the Julia runtime
            if b == "julia":
                raise
            warnings.warn(f"Julia edge transports failed ({e!r}); falling back to Python.")
    M = np.einsum("nai,naj->nij", F[pairs[:, 1]], F[pairs[:, 0]])   # F_j^T F_i
    sv = np.linalg.svd(M, compute_uv=False)
    return np.sign(np.linalg.det(M)).astype(np.int64), sv[:, -1]


class OrientationReport(BaseModel):
    """Orientability of a sampled d-manifold via the orientation double cover.

    Attributes:
        orientable (bool): No edge of the graph disagrees with the propagated signs.
        certified (bool): No unreliable transport anywhere.
        n_edges (int): Edges examined (all of them).
        n_tree_edges (int): Edges of the spanning forest.
        n_inconsistent (int): Non-tree edges whose transport sign contradicts the signs.
        inconsistent_edges (list): Those edges.
        n_unreliable (int): Edges whose tangent spaces are too far apart.
        unreliable_edges (list): Those edges.
        n_components (int): Components of the graph.
        signs (np.ndarray): The propagated orientation sign of every point.
        notes (list[str]): Human-readable notes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    orientable: bool
    certified: bool
    n_edges: int
    n_tree_edges: int
    n_inconsistent: int
    inconsistent_edges: List[Tuple[int, int]]
    n_unreliable: int
    unreliable_edges: List[Tuple[int, int]]
    n_components: int
    signs: np.ndarray
    notes: List[str] = Field(default_factory=list)

    def __str__(self) -> str:
        return (
            ("ORIENTABLE" if self.orientable else "NON-ORIENTABLE")
            + ("" if self.certified else " (NOT certified: unreliable transports)")
            + f" -- {self.n_inconsistent} of {self.n_edges - self.n_tree_edges} non-tree edges "
              f"disagree with the sign assignment; {self.n_unreliable} unreliable transports; "
              f"{self.n_components} component(s)"
            + "".join("\n  " + n for n in self.notes)
        )


def _normalise_edges(X: np.ndarray, edges, k: int) -> List[Tuple[int, int]]:
    if edges is None:
        return knn_graph(X, k=k)
    return sorted({(min(int(i), int(j)), max(int(i), int(j))) for i, j in edges if int(i) != int(j)})


def orientation_report(
    X: np.ndarray,
    d: int,
    edges=None,
    k: int = 8,
    k_pca: Optional[int] = None,
    min_cos: float = 0.5,
    backend: str = "auto",
) -> OrientationReport:
    """Is the sampled d-manifold orientable? Decided over EVERY edge of the graph.

    What is Being Computed?:
        The orientation double cover of the graph (``edges``, e.g. a complex's 1-skeleton;
        default the k-nearest-neighbour graph) with the transport signs of the local-PCA
        frames: signs propagate along a spanning forest, and every non-tree edge is
        checked against them. Orientable iff none disagrees -- iff ``w_1`` vanishes on
        every cycle of the graph.

    Args:
        X: ``(n, ambient)`` sample.
        d: Intrinsic dimension.
        edges: The graph (default: kNN graph with ``k``).
        k: Neighbours of the default graph.
        k_pca: Neighbours for local PCA (default ``max(2d + 2, k)``).
        min_cos: An edge is unreliable when the smallest singular value of
            ``F_j^T F_i`` -- the cosine of the largest principal angle between the two
            tangent spaces -- is below this: its transport sign is then a statement about
            the frame estimate rather than the manifold.
        backend: 'auto', 'julia' or 'python'.

    Returns:
        An ``OrientationReport``.
    """
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    F = tangent_frames(X, d, k=(k_pca or max(2 * d + 2, k)))
    E = _normalise_edges(X, edges, k)
    parent, tree, _root, order = _spanning_forest(n, E)
    tree_pairs = np.array([(parent[u], u) for u in order if parent[u] >= 0], dtype=np.int64).reshape(-1, 2)
    other = [e for e in E if e not in tree]
    other_pairs = np.array(other, dtype=np.int64).reshape(-1, 2)
    t_sign, t_cos = _edge_data(F, tree_pairs, backend)
    o_sign, o_cos = _edge_data(F, other_pairs, backend)
    signs = np.ones(n, dtype=np.int64)
    unreliable: List[Tuple[int, int]] = []
    tree_index = {int(u): t for t, u in enumerate(tree_pairs[:, 1])}
    for u in order:
        p = parent[u]
        if p >= 0:
            t = tree_index[u]
            sg = int(t_sign[t])
            signs[u] = signs[p] * (sg if sg != 0 else 1)
            if t_cos[t] < min_cos or sg == 0:
                unreliable.append((min(p, u), max(p, u)))
    bad = []
    for t, (i, j) in enumerate(other):
        sg = int(o_sign[t])
        if o_cos[t] < min_cos or sg == 0:
            unreliable.append((i, j))
        if signs[i] * sg != signs[j]:
            bad.append((i, j))
    n_comp = sum(1 for i in range(n) if parent[i] < 0)
    notes = []
    if n_comp > 1:
        notes.append(f"the graph has {n_comp} components; orientability is decided per "
                     f"component and the verdict is for all of them")
    return OrientationReport(
        orientable=not bad, certified=not unreliable, n_edges=len(E), n_tree_edges=len(tree),
        n_inconsistent=len(bad), inconsistent_edges=bad, n_unreliable=len(set(unreliable)),
        unreliable_edges=sorted(set(unreliable)), n_components=n_comp, signs=signs, notes=notes,
    )


def holonomy_along(F: np.ndarray, loop: Sequence[int]) -> np.ndarray:
    """Holonomy around a closed loop of vertex indices (the last transports back to the first).

    Args:
        F: ``(n, ambient, d)`` frames.
        loop: Vertex indices of the loop.

    Returns:
        The holonomy in O(d).
    """
    loop = list(loop)
    H = np.eye(F.shape[2])
    if len(loop) < 2:
        return H
    for a, b in zip(loop, loop[1:] + loop[:1]):
        H = transport(F[a], F[b]) @ H
    return H


def rotation_angles(H: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """The rotation angles of an orthogonal matrix, sorted descending.

    One angle in (0, pi) per complex-conjugate pair of eigenvalues ``e^(+-i theta)``, and 0
    or pi for each real eigenvalue +1 or -1 (a reflection contributes a pi).
    Gauge-invariant: conjugation preserves eigenvalues.

    Args:
        H: An orthogonal matrix.
        tol: Tolerance for a real eigenvalue.

    Returns:
        The angles.
    """
    w = np.linalg.eigvals(np.asarray(H, dtype=np.float64))
    pairs = [abs(np.angle(z)) for z in w if z.imag > tol]
    reals = [0.0 if z.real > 0 else np.pi for z in w if abs(z.imag) <= tol]
    return np.array(sorted(pairs + reals, reverse=True))


def rotation_angle(H: np.ndarray) -> float:
    """Signed angle for d = 2 with det = +1 (the Gauss-Bonnet angle, in (-pi, pi]).

    The largest rotation angle otherwise. The SIGN of the d = 2 angle depends on the
    orientation of the frame at the base point; its absolute value is gauge-invariant.

    Args:
        H: An orthogonal matrix.

    Returns:
        The angle in radians.
    """
    if H.shape == (2, 2) and np.linalg.det(H) > 0:
        return float(np.arctan2(H[1, 0], H[0, 0]))
    return float(rotation_angles(H)[0])


def fundamental_loops(n: int, edges) -> List[List[int]]:
    """One loop per non-tree edge of a spanning forest -- ALL of them.

    Together they generate ``H_1`` of the graph (the cycle space), so their determinants
    decide orientability and their angles are the holonomy's full fingerprint.

    Args:
        n: Number of vertices.
        edges: Undirected edges.

    Returns:
        The loops as vertex lists.
    """
    E = sorted({(min(i, j), max(i, j)) for i, j in edges})
    parent, tree, root_of, _order = _spanning_forest(n, E)

    def path_to_root(v):
        p = [v]
        while parent[p[-1]] >= 0:
            p.append(parent[p[-1]])
        return p

    loops = []
    for (i, j) in E:
        if (i, j) in tree or root_of[i] != root_of[j]:
            continue
        pi, pj = path_to_root(i), path_to_root(j)
        sj = set(pj)
        meet = next(x for x in pi if x in sj)
        a = pi[:pi.index(meet) + 1]
        b = pj[:pj.index(meet) + 1]
        loops.append([int(v) for v in a + b[:-1][::-1]])
    return loops


class HolonomyReport(BaseModel):
    """Determinants and rotation angles of the holonomy around every fundamental loop.

    Attributes:
        d (int): Intrinsic dimension.
        n_loops (int): Number of fundamental loops.
        dets (np.ndarray): Holonomy determinant per loop.
        angles (list[np.ndarray]): Rotation angles per loop.
        n_negative (int): Loops with determinant -1 (orientation-reversing).
        orientation (OrientationReport): The double-cover verdict.
        loop_lengths (list[int]): Length of each loop.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    d: int
    n_loops: int
    dets: np.ndarray
    angles: List[np.ndarray]
    n_negative: int
    orientation: OrientationReport
    loop_lengths: List[int] = Field(default_factory=list)


def holonomy_report(
    X: np.ndarray,
    d: int,
    edges=None,
    k: int = 8,
    k_pca: Optional[int] = None,
    min_cos: float = 0.5,
    backend: str = "auto",
) -> HolonomyReport:
    """Holonomy around EVERY fundamental loop of the graph, plus the orientation verdict.

    Args:
        X: ``(n, ambient)`` sample.
        d: Intrinsic dimension.
        edges: The graph (default: kNN graph with ``k``).
        k: Neighbours of the default graph.
        k_pca: Neighbours for local PCA (default ``max(2d + 2, k)``).
        min_cos: Reliability threshold of a transport (see ``orientation_report``).
        backend: 'auto', 'julia' or 'python'.

    Returns:
        A ``HolonomyReport``.
    """
    X = np.asarray(X, dtype=np.float64)
    F = tangent_frames(X, d, k=(k_pca or max(2 * d + 2, k)))
    E = _normalise_edges(X, edges, k)
    loops = fundamental_loops(len(X), E)
    dets, angles = [], []
    for lp in loops:
        H = holonomy_along(F, lp)
        dets.append(float(np.linalg.det(H)))
        angles.append(rotation_angles(H))
    ori = orientation_report(X, d, edges=E, k=k, k_pca=k_pca, min_cos=min_cos, backend=backend)
    dets = np.array(dets)
    return HolonomyReport(
        d=int(d), n_loops=len(loops), dets=dets, angles=angles, n_negative=int(np.sum(dets < 0)),
        orientation=ori, loop_lengths=[len(lp) for lp in loops],
    )
