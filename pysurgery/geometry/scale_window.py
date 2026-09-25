r"""Choosing the scale of a complex on a point cloud: the radii at which it is licensed.

Overview:
    Every complex built on a sample carries a scale, and a complex built at a bad scale
    is exactly certified as the wrong thing (``topology.local_homology``). This module
    computes the three constraints on the radius r of a union-of-balls model (Cech /
    alpha / Delaunay-Cech complexes, all homotopy equivalent to the union of r-balls),
    on EVERY point, in RADIUS units:

        r >= r_min          the union of r-balls is connected -- exactly half the longest
                            edge of the Euclidean minimum spanning tree
        r <  sqrt(3/5) tau  Niyogi-Smale-Weinberger (2008): the union of r-balls
                            deformation retracts onto the manifold, PROVIDED the sample
                            is (r/2)-dense in it; tau is the reach, estimated by
                            Federer's formula (``federer_reach``)
        r <  gap / 2        the balls of two classes do not meet (exact)

    An EMPTY window is a result: the sample does not resolve the class as a separate
    manifold at any scale.

Key Concepts:
    - **Federer's reach formula.** ``tau = inf_{x != y} |y - x|^2 / (2 d(y - x, T_x M))``
      over ALL pairs of sample points, with ``T_x M`` from local PCA -- the tangent
      spaces are the one estimated input; the infimum is taken over every pair exactly.
      A naive minimum distance would measure the sampling resolution, not the geometry.
    - **What cannot be checked.** The NSW density hypothesis is about the manifold
      between the samples; it is an assumption, not a certificate.

Common Workflows:
    1. **Radius range for one class** -> ``scale_window(X, dim=2)``.
    2. **Separating two classes** -> ``scale_window(X_a, X_b, dim=1)``.

Coefficient Ring:
    N/A (metric quantities).
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from ..bridge.julia_bridge import julia_engine

__all__ = ["connectivity_radius", "federer_reach", "ScaleWindow", "scale_window"]


def connectivity_radius(X: np.ndarray) -> float:
    """The smallest r at which the union of r-balls around X is connected.

    Exactly half the longest edge of the Euclidean minimum spanning tree; below it the
    Cech / alpha complex of X has ``beta_0 > 1``.

    Args:
        X: ``(n, ambient)`` points (distinct).

    Returns:
        The connectivity radius (0 for fewer than two points).

    Raises:
        ValueError: If X has duplicate points.
    """
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial.distance import pdist, squareform

    X = np.asarray(X, dtype=np.float64)
    if len(X) < 2:
        return 0.0
    D = squareform(pdist(X))
    if not np.all(D[~np.eye(len(X), dtype=bool)] > 0):
        raise ValueError("X has duplicate points; remove them first")
    T = minimum_spanning_tree(D)
    return float(T.data.max()) / 2.0


def federer_reach(X: np.ndarray, dim: int, k: int = 30, backend: str = "auto") -> float:
    """Estimated reach of the manifold X samples, by Federer's formula over ALL pairs.

    What is Being Computed?:
        ``tau = inf_{x != y} |y - x|^2 / (2 d(y - x, T_x M))`` with ``T_x M`` the
        ``dim``-dimensional local-PCA tangent space at x (``k`` neighbours). Pairs with
        ``y - x`` in ``T_x M`` contribute nothing (infinite). The Julia kernel evaluates
        the pairs in parallel threads.

    Args:
        X: ``(n, ambient)`` sample.
        dim: Intrinsic dimension of the manifold.
        k: Neighbours for local PCA.
        backend: 'auto', 'julia' or 'python'.

    Returns:
        The reach estimate (inf when every pair lies in the tangent spaces, e.g. a flat
        sample).
    """
    from .intrinsic_dimension import local_pca_tangent_basis

    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    if n < 3:
        return float("inf")
    kk = min(int(k), n - 1)
    P = np.asarray(local_pca_tangent_basis(X, int(dim), neighborhood_size=kk).bases)
    b = str(backend).lower().strip()
    if b == "julia" or (b == "auto" and julia_engine.available):
        try:
            return float(julia_engine.federer_reach(X, P))
        except Exception as e:  # pragma: no cover - depends on the Julia runtime
            if b == "julia":
                raise
            warnings.warn(f"Julia reach failed ({e!r}); falling back to Python.")
    best = np.inf
    for i in range(n):
        v = X - X[i]
        n2 = (v ** 2).sum(1)
        perp = np.linalg.norm(v - (v @ P[i]) @ P[i].T, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            cand = n2 / (2 * perp)
        cand[(n2 == 0) | ~np.isfinite(cand)] = np.inf
        best = min(best, float(cand.min()))
    return best


class ScaleWindow(BaseModel):
    """Radii at which a complex on one class is licensed (all in RADIUS units).

    Attributes:
        r_min (float): Connectivity: below it the class falls apart (beta_0 > 1).
        r_max_reach (float): NSW bound ``sqrt(3/5) tau`` (inf without ``dim``).
        r_max_gap (float): Half the distance to the other class (inf without one).
        r_max (float): ``min(r_max_reach, r_max_gap)``.
        empty (bool): ``r_max <= r_min``: the class is not resolved at any scale.
        detail (dict): The reach estimate, the inter-class gap and the dimension.
    """

    r_min: float
    r_max_reach: float
    r_max_gap: float
    r_max: float
    empty: bool
    detail: dict = Field(default_factory=dict)

    def grid(self, n: int = 8) -> np.ndarray:
        """Geometric grid over ``[r_min, r_max]`` (over ``[r_min, 2 r_min]`` if empty).

        Args:
            n: Number of radii.

        Returns:
            The radii.
        """
        hi = self.r_max if not self.empty else 2 * self.r_min
        return np.geomspace(max(self.r_min, 1e-12), max(hi, self.r_min * (1 + 1e-9)), n)

    def __str__(self) -> str:
        s = (f"r in [{self.r_min:.4g}, {self.r_max:.4g}] (connectivity {self.r_min:.4g}, "
             f"NSW reach bound {self.r_max_reach:.4g}, half-gap {self.r_max_gap:.4g})")
        return s + ("  <-- EMPTY: not resolved as a separate manifold" if self.empty else "")


def scale_window(
    X_class: np.ndarray,
    X_other: Optional[np.ndarray] = None,
    dim: Optional[int] = None,
    k_tangent: int = 30,
    backend: str = "auto",
) -> ScaleWindow:
    """The three constraints on the radius of a complex on one class (module docstring).

    Args:
        X_class: The class's sample.
        X_other: Optional sample of another class that must stay separate.
        dim: Intrinsic dimension (enables the NSW reach bound).
        k_tangent: Neighbours for the local-PCA tangent spaces.
        backend: 'auto', 'julia' or 'python'.

    Returns:
        A ``ScaleWindow``.
    """
    X = np.asarray(X_class, dtype=np.float64)
    r_min = connectivity_radius(X)
    if dim is None:
        r_reach, tau = float("inf"), float("nan")
    else:
        tau = federer_reach(X, dim, k=k_tangent, backend=backend)
        r_reach = float(np.sqrt(3.0 / 5.0) * tau)
    if X_other is not None and len(X_other):
        from scipy.spatial import cKDTree

        gap = float(cKDTree(np.asarray(X_other, dtype=np.float64)).query(X)[0].min())
    else:
        gap = float("inf")
    r_max = float(min(r_reach, gap / 2.0))
    return ScaleWindow(
        r_min=r_min, r_max_reach=r_reach, r_max_gap=gap / 2.0, r_max=r_max,
        empty=bool(r_max <= r_min), detail={"reach": tau, "inter_class_gap": gap, "dim": dim},
    )
