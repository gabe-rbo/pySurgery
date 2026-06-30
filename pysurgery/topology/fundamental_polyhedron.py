"""Fundamental Polyhedron and Universal Cover Tiling (compatibility shim).

The implementation now lives in :mod:`pysurgery.topology.coverings`, which is the
single home for pySurgery's covering-space machinery (both the geometric tiling
engine that used to live here and the algebraic finite-cover engine that used to
live in :mod:`pysurgery.homology.controlled_cohomology`).

This module is retained so that existing imports
``from pysurgery.topology.fundamental_polyhedron import (...)`` keep working.
"""

from pysurgery.topology.coverings import (  # noqa: F401
    FacePairing,
    FundamentalPolyhedron,
    construct_fundamental_polyhedron,
)

__all__ = [
    "FacePairing",
    "FundamentalPolyhedron",
    "construct_fundamental_polyhedron",
]
