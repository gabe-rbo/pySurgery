#!/usr/bin/env python3
"""Validate Cocone reconstruction against the real motivating linked-tori dataset.

What is Being Computed?:
    Reconstructs each of the two real point clouds from the FutureLab
    "two linked tori" surgery-theory experiment -- ``torus_a.csv`` and
    ``separated_cylinder_b_unified.csv`` (both genuine tori; there is no standalone
    ``torus_b.csv`` -- the original torus B was only ever saved post-deformation) --
    via ``SimplicialComplex.from_cocone(tight=False)`` (``repair=True``'s perturbation loop
    is the reliable path to closure -- see ``cocone_reconstruction``'s docstring --
    ``tight=True``'s flood-fill closer is experimental and was measured to leak badly on
    this exact dataset, so it is deliberately not the primary path here), one torus at a
    time (not the combined 1600-point union, which risks spurious bridging simplices
    between the two nearby, linked surfaces). Reports ``is_homology_manifold`` / Betti-number
    verdicts directly against the alpha-complex baseline already measured for this dataset:
    at ``torus_a``'s first Betti-matching threshold (alpha=0.2375, ``b0=1, b1=2, b2=1``),
    ``is_homology_manifold`` reported 1600 vertex-link defects out of 800 points; by the
    end of the filtration sweep the defect count was still 800 and the Betti numbers had
    already degraded to ``b1=1``. No threshold in that sweep had both the right Betti
    numbers and manifold status.

Use When:
    - Manually re-checking Cocone reconstruction against real, noisier data (as opposed
      to the synthetic jittered sphere/torus fixtures in ``tests/test_reconstruction.py``)
      -- e.g. after retuning ``theta`` / ``reach_fraction`` / ``bounding_radius_factor``.
    - Confirming the module's real-data behavior matches what its docstrings describe,
      including the documented concave-region limitation.

What Was Actually Measured (as of the theta=40deg/reach_fraction=0.85 tuning used here):
    Neither real torus reaches the "0/800 defects, fully closed" ideal the alpha-complex
    comparison motivates: ``torus_a`` converges within budget but with only ~45% vertex
    coverage and one residual link defect (a genuine improvement in kind over the alpha
    complex's 1600/800 -- the surface that IS reconstructed is a real, mostly-clean
    manifold patch, not merely thinner defective tetrahedra everywhere -- but not the
    "flip to fully closed" outcome); ``separated_cylinder_b_unified`` (torus B, post-
    deformation) does not converge within 250 rounds at all, across a further
    ``reach_fraction`` sweep (0.6-0.85). This is consistent with, not contradictory to,
    the concave/high-curvature limitation ``cocone_reconstruction``'s docstring documents
    on synthetic tori -- the real data is evidently at least as hard as the synthetic
    fixture in this respect. Treat this script's output as the current, honest state of
    real-data reconstruction quality, not a demonstration that the limitation has been
    resolved.

This is a manual checkpoint, not a pass/fail gate -- it is intentionally excluded from
pytest collection (``pyproject.toml``'s ``testpaths = ["tests"]``), matching the existing
``scripts/`` convention.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from pysurgery.core.exceptions import ReconstructionRepairError
from pysurgery.topology.complexes import SimplicialComplex

_DATA_DIR = (
    Path.home()
    / "Desktop"
    / "FutureLab"
    / "Experiment - Topology of Neural Networks"
    / "surgery_experiment"
    / "operated_datasets"
    / "linked_tori"
    / "two_linked_tori"
    / "data"
)

_DATASETS = {
    "torus_a": _DATA_DIR / "torus_a.csv",
    "torus_b (separated_cylinder_b_unified)": _DATA_DIR / "separated_cylinder_b_unified.csv",
}

# Alpha-complex baseline, measured directly from this dataset's own filtration reports
# (torus_a_filtration_report.md / torus_b_filtration_report.md) at the first threshold
# where Betti numbers correctly read the torus signature (b0=1, b1=2, b2=1):
_ALPHA_BASELINE_DEFECTS = 1600
_ALPHA_BASELINE_N_POINTS = 800


def _load_points(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=";")


def _report_one(label: str, path: Path, *, theta: float, reach_fraction: float) -> dict:
    print(f"\n{'=' * 70}\n{label}  ({path})\n  theta={np.rad2deg(theta):.1f} deg  "
          f"reach_fraction={reach_fraction}\n{'=' * 70}")
    if not path.exists():
        print(f"  MISSING: {path}")
        return {"label": label, "status": "missing"}

    points = _load_points(path)
    print(f"  n_points = {len(points)}")

    t0 = time.time()
    try:
        sc = SimplicialComplex.from_cocone(
            points, theta=theta, reach_fraction=reach_fraction, tight=False, max_repair_rounds=250
        )
    except ReconstructionRepairError as e:
        print(f"  REPAIR FAILED TO CONVERGE: {e}")
        return {"label": label, "status": "repair_failed", "error": str(e)}
    elapsed = time.time() - t0

    is_mani, dim, diag = sc.is_homology_manifold(backend="julia")
    n_defects = len(diag)
    n_triangles = len(sc.n_simplices(2))
    covered = len({v for tri in sc.n_simplices(2) for v in tri})
    is_closed = sc.is_closed_manifold if is_mani else None
    betti = sc.betti_numbers(backend="julia") if is_mani else {}

    print(f"  elapsed = {elapsed:.1f}s")
    print(f"  is_homology_manifold = {is_mani}  (dimension={dim})")
    print(f"  n_defects = {n_defects}  (alpha-complex baseline: {_ALPHA_BASELINE_DEFECTS}"
          f"/{_ALPHA_BASELINE_N_POINTS} points)")
    print(f"  is_closed_manifold = {is_closed}")
    print(f"  betti_numbers = {betti}")
    print(f"  n_triangles = {n_triangles}, covered vertices = {covered}/{len(points)}")
    if not is_mani:
        print(f"  diagnostics (first 5): {list(diag.items())[:5]}")

    return {
        "label": label,
        "status": "ok",
        "is_homology_manifold": is_mani,
        "n_defects": n_defects,
        "is_closed_manifold": is_closed,
        "betti_numbers": betti,
        "covered_fraction": covered / len(points),
        "elapsed_s": elapsed,
    }


def main() -> int:
    # theta/reach_fraction tuned for the torus's concave inner region in
    # tests/test_reconstruction.py's synthetic torus fixture -- the real data is the
    # motivating case for that same tuning, so start from it rather than the sphere-optimized
    # library defaults.
    theta = np.deg2rad(40.0)
    reach_fraction = 0.85
    results = [
        _report_one(label, path, theta=theta, reach_fraction=reach_fraction)
        for label, path in _DATASETS.items()
    ]

    print(f"\n{'=' * 70}\nSummary\n{'=' * 70}")
    for r in results:
        print(f"  {r['label']}: {r['status']}"
              + (f", defects={r.get('n_defects')}, closed={r.get('is_closed_manifold')}"
                 if r["status"] == "ok" else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
