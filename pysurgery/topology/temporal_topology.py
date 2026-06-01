import warnings
from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel

from pysurgery.topology.complexes import SimplicialComplex
from pysurgery.topology.persistent_homology import compute_zigzag_persistence, compute_topological_loss as ph_loss


class TemporalBarcode(BaseModel):
    """Parameter-indexed intervals tracking the persistence of topological features across a temporal sequence."""
    dimension: int
    births: List[float]
    deaths: List[float]
    parameters: List[float]
    field: str = 'Z2'
    exact: bool = False  # zigzag with index truncation; not exact


class BifurcationEvent(BaseModel):
    """A critical moment (phase transition) where the topology undergoes significant structural changes."""
    parameter_value: float
    type: str
    evidence: Dict[str, float]
    exact: bool = False


def _detect_bifurcations_impl(
    temporal_barcodes: List[TemporalBarcode],
    threshold: float = 1.0,
) -> List[BifurcationEvent]:
    """Internal worker for bifurcation detection.

    Identifies critical moments using gradient spikes in the count of
    active features per parameter step.
    """
    bifurcations: List[BifurcationEvent] = []

    for t_bc in temporal_barcodes:
        if not t_bc.births:
            continue

        params = t_bc.parameters
        n_params = len(params)

        active_counts = np.zeros(n_params)
        for b, d in zip(t_bc.births, t_bc.deaths):
            try:
                b_idx = params.index(b)
                d_idx = params.index(d)
                for i in range(b_idx, d_idx + 1):
                    active_counts[i] += 1
            except ValueError:
                continue

        flux = np.abs(np.diff(active_counts))

        for i in range(len(flux)):
            if flux[i] > threshold:
                evidence = {
                    "flux": float(flux[i]),
                    "dimension": float(t_bc.dimension),
                    "active_before": float(active_counts[i]),
                    "active_after": float(active_counts[i + 1]),
                }
                bifurcations.append(BifurcationEvent(
                    parameter_value=params[i + 1],
                    type="Topological Phase Transition",
                    evidence=evidence,
                    exact=False,
                ))

    return bifurcations


def intersect_complexes(K1: SimplicialComplex, K2: SimplicialComplex) -> SimplicialComplex:
    """Deprecated: use ``SimplicialComplex.intersect()`` instead."""
    warnings.warn(
        "intersect_complexes is deprecated; use SimplicialComplex.intersect()",
        DeprecationWarning,
        stacklevel=2,
    )
    return K1.intersect(K2)


def build_union_intersection_zigzag(point_cloud_sequence: List[np.ndarray], epsilon: float = 0.5, max_dimension: int = 2, complex_type: str = 'Rips') -> List[SimplicialComplex]:
    r"""Construct the union-intersection zigzag sequence over discrete temporal parameters.

    K_0 <- K_0 \\cap K_1 -> K_1 <- K_1 \\cap K_2 -> ... -> K_n.

    Args:
        point_cloud_sequence: List of point clouds.
        epsilon: Distance threshold for Rips complex.
        max_dimension: Max dimension for Rips complex.
        complex_type: Type of complex to build (currently only supports 'Rips').

    Returns:
        zigzag_sequence: The alternating sequence of complexes and intersections.
    """
    if complex_type != 'Rips':
        raise ValueError(f"Unsupported complex_type: {complex_type}. Only 'Rips' is currently supported.")

    complexes = []
    for pts in point_cloud_sequence:
        sc = SimplicialComplex.from_vietoris_rips(pts, epsilon=epsilon, max_dimension=max_dimension)
        complexes.append(sc)

    zigzag_sequence = []
    for i in range(len(complexes) - 1):
        K_current = complexes[i]
        K_next = complexes[i + 1]

        I_current = K_current.intersect(K_next)

        zigzag_sequence.append(K_current)
        zigzag_sequence.append(I_current)

    if complexes:
        zigzag_sequence.append(complexes[-1])

    return zigzag_sequence


def compute_temporal_homology(zigzag_seq: List[SimplicialComplex], dimension: int, parameters: Optional[List[float]] = None, field: str = 'Z2') -> TemporalBarcode:
    """Compute temporal homology using the Julia persistence kernel.

    Extracts parameter-indexed intervals mapping algebraic intervals back to the temporal indices.

    Args:
        zigzag_seq: Sequence of complexes (e.g., K0, K0_int_K1, K1, ...).
        dimension: Homological dimension to track.
        parameters: List of original parameter values (e.g. timestamps). If None, uses integer indices.
        field: Coefficient field ('Z2' or 'Q').

    Returns:
        TemporalBarcode object containing tracking data.
    """
    barcode_res = compute_zigzag_persistence(zigzag_seq, field=field)

    n_params = (len(zigzag_seq) + 1) // 2
    if parameters is None:
        parameters = list(range(n_params))

    if len(parameters) != n_params:
        raise ValueError("Length of parameters must match the number of original point clouds.")

    births = []
    deaths = []

    for b in barcode_res.barcodes:
        if b.dim == dimension:
            birth_idx = min(b.birth // 2, n_params - 1)
            death_idx = min(b.death // 2, n_params - 1)

            b_val = parameters[birth_idx]
            d_val = parameters[death_idx]

            births.append(b_val)
            deaths.append(d_val)

    return TemporalBarcode(
        dimension=dimension,
        births=births,
        deaths=deaths,
        parameters=parameters,
        field=field,
        exact=False,
    )


def compute_topological_loss(barcodes: TemporalBarcode, target_features: TemporalBarcode, epsilon: float = 0.01) -> float:
    """Computes Gromov-Wasserstein metric (via JAX layer) for diagram distances.

    Args:
        barcodes: The computed TemporalBarcode.
        target_features: The target TemporalBarcode to compare against.
        epsilon: Entropic regularization parameter.

    Returns:
        Gromov-Wasserstein loss value.
    """
    class PseudoBarcode:
        def __init__(self, birth, death, dim):
            self.birth = birth
            self.death = death
            self.dim = dim

    b1_list = [PseudoBarcode(b, d, barcodes.dimension) for b, d in zip(barcodes.births, barcodes.deaths)]
    b2_list = [PseudoBarcode(b, d, target_features.dimension) for b, d in zip(target_features.births, target_features.deaths)]

    for b in b1_list:
        b.dim = 0
    for b in b2_list:
        b.dim = 0

    loss_val = ph_loss(b1_list, b2_list, epsilon=epsilon)
    return float(loss_val)


def detect_bifurcations(temporal_barcodes: List[TemporalBarcode], threshold: float = 1.0) -> List[BifurcationEvent]:
    """Deprecated: use ``TemporalAnalysisResult.detect_bifurcations()`` instead."""
    warnings.warn(
        "detect_bifurcations is deprecated; use TemporalAnalysisResult.detect_bifurcations()",
        DeprecationWarning,
        stacklevel=2,
    )
    return _detect_bifurcations_impl(temporal_barcodes, threshold)


class TemporalAnalysisResult(BaseModel):
    """Combined result of temporal topology analysis, containing barcodes and bifurcations."""
    barcodes: Dict[int, TemporalBarcode]
    bifurcations: List[BifurcationEvent]

    def detect_bifurcations(self, threshold: float = 1.0) -> List[BifurcationEvent]:
        """Recompute bifurcations from this result's barcodes at a new threshold.

        What is Being Computed?:
            Re-runs the topological-flux bifurcation detection over the
            barcodes carried by this result, at a user-supplied
            sensitivity threshold. The pre-stored ``self.bifurcations``
            field is left untouched.

        Args:
            threshold: Sensitivity threshold for the |Δ active| spike test.

        Returns:
            A fresh list of :class:`BifurcationEvent`.
        """
        return _detect_bifurcations_impl(list(self.barcodes.values()), threshold)


def analyze_temporal_evolution(
    point_cloud_sequence: List[np.ndarray],
    parameter_values: List[float],
    dimensions: List[int] = [0, 1],
    epsilon: float = 0.5,
    max_rips_dimension: int = 2,
    bifurcation_threshold: float = 1.0
) -> TemporalAnalysisResult:
    """Orchestrates the full temporal topology analysis pipeline.

    Args:
        point_cloud_sequence: Sequence of point clouds evolving in time.
        parameter_values: Corresponding temporal parameters (e.g., timestamps).
        dimensions: Homological dimensions to track.
        epsilon: Distance threshold for Rips complex.
        max_rips_dimension: Max dimension for Rips complex construction.
        bifurcation_threshold: Sensitivity threshold for detecting phase transitions.

    Returns:
        TemporalAnalysisResult containing the tracked barcodes and detected bifurcations.
    """
    if len(point_cloud_sequence) != len(parameter_values):
        raise ValueError("Length of point_cloud_sequence must match length of parameter_values.")

    zigzag_seq = build_union_intersection_zigzag(
        point_cloud_sequence,
        epsilon=epsilon,
        max_dimension=max_rips_dimension,
        complex_type='Rips'
    )

    barcodes = {}
    for dim in dimensions:
        tb = compute_temporal_homology(zigzag_seq, dimension=dim, parameters=parameter_values, field='Z2')
        barcodes[dim] = tb

    bifurcations = _detect_bifurcations_impl(list(barcodes.values()), bifurcation_threshold)

    return TemporalAnalysisResult(
        barcodes=barcodes,
        bifurcations=bifurcations
    )
