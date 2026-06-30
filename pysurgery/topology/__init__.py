def __getattr__(name):
    if name in ["ChainComplex", "CWComplex", "SimplicialComplex", "DynamicComplex"]:
        from .complexes import (  # noqa: F401
            ChainComplex,
            CWComplex,
            SimplicialComplex,
            DynamicComplex,
        )
        return locals()[name]
    if name in [
        "FundamentalGroup",
        "GroupPresentation",
        "extract_pi_1",
        "extract_pi_1_with_traces",
        "simplify_presentation",
        "infer_standard_group_descriptor",
    ]:
        from .fundamental_group import (  # noqa: F401
            FundamentalGroup,
            GroupPresentation,
            extract_pi_1,
            extract_pi_1_with_traces,
            simplify_presentation,
            infer_standard_group_descriptor,
        )
        return locals()[name]
    if name == "Graph":
        from .graphs import Graph  # noqa: F401
        return Graph
    if name in [
        "FacePairing",
        "FundamentalPolyhedron",
        "construct_fundamental_polyhedron",
        "FiniteGroupOrderResult",
        "UniversalCoverResult",
        "FiniteGroupRing",
        "UniversalCover",
        "Covering",
        "DeckTransformationGroup",
        "GroupAction",
        "GraphCovering",
        "cover_graph",
        "graph_universal_cover",
    ]:
        from .coverings import (  # noqa: F401
            FacePairing,
            FundamentalPolyhedron,
            construct_fundamental_polyhedron,
            FiniteGroupOrderResult,
            UniversalCoverResult,
            FiniteGroupRing,
            UniversalCover,
            Covering,
            DeckTransformationGroup,
            GroupAction,
            GraphCovering,
            cover_graph,
            graph_universal_cover,
        )
        return locals()[name]
    if name in [
        "Pi1Evidence",
        "GroupRingContext",
        "Phase2Readiness",
        "evaluate_phase2_readiness",
    ]:
        from .pi1_group_ring_scaffold import (  # noqa: F401
            Pi1Evidence,
            GroupRingContext,
            Phase2Readiness,
            evaluate_phase2_readiness,
        )
        return locals()[name]
    if name in [
        "Barcode",
        "BarcodeResult",
        "compute_barcodes_exact",
        "compute_zigzag_persistence",
    ]:
        from .persistent_homology import (  # noqa: F401
            Barcode,
            BarcodeResult,
            compute_barcodes_exact,
            compute_zigzag_persistence,
        )
        return locals()[name]
    if name in [
        "TemporalBarcode",
        "BifurcationEvent",
        "TemporalAnalysisResult",
        "analyze_temporal_evolution",
    ]:
        from .temporal_topology import (  # noqa: F401
            TemporalBarcode,
            BifurcationEvent,
            TemporalAnalysisResult,
            analyze_temporal_evolution,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ChainComplex",
    "CWComplex",
    "SimplicialComplex",
    "DynamicComplex",
    "FundamentalGroup",
    "GroupPresentation",
    "extract_pi_1",
    "extract_pi_1_with_traces",
    "simplify_presentation",
    "infer_standard_group_descriptor",
    "Graph",
    "FacePairing",
    "FundamentalPolyhedron",
    "construct_fundamental_polyhedron",
    "FiniteGroupOrderResult",
    "UniversalCoverResult",
    "FiniteGroupRing",
    "UniversalCover",
    "Covering",
    "DeckTransformationGroup",
    "GroupAction",
    "GraphCovering",
    "cover_graph",
    "graph_universal_cover",
    "Pi1Evidence",
    "GroupRingContext",
    "Phase2Readiness",
    "evaluate_phase2_readiness",
    "Barcode",
    "BarcodeResult",
    "compute_barcodes_exact",
    "compute_zigzag_persistence",
    "TemporalBarcode",
    "BifurcationEvent",
    "TemporalAnalysisResult",
    "analyze_temporal_evolution",
]
