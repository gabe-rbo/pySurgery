def __getattr__(name):
    if name in [
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
    ]:
        from .embedding import (  # noqa: F401
            EmbeddingResult,
            ImmersionResult,
            PLMap,
            ProjectionResult,
            SelfIntersectionReport,
            SimplexIntersectionWitness,
            analyze_embedding,
            check_immersion,
            detect_self_intersections,
            jitter_coordinates,
            project_coordinates,
        )
        return locals()[name]
    if name in [
        "IntrinsicDimensionMethodResult",
        "IntrinsicDimensionResult",
        "TangentBasisResult",
        "estimate_intrinsic_dimension",
        "levina_bickel_mle",
        "local_pca_tangent_space_dimension",
        "local_pca_tangent_basis",
        "twonn",
    ]:
        from .intrinsic_dimension import (  # noqa: F401
            IntrinsicDimensionMethodResult,
            IntrinsicDimensionResult,
            TangentBasisResult,
            estimate_intrinsic_dimension,
            levina_bickel_mle,
            local_pca_tangent_space_dimension,
            local_pca_tangent_basis,
            twonn,
        )
        return locals()[name]
    if name in [
        "exact_sign_of_determinant",
        "exact_signs_of_determinants_batch",
        "exact_sign_of_sum",
        "orientation2d",
        "orientation3d",
        "incircle2d",
        "insphere3d",
    ]:
        from .predicates import (  # noqa: F401
            exact_sign_of_determinant,
            exact_signs_of_determinants_batch,
            exact_sign_of_sum,
            orientation2d,
            orientation3d,
            incircle2d,
            insphere3d,
        )
        return locals()[name]
    if name in [
        "is_single_cycle",
        "is_single_path_or_cycle",
        "intersect_local_stars",
        "PerturbationRepairResult",
        "moser_tardos_repair",
    ]:
        from .perturbation import (  # noqa: F401
            is_single_cycle,
            is_single_path_or_cycle,
            intersect_local_stars,
            PerturbationRepairResult,
            moser_tardos_repair,
        )
        return locals()[name]
    if name in [
        "PoleEstimationResult",
        "estimate_voronoi_poles",
        "CoconeFilterResult",
        "cocone_filter",
        "PruneWalkResult",
        "prune_and_walk",
        "TightCoconeResult",
        "tight_cocone_close",
        "CoconeReconstructionResult",
        "cocone_reconstruction",
    ]:
        from .reconstruction import (  # noqa: F401
            PoleEstimationResult,
            estimate_voronoi_poles,
            CoconeFilterResult,
            cocone_filter,
            PruneWalkResult,
            prune_and_walk,
            TightCoconeResult,
            tight_cocone_close,
            CoconeReconstructionResult,
            cocone_reconstruction,
        )
        return locals()[name]
    if name in [
        "TangentialComplexResult",
        "tangential_complex_reconstruction",
    ]:
        from .tangential_complex import (  # noqa: F401
            TangentialComplexResult,
            tangential_complex_reconstruction,
        )
        return locals()[name]
    if name in [
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
    ]:
        from .geometrization_3d import (  # noqa: F401
            GeometrizationResult,
            NormalSurfaceCandidate,
            PieceDecomposition,
            Triangulated3Manifold,
            analyze_geometrization,
            crush_normal_surface,
            jsj_decomposition,
            normal_surface_candidates,
            normal_surface_matching_matrix,
            prime_decomposition,
        )
        return locals()[name]
    if name in [
        "SurfaceMesh",
        "SurfaceUniformizationResult",
        "circle_packing_uniformization",
        "cotangent_laplacian",
        "discrete_ricci_flow",
        "surface_target_curvature",
        "uniformize_surface",
        "vertex_gaussian_curvature",
    ]:
        from .uniformization import (  # noqa: F401
            SurfaceMesh,
            SurfaceUniformizationResult,
            circle_packing_uniformization,
            cotangent_laplacian,
            discrete_ricci_flow,
            surface_target_curvature,
            uniformize_surface,
            vertex_gaussian_curvature,
        )
        return locals()[name]
    if name in [
        "extract_stiefel_whitney_w2",
        "check_spin_structure",
        "extract_pontryagin_p1",
        "verify_hirzebruch_signature",
    ]:
        from .characteristic_classes import (  # noqa: F401
            extract_stiefel_whitney_w2,
            check_spin_structure,
            extract_pontryagin_p1,
            verify_hirzebruch_signature,
        )
        return locals()[name]
    if name in [
        "verify_gauss_bonnet_2d",
        "verify_chern_gauss_bonnet_4d",
        "chern_gauss_bonnet_integral_expected",
    ]:
        from .gauss_bonnet import (  # noqa: F401
            verify_gauss_bonnet_2d,
            verify_chern_gauss_bonnet_4d,
            chern_gauss_bonnet_integral_expected,
        )
        return locals()[name]
    if name in [
        "NonImmersibilityWitness",
        "ImmersibilityInconclusive",
        "StructuralObstruction",
        "immersion_obstruction_analysis",
    ]:
        from .immersion_obstructions import (  # noqa: F401
            NonImmersibilityWitness,
            ImmersibilityInconclusive,
            StructuralObstruction,
            immersion_obstruction_analysis,
        )
        return locals()[name]
    if name in [
        "PointCloud",
        "SpaceBlock",
    ]:
        from .point_cloud import (  # noqa: F401
            PointCloud,
            SpaceBlock,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
    "IntrinsicDimensionMethodResult",
    "IntrinsicDimensionResult",
    "TangentBasisResult",
    "estimate_intrinsic_dimension",
    "levina_bickel_mle",
    "local_pca_tangent_space_dimension",
    "local_pca_tangent_basis",
    "twonn",
    "exact_sign_of_determinant",
    "exact_signs_of_determinants_batch",
    "exact_sign_of_sum",
    "orientation2d",
    "orientation3d",
    "incircle2d",
    "insphere3d",
    "is_single_cycle",
    "is_single_path_or_cycle",
    "intersect_local_stars",
    "PerturbationRepairResult",
    "moser_tardos_repair",
    "PoleEstimationResult",
    "estimate_voronoi_poles",
    "CoconeFilterResult",
    "cocone_filter",
    "PruneWalkResult",
    "prune_and_walk",
    "TightCoconeResult",
    "tight_cocone_close",
    "CoconeReconstructionResult",
    "cocone_reconstruction",
    "TangentialComplexResult",
    "tangential_complex_reconstruction",
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
    "SurfaceMesh",
    "SurfaceUniformizationResult",
    "circle_packing_uniformization",
    "cotangent_laplacian",
    "discrete_ricci_flow",
    "surface_target_curvature",
    "uniformize_surface",
    "vertex_gaussian_curvature",
    "extract_stiefel_whitney_w2",
    "check_spin_structure",
    "extract_pontryagin_p1",
    "verify_hirzebruch_signature",
    "verify_gauss_bonnet_2d",
    "verify_chern_gauss_bonnet_4d",
    "chern_gauss_bonnet_integral_expected",
    "NonImmersibilityWitness",
    "ImmersibilityInconclusive",
    "StructuralObstruction",
    "immersion_obstruction_analysis",
    "PointCloud",
    "SpaceBlock",
]
