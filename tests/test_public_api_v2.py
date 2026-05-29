import importlib
import pytest

_REQUIRED = {
    "RationalDGA", "RationalCohomologyAlgebra", "RationalMinimalModelResult",
    "RationalHomotopyGroup", "FormalityResult", "MasseyProductsResult",
    "sullivan_minimal_model", "is_formal_space", "extract_massey_products",
    "rational_homotopy_group",
    "SteenrodAlgebra", "AdamsE2Page", "adams_e2_page",
    "compute_rational_and_adams", "synthesize_homotopy_group_with_e_infinity",
    "HomotopyGroupApproximation",
    "Handle", "HandleDecomposition", "cw_complex_to_handle_decomposition",
    "Barcode", "BarcodeResult", "compute_barcodes_exact",
    "compute_zigzag_persistence",
    "TemporalBarcode", "analyze_temporal_evolution",
    "NonImmersibilityWitness", "immersion_obstruction_analysis",
    "compute_l_group_rational", "prime_local_obstruction_report",
    "InteractiveAdamsResolver", "LeanFormalAdamsResolver",
    "ConvergedAdamsPage", "UserVerifiedDifferential", "LeanProofAttempt",
    "LinkingComputationError", "BettiTrackingError",
}


@pytest.mark.parametrize("name", sorted(_REQUIRED))
def test_v2_api_re_exported(name):
    pysurgery = importlib.import_module("pysurgery")
    assert hasattr(pysurgery, name), f"pysurgery.{name} is not re-exported"


def test_version_matches_pyproject():
    import pysurgery
    from importlib.metadata import version
    assert pysurgery.__version__ == version("pysurgery")