from pysurgery.core.foundations import (
    COVERAGE_MATRIX,
    CONTRACT_VERSION,
    coverage_status_counts,
    get_contract,
)


def test_coverage_matrix_has_entries_and_contract_version():
    assert CONTRACT_VERSION.startswith("2026.04")
    assert CONTRACT_VERSION.endswith("phase10")
    assert len(COVERAGE_MATRIX) >= 5


def test_coverage_status_counts_contains_partial_or_exact():
    counts = coverage_status_counts()
    assert counts.get("exact", 0) + counts.get("partial", 0) >= 1


def test_coverage_matrix_has_no_partial_or_unsupported_entries_after_phase10_closure():
    counts = coverage_status_counts()
    assert counts.get("partial", 0) == 0
    assert counts.get("unsupported", 0) == 0


def test_get_contract_builds_normalized_record():
    c = get_contract(
        analyzer="analyze_homeomorphism_high_dim_result",
        theorem="s-Cobordism / surgery classification",
        theorem_tag="highdim.scobordism.surgery",
        required_inputs=["pi_1", "Wall obstruction"],
    )
    assert c.contract_version == CONTRACT_VERSION
    assert c.required_inputs == ["pi_1", "Wall obstruction"]


def test_highdim_pi1_entry_is_exact_with_completion_certificate_requirement():
    entry = next(
        e
        for e in COVERAGE_MATRIX
        if e.dimension_class == "n>=5"
        and e.pi_family == "pi=1"
        and e.theorem_tag == "highdim.scobordism.surgery"
    )
    assert entry.status == "exact"
    assert any("completion" in req.lower() for req in entry.required_inputs)


def test_highdim_product_group_entry_mentions_assembly_requirement():
    entry = next(
        e
        for e in COVERAGE_MATRIX
        if e.dimension_class == "n>=5" and e.theorem_tag == "highdim.wall.group_ring"
    )
    assert entry.status == "exact"
    assert any("assembly" in req.lower() for req in entry.required_inputs + entry.notes)


def test_3d_entry_mentions_recognition_certificate_requirement():
    entry = next(
        e
        for e in COVERAGE_MATRIX
        if e.dimension_class == "3D" and e.theorem_tag == "3d.poincare.geometrization"
    )
    assert any("recognition" in req.lower() for req in entry.required_inputs)


def test_4d_entry_is_exact_and_mentions_definite_certificate_path():
    entry = next(
        e
        for e in COVERAGE_MATRIX
        if e.dimension_class == "4D" and e.theorem_tag == "4d.freedman.simply_connected"
    )
    assert entry.status == "exact"
    assert any("certificate" in req.lower() for req in entry.required_inputs)
