# AGENTS.md

## Project map
- `pySurgery` is an exact-first computational algebraic topology / surgery-theory library.
- The public API is re-exported from `pysurgery/__init__.py`; internal structure starts in `pysurgery/core/`.
- Core contract metadata lives in `pysurgery/core/foundations.py` (`CONTRACT_VERSION`, `COVERAGE_MATRIX`, `AnalyzerContract`).
- Decision layers are split by responsibility: `pysurgery/homeomorphism.py` computes structured results, and `pysurgery/homeomorphism_witness.py` upgrades them into certificates/witnesses.
- Surgery exact-sequence and obstruction modeling live in `pysurgery/structure_set.py` and `pysurgery/wall_groups.py`.

## What to read first
- `README.md` for the project’s exactness-first philosophy and notebook-driven examples.
- `examples/08_certificate_workflows.ipynb` and `examples/10_witness_builder_reference.ipynb` for witness/certificate patterns.
- `examples/11_capstone_end_to_end_workflow.ipynb` for the full pipeline from `extract_pi_1_with_traces(...)` to `build_homeomorphism_witness(...)`.

## Project conventions
- Exactness is the default: APIs usually expose `allow_approx=False`; only opt into approximation for exploratory work.
- Result objects are typed and status-driven (`success`, `impediment`, `inconclusive`, `surgery_required`) with `exact`, `missing_data`, and `certificates` fields.
- Certificate objects typically require `provided`, `exact`, and `validated`, and many expose `decision_ready()` helpers.
- Use theorem tags from `pysurgery/core/theorem_tags.py`; the canonical tags are mirrored in the coverage matrix.

## Coverage and contracts
- Notebook coverage is enforced by `pysurgery/utils/tutorial_coverage_validator.py` against `docs/tutorial_coverage.json`.
- That validator checks both notebook snippets and theorem-tag coverage against `COVERAGE_MATRIX`.
- Keep `docs/tutorial_coverage.json` and `pysurgery/coverage_matrix.yaml` aligned with `pysurgery/core/foundations.py`.
- If `CONTRACT_VERSION` changes, update the coverage files together; the current phase is `2026.04-phase10`.

## Developer workflows
- Install from source with `pip install -e .`; optional extras are `pysurgery[tda]`, `pysurgery[mesh]`, and `pysurgery[all]`.
- Validate tutorial coverage with `python scripts/validate_tutorial_coverage.py`.
- Benchmark fundamental-group generator modes with `python scripts/benchmark_pi1_modes.py` or `--force-python`.
- Run `pytest` after touching contracts, witness builders, or integration adapters.

## Tests worth targeting
- `tests/test_foundations.py` for contract and matrix invariants.
- `tests/test_tutorial_coverage_matrix.py` for notebook/coverage sync.
- `tests/test_homeomorphism_witness.py` for certificate assembly and witness status propagation.
- `tests/test_integration_scope.py` for optional-dependency import guards.

## Integration boundaries
- Optional dependencies are isolated behind bridge modules in `pysurgery/bridge/` and `pysurgery/integrations/`.
- Preserve import guards and fallback behavior when editing `gudhi`, `trimesh`, `jax`, or Julia-facing code.

## Source of truth
- Treat `pysurgery/core/foundations.py`, `pysurgery/coverage_matrix.yaml`, and `docs/tutorial_coverage.json` as the contract trio.
- Update this file whenever the contract version, theorem tags, or coverage workflow changes.
