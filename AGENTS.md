# AGENTS.md

## Project orientation
- `pysurgery` is an exact-first computational topology/surgery library; core goal is certified integer invariants and theorem-scoped homeomorphism decisions (`README.md`, `pysurgery/core/foundations.py`).
- Public API is re-export heavy from `pysurgery/__init__.py`; when adding/removing symbols, keep exports and tests aligned.
- Existing AI guidance source scan (`**/{... ,README.md}`) resolves to `README.md` only in this repo.

## Architecture map (read this first)
- `pysurgery/core/`: algebraic primitives (`ChainComplex`, intersection forms, pi1/group-ring utilities, theorem-tag/contract metadata).
- `pysurgery/homeomorphism.py`: dimension-dispatch analyzers returning structured `HomeomorphismResult` (`status`, `theorem_tag`, `exact`, certificates).
- `pysurgery/homeomorphism_witness.py`: promotes successful analyzer results into explicit witness objects (`HomeomorphismWitnessResult`).
- `pysurgery/bridge/julia_bridge.py`: singleton Julia backend (`juliacall`) used for sparse exact operations and fast triangulation/cup products.
- `pysurgery/integrations/*.py`: adapters from external ecosystems (GUDHI, trimesh, PyG, JAX, Lean).

## Data-flow patterns you should preserve
- Typical pipeline is: external data -> integration bridge -> boundary matrices/cells -> `ChainComplex` -> invariants/analyzer -> witness.
- Example end-to-end path: `examples/12_torus_surgery.ipynb` builds point clouds, triangulates via `triangulate_surface`, runs `analyze_homeomorphism_2d_result`, then `build_homeomorphism_witness`.
- High-dimensional analyzers accumulate a decision DAG and typed certificates; preserve certificate fields when refactoring (`decision_dag`, homotopy/product certificates in `homeomorphism.py`).

## Exactness and status conventions (project-specific)
- Exact-first is default; approximate behavior is explicit via `allow_approx=True` and should emit warnings, not silent downgrades (`pysurgery/integrations/gudhi_bridge.py`).
- Do not collapse result statuses: this codebase distinguishes `success`, `impediment`, `inconclusive`, `surgery_required` and tests depend on these semantics.
- `theorem_tag` + `contract_version` are part of API contract (`tests/test_theorem_tags.py`, `pysurgery/core/foundations.py`).

## Dependency and integration boundaries
- Optional deps are guarded via flags/import guards (`HAS_GUDHI`, `HAS_TRIMESH`, `HAS_TORCH`, Julia `available`). Maintain graceful fallback/ImportError behavior (`tests/test_integration_scope.py`).
- Julia path is preferred where implemented; Python fallback should warn once and keep shape/type compatibility (`tests/test_gudhi_bridge_unit.py`).
- Keep Python-side indices zero-based when calling Julia bridge methods; backend handles index lift (`pysurgery/bridge/julia_bridge.py`).

## Developer workflows
- Install editable package: `pip install -e .`
- Optional extras: `pip install -e ".[tda,mesh,ml]"`
- Run full tests: `pytest -q`
- Target critical suites while iterating:
  - `pytest -q tests/test_homeomorphism.py tests/test_homeomorphism_witness.py`
  - `pytest -q tests/test_gudhi_bridge_unit.py tests/test_integration_scope.py`
  - `pytest -q tests/test_tutorial_coverage_matrix.py`
- Validate tutorial coverage contract directly: `python scripts/validate_tutorial_coverage.py`

## Editing heuristics for agents
- Prefer adding new theorem/certificate behavior through typed dataclasses and normalization helpers, then wire through both analyzer and witness layers.
- Mirror existing test style: lightweight synthetic `ChainComplex` fixtures with sparse matrices to pin theorem-branch behavior.
- When changing integration bridges, add tests for both accelerated and fallback paths (monkeypatch `julia_engine.available` / optional dependency flags).

