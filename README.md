# pySurgery

[![Tests](https://github.com/gabe-rbo/pysurgery/actions/workflows/tests.yml/badge.svg)](https://github.com/gabe-rbo/pysurgery/actions/workflows/tests.yml)
[![Lint](https://github.com/gabe-rbo/pysurgery/actions/workflows/lint.yml/badge.svg)](https://github.com/gabe-rbo/pysurgery/actions/workflows/lint.yml)

**pySurgery** is a Python library for *computational algebraic topology* and *computational surgery theory*: it turns discrete data (CW/simplicial complexes, meshes, TDA pipelines) into **integer** invariants strong enough to go beyond Betti numbers—e.g. **torsion**, **cup products**, **intersection forms**, and (in key cases) **homeomorphism classification signals**.

If you’ve ever felt “persistent homology is informative, but not decisive,” pySurgery is aimed at the next layer of structure.

---

## Why this exists

Most numerical linear-algebra approaches over $\mathbb{R}$ lose information that lives over $\mathbb{Z}$ (especially torsion). pySurgery focuses on *exact* and *structure-aware* invariants:

* **Homology & cohomology over $\mathbb{Z}$** via **Smith Normal Form (SNF)** (captures torsion)
* **Coefficient support** for `Z`, `Q`, and `Z/pZ` (including composite moduli via UCT decomposition)
* **Cup product** (Alexander–Whitney) to extract intersection information
* **Intersection forms** for 4-manifolds (rank/signature/parity)
* Hooks for **surgery-theoretic** obstructions and higher-dimensional classification workflows

---

## Quickstart

### Install

Requires **Python >= 3.10+**.

Project metadata (including version and Python requirement) is defined in `pyproject.toml`.

#### Install from PyPI (recommended)

```bash
pip install pysurgery
```

PyPI package: https://pypi.org/project/pysurgery/

#### Install from source (development)

```bash
git clone https://github.com/gabe-rbo/pySurgery.git
cd pySurgery
pip install -e .
```

Optional dependency groups:

```bash
# Topological Data Analysis integration
pip install "pysurgery[tda]"

# Mesh integration
pip install "pysurgery[mesh]"

# ML-oriented integrations
pip install "pysurgery[ml]"

# Everything optional
pip install "pysurgery[all]"
```


### Julia backend (optional, for large-scale exact integer computations)

pySurgery can offload some exact integer workloads to Julia (recommended for very large sparse exact-$\mathbb{Z}$ computations).

* Ensure `julia` is on your `PATH`
* Install required Julia backend packages in your Julia environment:

```julia
import Pkg
Pkg.add("AbstractAlgebra")
Pkg.add("PrecompileTools")
```

---

## Examples and tutorials

A step-by-step interactive curriculum is provided in `examples/` (Jupyter notebooks), covering topics from algebraic topology basics to surgery-theoretic workflows:

1. `01_basic_homology_and_cohomology.ipynb` — CW complexes, SNF, UCT
2. `02_intersection_forms.ipynb` — 4D classification concepts, spin checks
3. `03_algebraic_surgery.ipynb` — isotropic classes and algebraic surgery mechanics
4. `04_advanced_tda_and_surgery_theory.ipynb` — raw data → GUDHI → cup product → intersection form
5. `05_omni_dimensional_homeomorphisms.ipynb` — how classification behavior changes by dimension
6. `06_kirby_calculus_and_characteristic_classes.ipynb` — characteristic classes & Kirby-style ideas
7. `07_fundamental_group_and_structure_set.ipynb` — $\pi_1$ extraction, Whitehead data, and typed structure-set workflows
8. `08_certificate_workflows.ipynb` — explicit homeomorphism witnesses, obstruction interference, and surgery planning
9. `09_branch_matrix_and_failure_modes.ipynb` — success/impediment/inconclusive/surgery-required branch matrix
10. `10_witness_builder_reference.ipynb` — witness-builder API reference and dispatcher coverage
11. `11_capstone_end_to_end_workflow.ipynb` — one full problem-solving pipeline using the library end-to-end
12. `12_torus_surgery.ipynb` — torus triangulation, homology extraction, and surgery-style 2D witness diagnostics

---

## Mathematical basis

pySurgery bridges discrete geometry to topological classification using several key concepts.

### 1) Homology & cohomology via Smith Normal Form (over $\mathbb{Z}$)

When computing homology,

$$
H_n(X)=\ker(d_n)/\mathrm{im}(d_{n+1})
$$

floating-point linear algebra misses **torsion**. pySurgery implements **Smith Normal Form** over $\mathbb{Z}$ to recover:

* Betti ranks (free part)
* torsion coefficients (e.g. $\mathbb{Z}_k$ factors)

Cohomology is computed via the **Universal Coefficient Theorem**.

### 2) Alexander–Whitney cup product

To classify 4-manifolds, surfaces intersect—cup products encode that intersection.

Given cocycles $\alpha, \beta$:

$$
(\alpha \smile \beta)([v_0,\dots,v_4])=\alpha([v_0,v_1,v_2])\cdot\beta([v_2,v_3,v_4]).
$$

Summing over a fundamental class yields intersection form entries.

pySurgery also provides simplicial cup-$i$ operations (`i >= 0`) for higher-order cochain interactions.

### 3) Intersection forms & 4D classification signals

For 4-manifolds, pySurgery constructs the **intersection form** $Q$, computes rank/signature/parity, and supports workflows inspired by Freedman-style classification (with attention to needed hypotheses/invariants).

### 4) Algebraic surgery

Given a class $x$ representing a “hole,” pySurgery supports algorithmic manipulations that mirror algebraic surgery operations—checking isotropy $Q(x,x)=0$, finding dual classes, and updating lattice data.

### 5) Characteristic classes & spin structures

pySurgery extracts characteristic-class information from intersection-form data; using **Wu’s formula**, it computes $w_2$ (mod 2) to test spin conditions.

### 6) High-dimensional structure signals

In dimensions $n \ge 5$, classification interacts with $\pi_1$, Whitehead torsion, and Wall’s surgery groups $L_n(\pi_1)$. pySurgery includes tooling aimed at discrete extraction of group presentations and surgery-sequence-adjacent invariants (where computationally feasible).

### 7) Dimension-aware behavior

* **2D**: orientability/genus signals via low-dimensional homology
* **3D**: homology-sphere style signals and 3-manifold context
* **4D**: intersection forms + parity/signature style invariants
* **5D+**: surgery-theoretic invariants and $\pi_1$-sensitive obstructions

### 8) Data-grounded optimal homology generators

For discrete simplicial data, pySurgery includes data-grounded generator extraction for $H_1$ (cycle representatives as edge-level elements of the input complex), with Julia-accelerated and Python fallback paths.

The generator/optimality pipeline follows the "Generators and Optimality" framework in:

* Tamal K. Dey and Yusu Wang, *Computational Topology for Data Analysis*.

---

## Computational optimizations

Discrete topology can blow up combinatorially (e.g., large point clouds). pySurgery includes:

1. **Sparse fallbacks** when exact dense integer work is too large
2. **NumPy/Numba acceleration** for inner loops (e.g., cup product sweeps)
3. **Julia bridge** for large exact-$\mathbb{Z}$ workloads when Python becomes the bottleneck

Approximate (floating-point) fallbacks exist in selected large/sparse workflows, but are opt-in where mathematically sensitive and emit explicit warnings when exact integer guarantees are weakened.

### Exactness policy

pySurgery defaults to exact integer-topology workflows whenever practical.

* **Default mode:** exact-first (`allow_approx=False` in APIs that expose it)
* **Approximate mode:** explicitly opt in via `allow_approx=True`
* **Warnings:** approximate paths emit warnings when torsion/exact-cycle guarantees may be weakened

For mathematically rigorous classification/proof-oriented workflows, prefer exact mode and treat approximate mode as exploratory.

#### Example: exact vs approximate fundamental class extraction

```python
from pysurgery.integrations.gudhi_bridge import simplex_tree_to_intersection_form

# Exact-first: raise if exact extraction cannot be completed.
q_exact = simplex_tree_to_intersection_form(st, allow_approx=False)

# Exploratory fallback: allows numerical approximation with warnings.
q_fast = simplex_tree_to_intersection_form(st, allow_approx=True)
```

#### Example: exact vs approximate dimension-aware homeomorphism analysis

```python
from pysurgery.homeomorphism import analyze_homeomorphism_2d

# Exact-first classification attempt.
ok_exact, msg_exact = analyze_homeomorphism_2d(c1, c2, allow_approx=False)

# Permissive fallback for noisy/large pipelines.
ok_fast, msg_fast = analyze_homeomorphism_2d(c1, c2, allow_approx=True)
```

### Homeomorphism witness builders

`pysurgery.homeomorphism` answers: **"is a homeomorphism certified by invariants/obstructions?"**

`pysurgery.homeomorphism_witness` goes further and returns a **structured witness object** when exact data is sufficient.

```python
from pysurgery.homeomorphism_witness import build_homeomorphism_witness

res = build_homeomorphism_witness(c1=c1, c2=c2, dim=2)
if res.status == "success":
    print(res.witness.kind)        # e.g. "surface_classification"
    print(res.witness.theorem)     # theorem branch used
    print(res.witness.certificates)
else:
    print(res.reasoning)
    print(res.missing_data)
```

#### Dimension-specific builders

```python
from pysurgery.homeomorphism_witness import (
    build_surface_homeomorphism_witness,
    build_3d_homeomorphism_witness,
    build_4d_homeomorphism_witness,
    build_high_dim_homeomorphism_witness,
)
```

* `build_surface_homeomorphism_witness(...)` (2D)
* `build_3d_homeomorphism_witness(...)` (3D)
* `build_4d_homeomorphism_witness(...)` (Freedman branch)
* `build_high_dim_homeomorphism_witness(...)` (`n >= 5`, surgery-theoretic branch)

For `n >= 5`, an explicit completion certificate can be supplied via
`homotopy_completion_certificate={"provided": True, "exact": True, "validated": True, ...}`.
Certified success requires this certificate to be decision-ready (`exact` and `validated`).
Legacy `homotopy_witness_hook` inputs are still accepted and bridged for compatibility.
For nontrivial product groups, factor-wise Wall decompositions are exposed as structured surrogate certificates;
exact success still requires assembly-certified product-group obstruction data.

For 3D geometric-recognition completion, you can provide
`recognition_certificate={"provided": True, "exact": True, "validated": True, ...}`.
This allows exact success in non-Poincare branches when the certificate is decision-ready.

For high-dimensional nontrivial product groups, provide
`product_assembly_certificate={"provided": True, "exact": True, "validated": True, ...}`
to upgrade surrogate product decompositions into decision-ready assembly evidence.

#### 4D explicit algebraic witness

In definite 4D cases, the witness may include an integer isometry matrix `U` (stored in `witness.certificates["isometry_matrix"]`) such that:

$$
U^T Q_1 U = Q_2.
$$

This is an explicit algebraic certificate for the lattice isomorphism part of the classification branch.

#### Interpreting witness results

* `status="success"`: exact theorem/certificate path produced a witness.
* `status="inconclusive"`: not enough exact data to construct a certified witness (check `missing_data`).
* `status="surgery_required"`: obstruction branch indicates surgery-level impediment instead of direct witness construction.

`witness.exact` indicates whether the returned witness is exact-topology certified (`True`) or exploratory/heuristic (`False`).

---

## Package layout

```text
pysurgery/
├── core/
│   ├── cup_product.py
│   ├── complexes.py
│   ├── characteristic_classes.py
│   ├── exceptions.py
│   ├── fundamental_group.py
│   ├── group_rings.py
│   ├── intersection_forms.py
│   ├── k_theory.py
│   ├── kirby_calculus.py
│   ├── math_core.py
│   └── quadratic_forms.py
├── bridge/
│   ├── julia_bridge.py
│   └── surgery_backend.jl
├── integrations/
│   ├── gudhi_bridge.py
│   ├── trimesh_bridge.py
│   ├── pytorch_geometric_bridge.py
│   ├── jax_bridge.py
│   └── lean_export.py
```

Key modules:

* `pysurgery.core.complexes` — `CWComplex`, `ChainComplex`
* `pysurgery.core.intersection_forms` — lattice math, parity checks, surgery engine
* `pysurgery.core.math_core` — SNF / exact integer computation core
* `pysurgery.homeomorphism` — dimension-aware analyzers
* `pysurgery.homeomorphism_witness` — explicit witness/certificate builders
* `pysurgery.integrations`

  * `gudhi_bridge` — TDA point clouds → topology objects
  * `trimesh_bridge` — meshes → topology objects
  * `pytorch_geometric_bridge` — graph/ML integration
  * `jax_bridge` — differentiable / approximation-oriented integration ideas

---

## Contributing

PRs are welcome—especially for:

* minimal examples (“one function call → one invariant” demos)
* improved docs around mathematical assumptions and failure modes
* performance improvements and additional exact-$\mathbb{Z}$ backends
* more datasets and reproducible example pipelines

If you’re unsure where to start, open an issue describing your use-case and dataset type.

---

## Roadmap (selected)

* Non-abelian / twisted coefficient systems ($\pi_1$-sensitive homology)
* Deeper algebraic surgery tooling (Ranicki-style formulations)
* Learning on structure-set features via GNNs (experimental)

---

## License

MIT License. See `LICENSE`.
