# pysurgery

`pysurgery` is a world-class, high-performance Python library for Computational Algebraic Topology, with a specialized focus on **Mathematical Surgery Theory**. 

It provides an exact, rigorous framework for transforming discrete geometric data (point clouds, meshes, simplicial trees) into profound topological invariants. It goes far beyond simply computing Betti numbers—`pysurgery` automatically derives intersection forms, computes Wall group surgery obstructions, and applies major topological classification theorems (Freedman, Perelman, Smale) to conclusively determine if two manifolds are homeomorphic.

## Features

- **Exact Homology & Cohomology**: Computes free ranks (Betti numbers) and exact torsion coefficients ($H_n(M; \mathbb{Z})$) using highly optimized Smith Normal Form (SNF) reductions.
- **Intersection Forms**: Automated derivation of intersection forms via the Alexander-Whitney Cup Product. Exact classification (Rank, Signature, Parity).
- **Algebraic Surgery**: Capable of "surgering out" topological impediments by extracting isotropic homology classes, computing orthogonal projections onto hyperbolic planes, and utilizing Hermite Normal Forms for exact lattice reduction.
- **Homeomorphism Analysis**: Contains an omni-dimensional analyzer that mathematically proves whether two manifolds are homeomorphic:
  - **2D**: Genus and orientability.
  - **3D**: Homology spheres and Perelman's resolution boundaries.
  - **4D**: Freedman's Classification Theorem.
  - **Higher Dimensions**: The s-Cobordism Theorem and Smale's generalized Poincaré Conjecture.
- **Deep Integration Bridges (`pysurgery.integrations`)**:
  - `gudhi`: SimplexTree extraction and persistent signature landscapes.
  - `trimesh`: 3D CAD mesh to CW Complex conversion.
  - `PyTorch Geometric`: Graph topology simplification.
  - `JAX`: Differentiable topology and signature-based neural network loss functions.
  - `Julia`: Offloads extreme sparse SNF calculations to highly optimized Julia backends.
  - `Lean 4`: Generation of Algebraic Isomorphism Certificates for formal theorem proving.

## Installation

Ensure you have Python 3.10+ installed.

```bash
git clone https://github.com/your-username/pysurgery.git
cd pysurgery
pip install .
```

To take full advantage of the integrations, you may also install the optional dependencies:
```bash
pip install gudhi trimesh sympy scipy pydantic jax jaxlib torch_geometric
```
If you wish to use the ultra-high performance Julia backend for sparse Smith Normal Form, ensure the Julia executable is in your PATH.

## How to Learn `pysurgery`

The `examples/` directory has been curated into a structured, step-by-step tutorial series designed to take you from basic Algebraic Topology to advanced Surgery Theory. We highly recommend running them in order:

### 1. `01_basic_homology_and_cohomology.py`
Learn the basics of constructing a `CWComplex`, building `ChainComplex` objects, and utilizing our heavily optimized Smith Normal Form engine to extract Homology (Betti numbers + Torsion) and Cohomology bases.

### 2. `02_intersection_forms.py`
Dive into 4-manifolds! Learn how to define `IntersectionForm` matrices, calculate signatures, identify parity (Type I vs Type II), and classify famous lattices like $S^2 \times S^2$, $\mathbb{CP}^2$, and $E_8$.

### 3. `03_algebraic_surgery.py`
The core of Surgery Theory. Learn how to identify topological "impediments," find isotropic vectors, and use `perform_algebraic_surgery` to mathematically transform one manifold into another (e.g., surgering $S^2 \times S^2$ into the Sphere $S^4$).

### 4. `04_advanced_tda_and_surgery_theory.ipynb`
An interactive Jupyter Notebook bridging Topological Data Analysis (TDA) with Surgery. It demonstrates the complete pipeline: `Discrete Point Cloud` -> `GUDHI Alpha Complex` -> `Cup Product` -> `Intersection Form` -> `Freedman's Homeomorphism Classification`.

### 5. `05_omni_dimensional_homeomorphisms.py`
Explore the generalized `pysurgery.analyze_homeomorphism_*` suite. See how the library shifts its mathematical logic depending on the dimension of the input data (2D surfaces vs 3D Thurston geometries vs 4D Freedman vs 5D+ s-Cobordism).

## Contributing

`pysurgery` is an active research tool. We welcome contributions, particularly in expanding the capabilities of the `JuliaBridge` or implementing twisted homology over group rings.
