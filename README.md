<div align="center">
  <table border="0" cellpadding="0" cellspacing="0" style="border: none !important; border-collapse: collapse; width: 100%; background-color: transparent;">
    <tr style="border: none !important; background-color: transparent;">
      <td width="60%" align="left" style="border: none !important; padding: 20px; vertical-align: top; background-color: transparent;">

<div align="center">

# pySurgery

**A high-performance Python library for Computational Surgery Theory.**

</div>

pySurgery is a high-performance Python library for exact computational algebraic topology, computational surgery theory, and geometric analysis. It is designed to compute discrete topological invariants—such as integer homology (including exact torsion), intersection forms, cup products, L-group obstructions, and homeomorphism certificates—at massive scale. The library leverages a tri-language architecture (**Python**, **Julia**, and **JAX/XLA**) to rigorously evaluate complex topological structures, scaling to point clouds exceeding 100,000 points while operating within strict memory bounds.

<div align="center">

[![Tests](https://github.com/gabe-rbo/pysurgery/actions/workflows/tests.yml/badge.svg)](https://github.com/gabe-rbo/pysurgery/actions/workflows/tests.yml)
[![Lint](https://github.com/gabe-rbo/pysurgery/actions/workflows/lint.yml/badge.svg)](https://github.com/gabe-rbo/pysurgery/actions/workflows/lint.yml)

</div>
  </td>
<td width="40%" align="center" style="border: none !important; padding: 10px; vertical-align: middle; background-color: transparent;">
       <img width="2000" height="2000" alt="pySurgery logo" src="https://github.com/user-attachments/assets/6be087fc-9c87-433d-aa2e-b9762df6fc89" style="display: block; border: none !important; max-width: 100%; height: auto;" />
    </td>
    </tr>
  </table>
</div>


## Architectural Principles

pySurgery relies on three foundational pillars to ensure both scale and mathematical fidelity:

1. **Exact Integer Homology:** Computations of $H_n(X; \mathbb{Z})$ and corresponding torsion invariants are resolved using exact Smith Normal Form (SNF) over $\mathbb{Z}$. To prevent $\mathcal{O}(N^4)$ coefficient swell on massive matrices, the library employs a multi-threaded Julia backend featuring an optimal $\mathcal{O}(V+E)$ leaf-peeling pre-processor.
2. **Native Sparse Complexes:** The package natively constructs Alpha, Vietoris-Rips, and Witness complexes without opaque external C++ wrappers. It utilizes memory-efficient combinatorial algorithms (bounded sparse DFS, exact algebraic circumradius filtrations) that drop memory complexity from $\mathcal{O}(N^2)$ to $\mathcal{O}(N)$.
3. **Hardware-Accelerated Geometric Metrics:** Continuous geometric operations, including Sinkhorn-approximated Gromov-Wasserstein distances, Local PCA tangent-space estimations, and continuous relaxations of topological invariants, are fully vectorized and JIT-compiled via **JAX/XLA**.

---

## Comprehensive Capabilities

pySurgery goes beyond standard persistent homology, exposing the deep algebraic structures required for manifold classification and surgery theory.

### 1. Combinatorial Topology & Complex Generation
* **Discrete Spaces:** Robust native classes for `SimplicialComplex`, `CWComplex`, and `ChainComplex` with lazy-evaluated, cached topological properties (f-vectors, boundaries).
* **Massive Point Clouds:** Native construction of memory-efficient **Alpha Complexes** (2D/3D via Delaunay circumradius filtration), **Vietoris-Rips** (via sparse clique enumeration), and **Witness Complexes** (via Farthest Point Sampling).
* **Homology & Cohomology:** Exact computation of Betti numbers and torsion coefficients over $\mathbb{Z}$, $\mathbb{Q}$, and $\mathbb{Z}/p\mathbb{Z}$. Includes Universal Coefficient Theorem (UCT) decompositions for composite moduli.
* **Optimal Generators:** Data-grounded $H_1$ generator extraction, yielding cycle representatives optimized by minimum geometric weight over $\mathbb{F}_2$ annotations.

### 2. Algebraic Topology & Cohomological Operations
* **Cup Products:** Full simplicial implementation of the Alexander-Whitney diagonal approximation to evaluate $\alpha \smile \beta$, exposing the ring structure of cohomology.
* **Characteristic Classes:** Extraction of Stiefel-Whitney classes (e.g., $w_2$) via Wu's formula to evaluate spin/pin structures and orientability.
* **Fundamental Group ($\pi_1$):** Extraction of group presentations via spanning-tree retractions, supporting both raw and optimized (reduced trace) generator modes.
* **Whitehead Torsion:** $K$-theoretic heuristics for evaluating Whitehead groups ($Wh(\pi_1)$) and s-cobordism obstructions.

### 3. 4-Manifold Topology & Intersection Forms
* **Intersection Forms ($Q$):** Rigorous extraction of symmetric bilinear forms $Q: H_2(M) \times H_2(M) \to \mathbb{Z}$ evaluated on fundamental classes.
* **Algebraic Classification:** Exact classification of $\mathbb{Z}$-forms by Rank, Signature, Parity (Type I/II), and Definiteness. Detects foundational components like $E_8$ and Hyperbolic ($H$) forms.
* **Quadratic Refinements:** Evaluation of $q(\alpha)$ refinements to compute the **Arf Invariant** over $\mathbb{Z}/2\mathbb{Z}$ via symplectic basis reductions.
* **Kirby Calculus:** Tracking 4-manifold surgery diagrams through algorithmic handle slides and blow-ups/blow-downs.

### 4. Surgery Theory & High-Dimensional Classification
* **Wall Groups ($L_n(\pi_1)$):** Algorithmic evaluation of surgery obstructions mapping into $L$-groups. Supports product-group decompositions and Shaneson splitting sequences.
* **Twisted Multisignatures:** Multi-threaded Julia kernels for computing exact twisted signatures over group rings $\mathbb{Z}[\pi]$ using roots of unity.
* **Structure Sets ($\mathcal{S}(M)$):** Navigation of the Surgery Exact Sequence, calculating normal invariants over $\mathbb{Z}$ and $\mathbb{Z}/2\mathbb{Z}$ to determine the existence and uniqueness of manifold structures.

### 5. Homeomorphism Certification & Witnesses
* **Dimension-Aware Analyzers:** Specialized homeomorphism classification signals tailored for 2D (genus/orientability), 3D (prime decomposition signals), 4D (Freedman/Donaldson invariants), and 5D+ (Surgery theory).
* **Structured Witnesses:** The `homeomorphism_witness` module does not just return True/False; it generates rigorous certificate objects containing the exact theorems invoked, explicit isometry matrices ($U^T Q_1 U = Q_2$), and explicit delineations of missing obstruction data if surgery is required.

### 6. Geometric Analysis & Immersion
* **PL Embeddings:** High-performance $\mathcal{O}(N \log N)$ KDTree-bounded broad-phase and exact narrow-phase checks for piecewise-linear self-intersections and immersions.
* **Intrinsic Dimension:** Hardware-accelerated manifold dimension estimators using Maximum Likelihood (Levina-Bickel), Two-NN, and Local PCA tangent-space approximations.
* **Metric Alignment:** Orthogonal Procrustes, discrete Fréchet distances, and JAX-accelerated Entropic Gromov-Wasserstein alignment for comparing ambient metric spaces.
* **Geometrization & Uniformization:** Heuristics for Thurston's 8 geometries, normal surface residual norms, and discrete conformal equivalence metrics for 2D meshes.

### 7. Integrations & Interoperability
* **JAX:** Differentiable soft-signatures and high-throughput metric tensors.
* **Lean 4:** Export functionality to translate discrete simplicial complexes into formal theorem-prover syntax.
* **PyTorch Geometric:** Bridging topological complexes to graph neural network (GNN) architectures.
* **Trimesh:** Direct import of 3D asset geometries into rigorous CW/Simplicial complexes.

---

## Installation

**Requirements:** Python $\ge 3.10$.

### 1. Python Package

Install directly from PyPI:

```bash
pip install pysurgery
```

Or install from source:

```bash
git clone https://github.com/gabe-rbo/pySurgery.git
cd pySurgery
pip install -e .
```

### 2. High-Performance Backends (Required for Scale)

To process point clouds exceeding $N=1,000$ or to compute exact integer torsion on dense manifolds, the underlying accelerator backends must be configured.

**JAX (GPU/TPU Acceleration):**
```bash
pip install "pysurgery[ml]"
```

**Julia (Exact Integer Engine):**
Ensure `julia` is available on your `PATH`, then install the required exact algebra dependencies:
```julia
import Pkg
Pkg.add(["AbstractAlgebra", "PrecompileTools", "Combinatorics"])
```

The bridge will automatically detect the Julia environment and distribute SNF and multisignature kernels across available CPU threads.

---

## Documentation and Examples

For comprehensive theoretical backgrounds and executable pipelines, refer to the tutorial notebooks located in `examples/`. 

The curriculum covers:
1. Exact algebra and chain complexes from scratch.
2. Homology, cohomology, and the Alexander-Whitney cup product.
3. Surgery exact sequences and structure-set navigation.
4. End-to-end topological certification and homeomorphism witness generation.

---

## Academic Reference

If you utilize pySurgery in your research, please refer to the `CITATION.cff` file for appropriate attribution. 

## Sessions Reference

The algorithms implemented in pySurgery are grounded in several foundational publications:

* **Alexander-Whitney Cup Product:** Grounded in the classical simplicial diagonal approximation (Alexander & Whitney, 1949).
* **Bass-Heller-Swan Decomposition:** Used for Whitehead group obstructions and $K$-theory computations.
* **CkNN (Continuous k-Nearest Neighbors):** Graph construction robust to varying density (Berry & Sauer, 2016).
* **Freedman's Classification:** Topological classification of simply-connected 4-manifolds (Freedman, 1982).
* **Gromov-Wasserstein Distance:** Entropic approximation using JAX-accelerated Sinkhorn iterations (Peyré, Cuturi, et al., 2016).
* **Kirby Calculus:** Implementation of handle slide and blow-up mechanics (Kirby, 1970).
* **Levina-Bickel MLE:** Intrinsic dimension estimation via maximum likelihood (Levina & Bickel, 2004).
* **Orthogonal Procrustes:** Matrix alignment for geometric comparison (Schönemann, 1966).
* **QuickMapper:** High-performance topological mapping and simplification (Narang et al., 2015).
* **Smith Normal Form (SNF):** Exact integer matrix decomposition used for homology and torsion computations.
* **Stiefel-Whitney Classes:** Computation via Wu's formula for orientability and Spin structures (Wu, 1950).
* **Thurston's Geometrization:** Heuristics for 3-manifold classification (Thurston, 1982).
* **TwoNN:** Intrinsic dimension estimation using two nearest neighbors (Facco et al., 2017).
* **Wall Groups ($L$-theory):** Based on the surgery obstruction classification (Wall, 1970).
* **Whitney Embedding:** Piecewise-linear immersion and embedding checks (Whitney, 1944).
* **Computational Topology:** Algorithms extending frameworks from *Computational Topology for Data Analysis* (Dey & Wang).

---

## License

Released under the MIT License. See `LICENSE` for details.
