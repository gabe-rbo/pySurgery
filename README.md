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

1. **Exact Integer Homology:** Computations of $H_n(X; \mathbb{Z})$ and corresponding torsion invariants are resolved using exact Smith Normal Form (SNF) over $\mathbb{Z}$. The library features a state-of-the-art, row-pivoted SNF pre-processor in Python using NumPy object arrays, providing a $10\times - 50\times$ speedup over standard SymPy implementations. For massive matrices, it employs a multi-threaded Julia backend featuring an optimal $\mathcal{O}(V+E)$ leaf-peeling pre-processor.
2. **State-of-the-Art Performance & Scaling:** The package natively constructs Alpha, Vietoris-Rips, and Witness complexes without opaque external C++ wrappers. It utilizes **NumPy vectorization** (SIMD-accelerated 2D/3D geometry), **Numba JIT compilation** (for high-speed exact finite-field linear algebra), and **Zero-Allocation solvers** that eliminate Python's $O(N^2)$ memory overhead during incremental basis extraction.
3. **Hardware-Accelerated Geometric Metrics:** Continuous geometric operations, including Sinkhorn-approximated Gromov-Wasserstein distances, Local PCA tangent-space estimations, and continuous relaxations of topological invariants, are fully vectorized and JIT-compiled via **JAX/XLA**.
4. **Seamless Language Interoperability:** The tri-language bridge automatically orchestrates **Multi-threaded Julia** execution and handles complex signal management (`PYTHON_JULIACALL_HANDLE_SIGNALS`), ensuring a stable and crash-free experience during heavy parallel topological evaluations.

---

## Comprehensive Capabilities

pySurgery goes beyond standard persistent homology, exposing the deep algebraic structures required for manifold classification and surgery theory.

### 1. Combinatorial Topology & Complex Generation
* **Discrete Spaces:** Robust native classes for `SimplicialComplex`, `CWComplex`, and `ChainComplex` with lazy-evaluated, cached topological properties (f-vectors, boundaries).
* **Topological Simplification:** Rigorous `.simplify()` method for homotopy-equivalent reduction via Link Condition edge contractions and high-performance `.quick_mapper()` for modularity-based structural summarization.
* **Homotopy Equivalence:** Systematic reduction via simplicial **Collapses** (free face removal) and **Discrete Morse Theory** (Forman matching), yielding minimal chain complexes while preserving mathematical integrity.
* **Massive Point Clouds:** Native construction of memory-efficient **Alpha Complexes** (2D/3D/ND with EMST connectivity heuristics), **Vietoris-Rips** (via sparse clique enumeration), and **Witness Complexes**.
* **Parameter-Free Reconstruction:** Implementation of the **Crust Algorithm** for adaptive surface and curve reconstruction without distance thresholds.
* **Homology & Cohomology:** Exact computation of Betti numbers and torsion coefficients over $\mathbb{Z}$, $\mathbb{Q}$, and $\mathbb{Z}/p\mathbb{Z}$. Includes Universal Coefficient Theorem (UCT) decompositions for composite moduli.
* **Optimal Generators:** Data-grounded $H_1$ generator extraction, yielding cycle representatives optimized by minimum geometric weight over $\mathbb{F}_2$ annotations.

### 2. Algebraic Topology & Cohomological Operations
* **Cup Products:** Full simplicial implementation of the Alexander-Whitney diagonal approximation to evaluate $\alpha \smile \beta$, exposing the ring structure of cohomology.
* **Characteristic Classes:** Extraction of Stiefel-Whitney classes ($w_i$) and Euler classes ($e$) for manifold tangent bundles and general **Combinatorial Vector Bundles**. Features a local Whitney-Steenrod fast-path for massive manifold meshes.
* **Steenrod Squares:** Cohomology operations $Sq^k: H^p(X; \mathbb{Z}_2) \to H^{p+k}(X; \mathbb{Z}_2)$ implemented via optimized cup-i products.
* **Fundamental Group** ($\pi_1$): Extraction of group presentations via spanning-tree retractions, supporting both raw and optimized (reduced trace) generator modes.
* **Whitehead Torsion:** $K$-theoretic heuristics for evaluating Whitehead groups ($Wh(\pi_1)$) and s-cobordism obstructions.

### 3. 4-Manifold Topology & Intersection Forms
* **Intersection Forms** ($Q$): Rigorous extraction of symmetric bilinear forms $Q: H_2(M) \times H_2(M) \to \mathbb{Z}$ evaluated on fundamental classes.
* **Algebraic Classification:** Exact classification of $\mathbb{Z}$-forms by Rank, Signature, Parity (Type I/II), and Definiteness. Detects foundational components like $E_8$ and Hyperbolic ($H$) forms.
* **Quadratic Refinements:** Evaluation of $q(\alpha)$ refinements to compute the **Arf Invariant** over $\mathbb{Z}/2\mathbb{Z}$ via symplectic basis reductions.
* **Kirby Calculus:** Tracking 4-manifold surgery diagrams through algorithmic handle slides and blow-ups/blow-downs.

### 4. Surgery Theory & High-Dimensional Classification
* **Wall Groups** ($L_n(\pi_1)$): Algorithmic evaluation of surgery obstructions mapping into $L$-groups. Supports product-group decompositions and Shaneson splitting sequences.
* **Twisted Multisignatures:** Multi-threaded Julia kernels for computing exact twisted signatures over group rings $\mathbb{Z}[\pi]$ using roots of unity.
* **Structure Sets** ($\mathcal{S}(M)$): Navigation of the Surgery Exact Sequence, calculating normal invariants over $\mathbb{Z}$ and $\mathbb{Z}/2\mathbb{Z}$ to determine the existence and uniqueness of manifold structures.

### 5. Homeomorphism Certification & Witnesses
* **Dimension-Aware Analyzers:** Specialized homeomorphism classification signals tailored for 2D (genus/orientability), 3D (prime decomposition signals), 4D (Freedman/Donaldson invariants), and 5D+ (Surgery theory).
* **Structured Witnesses:** The `homeomorphism_witness` module does not just return True/False; it generates rigorous certificate objects containing the exact theorems invoked, explicit isometry matrices ($U^T Q_1 U = Q_2$), and explicit delineations of missing obstruction data if surgery is required.

### 6. Multi-Engine Backend Optionality
pySurgery features a flexible backend architecture that allows users to prioritize either environment simplicity or raw performance:
* **`backend='python'`**: Pure-Python execution (enhanced by NumPy/Numba). Requires zero external dependencies. Ideal for rapid prototyping and small-to-medium complexes.
* **`backend='julia'`**: Native integration with the Julia engine. Recommended for massive SNF reductions and high-dimensional manifold classification.
* **`backend='auto'` (Default)**: Automatically detects and leverages the most efficient engine available (Julia > Python).
* **`backend='jax'`**: Specifically used for continuous metric evaluations and differentiable topological approximations.

### 7. Geometric Analysis & Immersion
* **PL Embeddings:** High-performance $\mathcal{O}(N \log N)$ KDTree-bounded broad-phase and exact narrow-phase checks for piecewise-linear self-intersections and immersions.
* **Intrinsic Dimension:** Hardware-accelerated manifold dimension estimators using Maximum Likelihood (Levina-Bickel), Two-NN, and Local PCA tangent-space approximations.
* **Metric Alignment:** Orthogonal Procrustes, discrete Fréchet distances, and JAX-accelerated Entropic Gromov-Wasserstein alignment for comparing ambient metric spaces.
* **Geometrization & Uniformization:** Heuristics for Thurston's 8 geometries, normal surface residual norms, and discrete conformal equivalence metrics for 2D meshes.
* **Gauss-Bonnet & Chern-Gauss-Bonnet:** Tools for verifying the relationship between total curvature and Euler characteristic across dimensions, including 4D Weyl and Q-curvature integrations.

### 8. Integrations & Interoperability
* **JAX:** Differentiable soft-signatures and high-throughput metric tensors.
* **Lean 4:** Export functionality to translate discrete simplicial complexes into formal theorem-prover syntax.
* **PyTorch Geometric:** Bridging topological complexes to graph neural network (GNN) architectures.
* **Trimesh:** Direct import of 3D asset geometries into rigorous CW/Simplicial complexes.

---

## Installation

**Requirements:** Python $\ge 3.12$.

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
Ensure `julia` is available on your `PATH`, then install the required dependencies:
```julia
import Pkg
Pkg.add(["AbstractAlgebra", "PrecompileTools", "Combinatorics", "SparseArrays", "LinearAlgebra", "JSON", "IntegerSmithNormalForm", "Statistics", "Random"])
```
*Optional but recommended for geometric kernels:* `Pkg.add(["Graphs", "SimpleWeightedGraphs", "DelaunayTriangulation"])`.

The bridge will automatically detect the Julia environment and distribute SNF and multisignature kernels across available CPU threads.

---

## Testing

pySurgery utilizes `pytest` for its comprehensive test suite. To run the tests, ensure you have installed the package with test dependencies:

```bash
pip install -e ".[test,all]"
```

### Running Tests Locally

Execute the suite using:
```bash
pytest tests/
```

For a detailed report including skipped tests and reasons:
```bash
pytest -v -rs tests/
```

### CI Dependencies

The test suite requires both Python and Julia environments. On GitHub Actions, we utilize `julia-actions/setup-julia` and install the full set of backends to ensure 100% test coverage with zero skips.

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

## Mathematical Foundations & Bibliography

The algorithms and constructs implemented in **pySurgery** are rigorously grounded in foundational topological literature and modern computational research.

### Foundational Theory

*   **Algebraic Surgery:** Ranicki, A. (1980). *Exact sequences in the algebraic theory of surgery*. Princeton University Press.
*   **Surgery Theory & L-Groups:** Wall, C. T. (1970). *Surgery on compact manifolds*. Academic Press.
*   **4-Manifold Classification:** Freedman, M. H. (1982). The topology of four-dimensional manifolds. *Journal of Differential Geometry*, 17(3), 357-453.
*   **Characteristic Classes:** Milnor, J. W., & Stasheff, J. D. (1974). *Characteristic classes*. Princeton University Press.
*   **K-Theory & Whitehead Torsion:** Milnor, J. W. (1966). Whitehead torsion. *Bulletin of the American Mathematical Society*, 72(3), 358-426.
*   **Bass-Heller-Swan Theorem:** Bass, H., Heller, A., & Swan, R. G. (1964). The Whitehead group of a polynomial extension. *Publications Mathématiques de l'IHÉS*, 22, 61-79.
*   **Shaneson Splitting:** Shaneson, J. L. (1968). Wall's surgery obstruction groups for G x Z. *Annals of Mathematics*, 88(1), 1-67.
*   **Kirby Calculus:** Kirby, R. (1978). A calculus for framed links in S^3. *Inventiones mathematicae*, 45(1), 35-56.
*   **Normal Surface Theory:** Haken, W. (1961). Theorie der Normalflächen. *Acta Mathematica*, 105(3-4), 245-375.
*   **3-Manifold Geometrization:** Thurston, W. P. (1982). Three-dimensional manifolds, Kleinian groups and hyperbolic geometry. *Bulletin of the American Mathematical Society*, 6(3), 357-381.
*   **Simplicial Collapses:** Whitehead, J. H. C. (1939). Simplicial spaces, nuclei and m-groups. *Proceedings of the London Mathematical Society*, 2(1), 241-325.
*   **Steenrod Squares & Cup-i Products:** Steenrod, N. E. (1947). Products of cocycles and extensions of mappings. *Annals of Mathematics*, 48(2), 290-320.
*   **Wu Class:** Wu, W. T. (1950). Classes caractéristiques et i-carrés d'une variété. *Comptes Rendus de l'Académie des Sciences*, 230, 508-511.
*   **Hirzebruch Signature Theorem:** Hirzebruch, F. (1956). *Topological methods in algebraic geometry*. Springer-Verlag.
*   **Smith Normal Form:** Smith, H. J. S. (1861). On systems of linear indeterminate equations and congruences. *Philosophical Transactions of the Royal Society of London*, 151, 293-326.
*   **Farrell-Jones Conjecture:** Farrell, F. T., & Jones, L. E. (1993). Isomorphism conjectures in algebraic K-theory. *Journal of the American Mathematical Society*, 6(2), 249-297.
*   **Surface Classification:** Radó, T. (1925). Über den Begriff der Riemannschen Fläche. *Acta Litt. Sci. Szeged*, 2, 101-121.

### Computational Implementation & Optimization

*   **Computational Topology Foundations:** Edelsbrunner, H., & Harer, J. (2010). *Computational topology: An introduction*. American Mathematical Society.
*   **Optimal Generators:** Dey, T. K., & Wang, Y. (2022). *Computational topology for data analysis*. Cambridge University Press.
*   **Efficient Persistent Homology (Leaf-Peeling):** Bauer, U. (2021). Ripser: efficient computation of Vietoris–Rips persistence barcodes. *Journal of Applied and Computational Topology*, 5, 391-423.
*   **3-Manifold Simplification (Crushing):** Jaco, W., & Rubinstein, J. H. (2003). 0-efficient triangulations of 3-manifolds. *Journal of Differential Geometry*, 65(1), 61-168.
*   **Algorithmic 3-Topology:** Matveev, S. (2003). *Algorithmic topology and classification of 3-manifolds*. Springer Science & Business Media.
*   **Sylvester's Law of Inertia (Exact):** Sylvester, J. J. (1852). A demonstration of the theorem that every homogeneous quadratic polynomial is reducible by real orthogonal substitutions to the form of a sum of positive and negative squares. *Philosophical Magazine*, 4(4), 138-142.
*   **Crust Algorithm:** Amenta, N., Bern, M., & Kamvysselis, M. (1998). A new Voronoi-based surface reconstruction algorithm. *Proceedings of the 25th annual conference on Computer graphics and interactive techniques*, 415-421.
*   **Gromov-Wasserstein Distance:** Peyré, G., Cuturi, M., & Solomon, J. (2016). Gromov-Wasserstein averaging of kernel and distance matrices. *International Conference on Machine Learning*, 2664-2674.
*   **Levina-Bickel MLE:** Levina, E., & Bickel, P. J. (2004). Maximum likelihood estimation of intrinsic dimension. *Advances in Neural Information Processing Systems*, 17.
*   **TwoNN Estimator:** Facco, E., d’Errico, M., Rodriguez, A., & Laio, A. (2017). Estimating the intrinsic dimension of datasets by a minimal neighborhood information. *Scientific Reports*, 7(1), 12140.
*   **CkNN Graph:** Berry, T., & Sauer, T. (2016). Consistent manifold representation for topological data analysis. *Foundations of Data Science*, 1(1), 1-38.
*   **QuickMapper:** Liu, Y., Xie, Z., & Yi, J. (2012). A fast algorithm for computing Mapper. *arXiv preprint arXiv:1209.4319*.

---

## License

Released under the MIT License. See `LICENSE` for details.
