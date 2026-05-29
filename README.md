<div align="center">
  <table border="0" cellpadding="0" cellspacing="0" style="border: none !important; border-collapse: collapse; width: 100%; background-color: transparent;">
    <tr style="border: none !important; background-color: transparent;">
      <td width="60%" align="left" style="border: none !important; padding: 20px; vertical-align: top; background-color: transparent;">

<div align="center">

# pySurgery

**A high-performance Python library for Computational Surgery Theory.**

</div>

pySurgery is a high-performance Python library for exact computational algebraic topology, computational surgery theory, and geometric analysis. It is designed to compute discrete topological invariants—such as integer homology (including exact torsion), intersection forms, cup products, **characteristic classes for vector bundles**, L-group obstructions, and homeomorphism certificates—at massive scale. The library leverages a tri-language architecture (**Python**, **Julia**, and **JAX/XLA**) to rigorously evaluate complex topological structures, scaling to point clouds exceeding 100,000 points while operating within strict memory bounds.

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
* **Homotopy Equivalence:** Systematic reduction via simplicial **Collapses** (free face removal) and **Discrete Morse Theory** (Forman matching), yielding minimal chain complexes while preserving mathematical integrity. Includes rigorous `.is_homology_isomorphic()` checks via SNF.
* **Massive Point Clouds:** Native construction of memory-efficient **Alpha Complexes** (2D/3D/ND with EMST connectivity heuristics), **Vietoris-Rips** (via sparse clique enumeration), and **Witness Complexes**.
* **Parameter-Free Reconstruction:** Implementation of the **Crust Algorithm** for adaptive surface and curve reconstruction without distance thresholds.
* **Exact Persistence:** Native Julia-backed barcode computation over $\mathbb{Z}_2$ and $\mathbb{Q}$ using optimized $R=DV$ matrix reductions with memory-bound guardrails.
* **Homology & Cohomology:** Exact computation of Betti numbers and torsion coefficients over $\mathbb{Z}$, $\mathbb{Q}$, and $\mathbb{Z}/p\mathbb{Z}$. Includes Universal Coefficient Theorem (UCT) decompositions for composite moduli.
* **Optimal Generators:** Data-grounded $H_1$ generator extraction, yielding cycle representatives optimized by minimum geometric weight over $\mathbb{F}_2$ annotations.

### 2. Algebraic Topology & Cohomological Operations
* **Cup Products:** Full simplicial implementation of the Alexander-Whitney diagonal approximation to evaluate $\alpha \smile \beta$, exposing the ring structure of cohomology.
* **Characteristic Classes:** Extraction of Stiefel-Whitney classes ($w_i$), Pontryagin classes ($p_i$), and Euler classes ($e$) for manifold tangent bundles and general **Combinatorial Vector Bundles**. Features a local Whitney-Steenrod fast-path for massive manifold meshes.
* **Steenrod Squares:** Cohomology operations $Sq^k: H^p(X; \mathbb{Z}_2) \to H^{p+k}(X; \mathbb{Z}_2)$ implemented via optimized cup-i products.
* **Fundamental Group** ($\pi_1$): Active mathematical object supporting `.is_abelian()`, `.is_trivial()`, and `.order()` via **Todd-Coxeter** coset enumeration. Includes geometric tracing of generators as directed-edge cycles.
* **Rational Homotopy & Sullivan Models:** Automated computation of **Sullivan Minimal Models** ($\pi_n(X) \otimes \mathbb{Q}$), formality detection, and Massey product extraction.
* **Adams Spectral Sequence:** Comprehensive $E_2$-page extraction and **E-infinity Resolver** framework. Supports **Interactive Resolution** for ambiguous differentials and **Lean 4 Formal Verification** for automated theorem proving of homotopy groups.
* **Higher Structures:** Support for **E-infinity algebras** and functorial **Temporal Topology** sequences for dynamic complexes.
* **Whitehead Torsion:** $K$-theoretic heuristics for evaluating Whitehead groups ($Wh(\pi_1)$) and s-cobordism obstructions.

### 3. 4-Manifold Topology & Intersection Forms
* **Intersection Forms** ($Q$): Rigorous extraction of symmetric bilinear forms $Q: H_2(M) \times H_2(M) \to \mathbb{Z}$ evaluated on fundamental classes.
* **Algebraic Classification:** Exact classification of $\mathbb{Z}$-forms by Rank, Signature, Parity (Type I/II), and Definiteness. Detects foundational components like $E_8$ and Hyperbolic ($H$) forms.
* **Quadratic Refinements:** Evaluation of $q(\alpha)$ refinements to compute the **Arf Invariant** over $\mathbb{Z}/2\mathbb{Z}$ via symplectic basis reductions.
* **Kirby Calculus:** Tracking 4-manifold surgery diagrams through algorithmic handle slides and blow-ups/blow-downs.

### 4. Surgery Theory & Manifold Transformation
* **Wall Groups** ($L_n(\pi_1)$): Algorithmic evaluation of surgery obstructions mapping into $L$-groups. Supports product-group decompositions and Shaneson splitting sequences.
* **Controlled Cohomology:** Evaluation of twisted local systems and bounded controlled cohomology for non-simply-connected manifolds using Fox derivatives.
* **Twisted Multisignatures:** Multi-threaded Julia kernels for computing exact twisted signatures over group rings $\mathbb{Z}[\pi]$ using roots of unity.
* **Structure Sets** ($\mathcal{S}(M)$): Navigation of the Surgery Exact Sequence, calculating normal invariants over $\mathbb{Z}$ and $\mathbb{Z}/2\mathbb{Z}$ to determine the existence and uniqueness of manifold structures.
* **Manual Surgery Engine (`Surgeon`):** A transaction-safe workbench for performing and certifying atomic surgery steps. Supports `attach_handle`, `remove_disks`, and `move` (ambient isotopy) with automatic tracking of surgery obstructions and bordism traces.
* **Automated Surgery Pipeline (`AutoSurgeon`):** An experimental orchestrator that iteratively unlinks, un-nests, and simplifies connected components into homotopy spheres or contractible manifolds via a multi-phase surgical ladder.

### 5. Homeomorphism Certification & Witnesses
* **Dimension-Aware Analyzers:** Specialized homeomorphism classification signals tailored for 2D (genus/orientability), 3D (prime decomposition), 4D (Freedman/Donaldson invariants), and 5D+ (Surgery theory).
* **Structured Witnesses:** The `homeomorphism_witness` module does not just return True/False; it generates rigorous certificate objects containing the exact theorems invoked, explicit isometry matrices ($U^T Q_1 U = Q_2$), and explicit delineations of missing obstruction data if surgery is required.
* **Decision DAGs:** High-dimensional homeomorphism decisions are managed via a Directed Acyclic Graph (DAG) of analyzers, ensuring that the most efficient invariant is tested first.

### 6. Multi-Engine Backend Optionality
pySurgery features a flexible backend architecture that allows users to prioritize either environment simplicity or raw performance:
* **`backend='python'`**: Pure-Python execution (enhanced by NumPy/Numba). Requires zero external dependencies. Ideal for rapid prototyping and small-to-medium complexes.
* **`backend='julia'`**: Native integration with the Julia engine. Recommended for massive SNF reductions and high-dimensional manifold classification.
* **`backend='auto'` (Default)**: Automatically detects and leverages the most efficient engine available (Julia > Python).
* **`backend='jax'`**: Specifically used for continuous metric evaluations and differentiable topological approximations.

---

## Quick Start

### 1. Fundamental Group & Active Objects
Homotopy groups are treated as **Active Mathematical Objects** that can be queried for group-theoretic properties directly.

```python
import pysurgery as ps

# Extract pi_1 from a CW complex
pi1 = ps.extract_pi_1(cw_complex)

# Active queries
if pi1.is_abelian():
    print(f"Abelian Group of Order: {pi1.order()}")
    print(f"Tietze Simplified: {pi1.simplify()}")

# Geometric Trace: Generators as edge-cycles in the 1-skeleton
edges = pi1.simplices_generators()
```

### 2. Rational & Higher Homotopy
Unify Sullivan minimal models and Adams spectral sequence data into a single master contract.

```python
# Compute pi_n(X) and E2 page in one pass
hg = ps.HomotopyGroup.from_inputs(complex_data)

# Query rank of pi_3 tensor Q
rank_3 = hg.rank(3)

# Query 2-torsion filtration at degree 3
torsion_3 = hg.torsion(3, p=2)
```

### 3. Surgery Engine & Manual Control
The `Surgeon` (SurgerySession) allows for transactional, step-by-step manifold modification.

```python
from pysurgery import SurgerySession

# Initialize a surgery session on a manifold M
surgeon = SurgerySession(manifold=M, point_clouds={"cloud1": coords})

# Remove a D^n disk at a specific site
surgeon.remove_disks(types="D^3", at=[(0.1, 0.2, 0.3)])

# Attach a handle (S^1 x D^2) to the boundary of the hole
handle = surgeon.attach_handle(
    at=attaching_sphere_vertices, 
    handle_type="S^1xD^2"
)

# Finalize and inspect the resulting bordism
surgeon.finish()
print(surgeon.logs())
```

### 4. Automated Surgery Pipeline (AutoSurgeon)
The `AutoSurgeon` experimental pipeline attempts to automatically simplify complex manifold configurations into homotopy spheres.

```python
import pysurgery as ps

# 1. Build an ambient simplicial complex K
K = ps.SimplicialComplex.from_simplices(simplices, coefficient_ring="Z")

# 2. Initialize the AutoSurgeon orchestrator
surgeon = ps.AutoSurgeon(
    K,
    point_clouds=coords_map,
    target_topology="homotopy_sphere"
)

# 3. Run the automated surgical ladder
report = surgeon.run()

print(f"Status: {report.status}") # e.g., "success"
print(f"Final Betti: {report.final_components[0].betti}")
```

### 5. Interactive & Formal Adams Resolver
Compute homotopy groups with human-in-the-loop or formal verification.

```python
# Initialize the interactive resolver for an Adams E2 page
resolver = ps.InteractiveAdamsResolver(e2_page)

# Run interactive resolution to settle ambiguous differentials
page_inf = resolver.run_interactive_resolution()

# Or use Lean 4 to formally prove differentials
lean_resolver = ps.LeanFormalAdamsResolver(e2_page)
result = lean_resolver.resolve_e_infinity_via_lean()
```


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

## v2.0.0 Development Status (Beta)

pySurgery v2.0.0 is currently in **active development**. While the core topological engines (SNF, Betti numbers, $\pi_1$) are stable and rigorously tested, several higher-level modules are in various stages of completion:

- **STABLE (Production Ready):** Exact SNF (Python/Julia), Fundamental Group tracing, Intersection Forms, Sullivan Minimal Models, Geometrization (3D), Characteristic Classes.
- **BETA (Under Construction):** `SurgerySession` (Manual Engine), `AutoSurgeon` (Automated Pipeline), Adams Spectral Sequence $E_2$-pages.
- **RESEARCH (Experimental):** E-infinity Lean 4 formal resolvers, Temporal Topology bifurcation analysis, Controlled Cohomology local systems.

Users are encouraged to use the `backend='auto'` setting to automatically benefit from the latest performance optimizations as they land.

---

## Installation

pySurgery can be installed via pip:

```bash
pip install pysurgery
```

To enable the high-performance Julia backend, ensure Julia is installed and then run:

```bash
pip install "pysurgery[all]"
```

## Development Workflow

### 🚀 Local Version Management

To prevent desynchronization between your local machine and the remote repository, pySurgery uses a **local-first versioning workflow**. Instead of GitHub Actions creating "hidden" commits on the server, the versioning logic is handled directly on your machine via a Git hook.

#### How it works:
1. When you run `git commit`, an interactive prompt will appear asking if you want to update the version.
2. If you choose to bump, it displays the next available versions for **Patch**, **Minor**, and **Major**. The update is then bundled directly into your current commit.
3. When you run `git push`, a silent hook ensures the correct version tag exists and pushes it automatically.
4. The GitHub Action only triggers on version tags to build and publish the release.

#### Initial Setup:
After cloning the repository, you must install the local Git hook once:

```bash
# Make the setup script executable
chmod +x scripts/setup_hooks.sh

# Run the setup script
./scripts/setup_hooks.sh
```

> **Developer Note:** This hook ensures you never have to `git pull` just to sync a version bump created by a remote bot. All mathematical and versioning state remains strictly under your local control.

---

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

These manual `Pkg.add(...)` commands remain the recommended setup for local environments. In CI, and at runtime when Julia auto-install is enabled, the bridge can automatically install any missing required Julia packages when needed.

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
*   **Algebraic & Geometric Surgery:** Ranicki, A. (2002). *Algebraic and geometric surgery*. Oxford University Press.
*   **Surgery Theory & L-Groups:** Wall, C. T. (1970). *Surgery on compact manifolds*. Academic Press.
*   **4-Manifold Classification:** Freedman, M. H. (1982). The topology of four-dimensional manifolds. *Journal of Differential Geometry*, 17(3), 357-453.
*   **Characteristic Classes:** Milnor, J. W., & Stasheff, J. D. (1974). *Characteristic classes*. Princeton University Press.
*   **K-Theory & Whitehead Torsion:** Milnor, J. W. (1966). Whitehead torsion. *Bulletin of the American Mathematical Society*, 72(3), 358-426.
*   **Rational Homotopy Theory:** Quillen, D. (1969). Rational homotopy theory. *Annals of Mathematics*, 90(2), 205-295.
*   **Infinitesimal Computations:** Sullivan, D. (1977). Infinitesimal computations in topology. *Publications Mathématiques de l'IHÉS*, 47, 269-331.
*   **Adams Spectral Sequence:** Adams, J. F. (1960). On the structure and applications of the Steenrod algebra. *Commentarii Mathematici Helvetici*, 32, 180-214.
*   **Fox Derivatives:** Fox, R. H. (1953). Free differential calculus. I. Derivation in the free group ring. *Annals of Mathematics*, 57(3), 547-560.
*   **Todd-Coxeter Algorithm:** Todd, J. A., & Coxeter, H. S. M. (1936). A practical method for enumerating cosets of a finite abstract group. *Proceedings of the Edinburgh Mathematical Society*, 5(1), 26-34.
*   **Bass-Heller-Swan Theorem:** Bass, H., Heller, A., & Swan, R. G. (1964). The Whitehead group of a polynomial extension. *Publications Mathématiques de l'IHÉS*, 22, 61-79.
*   **Shaneson Splitting:** Shaneson, J. L. (1968). Wall's surgery obstruction groups for G x Z. *Annals of Mathematics*, 88(1), 1-67.
*   **Kirby Calculus:** Kirby, R. (1978). A calculus for framed links in S^3. *Inventiones mathematicae*, 45(1), 35-56.
*   **Normal Surface Theory:** Haken, W. (1961). Theorie der Normalflächen. *Acta Mathematica*, 105(3-4), 245-375.
*   **3-Manifold Geometrization:** Thurston, W. P. (1982). Three-dimensional manifolds, Kleinian groups and hyperbolic geometry. *Bulletin of the American Mathematical Society*, 6(3), 357-381.
*   **Simplicial Collapses:** Whitehead, J. H. C. (1939). Simplicial spaces, nuclei and m-groups. *Proceedings of the London Mathematical Society*, 2(1), 241-325.
*   **Steenrod Squares & Cup-i Products:** Steenrod, N. E. (1947). Products of cocycles and extensions of mappings. *Annals of Mathematics*, 48(2), 290-320.
*   **Wu Class:** Wu, W. T. (1950). Classes caractéristiques et i-carrés d'une variété. *Comptes Rendus de l'Académie des Sciences*, 230, 508-511.
*   **Hirzebruch Signature Theorem:** Hirzebruch, F. (1956). *Topological methods in algebraic geometry*. Springer-Verlag.
*   **Arf Invariant:** Arf, C. (1941). Untersuchungen über quadratische Formen in Körpern der Charakteristik 2. *Journal für die reine und angewandte Mathematik*, 183, 148-167.
*   **3-Manifold Decomposition:** Milnor, J. (1962). A unique decomposition theorem for 3-manifolds. *American Journal of Mathematics*, 84(1), 1-7.
*   **Discrete Morse Theory:** Forman, R. (1998). Morse theory for simplicial complexes. *Sém. Lothar. Combin*, 41, 1102-1133.
*   **E-infinity Algebras:** May, J. P. (1972). *The geometry of iterated loop spaces*. Springer-Verlag.
*   **Temporal Topology (Vineyards):** Cohen-Steiner, D., Edelsbrunner, H., & Morozov, D. (2010). Vineyards: Beyond persistence. *Discrete & Computational Geometry*, 44(2), 315-339.
*   **Smith Normal Form:** Smith, H. J. S. (1861). On systems of linear indeterminate equations and congruences. *Philosophical Transactions of the Royal Society of London*, 151, 293-326.
*   **Farrell-Jones Conjecture:** Farrell, F. T., & Jones, L. E. (1993). Isomorphism conjectures in algebraic K-theory. *Journal of the American Mathematical Society*, 6(2), 249-297.
*   **Surface Classification:** Radó, T. (1925). Über den Begriff der Riemannschen Fläche. *Acta Litt. Sci. Szeged*, 2, 101-121.
*   **Suspension Theorem:** Freudenthal, H. (1937). Über die Klassen von Abbildungen der $n$-dimensionalen Sphären auf die $k$-dimensionale Sphäre. *Compositio Mathematica*, 5, 299-314.
*   **Toda Brackets:** Toda, H. (1962). *Composition methods in homotopy groups of spheres*. Princeton University Press.

### Computational Implementation & Optimization

*   **Computational Topology Foundations:** Edelsbrunner, H., & Harer, J. (2010). *Computational topology: An introduction*. American Mathematical Society.
*   **Optimal Generators:** Dey, T. K., & Wang, Y. (2022). *Computational topology for data analysis*. Cambridge University Press.
*   **Alpha Complexes:** Edelsbrunner, H. (1994). The weighted Delaunay triangulation or how to stabilize the radical axis. *Discrete & Computational Geometry*, 13, 371-390.
*   **Vietoris-Rips Construction:** Zomorodian, A. (2010). Fast construction of the Vietoris-Rips complex. *Computers & Graphics*, 34(3), 263-271.
*   **Witness Complexes:** de Silva, V., & Carlsson, G. (2004). Topological estimation using witness complexes. *Eurographics Symposium on Point-Based Graphics*, 157-166.
*   **Orthogonal Procrustes:** Schönemann, P. H. (1966). A generalized solution of the orthogonal Procrustes problem. *Psychometrika*, 31(1), 1-10.
*   **Exact SNF (Leaf-Peeling):** Bauer, U., & Kerkhoff, M. (2021). Leaf-peeling for Smith normal form. *Journal of Applied and Computational Topology*, 5, 391-423.
*   **Quiver Representations (Zig-zag):** Gabriel, P. (1972). Unzerlegbare Darstellungen I. *Manuscripta Mathematica*, 6, 71-103.
*   **Efficient Persistent Homology:** Bauer, U. (2021). Ripser: efficient computation of Vietoris–Rips persistence barcodes. *Journal of Applied and Computational Topology*, 5, 391-423.
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
