# pysurgery

`pysurgery` is a Python library for Computational Algebraic Topology, built to apply the tools of **Surgery Theory** to discrete data.

Rather than just calculating Betti numbers for point clouds, `pysurgery` provides a framework to derive **Intersection Forms**, evaluate **Wall Group Surgery Obstructions**, perform algorithmic **Algebraic Surgery**, and determine if two datasets represent homeomorphic spaces.

From topological data analysis (TDA) pipelines to classification theorems (Freedman, Perelman, Smale), `pysurgery` aims to automate multi-dimensional topological analysis.

---

## ⚙️ Installation

Python 3.10+ is required.

```bash
git clone https://github.com/gabe-rbo/pysurgery.git
cd pysurgery
pip install .
```

For full integration support, install the optional dependencies:
```bash
pip install gudhi trimesh sympy scipy pydantic jax jaxlib torch_geometric
```

**Julia Integrations (Recommended for large-scale exact-Z calculations):**
If you want to use the Julia backend for sparse Smith Normal Form over PIDs, ensure the Julia executable is in your `PATH` and the `AbstractAlgebra` package is installed in your Julia environment.

---

## 📐 Mathematical Basis

`pysurgery` bridges discrete geometry to topological classification using several key mathematical concepts:

### 1. Homology & Cohomology via Smith Normal Form
When computing homology $H_n(X) = \ker(d_n) /  \mathop{\mathrm{im}}(d_{n+1})$, floating-point linear algebra misses **torsion** (topological twists). `pysurgery` implements the **Smith Normal Form (SNF)** over the integers $\mathbb{Z}$, reducing boundary matrices via the Extended Euclidean Algorithm to extract free ranks (Betti numbers) and specific torsion coefficients ($\mathbb{Z}_k$).
Cohomology $H^n(X)$ is computed via the **Universal Coefficient Theorem**.

### 2. The Alexander-Whitney Cup Product
To classify 4-manifolds, we evaluate how surfaces intersect by computing the **Cup Product** $\smile: H^2 \times H^2 \to H^4$ combinatorially. Given two 2-cocycles $\alpha, \beta$, the Alexander-Whitney formula is executed over every discrete 4-simplex:
$$(\alpha \smile \beta)([v_0, \dots, v_4]) = \alpha([v_0, v_1, v_2]) \cdot \beta([v_2, v_3, v_4])$$
Summing this over the fundamental class $[M]$ yields the Intersection Form matrix directly from raw data.

### 3. Intersection Forms & Freedman's Theorem
For 4-manifolds, `pysurgery` constructs the **Intersection Form** ($Q$), a symmetric unimodular integer matrix. It evaluates the matrix's Rank, Signature, and Parity (Type I/Type II). It then applies **Freedman’s Classification Theorem (1982)** to check if two spaces are homeomorphic based on these matrices and the Kirby-Siebenmann invariant.

### 4. Algebraic Surgery
If a manifold has a topological hole (a homology class $x$), `pysurgery` can \"cut it out\" and replace it to simplify the space. 
The algorithm verifies that the normal bundle is trivial by checking if $x$ is **isotropic** ($Q(x,x) = 0$). It then uses the Euclidean algorithm to find the dual class $y$, projects the lattice onto the orthogonal complement $\{x, y\}^\perp$, and uses **Hermite Normal Form (HNF)** lattice reduction to output the intersection form of the post-surgery manifold.

### 5. Characteristic Classes & Spin Structures
`pysurgery` extracts characteristic classes from intersection forms. Using **Wu's Formula**, it computes the 2nd Stiefel-Whitney class ($w_2 \in H^2(M; \mathbb{Z}_2)$) to check if a manifold admits a **Spin structure**. It also evaluates the **first Pontryagin class** ($p_1$) and uses the **Hirzebruch Signature Theorem** to verify if the topological intersection lattice aligns with the underlying geometric vector bundles.

### 6. Algebraic K-Theory & The Structure Set
In dimensions $n \ge 5$, classification utilizes the **Structure Set** $\mathcal{S}_{TOP}(M)$. `pysurgery` extracts the **Fundamental Group** ($\pi_1$) presentation via Edge-Path spanning trees from discrete data. It then applies **Algebraic K-Theory** to evaluate **Whitehead Torsion** ($Wh(\pi_1)$), identifying obstructions to the **s-Cobordism Theorem**.
Finally, it uses **Sullivan's Characteristic Variety Formula** to compute the rank of the set of **Normal Invariants** $[M, G/TOP]$, executing the High-Dimensional Surgery Exact Sequence to output a report on the exotic smooth/topological structures a manifold can admit.

### 7. Multi-Dimensional Homeomorphisms
`pysurgery` adapts its approach based on the dimension of the data:
- **2D**: Computes Orientability and Genus via $H_1$ and $H_2$.
- **3D**: Evaluates Homology Spheres and cites **Perelman's** resolution of the Poincaré Conjecture.
- **4D**: **Freedman's** Intersection Form analysis.
- **5D+**: Cites the **s-Cobordism Theorem**, evaluating Whitehead Torsion $Wh(\pi_1)$ and Wall's Surgery groups $L_n(\pi_1)$, using the Whitney Trick logic.

---

## 🚀 Computational Optimizations

Topology on discrete points can lead to combinatorial explosions (e.g., a 10,000-point cloud can generate 5,000,000 simplices). `pysurgery` uses several strategies to handle this:

1. **Sparse SVD Fallback**
   Exact integer math over $\mathbb{Z}$ requires dense matrices that consume significant RAM for large datasets. `pysurgery` uses a routing system: if a manifold has more than 1000 cells in a dimension, it offloads the math to **SciPy's ARPACK (Sparse SVD)**. It computes the intersection forms over $\mathbb{R}$, which reduces the memory footprint to $O(N)$ while keeping Freedman's signature invariants mathematically valid.

2. **Numba & NumPy Vectorization**
   The Alexander-Whitney Cup Product loop can iterate billions of times for large matrices. `pysurgery` minimizes Python's object overhead by flattening simplicial complexes into contiguous `np.int32` blocks and executing vectorized cache-friendly operations to reduce computation times.

3. **Julia AbstractAlgebra Bridge**
   If you need exact $\mathbb{Z}$-torsion for massive matrices without falling back to floats, the library transparently offloads execution to **Julia**. Using `AbstractAlgebra.jl`, it calculates the exact Smith Normal Form iteratively over the `SparseArray` COO layout, avoiding dense conversions.

---

## 📁 Library Architecture

- `pysurgery.core.complexes`: Abstractions for `CWComplex` and `ChainComplex`.
- `pysurgery.core.intersection_forms`: Lattice math, parity checks, and the `perform_algebraic_surgery` engine.
- `pysurgery.core.math_core`: The optimized integer Smith Normal Form algorithmic engine.
- `pysurgery.homeomorphism`: Multi-dimensional theorems and homeomorphism analyzers.
- `pysurgery.integrations`:
  - `gudhi_bridge`: Transitions raw TDA point clouds to Intersection Forms.
  - `trimesh_bridge`: Transitions 3D CAD meshes into topological frameworks.
  - `pytorch_geometric_bridge`: Topological simplification for Graph Neural Networks.
  - `jax_bridge`: Differentiable topology via hyperbolic tangent approximations.

---

## 📚 Examples and Tutorials

We provide a step-by-step interactive curriculum in the `examples/` directory using Jupyter Notebooks to cover concepts from basic Algebraic Topology to advanced Surgery Theory:

1. **`01_basic_homology_and_cohomology.ipynb`**: Building CW Complexes, SNF, and the Universal Coefficient Theorem.
2. **`02_intersection_forms.ipynb`**: 4D classification, Spin manifolds, and the $E_8$ manifold.
3. **`03_algebraic_surgery.ipynb`**: Using algebraic surgery to eliminate isotropic homology classes.
4. **`04_advanced_tda_and_surgery_theory.ipynb`**: The pipeline from Raw Discrete Data $\to$ GUDHI $\to$ Cup Product $\to$ Freedman's Theorem.
5. **`05_omni_dimensional_homeomorphisms.ipynb`**: Exploring how topological classification shifts behavior depending on dimension.
6. **`06_kirby_calculus_and_characteristic_classes.ipynb`**: Bridging algebra and 3D geometry via Kirby diagrams, handle slides, and evaluating Spin structures via Stiefel-Whitney classes ($w_2$).
7. **`07_fundamental_group_and_structure_set.ipynb`**: Extracting $\pi_1$ group presentations from discrete data and executing the High-Dimensional Surgery Exact Sequence to compute the Structure Set $\mathcal{S}_{TOP}(M)$.

---

## 🔮 Future Implementations

While `pysurgery` version 1.0 establishes a strong foundation, there are several areas for future expansion:

1. **Non-Abelian Twisted Homology**
   Currently, we extract the fundamental group $\pi_1(X)$, but computing the homology $H_n(M; \mathbb{Z}[\pi_1])$ for complex, infinite non-abelian groups is computationally intensive. Future iterations aim to expand the `JuliaBridge` to calculate exact twisted coefficients for asymmetric manifolds.

2. **The Algebraic $L$-Spectrum (Ranicki's Formulation)**
   Moving beyond classical Wall groups to implement Ranicki's full formulation of Algebraic Surgery using chain complexes of modules over rings with involution. This would allow `pysurgery` to perform surgery on spaces that aren't strictly manifolds (e.g., abstract Poincaré duality spaces).

3. **Machine Learning on the Structure Set**
   Integrating Graph Neural Networks (GNNs) via our `pytorch_geometric` and `jax` bridges. By training neural networks on the algebraic data of the Structure Set, the library could potentially predict optimal sequences of Kirby moves to minimize the geometric crossing number of complex 4-manifold diagrams.

---

## 🤝 Contributing

`pysurgery` is an active project spanning algebraic geometry and computational data science. We welcome pull requests, especially regarding expanding the `JuliaBridge` for twisted homology over non-abelian group rings.