# Discrete Surgery on Manifolds of Dimension < 3

**Finding cut sites and performing point-preserving surgery on simplicial complexes
built from data — the complete picture for dimensions 0, 1, and 2.**

Date: 2026-06-12
Scope: companion design document for pySurgery; generalizes the method prototyped in
`two_linked_tori_surgeon.ipynb` from a torus-specific geometric heuristic to a
provably terminating algorithm driven by the Classification of Surfaces.

---

## 1. Goal and constraints

We want to perform surgery on a simplicial complex `K` built from a discrete data set
(e.g. an alpha complex on a point cloud), removing topological features (handles,
crosscaps) one at a time, under one hard constraint:

> **Point bijection.** The vertex set of `K` is the data. Surgery may **remove or add
> simplices of dimension ≥ 1 only**. No 0-simplex is ever created or destroyed, so the
> data points before and after surgery are in bijection (in fact, identical).

This rules out the standard PL-topology tool of barycentric subdivision (it creates
vertices). The central result of this report is that in dimension ≤ 2 we do not need
it: the classification theorem plus two elementary lemmas guarantee that a valid cut
site **always exists inside the fixed 1-skeleton**, and that the surgery loop
**provably terminates** with a checkable certificate.

### 1.1 What "cut site" means

For a 1-cycle `γ` (the discrete analogue of an embedded circle), the cut site is the
**link of γ**, i.e. the boundary of its combinatorial regular neighborhood:

```
N(γ)  = union of closed stars of all simplices of γ      (the "tube" around γ)
Lk(γ) = { simplices of N(γ) containing no vertex of γ }  (the tube's boundary)
```

On a certified surface with `γ` *full* (chord-free), `Lk(γ)` is guaranteed to be one
or two disjoint simple cycles — the "parallel loops" found geometrically in the torus
notebook, now obtained with no coordinates, no PCA, and no assumptions about shape.

The cut itself is the rule already validated in the notebook (cell 27):

> remove every simplex containing at least one vertex of `γ` **and** at least one
> vertex of the chosen link component.

A 0-simplex has a single vertex, so it can never satisfy this condition: the point
bijection holds **by construction**, in every dimension, with a one-line proof.

---

## 2. Dimensions 0 and 1 (base cases)

**Dimension 0.** A compact 0-manifold is a finite set of points. There is no homology
above degree 0 and nothing to cut. These appear in practice as the "scar" components
produced by crosscap surgery (Section 4.4) and must simply be recognized and recorded.

**Dimension 1.** A compact connected 1-manifold is `S¹` or `[0,1]` (the classification
in dimension 1). An interval has `b₁ = 0`: nothing to cut. A circle has `b₁ = 1` and a
single essential class; surgery along it is the removal of any one edge:

```
ALGORITHM CutCircle(K)                          # K certified: closed 1-manifold
  1. e ← any 1-simplex of K
  2. remove e from K                            # K becomes an arc (path)
  3. assert vertex_set(K) unchanged
  4. assert b₁(K) == 0 and K connected
  return K
```

Every edge works because a circle minus an open edge is a path (connected). This is
the `n = 1` instance of the general rule: the cut site of a 0-sphere `{u, v} = ∂e`
inside `S¹` is its link, and removing the band (here: the edge itself) performs the
surgery. Trivial, but it is the base case the dimension-2 loop recurses into when
crosscap cuts produce residual circles.

---

## 3. Dimension 2: what the Classification Theorem buys

**Theorem (Classification of Surfaces).** Every compact connected surface without
boundary is homeomorphic to exactly one of:

* `S²` (the sphere),
* `T_g = T² # … # T²` (connected sum of `g ≥ 1` tori, orientable, `χ = 2 − 2g`),
* `N_k = RP² # … # RP²` (connected sum of `k ≥ 1` projective planes, non-orientable,
  `χ = 2 − k`).

The pair `(χ, orientability)` is a **complete invariant**, and both entries are
computable from `K` (`χ` by counting simplices; orientability via `H₂(K; ℤ) ≅ ℤ` vs
`0`). This turns cut-site finding from a heuristic into an algorithm, through four
guarantees:

**(G1) Existence.** A surface admits no essential (non-contractible, non-separating)
simple closed curve **iff** it is `S²` **iff** `χ = 2`. So "no cut site exists" is
never a search failure — it is the theorem certifying termination.

**(G2) Representability inside the fixed 1-skeleton.**

> **Lemma 1.** Every nonzero class in `H₁(K; ℤ/2)` is represented by a *simple* cycle
> (no repeated vertices) contained in the 1-skeleton of `K` as given.
>
> *Proof.* A mod-2 1-cycle is a set of edges in which every vertex has even degree.
> Such an edge set decomposes into edge-disjoint simple cycles (walk until a vertex
> repeats, split off the loop, induct). The classes of the pieces sum to the total
> class, so if the total is nonzero, some piece is a simple cycle with nonzero
> class. ∎

No subdivision, no geometry — the constraint that killed the textbook approach
(Section 1) costs nothing in dimension 2.

**(G3) Fullness for free.**

> **Lemma 2.** A minimum-length simple cycle `γ` with `[γ] ≠ 0` in `H₁(K; ℤ/2)` has no
> chord (no edge of `K` joining two non-consecutive vertices of `γ`). Hence `γ` is a
> full subcomplex of `K`.
>
> *Proof.* A chord `(u,v)` splits `γ` into two cycles `c₁, c₂` with
> `[c₁] + [c₂] = [γ] ≠ 0`, so some `cᵢ` has nonzero class; both are strictly shorter
> than `γ` (each uses a proper arc of `γ` plus one edge). This contradicts
> minimality. ∎

Fullness is exactly the hypothesis under which the simplicial neighborhood `N(γ)` is a
regular neighborhood and `Lk(γ)` is a closed 1-manifold (regular neighborhood theory,
Rourke–Sanderson) — i.e. the cut site is *guaranteed* to be clean: a disjoint union of
one or two circles, never the branched mess observed with naive star boundaries on
noisy complexes.

**(G4) The right curves come for free.**

> **Lemma 3.** A simple closed curve on a closed surface is non-separating **iff** its
> class in `H₁(K; ℤ/2)` is nonzero.
>
> *Proof sketch.* A separating curve bounds either side, hence is null-homologous.
> Conversely, if `γ` separates `K` into `A ∪ B` then `γ = ∂A` mod 2, so `[γ] = 0`. ∎

So selecting homology generators automatically selects exactly the curves whose
removal simplifies the surface; a cut is never wasted on a curve that merely splits
the complex without reducing genus.

Finally, the link of a full essential `γ` distinguishes the two families of the
classification:

* `π₀(Lk γ) = 2` components → `γ` is **two-sided** (annulus neighborhood): a *handle
  curve*. Always the case on orientable surfaces.
* `π₀(Lk γ) = 1` component → `γ` is **one-sided** (Möbius neighborhood; the link is the
  connected double cover of `γ`): a *crosscap curve*. Occurs only on `N_k`.

---

## 4. The algorithms

Throughout, `K` is a simplicial complex with vertex set = data points. All algorithms
touch only simplices of dimension ≥ 1 except where they merely *read* vertices.

### 4.0 Certification gate

```
ALGORITHM CertifyClosedSurface(K)
  Input : simplicial complex K (one connected component)
  Output: PASS, or FAIL with the offending simplices

  1. d ← dim K;  require d == 2
  2. for every edge e of K:
       c(e) ← number of triangles containing e
       if c(e) > 2:  FAIL("branching", e)          # fins
       if c(e) < 2:  FAIL("boundary/hole", e)      # not closed
  3. for every vertex v of K:
       L ← Lk(v)
       if L is not a single simple cycle:  FAIL("bad vertex link", v)
  4. return PASS
```

pySurgery: `is_homology_manifold()` (steps 2–3, with a fast incidence pre-check) and
`is_closed_manifold()` (closedness). On noisy alpha complexes, run `RepairCollar`
(Section 4.7) first; certification failures are *localized* and name the simplices to
repair.

### 4.1 Classification

```
ALGORITHM ClassifySurface(K)                      # K certified closed surface
  1. χ ← V − E + F                                 # euler_characteristic()
  2. orientable ← (H₂(K; ℤ) ≅ ℤ)                   # H₂ = 0 ⟺ non-orientable
  3. if χ == 2:               return  S²
     elif orientable:         return  T_g   with g = (2 − χ)/2
     else:                    return  N_k   with k =  2 − χ
```

By the classification theorem this is the **complete** homeomorphism type. It is used
twice per surgery step: to decide whether to halt, and to *predict* the type after the
cut so the post-cut recomputation can be checked against it (compute always, verify
against the prediction — never substitute the prediction for the computation).

### 4.2 Finding the cut curve

```
ALGORITHM EssentialSimpleCycle(K)                 # K certified, χ < 2
  Input : K with b₁(K; ℤ/2) > 0                    # guaranteed by (G1)
  Output: simple, chord-free cycle γ with [γ] ≠ 0

  1. z ← a generator of H₁(K; ℤ/2) of small support
       # compute_homology_basis(1, mode="optimal"): greedy short representative
  2. # Lemma 1 — extract a simple cycle from the mod-2 support:
     decompose support(z) into edge-disjoint simple cycles c₁, …, c_m
       (repeated vertex ⇒ split the walk there; iterate)
     γ ← any cᵢ with [cᵢ] ≠ 0                      # exists since Σ[cᵢ] = [z] ≠ 0
       # class test: chain_to_homology_class(1, cᵢ) ≠ 0
  3. # Lemma 2 — tighten until chord-free:
     loop:
       (a) if a triangle t ∈ K has 2 edges in γ:
             γ ← γ + ∂t                            # replace 2 edges by 1; class unchanged
       (b) elif an edge (u,v) ∈ K is a chord of γ:
             split γ at the chord into c₁, c₂      # [c₁]+[c₂] = [γ] ≠ 0
             γ ← the shorter of {c₁, c₂} with nonzero class
       (c) else: break                             # γ is full
     # terminates: |γ| strictly decreases in (a) and (b)
  4. return γ
```

Step 3 converges because the length of `γ` strictly decreases at every iteration and
is bounded below; by Lemma 2 its fixed points are exactly the chord-free cycles. The
result is a full subcomplex representing a nonzero class — by Lemma 3, a
non-separating curve; by (G3), a curve whose link is clean.

### 4.3 The cut site: link of a subcomplex

```
ALGORITHM SubcomplexLink(K, γ)
  Input : full simple cycle γ in certified surface K
  Output: cut site L = Lk(γ), and its sidedness

  1. N ← ∅
     for each simplex σ of γ (vertices and edges):
       N ← N ∪ closed_star(σ)                      # the simplicial neighborhood N(γ)
  2. L ← { τ ∈ N : τ ∩ V(γ) = ∅ }, closed under faces
  3. components ← connected_components(L)
  4. assert each component is a simple cycle       # every vertex has degree 2
       # guaranteed by fullness + certification; failure ⇒ bug or bad certification
  5. if |components| == 2: return (L, TWO_SIDED,  (L₁, L₂))
     if |components| == 1: return (L, ONE_SIDED,  (L₁,))
```

This is the coordinate-free replacement for the PCA/angular side-split of the torus
notebook: *the sides of γ are the connected components of its link.* pySurgery's
`link()` implements steps 1–2 for a single simplex; the subcomplex version is the
union over the simplices of `γ`.

### 4.4 The cut

```
ALGORITHM CutAlongCurve(K, γ, L₁)
  Input : K, cut curve γ, one chosen link component L₁
  Output: K′ with the band between γ and L₁ removed

  1. B ← { σ ∈ K : σ ∩ V(γ) ≠ ∅  and  σ ∩ V(L₁) ≠ ∅ }
  2. assert no σ ∈ B has dimension 0                # automatic: |σ| ≥ 2 by step 1
  3. K′ ← K \ B
  4. assert vertex_set(K′) == vertex_set(K)         # the point bijection
  return K′
```

Effects, by sidedness of `γ`:

* **Two-sided (handle cut).** `K′` is the surface cut along an annulus: a compact
  surface with two boundary circles, `γ` and `L₁`. `γ` stays attached to the `L₂`
  side. `χ` is unchanged (an annulus has `χ = 0`). Example: torus → cylinder.
* **One-sided (crosscap cut).** Removing the band between `γ` and its connected link
  detaches the open Möbius neighborhood of `γ`. `K′` has one boundary circle (`L₁`)
  **plus a residual component: the bare circle `γ`** — the core of the `RP²` summand
  just split off, surviving as a "scar" that carries its data points. This is the
  expected outcome, not damage; the verification step must whitelist it.

### 4.5 Optional capping (stay inside the closed classification)

Adding simplices among **existing** vertices also preserves the point bijection. By
capping each boundary circle with a fan, every iteration of the main loop faces a
*closed* surface, so `ClassifySurface` applies verbatim and `χ` becomes the
termination measure.

```
ALGORITHM CapBoundaryCircle(K, C = (v₀ v₁ … v_{m−1} v₀))
  Input : boundary circle C of K, m ≥ 3
  Output: K with a disk glued onto C, using no new vertices

  1. choose apex a ← v₀
  2. F ← { triangle (a, vᵢ, vᵢ₊₁) : 1 ≤ i ≤ m−2 }   # fan triangulation
     E ← { edge (a, vᵢ) : 2 ≤ i ≤ m−2 }             # the fan's chords
  3. if any simplex of E ∪ F already exists in K:
       retry with the next apex v₁, v₂, …           # avoid duplicated faces
       if all apices fail: return UNCAPPABLE        # fall back to cut-only mode
  4. K ← K ∪ E ∪ F
  5. assert CertifyClosedSurface on the capped component   # edge counts back to 2
  return K
```

`χ` bookkeeping: a handle cut leaves `χ` unchanged and caps two circles (`χ += 2`); a
crosscap cut removes a Möbius band (`χ += 1` net after capping its single circle).

If adding synthetic simplices is undesirable for the application, skip capping and run
the loop in **cut-only mode**: the classification of compact surfaces *with boundary*
(type + number of boundary circles, computable via `boundary()`) replaces the closed
classification, and `b₁` replaces `χ` as the progress measure.

### 4.6 The main loop

```
ALGORITHM SurfaceSurgeryLoop(K)
  Input : simplicial complex K (one component; run per component otherwise)
  Output: K reduced to S² (with capping) + list of recorded surgery steps
          (scars, removed bands, added caps), each with a verified certificate

  1. if CertifyClosedSurface(K) fails:
       K ← RepairCollar(K)                          # Section 4.7; or abort with diagnosis
  2. loop:
       T ← ClassifySurface(K)                       # (χ, orientability) — complete invariant
       if T == S²:  return DONE                     # (G1): no essential curve exists
       γ            ← EssentialSimpleCycle(K)       # (G2)+(G3): always exists, always full
       (L, side, ℂ) ← SubcomplexLink(K, γ)
       T_pred       ← PredictNextType(T, side)      # table below
       K            ← CutAlongCurve(K, γ, ℂ[0])
       record scar component γ if side == ONE_SIDED
       for each boundary circle C of K:             # skip in cut-only mode
            K ← CapBoundaryCircle(K, C)
       T_new ← ClassifySurface(K)                   # COMPUTE, then verify:
       assert T_new == T_pred                       # mismatch ⇒ bug or noise; halt loudly
```

Prediction table (`PredictNextType`):

| current type | side of γ   | after cut + cap                    |
|--------------|-------------|------------------------------------|
| `T_g`, g ≥ 1 | two-sided   | `T_{g−1}`                          |
| `N_k`        | one-sided   | `N_{k−1}`  (`N₀ := S²`)            |
| `N_k`, k ≥ 2 | two-sided   | recompute: `N_{k−2}` or `T_{(k−2)/2}` — accept either, the post-check disambiguates |

**Termination.** With capping, each iteration strictly increases `χ` (by 2 or 1) while
keeping `K` a closed connected surface; `χ ≤ 2` always, and `χ = 2` only for `S²`
(uniqueness in the classification!). Hence the loop halts after at most `2 − χ₀`
iterations — and you know the exact budget in advance: `g` cuts for `T_g`, `k` cuts
for `N_k`.

**Correctness.** Each step's postcondition is checked computationally (`T_new ==
T_pred`, vertex set unchanged, link components simple). The theorem supplies
predictions and existence; the pipeline never trusts them without recomputing.

### 4.7 Repair preprocessing (the only fallible stage)

Data-derived complexes (alpha complexes with noise) may fail certification near the
prospective cut. All failures are local and all repairs respect the point bijection:

```
ALGORITHM RepairCollar(K)
  1. fins    ← edges contained in ≥ 3 triangles    # branching (d−1)-faces
       resolve: delete the triangle(s) whose removal restores count == 2,
                preferring those whose deletion keeps vertex links connected
  2. badlinks ← vertices whose link is not a single cycle
       resolve: delete the chord edges / pinch triangles named in the link diagnosis
  3. guard: recompute b₀, b₁ of the affected collar before/after each deletion;
            revert any deletion that changes them unexpectedly
  4. rerun CertifyClosedSurface; if still failing, abort with the located defects
     (optionally: fall back to the geometric side-split heuristic of the torus
      notebook, which needs no certification but offers no guarantees)
```

This is the entire risk budget of the method in dimension ≤ 2: Sections 4.1–4.6 are
unconditional once certification passes.

---

## 5. Worked predictions (test cases for the implementation)

| start                | steps                                                                 |
|----------------------|-----------------------------------------------------------------------|
| Torus `T₁`           | handle cut (meridian) → cylinder → cap ×2 → `S²`. 1 cut, as predicted by `g = 1`. |
| Genus 2 `T₂`         | 2 handle cuts → `S²`.                                                  |
| `RP² = N₁`           | crosscap cut → disk + scar circle → cap → `S²` + scar. 1 cut (`k = 1`). |
| Klein bottle `N₂`    | either 2 crosscap cuts (`N₂ → N₁ → S²`), or 1 two-sided cut (`N₂ → S²` directly, χ: 0 → 2). Both paths valid; the post-check accepts whichever curve the generator search returns. |
| Circle `S¹` (dim 1)  | remove any edge → arc. `b₁: 1 → 0`.                                    |

The two linked tori notebook is the `T₁` row, where the geometric PCA construction
and `SubcomplexLink` must return the same two parallel loops — the regression test
tying the new machinery to the validated prototype.

---

## 6. Mapping to pySurgery

| report concept                  | existing API                                              | gap to fill |
|---------------------------------|-----------------------------------------------------------|-------------|
| certification                   | `is_homology_manifold()`, `is_closed_manifold()`          | restrict-to-collar variant (perf) |
| `χ`, orientability              | `euler_characteristic()`, `homology(2)`                   | — |
| mod-2 generator, short support  | `compute_homology_basis(1, mode="optimal")` (mod-2 engine)| Eulerian split + tighten loop (Alg. 4.2 steps 2–3) |
| class-preservation test         | `chain_to_homology_class(1, ·)`                           | — |
| `closed_star`, `link`           | `closed_star()`, `link()` (single simplex)                | subcomplex `link` (union over γ's simplices) |
| sides of γ                      | `connected_components()`                                  | — |
| cut                             | notebook cell-27 rule                                     | promote to `CutAlongCurve` |
| boundary handling / cut-only    | `boundary()`, `is_boundary_manifold`                      | — |
| post-cut verification           | `betti_numbers()`, `long_exact_sequence_of_pair()`        | `PredictNextType` table |

Suggested new surface-level API:

```
find_surgery_cut_site(K, generator, repair=True) -> CutSiteRegion
    # CutSiteRegion: tightened_cycle, neighborhood, link_components,
    #                two_sided, band_simplices, certificates

SurfaceSurgeon(K).run() -> SurgeryReport
    # the loop of Section 4.6; one step per essential class; full audit trail
```

(`CutSiteRegion` is deliberately distinct from `auto_surgery.CutSite`, which selects a
top simplex for unlinking — a different operation.)

---

## 7. Beyond dimension 2 (what breaks, for orientation)

The same skeleton — tighten, certify, take the link, cut the band — is stated
dimension-independently, but three of the four guarantees are special to surfaces:

1. **Representability fails:** an `H_k` class (`k ≥ 2`) need not be carried by an
   embedded submanifold at all (Steenrod); a pseudomanifold check on the generator's
   support becomes a genuine, possibly failing, gate.
2. **Minimal ⟹ full fails:** Lemma 2's chord argument is 1-dimensional; in higher
   dimension tightening may stall without subdivision.
3. **No complete invariant:** there is no classification to predict the post-surgery
   type or certify termination (in dimension ≥ 4 the homeomorphism problem is
   undecidable).

What survives everywhere: the link as cut site, sides = `π₀(link)`, the cell-27 cut
rule, and the 0-simplex argument for the point bijection. Dimension ≤ 2 is thus the
right foundation: every theoretical gap is closed, and every failure that remains is a
data-quality failure that the certification gate detects and localizes.
