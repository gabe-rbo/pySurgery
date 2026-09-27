# Exact invariants of complexes and embeddings

These modules were ported from **TabularTopology** (the topological-analysis library
built for the latent spaces of TabPFN) and adapted to pySurgery's types, backends and
conventions. They add invariants and properties that pySurgery did not compute before.
Three principles carry over unchanged:

* **Nothing is sampled.** Every simplex, every edge, every pair of points is checked.
  Restricting a computation is an explicit argument, never a random draw.
* **Exact where it can be.** Integer homology by unimodular elimination and exact Smith
  normal form; linking and winding numbers as intersection counts (integers, not
  integrals); knot invariants from one diagram. Every estimate says it is one.
* **A question with no answer gets a refusal** (an exception, or `defined=False` with the
  reason), never a number. Genericity of a ray, a cone apex or a projection is
  *certified*, not voted on.

Every accelerated function takes `backend='auto' | 'julia' | 'python'`, as in the rest of
pySurgery. The Julia kernels live in
`pysurgery/bridge/SurgeryBackend/src/TopologicalInvariants.jl`, are wrapped by
`JuliaBridge` methods of the same name, are precompiled and warmed up with the rest of
`SurgeryBackend`, and fall back to Python with a warning on failure. The tests run every
case on both backends and require identical answers.

## Map of the modules

| module | computes | Julia kernel |
|---|---|---|
| `topology.local_homology` | local homology at every simplex; sphere / acyclic / singular classification; the homology-manifold certificate (closed, or with a consistent boundary); pseudomanifold conditions; singular set; boundary complex | `classify_simplex_links_jl` (links in parallel threads) |
| `topology.fundamental_cycles` | coherent orientation of closed pseudomanifolds; integer and mod-2 fundamental cycles per strong component; a Z-basis of top homology; orientability | `coherent_orientation_jl` |
| `topology.finite_spaces` | finite T0 spaces: face posets, Stong cores, contractibility, weak points, quotients + T0 reflection, McCord certificates, order complexes; strong collapses of complexes | `stong_core_jl`, `strong_collapse_jl` |
| `topology.lower_star` | Robins-Wood-Sheppard gradients, signed Morse complex, gradient paths, cancellation classes, Z/2 lower-star persistence with simplices | `lower_star_gradient_jl` (lower stars in parallel), `z2_persistence_pairs_jl` |
| `knots.geometric_linking` | winding numbers of (m-1)-cycles and linking numbers of p- and q-cycles in R^m, exactly; the Gauss integral as an explicit estimate; definedness | `winding_numbers_jl`, `cone_intersection_count_jl` |
| `geometry.enclosure` | enclosure of points by a codimension-1 complex (degree vs parity); convex-hull membership; integer cycle generators | (through the linking kernels) |
| `knots.diagrams` | certified generic diagrams of polygons; diagrammatic linking numbers; writhe; Casson a2; Milnor mu(123) | `diagram_crossings_jl` |
| `knots.link_complement` | Wirtinger presentation as a `FundamentalGroup`; Alexander-duality check; exact Hom(G, S_n) counts; splitness and knottedness certificates | `count_homomorphisms_jl` |
| `geometry.holonomy` | discrete Levi-Civita transport of local-PCA frames; orientation double cover over every edge; holonomy around every fundamental loop | `edge_transport_data_jl` |
| `geometry.scale_window` | connectivity radius (EMST), Federer reach over every pair, the Niyogi-Smale-Weinberger scale window | `federer_reach_jl` |
| `algebra.sparse_elimination` | the exact Python engine: unit-pivot sparse elimination + dense SNF core, reduced homology of small complexes | - |

New methods on existing classes: `SimplicialComplex.local_homology`,
`.certify_homology_manifold`, `.pseudomanifold_report`, `.face_poset`,
`.strong_collapse`, `.is_strong_collapsible`, and
`FundamentalGroup.count_homomorphisms`. New exceptions in `core.exceptions`:
`NoFundamentalClassError`, `UndefinedInvariantError`, `NonGenericConfigurationError`.

## Local homology and the manifold certificate

For x in the open simplex sigma (dimension k),
`H_j(|K|, |K| - x) = H~_{j-k-1}(lk sigma)`, with `H~_{-1}(empty) = Z` (the link of a
maximal simplex). Relative to a dimension n a simplex is

* **sphere** - `H~_*(lk) = H~_*(S^{n-k-1})` and `dim lk = n-k-1` (an interior point);
* **acyclic** - `H~_*(lk) = 0`, `k < n`, and `lk` pure of dimension `n-k-1` (a boundary
  candidate);
* **other** - a singular point.

`certify_homology_manifold(K, n)` is the definition checked everywhere: closed when every
simplex is `sphere`; with boundary when there is no `other`, the acyclic simplices are
closed under faces (zero local homology alone cannot tell a boundary from the free end
of a dangling edge) and form a closed homology (n-1)-manifold. For n <= 3 this implies a
PL manifold (`implies_pl_manifold`); for n >= 4 it does not (a vertex link can be the
Poincare homology sphere), and the certificate says so.

This is strictly stronger than `SimplicialComplex.is_homology_manifold`, which reads
vertex links only: the test suite contains the cone on (S^2 wedge a disk), whose vertex
link has the homology of S^2 but whose edge (0, 1) has a link of two circles - the full
certificate names that edge.

## Fundamental cycles

For a closed pseudomanifold with nothing above dimension p, `H_p(K; Z) = Z_p(K)`, and a
p-cycle has constant absolute coefficient along the dual graph. Coherent signs
(`s_sigma [sigma:f] + s_tau [tau:f] = 0`) propagate by breadth-first search; a component is
orientable iff propagation closes up. Hence `H_p(K; Z) = Z^{#orientable components}` and
`H_p(K; F_2) = F_2^{#components}`. The functions refuse by name when K has simplices
above dimension p (the 2-simplices of a solid tetrahedron sum to a *boundary*), boundary
or branching faces, several components (`fundamental_cycle`), or a non-orientable
component over Z. Every returned cycle is re-verified, `d z = 0`, in exact arithmetic
independently of the backend that found the signs. `as_chain(K)` is the coefficient
vector over `K.n_simplices(p)` that `poincare_duality_verification` expects.

## Finite spaces and strong collapses

Convention: open sets are down-sets, `U(x) = {y <= x}`. A beat point has `U(x) - {x}` with
a maximum or `F(x) - {x}` with a minimum; removing beat points until none remain gives the
Stong core, and X is contractible iff its core is a point (Stong 1966) - an exact
decision. Weak points preserve the weak homotopy type. McCord's criterion
(`q^{-1}(U(y))` contractible for every y) certifies a weak equivalence; it is one-sided,
because Stong cores decide contractibility while McCord needs weak contractibility.

On the complex side, `strong_collapse(K)` deletes dominated vertices (every maximal
simplex through v contains some v' != v). By Barmak-Minian, K is strong collapsible iff
the face poset X(K) is contractible; the tests check the two independent algorithms agree
on random complexes where both outcomes occur. Strong collapsible implies collapsible
implies contractible, so this is a contractibility certificate for |K|.

## Lower-star Morse theory and persistence

`lower_star_gradient(K, g)` extends a vertex function to an acyclic matching whose
critical cells are the topology changes of the lower-star filtration (ties broken by
vertex label). The Morse complex (`morse_chain_complex()`) is a pySurgery `ChainComplex`
with the homology of K over Z, torsion included; the weak Morse inequalities hold over
every field and `sum (-1)^p m_p = chi(K)`. Pairs of critical cells are classified
`cancellable` (one gradient path: Forman's theorem applies), `chain_cancellable`
(|incidence| = 1, several paths: cancels algebraically) or `not_cancellable` (incidence 0
or |incidence| > 1 - which says nothing about homology on its own).

`lower_star_persistence(K, g)` returns the Z/2 persistence pairs with the simplices that
create and destroy each class. The number of essential classes is `dim H_p(K; Z/2)`,
*not* the rational Betti number when K has 2-torsion (RP^2 gives (1, 1, 1)). Checked
against gudhi with `homology_coeff_field=2` (gudhi's default field is Z/11).

## Linking and winding numbers

A chain is a list of `(simplex, coefficient)` pairs and the listed vertex order *is* the
orientation. `lk(A, B)` of a p-cycle and a q-cycle in R^m (p + q = m - 1) is the
intersection number of B with the cone on A from a certified generic apex (a Seifert
chain): one linear solve per pair of simplices, an integer. `lk(B, A) = (-1)^(pq+1) lk(A, B)`;
for curves in R^3 it agrees in sign with the Seifert-surface count, the crossing count
and the Gauss integral `(x - y).(dx x dy) / 4 pi`. The winding number is the signed count
of crossings of one certified generic ray. Linking outside p + q = m - 1, a point on the
cycle, or cycles that (nearly) meet are refusals.

`enclosure_report(points, K_A, coords, m)` decides whether points lie in a bounded
component of `R^m - |K_A|` from the winding numbers against a spanning set of
`H_{m-1}(K_A)` cycles (fundamental cycles for a closed pseudomanifold, the exact SNF
kernel otherwise), with the even-odd parity as an independent check whose disagreement
inside nested shells is expected and reported point by point.

## Knot diagrams and link complements

A projection direction is used only after checking every pair of segments: no crossing
at an endpoint, no collinear overlap, no triple point, and depth separation at every
crossing (otherwise the polygons intersect and the knot type is undefined). Crossing
sign: positive when `cross2d(t_over, t_under) > 0`. From one certified diagram:
`diagram_linking_number` (over-crossings of A over B, checked equal to B over A),
`writhe`, `casson_a2` (Polyak-Viro; the z^2 coefficient of the Conway polynomial: 1 on
the trefoil, -1 on the figure-eight, (p^2-1)(q^2-1)/24 on T(p, q)), and
`diagram_milnor_mu123` (defined only when all pairwise linking numbers vanish;
|mu| = 1 on the Borromean rings). `diagram_milnor_mu(curves, I)` computes Milnor's
mu-bar(I) for any multi-index by Milnor's algorithm (meridians of arcs substituted into
the Magnus expansion until it is exact to degree |I| - 1, read off the 0-framed
longitude), refusing unless every invariant obtained by deleting indices vanishes; e.g.
mu-bar(1122) = -beta, the Sato-Levine invariant, is +-1 on the Whitehead link.

The Wirtinger presentation (one generator per arc, one relator per crossing, one
redundant relator dropped) is returned as a `FundamentalGroup` and must pass the
Alexander-duality check (`H_1 = Z^k`) before any certificate is issued. Splitness and
triviality of finitely presented groups are undecidable, so certificates are
asymmetric: *split* from a verified separating plane or a Tietze reduction to a free
group of rank k with no relators; *non-split* when the exact counts `|Hom(G, S_n)|`
differ from the free-product count of every bipartition; *unknotted* when Tietze reaches
`<x | >`; *knotted* when `|Hom(G, S_n)| != n!`; otherwise *undetermined*. A budgeted count
that did not finish is flagged `exact=False` and never used for a certificate.

`FundamentalGroup.count_homomorphisms(n)` makes the same exact count available for every
pySurgery pi_1: a count above 1 proves the group nontrivial.

## Holonomy and scale

Transport along an edge is the polar factor of `F_j^T F_i` for local-PCA frames; the
holonomy around a loop is gauge-invariant up to conjugation, so its determinant and
rotation angles are invariants. `orientation_report` propagates signs over a spanning
forest and checks **every** other edge (the orientation double cover), so every cycle
of the graph is examined; an edge whose smallest singular value is below `min_cos` makes
the verdict *not certified*. Gauss-Bonnet on a latitude circle is the ground truth for
the angle. `scale_window` bounds the radius of a union-of-balls model from below by
connectivity (half the longest EMST edge) and from above by the NSW bound
`sqrt(3/5) * reach` and half the gap to another class; an empty window is a result.

## What was not ported, and why

* TabPFN access, layer Jacobians and folds (model-specific, not invariants of a complex).
* The dual active-set alpha complex and the Delaunay-Cech complex (constructions:
  pySurgery already builds alpha, Delaunay-Cech and Delaunay-Rips complexes).
* Intrinsic dimension and local PCA (already in pySurgery; the TwoNN censoring
  correction found in TabularTopology does not apply to pySurgery's TwoNN, which fits the
  full empirical CDF and discards nothing).
* The voxel model of a link complement (a construction; for polygons the Wirtinger
  presentation is the exact route to pi_1).

## Tests

`tests/test_local_homology.py`, `test_fundamental_cycles.py`, `test_finite_spaces.py`,
`test_lower_star.py`, `test_geometric_linking.py`, `test_enclosure.py`,
`test_knot_diagrams.py`, `test_link_complement.py`, `test_holonomy.py`, with ground truth
in `tests/exact_triangulations.py` (spheres, tori, the Klein bottle, RP^2, the Mobius band,
singular examples) and `tests/synthetic_curves.py` (knots and links as polygons, sampled
surfaces). Each test is parametrized over the available backends.
