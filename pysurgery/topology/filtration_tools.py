"""Filtration reports for point-cloud topology.

Every supported complex (Vietoris-Rips, CkNN, Alpha, Delaunay-Rips,
Delaunay-Cech, Witness) is a *monotone* filtration, so they all share one
engine: build the maximal complex once, tag each simplex with its appearance
value (see :mod:`pysurgery.topology.filtration_values`), and read the entire
Betti curve from a single Z2 persistence pass. Each method differs only in how
the maximal complex and its appearance values are produced -- that is the one
method ``_build_maximal_and_values`` each subclass overrides.

Use the concrete classes directly::

    RipsFiltrationReport(points)
    CknnFiltrationReport(points, k=8)
    AlphaFiltrationReport(points)
    DelaunayRipsFiltrationReport(points)
    DelaunayCechFiltrationReport(points)
    WitnessFiltrationReport(points, n_landmarks=50)

or the backward-compatible factory ``FiltrationReport(points, mode=...)``.
"""

import hashlib
import warnings
import numpy as np
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from pysurgery.geometry.point_cloud import PointCloud



def _format_padded_table(rows: List[List[str]]) -> str:
    """Format a list of rows into a markdown table with uniform column widths."""
    if not rows:
        return ""
    max_w = 0
    for row in rows:
        for cell in row:
            max_w = max(max_w, len(str(cell)))
    col_width = max_w + 2
    lines = []
    header = rows[0]
    lines.append("| " + " | ".join(str(cell).ljust(col_width) for cell in header) + " |")
    sep_cells = ["-" * (col_width + 2) for _ in header]
    lines.append("|" + "|".join(sep_cells) + "|")
    for row in rows[1:]:
        lines.append("| " + " | ".join(str(cell).ljust(col_width) for cell in row) + " |")
    return "\n".join(lines)


class _BaseFiltrationReport:
    """Shared engine for point-cloud filtration reports.

    Overview:
        Every supported complex is a monotone filtration, so this base owns the
        whole pipeline: it builds the maximal complex once, computes the full Betti
        curve from a single Z2 persistent-homology pass, and (optionally) runs
        per-threshold homology-manifold and connected-component analysis, before
        rendering a Markdown report or Plotly figure. Subclasses supply only the
        method-specific geometry.

    Key Concepts:
        - **Monotone filtration**: each simplex enters at one appearance value and
          never leaves, so persistence is a single boundary-matrix reduction.
        - **Build-once / slice-many**: sub-complexes are filtered from the maximal
          complex by appearance value rather than rebuilt at each threshold.
        - **Mod-2 Betti**: numbers come from Z2 persistence (equal to the integer
          ranks unless the complex carries 2-torsion).

    Subclass Contract:
        Set ``param_label`` / ``method_name`` and implement
        ``_build_maximal_and_values()`` returning ``(maximal_complex, {simplex: value})``.

    Attributes:
        epsilons (List[float]): The thresholds reported.
        results (List[dict]): Per-threshold invariants (Betti, manifold, components).
        barcode (List[tuple]): Persistence intervals ``(dim, birth, death)``.
        stability_data (List[tuple]): Topological regimes ranked by persistence range.
    """

    #: Parameter axis label in reports ("Eps", "Delta", "Alpha", "Radius").
    param_label: str = "Eps"
    #: Human-readable method name shown in report headers.
    method_name: str = "Filtration"
    #: Extra kwargs passed when constructing the tiny warmup instance.
    _warmup_kwargs: dict = {}

    # Above this simplex count the per-threshold homology-manifold check is
    # auto-skipped (per-vertex link homology is the slowest step).
    _MANIFOLD_MAX_SIMPLICES = 50_000

    def __init__(
        self,
        points: Union[np.ndarray, "PointCloud"],
        epsilons: Optional[List[float]] = None,
        max_dimension: int = 2,
        coefficient_ring: str = "Z",
        backend: str = "auto",
        track_connected_components: bool = False,
        *,
        n_samples: Optional[int] = None,
        eps_max: Optional[float] = None,
        analyze_manifolds: bool = False,
        compute_torsion: bool = False,
        manifold_analysis: Optional[bool] = None,
        **kwargs,
    ):
        """Build and compute the filtration report.

        Args:
            points: (N, D) array of point coordinates.
            epsilons: Explicit thresholds. If None, a bounded grid is derived from
                the distribution of appearance values.
            max_dimension: Maximum simplex dimension to build.
            coefficient_ring: Coefficient ring (manifold/component path).
            backend: 'auto', 'julia', or 'python'. Controls both the geometry
                builders and the persistence reducer; 'python' forces the
                pure-Python column reduction (the slow reference path).
            track_connected_components: Track per-component evolution (costly).
            n_samples: If given, select this many evenly-spaced thresholds from
                the full set of distinct appearance values. If None, all distinct
                appearance values are used.
            eps_max: Cap for the parameter range. Method-dependent meaning; if None
                each method picks a sensible default.
            analyze_manifolds: Run the per-threshold homology-manifold check
                (auto-skipped above _MANIFOLD_MAX_SIMPLICES simplices).
            compute_torsion: If True, additionally compute exact *integer*
                homology (free rank + torsion coefficients) of the sub-complex at
                every reported threshold and surface the torsion in the report.
                Default False, in which case Betti numbers come from the fast Z2
                persistence barcode (mod-2 ranks). Torsion uses Smith-normal-form
                per threshold and is markedly heavier — prefer it with a coarse
                grid (``n_samples``) or a modest ``eps_max``.
            manifold_analysis: Alias/override for analyze_manifolds. If provided,
                overrides analyze_manifolds.
            **kwargs: Method-specific options (e.g. ``k``, ``n_landmarks``).
        """
        self.points = np.asarray(points, dtype=np.float64)
        self.max_dimension = max_dimension
        self.coefficient_ring = coefficient_ring
        self.backend = backend
        self.track_connected_components = track_connected_components
        self.n_samples = int(n_samples) if n_samples is not None else None
        self.eps_max = eps_max
        if manifold_analysis is not None:
            analyze_manifolds = manifold_analysis
        self.analyze_manifolds = bool(analyze_manifolds)
        self.compute_torsion = bool(compute_torsion)
        self.kwargs = kwargs

        self._user_epsilons = list(epsilons) if epsilons is not None else None
        self.dynamic_mode = epsilons is None
        self.epsilons: List[float] = []
        self.results: List[dict] = []
        self.all_betti_dims = set()
        self.max_comp_rows = 0
        self.barcode: List = []
        self._precomputed_manifolds = None
        self._compute()

    # ------------------------------------------------------------------
    # Method hook (overridden per complex type)
    # ------------------------------------------------------------------
    def _build_maximal_and_values(self):
        """Build the maximal complex and its per-simplex appearance values.

        Returns:
            Tuple ``(maximal_complex, {simplex: appearance_value})``.

        Raises:
            NotImplementedError: Always in the base class; subclasses override.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _subsample_indices(n: int, cap: int, seed: int = 0) -> np.ndarray:
        """Indices of a deterministic subsample of size at most ``cap``.

        Args:
            n: Population size.
            cap: Maximum subsample size.
            seed: RNG seed for reproducible selection.

        Returns:
            An int array of indices (``arange(n)`` when ``n <= cap``).
        """
        if n <= cap:
            return np.arange(n)
        return np.random.default_rng(seed).choice(n, size=cap, replace=False)

    @classmethod
    def _estimate_eps_max(cls, points: Union[np.ndarray, "PointCloud"], factor: float = 1.2,
                          k: Optional[int] = None, quantile: float = 0.95) -> float:
        """Local k-NN scale: high-quantile distance to the k-th nearest neighbour.

        Keeps unbounded complexes (Rips) sparse -- every point links to ~k
        neighbours -- instead of using the connectivity scale, which over-connects
        dense regions into a complex whose construction dominates runtime.

        Args:
            points: (N, D) array of point coordinates.
            factor: Multiplier applied to the estimated scale.
            k: Neighbour rank to measure; defaults to ``min(N - 1, 12)``.
            quantile: Quantile of the k-th-neighbour distances to use.

        Returns:
            The estimated maximum filtration parameter (a distance).
        """
        from scipy.spatial import cKDTree
        pts = np.asarray(points, dtype=np.float64)
        n = len(pts)
        if n <= 1:
            return 1.0
        if k is None:
            k = min(n - 1, 12)
        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=k + 1)
        kth = np.atleast_2d(dists)[:, -1]
        emax = float(np.quantile(kth, quantile))
        if not np.isfinite(emax) or emax <= 0.0:
            emax = float(np.max(kth)) or 1.0
        return emax * factor

    def _default_grid_from_values(self, filt: Dict) -> List[float]:
        """All distinct appearance values as the threshold grid (from a value map)."""
        return self._grid_from_values(filt.values() if filt else [])

    def _grid_from_values(self, values) -> List[float]:
        """All distinct appearance values as the threshold grid.

        Every unique simplex-entry value becomes a threshold so no topological
        event is skipped. ``eps_max``, if set, caps the range; ``n_samples``, if
        set, evenly subsamples it.

        Args:
            values: Any iterable of appearance values -- a ``{simplex: value}``
                map's values, or the distinct grid the fused Julia builder already
                returns.

        Returns:
            A sorted list of every distinct appearance value (capped at
            ``eps_max`` if set), always including ``0.0``.
        """
        if isinstance(values, (list, np.ndarray)):
            vals = np.asarray(values, dtype=np.float64)
        else:
            vals = np.fromiter(values, dtype=np.float64)
        vals = np.unique(vals)

        if vals.size == 0:
            vals = np.array([0.0])
        if self.eps_max is not None:
            vals = vals[vals <= float(self.eps_max)]
        if vals.size == 0:
            return [0.0]
        if self.n_samples is not None and vals.size > self.n_samples:
            indices = np.unique(np.round(np.linspace(0, vals.size - 1, self.n_samples)).astype(int))
            vals = vals[indices]
        if not np.any(vals == 0.0):
            vals = np.concatenate(([0.0], vals))
            vals = np.unique(vals)
        return vals.tolist()

    @staticmethod
    def _complex_from_values(SC, values: Dict, coords, ring: str):
        """Build a SimplicialComplex from a ``{simplex: value}`` map.

        The value map is assumed closed under faces (as all our appearance-value
        maps are), so its keys already form a valid complex.

        Args:
            SC: The SimplicialComplex class.
            values: Mapping of simplex to appearance value.
            coords: (N, D) point coordinates to attach.
            ring: Coefficient ring label.

        Returns:
            A SimplicialComplex over the simplices in ``values``.
        """
        table: Dict[int, list] = {}
        for s in values:
            table.setdefault(len(s) - 1, []).append(s)
        table = {d: sorted(v) for d, v in table.items()}
        sc = SC(simplices=table, coefficient_ring=ring)
        sc._coordinates = np.asarray(coords, dtype=np.float64)
        return sc

    def _compute_barcode(self, simplices_table: dict, filt: dict) -> list:
        """Z2 persistence barcode, via the Julia twist/clearing reducer when available.

        Dispatches the one expensive step (the boundary-matrix reduction) to the
        compiled Julia kernel ``compute_filtration_persistence`` -- an exact,
        filtration-ordered reduction with Chen-Kerber clearing that returns the
        identical barcode to :meth:`_z2_persistence_barcode` but orders of
        magnitude faster. Falls back to the pure-Python reference reducer when
        ``backend == 'python'`` or Julia is unavailable / errors.

        Args:
            simplices_table: dim -> list of simplices (sorted vertex tuples).
            filt: Mapping of simplex to appearance value.

        Returns:
            A list of ``(dim, birth, death)`` intervals (``death == inf`` for
            essential classes), identical in content to the reference reducer.
        """
        if self.backend != "python":
            try:
                from pysurgery.bridge.julia_bridge import julia_engine
                if julia_engine.available:
                    flat, ptr, vals = self._flatten_for_persistence(simplices_table, filt)
                    return julia_engine.compute_filtration_persistence(flat, ptr, vals)
            except Exception as exc:  # pragma: no cover - exercised only when Julia errors
                warnings.warn(
                    f"Julia persistence reducer failed ({exc!r}); falling back to the "
                    "pure-Python reduction.",
                    stacklevel=2,
                )
        return self._z2_persistence_barcode(simplices_table, filt)

    @staticmethod
    def _flatten_for_persistence(simplices_table: dict, filt: dict):
        """Flatten ``{dim: [simplices]}`` + values into the Julia reducer's arrays.

        Args:
            simplices_table: dim -> list of simplices (sorted vertex tuples).
            filt: Mapping of simplex to appearance value.

        Returns:
            Tuple ``(simplices_flat, simplex_ptr, vals)`` of numpy arrays: all
            vertices concatenated, the per-simplex offsets (length ``M + 1``), and
            each simplex's appearance value. Order is arbitrary -- the Julia kernel
            re-derives the filtration order itself.
        """
        flat: List[int] = []
        ptr: List[int] = [0]
        vals: List[float] = []
        for d in sorted(simplices_table.keys()):
            for s in simplices_table[d]:
                flat.extend(s)
                ptr.append(len(flat))
                vals.append(float(filt.get(s, 0.0)))
        return (np.asarray(flat, dtype=np.int64),
                np.asarray(ptr, dtype=np.int64),
                np.asarray(vals, dtype=np.float64))

    @staticmethod
    def _z2_persistence_barcode(simplices_table: dict, filt: dict) -> list:
        """Persistent-homology barcode over Z2 by standard column reduction.

        Pure-Python reference implementation: kept as the correctness oracle and
        as the fallback when Julia is unavailable. One reduction pass yields every
        class's birth/death, so Betti numbers at all thresholds come from a single
        computation (the Ripser/GUDHI approach); reported Betti numbers are
        therefore mod-2 (equal to integer ranks unless the complex carries
        2-torsion). Returns ``(dim, birth, death)`` intervals with ``death == inf``
        for essential classes.
        """
        import math
        ordered = []
        for d, simps in simplices_table.items():
            for s in simps:
                ordered.append((filt.get(s, 0.0), len(s) - 1, s))
        ordered.sort(key=lambda x: (x[0], x[1], x[2]))
        index = {s: i for i, (_, _, s) in enumerate(ordered)}
        order_filt = [f for f, _, _ in ordered]
        order_dim = [dm for _, dm, _ in ordered]

        columns = []
        for _, dm, s in ordered:
            if dm == 0:
                columns.append(set())
            else:
                columns.append({index[s[:i] + s[i + 1:]] for i in range(len(s))})

        low_to_col: dict = {}
        paired_birth = set()
        bars = []
        for j in range(len(ordered)):
            col = columns[j]
            while col:
                low = max(col)
                owner = low_to_col.get(low)
                if owner is None:
                    break
                col ^= columns[owner]
            columns[j] = col
            if col:
                low = max(col)
                low_to_col[low] = j
                if order_filt[j] > order_filt[low]:
                    bars.append((order_dim[low], order_filt[low], order_filt[j]))
                paired_birth.add(low)

        for j in range(len(ordered)):
            if not columns[j] and j not in paired_birth:
                bars.append((order_dim[j], order_filt[j], math.inf))
        return bars

    @staticmethod
    def _betti_from_barcode(barcode: list, eps: float, tol: float = 1e-9) -> dict:
        """Betti numbers at a single threshold from a persistence barcode."""
        b: dict = {}
        e = eps + abs(eps) * tol + 1e-12
        for dim, birth, death in barcode:
            if birth <= e < death:
                b[dim] = b.get(dim, 0) + 1
        return b

    @staticmethod
    def _all_betti_from_barcode(barcode: list, epsilons: list) -> list:
        """Betti numbers at all thresholds via per-dimension prefix counts.

        A bar ``(birth, death)`` is alive at ``e`` iff ``birth <= e < death``, so
        ``betti_d(e) = #{birth_d <= e} - #{death_d <= e}``. Each side is a single
        ``searchsorted`` over the sorted births/deaths of dimension ``d`` -- O((bars
        + thresholds) log bars) time and *linear* memory. This replaces the former
        dense ``(bars x thresholds)`` boolean matrix, which reached tens of GiB (and
        swapped for hours) on large complexes. Returns one dict per epsilon.
        """
        import math
        if not barcode or not epsilons:
            return [{} for _ in epsilons]
        eps_arr = np.asarray(epsilons, dtype=np.float64)
        e_arr = eps_arr + np.abs(eps_arr) * 1e-9 + 1e-12
        births = np.array([b for _, b, _ in barcode], dtype=np.float64)
        deaths = np.array([d if d != math.inf else np.inf for _, _, d in barcode], dtype=np.float64)
        dims_arr = np.array([d for d, _, _ in barcode], dtype=np.int64)

        result: list = [dict() for _ in epsilons]
        for d in np.unique(dims_arr):
            mask = dims_arr == d
            born = np.searchsorted(np.sort(births[mask]), e_arr, side="right")
            died = np.searchsorted(np.sort(deaths[mask]), e_arr, side="right")
            betti = born - died
            di = int(d)
            for j in np.nonzero(betti)[0]:
                result[int(j)][di] = int(betti[j])
        return result

    @staticmethod
    def _slice_cutoff(eps: float) -> float:
        """Inclusive appearance-value cutoff for the slice at ``eps``.

        Simplices with ``filt[s] <= _slice_cutoff(eps)`` belong to the slice; the
        small tolerance absorbs float round-off in the appearance values. Shared
        by :meth:`_slice_complex` and :meth:`_incremental_slices` so the two
        cannot drift apart.
        """
        return eps + abs(eps) * 1e-9 + 1e-12

    def _slice_complex(self, SC, max_sc, filt, eps, coords):
        """Sub-complex of ``max_sc`` containing simplices appearing by ``eps``.

        A cheap dictionary filter on appearance value; the result is closed under
        faces because appearance values are monotone, so it equals the complex the
        method would build directly at ``eps``. This is the per-threshold
        reference for a slice; :meth:`_incremental_slices` reproduces the same
        slices across the grid without rescanning.

        Args:
            SC: The SimplicialComplex class.
            max_sc: The maximal complex.
            filt: Mapping of simplex to appearance value.
            eps: The filtration threshold.
            coords: Point coordinates to attach to the sub-complex.

        Returns:
            The sub-complex at ``eps`` as a SimplicialComplex.
        """
        cutoff = self._slice_cutoff(eps)
        table = {}
        for d in sorted(max_sc._simplices_table.keys()):
            kept = [s for s in max_sc.n_simplices(d) if filt.get(s, 0.0) <= cutoff]
            if kept:
                table[d] = kept
        sub = SC(simplices=table, coefficient_ring=self.coefficient_ring)
        sub._coordinates = coords
        if self.track_connected_components:
            sub._generate_point_cloud_mappings(coords)
        return sub

    @staticmethod
    def _component_content_key(table) -> str:
        """Content-complete key for a component's intrinsic topology.

        A filtration is a merge tree, so most components are byte-identical to
        their predecessor at the previous threshold. Their topological
        invariants (manifold status, dimension, closedness, integer homology)
        depend only on the global simplex set, so we key a cross-threshold cache
        by a hash of that set. Components keep global vertex indices, so an
        unchanged component hashes identically across thresholds and distinct
        components never collide. We hash the *content* (not the f-vector
        signature, which would collide for different complexes with the same
        simplex counts), and use a fixed-size digest so the cache stays bounded.

        Args:
            table: A ``{dim: [simplices]}`` mapping (e.g. one entry of
                :meth:`SimplicialComplex.component_simplex_tables`).
        """
        h = hashlib.blake2b(digest_size=16)
        for d in sorted(table.keys()):
            h.update(b"#%d" % d)
            for s in sorted(table[d]):
                h.update(b"|")
                h.update(",".join(map(str, s)).encode("ascii"))
        return h.hexdigest()

    def _build_component_complex(self, SC, table, coords):
        """Materialize a single component subcomplex from its simplex table.

        Used on a cache miss to build only the components whose heavy invariant
        (manifold status, integer homology) is not already memoized. The table
        is a connected-component slice of a face-closed complex, so it is closed
        under faces and is constructed directly — without re-deriving faces —
        mirroring :meth:`_slice_complex`. Global coordinates are attached so the
        manifold check matches the eager ``explode()``-built component.
        """
        sub = SC(simplices={d: list(ss) for d, ss in table.items()},
                 coefficient_ring=self.coefficient_ring)
        sub._coordinates = coords
        return sub

    @staticmethod
    def _aggregate_component_torsion(comp_homs) -> dict:
        """Aggregate per-component homology into the whole-complex torsion dict.

        Reproduces :meth:`SimplicialComplex.homology`'s torsion contract exactly:
        for a single component the native (Smith-normal-form) torsion order is
        preserved, matching the ``_homology_direct`` path; for several components
        the direct sum H_n(⊔Xᵢ)=⊕H_n(Xᵢ) is taken and the pooled torsion is
        sorted, matching ``_homology_summed``. Only degrees with torsion appear.

        Args:
            comp_homs: One ``{dim: (rank, torsion)}`` dict per component.
        """
        if not comp_homs:
            return {}
        if len(comp_homs) == 1:
            return {int(d): [int(t) for t in tors]
                    for d, (_r, tors) in comp_homs[0].items() if tors}
        agg: Dict[int, list] = {}
        for chz in comp_homs:
            for d, (_r, tors) in chz.items():
                if tors:
                    agg.setdefault(int(d), []).extend(int(t) for t in tors)
        return {d: sorted(tl) for d, tl in agg.items()}

    def _whole_complex_from_table(self, SC, full_table, coords):
        """Build the whole sub-complex at a threshold from an incremental table.

        The incremental engine (:meth:`_incremental_slices`) accumulates the
        slice's simplices in appearance order; this wraps a *copy* of that table
        in a SimplicialComplex for the whole-complex manifold check, mirroring
        :meth:`_slice_complex`'s object setup (coordinates always; point-cloud
        mappings only when components are tracked). The table is copied because
        the engine keeps mutating its lists as eps grows.
        """
        sub = SC(simplices={d: list(ss) for d, ss in full_table.items()},
                 coefficient_ring=self.coefficient_ring)
        sub._coordinates = coords
        if self.track_connected_components:
            sub._generate_point_cloud_mappings(coords)
        return sub

    def _incremental_slices(self, max_sc, filt, need_components, need_whole):
        """Yield per-threshold slice state, built incrementally across the grid.

        A filtration is monotone — simplices only ever *enter* as eps grows and
        connected components only ever *merge* — so rather than rescanning the
        maximal complex and re-running a DFS partition at every threshold, this
        walks the epsilon grid once:

        * simplices are sorted a single time by ``(appearance value, dimension)``
          (so every face precedes its cofaces) and consumed by a moving pointer;
        * connected components are maintained with a weighted union-find keyed on
          the 1-skeleton, with each root carrying its vertex set and per-dimension
          simplex bucket — a component that does not merge is never re-extracted.

        Yields, per threshold (in grid order), ``(comp_tables, full_table, cdim)``:

        * ``comp_tables`` — when ``need_components`` — the descending-size-ordered
          list of ``(vertex_frozenset, {dim: simplices})`` components, identical
          in content and order to
          :meth:`SimplicialComplex.component_simplex_tables` on the slice (the
          ``(-size, min-vertex)`` order reproduces the DFS partition's ordering);
          else ``None``;
        * ``full_table`` — when ``need_whole`` — the slice's ``{dim: simplices}``
          table (for the whole-complex manifold check); else ``None``;
        * ``cdim`` — the slice's top dimension (matches ``sc.dimension``).

        Orderings within a dimension may differ from a fresh rebuild, but every
        reported invariant (homology, manifold status, content hash) is
        order-independent, so the output is unchanged.
        """
        items = [(filt.get(s, 0.0), d, s)
                 for d, simps in max_sc._simplices_table.items() for s in simps]
        # (value, dim) sort guarantees faces precede cofaces: a face's value is
        # <= its coface's, and on ties the lower dimension comes first.
        items.sort(key=lambda it: (it[0], it[1]))
        n = len(items)

        parent: Dict[int, int] = {}
        csize: Dict[int, int] = {}                  # union weight (simplex count)
        cverts: Dict[int, set] = {}                 # root -> vertex set
        cbucket: Dict[int, Dict[int, list]] = {}    # root -> {dim: [simplices]}
        roots: set = set()
        full_table: Dict[int, list] = {}
        cdim = -1

        def find(x: int) -> int:
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:          # path compression
                parent[x], x = root, parent[x]
            return root

        def union(a: int, b: int) -> int:
            ra, rb = find(a), find(b)
            if ra == rb:
                return ra
            if csize[ra] < csize[rb]:         # merge smaller bucket into larger
                ra, rb = rb, ra
            parent[rb] = ra
            csize[ra] += csize[rb]
            cverts[ra] |= cverts[rb]
            dst = cbucket[ra]
            for dim, lst in cbucket[rb].items():
                if dim in dst:
                    dst[dim].extend(lst)
                else:
                    dst[dim] = lst
            del cverts[rb], cbucket[rb], csize[rb]
            roots.discard(rb)
            return ra

        ptr = 0
        for eps in self.epsilons:
            cutoff = self._slice_cutoff(eps)
            while ptr < n and items[ptr][0] <= cutoff:
                _val, d, s = items[ptr]
                ptr += 1
                if d > cdim:
                    cdim = d
                if need_whole:
                    full_table.setdefault(d, []).append(s)
                if not need_components:
                    continue
                if d == 0:
                    v = s[0]
                    parent[v] = v
                    csize[v] = 1
                    cverts[v] = {v}
                    cbucket[v] = {0: [s]}
                    roots.add(v)
                else:
                    # Connectivity is the 1-skeleton's, so only edges union;
                    # higher simplices attach to their (already-unified) root.
                    r = union(s[0], s[1]) if d == 1 else find(s[0])
                    csize[r] += 1
                    cbucket[r].setdefault(d, []).append(s)

            comp_tables = None
            if need_components:
                comp_tables = [(frozenset(cverts[r]), cbucket[r]) for r in roots]
                comp_tables.sort(key=lambda vt: (-len(vt[0]), min(vt[0])))
            yield comp_tables, (full_table if need_whole else None), cdim

    # ------------------------------------------------------------------
    # Engine
    # ------------------------------------------------------------------
    def _assemble(self):
        """Build the maximal complex, reduce its barcode, and gather grid/dim stats.

        Returns ``(max_sc, filt, barcode, dim_first_appear, total, grid_values)``:
        the maximal complex and its ``{simplex: value}`` map, the Z2 barcode, the
        per-dimension first (minimum) appearance value, the total simplex count, and
        an iterable of appearance values from which the threshold grid is derived.

        This base implementation builds the explicit complex and reduces it (the
        staged path). :class:`RipsFiltrationReport` overrides it to fuse the build
        and reduction inside Julia, keeping the complex implicit for large clouds
        (then ``max_sc`` / ``filt`` come back ``None``).
        """
        max_sc, filt = self._build_maximal_and_values()
        barcode = self._compute_barcode(max_sc._simplices_table, filt) \
            if max_sc._simplices_table else []
        dim_first_appear: dict = {}
        for d, simps in max_sc._simplices_table.items():
            if simps:
                dim_first_appear[d] = min(filt.get(s, 0.0) for s in simps)
        total = sum(len(v) for v in max_sc._simplices_table.values())
        return max_sc, filt, barcode, dim_first_appear, total, filt.values()

    def _compute(self):
        """Run the filtration engine and populate ``results``/``barcode``/``stability_data``.

        Builds the maximal complex once, derives the threshold grid, reduces the
        Z2 persistence barcode (via the Julia twist/clearing kernel when available),
        and walks the grid collecting Betti numbers (read from the barcode by a
        memory-flat searchsorted pass) plus optional manifold, integer-torsion, and
        connected-component invariants.
        """
        from pysurgery.topology.complexes import SimplicialComplex
        SC = SimplicialComplex

        max_sc, filt, barcode, dim_first_appear, total, grid_values = self._assemble()
        self.max_sc, self._filt = max_sc, filt
        coords = getattr(max_sc, "_coordinates", self.points)

        if self._precomputed_manifolds is not None:
            self.epsilons = self._precomputed_manifolds["epsilons"]
        else:
            self.epsilons = self._user_epsilons if self._user_epsilons is not None \
                else self._grid_from_values(grid_values)

        self.barcode = barcode

        is_julia_available = False
        try:
            from pysurgery.bridge.julia_bridge import julia_engine
            is_julia_available = julia_engine.available
        except Exception:
            pass

        if self.analyze_manifolds and total > self._MANIFOLD_MAX_SIMPLICES:
            if not is_julia_available or self.backend == "python":
                warnings.warn(
                    f"Skipping per-threshold manifold analysis: maximal complex has "
                    f"{total:,} simplices (> {self._MANIFOLD_MAX_SIMPLICES:,}). Betti "
                    "curves are still computed; pass a smaller eps_max to force it.",
                    stacklevel=2,
                )
                self.analyze_manifolds = False

        if self.compute_torsion and total > self._MANIFOLD_MAX_SIMPLICES:
            warnings.warn(
                f"compute_torsion=True on a large complex ({total:,} simplices): exact "
                "integer homology runs a Smith-normal-form solve at every reported "
                "threshold and may be slow. Coarsen with n_samples / eps_max if needed.",
                stacklevel=2,
            )

        def complex_dim_at(eps):
            ds = [d for d, f0 in dim_first_appear.items() if f0 <= eps + 1e-12]
            return max(ds) if ds else -1

        all_bettis = self._all_betti_from_barcode(barcode, self.epsilons)

        last_invariants = None
        active_rows: dict = {}
        merged_rows_history: dict = {}
        next_available_row_idx = 0
        # Cross-threshold caches keyed by the component's content hash. Stable
        # components (the common case in a merge filtration) reuse the result
        # computed when the component first appeared, skipping the expensive
        # per-component manifold check / Smith-normal-form solve on every
        # threshold after the first. Both are bounded by the number of distinct
        # components (~merge-tree size); the info value is a short string and
        # the torsion value is a small homology dict.
        component_info_cache: Dict[str, str] = {}
        component_homology_cache: Dict[str, dict] = {}

        # Incremental slice engine: a single sorted pass over the simplices plus a
        # union-find for components, instead of re-slicing and re-partitioning the
        # maximal complex at every threshold. Only engaged when a path actually
        # needs the explicit complex (manifold / torsion / component tracking);
        # otherwise Betti numbers come straight off the barcode. ``need_*`` is
        # loop-invariant (``analyze_manifolds`` is finalized by the auto-skip above).
        need_components = self.track_connected_components or self.compute_torsion
        need_whole = self.analyze_manifolds and (self._precomputed_manifolds is None)
        need_complex = need_components or need_whole
        slices = (self._incremental_slices(max_sc, filt, need_components, need_whole)
                  if need_complex else None)

        for idx, eps in enumerate(self.epsilons):
            bettis = all_bettis[idx]
            # Pull this threshold's slice state from the incremental engine (in
            # lockstep with the grid); the yielded tables are consumed before the
            # generator resumes and mutates them again.
            if slices is not None:
                comp_tables, full_table, cdim_inc = next(slices)
            else:
                comp_tables, full_table, cdim_inc = None, None, -1

            torsion = {}
            if self.compute_torsion:
                # Direct-sum torsion over components, memoizing each component's
                # integer homology across thresholds by content hash. Equivalent
                # to sc.homology() (H_n(⊔Xᵢ)=⊕H_n(Xᵢ)) — see
                # _aggregate_component_torsion for the exact ordering contract.
                comp_homs = []
                for _vset, table in comp_tables:
                    key = self._component_content_key(table)
                    chz = component_homology_cache.get(key)
                    if chz is None:
                        comp = self._build_component_complex(SC, table, coords)
                        hz = comp.homology(backend=self.backend)
                        chz = hz if isinstance(hz, dict) else {}
                        component_homology_cache[key] = chz
                    comp_homs.append(chz)
                torsion = self._aggregate_component_torsion(comp_homs)

            if self.analyze_manifolds:
                if self._precomputed_manifolds is not None:
                    is_manifold = self._precomputed_manifolds["is_manifold"][idx]
                    dim = self._precomputed_manifolds["dimensions"][idx]
                    failures = self._precomputed_manifolds["failures"][idx]
                    is_closed = self._precomputed_manifolds["is_closed"][idx]
                    
                    manifold_str = "Yes" if is_manifold else f"No ({failures} dft)"
                    closed = "Yes" if is_closed else "No"
                    dim_repr = str(dim) if dim is not None and dim >= 0 else "N/A"
                else:
                    sc = self._whole_complex_from_table(SC, full_table, coords)
                    is_manifold, dim, diag = sc.is_homology_manifold(backend=self.backend)
                    manifold_str = "Yes" if is_manifold else f"No ({len(diag)} dft)"
                    closed = "N/A"
                    if is_manifold:
                        closed = "Yes" if sc.is_closed_manifold else "No"
                    dim_repr = str(dim) if dim is not None else "N/A"
            else:
                manifold_str = "N/A"
                closed = "N/A"
                cdim = cdim_inc if slices is not None else complex_dim_at(eps)
                dim_repr = str(cdim) if cdim >= 0 else "N/A"

            current_invariants = {
                "bettis": bettis,
                "is_manifold_str": manifold_str,
                "is_closed": closed,
                "dimension": dim_repr,
                "torsion": torsion,
            }

            comp_info_dict = {}
            if self.track_connected_components:
                next_active_rows = {}
                new_merges_this_step = {}
                for vset_fz, table in comp_tables:
                    vset = set(vset_fz)
                    absorbed = [r for r, prev_vset in active_rows.items() if prev_vset.issubset(vset)]
                    if absorbed:
                        survivor = min(absorbed)
                        next_active_rows[survivor] = vset
                        for r in absorbed:
                            if r != survivor:
                                new_merges_this_step[r] = survivor
                    else:
                        r_new = next_available_row_idx
                        next_available_row_idx += 1
                        next_active_rows[r_new] = vset

                    key = self._component_content_key(table)
                    info = component_info_cache.get(key)
                    if info is None:
                        # Build the component only on a miss — stable components
                        # reuse the cached info without re-materializing.
                        sub_sc = self._build_component_complex(SC, table, coords)
                        c_is_mani, c_dim, c_diag = sub_sc.is_homology_manifold(backend=self.backend)
                        if c_is_mani:
                            c_closed = "Closed" if sub_sc.is_closed_manifold else "Bound"
                            info = f"M(D:{c_dim}, {c_closed})"
                        else:
                            info = f"Non-M ({len(c_diag)} dft)"
                        component_info_cache[key] = info

                    for r_idx, vs in next_active_rows.items():
                        if vs == vset:
                            comp_info_dict[r_idx] = info
                            break

                for r_idx, target_r in new_merges_this_step.items():
                    comp_info_dict[r_idx] = f"Merged (C_{target_r + 1})"
                    merged_rows_history[r_idx] = True

                for r_idx in merged_rows_history:
                    if r_idx not in comp_info_dict:
                        comp_info_dict[r_idx] = "-"

                active_rows = next_active_rows
                current_invariants["comp_info"] = [comp_info_dict.get(r, "-") for r in range(next_available_row_idx)]

            if self.dynamic_mode and last_invariants is not None and current_invariants == last_invariants:
                continue

            last_invariants = current_invariants
            self.all_betti_dims.update(bettis.keys())
            self.results.append({
                "epsilon": eps,
                "bettis": bettis,
                "is_manifold": manifold_str,
                "is_closed": closed,
                "dimension": dim_repr,
                "torsion": torsion,
                "comp_info_map": comp_info_dict.copy() if self.track_connected_components else {},
            })

        self.max_comp_rows = next_available_row_idx

        self.stability_data = []
        for j in range(len(self.results)):
            start_eps = self.results[j]["epsilon"]
            end_eps = self.results[j + 1]["epsilon"] if j + 1 < len(self.results) \
                else (self.epsilons[-1] if not self.dynamic_mode else start_eps * 1.1)
            self.stability_data.append((end_eps - start_eps, self.results[j]["bettis"], start_eps, end_eps))
        self.stability_data.sort(key=lambda x: x[0], reverse=True)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def to_markdown(self) -> str:
        """Render the filtration report as Markdown.

        Returns:
            A Markdown string with a unified table containing Betti numbers, 
            manifold-status, integer-torsion (when ``compute_torsion`` is set), 
            and connected-components (when tracked), plus a list of the most 
            persistent homologies.
        """
        if not self.results:
            return "No filtration steps recorded."

        sorted_betti_dims = sorted(list(self.all_betti_dims))
        reported_epsilons = [res["epsilon"] for res in self.results]
        eps_label = self.param_label

        def format_table_with_dynamic_eps(rows_without_eps_formatted, eps_values):
            max_w = 0
            for row in rows_without_eps_formatted:
                for cell in row:
                    max_w = max(max_w, len(str(cell)))
            max_w = max(max_w, 10)
            formatted_eps = []
            for e in eps_values:
                s_int = str(int(e))
                precision = max_w - len(s_int) - 1
                if precision < 0:
                    formatted_eps.append(str(e)[:max_w].ljust(max_w))
                else:
                    formatted_eps.append(f"{e:.{precision}f}")
            final_rows = [[rows_without_eps_formatted[0][0]] + formatted_eps]
            for row in rows_without_eps_formatted[1:]:
                final_rows.append(row)
            return _format_padded_table(final_rows)

        merged_rows = [[f"Property \\ {eps_label}"]]
        
        # 1. Betti Numbers
        for d in sorted_betti_dims:
            merged_rows.append([f"b_{d}"] + [str(res["bettis"].get(d, 0)) for res in self.results])
            
        # 2. Manifold Status
        merged_rows.append(["Is Homology Manifold"] + [res["is_manifold"] for res in self.results])
        merged_rows.append(["Is Closed"] + [res["is_closed"] for res in self.results])
        with_boundary = [("No" if r["is_closed"] == "Yes" else "Yes") if "Yes" in r["is_manifold"] else "N/A"
                         for r in self.results]
        merged_rows.append(["With Boundary"] + with_boundary)
        merged_rows.append(["Dimension"] + [res["dimension"] for res in self.results])
        
        # 3. Torsion
        if self.compute_torsion:
            tors_dims = sorted({d for res in self.results for d in res.get("torsion", {})})
            def _fmt_torsion(coeffs):
                if not coeffs:
                    return "0"
                return " + ".join(f"Z/{t}" for t in sorted(coeffs))
            
            if tors_dims:
                for d in tors_dims:
                    merged_rows.append(
                        [f"Torsion H_{d}"] + [_fmt_torsion(res.get("torsion", {}).get(d, [])) for res in self.results]
                    )
            else:
                merged_rows.append(["Torsion"] + ["0" for _ in self.results])
                
        # 4. Connected Components
        if self.track_connected_components:
            for i in range(self.max_comp_rows):
                merged_rows.append([f"C_{i+1}"] + [res["comp_info_map"].get(i, "-") for res in self.results])

        unified_table = format_table_with_dynamic_eps(merged_rows, reported_epsilons)
        report = (f"# Unified Filtration Report (Method: {self.method_name})\n\n" + unified_table)

        stab_lines = ["\n# Most Persistent Homologies"]
        for i, (dist, b, s, e) in enumerate(self.stability_data[:5]):
            b_str = ", ".join(f"b_{d}={b.get(d, 0)}" for d in sorted_betti_dims if b.get(d, 0) > 0)
            stab_lines.append(f"{i+1}. Range: {dist:.6f} [{s:.6f} -> {e:.6f}] | {b_str}")

        return report + "\n" + "\n".join(stab_lines)

    def plot(self, barcode: bool = False) -> Any:
        """Build an interactive Plotly figure of the Betti curves or persistence barcode.

        Args:
            barcode: If True, plot the persistence barcode. Otherwise, plot Betti curves.

        Returns:
            A ``plotly.graph_objects.Figure``.

        Raises:
            ImportError: If Plotly is not installed.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("Plotly is required for .plot(). Install with 'pip install plotly'.")

        if not barcode:
            steps_eps = [res["epsilon"] for res in self.results]
            betti_history = defaultdict(list)
            all_dims = sorted(list(self.all_betti_dims))
            for res in self.results:
                for d in all_dims:
                    betti_history[d].append(res["bettis"].get(d, 0))

            fig = go.Figure()
            for d in all_dims:
                fig.add_trace(go.Scatter(x=steps_eps, y=betti_history[d], mode="lines+markers",
                                         name=f"b_{d}", line=dict(shape="hv")))
            fig.update_layout(title=f"Betti Curves ({self.method_name})", xaxis_title=self.param_label,
                              yaxis_title="Betti Number", template="plotly_white", hovermode="x unified")
            return fig

        import math
        if not self.barcode:
            fig = go.Figure()
            fig.update_layout(title=f"Persistence Barcode ({self.method_name}) - Empty",
                              xaxis_title=self.param_label, template="plotly_white")
            return fig

        finite_vals = []
        for dim, birth, death in self.barcode:
            if birth is not None and not math.isinf(birth) and not math.isnan(birth):
                finite_vals.append(birth)
            if death is not None and not math.isinf(death) and not math.isnan(death):
                finite_vals.append(death)
        
        max_finite = max(finite_vals) if finite_vals else 1.0
        cap_val = max_finite * 1.15 if max_finite > 0 else 1.15

        bars_by_dim = defaultdict(list)
        for dim, birth, death in self.barcode:
            bars_by_dim[dim].append((birth, death))

        sorted_dims = sorted(bars_by_dim.keys())
        fig = go.Figure()

        y_val = 0
        y_ticks_vals = []
        y_ticks_text = []
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

        for color_idx, dim in enumerate(sorted_dims):
            dim_bars = bars_by_dim[dim]
            dim_bars.sort(key=lambda x: (x[0], -x[1] if not math.isinf(x[1]) else -math.inf))
            dim_color = colors[color_idx % len(colors)]
            
            dim_y_start = y_val
            for birth, death in dim_bars:
                is_inf = math.isinf(death) or death is None
                end_x = cap_val if is_inf else death
                
                fig.add_trace(go.Scatter(
                    x=[birth, end_x],
                    y=[y_val, y_val],
                    mode="lines",
                    line=dict(color=dim_color, width=3),
                    hoverinfo="skip",
                    showlegend=False
                ))
                
                if is_inf:
                    fig.add_trace(go.Scatter(
                        x=[end_x],
                        y=[y_val],
                        mode="markers",
                        marker=dict(symbol="triangle-right", size=8, color=dim_color),
                        hoverinfo="skip",
                        showlegend=False
                    ))
                y_val += 1
            
            dim_y_end = y_val - 1
            if dim_y_end >= dim_y_start:
                y_ticks_vals.append((dim_y_start + dim_y_end) / 2.0)
                y_ticks_text.append(f"H_{dim}")
                
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color=dim_color, width=3),
                name=f"H_{dim}"
            ))

            if dim != sorted_dims[-1]:
                fig.add_shape(
                    type="line",
                    x0=0,
                    y0=y_val - 0.5,
                    x1=cap_val,
                    y1=y_val - 0.5,
                    line=dict(color="rgba(0,0,0,0.1)", width=1, dash="dash")
                )

        # Add invisible trace across the epsilons grid to show the Betti numbers in unified hover
        betti_x = [res["epsilon"] for res in self.results]
        betti_text = []
        all_dims = sorted(list(self.all_betti_dims))
        for res in self.results:
            eps = res["epsilon"]
            bettis = res["bettis"]
            b_str = ", ".join(f"b_{d}={bettis.get(d, 0)}" for d in all_dims)
            betti_text.append(f"Betti: {b_str}")

        fig.add_trace(go.Scatter(
            x=betti_x,
            y=[y_val / 2.0] * len(betti_x),
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            name="Invariants",
            hoverinfo="text",
            hovertext=betti_text,
            showlegend=False
        ))

        fig.update_layout(
            title=f"Persistence Barcode ({self.method_name})",
            xaxis_title=self.param_label,
            yaxis=dict(
                tickvals=y_ticks_vals,
                ticktext=y_ticks_text,
                showgrid=False,
                zeroline=False
            ),
            template="plotly_white",
            showlegend=True,
            hovermode="x unified"
        )
        fig.update_xaxes(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikedash="dash",
            spikecolor="rgba(0,0,0,0.3)",
            spikethickness=1
        )
        return fig

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------
    @staticmethod
    def _warmup_points() -> np.ndarray:
        """A tiny fixed point cloud in general position for end-to-end warmup.

        Returns:
            A deterministic ``(12, 3)`` array of coordinates.
        """
        return np.random.default_rng(0).standard_normal((12, 3))

    @classmethod
    def warmup(cls, points: Optional[Union[np.ndarray, "PointCloud"]] = None) -> bool:
        """Run the full report path on a tiny complex to compile/JIT everything.

        Failures are non-fatal (warned), so warming one report type never blocks
        another.

        Args:
            points: Optional point cloud to warm on; defaults to a tiny fixed cloud.

        Returns:
            True if the warmup report built successfully, else False.
        """
        pts = cls._warmup_points() if points is None else np.asarray(points, dtype=np.float64)
        try:
            cls(pts, n_samples=5, analyze_manifolds=False, **cls._warmup_kwargs)
            return True
        except Exception as exc:  # pragma: no cover - warmup is best-effort
            warnings.warn(f"{cls.__name__}.warmup failed: {exc!r}", stacklevel=2)
            return False

    def __str__(self) -> str:
        return self.to_markdown()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} steps={len(self.results)} bars={len(self.barcode)}>"


# ──────────────────────────────────────────────────────────────────────────────
# Concrete report types
# ──────────────────────────────────────────────────────────────────────────────
class RipsFiltrationReport(_BaseFiltrationReport):
    """Vietoris-Rips filtration: simplices enter at their longest edge.

    Args:
        points: (N, D) array of point coordinates.
        epsilons: Explicit thresholds. If None, a bounded grid is derived.
        max_dimension: Maximum simplex dimension to build.
        coefficient_ring: Coefficient ring (manifold/component path).
        backend: 'auto', 'julia', or 'python'.
        track_connected_components: Track per-component evolution.
        n_samples: Number of evenly-spaced thresholds.
        eps_max: Cap for the parameter range (defaults to maximum pairwise distance).
        analyze_manifolds: Run per-threshold homology-manifold check.
        compute_torsion: Additionally compute exact integer homology.
        manifold_analysis: Alias/override for analyze_manifolds.
        rips_engine: Which fused Julia engine to use: 'clique', 'cohomology', or 'auto'.
    """

    param_label = "Eps"
    method_name = "Vietoris-Rips"

    # Below this many points the staged build is already cheap, so we skip the
    # fused probe and keep the exact staged path (which the test-suite exercises).
    _RIPS_FUSED_MIN_POINTS = 256

    def _rips_eps_max(self) -> float:
        """The diameter cap for the maximal complex (explicit ``eps_max`` or pdist max)."""
        from scipy.spatial.distance import pdist
        if self.eps_max is not None:
            return float(self.eps_max)
        return float(pdist(self.points).max()) if len(self.points) > 1 else 1.0

    def _build_maximal_and_values(self):
        """Build the Vietoris-Rips complex at the full pairwise diameter with longest-edge values."""
        from pysurgery.topology.complexes import SimplicialComplex as SC
        from pysurgery.topology.filtration_values import rips_filtration_values
        sc = SC.from_vietoris_rips(self.points, self._rips_eps_max(), self.max_dimension,
                                   coefficient_ring=self.coefficient_ring, backend=self.backend)
        return sc, rips_filtration_values(sc._simplices_table, self.points)

    def _select_rips_engine(self) -> str:
        """Which fused Julia engine to use: 'clique' (homology) or 'cohomology'.

        ``rips_engine`` kwarg, default ``'auto'``:
          - ``'auto'`` -> implicit cohomology for ``max_dimension >= 3`` (where the
            homology reduction blows up and cohomology wins), else the clique engine
            (faster in the common low-dimensional case);
          - ``'clique'`` / ``'cohomology'`` force the respective engine.
        Both are exact and return the identical barcode.
        """
        engine = str(self.kwargs.get("rips_engine", "auto")).lower()
        if engine == "auto":
            return "cohomology" if self.max_dimension >= 3 else "clique"
        if engine not in ("clique", "cohomology"):
            warnings.warn(f"Unknown rips_engine={engine!r}; using 'auto'.", stacklevel=2)
            return "cohomology" if self.max_dimension >= 3 else "clique"
        return engine

    def _assemble(self):
        """Fuse the VR build, longest-edge filtration, and reduction inside Julia.

        For large clouds this runs the entire hot path in a single Julia call and
        keeps the maximal complex *implicit* (``max_sc = None``): the per-threshold
        loop needs only the barcode and per-dimension stats when no manifold /
        torsion / component analysis is requested (manifold analysis is itself
        auto-skipped above ``_MANIFOLD_MAX_SIMPLICES``). When the complex is small
        enough to be wanted -- or any path needs it explicitly -- we defer to the
        staged base build, so small-complex behaviour is byte-identical to before.

        The reduction runs in one of two exact engines (see :meth:`_select_rips_engine`):
        the clique homology engine or the implicit-cohomology engine; both return
        the identical barcode.
        """
        need_explicit = self.track_connected_components or self.compute_torsion
        if (need_explicit or self.backend == "python"
                or len(self.points) < self._RIPS_FUSED_MIN_POINTS):
            return super()._assemble()
        try:
            from pysurgery.bridge.julia_bridge import julia_engine
            if not julia_engine.available:
                return super()._assemble()
            eps_max = self._rips_eps_max()
            if self._select_rips_engine() == "cohomology":
                payload = julia_engine.compute_rips_cohomology(
                    self.points, eps_max, self.max_dimension,
                    analyze_manifolds=self.analyze_manifolds,
                    n_samples=self.n_samples,
                )
            else:
                payload = julia_engine.compute_rips_filtration(
                    self.points, eps_max, self.max_dimension,
                    analyze_manifolds=self.analyze_manifolds,
                    n_samples=self.n_samples,
                )
        except Exception as exc:  # pragma: no cover - only when Julia errors
            warnings.warn(
                f"Fused Julia Rips filtration failed ({exc!r}); falling back to the "
                "staged build.", stacklevel=2)
            return super()._assemble()

        if payload["total"] <= self._MANIFOLD_MAX_SIMPLICES:
            # Small enough that manifold analysis / introspection wants the explicit
            # complex; the staged build is cheap at this size and keeps full fidelity.
            return super()._assemble()

        # Large: keep the complex implicit -- this is the whole point of the fusion.
        if "manifold_data" in payload and payload["manifold_data"] is not None:
            self._precomputed_manifolds = payload["manifold_data"]

        return (None, None, payload["barcode"], payload["dim_first_appear"],
                payload["total"], payload["eps_values"])


class CknnFiltrationReport(_BaseFiltrationReport):
    """Continuous k-NN filtration: edge (i, j) enters at d(i,j)/sqrt(rho_i*rho_j).

    Args:
        points: (N, D) array of point coordinates.
        epsilons: Explicit thresholds. If None, a bounded grid is derived.
        max_dimension: Maximum simplex dimension to build.
        coefficient_ring: Coefficient ring (manifold/component path).
        backend: 'auto', 'julia', or 'python'.
        track_connected_components: Track per-component evolution.
        n_samples: Number of evenly-spaced thresholds.
        eps_max: Cap for the parameter range (defaults to 2.0).
        analyze_manifolds: Run per-threshold homology-manifold check.
        compute_torsion: Additionally compute exact integer homology.
        manifold_analysis: Alias/override for analyze_manifolds.
        k: Number of nearest neighbors to use for CkNN density scaling (default 8).
    """

    param_label = "Delta"
    method_name = "CkNN"

    def _build_maximal_and_values(self):
        """Build the CkNN complex and its d/sqrt(rho_i*rho_j) edge-based values."""
        from pysurgery.topology.complexes import SimplicialComplex as SC
        from pysurgery.topology.filtration_values import cknn_filtration_values
        from scipy.spatial import cKDTree
        k = int(self.kwargs.get("k", 8))
        delta_max = float(self.eps_max) if self.eps_max is not None else 2.0
        sc = SC.from_point_cloud_cknn(self.points, k=k, delta=delta_max,
                                      max_dimension=self.max_dimension,
                                      coefficient_ring=self.coefficient_ring, backend=self.backend)
        n = len(self.points)
        kk = min(k, max(1, n - 1))
        dists, _ = cKDTree(self.points).query(self.points, k=kk + 1)
        rho = np.atleast_2d(dists)[:, -1]
        return sc, cknn_filtration_values(sc._simplices_table, self.points, rho)


class AlphaFiltrationReport(_BaseFiltrationReport):
    """Alpha complex filtration on the Delaunay triangulation.

    Appearance values are circumradius / Gabriel based.

    Args:
        points: (N, D) array of point coordinates.
        epsilons: Explicit thresholds. If None, a bounded grid is derived.
        max_dimension: Maximum simplex dimension to build.
        coefficient_ring: Coefficient ring (manifold/component path).
        backend: 'auto', 'julia', or 'python'.
        track_connected_components: Track per-component evolution.
        n_samples: Number of evenly-spaced thresholds.
        eps_max: Cap for the parameter range.
        analyze_manifolds: Run per-threshold homology-manifold check.
        compute_torsion: Additionally compute exact integer homology.
        manifold_analysis: Alias/override for analyze_manifolds.
    """

    param_label = "Alpha"
    method_name = "Alpha"
    _ALPHA_FUSED_MIN_POINTS = 256

    def _assemble(self):
        """Fuse the Alpha complex build, Gabriel empty-sphere test, and Z2 reduction inside Julia.

        For large clouds, this runs the entire hot path inside Julia, bypassing the
        staged build and python loops entirely. For small complexes, we defer to the
        staged build.
        """
        need_explicit = self.track_connected_components or self.compute_torsion
        if (need_explicit or self.backend == "python"
                or len(self.points) < self._ALPHA_FUSED_MIN_POINTS):
            return super()._assemble()
        try:
            from scipy.spatial import Delaunay
            from pysurgery.bridge.julia_bridge import julia_engine
            if not julia_engine.available:
                return super()._assemble()
            
            pts = self.points
            n, dim = pts.shape
            if n < dim + 1:
                return super()._assemble()
                
            dt = Delaunay(pts, qhull_options="QJ")
            payload = julia_engine.compute_alpha_filtration(
                pts, dt.simplices, self.max_dimension,
                analyze_manifolds=self.analyze_manifolds,
                n_samples=self.n_samples,
                eps_max=self.eps_max,
            )
        except Exception as exc:
            warnings.warn(
                f"Fused Julia Alpha filtration failed ({exc!r}); falling back to the "
                "staged build.", stacklevel=2)
            return super()._assemble()

        if payload["total"] <= self._MANIFOLD_MAX_SIMPLICES:
            return super()._assemble()

        if "manifold_data" in payload and payload["manifold_data"] is not None:
            self._precomputed_manifolds = payload["manifold_data"]

        return (None, None, payload["barcode"], payload["dim_first_appear"],
                payload["total"], payload["eps_values"])

    def _build_maximal_and_values(self):
        """Build the Delaunay complex with circumradius/Gabriel alpha values."""
        from pysurgery.topology.complexes import SimplicialComplex as SC
        from pysurgery.topology.filtration_values import alpha_filtration_values
        from scipy.spatial import Delaunay
        pts = self.points
        n, dim = pts.shape
        if n < dim + 1:
            vals = {(i,): 0.0 for i in range(n)}
            return self._complex_from_values(SC, vals, pts, self.coefficient_ring), vals
        dt = Delaunay(pts, qhull_options="QJ")
        vals = alpha_filtration_values(pts, dt.simplices, self.max_dimension)
        return self._complex_from_values(SC, vals, pts, self.coefficient_ring), vals


class DelaunayRipsFiltrationReport(_BaseFiltrationReport):
    """Rips filtration restricted to Delaunay edges (longest-edge values).

    Args:
        points: (N, D) array of point coordinates.
        epsilons: Explicit thresholds. If None, a bounded grid is derived.
        max_dimension: Maximum simplex dimension to build.
        coefficient_ring: Coefficient ring (manifold/component path).
        backend: 'auto', 'julia', or 'python'.
        track_connected_components: Track per-component evolution.
        n_samples: Number of evenly-spaced thresholds.
        eps_max: Cap for the parameter range.
        analyze_manifolds: Run per-threshold homology-manifold check.
        compute_torsion: Additionally compute exact integer homology.
        manifold_analysis: Alias/override for analyze_manifolds.
    """

    param_label = "Eps"
    method_name = "Delaunay-Rips"

    def _build_maximal_and_values(self):
        """Build the Delaunay-Rips complex (longest-edge values on Delaunay edges)."""
        from pysurgery.topology.complexes import SimplicialComplex as SC
        sc = SC.from_delaunay_rips(self.points, threshold=None, max_dimension=self.max_dimension,
                                   coefficient_ring=self.coefficient_ring, backend=self.backend)
        return sc, dict(sc.filtration)


class DelaunayCechFiltrationReport(_BaseFiltrationReport):
    """Cech filtration on the Delaunay complex (smallest-enclosing-ball values).

    Args:
        points: (N, D) array of point coordinates.
        epsilons: Explicit thresholds. If None, a bounded grid is derived.
        max_dimension: Maximum simplex dimension to build.
        coefficient_ring: Coefficient ring (manifold/component path).
        backend: 'auto', 'julia', or 'python'.
        track_connected_components: Track per-component evolution.
        n_samples: Number of evenly-spaced thresholds.
        eps_max: Cap for the parameter range.
        analyze_manifolds: Run per-threshold homology-manifold check.
        compute_torsion: Additionally compute exact integer homology.
        manifold_analysis: Alias/override for analyze_manifolds.
    """

    param_label = "Radius"
    method_name = "Delaunay-Cech"

    def _build_maximal_and_values(self):
        """Build the Delaunay-Cech complex (smallest-enclosing-ball radius values)."""
        from pysurgery.topology.complexes import SimplicialComplex as SC
        sc = SC.from_delaunay_cech(self.points, threshold=None, max_dimension=self.max_dimension,
                                   coefficient_ring=self.coefficient_ring, backend=self.backend)
        return sc, dict(sc.filtration)


class WitnessFiltrationReport(_BaseFiltrationReport):
    """Lazy-witness filtration on landmark points.

    Edge ``(i, j)`` enters at ``min over witnesses p of max(d(p, l_i), d(p, l_j))``
    and higher simplices at the max over their edges (flag complex).

    Args:
        points: (N, D) array of point coordinates.
        epsilons: Explicit thresholds. If None, a bounded grid is derived.
        max_dimension: Maximum simplex dimension to build.
        coefficient_ring: Coefficient ring (manifold/component path).
        backend: 'auto', 'julia', or 'python'.
        track_connected_components: Track per-component evolution.
        n_samples: Number of evenly-spaced thresholds.
        eps_max: Cap for the parameter range (defaults to median edge distance).
        analyze_manifolds: Run per-threshold homology-manifold check.
        compute_torsion: Additionally compute exact integer homology.
        manifold_analysis: Alias/override for analyze_manifolds.
        n_landmarks: Number of landmarks to subsample from the points (default min(N, 50)).
    """

    param_label = "Alpha"
    method_name = "Witness"
    _warmup_kwargs = {"n_landmarks": 6}

    def _build_maximal_and_values(self):
        """Build the lazy-witness flag complex on landmarks with witness edge values."""
        from pysurgery.topology.complexes import SimplicialComplex as SC
        from scipy.spatial.distance import cdist
        pts = self.points
        n = len(pts)
        n_land = max(1, min(int(self.kwargs.get("n_landmarks", min(n, 50))), n))
        landmarks = self._subsample_indices(n, n_land)
        L = pts[landmarks]
        nL = L.shape[0]
        Dl = cdist(pts, L)                                # (N, nL): witness->landmark distances

        edge_val: dict = {}
        for i in range(nL):
            di = Dl[:, i]
            for j in range(i + 1, nL):
                edge_val[(i, j)] = float(np.min(np.maximum(di, Dl[:, j])))

        if self.eps_max is not None:
            alpha_max = float(self.eps_max)
        else:
            vv = np.array(list(edge_val.values())) if edge_val else np.array([1.0])
            alpha_max = float(np.quantile(vv, 0.5))

        edges = [list(e) for e, v in edge_val.items() if v <= alpha_max]
        sc = SC.from_simplices([[i] for i in range(nL)] + edges,
                               coefficient_ring=self.coefficient_ring, close_under_faces=True)
        if self.max_dimension > 1 and edges:
            sc = sc.expand(self.max_dimension)
        sc._coordinates = L

        filt: dict = {}
        for d in sorted(sc._simplices_table.keys()):
            for s in sc.n_simplices(d):
                if len(s) == 1:
                    filt[s] = 0.0
                else:
                    m = 0.0
                    for a in range(len(s)):
                        for b in range(a + 1, len(s)):
                            e = (s[a], s[b]) if s[a] < s[b] else (s[b], s[a])
                            m = max(m, edge_val.get(e, 0.0))
                    filt[s] = m
        return sc, filt


# ──────────────────────────────────────────────────────────────────────────────
# Backward-compatible factory + warmup-all + legacy wrappers
# ──────────────────────────────────────────────────────────────────────────────
_MODE_TO_CLASS = {
    "vietoris_rips": RipsFiltrationReport,
    "rips": RipsFiltrationReport,
    "cknn": CknnFiltrationReport,
    "alpha": AlphaFiltrationReport,
    "delaunay": AlphaFiltrationReport,
    "delaunay_rips": DelaunayRipsFiltrationReport,
    "delaunay_cech": DelaunayCechFiltrationReport,
    "witness": WitnessFiltrationReport,
}


def FiltrationReport(
    points: Union[np.ndarray, "PointCloud"],
    epsilons: Optional[List[float]] = None,
    max_dimension: int = 2,
    coefficient_ring: str = "Z",
    backend: str = "auto",
    track_connected_components: bool = False,
    mode: str = "vietoris_rips",
    **kwargs,
):
    """Backward-compatible factory dispatching ``mode`` to a report class.

    Prefer the explicit classes (``RipsFiltrationReport`` etc.); this exists so
    older ``FiltrationReport(points, mode=...)`` call sites keep working.

    Args:
        points: (N, D) array of point coordinates.
        epsilons: Explicit thresholds, or None for a value-derived grid.
        max_dimension: Maximum simplex dimension to build.
        coefficient_ring: Coefficient ring label.
        backend: 'auto', 'julia', or 'python'.
        track_connected_components: Track per-component evolution.
        mode: One of ``vietoris_rips``/``rips``, ``cknn``, ``alpha``/``delaunay``,
            ``delaunay_rips``, ``delaunay_cech``, ``witness``.
        **kwargs: Method-specific options forwarded to the report class:
            - ``n_samples``: Number of evenly-spaced thresholds.
            - ``eps_max``: Cap for the parameter range.
            - ``analyze_manifolds``: Run per-threshold homology-manifold check.
            - ``compute_torsion``: Additionally compute exact integer homology.
            - ``manifold_analysis``: Alias/override for analyze_manifolds.
            - ``rips_engine``: Which fused Julia engine to use (for Rips).
            - ``k``: Number of nearest neighbors (for CkNN).
            - ``n_landmarks``: Number of landmarks to subsample (for Witness).

    Returns:
        A computed report instance of the class selected by ``mode``.

    Raises:
        ValueError: If ``mode`` is not recognised.
    """
    try:
        cls = _MODE_TO_CLASS[mode]
    except KeyError:
        raise ValueError(f"Unknown filtration mode {mode!r}; choose from {sorted(_MODE_TO_CLASS)}")
    return cls(
        points,
        epsilons=epsilons,
        max_dimension=max_dimension,
        coefficient_ring=coefficient_ring,
        backend=backend,
        track_connected_components=track_connected_components,
        **kwargs,
    )


def warm_all(points: Optional[Union[np.ndarray, "PointCloud"]] = None) -> Dict[str, bool]:
    """Warm up every filtration report type on a tiny complex (best-effort).

    Exercises the full Python+Julia path for each method so the first real report
    is fast. The underlying Julia builder kernels are also compiled by
    ``julia_engine.warmup()``.

    Args:
        points: Optional point cloud to warm on; defaults to a tiny fixed cloud.

    Returns:
        Dict mapping each method name to True/False for warmup success.
    """
    report_classes = [
        RipsFiltrationReport, CknnFiltrationReport, AlphaFiltrationReport,
        DelaunayRipsFiltrationReport, DelaunayCechFiltrationReport, WitnessFiltrationReport,
    ]
    return {c.method_name: c.warmup(points) for c in report_classes}


def generate_filtration_report(*args, **kwargs) -> str:
    """Legacy wrapper for ``FiltrationReport(...).to_markdown()``."""
    return str(FiltrationReport(*args, **kwargs))


def plot_filtration_report(*args, **kwargs):
    """Legacy wrapper for ``FiltrationReport(...).plot()``."""
    return FiltrationReport(*args, **kwargs).plot()
