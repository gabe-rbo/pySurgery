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

import warnings
import numpy as np
from collections import defaultdict
from typing import Any, Dict, List, Optional


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
        points: np.ndarray,
        epsilons: Optional[List[float]] = None,
        max_dimension: int = 2,
        coefficient_ring: str = "Z",
        backend: str = "auto",
        track_connected_components: bool = False,
        *,
        n_steps: int = 48,
        eps_max: Optional[float] = None,
        analyze_manifolds: bool = True,
        **kwargs,
    ):
        """Build and compute the filtration report.

        Args:
            points: (N, D) array of point coordinates.
            epsilons: Explicit thresholds. If None, a bounded grid is derived from
                the distribution of appearance values.
            max_dimension: Maximum simplex dimension to build.
            coefficient_ring: Coefficient ring (manifold/component path).
            backend: 'auto', 'julia', or 'python'.
            track_connected_components: Track per-component evolution (costly).
            n_steps: Number of thresholds in the default grid.
            eps_max: Cap for the parameter range. Method-dependent meaning; if None
                each method picks a sensible default.
            analyze_manifolds: Run the per-threshold homology-manifold check
                (auto-skipped above _MANIFOLD_MAX_SIMPLICES simplices).
            **kwargs: Method-specific options (e.g. ``k``, ``n_landmarks``).
        """
        self.points = np.asarray(points, dtype=np.float64)
        self.max_dimension = max_dimension
        self.coefficient_ring = coefficient_ring
        self.backend = backend
        self.track_connected_components = track_connected_components
        self.n_steps = max(2, int(n_steps))
        self.eps_max = eps_max
        self.analyze_manifolds = analyze_manifolds
        self.kwargs = kwargs

        self._user_epsilons = list(epsilons) if epsilons is not None else None
        self.dynamic_mode = epsilons is None
        self.epsilons: List[float] = []
        self.results: List[dict] = []
        self.all_betti_dims = set()
        self.max_comp_rows = 0
        self.barcode: List = []
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
    def _estimate_eps_max(cls, points: np.ndarray, factor: float = 1.2,
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
        """Bounded, deduplicated threshold grid from the appearance-value spread.

        Args:
            filt: Mapping of simplex to appearance value.

        Returns:
            A sorted list of at most ``n_steps`` thresholds (quantiles of the
            distinct appearance values, capped at ``eps_max`` if set), always
            including ``0.0``.
        """
        vals = np.array(sorted(set(filt.values())), dtype=np.float64) if filt else np.array([0.0])
        if self.eps_max is not None:
            vals = vals[vals <= float(self.eps_max)]
        if vals.size == 0:
            return [0.0]
        if vals.size <= self.n_steps:
            grid = vals
        else:
            grid = np.quantile(vals, np.linspace(0.0, 1.0, self.n_steps))
        return sorted({0.0, *(float(x) for x in grid)})

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

    @staticmethod
    def _z2_persistence_barcode(simplices_table: dict, filt: dict) -> list:
        """Persistent-homology barcode over Z2 by standard column reduction.

        One reduction pass yields every class's birth/death, so Betti numbers at
        all thresholds come from a single computation (the Ripser/GUDHI approach);
        reported Betti numbers are therefore mod-2 (equal to integer ranks unless
        the complex carries 2-torsion). Returns ``(dim, birth, death)`` intervals
        with ``death == inf`` for essential classes.
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
        """Betti numbers at a threshold from a persistence barcode.

        Args:
            barcode: List of ``(dim, birth, death)`` intervals.
            eps: The filtration threshold.
            tol: Relative tolerance for the half-open ``birth <= eps < death`` test.

        Returns:
            Dict mapping each dimension to the number of bars alive at ``eps``.
        """
        b: dict = {}
        e = eps + abs(eps) * tol + 1e-12
        for dim, birth, death in barcode:
            if birth <= e < death:
                b[dim] = b.get(dim, 0) + 1
        return b

    def _slice_complex(self, SC, max_sc, filt, eps, coords):
        """Sub-complex of ``max_sc`` containing simplices appearing by ``eps``.

        A cheap dictionary filter on appearance value; the result is closed under
        faces because appearance values are monotone, so it equals the complex the
        method would build directly at ``eps``.

        Args:
            SC: The SimplicialComplex class.
            max_sc: The maximal complex.
            filt: Mapping of simplex to appearance value.
            eps: The filtration threshold.
            coords: Point coordinates to attach to the sub-complex.

        Returns:
            The sub-complex at ``eps`` as a SimplicialComplex.
        """
        tol = abs(eps) * 1e-9 + 1e-12
        table = {}
        for d in sorted(max_sc._simplices_table.keys()):
            kept = [s for s in max_sc.n_simplices(d) if filt.get(s, 0.0) <= eps + tol]
            if kept:
                table[d] = kept
        sub = SC(simplices=table, coefficient_ring=self.coefficient_ring)
        sub._coordinates = coords
        if self.track_connected_components:
            sub._generate_point_cloud_mappings(coords)
        return sub

    # ------------------------------------------------------------------
    # Engine
    # ------------------------------------------------------------------
    def _compute(self):
        """Run the filtration engine and populate ``results``/``barcode``/``stability_data``.

        Builds the maximal complex once, derives the threshold grid, reduces the
        Z2 persistence barcode, and walks the grid collecting Betti numbers (from
        the barcode) plus optional manifold and connected-component invariants.
        """
        from pysurgery.topology.complexes import SimplicialComplex
        SC = SimplicialComplex

        max_sc, filt = self._build_maximal_and_values()
        self.max_sc, self._filt = max_sc, filt
        coords = getattr(max_sc, "_coordinates", self.points)

        self.epsilons = self._user_epsilons if self._user_epsilons is not None \
            else self._default_grid_from_values(filt)

        barcode = self._z2_persistence_barcode(max_sc._simplices_table, filt) \
            if max_sc._simplices_table else []
        self.barcode = barcode

        dim_first_appear: dict = {}
        for d, simps in max_sc._simplices_table.items():
            if simps:
                dim_first_appear[d] = min(filt.get(s, 0.0) for s in simps)

        total = sum(len(v) for v in max_sc._simplices_table.values())
        if self.analyze_manifolds and total > self._MANIFOLD_MAX_SIMPLICES:
            warnings.warn(
                f"Skipping per-threshold manifold analysis: maximal complex has "
                f"{total:,} simplices (> {self._MANIFOLD_MAX_SIMPLICES:,}). Betti "
                "curves are still computed; pass a smaller eps_max to force it.",
                stacklevel=2,
            )
            self.analyze_manifolds = False

        def complex_dim_at(eps):
            ds = [d for d, f0 in dim_first_appear.items() if f0 <= eps + 1e-12]
            return max(ds) if ds else -1

        last_invariants = None
        active_rows: dict = {}
        merged_rows_history: dict = {}
        next_available_row_idx = 0

        for eps in self.epsilons:
            need_complex = self.analyze_manifolds or self.track_connected_components
            sc = self._slice_complex(SC, max_sc, filt, eps, coords) if need_complex else None
            bettis = self._betti_from_barcode(barcode, eps)

            if self.analyze_manifolds:
                is_manifold, dim, diag = sc.is_homology_manifold(backend=self.backend)
                manifold_str = "Yes" if is_manifold else f"No ({len(diag)} dft)"
                closed = "N/A"
                if is_manifold:
                    closed = "Yes" if sc.is_closed_manifold else "No"
                dim_repr = str(dim) if dim is not None else "N/A"
            else:
                manifold_str = "N/A"
                closed = "N/A"
                cdim = sc.dimension if sc is not None else complex_dim_at(eps)
                dim_repr = str(cdim) if cdim >= 0 else "N/A"

            current_invariants = {
                "bettis": bettis,
                "is_manifold_str": manifold_str,
                "is_closed": closed,
                "dimension": dim_repr,
            }

            comp_info_dict = {}
            if self.track_connected_components:
                components_sc = sc.explode()
                next_active_rows = {}
                new_merges_this_step = {}
                for sub_sc in components_sc:
                    vlist = [v[0] for v in sub_sc.n_simplices(0)]
                    vset = set(vlist)
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

                    c_is_mani, c_dim, c_diag = sub_sc.is_homology_manifold(backend=self.backend)
                    if c_is_mani:
                        c_closed = "Closed" if sub_sc.is_closed_manifold else "Bound"
                        info = f"M(D:{c_dim}, {c_closed})"
                    else:
                        info = f"Non-M ({len(c_diag)} dft)"

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
            A Markdown string with the Betti-number table, the most-persistent
            homologies, the manifold-status table, and (when tracked) the
            connected-components table.
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

        betti_rows_init = [[f"Betti \\ {eps_label}"]]
        for d in sorted_betti_dims:
            betti_rows_init.append([f"b_{d}"] + [str(res["bettis"].get(d, 0)) for res in self.results])
        betti_report = (f"# Betti Numbers Report (Method: {self.method_name})\n"
                        + format_table_with_dynamic_eps(betti_rows_init, reported_epsilons))

        stab_lines = ["\n# Most Persistent Homologies"]
        for i, (dist, b, s, e) in enumerate(self.stability_data[:5]):
            b_str = ", ".join(f"b_{d}={b.get(d, 0)}" for d in sorted_betti_dims if b.get(d, 0) > 0)
            stab_lines.append(f"{i+1}. Range: {dist:.6f} [{s:.6f} -> {e:.6f}] | {b_str}")

        mani_rows_init = [[f"Property \\ {eps_label}"]]
        mani_rows_init.append(["Is Manifold"] + [res["is_manifold"] for res in self.results])
        mani_rows_init.append(["Is Closed"] + [res["is_closed"] for res in self.results])
        with_boundary = [("No" if r["is_closed"] == "Yes" else "Yes") if "Yes" in r["is_manifold"] else "N/A"
                         for r in self.results]
        mani_rows_init.append(["With Boundary"] + with_boundary)
        mani_rows_init.append(["Dimension"] + [res["dimension"] for res in self.results])
        mani_report = "\n# Manifold Status Report\n" + format_table_with_dynamic_eps(mani_rows_init, reported_epsilons)

        comp_report = ""
        if self.track_connected_components:
            comp_rows_init = [[f"Component \\ {eps_label}"]]
            for i in range(self.max_comp_rows):
                comp_rows_init.append([f"C_{i+1}"] + [res["comp_info_map"].get(i, "-") for res in self.results])
            comp_report = "\n\n# Connected Components Report\n" + format_table_with_dynamic_eps(comp_rows_init, reported_epsilons)

        return betti_report + "\n" + "\n".join(stab_lines) + "\n" + mani_report + comp_report

    def plot(self) -> Any:
        """Build an interactive Plotly figure of the Betti curves.

        Returns:
            A ``plotly.graph_objects.Figure`` with one step plot per dimension.

        Raises:
            ImportError: If Plotly is not installed.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("Plotly is required for .plot(). Install with 'pip install plotly'.")

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
    def warmup(cls, points: Optional[np.ndarray] = None) -> bool:
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
            cls(pts, n_steps=5, analyze_manifolds=False, **cls._warmup_kwargs)
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
    """Vietoris-Rips filtration: simplices enter at their longest edge."""

    param_label = "Eps"
    method_name = "Vietoris-Rips"

    def _build_maximal_and_values(self):
        """Build the Vietoris-Rips complex at the k-NN scale with longest-edge values."""
        from pysurgery.topology.complexes import SimplicialComplex as SC
        from pysurgery.topology.filtration_values import rips_filtration_values
        eps_max = self.eps_max if self.eps_max is not None else self._estimate_eps_max(self.points)
        sc = SC.from_vietoris_rips(self.points, eps_max, self.max_dimension,
                                   coefficient_ring=self.coefficient_ring, backend=self.backend)
        return sc, rips_filtration_values(sc._simplices_table, self.points)


class CknnFiltrationReport(_BaseFiltrationReport):
    """Continuous k-NN filtration: edge (i, j) enters at d(i,j)/sqrt(rho_i*rho_j)."""

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
    """

    param_label = "Alpha"
    method_name = "Alpha"

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
    """Rips filtration restricted to Delaunay edges (longest-edge values)."""

    param_label = "Eps"
    method_name = "Delaunay-Rips"

    def _build_maximal_and_values(self):
        """Build the Delaunay-Rips complex (longest-edge values on Delaunay edges)."""
        from pysurgery.topology.complexes import SimplicialComplex as SC
        sc = SC.from_delaunay_rips(self.points, threshold=None, max_dimension=self.max_dimension,
                                   coefficient_ring=self.coefficient_ring, backend=self.backend)
        return sc, dict(sc.filtration)


class DelaunayCechFiltrationReport(_BaseFiltrationReport):
    """Cech filtration on the Delaunay complex (smallest-enclosing-ball values)."""

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
    points: np.ndarray,
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
        **kwargs: Method-specific options forwarded to the report class (e.g.
            ``k``, ``n_landmarks``, ``n_steps``, ``eps_max``, ``analyze_manifolds``).

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


def warm_all(points: Optional[np.ndarray] = None) -> Dict[str, bool]:
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
