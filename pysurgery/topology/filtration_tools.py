"""
Filtration tools for point cloud analysis and manifold verification.
"""

import numpy as np
from collections import defaultdict
from typing import List, Literal, Optional, Any


def _format_padded_table(rows: List[List[str]]) -> str:
    """Format a list of rows into a markdown table with uniform column widths."""
    if not rows:
        return ""
    
    # Calculate the maximum width needed across all cells in the entire table
    # to ensure all columns have the same "uniform" size as requested.
    max_w = 0
    for row in rows:
        for cell in row:
            max_w = max(max_w, len(str(cell)))
    
    col_width = max_w + 2
    lines = []
    # Header
    header = rows[0]
    lines.append("| " + " | ".join(str(cell).ljust(col_width) for cell in header) + " |")
    
    # Let's rewrite separator logic for clarity
    sep_cells = ["-" * (col_width + 2) for _ in header]
    lines.append("|" + "|".join(sep_cells) + "|")
    
    # Data rows
    for row in rows[1:]:
        lines.append("| " + " | ".join(str(cell).ljust(col_width) for cell in row) + " |")
        
    return "\n".join(lines)


class FiltrationReport:
    """A comprehensive report on point cloud filtration with manifold analysis.
    
    Overview:
        The FiltrationReport object computes and stores the topological life cycle of a 
        point cloud. It identifies Betti numbers, manifold properties, and connected 
        component evolution across a range of distance thresholds.

    Attributes:
        results (List[Dict]): Raw data for each filtration step.
        stability_data (List[Tuple]): Ranked topological regimes by persistence range.
        mode (str): Complex construction mode used.
        epsilons (List[float]): The thresholds used for filtration.
    """

    def __init__(
        self,
        points: np.ndarray,
        epsilons: Optional[List[float]] = None,
        max_dimension: int = 2,
        coefficient_ring: str = "Z",
        backend: str = "auto",
        track_connected_components: bool = False,
        mode: Literal["vietoris_rips", "alpha", "witness", "cknn"] = "vietoris_rips",
        **kwargs
    ):
        """Initialize and compute the filtration report."""
        from scipy.spatial.distance import pdist

        self.points = points
        self.max_dimension = max_dimension
        self.coefficient_ring = coefficient_ring
        self.backend = backend
        self.track_connected_components = track_connected_components
        self.mode = mode
        self.kwargs = kwargs

        # 1. Epsilon Selection
        if epsilons is None:
            dists = pdist(points)
            unique_dists = sorted(np.unique(dists))
            if mode == "alpha":
                self.epsilons = [0.0] + [d/2.0 for d in unique_dists]
            elif mode == "cknn":
                self.epsilons = [0.1, 0.5, 1.0, 1.5, 2.0]
            else:
                self.epsilons = sorted(unique_dists)
                if self.epsilons and self.epsilons[0] > 0:
                    self.epsilons = [0.0] + self.epsilons
                elif not self.epsilons:
                    self.epsilons = [0.0]
            self.dynamic_mode = True
        else:
            self.epsilons = epsilons
            self.dynamic_mode = False

        self.results = []
        self.all_betti_dims = set()
        self.max_comp_rows = 0
        self._compute()

    def _compute(self):
        """Internal engine to compute invariants across the filtration."""
        from pysurgery.topology.complexes import SimplicialComplex
        
        last_invariants = None
        active_rows = {} # row_idx -> set(vertex_ids)
        merged_rows_history = {} # row_idx -> True (has shown "Merged" message)
        next_available_row_idx = 0

        for eps in self.epsilons:
            # Dispatch based on mode
            if self.mode == "vietoris_rips":
                sc = SimplicialComplex.from_vietoris_rips(self.points, eps, self.max_dimension, coefficient_ring=self.coefficient_ring, backend=self.backend)
            elif self.mode == "alpha":
                sc = SimplicialComplex.from_alpha_complex(self.points, alpha=eps, coefficient_ring=self.coefficient_ring, backend=self.backend)
            elif self.mode == "witness":
                n_landmarks = self.kwargs.get("n_landmarks", min(len(self.points), 50))
                sc = SimplicialComplex.from_witness(self.points, n_landmarks, alpha=eps, max_dimension=self.max_dimension, coefficient_ring=self.coefficient_ring)
            elif self.mode == "cknn":
                k = self.kwargs.get("k", 5)
                sc = SimplicialComplex.from_point_cloud_cknn(self.points, k=k, delta=eps, max_dimension=self.max_dimension, coefficient_ring=self.coefficient_ring, backend=self.backend)
            
            bettis = sc.betti_numbers(backend=self.backend)
            is_manifold, dim, diag = sc.is_homology_manifold(backend=self.backend)
            
            n_defects = len(diag)
            manifold_str = "Yes" if is_manifold else f"No ({n_defects} dft)"
            closed = "N/A"
            if is_manifold:
                closed = "Yes" if sc.is_closed_manifold else "No"
                
            current_invariants = {
                "bettis": bettis,
                "is_manifold_str": manifold_str,
                "is_closed": closed,
                "dimension": dim
            }
            
            comp_info_dict = {}
            if self.track_connected_components:
                # Use the new native .explode() method!
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
                    
                    # Analyze the component subcomplex directly from .explode()
                    c_is_mani, c_dim, c_diag = sub_sc.is_homology_manifold(backend=self.backend)
                    if c_is_mani:
                        c_closed = "Closed" if sub_sc.is_closed_manifold else "Bound"
                        info = f"M(D:{c_dim}, {c_closed})"
                    else:
                        info = f"Non-M ({len(c_diag)} dft)"
                    
                    # Identify the row we just assigned this sub_sc to
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
                "dimension": str(dim) if dim is not None else "N/A",
                "comp_info_map": comp_info_dict.copy() if self.track_connected_components else {}
            })
        
        self.max_comp_rows = next_available_row_idx

        # Stability Calculation
        self.stability_data = []
        for j in range(len(self.results)):
            start_eps = self.results[j]["epsilon"]
            end_eps = self.results[j+1]["epsilon"] if j+1 < len(self.results) else (self.epsilons[-1] if not self.dynamic_mode else start_eps * 1.1)
            dist = end_eps - start_eps
            self.stability_data.append((dist, self.results[j]["bettis"], start_eps, end_eps))
        self.stability_data.sort(key=lambda x: x[0], reverse=True)

    def to_markdown(self) -> str:
        """Return the formatted Markdown report as a string."""
        if not self.results:
            return "No filtration steps recorded."

        sorted_betti_dims = sorted(list(self.all_betti_dims))
        reported_epsilons = [res["epsilon"] for res in self.results]
        eps_label = {"alpha": "Alpha", "cknn": "Delta", "witness": "Alpha"}.get(self.mode, "Eps")

        def format_table_with_dynamic_eps(rows_without_eps_formatted, eps_values):
            # 1. Determine max_w of all existing string cells
            max_w = 0
            for row in rows_without_eps_formatted:
                for cell in row:
                    max_w = max(max_w, len(str(cell)))
            
            # Also consider a minimum width for epsilons (e.g. 10 chars)
            max_w = max(max_w, 10)
            
            # 2. Format epsilons to exactly match max_w
            formatted_eps = []
            for e in eps_values:
                # Total length = len(integer_part) + 1 (dot) + precision
                # So precision = max_w - len(integer_part) - 1
                s_int = str(int(e))
                precision = max_w - len(s_int) - 1
                if precision < 0:
                    formatted_eps.append(str(e)[:max_w].ljust(max_w))
                else:
                    formatted_eps.append(f"{e:.{precision}f}")
            
            # 3. Reconstruct the full rows list
            final_rows = []
            header = [rows_without_eps_formatted[0][0]] + formatted_eps
            final_rows.append(header)
            
            for i, row in enumerate(rows_without_eps_formatted[1:]):
                final_rows.append(row)
                
            return _format_padded_table(final_rows)

        # Betti Numbers Table
        betti_rows_init = [[f"Betti \\ {eps_label}"]]
        for d in sorted_betti_dims:
            betti_rows_init.append([f"b_{d}"] + [str(res["bettis"].get(d, 0)) for res in self.results])
        
        betti_report = f"# Betti Numbers Report (Mode: {self.mode})\n" + format_table_with_dynamic_eps(betti_rows_init, reported_epsilons)
        
        # Stability
        stab_lines = ["\n# Most Persistent Homologies"]
        for i, (dist, b, s, e) in enumerate(self.stability_data[:5]):
            b_str = ", ".join(f"b_{d}={b.get(d, 0)}" for d in sorted_betti_dims if b.get(d, 0) > 0)
            stab_lines.append(f"{i+1}. Range: {dist:.6f} [{s:.6f} -> {e:.6f}] | {b_str}")

        # Manifold Status Table
        mani_rows_init = [[f"Property \\ {eps_label}"]]
        mani_rows_init.append(["Is Manifold"] + [res["is_manifold"] for res in self.results])
        mani_rows_init.append(["Is Closed"] + [res["is_closed"] for res in self.results])
        with_boundary = [("No" if r["is_closed"] == "Yes" else "Yes") if "Yes" in r["is_manifold"] else "N/A" for r in self.results]
        mani_rows_init.append(["With Boundary"] + with_boundary)
        mani_rows_init.append(["Dimension"] + [res["dimension"] for res in self.results])
        
        mani_report = "\n# Manifold Status Report\n" + format_table_with_dynamic_eps(mani_rows_init, reported_epsilons)
        
        # Connected Components Table
        comp_report = ""
        if self.track_connected_components:
            comp_rows_init = [[f"Component \\ {eps_label}"]]
            for i in range(self.max_comp_rows):
                comp_rows_init.append([f"C_{i+1}"] + [res["comp_info_map"].get(i, "-") for res in self.results])
            comp_report = "\n\n# Connected Components Report\n" + format_table_with_dynamic_eps(comp_rows_init, reported_epsilons)
        
        return betti_report + "\n" + "\n".join(stab_lines) + "\n" + mani_report + comp_report

    def plot(self) -> Any:
        """Generate an interactive Plotly visualization of Betti curves."""
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
        eps_label = {"alpha": "Alpha", "cknn": "Delta", "witness": "Alpha"}.get(self.mode, "Epsilon")

        for d in all_dims:
            fig.add_trace(go.Scatter(
                x=steps_eps, 
                y=betti_history[d], 
                mode='lines+markers',
                name=f'b_{d}',
                line=dict(shape='hv')
            ))

        fig.update_layout(
            title=f"Betti Curves Filtration (Mode: {self.mode})",
            xaxis_title=eps_label,
            yaxis_title="Betti Number",
            template="plotly_white",
            hovermode="x unified"
        )
        return fig

    def __str__(self) -> str:
        return self.to_markdown()

    def __repr__(self) -> str:
        return f"<FiltrationReport mode={self.mode} steps={len(self.results)}>"


# --- Convenience Wrappers (Backward Compatibility) ---

def generate_filtration_report(*args, **kwargs) -> str:
    """Legacy wrapper for FiltrationReport(...).to_markdown()"""
    return str(FiltrationReport(*args, **kwargs))

def plot_filtration_report(*args, **kwargs):
    """Legacy wrapper for FiltrationReport(...).plot()"""
    return FiltrationReport(*args, **kwargs).plot()
