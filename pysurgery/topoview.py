import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Tuple, Dict, Any, Literal, Union
import warnings
import math

try:
    import umap
except ImportError:
    umap = None

from .core import (
    SimplicialComplex,
    compute_homology_basis_from_complex,
    extract_pi_1_with_traces,
    HomologyGenerator,
    HomologyBasisResult,
    Pi1GeneratorTrace,
    Pi1PresentationWithTraces,
    vertex_gaussian_curvature,
    detect_self_intersections,
    PLMap,
    extract_stiefel_whitney_w2,
)


def _get_coordinates(
    sc: SimplicialComplex, 
    dimension: int = 3, 
    points: Optional[np.ndarray] = None
) -> np.ndarray:
    """Obtain or compute coordinates for the simplicial complex vertices."""
    # Collect all vertex indices present in the complex
    vertices = sorted([v[0] for v in sc.n_simplices(0)])
    if not vertices:
        return np.empty((0, dimension))
    
    max_v = max(vertices)
    n_vertices = max_v + 1

    if points is not None:
        if points.shape[0] < n_vertices:
            # Pad points if they don't cover all vertices
            padded = np.zeros((n_vertices, points.shape[1]))
            padded[:points.shape[0]] = points
            points = padded
            
        if points.shape[1] == dimension:
            return points
        
        if umap is not None:
            reducer = umap.UMAP(n_components=dimension, random_state=42)
            return reducer.fit_transform(points)
        
        from sklearn.decomposition import PCA
        return PCA(n_components=dimension).fit_transform(points)

    # Fallback to Graph Layout if no points provided
    try:
        import networkx as nx
        edges = [tuple(e) for e in sc.n_simplices(1)]
        G = nx.Graph()
        G.add_nodes_from(range(n_vertices))
        G.add_edges_from(edges)
        
        pos = nx.spring_layout(G, dim=dimension, seed=42)
        coords = np.zeros((n_vertices, dimension))
        for i in range(n_vertices):
            coords[i] = pos[i]
        return coords
    except ImportError:
        # Final fallback: random coordinates
        np.random.seed(42)
        return np.random.rand(n_vertices, dimension)


def visualize_topoview(
    sc: SimplicialComplex,
    dimension: Literal[2, 3] = 3,
    points: Optional[np.ndarray] = None,
    h0_basis: Optional[HomologyBasisResult] = None,
    h1_basis: Optional[HomologyBasisResult] = None,
    h2_basis: Optional[HomologyBasisResult] = None,
    pi1_result: Optional[Pi1PresentationWithTraces] = None,
    title: str = "Topological Invariants Workspace",
    show: bool = True,
    features: Optional[List[str]] = None,
    max_individual_plots: Optional[int] = None,
) -> List[go.Figure]:
    """
    Create an interactive visual workspace for SimplicialComplex invariants.
    
    Layout (Web-page style):
    - Generates a list of figures, one for each invariant/generator.
    - Each Figure contains:
        - Left: Metadata Table.
        - Right: Geometric UMAP Plot.
    - Background is white.
    """
    if features is None:
        features = []

    # 1. Prepare Coordinates (UMAP if needed)
    coords = _get_coordinates(sc, dimension=dimension, points=points)
    
    # 2. Collect all items to visualize
    items: List[Dict[str, Any]] = []
    
    if pi1_result:
        for i, tr in enumerate(pi1_result.traces):
            items.append({"type": "pi1", "id": i, "data": tr, "title": f"Fundamental Group Generator: {tr.generator}"})
            
    if h1_basis:
        for i, gen in enumerate(h1_basis.generators):
            items.append({"type": "H1", "id": i, "data": gen, "title": f"H1 Homology Generator {i}"})
    
    if h2_basis:
        for i, gen in enumerate(h2_basis.generators):
            items.append({"type": "H2", "id": i, "data": gen, "title": f"H2 Homology Generator {i}"})

    if h0_basis:
        for i, gen in enumerate(h0_basis.generators):
            items.append({"type": "H0", "id": i, "data": gen, "title": f"Connected Component (H0) {i}"})

    if max_individual_plots is not None:
        items = items[:max_individual_plots]

    figures = []

    # Helper for adding base skeleton
    def add_base(fig, row, col):
        faces = [s for s in sc.n_simplices(2) if len(s) == 3]
        if dimension == 3:
            fig.add_trace(go.Mesh3d(
                x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                i=[f[0] for f in faces], j=[f[1] for f in faces], k=[f[2] for f in faces],
                opacity=0.05, color="gray", name="Base Surface", showlegend=False
            ), row=row, col=col)
        else:
            edges = [tuple(e) for e in sc.n_simplices(1)]
            xs, ys = [], []
            for u, v in edges:
                xs.extend([coords[u, 0], coords[v, 0], None])
                ys.extend([coords[u, 1], coords[v, 1], None])
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color="rgba(200, 200, 200, 0.3)", width=1),
                showlegend=False
            ), row=row, col=col)

    # 3. Create one figure per item
    for item in items:
        g_type = item["type"]
        g_data = item["data"]
        
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.3, 0.7],
            specs=[[{"type": "table"}, {"type": "scene" if dimension == 3 else "xy"}]]
        )
        
        # --- Metadata Table ---
        header = ["Property", "Value"]
        cells = []
        if g_type == "pi1":
            cells = [
                ["Type", "pi1 Generator"],
                ["Word", g_data.generator],
                ["Path Length", str(len(g_data.vertex_path))],
            ]
        else:
            cells = [
                ["Type", f"Homology ({g_type})"],
                ["Weight", f"{g_data.weight:.4f}"],
                ["Simplices", str(len(g_data.support_simplices))],
            ]
        if points is not None:
            cells.append(["Ambient Dim", str(points.shape[1])])
        
        fig.add_trace(go.Table(
            header=dict(values=header, fill_color='royalblue', font=dict(color='white'), align='left'),
            cells=dict(values=list(zip(*cells)), fill_color='aliceblue', align='left')
        ), row=1, col=1)

        # --- Geometric Plot ---
        add_base(fig, 1, 2)
        
        if g_type == "pi1":
            path = g_data.vertex_path
            xs, ys, zs = [], [], []
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                xs.extend([coords[u, 0], coords[v, 0], None]); ys.extend([coords[u, 1], coords[v, 1], None])
                if dimension == 3: zs.extend([coords[u, 2], coords[v, 2], None])
            
            trace = go.Scatter3d(x=xs, y=ys, z=zs, mode="lines+markers", line=dict(color="magenta", width=6), name=item["title"]) if dimension == 3 else \
                    go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color="magenta", width=4), name=item["title"])
            fig.add_trace(trace, row=1, col=2)

        elif g_type == "H1":
            edges = g_data.support_edges
            xs, ys, zs = [], [], []
            for u, v in edges:
                xs.extend([coords[u, 0], coords[v, 0], None]); ys.extend([coords[u, 1], coords[v, 1], None])
                if dimension == 3: zs.extend([coords[u, 2], coords[v, 2], None])
            
            trace = go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="red", width=8), name=item["title"]) if dimension == 3 else \
                    go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="red", width=5), name=item["title"])
            fig.add_trace(trace, row=1, col=2)

        elif g_type == "H2":
            faces = [s for s in g_data.support_simplices if len(s) == 3]
            if dimension == 3:
                fig.add_trace(go.Mesh3d(
                    x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                    i=[f[0] for f in faces], j=[f[1] for f in faces], k=[f[2] for f in faces],
                    color="gold", opacity=0.9, name=item["title"]
                ), row=1, col=2)
            else:
                pts_idx = list(set([v for f in faces for v in f]))
                fig.add_trace(go.Scatter(x=coords[pts_idx, 0], y=coords[pts_idx, 1], mode="markers", marker=dict(color="gold", size=10), name=item["title"]), row=1, col=2)

        elif g_type == "H0":
            pts_idx = [s[0] for s in g_data.support_simplices if len(s) == 1]
            trace = go.Scatter3d(x=coords[pts_idx, 0], y=coords[pts_idx, 1], z=coords[pts_idx, 2], mode="markers", marker=dict(size=14, color="black"), name=item["title"]) if dimension == 3 else \
                    go.Scatter(x=coords[pts_idx, 0], y=coords[pts_idx, 1], mode="markers", marker=dict(size=14, color="black"), name=item["title"])
            fig.add_trace(trace, row=1, col=2)

        fig.update_layout(
            title=f"<b>{item['title']}</b>",
            template="plotly_white",
            height=500,
            margin=dict(t=50, b=20, l=20, r=20)
        )
        figures.append(fig)

    # 4. Final Overview Figure
    ov_fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.3, 0.7],
        specs=[[{"type": "table"}, {"type": "scene" if dimension == 3 else "xy"}]]
    )
    add_base(ov_fig, 1, 2)
    ov_cells = [
        ["Total Vertices", str(len(sc.n_simplices(0)))],
        ["Total Edges", str(len(sc.n_simplices(1)))],
        ["Total Faces", str(len(sc.n_simplices(2)))],
        ["Ambient Dim", str(points.shape[1] if points is not None else "N/A")],
    ]
    ov_fig.add_trace(go.Table(
        header=dict(values=["Overview", "Value"], fill_color='darkgray', font=dict(color='white'), align='left'),
        cells=dict(values=list(zip(*ov_cells)), fill_color='whitesmoke', align='left')
    ), row=1, col=1)
    ov_fig.update_layout(title="<b>Complex Overview</b>", template="plotly_white", height=400)
    figures.append(ov_fig)

    if show:
        for f in figures:
            f.show()
            
    return figures
