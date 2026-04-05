import numpy as np
import scipy.sparse as sp
from pysurgery.core.complexes import ChainComplex, CWComplex

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def pyg_to_cw_complex(data) -> CWComplex:
    """
    Converts a PyTorch Geometric (PyG) Graph Data object into a 1-dimensional CW Complex.
    This enables topological surgery analysis on graph data structures, computing 
    homology and identifying fundamental cycles for graph simplification.
    
    Parameters
    ----------
    data : torch_geometric.data.Data
        A PyG Data object.
        
    Returns
    -------
    CWComplex
        The 1-dimensional CW complex representation.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required. Install via 'pip install torch'.")

    if not hasattr(data, 'edge_index'):
        raise TypeError("Input must have an 'edge_index' attribute (like PyG Data).")

    n_vertices = data.num_nodes
    edge_index = data.edge_index.cpu().numpy()
    
    # We treat the graph as an undirected graph, but boundary matrices require orientation.
    # PyG edge_index usually contains both (u, v) and (v, u) for undirected graphs.
    # We will filter to only include edges where u < v to define a canonical orientation.
    
    mask = edge_index[0, :] < edge_index[1, :]
    unique_edges = edge_index[:, mask]
    n_edges = unique_edges.shape[1]
    
    cells = {0: n_vertices, 1: n_edges}
    
    # Boundary 1: d_1 (Edges -> Vertices)
    d1_rows = []
    d1_cols = []
    d1_data = []
    
    for j in range(n_edges):
        u, v = unique_edges[0, j], unique_edges[1, j]
        # Boundary of oriented edge (u, v) is v - u
        d1_rows.extend([u, v])
        d1_cols.extend([j, j])
        d1_data.extend([-1, 1])
        
    d1 = sp.csr_matrix((d1_data, (d1_rows, d1_cols)), shape=(n_vertices, n_edges), dtype=np.int64)
    
    attaching_maps = {1: d1}
    
    return CWComplex(cells=cells, attaching_maps=attaching_maps)
