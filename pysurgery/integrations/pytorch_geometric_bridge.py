import numpy as np
import scipy.sparse as sp
from pysurgery.core.complexes import CWComplex
import importlib.util

HAS_TORCH = importlib.util.find_spec("torch") is not None


def pyg_to_cw_complex(data) -> CWComplex:
    """Converts a PyTorch Geometric (PyG) Graph Data object into a CW Complex.

    What is Being Computed?:
        Translates a graph-structured PyG Data object (optionally with faces) 
        into a CWComplex representation. This enables the application of 
        algebraic surgery, homology computation, and persistent cycle analysis 
        directly on Graph Neural Network (GNN) inputs.

    Algorithm:
        1. Extract `edge_index` and node counts from the PyG object.
        2. Symmetrize edges and select a canonical orientation (u < v).
        3. Assemble the d1 boundary matrix mapping edges to their endpoint vertices.
        4. If faces are present:
           a. Map each face to its constituent oriented edges.
           b. Assemble the d2 boundary matrix.
           c. Validate the chain complex condition ∂₁ ∘ ∂₂ = 0.
        5. Initialize and return a CWComplex with these attaching maps.

    Preserved Invariants:
        - Preserves the connectivity (π₀) and cycle structure (H₁) of the graph.
        - Preserves higher-order topology if the PyG object includes face definitions.

    Args:
        data: A PyG `Data` object containing at least an `edge_index` tensor.

    Returns:
        CWComplex: The equivalent cell complex representation.

    Use When:
        - Applying topological denoising to graph datasets.
        - Computing homology groups of meshes represented as PyG objects.
        - Performing surgery to simplify graph topology while preserving invariants.

    Example:
        cw = pyg_to_cw_complex(pyg_data)
        h1 = cw.homology(1)
        print(f"Number of fundamental cycles: {h1.rank}")
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required. Install via 'pip install torch'.")

    if not hasattr(data, "edge_index"):
        raise TypeError("Input must have an 'edge_index' attribute (like PyG Data).")

    n_vertices = data.num_nodes
    edge_index = data.edge_index.cpu().numpy()
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")

    # We treat the graph as an undirected graph, but boundary matrices require orientation.
    # PyG edge_index usually contains both (u, v) and (v, u) for undirected graphs.
    # We will filter to only include edges where u < v to define a canonical orientation.

    mask = edge_index[0, :] < edge_index[1, :]
    unique_edges = edge_index[:, mask]
    n_edges = unique_edges.shape[1]
    edge_to_idx = {}
    for j in range(n_edges):
        u, v = int(unique_edges[0, j]), int(unique_edges[1, j])
        edge_to_idx[(u, v)] = j

    has_faces = hasattr(data, "face") and data.face is not None
    face_obj = data.face if has_faces else None
    n_faces = int(face_obj.shape[1]) if has_faces else 0

    cells = {0: n_vertices, 1: n_edges}
    if n_faces > 0:
        cells[2] = n_faces

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

    d1 = sp.csr_matrix(
        (d1_data, (d1_rows, d1_cols)), shape=(n_vertices, n_edges), dtype=np.int64
    )

    attaching_maps = {1: d1}

    if n_faces > 0:
        face_arr = (
            face_obj.cpu().numpy() if hasattr(face_obj, "cpu") else np.asarray(face_obj)
        )
        if face_arr.ndim != 2 or face_arr.shape[0] < 3:
            raise ValueError("face must have shape [k, F] with k >= 3.")

        d2_rows = []
        d2_cols = []
        d2_data = []

        for j in range(n_faces):
            cycle = [int(v) for v in face_arr[:, j].tolist()]
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                key = (u, v) if u < v else (v, u)
                if key not in edge_to_idx:
                    raise ValueError(
                        f"Face references edge ({u}, {v}) that is missing from edge_index."
                    )
                sign = 1 if key == (u, v) else -1
                d2_rows.append(edge_to_idx[key])
                d2_cols.append(j)
                d2_data.append(sign)

        d2 = sp.csr_matrix(
            (d2_data, (d2_rows, d2_cols)), shape=(n_edges, n_faces), dtype=np.int64
        )
        boundary_check = d1 @ d2
        if boundary_check.nnz > 0 and np.any(boundary_check.data != 0):
            raise ValueError(
                "Invalid face orientation: boundary operator d_1 o d_2 != 0."
            )
        attaching_maps[2] = d2

    return CWComplex(cells=cells, attaching_maps=attaching_maps)
