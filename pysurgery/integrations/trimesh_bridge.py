import numpy as np
import scipy.sparse as sp
import warnings
from pysurgery.core.complexes import CWComplex
from pysurgery.bridge.julia_bridge import julia_engine

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

def trimesh_to_cw_complex(mesh) -> CWComplex:
    """
    Converts a Trimesh object (3D geometric mesh) into a topological CW Complex.
    This extracts the 0-cells (vertices), 1-cells (edges), and 2-cells (faces)
    and constructs the exact boundary operators (attaching maps) over Z.
    
    Uses Julia acceleration for large face meshes, with pure-Python fallback.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        A loaded Trimesh object.
        
    Returns
    -------
    CWComplex
        The abstract topological representation of the mesh.
    """
    if not HAS_TRIMESH:
        raise ImportError("The 'trimesh' library is required. Install via 'pip install trimesh'.")

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("Input must be a trimesh.Trimesh object.")

    if mesh.faces.ndim != 2 or mesh.faces.shape[1] < 3:
        raise ValueError("trimesh_to_cw_complex expects polygonal faces with at least 3 vertices.")
    if len(mesh.faces) == 0:
        raise ValueError("Mesh has no 2-cells (faces); unsupported for current 2-skeleton conversion.")

    n_vertices = len(mesh.vertices)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    n_faces = len(faces)

    # Try Julia acceleration for large meshes
    if julia_engine.available and n_faces > 1000:
        try:
            payload = julia_engine.compute_trimesh_boundary_data([tuple(f) for f in faces], n_vertices)
            d1 = sp.csr_matrix(
                (payload["d1_data"], (payload["d1_rows"], payload["d1_cols"])),
                shape=(payload["n_vertices"], payload["n_edges"]),
                dtype=np.int64,
            )
            d2 = sp.csr_matrix(
                (payload["d2_data"], (payload["d2_rows"], payload["d2_cols"])),
                shape=(payload["n_edges"], payload["n_faces"]),
                dtype=np.int64,
            )
            cells = {0: payload["n_vertices"], 1: payload["n_edges"], 2: payload["n_faces"]}
            attaching_maps = {1: d1, 2: d2}
            return CWComplex(cells=cells, attaching_maps=attaching_maps)
        except Exception as e:
            warnings.warn(
                f"Topological Hint: Julia trimesh boundary assembly failed ({e!r}). "
                "Falling back to pure Python."
            )

    # Python fallback (original implementation)
    # 1-cells: build unique undirected edges from polygon face cycles
    edge_to_idx = {}
    edges = []
    face_boundary_edges = []
    for f in faces:
        cycle = [int(v) for v in f.tolist()]
        cyc_edges = []
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            key = (u, v) if u < v else (v, u)
            if key not in edge_to_idx:
                edge_to_idx[key] = len(edges)
                edges.append(key)
            cyc_edges.append((u, v, edge_to_idx[key]))
        face_boundary_edges.append(cyc_edges)
    n_edges = len(edges)

    cells = {0: n_vertices, 1: n_edges, 2: n_faces}
    
    # Boundary 1: d_1 (Edges -> Vertices)
    d1_rows = []
    d1_cols = []
    d1_data = []
    
    for j, (v1, v2) in enumerate(edges):
        d1_rows.extend([v1, v2])
        d1_cols.extend([j, j])
        d1_data.extend([-1, 1])
        
    d1 = sp.csr_matrix((d1_data, (d1_rows, d1_cols)), shape=(n_vertices, n_edges), dtype=np.int64)
    
    # Boundary 2: d_2 (Faces -> Edges)
    d2_rows = []
    d2_cols = []
    d2_data = []
    for j, edge_cycle in enumerate(face_boundary_edges):
        for u, v, idx in edge_cycle:
            sign = 1 if (u, v) == edges[idx] else -1
            d2_rows.append(idx)
            d2_cols.append(j)
            d2_data.append(sign)

    d2 = sp.csr_matrix((d2_data, (d2_rows, d2_cols)), shape=(n_edges, n_faces), dtype=np.int64)
    
    boundary_check = d1 @ d2
    if boundary_check.nnz > 0 and np.any(boundary_check.data != 0):
        from pysurgery.core.exceptions import DimensionError
        raise DimensionError("Invalid mesh topology: boundary operator d_1 o d_2 != 0. The mesh faces are not consistently oriented or the mesh contains non-manifold geometry.")
    
    attaching_maps = {1: d1, 2: d2}
    
    return CWComplex(cells=cells, attaching_maps=attaching_maps)

def heal_mesh_topology(mesh) -> str:
    """
    Identifies high-genus topology in a mesh (e.g., unintended handles/tunnels)
    by computing H_1(M; Z). Returns a report indicating if surgery is required.
    """
    cw = trimesh_to_cw_complex(mesh)
    chain = cw.cellular_chain_complex()
    
    betti_1, torsion_1 = chain.homology(1)
    
    if betti_1 == 0:
        return "Mesh is topologically simple (H_1 = 0). No handles detected. Healing not required."
    else:
        return f"Mesh has genus > 0 (H_1 rank = {betti_1}). Contains {betti_1} topological handles/tunnels. Surgery recommended to heal to a sphere."
