import numpy as np
import scipy.sparse as sp
from pysurgery.core.complexes import ChainComplex, CWComplex
from pysurgery.core.exceptions import DimensionError

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
        
    # 0-cells: Vertices
    n_vertices = len(mesh.vertices)
    
    # 1-cells: Edges
    # Trimesh provides mesh.edges_unique which are sorted (v1, v2) pairs
    edges = mesh.edges_unique
    n_edges = len(edges)
    edge_to_idx = {tuple(e): i for i, e in enumerate(edges)}
    
    # 2-cells: Faces
    faces = mesh.faces
    n_faces = len(faces)
    
    cells = {0: n_vertices, 1: n_edges, 2: n_faces}
    
    # Boundary 1: d_1 (Edges -> Vertices)
    # For each edge (v1, v2), boundary is v2 - v1
    d1_rows = []
    d1_cols = []
    d1_data = []
    
    for j, (v1, v2) in enumerate(edges):
        d1_rows.extend([v1, v2])
        d1_cols.extend([j, j])
        d1_data.extend([-1, 1])
        
    d1 = sp.csr_matrix((d1_data, (d1_rows, d1_cols)), shape=(n_vertices, n_edges), dtype=np.int64)
    
    # Boundary 2: d_2 (Faces -> Edges)
    # For each face (v1, v2, v3), boundary is (v1,v2) + (v2,v3) - (v1,v3)
    d2_rows = []
    d2_cols = []
    d2_data = []
    
    for j, (v1, v2, v3) in enumerate(faces):
        # We need to find the index of each edge and its orientation
        face_edges = [(v1, v2), (v2, v3), (v1, v3)]
        signs = [1, 1, -1]
        
        for e, sign in zip(face_edges, signs):
            # Check if edge is in unique edges as (v1, v2) or (v2, v1)
            sorted_e = tuple(sorted(e))
            if sorted_e in edge_to_idx:
                idx = edge_to_idx[sorted_e]
                # Adjust sign if the original edge was reversed
                if sorted_e != e:
                    sign *= -1
                
                d2_rows.append(idx)
                d2_cols.append(j)
                d2_data.append(sign)
                
    d2 = sp.csr_matrix((d2_data, (d2_rows, d2_cols)), shape=(n_edges, n_faces), dtype=np.int64)
    
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
