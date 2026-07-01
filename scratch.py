import numpy as np
from pysurgery.topology.complexes import SimplicialComplex
from pysurgery.geometry.embedding import PLMap

sc = SimplicialComplex.from_simplices([(0, 1, 2), (3, 4, 5)])
coords = np.array([
    [0.0, 0, 0], [1, 0, 0], [0, 1, 0],
    [0.25, 0.25, -1], [0.25, 0.25, 1], [0.75, 0.75, 0]
], dtype=np.float64)

pl_map = PLMap.from_source(sc, coords)
print("D:", pl_map.ambient_dimension)

from pysurgery.geometry.embedding import _all_simplices_by_dim
source_simplices = _all_simplices_by_dim(pl_map.source_complex)
simplices = [
    s
    for dim in sorted(source_simplices.keys())
    for s in sorted(source_simplices[dim])
    if len(s) >= 2
]

barycenters = [pl_map.simplex_barycenter(s) for s in simplices]
print("shapes:", [b.shape for b in barycenters])
print("types:", [type(b) for b in barycenters])

try:
    centroids = np.array(barycenters, dtype=np.float64)
    print("centroids shape:", centroids.shape)
except Exception as e:
    print("Error:", e)
