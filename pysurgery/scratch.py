import pysurgery as ps
import numpy as np

# 1. Create a 500-point sphere
n_points = 500
pts = np.random.normal(size=(n_points, 3))
pts /= np.linalg.norm(pts, axis=1, keepdims=True)

sc = ps.SimplicialComplex.from_point_cloud_cknn(pts, k=5, delta=1.0, max_dimension=2)
print(f"Original vertices: {sc.count_simplices(0)}")

# 2. Run simplify to preserve topology
sc_qm, s_map = sc.simplify()
print(f"Reduced vertices: {sc_qm.count_simplices(0)}")

# 3. Verify mappings
print(f"Simplex map size: {len(s_map)}")

# Check that total vertices in s_map equals original
total_orig_v = sum(len(vs) for vs in s_map.values())
assert total_orig_v == sc.count_simplices(0)
print(f"Verified: All {total_orig_v} original vertices are preserved in the map.")

# Verify reduction
if sc_qm.count_simplices(0) < sc.count_simplices(0):
    print("Success: Quick Mapper reduced vertices aggressively!")
