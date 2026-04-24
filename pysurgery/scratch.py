from pysurgery.topoview import visualize_topoview
from tests.discrete_surface_data import build_torus
import numpy as np

# Create a torus
sc = build_torus()
# Mock coordinates for 9 vertices of build_torus
points = np.array([
    [0, 0, 0], [1, 0, 0], [2, 0, 0],
    [0, 1, 0], [1, 1, 0], [2, 1, 0],
    [0, 2, 0], [1, 2, 0], [2, 2, 0]
], dtype=float)

# Test with features
print("Testing visualize_topoview with curvature and dual graph...")
fig = visualize_topoview(sc, dimension=3, points=points, 
                        features=["curvature", "dual"], show=False)
print(f"Figure with features created successfully. Layout height: {fig.layout.height}")

# Test Stiefel-Whitney (on a torus it might be trivial but let's see if it runs)
print("Testing with w2 feature...")
fig_w2 = visualize_topoview(sc, dimension=3, points=points, 
                           features=["w2"], show=False)
print("Figure with w2 created successfully.")
