from pysurgery.core.homology_generators import _components_h0_generators

edges = [(0, 1), (2, 3)]
res = _components_h0_generators(edges, 4)
print(res.rank)
print(res.generators)
