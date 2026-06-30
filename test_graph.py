from pysurgery.topology.graphs import Graph

# Multigraph with 2 vertices, 3 parallel edges
g = Graph.from_edges([(0, 1), (0, 1), (0, 1)], num_vertices=2)

print("Cyclomatic number:", g.cyclomatic_number)
print("Simplicial b1:", g.betti_number(1))
print("Laplacian (0-th hodge):")
print(g.laplacian().toarray())

# Inherited hodge_laplacian(1)
try:
    print("Inherited L1:")
    print(g.hodge_laplacian(1).toarray())
except Exception as e:
    print("Error:", e)

