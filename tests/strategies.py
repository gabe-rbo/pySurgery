import numpy as np
from hypothesis import strategies as st

@st.composite
def point_clouds(draw, min_pts=4, max_pts=20, dim=3):
    """Strategy to generate random point clouds as (N, D) numpy arrays.

    What is Being Computed?:
        Generates a random point cloud in D-dimensional Euclidean space for property-based testing.

    Algorithm:
        1. Draw number of points N from [min_pts, max_pts]
        2. Draw dimension D from [2, dim]
        3. Draw N unique D-tuples of floats in range [-100, 100]
        4. Convert to (N, D) numpy array

    Preserved Invariants:
        - Points are unique to avoid immediate Delaunay/triangulation degeneracies.
        - Precision is maintained at float64.

    Args:
        draw: Hypothesis draw function.
        min_pts (int): Minimum number of points in the cloud.
        max_pts (int): Maximum number of points in the cloud.
        dim (int): Maximum dimension of the ambient space.

    Returns:
        np.ndarray: A (N, D) numpy array representing the point cloud.

    Use When:
        - Fuzzing algorithms that take raw coordinate data (e.g., Witness complexes, Rips).
        - Testing geometric robustness of intrinsic dimension estimators.

    Example:
        @given(point_clouds(min_pts=10, dim=3))
        def test_point_algorithm(cloud):
            assert cloud.shape[1] <= 3
    """
    n_pts = draw(st.integers(min_value=min_pts, max_value=max_pts))
    d = draw(st.integers(min_value=2, max_value=dim))
    # Using a larger range and unique points to avoid Delaunay issues
    points = draw(st.lists(
        st.tuples(*(st.floats(min_value=-100.0, max_value=100.0) for _ in range(d))),
        min_size=n_pts, max_size=n_pts, unique=True
    ))
    return np.array(points, dtype=np.float64)

@st.composite
def connected_simplicial_complexes_raw(draw, min_vertices=4, max_vertices=12):
    """Strategy to generate a connected simplicial complex as a list of simplices.

    What is Being Computed?:
        Generates a combinatorial simplicial complex that is guaranteed to be path-connected.

    Algorithm:
        1. Generate N vertices.
        2. Create a spanning tree to ensure 1-skeleton connectivity.
        3. Randomly add higher-dimensional simplices (up to 3-simplices).
        4. Take the skeletal closure (add all faces of every simplex).

    Preserved Invariants:
        - Path-connectivity (H_0 rank is always 1).
        - Skeletal closure (every face of every simplex is present).

    Args:
        draw: Hypothesis draw function.
        min_vertices (int): Minimum number of vertices.
        max_vertices (int): Maximum number of vertices.

    Returns:
        list[tuple]: A list of tuples, each representing a simplex in the complex.

    Use When:
        - Testing algorithms that assume a connected input (e.g., fundamental group extraction).
        - Verifying homology computations where H_0 is known.

    Example:
        @given(connected_simplicial_complexes_raw())
        def test_is_connected(simplices):
            sc = SimplicialComplex.from_simplices(simplices)
            assert sc.betti_numbers()[0] == 1
    """
    n_v = draw(st.integers(min_value=min_vertices, max_value=max_vertices))
    vertices = list(range(n_v))
    
    # Ensure connectivity with a spanning tree
    simplices = set((v,) for v in vertices)
    for i in range(1, n_v):
        parent = draw(st.integers(min_value=0, max_value=i-1))
        simplices.add(tuple(sorted((parent, i))))
    
    # Add some random higher dimensional simplices
    n_extra = draw(st.integers(min_value=0, max_value=n_v))
    for _ in range(n_extra):
        size = draw(st.integers(min_value=2, max_value=min(n_v, 4)))
        s = draw(st.sets(st.sampled_from(vertices), min_size=size, max_size=size))
        if s:
            s_tuple = tuple(sorted(list(s)))
            # Add all faces
            for i in range(1, 1 << len(s_tuple)):
                face = tuple(sorted(s_tuple[j] for j in range(len(s_tuple)) if (i >> j) & 1))
                simplices.add(face)
    
    return list(simplices)

@st.composite
def simplicial_complexes_raw(draw, min_vertices=4, max_vertices=15):
    """Strategy to generate a list of simplices that form a valid (potentially disconnected) complex.

    What is Being Computed?:
        Generates a general combinatorial simplicial complex by closing random simplices under faces.

    Algorithm:
        1. Generate N vertices.
        2. Pick random "maximal" simplices (dimension up to 3).
        3. Compute the downward closure (all faces) for each selected simplex.
        4. Return the union of all faces.

    Preserved Invariants:
        - Skeletal closure: The result is always a valid simplicial complex.

    Args:
        draw: Hypothesis draw function.
        min_vertices (int): Minimum number of vertices.
        max_vertices (int): Maximum number of vertices.

    Returns:
        list[tuple]: A list of simplices as vertex tuples.

    Use When:
        - Testing general-purpose homology/cohomology algorithms.
        - Verifying complex construction and simplification logic.

    Example:
        @given(simplicial_complexes_raw())
        def test_complex_validity(simplices):
            sc = SimplicialComplex.from_simplices(simplices)
            assert sc.num_simplices() >= len(simplices)
    """
    n_v = draw(st.integers(min_value=min_vertices, max_value=max_vertices))
    vertices = list(range(n_v))
    
    # Randomly pick some "maximal" simplices
    n_max = draw(st.integers(min_value=1, max_value=n_v))
    max_simplices = []
    for _ in range(n_max):
        size = draw(st.integers(min_value=1, max_value=min(n_v, 4))) # Keep dim <= 3 for speed
        if size > 0:
            s = draw(st.sets(st.sampled_from(vertices), min_size=size, max_size=size))
            if s:
                max_simplices.append(tuple(sorted(list(s))))
    
    # Close under faces
    all_simplices = set()
    for s in max_simplices:
        for i in range(1, 1 << len(s)):
            face = tuple(sorted(s[j] for j in range(len(s)) if (i >> j) & 1))
            all_simplices.add(face)
            
    return list(all_simplices)
