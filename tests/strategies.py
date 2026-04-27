import numpy as np
from hypothesis import strategies as st
import itertools

@st.composite
def point_clouds(draw, min_pts=4, max_pts=20, dim=3):
    """Strategy to generate random point clouds as (N, D) numpy arrays."""
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
    """Strategy to generate a connected simplicial complex."""
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
    """Strategy to generate a list of simplices that form a valid complex."""
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
