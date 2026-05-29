import numpy as np

from pysurgery.topology.complexes import SimplicialComplex
from pysurgery.auto_surgery import (
    _reduce_backtracking,
    _embed_loop,
    compute_pi1_generators_as_cycles,
    detect_components_with_status,
    _snapshot_topology,
    detect_linked_pairs,
    _local_strand_count,
    _find_cut_site,
    _plan_threading_isotopy,
    CutSite,
)

# ── 1. Reduce Backtracking Tests ──────────────────────────────────────────────

def test_reduce_backtracking_basic():
    # Simple backtracking: 0 -> 1 -> 2 -> 1 -> 3
    walk = [(0, 1), (1, 2), (2, 1), (1, 3)]
    reduced = _reduce_backtracking(walk)
    assert reduced == [(0, 1), (1, 3)]

def test_reduce_backtracking_complete():
    # Backtrack all the way to empty: 0 -> 1 -> 0
    walk = [(0, 1), (1, 0)]
    assert _reduce_backtracking(walk) == []

def test_reduce_backtracking_cycle():
    # A true cycle has no backtracking: 0 -> 1 -> 2 -> 0
    walk = [(0, 1), (1, 2), (2, 0)]
    assert _reduce_backtracking(walk) == walk

def test_reduce_backtracking_boundary():
    # Boundary backtracking: 0 -> 1 -> 2 -> 1 -> 0
    walk = [(0, 1), (1, 2), (2, 1), (1, 0)]
    assert _reduce_backtracking(walk) == []


# ── 2. Loop Embedding Tests ───────────────────────────────────────────────────

def test_embed_loop_simple_circle():
    # Circle K: 0-1-2-0
    K = SimplicialComplex.from_simplices(
        [(0, 1), (1, 2), (2, 0)],
        close_under_faces=True,
    )
    # A walk with a repeated vertex: 0 -> 1 -> 2 -> 1 -> 2 -> 0
    # This has a sub-loop 1 -> 2 -> 1 (contractible)
    walk = [(0, 1), (1, 2), (2, 1), (1, 2), (2, 0)]
    
    embedded = _embed_loop(K, walk, None)
    assert embedded is not None
    # Should reduce to the minimal circle walk
    assert set(embedded) == {(0, 1), (1, 2), (2, 0)}


# ── 3. Pi1 Generators as Cycles Tests ─────────────────────────────────────────

def test_compute_pi1_generators_as_cycles_circle():
    K = SimplicialComplex.from_simplices(
        [(0, 1), (1, 2), (2, 0)],
        close_under_faces=True,
    )
    cycles = compute_pi1_generators_as_cycles(K)
    assert len(cycles) == 1
    assert cycles[0].name != ""
    assert set(cycles[0].cycle) == {(0, 1), (1, 2), (2, 0)} or set(cycles[0].cycle) == {(0, 2), (2, 1), (1, 0)}


# ── 4. Component Detection & Status Tests ─────────────────────────────────────

def test_detect_components_disjoint_spheres():
    # Component 0: A circle S¹ on vertices 0, 1, 2
    # Component 1: A sphere S² (tetrahedron boundary) on vertices 10, 11, 12, 13
    s1_simps = [(0, 1), (1, 2), (2, 0)]
    s2_simps = [
        (10, 11, 12), (10, 11, 13), (10, 12, 13), (11, 12, 13)
    ]
    K = SimplicialComplex.from_simplices(
        s1_simps + s2_simps,
        close_under_faces=True,
    )
    
    components = detect_components_with_status(K)
    assert len(components) == 2
    
    # Sort by vertex counts or name to identify which is which
    components.sort(key=lambda c: len(c.vertex_ids))
    
    c_s1 = components[0]
    c_s2 = components[1]
    
    assert c_s1.is_manifold is True
    assert c_s1.is_closed is True
    assert c_s1.betti.get(0, 0) == 1
    assert c_s1.betti.get(1, 0) == 1
    
    assert c_s2.is_manifold is True
    assert c_s2.is_closed is True
    assert c_s2.betti.get(0, 0) == 1
    assert c_s2.betti.get(2, 0) == 1


# ── 5. Snapshot Topology Tests ───────────────────────────────────────────────

def test_snapshot_topology_circle():
    K = SimplicialComplex.from_simplices(
        [(0, 1), (1, 2), (2, 0)],
        close_under_faces=True,
    )
    snap = _snapshot_topology(K)
    assert snap["is_mani"] is True
    assert snap["is_closed"] is True
    assert snap["betti"].get(0) == 1
    assert snap["betti"].get(1) == 1


# ── 6. Linked Pairs Detection Tests ───────────────────────────────────────────

def test_detect_linked_pair_unlinked():
    # Two disjoint circles in 3D (unlinked)
    # Ka: 0-1-2-0
    # Kb: 10-11-12-10
    # ambient 3-simplex: 20-21-22-23
    K = SimplicialComplex.from_simplices(
        [(0, 1), (1, 2), (2, 0),
         (10, 11), (11, 12), (12, 10),
         (20, 21, 22, 23)],
        close_under_faces=True,
    )
    components = detect_components_with_status(K)
    # We filter out only the circles (dim = 1)
    circles = [c for c in components if c.dimension == 1]
    assert len(circles) == 2
    
    linked = detect_linked_pairs(K, circles)
    assert len(linked) == 0


# ── 7. Local Strand Count Tests ───────────────────────────────────────────────

def test_local_strand_count_with_coords():
    # C_a is a circle, C_b is another circle
    C_a = SimplicialComplex.from_simplices([(0, 1), (1, 2), (2, 0)], close_under_faces=True)
    C_b = SimplicialComplex.from_simplices([(3, 4), (4, 5), (5, 3)], close_under_faces=True)
    
    # Global coordinate array indexed by vertex IDs 0-5
    coords = np.array([
        [0.0, 0.0, 0.0],     # vertex 0
        [1.0, 0.0, 0.0],     # vertex 1
        [0.0, 1.0, 0.0],     # vertex 2
        [0.0, 0.0, 0.0],     # vertex 3 (very close to C_a)
        [0.05, 0.0, 0.0],    # vertex 4 (very close to C_a)
        [10.0, 10.0, 10.0]   # vertex 5 (far away)
    ])
    
    # σ is the edge (0, 1) in C_a
    σ = (0, 1)
    
    # With a small tube radius, we should only see 1 connected component of C_b near σ
    strands = _local_strand_count(σ, C_a, C_b, coords, coords, tube_radius=1.0)
    assert strands == 1


# ── 8. Cut Site Selection Tests ───────────────────────────────────────────────

def test_find_cut_site_basic():
    # Two disjoint components, C_a is a circle (vertices 0-2), C_b is a point (vertex 10)
    # K is the union, with 2D simplex to make K 2D
    K = SimplicialComplex.from_simplices(
        [(0, 1), (1, 2), (2, 0), (10,), (20, 21, 22)],
        close_under_faces=True,
    )
    
    coords = np.zeros((23, 3))
    coords[0] = [0.0, 0.0, 0.0]
    coords[1] = [1.0, 0.0, 0.0]
    coords[2] = [0.0, 1.0, 0.0]
    coords[10] = [0.5, 0.5, 0.5] # very close to (0, 1, 2)
    
    K._coordinates = coords
    
    C_a = SimplicialComplex.from_simplices([(0, 1), (1, 2), (2, 0)], close_under_faces=True)
    C_b = SimplicialComplex.from_simplices([(10,)], close_under_faces=True)
    
    cut = _find_cut_site(K, C_a, C_b, coords_a=coords, coords_b=coords)
    assert cut is not None
    assert cut.keeps_component_connected is True
    assert cut.local_strands == 1


# ── 9. Threading Isotopy Planning Tests ────────────────────────────────────────

def test_plan_threading_isotopy_basic():
    coords_b = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0]
    ])
    cut = CutSite(
        simplex=(0, 1),
        centroid=(0.1, 0.1, 0.1),
        score=0.1,
        keeps_component_connected=True,
        local_strands=1,
    )
    
    iso, translation = _plan_threading_isotopy(coords_b, cut, lk_initial=1)
    assert iso is not None
    assert len(translation) == 3
    # Check that at t = 0.5, the translation passes exactly through the via point (cs)
    # Let's test the isotopy on the start point
    moved_p = iso(np.array([iso.start]), 0.5)
    assert np.allclose(moved_p[0], iso.via)
