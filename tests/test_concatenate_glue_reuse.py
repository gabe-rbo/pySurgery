"""Cache reuse on disjoint union (`concatenate`) and the quotient `glue`.

`concatenate` (disjoint union) is cache-safe under the uniform vertex shift it
applies: it carries each input's connected components — with their
homology/cohomology (rank,torsion) results and re-indexed π₁ generator cycles —
into the result, so the result's invariants are read back without re-running
Smith Normal Form (SNF) or the π₁ extraction. `glue` (adjunction along an
identification) changes topology, so it is computed fresh.

All complexes here are intentionally tiny (triangles, arcs) to stay well within
memory bounds.
"""

import warnings

import numpy as np
import pytest

import pysurgery.topology.complexes as cx
import pysurgery.topology.fundamental_group as fg
from pysurgery.topology.complexes import SimplicialComplex as SC
from pysurgery.auto_surgery import compute_pi1_generators_as_cycles as direct_cycles


# ── instrumentation ────────────────────────────────────────────────────────


@pytest.fixture
def snf_counter(monkeypatch):
    """Count calls to the sparse SNF kernel that backs homology/cohomology."""
    calls = {"n": 0}
    orig = cx.get_sparse_snf_diagonal

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(cx, "get_sparse_snf_diagonal", counting)
    return calls


@pytest.fixture
def pi1_counter(monkeypatch):
    """Count calls to the heavy π₁ presentation extraction."""
    calls = {"n": 0}
    orig = fg.extract_pi_1_with_traces

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    # auto_surgery imports the symbol lazily from the module, so patching the
    # module attribute is enough for the counter to be observed.
    monkeypatch.setattr(fg, "extract_pi_1_with_traces", counting)
    return calls


# ── helpers ────────────────────────────────────────────────────────────────


def _circle(base=0):
    """Hollow triangle on vertices base..base+2 (a 1-cycle)."""
    return SC.from_simplices(
        [(base + 0, base + 1), (base + 1, base + 2), (base + 0, base + 2)],
        close_under_faces=True,
    )


def _solid_triangle(base=0):
    return SC.from_simplices([(base, base + 1, base + 2)], close_under_faces=True)


def _betti(K):
    return {d: r for d, (r, _t) in K.homology().items()}


def _cyc_sig(cycles):
    return sorted(
        (
            g.name,
            tuple(sorted(tuple(sorted(e)) for e in g.cycle)),
            g.component_root,
            g.orientation_character,
        )
        for g in cycles
    )


# ── concatenate: homology / cohomology reuse ────────────────────────────────


def test_concatenate_reuses_homology_no_snf(snf_counter):
    A, B = _circle(0), _circle(0)
    A.homology()
    B.homology()
    U = SC.concatenate([A, B])

    before = snf_counter["n"]
    hU = U.homology()
    during = snf_counter["n"] - before

    fresh = SC.from_simplices(
        [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)], close_under_faces=True
    )
    fresh.clear_cache()
    assert hU == fresh.homology()
    assert during == 0  # reused the re-indexed component caches


def test_concatenate_reuses_cohomology_no_snf(snf_counter):
    A, B = _circle(0), _circle(0)
    A.cohomology()
    B.cohomology()
    U = SC.concatenate([A, B])

    before = snf_counter["n"]
    cU = U.cohomology()
    during = snf_counter["n"] - before

    assert cU == {0: (2, []), 1: (2, [])}
    assert during == 0


def test_concatenate_per_degree_homology_is_free(snf_counter):
    A, B = _circle(0), _circle(0)
    A.homology()
    B.homology()
    U = SC.concatenate([A, B])

    before = snf_counter["n"]
    h1 = U.homology(1)
    during = snf_counter["n"] - before
    assert h1 == (2, [])
    assert during == 0


def test_concatenate_single_connected_input_reuses(snf_counter):
    A = _circle(0)
    A.homology()
    U = SC.concatenate([A])
    assert U.num_connected_components() == 1

    before = snf_counter["n"]
    hU = U.homology()
    during = snf_counter["n"] - before
    assert hU == {0: (1, []), 1: (1, [])}
    assert during == 0


def test_concatenate_seeds_components_matching_fresh():
    A, B = _circle(0), _circle(10)  # disjoint label ranges
    U = SC.concatenate([A, B])
    # Seeded component vertex sets must match a from-scratch decomposition.
    fresh = SC.from_simplices(list(U.simplices), close_under_faces=True)
    fresh.clear_cache()
    seeded = [set(vs) for vs in U._component_vertex_sets()]
    recomputed = [set(vs) for vs in fresh._component_vertex_sets()]
    assert seeded == recomputed


# ── concatenate: π₁ generator-cycle reuse ───────────────────────────────────


def test_concatenate_reuses_pi1_cycles_no_extraction(pi1_counter):
    A, B = _circle(0), _circle(0)
    A.pi1_generator_cycles()
    B.pi1_generator_cycles()
    U = SC.concatenate([A, B])

    before = pi1_counter["n"]
    uc = U.pi1_generator_cycles()
    during = pi1_counter["n"] - before

    fresh = SC.from_simplices(
        [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)], close_under_faces=True
    )
    fresh.clear_cache()
    assert _cyc_sig(uc) == _cyc_sig(fresh.pi1_generator_cycles())
    assert during == 0


def test_concatenate_pi1_roots_are_shifted():
    A, B = _circle(0), _circle(0)
    A.pi1_generator_cycles()
    B.pi1_generator_cycles()
    U = SC.concatenate([A, B])
    uc = U.pi1_generator_cycles()
    assert sorted(g.component_root for g in uc) == [0, 3]
    # every cycle edge lives in the right component of U
    edges = {tuple(sorted(e)) for e in U.n_simplices(1)}
    for g in uc:
        for e in g.cycle:
            assert tuple(sorted(e)) in edges
    # names de-duplicated across components
    assert len({g.name for g in uc}) == len(uc)


# ── π₁ generator cycles: decomposition correctness ──────────────────────────


def test_pi1_cycles_connected_is_byte_identical():
    K = _circle(0)
    assert _cyc_sig(K.pi1_generator_cycles()) == _cyc_sig(direct_cycles(K))


def test_pi1_cycles_cached(pi1_counter):
    K = _circle(0)
    first = K.pi1_generator_cycles()
    after_first = pi1_counter["n"]
    second = K.pi1_generator_cycles()
    assert pi1_counter["n"] == after_first  # no recompute
    assert _cyc_sig(first) == _cyc_sig(second)


def test_pi1_cycles_disconnected_union_handles_shifted_component():
    # The high component {3,4,5} would crash if computed in isolation without
    # the local↔global remap; the decomposition must handle it.
    U = SC.from_simplices(
        [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)], close_under_faces=True
    )
    assert U.num_connected_components() == 2
    uc = U.pi1_generator_cycles()
    assert len(uc) == 2
    edges = {tuple(sorted(e)) for e in U.n_simplices(1)}
    for g in uc:
        for e in g.cycle:
            assert tuple(sorted(e)) in edges
    assert len({g.name for g in uc}) == 2


# ── glue: quotient topology ─────────────────────────────────────────────────


def test_glue_two_triangles_along_edge_is_a_disk():
    T1, T2 = _solid_triangle(0), _solid_triangle(0)
    glued = T1.glue(T2, identify=[(1, 0), (2, 1)])  # T1 edge (1,2) ~ T2 edge (0,1)
    assert glued.num_connected_components() == 1
    b = _betti(glued)
    assert b.get(0) == 1 and b.get(1, 0) == 0 and b.get(2, 0) == 0


def test_glue_accepts_simplex_pairs():
    T1, T2 = _solid_triangle(0), _solid_triangle(0)
    glued = T1.glue(T2, identify=[((1, 2), (0, 1))])
    b = _betti(glued)
    assert b.get(0) == 1 and b.get(1, 0) == 0


def test_glue_two_arcs_into_a_circle():
    A = SC.from_simplices([(0, 1), (1, 2)], close_under_faces=True)
    B = SC.from_simplices([(0, 1), (1, 2)], close_under_faces=True)
    circ = A.glue(B, identify=[(0, 0), (2, 2)])
    assert circ.num_connected_components() == 1
    b = _betti(circ)
    assert b.get(0) == 1 and b.get(1) == 1


def test_glue_share_points_exact_match():
    P1 = _solid_triangle(0)
    P1._coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    P1._generate_point_cloud_mappings(P1._coordinates)
    P2 = _solid_triangle(0)
    P2._coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])  # shares 2 pts
    P2._generate_point_cloud_mappings(P2._coordinates)

    glued = P1.glue(P2, share_points=True)
    assert len(glued._coordinates) == 4  # 3 + 3 - 2 shared
    assert glued.num_connected_components() == 1
    assert _betti(glued).get(1, 0) == 0
    assert len(glued.point_cloud_to_simplices) == 4


def test_glue_without_identification_is_disjoint():
    A = SC.from_simplices([(0, 1), (1, 2)], close_under_faces=True)
    B = SC.from_simplices([(0, 1), (1, 2)], close_under_faces=True)
    g = A.glue(B)
    assert g.num_connected_components() == 2


def test_glue_does_not_reuse_caches():
    # Gluing changes topology; the result must not inherit input invariants.
    A, B = _solid_triangle(0), _solid_triangle(0)
    A.homology()
    B.homology()
    glued = A.glue(B, identify=[(1, 0), (2, 1)])
    # Fresh, correct invariant for the disk (not a stale copy of a triangle).
    assert _betti(glued) == {d: r for d, (r, _t) in glued.homology().items()}
    assert glued.num_connected_components() == 1


def test_glue_errors_on_bad_identify():
    T1, T2 = _solid_triangle(0), _solid_triangle(0)
    with pytest.raises(ValueError):
        T1.glue(T2, identify=[((0, 1), (0, 1, 2))])  # length mismatch
    with pytest.raises(TypeError):
        T1.glue(T2, identify=[(0, (1, 2))])  # mixed shapes
    with pytest.raises(ValueError):
        T1.glue(T2, share_points=True)  # no coordinates
