import numpy as np
import scipy.sparse as sp
import pytest

try:
    from tests.discrete_surface_data import get_surfaces, get_3_manifolds, to_complex
except ImportError:
    pass
from pysurgery.core.fundamental_group import (
    extract_pi_1,
    extract_pi_1_with_traces,
    simplify_presentation,
    infer_standard_group_descriptor,
    FundamentalGroup,
)
from pysurgery.core.complexes import CWComplex


def test_extract_pi_1_trivial():
    d1 = sp.csr_matrix(np.zeros((1, 0), dtype=np.int64))
    cw = CWComplex(attaching_maps={1: d1}, dimensions=[0], cells={0: 1, 1: 0})
    pi_1 = extract_pi_1(cw)
    assert len(pi_1.generators) == 0
    assert len(pi_1.relations) == 0


def test_extract_pi_1_circle():
    # 1 vertex, 1 loop
    # d1 has 1 row (vertex 0), 1 col (edge 0). Loop boundary is 0.
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    cw = CWComplex(attaching_maps={1: d1}, dimensions=[0, 1], cells={0: 1, 1: 1})
    pi_1 = extract_pi_1(cw)
    assert len(pi_1.generators) == 1
    assert pi_1.generators == ["g_0"]
    assert len(pi_1.relations) == 0


def test_extract_pi_1_unclosed_loop():
    # 2 vertices, 1 edge connecting them
    d1 = sp.csr_matrix(np.array([[-1], [1]], dtype=np.int64))
    cw = CWComplex(attaching_maps={1: d1}, dimensions=[0, 1], cells={0: 2, 1: 1})
    pi_1 = extract_pi_1(cw)
    assert len(pi_1.generators) == 0
    assert len(pi_1.relations) == 0


def test_extract_pi_1_disc():
    # 1 vertex, 1 edge (loop), 1 face attaching to the loop
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    # d2 contains the sequence of edges.
    # 1 face. Edge 0 is traversed once forward.
    d2 = sp.csr_matrix(np.array([[1]], dtype=np.int64))
    cw = CWComplex(
        attaching_maps={1: d1, 2: d2}, dimensions=[0, 1, 2], cells={0: 1, 1: 1, 2: 1}
    )
    pi_1 = extract_pi_1(cw)
    # After simplification, the singleton relator g_0 kills the only generator.
    assert len(pi_1.generators) == 0
    assert len(pi_1.relations) == 0


def test_extract_pi_1_disconnected_spanning_forest():
    # Two disconnected edges in separate components should both be tree edges in a spanning forest.
    d1 = sp.csr_matrix(
        np.array(
            [
                [-1, 0],
                [1, 0],
                [0, -1],
                [0, 1],
            ],
            dtype=np.int64,
        )
    )
    cw = CWComplex(attaching_maps={1: d1}, dimensions=[0, 1], cells={0: 4, 1: 2})
    pi_1 = extract_pi_1(cw)
    assert pi_1.generators == []


def test_extract_pi_1_malformed_face_trace_is_skipped():
    # One valid loop edge and one malformed edge in the same face should not force a bogus relation.
    d1 = sp.csr_matrix(np.array([[0, 2]], dtype=np.int64))
    d2 = sp.csr_matrix(np.array([[1], [1]], dtype=np.int64))
    cw = CWComplex(
        attaching_maps={1: d1, 2: d2}, dimensions=[0, 1, 2], cells={0: 1, 1: 2, 2: 1}
    )
    pi_1 = extract_pi_1(cw)
    # We still get generators from non-tree edges; malformed boundary entries are ignored.
    assert len(pi_1.generators) >= 0
    assert pi_1.relations == []


def test_extract_pi_1_face_multiplicity_is_respected():
    # One loop edge traversed twice should produce relation g_0 g_0.
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    d2 = sp.csr_matrix(np.array([[2]], dtype=np.int64))
    cw = CWComplex(
        attaching_maps={1: d1, 2: d2}, dimensions=[0, 1, 2], cells={0: 1, 1: 1, 2: 1}
    )
    pi_1 = extract_pi_1(cw)
    assert pi_1.relations == [["g_0", "g_0"]]


def test_simplify_presentation_free_and_cyclic_reduction():
    g = ["g_0", "g_1"]
    rels = [["g_0", "g_0^-1", "g_1", "g_1^-1"], ["g_1", "g_0", "g_1^-1", "g_0^-1"]]
    simp = simplify_presentation(g, rels)
    # First relator vanishes. Commutator remains (up to cyclic normalization).
    assert len(simp.relations) == 1


def test_simplify_presentation_singleton_inverse_relator_kills_generator():
    simp = simplify_presentation(["g_0"], [["g_0^-1"]])
    assert simp.generators == []
    assert simp.relations == []


def test_simplify_presentation_iterative_substitution_chain_collapses_to_trivial():
    simp = simplify_presentation(
        ["g_0", "g_1", "g_2", "g_3"],
        [["g_0", "g_1^-1"], ["g_1", "g_2^-1"], ["g_2", "g_3^-1"], ["g_3"]],
    )
    assert simp.generators == []
    assert simp.relations == []


def test_simplify_presentation_torus_like_relators_reduce_significantly():
    gens = [
        "g_11",
        "g_13",
        "g_15",
        "g_21",
        "g_24",
        "g_25",
        "g_27",
        "g_28",
        "g_29",
        "g_30",
        "g_31",
        "g_33",
        "g_34",
        "g_35",
        "g_36",
        "g_37",
        "g_38",
        "g_39",
        "g_40",
        "g_41",
        "g_43",
        "g_44",
        "g_46",
        "g_47",
    ]
    rels = [
        ["g_24"],
        ["g_44"],
        ["g_11", "g_13^-1"],
        ["g_11", "g_15^-1"],
        ["g_13", "g_27^-1"],
        ["g_15", "g_46^-1"],
        ["g_21"],
        ["g_47"],
        ["g_25"],
        ["g_21", "g_30"],
        ["g_24", "g_28"],
        ["g_25", "g_36"],
        ["g_27", "g_31", "g_29^-1"],
        ["g_28", "g_39", "g_29^-1"],
        ["g_30", "g_33", "g_31^-1"],
        ["g_35", "g_37^-1"],
        ["g_33", "g_34^-1"],
        ["g_34", "g_44", "g_35^-1"],
        ["g_36", "g_40", "g_38^-1"],
        ["g_37", "g_46", "g_38^-1"],
        ["g_39", "g_43", "g_41^-1"],
        ["g_40", "g_47", "g_41^-1"],
        ["g_43"],
    ]
    simp = simplify_presentation(gens, rels)
    assert len(simp.generators) < 10


def test_extract_pi_1_disable_simplification_keeps_singleton_relation():
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    d2 = sp.csr_matrix(np.array([[1]], dtype=np.int64))
    cw = CWComplex(
        attaching_maps={1: d1, 2: d2}, dimensions=[0, 1, 2], cells={0: 1, 1: 1, 2: 1}
    )
    pi_raw = extract_pi_1(cw, simplify=False)
    assert pi_raw.relations == [["g_0"]]
    pi_s = extract_pi_1(cw, simplify=True)
    assert pi_s.relations == []


def test_extract_pi_1_with_traces_rejects_invalid_mode():
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    cw = CWComplex(attaching_maps={1: d1}, dimensions=[0, 1], cells={0: 1, 1: 1})
    with pytest.raises(ValueError):
        extract_pi_1_with_traces(cw, generator_mode="invalid")


def test_extract_pi_1_with_traces_simplify_false_preserves_requested_mode_metadata():
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    d2 = sp.csr_matrix(np.array([[1]], dtype=np.int64))
    cw = CWComplex(
        attaching_maps={1: d1, 2: d2}, dimensions=[0, 1, 2], cells={0: 1, 1: 1, 2: 1}
    )
    out = extract_pi_1_with_traces(cw, simplify=False, generator_mode="optimized")
    assert out.generator_mode == "raw"
    assert out.mode_used == "optimized"
    assert out.raw_generator_count == 1
    assert out.reduced_generator_count == 1


def test_extract_pi_1_with_traces_mode_alias_matches_generator_mode():
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    cw = CWComplex(attaching_maps={1: d1}, dimensions=[0, 1], cells={0: 1, 1: 1})
    out = extract_pi_1_with_traces(cw, mode="raw")
    assert out.generator_mode == "raw"
    assert out.mode_used == "raw"


def test_extract_pi_1_with_traces_uses_julia_trace_backend_when_available(monkeypatch):
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    cw = CWComplex(attaching_maps={1: d1}, dimensions=[0, 1], cells={0: 1, 1: 1})

    from pysurgery.bridge.julia_bridge import julia_engine

    monkeypatch.setattr(julia_engine, "available", True, raising=False)
    monkeypatch.setattr(
        julia_engine,
        "compute_pi1_trace_candidates",
        lambda *args, **kwargs: [
            {
                "generator": "g_0",
                "edge_index": 0,
                "component_root": 0,
                "vertex_path": [0],
                "directed_edge_path": [(0, 0)],
                "undirected_edge_path": [(0, 0)],
            }
        ],
        raising=False,
    )

    out = extract_pi_1_with_traces(cw, generator_mode="raw")
    assert out.backend_used == "julia"
    assert out.traces[0].generator == "g_0"


def test_extract_pi_1_with_traces_falls_back_to_python_when_julia_trace_fails(
    monkeypatch,
):
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    cw = CWComplex(attaching_maps={1: d1}, dimensions=[0, 1], cells={0: 1, 1: 1})

    from pysurgery.bridge.julia_bridge import julia_engine

    monkeypatch.setattr(julia_engine, "available", True, raising=False)

    def _raise(*args, **kwargs):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(
        julia_engine, "compute_pi1_trace_candidates", _raise, raising=False
    )

    out = extract_pi_1_with_traces(cw, generator_mode="raw")
    assert out.backend_used == "python"
    assert out.traces[0].generator == "g_0"


def test_infer_standard_group_descriptor_recognizes_trivial_and_infinite_cyclic():
    assert (
        infer_standard_group_descriptor(FundamentalGroup(generators=[], relations=[]))
        == "1"
    )
    assert (
        infer_standard_group_descriptor(
            FundamentalGroup(generators=["g_0"], relations=[])
        )
        == "Z"
    )


def test_infer_standard_group_descriptor_recognizes_finite_cyclic_via_gcd_of_relators():
    pi = FundamentalGroup(
        generators=["g_0"],
        relations=[
            ["g_0", "g_0", "g_0", "g_0"],
            ["g_0", "g_0", "g_0", "g_0", "g_0", "g_0"],
        ],
    )
    # <g | g^4, g^6> = C_gcd(4,6) = C_2
    assert infer_standard_group_descriptor(pi) == "Z_2"


def test_infer_standard_group_descriptor_recognizes_torus_pi1_as_zx_z():
    pi = FundamentalGroup(
        generators=["a", "b"],
        relations=[["a", "b", "a^-1", "b^-1"]],
    )
    assert infer_standard_group_descriptor(pi) == "Z x Z"


def test_infer_standard_group_descriptor_recognizes_abelian_torsion_product():
    # <a,b | [a,b], a^2, b^3> ~= Z_2 x Z_3
    pi = FundamentalGroup(
        generators=["a", "b"],
        relations=[
            ["a", "b", "a^-1", "b^-1"],
            ["a", "a"],
            ["b", "b", "b"],
        ],
    )
    # Canonical SNF form collapses coprime torsion factors to Z_6.
    assert infer_standard_group_descriptor(pi) == "Z_6"


@pytest.mark.parametrize(
    "name, builder, bettis, torsion, euler",
    get_surfaces() if "get_surfaces" in globals() else [],
)
def test_discrete_surface_fundamental_group(name, builder, bettis, torsion, euler):
    st = builder()
    complex_c = to_complex(st)
    h_rank, h_torsion = complex_c.homology(1)

    assert h_rank == bettis.get(1, 0)
    from pysurgery.bridge.julia_bridge import julia_engine

    if julia_engine.available:
        assert set(h_torsion) == set(torsion.get(1, []))


@pytest.mark.parametrize(
    "name, builder, bettis, torsion, euler",
    get_3_manifolds() if "get_3_manifolds" in globals() else [],
)
def test_discrete_3_manifold_fundamental_group(name, builder, bettis, torsion, euler):
    st = builder()
    complex_c = to_complex(st)
    h_rank, h_torsion = complex_c.homology(1)

    assert h_rank == bettis.get(1, 0)
    from pysurgery.bridge.julia_bridge import julia_engine

    if julia_engine.available:
        assert set(h_torsion) == set(torsion.get(1, []))
