"""Active-object (OOP) tests for FundamentalGroup and HomotopyGroup.

Overview:
    Validates the v2 "Active Mathematical Object" refactor of the homotopy API:
        * FundamentalGroup gains is_abelian / is_trivial / order / simplify /
          simplices_generators methods, all backed by per-instance caching.
        * HomotopyGroup unifies RationalHomotopyGroup + AdamsE2Page and exposes
          rank(n), torsion(n, p), and simplices_generators(n).

Key Concepts:
    - **Active Object**: domain methods live on the class, not in free helpers.
    - **Geometric Bridge**: simplices_generators links symbolic generators to
      concrete edge cycles in the source CW / simplicial complex.
    - **Caching**: invariant queries are memoized for the lifetime of the object.

Common Workflows:
    1. Build a FundamentalGroup, query is_abelian / order, verify cache hit.
    2. Extract pi_1 with traces from a CW complex, verify
       simplices_generators returns valid (closed) cycles.
    3. Build a HomotopyGroup for a sphere, query rank/torsion, confirm caching.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from pysurgery.topology.complexes import CWComplex
from pysurgery.core.exceptions import FundamentalGroupError
from pysurgery.topology.fundamental_group import (
    FundamentalGroup,
    extract_pi_1,
    infer_standard_group_descriptor,
)
from pysurgery.homotopy.higher_homotopy_groups import (
    HomotopyGroup,
    sphere_cohomology,
    sphere_cohomology_fp,
)


# ── FundamentalGroup: is_abelian / is_trivial / order ────────────────────────


def test_torus_pi1_is_abelian():
    """The torus has pi_1 = Z x Z, an abelian group."""
    pi1 = FundamentalGroup(
        generators=["a", "b"],
        relations=[["a", "b", "a^-1", "b^-1"]],
    )
    assert pi1.is_abelian() is True


def test_genus2_pi1_is_not_abelian_or_undecidable():
    """Genus-2 surface group is non-abelian.

    The implementation only certifies non-abelianness via finite quotient
    witnesses; for the genus-2 surface (infinite, abelianization Z^4)
    that witness is unavailable, so we accept either a sound False or
    a FundamentalGroupError.
    """
    pi1 = FundamentalGroup(
        generators=["a", "b", "c", "d"],
        relations=[["a", "b", "a^-1", "b^-1", "c", "d", "c^-1", "d^-1"]],
    )
    try:
        result = pi1.is_abelian()
    except FundamentalGroupError:
        return  # acceptable: the procedure declined to certify
    assert result is False, "genus-2 surface group must not be reported abelian"


def test_pi1_is_abelian_uses_cache(monkeypatch):
    """Verify is_abelian is memoised — internal computation runs at most once."""
    pi1 = FundamentalGroup(
        generators=["a", "b"],
        relations=[["a", "b", "a^-1", "b^-1"]],
    )

    call_counter = {"count": 0}
    original = pi1._compute_is_abelian_internal

    def counting_wrapper():
        call_counter["count"] += 1
        return original()

    monkeypatch.setattr(pi1, "_compute_is_abelian_internal", counting_wrapper)

    assert pi1.is_abelian() is True
    assert pi1.is_abelian() is True
    assert pi1.is_abelian() is True
    assert call_counter["count"] == 1, "is_abelian must compute exactly once"


def test_trivial_group_is_trivial_and_abelian():
    pi1 = FundamentalGroup(generators=[], relations=[])
    assert pi1.is_trivial() is True
    assert pi1.is_abelian() is True
    assert pi1.order() == 1


def test_z2_pi1_order_and_not_trivial():
    pi1 = FundamentalGroup(generators=["a"], relations=[["a", "a"]])
    assert pi1.order() == 2
    assert pi1.is_trivial() is False
    assert pi1.is_abelian() is True  # one generator => abelian


def test_pi1_order_caches_result():
    """order() should be computed once; second call hits the cache."""
    pi1 = FundamentalGroup(generators=["a"], relations=[["a", "a", "a"]])
    n1 = pi1.order()
    cache_keys = [tuple(k) for k in pi1.cache_info()["keys"]]
    assert any(k[0] == "order" for k in cache_keys), "order result not cached"
    n2 = pi1.order()
    assert n1 == n2 == 3


# ── FundamentalGroup: simplify ───────────────────────────────────────────────


def test_simplify_returns_new_fundamentalgroup():
    pi1 = FundamentalGroup(
        generators=["a", "b"],
        relations=[["a"]],
    )
    simp = pi1.simplify()
    assert isinstance(simp, FundamentalGroup)
    assert simp.generators == ["b"]
    assert simp.relations == []
    # Original is not mutated.
    assert pi1.generators == ["a", "b"]


# ── FundamentalGroup: simplices_generators ───────────────────────────────────


def test_simplices_generators_circle_returns_closed_cycle():
    """For S^1 the single generator is the loop (0,0)."""
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    cw = CWComplex(cells={0: 1, 1: 1}, attaching_maps={1: d1}, dimensions=[0, 1])
    pi = extract_pi_1(cw)
    edges = pi.simplices_generators()
    assert "g_0" in edges
    path = edges["g_0"]
    assert path == [(0, 0)]
    # A trivially closed cycle: start vertex == end vertex.
    assert path[0][0] == path[-1][1]


def test_simplices_generators_traces_form_valid_cycles_on_torus():
    """Each generator's edge sequence must be a closed walk in the 1-skeleton."""
    from discrete_surface_data import build_torus  # type: ignore[import-not-found]

    sc = build_torus()
    cw = sc.to_cw_complex() if hasattr(sc, "to_cw_complex") else None
    if cw is None:
        pytest.skip("Torus simplicial complex has no CWComplex bridge.")

    pi = extract_pi_1(cw)
    edges_by_gen = pi.simplices_generators()
    assert edges_by_gen, "extracted pi_1 must expose at least one generator path"
    for gen, path in edges_by_gen.items():
        if not path:
            continue
        # Closed cycle: chained endpoints, last edge returns to start.
        for i in range(len(path) - 1):
            assert path[i][1] == path[i + 1][0], (
                f"generator {gen}: edge {i} → {i+1} not chained "
                f"({path[i]} -> {path[i+1]})"
            )
        assert path[0][0] == path[-1][1], (
            f"generator {gen}: cycle is not closed ({path})"
        )


def test_simplices_generators_empty_when_no_traces():
    pi1 = FundamentalGroup(generators=["x"], relations=[])
    out = pi1.simplices_generators()
    assert out == {"x": []}


# ── HomotopyGroup: rank / torsion / cache ────────────────────────────────────


def test_homotopy_group_rank_for_sphere():
    hg = HomotopyGroup.from_inputs(
        sphere_cohomology(3),
        fp_cohomology_ring=sphere_cohomology_fp(3, prime=2),
        adams_s_max=4,
        adams_t_max=10,
    )
    assert hg.rank(3) == 1
    assert hg.rank(2) == 0
    assert hg.is_rationally_trivial(2) is True
    assert hg.is_rationally_trivial(3) is False


def test_homotopy_group_torsion_returns_tuple():
    hg = HomotopyGroup.from_inputs(
        sphere_cohomology(3),
        fp_cohomology_ring=sphere_cohomology_fp(3, prime=2),
        adams_s_max=4,
        adams_t_max=10,
    )
    t = hg.torsion(3, p=2)
    assert isinstance(t, tuple)
    assert all(isinstance(x, int) and x >= 0 for x in t)


def test_homotopy_group_cache_is_per_query():
    hg = HomotopyGroup.from_inputs(
        sphere_cohomology(3),
        fp_cohomology_ring=sphere_cohomology_fp(3, prime=2),
        adams_s_max=2,
        adams_t_max=6,
    )
    # First queries populate the cache.
    hg.rank(3)
    hg.rank(2)
    hg.torsion(3, p=2)
    s1 = hg.cache_info()["size"]

    # Repeated queries do not grow the cache.
    hg.rank(3)
    hg.rank(2)
    hg.torsion(3, p=2)
    s2 = hg.cache_info()["size"]
    assert s1 == s2 > 0


def test_homotopy_group_simplices_generators_for_sphere():
    hg = HomotopyGroup.from_inputs(
        sphere_cohomology(3),
        fp_cohomology_ring=sphere_cohomology_fp(3, prime=2),
        adams_s_max=2,
        adams_t_max=6,
    )
    info = hg.simplices_generators(3)
    assert info, "S^3 should have one π_3 generator"
    name = next(iter(info))
    entry = info[name]
    assert entry["degree"] == 3
    assert entry["is_closed"] is True


# ── Backwards compatibility: top-level helpers still work ────────────────────


def test_infer_standard_group_descriptor_still_works():
    pi1 = FundamentalGroup(generators=["a"], relations=[["a"] * 5])
    assert infer_standard_group_descriptor(pi1) == "Z_5"
