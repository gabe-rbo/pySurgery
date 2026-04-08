from pysurgery.core.k_theory import compute_whitehead_group, cyclic_whitehead_rank
from pysurgery.core.fundamental_group import FundamentalGroup

def test_compute_whitehead_group_trivial():
    pi_1 = FundamentalGroup(generators=[], relations=[])
    wg = compute_whitehead_group(pi_1)
    assert wg.rank == 0
    assert wg.computable
    assert wg.exact

def test_compute_whitehead_group_circle():
    pi_1 = FundamentalGroup(generators=["g_0"], relations=[])
    wg = compute_whitehead_group(pi_1)
    assert wg.rank == 0
    assert wg.computable

def test_compute_whitehead_group_Z5():
    # Z_5 has relation g^5 = 1.
    pi_1 = FundamentalGroup(generators=["g_0"], relations=[["g_0", "g_0", "g_0", "g_0", "g_0"]])
    wg = compute_whitehead_group(pi_1)
    
    from pysurgery.bridge.julia_bridge import julia_engine
    if julia_engine.available:
        # Wh(Z_5) has rank 1
        assert wg.rank == 1
        assert "definitively exists" in wg.description
    else:
        assert not wg.computable
        assert wg.rank == -1


def test_cyclic_whitehead_rank_formula():
    assert cyclic_whitehead_rank(1) == 0
    assert cyclic_whitehead_rank(5) == 1
    assert cyclic_whitehead_rank(8) == 0
    assert cyclic_whitehead_rank(11) == 4
    assert cyclic_whitehead_rank(20) == 4

