from pysurgery.core.k_theory import compute_whitehead_group
from pysurgery.core.fundamental_group import FundamentalGroup

def test_compute_whitehead_group_trivial():
    pi_1 = FundamentalGroup(generators=[], relations=[])
    wg = compute_whitehead_group(pi_1)
    assert wg.rank == 0

def test_compute_whitehead_group_circle():
    pi_1 = FundamentalGroup(generators=["g_0"], relations=[])
    wg = compute_whitehead_group(pi_1)
    assert wg.rank == 0

def test_compute_whitehead_group_Z5():
    # Z_5 has relation g^5 = 1.
    pi_1 = FundamentalGroup(generators=["g_0"], relations=[["g_0", "g_0", "g_0", "g_0", "g_0"]])
    wg = compute_whitehead_group(pi_1)
    
    from pysurgery.bridge.julia_bridge import julia_engine
    if julia_engine.available:
        # Wh(Z_5) has rank 1
        assert wg.rank == 1
        assert "definitively exists" in wg.description
