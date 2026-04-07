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
