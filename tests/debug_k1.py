from pysurgery.topology.fundamental_group import FundamentalGroup
from pysurgery.algebra.k_theory import compute_k0_group, compute_k1_group

print("Before julia_engine import")

print("Running trivial group")
pi1_1 = FundamentalGroup(generators=[], relations=[])
k1_1 = compute_k1_group(pi1_1)
print("Done trivial group")

pi1_z2 = FundamentalGroup(generators=["a"], relations=[["a", "a"]])
print("Running compute_k0_group")
k0_z2 = compute_k0_group(pi1_z2)
print("Done compute_k0_group")

pi1_t2 = FundamentalGroup(generators=["a", "b"], relations=[["a", "b", "a^-1", "b^-1"]])
print("Running compute_k1_group")
k1_t2 = compute_k1_group(pi1_t2)
print("Done compute_k1_group. Rank =", k1_t2.rank)
print("Exiting normally.")
