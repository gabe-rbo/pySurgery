"""Multi-prime CRT synthesis of finite abelian groups.

Given p-primary invariant-factor lists ``{p: [d_{p,1} ≥ d_{p,2} ≥ ...]}``,
this module assembles the global invariant-factor decomposition of the
finite abelian group ``⊕_p ⊕_i Z/d_{p,i}``.

Used by ``compute_pi_n`` to merge Adams-derived per-prime torsion into a
single torsion summary. CRT is exact and well-defined; this module has no
external dependencies and is purely arithmetic.

References:
    Dummit & Foote, *Abstract Algebra* 3e, §5.2 (Fundamental Theorem of
        Finitely Generated Abelian Groups).
"""
from __future__ import annotations

from typing import Dict, List


def _check_prime_power_factors(p: int, factors: List[int]) -> None:
    if p < 2:
        raise ValueError(f"prime must be ≥ 2, got {p}")
    for d in factors:
        if d < 2:
            raise ValueError(f"invariant factor must be ≥ 2, got {d} for p={p}")
        n = d
        while n % p == 0:
            n //= p
        if n != 1:
            raise ValueError(
                f"factor {d} is not a power of {p} (p-primary list malformed)"
            )


def _padded(factors: List[int], length: int) -> List[int]:
    """Right-pad a descending list with 1's to a fixed length.

    A trailing ``1`` is the identity of Z/1 = 0, the convention used here
    to align invariant-factor lists of different lengths across primes.
    """
    return list(factors) + [1] * (length - len(factors))


def synthesize_torsion(
    p_primary: Dict[int, List[int]],
) -> List[int]:
    """Combine per-prime invariant factors into global invariant factors.

    Inputs ``p_primary[p]`` are the invariant factors of the p-primary
    part, sorted descending (largest first), and each is a power of p.

    Output is the invariant-factor list of the global torsion abelian
    group: a descending list ``[d_1, d_2, ...]`` with ``d_{i+1} | d_i``.

    Example:
        synthesize_torsion({2: [4], 3: [3]}) == [12]
        # Z/4 ⊕ Z/3 ≅ Z/12 by CRT.

        synthesize_torsion({2: [2, 2]}) == [2, 2]
        # Z/2 ⊕ Z/2 has no nontrivial CRT collapse.

        synthesize_torsion({2: [8, 2], 3: [3]}) == [24, 2]
        # Pad 3-side to length 2 (Z/3, Z/1), then CRT column-by-column:
        # max d_1 = lcm(8, 3) = 24; d_2 = lcm(2, 1) = 2.
    """
    if not p_primary:
        return []
    cleaned: Dict[int, List[int]] = {}
    for p, factors in p_primary.items():
        if not factors:
            continue
        flist = sorted((int(d) for d in factors), reverse=True)
        _check_prime_power_factors(int(p), flist)
        cleaned[int(p)] = flist
    if not cleaned:
        return []
    max_len = max(len(v) for v in cleaned.values())
    padded = {p: _padded(v, max_len) for p, v in cleaned.items()}
    out: List[int] = []
    for i in range(max_len):
        prod = 1
        for p, v in padded.items():
            prod *= v[i]
        if prod > 1:
            out.append(prod)
    return out


def torsion_to_p_primary(invariant_factors: List[int]) -> Dict[int, List[int]]:
    """Inverse of ``synthesize_torsion``: split a global decomposition by prime.

    Used by the verifier to compare an algorithm's p-by-p output against
    a published global torsion description.
    """
    by_prime: Dict[int, List[int]] = {}
    for d in invariant_factors:
        n = int(d)
        if n < 2:
            continue
        p = 2
        while n > 1:
            if n % p == 0:
                k = 0
                while n % p == 0:
                    n //= p
                    k += 1
                by_prime.setdefault(p, []).append(p**k)
            else:
                p += 1
                if p * p > n and n > 1:
                    # remaining n is prime
                    by_prime.setdefault(n, []).append(n)
                    n = 1
    for p in by_prime:
        by_prime[p].sort(reverse=True)
    return by_prime


def group_string(
    free_rank: int,
    invariant_factors: List[int],
) -> str:
    """Render ``Z^r ⊕ Z/d_1 ⊕ ... ⊕ Z/d_k`` or ``0``."""
    parts: List[str] = []
    if free_rank == 1:
        parts.append("Z")
    elif free_rank > 1:
        parts.append(f"Z^{free_rank}")
    for d in invariant_factors:
        parts.append(f"Z/{d}")
    return " ⊕ ".join(parts) if parts else "0"


__all__ = [
    "synthesize_torsion",
    "torsion_to_p_primary",
    "group_string",
]
