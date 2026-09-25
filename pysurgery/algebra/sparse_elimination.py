"""Exact sparse unimodular elimination for integer boundary operators.

Overview:
    The Python engine behind the local-homology, fundamental-cycle and lower-star
    Morse modules. A boundary matrix of a simplicial complex is overwhelmingly made
    of ``+-1`` entries, and pivoting on a ``+-1`` entry is a *unimodular* change of
    basis on both sides:

        SNF(A) = diag(1, SNF(S)),   S the Schur complement of the pivot.

    ``sparse_smith_invariants`` pivots out every unit it can find (a Markowitz-style
    choice: the sparsest column, then the sparsest row inside it, to limit fill-in),
    and hands whatever is left -- usually nothing, or a small dense core -- to the
    exact dense Smith normal form of ``pysurgery.algebra.math_core``. The answer is
    the Smith normal form of the matrix handed in, not an estimate of it: all
    arithmetic is on Python integers (arbitrary precision), and nothing is ever
    rounded.

Key Concepts:
    - **Sparse column**: a ``{row_index: value}`` dict with no zero values.
    - **Unit pivot**: over Z the units are ``+1`` and ``-1``; over F_p every nonzero
      residue is a unit, so elimination mod a prime never gets stuck.
    - **Rank vs torsion**: ``rank`` is the rank over Q; the elementary divisors
      ``> 1`` are the torsion coefficients of the cokernel.

Common Workflows:
    1. **Integer homology of a small complex** ->
       ``reduced_homology_from_simplices(maximal_simplices)``.
    2. **Rank / torsion of one operator** -> ``sparse_smith_invariants(columns)``.
    3. **Rank over F_p** -> ``sparse_rank_mod_p(columns, p)``.

Coefficient Ring:
    Z (exact, arbitrary precision) and F_p (p prime).
"""

from __future__ import annotations

import heapq
import itertools
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Column = Dict[int, int]

__all__ = [
    "sparse_smith_invariants",
    "sparse_rank_mod_p",
    "boundary_columns",
    "close_under_faces",
    "reduced_homology_from_simplices",
]


def _eliminate(cols: Sequence[Column], modulus: Optional[int]) -> Tuple[int, List[Column]]:
    """Pivot out entries that are units, exactly.

    Args:
        cols: Sparse columns ``{row: value}``.
        modulus: ``None`` for Z (units are +-1), or a prime p for F_p.

    Returns:
        ``(number_of_pivots, surviving_columns)``. Each pivot is a unimodular change of
        basis on rows and columns, so the Smith normal form of the input equals
        ``diag(1, ..., 1, SNF(surviving block))``.
    """
    col: Dict[int, Column] = {}
    row: Dict[int, set] = {}
    for j, c in enumerate(cols):
        c = {int(i): (int(v) % modulus if modulus else int(v)) for i, v in c.items()}
        c = {i: v for i, v in c.items() if v}
        if c:
            col[j] = c
            for i in c:
                row.setdefault(i, set()).add(j)

    def is_unit(v: int) -> bool:
        return v != 0 if modulus else (v == 1 or v == -1)

    heap = [(len(c), j) for j, c in col.items()]
    heapq.heapify(heap)
    pivots = 0
    while heap:
        n, j = heapq.heappop(heap)
        c = col.get(j)
        if c is None or len(c) != n:
            continue  # stale heap entry
        best = None
        for i, v in c.items():
            if is_unit(v):
                ri = len(row[i])
                if best is None or ri < best[0]:
                    best = (ri, i, v)
                    if ri == 1:
                        break
        if best is None:
            continue  # no unit in this column (yet)
        _, r, a = best
        a_inv = pow(a, -1, modulus) if modulus else a  # +-1 is its own inverse
        prow = {jj: col[jj][r] for jj in row[r] if jj != j}
        for i, a_ic in c.items():
            if i == r:
                continue
            f = a_ic * a_inv
            if modulus:
                f %= modulus
            ri = row[i]
            for jj, a_rj in prow.items():
                cj = col[jj]
                new = cj.get(i, 0) - f * a_rj
                if modulus:
                    new %= modulus
                if new:
                    cj[i] = new
                    ri.add(jj)
                elif i in cj:
                    del cj[i]
                    ri.discard(jj)
        for jj in prow:
            del col[jj][r]
        for i in c:
            if i != r:
                row[i].discard(j)
        del col[j]
        del row[r]
        pivots += 1
        for jj in prow:
            cj = col[jj]
            if cj:
                heapq.heappush(heap, (len(cj), jj))
            else:
                del col[jj]
    return pivots, [c for c in col.values() if c]


def sparse_smith_invariants(columns: Sequence[Column]) -> Tuple[int, List[int]]:
    """Rank and the elementary divisors ``> 1`` of an integer matrix, exactly.

    What is Being Computed?:
        For an integer matrix A (given by sparse columns), the rank of A over Q and the
        invariant factors ``d_i > 1`` of its Smith normal form -- the torsion
        coefficients of ``coker(A)``.

    Algorithm:
        1. Unit-pivot elimination (``+-1`` pivots, Markowitz order): each pivot
           contributes an invariant factor 1.
        2. The surviving columns (no unit left) form a small dense core whose exact
           Smith normal form is computed by ``math_core.get_snf_diagonal``
           (arbitrary-precision object arithmetic).

    Preserved Invariants:
        Every step is unimodular, so the result is the Smith normal form of the input.

    Args:
        columns: The matrix as a sequence of ``{row: value}`` dicts.

    Returns:
        ``(rank, torsion)`` with ``torsion`` the sorted elementary divisors ``> 1``.

    Example:
        >>> sparse_smith_invariants([{0: 2}])
        (1, [2])
    """
    pivots, rest = _eliminate(columns, None)
    if not rest:
        return pivots, []
    from .math_core import get_snf_diagonal

    rows_left = sorted({i for c in rest for i in c})
    pos = {r: k for k, r in enumerate(rows_left)}
    dense = np.zeros((len(rows_left), len(rest)), dtype=object)
    for j, c in enumerate(rest):
        for i, v in c.items():
            dense[pos[i], j] = int(v)
    diag = [abs(int(d)) for d in get_snf_diagonal(dense)]
    return pivots + len(diag), sorted(d for d in diag if d > 1)


def sparse_rank_mod_p(columns: Sequence[Column], p: int) -> int:
    """Rank over the prime field F_p of an integer matrix, exactly.

    Args:
        columns: The matrix as a sequence of ``{row: value}`` dicts.
        p: A prime.

    Returns:
        The rank of the matrix reduced mod p.

    Raises:
        ValueError: If ``p`` is not prime.
    """
    p = int(p)
    if p < 2 or any(p % k == 0 for k in range(2, int(p ** 0.5) + 1)):
        raise ValueError(f"p = {p} is not prime; F_p is not a field")
    pivots, rest = _eliminate(columns, p)
    if rest:  # pragma: no cover - elimination over a field cannot get stuck
        raise RuntimeError("elimination over a prime field left a non-unit block")
    return pivots


def close_under_faces(simplices: Iterable[Iterable[int]]) -> Dict[int, List[Tuple[int, ...]]]:
    """All faces of the given simplices, grouped by dimension and sorted.

    Args:
        simplices: Generating simplices (any vertex order).

    Returns:
        ``{dimension: sorted list of sorted vertex tuples}``; empty for no input.
    """
    faces: Dict[int, set] = {}
    for s in simplices:
        v = tuple(sorted({int(x) for x in s}))
        if not v:
            continue
        for r in range(1, len(v) + 1):
            bucket = faces.setdefault(r - 1, set())
            for f in itertools.combinations(v, r):
                bucket.add(f)
    return {d: sorted(fs) for d, fs in sorted(faces.items())}


def boundary_columns(
    simplices_d: Sequence[Tuple[int, ...]], index_dm1: Dict[Tuple[int, ...], int]
) -> List[Column]:
    """Sparse columns of the simplicial boundary ``d: C_d -> C_{d-1}``.

    Uses the standard orientation convention of pySurgery: a simplex is oriented by its
    sorted vertex order and ``d[v_0..v_d] = sum_i (-1)^i [v_0..^v_i..v_d]``.

    Args:
        simplices_d: The d-simplices (sorted tuples).
        index_dm1: Row index of every (d-1)-simplex.

    Returns:
        One ``{row: +-1}`` column per d-simplex.
    """
    cols: List[Column] = []
    for s in simplices_d:
        c: Column = {}
        for i in range(len(s)):
            c[index_dm1[s[:i] + s[i + 1:]]] = -1 if i % 2 else 1
        cols.append(c)
    return cols


def reduced_homology_from_simplices(
    simplices: Iterable[Iterable[int]],
) -> Tuple[Dict[int, Tuple[int, List[int]]], int]:
    """Reduced integer homology of the complex generated by ``simplices``, exactly.

    What is Being Computed?:
        ``H~_d(L; Z)`` for every degree d where it is nonzero, **including degree -1**:
        the empty complex has ``H~_{-1} = Z`` (the augmented chain complex of the empty
        set is Z in degree -1) and nothing else. This convention is what makes the link
        formula ``H_j(|K|, |K| - x) = H~_{j - dim(sigma) - 1}(lk sigma)`` correct at a
        maximal simplex, whose link is empty.

    Algorithm:
        1. Close the generators under faces.
        2. Build every boundary operator as sparse +-1 columns.
        3. ``sparse_smith_invariants`` on each operator gives its rank and the torsion
           of the homology one degree below.
        4. ``beta_d = n_d - rank(d_d) - rank(d_{d+1})``, minus one in degree 0.

    Args:
        simplices: Generators (e.g. the maximal simplices of a link).

    Returns:
        ``(reduced, dimension)``: ``reduced`` maps degree -> ``(rank, torsion)`` for the
        nonzero groups only; ``dimension`` is the dimension of the complex (-1 if empty).
    """
    faces = close_under_faces(simplices)
    if not faces:
        return {-1: (1, [])}, -1
    top = max(faces)
    index = {d: {s: i for i, s in enumerate(faces.get(d, []))} for d in range(top + 1)}
    ranks: Dict[int, int] = {0: 0, top + 1: 0}
    tors: Dict[int, List[int]] = {}
    for d in range(1, top + 1):
        r, t = sparse_smith_invariants(boundary_columns(faces.get(d, []), index[d - 1]))
        ranks[d] = r
        tors[d - 1] = t
    out: Dict[int, Tuple[int, List[int]]] = {}
    for d in range(top + 1):
        beta = len(faces.get(d, [])) - ranks.get(d, 0) - ranks.get(d + 1, 0)
        if d == 0:
            beta -= 1
        t = tors.get(d, [])
        if beta or t:
            out[d] = (int(beta), list(t))
    return out, top
