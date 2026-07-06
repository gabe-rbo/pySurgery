"""Exact and adaptive-precision geometric sign predicates.

Overview:
    Manifold-reconstruction algorithms (Cocone, tangential Delaunay complexes, ...) make
    many small yes/no geometric decisions -- which side of a plane a point falls on, whether
    a point lies inside a circumscribed circle/sphere, which of two candidate triangles is
    "more Delaunay" -- and the *topology* of the reconstructed complex depends on getting
    every one of these signs right. A single wrong sign near a degenerate configuration can
    silently produce a combinatorially inconsistent complex. This module makes those sign
    decisions exact rather than float64-with-an-epsilon-tolerance.

Key Concepts:
    - **Two-tier exactness**: every predicate here first evaluates a fixed Leibniz/cofactor
      formula in float64 and bounds its worst-case forward error using Higham's standard
      running-error bound ``gamma_k = k*u / (1 - k*u)`` (``u`` = the IEEE-754 double unit
      roundoff). If the float result clears that bound, its sign is trusted. Otherwise the
      *same* formula is re-evaluated in exact ``fractions.Fraction`` arithmetic and that sign
      is returned instead.
    - **Why ``Fraction`` and not Shewchuk's adaptive expansions**: every IEEE-754 double is an
      exact dyadic rational, so ``Fraction(float(x))`` loses no information -- recomputing the
      determinant in ``Fraction`` arithmetic is exact, full stop, with no need to hand-roll
      Shewchuk's adaptive floating-point expansion cascade (a correct, from-scratch
      implementation of which is a substantial undertaking in its own right). The trade is
      worst-case speed (big-integer arithmetic) for a much simpler, easier-to-verify-correct
      implementation; this is the same two-tier philosophy as Shewchuk predicates (fast filter,
      exact fallback), built on Python's native arbitrary-precision rationals instead.
    - **Scope**: these predicates re-derive topology-*determining* decisions built on top of a
      triangulation (pole signs, cocone membership, umbrella-walk ties, local Delaunay
      diagnostics) -- they do not replace the triangulation algorithm itself. The base
      triangulation topology still comes from ``scipy.spatial.Delaunay``/``Voronoi``, exactly
      as it already does elsewhere in pySurgery (``from_alpha_complex``, ``from_crust_algorithm``).

Common Workflows:
    1. **Pole / halfspace sign tests** -> ``exact_sign_of_sum`` on a dot product's per-term
       array.
    2. **Which side of a line/plane** -> ``orientation2d`` / ``orientation3d``.
    3. **Circumcircle/circumsphere membership** -> ``incircle2d`` / ``insphere3d``.
    4. **Bulk predicate evaluation over many candidate simplices** ->
       ``exact_signs_of_determinants_batch`` (vectorized float filter, Julia-accelerated
       exact fallback available via ``backend="julia"``).
"""

from __future__ import annotations

import itertools
import warnings
from fractions import Fraction
from typing import Sequence

import numpy as np

_EPS_MACHINE = 2.0**-53  # IEEE-754 double unit roundoff, u.
_MAX_PREDICATE_N = 6  # Largest square-matrix size this module precomputes permutations for.


def _build_signed_permutations(n: int) -> list[tuple[tuple[int, ...], int]]:
    """Enumerate every permutation of ``range(n)`` together with its sign.

    Algorithm:
        1. Generate all ``n!`` permutations of ``range(n)``.
        2. For each, count inversions (pairs out of order); parity of that count gives the
           permutation's sign (+1 even, -1 odd).

    Args:
        n: Size of the permutations to enumerate.

    Returns:
        A list of ``(permutation, sign)`` pairs, one per element of ``S_n``.
    """
    perms = []
    for perm in itertools.permutations(range(n)):
        inversions = sum(1 for i in range(n) for j in range(i + 1, n) if perm[i] > perm[j])
        sign = -1 if (inversions % 2) else 1
        perms.append((perm, sign))
    return perms


_PERMUTATIONS: dict[int, list[tuple[tuple[int, ...], int]]] = {
    n: _build_signed_permutations(n) for n in range(1, _MAX_PREDICATE_N + 1)
}


def _leibniz_det(matrix: Sequence[Sequence]):
    """Evaluate ``det(matrix)`` via the Leibniz (cofactor) formula.

    Overview:
        Deliberately not ``numpy.linalg.det`` (LU with partial pivoting): the Leibniz formula
        is a plain sum of ``n!`` signed products, which (a) is generic over the entry type --
        the identical arithmetic sequence runs unchanged over ``float`` or ``fractions.Fraction``
        entries -- and (b) has an error behavior that is trivial to bound a priori (see
        ``_leibniz_error_bound``), unlike pivoted LU.

    Args:
        matrix: An ``n``x``n`` sequence of sequences of numbers (``float`` or ``Fraction``).

    Returns:
        The determinant, as whatever number type the entries were (float in, float out;
        Fraction in, Fraction out).
    """
    n = len(matrix)
    total = 0
    for perm, sign in _PERMUTATIONS[n]:
        term = sign
        for i in range(n):
            term = term * matrix[i][perm[i]]
        total = total + term
    return total


def _gamma(k: int) -> float:
    """Higham's standard forward-error growth factor ``gamma_k = k*u / (1 - k*u)``.

    What is Being Computed?:
        A conservative bound on the relative forward error accumulated by ``k`` sequential
        floating-point rounding steps (``u`` = the unit roundoff, ``2**-53`` for float64).
        Used to certify (or refuse to certify) the sign of a float64 Leibniz-formula result.

    Args:
        k: A conservative count of rounding steps in the computation being bounded.

    Returns:
        The growth factor ``gamma_k``.
    """
    ku = k * _EPS_MACHINE
    return ku / (1.0 - ku)


def _leibniz_error_bound(matrix_abs: np.ndarray, n: int) -> float:
    """Bound the float64 forward error of ``_leibniz_det`` on a matrix of the given entries.

    Algorithm:
        1. Re-run the exact same permutation structure as ``_leibniz_det``, but on the
           entrywise-absolute-value matrix, to get ``sum_of_|each signed product|`` -- the
           standard "run the same algorithm on |a_ij|" running-error-bound technique.
        2. Multiply by ``gamma_k`` with ``k = n! * n``, a conservative (over-)count of the
           rounding steps in ``n!`` products of ``n`` factors each, summed.

    Preserved Invariants:
        The bound is intentionally conservative (never too tight): a looser bound only makes
        the exact ``Fraction`` fallback trigger somewhat more often than a razor-tight,
        predicate-specific bound would -- it can never accept a wrong float sign.

    Args:
        matrix_abs: The entrywise absolute value of the float64 matrix.
        n: The matrix dimension.

    Returns:
        A conservative upper bound on ``|computed_det - true_det|``.
    """
    n_perms = len(_PERMUTATIONS[n])
    total_abs = 0.0
    for perm, _sign in _PERMUTATIONS[n]:
        term = 1.0
        for i in range(n):
            term *= matrix_abs[i, perm[i]]
        total_abs += term
    k = n_perms * n
    return _gamma(k) * total_abs


def exact_sign_of_determinant(matrix: np.ndarray | Sequence[Sequence[float]]) -> int:
    """Exact sign of ``det(matrix)`` for a small (``n <= 6``) square matrix.

    What is Being Computed?:
        ``sign(det(matrix))`` as ``+1``, ``-1``, or ``0`` -- exactly, never an approximation.

    Algorithm:
        1. Evaluate the Leibniz formula in float64.
        2. Bound its forward error via ``_leibniz_error_bound``.
        3. If the float result's magnitude clears the bound, trust its sign.
        4. Otherwise convert every entry to ``fractions.Fraction(float(x))`` (exact -- every
           IEEE-754 double is an exact dyadic rational) and re-evaluate the same formula
           exactly; return that sign.

    Args:
        matrix: A square array-like of shape ``(n, n)``, ``1 <= n <= 6``.

    Returns:
        int: ``1`` if ``det > 0``, ``-1`` if ``det < 0``, ``0`` if exactly singular.

    Raises:
        ValueError: If ``matrix`` is not square or ``n`` is out of the supported range.

    Example:
        >>> exact_sign_of_determinant([[1.0, 0.0], [0.0, 1.0]])
        1
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"matrix must be square, got shape {m.shape}.")
    n = m.shape[0]
    if n < 1 or n > _MAX_PREDICATE_N:
        raise ValueError(f"matrix size must be in [1, {_MAX_PREDICATE_N}], got {n}.")

    float_det = _leibniz_det(m.tolist())
    bound = _leibniz_error_bound(np.abs(m), n)
    if abs(float_det) > bound:
        return 1 if float_det > 0.0 else -1

    frac_matrix = [[Fraction(float(x)) for x in row] for row in m]
    exact_det = _leibniz_det(frac_matrix)
    if exact_det > 0:
        return 1
    if exact_det < 0:
        return -1
    return 0


def exact_signs_of_determinants_batch(
    matrices: np.ndarray, *, backend: str = "auto"
) -> np.ndarray:
    """Exact signs of ``det(matrices[i])`` for a batch of small square matrices.

    What is Being Computed?:
        The same certificate as ``exact_sign_of_determinant``, vectorized over a batch --
        the float64 filter runs on the whole batch at once (via NumPy), and only the
        (expected rare) matrices that fail the filter fall back to per-item exact arithmetic.

    Args:
        matrices: Array of shape ``(M, n, n)``, ``1 <= n <= 6``, one fixed ``n`` per call.
        backend: ``'auto'``, ``'julia'``, or ``'python'``. ``'julia'`` (or ``'auto'`` with
            Julia available) dispatches the per-item exact fallback to
            ``JuliaBridge.exact_signs_of_determinants_batch`` (``Rational{BigInt}`` exact
            arithmetic); on failure it warns and falls back to Python unless ``backend`` was
            forced to ``'julia'``, in which case the error is re-raised.

    Returns:
        np.ndarray: Shape ``(M,)`` int64 array of signs in ``{-1, 0, 1}``.

    Raises:
        ValueError: If ``matrices`` is not shaped ``(M, n, n)`` or ``n`` is out of range.

    Use When:
        - Filtering many candidate Delaunay facets/tetrahedra at once (Cocone, tangential
          complex star-consistency) where a per-item Python loop over
          ``exact_sign_of_determinant`` would dominate runtime.
    """
    mats = np.asarray(matrices, dtype=np.float64)
    if mats.ndim != 3 or mats.shape[1] != mats.shape[2]:
        raise ValueError(f"matrices must have shape (M, n, n), got {mats.shape}.")
    m_count, n, _ = mats.shape
    if n < 1 or n > _MAX_PREDICATE_N:
        raise ValueError(f"matrix size must be in [1, {_MAX_PREDICATE_N}], got {n}.")

    perms = _PERMUTATIONS[n]
    float_dets = np.zeros(m_count, dtype=np.float64)
    abs_dets = np.zeros(m_count, dtype=np.float64)
    for perm, sign in perms:
        term = np.full(m_count, float(sign), dtype=np.float64)
        term_abs = np.ones(m_count, dtype=np.float64)
        for i in range(n):
            col = mats[:, i, perm[i]]
            term = term * col
            term_abs = term_abs * np.abs(col)
        float_dets += term
        abs_dets += term_abs

    bound = _gamma(len(perms) * n) * abs_dets
    signs = np.sign(float_dets).astype(np.int64)
    uncertain = np.nonzero(np.abs(float_dets) <= bound)[0]
    if uncertain.size == 0:
        return signs

    backend_norm = str(backend).lower().strip()
    from ..bridge.julia_bridge import julia_engine

    use_julia = (backend_norm == "julia") or (backend_norm == "auto" and julia_engine.available)
    if use_julia:
        try:
            exact_signs = julia_engine.exact_signs_of_determinants_batch(mats[uncertain])
            signs[uncertain] = exact_signs
            return signs
        except Exception as e:
            if backend_norm == "julia":
                raise e
            warnings.warn(f"Julia exact_signs_of_determinants_batch failed ({e!r}). Falling back to pure Python.")

    for i in uncertain:
        signs[i] = exact_sign_of_determinant(mats[i])
    return signs


def exact_sign_of_sum(terms: np.ndarray | Sequence[float]) -> int:
    """Exact sign of a sum of already-formed float terms (e.g. a dot product's summands).

    What is Being Computed?:
        ``sign(sum(terms))``, exactly. Useful for halfspace/sign tests that are naturally a
        sum of a handful of products (e.g. ``(v - p) . n`` expressed as its 3 per-coordinate
        products) without going through the full determinant machinery.

    Algorithm:
        Same two-tier philosophy as ``exact_sign_of_determinant``: float64 sum, Higham
        ``gamma_k`` bound (``k = len(terms)``) against ``sum(|terms|)``, exact
        ``fractions.Fraction`` fallback on failure.

    Args:
        terms: A 1-D array-like of the summands (already multiplied out).

    Returns:
        int: ``1``, ``-1``, or ``0``.
    """
    arr = np.asarray(terms, dtype=np.float64)
    total = float(np.sum(arr))
    bound = _gamma(arr.size) * float(np.sum(np.abs(arr)))
    if abs(total) > bound:
        return 1 if total > 0.0 else -1
    exact_total = sum((Fraction(float(x)) for x in arr), Fraction(0))
    if exact_total > 0:
        return 1
    if exact_total < 0:
        return -1
    return 0


def orientation2d(a, b, c) -> int:
    """Orientation predicate for three points in the plane.

    What is Being Computed?:
        The sign of ``det([[a_x,a_y,1],[b_x,b_y,1],[c_x,c_y,1]])``.

    Args:
        a: First point, length-2 array-like.
        b: Second point, length-2 array-like.
        c: Third point, length-2 array-like.

    Returns:
        int: ``+1`` if ``a, b, c`` are in counterclockwise order, ``-1`` if clockwise,
        ``0`` if collinear.
    """
    m = [[float(a[0]), float(a[1]), 1.0],
         [float(b[0]), float(b[1]), 1.0],
         [float(c[0]), float(c[1]), 1.0]]
    return exact_sign_of_determinant(m)


def orientation3d(a, b, c, d) -> int:
    """Orientation predicate for four points in space.

    What is Being Computed?:
        The sign of ``det([[a,1],[b,1],[c,1],[d,1]])`` (each row is a point's 3 coordinates
        plus a homogeneous ``1``).

    Args:
        a: First point, length-3 array-like.
        b: Second point, length-3 array-like.
        c: Third point, length-3 array-like.
        d: Fourth point, length-3 array-like.

    Returns:
        int: ``+1`` if ``d`` is on the positive side of the oriented plane through
        ``a, b, c``, ``-1`` if on the negative side, ``0`` if coplanar.
    """
    m = [[float(a[0]), float(a[1]), float(a[2]), 1.0],
         [float(b[0]), float(b[1]), float(b[2]), 1.0],
         [float(c[0]), float(c[1]), float(c[2]), 1.0],
         [float(d[0]), float(d[1]), float(d[2]), 1.0]]
    return exact_sign_of_determinant(m)


def incircle2d(a, b, c, d) -> int:
    """In-circle predicate for four points in the plane.

    What is Being Computed?:
        The sign of the standard lifted-paraboloid determinant with rows
        ``[p_x, p_y, p_x**2 + p_y**2, 1]`` for ``p in (a, b, c, d)``.

    Args:
        a: First point (defines the circle, together with ``b``, ``c``), length-2 array-like.
        b: Second point, length-2 array-like.
        c: Third point, length-2 array-like.
        d: Query point, length-2 array-like.

    Returns:
        int: If ``a, b, c`` are in counterclockwise order, ``+1`` means ``d`` lies strictly
        inside the circle through ``a, b, c``, ``-1`` strictly outside, ``0`` exactly on it
        (a cocircular configuration). The sign flips if ``a, b, c`` are clockwise.
    """
    def _row(p):
        x, y = float(p[0]), float(p[1])
        return [x, y, x * x + y * y, 1.0]

    m = [_row(a), _row(b), _row(c), _row(d)]
    return exact_sign_of_determinant(m)


def insphere3d(a, b, c, d, e) -> int:
    """In-sphere predicate for five points in space.

    What is Being Computed?:
        The sign of the standard lifted-paraboloid determinant with rows
        ``[p_x, p_y, p_z, p_x**2 + p_y**2 + p_z**2, 1]`` for ``p in (a, b, c, d, e)``.

    Args:
        a: First point (defines the sphere, together with ``b``, ``c``, ``d``),
            length-3 array-like.
        b: Second point, length-3 array-like.
        c: Third point, length-3 array-like.
        d: Fourth point, length-3 array-like.
        e: Query point, length-3 array-like.

    Returns:
        int: If ``a, b, c, d`` are positively oriented (``orientation3d(a,b,c,d) > 0``),
        ``+1`` means ``e`` lies strictly inside their circumsphere, ``-1`` strictly outside,
        ``0`` exactly on it (a cospherical configuration). The sign flips if
        ``a, b, c, d`` are negatively oriented.
    """
    def _row(p):
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        return [x, y, z, x * x + y * y + z * z, 1.0]

    m = [_row(a), _row(b), _row(c), _row(d), _row(e)]
    return exact_sign_of_determinant(m)


__all__ = [
    "exact_sign_of_determinant",
    "exact_signs_of_determinants_batch",
    "exact_sign_of_sum",
    "orientation2d",
    "orientation3d",
    "incircle2d",
    "insphere3d",
]
