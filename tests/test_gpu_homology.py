"""Tests for exact homology on the GPU (pysurgery.gpu.homology).

Overview:
    Every answer of the GPU engine is exact, so every test compares it with an exact
    authority that shares none of its machinery: sympy's Smith normal form, the
    pure-Python elimination already in pySurgery (``_rank_mod_p``, the CPU
    homology path), and textbook homology of small spaces (RP^2, the Klein bottle,
    the torus, S^3, the lens space L(3,1)). The universal coefficient theorem ties
    the modular and integral answers together. Matrices are tiny, so the suite is
    light on memory; the overflow guard is exercised with entries near 2^30.
"""
import numpy as np
import pytest
import scipy.sparse as sp

torch = pytest.importorskip("torch")

from sympy import Matrix, ZZ  # noqa: E402
from sympy.matrices.normalforms import smith_normal_form  # noqa: E402

from pysurgery.gpu import homology as H  # noqa: E402
from pysurgery.topology.complexes import ChainComplex, SimplicialComplex, _rank_mod_p  # noqa: E402

RP2 = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 1],
       [1, 2, 4], [2, 3, 5], [3, 4, 1], [4, 5, 2], [5, 1, 3]]
TORUS = [[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4], [2, 0, 3], [2, 3, 5],
         [3, 4, 7], [3, 7, 6], [4, 5, 8], [4, 8, 7], [5, 3, 6], [5, 6, 8],
         [6, 7, 1], [6, 1, 0], [7, 8, 2], [7, 2, 1], [8, 6, 0], [8, 0, 2]]


def _sympy_factors(A):
    S = smith_normal_form(Matrix(A), domain=ZZ)
    return sorted(abs(int(S[i, i])) for i in range(min(S.shape)) if S[i, i] != 0)


def _cw(boundaries, cells, ring="Z"):
    bd = {k: sp.csr_matrix(np.asarray(v, dtype=np.int64)) for k, v in boundaries.items()}
    return ChainComplex(boundaries=bd, dimensions=sorted(cells), cells=cells, coefficient_ring=ring)


# ─────────────────────────────────────────────────────────── one matrix at a time

def test_invariant_factors_are_the_divisibility_chain():
    assert H.invariant_factors_from_diagonal([2, 3]) == [1, 6]
    assert H.invariant_factors_from_diagonal([4, 0, -6, 1, 2]) == [1, 2, 2, 12]
    assert H.invariant_factors_from_diagonal([]) == []


def test_smith_factors_match_sympy_on_random_matrices():
    rng = np.random.default_rng(0)
    for _ in range(60):
        m, n = rng.integers(1, 8, size=2)
        A = rng.integers(-4, 5, size=(m, n)) * (rng.uniform(size=(m, n)) < rng.uniform(0.2, 0.9))
        want = _sympy_factors(A)
        assert list(H.smith_invariant_factors(A)) == want
        assert list(H.smith_invariant_factors(A, presimplify=False)) == want   # device path only


def test_ranks_mod_p_match_python_elimination_in_int32_and_int64():
    rng = np.random.default_rng(1)
    for _ in range(30):
        m, n = rng.integers(1, 9, size=2)
        A = rng.integers(-6, 7, size=(m, n))
        for p in (2, 3, 7, 46337, 2_147_483_647):        # 46337: int32; 2^31 - 1: int64
            assert H.rank_mod_p(A, p) == _rank_mod_p(np.mod(A, p), p)
            assert H.rank_mod_p(A, p, presimplify=False) == _rank_mod_p(np.mod(A, p), p)


def test_peeling_is_exact_and_arithmetic_free():
    A = sp.csr_matrix(np.array([[1, 0, 2], [1, 1, 0], [0, 2, 2], [0, 0, 1]]))
    k, R = H.peel_unit_singletons(A)
    assert k >= 1
    assert [1] * k + _sympy_factors(R.toarray()) == _sympy_factors(A.toarray())


def test_the_overflow_guard_hands_large_entries_to_the_exact_finish():
    """Eliminating the unit at (0, 0) creates 1 - 2^60: past the int64-safe bound."""
    big = 2 ** 30
    A = np.array([[1, big, 3], [big, 1, 5], [7, 11, 2 * big]], dtype=np.int64)
    res = H.smith_invariant_factors(A, presimplify=False, return_details=True)
    assert res.factors == _sympy_factors(A)
    assert res.residual_shape != (0, 0)               # the CPU finish was needed


def test_budget_and_argument_refusals():
    A = np.random.default_rng(2).integers(-3, 4, size=(40, 40))
    with pytest.raises(H.DenseBudgetExceeded):
        H.smith_invariant_factors(A, presimplify=False, max_dense_bytes=1024)
    with pytest.raises(ValueError):
        H.rank_mod_p(A, 4)                            # not a field
    with pytest.raises(ValueError):
        H.rank_mod_p(A, 2_147_483_659)                # prime, but above 2^31
    with pytest.raises(ValueError):
        H.smith_invariant_factors(np.array([[0.5, 1.0]]))


# ───────────────────────────────────────────────────────────── known spaces

def test_integral_homology_of_known_spaces():
    rp2 = SimplicialComplex.from_simplices(RP2)
    assert H.gpu_homology(rp2) == {0: (1, []), 1: (0, [2]), 2: (0, [])}
    torus = SimplicialComplex.from_simplices(TORUS)
    assert H.gpu_homology(torus) == {0: (1, []), 1: (2, []), 2: (1, [])}
    s3 = SimplicialComplex.from_simplices([[j for j in range(5) if j != i] for i in range(5)])
    assert H.gpu_homology(s3) == {0: (1, []), 1: (0, []), 2: (0, []), 3: (1, [])}
    # CW Klein bottle: one vertex, edges a, b, face a b a^-1 b, so d2 = (0, 2)^T
    klein = _cw({1: [[0, 0]], 2: [[0], [2]]}, {0: 1, 1: 2, 2: 1})
    assert H.gpu_homology(klein) == {0: (1, []), 1: (1, [2]), 2: (0, [])}
    lens = _cw({1: [[0]], 2: [[3]], 3: [[0]]}, {0: 1, 1: 1, 2: 1, 3: 1})
    assert H.gpu_homology(lens) == {0: (1, []), 1: (0, [3]), 2: (0, []), 3: (1, [])}


def test_torsion_is_reported_as_invariant_factors():
    """Z/2 + Z/3 is Z/6: one invariant factor, not two elementary divisors."""
    cc = _cw({1: [[0, 0]], 2: [[2, 0], [0, 3]]}, {0: 1, 1: 2, 2: 2})
    assert H.gpu_homology(cc, 1) == (0, [6])


def test_universal_coefficients_tie_the_modular_and_integral_answers():
    rp2 = SimplicialComplex.from_simplices(RP2)
    lens = _cw({1: [[0]], 2: [[3]], 3: [[0]]}, {0: 1, 1: 1, 2: 1, 3: 1})
    for X in (rp2, lens):
        hz = H.gpu_homology(X)
        for p in (2, 3, 5):
            t = {d: sum(1 for f in tors if f % p == 0) for d, (_r, tors) in hz.items()}
            want = {d: r + t[d] + t.get(d - 1, 0) for d, (r, _t) in hz.items()}
            assert H.gpu_betti_numbers(X, p=p) == want, (p, hz)


def test_composite_modulus_goes_through_the_universal_coefficient_theorem():
    lens = _cw({1: [[0]], 2: [[3]], 3: [[0]]}, {0: 1, 1: 1, 2: 1, 3: 1}, ring="Z/6Z")
    assert lens.homology(backend="gpu") == lens.homology(backend="python")


# ───────────────────────────────────────────── the backend="gpu" hook and the screen

@pytest.mark.parametrize("ring", ["Z", "Q", "Z/2Z", "Z/3Z"])
def test_gpu_backend_equals_python_backend(ring):
    for faces in (RP2, TORUS):
        base = SimplicialComplex.from_simplices(faces)
        sc = SimplicialComplex(simplices=base._simplices_table, coefficient_ring=ring)
        assert sc.homology(backend="gpu") == sc.homology(backend="python")
        assert sc.homology(1, backend="gpu:cpu") == sc.homology(1, backend="python")
    union = SimplicialComplex.concatenate([SimplicialComplex.from_simplices(RP2),
                                           SimplicialComplex.from_simplices(TORUS)])
    assert union.homology(backend="gpu") == {0: (2, []), 1: (2, [2]), 2: (1, [])}
    assert union.betti_numbers(backend="gpu") == {0: 2, 1: 2, 2: 1}


def test_torsion_prime_screen_detects_only_what_it_can_prove():
    screen = H.torsion_prime_screen(SimplicialComplex.from_simplices(RP2))
    assert screen.detected == {1: {2: 1}}                       # 2-torsion in H_1
    assert H.torsion_prime_screen(SimplicialComplex.from_simplices(TORUS)).detected == {}


def test_inputs_are_never_mutated():
    """Reducing modulo p drops entries; that must happen on a copy, not the caller's matrix."""
    d2 = sp.csr_matrix(np.array([[3, 0], [0, 6]], dtype=np.int64))
    before = (d2.data.copy(), d2.indices.copy(), d2.indptr.copy())
    assert H.rank_mod_p(d2, 3) == 0 and H.rank_mod_p(d2, 5) == 2
    H.smith_invariant_factors(d2)
    H.peel_unit_singletons(d2)
    after = (d2.data, d2.indices, d2.indptr)
    assert all(np.array_equal(a, b) for a, b in zip(before, after))


def test_a_device_without_the_ops_falls_back_to_the_cpu_exactly():
    """PyTorch's 'meta' device implements no data-dependent op: a stand-in for an MPS gap."""
    A = np.array([[1, 2, 0], [3, 4, 5], [0, 6, 7]])     # det = -44
    with pytest.warns(RuntimeWarning, match="recomputing on the CPU"):
        assert H.rank_mod_p(A, 5, device="meta", presimplify=False) == 3
    with pytest.warns(RuntimeWarning, match="recomputing on the CPU"):
        res = H.smith_invariant_factors(A, device="meta", presimplify=False, return_details=True)
    assert res.factors == [1, 1, 44] and res.device == "cpu"
