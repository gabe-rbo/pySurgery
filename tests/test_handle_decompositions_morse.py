"""Tests for the exact discrete Morse boundary in handle_decompositions.

These verify that ``cw_complex_to_handle_decomposition`` produces the true
Morse boundary on the critical chain complex via the integer Schur
complement, on three reference manifolds and one CW complex that
genuinely exercises the Schur reduction.
"""

import numpy as np
from scipy.sparse import csr_matrix

from pysurgery.topology.complexes import CWComplex
from pysurgery.manifolds.handle_decompositions import (
    cw_complex_to_handle_decomposition,
)


def test_morse_sphere_s2_minimal_cw():
    """S^2 with the minimal CW structure: one 0-handle, one 2-handle, exact."""
    cw = CWComplex(
        cells={0: 1, 2: 1},
        attaching_maps={},
        dimensions=[0, 2],
    )
    hd = cw_complex_to_handle_decomposition(cw)

    assert sorted(h.index for h in hd.handles) == [0, 2]
    assert hd.exact is True
    assert "Schur complement" in hd.notes
    for d in (1, 2):
        if d in hd.boundaries:
            assert np.all(hd.boundaries[d].toarray() == 0)


def test_morse_torus_t2_minimal_cw():
    """T^2 minimal CW (1, 2, 1): boundary is zero over Z (commutator)."""
    cw = CWComplex(
        cells={0: 1, 1: 2, 2: 1},
        attaching_maps={
            1: csr_matrix(np.zeros((1, 2), dtype=np.int64)),
            2: csr_matrix(np.zeros((2, 1), dtype=np.int64)),
        },
        dimensions=[0, 1, 2],
    )
    hd = cw_complex_to_handle_decomposition(cw)

    indices = sorted(h.index for h in hd.handles)
    assert indices == [0, 1, 1, 2]
    assert hd.exact is True
    assert np.all(hd.boundaries[1].toarray() == 0)
    assert np.all(hd.boundaries[2].toarray() == 0)


def test_morse_real_projective_plane_rp2_minimal_cw():
    """RP^2 minimal CW (1, 1, 1): ∂^M_2 = ±2 on the lone critical 2-cell."""
    cw = CWComplex(
        cells={0: 1, 1: 1, 2: 1},
        attaching_maps={
            1: csr_matrix(np.array([[0]], dtype=np.int64)),
            2: csr_matrix(np.array([[2]], dtype=np.int64)),
        },
        dimensions=[0, 1, 2],
    )
    hd = cw_complex_to_handle_decomposition(cw)

    indices = sorted(h.index for h in hd.handles)
    assert indices == [0, 1, 2]
    assert hd.exact is True
    assert hd.boundaries[1].toarray().shape == (1, 1)
    assert hd.boundaries[1].toarray()[0, 0] == 0
    assert hd.boundaries[2].toarray().shape == (1, 1)
    assert abs(hd.boundaries[2].toarray()[0, 0]) == 2


def test_morse_schur_complement_cancels_indirect_path():
    """A CW where Schur reduction is non-trivial: the indirect gradient
    path through a matched pair cancels the direct boundary contribution.

    Setup:
        Two 0-cells v0, v1. Two 1-cells e_a, e_b (both v0 → v1). Two
        2-cells F (boundary e_a - e_b) and G (boundary 2 e_a - 2 e_b).

    Greedy matches (e_a, F); critical cells are {v0, v1, e_b, G}. The
    naive projection would record ∂(e_b) under G as -2, but the true
    Morse boundary cancels this via the gradient path G → e_a → F → e_b
    (multiplicity +2). The Schur complement recovers ∂^M_2(G) = 0.
    """
    cells = {0: 2, 1: 2, 2: 2}
    d1 = csr_matrix(np.array([
        [-1, -1],
        [1, 1],
    ], dtype=np.int64))
    d2 = csr_matrix(np.array([
        [1, 2],
        [-1, -2],
    ], dtype=np.int64))

    cw = CWComplex(
        cells=cells,
        attaching_maps={1: d1, 2: d2},
        dimensions=[0, 1, 2],
    )
    hd = cw_complex_to_handle_decomposition(cw)

    indices = sorted(h.index for h in hd.handles)
    assert indices == [0, 0, 1, 2]
    assert hd.exact is True

    assert hd.boundaries[2].toarray().shape == (1, 1)
    assert hd.boundaries[2].toarray()[0, 0] == 0

    # Sanity: ∂_1 restricted to (crit_0, crit_1) is the e_b column = (-1, 1).
    assert hd.boundaries[1].toarray().shape == (2, 1)
    assert sorted(hd.boundaries[1].toarray().flatten().tolist()) == [-1, 1]


def test_morse_existing_translation_test_still_passes():
    """Regression: the original CW (1, 2, 1) translation case should keep
    producing 1 0-handle, 1 1-handle, ∂^M_1 = 0, and now mark exact=True.
    """
    cells = {0: 1, 1: 2, 2: 1}
    d1 = csr_matrix(np.zeros((1, 2), dtype=np.int64))
    d2 = csr_matrix(np.array([[1], [1]], dtype=np.int64))
    cw = CWComplex(
        cells=cells,
        attaching_maps={1: d1, 2: d2},
        dimensions=[0, 1, 2],
    )
    hd = cw_complex_to_handle_decomposition(cw)

    assert len(hd.handles) == 2
    assert sorted(h.index for h in hd.handles) == [0, 1]
    assert hd.exact is True
    assert hd.boundaries[1].shape == (1, 1)
    assert hd.boundaries[1][0, 0] == 0
