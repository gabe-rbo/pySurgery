"""Tests for Handle Decomposition & Framed Manifold Theory.

Overview:
    This suite validates the discrete Morse theory translation from CW complexes
    to handle decompositions. It also verifies algebraic tracking during framed
    handle attachment and validates intersection form extraction.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pysurgery.topology.complexes import CWComplex
from pysurgery.manifolds.handle_decompositions import (
    HandleDecomposition,
    cw_complex_to_handle_decomposition,
)
from pysurgery.core.exceptions import KirbyMoveError


def test_handle_decomposition_attachment():
    """Verify handle attachment and dynamic boundary matrix updates."""
    hd = HandleDecomposition(handles=[])
    
    # Attach 0-handle
    h0 = hd.attach_handle(0)
    assert h0.index == 0
    assert h0.cell_id == 0
    assert len(hd.handles) == 1
    
    # Attach another 0-handle
    h0_1 = hd.attach_handle(0)
    assert h0_1.index == 0
    assert h0_1.cell_id == 1
    
    # Attach 1-handle connecting them (boundary [1, -1])
    h1 = hd.attach_handle(1, boundary_vector=np.array([1, -1]))
    assert h1.index == 1
    assert h1.cell_id == 0
    assert hd.boundaries[1].shape == (2, 1)
    assert np.array_equal(hd.boundaries[1].toarray().flatten(), [1, -1])
    
    # Attach 2-handle with framing
    h2 = hd.attach_handle(2, boundary_vector=np.array([0]), framing=5)
    assert h2.index == 2
    assert h2.cell_id == 0
    assert h2.framing == 5
    
    # Check intersection form
    q = hd.get_intersection_form()
    assert q.shape == (1, 1)
    assert q[0, 0] == 5


def test_missing_framing_error():
    """Verify KirbyMoveError triggers on invalid topological operations."""
    hd = HandleDecomposition(handles=[])
    hd.attach_handle(0)
    hd.attach_handle(1, boundary_vector=np.array([0]))
    # Attach 2-handle WITHOUT framing
    hd.attach_handle(2, boundary_vector=np.array([0]))
    
    with pytest.raises(KirbyMoveError, match="missing a framing"):
        hd.get_intersection_form()


def test_cw_to_handle_translation():
    """Verify discrete Morse translation from CW complex to Handle Decomposition."""
    # Build a simple CW complex (e.g. 1 vertex, 1 edge, 1 face)
    # This simulates a topological space that can be simplified.
    # 0-cells: 1
    # 1-cells: 2
    # 2-cells: 1
    cells = {0: 1, 1: 2, 2: 1}
    
    # d1 maps 2 edges to 1 vertex. Say both edges are loops [0] -> [0]
    d1 = csr_matrix(np.zeros((1, 2), dtype=np.int64))
    
    # d2 maps 1 face to 2 edges. Say boundary is e1 + e2.
    d2 = csr_matrix(np.array([[1], [1]], dtype=np.int64))
    
    cw = CWComplex(cells=cells, attaching_maps={1: d1, 2: d2}, dimensions=[0, 1, 2])
    
    # In discrete Morse theory, one 1-cell and the 2-cell will pair up (since d2 incidence is 1)
    # This leaves 1 vertex (0-handle) and 1 edge (1-handle)
    hd = cw_complex_to_handle_decomposition(cw)
    
    assert len(hd.handles) == 2
    indices = [h.index for h in hd.handles]
    assert indices == [0, 1]
    
    # Verify boundaries are correctly isolated
    assert 1 in hd.boundaries
    assert hd.boundaries[1].shape == (1, 1)
    assert hd.boundaries[1][0, 0] == 0


def test_intersection_form_linking_data():
    """Verify intersection form computation with explicit linking numbers."""
    hd = HandleDecomposition(handles=[])
    hd.attach_handle(0)
    
    # Attach two 2-handles to the 0-handle
    h2_0 = hd.attach_handle(2, boundary_vector=np.array([]), framing=2)
    hd.attach_handle(2, boundary_vector=np.array([]), framing=-3)
    
    # Add linking data manually simulating attaching_sphere_data
    h2_0.attaching_sphere_data = {"linking": {1: 4}}
    
    q = hd.get_intersection_form()
    assert q.shape == (2, 2)
    assert q[0, 0] == 2
    assert q[1, 1] == -3
    assert q[0, 1] == 4
    assert q[1, 0] == 4
