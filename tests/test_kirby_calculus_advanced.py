"""Advanced tests for Kirby calculus operations and 4-manifold surgery moves.

Overview:
    This suite validates the implementation of Kirby moves—specifically handle 
    slides and blow-ups/blow-downs—on link diagrams representing 4-manifolds. 
    It ensures that these moves correctly transform the intersection form and 
    associated invariants like the signature.

Key Concepts:
    - **Kirby Diagram**: A framed link in S³ representing a 4-manifold (via 2-handle attachment).
    - **Handle Slide**: Moving one handle over another, which corresponds to row/column operations on the linking matrix.
    - **Blow-up (σ-process)**: Adding a ±1-framed unknot linked with nothing, which changes the signature by ±1.
"""
import numpy as np
import pytest
from pysurgery.core.kirby_calculus import KirbyDiagram
from pysurgery.core.exceptions import KirbyMoveError


def test_hopf_link_slide():
    """Verify the transformation of the linking matrix during a handle slide on a Hopf link.

    What is Being Computed?:
        The effect of sliding handle 0 over handle 1 in a Hopf link diagram.

    Algorithm:
        1. Initialize a 0-framed Hopf link linking matrix [[0, 1], [1, 0]].
        2. Perform a handle slide of handle 0 over handle 1.
        3. Verify the new linking matrix is [[2, 1], [1, 0]] and framing is [2, 0].

    Preserved Invariants:
        - The underlying 4-manifold topology (handle slides are diffeomorphism invariants).
    """
    linking = np.array([[0, 1], [1, 0]])
    framings = np.array([0, 0])
    diagram = KirbyDiagram(linking_matrix=linking, framings=framings)

    new_diag = diagram.handle_slide(source_idx=0, target_idx=1)

    expected = np.array([[2, 1], [1, 0]])
    assert np.array_equal(new_diag.linking_matrix, expected)
    assert np.array_equal(new_diag.framings, np.array([2, 0]))


def test_kirby_blowup_signature():
    """Verify that Kirby blow-ups correctly shift the signature of the intersection form.

    What is Being Computed?:
        The signature of the intersection form after performing a positive or negative blow-up.

    Algorithm:
        1. Initialize a trivial Kirby diagram.
        2. Perform a positive blow-up (sign=1) and assert signature increases by 1.
        3. Perform a negative blow-up (sign=-1) and assert signature decreases by 1.

    Preserved Invariants:
        - Homotopy type (up to stabilization by CP² or CP²-bar).
    """
    linking = np.array([[0]])
    diag = KirbyDiagram(linking_matrix=linking, framings=np.array([0]))

    diag_plus = diag.blow_up(sign=1)
    assert diag_plus.extract_intersection_form().signature() == 1
    assert diag_plus.extract_intersection_form().rank() == 1

    diag_minus = diag.blow_up(sign=-1)
    assert diag_minus.extract_intersection_form().signature() == -1
    assert diag_minus.extract_intersection_form().rank() == 1


def test_invalid_kirby_moves():
    """Ensure that illegal Kirby moves are caught and raise KirbyMoveError.

    Algorithm:
        1. Attempt to blow up with an invalid sign (not 1 or -1).
        2. Attempt to slide a handle over itself.
        3. Assert KirbyMoveError is raised in both cases.
    """
    linking = np.array([[0, 1], [1, 0]])
    diagram = KirbyDiagram(linking_matrix=linking, framings=np.array([0, 0]))

    with pytest.raises(KirbyMoveError):
        diagram.blow_up(sign=2)

    with pytest.raises(KirbyMoveError):
        diagram.handle_slide(source_idx=0, target_idx=0)
