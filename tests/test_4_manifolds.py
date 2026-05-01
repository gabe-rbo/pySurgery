"""Tests for 4-manifold invariants and topological classification.

Overview:
    This suite verifies the computation of intersection forms, characteristic
    classes (Stiefel-Whitney and Pontryagin), and Freedman's classification for
    simply-connected 4-manifolds.

Key Concepts:
    - **Intersection Form**: A symmetric bilinear form on H₂(M; ℤ) that classifies 4-manifolds.
    - **Signature and Rank**: Fundamental invariants derived from the intersection form.
    - **Kirby-Siebenmann Invariant**: Obstruction to being a smooth manifold (PL/Top difference).
    - **Characteristic Classes**: w₂ and p₁ invariants computed from the intersection form.
"""
import numpy as np
from pysurgery.core.intersection_forms import IntersectionForm
from pysurgery.core.characteristic_classes import (
    extract_stiefel_whitney_w2 as wu_class,
    extract_pontryagin_p1 as pontryagin_class,
)
from pysurgery.homeomorphism import analyze_homeomorphism_4d


def test_CP2():
    """Verify invariants for the complex projective plane CP².

    What is Being Computed?:
        Intersection form invariants (signature, parity) and characteristic classes.

    Algorithm:
        1. Define the 1x1 intersection form [1].
        2. Compute signature, type, p₁, and w₂.

    Preserved Invariants:
        - Signature(CP²) = 1.
        - Type I (odd).
        - p₁(CP²) = 3.
        - w₂(CP²) = 1 (non-zero).
    """
    Q = np.array([[1]])
    form = IntersectionForm(matrix=Q, dimension=4)
    assert form.signature() == 1
    assert not form.is_even()
    assert form.type() == "I"
    assert pontryagin_class(form) == 3
    assert np.array_equal(wu_class(form), np.array([1]))


def test_anti_CP2():
    """Verify invariants for CP² with reversed orientation (anti-CP²).

    What is Being Computed?:
        Invariants for CP² with intersection form [-1].

    Algorithm:
        1. Define the 1x1 intersection form [-1].
        2. Compute signature, type, p₁, and w₂.

    Preserved Invariants:
        - Signature(anti-CP²) = -1.
        - Type I (odd).
        - p₁(anti-CP²) = -3.
        - w₂(anti-CP²) = 1 (orientation independent).
    """
    Q = np.array([[-1]])
    form = IntersectionForm(matrix=Q, dimension=4)
    assert form.signature() == -1
    assert not form.is_even()
    assert form.type() == "I"
    assert pontryagin_class(form) == -3
    assert np.array_equal(wu_class(form), np.array([1]))


def test_S2_times_S2():
    """Verify invariants for the product space S² × S².

    What is Being Computed?:
        Invariants for the hyperbolic form H = [[0, 1], [1, 0]].

    Algorithm:
        1. Define the 2x2 intersection form H.
        2. Compute signature, type, p₁, and w₂.

    Preserved Invariants:
        - Signature(S² × S²) = 0.
        - Type II (even).
        - p₁(S² × S²) = 0.
        - w₂(S² × S²) = [0, 0] (even).
    """
    Q = np.array([[0, 1], [1, 0]])
    form = IntersectionForm(matrix=Q, dimension=4)
    assert form.signature() == 0
    assert form.is_even()
    assert form.type() == "II"
    assert pontryagin_class(form) == 0
    assert np.array_equal(wu_class(form), np.array([0, 0]))


def test_K3_surface():
    """Verify invariants for the K3 surface.

    What is Being Computed?:
        Invariants for the intersection form 2(-E₈) ⊕ 3H.

    Algorithm:
        1. Construct the 22x22 K3 intersection matrix using block diagonal sum.
        2. Compute rank, signature, type, p₁, and w₂.

    Preserved Invariants:
        - Rank(K3) = 22.
        - Signature(K3) = -16.
        - Type II (even).
        - p₁(K3) = -48.
        - w₂(K3) = 0 (spin).
    """
    e8 = np.array(
        [
            [2, -1, 0, 0, 0, 0, 0, 0],
            [-1, 2, -1, 0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0, 0, 0, -1],
            [0, 0, -1, 2, -1, 0, 0, 0],
            [0, 0, 0, -1, 2, -1, 0, 0],
            [0, 0, 0, 0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0, -1, 2, 0],
            [0, 0, -1, 0, 0, 0, 0, 2],
        ]
    )
    h = np.array([[0, 1], [1, 0]])
    from scipy.linalg import block_diag

    k3_matrix = block_diag(-e8, -e8, h, h, h)
    form = IntersectionForm(matrix=k3_matrix, dimension=4)

    assert form.rank() == 22
    assert form.signature() == -16
    assert form.is_even()
    assert form.type() == "II"
    assert pontryagin_class(form) == -48
    assert np.all(wu_class(form) == 0)


def test_K3_homeomorphism():
    """Test Freedman's homeomorphism classification for K3 surfaces.

    What is Being Computed?:
        Homeomorphism between two K3 surfaces (identical intersection forms).

    Algorithm:
        1. Construct two identical K3 intersection forms.
        2. Use analyze_homeomorphism_4d with simply_connected=True.

    Preserved Invariants:
        - Freedman's Theorem states that (rank, signature, type, KS-invariant)
          is a complete homeomorphism invariant for simply-connected 4-manifolds.
    """
    e8 = np.array(
        [
            [2, -1, 0, 0, 0, 0, 0, 0],
            [-1, 2, -1, 0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0, 0, 0, -1],
            [0, 0, -1, 2, -1, 0, 0, 0],
            [0, 0, 0, -1, 2, -1, 0, 0],
            [0, 0, 0, 0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0, -1, 2, 0],
            [0, 0, -1, 0, 0, 0, 0, 2],
        ]
    )
    h = np.array([[0, 1], [1, 0]])
    from scipy.linalg import block_diag

    k3_matrix = block_diag(-e8, -e8, h, h, h)
    form1 = IntersectionForm(matrix=k3_matrix, dimension=4)
    form2 = IntersectionForm(matrix=k3_matrix.copy(), dimension=4)

    is_homeo, reason = analyze_homeomorphism_4d(
        form1, form2, ks1=0, ks2=0, simply_connected=True
    )
    assert is_homeo
    assert "SUCCESS" in reason
