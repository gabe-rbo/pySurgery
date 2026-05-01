"""Property-based tests for surgery invariants and intersection forms.

Overview:
    This suite validates the consistency of surgery-theoretic invariants, 
    specifically intersection forms, signatures, and Pontryagin classes. 
    It tests foundational theorems like Hirzebruch signature theorem and 
    stability under hyperbolic stabilization using generated symmetric matrices.

Key Concepts:
    - **Intersection Form (Q)**: A symmetric bilinear form on the middle homology of a manifold.
    - **Hirzebruch Signature Theorem**: Relates the signature (σ) to the first Pontryagin class (p₁).
    - **Hyperbolic Stabilization**: Adding the hyperbolic form H to a quadratic form (preserves signature).
"""

from hypothesis import given, settings, strategies as st
import numpy as np
from pysurgery.core.intersection_forms import IntersectionForm
from pysurgery.core.characteristic_classes import extract_pontryagin_p1, verify_hirzebruch_signature

@st.composite
def symmetric_matrices(draw, min_size=1, max_size=8):
    """Hypothesis strategy for generating symmetric integer matrices.

    What is Being Computed?:
        A square symmetric matrix with integer entries.

    Algorithm:
        1. Draw an integer size.
        2. Draw elements for the upper triangular part.
        3. Mirror elements to ensure symmetry (M = Mᵀ).

    Returns:
        np.ndarray: A symmetric (size, size) matrix.
    """
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    # Generate upper triangular part
    data = draw(st.lists(
        st.integers(min_value=-10, max_value=10),
        min_size=(size * (size + 1)) // 2,
        max_size=(size * (size + 1)) // 2
    ))
    matrix = np.zeros((size, size), dtype=np.int64)
    idx = 0
    for i in range(size):
        for j in range(i, size):
            matrix[i, j] = matrix[j, i] = data[idx]
            idx += 1
    return matrix

@settings(max_examples=100, deadline=None)
@given(symmetric_matrices())
def test_intersection_form_symmetry_property(matrix):
    """Verify that IntersectionForm enforces and maintains matrix symmetry.

    What is Being Computed?:
        Symmetry check for the underlying matrix of an IntersectionForm.

    Algorithm:
        1. Initialize IntersectionForm with a generated symmetric matrix.
        2. Assert that the internal matrix representation remains perfectly symmetric.

    Preserved Invariants:
        - Symmetry of the bilinear form.
    """
    form = IntersectionForm(matrix=matrix, dimension=4)
    m = form.matrix
    assert np.all(m == m.T)

@settings(max_examples=100, deadline=None)
@given(symmetric_matrices(min_size=1, max_size=6))
def test_hirzebruch_signature_theorem_property(matrix):
    """Verify the Hirzebruch Signature Theorem identity (3σ = p₁) for 4-manifolds.

    What is Being Computed?:
        The relation between the signature of the intersection form and the first 
        Pontryagin class p₁(M).

    Algorithm:
        1. Construct an IntersectionForm.
        2. Compute its signature σ via eigenvalue analysis or SNF-based methods.
        3. Extract the p₁ class from the form metadata.
        4. Assert that the theorem identity 3σ = p₁ holds exactly.

    Preserved Invariants:
        - Signature-Pontryagin identity for oriented closed 4-manifolds.
    """
    # This identity is a definition/verification pair in characteristic_classes.py
    # and should always be consistent by construction or theorem.
    form = IntersectionForm(matrix=matrix, dimension=4)
    sig = form.signature()
    p1 = extract_pontryagin_p1(form)
    
    assert 3 * sig == p1
    assert verify_hirzebruch_signature(form, p1)

@settings(max_examples=50, deadline=None)
@given(symmetric_matrices(min_size=2, max_size=4))
def test_hyperbolic_stabilization_signature(matrix):
    """Verify that signature is invariant under hyperbolic stabilization.

    What is Being Computed?:
        The signature of a stabilized form Q ⊕ H, where H = [[0, 1], [1, 0]].

    Algorithm:
        1. Compute signature of original form Q.
        2. Direct-sum Q with the hyperbolic form H.
        3. Compute signature of the stabilized form.
        4. Assert signature(Q ⊕ H) == signature(Q).

    Preserved Invariants:
        - Signature is a stable invariant under hyperbolic sum (Witt group invariant).
    """
    form = IntersectionForm(matrix=matrix, dimension=4)
    sig_orig = form.signature()
    
    # Add H = [[0, 1], [1, 0]]
    h = np.array([[0, 1], [1, 0]], dtype=np.int64)
    size = matrix.shape[0]
    new_matrix = np.zeros((size + 2, size + 2), dtype=np.int64)
    new_matrix[:size, :size] = matrix
    new_matrix[size:, size:] = h
    
    form_stabilized = IntersectionForm(matrix=new_matrix, dimension=4)
    assert form_stabilized.signature() == sig_orig
