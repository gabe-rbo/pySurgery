"""Tests for Algebraic Surgery and the Surgery Exact Sequence.

Overview:
    This suite verifies the implementation of the assembly map, surgery 
    obstructions, and the evaluation of the surgery exact sequence. It focuses
    on the mapping between normal invariants and Wall groups (L-groups).

Key Concepts:
    - **Surgery Obstruction**: An element in the Wall group Lₙ(π) that determines
      if a normal map is cobordant to a homotopy equivalence.
    - **Assembly Map**: The map A: Hₙ(Bπ; 𝕃) → Lₙ(π) in the surgery exact sequence.
    - **Structure Set S(M)**: The set of manifolds homotopy equivalent to M.
    - **Normal Invariants**: The set [M, G/TOP] representing normal maps.
"""
import numpy as np
from pysurgery.algebraic_poincare import AlgebraicPoincareComplex
from pysurgery.core.complexes import ChainComplex
from pysurgery.algebraic_surgery import AlgebraicSurgeryComplex
from pysurgery.core.intersection_forms import IntersectionForm


def test_assembly_map():
    """Verify the surgery assembly map for a trivial fundamental group.

    What is Being Computed?:
        The assembly map A: H₄(pt; 𝕃) → L₄(1) ≅ ℤ.

    Algorithm:
        1. Construct a trivial AlgebraicPoincareComplex.
        2. Define a hyperbolic intersection form (signature 0).
        3. Compute the assembly map value and verify it is 0.

    Preserved Invariants:
        - The L₄(1) obstruction is determined by signature(form)/8.
    """
    cc = ChainComplex(boundaries={}, dimensions=[0, 1], cells={0: 1, 1: 0})
    apc1 = AlgebraicPoincareComplex(
        chain_complex=cc, fundamental_class=np.array([1]), dimension=4
    )
    apc2 = AlgebraicPoincareComplex(
        chain_complex=cc, fundamental_class=np.array([1]), dimension=4
    )

    asc = AlgebraicSurgeryComplex(domain=apc1, codomain=apc2, degree=1)

    matrix = np.array([[0, 1], [1, 0]])
    form = IntersectionForm(matrix=matrix, dimension=4)
    obstruction = asc.assembly_map(pi_1_group="1", form=form)

    # signature is 0, so the obstruction class is 0.
    assert obstruction == 0


def test_assembly_map_result_typed():
    """Verify the structured result of the assembly map computation.

    What is Being Computed?:
        A SurgeryResult object containing the obstruction value and metadata.

    Algorithm:
        1. Call assembly_map_result with a hyperbolic form.
        2. Verify computable=True, exact=True, and value=0.
    """
    cc = ChainComplex(boundaries={}, dimensions=[0, 1], cells={0: 1, 1: 0})
    apc1 = AlgebraicPoincareComplex(
        chain_complex=cc, fundamental_class=np.array([1]), dimension=4
    )
    apc2 = AlgebraicPoincareComplex(
        chain_complex=cc, fundamental_class=np.array([1]), dimension=4
    )
    asc = AlgebraicSurgeryComplex(domain=apc1, codomain=apc2, degree=1)

    form = IntersectionForm(matrix=np.array([[0, 1], [1, 0]]), dimension=4)
    res = asc.assembly_map_result(pi_1_group="1", form=form)
    assert res.computable
    assert res.exact
    assert res.value == 0


def test_evaluate_structure_set_typed():
    """Verify the evaluation of the surgery exact sequence.

    What is Being Computed?:
        The SurgeryExactSequence object for a 5D manifold with trivial π₁.

    Algorithm:
        1. Define a 5D APC and ASC.
        2. Call evaluate_structure_set to compute the sequence components.
        3. Verify the presence of normal invariants.

    Preserved Invariants:
        - The exactness of the sequence: S(M) → [M, G/TOP] → Lₙ(π).
    """
    cc = ChainComplex(boundaries={}, dimensions=[0], cells={0: 1})
    apc = AlgebraicPoincareComplex(
        chain_complex=cc, fundamental_class=np.array([1]), dimension=5
    )
    asc = AlgebraicSurgeryComplex(domain=apc, codomain=apc, degree=1)
    seq = asc.evaluate_structure_set(cc, fundamental_group="1")
    assert seq.computable
    assert seq.exact
    assert seq.normal_invariants is not None

def test_assembly_map_nonzero_signature():
    """Verify the surgery obstruction for a non-trivial signature (E₈ form).

    What is Being Computed?:
        The surgery obstruction in L₄(1) ≅ ℤ for the E₈ form.

    Algorithm:
        1. Define the 8x8 E₈ intersection matrix.
        2. Compute signature(E₈) = 8.
        3. Verify that the assembly map returns 1 (since 8/8 = 1).

    Preserved Invariants:
        - Signature / 8 is the complete invariant for L₄(1).
    """
    cc = ChainComplex(boundaries={}, dimensions=[0, 1], cells={0: 1, 1: 0})
    apc1 = AlgebraicPoincareComplex(
        chain_complex=cc, fundamental_class=np.array([1]), dimension=4
    )
    apc2 = AlgebraicPoincareComplex(
        chain_complex=cc, fundamental_class=np.array([1]), dimension=4
    )
    asc = AlgebraicSurgeryComplex(domain=apc1, codomain=apc2, degree=1)

    # E8 has signature 8. 8/8 = 1.
    E8 = np.array([
        [2, 0, -1, 0, 0, 0, 0, 0],
        [0, 2, 0, -1, 0, 0, 0, 0],
        [-1, 0, 2, -1, 0, 0, 0, 0],
        [0, -1, -1, 2, -1, 0, 0, 0],
        [0, 0, 0, -1, 2, -1, 0, 0],
        [0, 0, 0, 0, -1, 2, -1, 0],
        [0, 0, 0, 0, 0, -1, 2, -1],
        [0, 0, 0, 0, 0, 0, -1, 2]
    ])
    form = IntersectionForm(matrix=E8, dimension=4)
    obstruction = asc.assembly_map(pi_1_group="1", form=form)

    # In our model for L4(1), it returns signature/8
    assert obstruction == 1
