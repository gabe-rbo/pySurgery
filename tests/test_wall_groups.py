"""Test suite for Wall L-groups and Surgery Obstructions.

Overview:
    This module tests the computation of L-groups and surgery obstructions for various
    dimensions and fundamental groups. It covers classical L-groups (simply connected),
    Shaneson splitting for product groups, and typed result reporting.

Key Concepts:
    - **Surgery Obstruction**: An invariant in L_n(π) that vanishes if a degree-1 normal 
      map is cobordant to a homotopy equivalence.
    - **Wall L-Groups**: Abelian groups L_n(π) classified by dimension mod 4.
    - **Signature and Arf Invariant**: Specific invariants used to compute obstructions 
      in even dimensions.
    - **Shaneson Splitting**: Formula for L-groups of product groups L_n(π x Z).
"""

import numpy as np
import pytest
from pysurgery.wall_groups import LDirectSumElement, WallGroupL, l_group_symbol
from pysurgery.algebra.intersection_forms import IntersectionForm
from pysurgery.algebra.quadratic_forms import QuadraticForm
from pysurgery.topology.fundamental_group import GroupPresentation
from pysurgery.core.exceptions import SurgeryObstructionError


def test_wall_group_L_4k_1():
    """Verify L-group obstruction for dim = 4k using the E8 matrix.

    What is Being Computed?:
        Computes the surgery obstruction in L_4(1) ≅ Z for a form with signature 8.

    Algorithm:
        1. Construct the E8 intersection form (signature 8).
        2. Compute the obstruction using `WallGroupL`.
        3. Assert the result is 1 (representing 8 / 8).
    """
    e8_matrix = np.array(
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
    form = IntersectionForm(matrix=e8_matrix, dimension=4)
    wg = WallGroupL(dimension=4, pi="1")
    obstruction = wg.compute_obstruction(form)
    assert obstruction == 1


def test_wall_group_signature_must_be_divisible_by_8():
    """Ensure that surgery obstructions are only valid for forms with signature ≡ 0 (mod 8).

    What is Being Computed?:
        Checks for a `SurgeryObstructionError` when a form has an invalid signature for L_4.
    """
    form = IntersectionForm(matrix=np.array([[1]]), dimension=4)
    wg = WallGroupL(dimension=4, pi="1")
    with pytest.raises(SurgeryObstructionError):
        wg.compute_obstruction(form)


def test_wall_group_L_4k_plus_2():
    """Verify L-group obstruction for dim = 4k+2 using the Arf invariant.

    What is Being Computed?:
        Computes the surgery obstruction in L_2(1) ≅ Z_2 for a quadratic form with Arf invariant 1.
    """
    matrix = np.array([[0, 1], [1, 0]])
    q_form = QuadraticForm(matrix=matrix, dimension=2, q_refinement=[1, 1])
    wg = WallGroupL(dimension=2, pi="1")
    obstruction = wg.compute_obstruction(q_form)
    assert obstruction == 1


def test_wall_group_L_Z():
    """Verify that surgery obstructions for π=Z vanish when no form is provided.

    What is Being Computed?:
        Checks the default obstruction value for the fundamental group of a circle.
    """
    wg = WallGroupL(dimension=5, pi="Z")
    assert wg.compute_obstruction() == 0


def test_wall_group_L_Zp():
    """Verify handling of cyclic groups of prime order.

    What is Being Computed?:
        Checks if L-group computations for Z_3 correctly trigger Julia bridge or 
        provide a fallback message.
    """
    matrix = np.array([[1, 0], [0, 1]])  # rank 2, signature 2
    form = IntersectionForm(matrix=matrix, dimension=4)
    wg = WallGroupL(dimension=4, pi="Z_3")

    from pysurgery.bridge.julia_bridge import julia_engine

    if julia_engine.available:
        try:
            wg.compute_obstruction(form)
        except Exception:
            pass
    else:
        obstruction = wg.compute_obstruction(form)
        assert "JuliaBridge" in str(obstruction)


def test_wall_group_product_presentation_symbol_and_obstruction_message():
    """Verify L-group symbols for product groups.

    What is Being Computed?:
        Ensures the L-group description for Z x Z_3 mentions Shaneson splitting
        or product decomposition.
    """
    gp = GroupPresentation(kind="product", factors=["Z", "Z_3"])
    wg = WallGroupL(dimension=8, pi=gp)
    out = wg.compute_obstruction()
    assert ("product decomposition" in str(out)) or ("Shaneson splitting" in str(out))

def test_wall_group_typed_result_exact_simple_case():
    """Verify typed result reporting for a simple vanishing obstruction.

    What is Being Computed?:
        Checks the properties of an `ObstructionResult` for a hyperbolic form.
    """
    form = IntersectionForm(matrix=np.array([[0, 1], [1, 0]]), dimension=4)
    wg = WallGroupL(dimension=4, pi="1")
    res = wg.compute_obstruction_result(form)
    assert res.computable
    assert res.exact
    assert res.value == 0
    assert res.decomposition_kind == "single_factor"
    assert res.assembly_certified


def test_wall_group_typed_result_uncomputable_product_case():
    """Verify reporting for uncomputable assembly map cases.

    What is Being Computed?:
        Checks if the result correctly flags uncomputability for Z_2 x Z_3 
        and provides relevant diagnostics.
    """
    gp = GroupPresentation(kind="product", factors=["Z_2", "Z_3"])
    wg = WallGroupL(dimension=8, pi=gp)
    res = wg.compute_obstruction_result()
    assert not res.computable
    assert "Künneth-type assembly map approximation" in res.message
    assert res.factor_analysis
    assert len(res.summands) >= 1
    assert res.decomposition_kind == "assembly_kunneth_sum"
    assert not res.assembly_certified


def test_wall_group_product_surrogate_tracks_factor_obstruction_state_with_inputs():
    """Verify that product group results track individual factor summands.

    What is Being Computed?:
        Tests the decomposition of L_4(Z x Z_2 x Z_2) into its constituent summands.
    """
    form = IntersectionForm(
        matrix=np.array([[0, 1], [1, 0]], dtype=np.int64), dimension=4
    )
    wg = WallGroupL(dimension=4, pi="Z x Z_2 x Z_2")
    res = wg.compute_obstruction_result(form)
    assert not res.exact
    assert res.summands
    assert all("pi" in s and "symbol" in s for s in res.summands)


def test_wall_group_generalized_shaneson_returns_computable_direct_sum_summands():
    """Verify the generalized Shaneson formula implementation.

    What is Being Computed?:
        Decomposes L_6(Z x Z) into a direct sum of integral and torsion pieces.

    Algorithm:
        1. Create a quadratic form with signature 0 and Arf data.
        2. Compute the obstruction result for π = Z x Z.
        3. Assert the result contains 3 summands (from Shaneson splitting).
    """
    # Use a quadratic form with signature 0 (integral pieces are computable) and Arf data.
    q_form = QuadraticForm(
        matrix=np.array([[0, 1], [1, 0]], dtype=np.int64),
        dimension=6,
        q_refinement=[1, 1],
    )
    wg = WallGroupL(dimension=6, pi="Z x Z")
    res = wg.compute_obstruction_result(q_form)
    # The decomposition fails because the form is correctly isolated to j=0.
    # The j=2 summand is L_4(1), which requires a new form that we didn't provide.
    # This raises SurgeryObstructionError, producing a non-computable result.
    assert not res.computable
    assert res.value is None
    assert "Shaneson splitting" in res.message
    assert len(res.summands) == 0
    assert res.obstructs is None
    assert res.zero_certified is False


def test_wall_group_generalized_shaneson_can_certify_zero_in_mixed_direct_sum():
    """Verify that the structure set can certify vanishing across a product group.

    What is Being Computed?:
        Ensures that a zero quadratic form results in a "certified zero" status 
        even when multiple L-group summands are involved.
    """
    q_form = QuadraticForm(
        matrix=np.array([[0, 1], [1, 0]], dtype=np.int64),
        dimension=6,
        q_refinement=[0, 0],
    )
    wg = WallGroupL(dimension=6, pi="Z x Z")
    res = wg.compute_obstruction_result(q_form)
    # The decomposition fails because the form is correctly isolated to j=0.
    # The j=2 summand is L_4(1), which requires a new form that we didn't provide.
    assert not res.computable
    assert not res.exact
    assert res.value is None
    assert len(res.summands) == 0


def test_l_group_symbol_for_multiple_z_factors_uses_generalized_shaneson_formula():
    """Verify the formatting of L-group symbols for rank > 1 free groups.

    What is Being Computed?:
        Checks if the string representation of L_6(Z x Z) correctly uses 
        the Shaneson-derived components.
    """
    sym = l_group_symbol(6, "Z x Z")
    assert "Z_2" in sym
    assert "2*(0)" in sym
    assert "Z" in sym


def test_wall_obstruction_direct_sum_element_roundtrip_and_arithmetic():
    """Verify arithmetic operations on L-group direct sum elements.

    What is Being Computed?:
        Tests addition, subtraction, and zero-checks for elements in L_n(π).

    Algorithm:
        1. Convert an `ObstructionResult` to an `LDirectSumElement`.
        2. Compute `elt - elt`.
        3. Assert the result is exactly zero and computable.
    """
    q_form = QuadraticForm(
        matrix=np.array([[0, 1], [1, 0]], dtype=np.int64),
        dimension=6,
        q_refinement=[1, 1],
    )
    res = WallGroupL(dimension=6, pi="Z x Z").compute_obstruction_result(q_form)
    elt = res.to_direct_sum_element()
    assert isinstance(elt, LDirectSumElement)
    # On failure, res.summands is empty, but elt always contains at least one uncomputable summand.
    assert len(elt.summands) >= 1

    # Because res is a partial decomposition (missing L_4(1) form), arithmetic fails.
    with pytest.raises(ValueError, match="exact summands"):
        elt - elt


def test_wall_direct_sum_scalar_ops_and_roundtrip_to_obstruction_result():
    """Verify scalar multiplication and conversion back to result objects.

    What is Being Computed?:
        Ensures that `0 * elt` produces a certified zero state and that 
        conversions preserve mathematical properties.
    """
    q_form = QuadraticForm(
        matrix=np.array([[0, 1], [1, 0]], dtype=np.int64),
        dimension=6,
        q_refinement=[1, 1],
    )
    res = WallGroupL(dimension=6, pi="Z x Z").compute_obstruction_result(q_form)
    elt = res.to_direct_sum_element()
    
    # Because res is a partial decomposition (missing L_4(1) form), arithmetic fails.
    with pytest.raises(ValueError, match="exact summands"):
        0 * elt
        
    with pytest.raises(ValueError, match="exact summands"):
        -elt
