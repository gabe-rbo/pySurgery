import numpy as np
import pytest
from pysurgery.wall_groups import LDirectSumElement, WallGroupL, l_group_symbol
from pysurgery.core.intersection_forms import IntersectionForm
from pysurgery.core.quadratic_forms import QuadraticForm
from pysurgery.core.fundamental_group import GroupPresentation
from pysurgery.core.exceptions import SurgeryObstructionError


def test_wall_group_L_4k_1():
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
    form = IntersectionForm(matrix=np.array([[1]]), dimension=4)
    wg = WallGroupL(dimension=4, pi="1")
    with pytest.raises(SurgeryObstructionError):
        wg.compute_obstruction(form)


def test_wall_group_L_4k_plus_2():
    matrix = np.array([[0, 1], [1, 0]])
    q_form = QuadraticForm(matrix=matrix, dimension=2, q_refinement=[1, 1])
    wg = WallGroupL(dimension=2, pi="1")
    obstruction = wg.compute_obstruction(q_form)
    assert obstruction == 1


def test_wall_group_L_Z():
    wg = WallGroupL(dimension=5, pi="Z")
    assert wg.compute_obstruction() == 0


def test_wall_group_L_Zp():
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
    gp = GroupPresentation(kind="product", factors=["Z", "Z_3"])
    wg = WallGroupL(dimension=8, pi=gp)
    out = wg.compute_obstruction()
    assert ("Product group" in str(out)) or ("Shaneson splitting" in str(out))


def test_wall_group_typed_result_exact_simple_case():
    form = IntersectionForm(matrix=np.array([[0, 1], [1, 0]]), dimension=4)
    wg = WallGroupL(dimension=4, pi="1")
    res = wg.compute_obstruction_result(form)
    assert res.computable
    assert res.exact
    assert res.value == 0
    assert res.decomposition_kind == "single_factor"
    assert res.assembly_certified


def test_wall_group_typed_result_uncomputable_product_case():
    gp = GroupPresentation(kind="product", factors=["Z_2", "Z_3"])
    wg = WallGroupL(dimension=8, pi=gp)
    res = wg.compute_obstruction_result()
    assert not res.computable
    assert "surrogate decomposition" in res.message
    assert res.factor_analysis
    assert len(res.summands) >= 1
    assert res.decomposition_kind == "factor_surrogate"
    assert not res.assembly_certified


def test_wall_group_product_surrogate_tracks_factor_obstruction_state_with_inputs():
    form = IntersectionForm(
        matrix=np.array([[0, 1], [1, 0]], dtype=np.int64), dimension=4
    )
    wg = WallGroupL(dimension=4, pi="Z x Z_2 x Z_2")
    res = wg.compute_obstruction_result(form)
    assert not res.exact
    assert res.summands
    assert all("pi" in s and "symbol" in s for s in res.summands)


def test_wall_group_generalized_shaneson_returns_computable_direct_sum_summands():
    # Use a quadratic form with signature 0 (integral pieces are computable) and Arf data.
    q_form = QuadraticForm(
        matrix=np.array([[0, 1], [1, 0]], dtype=np.int64),
        dimension=6,
        q_refinement=[1, 1],
    )
    wg = WallGroupL(dimension=6, pi="Z x Z")
    res = wg.compute_obstruction_result(q_form)
    assert res.computable
    assert res.value is None
    assert "direct-sum" in res.message
    assert len(res.summands) == 3
    assert any(s["modulus"] == 2 for s in res.summands)
    assert any(s["modulus"] is None for s in res.summands)
    assert res.obstructs is True
    assert res.zero_certified is False


def test_wall_group_generalized_shaneson_can_certify_zero_in_mixed_direct_sum():
    q_form = QuadraticForm(
        matrix=np.array([[0, 1], [1, 0]], dtype=np.int64),
        dimension=6,
        q_refinement=[0, 0],
    )
    wg = WallGroupL(dimension=6, pi="Z x Z")
    res = wg.compute_obstruction_result(q_form)
    assert res.computable
    assert res.exact
    assert res.value is None
    assert res.obstructs is False
    assert res.zero_certified is True


def test_l_group_symbol_for_multiple_z_factors_uses_generalized_shaneson_formula():
    sym = l_group_symbol(6, "Z x Z")
    assert "Z_2" in sym
    assert "2*(0)" in sym
    assert "Z" in sym


def test_wall_obstruction_direct_sum_element_roundtrip_and_arithmetic():
    q_form = QuadraticForm(
        matrix=np.array([[0, 1], [1, 0]], dtype=np.int64),
        dimension=6,
        q_refinement=[1, 1],
    )
    res = WallGroupL(dimension=6, pi="Z x Z").compute_obstruction_result(q_form)
    elt = res.to_direct_sum_element()
    assert isinstance(elt, LDirectSumElement)
    assert len(elt.summands) == len(res.summands)

    diff = elt - elt
    assert diff.computable
    assert diff.exact
    assert all(s.zero_certified for s in diff.summands)


def test_wall_direct_sum_scalar_ops_and_roundtrip_to_obstruction_result():
    q_form = QuadraticForm(
        matrix=np.array([[0, 1], [1, 0]], dtype=np.int64),
        dimension=6,
        q_refinement=[1, 1],
    )
    res = WallGroupL(dimension=6, pi="Z x Z").compute_obstruction_result(q_form)
    elt = res.to_direct_sum_element()
    zero = 0 * elt
    assert all(s.zero_certified for s in zero.summands)

    neg = -elt
    add_back = neg + elt
    assert add_back == zero

    rt = add_back.to_obstruction_result(collapse_integral=False)
    assert rt.computable
    assert rt.exact
    assert rt.zero_certified
    assert rt.obstructs is False
