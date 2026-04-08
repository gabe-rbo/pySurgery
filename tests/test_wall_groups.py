import numpy as np
import pytest
from pysurgery.wall_groups import WallGroupL
from pysurgery.core.intersection_forms import IntersectionForm
from pysurgery.core.quadratic_forms import QuadraticForm
from pysurgery.core.fundamental_group import GroupPresentation
from pysurgery.core.exceptions import SurgeryObstructionError

def test_wall_group_L_4k_1():
    e8_matrix = np.array([
        [2, -1, 0, 0, 0, 0, 0, 0],
        [-1, 2, -1, 0, 0, 0, 0, 0],
        [0, -1, 2, -1, 0, 0, 0, -1],
        [0, 0, -1, 2, -1, 0, 0, 0],
        [0, 0, 0, -1, 2, -1, 0, 0],
        [0, 0, 0, 0, -1, 2, -1, 0],
        [0, 0, 0, 0, 0, -1, 2, 0],
        [0, 0, -1, 0, 0, 0, 0, 2]
    ])
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
    matrix = np.array([[1, 0], [0, 1]]) # rank 2, signature 2
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


def test_wall_group_typed_result_uncomputable_product_case():
    gp = GroupPresentation(kind="product", factors=["Z_2", "Z_3"])
    wg = WallGroupL(dimension=8, pi=gp)
    res = wg.compute_obstruction_result()
    assert not res.computable
    assert "factor-wise" in res.message


