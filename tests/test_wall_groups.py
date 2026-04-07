import numpy as np
from pysurgery.wall_groups import WallGroupL
from pysurgery.core.intersection_forms import IntersectionForm
from pysurgery.core.quadratic_forms import QuadraticForm

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

def test_wall_group_L_4k_plus_2():
    matrix = np.array([[0, 1], [1, 0]])
    q_form = QuadraticForm(matrix=matrix, dimension=2, q_refinement=[1, 1])
    wg = WallGroupL(dimension=2, pi="1")
    obstruction = wg.compute_obstruction(q_form)
    assert obstruction == 1

def test_wall_group_L_Z():
    wg = WallGroupL(dimension=5, pi="Z")
    assert wg.compute_obstruction() == "Z"
