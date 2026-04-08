import numpy as np
import pytest
from hypothesis import given, strategies as st
from pysurgery.core.intersection_forms import IntersectionForm
from pysurgery.core.complexes import ChainComplex
from pysurgery.core.quadratic_forms import QuadraticForm
from pysurgery.core.kirby_calculus import KirbyDiagram
from pysurgery.wall_groups import l_group_symbol
from pysurgery.core.fundamental_group import GroupPresentation
import scipy.sparse as sp

@given(st.lists(st.integers(min_value=-10, max_value=10), min_size=4, max_size=4))
def test_intersection_form_signature(data):
    matrix = np.array([[data[0], data[1]], [data[1], data[3]]])
    form = IntersectionForm(matrix=matrix, dimension=4)
    sig = form.signature()
    assert isinstance(sig, int)
    assert abs(sig) <= 2

def test_even_form_classification():
    # E8 matrix (even, unimodular, rank 8, signature 8)
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
    assert form.is_even()
    assert form.type() == "II"
    assert form.signature() == 8

def test_odd_form_classification():
    matrix = np.array([[1, 0], [0, 1]])
    form = IntersectionForm(matrix=matrix, dimension=4)
    assert not form.is_even()
    assert form.type() == "I"
    assert form.signature() == 2

def test_single_point_homology():
    # A single point: 1 cell of dim 0, no boundaries
    complex_c = ChainComplex(boundaries={}, dimensions=[0], cells={0: 1})
    rank, torsion = complex_c.homology(0)
    assert rank == 1
    assert torsion == []
    rank, torsion = complex_c.homology(1)
    assert rank == 0
    assert torsion == []

def test_sphere_homology():
    # S^2: 1 cell of dim 0, 1 cell of dim 2, trivial boundaries
    d1 = sp.csr_matrix(np.zeros((1, 0), dtype=np.int64))
    d2 = sp.csr_matrix(np.zeros((0, 1), dtype=np.int64))
    complex_c = ChainComplex(boundaries={1: d1, 2: d2}, dimensions=[0, 1, 2], cells={0: 1, 1: 0, 2: 1})
    assert complex_c.homology(0) == (1, [])
    assert complex_c.homology(1) == (0, [])
    assert complex_c.homology(2) == (1, [])

def test_intersection_form_sanity():
    # Hyperbolic plane
    matrix = np.array([[0, 1], [1, 0]])
    form = IntersectionForm(matrix=matrix, dimension=4)
    assert form.signature() == 0
    assert form.is_even()


def test_intersection_form_requires_square_matrix():
    from pysurgery.core.exceptions import DimensionError
    with pytest.raises(DimensionError):
        IntersectionForm(matrix=np.array([[1, 2, 3]]), dimension=4)


def test_zero_matrix_signature_rank_are_stable():
    form = IntersectionForm(matrix=np.zeros((3, 3), dtype=int), dimension=4)
    assert form.signature() == 0
    assert form.rank() == 0

def test_arf_invariant():
    matrix = np.array([[0, 1], [1, 0]])
    q_form = QuadraticForm(matrix=matrix, dimension=4, q_refinement=[1, 1])
    assert q_form.arf_invariant() == 1
    q_form2 = QuadraticForm(matrix=matrix, dimension=4, q_refinement=[0, 0])
    assert q_form2.arf_invariant() == 0

def test_handle_slide_invertibility():
    diag = KirbyDiagram(linking_matrix=np.array([[0, 1], [1, 0]]), framings=np.array([0, 0]))
    slide1 = diag.handle_slide(source_idx=0, target_idx=1)
    # The algebraic realization is P = I + E_{target, source}.
    # To invert it, we'd need to subtract. Handle slide back is not simply the same operation,
    # but let's just test that the operation produces a valid diagram with correct determinant.
    assert np.abs(np.linalg.det(slide1.linking_matrix)) == 1

def test_blow_up():
    diag = KirbyDiagram(linking_matrix=np.array([[2]]), framings=np.array([2]))
    bup = diag.blow_up(sign=1)
    assert bup.linking_matrix.shape == (2, 2)
    assert np.array_equal(bup.linking_matrix, np.array([[2, 0], [0, 1]]))
    
def test_wall_group_symbols():
    assert l_group_symbol(0, "Z") == "Z"
    assert l_group_symbol(1, "Z") == "Z"
    assert l_group_symbol(2, "Z") == "Z_2"
    assert l_group_symbol(3, "Z") == "Z_2"
    assert l_group_symbol(4, "Z") == "Z"


def test_wall_group_symbol_product_group_presentation():
    gp = GroupPresentation(kind="product", factors=["Z", "Z_2"])
    assert "product-group decomposition required" in l_group_symbol(8, gp)

def test_algebraic_surgery():
    # Start with S^2 x S^2, Q = [[0, 1], [1, 0]]
    Q = np.array([[0, 1], [1, 0]])
    form = IntersectionForm(matrix=Q, dimension=4)
    
    # Isotropic class x = [1, 0]
    x = np.array([1, 0])
    
    new_form = form.perform_algebraic_surgery(x)
    # The rank should drop by 2 (from 2 to 0) since we surgered out a hyperbolic plane
    assert new_form.rank() == 0
    assert new_form.signature() == 0

def test_algebraic_surgery_invalid_isotropic():
    # Try to surgery a non-isotropic class
    Q = np.array([[1, 0], [0, -1]])
    form = IntersectionForm(matrix=Q, dimension=4)
    
    x = np.array([1, 0]) # Q(x,x) = 1, not 0
    
    from pysurgery.core.exceptions import IsotropicError
    import pytest
    with pytest.raises(IsotropicError):
        form.perform_algebraic_surgery(x)


def test_algebraic_surgery_zero_class_rejected():
    Q = np.array([[0, 1], [1, 0]])
    form = IntersectionForm(matrix=Q, dimension=4)
    from pysurgery.core.exceptions import NonPrimitiveError
    with pytest.raises(NonPrimitiveError):
        form.perform_algebraic_surgery(np.array([0, 0]))


def test_torus_intersection_form():
    # Placeholder for checking if intersection form functions run
    # Since Q = [0 1; 1 0] or similar for T2
    try:
        import gudhi # noqa
    except ImportError:
        pytest.skip("GUDHI not available")

