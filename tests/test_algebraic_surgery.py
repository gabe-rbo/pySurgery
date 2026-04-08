import numpy as np
from pysurgery.algebraic_poincare import AlgebraicPoincareComplex
from pysurgery.core.complexes import ChainComplex
from pysurgery.algebraic_surgery import AlgebraicSurgeryComplex
from pysurgery.core.intersection_forms import IntersectionForm

def test_assembly_map():
    cc = ChainComplex(boundaries={}, dimensions=[0, 1], cells={0: 1, 1: 0})
    apc1 = AlgebraicPoincareComplex(chain_complex=cc, fundamental_class=np.array([1]), dimension=4)
    apc2 = AlgebraicPoincareComplex(chain_complex=cc, fundamental_class=np.array([1]), dimension=4)
    
    asc = AlgebraicSurgeryComplex(domain=apc1, codomain=apc2, degree=1)
    
    matrix = np.array([[0, 1], [1, 0]])
    form = IntersectionForm(matrix=matrix, dimension=4)
    obstruction = asc.assembly_map(pi_1_group="1", form=form)
    
    # signature is 0, so the obstruction class is 0.
    assert obstruction == 0


def test_assembly_map_result_typed():
    cc = ChainComplex(boundaries={}, dimensions=[0, 1], cells={0: 1, 1: 0})
    apc1 = AlgebraicPoincareComplex(chain_complex=cc, fundamental_class=np.array([1]), dimension=4)
    apc2 = AlgebraicPoincareComplex(chain_complex=cc, fundamental_class=np.array([1]), dimension=4)
    asc = AlgebraicSurgeryComplex(domain=apc1, codomain=apc2, degree=1)

    form = IntersectionForm(matrix=np.array([[0, 1], [1, 0]]), dimension=4)
    res = asc.assembly_map_result(pi_1_group="1", form=form)
    assert res.computable
    assert res.exact
    assert res.value == 0


def test_evaluate_structure_set_typed():
    cc = ChainComplex(boundaries={}, dimensions=[0], cells={0: 1})
    apc = AlgebraicPoincareComplex(chain_complex=cc, fundamental_class=np.array([1]), dimension=5)
    asc = AlgebraicSurgeryComplex(domain=apc, codomain=apc, degree=1)
    seq = asc.evaluate_structure_set(cc, fundamental_group="1")
    assert seq.computable
    assert seq.exact
    assert seq.normal_invariants is not None

