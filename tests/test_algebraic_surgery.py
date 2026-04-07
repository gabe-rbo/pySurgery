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
    
    matrix = np.array([[1, 0], [0, 1]])
    form = IntersectionForm(matrix=matrix, dimension=4)
    obstruction = asc.assembly_map(pi_1_group="1", form=form)
    
    # signature is 2. 2 // 8 is 0.
    assert obstruction == 0
