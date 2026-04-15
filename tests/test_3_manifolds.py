import numpy as np
import scipy.sparse as sp
from pysurgery.core.complexes import ChainComplex
from pysurgery.homeomorphism import analyze_homeomorphism_3d


def test_lens_space_L31_homology():
    d1 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))
    d2 = sp.csr_matrix(np.array([[3]], dtype=np.int64))
    d3 = sp.csr_matrix(np.zeros((1, 1), dtype=np.int64))

    cc = ChainComplex(
        boundaries={1: d1, 2: d2, 3: d3},
        dimensions=[0, 1, 2, 3],
        cells={0: 1, 1: 1, 2: 1, 3: 1},
    )

    assert cc.homology(0) == (1, [])

    from pysurgery.bridge.julia_bridge import julia_engine

    if julia_engine.available:
        assert cc.homology(1) == (0, [3])
    else:
        assert cc.homology(1)[0] == 0

    assert cc.homology(2) == (0, [])
    assert cc.homology(3) == (1, [])


def test_poincare_homology_sphere():
    d1 = sp.csr_matrix(np.zeros((1, 0), dtype=np.int64))
    d2 = sp.csr_matrix(np.zeros((0, 0), dtype=np.int64))
    d3 = sp.csr_matrix(np.zeros((0, 1), dtype=np.int64))
    cc_phs = ChainComplex(
        boundaries={1: d1, 2: d2, 3: d3},
        dimensions=[0, 1, 2, 3],
        cells={0: 1, 1: 0, 2: 0, 3: 1},
    )
    cc_s3 = ChainComplex(
        boundaries={1: d1, 2: d2, 3: d3},
        dimensions=[0, 1, 2, 3],
        cells={0: 1, 1: 0, 2: 0, 3: 1},
    )

    is_homeo, reason = analyze_homeomorphism_3d(cc_phs, cc_s3)
    assert not is_homeo
    assert "INCONCLUSIVE: Both are homology-sphere candidates" in reason
