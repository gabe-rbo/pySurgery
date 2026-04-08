import numpy as np
import pytest
import scipy.sparse as sp
try:
    from tests.discrete_surface_data import build_tetrahedron, build_octahedron, build_icosahedron, build_torus, to_complex
    from pysurgery.homeomorphism import analyze_homeomorphism_2d
except ImportError:
    pass
from pysurgery.homeomorphism import (
    analyze_homeomorphism_2d_result,
    analyze_homeomorphism_4d,
    analyze_homeomorphism_4d_result,
    analyze_homeomorphism_high_dim_result,
    surgery_to_remove_impediments,
)
from pysurgery.core.intersection_forms import IntersectionForm
from pysurgery.core.complexes import ChainComplex
from pysurgery.core.fundamental_group import FundamentalGroup
from pysurgery.core.k_theory import WhiteheadGroup
from pysurgery.wall_groups import ObstructionResult

def test_analyze_homeomorphism_4d_indefinite():
    matrix1 = np.array([[0, 1], [1, 0]])
    form1 = IntersectionForm(matrix=matrix1, dimension=4)
    
    matrix2 = np.array([[0, 1], [1, 0]])
    form2 = IntersectionForm(matrix=matrix2, dimension=4)
    
    is_homeo, reason = analyze_homeomorphism_4d(form1, form2, ks1=0, ks2=0, simply_connected=True)
    assert is_homeo
    assert "SUCCESS" in reason

def test_analyze_homeomorphism_4d_impediment():
    matrix1 = np.array([[0, 1], [1, 0]])
    form1 = IntersectionForm(matrix=matrix1, dimension=4)
    
    matrix2 = np.array([[1, 0], [0, -1]])
    form2 = IntersectionForm(matrix=matrix2, dimension=4)
    
    is_homeo, reason = analyze_homeomorphism_4d(form1, form2, ks1=0, ks2=0, simply_connected=True)
    assert not is_homeo
    assert "Parity mismatch" in reason


def test_analyze_homeomorphism_4d_definite_exact_match_is_homeomorphic():
    form1 = IntersectionForm(matrix=np.array([[1, 0], [0, 1]]), dimension=4)
    form2 = IntersectionForm(matrix=np.array([[1, 0], [0, 1]]), dimension=4)
    is_homeo, reason = analyze_homeomorphism_4d(form1, form2, ks1=0, ks2=0, simply_connected=True)
    assert is_homeo is True
    assert "SUCCESS" in reason


def test_analyze_homeomorphism_2d_homology_failure_fallback():
    class BrokenComplex:
        def homology(self, n):
            raise RuntimeError("boom")

    from pysurgery.homeomorphism import analyze_homeomorphism_2d
    is_homeo, reason = analyze_homeomorphism_2d(BrokenComplex(), BrokenComplex())
    assert is_homeo is None
    assert "INCONCLUSIVE" in reason

    with pytest.warns(UserWarning) as rec:
        is_homeo2, reason2 = analyze_homeomorphism_2d(BrokenComplex(), BrokenComplex(), allow_approx=True)
    assert is_homeo2 is None
    assert "INCONCLUSIVE" in reason2
    warning_text = "\n".join(str(w.message) for w in rec)
    assert "boom" in warning_text
    assert "{e}" not in warning_text

def test_surgery_to_remove_impediments():
    matrix1 = np.array([[1, 0], [0, 1]]) # sig = 2
    form1 = IntersectionForm(matrix=matrix1, dimension=4)
    
    can_remove, reason = surgery_to_remove_impediments(form1, 10)
    assert can_remove
    
    can_remove, reason = surgery_to_remove_impediments(form1, 4)
    assert can_remove


def test_analyze_homeomorphism_4d_requires_explicit_hypotheses():
    form1 = IntersectionForm(matrix=np.array([[0, 1], [1, 0]]), dimension=4)
    form2 = IntersectionForm(matrix=np.array([[0, 1], [1, 0]]), dimension=4)
    res = analyze_homeomorphism_4d_result(form1, form2)
    assert res.is_homeomorphic is None
    assert res.status == "inconclusive"
    assert "Simply-connectedness" in res.reasoning


def test_high_dim_reports_surgery_required_for_nonzero_wall_obstruction():
    d1 = sp.csr_matrix(np.zeros((1, 0), dtype=np.int64))
    d5 = sp.csr_matrix(np.zeros((0, 1), dtype=np.int64))
    c = ChainComplex(
        boundaries={1: d1, 5: d5},
        dimensions=[0, 1, 2, 3, 4, 5],
        cells={0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1},
    )

    wh = WhiteheadGroup(rank=0, description="Wh(1)=0", computable=True, exact=True)
    wall = ObstructionResult(
        dimension=5,
        pi="1",
        computable=True,
        exact=True,
        value=1,
        modulus=None,
        message="",
        assumptions=[],
    )
    res = analyze_homeomorphism_high_dim_result(
        c,
        c,
        dim=5,
        pi1=FundamentalGroup(generators=[], relations=[]),
        whitehead_group=wh,
        wall_obstruction=wall,
    )
    assert res.status == "surgery_required"
    assert res.is_homeomorphic is False


def test_2d_detects_cohomology_mismatch_even_when_homology_matches():
    class FakeSurface:
        coefficient_ring = "Z"

        def __init__(self, h1_torsion, h1co_torsion):
            self._h1_torsion = h1_torsion
            self._h1co_torsion = h1co_torsion

        def homology(self, n):
            if n == 2:
                return 1, []
            if n == 1:
                return 2, list(self._h1_torsion)
            if n == 0:
                return 1, []
            return 0, []

        def cohomology(self, n):
            if n == 2:
                return 1, []
            if n == 1:
                return 2, list(self._h1co_torsion)
            if n == 0:
                return 1, []
            return 0, []

    c1 = FakeSurface(h1_torsion=[], h1co_torsion=[])
    c2 = FakeSurface(h1_torsion=[], h1co_torsion=[2])
    res = analyze_homeomorphism_2d_result(c1, c2)
    assert res.status == "impediment"
    assert res.is_homeomorphic is False
    assert "Cohomology groups differ" in res.reasoning


def test_high_dim_requires_shared_cohomology_coefficients():
    d1 = sp.csr_matrix(np.zeros((1, 0), dtype=np.int64))
    d5 = sp.csr_matrix(np.zeros((0, 1), dtype=np.int64))
    c_z = ChainComplex(
        boundaries={1: d1, 5: d5},
        dimensions=[0, 1, 2, 3, 4, 5],
        cells={0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1},
        coefficient_ring="Z",
    )
    c_q = ChainComplex(
        boundaries={1: d1, 5: d5},
        dimensions=[0, 1, 2, 3, 4, 5],
        cells={0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1},
        coefficient_ring="Q",
    )
    res = analyze_homeomorphism_high_dim_result(c_z, c_q, dim=5)
    assert res.status == "inconclusive"
    assert "shared coefficient ring" in res.reasoning


def test_2d_detects_cup_product_signature_mismatch():
    class FakeSurface:
        coefficient_ring = "Z"

        def homology(self, n):
            if n == 2:
                return 1, []
            if n == 1:
                return 2, []
            if n == 0:
                return 1, []
            return 0, []

        def cohomology(self, n):
            if n == 2:
                return 1, []
            if n == 1:
                return 2, []
            if n == 0:
                return 1, []
            return 0, []

    c1 = FakeSurface()
    c2 = FakeSurface()
    res = analyze_homeomorphism_2d_result(
        c1,
        c2,
        cup_product_signature_1={"u^2": 1},
        cup_product_signature_2={"u^2": 0},
    )
    assert res.status == "impediment"
    assert "cup-product incompatibility" in res.reasoning


def test_high_dim_cup_product_requires_both_signatures_when_used():
    d1 = sp.csr_matrix(np.zeros((1, 0), dtype=np.int64))
    d5 = sp.csr_matrix(np.zeros((0, 1), dtype=np.int64))
    c = ChainComplex(
        boundaries={1: d1, 5: d5},
        dimensions=[0, 1, 2, 3, 4, 5],
        cells={0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1},
    )
    res = analyze_homeomorphism_high_dim_result(
        c,
        c,
        dim=5,
        cup_product_signature_1={"x": 1},
        cup_product_signature_2=None,
    )
    assert res.status == "inconclusive"
    assert "cup-product signatures" in res.reasoning


def test_s2_models_homeomorphism():
    try:
        c1 = to_complex(build_tetrahedron())
        c2 = to_complex(build_octahedron())
        c3 = to_complex(build_icosahedron())
        
        # They should all be homeomorphic to each other
        is_homeo_1, _ = analyze_homeomorphism_2d(c1, c2)
        is_homeo_2, _ = analyze_homeomorphism_2d(c2, c3)
        assert is_homeo_1
        assert is_homeo_2
    except NameError:
        pytest.skip("GUDHI not available")

def test_s2_vs_torus_homeomorphism():
    try:
        c1 = to_complex(build_tetrahedron())
        c2 = to_complex(build_torus())
        
        is_homeo, _ = analyze_homeomorphism_2d(c1, c2)
        assert not is_homeo
    except NameError:
        pytest.skip("GUDHI not available")

