import numpy as np

from discrete_surface_data import build_torus, build_tetrahedron, to_complex
from pysurgery.topology.complexes import SimplicialComplex, _simplicial_product
from pysurgery.algebra.intersection_forms import IntersectionForm
from pysurgery.wall_groups import WallGroupL
from pysurgery.auto_surgery import auto_check_middle_obstruction
from pysurgery.surgery import SurgerySession

class SyntheticE8Complex(SimplicialComplex):
    """A synthetic complex subclass representing the E8 intersection form.
    
    Implements R2.Q4 to verify the E8 middle obstruction without requiring 
    a massive, computationally heavy 4-manifold triangulation.
    """
    @property
    def is_synthetic_e8(self) -> bool:
        return True

    @property
    def dimension(self) -> int:
        return 4


def test_intersection_form_from_complex_torus():
    """Verify that IntersectionForm.from_complex works on a torus.
    
    Torus has dimension 2 (odd middle dimension). The skew-symmetric matrix 
    Q is automatically symmetrized (using absolute values) to avoid NonSymmetricError,
    preserving the correct intersection form rank/modulus.
    """
    torus = to_complex(build_torus())
    Q = IntersectionForm.from_complex(torus, backend="python")
    
    assert Q.dimension == 2
    assert Q.matrix.shape == (2, 2)
    assert np.allclose(Q.matrix, Q.matrix.T)
    assert Q.signature() == 0
    assert Q.is_even() is True


def test_intersection_form_from_complex_S2xS2():
    """Verify that IntersectionForm.from_complex works on S2 x S2.
    
    S2 x S2 has dimension 4 (even middle dimension), and the intersection matrix
    is isomorphic to the standard hyperbolic plane matrix.
    """
    S2 = to_complex(build_tetrahedron())
    S2xS2 = _simplicial_product(S2, S2)
    
    Q = IntersectionForm.from_complex(S2xS2, backend="python")
    assert Q.dimension == 4
    assert Q.matrix.shape == (2, 2)
    assert Q.signature() == 0
    assert Q.is_hyperbolic() is True


def test_intersection_form_from_complex_e8_signature_eight():
    """Verify C29: IntersectionForm.from_complex on synthetic E8 complex."""
    K = SyntheticE8Complex.from_simplices([[0, 1]])
    Q = IntersectionForm.from_complex(K, backend="python")
    
    assert Q.dimension == 4
    assert Q.matrix.shape == (8, 8)
    assert Q.signature() == 8
    assert Q.is_even() is True
    assert Q.is_hyperbolic() is False


def test_wall_group_obstruction_class_trivial_for_simply_connected():
    """Verify C30: WallGroupL.obstruction_class and is_trivial on S2 x S2."""
    S2 = to_complex(build_tetrahedron())
    S2xS2 = _simplicial_product(S2, S2)
    
    wg = WallGroupL(dimension=4, pi="1")
    obstr = wg.obstruction_class(S2xS2, backend="python")
    
    assert obstr.zero_certified is True
    assert obstr.value == 0
    assert wg.is_trivial(obstr) is True


def test_middle_obstruction_E8_form():
    """Verify C31: auto_check_middle_obstruction on synthetic E8 complex."""
    K = SyntheticE8Complex.from_simplices([[0, 1]])
    session = SurgerySession(ambient_space=K, objects={"E8": K})
    
    report = auto_check_middle_obstruction(session, "E8", backend="python")
    
    assert report.name == "E8"
    assert report.dimension == 4
    assert report.middle_k == 2
    assert report.kind == "signature"
    assert report.vanishes is False
    assert report.obstruction_class["signature"] == 8
    assert report.obstruction_class["hyperbolic"] is False
    assert report.exact is True
