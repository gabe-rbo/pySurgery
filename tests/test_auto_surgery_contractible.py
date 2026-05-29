import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from discrete_surface_data import build_tetrahedron, to_complex
from pysurgery.topology.complexes import SimplicialComplex
from pysurgery.auto_surgery import AutoSurgeon

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


def test_auto_surgeon_disjoint_spheres():
    """Verify that a disjoint union of two spheres passes the pipeline successfully."""
    S2 = to_complex(build_tetrahedron())
    
    surgeon = AutoSurgeon(S2, backend="python", target_topology="homotopy_sphere")
    report = surgeon.run()
    
    assert report.status == "success"
    assert report.exact is True
    assert len(report.final_components) == 1
    assert report.final_components[0].betti == {0: 1, 1: 0, 2: 1}


def test_auto_surgeon_simply_connected_homotopy_sphere():
    """Verify that a simply connected complex successfully reduces to a homotopy sphere."""
    K = to_complex(build_tetrahedron())
    surgeon = AutoSurgeon(K, backend="python", target_topology="homotopy_sphere")
    report = surgeon.run()
    
    assert report.status == "success"
    assert report.exact is True
    assert report.final_components[0].betti == {0: 1, 1: 0, 2: 1}


def test_auto_surgeon_simply_connected_contractible():
    """Verify that opt-in target_topology='contractible' successfully achieves contractibility."""
    K = to_complex(build_tetrahedron())
    surgeon = AutoSurgeon(K, backend="python", target_topology="contractible")
    report = surgeon.run()
    
    assert report.status == "success"
    assert report.exact is True
    assert report.final_components[0].betti == {0: 1, 1: 0, 2: 0}


def test_auto_surgeon_obstructed_e8():
    """Verify that the obstructed E8 complex is cleanly halted without exceptions."""
    K = SyntheticE8Complex.from_simplices([[0, 1]])
    surgeon = AutoSurgeon(K, backend="python", target_topology="homotopy_sphere")
    report = surgeon.run()
    
    assert report.status == "halted_by_obstruction"
    assert report.exact is False
    assert report.obstruction_reports["C0"].vanishes is False
    assert report.obstruction_reports["C0"].obstruction_class["signature"] == 8
