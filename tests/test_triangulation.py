import numpy as np
import pytest
from pysurgery import SimplicialComplex


def test_vietoris_rips_replaces_gudhi_bridge():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]])
    sc = SimplicialComplex.from_vietoris_rips(pts, epsilon=1.05, max_dimension=2)
    assert sc.dimension >= 1
    assert sc.count_simplices(0) == 3


def test_gudhi_bridge_module_removed():
    with pytest.raises(ModuleNotFoundError):
        import pysurgery.integrations.gudhi_bridge  # noqa: F401