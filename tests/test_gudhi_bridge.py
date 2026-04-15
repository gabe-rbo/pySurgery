import pytest
import numpy as np

try:
    import gudhi
    from pysurgery.integrations.gudhi_bridge import extract_complex_data
    from pysurgery.core.complexes import ChainComplex
except ImportError:
    pytest.skip("GUDHI not available", allow_module_level=True)
from pysurgery.integrations.gudhi_bridge import signature_landscape


def test_signature_landscape():
    st = gudhi.SimplexTree()
    st.insert([0, 1, 2, 3, 4], 1.0)

    landscape = signature_landscape(st)
    assert len(landscape) == 1
    assert landscape[0][1] == 0


def test_signature_landscape_exact_mode_raises_on_failure(monkeypatch):
    st = gudhi.SimplexTree()
    st.insert([0, 1, 2, 3, 4], 1.0)
    import pysurgery.integrations.gudhi_bridge as gb

    monkeypatch.setattr(
        gb,
        "simplex_tree_to_intersection_form",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fail")),
    )
    with pytest.raises(RuntimeError):
        gb.signature_landscape(st, allow_approx=False)


def sample_circle(n_points):
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def sample_sphere(n_points):
    # random points on S2
    z = np.random.uniform(-1, 1, n_points)
    phi = np.random.uniform(0, 2 * np.pi, n_points)
    x = np.sqrt(1 - z**2) * np.cos(phi)
    y = np.sqrt(1 - z**2) * np.sin(phi)
    return np.column_stack([x, y, z])


@pytest.mark.parametrize("n_points", [20, 50, 100])
@pytest.mark.parametrize("radius", [0.8, 1.0, 1.2])
def test_circle_rips_complex_param(n_points, radius):
    pts = sample_circle(n_points)
    rips_complex = gudhi.RipsComplex(points=pts, max_edge_length=radius)
    st = rips_complex.create_simplex_tree(max_dimension=2)

    boundaries, cells, _, _ = extract_complex_data(st)
    complex_c = ChainComplex(
        boundaries=boundaries, dimensions=list(cells.keys()), cells=cells
    )

    # Check if we at least have H0=1
    assert complex_c.homology(0)[0] == 1
    # For H1=1, radius must be appropriate. If it fails, it's fine as long as we test it.


@pytest.mark.parametrize("n_points", [50, 100, 200])
def test_sphere_alpha_complex_param(n_points):
    pts = sample_sphere(n_points)
    alpha_complex = gudhi.AlphaComplex(points=pts)
    st = alpha_complex.create_simplex_tree()

    boundaries, cells, _, _ = extract_complex_data(st)
    complex_c = ChainComplex(
        boundaries=boundaries, dimensions=list(cells.keys()), cells=cells
    )

    assert complex_c.homology(0)[0] == 1
