import numpy as np
import pytest
import sympy as sp
from pysurgery.surgery import SurgerySession, TranslateIsotopy, RotateIsotopy, ScaleIsotopy, SymbolicTransformation

def test_isotopy_basic_application():
    cloud = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    
    # Translate
    iso_t = TranslateIsotopy(offset=(1.0, 2.0, 3.0))
    res_t = iso_t(cloud, 1.0)
    expected_t = cloud + np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(res_t, expected_t)
    
    # Scale
    iso_s = ScaleIsotopy(factors=2.0)
    res_s = iso_s(cloud, 1.0)
    expected_s = cloud * 2.0
    np.testing.assert_allclose(res_s, expected_s)
    
    # Rotate (90 deg in 0,1 plane)
    iso_r = RotateIsotopy(angle=np.pi/2, plane=(0, 1))
    res_r = iso_r(cloud, 1.0)
    # [1,0,0] -> [0,1,0], [0,1,0] -> [-1,0,0]
    expected_r = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    np.testing.assert_allclose(res_r, expected_r, atol=1e-7)

def test_surgery_session_isotopy_integration():
    session = SurgerySession(ambient_space="R^3", point_clouds={"T1": np.array([[1.0, 0.0, 0.0]])})
    
    session.move(offset=(1.0, 0.0, 0.0), target="T1")
    np.testing.assert_allclose(session.point_clouds["T1"], [[2.0, 0.0, 0.0]])
    
    session.rotate(angle=np.pi/2, plane=(0, 1), target="T1")
    # [2,0,0] rotated 90 deg in XY plane -> [0,2,0]
    np.testing.assert_allclose(session.point_clouds["T1"], [[0.0, 2.0, 0.0]], atol=1e-7)
    
    session.scale(factors=0.5, target="T1")
    # [0,2,0] scaled by 0.5 -> [0,1,0]
    np.testing.assert_allclose(session.point_clouds["T1"], [[0.0, 1.0, 0.0]], atol=1e-7)
    
    assert len(session._isotopy_log) == 3
    
    logs = session.logs()
    assert "II.5 SYMBOLIC COMPOSITION" in logs
    assert "Global Composition" in logs

def test_symbolic_transformation():
    # f(x, t) = [x0 + t, x1 * (1+t), x2]
    def expr_fn(d):
        t = sp.Symbol("t")
        xs = [sp.Symbol(f"x_{i}") for i in range(d)]
        return sp.Matrix([xs[0] + t, xs[1] * (1 + t), xs[2]])
    
    iso = SymbolicTransformation(expr_fn)
    cloud = np.array([[1.0, 2.0, 3.0]])
    
    res = iso(cloud, 0.5)
    # [1+0.5, 2*(1+0.5), 3] = [1.5, 3.0, 3.0]
    expected = np.array([[1.5, 3.0, 3.0]])
    np.testing.assert_allclose(res, expected)

def test_collision_detection():
    # T1 at [0,0,0], T2 at [2,0,0]
    point_clouds = {
        "T1": np.array([[0.0, 0.0, 0.0]]),
        "T2": np.array([[2.0, 0.0, 0.0]])
    }
    session = SurgerySession(ambient_space="R^3", point_clouds=point_clouds)
    
    # Moving T1 by [4,0,0] should collide with T2 at t=0.5
    with pytest.raises(ValueError, match="Isotopy collision"):
        session.move(offset=(4.0, 0.0, 0.0), target="T1")

    # Try with warn policy
    session._isotopy_collision_policy = "warn"
    with pytest.warns(UserWarning, match="translate path intersects 'T2'"):
        session.move(offset=(4.0, 0.0, 0.0), target="T1")

def test_composition_ledger():
    session = SurgerySession(ambient_space="R^3", point_clouds={"T1": np.array([[1.0, 0.0, 0.0]])})
    session.translate(offset=(1.0, 0.0, 0.0), target="T1") # x0 -> x0 + 1
    session.scale(factors=2.0, target="T1")               # x -> 2x
    
    # Composition: f2(f1(x)) = 2*(x + [1,0,0]) = [2*x0 + 2, 2*x1, 2*x2]
    expr = session._isotopy_log.full_expr(d=3)
    
    t = sp.Symbol("t")
    x0, x1, x2 = sp.symbols("x_0 x_1 x_2")
    # At t=1:
    expected_expr = sp.Matrix([2*(x0 + 1), 2*x1, 2*x2])
    
    # full_expr returns expression with 't'
    # We substitute t=1 to check the final map
    final_map = expr.subs(t, 1.0)
    
    # Use simplify and check if zero matrix
    diff = sp.simplify(final_map - expected_expr)
    assert diff == sp.zeros(3, 1)
