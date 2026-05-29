import numpy as np
from pysurgery.geometry.immersion_obstructions import (
    check_dual_stiefel_whitney_non_immersibility,
    rational_pontryagin_classes,
    combinatorial_euler_class,
    compute_rational_pontryagin_obstruction,
    NonImmersibilityWitness,
    PontryaginClasses,
    EulerClass
)
from pysurgery.topology.complexes import SimplicialComplex
from pysurgery.algebra.intersection_forms import IntersectionForm

def test_rational_pontryagin_obstruction_cp2(monkeypatch):
    cp2 = SimplicialComplex(coefficient_ring='Z')
    monkeypatch.setattr(SimplicialComplex, 'dimension', 4)
    monkeypatch.setattr('pysurgery.geometry.immersion_obstructions.extract_euler_class', lambda m: 3)
    
    q = IntersectionForm(matrix=np.array([[1]]), dimension=4)
    
    res = compute_rational_pontryagin_obstruction(cp2, target_dim=6, intersection_form=q)
    assert isinstance(res, NonImmersibilityWitness)
    assert res.immersible is False
    assert res.exact is True

def test_lens_space_immersion(monkeypatch):
    l31 = SimplicialComplex(coefficient_ring='Z')
    monkeypatch.setattr(SimplicialComplex, 'dimension', 3)
    monkeypatch.setattr('pysurgery.geometry.immersion_obstructions.extract_euler_class', lambda m: 0)
    
    res = compute_rational_pontryagin_obstruction(l31, target_dim=4)
    assert not isinstance(res, NonImmersibilityWitness)

def test_dual_stiefel_whitney_rp2(monkeypatch):
    rp2 = SimplicialComplex(coefficient_ring='Z')
    monkeypatch.setattr(SimplicialComplex, 'dimension', 2)
    monkeypatch.setattr(SimplicialComplex, 'count_simplices', lambda self, k: 1)
    monkeypatch.setattr(SimplicialComplex, 'n_simplices', lambda self, k: [(0,)])
    monkeypatch.setattr(SimplicialComplex, 'simplex_to_index', lambda self, k: {(0,): 0})
    
    w0 = np.array([1])
    w1 = np.array([1]) # non-orientable
    w2 = np.array([1]) # euler mod 2 = 1
    
    def mock_extract(m, k):
        if k == 0:
            return w0
        if k == 1:
            return w1
        if k == 2:
            return w2
        return np.array([0])
        
    monkeypatch.setattr('pysurgery.geometry.immersion_obstructions.extract_stiefel_whitney_tangent', mock_extract)
    monkeypatch.setattr('pysurgery.geometry.immersion_obstructions.alexander_whitney_cup', lambda alpha, beta, **kwargs: (alpha * beta) % 2)
    
    # w_bar_0 = [1]
    # w_bar_1 = w1 * w_bar_0 = [1] * [1] = [1]
    # w_bar_2 = w2 * w_bar_0 + w1 * w_bar_1 = 1*1 + 1*1 = 2 = 0
    # highest_non_zero_degree is 1.
    # Target dimension 2 (k=0). 1 > 0 -> should fail immersion
    res_2 = check_dual_stiefel_whitney_non_immersibility(rp2, target_dim=2)
    assert isinstance(res_2, NonImmersibilityWitness)
    assert res_2.immersible is False
    assert res_2.exact is True

def test_rational_pontryagin_torus(monkeypatch):
    torus = SimplicialComplex(coefficient_ring='Z')
    monkeypatch.setattr(SimplicialComplex, 'dimension', 2)
    
    p_classes = rational_pontryagin_classes(torus)
    assert isinstance(p_classes, PontryaginClasses)
    assert p_classes.exact is True
    assert len(p_classes.classes) == 0

def test_combinatorial_euler_class(monkeypatch):
    torus = SimplicialComplex(coefficient_ring='Z')
    monkeypatch.setattr('pysurgery.geometry.immersion_obstructions.extract_euler_class', lambda m: 0)
    e_class = combinatorial_euler_class(torus)
    assert isinstance(e_class, EulerClass)
    assert e_class.value == 0

def test_rational_pontryagin_obstruction_torus(monkeypatch):
    torus = SimplicialComplex(coefficient_ring='Z')
    monkeypatch.setattr(SimplicialComplex, 'dimension', 2)
    monkeypatch.setattr('pysurgery.geometry.immersion_obstructions.extract_euler_class', lambda m: 0)
    
    res = compute_rational_pontryagin_obstruction(torus, target_dim=3)
    assert not isinstance(res, NonImmersibilityWitness)
