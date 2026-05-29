from pysurgery.homeomorphism_witness import build_homeomorphism_witness
from discrete_surface_data import build_rp2, build_torus
from pysurgery.geometry.immersion_obstructions import NonImmersibilityWitness, ImmersibilityInconclusive

def test_homeomorphism_pipeline_integration_rp2(monkeypatch):
    rp2 = build_rp2()
    
    # We monkeypatch the analysis so we can cleanly verify the pipeline hooks it in
    
    # Run the full pipeline with immersion_target_dim = 2
    res = build_homeomorphism_witness(
        c1=rp2, c2=rp2, dim=2, immersion_target_dim=2, allow_approx=False
    )
    
    assert res.status == "success"
    assert res.witness is not None
    assert "immersion_obstruction" in res.witness.certificates
    
    obstruction = res.witness.certificates["immersion_obstruction"]
    # For target_dim=2, k=0. RP2 w_bar_1 or w_bar_2 will not vanish, so it shouldn't immerse
    assert isinstance(obstruction, NonImmersibilityWitness)
    assert obstruction.exact is True
    assert obstruction.immersible is False

def test_homeomorphism_pipeline_integration_torus():
    torus = build_torus()
    
    # Target dim 3 -> codimension 1
    res = build_homeomorphism_witness(
        c1=torus, c2=torus, dim=2, immersion_target_dim=3, allow_approx=False
    )
    
    assert res.status == "success"
    assert res.witness is not None
    assert "immersion_obstruction" in res.witness.certificates
    
    obstruction = res.witness.certificates["immersion_obstruction"]
    # Torus immerses in R^3 (and even R^2?) No, R^3.
    # The obstruction should be inconclusive (meaning no obstruction found)
    assert isinstance(obstruction, ImmersibilityInconclusive)
    assert obstruction.exact is True

