import pytest
from unittest.mock import patch

from pysurgery.topology.complexes import _simplicial_product
from pysurgery.core.exceptions import LadderProgressError
from pysurgery.surgery import SurgerySession
from pysurgery.core.generator_models import HomologyGenerator
from pysurgery.homology.homology_generators import hk_generators_z
from discrete_surface_data import build_torus, build_rp2, build_tetrahedron
from pysurgery.auto_surgery import (
    auto_kill_pi1,
    auto_kill_homology_dim,
    Pi1KillReport,
    HKillReport,
)

# ── Test 1: hk_generators_z ───────────────────────────────────────────────────

def test_homology_generators_z_torus():
    """Verify that hk_generators_z correctly extracts free integer homology generators for T²."""
    T2 = build_torus()
    
    # H1(T2) has rank 2, both free (summand_label = "Z")
    gens = hk_generators_z(T2, k=1, backend="python")
    assert len(gens) == 2
    for g in gens:
        assert isinstance(g, HomologyGenerator)
        assert g.dimension == 1
        assert g.summand_label == "Z"
        assert len(g.support_simplices) > 0


def test_homology_generators_z_rp2():
    """Verify that hk_generators_z correctly extracts torsion integer homology generators for RP²."""
    rp2 = build_rp2()
    
    # H1(RP2) has 1 torsion generator of order 2 (summand_label = "Z/2")
    gens = hk_generators_z(rp2, k=1, backend="python")
    assert len(gens) == 1
    g = gens[0]
    assert isinstance(g, HomologyGenerator)
    assert g.dimension == 1
    assert g.summand_label == "Z/2"
    assert len(g.support_simplices) > 0


def test_homology_generators_z_sphere():
    """Verify that hk_generators_z correctly extracts free integer homology generators for S²."""
    S2 = build_tetrahedron()
    
    # H2(S2) has rank 1, free (summand_label = "Z")
    gens = hk_generators_z(S2, k=2, backend="python")
    assert len(gens) == 1
    g = gens[0]
    assert isinstance(g, HomologyGenerator)
    assert g.dimension == 2
    assert g.summand_label == "Z"
    assert len(g.support_simplices) > 0


# ── Test 2: auto_kill_pi1 ─────────────────────────────────────────────────────

def test_auto_kill_pi1_on_torus():
    """Verify that auto_kill_pi1 identifies and surgers fundamental group generators of Torus."""
    T2 = build_torus()
    
    session = SurgerySession(ambient_space=T2, objects={"T2": T2})
    
    # Mock attach_handle to prevent actual simplicial mutations during loops
    with patch.object(session.AmbientSpace, "attach_handle"):
        report = auto_kill_pi1(session, "T2", backend="python")
        
    assert isinstance(report, Pi1KillReport)
    assert report.name == "T2"
    assert len(report.steps) > 0
    for step in report.steps:
        assert step.attached is True
        assert step.verified_killed is True
        assert len(step.cycle) > 0


# ── Test 3: auto_kill_homology_dim ────────────────────────────────────────────

def test_auto_kill_homology_dim_on_S2xS2():
    """Verify that auto_kill_homology_dim detects and surgers homology generators on S² x S²."""
    S2 = build_tetrahedron()
    S2xS2 = _simplicial_product(S2, S2)
    
    session = SurgerySession(ambient_space=S2xS2, objects={"S2xS2": S2xS2})
    
    # Verify H2(S2 x S2) initially has rank 2
    betti_before = S2xS2.betti_numbers(backend="python")
    assert betti_before.get(2, 0) == 2
    
    # Mock attach_handle to verify step execution and report outcomes
    with patch.object(session.AmbientSpace, "attach_handle"):
        report = auto_kill_homology_dim(session, "S2xS2", k=2, backend="python")
        
    assert isinstance(report, HKillReport)
    assert report.name == "S2xS2"
    assert report.k == 2
    assert len(report.steps) > 0
    for step in report.steps:
        assert step.attached is True
        assert step.summand == "Z"
        assert step.exact is True


def test_auto_kill_homology_dim_hurewicz_precondition_check():
    """Verify that auto_kill_homology_dim enforces Hurewicz preconditions strictly."""
    from discrete_surface_data import build_s1, build_s3
    
    S1 = build_s1()
    S3 = build_s3()
    S1xS3 = _simplicial_product(S1, S3)
    
    session = SurgerySession(ambient_space=S1xS3, objects={"S1xS3": S1xS3})
    
    # Attempting to kill H2 should raise LadderProgressError since H1(S1xS3) = Z != 0
    with pytest.raises(LadderProgressError) as exc:
        auto_kill_homology_dim(session, "S1xS3", k=2, backend="python")
        
    assert "H_1" in str(exc.value)
    assert "Hurewicz" in str(exc.value)

