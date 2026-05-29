import pytest
from pysurgery.topology.persistent_homology import compute_barcodes_exact
from pysurgery.topology.complexes import SimplicialComplex


def test_python_fallback_does_not_require_julia(monkeypatch):
    sc = SimplicialComplex.from_simplices([(0,), (1,), (2,), (0, 1), (1, 2)],
                                          close_under_faces=True)
    res = compute_barcodes_exact(sc, dimension=1, field="Z2", backend="python")
    assert res.field == "Z2"
    assert res.exact in (True, False)


def test_julia_backend_requested_but_missing_raises(monkeypatch):
    from pysurgery.topology import persistent_homology as ph
    monkeypatch.setattr(ph.JuliaBridge, "require_julia",
                        lambda self: (_ for _ in ()).throw(RuntimeError("no julia")))
    sc = SimplicialComplex.from_simplices([(0,), (1,), (0, 1)],
                                          close_under_faces=True)
    with pytest.raises(ph.BackendUnavailable):
        compute_barcodes_exact(sc, dimension=1, field="Z2", backend="julia")
