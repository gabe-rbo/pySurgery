import numpy as np
from pysurgery.algebra.intersection_forms import IntersectionForm
from pysurgery.manifolds.rational_surgery import compute_l_group_rational
from pysurgery.topology.persistent_homology import Barcode, BarcodeResult


def test_rational_l_group_marks_float_path_inexact(monkeypatch):
    # Force the float-eigenvalue fallback by disabling the exact SNF path.
    import pysurgery.utils.signature as sig_mod
    monkeypatch.setattr(sig_mod, 'signature_via_snf',
                        lambda f: (_ for _ in ()).throw(RuntimeError("test: forced fail")))
    form = IntersectionForm(matrix=np.array([[0, 1], [1, 0]]), dimension=4)
    obs = compute_l_group_rational(dimension=4, pi="trivial", form=form)
    if obs.signature is not None:
        assert obs.exact is False, "float-eigenvalue path must not claim exact=True"


def test_barcode_carries_exact_field():
    bc = Barcode(birth=0, death=1, dim=0)
    assert hasattr(bc, "exact")
    res = BarcodeResult(barcodes=[bc], field="Z2")
    assert hasattr(res, "exact")


def test_temporal_default_inexact():
    from pysurgery.topology.temporal_topology import TemporalBarcode
    tb = TemporalBarcode(dimension=0, births=[], deaths=[], parameters=[])
    assert tb.exact is False
