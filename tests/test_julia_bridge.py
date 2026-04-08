import pytest

from pysurgery.bridge.julia_bridge import JuliaBridge, julia_engine


def test_julia_bridge_singleton_identity():
    a = JuliaBridge()
    b = JuliaBridge()
    assert a is b
    assert a is julia_engine


def test_julia_bridge_require_julia_behavior():
    if julia_engine.available:
        # Should not raise when backend is available.
        julia_engine.require_julia()
    else:
        from pysurgery.core.exceptions import SurgeryError
        with pytest.raises(SurgeryError):
            julia_engine.require_julia()

