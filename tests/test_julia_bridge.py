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


def test_julia_bridge_warmup_unavailable_is_nonfatal(monkeypatch):
    monkeypatch.setattr(julia_engine, "_initialized", True, raising=False)
    monkeypatch.setattr(julia_engine, "_available", False, raising=False)
    monkeypatch.setattr(julia_engine, "error", "missing juliacall", raising=False)

    report = julia_engine.warmup()
    assert report["available"] is False
    assert report["mode"] == "full"
    assert isinstance(report["failed"], dict)


def test_julia_bridge_warmup_full_executes_and_caches(monkeypatch):
    monkeypatch.setattr(julia_engine, "_initialized", True, raising=False)
    monkeypatch.setattr(julia_engine, "_available", True, raising=False)
    monkeypatch.setattr(julia_engine, "jl", object(), raising=False)
    monkeypatch.setattr(julia_engine, "backend", object(), raising=False)
    monkeypatch.setattr(julia_engine, "_warmup_level", 0, raising=False)
    monkeypatch.setattr(julia_engine, "_warmup_report", {}, raising=False)

    calls = {"minimal": 0, "full": 0}

    def _minimal_workloads():
        return [
            ("min_probe", lambda: calls.__setitem__("minimal", calls["minimal"] + 1))
        ]

    def _full_workloads():
        return [("full_probe", lambda: calls.__setitem__("full", calls["full"] + 1))]

    monkeypatch.setattr(
        julia_engine, "_minimal_warmup_workloads", _minimal_workloads, raising=False
    )
    monkeypatch.setattr(
        julia_engine, "_full_warmup_workloads", _full_workloads, raising=False
    )

    report_first = julia_engine.warmup()
    report_second = julia_engine.warmup()

    assert calls["minimal"] == 1
    assert calls["full"] == 1
    assert report_first["available"] is True
    assert report_first["mode"] == "full"
    assert report_first["cached"] is False
    assert report_second["cached"] is True
