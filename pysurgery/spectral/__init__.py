def __getattr__(name):
    if name == "SerreSpectralSequence":
        from .spectral_sequences import SerreSpectralSequence  # noqa: F401
        return SerreSpectralSequence
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "SerreSpectralSequence",
]
