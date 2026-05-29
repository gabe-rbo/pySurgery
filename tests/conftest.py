import os

import pytest

# CRITICAL FIX for Segmentation Faults:
# Set these environment variables before any test file imports juliacall.
# This ensures that when juliacall initializes the C-level libjulia runtime,
# the required signal handlers are correctly bound for multi-threaded execution.
if "PYTHON_JULIACALL_HANDLE_SIGNALS" not in os.environ:
    os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
if "JULIA_NUM_THREADS" not in os.environ:
    os.environ["JULIA_NUM_THREADS"] = str(os.cpu_count() or 1)


@pytest.fixture(scope="session", autouse=True)
def _julia_warmup():
    """Run the full Julia warm-up exactly once at session start.

    Pays the JIT cost in one shot so every individual test (knot linking,
    Seifert pairings, SNF, etc.) hits an already-compiled hot kernel.
    """
    try:
        from pysurgery.bridge.julia_bridge import julia_engine
    except Exception:
        return
    if julia_engine.available:
        try:
            julia_engine.warmup()
        except Exception:
            pass
    yield
