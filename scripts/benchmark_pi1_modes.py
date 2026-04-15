from __future__ import annotations

import argparse
import time

import numpy as np

from pysurgery.core.complexes import CWComplex
from pysurgery.core.fundamental_group import extract_pi_1_with_traces
from pysurgery.bridge.julia_bridge import julia_engine
from pysurgery.integrations.gudhi_bridge import extract_complex_data


def build_torus_4x4_chain_complex(R: float = 3.0, r: float = 1.0):
    import gudhi

    nu = nv = 4
    pts = []

    def idx(i: int, j: int) -> int:
        return i * nv + j

    st = gudhi.SimplexTree()
    for i in range(nu):
        u = 2.0 * np.pi * i / nu
        for j in range(nv):
            v = 2.0 * np.pi * j / nv
            x = (R + r * np.cos(v)) * np.cos(u)
            y = (R + r * np.cos(v)) * np.sin(u)
            z = r * np.sin(v)
            pts.append([x, y, z])
            st.insert([idx(i, j)])

    for i in range(nu):
        ip = (i + 1) % nu
        for j in range(nv):
            jp = (j + 1) % nv
            a = idx(i, j)
            b = idx(ip, j)
            c = idx(ip, jp)
            d = idx(i, jp)
            st.insert([a, b, c])
            st.insert([a, c, d])

    boundaries, cells, _, _ = extract_complex_data(st)
    cw = CWComplex(cells=cells, attaching_maps=boundaries, coefficient_ring="Z")
    return cw


def _avg_trace_len(pi1) -> float:
    if not pi1.traces:
        return 0.0
    return float(sum(len(t.undirected_edge_path) for t in pi1.traces) / len(pi1.traces))


def run_case(cw: CWComplex, *, mode: str, force_python: bool) -> dict:
    original_available = julia_engine.available
    if force_python:
        julia_engine.available = False

    try:
        t0 = time.perf_counter()
        out = extract_pi_1_with_traces(cw, simplify=True, generator_mode=mode)
        elapsed = time.perf_counter() - t0
    finally:
        if force_python:
            julia_engine.available = original_available

    return {
        "mode": mode,
        "backend": out.backend_used,
        "runtime_s": elapsed,
        "raw_count": out.raw_generator_count,
        "optimized_count": out.optimized_generator_count,
        "selected_count": len(out.generators),
        "avg_trace_len": _avg_trace_len(out),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark pi1 raw/optimized generator extraction."
    )
    parser.add_argument(
        "--force-python",
        action="store_true",
        help="Disable Julia backend for this benchmark run.",
    )
    args = parser.parse_args()

    cw = build_torus_4x4_chain_complex()
    rows = [
        run_case(cw, mode="raw", force_python=args.force_python),
        run_case(cw, mode="optimized", force_python=args.force_python),
    ]

    print("pi1 benchmark results")
    print(
        "mode\tbackend\truntime_s\traw_count\toptimized_count\tselected_count\tavg_trace_len"
    )
    for r in rows:
        print(
            f"{r['mode']}\t{r['backend']}\t{r['runtime_s']:.6f}\t"
            f"{r['raw_count']}\t{r['optimized_count']}\t{r['selected_count']}\t{r['avg_trace_len']:.3f}"
        )


if __name__ == "__main__":
    main()
