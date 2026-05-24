"""Measure the cold-start cost of the LBM path.

Reports three numbers per JIT step variant:

  1. Cold compile time -- how long the @njit decorator takes on the
     first call, with Numba's disk cache cleared.
  2. Warm call time   -- a single step on the same tiny grid after the
     function has been compiled.
  3. Total page-load delta -- sum of cold compiles, which is what
     ``src.warmup.warm()`` pays at app startup (and what a fresh
     Streamlit Cloud container will spend in the loading spinner).

To get a true cold measurement we delete ``__pycache__/`` for ``src.lbm``
and ``src.shapes`` before importing them. Run this on a quiet machine;
results depend heavily on CPU clock and Numba version.

Usage::

    python scripts/dev_measure_cold_start.py

The numbers reported are LOCAL machine numbers. Streamlit Cloud's free
tier shared CPU is typically 1.5-2x slower than a modern laptop, so
multiply by ~2 for a Cloud expectation.
"""
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _purge_pycache():
    """Delete __pycache__/ under src/ to force Numba to re-compile.

    Numba caches compiled @njit functions on disk (when ``cache=True``
    is set on the decorator). Deleting the cache directory is the
    cleanest way to force a true cold compile.
    """
    repo_root = Path(__file__).resolve().parents[1]
    src_pycache = repo_root / "src" / "__pycache__"
    if src_pycache.exists():
        shutil.rmtree(src_pycache)
        print(f"[cold-start] purged {src_pycache}")
    else:
        print(f"[cold-start] no pycache at {src_pycache} (already cold)")


def _import_and_compile():
    """Import LBM + call each JIT step variant once to trigger compile.

    Returns a dict of {variant_name: cold_compile_seconds} plus a
    'total' entry covering all three.
    """
    import numpy as np

    # Drop any previously imported src.lbm so this is a true fresh
    # import (matters when this script is re-run in the same process).
    for mod_name in [m for m in sys.modules if m.startswith("src.")]:
        del sys.modules[mod_name]

    timings = {}
    t_total = time.perf_counter()

    print("[cold-start] importing src.lbm + src.shapes (cold)...")
    t = time.perf_counter()
    from src.lbm import (
        equilibrium,
        step_njit_mrt_no_force,
        step_njit_mrt_with_force,
        step_njit_with_force,
    )
    from src.shapes import no_bouzidi_q_field
    timings["import"] = time.perf_counter() - t
    print(f"[cold-start]   import: {timings['import']:.2f} s")

    Nx, Ny = 16, 8
    rho0 = np.ones((Nx, Ny), dtype=np.float64)
    u0 = np.zeros((2, Nx, Ny), dtype=np.float64)
    u0[0] = 0.1
    f = equilibrium(rho0, u0)
    solid = np.zeros((Nx, Ny), dtype=bool)
    q = no_bouzidi_q_field(Nx, Ny)
    f_inflow = f[:, 0, 0].copy()
    tau = 0.6

    variants = [
        ("step_njit_with_force (BGK + force)", step_njit_with_force),
        ("step_njit_mrt_with_force (MRT + force)", step_njit_mrt_with_force),
        ("step_njit_mrt_no_force (MRT, viz path)", step_njit_mrt_no_force),
    ]

    for label, step_fn in variants:
        t = time.perf_counter()
        step_fn(f.copy(), tau, solid, q, f_inflow, True, True)
        cold = time.perf_counter() - t
        t = time.perf_counter()
        step_fn(f.copy(), tau, solid, q, f_inflow, True, True)
        warm = time.perf_counter() - t
        timings[label] = {"cold_s": cold, "warm_ms": warm * 1000.0}
        print(f"[cold-start]   {label}")
        print(f"[cold-start]     cold compile : {cold:.2f} s")
        print(f"[cold-start]     warm call    : {warm * 1000:.2f} ms")

    timings["total"] = time.perf_counter() - t_total
    return timings


def main():
    print("=" * 60)
    print("AeroLab JIT cold-start measurement")
    print("=" * 60)
    _purge_pycache()
    timings = _import_and_compile()
    print()
    print(f"Total cold-start: {timings['total']:.2f} s")
    print()
    print("Streamlit Cloud free tier estimate (1.5-2x local):")
    print(f"  ~ {timings['total'] * 1.5:.0f}-{timings['total'] * 2.0:.0f} s")
    print()
    print("This is what src.warmup.warm() costs on a fresh container.")
    print("Subsequent user clicks pay only the simulation time, not the compile.")


if __name__ == "__main__":
    main()
