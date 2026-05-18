"""Pre-warm the Numba JIT cache for the LBM step variants.

The three @njit fused-step functions in ``src.lbm`` each take ~20-30 s to
compile from cold. By default the first LBM run in a fresh Streamlit
session pays this cost on top of the actual simulation time, which makes
the first click feel broken (the user clicks Run and stares at a frozen
progress bar for nearly a minute before the simulation even starts).

This module compiles all three variants on a tiny 16x8 grid at app
startup. Numba caches compiled functions by source hash, so subsequent
calls (including the real Streamlit user click on the 320x100 grid) are
near-instant. The actual simulation work is unaffected -- only the
compile cost is moved from "first user click" to "app boot".

Usage from app.py::

    @st.cache_resource(show_spinner="Pre-warming Numba JIT (one-time)...")
    def _warm_jit():
        from src.warmup import warm
        return warm()

    if mode == "Real CFD (LBM)":
        _warm_jit()

The ``@st.cache_resource`` decorator ensures ``warm()`` runs exactly once
per Streamlit process, regardless of how many times the script reruns.

The module-level ``_WARMED`` sentinel guards against accidental repeat
calls when this module is imported outside of Streamlit (tests, scripts).
"""
import numpy as np

_WARMED = False


def warm():
    """Trigger JIT compilation of all three LBM step variants.

    Returns the elapsed wall-clock seconds (useful for logging /
    measurement scripts). After the first successful call, further
    invocations short-circuit and return 0.0.
    """
    global _WARMED
    if _WARMED:
        return 0.0

    import time
    t0 = time.perf_counter()

    # Local imports keep warmup.py importable in test environments where
    # numba may not be available yet, and avoid eagerly importing numba
    # for callers that just want to query the WARMED sentinel.
    from src.lbm import (
        equilibrium, step_njit_mrt_no_force, step_njit_mrt_with_force,
        step_njit_with_force,
    )
    from src.shapes import no_bouzidi_q_field

    Nx, Ny = 16, 8
    # Initialise f at a proper equilibrium (rho=1, ux=0.1, uy=0) -- a zero
    # f_inflow would trigger division-by-zero inside the Zou-He block.
    rho0 = np.ones((Nx, Ny), dtype=np.float64)
    u0 = np.zeros((2, Nx, Ny), dtype=np.float64)
    u0[0] = 0.1
    f = equilibrium(rho0, u0)
    solid = np.zeros((Nx, Ny), dtype=bool)
    q = no_bouzidi_q_field(Nx, Ny)
    f_inflow = f[:, 0, 0].copy()  # 1D ghost values at the inlet, ux=0.1
    inflow_dirs = np.array([1, 5, 8], dtype=np.int32)
    outflow_dirs = np.array([3, 6, 7], dtype=np.int32)
    tau = 0.6

    step_njit_with_force(f.copy(), tau, solid, q, f_inflow, inflow_dirs, outflow_dirs)
    step_njit_mrt_with_force(f.copy(), tau, solid, q, f_inflow, inflow_dirs, outflow_dirs)
    step_njit_mrt_no_force(f.copy(), tau, solid, q, f_inflow, inflow_dirs, outflow_dirs)

    _WARMED = True
    return time.perf_counter() - t0
