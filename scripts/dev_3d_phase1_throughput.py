"""Phase 1 throughput re-measurement: did the clean kernel close the
Phase 0 proto-3 shortfall?

Phase 0 proto 3 measured 11.5 Mcell/s on 96^3 using the boundary-laden
``src.lbm_3d.step_bgk_3d`` kernel. The Phase 1 production kernel
``src.lbm_3d_trt.trt_periodic_step`` has no inflow / outflow / wall /
body branching in its hot loop and uses pull-streaming with signed
wrap instead of modulo. This script measures the new kernel head to
head against the old one.

Pass criterion: >= 20 Mcell/s at 96^3 (the Cloud floor the Phase 0
proto-3 estimate cited; both kernels are measured serial, no
``parallel=True``, so the comparison is apples-to-apples).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ_ROOT))

from src.lbm_3d import (  # noqa: E402
    LATTICE_VELOCITIES_3D,
    LATTICE_WEIGHTS_3D,
    OPPOSITE_3D,
    init_population,
    step_bgk_3d,
)
from src.lbm_3d_trt import (  # noqa: E402
    init_tgv,
    init_tgv_aos,
    omegas_for_trt,
    trt_periodic_step,
    trt_periodic_step_aos,
)


def time_phase0(N: int, n_warm: int = 5, n_meas: int = 30) -> float:
    body = np.zeros((N, N, N), dtype=np.bool_)
    omega = np.float32(1.0 / (3.0 * 0.02 + 0.5))
    f = init_population(N, N, N, u_in=0.04)
    f_next = f.copy()
    # JIT warm-up
    step_bgk_3d(f, f_next, body, omega, np.float32(0.04))
    f, f_next = f_next, f
    for _ in range(n_warm):
        step_bgk_3d(f, f_next, body, omega, np.float32(0.04))
        f, f_next = f_next, f
    t0 = time.time()
    for _ in range(n_meas):
        step_bgk_3d(f, f_next, body, omega, np.float32(0.04))
        f, f_next = f_next, f
    return (time.time() - t0) / n_meas


def time_phase1(N: int, n_warm: int = 5, n_meas: int = 30) -> float:
    s_plus, s_minus = omegas_for_trt(0.01)
    s_plus = np.float32(s_plus)
    s_minus = np.float32(s_minus)
    vel = LATTICE_VELOCITIES_3D.astype(np.int32)
    weights = LATTICE_WEIGHTS_3D.astype(np.float32)
    opp = OPPOSITE_3D.astype(np.int32)
    f = init_tgv(N, 0.04, dtype=np.float32)
    f_next = f.copy()
    trt_periodic_step(f, f_next, s_plus, s_minus, vel, weights, opp)
    f, f_next = f_next, f
    for _ in range(n_warm):
        trt_periodic_step(f, f_next, s_plus, s_minus, vel, weights, opp)
        f, f_next = f_next, f
    t0 = time.time()
    for _ in range(n_meas):
        trt_periodic_step(f, f_next, s_plus, s_minus, vel, weights, opp)
        f, f_next = f_next, f
    return (time.time() - t0) / n_meas


def time_phase1_aos(N: int, n_warm: int = 5, n_meas: int = 30) -> float:
    """AoS-layout TRT kernel — same math as time_phase1 but f shape
    is (Nx, Ny, Nz, 19) so per-cell reads are unit-stride."""
    s_plus, s_minus = omegas_for_trt(0.01)
    s_plus = np.float32(s_plus)
    s_minus = np.float32(s_minus)
    vel = LATTICE_VELOCITIES_3D.astype(np.int32)
    weights = LATTICE_WEIGHTS_3D.astype(np.float32)
    opp = OPPOSITE_3D.astype(np.int32)
    f = init_tgv_aos(N, 0.04, dtype=np.float32)
    f_next = f.copy()
    trt_periodic_step_aos(f, f_next, s_plus, s_minus, vel, weights, opp)
    f, f_next = f_next, f
    for _ in range(n_warm):
        trt_periodic_step_aos(f, f_next, s_plus, s_minus, vel, weights, opp)
        f, f_next = f_next, f
    t0 = time.time()
    for _ in range(n_meas):
        trt_periodic_step_aos(f, f_next, s_plus, s_minus, vel, weights, opp)
        f, f_next = f_next, f
    return (time.time() - t0) / n_meas


def main() -> int:
    print("# Phase 1 throughput: AoS TRT vs SoA TRT vs Phase 0 BGK (SoA, boundary-laden)")
    print(f"{'N':>5} {'Ph0 Mcell/s':>13} {'Ph1 SoA':>10} "
          f"{'Ph1 AoS':>10} {'AoS / SoA':>11} {'AoS / Ph0':>11}")
    target = 20.0
    rows = []
    for N in (48, 64, 96):
        cells = float(N) ** 3
        t0 = time_phase0(N)
        t1_soa = time_phase1(N)
        t1_aos = time_phase1_aos(N)
        mc0 = cells * 1e-6 / t0
        mc_soa = cells * 1e-6 / t1_soa
        mc_aos = cells * 1e-6 / t1_aos
        rows.append((N, mc0, mc_soa, mc_aos))
        print(f"{N:>5} {mc0:>13.2f} {mc_soa:>10.2f} {mc_aos:>10.2f} "
              f"{mc_aos/mc_soa:>10.2f}x {mc_aos/mc0:>10.2f}x")
    print()
    r96 = next(r for r in rows if r[0] == 96)
    print(f"[verdict] AoS 96^3 throughput = {r96[3]:.2f} Mcell/s "
          f"(Cloud floor {target:.0f})")
    if r96[3] >= target:
        print(f"[PASS] {r96[3]/target:.1f}x the Cloud floor. AoS closed "
              "the proto-3 throughput shortfall. The plan-§3.D-6 fallback "
              "from SoA to AoS pays off as the doc predicted.")
        return 0
    print(f"[PARTIAL] AoS at 96^3 = {r96[3]:.2f} Mcell/s is still under "
          f"the 20 Mcell/s floor. The remaining levers are AoS + "
          f"parallel=True (offline), drop to 64^3 (Interactive3D toy), "
          f"or AA streaming (Phase 5 reconsider).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
