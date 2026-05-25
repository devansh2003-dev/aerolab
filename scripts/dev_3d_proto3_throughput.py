"""Phase 0 prototype #3: D3Q19 Numba kernel throughput + JIT compile time.

Confirms the compute budget from 3D_RESEARCH_PLAN.md section 1.2: that
the Cloud single-vCPU path can hit ~20 to 50 Mcell-updates/s at the
~96 cubed Interactive3D target, and that the laptop offline path is
in the 50 to 150 Mcell/s range. Uses the existing src/lbm_3d
`step_bgk_3d` kernel -- whose physics will be replaced in Phase 1
(BGK -> TRT) but whose memory layout and JIT shape are the production
target.

Pass criterion: 96 cubed single-core single-thread throughput
>= 20 Mcell-updates/s on this laptop, the Cloud target floor.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ_ROOT))

from src.lbm_3d import init_population, step_bgk_3d  # noqa: E402


def time_grid(N: int, n_warm: int = 5, n_meas: int = 30) -> dict:
    body = np.zeros((N, N, N), dtype=np.bool_)
    omega = np.float32(1.0 / (3.0 * 0.02 + 0.5))
    f = init_population(N, N, N, u_in=0.04)
    f_next = f.copy()
    # First call: JIT compile + 1 step. The compile dominates.
    t0 = time.time()
    step_bgk_3d(f, f_next, body, omega, np.float32(0.04))
    f, f_next = f_next, f
    t_first = time.time() - t0
    # Warm steps (discard).
    for _ in range(n_warm):
        step_bgk_3d(f, f_next, body, omega, np.float32(0.04))
        f, f_next = f_next, f
    # Measurement.
    t0 = time.time()
    for _ in range(n_meas):
        step_bgk_3d(f, f_next, body, omega, np.float32(0.04))
        f, f_next = f_next, f
    elapsed = time.time() - t0
    per_step = elapsed / n_meas
    cells = float(N) ** 3
    mcells = cells * 1e-6 / per_step
    # Memory: 2 * (19 * N^3) * 4 bytes (float32 double-buffer)
    mem_mb = 2 * 19 * cells * 4 / (1024 * 1024)
    return {
        "N": N,
        "t_first": t_first,
        "per_step": per_step,
        "mcells_per_s": mcells,
        "mem_mb": mem_mb,
    }


def main() -> int:
    print(f"# Proto 3: D3Q19 single-core throughput (Numba @njit, fastmath, serial)")
    print(f"{'N':>5} {'mem MB (2 bufs)':>16} {'JIT first step':>15} "
          f"{'per step (ms)':>14} {'Mcell/s':>10}")
    rows = []
    for N in (48, 64, 96):
        r = time_grid(N)
        rows.append(r)
        print(f"{r['N']:>5} {r['mem_mb']:>16.1f} {r['t_first']:>13.2f} s  "
              f"{r['per_step']*1000:>14.2f} {r['mcells_per_s']:>10.2f}")
    print()
    target = 20.0
    r96 = next(r for r in rows if r["N"] == 96)
    print(f"[verdict] 96 cubed throughput = {r96['mcells_per_s']:.2f} Mcell/s "
          f"(target floor {target:.0f})")
    if r96["mcells_per_s"] >= target:
        margin = r96["mcells_per_s"] / target
        print(f"[PASS] {margin:.1f}x the Cloud floor. With ~5000 steps a "
              f"96 cubed run lands at "
              f"~{r96['per_step']*5000:.0f} s on this laptop.")
        return 0
    print("[FAIL] 96 cubed throughput below the Cloud floor. Either drop to "
          "64 cubed for the Interactive preset, or pursue the in-place AA "
          "scheme to amortise.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
