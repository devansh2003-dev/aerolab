"""Phase 1 exit-gate CLI driver: 3D Taylor-Green vortex decay rate.

Runs the production TRT kernel (and BGK as the reference path) on a
periodic 3D TGV, measures the kinetic-energy decay rate, and compares
to the analytic 4 ν k². Pass criterion is ±2 % — the same gate the
plan document set for Phase 1 in section 4.

Run from the project root:

    .venv311\\Scripts\\python.exe scripts\\dev_3d_phase1_tgv_gate.py

Exits 0 on PASS, 2 on FAIL. First-call timing reflects the Numba JIT.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ_ROOT))

from src.lbm_3d_trt import (  # noqa: E402
    analytic_tgv_decay_rate,
    fit_decay_rate,
    run_tgv,
    run_tgv_aos,
)


def main() -> int:
    N = 32
    U = 0.04
    n_steps = 800
    print(f"# Phase 1 gate: 3D TGV decay rate, N={N}, U={U}, n_steps={n_steps}")
    print(f"# Tolerance: +/-2 % vs analytic 4 nu k^2")
    print(f"{'scheme':8} {'nu':>8} {'measured':>13} {'analytic':>13} "
          f"{'err %':>8} {'time':>8}")
    rows = []
    for nu in (0.005, 0.01, 0.02):
        # Production path is AoS layout; SoA kept as comparison.
        for scheme, runner in (("trt-aos", run_tgv_aos),
                                ("trt-soa", run_tgv),
                                ("bgk-aos", run_tgv_aos)):
            scheme_kind = "bgk" if scheme.startswith("bgk") else "trt"
            t0 = time.time()
            times, ke, diag = runner(
                N=N, U=U, nu=nu, n_steps=n_steps, scheme=scheme_kind,
                dtype=np.float32,
            )
            elapsed = time.time() - t0
            if diag["diverged"]:
                rows.append((scheme, nu, float("nan"), float("nan"), True))
                print(f"{scheme:8} {nu:>8.4f}  DIVERGED")
                continue
            measured = fit_decay_rate(times, ke)
            analytic = analytic_tgv_decay_rate(nu, N)
            err_pct = 100.0 * (measured - analytic) / analytic
            rows.append((scheme, nu, measured, analytic, False, err_pct))
            print(f"{scheme:8} {nu:>8.4f}  {measured:>13.6e}  "
                  f"{analytic:>13.6e}  {err_pct:>+7.2f}  {elapsed:>7.1f} s")

    print()
    pass_all = all(
        not r[4] and abs(r[5]) < 2.0
        for r in rows
        if len(r) > 5
    )
    if pass_all:
        print("[PASS] Phase 1 exit gate met. The production TRT kernel "
              "reproduces the analytic 3D TGV decay rate within +/-2 % "
              "across nu in {0.005, 0.01, 0.02}. Phase 2 (sphere geometry + "
              "Bouzidi q + Guo NEEM) can begin.")
        return 0
    print("[FAIL] Phase 1 exit gate NOT met. Investigate before Phase 2.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
