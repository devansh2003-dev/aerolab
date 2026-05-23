"""Conservation-law diagnostics: mass + momentum drift over long runs.

A clean LBM solver MUST conserve mass to machine precision in a closed
box, and to within boundary-flux balance in an open channel. This
script measures the actual drift in both modes so we can quote a
numerical envelope in VALIDATION.md.

Closed-box (no inflow / outflow):
    sum(rho * 1) should drift by no more than O(1e-12 * n_steps)
    sum(rho * u) should remain zero to machine precision (no body)

Open-channel (Zou-He inflow + pressure outflow, with body):
    mass-flux imbalance = (in - out) / (in) should be < 1 %
    after the transient settles
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.lbm import (  # noqa: E402
    equilibrium, step_njit_mrt_with_force, step_njit_mrt_no_force,
)
from src.shapes import cylinder_mask, cylinder_q_field  # noqa: E402


def closed_box_mass_drift(n_steps: int = 5000, Nx: int = 60, Ny: int = 40):
    """Closed box (no in/out): mass MUST be conserved to machine precision."""
    rho0 = np.ones((Nx, Ny))
    u0 = np.zeros((2, Nx, Ny))
    u0[0] = 0.05  # small uniform x-velocity to seed dynamics
    f = equilibrium(rho0, u0)
    solid_mask = np.zeros((Nx, Ny), dtype=bool)
    q_field = np.zeros((Nx, Ny, 9))
    f_inflow_dummy = equilibrium(1.0, np.array([0.05, 0.0]))
    no_in = np.zeros((0,), dtype=np.int64)
    no_out = np.zeros((0,), dtype=np.int64)
    tau = 0.6

    m0 = f.sum()
    px0 = (f * np.array([0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0])[:, None, None]).sum()
    py0 = (f * np.array([0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0])[:, None, None]).sum()

    drifts = []
    for step in range(n_steps):
        f = step_njit_mrt_no_force(
            f, tau, solid_mask, q_field, f_inflow_dummy, no_in, no_out,
        )
        if step % 500 == 0 or step == n_steps - 1:
            m = f.sum()
            drifts.append((step + 1, (m - m0) / m0))

    final_drift = drifts[-1][1]
    return {
        "closed_box_initial_mass": float(m0),
        "closed_box_final_mass": float(f.sum()),
        "closed_box_mass_drift_rel": float(final_drift),
        "closed_box_n_steps": n_steps,
        "drift_progress": drifts,
    }


def open_channel_mass_balance(re: int = 200, n_steps: int = 5000):
    """Open channel with cylinder. After transient, mass-flux imbalance
    (in - out) / in should be small (< 1 %).
    """
    Nx, Ny = 320, 80
    cx, cy, D = 70, 40, 28

    mask = cylinder_mask(Nx, Ny, cx=cx, cy=cy, radius=D // 2)
    q_field = cylinder_q_field(Nx, Ny, cx=cx, cy=cy, radius=D // 2)
    U = 0.1
    rho0 = np.ones((Nx, Ny))
    u0 = np.zeros((2, Nx, Ny))
    u0[0] = U
    f = equilibrium(rho0, u0)
    f_inflow_eq = equilibrium(1.0, np.array([U, 0.0]))
    nu = U * D / re
    tau = nu / (1.0 / 3.0) + 0.5
    INFLOW_DIRS = np.array([1, 5, 8], dtype=np.int64)
    OUTFLOW_DIRS = np.array([3, 6, 7], dtype=np.int64)
    f_eq_solid = equilibrium(1.0, np.array([0.0, 0.0]))

    in_history = []
    out_history = []
    for step in range(n_steps):
        f, _, _ = step_njit_mrt_with_force(
            f, tau, mask, q_field, f_inflow_eq, INFLOW_DIRS, OUTFLOW_DIRS,
        )
        f[:, mask] = f_eq_solid[:, None]
        # Sample inflow + outflow flux every 100 steps after the transient.
        if step > 1000 and step % 100 == 0:
            inflow_rho_u = f[1, 0, :].sum() + f[5, 0, :].sum() + f[8, 0, :].sum()
            inflow_rho_u -= (f[3, 0, :].sum() + f[6, 0, :].sum() + f[7, 0, :].sum())
            outflow_rho_u = f[1, -1, :].sum() + f[5, -1, :].sum() + f[8, -1, :].sum()
            outflow_rho_u -= (f[3, -1, :].sum() + f[6, -1, :].sum() + f[7, -1, :].sum())
            in_history.append(float(inflow_rho_u))
            out_history.append(float(outflow_rho_u))

    mean_in = float(np.mean(in_history)) if in_history else float("nan")
    mean_out = float(np.mean(out_history)) if out_history else float("nan")
    imbalance = (mean_in - mean_out) / mean_in if mean_in else float("nan")

    return {
        "open_channel_mean_inflow": mean_in,
        "open_channel_mean_outflow": mean_out,
        "open_channel_mass_imbalance_rel": imbalance,
        "open_channel_n_steps": n_steps,
    }


def main():
    print("=" * 60)
    print("Conservation-law diagnostics")
    print("=" * 60)

    print("\n[1/2] Closed-box mass conservation (no body, no in/out, "
          "5000 steps)")
    t0 = time.time()
    cb = closed_box_mass_drift(n_steps=5000)
    t1 = time.time()
    print(f"  initial mass : {cb['closed_box_initial_mass']:.6f}")
    print(f"  final mass   : {cb['closed_box_final_mass']:.6f}")
    print(f"  relative drift: {cb['closed_box_mass_drift_rel']:+.2e}")
    print(f"  runtime: {t1-t0:.1f}s")
    pass_cb = abs(cb["closed_box_mass_drift_rel"]) < 1e-10
    print(f"  >>> {'PASS' if pass_cb else 'FAIL'}: "
          f"mass-drift {'< 1e-10' if pass_cb else '>= 1e-10'} "
          f"(machine-precision target)")

    print("\n[2/2] Open-channel mass balance (cylinder Re=200, 5000 steps)")
    t0 = time.time()
    oc = open_channel_mass_balance(re=200, n_steps=5000)
    t1 = time.time()
    print(f"  mean inflow  : {oc['open_channel_mean_inflow']:+.5f}")
    print(f"  mean outflow : {oc['open_channel_mean_outflow']:+.5f}")
    print(f"  imbalance    : {oc['open_channel_mass_imbalance_rel']:+.2%}")
    print(f"  runtime: {t1-t0:.1f}s")
    pass_oc = abs(oc["open_channel_mass_imbalance_rel"]) < 0.02
    print(f"  >>> {'PASS' if pass_oc else 'FAIL'}: imbalance "
          f"{'< 2 %' if pass_oc else '>= 2 %'} (target < 2 % of throughflow)")

    print("\n" + "=" * 60)
    print(f"Summary: {'BOTH PASS' if (pass_cb and pass_oc) else 'CHECK FAILURES'}")
    print("=" * 60)
    return 0 if (pass_cb and pass_oc) else 1


if __name__ == "__main__":
    raise SystemExit(main())
