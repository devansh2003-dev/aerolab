"""Long-time stability sweep, cylinder Re = 100, Standard preset.

Card #10 (David Artemyev review, 2026-05-27): SciML's standing critique
of low-cost solvers is that long-time extrapolation is the unsolved
problem -- the math may be right step-by-step but accumulated drift /
boundary-condition error eventually wins. AeroLab makes no claim to
solve that, but we should document **what fails first** when we push
the validated case past the validated window.

We pick cylinder Re = 100, Standard preset (D = 28, B = 0.35) since
that case is the most-extensively-instrumented in the suite, and run
it at ascending step counts. For each run we record:

  - whether the run finished without nan/inf (solve_lbm raises if not)
  - mass drift in percent (mean rho over the domain, t_end vs t_0)
  - peak velocity in lattice units (max over all snapshots)
  - Cd_mean over the **last 50 D/U** of the record (so the comparison
    is windowed-equivalent across run lengths)
  - St measured by the same FFT-on-last-half as the validation runs

The result is committed to ``data/validation/long_time_cylinder_re100.json``
and referenced from VALIDATION.md's "Long-time behaviour" appendix.
Reproducible via::

    python scripts/long_time_stability_cylinder.py

Typical wall time: ~7 minutes on a 4-core laptop (sequential runs at
n_frames = 100, 200, 400, 800 -- corresponding to 12.5 / 25 / 50 / 100
D/U on the Standard preset).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# Add project root so `from src.foo import ...` works from any CWD.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.lbm_render import solve_lbm  # noqa: E402

# Lattice constants that solve_lbm uses internally for the Standard
# preset. Mirrored here so we can convert D/U to step counts honestly.
U_INFLOW = 0.1
STEPS_PER_FRAME = 35
CHAR_LENGTH_STANDARD = 28  # cylinder diameter in lattice cells

# n_frames -> approximate t_end in D/U units.
# steps = n_frames * STEPS_PER_FRAME; t_in_DU = steps * U_INFLOW / D.
N_FRAMES_SWEEP = [100, 200, 400, 800]

OUT_PATH = (
    _PROJECT_ROOT / "data" / "validation" / "long_time_cylinder_re100.json"
)


def _t_end_in_du(n_frames: int) -> float:
    return (n_frames * STEPS_PER_FRAME) * U_INFLOW / CHAR_LENGTH_STANDARD


def _diagnostics_from_solve(out: dict, n_frames: int) -> dict:
    """Extract the long-time diagnostics from a solve_lbm() return."""
    snapshots = out["snapshots"]
    cd_hist = np.asarray(out["cd_history"], dtype=np.float64)
    cl_hist = np.asarray(out["cl_history"], dtype=np.float64)
    n_steps = int(out["n_steps"])
    char_length = float(out["char_length"])

    # Mass drift: mean density now vs at t_0. solve_lbm seeds f from a
    # uniform rho=1, U_INFLOW velocity, so initial mean rho is ~1.0
    # exactly. Drift is signed because the Zou-He outlet leaks slowly.
    initial_mass = 1.0  # known from the equilibrium seed
    final_rho = snapshots[-1]["rho"]
    final_mass = float(final_rho.mean())
    mass_drift_pct = 100.0 * (final_mass - initial_mass) / initial_mass

    # Peak velocity (vector magnitude). Stays well below sqrt(3)*c_s
    # for a stable run; close to that is the populations-go-negative
    # warning signal.
    u_peak = 0.0
    for snap in snapshots:
        ux = snap["u_x"]
        uy = snap["u_y"]
        mag = float(np.max(np.hypot(ux, uy)))
        u_peak = max(u_peak, mag)

    # Cd_mean over the LAST 50 D/U so window-equivalent across runs.
    # 50 D/U = 50 * D / U = 14000 steps at the Standard preset (D=28,
    # U=0.1). If the run is shorter than 50 D/U we fall back to the
    # last third (matches solve_lbm's own stable-tail convention).
    window_steps = int(50.0 * char_length / U_INFLOW)
    if n_steps > window_steps:
        cd_mean = float(cd_hist[-window_steps:].mean())
        cl_mean = float(cl_hist[-window_steps:].mean())
        cd_window_label = "last 50 D/U"
    else:
        tail_start = max(1, n_steps // 3 * 2)
        cd_mean = float(cd_hist[tail_start:].mean())
        cl_mean = float(cl_hist[tail_start:].mean())
        cd_window_label = "last 1/3 (run shorter than 50 D/U)"

    return {
        "n_frames": int(n_frames),
        "n_steps": n_steps,
        "t_end_DU": round(_t_end_in_du(n_frames), 2),
        "finished_clean": bool(np.isfinite(cd_hist).all() and
                                np.isfinite(cl_hist).all()),
        "mass_drift_pct": round(mass_drift_pct, 4),
        "u_peak_lattice": round(u_peak, 5),
        "cd_mean": round(cd_mean, 4),
        "cl_mean": round(cl_mean, 4),
        "cd_window": cd_window_label,
        "strouhal": (round(float(out["strouhal"]), 4)
                     if np.isfinite(float(out["strouhal"])) else None),
        "strouhal_bin_width": round(float(out.get("strouhal_bin_width", 0.0)), 5),
        "strouhal_n_cycles": round(float(out.get("strouhal_n_cycles", 0.0)), 2),
    }


def run_sweep() -> list[dict]:
    rows: list[dict] = []
    for n_frames in N_FRAMES_SWEEP:
        print(
            f"[{time.strftime('%H:%M:%S')}] running n_frames={n_frames} "
            f"(~{_t_end_in_du(n_frames):.1f} D/U) ..."
        )
        t0 = time.time()
        try:
            out = solve_lbm(
                "Cylinder", 100, 0.0, "Standard (320 x 80)",
                n_frames=n_frames,
            )
        except ValueError as exc:
            # Divergence (nan/inf) raises ValueError from solve_lbm with
            # the offending frame index baked in. Record as a row so the
            # table shows WHERE the run gave up.
            rows.append({
                "n_frames": int(n_frames),
                "n_steps": int(n_frames * STEPS_PER_FRAME),
                "t_end_DU": round(_t_end_in_du(n_frames), 2),
                "finished_clean": False,
                "first_failure": str(exc),
            })
            print(f"  -> diverged: {exc}")
            continue
        runtime = time.time() - t0
        row = _diagnostics_from_solve(out, n_frames)
        row["runtime_sec"] = round(runtime, 2)
        rows.append(row)
        print(
            f"  -> Cd_mean={row['cd_mean']:.3f} "
            f"mass_drift={row['mass_drift_pct']:.3f}% "
            f"u_peak={row['u_peak_lattice']:.4f} "
            f"({runtime:.1f}s)"
        )
        # Release snapshot memory before the next run.
        del out
    return rows


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = run_sweep()
    payload = {
        "case": "Cylinder Re=100, Standard preset (D=28, B=0.35)",
        "preset": "Standard (320 x 80)",
        "U_inflow_lattice": U_INFLOW,
        "char_length_cells": CHAR_LENGTH_STANDARD,
        "results": rows,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
