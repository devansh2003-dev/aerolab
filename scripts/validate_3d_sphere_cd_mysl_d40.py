"""Sphere Re = 100 Cd validation at D = 40 with MYSL momentum exchange.

Companion to ``scripts/validate_3d_sphere_cd_d40.py`` and the
**second half** of audit item #8 from the v0.6.5.1 senior re-audit.

The §8.3.3 D = 40 bake (Ladd 1994 simplified) measured Cd = 1.528
(+40.2 % vs CGW 1.09). The grid-resolution contribution to the
+50.9 % D = 20 baseline bias was only ~ 7 percentage points; the
remaining ~ 33 % lives in the simplified Ladd 1994 momentum-exchange
formula, which does not weight wall links by their Bouzidi q-fraction.

This script re-runs the IDENTICAL D = 40 bake and reports BOTH the
Ladd post-stream and MYSL 2002 forces at the end. The flow itself is
unchanged (same Bouzidi BB, same TRT collision, same Guo NEEM
boundary conditions) -- only the force POST-PROCESSING differs.

Reference: Mei, R., Yu, D., Shyy, W., Luo, L.-S. (2002). "Force
evaluation in the lattice Boltzmann method involving curved
geometry." Phys. Rev. E 65, 041203.

Reference Cd from Clift-Grace-Weber 1978: Cd = 1.09 at Re = 100.

Outputs
-------
data/validation_3d_sphere_re100_d40_mysl.json
    -- includes BOTH Cd values so the comparison is committed.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.forces_3d import (  # noqa: E402
    drag_coefficient_3d,
    momentum_exchange_force_3d_mysl,
    momentum_exchange_force_3d_post_stream,
)
from src.lbm_3d_bouzidi import make_sphere_mask, sphere_wall_links  # noqa: E402
from src.lbm_3d_trt import run_channel_smoke_trt  # noqa: E402

# Mirror of the D = 40 Ladd bake configuration. The flow is
# UNCHANGED -- only the force formula at the end differs.
CONFIG = {
    "Nx": 320,
    "Ny": 160,
    "Nz": 160,
    "cx": 60.0,
    "cy": 80.0,
    "cz": 80.0,
    "R": 20.0,
    "u_in": 0.04,
    "nu": 0.016,
    "n_steps": 5000,
    "scheme": "trt",
    "use_guo_neem": True,
    "rho_outflow": 1.0,
    "outflow_scheme": "regularised",
}

CGW_CD_REF = 1.09

# Wide tolerance band same as the Ladd D = 40 case -- this is a
# measurement, not a percent-level claim.
CD_TOLERANCE_PCT = 60.0


def _make_progress_callback(n_steps: int, t_start: float):
    def cb(fraction: float, message: str) -> None:
        elapsed = time.time() - t_start
        if fraction <= 1e-9:
            eta_str, rate_str = "TBD", "TBD"
        else:
            total_est = elapsed / fraction
            remaining = total_est - elapsed
            eta_h, eta_rem = divmod(int(remaining), 3600)
            eta_m, eta_s = divmod(eta_rem, 60)
            eta_str = f"{eta_h:d}h {eta_m:02d}m {eta_s:02d}s"
            steps_done = int(fraction * n_steps)
            rate_str = f"{steps_done / elapsed:.1f} steps/s"
        print(
            f"  [progress] {message}  -- "
            f"{100*fraction:5.1f} %  |  elapsed {elapsed/60:5.1f} min  |  "
            f"ETA {eta_str}  |  {rate_str}",
            flush=True,
        )
    return cb


def main() -> dict:
    cfg = CONFIG
    Nx, Ny, Nz = cfg["Nx"], cfg["Ny"], cfg["Nz"]

    n_cells = Nx * Ny * Nz
    mem_mb = (n_cells * 19 * 2 * 4) / 1024**2

    print(f"Sphere Re = 100 at D = 40 with MYSL momentum exchange")
    print(f"  grid: {Nx} x {Ny} x {Nz} = {n_cells/1e6:.2f} M cells")
    print(f"  memory: ~ {mem_mb:.0f} MB peak (f x 2 buffers, float32)")
    print(f"  R = {cfg['R']}, B = {2*cfg['R']/Ny*100:.0f} %, "
          f"L_x / D = {Nx/(2*cfg['R']):.1f}")
    print(f"  u_in = {cfg['u_in']}, nu = {cfg['nu']}, "
          f"Re = {cfg['u_in']*2*cfg['R']/cfg['nu']:.1f}")
    print(f"  tau = {0.5 + 3*cfg['nu']:.4f}")
    print(f"  n_steps = {cfg['n_steps']} "
          f"({cfg['n_steps']*cfg['u_in']/(2*cfg['R']):.1f} D/U)")
    print(f"  expected wall-time: ~ 2.2 hours on a 4-core CPU "
          f"(same as the Ladd D = 40 bake)")
    print()

    body = make_sphere_mask(Nx, Ny, Nz, cfg["cx"], cfg["cy"], cfg["cz"], cfg["R"])
    wall_links = sphere_wall_links(
        Nx, Ny, Nz, cfg["cx"], cfg["cy"], cfg["cz"], cfg["R"],
    )
    print(f"  body mask + wall links built; "
          f"n_solid = {int(body.sum())}, n_wall_links = {wall_links.n_links}")
    print(f"  wall-link q distribution: "
          f"mean = {float(wall_links.q.mean()):.3f}, "
          f"std = {float(wall_links.q.std()):.3f}, "
          f"frac off 0.5 = "
          f"{100 * (np.abs(wall_links.q - 0.5) > 0.05).mean():.1f} %")
    print()

    t0 = time.time()
    rho, ux, uy, uz, diag, f_final = run_channel_smoke_trt(
        Nx=Nx, Ny=Ny, Nz=Nz,
        u_in=cfg["u_in"], nu=cfg["nu"],
        n_steps=cfg["n_steps"],
        body=body, wall_links=wall_links,
        use_guo_neem=cfg["use_guo_neem"],
        rho_outflow=cfg["rho_outflow"],
        outflow_scheme=cfg["outflow_scheme"],
        scheme=cfg["scheme"],
        return_populations=True,
        progress_callback=_make_progress_callback(cfg["n_steps"], t0),
    )
    t_solve = time.time() - t0

    # Both force evaluations on the same converged state.
    F_ladd = momentum_exchange_force_3d_post_stream(f_final, body)
    F_mysl = momentum_exchange_force_3d_mysl(
        f_final, wall_links, body, cfg["nu"],
    )

    A_proj = np.pi * cfg["R"] * cfg["R"]
    Cd_ladd = drag_coefficient_3d(
        float(F_ladd[0]), rho_ref=1.0, u_ref=cfg["u_in"], A_proj=A_proj,
    )
    Cd_mysl = drag_coefficient_3d(
        float(F_mysl[0]), rho_ref=1.0, u_ref=cfg["u_in"], A_proj=A_proj,
    )

    err_ladd = 100.0 * (Cd_ladd - CGW_CD_REF) / CGW_CD_REF
    err_mysl = 100.0 * (Cd_mysl - CGW_CD_REF) / CGW_CD_REF
    delta_mysl_vs_ladd = 100.0 * (Cd_mysl - Cd_ladd) / Cd_ladd

    blockage_pct = (2.0 * cfg["R"] / Ny) * 100.0

    result = {
        "ref": "Clift-Grace-Weber 1978",
        "Re_target": 100,
        "Re_actual": cfg["u_in"] * 2.0 * cfg["R"] / cfg["nu"],
        "u_in": cfg["u_in"],
        "nu": cfg["nu"],
        "tau": 0.5 + 3.0 * cfg["nu"],
        "grid": [Nx, Ny, Nz],
        "n_cells": n_cells,
        "n_steps": cfg["n_steps"],
        "advective_times": cfg["u_in"] * cfg["n_steps"] / (2.0 * cfg["R"]),
        "blockage_pct": blockage_pct,
        "scheme": cfg["scheme"],
        "outflow_scheme": cfg["outflow_scheme"],
        "use_bouzidi": True,
        # ---- Ladd (baseline, for comparison)
        "ladd": {
            "momentum_exchange": "Ladd 1994 simplified",
            "F_drag_lattice": float(F_ladd[0]),
            "F_lift_lattice": float(F_ladd[1]),
            "F_side_lattice": float(F_ladd[2]),
            "Cd_raw": Cd_ladd,
            "Cd_error_pct": err_ladd,
        },
        # ---- MYSL (this run's headline)
        "mysl": {
            "momentum_exchange": "MYSL 2002 (Mei-Yu-Shyy-Luo, q-aware Bouzidi)",
            "F_drag_lattice": float(F_mysl[0]),
            "F_lift_lattice": float(F_mysl[1]),
            "F_side_lattice": float(F_mysl[2]),
            "Cd_raw": Cd_mysl,
            "Cd_error_pct": err_mysl,
        },
        "delta_mysl_vs_ladd_pct": delta_mysl_vs_ladd,
        "Cd_ref_clift_grace_weber": CGW_CD_REF,
        "Cd_tolerance_pct": CD_TOLERANCE_PCT,
        "u_peak_lattice": diag["u_peak"],
        "mass_drift_rel": diag["mass_drift_rel"],
        "solve_wall_time_s": t_solve,
        "_provenance": {
            "script": "scripts/validate_3d_sphere_cd_mysl_d40.py",
            "purpose": (
                "Second half of audit item #8 from v0.6.5.1: same D = 40 "
                "bake as VALIDATION.md sec 8.3.3, but with the MYSL 2002 "
                "Bouzidi-aware momentum-exchange formula applied at the "
                "end. The flow is unchanged; only the force "
                "post-processing changes. This isolates the momentum-"
                "exchange contribution to the +40 % Cd bias the Ladd "
                "form carries at this resolution."
            ),
        },
    }

    out_dir = _ROOT / "data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "validation_3d_sphere_re100_d40_mysl.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))

    print()
    print(f"Solve: {t_solve:.1f} s ({t_solve/3600:.2f} h)")
    print(f"u_peak (lattice): {diag['u_peak']:.4f}")
    print(f"mass drift: {diag['mass_drift_rel']:+.2%}")
    print()
    print(f"--- Force comparison ---")
    print(f"  Ladd 1994 simplified:")
    print(f"    F_drag = {float(F_ladd[0]):.6f}, "
          f"F_lift = {float(F_ladd[1]):.6f}, "
          f"F_side = {float(F_ladd[2]):.6f}")
    print(f"    Cd = {Cd_ladd:.4f}  ({err_ladd:+.2f} % vs CGW {CGW_CD_REF:.3f})")
    print(f"  MYSL 2002 (q-aware Bouzidi):")
    print(f"    F_drag = {float(F_mysl[0]):.6f}, "
          f"F_lift = {float(F_mysl[1]):.6f}, "
          f"F_side = {float(F_mysl[2]):.6f}")
    print(f"    Cd = {Cd_mysl:.4f}  ({err_mysl:+.2f} % vs CGW {CGW_CD_REF:.3f})")
    print()
    print(f"  Delta (MYSL vs Ladd): {delta_mysl_vs_ladd:+.2f} %")
    print(f"  Bias reduction: "
          f"{(abs(err_ladd) - abs(err_mysl)):+.1f} percentage points")
    print(f"-> {out_path}")
    return result


if __name__ == "__main__":
    main()
