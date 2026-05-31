"""Sphere Re = 100 Cd validation at D = 40 (Mei-Luo-Shyy 1999 grade).

Companion to ``scripts/validate_3d_sphere_cd_lowblock.py`` and the
**first half** of item #8 from the v0.6.5.1 senior re-audit:

    > MYSL Bouzidi-aware momentum exchange + D >= 40 sphere bake
    > (turns the +44 % into a percent-level claim).

The audit item bundles two refinements -- grid (D >= 40 per Mei-Luo-
Shyy 1999) and momentum exchange (MYSL 2002 q-weighted). This script
runs the grid half ONLY, keeping the same simplified Ladd 1994
momentum-exchange formula as the shipped Re = 100 / D = 20 baseline.

The point is to **isolate**: if Cd drops substantially at D = 40
(toward CGW 1.09), grid resolution is the dominant residual bias.
If Cd stays near 1.5 - 1.6, momentum exchange is the dominant bias
and the MYSL upgrade becomes the critical next step. Either outcome
updates the VALIDATION.md sec 8.3.1 error budget with a measured
data point instead of a budgeted one.

Reference: Clift-Grace-Weber 1978, Cd = 1.09 at Re = 100.

Grid + physics
--------------
Domain matches the lowblock physical extent (8 D x 4 D x 4 D) so the
only changed dimensionless number is grid spacing per body diameter:

    lowblock (D = 20):  Nx = 160, Ny = Nz =  80, B = 25 %
    this run (D = 40):  Nx = 320, Ny = Nz = 160, B = 25 %  <-- 8 x cells

    u_in unchanged at 0.04 (keeps Mach ~ 0.07, well incompressible).
    nu   scaled to u_in * D / Re_target = 0.04 * 40 / 100 = 0.016.
    tau  = 0.5 + 3 * nu = 0.548 -- safer than lowblock's 0.524.

    n_steps = 5000 (5 D/U advective settle = 5 * D / u_in = 5 * 40 / 0.04).

Runtime
-------
160 cells x 80 cells x 80 cells x 2500 steps = 0.8 G cell-steps; the
lowblock run takes ~ 20 min on a 4-core laptop, so the D=40 case at
12.8 G cell-steps is ~ **5 - 6 hours**. Run overnight, ideally in
a tmux-like session to survive lid-close / WSL hibernation.

Memory
------
320 x 160 x 160 = 8.2 M lattice cells, 19 directions, 2 buffers,
float32: 1.24 GB peak resident. Comfortably below a 16 GB laptop's
working set; not feasible on the 1-vCPU Cloud worker.

Outputs
-------
data/validation_3d_sphere_re100_d40.json
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
    momentum_exchange_force_3d_post_stream,
)
from src.lbm_3d_bouzidi import make_sphere_mask, sphere_wall_links  # noqa: E402
from src.lbm_3d_trt import run_channel_smoke_trt  # noqa: E402

CONFIG = {
    "Nx": 320,                        # 8 D upstream + downstream, matches lowblock physical extent
    "Ny": 160,                        # B = 40 / 160 = 25 %
    "Nz": 160,                        # B = 40 / 160 = 25 %
    "cx": 60.0,                       # 1.5 D upstream margin
    "cy": 80.0,                       # mid-channel
    "cz": 80.0,                       # mid-channel
    "R": 20.0,                        # D = 40
    "u_in": 0.04,                     # Mach ~ 0.07
    "nu": 0.016,                      # Re = 0.04 * 40 / 0.016 = 100.0  exact
    "n_steps": 5000,                  # 5 D/U advective settle (matches lowblock)
    "scheme": "trt",
    "use_guo_neem": True,
    "rho_outflow": 1.0,
    "outflow_scheme": "regularised",
}

CGW_CD_REF = 1.09

# Tolerance band: we expect this run to LAND somewhere; the question
# is where. The lowblock at D=20 gave Cd = 1.65 (+51 %). If D=40 lands
# near 1.1 - 1.3, grid resolution was the dominant bias; if it lands
# near 1.4 - 1.6, momentum exchange is the dominant bias. The band
# below gates on "the measurement is in the physically-plausible
# bluff-body Cd range," not on the budgeted target.
CD_TOLERANCE_PCT = 80.0


def main() -> dict:
    cfg = CONFIG
    Nx, Ny, Nz = cfg["Nx"], cfg["Ny"], cfg["Nz"]

    n_cells = Nx * Ny * Nz
    mem_mb = (n_cells * 19 * 2 * 4) / 1024**2
    n_advective = (cfg["u_in"] * cfg["n_steps"]) / (2.0 * cfg["R"])

    print(f"Sphere Re = 100 at D = 40 (Mei-Luo-Shyy 1999 grade)")
    print(f"  grid: {Nx} x {Ny} x {Nz} = {n_cells/1e6:.2f} M cells")
    print(f"  memory: ~ {mem_mb:.0f} MB peak (f x 2 buffers, float32)")
    print(f"  R = {cfg['R']}, B = {2*cfg['R']/Ny*100:.0f} %, "
          f"L_x / D = {Nx/(2*cfg['R']):.1f}")
    print(f"  u_in = {cfg['u_in']}, nu = {cfg['nu']}, "
          f"Re = {cfg['u_in']*2*cfg['R']/cfg['nu']:.1f}")
    print(f"  tau = {0.5 + 3*cfg['nu']:.4f} (lowblock baseline: 0.524)")
    print(f"  n_steps = {cfg['n_steps']} ({n_advective:.1f} D/U)")
    print(f"  expected wall-time: ~ 5 - 6 hours on a 4-core CPU")
    print()

    body = make_sphere_mask(Nx, Ny, Nz, cfg["cx"], cfg["cy"], cfg["cz"], cfg["R"])
    wall_links = sphere_wall_links(
        Nx, Ny, Nz, cfg["cx"], cfg["cy"], cfg["cz"], cfg["R"],
    )
    print(f"  body mask + wall links built; "
          f"n_solid = {int(body.sum())}, n_wall_links = {wall_links.n_links}")
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
    )
    t_solve = time.time() - t0

    F = momentum_exchange_force_3d_post_stream(f_final, body)
    F_drag = float(F[0])
    F_lift = float(F[1])
    F_side = float(F[2])

    A_proj = np.pi * cfg["R"] * cfg["R"]
    Cd_raw = drag_coefficient_3d(
        F_drag,
        rho_ref=1.0,
        u_ref=cfg["u_in"],
        A_proj=A_proj,
    )

    blockage_pct = (2.0 * cfg["R"] / Ny) * 100.0
    err_pct = 100.0 * (Cd_raw - CGW_CD_REF) / CGW_CD_REF

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
        "advective_times": n_advective,
        "blockage_pct": blockage_pct,
        "scheme": cfg["scheme"],
        "outflow_scheme": cfg["outflow_scheme"],
        "use_bouzidi": True,
        "momentum_exchange": "Ladd 1994 simplified",
        "F_drag_lattice": F_drag,
        "F_lift_lattice": F_lift,
        "F_side_lattice": F_side,
        "A_proj_lattice": float(A_proj),
        "Cd_raw": Cd_raw,
        "Cd_ref_clift_grace_weber": CGW_CD_REF,
        "Cd_error_pct": err_pct,
        "Cd_tolerance_pct": CD_TOLERANCE_PCT,
        "Cd_in_band": abs(err_pct) <= CD_TOLERANCE_PCT,
        "u_peak_lattice": diag["u_peak"],
        "mass_drift_rel": diag["mass_drift_rel"],
        "solve_wall_time_s": t_solve,
        "_provenance": {
            "script": "scripts/validate_3d_sphere_cd_d40.py",
            "purpose": (
                "First half of audit item #8: D = 40 sphere bake. Isolates "
                "the grid-resolution contribution to the +44 % Cd bias on "
                "the shipped D = 20 baseline by holding everything else "
                "constant (same B = 25 %, same simplified Ladd 1994 "
                "momentum exchange, same TRT collision, same Guo NEEM "
                "regularised outflow)."
            ),
            "diagnosis": (
                "Cd around 1.1 - 1.3 -> grid resolution was the dominant "
                "bias (Mei-Luo-Shyy 1999 D >= 40 guideline binds). "
                "Cd around 1.4 - 1.6 -> momentum-exchange formula is "
                "the dominant bias; MYSL 2002 upgrade is the critical "
                "next step (second half of audit item #8)."
            ),
        },
    }

    out_dir = _ROOT / "data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "validation_3d_sphere_re100_d40.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))

    print(f"Solve: {t_solve:.1f} s ({t_solve/3600:.2f} h)")
    print(f"F_drag (lattice): {F_drag:.6f}")
    print(f"F_lift (lattice): {F_lift:.6f} (expected ~ 0 by axisymmetry)")
    print(f"F_side (lattice): {F_side:.6f} (expected ~ 0 by axisymmetry)")
    print(f"u_peak (lattice): {diag['u_peak']:.4f}")
    print(f"mass drift: {diag['mass_drift_rel']:+.2%}")
    print(f"Cd (raw): {Cd_raw:.3f}")
    print(f"Cd (CGW 1978 at Re=100): {CGW_CD_REF:.3f}")
    print(f"error: {err_pct:+.1f} %")
    print()
    print(f"Comparison to D = 20 lowblock baseline (Cd = 1.645, +51 %):")
    print(f"  D = 40 measurement: Cd = {Cd_raw:.3f}, error = {err_pct:+.1f} %")
    print(f"  D-resolution effect: {Cd_raw - 1.645:+.3f} ({(Cd_raw - 1.645)/1.645*100:+.1f} %)")
    print(f"-> {out_path}")
    return result


if __name__ == "__main__":
    main()
