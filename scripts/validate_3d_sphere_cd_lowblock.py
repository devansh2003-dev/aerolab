"""Sphere Re=100 Cd validation at a lower-blockage grid.

Companion to ``scripts/validate_3d_sphere_cd.py``. Same physics
(D3Q19 TRT, Bouzidi interpolated bounce-back, regularised outflow)
and same reference (Clift-Grace-Weber 1978 at Re = 100), but on a
160 x 80 x 80 grid with R = 10 (D = 20). Blockage drops from 42 %
(the shipped sphere_re100 bake) to 25 %, isolating the
blockage-induced bias from the grid and momentum-exchange biases.

Writes ``data/validation_3d_sphere_re100_lowblock.json`` next to the
high-blockage result. The companion test
``tests/test_validation_3d_sphere_cd_lowblock.py`` reads that JSON
and gates the value.

Runtime: ~20 min on a laptop CPU (1.024 M cells x 2500 steps).

Why two data points instead of one
----------------------------------
The high-blockage measurement reads Cd = 1.57, +44 % above the
CGW reference. The +44 % budget breaks down (per VALIDATION.md
§8.3) into roughly:

- ~+25 % from blockage at B = 42 %
- ~+10 % from halfway bounce-back momentum exchange (Mei-Yu-Shyy-Luo
  2002 Bouzidi-aware variant would shrink this)
- ~+5 % from D = 20 cell resolution (Mei-Luo-Shyy 1999 recommend D >= 40)
- ~+5 % residual / nonlinear combinations

Re-running at B = 25 % (this script) should drop the blockage term
from ~+25 % to ~+10 %, predicting Cd ~ 1.30 - 1.40. If the
measurement lands there, the budget breakdown is supported. If it
lands further off, the budget needs revisiting.
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

# Low-blockage variant of the sphere_re100 setup:
#   Nx 96 -> 160:  more downstream wake budget (8 D vs 4.6 D)
#   Ny 48 -> 80:   blockage 42 % -> 25 %
#   Nz 48 -> 80:   blockage 42 % -> 25 %
#   R    unchanged at 10 (keeps D = 20)
#   u_in unchanged at 0.04 (keeps tau = 0.524 well off the stability edge)
#   nu   unchanged at 0.008 (keeps Re = u_in * D / nu = 100)
#   n_steps = 2500 (same 5 D/u advective settle as the high-blockage run)
CONFIG = {
    "Nx": 160,
    "Ny": 80,
    "Nz": 80,
    "cx": 30.0,                       # 1.5 D upstream margin
    "cy": 40.0,                       # mid-channel
    "cz": 40.0,                       # mid-channel
    "R": 10.0,
    "u_in": 0.04,
    "nu": 0.008,                      # Re = u_in * D / nu = 0.04 * 20 / 0.008 = 100
    "n_steps": 2500,
    "scheme": "trt",
    "use_guo_neem": True,
    "rho_outflow": 1.0,
    "outflow_scheme": "regularised",
}

CGW_CD_REF = 1.09
# Tolerance band keeps the same 0.7-absolute slack as the
# high-blockage run -- we are not claiming a tighter measurement
# here, just demonstrating that blockage was a (large) chunk of
# the +44 % error. A genuinely tight Cd claim still needs D >= 40
# and the Mei-Yu-Shyy-Luo 2002 momentum-exchange refinement.
CD_TOLERANCE = 0.7


def main() -> dict:
    cfg = CONFIG
    Nx, Ny, Nz = cfg["Nx"], cfg["Ny"], cfg["Nz"]

    body = make_sphere_mask(Nx, Ny, Nz, cfg["cx"], cfg["cy"], cfg["cz"], cfg["R"])
    wall_links = sphere_wall_links(
        Nx, Ny, Nz, cfg["cx"], cfg["cy"], cfg["cz"], cfg["R"],
    )

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

    advective_times = (cfg["u_in"] * cfg["n_steps"]) / (2.0 * cfg["R"])
    blockage_pct = (2.0 * cfg["R"] / Ny) * 100.0

    result = {
        "ref": "Clift-Grace-Weber 1978",
        "Re_target": 100,
        "Re_actual": cfg["u_in"] * 2.0 * cfg["R"] / cfg["nu"],
        "u_in": cfg["u_in"],
        "nu": cfg["nu"],
        "grid": [Nx, Ny, Nz],
        "n_steps": cfg["n_steps"],
        "advective_times": advective_times,
        "blockage_pct": blockage_pct,
        "scheme": cfg["scheme"],
        "outflow_scheme": cfg["outflow_scheme"],
        "use_bouzidi": True,
        "F_drag_lattice": F_drag,
        "F_lift_lattice": F_lift,
        "F_side_lattice": F_side,
        "A_proj_lattice": float(A_proj),
        "Cd_raw": Cd_raw,
        "Cd_ref_clift_grace_weber": CGW_CD_REF,
        "Cd_error_pct": 100.0 * (Cd_raw - CGW_CD_REF) / CGW_CD_REF,
        "Cd_tolerance_band": CD_TOLERANCE,
        "Cd_in_band": abs(Cd_raw - CGW_CD_REF) <= CD_TOLERANCE,
        "u_peak_lattice": diag["u_peak"],
        "mass_drift_rel": diag["mass_drift_rel"],
        "solve_wall_time_s": t_solve,
    }

    out_dir = _ROOT / "data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "validation_3d_sphere_re100_lowblock.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))

    print(f"Solve: {t_solve:.1f} s")
    print(f"Grid: {Nx} x {Ny} x {Nz}, R = {cfg['R']}, blockage = {blockage_pct:.0f} %")
    print(f"Advective times: {advective_times:.1f} D/u  ({cfg['n_steps']} steps)")
    print(f"F_drag (lattice): {F_drag:.6f}")
    print(f"F_lift (lattice): {F_lift:.6f} (expected ~ 0 by axisymmetry)")
    print(f"F_side (lattice): {F_side:.6f} (expected ~ 0 by axisymmetry)")
    print(f"u_peak (lattice): {diag['u_peak']:.4f}")
    print(f"mass drift: {diag['mass_drift_rel']:+.2%}")
    print(f"Cd (raw):  {Cd_raw:.3f}")
    print(f"Cd (Clift-Grace-Weber Re=100): {CGW_CD_REF:.3f}")
    print(f"error: {result['Cd_error_pct']:+.1f} %")
    print(
        f"in {CD_TOLERANCE:.2f} band: "
        f"{'PASS' if result['Cd_in_band'] else 'FAIL'}"
    )
    print(f"-> {out_path}")
    return result


if __name__ == "__main__":
    main()
