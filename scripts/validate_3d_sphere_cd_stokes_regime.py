"""Steady-wake sphere validation at Re = 20 (Stokes-regime companion).

Closes item #9 from the v0.6.5.1 senior re-audit:

    > A second 3D validation (3D lid-driven cavity or a Stokes-regime
    > sphere) so 3D rests on more than TGV + one bluff body.

The shipped 3D drag validation (`validate_3d_sphere_cd_lowblock.py`)
runs sphere Re = 100 -- a regime where the wake is steady but where
the simplified Ladd 1994 momentum exchange + D = 20 grid biases the
measurement to Cd = 1.57 vs CGW 1.09 (+44 %). One Re-point is not
enough to characterise the bias: at Re = 100 the drag is roughly
half-pressure / half-viscous, so the +44 % could be coming from
either bucket.

This script runs the SAME body and grid at **Re = 20** -- well below
the Roos-Willmarth 1971 wake-asymmetry threshold (Re ~ 210) and the
shedding threshold (Re ~ 270 for spheres), so the flow is steady,
symmetric, and entirely viscous-dominated. Pressure drag contributes
~ 30 % of the total at Re = 20 versus ~ 50 % at Re = 100, so the
viscous-vs-pressure error split is testable.

Reference: Clift-Grace-Weber 1978 standard sphere drag curve at
Re = 20:

    Cd_CGW = (24/Re) * (1 + 0.1806 * Re^0.6459)  +  0.4251 / (1 + 6880.95/Re)
           = 1.2 * (1 + 0.1806 * 20^0.6459)      +  0.4251 / (1 + 344.05)
           = 1.2 * 2.272                         +  0.00123
           = 2.728

so we report against Cd_ref = 2.73.

Outputs:
    data/validation_3d_sphere_re20_stokes_regime.json

Runtime: ~ 15 - 25 min on a 4-core laptop (1.024 M cells x 2500
steps, exactly the same compute budget as the Re = 100 lowblock
companion -- nu was scaled up to 0.04 so tau = 0.62 keeps the
stability margin and u_in stays at 0.04, making one D/U the same
500 steps as the lowblock run).
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

# Mirror of the Re = 100 low-blockage grid (160 x 80 x 80, R = 10
# → D = 20, B = 25 %). The only physics changes are nu and Re.
#
#   Re_target = 20
#   To keep tau in its stability band AND keep the same wall-time
#   budget as the Re = 100 lowblock run, we scale nu up by 5x and
#   keep u_in at 0.04:
#     u_in = 0.04, nu = 0.04 -> tau = 0.5 + 3 nu = 0.62 (safe)
#     Re = u_in * D / nu = 0.04 * 20 / 0.04 = 20.0  exact
#   At Re = 20 the wake is short and symmetric; 5 D/U advective
#   settle (2500 steps with these constants) is more than enough.
CONFIG = {
    "Nx": 160,
    "Ny": 80,
    "Nz": 80,
    "cx": 30.0,
    "cy": 40.0,
    "cz": 40.0,
    "R": 10.0,
    "u_in": 0.04,
    "nu": 0.04,                       # 5 x the Re=100 lowblock value
    "n_steps": 2500,
    "scheme": "trt",
    "use_guo_neem": True,
    "rho_outflow": 1.0,
    "outflow_scheme": "regularised",
}

# CGW 1978 standard drag curve at Re = 20.
CGW_CD_REF = 2.73

# We expect the same Ladd-1994-on-D=20 bias the Re=100 case carries
# (~ +30 - 50 %); the band below is wide so the test passes /
# fails on "is the bias the same sign and order of magnitude," not on
# the exact %. Tightening this band waits for the MYSL momentum
# exchange + D >= 40 grid (audit item #8).
CD_TOLERANCE_PCT = 60.0


def main() -> dict:
    cfg = CONFIG
    Nx, Ny, Nz = cfg["Nx"], cfg["Ny"], cfg["Nz"]

    body = make_sphere_mask(Nx, Ny, Nz, cfg["cx"], cfg["cy"], cfg["cz"], cfg["R"])
    wall_links = sphere_wall_links(
        Nx, Ny, Nz, cfg["cx"], cfg["cy"], cfg["cz"], cfg["R"],
    )

    print(f"Sphere Re = 20 (steady-wake / Stokes-regime companion)")
    print(f"  grid: {Nx} x {Ny} x {Nz}, R = {cfg['R']}, B = {2*cfg['R']/Ny*100:.0f} %")
    print(f"  u_in = {cfg['u_in']}, nu = {cfg['nu']}, "
          f"Re = {cfg['u_in']*2*cfg['R']/cfg['nu']:.1f}")
    print(f"  n_steps = {cfg['n_steps']} "
          f"({cfg['n_steps']*cfg['u_in']/(2*cfg['R']):.1f} D/U)")
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

    advective_times = (cfg["u_in"] * cfg["n_steps"]) / (2.0 * cfg["R"])
    blockage_pct = (2.0 * cfg["R"] / Ny) * 100.0
    err_pct = 100.0 * (Cd_raw - CGW_CD_REF) / CGW_CD_REF

    result = {
        "ref": "Clift-Grace-Weber 1978",
        "Re_target": 20,
        "Re_actual": cfg["u_in"] * 2.0 * cfg["R"] / cfg["nu"],
        "u_in": cfg["u_in"],
        "nu": cfg["nu"],
        "tau": 0.5 + 3.0 * cfg["nu"],
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
        "Cd_error_pct": err_pct,
        "Cd_tolerance_pct": CD_TOLERANCE_PCT,
        "Cd_in_band": abs(err_pct) <= CD_TOLERANCE_PCT,
        "u_peak_lattice": diag["u_peak"],
        "mass_drift_rel": diag["mass_drift_rel"],
        "solve_wall_time_s": t_solve,
        "_provenance": {
            "script": "scripts/validate_3d_sphere_cd_stokes_regime.py",
            "purpose": (
                "Second 3D validation data point at a regime that the "
                "Re=100 case cannot probe: Re=20 is steady, symmetric, "
                "and viscous-dominated -- 30 % pressure / 70 % viscous "
                "drag versus the ~ 50 / 50 split at Re=100. A second "
                "regime tests whether the Ladd 1994 + D=20 bias is "
                "constant across Re or Re-dependent."
            ),
        },
    }

    out_dir = _ROOT / "data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "validation_3d_sphere_re20_stokes_regime.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))

    print(f"Solve: {t_solve:.1f} s")
    print(f"F_drag (lattice): {F_drag:.6f}")
    print(f"F_lift (lattice): {F_lift:.6f} (expected ~ 0 by axisymmetry)")
    print(f"F_side (lattice): {F_side:.6f} (expected ~ 0 by axisymmetry)")
    print(f"u_peak (lattice): {diag['u_peak']:.4f}")
    print(f"mass drift: {diag['mass_drift_rel']:+.2%}")
    print(f"Cd (raw):  {Cd_raw:.3f}")
    print(f"Cd (CGW 1978 at Re=20): {CGW_CD_REF:.3f}")
    print(f"error: {err_pct:+.1f} %  "
          f"({'IN' if result['Cd_in_band'] else 'OUT OF'} +/-{CD_TOLERANCE_PCT:.0f} % band)")
    print(f"-> {out_path}")
    return result


if __name__ == "__main__":
    main()
