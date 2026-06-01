"""Sphere Re = 20 Cd validation at D = 40 with MYSL momentum exchange.

Method-consistent companion to §8.3.4 (Re = 100 + MYSL + D = 40).
Closes audit Task 7 from the 2026-06-01 forensic re-audit: bring the
Re = 20 data point onto the same methodology as the Re = 100 headline
(MYSL 2002 q-aware Bouzidi + D = 40) so the two-Re-point
"bias-vs-Re-trend" claim rests on apples-to-apples measurements.

The earlier `validate_3d_sphere_cd_stokes_regime.py` ran at D = 20 +
Ladd 1994 simplified (Cd = 4.27, +56.2 %). This script re-runs the
SAME Re = 20 physics on the D = 40 grid + MYSL force formula -- same
grid as §8.3.4, with nu / U scaled to land at Re = 20 instead of 100.

Reference: Clift-Grace-Weber 1978 standard sphere drag curve at
Re = 20:

    Cd_CGW = (24/Re) * (1 + 0.1806 * Re^0.6459)  +  0.4251 / (1 + 6880.95/Re)
           = 2.728

Outputs
-------
data/validation_3d_sphere_re20_mysl_d40.json
    -- includes BOTH Ladd and MYSL Cd from the same converged flow so
       the side-by-side reduction is committed (same pattern as §8.3.4).
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

# D = 40 grid, exactly matching §8.3.4 / sec 8.3.3.
# Re = 20 via nu = u_in * D / 20 = 0.04 * 40 / 20 = 0.08.
# tau = 0.5 + 3 nu = 0.74 -- well within the LBM stability band.
#
# At Re = 20 the flow is steady symmetric (no shedding). 5 D/U
# advective settle (5000 steps with these constants) is more than
# enough; same step count as the Re = 100 case for direct comparison.
CONFIG = {
    "Nx": 320,
    "Ny": 160,
    "Nz": 160,
    "cx": 60.0,
    "cy": 80.0,
    "cz": 80.0,
    "R": 20.0,
    "u_in": 0.04,
    "nu": 0.08,                     # Re = u_in * D / nu = 0.04 * 40 / 0.08 = 20.0
    "n_steps": 5000,
    "scheme": "trt",
    "use_guo_neem": True,
    "rho_outflow": 1.0,
    "outflow_scheme": "regularised",
}

CGW_CD_REF = 2.728

# Wider band than Re = 100 / MYSL (10 %) -- this is the first MYSL
# measurement at low Re; tightening waits until the result lands.
CD_TOLERANCE_PCT = 30.0


def _make_progress_callback(n_steps: int, t_start: float):
    def cb(fraction: float, message: str) -> None:
        elapsed = time.time() - t_start
        if fraction <= 1e-9:
            eta_str, rate_str = "TBD", "TBD"
        else:
            remaining = elapsed / fraction - elapsed
            eta_h, eta_rem = divmod(int(remaining), 3600)
            eta_m, eta_s = divmod(eta_rem, 60)
            eta_str = f"{eta_h:d}h {eta_m:02d}m {eta_s:02d}s"
            rate_str = f"{int(fraction * n_steps) / elapsed:.1f} steps/s"
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

    print(f"Sphere Re = 20 at D = 40 with MYSL momentum exchange")
    print(f"  grid: {Nx} x {Ny} x {Nz} = {Nx*Ny*Nz/1e6:.2f} M cells")
    print(f"  R = {cfg['R']}, B = {2*cfg['R']/Ny*100:.0f} %")
    print(f"  u_in = {cfg['u_in']}, nu = {cfg['nu']}, "
          f"Re = {cfg['u_in']*2*cfg['R']/cfg['nu']:.1f}")
    print(f"  tau = {0.5 + 3*cfg['nu']:.4f}")
    print(f"  n_steps = {cfg['n_steps']} "
          f"({cfg['n_steps']*cfg['u_in']/(2*cfg['R']):.1f} D/U)")
    print(f"  expected wall-time: ~ 2.2 hours on a 4-core CPU "
          f"(same compute as the Re = 100 MYSL D = 40 bake)")
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
        progress_callback=_make_progress_callback(cfg["n_steps"], t0),
    )
    t_solve = time.time() - t0

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

    result = {
        "ref": "Clift-Grace-Weber 1978",
        "Re_target": 20,
        "Re_actual": cfg["u_in"] * 2.0 * cfg["R"] / cfg["nu"],
        "u_in": cfg["u_in"],
        "nu": cfg["nu"],
        "tau": 0.5 + 3.0 * cfg["nu"],
        "grid": [Nx, Ny, Nz],
        "n_steps": cfg["n_steps"],
        "advective_times": cfg["u_in"] * cfg["n_steps"] / (2.0 * cfg["R"]),
        "blockage_pct": (2.0 * cfg["R"] / Ny) * 100.0,
        "scheme": cfg["scheme"],
        "outflow_scheme": cfg["outflow_scheme"],
        "use_bouzidi": True,
        "ladd": {
            "momentum_exchange": "Ladd 1994 simplified",
            "F_drag_lattice": float(F_ladd[0]),
            "F_lift_lattice": float(F_ladd[1]),
            "F_side_lattice": float(F_ladd[2]),
            "Cd_raw": Cd_ladd,
            "Cd_error_pct": err_ladd,
        },
        "mysl": {
            "momentum_exchange": "MYSL 2002 (Mei-Yu-Shyy-Luo, q-aware Bouzidi)",
            "F_drag_lattice": float(F_mysl[0]),
            "F_lift_lattice": float(F_mysl[1]),
            "F_side_lattice": float(F_mysl[2]),
            "Cd_raw": Cd_mysl,
            "Cd_error_pct": err_mysl,
        },
        "delta_mysl_vs_ladd_pct": 100.0 * (Cd_mysl - Cd_ladd) / Cd_ladd,
        "Cd_ref_clift_grace_weber": CGW_CD_REF,
        "Cd_tolerance_pct": CD_TOLERANCE_PCT,
        "u_peak_lattice": diag["u_peak"],
        "mass_drift_rel": diag["mass_drift_rel"],
        "solve_wall_time_s": t_solve,
        "_provenance": {
            "script": "scripts/validate_3d_sphere_cd_stokes_regime_mysl_d40.py",
            "purpose": (
                "Audit Task 7 (forensic re-audit 2026-06-01): bring the "
                "Re = 20 data point onto the same MYSL + D = 40 method "
                "as the Re = 100 headline (sec 8.3.4) so the two-point "
                "trend rests on apples-to-apples measurements."
            ),
        },
    }

    out_dir = _ROOT / "data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "validation_3d_sphere_re20_mysl_d40.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))

    print()
    print(f"Solve: {t_solve:.1f} s ({t_solve/3600:.2f} h)")
    print(f"u_peak: {diag['u_peak']:.4f}, mass drift: {diag['mass_drift_rel']:+.2%}")
    print()
    print(f"--- Force comparison at Re = 20, D = 40 ---")
    print(f"  Ladd 1994 simplified:")
    print(f"    Cd = {Cd_ladd:.4f}  ({err_ladd:+.2f} % vs CGW {CGW_CD_REF:.3f})")
    print(f"  MYSL 2002 (q-aware Bouzidi):")
    print(f"    Cd = {Cd_mysl:.4f}  ({err_mysl:+.2f} % vs CGW {CGW_CD_REF:.3f})")
    print()
    print(f"  Bias reduction: {abs(err_ladd) - abs(err_mysl):+.1f} percentage points")
    print(f"-> {out_path}")
    return result


if __name__ == "__main__":
    main()
