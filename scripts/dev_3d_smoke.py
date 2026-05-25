"""3D D3Q19 channel-flow smoke driver (CLI).

Runs ``src.lbm_3d.run_channel_smoke`` at a small grid and prints
diagnostics. Used for local iteration while the 3D solver is still
in development -- the Streamlit "3D (local, in development)" tab is
the visual equivalent of this script.

The 2D solver in ``src.lbm`` is the validated production path; this
3D bench is local-only because D3Q19 populations push past Streamlit
Cloud's 1 GB memory cap at any non-trivial grid. See section
"Senior-engineer heads-up before 3D" in the conversation log for
the memory math.

Usage:
    python scripts/dev_3d_smoke.py
    python scripts/dev_3d_smoke.py --nx 96 --ny 32 --nz 32 --n-steps 800

Expected first run: ~15 s JIT compile + ~2-5 s of streaming for
the default 64 x 24 x 24 / 400-step case.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ_ROOT))

from src.lbm_3d import run_channel_smoke  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(
        description="3D LBM channel-flow smoke (D3Q19, BGK).",
    )
    p.add_argument("--nx", type=int, default=64, help="streamwise cells")
    p.add_argument("--ny", type=int, default=24, help="wall-normal cells")
    p.add_argument("--nz", type=int, default=24, help="spanwise cells")
    p.add_argument("--u-in", type=float, default=0.04,
                   help="inflow velocity (lattice units)")
    p.add_argument("--nu", type=float, default=0.02,
                   help="kinematic viscosity (lattice units)")
    p.add_argument("--n-steps", type=int, default=400)
    args = p.parse_args()

    print(f"[3d-smoke] grid={args.nx}x{args.ny}x{args.nz}, "
          f"u_in={args.u_in}, nu={args.nu}, n_steps={args.n_steps}")
    print(f"[3d-smoke] Re_channel ~ {args.u_in*args.ny/args.nu:.1f}")

    t0 = time.time()
    rho, ux, _, _, diag = run_channel_smoke(
        Nx=args.nx, Ny=args.ny, Nz=args.nz,
        u_in=args.u_in, nu=args.nu, n_steps=args.n_steps,
    )
    elapsed = time.time() - t0

    print()
    print("=" * 64)
    print(f"  wall time         {elapsed:.2f} s "
          f"({args.n_steps/elapsed:.1f} steps/s)")
    print(f"  u_peak            {diag['u_peak']:.5f}")
    print(f"  u_mean            {diag['u_mean']:.5f}")
    print(f"  centerline ratio  {diag['centerline_ratio']:.4f}   "
          f"(1.5 = fully-developed plane Poiseuille)")
    print(f"  mass drift (rel)  {diag['mass_drift_rel']*100:+.4f} %")
    print(f"  rho   min/max     {float(rho.min()):.6f} / {float(rho.max()):.6f}")
    print(f"  ux    min/max     {float(ux.min()):.6f} / {float(ux.max()):.6f}")
    print("=" * 64)

    # Wall-normal centerline profile at midchannel x, mid z. The shape
    # should be a clean symmetric parabola.
    mid_x = args.nx // 2
    mid_z = args.nz // 2
    profile = ux[mid_x, :, mid_z]
    print("\n  ux(y) at midchannel x, mid z:")
    for y, v in enumerate(profile):
        bar = "*" * int(60 * float(v) / max(float(profile.max()), 1e-12))
        print(f"    y={y:2d}  ux={float(v):+.5f}  |{bar}")

    # Quick verdict.
    print()
    if abs(diag["mass_drift_rel"]) > 0.05:
        print("[3d-smoke] FAIL: mass drift > 5 %. Streaming or boundary "
              "code is leaking mass.")
        return 2
    if diag["u_peak"] < 0.5 * args.u_in:
        print("[3d-smoke] FAIL: peak u far below inflow. Inflow not "
              "propagating?")
        return 2
    if abs(profile - profile[::-1]).max() > 1e-3 * float(profile.max()):
        print("[3d-smoke] WARN: profile not symmetric in y. Wall "
              "bounce-back may be biased.")
    print("[3d-smoke] OK: smoke clean. 3D BGK kernel + bounce-back + "
          "inflow / outflow handling are internally consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
