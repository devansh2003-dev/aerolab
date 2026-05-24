"""Phase 1 Week 2 shape gallery: same flow, different shapes.

Runs five presets through the LBM pipeline -- cylinder, square, ellipse,
NACA 0012 at AoA=5deg, NACA 4412 at AoA=5deg -- and saves a vorticity wake
plot per shape plus a combined gallery figure. Precursor to the Streamlit
preset dropdown.

Common flow conditions (U=0.1, tau=0.56, nu=0.02) for all shapes. Characteristic
length varies per body, so this is a *visual* gallery, not a strict Re-matched
comparison. Wake geometry reveals shape-dependent flow separation:
  - Bluff bodies (cylinder, square): wide periodic von Karman wake
  - Streamlined ellipse: narrower wake, milder shedding
  - Airfoils: attached flow on the suction side, narrow wake from the TE

Run from project root:
    python scripts/shape_gallery.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from src.lbm import CS2, equilibrium, macroscopic, step_njit_with_force
from src.shapes import (
    cylinder_mask,
    ellipse_mask,
    naca4_airfoil_mask,
    no_bouzidi_q_field,
    square_mask,
)

# --- Common flow setup ---
Nx, Ny = 500, 100
U_inflow = 0.1
tau = 0.56
nu = (tau - 0.5) * CS2          # ~ 0.02
n_steps = 8_000
body_x = 80                     # LE / center of every body
cy_center = Ny // 2             # 50

KICK_START = 100
KICK_END = 500
KICK_AMPLITUDE = 0.005
KICK_Y_OFFSET = 2

# Pre-compute inflow equilibrium (constant in time)
f_inflow_eq = equilibrium(1.0, np.array([U_inflow, 0.0]))


# --- Preset factories: each returns (label, mask, kick_x) ---
# kick_x is placed just downstream of the body so the kick reliably perturbs
# the developing wake.

def _cylinder_preset():
    return ("Cylinder  D=20",
            cylinder_mask(Nx, Ny, cx=body_x, cy=cy_center, radius=10),
            body_x + 20)


def _square_preset():
    return ("Square  side=20",
            square_mask(Nx, Ny, cx=body_x, cy=cy_center, side=20),
            body_x + 20)


def _ellipse_preset():
    return ("Ellipse  a=20  b=10",
            ellipse_mask(Nx, Ny, cx=body_x, cy=cy_center, a=20, b=10),
            body_x + 30)


def _naca0012_preset():
    return ("NACA 0012  chord=40  AoA=5 deg",
            naca4_airfoil_mask(Nx, Ny, cx=body_x, cy=cy_center,
                               chord=40, naca_code="0012", aoa_deg=5),
            body_x + 50)


def _naca4412_preset():
    return ("NACA 4412  chord=40  AoA=5 deg",
            naca4_airfoil_mask(Nx, Ny, cx=body_x, cy=cy_center,
                               chord=40, naca_code="4412", aoa_deg=5),
            body_x + 50)


PRESETS = [
    _cylinder_preset,
    _square_preset,
    _ellipse_preset,
    _naca0012_preset,
    _naca4412_preset,
]


def run_one(label, solid_mask, kick_x):
    """Time-march one preset. Returns (label, mask, u_mag_plot, vorticity_plot)."""
    kick_y = cy_center + KICK_Y_OFFSET
    # Halfway BB across all 5 presets to keep this gallery comparison apples-to-apples.
    # Production single-shape runs (Streamlit app) use per-shape Bouzidi where available.
    q_field = no_bouzidi_q_field(Nx, Ny)

    # Initial: uniform inflow everywhere
    rho0 = np.ones((Nx, Ny))
    u0 = np.zeros((2, Nx, Ny))
    u0[0] = U_inflow
    f = equilibrium(rho0, u0)

    print(f"\n  [{label}]")
    print(f"    grid {Nx}x{Ny}, kick at ({kick_x}, {kick_y}), {n_steps} steps")

    t_start = time.perf_counter()
    for step in range(n_steps):
        f, _Fx, _Fy = step_njit_with_force(
            f, tau, solid_mask, q_field, f_inflow_eq, True, True
        )
        if KICK_START <= step < KICK_END:
            f[2, kick_x, kick_y] += KICK_AMPLITUDE
            f[4, kick_x, kick_y] -= KICK_AMPLITUDE

        if (step + 1) % 2000 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"      step {step + 1}/{n_steps}  ({elapsed:.0f} s)")

    runtime = time.perf_counter() - t_start
    print(f"    done in {runtime:.0f} s ({runtime / n_steps * 1000:.2f} ms/step)")

    _, u = macroscopic(f)
    u_mag = np.sqrt(u[0] ** 2 + u[1] ** 2)

    # Central-difference vorticity in the interior
    dv_dx = np.zeros_like(u[1])
    du_dy = np.zeros_like(u[0])
    dv_dx[1:-1, :] = (u[1, 2:, :] - u[1, :-2, :]) / 2
    du_dy[:, 1:-1] = (u[0, :, 2:] - u[0, :, :-2]) / 2
    vorticity = dv_dx - du_dy

    u_mag_plot = np.where(solid_mask, np.nan, u_mag)
    vorticity_plot = np.where(solid_mask, np.nan, vorticity)

    return label, solid_mask, u_mag_plot, vorticity_plot


def _slugify(label):
    """Filesystem-safe filename slug from a preset label."""
    out = label.lower()
    for ch in [" ", "=", ",", "."]:
        out = out.replace(ch, "_")
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def main():
    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(exist_ok=True)

    print("=" * 72)
    print("Shape gallery -- 5 presets, common flow (U=0.1, tau=0.56)")
    print(f"  Grid {Nx}x{Ny}, {n_steps} steps each (~15 min total estimated)")
    print("=" * 72)

    results = []
    t_global = time.perf_counter()
    for factory in PRESETS:
        result = run_one(*factory())
        results.append(result)
    t_total = time.perf_counter() - t_global
    print(f"\nGallery complete in {t_total / 60:.1f} min")

    # --- Individual per-shape PNGs (|u| + vorticity) ---
    print("\nSaving individual wake plots:")
    for label, _, u_mag_plot, vorticity_plot in results:
        fig, (ax_u, ax_v) = plt.subplots(2, 1, figsize=(13, 4.5), sharex=True)

        mesh_u = ax_u.pcolormesh(
            np.arange(Nx), np.arange(Ny), u_mag_plot.T,
            shading="auto", cmap="viridis",
        )
        ax_u.set_aspect("equal")
        ax_u.set_ylabel("y")
        ax_u.set_title(f"|u|   --   {label}")
        plt.colorbar(mesh_u, ax=ax_u, label="|u|", pad=0.01, fraction=0.015)

        v_max = float(np.nanmax(np.abs(vorticity_plot)))
        mesh_v = ax_v.pcolormesh(
            np.arange(Nx), np.arange(Ny), vorticity_plot.T,
            shading="auto", cmap="RdBu_r", vmin=-v_max, vmax=v_max,
        )
        ax_v.set_aspect("equal")
        ax_v.set_xlabel("x")
        ax_v.set_ylabel("y")
        ax_v.set_title(f"vorticity   --   {label}")
        plt.colorbar(mesh_v, ax=ax_v, label=r"$\omega$", pad=0.01, fraction=0.015)

        plt.tight_layout()
        fname = out_dir / f"wake_{_slugify(label)}.png"
        plt.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  {fname.relative_to(out_dir.parent)}")

    # --- Combined gallery PNG: vorticity stacked, one row per shape ---
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.0 * n + 1), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (label, _, _, vorticity_plot) in zip(axes, results, strict=True):
        v_max = float(np.nanmax(np.abs(vorticity_plot)))
        mesh = ax.pcolormesh(
            np.arange(Nx), np.arange(Ny), vorticity_plot.T,
            shading="auto", cmap="RdBu_r", vmin=-v_max, vmax=v_max,
        )
        ax.set_aspect("equal")
        ax.set_ylabel("y")
        ax.set_title(label, fontsize=11, loc="left")
        plt.colorbar(mesh, ax=ax, label=r"$\omega$", pad=0.01, fraction=0.015)
    axes[-1].set_xlabel("x")
    fig.suptitle(
        "LBM shape gallery  --  vorticity wake at U=0.1, tau=0.56  (Re varies with characteristic length)",
        fontsize=12,
    )
    plt.tight_layout()
    fname_gallery = out_dir / "shape_gallery_vorticity.png"
    plt.savefig(fname_gallery, dpi=120, bbox_inches="tight")
    print(f"\nSaved combined gallery: {fname_gallery.relative_to(out_dir.parent)}")
    plt.show()


if __name__ == "__main__":
    main()
