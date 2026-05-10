"""Phase 1 Week 1: flow over a circular cylinder, visual wake.

Channel with a cylinder obstacle at moderate Reynolds number. Inflow at left,
outflow at right, periodic top/bottom (wake doesn't see itself if Ny is large
enough). Cylinder is bounce-back via a shape mask.

The goal of THIS script is qualitative: see the von Karman vortex street form
behind the cylinder. Quantitative drag-coefficient validation (Cd ~ 1.4 at
Re=100, the Phase 1 acceptance gate) is a separate run with force calculation
in week1_cylinder.py once src/forces.py is in place.

Run from the project root:
    python scripts/week1_cylinder.py
"""
import sys
import time
from pathlib import Path

# Allow `from src.* import ...` when launched from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from src.lbm import CS2, bounce_back, collide, equilibrium, macroscopic, stream
from src.shapes import cylinder_mask

# --- Configuration ---
Nx, Ny = 300, 100        # channel grid
D = 20                   # cylinder diameter (lattice units)
# Cylinder center y is OFFSET BY 0.5 cells (off the integer grid). On a perfectly
# symmetric mesh the symmetric wake at Re=100 is unstable but won't depart from
# it without a perturbation -- we'd be sitting on a knife edge. The half-cell
# offset breaks the discrete symmetry and lets vortex shedding develop in O(1000)
# steps instead of O(50000+).
cx, cy = 80.0, Ny / 2 + 0.5
U_inflow = 0.1           # inflow velocity in lattice units (Mach ~ 0.17, OK)

target_Re = 100.0
nu = U_inflow * D / target_Re                 # 0.02
tau = nu / CS2 + 0.5                          # 0.56, comfortably > 0.5

n_steps = 12_000          # ~10 vortex shedding cycles after transient

print(f"Cylinder in channel: {Nx}x{Ny}")
print(f"  Cylinder D={D} centered at ({cx}, {cy})")
print(f"  Re={target_Re:.0f}, U_inflow={U_inflow}, nu={nu:.4f}, tau={tau:.3f}")
print(f"  Running {n_steps} timesteps (pure NumPy, no JIT)...")

# --- Solid mask: just the cylinder. Top/bottom are periodic (np.roll wraps). ---
solid_mask = cylinder_mask(Nx, Ny, cx, cy, D / 2)

# --- Pre-compute inflow equilibrium (constant in time) ---
f_inflow = equilibrium(1.0, np.array([U_inflow, 0.0]))   # shape (9,)

# --- Selective boundary direction sets ---
# At the LEFT wall, east-pointing populations (1, 5, 8) are streamed in from
# "outside the domain" (incorrectly via np.roll wrap) -- override these with
# inflow equilibrium values. The other directions came from the interior or
# stay put, so leave them alone.
INFLOW_DIRS = np.array([1, 5, 8])
# At the RIGHT wall, west-pointing populations (3, 6, 7) similarly need to
# come from outside. Zero-gradient: copy from the second-to-last column.
OUTFLOW_DIRS = np.array([3, 6, 7])

# --- Initial condition: impulsive start with uniform inflow velocity ---
# Plus a tiny random y-velocity perturbation (~1e-3, three orders of magnitude
# below U_inflow) to bootstrap the vortex shedding instability. Without it,
# perfect symmetry suppresses shedding indefinitely; with it, shedding starts
# in ~3000-5000 steps.
rng = np.random.default_rng(seed=42)
rho0 = np.ones((Nx, Ny))
u0 = np.zeros((2, Nx, Ny))
u0[0] = U_inflow                                  # uniform x-velocity
u0[1] = rng.normal(0, 1e-3, size=(Nx, Ny))        # small random y-perturbation
f = equilibrium(rho0, u0)

# --- Time loop ---
t_start = time.perf_counter()
for step in range(n_steps):
    f = collide(f, tau)
    f = bounce_back(f, solid_mask)
    f = stream(f)

    # Inflow: override east-pointing populations at x=0 with inflow equilibrium.
    # f_inflow has shape (9,); index it down to the inflow directions and
    # broadcast across all Ny rows.
    f[INFLOW_DIRS, 0, :] = f_inflow[INFLOW_DIRS, None]

    # Outflow: zero-gradient for west-pointing populations at x=Nx-1.
    f[OUTFLOW_DIRS, -1, :] = f[OUTFLOW_DIRS, -2, :]

    if (step + 1) % 1000 == 0:
        _, u_now = macroscopic(f)
        u_mag_max = float(np.sqrt(u_now[0] ** 2 + u_now[1] ** 2).max())
        elapsed = time.perf_counter() - t_start
        print(f"  step {step + 1:5d}/{n_steps}  |  max|u| = {u_mag_max:.4f}  |  {elapsed:5.1f} s")

t_total = time.perf_counter() - t_start
print(f"Done in {t_total:.1f} s ({t_total / n_steps * 1000:.2f} ms/step)\n")

# --- Macroscopic fields + vorticity ---
rho, u = macroscopic(f)
u_mag = np.sqrt(u[0] ** 2 + u[1] ** 2)

# Vorticity = dv/dx - du/dy. Central differences in the interior, zero at edges
# (we never look at the very edges in the plot anyway).
dv_dx = np.zeros_like(u[1])
du_dy = np.zeros_like(u[0])
dv_dx[1:-1, :] = (u[1, 2:, :] - u[1, :-2, :]) / 2
du_dy[:, 1:-1] = (u[0, :, 2:] - u[0, :, :-2]) / 2
vorticity = dv_dx - du_dy

# Mask solid cells so they appear as "no data" in the colormap.
u_mag_plot = np.where(solid_mask, np.nan, u_mag)
vorticity_plot = np.where(solid_mask, np.nan, vorticity)

# --- Plot: two stacked panels ---
fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)

# Top panel: velocity magnitude.
mesh_u = axes[0].pcolormesh(
    np.arange(Nx), np.arange(Ny), u_mag_plot.T,
    shading="auto", cmap="viridis",
)
axes[0].set_aspect("equal")
axes[0].set_ylabel("y")
axes[0].set_title(f"|u|   (Re={target_Re:.0f}, step {n_steps}, {Nx}x{Ny} grid)")
plt.colorbar(mesh_u, ax=axes[0], label="|u|", pad=0.01, fraction=0.015)

# Bottom panel: vorticity. Diverging colormap centered at zero so positive
# (counter-clockwise) and negative (clockwise) vortices read as opposite colors.
v_max = float(np.nanmax(np.abs(vorticity_plot)))
mesh_v = axes[1].pcolormesh(
    np.arange(Nx), np.arange(Ny), vorticity_plot.T,
    shading="auto", cmap="RdBu_r", vmin=-v_max, vmax=v_max,
)
axes[1].set_aspect("equal")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].set_title("vorticity   (red = clockwise, blue = counter-clockwise)")
plt.colorbar(mesh_v, ax=axes[1], label=r"$\omega$", pad=0.01, fraction=0.015)

plt.tight_layout()

out_dir = Path(__file__).resolve().parent.parent / "data"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "cylinder_wake.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"Saved figure: {out_path.relative_to(out_dir.parent)}")

plt.show()
