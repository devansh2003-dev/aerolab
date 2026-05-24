"""Lid-driven cavity smoke test for the LBM solver.

The classical first benchmark for any incompressible flow solver: a square box
of fluid with three no-slip walls (bounce-back) and a moving top wall (lid)
that drags fluid in +x direction. After O(10^4) timesteps the flow reaches
a quasi-steady recirculating vortex.

What you should see in the saved PNG:
- High velocity magnitude in a thin layer just below the top lid
- A single clockwise recirculation vortex with its center slightly above and
  right of the cavity mid-point. For Re=100 (Ghia et al. 1982 benchmark),
  the center lands near (0.61 * Nx, 0.74 * Ny) -- roughly (61, 74) on a
  100x100 grid.
- Slow flow in the bottom corners

Run from the project root:
    python scripts/lid_cavity_smoke.py
"""
import sys
import time
from pathlib import Path

# Allow `from src.lbm import ...` when launched from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from src.lbm import CS2, bounce_back, collide, equilibrium, macroscopic, stream

# --- Configuration ---
Nx, Ny = 100, 100        # grid size (square cavity)
n_steps = 10_000         # timesteps to integrate
U_lid = 0.1              # lid velocity in lattice units (must stay << cs ~ 0.577)

# Set kinematic viscosity to give a target Reynolds number based on lid speed
# and cavity size: Re = U_lid * Nx / nu.
target_Re = 100.0
nu = U_lid * Nx / target_Re                # = 0.1 here
tau = nu / CS2 + 0.5                       # = 0.8 here, comfortably > 0.5

print(f"Lid-driven cavity: {Nx}x{Ny} grid, target Re = {target_Re:.0f}")
print(f"  U_lid = {U_lid}, nu = {nu:.4f}, tau = {tau:.3f}")
print(f"  Running {n_steps} timesteps...")

# --- Solid mask: bottom row, left column, right column. Top row is the lid. ---
solid_mask = np.zeros((Nx, Ny), dtype=bool)
solid_mask[:, 0] = True    # bottom (y = 0)
solid_mask[0, :] = True    # left   (x = 0)
solid_mask[-1, :] = True   # right  (x = Nx-1)
# Top row left as fluid -- we overwrite it with lid equilibrium each step.

# --- Pre-compute the lid equilibrium (constant in time) ---
# rho = 1.0, u = (U_lid, 0). equilibrium() returns shape (9,) for point input.
f_lid = equilibrium(1.0, np.array([U_lid, 0.0]))

# --- Initial condition: fluid at rest, uniform density ---
rho0 = np.ones((Nx, Ny))
u0 = np.zeros((2, Nx, Ny))
f = equilibrium(rho0, u0)

# --- Time loop ---
t_start = time.perf_counter()
for step in range(n_steps):
    f = collide(f, tau)
    f = bounce_back(f, solid_mask)
    f = stream(f)
    # Enforce moving lid by overwriting the top row each step. f_lid[:, None]
    # broadcasts the (9,) per-direction populations across all Nx columns.
    f[:, :, -1] = f_lid[:, None]

    if (step + 1) % 1000 == 0:
        _, u_now = macroscopic(f)
        u_mag_max = float(np.sqrt(u_now[0] ** 2 + u_now[1] ** 2).max())
        elapsed = time.perf_counter() - t_start
        print(f"  step {step + 1:5d}/{n_steps}  |  max|u| = {u_mag_max:.4f}  |  elapsed {elapsed:5.1f} s")

t_total = time.perf_counter() - t_start
print(f"Done in {t_total:.1f} s ({t_total / n_steps * 1000:.2f} ms/step)\n")

# --- Macroscopic fields + diagnostics ---
rho, u = macroscopic(f)
u_mag = np.sqrt(u[0] ** 2 + u[1] ** 2)

# Vortex center: minimum |u| in the DEEP interior, at least BUFFER cells away
# from any wall. The naive "min |u| anywhere in fluid" trick gets fooled by
# boundary layers and corner stagnation -- those cells are slow because of the
# no-slip wall, not because they're the rotation center. A 10-cell buffer is
# enough on a 100x100 grid to skip past those regions.
BUFFER = 10
interior = np.zeros_like(solid_mask)
interior[BUFFER:-BUFFER, BUFFER:-BUFFER] = True
u_mag_interior = np.where(interior, u_mag, np.inf)
ic_x, ic_y = np.unravel_index(np.argmin(u_mag_interior), u_mag_interior.shape)
print(f"Vortex center (min |u| in interior): (x={ic_x}, y={ic_y})")
print("  Reference for Re=100 (Ghia et al. 1982): approx (61, 74) on a 100x100 grid")

# --- Plot velocity magnitude with streamlines overlaid ---
fig, ax = plt.subplots(figsize=(7, 6.5))

# pcolormesh wants the data in (y, x) layout; we have (x, y) so transpose.
u_mag_plot = np.where(solid_mask, np.nan, u_mag)
mesh = ax.pcolormesh(
    np.arange(Nx),
    np.arange(Ny),
    u_mag_plot.T,
    shading="auto",
    cmap="viridis",
)

# streamplot is also (y, x) layout for the velocity components.
ax.streamplot(
    np.arange(Nx),
    np.arange(Ny),
    u[0].T,
    u[1].T,
    color="white",
    linewidth=0.6,
    density=1.5,
    arrowsize=1.0,
)

# Mark the vortex center for visual confirmation.
ax.plot(ic_x, ic_y, marker="x", color="red", markersize=10, markeredgewidth=2,
        label=f"vortex ({ic_x}, {ic_y})")
ax.legend(loc="lower left", fontsize=9, facecolor="white", framealpha=0.8)

ax.set_xlabel("x (lattice units)")
ax.set_ylabel("y (lattice units)")
ax.set_title(f"Lid-driven cavity  |  Re = {target_Re:.0f}  |  step {n_steps}")
ax.set_aspect("equal")
plt.colorbar(mesh, ax=ax, label="|u| (lattice units)")
plt.tight_layout()

# --- Save ---
out_dir = Path(__file__).resolve().parent.parent / "data"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "cavity_velocity_streamlines.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"\nSaved figure: {out_path.relative_to(out_dir.parent)}")

plt.show()
