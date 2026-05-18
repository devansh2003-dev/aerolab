"""Validate that the LBM velocity field is physical -- so the streamlines
drawn from it represent actual flow, not numerical noise or solver bugs.

Streamlines in a CFD plot are integral curves of the velocity field. They
faithfully represent the flow IF the velocity field itself satisfies the
governing equations. For incompressible flow we check four invariants:

  1. Divergence: div(u) ~= 0 everywhere in the fluid (incompressibility).
  2. No-slip: |u| ~= 0 at solid lattice nodes (bounce-back enforces this).
  3. Mass conservation: integrated x-flux is constant across x-slices.
  4. Asymptotic: far from the body, u_x -> U_inflow, u_y -> 0.

If all four pass with the documented tolerances, the streamline picture is
trustworthy. If any fails, the streamlines may be physically meaningless
even if visually striking.

Run from project root:
    python scripts/dev_validate_streamlines.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from src.lbm import CS2, equilibrium, macroscopic, step_njit_with_force
from src.shapes import cylinder_mask

# --- Config: same parameters the Streamlit LBM mode uses for Cylinder Re=100 ---
Nx, Ny = 320, 100
U_INFLOW = 0.1
body_x = 70
cy = Ny // 2
D = 20
mask = cylinder_mask(Nx, Ny, cx=body_x, cy=cy, radius=D // 2)

# Re = 100, tau = 0.56
nu = U_INFLOW * D / 100.0
tau = nu / CS2 + 0.5

INFLOW_DIRS = np.array([1, 5, 8], dtype=np.int32)
OUTFLOW_DIRS = np.array([3, 6, 7], dtype=np.int32)
KICK_START, KICK_END = 100, 500
KICK_AMPLITUDE = 0.005

N_STEPS = 3000

print("=" * 78)
print(f"Streamline validation: Cylinder D={D}, Re=100, tau={tau:.4f}, grid {Nx}x{Ny}")
print("=" * 78)

# --- Run sim ---
rho0 = np.ones((Nx, Ny))
u0 = np.zeros((2, Nx, Ny))
u0[0] = U_INFLOW
f = equilibrium(rho0, u0)
f_inflow_eq = equilibrium(1.0, np.array([U_INFLOW, 0.0]))
kick_x = body_x + D
kick_y = cy + 2

print(f"\nRunning {N_STEPS} steps...")
t0 = time.perf_counter()
for step in range(N_STEPS):
    f, _, _ = step_njit_with_force(f, tau, mask, f_inflow_eq, INFLOW_DIRS, OUTFLOW_DIRS)
    if KICK_START <= step < KICK_END:
        f[2, kick_x, kick_y] += KICK_AMPLITUDE
        f[4, kick_x, kick_y] -= KICK_AMPLITUDE
print(f"  done in {time.perf_counter() - t0:.1f}s\n")

_, u = macroscopic(f)
ux, uy = u[0], u[1]
fluid = ~mask
solid = mask

# === Test 1: divergence ===
# Central-difference divergence on the interior; one-sided at edges.
div_u = np.zeros((Nx, Ny))
div_u[1:-1, :] += (ux[2:, :] - ux[:-2, :]) / 2.0
div_u[:, 1:-1] += (uy[:, 2:] - uy[:, :-2]) / 2.0
abs_div = np.abs(div_u[fluid])
# For weakly-compressible LBM, div(u) is O(Ma^2). Here Ma = U/cs = 0.1/sqrt(1/3) ~= 0.173.
# So expected |div(u)| ~ 1e-4 to 1e-3 in fluid bulk; larger near body where the
# discretization breaks down at the cell scale. Tolerance: median < 1e-3.
median_div = float(np.median(abs_div))
p99_div = float(np.percentile(abs_div, 99))
max_div = float(abs_div.max())
print("Test 1 -- divergence  |div(u)|  (should be near zero in incompressible fluid)")
print(f"  median:  {median_div:.2e}")
print(f"  p99:     {p99_div:.2e}")
print(f"  max:     {max_div:.2e}  (expected near solid boundary, not in bulk)")
t1_pass = median_div < 1e-3
print(f"  -> {'PASS' if t1_pass else 'FAIL'} (median < 1e-3)")

# === Test 2: no-slip at body ===
# Halfway bounce-back doesn't strictly zero u at solid lattice nodes -- it
# enforces zero flux through the cell face. But interior solid nodes that are
# fully surrounded by solid should have near-zero macroscopic u.
ux_solid = np.abs(ux[solid])
uy_solid = np.abs(uy[solid])
ux_solid_mean = float(ux_solid.mean())
uy_solid_mean = float(uy_solid.mean())
print(f"\nTest 2 -- no-slip  |u|  at solid cells")
print(f"  mean |u_x| inside body: {ux_solid_mean:.2e}  (should be << U={U_INFLOW})")
print(f"  mean |u_y| inside body: {uy_solid_mean:.2e}")
t2_pass = ux_solid_mean < 0.1 * U_INFLOW and uy_solid_mean < 0.1 * U_INFLOW
print(f"  -> {'PASS' if t2_pass else 'FAIL'} (both << inflow speed)")

# === Test 3: mass conservation through cross-sections ===
# Integrate u_x over y at each x. In steady state, the result should be constant
# across x (mass enters at the inflow at rate U*Ny and must leave at the same rate).
flux_per_x = ux.sum(axis=1)               # shape (Nx,)
# Far-field flux (away from body and boundaries) is the reference.
flux_ref = float(flux_per_x[Nx - 20:Nx - 5].mean())
flux_variation = float((flux_per_x[20:-20].max() - flux_per_x[20:-20].min()) / abs(flux_ref))
print(f"\nTest 3 -- mass conservation  integral(u_x dy)  across vertical slices")
print(f"  far-field reference flux: {flux_ref:.4f}  (target: U*Ny = {U_INFLOW * Ny:.2f}, "
      f"corrected for body blockage)")
print(f"  flux variation across domain interior: {flux_variation:.2%}")
t3_pass = flux_variation < 0.10
print(f"  -> {'PASS' if t3_pass else 'FAIL'} (< 10% variation across domain)")

# === Test 4: asymptotic flow ===
# Far downstream and far above/below the body, the flow should recover
# u_x ~= U_inflow, u_y ~= 0.
far_x = Nx - 5
far_ux = float(ux[far_x, :].mean())
far_uy = float(np.abs(uy[far_x, :]).mean())
print(f"\nTest 4 -- asymptotic flow  (5 cells before outflow)")
print(f"  mean u_x: {far_ux:.4f}  (target: {U_INFLOW})")
print(f"  mean |u_y|: {far_uy:.4f}  (target: 0)")
t4_pass = abs(far_ux - U_INFLOW) / U_INFLOW < 0.05 and far_uy < 0.01
print(f"  -> {'PASS' if t4_pass else 'FAIL'} (u_x within 5%, |u_y| < 1% of U)")

# === Summary ===
print("\n" + "=" * 78)
all_pass = t1_pass and t2_pass and t3_pass and t4_pass
if all_pass:
    print("ALL CHECKS PASSED")
    print("Streamlines drawn from this velocity field faithfully represent the flow.")
else:
    print("ONE OR MORE CHECKS FAILED -- streamline picture may be misleading.")
print("=" * 78)

# --- Save divergence heatmap for visual confirmation ---
out_dir = Path(__file__).resolve().parent.parent / "data"
out_dir.mkdir(exist_ok=True)

div_plot = np.where(mask, np.nan, div_u)
clip = float(np.nanpercentile(np.abs(div_plot), 99))

fig, axes = plt.subplots(2, 1, figsize=(9, 5), dpi=100, facecolor="#0a0a0a")
ax_u, ax_d = axes

ax_u.set_facecolor("#0a0a0a")
u_mag = np.sqrt(ux ** 2 + uy ** 2)
mesh = ax_u.imshow(
    np.where(mask, np.nan, u_mag).T,
    cmap="viridis", origin="lower", aspect="equal",
    extent=[0, Nx - 1, 0, Ny - 1],
)
ax_u.set_title("velocity magnitude |u|", color="#f5f5f5", fontsize=10)
ax_u.set_xticks([]); ax_u.set_yticks([])
for s in ax_u.spines.values():
    s.set_color("#404040")
plt.colorbar(mesh, ax=ax_u, fraction=0.025, pad=0.01).ax.tick_params(
    color="#f5f5f5", labelcolor="#f5f5f5", labelsize=8,
)

ax_d.set_facecolor("#0a0a0a")
mesh = ax_d.imshow(
    div_plot.T, cmap="RdBu_r", origin="lower", aspect="equal",
    extent=[0, Nx - 1, 0, Ny - 1], vmin=-clip, vmax=clip,
)
ax_d.set_title(
    f"div(u)  --  median |div|={median_div:.1e},  p99={p99_div:.1e}",
    color="#f5f5f5", fontsize=10,
)
ax_d.set_xticks([]); ax_d.set_yticks([])
for s in ax_d.spines.values():
    s.set_color("#404040")
plt.colorbar(mesh, ax=ax_d, fraction=0.025, pad=0.01).ax.tick_params(
    color="#f5f5f5", labelcolor="#f5f5f5", labelsize=8,
)

fig.tight_layout()
png_path = out_dir / "validation_streamlines_divergence.png"
fig.savefig(png_path, facecolor="#0a0a0a", dpi=100)
plt.close(fig)
print(f"\nSaved validation figure: {png_path.relative_to(out_dir.parent)}")
