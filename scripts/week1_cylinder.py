"""Phase 1 Week 1 acceptance gate: cylinder Re=100 validation.

Goal of THIS script: produce a time-averaged Cd and a Strouhal number, both
to be compared against textbook values (Cd ~ 1.4, St ~ 0.165 at Re=100).
The Phase 1 gate is BOTH numbers within +/-10% of textbook.

Approach
--------
- 300 x 100 channel, periodic top/bottom (free via np.roll wrapping in stream).
- Cylinder D=20 centered at (80, 50). ON-GRID -- no built-in mask asymmetry.
- Brief asymmetric "kick" during steps 100-500: inject +y momentum at one cell
  immediately behind the cylinder and slightly above the centerline. This
  bootstraps the symmetry-breaking instability that drives vortex shedding.
  Kick is OFF after step 500; by the time we time-average (last 10000 steps)
  it has long washed out.
- Numba @njit fused step function for 20-30x speedup over pure NumPy.
- 40000 timesteps total. Last 10000 are used for the time-average.
- Strouhal from FFT of CL(t) over the last 30000 steps.

Outputs (data/)
---------------
- cylinder_wake.png            : |u| + vorticity at final state
- cylinder_cd_history.png      : Cd(t) and CL(t), gate band shaded
- cylinder_cd_history.csv      : raw time series for replotting

Run from project root:
    python scripts/week1_cylinder.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.lbm import CS2, equilibrium, macroscopic, step_njit_with_force
from src.shapes import cylinder_mask

# --- Configuration ---
# Ny widened from 100 to 400 (2026-05-12) to drop blockage from 20% to 5%.
# Free-stream Cd ~ 1.4 only emerges when Ny/D >= 15-20; at the old 20% blockage
# the periodic-channel Cd was ~2.33 (correct physics, wrong benchmark).
Nx, Ny = 300, 400
D = 20
cx, cy = 80, 200             # on-grid integer center, channel midline
U_inflow = 0.1
target_Re = 100.0
nu = U_inflow * D / target_Re   # 0.02
tau = nu / CS2 + 0.5            # 0.56

# Transient kick: small +y momentum injection just behind the cylinder.
KICK_START = 100
KICK_END = 500
KICK_X = cx + D                  # 100 -- 1D downstream of cylinder center
KICK_Y_OFFSET = 2                # slightly above centerline (asymmetric)
KICK_AMPLITUDE = 0.005           # 5% of U_inflow

n_steps = 40_000
n_average = 10_000               # last N steps for Cd time-average
n_fft = 30_000                   # last N steps for Strouhal FFT

print(f"Cylinder validation: {Nx}x{Ny} grid")
print(f"  Cylinder D={D} at ({cx}, {cy})  -- ON-GRID center")
print(f"  Re={target_Re:.0f}, U_inflow={U_inflow}, nu={nu:.4f}, tau={tau:.3f}")
print(f"  Transient kick: steps {KICK_START}-{KICK_END} at ({KICK_X}, {cy + KICK_Y_OFFSET})")
print(f"  Running {n_steps} timesteps (Numba JIT)...\n")

# --- Setup ---
solid_mask = cylinder_mask(Nx, Ny, cx, cy, D / 2)
f_inflow = equilibrium(1.0, np.array([U_inflow, 0.0]))
INFLOW_DIRS = np.array([1, 5, 8], dtype=np.int32)
OUTFLOW_DIRS = np.array([3, 6, 7], dtype=np.int32)

# Initial condition: uniform inflow, no noise (kick will break the symmetry).
rho0 = np.ones((Nx, Ny))
u0 = np.zeros((2, Nx, Ny))
u0[0] = U_inflow
f = equilibrium(rho0, u0)

# Cd normalization: 0.5 * rho * U^2 * D in lattice units (rho=1, U=U_inflow, D=20).
half_rho_u2_D = 0.5 * 1.0 * U_inflow * U_inflow * D    # 0.1

# Storage for time series (preallocated for speed).
cd_history = np.zeros(n_steps)
cl_history = np.zeros(n_steps)

# Trigger JIT compile on a throwaway call so the timing below is clean.
print("  Compiling JIT step function (first call only)...")
t_jit = time.perf_counter()
_ = step_njit_with_force(f.copy(), tau, solid_mask, f_inflow, INFLOW_DIRS, OUTFLOW_DIRS)
print(f"  Compile done in {time.perf_counter() - t_jit:.1f} s\n")

# --- Time loop ---
t_start = time.perf_counter()
for step in range(n_steps):
    f, Fx, Fy = step_njit_with_force(f, tau, solid_mask, f_inflow, INFLOW_DIRS, OUTFLOW_DIRS)

    # Transient kick: inject +y momentum at one cell behind cylinder.
    # Adding +amp to f[2] (north) and -amp to f[4] (south) shifts the local
    # momentum by +2*amp in y while preserving local mass exactly.
    if KICK_START <= step < KICK_END:
        f[2, KICK_X, cy + KICK_Y_OFFSET] += KICK_AMPLITUDE
        f[4, KICK_X, cy + KICK_Y_OFFSET] -= KICK_AMPLITUDE

    cd_history[step] = Fx / half_rho_u2_D
    cl_history[step] = Fy / half_rho_u2_D

    if (step + 1) % 5000 == 0:
        recent = cd_history[max(0, step - 999): step + 1].mean()
        elapsed = time.perf_counter() - t_start
        print(f"  step {step + 1:6d}/{n_steps}  |  recent <Cd>={recent:6.3f}  |  Fx={Fx:.5f}  |  {elapsed:5.1f} s")

t_total = time.perf_counter() - t_start
print(f"\nDone in {t_total:.1f} s ({t_total / n_steps * 1000:.2f} ms/step)")

# --- Analysis ---
cd_window = cd_history[-n_average:]
cd_average = float(cd_window.mean())
cd_std = float(cd_window.std())

# Strouhal from rFFT of CL(t) over last n_fft steps.
cl_window = cl_history[-n_fft:]
cl_centered = cl_window - cl_window.mean()
fft = np.fft.rfft(cl_centered)
freqs = np.fft.rfftfreq(len(cl_centered), d=1.0)   # cycles per timestep
power = np.abs(fft) ** 2
peak_idx = 1 + int(np.argmax(power[1:]))           # skip DC bin
peak_freq = float(freqs[peak_idx])
St = peak_freq * D / U_inflow

# Gate evaluation.
cd_ref, st_ref = 1.4, 0.165
cd_err = abs(cd_average - cd_ref) / cd_ref * 100
st_err = abs(St - st_ref) / st_ref * 100
cd_pass = cd_err <= 10
st_pass = st_err <= 10

print(f"\n=== Phase 1 Week 1 acceptance gate ===")
print(f"  Cd time-averaged over last {n_average} steps : {cd_average:7.3f}  (std {cd_std:.4f})")
print(f"  Cd reference (textbook, Re=100)              : {cd_ref:7.3f}")
print(f"  Cd error                                     : {cd_err:6.1f} %   {'PASS' if cd_pass else 'FAIL'}")
print(f"  Strouhal St (from CL FFT, last {n_fft} steps) : {St:7.4f}")
print(f"  St reference (textbook, Re=100)              : {st_ref:7.4f}")
print(f"  St error                                     : {st_err:6.1f} %   {'PASS' if st_pass else 'FAIL'}")
print(f"  Gate                                         : {'PASS' if (cd_pass and st_pass) else 'FAIL'}")

# --- Visualization ---
rho, u = macroscopic(f)
u_mag = np.sqrt(u[0] ** 2 + u[1] ** 2)

dv_dx = np.zeros_like(u[1])
du_dy = np.zeros_like(u[0])
dv_dx[1:-1, :] = (u[1, 2:, :] - u[1, :-2, :]) / 2
du_dy[:, 1:-1] = (u[0, :, 2:] - u[0, :, :-2]) / 2
vorticity = dv_dx - du_dy

u_mag_plot = np.where(solid_mask, np.nan, u_mag)
vorticity_plot = np.where(solid_mask, np.nan, vorticity)

out_dir = Path(__file__).resolve().parent.parent / "data"
out_dir.mkdir(exist_ok=True)

# Plot 1: wake (final state)
fig1, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)
mesh_u = axes[0].pcolormesh(
    np.arange(Nx), np.arange(Ny), u_mag_plot.T, shading="auto", cmap="viridis",
)
axes[0].set_aspect("equal")
axes[0].set_ylabel("y")
axes[0].set_title(f"|u|   (Re={target_Re:.0f}, step {n_steps}, time-avg Cd = {cd_average:.3f})")
plt.colorbar(mesh_u, ax=axes[0], label="|u|", pad=0.01, fraction=0.015)

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
plt.savefig(out_dir / "cylinder_wake.png", dpi=120, bbox_inches="tight")

# Plot 2: Cd and CL time histories
fig2, (ax_cd, ax_cl) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
steps_arr = np.arange(n_steps)

ax_cd.plot(steps_arr, cd_history, color="tab:blue", linewidth=0.6)
ax_cd.axhline(cd_average, color="black", linestyle="--", alpha=0.7,
              label=f"<Cd>$_{{last {n_average}}}$ = {cd_average:.3f}")
ax_cd.axhline(cd_ref, color="red", linestyle=":", alpha=0.7, label=f"textbook ~{cd_ref}")
ax_cd.fill_between(steps_arr, cd_ref * 0.9, cd_ref * 1.1, color="red", alpha=0.08,
                    label=r"$\pm$10% gate")
ax_cd.set_ylabel("Cd")
ax_cd.set_title("Drag coefficient over time")
ax_cd.legend(loc="best", fontsize=9)
ax_cd.grid(alpha=0.3)

ax_cl.plot(steps_arr, cl_history, color="tab:green", linewidth=0.6)
ax_cl.set_xlabel("timestep")
ax_cl.set_ylabel("CL")
ax_cl.set_title(f"Lift coefficient over time   (St = {St:.4f} from FFT, textbook ~{st_ref})")
ax_cl.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / "cylinder_cd_history.png", dpi=120, bbox_inches="tight")

# CSV for replotting / external analysis
pd.DataFrame({"step": steps_arr, "Cd": cd_history, "CL": cl_history}).to_csv(
    out_dir / "cylinder_cd_history.csv", index=False
)

print(f"\nSaved:")
print(f"  {(out_dir / 'cylinder_wake.png').relative_to(out_dir.parent)}")
print(f"  {(out_dir / 'cylinder_cd_history.png').relative_to(out_dir.parent)}")
print(f"  {(out_dir / 'cylinder_cd_history.csv').relative_to(out_dir.parent)}")

plt.show()
