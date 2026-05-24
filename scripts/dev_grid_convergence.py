"""Grid convergence study for AeroLab's LBM solver.

Runs the SAME physical case (cylinder Re=100, U_inflow=0.1) on BOTH
production presets and reports the grid-resolution gap:

  * Standard:  320 x 100 grid, D=20 cells across body  (20.0% blockage)
  * Detailed:  720 x 240 grid, D=45 cells across body  (18.8% blockage)

The two presets have nearly identical channel blockage (within 1.2%),
so any difference in Cd or Strouhal between them is dominated by grid
resolution -- not by changing geometry. That makes this a clean
convergence test.

What grid convergence tells us:
  * If Cd_coarse ~= Cd_fine: the simulation is grid-converged at the
    coarser grid. Production users on Standard are getting the same
    physics as Detailed, just at lower visual resolution.
  * If they disagree significantly: the coarser grid is undercooked.
    Production Standard-preset numbers don't match Detailed-preset
    numbers, which means we can't yet say which (if either) matches
    physical reality.

LBM is formally 2nd-order accurate in space, so the (Cd_standard -
Cd_detailed) gap should scale as ~(h_std/h_det)^2 = (45/20)^2 ~= 5x
the remaining error on Detailed. The script reports a Richardson
extrapolation as a rough estimate of grid-converged Cd.

Run from the project root:
    python scripts/dev_grid_convergence.py

The script takes ~3-5 minutes total (Standard + Detailed combined).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")  # headless: don't try to open a window
import matplotlib.pyplot as plt
import numpy as np

from src.lbm import (
    C_SMAG,
    CS2,
    equilibrium,
    macroscopic,
    step_njit_mrt_with_force,
)
from src.shapes import cylinder_mask, cylinder_q_field

# Both presets, defined together for easy comparison. body_x is chosen so the
# cylinder sits ~3.5 D downstream of the inlet (gives the flow room to develop
# before hitting the body) and the wake has ~12-15 D of channel for vortex
# shedding before the outflow boundary.
PRESETS = {
    "Standard": dict(Nx=320, Ny=100, body_x=70,  cy=50,  D=20, n_steps=8000),
    "Detailed": dict(Nx=720, Ny=240, body_x=160, cy=120, D=45, n_steps=14000),
}

# Common physics. We pin everything except the grid + body size so the
# comparison is genuinely grid-vs-grid.
RE = 100
U = 0.1
KICK_START, KICK_END = 30, 200
KICK_AMPLITUDE = 0.008

# Textbook references for cylinder at Re=100. We don't expect either preset
# to hit these exactly (channel blockage + halfway-BB artifacts shift them).
# Reporting them is for sanity, not as a pass/fail.
TEXTBOOK_CD = 1.4
TEXTBOOK_ST = 0.165


def run_preset(name: str, cfg: dict) -> dict:
    """Run a single preset's full simulation and extract Cd + St."""
    Nx, Ny = cfg["Nx"], cfg["Ny"]
    D = cfg["D"]
    body_x, cy = cfg["body_x"], cfg["cy"]
    n_steps = cfg["n_steps"]

    nu = U * D / RE
    tau = nu / CS2 + 0.5

    mask = cylinder_mask(Nx, Ny, cx=body_x, cy=cy, radius=D // 2)
    q_field = cylinder_q_field(Nx, Ny, cx=body_x, cy=cy, radius=D // 2)

    rho0 = np.ones((Nx, Ny))
    u0 = np.zeros((2, Nx, Ny))
    u0[0] = U
    f = equilibrium(rho0, u0)
    f_inflow = equilibrium(1.0, np.array([U, 0.0]))
    kick_x = body_x + D
    kick_y = cy + 2

    Fx_history = np.zeros(n_steps)
    Fy_history = np.zeros(n_steps)
    mass_initial = float(f.sum())

    print(f"\n--- {name}: {Nx}x{Ny}, D={D} ({D/Ny*100:.1f}% blockage), "
          f"{n_steps} steps, tau={tau:.4f} ---")
    t0 = time.perf_counter()
    for step in range(n_steps):
        f, Fx, Fy = step_njit_mrt_with_force(
            f, tau, mask, q_field, f_inflow, True, True,
        )
        if KICK_START <= step < KICK_END:
            f[2, kick_x, kick_y] += KICK_AMPLITUDE
            f[4, kick_x, kick_y] -= KICK_AMPLITUDE
        Fx_history[step] = Fx
        Fy_history[step] = Fy
        if (step + 1) % 2000 == 0:
            elapsed = time.perf_counter() - t0
            rate = (step + 1) / elapsed
            eta = (n_steps - step - 1) / rate
            print(f"  step {step + 1:>5d}/{n_steps}  "
                  f"({rate:>5.0f} steps/s, ETA {eta:.0f}s)")
    elapsed = time.perf_counter() - t0
    print(f"  done in {elapsed:.1f}s")

    # Time-mean Cd: drop first half as transient. Cd = 2 Fx / (rho * U^2 * D).
    transient = n_steps // 2
    Fx_steady = Fx_history[transient:]
    Fy_steady = Fy_history[transient:]
    cd_mean = 2.0 * float(Fx_steady.mean()) / (1.0 * U ** 2 * D)
    cd_std = 2.0 * float(Fx_steady.std()) / (1.0 * U ** 2 * D)

    # Strouhal: FFT of Fy (lift), pick the dominant frequency.
    Fy_detrended = Fy_steady - Fy_steady.mean()
    fft = np.fft.rfft(Fy_detrended)
    freqs = np.fft.rfftfreq(len(Fy_steady), d=1.0)
    power = np.abs(fft) ** 2
    peak_idx = 1 + int(np.argmax(power[1:]))
    f_peak = float(freqs[peak_idx])
    st = f_peak * D / U
    # Number of full vortex cycles captured in the steady window -- if this
    # is <8 the FFT is too coarse to trust the St number.
    n_cycles = (n_steps - transient) * f_peak

    # Mass drift over the run.
    mass_final = float(f.sum())
    mass_drift_pct = abs(mass_final - mass_initial) / mass_initial * 100

    # u-field for visual artifact.
    _, u_field = macroscopic(f)

    return dict(
        Nx=Nx, Ny=Ny, D=D, blockage=D / Ny, n_steps=n_steps, tau=tau,
        cd_mean=cd_mean, cd_std=cd_std, st=st, n_cycles=n_cycles,
        mass_drift_pct=mass_drift_pct,
        u_x=u_field[0], u_y=u_field[1], mask=mask,
        Fy_history=Fy_history, transient=transient,
        elapsed_s=elapsed,
    )


# -- Run both presets ------------------------------------------------------
print("=" * 78)
print("AeroLab grid convergence study  --  cylinder, Re=100, U=0.1")
print(f"Solver: MRT + Smagorinsky LES (C_SMAG = {C_SMAG})")
print("=" * 78)
results = {}
for name, cfg in PRESETS.items():
    results[name] = run_preset(name, cfg)

std = results["Standard"]
det = results["Detailed"]

# -- Convergence report ---------------------------------------------------
print()
print("=" * 78)
print("GRID CONVERGENCE REPORT")
print("=" * 78)
print()
print(f"{'Metric':<32}  {'Standard':>14}  {'Detailed':>14}  {'Delta':>10}")
print("-" * 78)
grid_std = f"{std['Nx']}x{std['Ny']}"
grid_det = f"{det['Nx']}x{det['Ny']}"
d_ratio = f"{det['D']/std['D']:.2f}x"
print(f"{'grid (Nx x Ny)':<32}  {grid_std:>14}  {grid_det:>14}  {'':>10}")
print(f"{'body diameter D (cells)':<32}  {std['D']:>14}  {det['D']:>14}  {d_ratio:>10}")
print(f"{'channel blockage':<32}  {std['blockage']:>13.1%}  "
      f"{det['blockage']:>13.1%}  "
      f"{(det['blockage']-std['blockage'])*100:>+9.2f}pp")
print(f"{'sim steps':<32}  {std['n_steps']:>14}  {det['n_steps']:>14}  {'':>10}")
print(f"{'relaxation time tau':<32}  {std['tau']:>14.4f}  {det['tau']:>14.4f}  "
      f"{'':>10}")
print(f"{'mass drift over run (%)':<32}  {std['mass_drift_pct']:>14.3f}  "
      f"{det['mass_drift_pct']:>14.3f}  {'':>10}")
print(f"{'vortex cycles in FFT window':<32}  {std['n_cycles']:>14.1f}  "
      f"{det['n_cycles']:>14.1f}  {'':>10}")
print("-" * 78)

# Main physics quantities.
cd_delta_pct = (std["cd_mean"] / det["cd_mean"] - 1) * 100
st_delta_pct = (std["st"] / det["st"] - 1) * 100
print(f"{'Cd (time-mean drag coeff)':<32}  {std['cd_mean']:>14.3f}  "
      f"{det['cd_mean']:>14.3f}  {cd_delta_pct:>+9.1f}%")
print(f"{'  vs textbook Cd=1.4':<32}  "
      f"{(std['cd_mean']/TEXTBOOK_CD-1)*100:>+13.0f}%  "
      f"{(det['cd_mean']/TEXTBOOK_CD-1)*100:>+13.0f}%  {'':>10}")
print(f"{'St (Strouhal from lift FFT)':<32}  {std['st']:>14.3f}  "
      f"{det['st']:>14.3f}  {st_delta_pct:>+9.1f}%")
print(f"{'  vs textbook St=0.165':<32}  "
      f"{(std['st']/TEXTBOOK_ST-1)*100:>+13.0f}%  "
      f"{(det['st']/TEXTBOOK_ST-1)*100:>+13.0f}%  {'':>10}")

# Richardson extrapolation: assume Cd ~ Cd_exact + A * h^2 where h is grid
# spacing (we use 1/D as a proxy for h). With refinement ratio r = D_det/D_std:
#   Cd_exact = Cd_det + (Cd_det - Cd_std) / (r^2 - 1)
# This is the leading-order grid-converged estimate IF the error is purely
# 2nd-order in h. It will be wrong if (a) the BC error has a different
# order, or (b) the asymptotic regime hasn't been reached yet.
r = det["D"] / std["D"]
cd_richardson = det["cd_mean"] + (det["cd_mean"] - std["cd_mean"]) / (r ** 2 - 1)
st_richardson = det["st"] + (det["st"] - std["st"]) / (r ** 2 - 1)
print()
print("Richardson extrapolation (assumes 2nd-order convergence in h):")
print(f"  Cd extrapolated to h -> 0:  {cd_richardson:.3f}  "
      f"(textbook 1.4, ratio {cd_richardson/TEXTBOOK_CD:.2f}x)")
print(f"  St extrapolated to h -> 0:  {st_richardson:.3f}  "
      f"(textbook 0.165, ratio {st_richardson/TEXTBOOK_ST:.2f}x)")
print()
print("Interpretation guide:")
print(f"  |Cd_std - Cd_det| / Cd_det = {abs(cd_delta_pct):.1f}%")
print(f"  |St_std - St_det| / St_det = {abs(st_delta_pct):.1f}%")
print("  Industry-standard 'grid-converged' threshold: < 5%")
print("  If either delta > 10%: production Standard preset is undercooked.")
print("  If even Richardson estimate is far from textbook: the leading error")
print("  is NOT discretization -- it's the boundary conditions (halfway BB,")
print("  channel blockage, equilibrium inflow). Those are III-4 and III-5.")
print("=" * 78)

# -- Save artifact ---------------------------------------------------------
out_dir = Path(__file__).resolve().parent.parent / "data"
out_dir.mkdir(exist_ok=True)

fig, axes = plt.subplots(
    2, 1, figsize=(11, 6), dpi=100, facecolor="#0a0a0a",
)
for ax, name in zip(axes, ("Standard", "Detailed"), strict=True):
    r = results[name]
    u_mag = np.sqrt(r["u_x"] ** 2 + r["u_y"] ** 2)
    u_plot = np.where(r["mask"], np.nan, u_mag)
    mesh = ax.imshow(
        u_plot.T, cmap="viridis", origin="lower", aspect="equal",
        extent=[0, r["Nx"] - 1, 0, r["Ny"] - 1], vmin=0, vmax=2 * U,
    )
    ax.set_title(
        f"{name}  {r['Nx']}x{r['Ny']}, D={r['D']}  "
        f"Cd={r['cd_mean']:.2f}  St={r['st']:.3f}  "
        f"(textbook Cd=1.4 St=0.165)",
        color="#f5f5f5", fontsize=10,
    )
    ax.set_facecolor("#0a0a0a")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(mesh, ax=ax, fraction=0.025, pad=0.01).ax.tick_params(
        color="#f5f5f5", labelcolor="#f5f5f5", labelsize=8,
    )

fig.suptitle(
    f"Grid convergence: cylinder Re={RE}, U={U}, MRT + LES (C_SMAG={C_SMAG})",
    color="#f5f5f5", fontsize=11,
)
fig.tight_layout()
png_path = out_dir / "validation_grid_convergence.png"
fig.savefig(png_path, facecolor="#0a0a0a", dpi=100)
plt.close(fig)
print(f"\nSaved artifact: {png_path.relative_to(out_dir.parent)}")
