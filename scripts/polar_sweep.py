"""Sweep angle of attack for a NACA airfoil and plot its polars.

Generates a 3-panel figure: CL-vs-alpha, CD-vs-alpha, and the drag polar (CL vs CD).
Saves a PNG to data/ and pops up an interactive matplotlib window.

Run from the project root:
    python scripts/polar_sweep.py
"""
import sys
from pathlib import Path

# Add project root to sys.path so we can import from src/ without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from src.airfoils import analyze_airfoil

# --- Configuration ---
airfoil_name = "naca4412"
alphas = np.linspace(-5, 15, 41)   # 41 points from -5 deg to 15 deg, 0.5 deg step
reynolds = 500_000

# --- Run the sweep (one batched NeuralFoil call) ---
aero = analyze_airfoil(airfoil_name, alphas, reynolds)
cl, cd, ld = aero["CL"], aero["CD"], aero["LD"]

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# CL vs alpha
axes[0].plot(alphas, cl, color="tab:blue", linewidth=1.8)
axes[0].axhline(0, color="k", linewidth=0.5)
axes[0].axvline(0, color="k", linewidth=0.5)
axes[0].set_xlabel(r"Angle of attack $\alpha$ (deg)")
axes[0].set_ylabel(r"Lift coefficient $C_L$")
axes[0].set_title(r"$C_L$ vs $\alpha$")
axes[0].grid(True, alpha=0.3)

# CD vs alpha
axes[1].plot(alphas, cd, color="tab:red", linewidth=1.8)
axes[1].set_xlabel(r"Angle of attack $\alpha$ (deg)")
axes[1].set_ylabel(r"Drag coefficient $C_D$")
axes[1].set_title(r"$C_D$ vs $\alpha$")
axes[1].grid(True, alpha=0.3)

# Drag polar -- the classic aerodynamics chart: lift vs drag, parametric in alpha.
axes[2].plot(cd, cl, color="tab:green", linewidth=1.8)
axes[2].set_xlabel(r"Drag coefficient $C_D$")
axes[2].set_ylabel(r"Lift coefficient $C_L$")
axes[2].set_title("Drag polar")
axes[2].grid(True, alpha=0.3)

fig.suptitle(
    f"{airfoil_name.upper()} polars  |  Re = {reynolds:,}  |  NeuralFoil (xxxlarge)",
    fontsize=13,
)
fig.tight_layout()

# --- Save figure to data/ (git-ignored) ---
out_dir = Path(__file__).resolve().parent.parent / "data"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / f"polar_{airfoil_name}_re{reynolds}.png"
fig.savefig(out_path, dpi=120, bbox_inches="tight")

# --- Text summary ---
i_ld_max = int(np.argmax(ld))
i_cl_max = int(np.argmax(cl))
print(f"{airfoil_name.upper()}  |  Re = {reynolds:,}  |  swept {len(alphas)} points")
print(f"  Max L/D = {ld[i_ld_max]:.1f}  at alpha = {alphas[i_ld_max]:+.1f} deg")
print(f"  Max CL  = {cl[i_cl_max]:.3f}  at alpha = {alphas[i_cl_max]:+.1f} deg")
print(f"  Saved figure: {out_path.relative_to(out_dir.parent)}")

plt.show()  # blocks until you close the window; comment out for headless runs
