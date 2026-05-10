"""Sweep angle of attack for a NACA airfoil and plot its polars.

Generates a 3-panel figure: CL-vs-alpha, CD-vs-alpha, and the drag polar (CL vs CD).
Saves a PNG to data/ and pops up an interactive matplotlib window.

Run from the project root:
    python scripts/polar_sweep.py
"""
from pathlib import Path

import aerosandbox as asb
import matplotlib.pyplot as plt
import neuralfoil as nf
import numpy as np

# --- Configuration ---
airfoil_name = "naca4412"
alphas = np.linspace(-5, 15, 41)   # 41 points from -5 deg to 15 deg, 0.5 deg step
reynolds = 500_000

# --- Build the airfoil and run NeuralFoil ---
airfoil = asb.Airfoil(airfoil_name)

# NeuralFoil is vectorized: passing an array of alphas returns arrays for each output.
# One batched call is far faster than looping (the neural net runs once on a batch).
aero = nf.get_aero_from_airfoil(
    airfoil=airfoil,
    alpha=alphas,
    Re=reynolds,
    model_size="xxxlarge",
)

cl = aero["CL"]
cd = aero["CD"]
ld = cl / cd

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

# Drag polar -- this is the classic aerodynamics chart: lift vs drag, parametric in alpha.
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
# Path(__file__).parent.parent is the project root regardless of where you run from.
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
