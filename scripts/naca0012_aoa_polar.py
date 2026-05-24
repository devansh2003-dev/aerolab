"""NACA 0012 angle-of-attack polar.

Sweeps NACA 0012 across a range of AoA, time-averages CL and Cd at each, and
plots the canonical three-panel airfoil polar: CL-vs-alpha, Cd-vs-alpha, and
the drag polar CL-vs-Cd.

Reuses the existing JIT step + transient kick + force-calc machinery. With
chord=40 and U=0.1 nu=0.02, Re_c = U*chord/nu = 200 -- moderate Reynolds
where viscous effects are visible but flow is still well-defined.

Note on absolute magnitudes: the BGK-tau wall-correction artifact (documented
in Phase 1 W1 closure) biases Cd high at low tau. The SHAPE of the curves
remains physical:
  - CL(0) ~= 0 for the symmetric NACA 0012
  - CL roughly linear in alpha until stall onset
  - Cd has a minimum near alpha = 0 (drag bucket)
  - Stall (CL drop) somewhere in alpha = 10-15 deg at this Re

Run from project root:
    python scripts/naca0012_aoa_polar.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.lbm import CS2, equilibrium, step_njit_with_force
from src.shapes import naca4_airfoil_mask, no_bouzidi_q_field

# --- Configuration ---
Nx, Ny = 600, 200
chord = 40
U_inflow = 0.1
target_Re_c = 200.0
nu = U_inflow * chord / target_Re_c    # 0.02
tau = nu / CS2 + 0.5                   # 0.56
body_x = 120                           # LE at (120, cy) -- 3 chord upstream
cy = Ny // 2                           # 100

ALPHAS_DEG = [-5, 0, 2.5, 5, 7.5, 10, 12.5, 15]
n_steps = 20_000
n_average = 6_000        # last N steps for time average
n_fft = 12_000           # last N steps for FFT (Strouhal estimate)

KICK_START = 100
KICK_END = 500
KICK_AMPLITUDE = 0.005
KICK_Y_OFFSET = 2

f_inflow = equilibrium(1.0, np.array([U_inflow, 0.0]))
half_rho_u2_c = 0.5 * 1.0 * U_inflow ** 2 * chord    # normalize by chord


def run_one_aoa(alpha_deg):
    """Run NACA 0012 at given AoA. Returns dict of (Cd, CL, St, runtime)."""
    mask = naca4_airfoil_mask(
        Nx, Ny, cx=body_x, cy=cy, chord=chord,
        naca_code="0012", aoa_deg=alpha_deg,
    )
    # Halfway BB -- airfoil Bouzidi q-field is III-5d (pending).
    q_field = no_bouzidi_q_field(Nx, Ny)

    rho0 = np.ones((Nx, Ny))
    u0 = np.zeros((2, Nx, Ny))
    u0[0] = U_inflow
    f = equilibrium(rho0, u0)

    kick_x = body_x + chord + 10   # ~10 cells downstream of TE
    kick_y = cy + KICK_Y_OFFSET

    cd_hist = np.zeros(n_steps)
    cl_hist = np.zeros(n_steps)

    t0 = time.perf_counter()
    for step in range(n_steps):
        f, Fx, Fy = step_njit_with_force(
            f, tau, mask, q_field, f_inflow, True, True,
        )
        if KICK_START <= step < KICK_END:
            f[2, kick_x, kick_y] += KICK_AMPLITUDE
            f[4, kick_x, kick_y] -= KICK_AMPLITUDE
        cd_hist[step] = Fx / half_rho_u2_c
        cl_hist[step] = Fy / half_rho_u2_c
    elapsed = time.perf_counter() - t0

    cd_avg = float(cd_hist[-n_average:].mean())
    cd_std = float(cd_hist[-n_average:].std())
    cl_avg = float(cl_hist[-n_average:].mean())
    cl_std = float(cl_hist[-n_average:].std())

    # Strouhal from CL FFT (when shedding is present)
    cl_window = cl_hist[-n_fft:]
    cl_centered = cl_window - cl_window.mean()
    fft = np.fft.rfft(cl_centered)
    freqs = np.fft.rfftfreq(len(cl_centered), d=1.0)
    if len(fft) > 1:
        peak_idx = 1 + int(np.argmax(np.abs(fft[1:])))
        St = float(freqs[peak_idx] * chord / U_inflow)
    else:
        St = float("nan")

    return {
        "alpha_deg": alpha_deg,
        "Cd": cd_avg, "Cd_std": cd_std,
        "CL": cl_avg, "CL_std": cl_std,
        "L_over_D": cl_avg / cd_avg if cd_avg > 0 else float("nan"),
        "St": St,
        "runtime_s": elapsed,
    }


def main():
    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "naca0012_aoa_polar.csv"
    png_path = out_dir / "naca0012_aoa_polar.png"

    print("=" * 72)
    print(f"NACA 0012 AoA polar -- Re_c = {target_Re_c:.0f}, chord = {chord}")
    print(f"  {len(ALPHAS_DEG)} angles: {ALPHAS_DEG}")
    print(f"  Grid {Nx}x{Ny}, {n_steps} steps each (~30-40 min total estimated)")
    print(f"  Partial CSV at {csv_path.relative_to(out_dir.parent)} after each angle")
    print("=" * 72)

    results = []
    t_global = time.perf_counter()
    for i, alpha in enumerate(ALPHAS_DEG):
        print(f"\n  [{i + 1}/{len(ALPHAS_DEG)}]  AoA = {alpha:+.1f} deg")
        result = run_one_aoa(alpha)
        results.append(result)
        pd.DataFrame(results).to_csv(csv_path, index=False)   # partial save
        print(
            f"    Cd = {result['Cd']:.3f}  CL = {result['CL']:+.3f}  "
            f"L/D = {result['L_over_D']:+.2f}  St = {result['St']:.3f}  "
            f"({result['runtime_s']:.0f} s)"
        )
    t_total = time.perf_counter() - t_global
    print(f"\nPolar complete in {t_total / 60:.1f} min\n")

    df = pd.DataFrame(results)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- Plot: three-panel polar ---
    fig, (ax_cl, ax_cd, ax_polar) = plt.subplots(1, 3, figsize=(15, 4.8))

    ax_cl.plot(df["alpha_deg"], df["CL"], "o-", color="tab:blue",
               linewidth=2, markersize=8, label="LBM (BGK)")
    ax_cl.axhline(0, color="k", linewidth=0.5)
    ax_cl.axvline(0, color="k", linewidth=0.5)
    ax_cl.set_xlabel(r"angle of attack $\alpha$ (deg)")
    ax_cl.set_ylabel(r"$C_L$")
    ax_cl.set_title("Lift curve")
    ax_cl.legend(loc="best", fontsize=9)
    ax_cl.grid(alpha=0.3)

    ax_cd.plot(df["alpha_deg"], df["Cd"], "o-", color="tab:red",
               linewidth=2, markersize=8)
    ax_cd.set_xlabel(r"angle of attack $\alpha$ (deg)")
    ax_cd.set_ylabel(r"$C_D$")
    ax_cd.set_title("Drag curve")
    ax_cd.grid(alpha=0.3)

    ax_polar.plot(df["Cd"], df["CL"], "o-", color="tab:green",
                  linewidth=2, markersize=8)
    for _, row in df.iterrows():
        ax_polar.annotate(
            f"{row['alpha_deg']:+.1f}",
            (row["Cd"], row["CL"]),
            textcoords="offset points", xytext=(8, 4), fontsize=8,
        )
    ax_polar.set_xlabel(r"$C_D$")
    ax_polar.set_ylabel(r"$C_L$")
    ax_polar.set_title("Drag polar")
    ax_polar.grid(alpha=0.3)

    fig.suptitle(
        f"NACA 0012 AoA polar  --  Re$_c$ = {target_Re_c:.0f}, "
        f"chord = {chord}, BGK LBM at $\\tau = {tau}$",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(png_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved CSV: {csv_path.relative_to(out_dir.parent)}")
    print(f"Saved PNG: {png_path.relative_to(out_dir.parent)}")
    plt.show()


if __name__ == "__main__":
    main()
