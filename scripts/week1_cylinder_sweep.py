"""Phase 1 Week 1 convergence sweep: cylinder Cd at Re=100.

Demonstrates how Cd approaches the free-stream textbook value (~1.4) as we
reduce the two dominant LBM error sources:

  1. Mach compressibility -- Cd error grows ~Ma^2 above Ma ~ 0.1
  2. Cylinder discretization -- halfway bounce-back is 1st-order at curved walls

The single-point gate ("Cd within 10% of 1.4") was failed at our base config
(D=20, Ma=0.17) with Cd=2.04. This sweep shows that the failure is convergence,
not a bug -- as we improve Ma and D, Cd should approach 1.4 monotonically.

Outputs:
    data/cylinder_convergence.csv  -- one row per config
    data/cylinder_convergence.png  -- two-panel: Cd vs Ma and Cd vs D

Run from project root:
    python scripts/week1_cylinder_sweep.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.lbm import CS2, equilibrium, step_njit_with_force
from src.shapes import cylinder_mask


# Sweep configurations: a Mach axis at D=20, then a D axis at Ma=0.087.
# Domain scales so each config has ~4D upstream, ~10D downstream, ~10D side margin.
CONFIGS = [
    # --- Mach sweep at D=20 ---
    dict(label="A: Ma=0.17, D=20",  Nx=300, Ny=400, D=20, U_inflow=0.10),
    dict(label="B: Ma=0.12, D=20",  Nx=300, Ny=400, D=20, U_inflow=0.07),
    dict(label="C: Ma=0.087, D=20", Nx=300, Ny=400, D=20, U_inflow=0.05),
    # --- Resolution sweep at Ma=0.087 ---
    dict(label="D: Ma=0.087, D=30", Nx=460, Ny=600, D=30, U_inflow=0.05),
    dict(label="E: Ma=0.087, D=40", Nx=600, Ny=800, D=40, U_inflow=0.05),
]

N_STEPS = 30_000
N_AVERAGE = 10_000   # last N steps for Cd time-average
N_FFT = 20_000       # last N steps for Strouhal FFT
TARGET_RE = 100.0
CD_REF = 1.4
ST_REF = 0.165

# Transient kick parameters (same for every config -- bootstraps shedding instability)
KICK_START = 100
KICK_END = 500
KICK_Y_OFFSET = 2
KICK_AMPLITUDE = 0.005


def run_one_config(Nx, Ny, D, U_inflow, target_Re, n_steps, n_average, n_fft, label):
    """Run a single cylinder configuration. Returns a result dict."""
    # Derived parameters
    cx = max(80, 4 * D)             # 4D from inlet, but at least 80 cells
    cy = Ny // 2                    # on-grid integer center
    nu = U_inflow * D / target_Re
    tau = nu / CS2 + 0.5
    Ma = U_inflow / np.sqrt(CS2)
    half_rho_u2_D = 0.5 * 1.0 * U_inflow * U_inflow * D

    # Geometry + boundary state
    solid_mask = cylinder_mask(Nx, Ny, cx, cy, D / 2)
    f_inflow = equilibrium(1.0, np.array([U_inflow, 0.0]))
    inflow_dirs = np.array([1, 5, 8], dtype=np.int32)
    outflow_dirs = np.array([3, 6, 7], dtype=np.int32)

    # Initial condition: uniform inflow (kick will break symmetry)
    rho0 = np.ones((Nx, Ny))
    u0 = np.zeros((2, Nx, Ny))
    u0[0] = U_inflow
    f = equilibrium(rho0, u0)

    kick_x = int(cx + D)
    kick_y = int(cy + KICK_Y_OFFSET)

    cd_history = np.zeros(n_steps)
    cl_history = np.zeros(n_steps)

    print(f"\n  [{label}]")
    print(f"    grid {Nx}x{Ny}, D={D}, U={U_inflow}, Ma={Ma:.3f}, tau={tau:.3f}, nu={nu:.4f}")
    print(f"    cylinder at ({cx}, {cy}), running {n_steps} steps...")

    t_start = time.perf_counter()
    for step in range(n_steps):
        f, Fx, Fy = step_njit_with_force(f, tau, solid_mask, f_inflow, inflow_dirs, outflow_dirs)

        # Transient asymmetry kick
        if KICK_START <= step < KICK_END:
            f[2, kick_x, kick_y] += KICK_AMPLITUDE
            f[4, kick_x, kick_y] -= KICK_AMPLITUDE

        cd_history[step] = Fx / half_rho_u2_D
        cl_history[step] = Fy / half_rho_u2_D

        # Periodic status print
        if (step + 1) % 5000 == 0:
            recent = float(cd_history[max(0, step - 999): step + 1].mean())
            elapsed = time.perf_counter() - t_start
            print(f"      step {step + 1:6d}/{n_steps}  recent<Cd>={recent:6.3f}  ({elapsed:6.0f} s)")

        # Stability sanity check
        if step % 1000 == 999 and not np.isfinite(Fx):
            print(f"      !! NaN detected at step {step + 1}. Aborting this config.")
            break

    t_total = time.perf_counter() - t_start

    # --- Analysis ---
    cd_window = cd_history[-n_average:]
    cd_avg = float(cd_window.mean())
    cd_std = float(cd_window.std())

    cl_window = cl_history[-n_fft:]
    cl_centered = cl_window - cl_window.mean()
    fft = np.fft.rfft(cl_centered)
    freqs = np.fft.rfftfreq(len(cl_centered), d=1.0)
    if len(fft) > 1:
        peak_idx = 1 + int(np.argmax(np.abs(fft[1:])))
        peak_freq = float(freqs[peak_idx])
        St = peak_freq * D / U_inflow
    else:
        St = float("nan")

    cd_err = abs(cd_avg - CD_REF) / CD_REF * 100
    st_err = abs(St - ST_REF) / ST_REF * 100

    print(f"    --> Cd = {cd_avg:.3f} (std {cd_std:.4f})  err {cd_err:.1f}% vs {CD_REF}")
    print(f"    --> St = {St:.4f}                err {st_err:.1f}% vs {ST_REF}")
    print(f"    --> runtime {t_total:.0f} s ({t_total / n_steps * 1000:.2f} ms/step)")

    return {
        "label": label,
        "Nx": Nx, "Ny": Ny, "D": D, "U": U_inflow,
        "Re": target_Re, "Ma": Ma, "tau": tau,
        "Cd": cd_avg, "Cd_std": cd_std,
        "St": St,
        "Cd_err_pct": cd_err, "St_err_pct": st_err,
        "n_steps": n_steps, "runtime_s": t_total,
    }


def main():
    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "cylinder_convergence.csv"
    png_path = out_dir / "cylinder_convergence.png"

    print("=" * 72)
    print("Cylinder Re=100 convergence sweep")
    print(f"  {len(CONFIGS)} configurations, ~1 hour total estimated runtime")
    print(f"  Partial results saved to {csv_path.relative_to(out_dir.parent)} after each config")
    print("=" * 72)

    results = []
    t_global = time.perf_counter()

    for i, cfg in enumerate(CONFIGS):
        print(f"\n[{i + 1}/{len(CONFIGS)}]")
        result = run_one_config(
            **cfg,
            target_Re=TARGET_RE,
            n_steps=N_STEPS,
            n_average=N_AVERAGE,
            n_fft=N_FFT,
        )
        results.append(result)
        # Save partial CSV after each config (crash insurance)
        pd.DataFrame(results).to_csv(csv_path, index=False)

    t_total = time.perf_counter() - t_global
    df = pd.DataFrame(results)

    print(f"\n{'=' * 72}")
    print(f"Sweep complete in {t_total / 60:.1f} min")
    print(f"{'=' * 72}\n")

    # Summary table
    summary_cols = ["label", "Nx", "Ny", "D", "U", "Ma", "tau", "Cd", "Cd_err_pct", "St", "St_err_pct"]
    print(df[summary_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved CSV: {csv_path.relative_to(out_dir.parent)}")

    # --- Convergence plot ---
    fig, (ax_ma, ax_d) = plt.subplots(1, 2, figsize=(13, 5))

    # Left panel: Cd vs Ma at D=20 (subset A, B, C)
    ma_subset = df[df["D"] == 20].sort_values("Ma")
    ax_ma.plot(ma_subset["Ma"], ma_subset["Cd"], "o-",
               color="tab:blue", linewidth=2, markersize=9, label="this sweep")
    for _, row in ma_subset.iterrows():
        ax_ma.annotate(f"{row['Cd']:.2f}", (row["Ma"], row["Cd"]),
                        textcoords="offset points", xytext=(8, 5), fontsize=9)
    ax_ma.axhline(CD_REF, color="red", linestyle="--", alpha=0.6, label=f"textbook free-stream ({CD_REF})")
    ax_ma.axhspan(CD_REF * 0.9, CD_REF * 1.1, color="red", alpha=0.08, label=r"$\pm$10% gate")
    ax_ma.set_xlabel("Mach number  Ma = U/cs")
    ax_ma.set_ylabel(r"$C_d$  (time-averaged)")
    ax_ma.set_title("Compressibility convergence  (D=20, Re=100)")
    ax_ma.legend(loc="best", fontsize=9)
    ax_ma.grid(alpha=0.3)

    # Right panel: Cd vs D at Ma=0.087 (subset C, D, E)
    d_subset = df[df["U"] == 0.05].sort_values("D")
    ax_d.plot(d_subset["D"], d_subset["Cd"], "o-",
              color="tab:green", linewidth=2, markersize=9, label="this sweep")
    for _, row in d_subset.iterrows():
        ax_d.annotate(f"{row['Cd']:.2f}", (row["D"], row["Cd"]),
                       textcoords="offset points", xytext=(8, 5), fontsize=9)
    ax_d.axhline(CD_REF, color="red", linestyle="--", alpha=0.6, label=f"textbook free-stream ({CD_REF})")
    ax_d.axhspan(CD_REF * 0.9, CD_REF * 1.1, color="red", alpha=0.08, label=r"$\pm$10% gate")
    ax_d.set_xlabel("Cylinder diameter  D  (lattice cells)")
    ax_d.set_ylabel(r"$C_d$  (time-averaged)")
    ax_d.set_title("Discretization convergence  (Ma=0.087, Re=100)")
    ax_d.legend(loc="best", fontsize=9)
    ax_d.grid(alpha=0.3)

    fig.suptitle("LBM cylinder Cd convergence study  (Re=100, periodic top/bottom)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(png_path, dpi=120, bbox_inches="tight")
    print(f"Saved PNG: {png_path.relative_to(out_dir.parent)}")

    plt.show()


if __name__ == "__main__":
    main()
