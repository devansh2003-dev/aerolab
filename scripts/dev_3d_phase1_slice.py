"""Phase 1 slice viewer: save mid-plane PNGs of a TGV run.

The Phase 1 viewer deliberately reuses the 2D matplotlib pipeline:
extract the z-mid plane of the 3D velocity field and pcolormesh it.
This is the cheapest debugging window into the 3D solver. The fancier
Q-criterion isosurface viewer arrives in Phase 2.

Run:

    .venv311\\Scripts\\python.exe scripts\\dev_3d_phase1_slice.py

Writes two PNGs to ``data/phase1/``:

  * ``tgv_slice_initial.png`` -- mid-z slice of |u| at t = 0
  * ``tgv_slice_final.png``   -- mid-z slice of |u| after n_steps

You should see the two clean sinusoidal vortices the analytic TGV
init lays down, smoothly decaying in amplitude after a few hundred
steps. If they get jagged or asymmetric, the kernel has a defect.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ_ROOT))

from src.lbm_3d import (  # noqa: E402
    LATTICE_VELOCITIES_3D,
    LATTICE_WEIGHTS_3D,
    OPPOSITE_3D,
    macroscopic_3d,
)
from src.lbm_3d_trt import init_tgv, omegas_for_trt, trt_periodic_step  # noqa: E402


def save_slice(u: np.ndarray, path: Path, title: str, vmax: float):
    """Mid-z slice of |u|. Plain pcolormesh; the body of the project
    uses matplotlib for the same job in 2D."""
    Nz = u.shape[3]
    mid_z = Nz // 2
    speed = np.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)[:, :, mid_z]
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    im = ax.pcolormesh(speed.T, cmap="viridis", vmin=0, vmax=vmax,
                       shading="auto")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="|u| (lattice units)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> int:
    N = 48
    U = 0.04
    nu = 0.01
    n_steps = 600
    out_dir = _PROJ_ROOT / "data" / "phase1"
    out_dir.mkdir(parents=True, exist_ok=True)

    f = init_tgv(N, U, dtype=np.float32)
    rho0, ux0, uy0, uz0 = macroscopic_3d(f)
    u0 = np.stack([ux0, uy0, uz0])
    vmax = float(np.sqrt((u0 ** 2).sum(axis=0)).max())
    save_slice(u0, out_dir / "tgv_slice_initial.png",
               f"TGV initial, N={N}, U={U}", vmax)
    print(f"[slice] wrote {out_dir / 'tgv_slice_initial.png'}")

    s_plus, s_minus = omegas_for_trt(nu)
    s_plus = np.float32(s_plus)
    s_minus = np.float32(s_minus)
    vel = LATTICE_VELOCITIES_3D.astype(np.int32)
    weights = LATTICE_WEIGHTS_3D.astype(np.float32)
    opp = OPPOSITE_3D.astype(np.int32)

    f_next = f.copy()
    for step in range(n_steps):
        trt_periodic_step(f, f_next, s_plus, s_minus, vel, weights, opp)
        f, f_next = f_next, f
    rho1, ux1, uy1, uz1 = macroscopic_3d(f)
    u1 = np.stack([ux1, uy1, uz1])
    # Re-use the initial vmax so the colour-scale comparison is honest.
    save_slice(u1, out_dir / "tgv_slice_final.png",
               f"TGV after {n_steps} steps (nu={nu})", vmax)
    print(f"[slice] wrote {out_dir / 'tgv_slice_final.png'}")
    print(f"[slice] KE ratio (final / initial) = "
          f"{float((u1 ** 2).sum() / (u0 ** 2).sum()):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
