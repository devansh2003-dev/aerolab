"""Bake a steady-state 3D flow field for the consumer gallery (D-10).

Usage
-----

    python scripts/bake_3d_field.py --preset sphere_re40
    python scripts/bake_3d_field.py --preset sphere_re40 --out data/baked/

Writes a compressed .npz to ``data/baked/<preset_name>.npz`` (or the
``--out`` directory). The file holds ``(rho, ux, uy, uz, body)`` as
float16 plus a JSON manifest with the bake parameters and a canonical
hash.

The consumer Streamlit page (next phase) will list the available .npz
files and replay each via the existing smoke-particle viz -- no kernel
recompute, no Cloud-side compute pressure.

The preset registry below is the source of truth for what we bake.
``sphere_re40`` is the conservative starter (clean steady wake, ~12 s
on laptop CPU, default ``outflow_scheme="guo"``). ``sphere_re100``
goes to Re ~ 100 -- a regime where the old Guo-NEEM-only outflow
went populations-negative around step 500-800; it stays clean here
because the preset opts into ``outflow_scheme="regularised"``
(Latt-Chopard 2008), which filters the non-hydrodynamic ghost moments
that drive the instability. See ``src/lbm_3d.py:apply_regularised_outflow``
for the projection and ``tests/test_lbm_3d_outflow.py`` for the gate.

To add a preset, add a new entry to ``PRESETS`` below.

**Re-bake determinism note** (matters for the .gitignore exception that
lets us ship pre-baked artifacts to Cloud). The hash in the manifest is
content-deterministic -- two runs at the same parameters produce
byte-identical hashes. The .npz file itself is NOT byte-identical
across re-bakes because (1) ``save_baked_field`` stamps
``ts_baked = datetime.now(UTC).isoformat()`` into the manifest, and
(2) ``np.savez_compressed`` embeds zip-archive timestamps. So a fresh
``python scripts/bake_3d_field.py --preset sphere_re40`` will produce
a working file with the same hash as the committed one but different
on-disk bytes -- git diff will flag it as modified. Revert with
``git restore data/baked/sphere_re40.npz`` if the re-bake wasn't
meant to update the shipped artifact.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to sys.path so `from src.foo import ...` works when
# this script is run from anywhere.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.baked_fields import save_baked_field  # noqa: E402
from src.lbm_3d_bouzidi import (  # noqa: E402
    make_cylinder_mask,
    make_sphere_mask,
    sphere_wall_links,
)
from src.lbm_3d_trt import run_channel_smoke_trt  # noqa: E402
from src.voxelize import voxel_wall_links  # noqa: E402

# ---------------------------------------------------------------------------
# Preset registry
#
# Each entry is a dict with the parameters needed to drive the bake.
# Velocity / viscosity are tuned so the resulting Re = u_in * D / nu is
# in a regime that gives visible Q-criterion structure on the chosen
# grid without exceeding Ma ~ 0.1 (BGK / TRT stability).
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict[str, Any]] = {
    # Conservative-Mach starter scenes. Ma = u_in * sqrt(3) ~ 0.07
    # at u_in = 0.04 -- well inside the BGK / TRT stability envelope.
    # Higher-Re presets need larger grids to keep blockage < ~0.30 and
    # Mach < ~0.10 simultaneously; the dev-grade scenes below trade Re
    # for snap (10-20 s on a laptop).
    "sphere_re40": {
        "body_type": "sphere",
        "Nx": 64, "Ny": 32, "Nz": 32,
        "body_params": {"cx": 16.0, "cy": 16.0, "cz": 16.0, "R": 4.0},
        "u_in": 0.04,
        "nu": 0.008,                     # Re = 0.04 * 8 / 0.008 = 40
        "n_steps": 800,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "guo",
    },
    # sphere_re100: same body geometry as sphere_re40 but 2.5x the
    # Reynolds number on a 1.5x grid. The old Guo-NEEM-only outflow
    # went populations-negative on this config by ~step 500-800; the
    # regularised outflow holds the field steady through step 800
    # (u_peak ~ 0.044, mass drift < 0.25%). A separate late-onset
    # instability sets in between step 800-1000 -- own task; for now
    # the bake stops at n_steps=800. That is ~2.7 advective times
    # (D/u_in = 300 lattice steps), so the wake is past startup but
    # NOT yet the asymptotic steady Re=100 wake -- it is a transient
    # snapshot with a recognisable recirculation downstream of the
    # body. Cost ~45 s on laptop CPU.
    "sphere_re100": {
        "body_type": "sphere",
        "Nx": 96, "Ny": 48, "Nz": 48,
        "body_params": {"cx": 24.0, "cy": 24.0, "cz": 24.0, "R": 6.0},
        "u_in": 0.04,
        "nu": 0.0048,                    # Re = 0.04 * 12 / 0.0048 = 100
        "n_steps": 800,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    # cylinder_re100: 3D extrude of the validated 2D cylinder along
    # the spanwise (z) axis. Wake should be the classic steady twin
    # vortices below Re~47 and onset of von Karman shedding above --
    # at Re=100 the shedding is mature in 2D. In 3D the same Re=100
    # is just past the onset of 3D wake instabilities (mode A around
    # Re=190 per Williamson 1996), so this scene shows the upstream
    # span-coherent wake -- visually the cleanest 3D cylinder shot.
    # Wall links use the smoothed-mask q approach from voxelize.py
    # because the analytic cylinder q hasn't been derived yet -- the
    # voxel approach is good to ~0.05 voxel accuracy, plenty for viz.
    "cylinder_re100": {
        "body_type": "cylinder",
        "Nx": 96, "Ny": 48, "Nz": 48,
        # Cylinder centre at (24, 24) in (x, y), radius 6, spans full z.
        # D = 12, blockage = D/Ny = 0.25 (matches sphere_re100).
        "body_params": {"cx": 24.0, "cy": 24.0, "R": 6.0},
        "u_in": 0.04,
        "nu": 0.0048,                    # Re = 0.04 * 12 / 0.0048 = 100
        "n_steps": 800,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
}


def _build_body(
    body_type: str,
    Nx: int, Ny: int, Nz: int,
    body_params: dict[str, Any],
):
    """Return (body_mask, wall_links_or_None) for a preset's body."""
    if body_type == "sphere":
        cx = float(body_params["cx"])
        cy = float(body_params["cy"])
        cz = float(body_params["cz"])
        R = float(body_params["R"])
        mask = make_sphere_mask(Nx, Ny, Nz, cx, cy, cz, R)
        wall_links = sphere_wall_links(Nx, Ny, Nz, cx, cy, cz, R)
        return mask, wall_links
    if body_type == "cylinder":
        # Spanwise cylinder along z. Wall links via the smoothed-mask
        # approach (voxelize.voxel_wall_links) -- accurate to ~0.05
        # voxels, no analytic q derivation required. The mask is
        # binary so the Gaussian smooth picks up the cylinder surface
        # cleanly.
        cx = float(body_params["cx"])
        cy = float(body_params["cy"])
        R = float(body_params["R"])
        mask = make_cylinder_mask(Nx, Ny, Nz, cx, cy, R)
        wall_links = voxel_wall_links(mask)
        return mask, wall_links
    raise ValueError(f"unknown body_type: {body_type!r}")


def bake(preset_name: str, out_dir: Path) -> Path:
    """Run the preset's simulation and save the result.

    Returns the path of the .npz that was written.
    """
    if preset_name not in PRESETS:
        raise KeyError(
            f"unknown preset {preset_name!r}. "
            f"Available: {sorted(PRESETS)}"
        )
    params = PRESETS[preset_name]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{preset_name}.npz"

    body, wall_links = _build_body(
        params["body_type"],
        params["Nx"], params["Ny"], params["Nz"],
        params["body_params"],
    )

    # Bouzidi correction is optional -- the preset opts in by default
    # because every shipped preset wants the validated boundary path.
    use_bouzidi = bool(params.get("use_bouzidi", True))

    t0 = time.time()
    rho, ux, uy, uz, diag = run_channel_smoke_trt(
        Nx=params["Nx"], Ny=params["Ny"], Nz=params["Nz"],
        u_in=params["u_in"], nu=params["nu"],
        n_steps=params["n_steps"],
        body=body,
        wall_links=wall_links if use_bouzidi else None,
        use_guo_neem=params["use_guo_neem"],
        rho_outflow=params.get("rho_outflow", 1.0),
        outflow_scheme=params.get("outflow_scheme", "guo"),
        scheme=params["scheme"],
    )
    elapsed = time.time() - t0

    # bake_params is the canonical-hash input. We strip diag (which
    # has float values that depend on the run) and ts (added by save).
    bake_params = dict(params)
    bake_params["solver_diag"] = {
        "mass_drift_rel": float(diag["mass_drift_rel"]),
        "u_peak": float(diag["u_peak"]),
        "u_mean": float(diag["u_mean"]),
        "centerline_ratio": float(diag["centerline_ratio"]),
    }
    bake_params["bake_wall_time_s"] = round(float(elapsed), 3)

    meta = save_baked_field(
        out_path,
        rho=rho, ux=ux, uy=uy, uz=uz, body=body,
        preset_name=preset_name,
        bake_params=bake_params,
    )

    return out_path, meta, elapsed


def _print_summary(out_path: Path, meta: dict[str, Any], elapsed: float) -> None:
    """Pretty-print the bake outcome to stdout."""
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print()
    print(f"  baked: {out_path}")
    print(f"  size:  {size_mb:.2f} MB (.npz, float16 storage)")
    print(f"  time:  {elapsed:.1f} s")
    print(f"  grid:  {meta['Nx']} x {meta['Ny']} x {meta['Nz']}")
    sd = meta.get("solver_diag", {})
    if sd:
        print(
            f"  diag:  u_peak={sd.get('u_peak', 0):.5f} "
            f"u_mean={sd.get('u_mean', 0):.5f} "
            f"mass_drift={sd.get('mass_drift_rel', 0):.4f}"
        )
    print(f"  hash:  {meta['hash'][:16]}...")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--preset", required=True,
        choices=sorted(PRESETS),
        help="Preset to bake (registry: scripts/bake_3d_field.py).",
    )
    p.add_argument(
        "--out", default="data/baked",
        help="Output directory (default: data/baked).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    out_dir = Path(args.out)
    print(f"baking {args.preset!r} -> {out_dir}/{args.preset}.npz ...")
    out_path, meta, elapsed = bake(args.preset, out_dir)
    _print_summary(out_path, meta, elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
