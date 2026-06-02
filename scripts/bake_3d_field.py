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
    make_cube_mask,
    make_cylinder_mask,
    make_naca_mask,
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
        # R bumped 6 -> 8 (2026-05-29 round 2 user note: bodies still
        # too small, can't see streamlines wrapping). New blockage =
        # D/Ny = 16/32 = 50 %. nu rescaled to keep Re = 40 exactly.
        # At Re=40 with 50% blockage the wake is steady and the solver
        # is far from any stability boundary (tau = 3*0.016+0.5 = 0.548).
        "body_params": {"cx": 16.0, "cy": 16.0, "cz": 16.0, "R": 8.0},
        "u_in": 0.04,
        "nu": 0.016,                     # Re = 0.04 * 16 / 0.016 = 40
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
        # R bumped 7 -> 10. Blockage = D/Ny = 20/48 = 42 %.  Bigger
        # body intensifies the wake and pushes stability; we stay at
        # 800 steps as before (the late-onset divergence past ~800
        # is unaffected by R, the regularised outflow holds through
        # step 800). nu rescaled to keep Re = 100 exactly.
        "body_params": {"cx": 24.0, "cy": 24.0, "cz": 24.0, "R": 10.0},
        "u_in": 0.04,
        "nu": 0.008,                     # Re = 0.04 * 20 / 0.008 = 100
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
        # R bumped 7 -> 10 (matches sphere_re100). Blockage = 20/48 =
        # 42 %. Cylinder centre at (24, 24) in (x, y), spans full z.
        "body_params": {"cx": 24.0, "cy": 24.0, "R": 10.0},
        "u_in": 0.04,
        "nu": 0.008,                     # Re = 0.04 * 20 / 0.008 = 100
        "n_steps": 800,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    # ---- Additional Re bands (2026-05-29) ---------------------------
    # The 3D gallery exposes a Reynolds slider with two snap-to bakes
    # per shape (Re = 40 and 100). Re = 200 was attempted but the
    # 96x48x48 BGK/TRT bake diverges to NaN (tau ~ 0.512, on the edge
    # of stability with 42 % blockage) -- a larger grid or lower u_in
    # would be needed to push past Re = 100, and is left as follow-up.
    # All shipped bakes use TRT collision and the regularised outflow
    # that held cylinder_re100 stable.
    "cylinder_re40": {
        "body_type": "cylinder",
        "Nx": 64, "Ny": 32, "Nz": 32,
        "body_params": {"cx": 16.0, "cy": 16.0, "R": 8.0},
        "u_in": 0.04,
        "nu": 0.016,                     # Re = 0.04 * 16 / 0.016 = 40
        "n_steps": 800,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "cube_re100": {
        "body_type": "cube",
        "Nx": 96, "Ny": 48, "Nz": 48,
        # half_extent 9 -> cube spans 18 lattice units on each side.
        # Blockage 18/48 = 37 %. Re = u_in * (2 * half_extent) / nu.
        "body_params": {"cx": 24.0, "cy": 24.0, "cz": 24.0, "half_extent": 9.0},
        "u_in": 0.04,
        "nu": 0.0072,                    # Re = 0.04 * 18 / 0.0072 = 100
        "n_steps": 800,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "cube_re40": {
        "body_type": "cube",
        "Nx": 64, "Ny": 32, "Nz": 32,
        "body_params": {"cx": 16.0, "cy": 16.0, "cz": 16.0, "half_extent": 7.0},
        "u_in": 0.04,
        "nu": 0.014,                     # Re = 0.04 * 14 / 0.014 = 40
        "n_steps": 800,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    # NACA 4-digit wings, span axis = y (horizontal wing --
    # the conventional aircraft-photo orientation). AoA bands
    # at {-30, -15, -5, 0, +5, +15, +30} deg per Re band so the
    # sidebar AoA slider can land on a real baked snapshot within
    # ~5 deg of any 1-deg slider position. NACA 0012 (symmetric)
    # and NACA 4412 (cambered: 4 % camber at 40 % chord, 12 %
    # thickness). chord_offset is the z coordinate of the chord
    # midline; with span_axis='y' the airfoil cross-section lives
    # in (x, z) and the wing extrudes side-to-side in y.
    "naca0012_aoa-30_re40": {
        "body_type": "naca",
        "Nx": 80, "Ny": 40, "Nz": 32,
        "body_params": {
            "x_le": 12.0, "chord_offset": 16.0, "chord": 20.0,
            "m": 0.0, "p": 0.0, "thickness": 0.12,
            "aoa_deg": -30.0,
            "span_axis": "y",
        },
        "u_in": 0.04,
        "nu": 0.02,
        "n_steps": 8000,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "naca0012_aoa-15_re40": {
        "body_type": "naca",
        "Nx": 80, "Ny": 40, "Nz": 32,
        "body_params": {
            "x_le": 12.0, "chord_offset": 16.0, "chord": 20.0,
            "m": 0.0, "p": 0.0, "thickness": 0.12,
            "aoa_deg": -15.0,
            "span_axis": "y",
        },
        "u_in": 0.04,
        "nu": 0.02,
        "n_steps": 800,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "naca0012_aoa-5_re40": {
        "body_type": "naca",
        "Nx": 80, "Ny": 40, "Nz": 32,
        "body_params": {
            "x_le": 12.0, "chord_offset": 16.0, "chord": 20.0,
            "m": 0.0, "p": 0.0, "thickness": 0.12,
            "aoa_deg": -5.0,
            "span_axis": "y",
        },
        "u_in": 0.04,
        "nu": 0.02,
        "n_steps": 800,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "naca0012_re40": {
        "body_type": "naca",
        "Nx": 80, "Ny": 40, "Nz": 32,
        "body_params": {
            "x_le": 12.0, "chord_offset": 16.0, "chord": 20.0,
            "m": 0.0, "p": 0.0, "thickness": 0.12,
            "aoa_deg": 0.0,
            "span_axis": "y",
        },
        "u_in": 0.04,
        "nu": 0.02,
        "n_steps": 800,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "naca0012_aoa5_re40": {
        "body_type": "naca",
        "Nx": 80, "Ny": 40, "Nz": 32,
        "body_params": {
            "x_le": 12.0, "chord_offset": 16.0, "chord": 20.0,
            "m": 0.0, "p": 0.0, "thickness": 0.12,
            "aoa_deg": 5.0,
            "span_axis": "y",
        },
        "u_in": 0.04,
        "nu": 0.02,
        "n_steps": 800,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "naca0012_aoa15_re40": {
        "body_type": "naca",
        "Nx": 80, "Ny": 40, "Nz": 32,
        "body_params": {
            "x_le": 12.0, "chord_offset": 16.0, "chord": 20.0,
            "m": 0.0, "p": 0.0, "thickness": 0.12,
            "aoa_deg": 15.0,
            "span_axis": "y",
        },
        "u_in": 0.04,
        "nu": 0.02,
        "n_steps": 800,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "naca0012_aoa30_re40": {
        "body_type": "naca",
        "Nx": 80, "Ny": 40, "Nz": 32,
        "body_params": {
            "x_le": 12.0, "chord_offset": 16.0, "chord": 20.0,
            "m": 0.0, "p": 0.0, "thickness": 0.12,
            "aoa_deg": 30.0,
            "span_axis": "y",
        },
        "u_in": 0.04,
        "nu": 0.02,
        "n_steps": 8000,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "naca0012_aoa-30_re100": {
        "body_type": "naca",
        "Nx": 96, "Ny": 48, "Nz": 32,
        "body_params": {
            "x_le": 14.0, "chord_offset": 16.0, "chord": 24.0,
            "m": 0.0, "p": 0.0, "thickness": 0.12,
            "aoa_deg": -30.0,
            "span_axis": "y",
        },
        "u_in": 0.04,
        "nu": 0.0096,
        "n_steps": 12000,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "naca0012_aoa-15_re100": {
        "body_type": "naca",
        "Nx": 96, "Ny": 48, "Nz": 32,
        "body_params": {
            "x_le": 14.0, "chord_offset": 16.0, "chord": 24.0,
            "m": 0.0, "p": 0.0, "thickness": 0.12,
            "aoa_deg": -15.0,
            "span_axis": "y",
        },
        "u_in": 0.04,
        "nu": 0.0096,
        "n_steps": 800,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "naca0012_aoa-5_re100": {
        "body_type": "naca",
        "Nx": 96, "Ny": 48, "Nz": 32,
        "body_params": {
            "x_le": 14.0, "chord_offset": 16.0, "chord": 24.0,
            "m": 0.0, "p": 0.0, "thickness": 0.12,
            "aoa_deg": -5.0,
            "span_axis": "y",
        },
        "u_in": 0.04,
        "nu": 0.0096,
        "n_steps": 800,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "naca0012_re100": {
        "body_type": "naca",
        "Nx": 96, "Ny": 48, "Nz": 32,
        "body_params": {
            "x_le": 14.0, "chord_offset": 16.0, "chord": 24.0,
            "m": 0.0, "p": 0.0, "thickness": 0.12,
            "aoa_deg": 0.0,
            "span_axis": "y",
        },
        "u_in": 0.04,
        "nu": 0.0096,
        "n_steps": 800,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "naca0012_aoa5_re100": {
        "body_type": "naca",
        "Nx": 96, "Ny": 48, "Nz": 32,
        "body_params": {
            "x_le": 14.0, "chord_offset": 16.0, "chord": 24.0,
            "m": 0.0, "p": 0.0, "thickness": 0.12,
            "aoa_deg": 5.0,
            "span_axis": "y",
        },
        "u_in": 0.04,
        "nu": 0.0096,
        "n_steps": 800,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "naca0012_aoa15_re100": {
        "body_type": "naca",
        "Nx": 96, "Ny": 48, "Nz": 32,
        "body_params": {
            "x_le": 14.0, "chord_offset": 16.0, "chord": 24.0,
            "m": 0.0, "p": 0.0, "thickness": 0.12,
            "aoa_deg": 15.0,
            "span_axis": "y",
        },
        "u_in": 0.04,
        "nu": 0.0096,
        "n_steps": 800,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "naca0012_aoa30_re100": {
        "body_type": "naca",
        "Nx": 96, "Ny": 48, "Nz": 32,
        "body_params": {
            "x_le": 14.0, "chord_offset": 16.0, "chord": 24.0,
            "m": 0.0, "p": 0.0, "thickness": 0.12,
            "aoa_deg": 30.0,
            "span_axis": "y",
        },
        "u_in": 0.04,
        "nu": 0.0096,
        "n_steps": 12000,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "naca4412_aoa-30_re40": {
        "body_type": "naca",
        "Nx": 80, "Ny": 40, "Nz": 32,
        "body_params": {
            "x_le": 12.0, "chord_offset": 16.0, "chord": 20.0,
            "m": 0.04, "p": 0.4, "thickness": 0.12,
            "aoa_deg": -30.0,
            "span_axis": "y",
        },
        "u_in": 0.04,
        "nu": 0.02,
        "n_steps": 8000,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "naca4412_aoa-15_re40": {
        "body_type": "naca",
        "Nx": 80, "Ny": 40, "Nz": 32,
        "body_params": {
            "x_le": 12.0, "chord_offset": 16.0, "chord": 20.0,
            "m": 0.04, "p": 0.4, "thickness": 0.12,
            "aoa_deg": -15.0,
            "span_axis": "y",
        },
        "u_in": 0.04,
        "nu": 0.02,
        "n_steps": 800,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "naca4412_aoa-5_re40": {
        "body_type": "naca",
        "Nx": 80, "Ny": 40, "Nz": 32,
        "body_params": {
            "x_le": 12.0, "chord_offset": 16.0, "chord": 20.0,
            "m": 0.04, "p": 0.4, "thickness": 0.12,
            "aoa_deg": -5.0,
            "span_axis": "y",
        },
        "u_in": 0.04,
        "nu": 0.02,
        "n_steps": 800,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "naca4412_re40": {
        "body_type": "naca",
        "Nx": 80, "Ny": 40, "Nz": 32,
        "body_params": {
            "x_le": 12.0, "chord_offset": 16.0, "chord": 20.0,
            "m": 0.04, "p": 0.4, "thickness": 0.12,
            "aoa_deg": 0.0,
            "span_axis": "y",
        },
        "u_in": 0.04,
        "nu": 0.02,
        "n_steps": 800,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "naca4412_aoa5_re40": {
        "body_type": "naca",
        "Nx": 80, "Ny": 40, "Nz": 32,
        "body_params": {
            "x_le": 12.0, "chord_offset": 16.0, "chord": 20.0,
            "m": 0.04, "p": 0.4, "thickness": 0.12,
            "aoa_deg": 5.0,
            "span_axis": "y",
        },
        "u_in": 0.04,
        "nu": 0.02,
        "n_steps": 800,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "naca4412_aoa15_re40": {
        "body_type": "naca",
        "Nx": 80, "Ny": 40, "Nz": 32,
        "body_params": {
            "x_le": 12.0, "chord_offset": 16.0, "chord": 20.0,
            "m": 0.04, "p": 0.4, "thickness": 0.12,
            "aoa_deg": 15.0,
            "span_axis": "y",
        },
        "u_in": 0.04,
        "nu": 0.02,
        "n_steps": 800,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "naca4412_aoa30_re40": {
        "body_type": "naca",
        "Nx": 80, "Ny": 40, "Nz": 32,
        "body_params": {
            "x_le": 12.0, "chord_offset": 16.0, "chord": 20.0,
            "m": 0.04, "p": 0.4, "thickness": 0.12,
            "aoa_deg": 30.0,
            "span_axis": "y",
        },
        "u_in": 0.04,
        "nu": 0.02,
        "n_steps": 8000,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "naca4412_aoa-30_re100": {
        "body_type": "naca",
        "Nx": 96, "Ny": 48, "Nz": 32,
        "body_params": {
            "x_le": 14.0, "chord_offset": 16.0, "chord": 24.0,
            "m": 0.04, "p": 0.4, "thickness": 0.12,
            "aoa_deg": -30.0,
            "span_axis": "y",
        },
        "u_in": 0.04,
        "nu": 0.0096,
        "n_steps": 12000,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "naca4412_aoa-15_re100": {
        "body_type": "naca",
        "Nx": 96, "Ny": 48, "Nz": 32,
        "body_params": {
            "x_le": 14.0, "chord_offset": 16.0, "chord": 24.0,
            "m": 0.04, "p": 0.4, "thickness": 0.12,
            "aoa_deg": -15.0,
            "span_axis": "y",
        },
        "u_in": 0.04,
        "nu": 0.0096,
        "n_steps": 800,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "naca4412_aoa-5_re100": {
        "body_type": "naca",
        "Nx": 96, "Ny": 48, "Nz": 32,
        "body_params": {
            "x_le": 14.0, "chord_offset": 16.0, "chord": 24.0,
            "m": 0.04, "p": 0.4, "thickness": 0.12,
            "aoa_deg": -5.0,
            "span_axis": "y",
        },
        "u_in": 0.04,
        "nu": 0.0096,
        "n_steps": 800,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "naca4412_re100": {
        "body_type": "naca",
        "Nx": 96, "Ny": 48, "Nz": 32,
        "body_params": {
            "x_le": 14.0, "chord_offset": 16.0, "chord": 24.0,
            "m": 0.04, "p": 0.4, "thickness": 0.12,
            "aoa_deg": 0.0,
            "span_axis": "y",
        },
        "u_in": 0.04,
        "nu": 0.0096,
        "n_steps": 800,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "naca4412_aoa5_re100": {
        "body_type": "naca",
        "Nx": 96, "Ny": 48, "Nz": 32,
        "body_params": {
            "x_le": 14.0, "chord_offset": 16.0, "chord": 24.0,
            "m": 0.04, "p": 0.4, "thickness": 0.12,
            "aoa_deg": 5.0,
            "span_axis": "y",
        },
        "u_in": 0.04,
        "nu": 0.0096,
        "n_steps": 800,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "naca4412_aoa15_re100": {
        "body_type": "naca",
        "Nx": 96, "Ny": 48, "Nz": 32,
        "body_params": {
            "x_le": 14.0, "chord_offset": 16.0, "chord": 24.0,
            "m": 0.04, "p": 0.4, "thickness": 0.12,
            "aoa_deg": 15.0,
            "span_axis": "y",
        },
        "u_in": 0.04,
        "nu": 0.0096,
        "n_steps": 800,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "naca4412_aoa30_re100": {
        "body_type": "naca",
        "Nx": 96, "Ny": 48, "Nz": 32,
        "body_params": {
            "x_le": 14.0, "chord_offset": 16.0, "chord": 24.0,
            "m": 0.04, "p": 0.4, "thickness": 0.12,
            "aoa_deg": 30.0,
            "span_axis": "y",
        },
        "u_in": 0.04,
        "nu": 0.0096,
        "n_steps": 12000,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    # === AoA = +/- 45 deg "stall" presets (added 2026-06-02) ===
    # User request: "I want to see the wing stall at like 45 deg." Thin
    # airfoils typically stall around 12 - 15 deg; 45 deg is deep stall,
    # a separated free-shear-layer regime that LBM at Re=40-100 will
    # render as a steady (Re=40) or weakly unsteady (Re=100) wake. The
    # visual point is the massive wake separation behind the wing -- not
    # a quantitative aerodynamic claim, since the validated airfoil
    # regime is attached flow (NeuralFoil, AoA up to ~ 10 deg).
    # Ny is bumped one notch above the 30 deg presets so the projected
    # vertical extent of the wing fits with margin.
    "naca0012_aoa45_re40": {
        "body_type": "naca",
        "Nx": 80, "Ny": 48, "Nz": 32,
        "body_params": {
            "x_le": 12.0, "chord_offset": 16.0, "chord": 20.0,
            "m": 0.0, "p": 0.0, "thickness": 0.12,
            "aoa_deg": 45.0,
            "span_axis": "y",
        },
        "u_in": 0.04, "nu": 0.02, "n_steps": 8000,
        "scheme": "trt", "use_guo_neem": True, "use_bouzidi": True,
        "rho_outflow": 1.0, "outflow_scheme": "regularised",
    },
    "naca0012_aoa-45_re40": {
        "body_type": "naca",
        "Nx": 80, "Ny": 48, "Nz": 32,
        "body_params": {
            "x_le": 12.0, "chord_offset": 16.0, "chord": 20.0,
            "m": 0.0, "p": 0.0, "thickness": 0.12,
            "aoa_deg": -45.0,
            "span_axis": "y",
        },
        "u_in": 0.04, "nu": 0.02, "n_steps": 8000,
        "scheme": "trt", "use_guo_neem": True, "use_bouzidi": True,
        "rho_outflow": 1.0, "outflow_scheme": "regularised",
    },
    "naca0012_aoa45_re100": {
        "body_type": "naca",
        "Nx": 96, "Ny": 56, "Nz": 32,
        "body_params": {
            "x_le": 14.0, "chord_offset": 16.0, "chord": 24.0,
            "m": 0.0, "p": 0.0, "thickness": 0.12,
            "aoa_deg": 45.0,
            "span_axis": "y",
        },
        "u_in": 0.04, "nu": 0.0096, "n_steps": 12000,
        "scheme": "trt", "use_guo_neem": True, "use_bouzidi": True,
        "rho_outflow": 1.0, "outflow_scheme": "regularised",
    },
    "naca0012_aoa-45_re100": {
        "body_type": "naca",
        "Nx": 96, "Ny": 56, "Nz": 32,
        "body_params": {
            "x_le": 14.0, "chord_offset": 16.0, "chord": 24.0,
            "m": 0.0, "p": 0.0, "thickness": 0.12,
            "aoa_deg": -45.0,
            "span_axis": "y",
        },
        "u_in": 0.04, "nu": 0.0096, "n_steps": 12000,
        "scheme": "trt", "use_guo_neem": True, "use_bouzidi": True,
        "rho_outflow": 1.0, "outflow_scheme": "regularised",
    },
    "naca4412_aoa45_re40": {
        "body_type": "naca",
        "Nx": 80, "Ny": 48, "Nz": 32,
        "body_params": {
            "x_le": 12.0, "chord_offset": 16.0, "chord": 20.0,
            "m": 0.04, "p": 0.4, "thickness": 0.12,
            "aoa_deg": 45.0,
            "span_axis": "y",
        },
        "u_in": 0.04, "nu": 0.02, "n_steps": 8000,
        "scheme": "trt", "use_guo_neem": True, "use_bouzidi": True,
        "rho_outflow": 1.0, "outflow_scheme": "regularised",
    },
    "naca4412_aoa-45_re40": {
        "body_type": "naca",
        "Nx": 80, "Ny": 48, "Nz": 32,
        "body_params": {
            "x_le": 12.0, "chord_offset": 16.0, "chord": 20.0,
            "m": 0.04, "p": 0.4, "thickness": 0.12,
            "aoa_deg": -45.0,
            "span_axis": "y",
        },
        "u_in": 0.04, "nu": 0.02, "n_steps": 8000,
        "scheme": "trt", "use_guo_neem": True, "use_bouzidi": True,
        "rho_outflow": 1.0, "outflow_scheme": "regularised",
    },
    "naca4412_aoa45_re100": {
        "body_type": "naca",
        "Nx": 96, "Ny": 56, "Nz": 32,
        "body_params": {
            "x_le": 14.0, "chord_offset": 16.0, "chord": 24.0,
            "m": 0.04, "p": 0.4, "thickness": 0.12,
            "aoa_deg": 45.0,
            "span_axis": "y",
        },
        "u_in": 0.04, "nu": 0.0096, "n_steps": 12000,
        "scheme": "trt", "use_guo_neem": True, "use_bouzidi": True,
        "rho_outflow": 1.0, "outflow_scheme": "regularised",
    },
    "naca4412_aoa-45_re100": {
        "body_type": "naca",
        "Nx": 96, "Ny": 56, "Nz": 32,
        "body_params": {
            "x_le": 14.0, "chord_offset": 16.0, "chord": 24.0,
            "m": 0.04, "p": 0.4, "thickness": 0.12,
            "aoa_deg": -45.0,
            "span_axis": "y",
        },
        "u_in": 0.04, "nu": 0.0096, "n_steps": 12000,
        "scheme": "trt", "use_guo_neem": True, "use_bouzidi": True,
        "rho_outflow": 1.0, "outflow_scheme": "regularised",
    },
    # Cube with AoA rotation about the y axis. AoA=0 is the existing
    # ``cube_re{N}`` preset. AoA=45 reads as the classic 3D diamond
    # (corners pointing at the flow); 90 deg symmetry means the band
    # {0, 15, 30, 45} covers every unique orientation.
    "cube_aoa15_re40": {
        "body_type": "cube",
        "Nx": 64, "Ny": 32, "Nz": 32,
        "body_params": {"cx": 16.0, "cy": 16.0, "cz": 16.0, "half_extent": 7.0, "aoa_deg": 15.0},
        "u_in": 0.04,
        "nu": 0.014,
        "n_steps": 800,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "cube_aoa30_re40": {
        "body_type": "cube",
        "Nx": 64, "Ny": 32, "Nz": 32,
        "body_params": {"cx": 16.0, "cy": 16.0, "cz": 16.0, "half_extent": 7.0, "aoa_deg": 30.0},
        "u_in": 0.04,
        "nu": 0.014,
        "n_steps": 800,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "cube_aoa45_re40": {
        "body_type": "cube",
        "Nx": 64, "Ny": 32, "Nz": 32,
        "body_params": {"cx": 16.0, "cy": 16.0, "cz": 16.0, "half_extent": 7.0, "aoa_deg": 45.0},
        "u_in": 0.04,
        "nu": 0.014,
        "n_steps": 800,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "cube_aoa15_re100": {
        "body_type": "cube",
        "Nx": 96, "Ny": 48, "Nz": 48,
        "body_params": {"cx": 24.0, "cy": 24.0, "cz": 24.0, "half_extent": 9.0, "aoa_deg": 15.0},
        "u_in": 0.04,
        "nu": 0.0072,
        "n_steps": 800,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "cube_aoa30_re100": {
        "body_type": "cube",
        "Nx": 96, "Ny": 48, "Nz": 48,
        "body_params": {"cx": 24.0, "cy": 24.0, "cz": 24.0, "half_extent": 9.0, "aoa_deg": 30.0},
        "u_in": 0.04,
        "nu": 0.0072,
        "n_steps": 800,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "cube_aoa45_re100": {
        "body_type": "cube",
        "Nx": 96, "Ny": 48, "Nz": 48,
        "body_params": {"cx": 24.0, "cy": 24.0, "cz": 24.0, "half_extent": 9.0, "aoa_deg": 45.0},
        "u_in": 0.04,
        "nu": 0.0072,
        "n_steps": 800,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "cube_aoa-15_re40": {
        "body_type": "cube",
        "Nx": 64, "Ny": 32, "Nz": 32,
        "body_params": {"cx": 16.0, "cy": 16.0, "cz": 16.0, "half_extent": 7.0, "aoa_deg": -15.0},
        "u_in": 0.04,
        "nu": 0.014,
        "n_steps": 800,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "cube_aoa-30_re40": {
        "body_type": "cube",
        "Nx": 64, "Ny": 32, "Nz": 32,
        "body_params": {"cx": 16.0, "cy": 16.0, "cz": 16.0, "half_extent": 7.0, "aoa_deg": -30.0},
        "u_in": 0.04,
        "nu": 0.014,
        "n_steps": 800,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "cube_aoa-45_re40": {
        "body_type": "cube",
        "Nx": 64, "Ny": 32, "Nz": 32,
        "body_params": {"cx": 16.0, "cy": 16.0, "cz": 16.0, "half_extent": 7.0, "aoa_deg": -45.0},
        "u_in": 0.04,
        "nu": 0.014,
        "n_steps": 800,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "cube_aoa-15_re100": {
        "body_type": "cube",
        "Nx": 96, "Ny": 48, "Nz": 48,
        "body_params": {"cx": 24.0, "cy": 24.0, "cz": 24.0, "half_extent": 9.0, "aoa_deg": -15.0},
        "u_in": 0.04,
        "nu": 0.0072,
        "n_steps": 800,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "cube_aoa-30_re100": {
        "body_type": "cube",
        "Nx": 96, "Ny": 48, "Nz": 48,
        "body_params": {"cx": 24.0, "cy": 24.0, "cz": 24.0, "half_extent": 9.0, "aoa_deg": -30.0},
        "u_in": 0.04,
        "nu": 0.0072,
        "n_steps": 800,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    "cube_aoa-45_re100": {
        "body_type": "cube",
        "Nx": 96, "Ny": 48, "Nz": 48,
        "body_params": {"cx": 24.0, "cy": 24.0, "cz": 24.0, "half_extent": 9.0, "aoa_deg": -45.0},
        "u_in": 0.04,
        "nu": 0.0072,
        "n_steps": 800,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    # ===================================================================
    # A3: extra Re bands so the gallery speed slider has more snap-points.
    # Re=20 = creeping (clearly attached), Re=200 = vortex-shedding.
    # Re=200 is close to the TRT stability boundary (tau ~ 0.51);
    # refined grids + bumped n_steps for stability margin. Scope of
    # this batch: AoA = 0, +/-30, +/-45 + bluff bodies. Lower AoAs follow.
    # ===================================================================
    "sphere_re20": {
        "body_type": "sphere",
        "Nx": 32, "Ny": 32, "Nz": 32,
        "body_params": {"cx": 16.0, "cy": 16.0, "cz": 16.0, "R": 8.0},
        "u_in": 0.04,
        "nu": 0.032,                     # Re = 0.04 * 16 / 0.032 = 20
        "n_steps": 2000,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    # NOTE: sphere_re200 / cylinder_re200 attempted on 2026-06-02 with
    # u_in=0.04, nu=0.0048, tau=0.5144 but the sphere diverged to u_peak=NaN
    # at ~37% blockage. The wing pilot at the same nu survived because the
    # projected cross-section was much thinner. Conclusion: TRT at
    # Re=200 + 30%+ blockage is outside the stable envelope on the
    # consumer-mode gallery grid. The wing Re=200 presets stay (they're
    # the user-visible value); bluff bodies stop at Re=100. To revive,
    # we'd need a cumulant collision (see VALIDATION.md §8.8 #3) or a
    # much larger grid (Ny>=128) -- both deferred.
    "cylinder_re20": {
        "body_type": "cylinder",
        "Nx": 64, "Ny": 32, "Nz": 32,
        "body_params": {"cx": 16.0, "cy": 16.0, "R": 8.0},
        "u_in": 0.04,
        "nu": 0.032,                     # Re = 0.04 * 16 / 0.032 = 20
        "n_steps": 2000,
        "scheme": "trt",
        "use_guo_neem": True,
        "use_bouzidi": True,
        "rho_outflow": 1.0,
        "outflow_scheme": "regularised",
    },
    # NACA0012 at Re=20 (chord 20, tau=0.62) -- 9 chord-transits.
    "naca0012_re20": {
        "body_type": "naca",
        "Nx": 80, "Ny": 40, "Nz": 32,
        "body_params": {
            "x_le": 12.0, "chord_offset": 16.0, "chord": 20.0,
            "m": 0.0, "p": 0.0, "thickness": 0.12,
            "aoa_deg": 0.0,
            "span_axis": "y",
        },
        "u_in": 0.04, "nu": 0.04, "n_steps": 6000,
        "scheme": "trt", "use_guo_neem": True, "use_bouzidi": True,
        "rho_outflow": 1.0, "outflow_scheme": "regularised",
    },
    "naca0012_aoa30_re20": {
        "body_type": "naca",
        "Nx": 80, "Ny": 40, "Nz": 32,
        "body_params": {
            "x_le": 12.0, "chord_offset": 16.0, "chord": 20.0,
            "m": 0.0, "p": 0.0, "thickness": 0.12,
            "aoa_deg": 30.0,
            "span_axis": "y",
        },
        "u_in": 0.04, "nu": 0.04, "n_steps": 6000,
        "scheme": "trt", "use_guo_neem": True, "use_bouzidi": True,
        "rho_outflow": 1.0, "outflow_scheme": "regularised",
    },
    "naca0012_aoa-30_re20": {
        "body_type": "naca",
        "Nx": 80, "Ny": 40, "Nz": 32,
        "body_params": {
            "x_le": 12.0, "chord_offset": 16.0, "chord": 20.0,
            "m": 0.0, "p": 0.0, "thickness": 0.12,
            "aoa_deg": -30.0,
            "span_axis": "y",
        },
        "u_in": 0.04, "nu": 0.04, "n_steps": 6000,
        "scheme": "trt", "use_guo_neem": True, "use_bouzidi": True,
        "rho_outflow": 1.0, "outflow_scheme": "regularised",
    },
    "naca0012_aoa45_re20": {
        "body_type": "naca",
        "Nx": 80, "Ny": 48, "Nz": 32,
        "body_params": {
            "x_le": 12.0, "chord_offset": 16.0, "chord": 20.0,
            "m": 0.0, "p": 0.0, "thickness": 0.12,
            "aoa_deg": 45.0,
            "span_axis": "y",
        },
        "u_in": 0.04, "nu": 0.04, "n_steps": 6000,
        "scheme": "trt", "use_guo_neem": True, "use_bouzidi": True,
        "rho_outflow": 1.0, "outflow_scheme": "regularised",
    },
    "naca0012_aoa-45_re20": {
        "body_type": "naca",
        "Nx": 80, "Ny": 48, "Nz": 32,
        "body_params": {
            "x_le": 12.0, "chord_offset": 16.0, "chord": 20.0,
            "m": 0.0, "p": 0.0, "thickness": 0.12,
            "aoa_deg": -45.0,
            "span_axis": "y",
        },
        "u_in": 0.04, "nu": 0.04, "n_steps": 6000,
        "scheme": "trt", "use_guo_neem": True, "use_bouzidi": True,
        "rho_outflow": 1.0, "outflow_scheme": "regularised",
    },
    # NACA4412 at Re=20 (same grid as 0012).
    "naca4412_re20": {
        "body_type": "naca",
        "Nx": 80, "Ny": 40, "Nz": 32,
        "body_params": {
            "x_le": 12.0, "chord_offset": 16.0, "chord": 20.0,
            "m": 0.04, "p": 0.4, "thickness": 0.12,
            "aoa_deg": 0.0,
            "span_axis": "y",
        },
        "u_in": 0.04, "nu": 0.04, "n_steps": 6000,
        "scheme": "trt", "use_guo_neem": True, "use_bouzidi": True,
        "rho_outflow": 1.0, "outflow_scheme": "regularised",
    },
    "naca4412_aoa30_re20": {
        "body_type": "naca",
        "Nx": 80, "Ny": 40, "Nz": 32,
        "body_params": {
            "x_le": 12.0, "chord_offset": 16.0, "chord": 20.0,
            "m": 0.04, "p": 0.4, "thickness": 0.12,
            "aoa_deg": 30.0,
            "span_axis": "y",
        },
        "u_in": 0.04, "nu": 0.04, "n_steps": 6000,
        "scheme": "trt", "use_guo_neem": True, "use_bouzidi": True,
        "rho_outflow": 1.0, "outflow_scheme": "regularised",
    },
    "naca4412_aoa-30_re20": {
        "body_type": "naca",
        "Nx": 80, "Ny": 40, "Nz": 32,
        "body_params": {
            "x_le": 12.0, "chord_offset": 16.0, "chord": 20.0,
            "m": 0.04, "p": 0.4, "thickness": 0.12,
            "aoa_deg": -30.0,
            "span_axis": "y",
        },
        "u_in": 0.04, "nu": 0.04, "n_steps": 6000,
        "scheme": "trt", "use_guo_neem": True, "use_bouzidi": True,
        "rho_outflow": 1.0, "outflow_scheme": "regularised",
    },
    "naca4412_aoa45_re20": {
        "body_type": "naca",
        "Nx": 80, "Ny": 48, "Nz": 32,
        "body_params": {
            "x_le": 12.0, "chord_offset": 16.0, "chord": 20.0,
            "m": 0.04, "p": 0.4, "thickness": 0.12,
            "aoa_deg": 45.0,
            "span_axis": "y",
        },
        "u_in": 0.04, "nu": 0.04, "n_steps": 6000,
        "scheme": "trt", "use_guo_neem": True, "use_bouzidi": True,
        "rho_outflow": 1.0, "outflow_scheme": "regularised",
    },
    "naca4412_aoa-45_re20": {
        "body_type": "naca",
        "Nx": 80, "Ny": 48, "Nz": 32,
        "body_params": {
            "x_le": 12.0, "chord_offset": 16.0, "chord": 20.0,
            "m": 0.04, "p": 0.4, "thickness": 0.12,
            "aoa_deg": -45.0,
            "span_axis": "y",
        },
        "u_in": 0.04, "nu": 0.04, "n_steps": 6000,
        "scheme": "trt", "use_guo_neem": True, "use_bouzidi": True,
        "rho_outflow": 1.0, "outflow_scheme": "regularised",
    },
    # NACA0012 at Re=200 (chord 32 -> bigger grid, tau=0.5192).
    "naca0012_re200": {
        "body_type": "naca",
        "Nx": 128, "Ny": 64, "Nz": 40,
        "body_params": {
            "x_le": 20.0, "chord_offset": 32.0, "chord": 32.0,
            "m": 0.0, "p": 0.0, "thickness": 0.12,
            "aoa_deg": 0.0,
            "span_axis": "y",
        },
        "u_in": 0.04, "nu": 0.0064, "n_steps": 16000,
        "scheme": "trt", "use_guo_neem": True, "use_bouzidi": True,
        "rho_outflow": 1.0, "outflow_scheme": "regularised",
    },
    "naca0012_aoa30_re200": {
        "body_type": "naca",
        "Nx": 128, "Ny": 80, "Nz": 40,
        "body_params": {
            "x_le": 20.0, "chord_offset": 32.0, "chord": 32.0,
            "m": 0.0, "p": 0.0, "thickness": 0.12,
            "aoa_deg": 30.0,
            "span_axis": "y",
        },
        "u_in": 0.04, "nu": 0.0064, "n_steps": 16000,
        "scheme": "trt", "use_guo_neem": True, "use_bouzidi": True,
        "rho_outflow": 1.0, "outflow_scheme": "regularised",
    },
    "naca0012_aoa-30_re200": {
        "body_type": "naca",
        "Nx": 128, "Ny": 80, "Nz": 40,
        "body_params": {
            "x_le": 20.0, "chord_offset": 32.0, "chord": 32.0,
            "m": 0.0, "p": 0.0, "thickness": 0.12,
            "aoa_deg": -30.0,
            "span_axis": "y",
        },
        "u_in": 0.04, "nu": 0.0064, "n_steps": 16000,
        "scheme": "trt", "use_guo_neem": True, "use_bouzidi": True,
        "rho_outflow": 1.0, "outflow_scheme": "regularised",
    },
    # v1.7.3: Nz bumped 40 -> 48 and chord_offset 32 -> 24 so the
    # rotated chord (vertical extent ~26 LU at AoA=45) sits centered
    # with ~11 LU wake clearance on each Z face. v1.7.2 setting
    # clipped at the top wall (body z-bbox reached Nz-1).
    "naca0012_aoa45_re200": {
        "body_type": "naca",
        "Nx": 128, "Ny": 96, "Nz": 48,
        "body_params": {
            "x_le": 20.0, "chord_offset": 24.0, "chord": 32.0,
            "m": 0.0, "p": 0.0, "thickness": 0.12,
            "aoa_deg": 45.0,
            "span_axis": "y",
        },
        "u_in": 0.04, "nu": 0.0064, "n_steps": 16000,
        "scheme": "trt", "use_guo_neem": True, "use_bouzidi": True,
        "rho_outflow": 1.0, "outflow_scheme": "regularised",
    },
    "naca0012_aoa-45_re200": {
        "body_type": "naca",
        "Nx": 128, "Ny": 96, "Nz": 48,
        "body_params": {
            "x_le": 20.0, "chord_offset": 24.0, "chord": 32.0,
            "m": 0.0, "p": 0.0, "thickness": 0.12,
            "aoa_deg": -45.0,
            "span_axis": "y",
        },
        "u_in": 0.04, "nu": 0.0064, "n_steps": 16000,
        "scheme": "trt", "use_guo_neem": True, "use_bouzidi": True,
        "rho_outflow": 1.0, "outflow_scheme": "regularised",
    },
    # NACA4412 at Re=200.
    "naca4412_re200": {
        "body_type": "naca",
        "Nx": 128, "Ny": 64, "Nz": 40,
        "body_params": {
            "x_le": 20.0, "chord_offset": 32.0, "chord": 32.0,
            "m": 0.04, "p": 0.4, "thickness": 0.12,
            "aoa_deg": 0.0,
            "span_axis": "y",
        },
        "u_in": 0.04, "nu": 0.0064, "n_steps": 16000,
        "scheme": "trt", "use_guo_neem": True, "use_bouzidi": True,
        "rho_outflow": 1.0, "outflow_scheme": "regularised",
    },
    "naca4412_aoa30_re200": {
        "body_type": "naca",
        "Nx": 128, "Ny": 80, "Nz": 40,
        "body_params": {
            "x_le": 20.0, "chord_offset": 32.0, "chord": 32.0,
            "m": 0.04, "p": 0.4, "thickness": 0.12,
            "aoa_deg": 30.0,
            "span_axis": "y",
        },
        "u_in": 0.04, "nu": 0.0064, "n_steps": 16000,
        "scheme": "trt", "use_guo_neem": True, "use_bouzidi": True,
        "rho_outflow": 1.0, "outflow_scheme": "regularised",
    },
    "naca4412_aoa-30_re200": {
        "body_type": "naca",
        "Nx": 128, "Ny": 80, "Nz": 40,
        "body_params": {
            "x_le": 20.0, "chord_offset": 32.0, "chord": 32.0,
            "m": 0.04, "p": 0.4, "thickness": 0.12,
            "aoa_deg": -30.0,
            "span_axis": "y",
        },
        "u_in": 0.04, "nu": 0.0064, "n_steps": 16000,
        "scheme": "trt", "use_guo_neem": True, "use_bouzidi": True,
        "rho_outflow": 1.0, "outflow_scheme": "regularised",
    },
    "naca4412_aoa45_re200": {
        "body_type": "naca",
        "Nx": 128, "Ny": 96, "Nz": 48,
        "body_params": {
            "x_le": 20.0, "chord_offset": 24.0, "chord": 32.0,
            "m": 0.04, "p": 0.4, "thickness": 0.12,
            "aoa_deg": 45.0,
            "span_axis": "y",
        },
        "u_in": 0.04, "nu": 0.0064, "n_steps": 16000,
        "scheme": "trt", "use_guo_neem": True, "use_bouzidi": True,
        "rho_outflow": 1.0, "outflow_scheme": "regularised",
    },
    "naca4412_aoa-45_re200": {
        "body_type": "naca",
        "Nx": 128, "Ny": 96, "Nz": 48,
        "body_params": {
            "x_le": 20.0, "chord_offset": 24.0, "chord": 32.0,
            "m": 0.04, "p": 0.4, "thickness": 0.12,
            "aoa_deg": -45.0,
            "span_axis": "y",
        },
        "u_in": 0.04, "nu": 0.0064, "n_steps": 16000,
        "scheme": "trt", "use_guo_neem": True, "use_bouzidi": True,
        "rho_outflow": 1.0, "outflow_scheme": "regularised",
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
    if body_type == "cube":
        # Axis-aligned cube, optionally rotated about the y axis. Same
        # voxelised wall-link approach as the cylinder; flat faces at
        # AoA=0 align with cell boundaries; the rotated cube uses
        # staircase walls.
        cx = float(body_params["cx"])
        cy = float(body_params["cy"])
        cz = float(body_params["cz"])
        h = float(body_params["half_extent"])
        aoa_deg = float(body_params.get("aoa_deg", 0.0))
        mask = make_cube_mask(
            Nx, Ny, Nz, cx, cy, cz, h, aoa_deg=aoa_deg,
        )
        wall_links = voxel_wall_links(mask)
        return mask, wall_links
    if body_type == "naca":
        # NACA 4-digit airfoil. Default span axis is now ``y`` so the
        # wing reads as a chord-section extending side-to-side (the
        # standard aircraft-wing camera angle). Legacy bakes can pin
        # ``span_axis='z'`` if needed.
        x_le = float(body_params["x_le"])
        # chord_offset is the position of the chord midline along the
        # cross-section perpendicular axis (z for horizontal wings,
        # y for legacy vertical wings).
        chord_offset = float(
            body_params.get("chord_offset",
                            body_params.get("y_chord"))
        )
        chord = float(body_params["chord"])
        m = float(body_params["m"])
        p = float(body_params["p"])
        thickness = float(body_params["thickness"])
        aoa_deg = float(body_params.get("aoa_deg", 0.0))
        span_axis = str(body_params.get("span_axis", "y"))
        mask = make_naca_mask(
            Nx, Ny, Nz, x_le, chord_offset, chord, m, p, thickness,
            aoa_deg=aoa_deg, span_axis=span_axis,
        )
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
