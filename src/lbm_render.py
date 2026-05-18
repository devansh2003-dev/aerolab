"""LBM simulation + GIF rendering pipeline.

`simulate_and_render(shape_preset, reynolds_target, aoa_deg, res_key)` runs
the D2Q9 LBM with MRT collision + Smagorinsky LES, then renders to a GIF
with RK4-advected particle streaklines on an alpha-modulated vorticity
heatmap. It returns a dict of bytes + metadata for the Streamlit caller.

The function takes an optional `progress_callback(frac, text)` so the
Streamlit UI can show progress without this module depending on Streamlit.

All tunables for the pipeline are at the top of the module, grouped by
what they control (simulation / grid presets / particles / vorticity /
visual). Don't bury new tunables inside functions.
"""
import io

import matplotlib as mpl
# Force the non-interactive Agg backend BEFORE pyplot is imported. Streamlit
# reads our output as raw GIF/PNG bytes (no interactive window needed), and
# pytest runs headless (Windows Tk on miniconda is fragile). Setting Agg here
# is the standard production-server pattern -- not a test workaround.
mpl.use("Agg", force=True)
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.patches import Polygon
from PIL import Image
from scipy.ndimage import gaussian_filter

from src.lbm import CS2, equilibrium, macroscopic, step_njit_mrt_no_force
from src.shapes import (
    cylinder_mask, cylinder_q_field, ellipse_mask, ellipse_q_field,
    naca4_airfoil_mask, naca4_outline_xy, naca4_q_field, no_bouzidi_q_field,
    square_mask, square_q_field,
)

# Modern system sans-serif for in-figure text. matplotlib picks the first
# font from this list that's actually installed -- Segoe UI Variable on
# Windows (Streamlit's host OS here), Inter / SF Pro on other systems,
# DejaVu Sans as a guaranteed fallback so this never errors at render.
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = [
    "Segoe UI Variable", "Segoe UI", "Inter", "SF Pro Text",
    "Helvetica Neue", "Arial", "DejaVu Sans",
]

# =====================================================================
#  LBM RENDER CONFIGURATION CONSTANTS
# =====================================================================
#  All tunables for the Real CFD path. Grouped by what they control so
#  you can find what you'd actually change without scrolling.
#
#  Rules of thumb:
#    * "Don't touch" = a load-bearing physics or correctness invariant.
#    * "Visual" = changes only how the GIF looks, not solver math.
#    * "Perf" = changes runtime cost, not visible output.
# =====================================================================

# --- Simulation: timestep schedule, inflow, body perturbation ---
# STEPS_PER_FRAME: 35 = 3.5 cells of inflow advance per recorded frame at
# U=0.1. Smaller = smoother playback at the cost of more LBM work per
# frame; larger = jumpy motion. Was 50 before the Cloud-perf pass --
# dropping to 35 cuts total LBM work by 30 % with only marginal
# inter-frame jump visible. The per-preset N_FRAMES lives in
# RESOLUTION_PRESETS below (Standard 60 frames, Detailed 100 frames).
STEPS_PER_FRAME = 35
# U_INFLOW: lattice inflow speed. 0.1 keeps Mach ~ 0.17 (well below the
# LBM stability limit of ~0.3). Don't increase past 0.15 without
# widening tau or compressibility errors get visible.
U_INFLOW = 0.1
# Kick: a small directional perturbation injected near the body during
# the first ~5 frames to break perfect symmetry (would otherwise leave
# cylinder wake non-shedding for thousands of steps due to FP-perfect
# mirror flow). KICK_AMPLITUDE 0.008 is stronger than the 0.005 we used
# to have warmup, since now the wake has to develop within the recording.
KICK_START, KICK_END = 30, 200
KICK_AMPLITUDE = 0.008
KICK_Y_OFFSET = 2
# Boundary-condition direction masks (don't touch -- D2Q9 lattice geometry).
INFLOW_DIRS = np.array([1, 5, 8], dtype=np.int32)
OUTFLOW_DIRS = np.array([3, 6, 7], dtype=np.int32)

# --- Grid presets ---
# Standard matches the dev_lbm_gif_preview reference geometry: body
# fills viewport, wake develops within 5000 steps. Detailed bumps grid
# ~5.6x in cells and recording to 10000 steps for full limit-cycle.
RESOLUTION_PRESETS = {
    "Standard (240 x 80)": dict(
        Nx=240, Ny=80, body_x=52, cy=40,
        cylinder_D=16, square_side=16,
        ellipse_a=18, ellipse_b=9, chord=36,
        n_frames=60,
        gif_palette=192,
    ),
    "Detailed (720 x 240)": dict(
        Nx=720, Ny=240, body_x=160, cy=120,
        cylinder_D=45, square_side=45,
        ellipse_a=50, ellipse_b=25, chord=100,
        n_frames=100,
        gif_palette=128,
    ),
    # Standard preset shrunk from 320x100 to 240x80 (44 % fewer cells)
    # for Cloud free-tier wall-time. Body sizes scaled with the grid so
    # blockage stays ~20 % (cylinder D/Ny = 16/80) -- matches the old
    # Standard's blockage. Detailed is unchanged: opt-in for quality.
    #
    # Budget at STEPS_PER_FRAME=35:
    #   Standard 60 frames * 35 steps = 2100 LBM steps  (~1.4 shedding
    #     periods at D=16). Cloud ~ 60-75 s; local ~ 12 s.
    #   Detailed 100 frames * 35 steps = 3500 LBM steps (~2.5 periods
    #     at D=45). Cloud ~ 3-4 min; local ~ 50 s.
    # GIF sizes: Standard ~3-4 MB, Detailed ~9-11 MB (palette 192/128).
}
# gif_palette: number of colors in the MEDIANCUT palette used to quantize
# each frame. Lower = smaller GIF, more posterization in smooth gradients.
# Measurements at Re=400, AoA=15 with the full plasma + RdBu_r render:
#   * 256 colors: Standard ~6-7 MB, Detailed ~18-20 MB (Detailed exceeds
#     practical bandwidth thresholds on free-tier Cloud).
#   * 192 colors: Standard ~4-5 MB (visually identical -- the alpha-faded
#     vorticity heatmap doesn't have 256 distinguishable hues).
#   * 128 colors: Detailed ~9-11 MB (gradient banding visible only on
#     close inspection of the vorticity heatmap; particle streaks
#     unaffected). Worth the size cut.
# Set both back to 256 if you ship a fundamentally different palette.

# --- Particle streaklines ---
# Two seeding sources:
#   (a) "Inflow column" -- n_seed_rows rows across the inflow, SPAWN_PER_SEED
#       particles per row per frame. Floor of 8 rows (Standard preset),
#       scales as Ny / SEED_ROW_CELL_SPACING on Detailed (Ny=240 -> 20 rows)
#       so seed density stays roughly constant as the channel grows. The
#       previous fixed 8 rows on Detailed left big gaps -- one stream per
#       30 cells of Ny, with half the channel unsampled.
#   (b) "Wake region" -- N_WAKE_SPAWN_PER_FRAME particles spawned at random
#       (x, y) downstream of the body each frame. Without this the wake
#       had no fresh particles by the time vortices shed -- the inflow
#       particles passed through 60 frames ago and aged out, leaving the
#       most physically interesting region visually empty.
# MAX_AGE 60: particles fade out after 60 frames; with 5000-step
# Standard sim and STEPS_PER_FRAME=50, 60 frames = 3000 lattice-time
# of trail -- the trail covers most of the channel before fading.
# RK4_SUBSTEPS 4: 4 RK4 substeps per frame, each advecting by
# STEPS_PER_FRAME/4 = 12.5 lattice-time. More substeps = smoother
# trajectories near vortices, marginal cost (perf).
# PARTICLE_BASE_SIZE 22: matplotlib scatter "s" param in px^2 at birth;
# tapers to ~60% at MAX_AGE.
SEED_ROW_CELL_SPACING = 12   # one inflow seed row per N cells of Ny
SEED_ROW_MIN = 8             # never fewer than this many inflow rows
WAKE_SPAWN_PER_NX = 0.05     # wake particles per frame, as fraction of Nx
WAKE_SPAWN_MIN = 12          # never fewer than this many wake particles/frame
SPAWN_PER_SEED = 3
MAX_AGE = 60
RK4_SUBSTEPS = 4
PARTICLE_BASE_SIZE = 22.0

# --- Vorticity heatmap ---
# Alpha-modulated RdBu_r: red = anti-clockwise, blue = clockwise,
# transparent at omega~0 so the dark background shows through.
# VORT_ALPHA_MAX 0.9 keeps the wake clearly visible while still letting
# bright particles read on top of vortex cores.
# v_clip = max(VORT_CLIP_PERCENTILE pct of |omega|, VORT_CLIP_FACTOR*U/L)
# The floor keeps low-Re wakes visible; the percentile auto-scales for
# high-Re. 92nd percentile + 1.5*U/L floor was tuned across Re=50-1500.
VORT_ALPHA_MAX = 0.9
VORT_CLIP_FACTOR = 1.5
VORT_CLIP_PERCENTILE = 92
# Re-adaptive Gaussian blur on the vorticity field before display.
# Sigma = BASE + RE_SCALE * log10(Re/100). At Re=100 sigma=1.0; at
# Re=1000 sigma=2.6 -- more smoothing for the noisier high-Re wake.
VORT_BLUR_SIGMA_BASE = 1.0
VORT_BLUR_SIGMA_RE_SCALE = 1.6
# WALL_FADE_CELLS 14: smoothstep alpha-fade in the bottom/top 14 cells
# to hide the bounce-back BL band from the heatmap (visual; physics
# unchanged). 14 is ~14% of Ny=100 -- proportional on both presets.
WALL_FADE_CELLS = 14
# VORT_UPSAMPLE 1: we let matplotlib's bicubic do the smoothing at
# display resolution instead of pre-upsampling with scipy.zoom (perf,
# ~10ms/frame saved). Don't bump above 1 unless you have a reason.
VORT_UPSAMPLE = 1

# --- Colors, body patch, figure, GIF ---
# Body patch is drawn just 0.3 cells outside the voxelized mask. Bouzidi
# interpolated bounce-back puts the physics wall at its analytic location,
# so the polygon overlay only has to cover the staircase voxel boundary --
# not the full near-wall region. A tighter margin keeps the boundary
# layer, separation point, and shear-layer initiation visible.
BG_COLOR = "#0a0a0a"
BODY_COLOR = "#1f2937"               # slate-800
BODY_OUTLINE_MARGIN = 0.3
# Body patch opacity: 0.7 so a faint trace of the near-wall vorticity
# bleeds through (helps the viewer see the boundary layer attaching to
# / separating from the surface).
BODY_ALPHA = 0.7
TEXT_COLOR = "#f5f5f5"
# Particle-speed colormap: perceptually uniform "plasma" truncated to
# [0.15, 1.0] so the dark-purple low end stays visible on the dark
# background. plasma is the matplotlib standard for sequential scientific
# data and matches the convention used by ParaView/Tecplot/ANSYS
# post-processors.
SPEED_CMAP = ListedColormap(
    plt.get_cmap("plasma")(np.linspace(0.15, 1.0, 256)),
    name="aerolab_speed",
)
# SPEED_CLIP_FACTOR 2.0: clip speed colorbar at 2*U_INFLOW so inflow
# speed lands at the midpoint and "slower/faster than freestream"
# reads symmetrically.
SPEED_CLIP_FACTOR = 2.0
# Figure: 968x308 px at 88 DPI. Wider aspect matches Nx/Ny ratios of
# both presets (3.0 each after the Standard shrink). DPI dropped from
# 100 to 88 in the Cloud-perf pass -- ~22 % fewer pixels per frame,
# proportionally faster matplotlib draw + smaller GIF, visually
# indistinguishable from 100 on a typical display.
FIG_W_IN, FIG_H_IN, FIG_DPI = 11.0, 3.5, 88
# GIF_FRAME_MS 67 -> 15 fps. Browser GIF playback caps at ~20 fps
# reliably; 15 is the safe choice across Chrome/Firefox/Safari.
GIF_FRAME_MS = 67


def expand_outline(xs, ys, margin):
    """Push each polygon vertex `margin` cells outward from the centroid.

    Used to draw the displayed body patch slightly larger than the
    voxelized physics mask, so the patch covers (a) the staircase mask
    boundary and (b) the bright vorticity ring of the bounce-back
    boundary layer. The mask itself is unchanged -- this is display only.
    """
    cx_poly = float(np.mean(xs))
    cy_poly = float(np.mean(ys))
    dx = xs - cx_poly
    dy = ys - cy_poly
    r = np.sqrt(dx * dx + dy * dy)
    r_safe = np.where(r > 1e-6, r, 1.0)
    return xs + margin * dx / r_safe, ys + margin * dy / r_safe


def body_outline_xy(shape_preset, res_cfg, aoa_deg):
    """Smooth analytic boundary (xs, ys) in grid coords for the body.

    The LBM mask is voxelized to grid cells, which makes the body's edge look
    staircase-y when rendered. We overlay this high-resolution outline as a
    filled patch so the displayed shape edge stays crisp regardless of grid
    resolution. The mask itself (used for physics) is unchanged.
    """
    body_x = res_cfg["body_x"]
    cy = res_cfg["cy"]
    aoa_rad = np.deg2rad(aoa_deg)
    cos_a = np.cos(aoa_rad)
    sin_a = np.sin(aoa_rad)
    if shape_preset == "Cylinder":
        r = res_cfg["cylinder_D"] / 2
        t = np.linspace(0.0, 2 * np.pi, 200)
        return body_x + r * np.cos(t), cy + r * np.sin(t)
    if shape_preset == "Square":
        s = res_cfg["square_side"] / 2
        xs_local = np.array([-s, s, s, -s, -s])
        ys_local = np.array([-s, -s, s, s, -s])
        return (
            body_x + cos_a * xs_local + sin_a * ys_local,
            cy + (-sin_a) * xs_local + cos_a * ys_local,
        )
    if shape_preset == "Ellipse":
        a = res_cfg["ellipse_a"]
        b = res_cfg["ellipse_b"]
        t = np.linspace(0.0, 2 * np.pi, 200)
        xs_local = a * np.cos(t)
        ys_local = b * np.sin(t)
        return (
            body_x + cos_a * xs_local + sin_a * ys_local,
            cy + (-sin_a) * xs_local + cos_a * ys_local,
        )
    # NACA 0012 / 4412. The LE is anchored at (body_x, cy); positive AoA
    # rotates LE up (negative-sine on y component, matching the mask).
    chord = res_cfg["chord"]
    naca_code = shape_preset.split()[1]
    poly_x, poly_y = naca4_outline_xy(naca_code)
    gx = body_x + chord * (poly_x * cos_a + poly_y * sin_a)
    gy = cy + chord * (-poly_x * sin_a + poly_y * cos_a)
    return gx, gy


def _bilerp(field, xs, ys):
    """Vectorized bilinear interpolation of a 2D field at FP positions.

    Used inside the particle RK4 loop to look up u, v at each particle's
    exact location (sub-cell precision -- without this, particles would
    jitter on grid lines).
    """
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.clip(x0, 0, field.shape[0] - 2)
    y0 = np.clip(y0, 0, field.shape[1] - 2)
    fx = xs - x0
    fy = ys - y0
    return (field[x0, y0]         * (1.0 - fx) * (1.0 - fy) +
            field[x0 + 1, y0]     *        fx  * (1.0 - fy) +
            field[x0, y0 + 1]     * (1.0 - fx) *        fy  +
            field[x0 + 1, y0 + 1] *        fx  *        fy)


def _render_horizontal_cbar(scalar_mappable, ticks, tick_labels):
    """Render a horizontal colorbar to PNG bytes.

    Used for the two legend strips below the main GIF (vorticity sign,
    particle speed). Standalone figure -> savefig to BytesIO -> returns
    PNG bytes for Streamlit's st.image().
    """
    cfig, cax = plt.subplots(figsize=(8.5, 0.55), dpi=FIG_DPI,
                              facecolor=BG_COLOR)
    cfig.subplots_adjust(left=0.06, right=0.94, bottom=0.55, top=0.95)
    cbar = cfig.colorbar(scalar_mappable, cax=cax, orientation="horizontal")
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)
    cbar.ax.xaxis.set_tick_params(color=TEXT_COLOR, labelcolor=TEXT_COLOR,
                                   labelsize=9, length=0, pad=4)
    cbar.outline.set_visible(False)
    buf = io.BytesIO()
    cfig.savefig(buf, format="png", facecolor=BG_COLOR, dpi=FIG_DPI)
    plt.close(cfig)
    return buf.getvalue()


def _noop_progress(frac, text):
    pass


def simulate_and_render(shape_preset, reynolds_target, aoa_deg, res_key,
                         *, progress_callback=None, n_frames=None):
    """Run LBM and render to GIF + colorbars.

    Pure function with no Streamlit dependency. The caller owns all I/O.

    Parameters
    ----------
    shape_preset : str
        One of "Cylinder", "Square", "Ellipse", "NACA 0012", "NACA 4412".
    reynolds_target : float
        Reynolds number based on the body's characteristic length (diameter
        for cylinder/ellipse, side for square, chord for airfoil).
    aoa_deg : float
        Body rotation / wing tilt in degrees. Ignored for Cylinder.
    res_key : str
        One of the keys in RESOLUTION_PRESETS ("Standard (240 x 80)" or
        "Detailed (720 x 240)").
    progress_callback : callable or None
        Signature ``(fraction: float in [0, 1], text: str) -> None``.
        Defaults to a no-op for headless use.
    n_frames : int or None
        Overrides the per-preset frame count. Used by end-to-end tests to
        run a 5-frame pipeline in ~5 s instead of 60/100. Production
        callers leave it None.

    Returns
    -------
    dict
        gif_bytes, vort_cbar_bytes, speed_cbar_bytes : bytes for st.image
        label                                        : short title string
        tau, nu, char_length                         : physics scalars
        lbm_nx, lbm_ny, n_frames, n_steps            : grid + sim sizes
        near_stable                                  : bool (tau < 0.51)
    """
    progress = progress_callback or _noop_progress

    res_cfg = RESOLUTION_PRESETS[res_key]
    LBM_NX = res_cfg["Nx"]
    LBM_NY = res_cfg["Ny"]
    BODY_X = res_cfg["body_x"]
    CY_CENTER = res_cfg["cy"]

    # Each shape returns (mask, q_field). All four shapes ship analytic
    # Bouzidi q-fields: cylinder + ellipse via quadratic line-shape
    # intersection, square via 4-face linear, NACA via polygon-segment.
    if shape_preset == "Cylinder":
        D = res_cfg["cylinder_D"]
        mask = cylinder_mask(LBM_NX, LBM_NY, cx=BODY_X, cy=CY_CENTER, radius=D // 2)
        q_field = cylinder_q_field(LBM_NX, LBM_NY, cx=BODY_X, cy=CY_CENTER, radius=D // 2)
        char_length = D
        kick_x = BODY_X + D
        label = "Cylinder"
    elif shape_preset == "Square":
        side = res_cfg["square_side"]
        mask = square_mask(
            LBM_NX, LBM_NY, cx=BODY_X, cy=CY_CENTER, side=side, aoa_deg=aoa_deg,
        )
        q_field = square_q_field(
            LBM_NX, LBM_NY, cx=BODY_X, cy=CY_CENTER, side=side, aoa_deg=aoa_deg,
        )
        char_length = side
        kick_x = BODY_X + int(side * 1.5)
        label = "Square" if abs(aoa_deg) < 0.25 else f"Square  ·  {aoa_deg:+.1f}° rotation"
    elif shape_preset == "Ellipse":
        a, b = res_cfg["ellipse_a"], res_cfg["ellipse_b"]
        mask = ellipse_mask(
            LBM_NX, LBM_NY, cx=BODY_X, cy=CY_CENTER, a=a, b=b, aoa_deg=aoa_deg,
        )
        q_field = ellipse_q_field(
            LBM_NX, LBM_NY, cx=BODY_X, cy=CY_CENTER, a=a, b=b, aoa_deg=aoa_deg,
        )
        char_length = 2 * b
        kick_x = BODY_X + a + 10
        label = "Ellipse" if abs(aoa_deg) < 0.25 else f"Ellipse  ·  {aoa_deg:+.1f}° rotation"
    else:
        chord = res_cfg["chord"]
        naca_code = shape_preset.split()[1]
        mask = naca4_airfoil_mask(
            LBM_NX, LBM_NY, cx=BODY_X, cy=CY_CENTER,
            chord=chord, naca_code=naca_code, aoa_deg=aoa_deg,
        )
        q_field = naca4_q_field(
            LBM_NX, LBM_NY, cx=BODY_X, cy=CY_CENTER,
            chord=chord, naca_code=naca_code, aoa_deg=aoa_deg,
        )
        char_length = chord
        kick_x = BODY_X + chord + 10
        label = f"{shape_preset}  ·  {aoa_deg:+.1f}° wing tilt"

    nu = U_INFLOW * char_length / reynolds_target
    tau = nu / CS2 + 0.5

    # Alpha-modulated RdBu_r cmap, built fresh per call so the ListedColormap
    # isn't shared across cache entries.
    _rdbu = plt.get_cmap("RdBu_r")(np.linspace(0.0, 1.0, 256))
    _alpha_t = np.abs(np.linspace(-1.0, 1.0, 256))
    _rdbu[:, 3] = VORT_ALPHA_MAX * _alpha_t ** 1.4
    vorticity_cmap = ListedColormap(_rdbu, name="rdbu_alpha70")
    vorticity_cmap.set_bad((0.0, 0.0, 0.0, 0.0))

    body_xs, body_ys = body_outline_xy(shape_preset, res_cfg, aoa_deg)
    body_xs, body_ys = expand_outline(body_xs, body_ys, BODY_OUTLINE_MARGIN)

    # Per-preset frame count. Detailed runs 100 frames at 50 steps/frame
    # = 5000 lattice timesteps -- enough for the wake to develop at least
    # 2 von Karman shedding periods at the bigger 720x240 grid. Standard
    # is 60 frames / 3000 steps (~1.5 periods on its smaller body).
    # The n_frames kwarg overrides for tests.
    n_frames_local = int(n_frames) if n_frames is not None else res_cfg["n_frames"]
    n_steps_local = n_frames_local * STEPS_PER_FRAME
    # GIF palette size from preset; .get() fallback keeps older preset
    # dicts (without gif_palette) working at the historical 256-colour
    # default.
    gif_palette_colors = res_cfg.get("gif_palette", 256)

    # Inflow seed y-positions. Number of rows scales with channel height so
    # Detailed (Ny=240) gets ~20 rows instead of being sampled at the same
    # 8 rows as Standard (which left big gaps).
    n_seed_rows = max(SEED_ROW_MIN, LBM_NY // SEED_ROW_CELL_SPACING)
    inflow_y = np.linspace(4.0, LBM_NY - 5.0, n_seed_rows)

    # Wake-region spawn box. Particles spawn at random (x, y) inside this
    # box every frame so vortices that shed off the body have fresh
    # streakline tracers passing through them. The box starts just
    # downstream of the body footprint (BODY_X + char_length is past every
    # supported shape, including NACA at 30 deg AoA) and ends short of the
    # outflow so escaped particles aren't pulled into the boundary BC zone.
    wake_x_min = BODY_X + char_length
    wake_x_max = LBM_NX * 0.78
    wake_y_min = LBM_NY * 0.08
    wake_y_max = LBM_NY * 0.92
    n_wake_spawn = max(WAKE_SPAWN_MIN, int(LBM_NX * WAKE_SPAWN_PER_NX))

    # === Phase 1: simulate, store snapshots ===
    rho0 = np.ones((LBM_NX, LBM_NY))
    u0 = np.zeros((2, LBM_NX, LBM_NY))
    u0[0] = U_INFLOW
    f = equilibrium(rho0, u0)
    f_inflow_eq = equilibrium(1.0, np.array([U_INFLOW, 0.0]))
    kick_y = CY_CENTER + KICK_Y_OFFSET

    progress(0.0, ":material/sync: Phase 1 of 2 -- simulating flow (MRT)...")
    snapshots = []
    step_counter = 0
    # No separate warmup -- frame 0 captures uniform inflow and the
    # wake develops visibly over the first ~30 frames. The kick fires
    # inside the early record steps to break perfect symmetry.
    for frame in range(n_frames_local):
        for _ in range(STEPS_PER_FRAME):
            f = step_njit_mrt_no_force(
                f, tau, mask, q_field, f_inflow_eq, INFLOW_DIRS, OUTFLOW_DIRS,
            )
            if KICK_START <= step_counter < KICK_END:
                f[2, kick_x, kick_y] += KICK_AMPLITUDE
                f[4, kick_x, kick_y] -= KICK_AMPLITUDE
            step_counter += 1

        _, u = macroscopic(f)
        dv_dx = np.zeros_like(u[1])
        du_dy = np.zeros_like(u[0])
        dv_dx[1:-1, :] = (u[1, 2:, :] - u[1, :-2, :]) / 2
        du_dy[:, 1:-1] = (u[0, :, 2:] - u[0, :, :-2]) / 2
        vorticity = dv_dx - du_dy

        snapshots.append({
            "vorticity": vorticity.astype(np.float32),
            "u_x": u[0].astype(np.float32),
            "u_y": u[1].astype(np.float32),
            "step": step_counter,
        })
        progress(
            0.5 * step_counter / n_steps_local,
            f":material/sync: Phase 1 of 2 -- "
            f"simulating frame {frame + 1} / {n_frames_local} (MRT+LES)",
        )

    # Blended v_clip: 92nd percentile of |omega| with a U/L-scaled floor.
    last_vort_fluid = np.where(mask, np.nan, snapshots[-1]["vorticity"])
    v_clip = max(
        float(np.nanpercentile(np.abs(last_vort_fluid), VORT_CLIP_PERCENTILE)),
        VORT_CLIP_FACTOR * U_INFLOW / max(char_length, 1.0),
    )
    blur_sigma = VORT_BLUR_SIGMA_BASE + VORT_BLUR_SIGMA_RE_SCALE * np.log10(
        max(reynolds_target / 100.0, 1.0)
    )

    # Wall-fade smoothstep weight (broadcasts across x).
    fade_hires = WALL_FADE_CELLS * VORT_UPSAMPLE
    ny_hi = LBM_NY * VORT_UPSAMPLE
    y_hi = np.arange(ny_hi)
    edge_dist = np.minimum(y_hi, ny_hi - 1 - y_hi) / fade_hires
    t_edge = np.clip(edge_dist, 0.0, 1.0)
    wall_fade = (t_edge * t_edge * (3.0 - 2.0 * t_edge))[None, :]

    u_clip = SPEED_CLIP_FACTOR * U_INFLOW
    speed_cmap = SPEED_CMAP
    speed_norm = Normalize(vmin=0.0, vmax=u_clip)

    # === Phase 2: render frames into a reused figure ===
    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), dpi=FIG_DPI,
                            facecolor=BG_COLOR)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.99)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, LBM_NX - 1)
    ax.set_ylim(0, LBM_NY - 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    im = ax.imshow(
        np.zeros((LBM_NY * VORT_UPSAMPLE, LBM_NX * VORT_UPSAMPLE)),
        cmap=vorticity_cmap, origin="lower",
        extent=[0, LBM_NX - 1, 0, LBM_NY - 1],
        aspect="equal", interpolation="bicubic",
        interpolation_stage="rgba",
        vmin=-v_clip, vmax=v_clip,
    )
    title_text = ax.text(
        0.012, 0.93, "", transform=ax.transAxes,
        ha="left", va="top",
        color=TEXT_COLOR, fontsize=10.5, fontweight="600",
        bbox=dict(
            boxstyle="round,pad=0.45,rounding_size=0.35",
            facecolor=(0.094, 0.110, 0.165, 0.78),
            edgecolor=(0.482, 0.541, 0.643, 0.65),
            linewidth=0.6,
        ),
        zorder=25,
    )

    body_patch = Polygon(
        np.column_stack([body_xs, body_ys]),
        closed=True, facecolor=BODY_COLOR, edgecolor="#94a3b8",
        linewidth=0.8, alpha=BODY_ALPHA,
        antialiased=True, zorder=10,
    )
    ax.add_patch(body_patch)

    # === Quantitative annotations: flow arrow + body-size scale bar ===
    # These get baked into the GIF (rather than overlaid by Streamlit) so
    # screenshots / shared GIFs carry the orientation + size context. A
    # viewer should never have to guess which way the flow is going or how
    # big the body actually is.

    # Flow direction arrow, top-left. Length = 12% of grid width. The arrow
    # is a clear "the air flows this way" cue, paired with a numeric label.
    arrow_y = LBM_NY * 0.88
    arrow_x0 = LBM_NX * 0.04
    arrow_x1 = LBM_NX * 0.16
    ax.annotate(
        "", xy=(arrow_x1, arrow_y), xytext=(arrow_x0, arrow_y),
        arrowprops=dict(
            arrowstyle="-|>", color=TEXT_COLOR, lw=1.3,
            mutation_scale=14, shrinkA=0, shrinkB=0,
        ),
        zorder=25,
    )
    ax.text(
        (arrow_x0 + arrow_x1) / 2, arrow_y + LBM_NY * 0.05,
        f"flow  U = {U_INFLOW:.2f}",
        color=TEXT_COLOR, fontsize=8.5, ha="center", va="bottom",
        zorder=25,
    )

    # Body-size scale bar, bottom-right. A horizontal line of length =
    # char_length, labeled with how many cells that is. Gives the viewer
    # a concrete sense of the body's pixel-scale.
    sb_y = LBM_NY * 0.08
    sb_x1 = LBM_NX - max(8, LBM_NX * 0.025)
    sb_x0 = sb_x1 - char_length
    ax.plot(
        [sb_x0, sb_x1], [sb_y, sb_y],
        color=TEXT_COLOR, linewidth=1.5, solid_capstyle="butt", zorder=25,
    )
    # Tiny end caps so the bar reads as a measurement, not just a line.
    cap_h = LBM_NY * 0.018
    for cap_x in (sb_x0, sb_x1):
        ax.plot(
            [cap_x, cap_x], [sb_y - cap_h, sb_y + cap_h],
            color=TEXT_COLOR, linewidth=1.5, zorder=25,
        )
    ax.text(
        (sb_x0 + sb_x1) / 2, sb_y - LBM_NY * 0.06,
        f"L = {int(round(char_length))} cells",
        color=TEXT_COLOR, fontsize=8.5, ha="center", va="top",
        zorder=25,
    )

    # Initialize particle state arrays (persistent across frames).
    # Each "particle" is a massless tracer drifting with the velocity
    # field. Seeds spawn continuously from the inflow column, get
    # RK4-advected through the wake, and fade out as they age or exit.
    particle_x = np.empty(0, dtype=np.float64)
    particle_y = np.empty(0, dtype=np.float64)
    particle_age = np.empty(0, dtype=np.int32)

    DT_SUB = STEPS_PER_FRAME / RK4_SUBSTEPS

    gif_frames = []
    master_palette_img = None
    for i, snap in enumerate(snapshots):
        vort_clipped = np.clip(snap["vorticity"], -3.0 * v_clip, 3.0 * v_clip)
        vort_smooth = gaussian_filter(vort_clipped, sigma=blur_sigma) * wall_fade

        # Lightly smooth the velocity for particle advection -- enough
        # to suppress sub-grid LBM oscillations that would jitter the
        # particles, far less than the heavy smoothing streamplot
        # needed (sigma=0.4 here vs 1.5 before).
        u_field = gaussian_filter(snap["u_x"], sigma=0.4)
        v_field = gaussian_filter(snap["u_y"], sigma=0.4)

        im.set_data(vort_smooth.T)
        title_text.set_text(f"{label}  ·  Re = {reynolds_target}")

        # Clear last frame's scatter; keep the persistent body patch.
        for col in list(ax.collections):
            col.remove()
        for patch in list(ax.patches):
            if patch is not body_patch:
                patch.remove()

        # --- 1) Spawn new particles ---
        # Deterministic RNG seeded per-frame so cache hits reproduce identically.
        spawn_rng = np.random.default_rng(seed=1000 + i)

        # (a) Inflow-column spawn: n_seed_rows * SPAWN_PER_SEED particles at x=3.
        inflow_new_x = np.full(n_seed_rows * SPAWN_PER_SEED, 3.0, dtype=np.float64)
        inflow_new_y = np.repeat(inflow_y, SPAWN_PER_SEED).astype(np.float64)
        inflow_new_y = inflow_new_y + spawn_rng.uniform(-0.7, 0.7, size=inflow_new_y.shape)

        # (b) Wake-region spawn: spawn at random (x, y) inside the wake box.
        # Reject any spawn that lands inside the body before adding it to the
        # particle pool (the cull step would catch these on the next frame
        # anyway, but rejecting now avoids a 1-frame visual artifact).
        wake_x_candidates = spawn_rng.uniform(wake_x_min, wake_x_max, size=n_wake_spawn)
        wake_y_candidates = spawn_rng.uniform(wake_y_min, wake_y_max, size=n_wake_spawn)
        xi_cand = np.clip(np.round(wake_x_candidates).astype(np.int32), 0, LBM_NX - 1)
        yi_cand = np.clip(np.round(wake_y_candidates).astype(np.int32), 0, LBM_NY - 1)
        valid_wake = ~mask[xi_cand, yi_cand]
        wake_new_x = wake_x_candidates[valid_wake]
        wake_new_y = wake_y_candidates[valid_wake]

        new_x = np.concatenate([inflow_new_x, wake_new_x])
        new_y = np.concatenate([inflow_new_y, wake_new_y])
        new_age = np.zeros(len(new_y), dtype=np.int32)

        particle_x = np.concatenate([particle_x, new_x])
        particle_y = np.concatenate([particle_y, new_y])
        particle_age = np.concatenate([particle_age, new_age])

        # --- 2) Advect all live particles by STEPS_PER_FRAME of
        # lattice time, using RK4 with the current velocity field ---
        if len(particle_x) > 0:
            for _ in range(RK4_SUBSTEPS):
                x = particle_x
                y = particle_y

                k1x = _bilerp(u_field, x, y)
                k1y = _bilerp(v_field, x, y)

                xm1 = x + 0.5 * DT_SUB * k1x
                ym1 = y + 0.5 * DT_SUB * k1y
                k2x = _bilerp(u_field, xm1, ym1)
                k2y = _bilerp(v_field, xm1, ym1)

                xm2 = x + 0.5 * DT_SUB * k2x
                ym2 = y + 0.5 * DT_SUB * k2y
                k3x = _bilerp(u_field, xm2, ym2)
                k3y = _bilerp(v_field, xm2, ym2)

                xe = x + DT_SUB * k3x
                ye = y + DT_SUB * k3y
                k4x = _bilerp(u_field, xe, ye)
                k4y = _bilerp(v_field, xe, ye)

                particle_x = x + DT_SUB * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
                particle_y = y + DT_SUB * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0

            particle_age = particle_age + 1

            # --- 3) Cull: out-of-domain, hit body, or aged out ---
            in_x = (particle_x >= 1.0) & (particle_x < LBM_NX - 1.5)
            in_y = (particle_y >= 1.0) & (particle_y < LBM_NY - 1.5)
            xi = np.clip(np.round(particle_x).astype(np.int32), 0, LBM_NX - 1)
            yi = np.clip(np.round(particle_y).astype(np.int32), 0, LBM_NY - 1)
            in_body = mask[xi, yi]
            keep = in_x & in_y & (~in_body) & (particle_age < MAX_AGE)

            particle_x = particle_x[keep]
            particle_y = particle_y[keep]
            particle_age = particle_age[keep]

        # --- 4) Render live particles as colored scatter ---
        if len(particle_x) > 0:
            sp = np.sqrt(_bilerp(u_field, particle_x, particle_y) ** 2 +
                         _bilerp(v_field, particle_x, particle_y) ** 2)
            rgba = speed_cmap(speed_norm(sp))
            age_frac = particle_age.astype(np.float64) / MAX_AGE
            # Alpha: fade IN over first 3 frames so newly-spawned
            # particles don't pop, then fade OUT linearly as age -> MAX_AGE.
            fade_in = np.minimum(particle_age / 3.0, 1.0)
            fade_out = 1.0 - age_frac
            rgba[:, 3] = fade_in * fade_out * 0.92
            # Size: gentle taper from full-size at birth down to ~60% at end.
            sizes = PARTICLE_BASE_SIZE * (0.6 + 0.4 * (1.0 - age_frac))
            ax.scatter(
                particle_x, particle_y,
                c=rgba, s=sizes, marker="o",
                edgecolors="none", zorder=8,
            )

        fig.canvas.draw()
        img_rgba = np.asarray(fig.canvas.buffer_rgba())
        # Global-palette GIF quantization, no dithering. The first frame
        # derives a MEDIANCUT palette of `gif_palette_colors` entries;
        # every subsequent frame maps onto it. Palette size is per-preset
        # (see RESOLUTION_PRESETS comments). Dropping Floyd-Steinberg
        # dither removes the speckled noise on smooth gradients.
        rgb_image = Image.fromarray(img_rgba[..., :3])
        if master_palette_img is None:
            img_p = rgb_image.quantize(
                colors=gif_palette_colors, method=Image.Quantize.MEDIANCUT,
                dither=Image.Dither.NONE,
            )
            master_palette_img = img_p
        else:
            img_p = rgb_image.quantize(
                palette=master_palette_img,
                dither=Image.Dither.NONE,
            )
        gif_frames.append(img_p)

        progress(
            0.5 + 0.5 * (i + 1) / n_frames_local,
            f":material/auto_awesome: Phase 2 of 2 -- "
            f"rendering frame {i + 1} of {n_frames_local}",
        )

    plt.close(fig)

    progress(1.0, ":material/movie: Encoding animation...")
    gif_buf_local = io.BytesIO()
    gif_frames[0].save(
        gif_buf_local, format="GIF",
        save_all=True, append_images=gif_frames[1:],
        duration=GIF_FRAME_MS, loop=0, optimize=True, disposal=2,
    )

    vort_cbar_b = _render_horizontal_cbar(
        cm.ScalarMappable(norm=Normalize(vmin=-v_clip, vmax=v_clip),
                          cmap=vorticity_cmap),
        ticks=[-v_clip, 0.0, v_clip],
        tick_labels=["Clockwise spin", "No rotation", "Anti-clockwise spin"],
    )
    speed_cbar_b = _render_horizontal_cbar(
        cm.ScalarMappable(norm=speed_norm, cmap=speed_cmap),
        ticks=[0.0, U_INFLOW, u_clip],
        tick_labels=["Stalled (slow)", "Inflow speed", "Accelerated (fast)"],
    )

    return {
        "gif_bytes": gif_buf_local.getvalue(),
        "vort_cbar_bytes": vort_cbar_b,
        "speed_cbar_bytes": speed_cbar_b,
        "label": label,
        "tau": float(tau),
        "nu": float(nu),
        "char_length": float(char_length),
        "lbm_nx": int(LBM_NX),
        "lbm_ny": int(LBM_NY),
        "n_frames": int(n_frames_local),
        "n_steps": int(n_steps_local),
        "near_stable": bool(tau < 0.51),
    }
