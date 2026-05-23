"""LBM simulation + GIF rendering pipeline.

`simulate_and_render(shape_preset, reynolds_target, aoa_deg, res_key)` runs
the D2Q9 LBM with MRT collision + Smagorinsky LES, then renders to a GIF
with RK4-advected particle streaklines on an alpha-modulated vorticity
heatmap. It returns a dict of bytes + metadata for the Streamlit caller.

Internally the pipeline is split into two public functions so the heavy
LBM solve can be cached independently of the viz-mode-dependent render:

  * solve_lbm(...)  -> dict of simulation snapshots + force history.
  * render_lbm(solve, viz_mode=...) -> dict of GIF + colorbar bytes.

simulate_and_render() is the legacy convenience wrapper that calls both
and merges their outputs into the public-API shape callers expect.

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

from src.lbm import CS2, equilibrium, macroscopic, step_njit_mrt_with_force
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
# RESOLUTION_PRESETS below (both presets now run 150 frames -- the
# sim was fast enough that more frames was a better visual win than
# more grid cells).
STEPS_PER_FRAME = 35
# U_INFLOW: lattice inflow speed. 0.1 keeps Mach ~ 0.17 (well below the
# LBM stability limit of ~0.3). Don't increase past 0.15 without
# widening tau or compressibility errors get visible.
U_INFLOW = 0.1
# Kick: a small directional perturbation injected near the body during
# the first ~5 frames to break perfect symmetry (would otherwise leave
# cylinder wake non-shedding for thousands of steps due to FP-perfect
# mirror flow). KICK_AMPLITUDE 0.008 is stronger than the 0.005 used
# when we had a separate warmup loop -- now the wake has to develop
# within the recording itself, so the kick needs to bite harder.
KICK_START, KICK_END = 30, 200
KICK_AMPLITUDE = 0.008
KICK_Y_OFFSET = 2
# Boundary-condition direction masks (don't touch -- D2Q9 lattice geometry).
INFLOW_DIRS = np.array([1, 5, 8], dtype=np.int32)
OUTFLOW_DIRS = np.array([3, 6, 7], dtype=np.int32)

# --- Grid presets ---
# Standard: 320x80, body fills viewport, wake develops within the 5250
# recorded steps. Detailed bumps grid ~9x in cells and 5250 steps for
# fuller limit-cycle. Both budgets defined explicitly in the dict below.
RESOLUTION_PRESETS = {
    "Standard (320 x 80)": dict(
        Nx=320, Ny=80, body_x=70, cy=40,
        cylinder_D=28, square_side=28,
        ellipse_a=32, ellipse_b=16, chord=60,
        custom_extent=60,
        n_frames=150,
        gif_palette=192,
    ),
    "Detailed (960 x 240)": dict(
        Nx=960, Ny=240, body_x=210, cy=120,
        cylinder_D=80, square_side=80,
        ellipse_a=90, ellipse_b=45, chord=170,
        custom_extent=170,
        n_frames=150,
        gif_palette=96,
    ),
    # Body sizes were bumped (~80 % bigger) in the prior release so each
    # shape fills the channel visibly. To keep the wake from running off
    # the right edge, Nx was then extended: Standard 240 -> 320 (wake
    # region 170 -> 250 cells, ~9D at D=28); Detailed 720 -> 960 (wake
    # region 510 -> 710 cells, ~9D at D=80). body_x is unchanged so the
    # inflow runway stays identical; the extra width goes entirely into
    # downstream wake visibility.
    #
    # Budget at STEPS_PER_FRAME=35:
    #   Standard 150 frames * 35 steps = 5250 LBM steps (~3.2 shedding
    #     periods at D=28). Cloud ~ 3.3 min; local ~ 40 s.
    #   Detailed 150 frames * 35 steps = 5250 LBM steps (~2 periods at
    #     D=80). Cloud ~ 6 min; local ~ 100 s.
    # GIF sizes: Standard ~9-13 MB (palette 192), Detailed ~17-20 MB
    # (palette 96).
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
#       particles aged out, leaving the most physically interesting region
#       visually empty. Spawn box extends close to the outflow so Detailed
#       (Nx=720) doesn't show a long empty stretch past the body.
# MAX_AGE 100: particles fade out after 100 frames. At STEPS_PER_FRAME=35
# and U=0.1 that's 350 cells of drift -- enough for one particle to traverse
# nearly half of the Detailed (720) channel before fading, which makes the
# wake trail-off gradual instead of abrupt.
# RK4_SUBSTEPS 4: 4 RK4 substeps per frame, each advecting by
# STEPS_PER_FRAME/4 = ~8.75 lattice-time. More substeps = smoother
# trajectories near vortices, marginal cost (perf).
# PARTICLE_BASE_SIZE 22: matplotlib scatter "s" param in px^2 at birth;
# tapers to ~60% at MAX_AGE.
SEED_ROW_CELL_SPACING = 12   # one inflow seed row per N cells of Ny
SEED_ROW_MIN = 8             # never fewer than this many inflow rows
WAKE_SPAWN_PER_NX = 0.05     # wake particles per frame, as fraction of Nx
WAKE_SPAWN_MIN = 12          # never fewer than this many wake particles/frame
WAKE_OUTFLOW_FRAC = 0.15     # last 15 % of channel is the trail-off zone (no wake spawn)
SPAWN_PER_SEED = 3
MAX_AGE = 100
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


def body_outline_xy(shape_preset, res_cfg, aoa_deg, custom_polygon=None):
    """Smooth analytic boundary (xs, ys) in grid coords for the body.

    The LBM mask is voxelized to grid cells, which makes the body's edge look
    staircase-y when rendered. We overlay this high-resolution outline as a
    filled patch so the displayed shape edge stays crisp regardless of grid
    resolution. The mask itself (used for physics) is unchanged.

    For shape_preset == "Custom", ``custom_polygon`` must be provided -- the
    extracted-from-image / drawn-on-canvas polygon in image-pixel coords.
    The smooth outline is then the polygon itself, transformed into lattice
    space via custom_shape.polygon_outline_xy.
    """
    body_x = res_cfg["body_x"]
    cy = res_cfg["cy"]
    if shape_preset == "Custom":
        if custom_polygon is None:
            raise ValueError("Custom shape requires custom_polygon")
        # Local import: keeps src.custom_shape's scikit-image import lazy,
        # so the LBM-only code path (tests, scripts) doesn't pay for it.
        from src.custom_shape import polygon_outline_xy
        target_extent = res_cfg.get("custom_extent", 30)
        return polygon_outline_xy(
            custom_polygon, res_cfg["Nx"], res_cfg["Ny"],
            body_x, cy, target_extent, aoa_deg,
        )
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


def render_shape_preview(shape_preset, res_cfg, aoa_deg, custom_polygon=None) -> bytes:
    """Render a quick PNG preview of where the body sits in the tunnel.

    Works for every shape_preset (Cylinder / Square / Ellipse / NACA / Custom)
    by going through body_outline_xy + the shared rendering helper. Shown
    in the sidebar before Run so the user sees scale, position, and AoA
    rotation BEFORE paying the simulation cost.
    """
    from src.custom_shape import render_outline_to_png
    xs, ys = body_outline_xy(
        shape_preset, res_cfg, aoa_deg, custom_polygon=custom_polygon,
    )
    return render_outline_to_png(xs, ys, res_cfg["Nx"], res_cfg["Ny"])


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


VIZ_MODES = ("Vorticity", "Velocity", "Pressure")

# Pressure mode applies a temporal moving-average over PRESSURE_AVG_FRAMES
# snapshots of rho before display. LBM is weakly compressible, so any
# disturbance (the kick, the inflow ramp, vortex shedding) emits acoustic
# waves that propagate at c_s = 1/sqrt(3) ~ 0.577 cells/step and look
# like blue/red lens-shaped bands floating downstream of the body. Over
# 5 frames * 35 steps/frame * 0.577 cells/step ~ 100 cells of travel,
# the waves shift far enough that averaging zeros them out. Body-bound
# pressure features (stagnation, suction) are quasi-static or oscillate
# at the von Karman shedding rate (~1430 steps for Re=200 cylinder), so
# they survive the 5-frame average intact.
PRESSURE_AVG_FRAMES = 5


def solve_lbm(shape_preset, reynolds_target, aoa_deg, res_key,
              *, n_frames=None, custom_polygon=None, progress_callback=None):
    """Run the LBM solve (Phase 1) and return the per-frame snapshots.

    Pure function with no Streamlit dependency. The caller owns all I/O.

    This is the heavy half of simulate_and_render(): all D2Q9 MRT+LES
    timestepping, force history accumulation, and the matplotlib force
    time-series plot. It is mode-independent -- viz_mode does NOT enter
    here, so a single solve can feed any number of render passes.

    Parameters
    ----------
    shape_preset : str
        One of "Cylinder", "Square", "Ellipse", "NACA 0012", "NACA 4412",
        or "Custom" (requires ``custom_polygon`` to be supplied).
    reynolds_target : float
        Reynolds number based on the body's characteristic length.
    aoa_deg : float
        Body rotation / wing tilt in degrees. Ignored for Cylinder.
    res_key : str
        One of the keys in RESOLUTION_PRESETS.
    n_frames : int or None
        Overrides the per-preset frame count. Used by tests.
    custom_polygon : np.ndarray or None
        Required when shape_preset == "Custom"; ignored otherwise.
    progress_callback : callable or None
        Signature ``(fraction: float in [0, 1], text: str) -> None``.
        This function reports progress in the 0.0 -> 0.5 range (Phase 1
        of the legacy 2-phase pipeline).

    Returns
    -------
    dict
        snapshots, mask, body_xs, body_ys, char_length, label, tau, nu,
        cd_history, cl_history, cd_mean, cl_mean, strouhal,
        force_plot_bytes, n_frames, n_steps, near_stable,
        lbm_nx, lbm_ny, reynolds_target, res_key.
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
    elif shape_preset == "Custom":
        if custom_polygon is None:
            raise ValueError("shape_preset='Custom' requires custom_polygon")
        # Local import keeps scikit-image out of the preset-only code path.
        from src.custom_shape import polygon_to_lbm_mask
        target_extent = res_cfg.get("custom_extent", 30)
        mask = polygon_to_lbm_mask(
            custom_polygon, LBM_NX, LBM_NY,
            cx=BODY_X, cy=CY_CENTER,
            target_extent_cells=target_extent, aoa_deg=aoa_deg,
        )
        # Uploaded / drawn polygons use halfway bounce-back for now -- analytic
        # Bouzidi q-field from a polygon requires per-link polygon-segment
        # intersection, scheduled for Phase 2 W5.5.
        q_field = no_bouzidi_q_field(LBM_NX, LBM_NY)
        char_length = float(target_extent)
        kick_x = BODY_X + int(target_extent) + 10
        label = (
            "Custom shape" if abs(aoa_deg) < 0.25
            else f"Custom shape  ·  {aoa_deg:+.1f}° rotation"
        )
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

    # Body outline (mode-independent -- the render path uses these to draw
    # the body patch on top of whichever background field it picked).
    body_xs, body_ys = body_outline_xy(
        shape_preset, res_cfg, aoa_deg, custom_polygon=custom_polygon,
    )
    body_xs, body_ys = expand_outline(body_xs, body_ys, BODY_OUTLINE_MARGIN)

    # Per-preset frame count. Both presets now run 150 frames at
    # STEPS_PER_FRAME=35 = 5250 lattice timesteps. On Standard (D=28)
    # that's ~3.2 von Karman shedding periods; on Detailed (D=80) it's
    # ~2 periods (the wake reaches full limit-cycle inside the loop).
    # The n_frames kwarg overrides for tests.
    n_frames_local = int(n_frames) if n_frames is not None else res_cfg["n_frames"]
    n_steps_local = n_frames_local * STEPS_PER_FRAME

    # === Pre-flight: mask sanity checks ===
    # Catch degenerate-shape geometry BEFORE we enter the @njit step path.
    # The Zou-He inflow/outflow boundaries divide by interior-column rho;
    # if a user's drawing pushes a solid cell into the inflow column or
    # leaves the outflow column with no fluid neighbours, that rho can
    # spike to zero on the first unstable step and trigger
    # ZeroDivisionError inside @njit. Cheap (one mask sum + slice) to
    # validate up-front and surface a clear ValueError instead.
    if shape_preset == "Custom":
        n_solid = int(mask.sum())
        n_fluid = mask.size - n_solid
        if n_solid < 8:
            raise ValueError(
                "Shape is too small after rasterisation (fewer than 8 grid cells "
                "solid). Try a larger source image or a chunkier drawing."
            )
        if n_fluid < mask.size // 4:
            raise ValueError(
                "Shape occupies too much of the channel (more than 75 % solid). "
                "Pick a smaller source image or reduce the body extent."
            )
        if mask[0, :].any():
            raise ValueError(
                "Shape touches the inflow wall (left edge). The simulation "
                "needs clear inflow -- shift the shape rightward or scale it down."
            )
        if mask[-1, :].any() or mask[-2, :].any():
            raise ValueError(
                "Shape touches the outflow wall (right edge). The simulation "
                "needs clear outflow -- shift the shape leftward or scale it down."
            )
        if mask[:, 0].any() or mask[:, -1].any():
            raise ValueError(
                "Shape touches the top/bottom wall. Centre the drawing vertically "
                "or scale it down."
            )

    # === Phase 1: simulate, store snapshots ===
    rho0 = np.ones((LBM_NX, LBM_NY))
    u0 = np.zeros((2, LBM_NX, LBM_NY))
    u0[0] = U_INFLOW
    f = equilibrium(rho0, u0)
    f_inflow_eq = equilibrium(1.0, np.array([U_INFLOW, 0.0]))
    # Ghost-cell equilibrium: solid cells are reset to (rho=1, u=0) after
    # every step. They aren't part of the physical fluid domain, so their
    # populations are free to be overwritten with a known-finite state.
    # This makes the solver immune to rho=0 propagation when the user
    # uploads a thin / spindly silhouette that the morphological cleanup
    # didn't fully smooth out. Tiny cost: one (9, K)-cell broadcast where
    # K = mask.sum(), well under one solver step's work.
    f_eq_solid = equilibrium(1.0, np.array([0.0, 0.0]))  # shape (9,)
    kick_y = CY_CENTER + KICK_Y_OFFSET

    progress(0.0, ":material/sync: Phase 1 of 2 -- simulating flow (MRT)...")
    snapshots = []
    step_counter = 0
    # Force history accumulators for Cd/Cl/Strouhal post-processing.
    # ~5250 float64 entries on either preset -> ~42 KB per array.
    fx_history = np.empty(n_steps_local, dtype=np.float64)
    fy_history = np.empty(n_steps_local, dtype=np.float64)
    # No separate warmup -- frame 0 captures uniform inflow and the
    # wake develops visibly over the first ~30 frames. The kick fires
    # inside the early record steps to break perfect symmetry.
    # Blow-up detection cadence. Checking np.isfinite on the full (9, Nx, Ny)
    # f-array every step is ~0.1 ms on Standard -- a few percent of step cost
    # we don't want to pay. Once per frame (every STEPS_PER_FRAME steps) is
    # tight enough that we catch the divergence before NaN propagates through
    # the rendering pipeline, while staying well under 1 % overhead.
    for frame in range(n_frames_local):
        for _ in range(STEPS_PER_FRAME):
            f, Fx_step, Fy_step = step_njit_mrt_with_force(
                f, tau, mask, q_field, f_inflow_eq, INFLOW_DIRS, OUTFLOW_DIRS,
            )
            fx_history[step_counter] = Fx_step
            fy_history[step_counter] = Fy_step
            # Defensive solid-cell reset: see f_eq_solid comment above.
            f[:, mask] = f_eq_solid[:, None]
            if KICK_START <= step_counter < KICK_END:
                f[2, kick_x, kick_y] += KICK_AMPLITUDE
                f[4, kick_x, kick_y] -= KICK_AMPLITUDE
            step_counter += 1

        # End-of-frame blow-up check. With error_model='numpy' the @njit
        # step path no longer raises ZeroDivisionError on degenerate cells
        # -- it produces inf/NaN silently and keeps going. We detect the
        # divergence here and bail before the rendering code sees NaN
        # snapshots and produces a black GIF. ValueError lets the polite
        # handler in app.py surface a "shape too unstable at this Re"
        # message that's actionable for the user.
        if not np.isfinite(f).all():
            raise ValueError(
                f"Simulation diverged at frame {frame + 1} of {n_frames_local} "
                f"(step {step_counter} of {n_steps_local}, shape={shape_preset!r}, "
                f"Re={reynolds_target}). This Reynolds number is at the edge of "
                f"the solver's stability for this shape -- try a lower Re (the "
                f"square cylinder is reliable up to Re ~ 1000)."
            )

        rho_field, u = macroscopic(f)
        dv_dx = np.zeros_like(u[1])
        du_dy = np.zeros_like(u[0])
        dv_dx[1:-1, :] = (u[1, 2:, :] - u[1, :-2, :]) / 2
        du_dy[:, 1:-1] = (u[0, :, 2:] - u[0, :, :-2]) / 2
        vorticity = dv_dx - du_dy

        snapshots.append({
            "vorticity": vorticity.astype(np.float32),
            "u_x": u[0].astype(np.float32),
            "u_y": u[1].astype(np.float32),
            "rho": rho_field.astype(np.float32),
            "step": step_counter,
        })
        progress(
            0.5 * step_counter / n_steps_local,
            f":material/sync: Phase 1 of 2 -- "
            f"simulating frame {frame + 1} / {n_frames_local} (MRT+LES)",
        )

    # Post-process forces -> Cd / Cl / Strouhal -----------------------------
    # In 2D lattice units with rho ~ 1, the drag/lift coefficients are
    # Cd = 2 * F_x / (rho * U^2 * L), Cl = 2 * F_y / (rho * U^2 * L),
    # where L = char_length and U = U_INFLOW (set above). step_njit_mrt_with_force
    # returns F_x / F_y per step; we keep the full history for the time-series
    # plot and compute summary stats over the last third of the run (to skip
    # the initial transient before vortex shedding locks in).
    cd_history = 2.0 * fx_history / (U_INFLOW ** 2 * char_length)
    cl_history = 2.0 * fy_history / (U_INFLOW ** 2 * char_length)
    _stable_tail_start = max(1, n_steps_local // 3 * 2)  # last third
    cd_mean = float(np.mean(cd_history[_stable_tail_start:]))
    cl_mean = float(np.mean(cl_history[_stable_tail_start:]))

    # Strouhal: dominant frequency of Cl oscillation (vortex shedding sheds
    # at the lift's oscillation rate, ~half the drag's rate for symmetric
    # bodies). FFT on the last-half of the run, in lattice-step units.
    # St = f * L / U where f is in cycles per lattice step.
    cl_tail = cl_history[len(cl_history) // 2:]
    cl_tail_centered = cl_tail - cl_tail.mean()
    if len(cl_tail_centered) >= 16:
        fft_mag = np.abs(np.fft.rfft(cl_tail_centered))
        freqs_per_step = np.fft.rfftfreq(len(cl_tail_centered), d=1.0)
        # Skip DC bin (index 0); pick the loudest non-DC frequency.
        peak_idx = int(np.argmax(fft_mag[1:])) + 1
        peak_freq = float(freqs_per_step[peak_idx])
        strouhal = peak_freq * char_length / U_INFLOW
    else:
        strouhal = float("nan")

    # Time-series plot of Cd + Cl on the same axis. Dark-theme to match the
    # vorticity GIF; fixed size that reads nicely under the GIF.
    fig_ts, ax_ts = plt.subplots(figsize=(8.0, 2.4), dpi=80)
    t_axis = np.arange(n_steps_local)
    ax_ts.plot(t_axis, cd_history, color="#94a3b8", linewidth=1.0, label="Cd (drag)")
    ax_ts.plot(t_axis, cl_history, color="#fbbf24", linewidth=1.0, label="Cl (lift)")
    ax_ts.axhline(0.0, color="#475569", linewidth=0.5)
    # Shade the "stable tail" so the user can see where the mean was taken.
    ax_ts.axvspan(
        _stable_tail_start, n_steps_local,
        color="#fbbf24", alpha=0.07, zorder=0,
    )
    ax_ts.set_xlabel("Lattice timestep", color="#94a3b8", fontsize=9)
    ax_ts.set_ylabel("Force coefficient", color="#94a3b8", fontsize=9)
    ax_ts.legend(
        loc="upper right", facecolor="#0b1220", edgecolor="#334155",
        labelcolor="#cbd5e1", fontsize=8,
    )
    ax_ts.set_facecolor("#0b1220")
    fig_ts.patch.set_facecolor("#0b1220")
    for spine in ax_ts.spines.values():
        spine.set_color("#334155")
    ax_ts.tick_params(axis="both", colors="#64748b", labelsize=8)
    ax_ts.grid(True, color="#1e293b", linestyle="--", linewidth=0.5)
    force_plot_buf = io.BytesIO()
    fig_ts.savefig(
        force_plot_buf, format="png", facecolor="#0b1220",
        bbox_inches="tight", pad_inches=0.1,
    )
    plt.close(fig_ts)
    force_plot_bytes = force_plot_buf.getvalue()

    return {
        "snapshots": snapshots,
        "mask": mask,
        "body_xs": body_xs,
        "body_ys": body_ys,
        "char_length": float(char_length),
        "label": label,
        "tau": float(tau),
        "nu": float(nu),
        "cd_history": cd_history.astype(np.float32),
        "cl_history": cl_history.astype(np.float32),
        "cd_mean": cd_mean,
        "cl_mean": cl_mean,
        "strouhal": float(strouhal),
        "force_plot_bytes": force_plot_bytes,
        "n_frames": int(n_frames_local),
        "n_steps": int(n_steps_local),
        "near_stable": bool(tau < 0.51),
        "lbm_nx": int(LBM_NX),
        "lbm_ny": int(LBM_NY),
        "reynolds_target": reynolds_target,
        "res_key": res_key,
    }


def render_lbm(solve, *, viz_mode="Vorticity", progress_callback=None):
    """Render the snapshots produced by solve_lbm() to a GIF + colorbars.

    Pure function with no Streamlit dependency. The caller owns all I/O.

    This is the light half of simulate_and_render(): per-frame matplotlib
    draw + GIF encode + per-mode colorbar PNGs. Switching viz_mode only
    re-pays this cost (~2 s for Standard), not the full LBM solve.

    Parameters
    ----------
    solve : dict
        The dict returned by solve_lbm(). Required keys:
        snapshots, mask, body_xs, body_ys, char_length, label,
        n_frames, reynolds_target, res_key.
    viz_mode : str
        Background heatmap selection. One of VIZ_MODES:
          * "Vorticity" -- bipolar red/blue, fluid rotation (curl of u).
          * "Velocity"  -- bipolar speed-minus-inflow.
          * "Pressure"  -- bipolar gauge pressure, temporal-averaged.
    progress_callback : callable or None
        Signature ``(fraction: float in [0, 1], text: str) -> None``.
        This function reports progress in the 0.5 -> 1.0 range (Phase 2
        of the legacy 2-phase pipeline); see module docstring.

    Returns
    -------
    dict
        gif_bytes, bg_cbar_bytes, vort_cbar_bytes (alias of bg_cbar_bytes
        for back-compat), bg_cbar_title, bg_cbar_blurb, viz_mode,
        speed_cbar_bytes.
    """
    progress = progress_callback or _noop_progress

    if viz_mode not in VIZ_MODES:
        raise ValueError(
            f"viz_mode must be one of {VIZ_MODES}; got {viz_mode!r}"
        )

    snapshots = solve["snapshots"]
    mask = solve["mask"]
    body_xs = solve["body_xs"]
    body_ys = solve["body_ys"]
    char_length = solve["char_length"]
    label = solve["label"]
    n_frames_local = solve["n_frames"]
    reynolds_target = solve["reynolds_target"]
    res_key = solve["res_key"]

    res_cfg = RESOLUTION_PRESETS[res_key]
    LBM_NX = res_cfg["Nx"]
    LBM_NY = res_cfg["Ny"]
    # GIF palette size from preset; .get() fallback keeps older preset
    # dicts (without gif_palette) working at the historical 256-colour
    # default.
    gif_palette_colors = res_cfg.get("gif_palette", 256)

    # Alpha-modulated RdBu_r cmap. All three viz modes are bipolar (vorticity
    # = curl of u, velocity = speed-minus-inflow, pressure = gauge pressure),
    # so they share the same cmap; only v_clip / cbar labels differ. Built
    # fresh per call so the ListedColormap isn't shared across cache entries.
    _rdbu = plt.get_cmap("RdBu_r")(np.linspace(0.0, 1.0, 256))
    _alpha_t = np.abs(np.linspace(-1.0, 1.0, 256))
    _rdbu[:, 3] = VORT_ALPHA_MAX * _alpha_t ** 1.4
    bipolar_cmap = ListedColormap(_rdbu, name="rdbu_alpha70")
    bipolar_cmap.set_bad((0.0, 0.0, 0.0, 0.0))
    # Alias kept for downstream variable names that still say "vorticity_cmap".
    vorticity_cmap = bipolar_cmap

    # Per-viz-mode calibration. v_clip = colorbar saturation point, vmin/vmax
    # = imshow window. All three modes use the bipolar cmap; only v_clip and
    # the cbar text change. Body interior is later NaN-masked in the per-frame
    # branch (Velocity / Pressure) so it doesn't paint colour through the
    # body patch.
    blur_sigma = VORT_BLUR_SIGMA_BASE + VORT_BLUR_SIGMA_RE_SCALE * np.log10(
        max(reynolds_target / 100.0, 1.0)
    )
    bg_cmap = bipolar_cmap

    if viz_mode == "Vorticity":
        # 92nd percentile of |omega| in fluid cells, with a U/L-scaled floor
        # so low-Re wakes stay visible.
        last_vort_fluid = np.where(mask, np.nan, snapshots[-1]["vorticity"])
        v_clip = max(
            float(np.nanpercentile(np.abs(last_vort_fluid), VORT_CLIP_PERCENTILE)),
            VORT_CLIP_FACTOR * U_INFLOW / max(char_length, 1.0),
        )
        bg_cbar_labels = ["Clockwise spin", "No rotation", "Anti-clockwise spin"]
        bg_cbar_title = "Background heatmap — air's rotation"
        bg_cbar_blurb = (
            "Red = anti-clockwise vortex, blue = clockwise. White = no spin. "
            "The two-coloured 'beads on a string' downstream of bluff bodies "
            "are the von Karman vortex street -- shed alternately off each "
            "side and carried downstream by the flow."
        )
    elif viz_mode == "Velocity":
        # Bipolar: speed - U_INFLOW. v_clip = U_INFLOW so blue saturates at
        # speed=0 (wake) and red saturates at speed=2*U_INFLOW (squeeze).
        v_clip = U_INFLOW
        bg_cbar_labels = [
            "Stalled (slower than wind)",
            "Inflow speed",
            "Accelerated (faster than wind)",
        ]
        bg_cbar_title = "Background heatmap — air's speed vs inflow"
        bg_cbar_blurb = (
            "Blue = slower than the inflow (wake behind bluff bodies, "
            "separated flow above a stalled wing). White = matching the "
            "inflow. Red = accelerated (the squeeze around a bump or the "
            "suction side of an airfoil at AoA)."
        )
    elif viz_mode == "Pressure":
        # Gauge pressure (rho - 1) / 3 with a temporal moving average (see
        # PRESSURE_AVG_FRAMES) to kill LBM acoustic waves. v_clip from the
        # *averaged* last-frame field, floored at U^2/2 (Bernoulli scale).
        _avg_start = max(0, len(snapshots) - PRESSURE_AVG_FRAMES)
        _last_rho_avg = np.mean(
            [s["rho"] for s in snapshots[_avg_start:]], axis=0,
        )
        last_p_fluid = np.where(mask, np.nan, (_last_rho_avg - 1.0) / 3.0)
        v_clip = max(
            float(np.nanpercentile(np.abs(last_p_fluid), VORT_CLIP_PERCENTILE)),
            0.5 * U_INFLOW ** 2,
        )
        bg_cbar_labels = [
            "Low pressure (suction)", "Static", "High pressure (stagnation)",
        ]
        bg_cbar_title = "Background heatmap — air's pressure"
        bg_cbar_blurb = (
            "Red = high pressure (where air piles up against the front of "
            "the body). Blue = low pressure / suction (on the upper surface "
            "of a wing at AoA, or in the cores of shed vortices). The "
            "asymmetry top-vs-bottom on a tilted airfoil is what generates "
            "lift. Averaged over a short rolling window to suppress the "
            "acoustic ripples a single LBM snapshot would otherwise show."
        )
    else:  # already validated above; belt-and-braces
        raise ValueError(f"Unknown viz_mode {viz_mode!r}")

    bg_vmin, bg_vmax = -v_clip, v_clip
    bg_cbar_ticks = [-v_clip, 0.0, v_clip]

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

    # Inflow seed y-positions. Number of rows scales with channel height so
    # Detailed (Ny=240) gets ~20 rows instead of being sampled at the same
    # 8 rows as Standard (which left big gaps).
    n_seed_rows = max(SEED_ROW_MIN, LBM_NY // SEED_ROW_CELL_SPACING)
    inflow_y = np.linspace(4.0, LBM_NY - 5.0, n_seed_rows)

    # Wake-region spawn box. Particles spawn at random (x, y) inside this
    # box every frame so vortices that shed off the body have fresh
    # streakline tracers passing through them.
    # * wake_x_min was BODY_X + char_length (one full body length past
    #   center). On Detailed NACA (chord=100) that left ~50 cells of
    #   empty channel between the trailing edge and the spawn box --
    #   visible as "streams trail off past the leading edge". We now use
    #   char_length/2 + 3: hugs the body's trailing edge at AoA=0 so the
    #   user can see vortices peeling off the moment they form (the most
    #   visually dramatic frames). Spawns that land INSIDE the rotated
    #   body footprint at AoA=45 are still rejected by the mask check
    #   below. The previous +6 buffer left a 3-cell deadzone where the
    #   wake's birth was unsampled -- visually the wake "appeared" two
    #   cells downstream of the body instead of right at the surface.
    # * wake_x_max = LBM_NX * (1 - WAKE_OUTFLOW_FRAC). Last 15 % of the
    #   channel is the trail-off zone: no fresh spawns, but aged wake
    #   particles drift through it and fade out. On Detailed (Nx=960)
    #   that's a 144-cell trail-off vs the old 0.22*Nx (=158-cell) one
    #   the user described as "trails off after the leading edge" --
    #   the channel-end emptiness was from wake_x_max being too far
    #   back, leaving the body-to-mid-channel region under-spawned.
    BODY_X = res_cfg["body_x"]
    wake_x_min = BODY_X + char_length / 2 + 3
    wake_x_max = LBM_NX * (1.0 - WAKE_OUTFLOW_FRAC)
    wake_y_min = LBM_NY * 0.08
    wake_y_max = LBM_NY * 0.92
    n_wake_spawn = max(WAKE_SPAWN_MIN, int(LBM_NX * WAKE_SPAWN_PER_NX))

    # Wake-spawn ramp-up: holds at 0 until inflow particles have nearly
    # reached wake_x_min, then quadratically ramps to full strength.
    # Earlier linear ramp still spawned a visible 2-3 wake particles by
    # frame ~5 -- physically wrong because flow hadn't yet propagated
    # into the wake region. Quadratic ease-in keeps n_wake near zero
    # until well into the ramp window. WAKE_HOLD_FRAC = 0.8 means we
    # don't begin ramping until inflow particles have travelled 80% of
    # the distance from x=3 to wake_x_min; the ramp completes ~10 frames
    # after they fully arrive. Net effect: wake region appears to fill
    # naturally from the flow that reaches it, not from random spawns.
    wake_arrival_frames = max(
        1.0, (wake_x_min - 3.0) / (U_INFLOW * STEPS_PER_FRAME)
    )
    wake_hold_frames = wake_arrival_frames * 0.8
    wake_ramp_window = max(8.0, wake_arrival_frames * 0.2 + 10.0)

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
        cmap=bg_cmap, origin="lower",
        extent=[0, LBM_NX - 1, 0, LBM_NY - 1],
        aspect="equal", interpolation="bicubic",
        interpolation_stage="rgba",
        vmin=bg_vmin, vmax=bg_vmax,
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

    # Pre-stack rho ONCE for Pressure mode's temporal moving average. Saves
    # ~50 ms/frame on Detailed vs the per-frame list-of-arrays-then-stack
    # the simpler implementation needs. Built only when actually needed --
    # the (n_frames, Nx, Ny) float32 array is ~15 MB on Standard, ~135 MB
    # on Detailed and isn't free.
    if viz_mode == "Pressure":
        rho_stack = np.stack(
            [s["rho"] for s in snapshots], axis=0,
        ).astype(np.float32)
    else:
        rho_stack = None  # unused; kept for the closure's static analysis

    gif_frames = []
    master_palette_img = None
    for i, snap in enumerate(snapshots):
        # Per-mode background field. Velocity / Pressure NaN-mask the body
        # AFTER the gaussian filter so the body patch isn't tinted by the
        # field. We do NOT pre-zero the body cells before filtering: for
        # Velocity, body cells naturally hold (speed - U_INFLOW) = -U_INFLOW
        # (since the solid-cell reset gives u = 0); zeroing them would dilute
        # the wake's true negative values when the filter spreads them. For
        # Pressure, body cells hold (rho - 1)/3 = 0 already (solid reset =>
        # rho = 1), so pre-zeroing is a no-op. Vorticity naturally goes to
        # ~0 inside the body so it doesn't need masking, and we keep its
        # render bit-identical to the visual baseline.
        if viz_mode == "Vorticity":
            clipped = np.clip(snap["vorticity"], -3.0 * v_clip, 3.0 * v_clip)
            bg_field = gaussian_filter(clipped, sigma=blur_sigma) * wall_fade
        elif viz_mode == "Velocity":
            speed = np.sqrt(snap["u_x"] ** 2 + snap["u_y"] ** 2)
            raw_field = speed - U_INFLOW
            clipped = np.clip(raw_field, -3.0 * v_clip, 3.0 * v_clip)
            bg_field = gaussian_filter(clipped, sigma=blur_sigma) * wall_fade
            bg_field = np.where(mask, np.nan, bg_field)
        else:  # "Pressure" -- temporal average over last PRESSURE_AVG_FRAMES
            _avg_start = max(0, i - PRESSURE_AVG_FRAMES + 1)
            # Slice + mean from the pre-stacked rho buffer (built once
            # before the render loop) -- vectorised, no per-frame list
            # allocations. Falls back gracefully on the early frames where
            # the window is shorter than PRESSURE_AVG_FRAMES.
            rho_avg = rho_stack[_avg_start:i + 1].mean(axis=0)
            raw_field = (rho_avg - 1.0) / 3.0
            clipped = np.clip(raw_field, -3.0 * v_clip, 3.0 * v_clip)
            bg_field = gaussian_filter(clipped, sigma=blur_sigma) * wall_fade
            bg_field = np.where(mask, np.nan, bg_field)

        # Lightly smooth the velocity for particle advection -- enough
        # to suppress sub-grid LBM oscillations that would jitter the
        # particles, far less than the heavy smoothing streamplot
        # needed (sigma=0.4 here vs 1.5 before).
        u_field = gaussian_filter(snap["u_x"], sigma=0.4)
        v_field = gaussian_filter(snap["u_y"], sigma=0.4)

        im.set_data(bg_field.T)
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

        # (b) Wake-region spawn: spawn at random (x, y) inside the wake box,
        # scaled by a held-then-quadratic ramp so frame 0 spawns 0 wake
        # particles, the next ~20 frames (Standard) / ~50 frames (Detailed)
        # spawn very few, and full strength only kicks in once inflow
        # particles have physically reached and crossed the wake_x_min
        # boundary. Reject any spawn that lands inside the body before
        # adding it to the particle pool (the cull step would catch these
        # on the next frame anyway, but rejecting now avoids a 1-frame
        # visual artifact).
        raw_ramp = max(0.0, (i - wake_hold_frames) / wake_ramp_window)
        wake_ramp = min(1.0, raw_ramp * raw_ramp)  # quadratic ease-in
        n_wake_this_frame = int(round(n_wake_spawn * wake_ramp))
        if n_wake_this_frame > 0:
            wake_x_candidates = spawn_rng.uniform(wake_x_min, wake_x_max, size=n_wake_this_frame)
            wake_y_candidates = spawn_rng.uniform(wake_y_min, wake_y_max, size=n_wake_this_frame)
            xi_cand = np.clip(np.round(wake_x_candidates).astype(np.int32), 0, LBM_NX - 1)
            yi_cand = np.clip(np.round(wake_y_candidates).astype(np.int32), 0, LBM_NY - 1)
            valid_wake = ~mask[xi_cand, yi_cand]
            wake_new_x = wake_x_candidates[valid_wake]
            wake_new_y = wake_y_candidates[valid_wake]
        else:
            wake_new_x = np.empty(0, dtype=np.float64)
            wake_new_y = np.empty(0, dtype=np.float64)

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

    # Background colorbar is per-viz-mode: vorticity gets the bipolar
    # RdBu_r, velocity gets unipolar plasma starting at 0, pressure gets
    # bipolar RdBu_r centred on 0. ticks + labels were assembled in the
    # viz-mode branch above.
    bg_cbar_b = _render_horizontal_cbar(
        cm.ScalarMappable(norm=Normalize(vmin=bg_vmin, vmax=bg_vmax),
                          cmap=bg_cmap),
        ticks=bg_cbar_ticks,
        tick_labels=bg_cbar_labels,
    )
    speed_cbar_b = _render_horizontal_cbar(
        cm.ScalarMappable(norm=speed_norm, cmap=speed_cmap),
        ticks=[0.0, U_INFLOW, u_clip],
        tick_labels=["Stalled (slow)", "Inflow speed", "Accelerated (fast)"],
    )

    return {
        "gif_bytes": gif_buf_local.getvalue(),
        "bg_cbar_bytes": bg_cbar_b,
        "vort_cbar_bytes": bg_cbar_b,  # alias kept for old tests / external scripts
        "bg_cbar_title": bg_cbar_title,
        "bg_cbar_blurb": bg_cbar_blurb,
        "viz_mode": viz_mode,
        "speed_cbar_bytes": speed_cbar_b,
    }


def simulate_and_render(shape_preset, reynolds_target, aoa_deg, res_key,
                         *, progress_callback=None, n_frames=None,
                         custom_polygon=None, viz_mode="Vorticity"):
    """Run LBM and render to GIF + colorbars (legacy convenience wrapper).

    Thin shim over solve_lbm() + render_lbm() that merges their outputs
    into the historical public-API dict shape. New callers that need to
    flip viz_mode without re-paying the LBM solve should call solve_lbm()
    and render_lbm() directly and cache each layer separately.

    Pure function with no Streamlit dependency. The caller owns all I/O.

    Parameters
    ----------
    shape_preset : str
        One of "Cylinder", "Square", "Ellipse", "NACA 0012", "NACA 4412",
        or "Custom" (requires ``custom_polygon`` to be supplied).
    custom_polygon : np.ndarray or None
        Shape (N, 2) polygon in image-pixel coords (PIL convention, y down).
        Required when shape_preset == "Custom"; ignored otherwise. The
        rasterizer centres, scales, flips, and rotates it onto the LBM grid
        using res_cfg["custom_extent"] cells as the longest-axis size.
    reynolds_target : float
        Reynolds number based on the body's characteristic length (diameter
        for cylinder/ellipse, side for square, chord for airfoil).
    aoa_deg : float
        Body rotation / wing tilt in degrees. Ignored for Cylinder.
    res_key : str
        One of the keys in RESOLUTION_PRESETS ("Standard (320 x 80)" or
        "Detailed (960 x 240)").
    progress_callback : callable or None
        Signature ``(fraction: float in [0, 1], text: str) -> None``.
        Defaults to a no-op for headless use.
    n_frames : int or None
        Overrides the per-preset frame count. Used by end-to-end tests to
        run a 5-frame pipeline in ~5 s instead of 60/100. Production
        callers leave it None.
    viz_mode : str
        Background heatmap selection. One of VIZ_MODES:
          * "Vorticity" -- bipolar red/blue, fluid rotation (curl of u).
          * "Velocity"  -- bipolar speed-minus-inflow: blue = slower than
            freestream (wake), white = matching, red = faster (squeeze).
          * "Pressure"  -- bipolar gauge pressure, temporal-averaged over
            ~5 frames to suppress the LBM acoustic waves a single
            snapshot would otherwise show.
        Particles + scale bars + body outline are unchanged across modes.

    Returns
    -------
    dict
        gif_bytes, vort_cbar_bytes, speed_cbar_bytes : bytes for st.image
        force_plot_bytes                             : PNG of Cd/Cl vs time
        cd_history, cl_history                       : float32 arrays
                                                       (length = n_steps)
        cd_mean, cl_mean                             : floats, mean over the
                                                       last third of the run
        strouhal                                     : float (NaN if too few
                                                       samples for FFT)
        label                                        : short title string
        tau, nu, char_length                         : physics scalars
        lbm_nx, lbm_ny, n_frames, n_steps            : grid + sim sizes
        near_stable                                  : bool (tau < 0.51)
    """
    solve = solve_lbm(
        shape_preset, reynolds_target, aoa_deg, res_key,
        n_frames=n_frames, custom_polygon=custom_polygon,
        progress_callback=progress_callback,
    )
    render = render_lbm(
        solve, viz_mode=viz_mode, progress_callback=progress_callback,
    )
    # Merge into the legacy public-API shape. Drop internal-only keys
    # (snapshots, mask, body_xs/ys, reynolds_target, res_key) so external
    # callers (tests, scripts) don't see them.
    result = {
        k: v for k, v in solve.items()
        if k not in (
            "snapshots", "mask", "body_xs", "body_ys",
            "reynolds_target", "res_key",
        )
    }
    result.update(render)
    return result
