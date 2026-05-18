"""AeroLab Streamlit app -- dual-mode airfoil aerodynamics.

Two modes via the sidebar toggle:
  - Fast (NeuralFoil): instant ML predictions, alpha sweeps, drag polar.
  - Real CFD (LBM): browser-based Lattice Boltzmann simulation (wired in Stage B).

Run from the project root:
    streamlit run app.py

The browser opens automatically at http://localhost:8501.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.airfoils import analyze_airfoil, get_airfoil

# --- Page config (must be the first Streamlit call) ---
st.set_page_config(page_title="AeroLab", layout="wide")

# Subtle wordmark; each mode below sets its own hero title.
st.markdown(
    "<div style='display:flex;align-items:center;gap:0.6rem;"
    "padding:0.2rem 0 0.4rem 0;color:#94a3b8;font-size:0.9rem;"
    "letter-spacing:0.04em;text-transform:uppercase;'>"
    "<span>AeroLab</span>"
    "<span style='opacity:0.4'>·</span>"
    "<span style='opacity:0.7'>browser-based aerodynamics</span>"
    "</div>",
    unsafe_allow_html=True,
)

# --- Mode toggle (sidebar, top) ---
with st.sidebar:
    st.markdown("### :material/tune: Mode")
    mode = st.radio(
        "Simulation mode",
        ["Fast (NeuralFoil)", "Real CFD (LBM)"],
        index=0,
        label_visibility="collapsed",
        help=(
            "**Fast**: instant ML predictions for NACA airfoils -- great "
            "for sweeping lots of cases.  "
            "**Real CFD**: full Lattice Boltzmann simulation -- watch the air "
            "actually move."
        ),
    )
    st.divider()

# --- Real CFD (LBM) mode: animated GIF playback of LBM run ---
if mode == "Real CFD (LBM)":
    # Lazy imports: keep Fast mode's cold-start untouched by Numba + matplotlib.
    import io

    import matplotlib as mpl
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
    from matplotlib.patches import Polygon
    from PIL import Image
    from scipy.ndimage import gaussian_filter, zoom

    # Modern system sans-serif for in-figure text. matplotlib picks the first
    # font from this list that's actually installed -- Segoe UI Variable on
    # Windows (Streamlit's host OS here), Inter / SF Pro on other systems,
    # DejaVu Sans as a guaranteed fallback so this never errors at render.
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = [
        "Segoe UI Variable", "Segoe UI", "Inter", "SF Pro Text",
        "Helvetica Neue", "Arial", "DejaVu Sans",
    ]

    from src.lbm import CS2, equilibrium, macroscopic, step_njit_mrt_with_force
    from src.shapes import (
        cylinder_mask, ellipse_mask, naca4_airfoil_mask, naca4_outline_xy,
        square_mask,
    )

    # --- Constants ---
    # Frame schedule: 75 frames at 40 steps each = 3000 recorded steps. Smaller
    # motion per frame (4 cells at U=0.1) gives fluider streamline playback.
    #
    # WARMUP_STEPS run BEFORE frame recording starts, with the kick applied
    # inside the warmup. By the time frame 0 is captured the wake is fully
    # developed -- so the animation doesn't open on "uniform flow, no wake"
    # and then visibly grow the wake during the first 30 frames (which the
    # user perceives as a glitchy start). The cost is a one-time ~6 s extra
    # in the simulate phase.
    WARMUP_STEPS = 1500
    N_FRAMES = 75
    STEPS_PER_FRAME = 40
    N_STEPS = WARMUP_STEPS + N_FRAMES * STEPS_PER_FRAME
    U_INFLOW = 0.1

    # Two grid/body presets. Detailed = 2.1x more cells, ~2x render time, but
    # bodies scale 1.5x so the wake has noticeably more discretized detail.
    RESOLUTION_PRESETS = {
        "Standard (320 x 100)": dict(
            Nx=320, Ny=100, body_x=70, cy=50,
            cylinder_D=20, square_side=20,
            ellipse_a=22, ellipse_b=11, chord=44,
        ),
        "Detailed (480 x 140)": dict(
            Nx=480, Ny=140, body_x=105, cy=70,
            cylinder_D=30, square_side=30,
            ellipse_a=33, ellipse_b=17, chord=66,
        ),
    }

    INFLOW_DIRS = np.array([1, 5, 8], dtype=np.int32)
    OUTFLOW_DIRS = np.array([3, 6, 7], dtype=np.int32)

    KICK_START, KICK_END = 100, 500
    KICK_AMPLITUDE = 0.005
    KICK_Y_OFFSET = 2

    BG_COLOR = "#0a0a0a"
    BODY_COLOR = "#1f2937"                     # slate-800
    # Body patch is drawn slightly larger than the voxelized mask to hide the
    # staircase-stepped boundary cells and the bright boundary-layer vorticity
    # ring that hugs them. ~1.5 cells of outward dilation is enough across all
    # presets without visibly fattening the body shape.
    BODY_OUTLINE_MARGIN = 1.5
    STREAMLINE_WIDTH = 1.4
    # Custom cyan -> pink -> yellow cmap. All three stops are luminous so the
    # speed encoding stays readable across the vorticity heatmap underneath.
    SPEED_CMAP = LinearSegmentedColormap.from_list(
        "aerolab_speed", ["#22d3ee", "#ff5e8a", "#fde047"],
    )
    SPEED_CLIP_FACTOR = 2.0
    # Vorticity heatmap: red/blue diverging, alpha-modulated so omega ~ 0 is
    # transparent and the dark background shows through. Max alpha capped at
    # 0.7 so the wake reads as a 70%-opacity wash rather than fully saturated
    # red/blue.
    VORT_ALPHA_MAX = 0.7
    # v_clip blends two things:
    #   1. A floor from the inflow / characteristic-length scale (so low-Re
    #      wakes with weak omega still saturate to visible color).
    #   2. The 75th percentile of |omega| in the fluid (so high-Re wakes
    #      with order-of-magnitude stronger omega scale up automatically).
    # A pure 92nd percentile was dominated by boundary-layer extremes and
    # the wake washed out at high Re; the 75th percentile sits in the wake
    # body and gives a clean colour-scale that tracks the regime.
    VORT_CLIP_FACTOR = 1.5
    VORT_CLIP_PERCENTILE = 92
    # Gaussian sigma applied to vorticity before upsampling. Scales with
    # log10(Re) so high-Re sub-grid wake aliasing (which looks like noise on
    # a 320x100 grid above Re~600) gets aggressively smoothed -- the wake's
    # large-scale shape stays, the noise washes out.
    VORT_BLUR_SIGMA_BASE = 1.0
    VORT_BLUR_SIGMA_RE_SCALE = 1.6
    # Vertical alpha fade at the top/bottom walls. The bounce-back walls
    # produce a thin no-slip boundary layer at y=0 and y=Ny-1 which the
    # vorticity heatmap renders as red/blue bands. The fade cosmetically
    # zeros out the heatmap within this many cells of each wall so the
    # visual matches a freestream box -- physics is unchanged.
    WALL_FADE_CELLS = 14
    # We rely on matplotlib's own bicubic interpolation_stage="rgba" to smooth
    # the colored field at display resolution. Pre-upsampling the source array
    # with scipy.ndimage.zoom barely improved appearance over native + bicubic
    # but cost 5-10ms per frame in the hot render loop -- skipped now.
    VORT_UPSAMPLE = 1
    # Light gaussian smoothing on the velocity field that drives streamplot.
    # Streamlines are integral curves of u, so jitter in u between frames
    # makes streamlines wobble. Pre-smoothing u gives noticeably more fluid
    # streamline motion in playback at moderate cost (~0.5 ms / frame).
    STREAM_BLUR_SIGMA = 0.8
    TEXT_COLOR = "#f5f5f5"
    GIF_FRAME_MS = 67                          # ~15 fps (0.75x of the 20 fps cap)
    FIG_W_IN, FIG_H_IN, FIG_DPI = 10.0, 3.0, 90      # 900x270 px -- balances crispness and per-frame render cost

    # Friendly display name -> internal preset key
    SHAPE_PRESETS = {
        "Cylinder  (round pipe)": "Cylinder",
        "Square  (boxy)": "Square",
        "Ellipse  (stretched oval)": "Ellipse",
        "NACA 0012  (symmetric wing)": "NACA 0012",
        "NACA 4412  (curved wing)": "NACA 4412",
    }

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

    def regime_label(re):
        if re <= 100:
            return "slow viscous flow", "honey-like"
        if re <= 175:
            return "moderate flow", "syrup-like"
        if re <= 400:
            return "transitional flow", "water-like"
        if re <= 800:
            return "early turbulent flow", "stirred-coffee"
        return "fully turbulent flow", "wind-tunnel"

    def tilt_label(deg):
        if deg <= -2:
            return f"tilted {abs(deg):.1f} deg nose-down"
        if deg < 2:
            return "roughly level with the wind"
        if deg <= 10:
            return f"tilted {deg:.1f} deg into the wind"
        return f"tilted {deg:.1f} deg -- approaching stall"

    # === Sidebar ===
    with st.sidebar:
        st.markdown("### :material/tune: Simulation setup")

        st.markdown(":material/category: **Shape**")
        shape_display = st.selectbox(
            "Shape preset",
            list(SHAPE_PRESETS.keys()),
            index=0,
            label_visibility="collapsed",
            help=(
                "What the wind flows past. Round and boxy shapes shed swirly "
                "wakes (think behind a bridge column). Wing shapes glide more "
                "smoothly. Try them and see the difference."
            ),
        )
        shape_preset = SHAPE_PRESETS[shape_display]

        st.markdown("")
        st.markdown(":material/speed: **Flow speed** &nbsp; :gray[(Reynolds number)]")
        reynolds_target = st.slider(
            "Reynolds number",
            min_value=50, max_value=1500, value=200, step=50,
            label_visibility="collapsed",
            help=(
                "How fast the air moves *relative to the object size*. "
                "Low Re = thick, syrupy flow (everything is gentle). "
                "High Re = thin, fast flow (chaotic swirls, turbulent wakes). "
                "Real airplane wings: millions. Our simulation: 50-1500. "
                "Above ~500 the BGK solver gets twitchy -- the wake structure "
                "stays qualitatively right but quantitative drag numbers "
                "drift high. Stress-test the solver and watch what happens."
            ),
        )
        reg, reg_feel = regime_label(reynolds_target)
        st.caption(f"Now showing **{reg}** -- air feels {reg_feel}")

        if shape_preset == "Cylinder":
            # A circle is rotationally invariant -- no point exposing a slider.
            aoa_deg = 0.0
        else:
            st.markdown("")
            is_airfoil = shape_preset in ("NACA 0012", "NACA 4412")
            if is_airfoil:
                st.markdown(":material/rotate_right: **Wing tilt** "
                            "&nbsp; :gray[(angle of attack)]")
                slider_min, slider_max, slider_default = -10.0, 20.0, 5.0
                slider_help = (
                    "How steeply the wing is angled into the wind. "
                    "More tilt = more lift -- but go too steep and the wing "
                    "**stalls** (lift collapses, drag spikes). Try +5 deg vs "
                    "+15 deg and watch the wake on top change."
                )
            else:
                st.markdown(":material/rotate_right: **Rotation** "
                            "&nbsp; :gray[(body angle vs. wind)]")
                slider_min, slider_max, slider_default = -45.0, 45.0, 0.0
                slider_help = (
                    "Rotate the body relative to the oncoming wind. A square "
                    "at 0 deg presents a flat face (huge wake, high drag); at "
                    "45 deg it's a diamond, with a much sharper leading edge. "
                    "An ellipse rotated end-on slips through the air; rotated "
                    "broadside it slams into it. Try the extremes."
                )
            aoa_deg = st.slider(
                "Body angle",
                min_value=slider_min, max_value=slider_max,
                value=slider_default, step=0.5,
                label_visibility="collapsed",
                help=slider_help,
            )
            if is_airfoil:
                st.caption(f"Wing {tilt_label(aoa_deg)}")
            elif abs(aoa_deg) < 0.25:
                st.caption("Body aligned with the wind")
            else:
                st.caption(f"Rotated {aoa_deg:+.1f} deg from horizontal")

        st.markdown("")
        st.markdown(":material/grid_view: **Resolution**")
        res_display = st.radio(
            "Resolution",
            list(RESOLUTION_PRESETS.keys()),
            index=0,
            label_visibility="collapsed",
            help=(
                "**Standard** runs in ~20s and is great for getting a feel. "
                "**Detailed** uses a 2x larger grid with bigger bodies, so "
                "the wake has more discretized swirls and finer streamline "
                "structure -- but takes ~45s per run."
            ),
        )
        res_cfg = RESOLUTION_PRESETS[res_display]

        st.markdown("---")
        run_clicked = st.button(
            ":material/play_arrow: &nbsp; **Run simulation**",
            type="primary", use_container_width=True,
        )
        if "Standard" in res_display:
            st.caption(":material/timer: First run ~50s (one-time compile). "
                       "Later runs ~20s.")
        else:
            st.caption(":material/timer: First run ~75s. Later runs ~45s.")

    # === Main page header ===
    st.title("Real CFD")
    st.markdown(
        "##### Watch how air actually moves around a shape -- "
        "the same physics that lets airplanes fly, slows cars down, "
        "and once tore a bridge apart."
    )

    if not run_clicked:
        with st.container(border=True):
            st.markdown(
                f"### :material/play_circle: Ready to run\n\n"
                f":material/arrow_back: **Set the inputs in the sidebar** "
                f"and press **Run simulation**.\n\n"
                f"A {res_cfg['Nx']} x {res_cfg['Ny']} Lattice Boltzmann "
                f"simulation runs {N_STEPS:,} steps, then plays the result "
                f"back as a smooth 20 fps animation. On the *first* click in "
                f"a fresh session the solver compiles itself (one-time, "
                f"about 30 extra seconds). Every click after that is just "
                f"the simulation time."
            )
        with st.expander(
                ":material/lightbulb: What you'll see -- a quick primer"):
            st.markdown(
                "The animation shows the air's flow on a dark canvas:\n\n"
                "- :material/rotate_left: **Red wash** = anti-clockwise spin\n"
                "- :material/rotate_right: **Blue wash** = clockwise spin\n"
                "- :material/timeline: **Glowing curves** = the direction air is "
                "moving (streamlines), **coloured by speed**: cyan = slow, "
                "yellow = fast.\n"
                "- :material/circle: **Dark shape** = the object\n\n"
                "Watch what happens *behind* the object. With a cylinder at "
                "moderate flow speeds, you'll see swirls peel off alternately "
                "from the top and bottom -- a **Karman vortex street**. It's "
                "the same pattern that makes telephone wires hum in the wind."
            )
        st.stop()

    # === Grid + body geometry from resolution preset ===
    LBM_NX = res_cfg["Nx"]
    LBM_NY = res_cfg["Ny"]
    BODY_X = res_cfg["body_x"]
    CY_CENTER = res_cfg["cy"]

    if shape_preset == "Cylinder":
        D = res_cfg["cylinder_D"]
        mask = cylinder_mask(LBM_NX, LBM_NY, cx=BODY_X, cy=CY_CENTER, radius=D // 2)
        char_length = D
        kick_x = BODY_X + D
        label = "Cylinder"
    elif shape_preset == "Square":
        side = res_cfg["square_side"]
        mask = square_mask(
            LBM_NX, LBM_NY, cx=BODY_X, cy=CY_CENTER, side=side, aoa_deg=aoa_deg,
        )
        char_length = side
        # A rotated square extends side*sqrt(2)/2 ~ 0.71*side from center along
        # each diagonal -- pad kick by 1.5x side so it lands well past the body
        # at any rotation angle in the slider range.
        kick_x = BODY_X + int(side * 1.5)
        label = "Square" if abs(aoa_deg) < 0.25 else f"Square  ·  {aoa_deg:+.1f}° rotation"
    elif shape_preset == "Ellipse":
        a, b = res_cfg["ellipse_a"], res_cfg["ellipse_b"]
        mask = ellipse_mask(
            LBM_NX, LBM_NY, cx=BODY_X, cy=CY_CENTER, a=a, b=b, aoa_deg=aoa_deg,
        )
        char_length = 2 * b
        kick_x = BODY_X + a + 10
        label = "Ellipse" if abs(aoa_deg) < 0.25 else f"Ellipse  ·  {aoa_deg:+.1f}° rotation"
    else:                                      # NACA 0012 or 4412
        chord = res_cfg["chord"]
        naca_code = shape_preset.split()[1]
        mask = naca4_airfoil_mask(
            LBM_NX, LBM_NY, cx=BODY_X, cy=CY_CENTER,
            chord=chord, naca_code=naca_code, aoa_deg=aoa_deg,
        )
        char_length = chord
        kick_x = BODY_X + chord + 10
        label = f"{shape_preset}  ·  {aoa_deg:+.1f}° wing tilt"

    nu = U_INFLOW * char_length / reynolds_target
    tau = nu / CS2 + 0.5

    if tau < 0.51:
        st.warning(
            ":material/warning: At this Reynolds number the MRT solver is "
            "right at the edge of its stable range (tau approaching 0.5). "
            "The wake structure stays qualitatively correct but small-scale "
            "turbulence is under-resolved on this grid. Try **Detailed** "
            "resolution for sharper structure, or step the Reynolds slider "
            "down for cleaner physics."
        )

    # Alpha-modulated RdBu_r: blue = clockwise, red = anti-clockwise rotation.
    # Alpha grows quadratically from omega=0 (transparent) to peak, capped at
    # VORT_ALPHA_MAX (0.7) so the wake reads as a 70%-opacity wash and the
    # dark background remains visible through it. Smoother alpha profile than
    # the prior linear ramp -- avoids sharp halo edges at the wake boundary.
    _rdbu = plt.get_cmap("RdBu_r")(np.linspace(0.0, 1.0, 256))
    _alpha_t = np.abs(np.linspace(-1.0, 1.0, 256))
    _rdbu[:, 3] = VORT_ALPHA_MAX * _alpha_t ** 1.4
    vorticity_cmap = ListedColormap(_rdbu, name="rdbu_alpha70")
    # We do NOT NaN-mask the body cells anymore -- the body patch is drawn on
    # top of the full heatmap with the outward margin, hiding the boundary
    # layer cleanly. Setting set_bad still serves as a safety net.
    vorticity_cmap.set_bad((0.0, 0.0, 0.0, 0.0))

    body_xs, body_ys = body_outline_xy(shape_preset, res_cfg, aoa_deg)
    body_xs, body_ys = expand_outline(body_xs, body_ys, BODY_OUTLINE_MARGIN)

    ds = 2
    xs_ds = np.arange(0, LBM_NX, ds)
    ys_ds = np.arange(0, LBM_NY, ds)

    # Deterministic inflow column. Forward-only integration from here yields
    # smooth, naturally-deflected streamlines that flow past the body and
    # slow / curve through the wake -- mirroring what dye lines look like in
    # a real wind-tunnel shot.
    #
    # We deliberately do NOT add wake seeds with bidirectional integration:
    # in chaotic recirculation, streamlines through fixed seeds are
    # hyper-sensitive to small velocity changes, which made the animation
    # look "jumpy" frame-to-frame. Rotation is conveyed by the heatmap;
    # streamlines just show the path and speed.
    n_inflow = max(LBM_NY // 12, 8)
    inflow_y = np.linspace(4, LBM_NY - 5, n_inflow)
    stream_seeds = np.column_stack([np.full(n_inflow, 3.0), inflow_y])

    # === Phase 1: simulate, store snapshots ===
    rho0 = np.ones((LBM_NX, LBM_NY))
    u0 = np.zeros((2, LBM_NX, LBM_NY))
    u0[0] = U_INFLOW
    f = equilibrium(rho0, u0)
    f_inflow_eq = equilibrium(1.0, np.array([U_INFLOW, 0.0]))
    kick_y = CY_CENTER + KICK_Y_OFFSET

    progress = st.progress(
        0.0, text=":material/sync: Phase 1 of 2 -- warming up flow...",
    )
    snapshots = []
    step_counter = 0
    # === Warmup: develop the wake before recording starts ===
    # The kick is applied during this phase, so by the time we start
    # capturing frames the Karman shedding is in its periodic regime
    # and the animation doesn't open on a wake-less freestream.
    warmup_report_every = max(WARMUP_STEPS // 20, 1)
    for step in range(WARMUP_STEPS):
        f, _Fx, _Fy = step_njit_mrt_with_force(
            f, tau, mask, f_inflow_eq, INFLOW_DIRS, OUTFLOW_DIRS,
        )
        if KICK_START <= step_counter < KICK_END:
            f[2, kick_x, kick_y] += KICK_AMPLITUDE
            f[4, kick_x, kick_y] -= KICK_AMPLITUDE
        step_counter += 1
        if (step + 1) % warmup_report_every == 0:
            progress.progress(
                0.5 * (step + 1) / N_STEPS,
                text=(
                    f":material/sync: Phase 1 of 2 -- warming up flow "
                    f"({step + 1:,} / {WARMUP_STEPS:,} steps)"
                ),
            )

    # === Record: every frame is a fully-developed snapshot ===
    for frame in range(N_FRAMES):
        for _ in range(STEPS_PER_FRAME):
            f, _Fx, _Fy = step_njit_mrt_with_force(
                f, tau, mask, f_inflow_eq, INFLOW_DIRS, OUTFLOW_DIRS,
            )
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
        progress.progress(
            0.5 * step_counter / N_STEPS,
            text=(
                f":material/sync: Phase 1 of 2 -- "
                f"recording frame {frame + 1} / {N_FRAMES}"
            ),
        )

    # Blended v_clip: 75th percentile of |omega| in the fluid (tracks the
    # wake's actual strength across the Re range), with a U/L-scaled floor
    # so very weak low-Re wakes still register against the heatmap.
    last_vort_fluid = np.where(mask, np.nan, snapshots[-1]["vorticity"])
    v_clip = max(
        float(np.nanpercentile(np.abs(last_vort_fluid), VORT_CLIP_PERCENTILE)),
        VORT_CLIP_FACTOR * U_INFLOW / max(char_length, 1.0),
    )

    # Re-adaptive blur sigma: smooths sub-grid wake aliasing at high Re
    # without over-washing the laminar Re=100 wake.
    blur_sigma = VORT_BLUR_SIGMA_BASE + VORT_BLUR_SIGMA_RE_SCALE * np.log10(
        max(reynolds_target / 100.0, 1.0)
    )

    # Pre-compute the wall-fade weight on the upsampled grid. Smoothstep
    # from 0 (at the wall) to 1 (WALL_FADE_CELLS into the fluid).
    fade_hires = WALL_FADE_CELLS * VORT_UPSAMPLE
    ny_hi = LBM_NY * VORT_UPSAMPLE
    nx_hi = LBM_NX * VORT_UPSAMPLE
    y_hi = np.arange(ny_hi)
    edge_dist = np.minimum(y_hi, ny_hi - 1 - y_hi) / fade_hires
    t = np.clip(edge_dist, 0.0, 1.0)
    fade_1d = t * t * (3.0 - 2.0 * t)
    # Broadcast onto (nx_hi, ny_hi). Stored as (1, ny_hi); broadcasts across x.
    wall_fade = fade_1d[None, :]

    # Speed colorbar: clip at 2x inflow so the inflow speed lands at the
    # midpoint and "slower / faster than freestream" reads symmetrically across
    # every shape. Local hot spots above 2 x U_INFLOW saturate to yellow --
    # acceptable for visualization, the goal is showing the gradient clearly.
    u_clip = SPEED_CLIP_FACTOR * U_INFLOW
    speed_cmap = SPEED_CMAP
    speed_norm = Normalize(vmin=0.0, vmax=u_clip)

    # === Phase 2: render frames into a reused figure ===
    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), dpi=FIG_DPI,
                            facecolor=BG_COLOR)
    # Title moved to an in-axes chip; reclaim the top padding for visualization.
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.99)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, LBM_NX - 1)
    ax.set_ylim(0, LBM_NY - 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Bicubic interpolation on a 3x upsampled, gaussian-blurred source. The
    # `interpolation_stage="rgba"` flag tells matplotlib to apply the colormap
    # FIRST then interpolate in RGBA space, which smooths the alpha channel
    # across the wake edges -- no more hard-edged voxel halos at omega = 0.
    im = ax.imshow(
        np.zeros((LBM_NY * VORT_UPSAMPLE, LBM_NX * VORT_UPSAMPLE)),
        cmap=vorticity_cmap, origin="lower",
        extent=[0, LBM_NX - 1, 0, LBM_NY - 1],
        aspect="equal", interpolation="bicubic",
        interpolation_stage="rgba",
        vmin=-v_clip, vmax=v_clip,
    )
    # Floating info chip in the top-left corner -- replaces the old centered
    # title with something that looks like a modern UI label instead of a
    # matplotlib figure heading.
    title_text = ax.text(
        0.012, 0.93, "", transform=ax.transAxes,
        ha="left", va="top",
        color=TEXT_COLOR, fontsize=10.5, fontweight="600",
        bbox=dict(
            boxstyle="round,pad=0.45,rounding_size=0.35",
            facecolor=(0.094, 0.110, 0.165, 0.78),    # slate-900 @ 78% alpha
            edgecolor=(0.482, 0.541, 0.643, 0.65),    # slate-400 hairline
            linewidth=0.6,
        ),
        zorder=25,
    )

    # Smooth analytic body outline drawn ONCE on top of imshow + streamlines.
    # Filled, no edge stroke -- the patch is sized with BODY_OUTLINE_MARGIN
    # extra cells so it cleanly covers (a) the voxelized mask staircase and
    # (b) the high-vorticity boundary-layer ring. zorder = 10 keeps it above
    # streamlines (default zorder ~5).
    body_patch = Polygon(
        np.column_stack([body_xs, body_ys]),
        closed=True, facecolor=BODY_COLOR, edgecolor="none",
        antialiased=True, zorder=10,
    )
    ax.add_patch(body_patch)

    gif_frames = []
    for i, snap in enumerate(snapshots):
        # Smooth the vorticity field (kill voxel artifacts at the grid scale),
        # then 3x upsample so the colorbar gradient looks continuous instead
        # of stair-stepped. Body cells are NOT masked here -- the body patch
        # is drawn on top with its outward margin to cover them cleanly.
        # Pre-clip omega to 3x v_clip, blur to kill grid-scale aliasing, then
        # apply the wall fade so the bounce-back boundary layers don't show.
        vort_clipped = np.clip(snap["vorticity"], -3.0 * v_clip, 3.0 * v_clip)
        vort_smooth = gaussian_filter(vort_clipped, sigma=blur_sigma) * wall_fade
        # Velocity field for streamplot: blur in numpy first so streamlines
        # are integral curves of a smoothed u, which moves more fluidly
        # between frames at high Re.
        u_x_blurred = gaussian_filter(snap["u_x"], sigma=STREAM_BLUR_SIGMA)
        u_y_blurred = gaussian_filter(snap["u_y"], sigma=STREAM_BLUR_SIGMA)
        u_x_plot = np.where(mask, 0.0, u_x_blurred)
        u_y_plot = np.where(mask, 0.0, u_y_blurred)
        u_mag = np.sqrt(u_x_blurred ** 2 + u_y_blurred ** 2)
        u_mag = np.where(mask, 0.0, u_mag)

        im.set_data(vort_smooth.T)
        # Title is static (no frame counter) -- clean look across the loop.
        title_text.set_text(f"{label}  ·  Re = {reynolds_target}")

        # Streamplot artists must be torn down + rebuilt per frame.
        for col in list(ax.collections):
            col.remove()
        # Keep the persistent body Polygon -- only remove streamplot arrow patches.
        for patch in list(ax.patches):
            if patch is not body_patch:
                patch.remove()

        # Streamlines: no path-effects halo. The halo was producing white-ish
        # artifacts where it interacted with the cmap arrowhead rendering, and
        # the speed cmap is luminous enough on its own that the halo isn't
        # needed for contrast on the red/blue background.
        ax.streamplot(
            xs_ds, ys_ds,
            u_x_plot[::ds, ::ds].T, u_y_plot[::ds, ::ds].T,
            start_points=stream_seeds,
            integration_direction="forward",
            density=2.0,
            color=u_mag[::ds, ::ds].T,
            cmap=speed_cmap, norm=speed_norm,
            linewidth=STREAMLINE_WIDTH, arrowsize=0.9,
        )

        fig.canvas.draw()
        img_rgba = np.asarray(fig.canvas.buffer_rgba())
        img_p = Image.fromarray(img_rgba[..., :3]).quantize(
            colors=128, method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.FLOYDSTEINBERG,
        )
        gif_frames.append(img_p)

        progress.progress(
            0.5 + 0.5 * (i + 1) / N_FRAMES,
            text=(
                f":material/auto_awesome: Phase 2 of 2 -- "
                f"rendering frame {i + 1} of {N_FRAMES}"
            ),
        )

    plt.close(fig)

    # --- Encode animated GIF + colorbar with plain-English tick labels ---
    progress.progress(1.0, text=":material/movie: Encoding animation...")
    gif_buf = io.BytesIO()
    gif_frames[0].save(
        gif_buf, format="GIF",
        save_all=True, append_images=gif_frames[1:],
        duration=GIF_FRAME_MS, loop=0, optimize=True, disposal=2,
    )

    def _render_horizontal_cbar(scalar_mappable, ticks, tick_labels):
        cfig, cax = plt.subplots(figsize=(8.5, 0.55), dpi=FIG_DPI,
                                  facecolor=BG_COLOR)
        cfig.subplots_adjust(left=0.06, right=0.94, bottom=0.55, top=0.95)
        cbar = cfig.colorbar(scalar_mappable, cax=cax, orientation="horizontal")
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)
        cbar.ax.xaxis.set_tick_params(color=TEXT_COLOR, labelcolor=TEXT_COLOR,
                                       labelsize=9, length=0, pad=4)
        # Flat / borderless: no axis outline, no tick marks -- just the gradient.
        cbar.outline.set_visible(False)
        buf = io.BytesIO()
        cfig.savefig(buf, format="png", facecolor=BG_COLOR, dpi=FIG_DPI)
        plt.close(cfig)
        return buf

    # Vorticity colorbar -- red = anti-clockwise, blue = clockwise. The cmap
    # is alpha-modulated and capped at 70% opacity so even at the extremes
    # the colorbar reads as washed red / washed blue, matching the heatmap.
    vort_cbar_buf = _render_horizontal_cbar(
        cm.ScalarMappable(norm=Normalize(vmin=-v_clip, vmax=v_clip),
                          cmap=vorticity_cmap),
        ticks=[-v_clip, 0.0, v_clip],
        tick_labels=["Clockwise spin", "No rotation", "Anti-clockwise spin"],
    )
    # Speed colorbar (streamline colors). Inflow lands at the midpoint thanks
    # to u_clip = 2 * U_INFLOW above, so the labels read symmetrically.
    speed_cbar_buf = _render_horizontal_cbar(
        cm.ScalarMappable(norm=speed_norm, cmap=speed_cmap),
        ticks=[0.0, U_INFLOW, u_clip],
        tick_labels=["Stalled (slow)", "Inflow speed", "Accelerated (fast)"],
    )

    progress.empty()

    # === Display: hero animation, colorbar, plain-English legend ===
    st.markdown("---")
    shape_name = shape_display.split("  (")[0]
    if shape_preset in ("NACA 0012", "NACA 4412"):
        st.markdown(
            f"### :material/air: {shape_name}, {reg}  ·  "
            f"wing tilt {aoa_deg:+.1f} deg"
        )
    else:
        st.markdown(f"### :material/air: {shape_name} in {reg}")

    with st.container(border=True):
        st.image(gif_buf.getvalue(), use_container_width=True)
        st.markdown(
            "<div style='color:#94a3b8;font-size:0.78rem;"
            "letter-spacing:0.05em;text-transform:uppercase;"
            "margin:0.4rem 0 0.1rem 0;'>"
            "Background heatmap — air's rotation"
            "</div>",
            unsafe_allow_html=True,
        )
        st.image(vort_cbar_buf.getvalue(), use_container_width=True)
        st.markdown(
            "<div style='color:#94a3b8;font-size:0.78rem;"
            "letter-spacing:0.05em;text-transform:uppercase;"
            "margin:0.4rem 0 0.1rem 0;'>"
            "Streamline colors — air's speed"
            "</div>",
            unsafe_allow_html=True,
        )
        st.image(speed_cbar_buf.getvalue(), use_container_width=True)

    st.markdown("##### :material/visibility: What you're looking at")
    leg_cols = st.columns(4)
    with leg_cols[0]:
        st.markdown(
            ":material/rotate_left: **Red wash**\n\n"
            "Air rotating *anti-clockwise* (counter-clockwise) -- vortices "
            "spinning one way."
        )
    with leg_cols[1]:
        st.markdown(
            ":material/rotate_right: **Blue wash**\n\n"
            "Air rotating *clockwise* -- vortices spinning the other way. "
            "Together they form the Karman street."
        )
    with leg_cols[2]:
        st.markdown(
            ":material/timeline: **Glowing lines**\n\n"
            "Direction the air is moving, **coloured by speed**: "
            "cyan = slow / recirculating, pink = inflow speed, "
            "yellow = accelerated."
        )
    with leg_cols[3]:
        st.markdown(
            ":material/circle: **Dark shape**\n\n"
            "The object. Air can't flow through it -- it has to go around."
        )

    # === Metric strip ===
    st.markdown("")
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric(":material/speed: Flow speed", f"Re {reynolds_target}", reg)
    with metric_cols[1]:
        st.metric(":material/footprint: Simulation steps", f"{N_STEPS:,}")
    with metric_cols[2]:
        st.metric(":material/movie: Playback",
                   f"{N_FRAMES} frames @ {round(1000 / GIF_FRAME_MS)} fps")
    with metric_cols[3]:
        st.metric(":material/calculate: Solver tau", f"{tau:.3f}",
                   help="LBM relaxation time; >0.5 for stability. Lower = faster flow.")

    # === Why this matters ===
    with st.expander(
            ":material/lightbulb: **Why this pattern matters in the real world**"):
        st.markdown(
            "The alternating red/blue swirls trailing the body are a "
            "**Karman vortex street** -- a flow pattern Theodore von Karman "
            "described in 1911. It shows up at human scales every day:\n\n"
            "- :material/cable: **Telephone wires hum in the wind.** At certain "
            "wind speeds, the swirls shed off the wire at audible frequencies, "
            "vibrating the cable like a string.\n"
            "- :material/sports_golf: **Golf balls have dimples** *specifically* "
            "to disrupt this pattern. The dimples force the air close to the ball "
            "to become turbulent earlier, which shrinks the wake and cuts drag -- "
            "letting a dimpled ball fly twice as far as a smooth one.\n"
            "- :material/account_balance: **The Tacoma Narrows Bridge collapse "
            "(1940).** Wind-driven oscillations of this pattern matched the "
            "bridge's natural twisting frequency. The bridge fed itself energy "
            "every cycle and tore itself apart in under an hour.\n"
            "- :material/sailing: **Sailboat wakes**, **smokestack plumes**, "
            "**river flow behind a piling** -- same physics, different scale.\n\n"
            "The rate at which the swirls peel off (the **Strouhal number**) is "
            "approximately constant for a given shape -- about 0.165 for a "
            "cylinder over a huge range of speeds. We hit 0.17 in our own "
            "validation runs."
        )

    # === Technical depth ===
    with st.expander(":material/science: **Under the hood -- the technical details**"):
        st.markdown(
            f"**Method:** D2Q9 Lattice Boltzmann with **MRT (multi-relaxation-time) "
            f"collision** on a {LBM_NX} x {LBM_NY} grid. Halfway bounce-back "
            f"boundaries for the solid; equilibrium inflow, zero-gradient outflow; "
            f"free-slip top/bottom (reflects y-momentum, no wall friction). "
            f"Numba-JIT compiled fused step (collide + force "
            f"+ bounce-back + stream + BCs in one function, parallel over x).\n\n"
            f"**Why MRT:** projects the 9 populations onto a moment basis "
            f"(density, energy, momentum, energy-flux, viscous stresses) and "
            f"relaxes each moment with its own rate. Viscous stresses use "
            f"s = 1/tau (same kinematic viscosity as BGK), but bulk-viscosity "
            f"and ghost-moment rates are free parameters tuned for stability. "
            f"That decoupling keeps the solver stable at tau approaching 0.5 "
            f"where BGK diverges -- giving us roughly 3x BGK's stable Re ceiling "
            f"on the same grid. Reference: Lallemand & Luo (2000), d'Humieres "
            f"et al. (2002).\n\n"
            f"**Heatmap:** signed vorticity omega = curl(u). Red = omega > 0 "
            f"(anti-clockwise), blue = omega < 0 (clockwise). RdBu_r colormap "
            f"is alpha-modulated -- omega ~ 0 is transparent so the dark "
            f"background shows through, peak alpha capped at 70% so the wake "
            f"reads as a wash. Color scale clipped at "
            f"{VORT_CLIP_FACTOR:.1f} x U / L (absolute, not a percentile of "
            f"the field) so the wake stays visible from Re=50 up through "
            f"Re=1500 -- a percentile clip lets the body's boundary-layer "
            f"vorticity dominate at high Re and wash the wake out.\n\n"
            f"**Top/bottom walls:** free-slip (no wall friction, no vertical "
            f"wraparound). The original periodic top/bottom was visually "
            f"misleading -- air exiting the bottom appeared to re-enter at "
            f"the top.\n\n"
            f"**This run:**\n"
            f"- {N_STEPS} time steps over {N_FRAMES} frames "
            f"(loop = {N_FRAMES * GIF_FRAME_MS / 1000:.1f} s)\n"
            f"- tau = {tau:.4f}  (kinematic relaxation; MRT stable to ~0.505)\n"
            f"- nu = {nu:.5f}  (kinematic viscosity, lattice units)\n"
            f"- Re = U L / nu = {U_INFLOW} x {char_length:.0f} / {nu:.5f} = {reynolds_target}\n"
            f"- characteristic length L = {char_length:.0f}  (lattice cells)\n"
            f"- MRT free rates: s_e = s_eps = s_q = 1.4  (tuned for high-Re stability)"
        )

    st.stop()

# --- Fast (NeuralFoil) mode ---
st.title("Instant Airfoil Analysis")
st.markdown(
    "##### Pick airfoils, set angle of attack, see lift and drag instantly "
    "-- powered by a neural network trained on millions of XFoil runs."
)
st.caption(
    ":material/bolt: For full fluid simulation with visible streamlines and "
    "wake structure, switch to **Real CFD** in the sidebar."
)


def normalize_naca(raw: str) -> str:
    """Accept '4412', 'naca4412', 'NACA 4412', etc., return 'naca4412'."""
    cleaned = raw.strip().lower().replace("naca", "").replace(" ", "")
    return f"naca{cleaned}"


def thickness_camber(coords: np.ndarray, x_stations: np.ndarray):
    """Compute t(x)/c and camber(x)/c at the given x/c stations.

    AeroSandbox airfoil coordinates start at the trailing edge, walk counter-clockwise
    over the upper surface to the leading edge, and back to the trailing edge along
    the lower surface. We split at the LE (min-x point) and interpolate each surface
    at common x stations to get thickness and camber.
    """
    le_idx = int(np.argmin(coords[:, 0]))
    upper = coords[: le_idx + 1][::-1]   # LE -> TE on upper surface
    lower = coords[le_idx:]               # LE -> TE on lower surface
    y_upper = np.interp(x_stations, upper[:, 0], upper[:, 1])
    y_lower = np.interp(x_stations, lower[:, 0], lower[:, 1])
    thickness = y_upper - y_lower
    camber = (y_upper + y_lower) / 2
    return thickness, camber


# NeuralFoil model size: smaller = faster inference, less accurate.
# Default to the largest (xxxlarge, ~1 ms/point, matches XFoil within ~few %)
# but expose the trade-off so the user can sweep large polars quickly with
# a smaller model.
NF_MODEL_PRESETS = {
    "Best (xxxlarge)": "xxxlarge",
    "Balanced (medium)": "medium",
    "Fast (xsmall)": "xsmall",
}

# --- Sidebar inputs ---
with st.sidebar:
    st.markdown("### :material/tune: Inputs")

    st.markdown(":material/airplane_ticket: **Airfoils**")
    raw_names = st.text_input(
        "Airfoils",
        value="4412, 0012",
        label_visibility="collapsed",
        help='Examples: "4412" for one, "0012, 4412, 2412" to compare. '
             'NACA prefix optional.',
    )

    st.markdown("")
    st.markdown(":material/rotate_right: **Angle of attack** &nbsp; :gray[(deg)]")
    alpha = st.slider(
        "alpha",
        -5.0, 15.0, 5.0, 0.25,
        label_visibility="collapsed",
        help="How much the wing is tilted into the wind. Positive = nose-up.",
    )

    st.markdown("")
    st.markdown(":material/speed: **Reynolds number**")
    reynolds = st.select_slider(
        "Reynolds",
        options=[1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7],
        value=5e5,
        format_func=lambda x: f"{x:.0e}",
        label_visibility="collapsed",
        help="Flow speed relative to chord. Light aircraft cruise: ~3-6 million.",
    )

    st.markdown("")
    st.markdown(":material/auto_awesome: **Model quality**")
    nf_model_display = st.radio(
        "nf_model_size",
        list(NF_MODEL_PRESETS.keys()),
        index=0,
        label_visibility="collapsed",
        help=(
            "NeuralFoil's accuracy/speed trade-off. **Best** matches XFoil to "
            "within ~3% (research-grade). **Balanced** is ~2x faster, ~5% accurate. "
            "**Fast** is ~5x faster, ~10% accurate -- great for sweeping wide "
            "alpha or Re ranges where you want shape not exactness."
        ),
    )
    nf_model_size = NF_MODEL_PRESETS[nf_model_display]

# Parse and dedupe airfoil names while preserving input order.
seen = set()
airfoil_names = []
for raw in raw_names.split(","):
    if raw.strip():
        n = normalize_naca(raw)
        if n not in seen:
            seen.add(n)
            airfoil_names.append(n)

if not airfoil_names:
    st.warning("Enter at least one airfoil name in the sidebar.")
    st.stop()

# Categorical color palette (Plotly's default qualitative set). Reused across all
# figures so the same airfoil keeps the same color in every chart.
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# --- Cached polar sweep ---
# Caching by (name, Re, model_size) means slider drags on alpha don't re-invoke
# NeuralFoil for the polar sweep -- only the alpha-dependent point evaluation
# re-runs. Switching model_size invalidates the cache (different model).
@st.cache_data(show_spinner=False)
def sweep_polar(name: str, Re: float, model_size: str):
    alphas = np.linspace(-5, 15, 81)  # 0.25 deg resolution for smooth curves
    aero = analyze_airfoil(name, alphas, Re, model_size=model_size)
    return alphas, aero["CL"], aero["CD"], aero["LD"]


# Resolve every requested airfoil up front so we know which names are valid before
# building any chart.
valid_names = []
airfoil_objs = {}
for name in airfoil_names:
    try:
        airfoil_objs[name] = get_airfoil(name)
        valid_names.append(name)
    except Exception as e:
        st.warning(f"Skipping {name.upper()!r}: {e}")

if not valid_names:
    st.error("None of the requested airfoils could be analyzed.")
    st.stop()

# === Section 1: Geometry ===
st.subheader("Geometry")

geom_fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("Airfoil shape", "Thickness & camber distribution"),
    horizontal_spacing=0.1,
)

x_stations = np.linspace(0, 1, 100)  # x/c stations for thickness/camber sampling

for i, name in enumerate(valid_names):
    af = airfoil_objs[name]
    color = PALETTE[i % len(PALETTE)]
    label = name.upper()

    # Shape: airfoil.coordinates is a closed loop (TE -> upper -> LE -> lower -> TE).
    geom_fig.add_trace(
        go.Scatter(
            x=af.coordinates[:, 0],
            y=af.coordinates[:, 1],
            mode="lines",
            name=label,
            legendgroup=label,
            showlegend=True,
            line=dict(color=color, width=1.5),
        ),
        row=1, col=1,
    )

    # Thickness (solid) and camber (dashed) sampled at common x/c stations.
    thickness, camber = thickness_camber(af.coordinates, x_stations)
    geom_fig.add_trace(
        go.Scatter(
            x=x_stations, y=thickness, mode="lines",
            name=f"{label} thickness", legendgroup=label, showlegend=False,
            line=dict(color=color, width=1.5),
        ),
        row=1, col=2,
    )
    geom_fig.add_trace(
        go.Scatter(
            x=x_stations, y=camber, mode="lines",
            name=f"{label} camber", legendgroup=label, showlegend=False,
            line=dict(color=color, width=1.5, dash="dash"),
        ),
        row=1, col=2,
    )

# Equal aspect on the shape subplot so airfoils don't get visually flattened.
geom_fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
geom_fig.update_xaxes(title_text="x/c", row=1, col=1)
geom_fig.update_yaxes(title_text="y/c", row=1, col=1)
geom_fig.update_xaxes(title_text="x/c", row=1, col=2)
geom_fig.update_yaxes(title_text="t/c (solid),  camber/c (dashed)", row=1, col=2)

geom_fig.update_layout(
    height=360,
    margin=dict(t=60, l=50, r=20, b=50),
    legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
    hovermode="x unified",
)

st.plotly_chart(geom_fig, width="stretch")

# === Section 2: Coefficient table at current alpha ===
table_rows = []
sweep_results = {}  # name -> (alphas, cl, cd, ld), reused below for the polar chart

for name in valid_names:
    p = analyze_airfoil(name, alpha, float(reynolds), model_size=nf_model_size)
    table_rows.append(
        {
            "Airfoil": name.upper(),
            "CL": round(p["CL"].item(), 4),
            "CD": round(p["CD"].item(), 4),
            "L/D": round(p["LD"].item(), 1),
        }
    )
    sweep_results[name] = sweep_polar(name, float(reynolds), nf_model_size)

st.subheader(
    f"Coefficients at alpha = {alpha:+.2f} deg, Re = {reynolds:.0e}  "
    f":gray[(model: {nf_model_display})]"
)
st.dataframe(table_rows, width="stretch", hide_index=True)

# === Section 3: Polar plots ===
polar_fig = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=("Lift curve", "Drag curve", "Drag polar"),
    horizontal_spacing=0.08,
)

for i, name in enumerate(valid_names):
    alphas_arr, cl, cd, _ = sweep_results[name]
    color = PALETTE[i % len(PALETTE)]
    label = name.upper()
    common = dict(mode="lines", line=dict(color=color, width=2), legendgroup=label)
    polar_fig.add_trace(
        go.Scatter(x=alphas_arr, y=cl, name=label, showlegend=True, **common),
        row=1, col=1,
    )
    polar_fig.add_trace(
        go.Scatter(x=alphas_arr, y=cd, name=label, showlegend=False, **common),
        row=1, col=2,
    )
    polar_fig.add_trace(
        go.Scatter(x=cd, y=cl, name=label, showlegend=False, **common),
        row=1, col=3,
    )

polar_fig.add_vline(x=alpha, line=dict(color="black", dash="dash", width=1), opacity=0.4, row=1, col=1)
polar_fig.add_vline(x=alpha, line=dict(color="black", dash="dash", width=1), opacity=0.4, row=1, col=2)

polar_fig.update_xaxes(title_text="alpha (deg)", row=1, col=1)
polar_fig.update_yaxes(title_text="CL", row=1, col=1)
polar_fig.update_xaxes(title_text="alpha (deg)", row=1, col=2)
polar_fig.update_yaxes(title_text="CD", row=1, col=2)
polar_fig.update_xaxes(title_text="CD", row=1, col=3)
polar_fig.update_yaxes(title_text="CL", row=1, col=3)

polar_fig.update_layout(
    height=440,
    margin=dict(t=60, l=50, r=20, b=50),
    legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
    hovermode="x unified",
)

st.plotly_chart(polar_fig, width="stretch")

st.caption(
    "Click a legend entry to toggle that airfoil across all three subplots. "
    "Hover for exact values. Drag to zoom; double-click to reset."
)

# === Section 4: CSV download ===
st.subheader("Export")

# Long-format polar CSV: one row per (airfoil, alpha) pair so the file is easy to
# load into pandas, Excel, or MATLAB and group by airfoil.
csv_frames = []
for name in valid_names:
    alphas_arr, cl, cd, ld = sweep_results[name]
    csv_frames.append(
        pd.DataFrame(
            {
                "airfoil": name.upper(),
                "Re": float(reynolds),
                "alpha_deg": alphas_arr,
                "CL": cl,
                "CD": cd,
                "LD": ld,
            }
        )
    )
csv_data = pd.concat(csv_frames, ignore_index=True).to_csv(index=False)

st.download_button(
    "Download polars (CSV)",
    data=csv_data,
    file_name=f"polars_re{int(reynolds)}.csv",
    mime="text/csv",
    help=f"All {len(valid_names)} airfoil(s), -5 to 15 deg in 0.25 deg steps.",
)

st.caption(
    "Geometry note: NeuralFoil predicts aerodynamic coefficients only, not the "
    "full Cp(x) distribution. We're showing thickness and camber distributions "
    "instead -- a real Cp(x) chart will need a panel method or XFoil and is "
    "queued for later."
)
