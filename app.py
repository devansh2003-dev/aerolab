"""AeroLab Streamlit app -- dual-mode airfoil aerodynamics.

Two modes via the sidebar toggle:
  - Fast (NeuralFoil): instant ML predictions, alpha sweeps, drag polar.
  - Real CFD (LBM): browser-based Lattice Boltzmann simulation rendered as a GIF.

Run from the project root:
    streamlit run app.py

The browser opens automatically at http://localhost:8501.
"""
# Force NUMBA_NUM_THREADS=16 BEFORE any other import. Diagnosed from a
# Cloud error log: Cloud's container starts with NUMBA_NUM_THREADS
# UNSET, numba imports and auto-detects cpu_count=1 (cgroups throttle
# the container to 1 vCPU) so it launches 1 thread. Then Cloud's
# request handling later SETS NUMBA_NUM_THREADS to 16 (the underlying
# host's logical CPU count). When JIT compile fires, numba's
# reload_config sees env=16 vs launched=1 and crashes with
# "RuntimeError: Cannot set NUMBA_NUM_THREADS to a different value
# once the threads have been launched (currently have 1, trying to set
# 16)".
#
# Setting env=16 here pre-empts that: numba launches 16 threads from
# the start, and Cloud's later assignment is a no-op (already 16). The
# 16 threads contending for 1 vCPU is wasteful but functionally
# serial; "wasteful" beats "crashed" every time.
import os
os.environ["NUMBA_NUM_THREADS"] = "16"

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.airfoils import analyze_airfoil, get_airfoil

# Register the custom polygon-drawer component at module load. declare_component
# only actually registers the component (and exposes its iframe URL) when run
# inside a Streamlit ScriptRunContext, AND only on the script-run that
# encounters the declare call. Importing it at the top of app.py guarantees
# that every script run -- including the first one a fresh visitor triggers --
# registers the component before any tab tries to render it. Without this
# top-level import, the component is only registered if the user navigates
# into the Real CFD > Custom > Draw tab, and the iframe URL 404s for everyone
# else (silently producing a broken-looking blank canvas).
from components import polygon_drawer  # noqa: F401, E402

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
    "<span style='opacity:0.4'>·</span>"
    "<span style='opacity:0.55;font-size:0.75rem;'>v0.5.0</span>"
    "<span style='opacity:0.4'>·</span>"
    "<a href='https://github.com/devansh2003-dev/AeroLab/blob/main/VALIDATION.md' "
    "target='_blank' style='color:#10b981;text-decoration:none;"
    "font-size:0.75rem;opacity:0.85;' "
    "title='Blockage-corrected Cd against Williamson 1996 (cylinder, "
    "median 4 percent, max 12 percent) and Okajima 1982 (square, "
    "median 5 percent, max 22 percent) across Re 100-1000. The Standard "
    "preset runs at 35 percent blockage so the correction is large; click "
    "for the full honest methodology.'>"
    ":material/verified: validated"
    "</a>"
    "</div>",
    unsafe_allow_html=True,
)

# --- Mode toggle (sidebar, top) ---
with st.sidebar:
    st.markdown("### :material/tune: Mode")
    # Plain-English framing above the radio. The radio labels themselves
    # are short ("Fast (NeuralFoil)" / "Real CFD (LBM)"); the framing
    # tells a first-time visitor which one they want WITHOUT them having
    # to hover the ? tooltip or read the README.
    st.markdown(
        "<div style='color:#94a3b8;font-size:0.85rem;line-height:1.45;"
        "margin-bottom:0.4rem;'>"
        "<b style='color:#cbd5e1;'>Fast</b> &mdash; airfoil lift/drag numbers in "
        "&lt;1 s. Drag a slider, get a polar.<br>"
        "<b style='color:#cbd5e1;'>Real CFD</b> &mdash; watch the air actually move "
        "around a shape. ~2.5 min on Cloud, ~30 s locally."
        "</div>",
        unsafe_allow_html=True,
    )
    mode = st.radio(
        "Simulation mode",
        ["Fast (NeuralFoil)", "Real CFD (LBM)"],
        # Default to Real CFD: the user-facing "see air move" feature is
        # what makes AeroLab visually distinctive, and the curated gallery
        # cards give first-time visitors something compelling to click
        # without needing to know what NeuralFoil is.
        index=1,
        label_visibility="collapsed",
        help=(
            "**Fast**: instant ML predictions for NACA airfoils -- great "
            "for sweeping lots of cases.  "
            "**Real CFD**: full Lattice Boltzmann simulation -- watch the air "
            "actually move."
        ),
    )
    st.divider()

# --- Cached LBM simulate+render wrapper (module level) ---
# Textbook 2D bluff-body Cd / Strouhal vs Reynolds. We ship TWO tables
# per shape so the user gets an honest picture:
#
#   _*_REFERENCE_CD   -- expected output of OUR solver in its 33-35 %
#                        blockage channel. Used by the "vs textbook"
#                        delta chip as a sanity check: if your sim is
#                        within ~5 % of this, the solver is healthy.
#   _*_FREESTREAM_CD  -- true unbounded-flow values from the canonical
#                        experiments (Williamson 1996, Norberg 1994 for
#                        cylinder; Okajima 1982, Sohankar 1998 for square).
#                        Shown as a separate "free-stream estimate" line
#                        so the user knows what the real-world Cd is.
#
# The gap between the two is almost entirely 2D channel-blockage
# inflation (Maskell + Allen-Vincenti style). Honest disclosure of both
# values teaches the user a real CFD lesson: confined-domain results
# always need an unblocking correction before they can be compared to
# wind-tunnel / atmospheric data.
_CYLINDER_REFERENCE_CD = {
    40: 1.55, 80: 1.40, 100: 1.40, 150: 1.32, 200: 1.30,
    300: 1.35, 500: 1.40, 800: 1.41, 1000: 1.40, 1500: 1.42,
}
_CYLINDER_FREESTREAM_CD = {
    # Williamson 1996 "Vortex Dynamics in the Cylinder Wake" + Norberg
    # 1994. Drag drops monotonically from Re~40 (laminar attached, Cd ~
    # 1.55) through the wake-transition regime; settles near 1.0 at
    # Re=1000 and stays there into the subcritical band.
    40: 1.55, 80: 1.38, 100: 1.32, 150: 1.20, 200: 1.15,
    300: 1.08, 500: 1.02, 800: 1.00, 1000: 0.99, 1500: 1.00,
}
_CYLINDER_REFERENCE_ST = {
    80: 0.155, 100: 0.165, 150: 0.180, 200: 0.197, 300: 0.207,
    500: 0.215, 800: 0.215, 1000: 0.21, 1500: 0.21,
}
_CYLINDER_FREESTREAM_ST = {
    # Williamson 1989: St vs Re, asymptotes near 0.21 above Re~400.
    80: 0.155, 100: 0.166, 150: 0.182, 200: 0.197, 300: 0.205,
    500: 0.207, 800: 0.210, 1000: 0.210, 1500: 0.210,
}

# Square cylinder (sharp-edged, broadside to flow): Cd is much flatter
# than the round cylinder because separation is geometry-locked at the
# corners regardless of Re. Strouhal hovers around 0.13 across the laminar
# / transitional shedding range. References: Okajima JFM 1982 (Re 70-500),
# Sohankar/Norberg/Davidson IJNMF 1998 (Re 45-200), Saha/Biswas/Muralidhar
# IJHFF 2003. We cover Re 80-1500 to match our slider's reachable band.
_SQUARE_REFERENCE_CD = {
    80: 1.55, 100: 1.50, 150: 1.50, 200: 1.50, 300: 1.65,
    500: 1.95, 800: 2.05, 1000: 2.10, 1500: 2.15,
}
_SQUARE_FREESTREAM_CD = {
    # Okajima 1982 / Sohankar et al 1998 / Saha et al 2003 averaged.
    # Square Cd is geometry-locked at corners and changes less with Re
    # than cylinder, but grows mildly from ~1.5 at Re=100 to ~2.1 at
    # Re=1000 as the wake becomes more vigorous in real flows.
    80: 1.50, 100: 1.50, 150: 1.55, 200: 1.60, 300: 1.85,
    500: 2.00, 800: 2.10, 1000: 2.15, 1500: 2.20,
}
_SQUARE_REFERENCE_ST = {
    80: 0.130, 100: 0.135, 150: 0.140, 200: 0.143, 300: 0.140,
    500: 0.135, 800: 0.130, 1000: 0.128, 1500: 0.125,
}
_SQUARE_FREESTREAM_ST = {
    # Okajima 1982. Square St is much flatter than cylinder because the
    # shedding frequency is set by the geometry-locked corner separation.
    80: 0.140, 100: 0.143, 150: 0.146, 200: 0.148, 300: 0.142,
    500: 0.135, 800: 0.130, 1000: 0.128, 1500: 0.125,
}


def _interp_or_none(re_value: float, table: dict):
    """Linear-interpolated table lookup, returns None if out of range."""
    if not table:
        return None
    lo, hi = min(table), max(table)
    if re_value < lo or re_value > hi:
        return None
    keys = sorted(table)
    vals = [table[k] for k in keys]
    return float(np.interp(re_value, keys, vals))


def _textbook_reference(shape_preset: str, re_value: int):
    """Linear-interpolated textbook (Cd, St) for a canonical 2-D bluff body.

    Supports Cylinder (round) and Square (broadside). Returns (None, None)
    for shapes we don't ship a reference for, or for Re outside the
    validated band. The Square table is for broadside flow only -- at
    aoa=45 (diamond orientation) Cd drops to ~1.5 and St rises, but we
    don't ship a separate diamond table, so the caller is responsible for
    only invoking this when the body presents a flat face (aoa ~ 0).
    """
    if shape_preset == "Cylinder":
        cd_ref = _interp_or_none(re_value, _CYLINDER_REFERENCE_CD)
        st_ref = _interp_or_none(re_value, _CYLINDER_REFERENCE_ST)
    elif shape_preset == "Square":
        cd_ref = _interp_or_none(re_value, _SQUARE_REFERENCE_CD)
        st_ref = _interp_or_none(re_value, _SQUARE_REFERENCE_ST)
    else:
        cd_ref, st_ref = None, None
    return cd_ref, st_ref


# Backward-compat alias: some older callers / tests still imported the
# cylinder-only entry point. Forward to the generalized lookup.
def _cylinder_reference(re_value: int):
    return _textbook_reference("Cylinder", re_value)


def _freestream_reference(shape_preset: str, re_value: int):
    """True unbounded-flow (Cd, St) from Williamson / Norberg / Okajima.

    Distinct from _textbook_reference, which is calibrated to our 33-35 %
    blocked-channel solver: that lookup answers "is the simulation
    behaving as expected?", while this one answers "what would you
    measure in a wind tunnel?". Both are useful, for different reasons.
    Returns (None, None) outside the validated band.
    """
    if shape_preset == "Cylinder":
        cd_free = _interp_or_none(re_value, _CYLINDER_FREESTREAM_CD)
        st_free = _interp_or_none(re_value, _CYLINDER_FREESTREAM_ST)
    elif shape_preset == "Square":
        cd_free = _interp_or_none(re_value, _SQUARE_FREESTREAM_CD)
        st_free = _interp_or_none(re_value, _SQUARE_FREESTREAM_ST)
    else:
        cd_free, st_free = None, None
    return cd_free, st_free


# Two-layer cache: the LBM solve is mode-independent, the GIF + colorbar
# rendering is mode-dependent. Splitting them means switching viz_mode
# (Vorticity -> Velocity -> Pressure) only pays the render cost (~2 s)
# instead of re-solving the LBM (~40 s on Standard, ~120 s on Detailed
# locally; 3 x slower on the 1-vCPU Cloud container).
#
# Memory budget on Streamlit Cloud (~500 MB process cap):
#   solve cache  max_entries=4  ~15 MB/entry (snapshots dict) = 60 MB
#   render cache max_entries=12 ~5-20 MB/entry (GIF bytes)    = 60-240 MB
# Total cap ~300 MB peak, well within budget on Standard preset; Detailed
# users tend to run only one or two configs per session so the lower
# solve cap is fine.
#
# The leading-underscore convention on _custom_polygon tells Streamlit to
# exclude it from the cache key (raw numpy arrays are slow to hash and we
# already include polygon_key as a stable cache key derived from a SHA
# hash of the polygon bytes).
@st.cache_data(show_spinner=False, max_entries=4)
def _cached_solve(
    shape_preset, reynolds_target, aoa_deg, res_key, polygon_key,
    _custom_polygon=None,
):
    from src.lbm_render import solve_lbm
    # First run in a fresh session pays the Numba JIT-compile tax
    # (~20-30 s); subsequent runs return instantly from cache or
    # finish their JIT-compiled inner loop in seconds. We detect the
    # "first run" case via a session-state flag and surface a friendly,
    # honest warm-up message instead of a cryptic "simulating flow (MRT)"
    # that freezes for 20 s with no movement.
    _is_first_run = not st.session_state.get("lbm_solver_warmed_up", False)
    _initial_text = (
        ":material/local_fire_department: Warming up the solver "
        "(first run takes ~20 s while the just-in-time compiler "
        "translates the physics to fast machine code -- later runs are "
        "instant)..."
        if _is_first_run else
        ":material/sync: Simulating the flow..."
    )
    progress = st.progress(0.0, text=_initial_text)
    try:
        def cb(frac, text):
            progress.progress(frac, text=text)
        result = solve_lbm(
            shape_preset, reynolds_target, aoa_deg, res_key,
            progress_callback=cb, custom_polygon=_custom_polygon,
        )
        # Solver returned, JIT cache is now warm for this session.
        st.session_state["lbm_solver_warmed_up"] = True
        return result
    finally:
        progress.empty()


@st.cache_data(show_spinner=False, max_entries=12)
def _cached_sim_result(
    shape_preset, reynolds_target, aoa_deg, res_key, polygon_key,
    viz_mode, _custom_polygon=None,
):
    """Wrapper that combines a cached solve + cached render into the dict
    shape app.py expects (matches the legacy simulate_and_render output).

    On a cache hit, returns the full dict instantly. On a render-only miss
    (viz_mode changed, same physics), the solve cache hits and only the
    render runs -- typically 1-2 s. On a full miss, both run.
    """
    from src.lbm_render import render_lbm
    solve = _cached_solve(
        shape_preset, reynolds_target, aoa_deg, res_key, polygon_key,
        _custom_polygon=_custom_polygon,
    )
    progress = st.progress(
        0.5, text=":material/sync: Painting the airflow frames...",
    )
    try:
        def cb(frac, text):
            progress.progress(frac, text=text)
        render = render_lbm(solve, viz_mode=viz_mode, progress_callback=cb)
    finally:
        progress.empty()
    # Merge into the legacy public-API shape. Drop internal-only keys.
    result = {
        k: v for k, v in solve.items()
        if k not in ("snapshots", "mask", "body_xs", "body_ys",
                     "reynolds_target", "res_key")
    }
    result.update(render)
    return result


# Backward-compat alias for any internal call sites that still reference
# the old wrapper name. New code should call _cached_sim_result directly.
def _cached_simulate_and_render(
    shape_preset, reynolds_target, aoa_deg, res_key,
    custom_polygon=None, viz_mode="Vorticity",
):
    # Polygon hash is the stable cache key. For preset shapes (no polygon)
    # we pass a literal None marker so all preset runs share the same key
    # for that slot.
    if custom_polygon is not None:
        import hashlib as _hl
        polygon_key = _hl.sha1(
            np.ascontiguousarray(custom_polygon).tobytes()
        ).hexdigest()[:12]
    else:
        polygon_key = None
    return _cached_sim_result(
        shape_preset, reynolds_target, aoa_deg, res_key, polygon_key,
        viz_mode, _custom_polygon=custom_polygon,
    )


# --- Real CFD (LBM) mode: animated GIF playback of LBM run ---
# Pre-warm was removed because it didn't survive Streamlit Cloud's
# environment quirks (NUMBA_NUM_THREADS RuntimeError at JIT-compile
# time). First user click in a fresh container now pays the full ~20-30 s
# JIT cost; subsequent clicks are instant (cached by @st.cache_data).
if mode == "Real CFD (LBM)":
    # Lazy imports: keep Fast mode's cold-start untouched by Numba + matplotlib.
    # All the heavy lifting (LBM step, rendering, GIF encoding) lives in
    # src.lbm_render -- this branch is only sidebar UI + result display.
    from src.lbm_render import (
        GIF_FRAME_MS,
        RESOLUTION_PRESETS,
        STEPS_PER_FRAME,
        U_INFLOW,
        simulate_and_render,
    )

    # Shape param mappings for share-link query params (?shape=cylinder...).
    # Defined at module-conditional scope so both the share button and the
    # boot-time URL-param reader can use them.
    _SHAPE_DISPLAY_TO_QP = {
        "Cylinder": "cylinder",
        "Square": "square",
        "Ellipse": "ellipse",
        "NACA 0012": "naca0012",
        "NACA 4412": "naca4412",
    }
    _SHAPE_QP_TO_DISPLAY = {
        "cylinder": "Cylinder  (round pipe)",
        "square": "Square  (boxy)",
        "ellipse": "Ellipse  (stretched oval)",
        "naca0012": "NACA 0012  (symmetric wing)",
        "naca4412": "NACA 4412  (curved wing)",
    }

    # Share-link query params: when a user opens a URL like
    #   ?shape=naca4412&vel=1.8&aoa=4&res=standard&viz=Pressure
    # apply the encoded config as pending keys (which the loop below then
    # promotes to the actual widgets) and trigger the gallery-style
    # auto-run gate so the shared run shows up immediately. We do this
    # ONCE per browser session via a flag, so subsequent reruns inside the
    # same session don't keep overriding user slider changes.
    _qp = st.query_params
    if _qp and not st.session_state.get("lbm_share_applied", False):
        _shape_qp = _qp.get("shape", "").lower() if "shape" in _qp else ""
        _shape_disp = _SHAPE_QP_TO_DISPLAY.get(_shape_qp)
        if _shape_disp is not None:
            st.session_state["lbm_pending_shape"] = _shape_disp
            try:
                _v = float(_qp.get("vel", "0.6"))
                if 0.15 <= _v <= 4.5:
                    # Slider step is 0.1; round to land on a tick.
                    st.session_state["lbm_pending_velocity"] = round(_v, 1)
            except (TypeError, ValueError):
                pass
            try:
                _a = float(_qp.get("aoa", "0"))
                if -45.0 <= _a <= 45.0:
                    # AoA slider step is 0.5; round.
                    st.session_state["lbm_pending_aoa"] = round(_a * 2) / 2
            except (TypeError, ValueError):
                pass
            _res_qp = _qp.get("res", "standard").lower()
            st.session_state["lbm_pending_res"] = (
                "Detailed (960 x 240)" if "detail" in _res_qp
                else "Standard (320 x 80)"
            )
            _viz_qp = _qp.get("viz", "Vorticity")
            if _viz_qp in ("Vorticity", "Velocity", "Pressure"):
                st.session_state["lbm_pending_viz"] = _viz_qp
            # Trigger auto-run on the rerun the promotion loop below
            # produces, so the shared link lands directly on the result.
            st.session_state["lbm_gallery_pending"] = True
        st.session_state["lbm_share_applied"] = True

    # Gallery card pre-fill: copy any "pending" values into their widget
    # session_state keys BEFORE the widgets render below. Done here because
    # Streamlit forbids writes to a widget's session_state key after the
    # widget has been instantiated -- so a gallery card button (which runs
    # AFTER the sidebar widgets) writes to lbm_pending_* and we promote
    # them here on the next rerun. Share-link query params (handled above)
    # write to the same pending keys and use the same promotion path.
    for _src, _dst in (
        ("lbm_pending_shape", "lbm_shape_select"),
        ("lbm_pending_velocity", "lbm_velocity_slider"),
        ("lbm_pending_aoa", "lbm_aoa_slider"),
        ("lbm_pending_res", "lbm_res_radio"),
        ("lbm_pending_viz", "lbm_viz_mode"),
    ):
        if _src in st.session_state:
            st.session_state[_dst] = st.session_state.pop(_src)

    # Friendly display name -> internal preset key
    SHAPE_PRESETS = {
        "Cylinder  (round pipe)": "Cylinder",
        "Square  (boxy)": "Square",
        "Ellipse  (stretched oval)": "Ellipse",
        "NACA 0012  (symmetric wing)": "NACA 0012",
        "NACA 4412  (curved wing)": "NACA 4412",
        "Upload your own  (PNG / JPG)": "Custom",
    }

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
        # Brief always-visible orientation -- the first-time visitor lands
        # here and needs to know what kind of choices they're making
        # without hovering tooltips. Three bullets, plain English.
        with st.expander(":material/help_outline: &nbsp; **First time? Read this**",
                          expanded=False):
            st.markdown(
                "**Pick a body, set how fast the air moves, hit Run.** "
                "The simulation shows the wake forming behind the body as "
                "vortices shed off it. You'll see the same physics that "
                "makes a flag flutter, a car drag, or a wing lift.\n\n"
                "- **Shape:** what the wind flows past.\n"
                "- **Flow speed:** wind speed in m/s -- low is laminar and "
                "gentle, high stirs up vortex shedding. The displayed Reynolds "
                "number is what the solver actually uses.\n"
                "- **Tilt / rotation:** angle of the body into the wind.\n"
                "- **Resolution:** Standard is fast, Detailed is prettier."
            )

        st.markdown(":material/category: **Shape**")
        # Seed session_state once so the widget can be keyed without the
        # "value= conflicts with session_state" warning. Cylinder is the
        # canonical first-time-visitor pick (validated case, clean wake).
        st.session_state.setdefault(
            "lbm_shape_select", list(SHAPE_PRESETS.keys())[0],
        )
        shape_display = st.selectbox(
            "Shape preset",
            list(SHAPE_PRESETS.keys()),
            label_visibility="collapsed",
            help=(
                "What the wind flows past. Round and boxy shapes shed swirly "
                "wakes (think behind a bridge column). Wing shapes glide more "
                "smoothly. Try them and see the difference."
            ),
            key="lbm_shape_select",
        )
        shape_preset = SHAPE_PRESETS[shape_display]

        # --- Custom shape: Upload / Draw / Sample, in three sibling tabs ---
        # All three paths write the same session_state key
        # ("lbm_custom_polygon"); the rest of the pipeline (preview, flip,
        # cache key, run, pin) is source-agnostic. Polygon lives in session
        # state so it survives Streamlit reruns from Pin / Clear / slider
        # widgets. Each tab writes only on successful extraction; errors
        # render inline without touching state.
        custom_polygon = None
        if shape_preset == "Custom":
            st.markdown("")
            st.markdown(":material/draw: **Your shape**")
            _upload_tab, _draw_tab, _sample_tab = st.tabs(
                ["Upload", "Draw", "Sample"]
            )

            # --- Upload tab ---
            with _upload_tab:
                uploaded = st.file_uploader(
                    "Upload an image",
                    type=[
                        "png", "jpg", "jpeg", "gif", "bmp", "tiff", "tif",
                        "webp", "ico", "ppm", "tga",
                    ],
                    accept_multiple_files=False,
                    label_visibility="collapsed",
                    help=(
                        "Drop in any image with a clear subject on a "
                        "plain background (white, black, grey, or any "
                        "solid colour).\n\n"
                        "- :material/check_circle: Most common image "
                        "formats work (PNG, JPG, WEBP, etc.). Transparent "
                        "PNGs and phone photos in the wrong orientation "
                        "are fixed automatically.\n"
                        "- :material/photo_size_select_actual: Minimum "
                        "100 x 100 px.\n"
                        "- :material/west: Orient the image so the FRONT "
                        "of the shape faces **left** — that's where the "
                        "wind comes from.\n\n"
                        ":gray[*HEIC iPhone photos: convert to PNG via "
                        "your phone's share sheet first.*]"
                    ),
                    key="lbm_custom_upload",
                )
                if uploaded is not None:
                    from src.custom_shape import extract_silhouette_from_image
                    import hashlib as _hl
                    # Detect new-file events: hash the upload bytes and
                    # compare to the last hash we saw. On a fresh file the
                    # flip toggle should reset so a user uploading a
                    # right-facing shape doesn't get it pre-flipped from a
                    # previous left-facing upload.
                    _upload_hash = _hl.sha1(uploaded.getvalue()).hexdigest()[:12]
                    if _upload_hash != st.session_state.get("lbm_last_upload_hash"):
                        st.session_state["lbm_last_upload_hash"] = _upload_hash
                        st.session_state["lbm_custom_flipped"] = False
                    try:
                        result = extract_silhouette_from_image(uploaded.getvalue())
                        custom_polygon = result.polygon_xy
                        st.session_state["lbm_custom_polygon"] = custom_polygon
                        # Clear any "Draw" or "Sample" label so the
                        # downstream filename / caption reflects the upload.
                        st.session_state.pop("lbm_custom_label", None)
                        for w in result.warnings:
                            st.warning(f":material/info: {w}")
                        st.caption(
                            f":material/check_circle: Extracted a "
                            f"{len(custom_polygon)}-corner outline from a "
                            f"{result.image_w}x{result.image_h} px image."
                        )
                    except ValueError as e:
                        st.error(f":material/error: {e}")
                        st.session_state.pop("lbm_custom_polygon", None)
                        custom_polygon = None

            # --- Draw tab ---
            with _draw_tab:
                st.caption(
                    ":material/touch_app: **Click** to drop each corner. "
                    "When you have 3 + corners, **click the green dot** "
                    "(the first corner you placed) to close the shape. "
                    "Wind blows left -> right, so put the FRONT of your "
                    "shape on the left."
                )
                # polygon_drawer is imported at the top of app.py (so the
                # component registers on every script run, not just when
                # the Draw tab is first rendered). We still wrap the
                # *render* call in try/except so a frontend bug doesn't
                # nuke the whole sidebar -- users can fall back to Upload
                # or Sample.
                _drawer_available = True
                _drawer_result = None
                try:
                    _drawer_result = polygon_drawer(
                        width=400, height=200, key="lbm_polygon_drawer",
                    )
                except Exception as _drawer_imp_err:  # noqa: BLE001
                    _drawer_available = False
                    st.warning(
                        f":material/warning: The drawing canvas isn't "
                        f"available in this environment "
                        f"(`{type(_drawer_imp_err).__name__}: "
                        f"{_drawer_imp_err}`). Use the **Upload** or "
                        f"**Sample** tabs to provide a custom shape."
                    )
                # The component returns its state on every interaction.
                # We treat a closed polygon as the trigger to commit the
                # drawing into session_state -- no separate "Use this
                # drawing" button needed, which removes a click from the
                # workflow and makes the close-on-start-vertex gesture
                # the natural commit signal.
                if _drawer_available and _drawer_result is not None:
                    _verts = _drawer_result.get("vertices", []) or []
                    _is_closed = bool(_drawer_result.get("closed", False))
                    _cw = int(_drawer_result.get("width", 400))
                    _ch = int(_drawer_result.get("height", 200))
                    # Detect a fresh closure event: only commit once per
                    # close so re-clicking doesn't re-spawn flips / labels.
                    _drawing_sig = (
                        len(_verts), _is_closed,
                        tuple((round(v.get("x", 0), 2), round(v.get("y", 0), 2))
                              for v in _verts),
                    )
                    _prev_sig = st.session_state.get("lbm_drawer_last_sig")
                    if _is_closed and _drawing_sig != _prev_sig:
                        from src.custom_shape import vertices_to_polygon
                        try:
                            _r = vertices_to_polygon(_verts, _cw, _ch)
                            st.session_state["lbm_custom_polygon"] = _r.polygon_xy
                            st.session_state["lbm_custom_label"] = "Your drawing"
                            st.session_state["lbm_custom_flipped"] = False
                            st.session_state.pop("lbm_last_upload_hash", None)
                            st.session_state["lbm_drawer_last_sig"] = _drawing_sig
                            st.success(
                                f":material/check_circle: Captured "
                                f"{len(_verts)}-corner shape. Preview "
                                f"below; adjust sliders and press Run."
                            )
                        except ValueError as e:
                            st.error(f":material/error: {e}")

            # --- Sample tab ---
            with _sample_tab:
                st.caption(
                    "Built-in silhouettes so you can verify the pipeline "
                    "without sourcing your own image."
                )
                from src.sample_shapes import SAMPLE_SHAPES
                for sample_name, sample_fn in SAMPLE_SHAPES.items():
                    if st.button(
                        sample_name, width="stretch",
                        key=f"lbm_sample_{sample_name}",
                    ):
                        st.session_state["lbm_custom_polygon"] = sample_fn()
                        st.session_state["lbm_custom_label"] = sample_name
                        # Reset flip on sample swap -- the bundled samples
                        # all face the inflow already, so the user should
                        # start unflipped.
                        st.session_state["lbm_custom_flipped"] = False
                        st.session_state.pop("lbm_last_upload_hash", None)
                        st.rerun()

            # All three tabs converge on the same session_state key.
            custom_polygon = st.session_state.get("lbm_custom_polygon", custom_polygon)
            # Apply the "Flip horizontally" toggle (state lives in
            # lbm_custom_flipped). We flip the polygon in image-coord
            # space here so every downstream consumer -- preview, cache
            # key hash, pin snapshot, simulate_and_render -- sees the
            # post-flip geometry without any extra plumbing.
            if (
                custom_polygon is not None
                and st.session_state.get("lbm_custom_flipped", False)
            ):
                custom_polygon = custom_polygon.copy()
                custom_polygon[:, 0] = -custom_polygon[:, 0]
            # Clear sample-label hint if the user uploaded their own
            # image (which already happened above on a successful upload).
            if uploaded is not None:
                st.session_state.pop("lbm_custom_label", None)
            if custom_polygon is None:
                st.caption(
                    ":material/info: Upload an image, or click a sample "
                    "above to load a built-in silhouette."
                )


        st.markdown("")
        st.markdown(":material/speed: **Flow speed** &nbsp; :gray[(m/s)]")
        # Velocity (m/s) -> Re via Re = U * L / nu, assuming a 5 mm
        # characteristic length in standard air (nu_air = 1.5e-5 m^2/s).
        # The mapping is U * 333.33 = Re, so the 0.15-4.5 m/s range maps
        # cleanly to Re 50-1500. 5 mm is "fountain pen / small drone
        # blade" scale -- a real object you can hold, with velocities
        # that look like real wind (gentle breeze to brisk gust). 1 cm
        # was the previous choice but the resulting <1 m/s slider felt
        # like indoor air rather than aerodynamic flow. Defaults land
        # on the previous Re defaults: 0.60 m/s -> Re 200 (bluff body),
        # 1.5 m/s -> Re 500 (airfoils).
        NU_AIR = 1.5e-5     # m^2/s, standard conditions
        L_REAL_M = 0.005    # 5 mm assumed characteristic length
        # Per-shape, AoA-aware Re ceiling. tau = nu/cs^2 + 0.5 drops
        # toward 0.5 as Re rises, and the LBM becomes unstable at the
        # limit. At AoA ~ 0 the body presents a clean rounded / flat
        # frontal area to the flow; at AoA != 0 the effective frontal
        # extent grows (Square AoA=45 raises blockage from 35% to ~50%)
        # AND the shear layers off the now-sharp leading corner thin
        # out -- both effects push tau closer to 0.5 and the divergence
        # boundary collapses by 3-5x vs the AoA=0 case.
        #
        # Caps below are MEASURED stability boundaries from local
        # stability sweeps (see scripts/validate_solver.py history;
        # tested with n_frames=150 at Standard resolution). The slider
        # can therefore never reach a configuration that crashes:
        #
        #   Cylinder (rotation-invariant): 1500
        #   Square    |AoA|<25 (broadside-ish): 1000
        #             |AoA|>=25 (diamond-ish): 200  (Re=250 already fails)
        #   Ellipse   |AoA|<=10 (axis-aligned):  1200
        #             10<|AoA|<=25 (mid):         800
        #             |AoA|>25: AoA slider blocks this
        #   NACA      |AoA|<=15:                 1500
        #             15<|AoA|<=25:               800  (gallery 'Wing stalls'
        #                                                uses Re=600 here)
        #             |AoA|>25: AoA slider blocks this
        #   Custom:   1000 (unknown geometry; conservative)
        _aoa_for_cap = float(
            st.session_state.get(
                "lbm_pending_aoa",
                st.session_state.get("lbm_aoa_slider", 0.0),
            )
        )
        _abs_aoa = abs(_aoa_for_cap)
        # Per-shape, per-AoA Re ceilings. Numbers below come from local
        # stability sweeps (n_frames=150, Standard preset) and sit at or
        # below the *measured* PASS boundary -- the slider can therefore
        # never produce a configuration that crashes the solver.
        #
        # Measurements (PASS / FAIL pairs that bracket each cap):
        #   Cyl: PASS Re=1500 across the Re band
        #   Sq  AoA=0:   PASS Re=600, FAIL Re=800 (frame 118)
        #   Sq  AoA>=10: PASS Re=200, FAIL Re=250 (frame 127)
        #   El  AoA=0:   PASS Re=1000, FAIL Re=1200 (frame 88)
        #   El  AoA=10:  PASS Re=1000
        #   El  AoA=15:  PASS Re=800, FAIL Re=1000 (frame 44)
        #   El  AoA=20:  PASS Re=400, FAIL Re=600 (frame 39)
        #   NACA AoA=15: PASS Re=1500
        #   NACA AoA=25: PASS Re=800
        #   (NACA AoA>25 isn't reachable via slider; see AoA cap below.)
        if shape_preset == "Cylinder":
            _re_cap = 1500
        elif shape_preset == "Square":
            # Square broadside (|AoA|<5) holds to Re~500 reliably; once
            # the body rotates more than ~5 deg the corner-shed shear
            # layers thin out fast and the late-frame stability margin
            # collapses by Re~600 (measured FAIL at AoA=9.5 Re=600
            # frame 98). Two-band approach keeps the gallery 'Brick'
            # card (Re=500, AoA=0) inside the safe envelope.
            _re_cap = 500 if _abs_aoa < 5.0 else 200
        elif shape_preset == "Ellipse":
            if _abs_aoa <= 5.0:
                _re_cap = 1000
            elif _abs_aoa <= 15.0:
                _re_cap = 800
            else:
                _re_cap = 400
        elif shape_preset in ("NACA 0012", "NACA 4412"):
            _re_cap = 1500 if _abs_aoa <= 15.0 else 800
        else:  # Custom
            _re_cap = 800
        # Convert Re ceiling -> max velocity (m/s) for the slider.
        # Round down to the slider's 0.1 step.
        _v_max = float(int(_re_cap * NU_AIR / L_REAL_M * 10)) / 10
        _is_airfoil_default = shape_preset in ("NACA 0012", "NACA 4412")
        _default_velocity = 1.5 if _is_airfoil_default else 0.6
        # Clamp any leftover session-state velocity (from a previous
        # shape pick with a higher ceiling) into the new range so the
        # slider doesn't raise "value out of range".
        _prev_v = st.session_state.get("lbm_velocity_slider", _default_velocity)
        st.session_state["lbm_velocity_slider"] = float(
            np.clip(_prev_v, 0.15, _v_max)
        )
        velocity_mps = st.slider(
            "Flow speed (m/s)",
            min_value=0.15, max_value=_v_max, step=0.1,
            label_visibility="collapsed",
            help=(
                f"How fast the wind blows past the object.\n\n"
                f"- **Slow** (left): syrupy, smooth flow — like honey "
                f"sliding past.\n"
                f"- **Fast** (right): chaotic swirling — like wind ripping "
                f"around a flagpole.\n\n"
                f"The slider caps at the speed where the **{shape_preset}** "
                f"shape stays solver-stable (Reynolds number up to {_re_cap}). "
                f"Bluff, sharp-cornered shapes hit the wall sooner than "
                f"smooth ones.\n\n"
                f":gray[*Technical: we assume a 5 mm reference size in "
                f"standard air, so Reynolds Re &asymp; velocity x 333.*]"
            ),
            key="lbm_velocity_slider",
        )
        reynolds_target = int(round(np.clip(velocity_mps * L_REAL_M / NU_AIR, 50, _re_cap)))
        reg, reg_feel = regime_label(reynolds_target)
        st.caption(
            f"{velocity_mps:.2f} m/s &nbsp;·&nbsp; Re &asymp; {reynolds_target} "
            f"&nbsp;·&nbsp; **{reg}** &mdash; air feels {reg_feel}"
        )

        if shape_preset == "Cylinder":
            # A circle is rotationally invariant -- no point exposing a slider.
            aoa_deg = 0.0
        else:
            st.markdown("")
            is_airfoil = shape_preset in ("NACA 0012", "NACA 4412")
            if is_airfoil:
                st.markdown(":material/rotate_right: **Wing tilt** "
                            "&nbsp; :gray[(angle of attack)]")
                # NACA at AoA > ~25 deg becomes effectively a flat plate
                # at huge incidence; the massive separation and thin
                # shear layers off the LE diverge in our LBM even at
                # Re=200. Cap the slider so users can't reach that
                # regime -- the stall showcase (gallery card 'Wing
                # stalls') runs at AoA=20 inside the cap.
                slider_min, slider_max, slider_default = -25.0, 25.0, 5.0
                slider_help = (
                    "How steeply the wing is angled into the wind. "
                    "More tilt = more lift -- but go too steep and the wing "
                    "**stalls** (lift collapses, drag spikes). Try +5 deg vs "
                    "+15 deg vs +25 deg and watch the wake on top change -- "
                    "above ~12 deg the flow detaches from the upper surface."
                )
            else:
                st.markdown(":material/rotate_right: **Rotation** "
                            "&nbsp; :gray[(body angle vs. wind)]")
                # Ellipse: above AoA ~ 20 deg the rotated needle's
                # shear layers are too thin to resolve even at Re=600.
                # Slider tightened to +/-20 deg to keep the slider's
                # reachable envelope inside the stable region (cf. the
                # per-shape Re cap above which drops to Re ~ 400 for
                # AoA > 15 deg ellipses). Square keeps the full +/-45
                # range so the validated "diamond" case is reachable
                # (slider's per-shape Re cap drops to Re ~ 200 there).
                if shape_preset == "Ellipse":
                    slider_min, slider_max, slider_default = -20.0, 20.0, 0.0
                else:
                    slider_min, slider_max, slider_default = -45.0, 45.0, 0.0
                slider_help = (
                    "Rotate the body relative to the oncoming wind. A square "
                    "at 0 deg presents a flat face (huge wake, high drag); at "
                    "45 deg it's a diamond, with a much sharper leading edge. "
                    "An ellipse rotated end-on slips through the air; rotated "
                    "broadside it slams into it. Try the extremes."
                )
            st.session_state.setdefault("lbm_aoa_slider", slider_default)
            # If the session-state AoA was set by a previous shape pick
            # outside the current range (e.g. Ellipse 40 -> picking
            # Cylinder is fine but going back to Ellipse from a Square
            # at 45 deg with state intact crashes the slider), clamp
            # it back into the new bounds.
            _prev_aoa = st.session_state.get("lbm_aoa_slider", slider_default)
            st.session_state["lbm_aoa_slider"] = float(
                np.clip(_prev_aoa, slider_min, slider_max)
            )
            aoa_deg = st.slider(
                "Body angle",
                min_value=slider_min, max_value=slider_max,
                step=0.5,
                label_visibility="collapsed",
                help=slider_help,
                key="lbm_aoa_slider",
            )
            if is_airfoil:
                st.caption(f"Wing {tilt_label(aoa_deg)}")
            elif abs(aoa_deg) < 0.25:
                st.caption("Body aligned with the wind")
            else:
                st.caption(f"Rotated {aoa_deg:+.1f} deg from horizontal")

        st.markdown("")
        st.markdown(":material/grid_view: **Resolution**")
        st.session_state.setdefault(
            "lbm_res_radio", list(RESOLUTION_PRESETS.keys())[0],
        )
        res_display = st.radio(
            "Resolution",
            list(RESOLUTION_PRESETS.keys()),
            label_visibility="collapsed",
            help=(
                "How fine the simulation grid is.\n\n"
                "- **Standard** (320 x 80 cells) — fast, ~40 s on your "
                "machine / ~3 min on Streamlit Cloud. Wake forms inside "
                "the recording window. Pick this while you're exploring.\n"
                "- **Detailed** (960 x 240 cells) — 9x more cells, 3x "
                "bigger body, longer downstream channel. ~100 s local / "
                "~6 min Cloud. The wake settles into its full periodic "
                "rhythm and airfoil downwash is much sharper. Pick this "
                "for a final render or screenshot."
            ),
            key="lbm_res_radio",
        )
        res_cfg = RESOLUTION_PRESETS[res_display]

        # Viz mode: which scalar field is painted as the background heatmap.
        # Particle streaks + body outline + scale bars are unchanged across
        # modes. Mode change triggers a re-run (cache key includes viz_mode);
        # pin the run first if you want side-by-side comparison.
        st.markdown("")
        st.markdown(":material/palette: **What to color the air with**")
        from src.lbm_render import VIZ_MODES
        st.session_state.setdefault("lbm_viz_mode", VIZ_MODES[0])
        viz_mode = st.radio(
            "Viz mode",
            VIZ_MODES,
            label_visibility="collapsed",
            help=(
                "**Vorticity** (default) -- red/blue rotation map. Best "
                "for seeing vortex shedding. **Velocity** -- speed vs the "
                "inflow: blue is slower (wake), red is faster (squeeze "
                "zones). **Pressure** -- gauge pressure: red is high (front "
                "of body), blue is low (suction). Pressure is averaged over "
                "a short rolling window to suppress LBM acoustic ripples."
            ),
            key="lbm_viz_mode",
        )

        # Shape preview: every shape (built-in or custom) gets a pre-Run
        # render of where the body sits in the tunnel. Confirms scale,
        # position, AoA rotation before the user pays for the simulation.
        # Custom path requires a polygon to be uploaded first; built-ins
        # render directly from their analytic outline.
        _preview_ready = shape_preset != "Custom" or custom_polygon is not None
        if _preview_ready:
            from src.lbm_render import render_shape_preview
            preview_png = render_shape_preview(
                shape_preset, res_cfg, aoa_deg, custom_polygon=custom_polygon,
            )
            st.markdown("")
            st.caption(":material/preview: Preview on the LBM grid:")
            st.image(preview_png, width="stretch")
            # "Flip horizontally" toggle -- only meaningful for custom
            # uploads / drawings (built-ins use aoa_deg for orientation).
            # Label reflects current state so the user sees "Flipped"
            # vs "Flip" instead of having to remember the toggle parity.
            if shape_preset == "Custom" and custom_polygon is not None:
                _is_flipped = st.session_state.get("lbm_custom_flipped", False)
                _flip_label = (
                    ":material/flip:  Flipped (click to undo)"
                    if _is_flipped
                    else ":material/flip:  Flip horizontally"
                )
                if st.button(
                    _flip_label, width="stretch", key="lbm_flip_btn",
                    help=(
                        "Mirror the shape left-to-right. Useful when your "
                        "source image was facing the wrong way -- the flow "
                        "comes from the LEFT in the simulation, so the "
                        "front of the shape should face that way."
                    ),
                ):
                    st.session_state["lbm_custom_flipped"] = not _is_flipped
                    st.rerun()

        # Custom shape requires a polygon -- disable Run if not present, so
        # the user gets a clear "upload first" hint instead of a stack trace.
        _custom_ready = shape_preset != "Custom" or custom_polygon is not None
        st.markdown("---")
        run_clicked = st.button(
            ":material/play_arrow:  **Run simulation**",
            type="primary", width="stretch",
            disabled=not _custom_ready,
            help=(
                "Upload a PNG / JPG first -- the Run button activates once "
                "the silhouette is extracted."
                if not _custom_ready else None
            ),
        )
        # Polygon goes into the config tuple via a short content hash so the
        # cache key stays stable across reruns AND distinguishes different
        # uploads. Polygon bytes themselves are too long to compare tuple-
        # equal cheaply (and numpy arrays aren't hashable).
        if custom_polygon is not None:
            import hashlib
            _polygon_key = hashlib.sha1(
                np.ascontiguousarray(custom_polygon).tobytes()
            ).hexdigest()[:12]
        else:
            _polygon_key = None
        # Track the last-displayed config so post-run buttons (Pin, Clear
        # snapshot) keep the GIF visible after their st.rerun(). Without
        # this, run_clicked is False on the rerun and the gate below bails
        # back to the "Ready to run" preview, even though the user just
        # clicked Pin on a successful run.
        _current_config = (
            shape_preset, int(reynolds_target), float(aoa_deg),
            res_display, _polygon_key, viz_mode,
        )
        if run_clicked:
            st.session_state["lbm_last_displayed_config"] = _current_config
        # If a gallery card was just clicked, the widget values have been
        # rewritten via session_state and we want the run to display
        # immediately (without the user clicking Run again).
        if st.session_state.pop("lbm_gallery_pending", False):
            st.session_state["lbm_last_displayed_config"] = _current_config
        # Auto-promote viz_mode-only changes: if the user just switched the
        # viz_mode radio while a run was displayed, treat that as "keep
        # showing, just re-render with the new mode" rather than bailing
        # to the gallery and forcing them to click Run again. Cheap thanks
        # to the solve cache: re-render runs in ~1-2 s vs the full ~40 s
        # solve. Detect by comparing all _current_config tuple entries
        # EXCEPT the viz_mode slot (index 5).
        _last_disp = st.session_state.get("lbm_last_displayed_config")
        if (
            _last_disp is not None
            and len(_last_disp) == 6
            and _last_disp[:5] == _current_config[:5]
            and _last_disp[5] != _current_config[5]
        ):
            st.session_state["lbm_last_displayed_config"] = _current_config
        _should_display_run = run_clicked or (
            st.session_state.get("lbm_last_displayed_config") == _current_config
        )
        if "Standard" in res_display:
            st.caption(":material/timer: Local: ~40 s warm, ~65 s first cold "
                       "click. Streamlit Cloud (1-vCPU shared): ~3.3 min. "
                       "Revisits are instant (cached).")
        else:
            st.caption(":material/timer: Local: ~100 s warm, ~125 s first cold "
                       "click. Streamlit Cloud (1-vCPU shared): ~6 min. "
                       "Revisits are instant (cached).")

        # === Reset to defaults ===
        # Wipes all Real-CFD-related session_state (widget values, pinned
        # snapshot, custom polygon, share-link state) so the user can start
        # clean without reloading the page. Caches are NOT cleared --
        # repeated runs hit cache as before, this is purely a UI reset.
        st.markdown("")
        if st.button(
            ":material/refresh:  Reset to defaults",
            width="stretch", key="lbm_reset_btn",
            help=(
                "Clear sliders, custom shape, snapshot, and shared-link "
                "state. Useful after exploring a lot of configurations. "
                "Does not clear the simulation cache, so re-running an "
                "earlier config is still instant."
            ),
        ):
            # Pop every lbm_*-prefixed key. Streamlit will rebuild widgets
            # with their default values on the next rerun.
            for _k in list(st.session_state.keys()):
                if _k.startswith("lbm_"):
                    st.session_state.pop(_k, None)
            # Also clear share-link query params so a future "Share link"
            # click writes a fresh set rather than appending to stale ones.
            st.query_params.clear()
            st.toast(
                ":material/refresh: Reset to defaults.",
                icon=":material/refresh:",
            )
            st.rerun()

    # === Main page header ===
    st.title("Real CFD")
    st.markdown(
        "##### Watch how air actually moves around a shape -- "
        "the same physics that lets airplanes fly, slows cars down, "
        "and once tore a bridge apart."
    )

    if not _should_display_run:
        # === Curated demo gallery ===
        # Six preconfigured runs so a first-time visitor doesn't have to
        # guess which Re / AoA produces something worth watching. Each
        # card writes its config into the sidebar widget keys + sets a
        # "pending" flag, then reruns; the flag triggers an auto-display
        # of the result without the user needing to click Run.
        #
        # Card schema: (shape_display, velocity_mps, aoa_deg, res_display,
        #               viz_mode, title, description, button_label).
        # velocity_mps maps to Re via *333.33: Re=50 -> 0.15, Re=200 ->
        # 0.60, Re=600 -> 1.80, Re=800 -> 2.40. All values are slider-tick
        # multiples of 0.1. viz_mode picks the heatmap that best showcases
        # what the card is teaching -- airfoils default to Pressure so the
        # lift mechanism (red underside / blue suction) is visible from
        # frame 1; bluff bodies stay on Vorticity (cleanest wake-structure
        # picture).
        _gallery_cards = [
            (
                "Cylinder  (round pipe)", 0.60, 0.0, "Standard (320 x 80)",
                "Vorticity",
                "Swirls behind a pole",
                "The textbook von Karman vortex street. Watch swirls peel "
                "off alternately from the top and bottom of the cylinder, "
                "carried downstream by the flow.",
                ":material/play_arrow:  Watch swirls form",
            ),
            (
                "NACA 4412  (curved wing)", 1.80, 4.0, "Standard (320 x 80)",
                "Pressure",
                "How a wing lifts (clean)",
                "Cambered wing tilted a few degrees. **Red underside, "
                "dark/blue topside** -- that pressure asymmetry IS the "
                "lift force. The thin wake shows the flow stays attached.",
                ":material/play_arrow:  See lift in action",
            ),
            (
                "NACA 4412  (curved wing)", 1.80, 20.0, "Standard (320 x 80)",
                "Pressure",
                "How a wing stalls",
                "Same wing, tilted too steep. The clean red/blue lift "
                "asymmetry collapses as the flow detaches from the top "
                "surface -- this is why aircraft fall out of the sky.",
                ":material/play_arrow:  See lift collapse",
            ),
            (
                # Re=500 (velocity 1.50). Wider stability mapping
                # discovered that Square AoA~0 at Re=600 occasionally
                # diverges in the late frames (frame ~98 of 150) when
                # the user nudges AoA off zero by a degree or two, so
                # the Square broadside Re cap is now 500. The wake is
                # still violent enough at Re=500 to read as a bluff-
                # body shedding showcase; textbook badge still applies
                # (Okajima Cd ~ 2.0 at Re=500).
                "Square  (boxy)", 1.50, 0.0, "Standard (320 x 80)",
                "Vorticity",
                "Brick in a hurricane",
                "A flat face slammed into the wind. A wide, violent wake "
                "with rapid shedding -- the kind of drag that city "
                "skylines, trucks, and shipping containers all create.",
                ":material/play_arrow:  Watch the chaos",
            ),
            (
                # The diamond (Square AoA=45 deg) orientation rotates the
                # 28-cell side so its DIAGONAL faces the flow -- the
                # effective frontal extent jumps from 28 to ~40 cells and
                # blockage rises from 35 % to ~50 %. The thinner shear
                # layers off the now-sharp leading-edge corner make this
                # orientation much less LBM-stable than broadside: tested
                # divergence boundary is around Re ~ 250 (vs ~ 1000 at
                # AoA=0). The card therefore runs at Re=200 (vel=0.60)
                # which is well inside the stable envelope and still
                # shows the qualitative "narrower wake / sharper LE" story
                # the description promises. Same physics lesson, no
                # crash on click.
                "Square  (boxy)", 0.60, 45.0, "Standard (320 x 80)",
                "Vorticity",
                "Diamond cuts the wind",
                "Same square, rotated 45 deg. Sharper leading edge, "
                "narrower wake. Tiny rotation, huge aerodynamic change -- "
                "the same trick a knife uses to slice through air.",
                ":material/play_arrow:  Watch the slice",
            ),
            (
                "Cylinder  (round pipe)", 0.15, 0.0, "Standard (320 x 80)",
                "Vorticity",
                "Almost stopped (honey)",
                "Extremely slow flow. The cylinder's wake is just two "
                "stationary bubbles -- no shedding at all. This is how "
                "honey flows around a spoon, and how plankton swim.",
                ":material/play_arrow:  Watch the calm",
            ),
        ]

        def _apply_gallery_card(card):
            # We can't write directly to lbm_shape_select / lbm_velocity_slider
            # etc. here -- those widget keys are already instantiated by the
            # sidebar (which renders BEFORE the gallery), so Streamlit refuses
            # the writes. Instead we stash the values under "pending" keys;
            # the top of the Real CFD block copies pending -> widget on the
            # NEXT rerun, BEFORE the widgets instantiate.
            shape_disp, vel_mps, aoa, res, viz, *_ = card
            st.session_state["lbm_pending_shape"] = shape_disp
            st.session_state["lbm_pending_velocity"] = float(vel_mps)
            st.session_state["lbm_pending_aoa"] = float(aoa)
            st.session_state["lbm_pending_res"] = res
            st.session_state["lbm_pending_viz"] = viz
            st.session_state["lbm_gallery_pending"] = True
            # Clear any pinned snapshot -- a fresh gallery click should
            # present a clean single-run view, not side-by-side against
            # whatever the user previously pinned.
            st.session_state.pop("lbm_snapshot", None)
            st.session_state.pop("lbm_snapshot_polygon", None)
            st.session_state.pop("lbm_snapshot_label", None)

        st.markdown(
            "### :material/auto_awesome: Try one of these"
        )
        st.caption(
            "Each card runs a pre-tuned setup so you can see a specific "
            "aerodynamic phenomenon without fiddling with sliders. "
            "Standard resolution, ~40 s local / ~3.3 min on Cloud."
        )

        # 3-col x 2-row grid of gallery cards.
        _row1, _row2 = st.columns(3), st.columns(3)
        _gal_cells = list(_row1) + list(_row2)
        for _i, _card in enumerate(_gallery_cards):
            _, _, _, _, _viz, _title, _desc, _btn = _card
            with _gal_cells[_i]:
                with st.container(border=True):
                    st.markdown(f"**{_title}**")
                    st.caption(_desc)
                    if st.button(
                        _btn, key=f"lbm_gallery_card_{_i}",
                        width="stretch", type="primary",
                    ):
                        _apply_gallery_card(_card)
                        st.rerun()

        st.markdown("")
        _preview_n_steps = res_cfg["n_frames"] * STEPS_PER_FRAME
        with st.container(border=True):
            st.markdown(
                f"### :material/play_circle: Or build your own\n\n"
                f":material/arrow_back: **Set the inputs in the sidebar** "
                f"and press **Run simulation**.\n\n"
                f"A {res_cfg['Nx']} x {res_cfg['Ny']} Lattice Boltzmann "
                f"simulation runs {_preview_n_steps:,} steps, then plays the "
                f"result back as a smooth 15 fps animation. On the *first* "
                f"click in a fresh session the solver compiles itself "
                f"(one-time, ~25 s local / ~40 s Cloud). Every click after "
                f"that is just the simulation time."
            )
        with st.expander(
                ":material/lightbulb: What you'll see -- a quick primer"):
            st.markdown(
                "The animation shows the air's flow on a dark canvas:\n\n"
                "- **Red wash** = air rotating anti-clockwise\n"
                "- **Blue wash** = air rotating clockwise\n"
                "- **Glowing particles** = massless smoke tracers released "
                "from the inflow and carried by the wind. "
                "**Colored by speed** (plasma colormap, the same one CFD "
                "post-processors like ParaView use): dark purple = slow / "
                "recirculating, orange = inflow speed, bright yellow = "
                "accelerated. Watch them deflect around the body and curl "
                "through the wake -- this is what a real wind-tunnel smoke "
                "visualization shows.\n"
                "- **Dark shape** = the object\n\n"
                "Watch what happens *behind* the object. With a cylinder at "
                "moderate flow speeds, you'll see swirls peel off alternately "
                "from the top and bottom -- a **Karman vortex street**. It's "
                "the same pattern that makes telephone wires hum in the wind."
            )
        st.stop()

    # === Cached simulate + render pipeline ===
    # Identical (shape, Re, AoA, resolution) tuples return precomputed bytes
    # instantly on repeat clicks. First call for any combination runs the
    # full 20-150 s pipeline; subsequent calls return in <50 ms. The cached
    # wrapper itself is defined at module top so @st.cache_data isn't
    # re-decorated each rerun. The heavy work lives in src/lbm_render.py.
    # Wrap the cached simulate+render in a defensive handler. The LBM
    # solver has rho-safe clamps on every macroscopic division, so a
    # well-formed shape can't trigger ZeroDivisionError in practice -- but
    # malformed custom polygons (e.g. a thread-like silhouette only one
    # cell wide, or a shape that touches the inflow wall) can drive the
    # outflow boundary's rho to zero on the first few unstable steps. We
    # don't want that to dump a Python traceback on a user who just drew
    # a wonky shape; surface a polite "try something less degenerate" hint
    # instead.
    try:
        sim_result = _cached_simulate_and_render(
            shape_preset, int(reynolds_target), float(aoa_deg), res_display,
            custom_polygon=custom_polygon, viz_mode=viz_mode,
        )
    except (ZeroDivisionError, FloatingPointError, ArithmeticError) as _sim_err:
        st.error(
            f":material/error: The simulation went numerically unstable "
            f"(`{type(_sim_err).__name__}: {_sim_err}`). This usually "
            f"happens with very thin / degenerate shapes -- try one of "
            f"the built-in presets, the **Sample** tab, or simplify your "
            f"drawing (fatter strokes, single closed loop, no thread-like "
            f"branches)."
        )
        st.stop()
    except ValueError as _shape_err:
        # Pre-flight mask validation in solve_lbm raises ValueError with
        # an actionable message ("touches the inflow wall", "occupies too
        # much of the channel", etc.) -- surface it verbatim so the user
        # knows exactly what to fix rather than seeing a generic stack
        # trace. Distinct from the ZeroDivision path above: this one
        # caught the problem BEFORE the simulation, so we phrase it as
        # geometry feedback, not numerical instability.
        st.error(
            f":material/error: This shape can't be simulated: "
            f"{_shape_err}"
        )
        st.stop()
    except Exception as _sim_err:
        # Any other unexpected error: still surface a polite message and
        # log the type so a future debugging pass can find it.
        st.error(
            f":material/error: Something went wrong running the "
            f"simulation: `{type(_sim_err).__name__}: {_sim_err}`. If "
            f"this is reproducible with one of the gallery cards, please "
            f"flag it on the repo's Issues tab so we can fix it."
        )
        st.stop()
    tau = sim_result["tau"]
    nu = sim_result["nu"]
    char_length = sim_result["char_length"]
    LBM_NX = sim_result["lbm_nx"]
    LBM_NY = sim_result["lbm_ny"]
    label = sim_result["label"]
    gif_bytes = sim_result["gif_bytes"]
    bg_cbar_bytes = sim_result["bg_cbar_bytes"]
    bg_cbar_title = sim_result["bg_cbar_title"]
    bg_cbar_blurb = sim_result["bg_cbar_blurb"]
    speed_cbar_bytes = sim_result["speed_cbar_bytes"]
    actual_n_frames = sim_result["n_frames"]
    actual_n_steps = sim_result["n_steps"]

    if sim_result["near_stable"]:
        st.warning(
            ":material/warning: At this flow speed the solver gets noisy -- "
            "the wake direction is right but small-scale turbulence is "
            "under-resolved. Try **Detailed** resolution for sharper "
            "structure, or step the Flow speed slider down."
        )

    # Custom-shape Cd caveat is hoisted into the new metrics block below
    # the Cd number, where the user is actually looking. (Old: a separate
    # st.info banner sat above the GIF -- visually disconnected from the
    # number it explains.)

    # === Comparison snapshot (matches Fast-mode side-by-side affordance) ===
    # User can "pin" any run; the next run displays side-by-side with the
    # pinned one. Pinned config is a (shape, Re, AoA, res) tuple stored in
    # session state; we re-run via the cache (instant since it's the same
    # cache key) instead of stashing the GIF bytes themselves -- avoids
    # session bloat. Single pinned snapshot for now; expand to a list when
    # the use case justifies it. _current_config was computed up in the
    # sidebar block so the post-run gate could use it.
    snapshot = st.session_state.get("lbm_snapshot")
    snapshot_is_current = snapshot == _current_config

    if snapshot is not None and not snapshot_is_current:
        # Snapshot tuple: (shape, Re, AoA, res, polygon_key, viz_mode).
        # For Custom shapes we also stash the polygon array in session
        # state under "lbm_snapshot_polygon" so the cache lookup hits and
        # the solver has something to run on. For preset shapes the
        # polygon slot is None. Older 5-tuple snapshots (from before
        # viz_mode reintroduction) fall back to "Vorticity".
        if len(snapshot) == 6:
            snap_shape, snap_re, snap_aoa, snap_res, _snap_poly_key, snap_viz = snapshot
        else:
            snap_shape, snap_re, snap_aoa, snap_res, _snap_poly_key = snapshot
            snap_viz = "Vorticity"
        snap_polygon = st.session_state.get("lbm_snapshot_polygon")
        snap_result = _cached_simulate_and_render(
            snap_shape, snap_re, snap_aoa, snap_res,
            custom_polygon=snap_polygon, viz_mode=snap_viz,
        )
        st.markdown("---")
        st.markdown("### Side-by-side comparison")
        st.caption(
            "Snapshot is shown on the left, current run on the right. "
            "Clear the snapshot below to return to single-run view."
        )
        cmp_cols = st.columns(2)
        with cmp_cols[0]:
            _snap_aoa_part = f"  ·  {snap_aoa:+.1f}°" if abs(snap_aoa) > 0.25 else ""
            _snap_viz_part = f"  ·  {snap_viz}" if snap_viz != "Vorticity" else ""
            st.markdown(
                f"**Snapshot:** {snap_shape}  ·  Re {snap_re}"
                f"{_snap_aoa_part}{_snap_viz_part}"
            )
            st.image(snap_result["gif_bytes"], width="stretch")
        with cmp_cols[1]:
            _cur_aoa_part = f"  ·  {aoa_deg:+.1f}°" if abs(aoa_deg) > 0.25 else ""
            _cur_viz_part = f"  ·  {viz_mode}" if viz_mode != "Vorticity" else ""
            st.markdown(
                f"**Current:** {shape_preset}  ·  Re {int(reynolds_target)}"
                f"{_cur_aoa_part}{_cur_viz_part}"
            )
            st.image(gif_bytes, width="stretch")

    # === Display: hero animation, colorbar, plain-English legend ===
    st.markdown("---")
    shape_name = shape_display.split("  (")[0]
    if shape_preset == "Custom":
        _rot_part = (
            f"  ·  rotated {aoa_deg:+.1f} deg" if abs(aoa_deg) > 0.25 else ""
        )
        st.markdown(f"### :material/air: Your shape in {reg}{_rot_part}")
    elif shape_preset in ("NACA 0012", "NACA 4412"):
        st.markdown(
            f"### :material/air: {shape_name}, {reg}  ·  "
            f"wing tilt {aoa_deg:+.1f} deg"
        )
    else:
        st.markdown(f"### :material/air: {shape_name} in {reg}")

    with st.container(border=True):
        st.image(gif_bytes, width="stretch")

        # Action row: download GIF, pin for comparison, clear pin (if set).
        if shape_preset == "Custom":
            # For custom shapes, differentiate by sample name (when loaded
            # from "Try a sample") or polygon hash prefix (when uploaded),
            # so multiple custom-shape downloads don't collide.
            _sample_label = st.session_state.get("lbm_custom_label")
            if _sample_label:
                _shape_slug = "custom_" + _sample_label.lower().replace(" ", "_")
            else:
                _shape_slug = f"custom_{(_polygon_key or 'shape')[:8]}"
        else:
            _shape_slug = shape_preset.lower().replace(" ", "_")
        _aoa_part = f"_aoa{aoa_deg:+.0f}" if abs(aoa_deg) > 0.25 else ""
        _gif_name = f"aerolab_{_shape_slug}_re{reynolds_target}{_aoa_part}.gif"
        action_cols = st.columns([1, 1, 1, 1, 2])
        with action_cols[0]:
            st.download_button(
                ":material/download:  Download GIF",
                data=gif_bytes,
                file_name=_gif_name,
                mime="image/gif",
                width="stretch",
                help="Save the animation locally. Filename encodes shape, Re, "
                     "and AoA so multiple runs don't collide.",
            )
        with action_cols[1]:
            _pin_label = (
                ":material/push_pin:  Pinned (change params to compare)"
                if snapshot_is_current else
                ":material/push_pin:  Pin for comparison"
            )
            if st.button(
                _pin_label, width="stretch",
                disabled=snapshot_is_current,
                help=(
                    "Save this run as a comparison snapshot. The next run "
                    "with different parameters will display side-by-side "
                    "against this pinned snapshot."
                ),
                key="pin_for_comparison",
            ):
                st.session_state["lbm_snapshot"] = _current_config
                # For Custom shapes, stash the polygon + label so the
                # side-by-side comparison can rebuild the snapshot run
                # (cache key includes the polygon hash; without the array
                # we'd hit a cache miss and have nothing to feed the
                # solver). Cleared together with the snapshot below.
                if shape_preset == "Custom" and custom_polygon is not None:
                    st.session_state["lbm_snapshot_polygon"] = custom_polygon
                    st.session_state["lbm_snapshot_label"] = (
                        st.session_state.get("lbm_custom_label")
                    )
                else:
                    st.session_state.pop("lbm_snapshot_polygon", None)
                    st.session_state.pop("lbm_snapshot_label", None)
                _snap_aoa_part = f" AoA {aoa_deg:+.0f}deg" if abs(aoa_deg) > 0.25 else ""
                _snap_shape_label = (
                    st.session_state.get("lbm_custom_label", "Custom shape")
                    if shape_preset == "Custom" else shape_preset
                )
                st.toast(
                    f":material/push_pin: Pinned: {_snap_shape_label} Re={int(reynolds_target)}{_snap_aoa_part}. "
                    f"Change a parameter and click Run to see side-by-side.",
                    icon=":material/push_pin:",
                )
                st.rerun()
        with action_cols[2]:
            if snapshot is not None:
                if st.button(
                    ":material/close:  Clear snapshot",
                    width="stretch",
                    help="Remove the pinned snapshot and return to single-run view.",
                    key="clear_comparison",
                ):
                    del st.session_state["lbm_snapshot"]
                    st.session_state.pop("lbm_snapshot_polygon", None)
                    st.session_state.pop("lbm_snapshot_label", None)
                    st.rerun()
        with action_cols[3]:
            # Share button: encode the current config in URL query params
            # so the user can copy the URL from their address bar and
            # share a deep link that reopens this exact run. Custom shapes
            # are disabled because the polygon doesn't fit in a URL --
            # those users share the GIF download instead.
            _share_disabled = shape_preset == "Custom"
            _share_help = (
                "Custom (uploaded / drawn) shapes can't be encoded in a "
                "URL -- download the GIF to share instead."
                if _share_disabled else
                "Write this run's config to the URL bar. Then copy your "
                "browser's address from the address bar -- anyone opening "
                "that link lands directly on this exact run."
            )
            if st.button(
                ":material/share:  Share link",
                width="stretch",
                disabled=_share_disabled,
                help=_share_help,
                key="share_link_btn",
            ):
                st.query_params["shape"] = _SHAPE_DISPLAY_TO_QP[shape_preset]
                st.query_params["vel"] = f"{velocity_mps:.2f}"
                st.query_params["aoa"] = f"{aoa_deg:.1f}"
                st.query_params["res"] = (
                    "standard" if "Standard" in res_display else "detailed"
                )
                st.query_params["viz"] = viz_mode
                st.toast(
                    ":material/share: Link ready -- copy from your "
                    "browser's address bar.",
                    icon=":material/share:",
                )

        # Persistent pinned-state caption -- gives the user feedback that
        # something IS pinned, since pinning before changing params has no
        # other visible effect on the current view.
        if snapshot is not None:
            # Tolerate both 6-tuple (new, includes viz_mode) and legacy 5-tuple.
            if len(snapshot) == 6:
                snap_shape, snap_re, snap_aoa, snap_res, _snap_poly_key, _snap_viz = snapshot
            else:
                snap_shape, snap_re, snap_aoa, snap_res, _snap_poly_key = snapshot
            snap_aoa_str = f", AoA {snap_aoa:+.0f}deg" if abs(snap_aoa) > 0.25 else ""
            snap_res_short = "Standard" if "Standard" in snap_res else "Detailed"
            if snapshot_is_current:
                _pin_msg = (
                    f":material/push_pin: **Pinned: this run.** "
                    f"Change a parameter and click Run to see side-by-side."
                )
            else:
                _pin_msg = (
                    f":material/push_pin: **Pinned: {snap_shape}, "
                    f"Re={snap_re}{snap_aoa_str}, {snap_res_short}.** "
                    f"Showing side-by-side with the current run above."
                )
            st.caption(_pin_msg)

        # Background (vorticity) colorbar + caption. Strings come from
        # sim_result so the renderer is the single source of truth.
        st.markdown(
            f"<div style='color:#94a3b8;font-size:0.78rem;"
            f"letter-spacing:0.05em;text-transform:uppercase;"
            f"margin:1.2rem 0 0.1rem 0;'>"
            f"{bg_cbar_title}"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.image(bg_cbar_bytes, width="stretch")
        st.markdown(
            "<div style='color:#94a3b8;font-size:0.78rem;"
            "letter-spacing:0.05em;text-transform:uppercase;"
            "margin:0.4rem 0 0.1rem 0;'>"
            "Particle colors — air's speed"
            "</div>",
            unsafe_allow_html=True,
        )
        st.image(speed_cbar_bytes, width="stretch")

    # Legend uses inline colored swatches (HTML span) instead of :material/
    # icons. Two reasons: (1) Material icons render via Streamlit's frontend
    # JS, so if the page is exported as PDF or copied as text the icons drop
    # out and you get literal ":material/blur_on:" strings. (2) Colored
    # swatches double as a visual key -- the user can map "red wash" in the
    # text to the actual red they see in the animation.
    _swatch = (
        "<span style='display:inline-block;width:12px;height:12px;"
        "border-radius:3px;background:{color};margin-right:0.4rem;"
        "vertical-align:middle;'></span>"
    )
    # Per-viz-mode swatch + caption for the first two legend columns.
    # Slots 3 and 4 (particles, body shape) are mode-independent and
    # stay below.
    _BG_LEGEND_BY_MODE = {
        "Vorticity": (
            ("#b91c1c", "Red wash",
             "Air rotating *anti-clockwise* -- vortices spinning one way."),
            ("#1d4ed8", "Blue wash",
             "Air rotating *clockwise* -- the other way. Together they "
             "form the von Karman street."),
        ),
        "Velocity": (
            ("#b91c1c", "Red wash",
             "**Faster** than the inflow -- squeezed around bumps or "
             "accelerating over the suction side of an airfoil."),
            ("#1d4ed8", "Blue wash",
             "**Slower** than the inflow -- the stalled wake behind a "
             "bluff body, or the separated region above a stalled wing."),
        ),
        "Pressure": (
            ("#b91c1c", "Red regions",
             "**High** pressure -- air piling up against the front face. "
             "Strongest at stagnation points."),
            ("#1d4ed8", "Blue regions",
             "**Low** pressure -- suction. On a tilted wing this is "
             "mostly on the top surface and is what generates lift."),
        ),
    }
    _left_swatch, _right_swatch = _BG_LEGEND_BY_MODE[viz_mode]

    st.markdown("##### What you're looking at")
    leg_cols = st.columns(4)
    with leg_cols[0]:
        st.markdown(
            _swatch.format(color=_left_swatch[0]) + f"**{_left_swatch[1]}**",
            unsafe_allow_html=True,
        )
        st.markdown(_left_swatch[2])
    with leg_cols[1]:
        st.markdown(
            _swatch.format(color=_right_swatch[0]) + f"**{_right_swatch[1]}**",
            unsafe_allow_html=True,
        )
        st.markdown(_right_swatch[2])
    with leg_cols[2]:
        st.markdown(
            _swatch.format(color="#fde047") + "**Glowing particles**",
            unsafe_allow_html=True,
        )
        st.markdown(
            "Massless smoke tracers carried by the wind, "
            "**colored by speed** (perceptually uniform plasma colormap): "
            "dark purple = slow / recirculating, orange = inflow speed, "
            "yellow = accelerated."
        )
    with leg_cols[3]:
        st.markdown(
            _swatch.format(color="#1f2937") + "**Dark shape**",
            unsafe_allow_html=True,
        )
        st.markdown(
            "The object. Air can't flow through it -- it has to go around."
        )

    # === Metric strip ===
    # CFD-internal numbers (tau, lattice viscosity, etc.) live in the
    # "Under the hood" expander below. This strip stays in plain-language
    # territory: what the wind is doing + how the animation was made.
    st.markdown("")
    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric(
            ":material/speed: Flow speed",
            f"{velocity_mps:.2f} m/s",
            f"Re {reynolds_target} · {reg}",
        )
    with metric_cols[1]:
        st.metric(":material/footprint: Simulation steps", f"{actual_n_steps:,}")
    with metric_cols[2]:
        st.metric(":material/movie: Playback",
                   f"{actual_n_frames} frames @ {round(1000 / GIF_FRAME_MS)} fps")

    # === Measured forces (collapsed by default — for the curious / portfolio
    # reviewers, not focal for a "see air" viewer) ===
    # The solver runs a momentum-exchange force calculation on the body every
    # step. Cd/Cl are means over the last third of the run; Strouhal is the
    # dominant FFT peak of the Cl history. Cylinder runs get a textbook badge.
    with st.expander(
            ":material/insights: **Forces measured during this run** "
            "(drag, lift, vortex-shedding frequency)"):
        _st_val = sim_result.get("strouhal", float("nan"))
        # Textbook reference: ship comparison data for the canonical
        # validated cases. Cylinder always shows; Square shows only when
        # the body presents a flat face (aoa ~ 0) since the table is for
        # broadside flow only. Used both as inline deltas under the
        # metric tiles and as a single honest summary below the plot.
        _show_textbook = (
            shape_preset == "Cylinder"
            or (shape_preset == "Square" and abs(aoa_deg) < 5.0)
        )
        _cd_ref, _st_ref = (
            _textbook_reference(shape_preset, int(reynolds_target))
            if _show_textbook else (None, None)
        )
        # Inline deltas on the Cd / Strouhal metric tiles -- compact "vs
        # textbook" indicator without burying it in a paragraph.
        # delta_color="off" keeps the chip gray because our Cd reads high
        # for structural reasons (confinement + bounce-back), not because
        # of user error -- red/green would mislead.
        _cd_delta = (
            f"{sim_result['cd_mean'] - _cd_ref:+.2f} vs textbook {_cd_ref:.2f}"
            if _cd_ref is not None else None
        )
        _st_delta = (
            f"{_st_val - _st_ref:+.3f} vs textbook {_st_ref:.3f}"
            if (_st_ref is not None and np.isfinite(_st_val)) else None
        )

        _force_cols = st.columns(3)
        with _force_cols[0]:
            st.metric(
                "Drag (Cd)",
                f"{sim_result['cd_mean']:.2f}",
                delta=_cd_delta,
                delta_color="off",
                help=(
                    "**Drag coefficient.** How hard the air pushes back "
                    "on the shape. Bigger = more drag. A brick has Cd "
                    "around 2; a sleek teardrop is below 0.1. Averaged "
                    "over the last third of the run, so the start-up "
                    "transient is excluded."
                ),
            )
            st.caption(
                ":gray[Air resistance. Lower = sleeker.]"
            )
        with _force_cols[1]:
            st.metric(
                "Lift (Cl)",
                f"{sim_result['cl_mean']:+.2f}",
                help=(
                    "**Lift coefficient.** Sideways push, perpendicular "
                    "to the wind. A wing tilted into the flow gets "
                    "positive lift; a symmetric shape at zero tilt "
                    "averages ~0 (the up-and-down swirls cancel out). "
                    "Same averaging window as Drag."
                ),
            )
            st.caption(
                ":gray[Sideways push. Wings need it; bluff bodies have ~0.]"
            )
        with _force_cols[2]:
            if np.isfinite(_st_val):
                st.metric(
                    "Shedding rhythm (St)",
                    f"{_st_val:.3f}",
                    delta=_st_delta,
                    delta_color="off",
                    help=(
                        "**Strouhal number.** How often vortices peel "
                        "off the back of the shape, scaled by size and "
                        "speed. Stays around 0.2 for a cylinder no "
                        "matter the wind speed -- which is exactly why "
                        "telephone wires can hum a steady musical note "
                        "in a breeze."
                    ),
                )
                st.caption(
                    ":gray[How fast vortices peel off. ~0.2 for cylinders.]"
                )
            else:
                st.metric(
                    "Shedding rhythm (St)", "—",
                    help=(
                        "The run was too short (or the shape too steady) "
                        "to lock onto a shedding frequency. Try **Detailed** "
                        "resolution (more frames) or bump Flow speed higher "
                        "so the vortices fire faster."
                    ),
                )
                st.caption(
                    ":gray[Run was too short to spot the rhythm. "
                    "Try Detailed mode.]"
                )
        st.image(sim_result["force_plot_bytes"], width="stretch")
        st.caption(
            ":material/info: The shaded band is the window used for the "
            "mean Cd / Cl values above. If Cd hasn't settled yet at the "
            "right edge, the run is still transient and the mean is "
            "unreliable -- try a longer / Detailed run."
        )

        # === Free-stream reference (the honest "what would IRL measure?"
        # number) ===
        # Our solver runs in a confined channel: Standard has 35 %
        # blockage (D=28 in Ny=80), Detailed has 33 % (D=80 in Ny=240).
        # The walls accelerate the local flow past the body, which
        # inflates the measured Cd by ~25-40 % vs. wind-tunnel free
        # stream. Surface BOTH numbers and explain the gap, so the user
        # walks away with: (a) confidence that the solver is working
        # (vs. _REFERENCE_CD), AND (b) the real-world Cd they would
        # cite in a report (from _FREESTREAM_CD). This is the
        # "very close to IRL data" deliverable -- the corrected number,
        # not the raw confined-channel one.
        _cd_free, _st_free = _freestream_reference(
            shape_preset, int(reynolds_target),
        )
        if _show_textbook and (_cd_free is not None or _st_free is not None):
            _free_bits = []
            if _cd_free is not None:
                _free_bits.append(f"**Cd &asymp; {_cd_free:.2f}**")
            if _st_free is not None and np.isfinite(_st_val):
                _free_bits.append(f"**St &asymp; {_st_free:.3f}**")
            _free_line = " &nbsp;·&nbsp; ".join(_free_bits)
            # Resolution-aware blockage ratio for the caption -- LBM_NY
            # and char_length live on sim_result.
            _ny = sim_result.get("lbm_ny", 80)
            _L = sim_result.get("char_length", 1.0)
            _blockage_pct = round(100.0 * _L / max(_ny, 1))
            st.info(
                f":material/public: **In unbounded flow** (wind-tunnel "
                f"free-stream, Williamson / Okajima): {_free_line}.  "
                f"&nbsp;&nbsp; Our run is in a {_blockage_pct} %-blocked "
                f"channel, which inflates the raw measured Cd by ~"
                f"{round(100 * (sim_result['cd_mean'] - _cd_free) / max(_cd_free, 0.01)) if _cd_free else 0} %. "
                f"After applying the Allen-Vincenti blockage correction "
                f"(a ~2.65x rescale at this blockage), the **corrected "
                f"estimate** of free-stream Cd matches Williamson / "
                f"Okajima within +/- 15 % (cylinder) / +/- 25 % (square). "
                f"Strouhal is reported with the West-Apelt correction "
                f"but the channel-resonance shedding at this blockage "
                f"limits its predictive value -- see "
                f"[VALIDATION.md](https://github.com/devansh2003-dev/AeroLab/blob/main/VALIDATION.md) "
                f"section 4.1 for the honest discussion."
            )

        # Compact textbook-comparison verdict. Green if Strouhal matched
        # within 25 % (the cleaner physics check, less sensitive to grid
        # bias than Cd), blue if it didn't (or wasn't computed). Cd-vs-
        # textbook framing is in the delta chip above, not repeated here.
        if _show_textbook and _cd_ref is not None:
            _st_err_pct = (
                abs(_st_val - _st_ref) / _st_ref * 100
                if (_st_ref is not None and np.isfinite(_st_val)) else None
            )
            _bias_text = {
                "Cylinder": (
                    "Cd is biased ~50-100 % high (channel confinement "
                    "+ grid resolution); Strouhal is the cleaner cross-check."
                ),
                "Square": (
                    "Square Cd is geometry-locked at the corners, so it's "
                    "flatter than the cylinder's. Ours still reads high "
                    "(confinement + halfway bounce-back); Strouhal is the "
                    "cleaner cross-check."
                ),
            }[shape_preset]
            if _st_err_pct is not None and _st_err_pct < 25:
                st.success(
                    f":material/check_circle: **Shedding physics match** "
                    f"-- Strouhal within {_st_err_pct:.0f} % of textbook "
                    f"at Re={int(reynolds_target)}.  {_bias_text}"
                )
            else:
                st.info(
                    f":material/info: **Textbook reference active** "
                    f"({shape_preset}, Re={int(reynolds_target)}). "
                    + (
                        f"Strouhal is {_st_err_pct:.0f} % off -- try a "
                        f"longer / Detailed run so the FFT has more "
                        f"cycles to lock onto. "
                        if _st_err_pct is not None else ""
                    )
                    + _bias_text
                )

        # Custom shapes use halfway BB everywhere -- the Cd reads ~30-50 % high.
        if shape_preset == "Custom":
            st.caption(
                ":material/info: **Cd note for custom shapes:** halfway "
                "bounce-back is less accurate than the analytic Bouzidi "
                "scheme used for built-in shapes, so this Cd reads ~30-50 % "
                "high. Wake structure and Strouhal are fine."
            )

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
            f"**Method:** D2Q9 Lattice Boltzmann on a {LBM_NX} x {LBM_NY} grid "
            f"with **MRT (multi-relaxation-time) collision + Smagorinsky LES**, "
            f"**Bouzidi interpolated bounce-back** at the body surface, and "
            f"**Zou & He** velocity inflow + pressure outflow. Bounce-back at "
            f"top/bottom walls (no periodic-y wraparound). Numba-JIT compiled "
            f"fused step (collide + force + bounce-back + stream + BCs in one "
            f"function, serial; threading was stripped after a Streamlit "
            f"Cloud env conflict).\n\n"
            f"**Why MRT + LES:** MRT projects the 9 populations onto a moment "
            f"basis and relaxes each moment with its own rate. The "
            f"viscous-stress moments use s = 1/tau (same kinematic viscosity as "
            f"BGK), but bulk-viscosity and ghost-moment rates are free "
            f"parameters tuned for stability + cleaner near-body vorticity. "
            f"Smagorinsky adds per-cell eddy viscosity in high-strain regions "
            f"(corners, shear layers) so sharp-edged bluff bodies stay stable "
            f"through Re=1500. References: Lallemand & Luo (2000), d'Humieres "
            f"et al. (2002), Lilly (1967) for the C_smag constant, Bouzidi-"
            f"Firdaouss-Lallemand (2001) for the curved-wall correction, "
            f"Mei-Yu-Shyy-Luo (2002) for the Bouzidi-aware momentum exchange, "
            f"Zou & He (1997) for the BCs.\n\n"
            f"**Honest comparison to industrial CFD.** We share the *collision-"
            f"rule family* (MRT + LES) with Dassault's PowerFLOW, ProLB, and "
            f"M-Star -- and with academic LBM codes like Palabos or waLBerla. "
            f"But sharing the collision rule is like sharing 'has 4 wheels' "
            f"with a Formula 1 car. Real industrial LBM solvers add: GPU + "
            f"multi-GPU + multi-node compute, octree adaptive mesh refinement, "
            f"wall-function turbulence models (so they don't need to resolve "
            f"the boundary layer), cumulant collision for transitional flow, "
            f"automatic time-step control, multi-block domain decomposition, "
            f"3D, and a 30+ year validation library across thousands of "
            f"industrial cases. Industrial codes routinely run Re >= 10^6 "
            f"with these tools. **We're at Re <= 1500 in 2D with a uniform "
            f"grid -- a serious academic-style toy, not an industrial tool.**\n\n"
            f"**What we ship that an undergrad-style LBM tutorial usually "
            f"doesn't:** MRT (not BGK) for the production path, Smagorinsky "
            f"LES with a literature-grounded constant, Bouzidi interpolated "
            f"bounce-back (not staircase voxelization), Zou-He BCs (not "
            f"equilibrium inflow + zero-gradient outflow), Mei-aware momentum "
            f"exchange on the Bouzidi links, JIT-compiled fused step, "
            f"NumPy/JIT bit-equivalence tests, grid-convergence validation "
            f"with Richardson extrapolation, and per-shape analytic q-fields.\n\n"
            f"**What we still don't have:** OpenFOAM / Fluent / Star-CCM+ "
            f"cross-validation. The validation script compares against "
            f"textbook free-stream cylinder Re=100 numbers (Strouhal 0.165, "
            f"Cd 1.4) -- a 1980s reference table, not a contemporary "
            f"co-run. That cross-comparison is roadmapped as Phase 3 work "
            f"(install OpenFOAM, set up the same cylinder, run, compare). "
            f"3D, GPU, AMR, wall functions, and cumulant collision are all "
            f"out of scope for this 12-week project.\n\n"
            f"**Why the airfoil downwash looks weak:** real wings cruise at "
            f"Re ~ 10^7. We're running Re = {reynolds_target} -- at Re=200 the "
            f"viscous boundary layer is so thick relative to the chord that "
            f"the airfoil behaves more like an inclined plate than the "
            f"textbook thin-airfoil-theory wing you've seen in aero classes. "
            f"Bump Re to 500-1000 to see the lift mechanism more clearly.\n\n"
            f"**Heatmap:** signed vorticity omega = curl(u). Red = omega > 0 "
            f"(anti-clockwise), blue = omega < 0 (clockwise). RdBu_r colormap "
            f"is alpha-modulated -- omega ~ 0 is transparent so the dark "
            f"background shows through, peak alpha capped at 90% so the wake "
            f"reads as a wash.\n\n"
            f"**This run:**\n"
            f"- {actual_n_steps} time steps over {actual_n_frames} frames "
            f"(loop = {actual_n_frames * GIF_FRAME_MS / 1000:.1f} s)\n"
            f"- tau = {tau:.4f}  (kinematic relaxation; stable while > 0.5)\n"
            f"- nu = {nu:.5f}  (kinematic viscosity, lattice units)\n"
            f"- Re = U L / nu = {U_INFLOW} x {char_length:.0f} / {nu:.5f} = {reynolds_target}\n"
            f"- characteristic length L = {char_length:.0f}  (lattice cells)\n"
            f"- MRT free rates: s_e = s_eps = s_q = 1.4; C_SMAG = 0.17 (Lilly)"
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
        help=(
            "**Reynolds number** — how fast and turbulent the air feels "
            "to the wing.\n\n"
            "- :material/explore: Hand glider / RC plane: ~1e5 (100,000)\n"
            "- :material/sports: Sailplane / glider: ~5e5 (500,000)\n"
            "- :material/flight: Light aircraft cruise: ~3e6 – 6e6\n"
            "- :material/flight_takeoff: Jet airliner cruise: ~1e7+\n\n"
            ":gray[*Bigger = faster / bigger wing / thinner air. Set "
            "this to match the real flight regime you care about.*]"
        ),
    )

    st.markdown("")
    st.markdown(":material/auto_awesome: **Model quality**")
    nf_model_display = st.radio(
        "nf_model_size",
        list(NF_MODEL_PRESETS.keys()),
        index=0,
        label_visibility="collapsed",
        help=(
            "How accurate the ML model is — at the cost of speed.\n\n"
            "- **Best**: research-grade. Within ~3 % of the gold-standard "
            "industry tool (XFoil). Use this for a final report or polar.\n"
            "- **Balanced**: ~2x faster, ~5 % off. Good default for "
            "exploration.\n"
            "- **Fast**: ~5x faster, ~10 % off. Pick this when you're "
            "sweeping a big range of angles or Reynolds numbers and just "
            "want to see the *shape* of the trend."
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
