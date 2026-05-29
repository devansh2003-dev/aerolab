"""AeroLab Streamlit app -- dual-mode airfoil aerodynamics.

Two modes via the sidebar toggle:
  - Fast (NeuralFoil): instant ML predictions, alpha sweeps, drag polar.
  - Real CFD (LBM): browser-based Lattice Boltzmann simulation rendered as a GIF.

Run from the project root:
    streamlit run app.py

The browser opens automatically at http://localhost:8501.
"""
# Set NUMBA_NUM_THREADS BEFORE any other import. Two failure modes to
# pre-empt, both diagnosed against real crashes:
#
#   Cloud: container starts with NUMBA_NUM_THREADS UNSET, numba
#       auto-detects cpu_count=1 (cgroup throttle on the 1-vCPU
#       container) and launches 1 thread. Cloud's request handler
#       later SETS NUMBA_NUM_THREADS to 16 (the host's logical CPU
#       count). The next JIT compile triggers `reload_config`, which
#       sees env=16 vs launched=1 and raises "Cannot set
#       NUMBA_NUM_THREADS to a different value once the threads have
#       been launched (currently 1, trying to set 16)".
#
#   Local: container starts with NUMBA_NUM_THREADS UNSET (just like
#       Cloud), we used to force env=16 here. Numba launched with 16,
#       and SOMETHING — Streamlit's runner, numba's own
#       multiprocessing helper, or a transitive C extension — later
#       clears the env var (no code we own does this; the reviewer
#       grepped and so did we). At the next JIT, `reload_config`
#       reads env-empty, falls back to NUMBA_DEFAULT_NUM_THREADS
#       (= cpu_count = 14 on the dev laptop), and crashes with the
#       MIRROR error: "currently 16, trying to set 14".
#
# The fix that survives BOTH races: pick a value that matches what
# numba's own auto-detect would land on, so even if env is cleared
# the fallback equals the launched count. Locally that's cpu_count().
# On Cloud, cpu_count() = 1 would re-introduce the original 1-vs-16
# race when Cloud's runtime forces 16; so on Cloud we keep "16",
# matching Cloud's later assignment. Cloud is detected by the mount
# point Streamlit Cloud uses, which does not exist locally on any
# OS.
import os
import sys

_IS_STREAMLIT_CLOUD = sys.platform == "linux" and os.path.isdir("/mount/src")
_NUMBA_THREADS_VALUE = "16" if _IS_STREAMLIT_CLOUD else str(os.cpu_count() or 1)
os.environ["NUMBA_NUM_THREADS"] = _NUMBA_THREADS_VALUE

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

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
from src.airfoils import analyze_airfoil, get_airfoil, normalize_naca, thickness_camber
from src.references import (
    blockage_corrected as _blockage_corrected,
)
from src.references import (
    freestream_reference as _freestream_reference,
)
from src.references import (
    textbook_reference as _textbook_reference,
)

# --- Page config (must be the first Streamlit call) ---
st.set_page_config(
    page_title="AeroLab",
    layout="wide",
    # Force the sidebar open on first load. With "auto" Streamlit collapses
    # it on narrow viewports / after manual close, and the collapse chevron
    # is easy to miss -- making the 2D/3D solver toggle effectively
    # undiscoverable for a first-time visitor.
    initial_sidebar_state="expanded",
)

# Hide Streamlit Cloud's "Manage app" / "Hosted with Streamlit" chrome so
# the GIF doesn't compete with the deploy badge for the bottom-right of
# the viewport. Streamlit's MainMenu and the footer get class names that
# are stable enough to target via attribute / data-testid selectors.
st.markdown(
    "<style>"
    # Pull Inter from Google Fonts and apply via the cascade only.
    # Earlier rev used `[class*='st-'] {font-family: Inter !important}`
    # which also matched Streamlit's material-icon spans -- overriding
    # the Material Symbols font and printing icon names as raw text
    # (e.g. 'keyboard_double_arrow_right') in every :material/...:
    # shortcode. Setting font-family on html/body only lets Streamlit's
    # per-element CSS for icon fonts win on the icon elements.
    "@import url('https://fonts.googleapis.com/css2?"
    "family=Inter:wght@400;500;600;700&display=swap');"
    "html, body {font-family: 'Inter', -apple-system, BlinkMacSystemFont, "
    "'Segoe UI', sans-serif;}"
    # Hide Streamlit chrome that competes with the GIF/Plotly canvas,
    # but do NOT blanket-hide [data-testid='stToolbar'] -- in Streamlit
    # 1.57 the sidebar's expand chevron ([data-testid='stExpandSidebarButton'])
    # is rendered in the same header region, and hiding the whole bar
    # took the chevron with it (regression caught 2026-05-26 -- user
    # reported the sidebar had no expand/collapse control at all).
    # Instead, hide only the SPECIFIC actions inside the toolbar.
    "#MainMenu {display: none;}"
    "footer {visibility: hidden;}"
    ".stDeployButton {display: none;}"
    "[data-testid='stToolbarActions'] {display: none;}"
    "[data-testid='stStatusWidget'] {display: none;}"
    # Force the sidebar collapse + expand buttons to render with high
    # z-index and full opacity, so neither our other rules nor Streamlit
    # cloud chrome can hide them.
    "[data-testid='stSidebarCollapseButton'], "
    "[data-testid='stExpandSidebarButton'] {"
    "display: flex !important; visibility: visible !important;"
    "opacity: 1 !important; z-index: 1000;}"
    # Help-icon tooltips need to draw OVER the result GIF / Plotly chart
    # layers; without an explicit z-index they get occluded by the viz
    # container's own stacking context (reviewer item #22, 2026-05-25).
    "[data-baseweb='tooltip'], [role='tooltip'] {z-index: 9999 !important;}"
    # --- Card / panel polish (2026-05-26) ---
    # Expander frames: rounded, subtle border that picks up the slate
    # palette instead of Streamlit's default grey. Removes the harsh
    # box look and unifies them with the new hero block.
    "[data-testid='stExpander'] {"
    "background: rgba(17,24,39,0.55); border: 1px solid #1f2937;"
    "border-radius: 10px; overflow: hidden; "
    "transition: border-color 0.18s ease, background 0.18s ease;}"
    "[data-testid='stExpander']:hover {border-color: #334155;}"
    "[data-testid='stExpander'] summary {padding: 0.55rem 0.85rem;}"
    # Buttons: rounded, subtle border, emerald accent on hover/active
    "div.stButton > button {border-radius: 8px; border: 1px solid #1f2937;"
    "font-weight: 500; transition: all 0.15s ease;}"
    "div.stButton > button:hover {border-color: #10b981;"
    "background: rgba(16,185,129,0.08);}"
    # Sliders: tighten thumb visuals, emerald track
    "[data-testid='stSlider'] [role='slider'] {"
    "box-shadow: 0 0 0 3px rgba(16,185,129,0.18); border: 1.5px solid #10b981;}"
    # Radio + selectbox: rounded options, subtle hover
    "div[data-baseweb='radio'] label, div[data-baseweb='select'] {"
    "border-radius: 8px;}"
    # Images / Plotly: rounded corners and subtle border so the result
    # canvas reads as a single framed artifact, not a raw png stuck on
    # the page.
    "[data-testid='stImage'] img, [data-testid='stPlotlyChart'] {"
    "border-radius: 10px; border: 1px solid #1f2937;"
    "box-shadow: 0 2px 8px rgba(0,0,0,0.35);}"
    # Sidebar section breathing room
    "section[data-testid='stSidebar'] [data-testid='stMarkdown'] {"
    "margin-bottom: 0.25rem;}"
    "section[data-testid='stSidebar'] hr {"
    "margin: 0.85rem 0; border-color: #1f2937;}"
    # Code / inline `<code>` reads as monospace tech detail, not as
    # a default browser style. No !important on font-family -- some
    # callers wrap icons in <code> for keyboard hints, and forcing the
    # monospace face would print the icon glyph name as text.
    "code {background: rgba(16,185,129,0.08);"
    "color: #6ee7b7; padding: 0.05rem 0.32rem;"
    "border-radius: 4px; font-size: 0.86em;"
    "font-family: ui-monospace, 'JetBrains Mono', monospace;}"
    # Horizontal radios (Simple/Detailed resolution toggle) read as a
    # segmented control: pill-shaped frame with selected option filled
    # in emerald. Targets the inner role='radiogroup' so vertical
    # radios elsewhere on the page (Fast/Real-CFD mode) keep their
    # default stacked layout.
    "div[role='radiogroup'][aria-orientation='horizontal'] {"
    "display: inline-flex; padding: 3px; background: #111827;"
    "border: 1px solid #1f2937; border-radius: 10px; gap: 2px;}"
    "div[role='radiogroup'][aria-orientation='horizontal'] label {"
    "margin: 0 !important; padding: 0.42rem 0.85rem; border-radius: 7px;"
    "cursor: pointer; transition: all 0.15s ease;"
    "color: #94a3b8;}"
    "div[role='radiogroup'][aria-orientation='horizontal'] label:hover {"
    "color: #f5f5f5; background: rgba(255,255,255,0.03);}"
    "div[role='radiogroup'][aria-orientation='horizontal'] "
    "label:has(input:checked) {"
    "background: rgba(16,185,129,0.16); color: #6ee7b7;"
    "box-shadow: inset 0 0 0 1px rgba(16,185,129,0.4);}"
    # Hide the actual radio dot inside the segmented pill -- the fill
    # state already communicates selection clearly.
    "div[role='radiogroup'][aria-orientation='horizontal'] label > "
    "div:first-child {display: none;}"
    # Tighten slider label spacing and emerald-tint the track.
    "[data-testid='stSlider'] label p {font-size: 0.92rem; font-weight: 500;}"
    "[data-testid='stSlider'] [data-baseweb='slider'] > div > div > div {"
    "background: linear-gradient(90deg, #10b981 0%, #34d399 100%);}"
    # --- Result panel: give the viz more room, less chrome ---
    # Bordered st.container (the result frame around the GIF) gets a
    # subtle slate frame matching the rest of the palette; the global
    # stImage border is suppressed inside it so we don't double-frame.
    "[data-testid='stVerticalBlockBorderWrapper'] {"
    "border-color: #1f2937 !important; border-radius: 12px;"
    "background: rgba(10,10,10,0.45);"
    "padding-top: 0.5rem; padding-bottom: 0.5rem;}"
    "[data-testid='stVerticalBlockBorderWrapper'] "
    "[data-testid='stImage'] img {"
    "border: none; box-shadow: none; border-radius: 8px;}"
    # H3 headings: enough breathing room above so they don't crash into
    # the element above (the earlier 0.4rem caused visible overlap on
    # narrow viewports where the bordered container above wrapped).
    ".stMarkdown h3 {margin-top: 0.95rem !important;"
    "margin-bottom: 0.55rem !important; font-weight: 600;"
    "letter-spacing: -0.01em; line-height: 1.3;}"
    # Subtler horizontal rulers -- the default Streamlit hr is loud.
    "hr {border: none; border-top: 1px solid #1f2937;"
    "margin: 1rem 0;}"
    # Captions: a bit dimmer, less noisy in the result strip.
    "[data-testid='stCaptionContainer'] {color: #64748b !important;}"
    # --- Sidebar must always be accessible ---
    # We force the sidebar to render with display: flex (some browsers
    # honour display: none from Streamlit's slide-out animation past the
    # transition); we do NOT pin its width because that would fight the
    # collapsed-state translateX(-100%). The collapse + expand buttons
    # are already forced visible up top alongside #MainMenu hiding.
    "section[data-testid='stSidebar'] {"
    "visibility: visible !important;}"
    # --- Motion + depth pass (2026-05-26) ---
    # Quiet micro-interactions: page fade-in, hover lifts on buttons /
    # expanders / cards, a barely-there emerald aurora behind the hero,
    # subtle gradient pan on the validated badge. Goal is "alive but
    # never competes with the GIF" -- every animation is < 0.3s, easing
    # is ease-out (decelerates into rest), and there is no perpetual
    # motion in the main viewport.
    "@keyframes aerolab-fade-in {"
    "from {opacity: 0; transform: translateY(4px);}"
    "to {opacity: 1; transform: translateY(0);}}"
    "@keyframes aerolab-shimmer {"
    "0% {background-position: 0% 50%;}"
    "100% {background-position: 200% 50%;}}"
    "@keyframes aerolab-pulse-glow {"
    "0%, 100% {box-shadow: 0 0 0 0 rgba(16,185,129,0.0);}"
    "50% {box-shadow: 0 0 0 4px rgba(16,185,129,0.18);}}"
    # Page fade-in: every block stagger-loads in for ~0.5s. Stops on
    # second-rerun (animation only runs on element creation).
    "[data-testid='stAppViewContainer'] > .main > .block-container > "
    "div > [data-testid='stVerticalBlock'] > * {"
    "animation: aerolab-fade-in 0.45s ease-out backwards;}"
    # Ambient emerald aurora behind the page -- two soft radial blooms
    # at the top corners. Read as ambience, not as a visible gradient.
    "[data-testid='stAppViewContainer']::before {"
    "content: ''; position: fixed; inset: 0; pointer-events: none;"
    "background:"
    "radial-gradient(circle at 12% -5%, rgba(16,185,129,0.10), "
    "transparent 45%),"
    "radial-gradient(circle at 92% -8%, rgba(56,189,248,0.06), "
    "transparent 50%);"
    "z-index: 0;}"
    "[data-testid='stAppViewContainer'] > .main {position: relative;"
    "z-index: 1;}"
    # Button: lift + emerald glow on hover, soft press on active. The
    # active-state press uses translateY(1px) to give a real tactile feel.
    "div.stButton > button {transition: transform 0.18s ease-out, "
    "box-shadow 0.18s ease-out, border-color 0.18s ease-out, "
    "background 0.18s ease-out;}"
    "div.stButton > button:hover {transform: translateY(-1px);"
    "box-shadow: 0 6px 16px rgba(16,185,129,0.22),"
    "0 0 0 1px rgba(16,185,129,0.25);}"
    "div.stButton > button:active {transform: translateY(1px);"
    "box-shadow: 0 1px 4px rgba(0,0,0,0.4);}"
    # Primary button (Run simulation, download): get a gentle pulse-glow
    # at idle so the eye finds them. Pauses on hover so it doesn't
    # double-animate with the lift.
    "div.stButton > button[kind='primary'] {"
    "animation: aerolab-pulse-glow 3.2s ease-in-out infinite;}"
    "div.stButton > button[kind='primary']:hover {animation: none;}"
    # Expanders: cursor pointer + slight scale on hover; the summary row
    # tints emerald when hovered so the user knows it is clickable.
    "[data-testid='stExpander'] {"
    "transition: border-color 0.2s ease-out, background 0.2s ease-out,"
    "transform 0.2s ease-out, box-shadow 0.2s ease-out;}"
    "[data-testid='stExpander']:hover {transform: translateY(-1px);"
    "box-shadow: 0 4px 14px rgba(0,0,0,0.30),"
    "0 0 0 1px rgba(16,185,129,0.25);}"
    "[data-testid='stExpander'] summary {cursor: pointer;"
    "transition: color 0.18s ease-out, background 0.18s ease-out;}"
    "[data-testid='stExpander'] summary:hover {color: #6ee7b7;"
    "background: rgba(16,185,129,0.04);}"
    # Bordered st.container (the result frame): lifts on hover so the GIF
    # frame feels like a tangible artifact, not a static panel.
    "[data-testid='stVerticalBlockBorderWrapper'] {"
    "transition: border-color 0.25s ease-out, "
    "box-shadow 0.25s ease-out, transform 0.25s ease-out;}"
    "[data-testid='stVerticalBlockBorderWrapper']:hover {"
    "border-color: #334155 !important;"
    "box-shadow: 0 6px 24px rgba(0,0,0,0.4),"
    "0 0 0 1px rgba(16,185,129,0.18);}"
    # Slider thumb: scale up on hover so the thumb feels grabbable.
    "[data-testid='stSlider'] [role='slider'] {"
    "transition: transform 0.16s ease-out,"
    "box-shadow 0.16s ease-out;}"
    "[data-testid='stSlider'] [role='slider']:hover {"
    "transform: scale(1.15);"
    "box-shadow: 0 0 0 6px rgba(16,185,129,0.22);}"
    # Validated badge: subtle emerald gradient pan to draw the eye
    # without being a flashing-billboard.
    "a[href*='VALIDATION.md'] {"
    "background-size: 200% 200% !important;"
    "background-image: linear-gradient(110deg,"
    "rgba(16,185,129,0.10) 0%,"
    "rgba(16,185,129,0.18) 45%,"
    "rgba(16,185,129,0.10) 100%) !important;"
    "animation: aerolab-shimmer 6s ease-in-out infinite;"
    "transition: transform 0.18s ease-out, "
    "border-color 0.18s ease-out;}"
    "a[href*='VALIDATION.md']:hover {transform: translateY(-1px);"
    "border-color: rgba(16,185,129,0.7) !important;}"
    # Segmented-control selection: smooth slide instead of pop.
    "div[role='radiogroup'][aria-orientation='horizontal'] label {"
    "transition: all 0.2s ease-out;}"
    # Images / Plotly inside results: gentle glow ring on hover so the
    # viz canvas reads as the centerpiece.
    "[data-testid='stImage'] img {transition: box-shadow 0.25s ease-out;}"
    "[data-testid='stImage'] img:hover {"
    "box-shadow: 0 8px 32px rgba(16,185,129,0.18),"
    "0 4px 12px rgba(0,0,0,0.4);}"
    "</style>",
    unsafe_allow_html=True,
)

# Hero block. Wordmark + tagline + validated-Cd badge in a single
# horizontal flex row. The earlier (pre-2026-05-26) "subtle wordmark"
# left the top of the app reading like a Streamlit default; this hero
# anchors AeroLab as a product first, then defers to the mode-specific
# subtitle below.
st.markdown(
    "<div style='display:flex;align-items:flex-end;justify-content:space-between;"
    "gap:1rem;padding:0.5rem 0 0.9rem 0;border-bottom:1px solid #1f2937;"
    "margin-bottom:1rem;flex-wrap:wrap;'>"
    # Left: brand + tagline stacked vertically
    "<div style='display:flex;flex-direction:column;gap:0.15rem;'>"
    "<div style='display:flex;align-items:baseline;gap:0.6rem;'>"
    "<span style='font-size:1.85rem;font-weight:700;letter-spacing:-0.02em;"
    "color:#f5f5f5;line-height:1;'>AeroLab</span>"
    "<span style='font-size:0.72rem;color:#64748b;font-weight:500;"
    "letter-spacing:0.05em;'>v0.5.0</span>"
    "</div>"
    "<div style='color:#94a3b8;font-size:0.95rem;line-height:1.35;'>"
    "Watch air move around any shape &mdash; in your browser."
    "</div>"
    "</div>"
    # Right: validated badge + a discoverability chip for the 3D bench.
    # The 3D toggle itself lives in the sidebar (under "Solver tab");
    # this chip is the hybrid signal so first-time visitors notice 3D
    # exists without us promoting it to a top-level tab (which would
    # imply consumer-readiness it doesn't have yet).
    "<div style='display:flex;flex-direction:column;align-items:flex-end;"
    "gap:0.35rem;'>"
    "<a href='https://github.com/devansh2003-dev/AeroLab/blob/main/VALIDATION.md' "
    "target='_blank' style='display:inline-flex;align-items:center;gap:0.35rem;"
    "padding:0.32rem 0.7rem;background:rgba(16,185,129,0.10);"
    "border:1px solid rgba(16,185,129,0.45);border-radius:999px;"
    "color:#34d399;text-decoration:none;font-size:0.78rem;font-weight:500;"
    "letter-spacing:0.01em;' "
    "title='Blockage-corrected Cd against Williamson 1996 (cylinder, "
    "median 4 percent, max 12 percent) and Okajima 1982 (square, "
    "median 9 percent, max 22 percent) across Re 100-1000. The Standard "
    "preset runs at 35 percent blockage so the correction is large; click "
    "for the full honest methodology.'>"
    # U+2713 check mark; :material/X: shortcodes don't render inside
    # raw HTML markdown, so we use the unicode glyph.
    "<span style='font-size:0.85rem;line-height:1;'>&#10003;</span>"
    "<span>Validated against published Cd</span>"
    "</a>"
    # Discoverability chip for the 3D experience. Subtle, dimmer than
    # the validated badge -- 3D is preview-quality, not the headline
    # claim. The gallery view loads pre-baked fields (Cloud-safe); the
    # dev bench runs the kernel live (local-only).
    "<span style='display:inline-flex;align-items:center;gap:0.3rem;"
    "padding:0.22rem 0.6rem;background:rgba(100,116,139,0.10);"
    "border:1px solid #1f2937;border-radius:999px;"
    "color:#94a3b8;font-size:0.72rem;letter-spacing:0.01em;' "
    "title='Switch via the &quot;Solver tab&quot; radio in the sidebar. "
    "Gallery view replays pre-baked flow fields with smoke particles; "
    "the dev bench runs the kernel live and stays local-only.'>"
    "<span style='color:#64748b;'>3D gallery</span>"
    "<span style='opacity:0.6;'>&middot;</span>"
    "<span>preview, in sidebar &rarr;</span>"
    "</span>"
    "</div>"
    "</div>",
    unsafe_allow_html=True,
)

# --- 2D / 3D switcher (sidebar, very top) -----------------------------------
# AeroLab ships the validated 2D D2Q9 solver online. The 3D D3Q19 scaffold
# lives behind this toggle as a LOCAL-ONLY development bench -- D3Q19
# populations are ~150 MB at modest grids, well beyond Streamlit Cloud's
# 1 GB process cap. The kernel is BGK with full-way OR Bouzidi
# interpolated bounce-back, full-way OR Guo NEEM inflow/outflow (both
# toggleable in the bench), and TRT is available as the validation-
# track collision -- no MRT yet. The 2D tab is the default and runs
# the same code path as before this toggle existed.
with st.sidebar:
    st.markdown("### :material/grid_view: Solver tab")
    # key= is REQUIRED here. Without it, Streamlit auto-generates a key
    # from the widget's parameters, and that auto-key can become unstable
    # across reruns (e.g. when surrounding markdown/CSS reflows), causing
    # the selection to silently reset to index=0 on every rerun.
    # Reviewer 2026-05-26 confirmed clicking "3D (local, in development)"
    # did not switch the view -- an explicit key fixes the binding.
    # The "3D dev bench (local)" option was removed 2026-05-29 -- it
    # was a developer-facing live-kernel view with consumer-hostile
    # chrome (per-frame body re-render, auto-resizing axes that made
    # the box look like it was breathing, animation that updated body
    # position alongside the air). Code block remains downstream so we
    # can re-enable it later behind a debug flag if needed, but it is
    # NOT in the user-facing radio.
    view = st.radio(
        "Solver dimensionality",
        [
            "2D playground (validated)",
            "3D gallery (preview)",
        ],
        index=0,
        label_visibility="collapsed",
        key="solver_view",
        help=(
            "**2D**: the shipped D2Q9 LBM solver. Validated against "
            "Williamson 1996 (cylinder) and Okajima 1982 (square) to "
            "single-digit percent up to Re ~ 200. See VALIDATION.md.\n\n"
            "**3D gallery**: pre-baked field replay. The kernel ran "
            "offline; this view loads the saved velocity field and "
            "streams smoke particles through it. No live compute, "
            "Cloud-safe."
        ),
    )
    st.divider()

if view == "3D dev bench (local)":
    st.markdown("# AeroLab 3D &mdash; *local development bench*")
    st.markdown(
        "<div style='color:#94a3b8;font-size:0.92rem;line-height:1.5;'>"
        "This tab drives the <b>D3Q19 BGK</b> scaffold in "
        "<code>src/lbm_3d.py</code>. It is NOT the shipped playground "
        "&mdash; the 2D solver in <code>src/lbm.py</code> is. The 3D "
        "kernel runs a channel-flow smoke and prints diagnostics. As "
        "of Phase A2 it ships a sphere body with Bouzidi interpolated "
        "bounce-back (analytic q-field) and Guo NEEM inflow/outflow; no "
        "GIF rendering yet, no Cd validation, no MRT, no mesh upload. "
        "The point of this bench is to make every increment of 3D solver "
        "work visible while it is still developing locally."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")  # spacer

    with st.expander(":material/check_circle: &nbsp; **What works today**",
                     expanded=False):
        st.markdown(
            "- D3Q19 lattice constants verified (weights sum to 1, "
            "second-moment matches `cs² = 1/3`, OPPOSITE is an involution, "
            "all velocities unique).\n"
            "- BGK collision + push-streaming compiled by Numba; TRT "
            "(Λ = 3/16) production kernel available for the validation "
            "track (`src/lbm_3d_trt.py`).\n"
            "- Plane-channel boundaries: equilibrium inflow + zero-gradient "
            "outflow as the legacy path, **Guo non-equilibrium "
            "extrapolation** for both inflow and outflow as the production "
            "path (toggleable). Bounce-back at y=0/Ny-1, periodic in z.\n"
            "- Sphere body with full-way OR **Bouzidi interpolated "
            "bounce-back** using the analytic q-field (toggleable). "
            "TRT-aware Bouzidi correction is also wired up via "
            "`apply_bouzidi_correction_trt` for the validation track.\n"
            "- RK4 smoke-particle advection (`src/lbm_3d_smoke_particles.py`) "
            "with inflow-only seeding and solid-cell culling -- the "
            "Plotly Scatter3d viz below uses it.\n"
            "- Channel smoke produces a symmetric parabolic ux(y) profile "
            "(plane Poiseuille shape) with `mass_drift_rel < 1 %` over "
            "400 steps."
        )

    with st.expander(":material/build: &nbsp; **What is queued**",
                     expanded=False):
        st.markdown(
            "- TRT-collision channel driver -- a `run_channel_smoke`-shaped "
            "wrapper around `trt_periodic_step` plus the Guo NEEM and "
            "Bouzidi post-passes so the TRT validation track is "
            "end-to-end runnable (the kernels exist; the driver does not).\n"
            "- MRT collision (D3Q19 moment basis from d'Humières 2002) "
            "as an alternative to TRT for the production path.\n"
            "- 3D cylinder Cd validation against Williamson 1996 at "
            "Re = 100 to 200 (the headline 3D check).\n"
            "- Slice rendering as an animated GIF (mirror the 2D path).\n"
            "- A 3D version of `validate_solver.py` so the same gate "
            "test discipline applies."
        )

    st.divider()
    st.markdown("### Channel-flow smoke")
    st.markdown(
        "Pick a grid, hit **Run smoke**. Returns the centerline y-profile "
        "(should be parabolic between bounce-back walls), peak velocity, "
        "and mass drift. Grid sizes are kept small enough to finish in "
        "under a minute on a laptop."
    )

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        nx = st.select_slider("Nx (streamwise)", options=[32, 48, 64, 96],
                              value=64)
    with col_b:
        ny = st.select_slider("Ny (wall-normal)", options=[16, 24, 32],
                              value=24)
    with col_c:
        nz = st.select_slider("Nz (spanwise)", options=[16, 24, 32],
                              value=24)
    with col_d:
        n_steps = st.select_slider("Steps", options=[200, 400, 800, 1600],
                                   value=400)

    u_in = st.slider("Inflow u (lattice units, Ma ≲ 0.1)",
                     min_value=0.01, max_value=0.10, value=0.04, step=0.01)
    nu = st.slider("Kinematic viscosity ν (lattice units)",
                   min_value=0.005, max_value=0.10, value=0.02, step=0.005)
    re_est = u_in * ny / nu if nu > 0 else float("nan")
    st.caption(
        f"Re estimate ≈ u_in · Ny / ν = "
        f"{u_in:.2f} · {ny} / {nu:.3f} ≈ **{re_est:.1f}** (channel "
        f"Reynolds, low-Re smoke regime)."
    )

    # Cached voxelisation: a single STL upload + grid + extent combo
    # should not re-voxelise every time the user nudges another slider.
    # Cache key includes the raw bytes (so two STLs with identical
    # geometry but different metadata still hit the same entry), grid
    # extents, and body extent. Max 4 entries -- four different uploads
    # at a time is plenty for an interactive session, and each entry
    # is a ~1 MB bool array on the dev-grid sizes. (D-9 Phase 2,
    # 2026-05-28.)
    @st.cache_data(show_spinner=False, max_entries=4)
    def _cached_voxelise_stl(
        stl_bytes: bytes,
        Nx: int, Ny: int, Nz: int,
        body_extent_cells: float,
    ):
        """Returns ``(mask, links)``.

        ``mask`` is the (Nx, Ny, Nz) bool body mask. ``links`` is the
        Bouzidi-style WallLinkList (D-9 Phase 3, smoothed-mask q-field)
        that the kernel consumes when Bouzidi BB is selected. The pair
        is cached together because building the links is fast (~50 ms
        even on a 100 k-voxel mask) and the kernel may want either or
        both depending on the toggle.
        """
        from pathlib import Path as _Path
        from tempfile import NamedTemporaryFile

        from src.voxelize import voxel_mask_and_links_for_lbm
        # voxel_mask_and_links_for_lbm takes a path; the uploaded bytes
        # live in memory, so we land them in a temp file for the
        # duration of the call, then remove it. We don't use
        # NamedTemporaryFile's delete=True because Windows holds the
        # file open inside the `with` block, blocking the voxeliser
        # from reading it.
        with NamedTemporaryFile(suffix=".stl", delete=False) as tf:
            tf.write(stl_bytes)
            tf_path = tf.name
        try:
            return voxel_mask_and_links_for_lbm(
                tf_path, Nx, Ny, Nz,
                body_extent_cells=body_extent_cells,
                padding_cells=(max(8.0, Nx * 0.20), Ny * 0.10, Nz * 0.10),
                close_iters=1,
            )
        finally:
            _Path(tf_path).unlink(missing_ok=True)

    # Phase A2 (2026-05-26): body selector. The existing lbm_3d.py
    # kernel already takes a `body` bool mask and does full-way
    # bounce-back on solid cells. Wiring the sphere here is what
    # turns the channel-flow demo into a recognisable wind-tunnel
    # scene: smoke deflecting around an obstacle. D-9 Phase 2
    # (2026-05-28) adds the "Upload STL" path: the STL is voxelised
    # via src/voxelize.py and the resulting bool mask feeds into the
    # same kernel slot as the analytic sphere.
    body_choice = st.radio(
        "Body",
        ["None (channel only)", "Sphere", "Upload STL"],
        index=1,                   # default to sphere -- the dramatic demo
        horizontal=True,
        key="body_3d_choice",
        help=(
            "**None**: plain channel flow, baseline Poiseuille profile.\n\n"
            "**Sphere**: an analytic sphere placed ~30 % downstream of "
            "the inflow. Carries an exact q-field for Bouzidi BB.\n\n"
            "**Upload STL**: voxelise a user-supplied STL onto the grid "
            "via odd-parity ray casting. Halfway bounce-back at the "
            "voxel surface (Bouzidi q-field for arbitrary polygons is "
            "Phase 3 of the D-9 roadmap)."
        ),
    )
    use_sphere = body_choice == "Sphere"
    use_stl = body_choice == "Upload STL"
    # Radius scales with the wall-normal dimension so the sphere fits
    # comfortably regardless of grid choice. ~Ny/5 leaves ~2 Ny/5 cells
    # of clearance on each side -- well outside the bounce-back wall
    # boundary layer, where flow is reasonably uniform.
    sphere_R = float(ny) / 5.0
    sphere_cx = int(nx * 0.30)
    sphere_cy = ny // 2
    sphere_cz = nz // 2

    # STL upload state (populated only when use_stl). Defaults make the
    # downstream Run block work uniformly whether or not the upload
    # path is active. ``stl_links`` is the Bouzidi WallLinkList paired
    # with the mask (D-9 Phase 3, voxel_wall_links via smoothed-mask
    # interpolation); the kernel ignores it unless the BB toggle is
    # set to Bouzidi.
    stl_bytes: bytes | None = None
    stl_filename: str | None = None
    stl_extent_cells = 8.0
    stl_mask: np.ndarray | None = None
    stl_links = None
    stl_error: str | None = None

    if use_sphere:
        _blockage = 2 * sphere_R / ny
        st.caption(
            f"Sphere: R = {sphere_R:.1f} cells, centre = "
            f"({sphere_cx}, {sphere_cy}, {sphere_cz}). "
            f"Blockage (D / Ny) = {_blockage:.2f}. "
            f"This bench is correctness scaffolding, not a "
            f"validation result -- the dev grid is small so the "
            f"blockage is intentionally well above the < 0.10 the "
            f"`Validation3D` preset will use for Cd numbers."
        )

        # Phase A2-FULL Part 2 (2026-05-26): BC toggle. Bouzidi-linear
        # interpolated bounce-back tracks the analytic q for each wall
        # link (built once via src/lbm_3d_bouzidi.sphere_wall_links).
        # At q = 0.5 the formula reduces to full-way BB exactly --
        # pinned by tests/test_lbm_3d_bouzidi.py.
        bc_choice = st.radio(
            "Bounce-back scheme",
            ["Full-way (q = 0.5 everywhere)", "Bouzidi (analytic q)"],
            index=1,                       # default to Bouzidi -- the accurate path
            horizontal=True,
            key="bc_3d_choice",
            help=(
                "**Full-way**: every wall link bounces back as if the "
                "wall sits at the on-link midpoint (q = 0.5). Cheap, "
                "introduces a viscosity-dependent error in Cd.\n\n"
                "**Bouzidi**: linearly interpolates the bounce-back "
                "using the actual wall fraction q from the analytic "
                "sphere quadratic (D-4 in the memo). Combined with "
                "TRT (D-2) this places the wall at the mid-link "
                "position INDEPENDENT of viscosity -- the property "
                "that directly serves Cd accuracy."
            ),
        )
        use_bouzidi = bc_choice.startswith("Bouzidi")
    elif use_stl:
        uploaded = st.file_uploader(
            "STL file (binary or ASCII)",
            type=["stl"],
            key="stl_uploader_3d",
            help=(
                "Drop an STL (.stl). The mesh is centred in the channel, "
                "scaled so its longest axis spans the chosen extent, and "
                "voxelised onto the LBM grid via odd-parity ray casting. "
                "Closed manifolds work best; small holes get filled by "
                "one round of morphological closing."
            ),
        )
        _max_extent = max(4.0, float(min(ny, nz)) * 0.45)
        stl_extent_cells = st.slider(
            "Body extent (cells, longest axis)",
            min_value=4.0, max_value=_max_extent,
            value=min(8.0, _max_extent),
            step=0.5,
            key="stl_extent_3d",
            help=(
                "Longest-axis extent of the body after scaling onto the "
                "LBM grid. Bigger -> richer wake structure but higher "
                "blockage; keep < ~0.5 × Ny for clean dev-bench runs."
            ),
        )
        # D-9 Phase 3 (2026-05-28): Bouzidi toggle for STL bodies. The
        # voxel WallLinkList carries a per-link q estimated by
        # linear interpolation on a sigma=0.6 gaussian smoothing of
        # the mask; the 0.5 level set approximates the true surface
        # to sub-voxel accuracy. Halfway BB (q = 0.5 everywhere) is
        # the cheap baseline; flip on Bouzidi for sharper Cd numbers
        # on the uploaded mesh.
        stl_bc_choice = st.radio(
            "Bounce-back scheme",
            ["Full-way (q = 0.5 everywhere)",
             "Bouzidi (voxel-q from smoothed mask)"],
            index=1,                         # default to Bouzidi
            horizontal=True,
            key="stl_bc_3d_choice",
            help=(
                "**Full-way**: halfway BB at every solid cell. Cheap, "
                "but ~5 - 10 % Cd bias on smooth surfaces from the "
                "stair-stepped voxel wall position.\n\n"
                "**Bouzidi (voxel-q)**: per-link q estimated by "
                "linear interpolation on a smoothed (sigma=0.6) "
                "copy of the mask. Sub-voxel surface localisation -- "
                "sharper Cd on smooth bodies. Triangle-exact q is a "
                "future Phase 4."
            ),
        )
        use_bouzidi = stl_bc_choice.startswith("Bouzidi")

        if uploaded is not None:
            stl_bytes = uploaded.getvalue()
            stl_filename = uploaded.name
            try:
                stl_mask, stl_links = _cached_voxelise_stl(
                    stl_bytes,
                    int(nx), int(ny), int(nz),
                    float(stl_extent_cells),
                )
            except ValueError as exc:
                stl_error = str(exc)

            if stl_error is not None:
                st.error(f":material/error: {stl_error}")
            elif stl_mask is not None:
                _n_solid = int(stl_mask.sum())
                _solid_idx = np.argwhere(stl_mask)
                _bbox_extent = (
                    _solid_idx.max(axis=0) - _solid_idx.min(axis=0) + 1
                    if _n_solid > 0 else np.zeros(3, dtype=int)
                )
                _blockage_stl = float(_bbox_extent[1]) / ny if ny else 0.0
                _bc_label = "Bouzidi (voxel-q)" if use_bouzidi else "halfway BB"
                _n_links_str = (
                    f" &middot; {stl_links.n_links:,} wall links built"
                    if stl_links is not None else ""
                )
                st.caption(
                    f"STL: **{stl_filename}** "
                    f"({len(stl_bytes) / 1024:.1f} KB) &middot; "
                    f"voxelised to **{_n_solid:,} solid cells** &middot; "
                    f"body bbox {tuple(int(v) for v in _bbox_extent)} cells "
                    f"&middot; blockage (y-extent / Ny) = "
                    f"{_blockage_stl:.2f} &middot; BC: {_bc_label}"
                    f"{_n_links_str}."
                )
        else:
            st.caption(
                ":material/upload_file: Drop an STL above to voxelise it "
                "onto the LBM grid. No file = no body; the Run button "
                "below will refuse to launch."
            )
    else:
        use_bouzidi = False

    # 2026-05-26 Guo NEEM toggle. The legacy path uses equilibrium-write
    # inflow + zero-gradient-copy outflow inside the BGK kernel; Guo
    # non-equilibrium extrapolation (Guo, Zheng, Shi 2002) replaces both
    # with proper post-passes that copy the non-equilibrium part from
    # the neighbour interior. Visually this means the wake leaves the
    # domain cleanly instead of being smoothed flat by the zero-gradient
    # copy.
    inflow_choice = st.radio(
        "Inflow / outflow scheme",
        ["Equilibrium + zero-gradient (legacy)", "Guo NEEM (non-equilibrium)"],
        index=1,                                   # default to Guo NEEM
        horizontal=True,
        key="inflow_3d_choice",
        help=(
            "**Equilibrium + zero-gradient**: the inlet writes f_eq at "
            "the prescribed velocity, the outlet copies the second-to-last "
            "x-slice to the last one. Cheap, but the zero-gradient copy "
            "forces a uniform outlet velocity which damps the wake and "
            "introduces low-grade reflections back into the domain.\n\n"
            "**Guo NEEM**: post-passes that prescribe u_in at the inlet "
            "and rho = 1 at the outlet, while extrapolating the "
            "non-equilibrium part of the populations from the interior. "
            "The wake leaves the domain naturally with no spurious "
            "reflections; this is what the validation track will use."
        ),
    )
    use_guo_neem = inflow_choice.startswith("Guo")

    # 2026-05-26 Collision toggle. BGK is the default reference path
    # (single relaxation rate, classic Chapman-Enskog viscosity); TRT
    # adds the antisymmetric mode at the magic parameter Λ = 3/16, which
    # places the no-slip wall at the mid-link position INDEPENDENT of
    # viscosity -- the property that buys Cd accuracy in the validation
    # track. At s_plus = s_minus the two reduce to each other exactly
    # (pinned by tests/test_lbm_3d_trt.py::test_trt_channel_matches_bgk_at_unit_lambda).
    collision_choice = st.radio(
        "Collision operator",
        ["BGK (single relaxation rate)", "TRT (Λ = 3/16, magic parameter)"],
        index=0,                                   # default to BGK -- snappier baseline
        horizontal=True,
        key="collision_3d_choice",
        help=(
            "**BGK**: one relaxation rate omega = 1/tau set by the "
            "viscosity. The simplest collision operator; what every "
            "tutorial LBM ships. Stable up to ~Re 200 on this grid.\n\n"
            "**TRT (Λ = 3/16)**: splits each direction into symmetric "
            "and antisymmetric modes with rates s_plus and s_minus, "
            "constrained so (1/s_plus - 1/2)(1/s_minus - 1/2) = 3/16. "
            "This magic parameter places the wall at the on-link "
            "midpoint regardless of viscosity, eliminating the "
            "tau-dependent error in Cd. Production target for the "
            "Validation3D preset."
        ),
    )
    use_trt = collision_choice.startswith("TRT")

    # Refuse to launch when STL was selected but no file is loaded or
    # the upload failed to voxelise -- otherwise the user clicks Run
    # and either gets a confusing channel-only result or a hard error
    # mid-kernel. The button is shown but disabled with a help string
    # so the user knows WHY they cannot run.
    _run_disabled = use_stl and (stl_mask is None or stl_error is not None)
    if _run_disabled:
        _run_help = (
            stl_error
            if stl_error
            else "Upload an STL above before running."
        )
    else:
        _run_help = "Compile the kernel and stream particles through it."
    if st.button(":material/play_arrow: &nbsp; Run smoke",
                 width="stretch",
                 disabled=_run_disabled,
                 help=_run_help):
        import time as _time

        from src.lbm_3d import _make_sphere_mask, run_channel_smoke
        progress = st.progress(0.0, text="Compiling kernel + streaming...")
        if use_sphere:
            body_mask_3d = _make_sphere_mask(
                nx, ny, nz, sphere_cx, sphere_cy, sphere_cz, sphere_R,
            )
        elif use_stl and stl_mask is not None:
            # Cached voxel mask from src/voxelize.py. Halfway BB is
            # implicit at every solid cell; no wall_links since voxel
            # walls have no analytic q-field. (Phase 3 D-9 will wire
            # an approximate q-field via 1-cell ray casts to neighbour
            # solids.)
            body_mask_3d = stl_mask
        else:
            body_mask_3d = None
        wall_links_3d = None
        if use_sphere and use_bouzidi:
            from src.lbm_3d_bouzidi import sphere_wall_links
            wall_links_3d = sphere_wall_links(
                nx, ny, nz, sphere_cx, sphere_cy, sphere_cz, sphere_R,
            )
        elif use_stl and use_bouzidi and stl_links is not None:
            # D-9 Phase 3 voxel WallLinkList. Same dataclass as the
            # analytic sphere path, so the kernel consumes it through
            # the existing wall_links hook without further changes.
            wall_links_3d = stl_links
        try:
            def _cb(frac, text):
                progress.progress(frac, text=text)
            _t0 = _time.time()
            if use_trt:
                # TRT path: separate driver in src/lbm_3d_trt.py. Uses
                # apply_bouzidi_correction_trt when wall_links is set.
                from src.lbm_3d_trt import run_channel_smoke_trt
                rho, ux, uy, uz, diag = run_channel_smoke_trt(
                    Nx=nx, Ny=ny, Nz=nz, u_in=u_in, nu=nu, n_steps=n_steps,
                    body=body_mask_3d,
                    wall_links=wall_links_3d,
                    use_guo_neem=use_guo_neem,
                    progress_callback=_cb,
                )
            else:
                rho, ux, uy, uz, diag = run_channel_smoke(
                    Nx=nx, Ny=ny, Nz=nz, u_in=u_in, nu=nu, n_steps=n_steps,
                    body=body_mask_3d,
                    wall_links=wall_links_3d,
                    use_guo_neem=use_guo_neem,
                    progress_callback=_cb,
                )
            elapsed = _time.time() - _t0
        finally:
            progress.empty()

        diag_table = pd.DataFrame(
            [
                ("Grid",            f"{nx} × {ny} × {nz}"),
                ("Steps",           f"{n_steps}"),
                ("Wall time",       f"{elapsed:.2f} s "
                                    f"({n_steps/elapsed:.1f} steps/s)"),
                ("u_peak",          f"{diag['u_peak']:.5f}"),
                ("u_mean",          f"{diag['u_mean']:.5f}"),
                ("Centerline ratio (≈1.5 fully developed)",
                                    f"{diag['centerline_ratio']:.3f}"),
                ("Mass drift (rel)",
                                    f"{diag['mass_drift_rel']*100:+.3f} %"),
            ],
            columns=["Metric", "Value"],
        )
        st.dataframe(diag_table, hide_index=True, width="stretch")

        # Midplane slice (z = Nz/2) of ux. Plotly heatmap is fast and
        # works without matplotlib; the 2D playground already pulls
        # plotly so no new dep.
        mid_z = nz // 2
        slice_ux = ux[:, :, mid_z].T  # rows = y, cols = x
        fig = go.Figure(
            data=go.Heatmap(
                z=slice_ux,
                colorscale="Viridis",
                colorbar={"title": "u<sub>x</sub>"},
                zmin=0.0, zmax=float(diag["u_peak"]) * 1.05,
            )
        )
        fig.update_layout(
            title=f"Midplane slice ux(x, y) at z = {mid_z}",
            xaxis_title="x (streamwise)",
            yaxis_title="y (wall-normal)",
            height=380,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig, width="stretch")

        # Centerline y-profile vs the analytic plane-Poiseuille parabola.
        mid_x = nx // 2
        y_profile = ux[mid_x, :, mid_z]
        ys = np.arange(ny)
        # Fit a parabola u_fit = u_peak * (1 - ((y - y_c) / h)^2),
        # h = (Ny - 1) / 2, y_c = (Ny - 1) / 2.
        y_c = (ny - 1) / 2
        h = (ny - 1) / 2
        u_peak = float(diag["u_peak"])
        u_parabolic = u_peak * (1.0 - ((ys - y_c) / h) ** 2)
        prof_fig = go.Figure()
        prof_fig.add_scatter(x=ys, y=y_profile, mode="lines+markers",
                              name="measured")
        prof_fig.add_scatter(x=ys, y=u_parabolic, mode="lines",
                              name="analytic Poiseuille",
                              line=dict(dash="dash"))
        prof_fig.update_layout(
            title="Wall-normal profile ux(y) at midchannel x, mid z",
            xaxis_title="y",
            yaxis_title="u<sub>x</sub>",
            height=300,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(prof_fig, width="stretch")

        if abs(diag["mass_drift_rel"]) > 0.05:
            st.warning(
                f"Mass drifted {diag['mass_drift_rel']*100:+.2f} % &mdash; "
                "the streaming or boundary code is leaking mass. Investigate "
                "before reading the velocity profile as physical."
            )
        elif diag["centerline_ratio"] < 1.0 or diag["centerline_ratio"] > 1.7:
            st.info(
                f"Centerline ratio {diag['centerline_ratio']:.2f} is outside "
                "the fully-developed Poiseuille band [1.0, 1.7]. Either the "
                "flow has not converged yet (try more steps) or the "
                "boundary code has a subtle bias."
            )
        else:
            st.success(
                "Smoke clean: parabolic profile, mass conserved to "
                "within 5 %, centerline ratio in the Poiseuille band."
            )

        # === Phase A1 prototype: smoke-particle 3D advection ===
        # Locked by D-8 in 3D_PHASE0_DECISIONS.md. This is the first
        # concrete demonstration of the consumer-product viz path:
        # take the steady (ux, uy, uz) the LBM solver produced, advect
        # massless tracers through it via RK4 (`src/lbm_3d_smoke_particles.py`),
        # render the final cloud as Plotly Scatter3d so the user can
        # rotate / zoom. On a body-less channel the trajectories are
        # straight; the dramatic flow-around-shape visuals come once
        # Phase A2 wires in a sphere.
        from src.lbm_3d_smoke_particles import (
            seed_inflow_particles,
            step_smoke,
            trilerp_3d,
        )

        _sphere_suffix = ""
        if use_sphere:
            _sphere_suffix = (
                " with sphere (Bouzidi)" if use_bouzidi
                else " with sphere (full-way BB)"
            )
        else:
            _sphere_suffix = " (no body)"
        _sphere_suffix += " · Guo NEEM" if use_guo_neem else " · Eq inflow"
        _sphere_suffix += " · TRT" if use_trt else " · BGK"
        with st.expander(
            ":material/visibility: &nbsp; **Smoke particles** &mdash; "
            "Phase A2 viz" + _sphere_suffix,
            expanded=True,
        ):
            _bc_label = "Bouzidi interpolated" if use_bouzidi else "full-way"
            st.caption(
                "RK4 advection of ~150 massless tracers through the "
                "steady velocity field above. Seeded at the inflow "
                "(x=2) only -- no mid-domain spawn (the 2D streakline "
                "design rule carries over). Rotate / zoom the scene. "
                + (
                    f"Particles deflect around the sphere; the LBM solver "
                    f"computed the flow field with **{_bc_label}** bounce-back "
                    f"on the sphere surface, and the advector culls any "
                    f"tracer whose nearest cell is solid."
                    if use_sphere
                    else "On a body-less channel the streaks are straight; "
                    "switch the **Body** above to **Sphere** to see the "
                    "flow-around-obstacle demo."
                )
            )

            # Build the particle pool by repeatedly seeding + advecting.
            # The LBM solve produced a steady field; advection cost is
            # dominated by the trilerp inner loops (pure numpy,
            # ~few ms / frame at this particle count).
            #
            # Denser seeding rows than before (Ny / 3, Nz / 3 with a
            # floor of 5) -- ported from the gallery so the smoke
            # reads as a cloud rather than a constellation. Costs
            # ~50 % more advection work per frame; still well inside
            # the per-click budget at this grid size.
            n_y_rows = max(5, ny // 3)
            n_z_rows = max(5, nz // 3)
            y_rows_ad = np.linspace(2.0, ny - 3.0, n_y_rows)
            z_rows_ad = np.linspace(2.0, nz - 3.0, n_z_rows)

            rng_smoke = np.random.default_rng(0)
            px = np.empty(0, dtype=np.float64)
            py = np.empty(0, dtype=np.float64)
            pz = np.empty(0, dtype=np.float64)
            age = np.empty(0, dtype=np.int32)

            # dt MUST scale with u_in so particles actually move
            # visibly. step_smoke's dt is total advection time per
            # frame; at dt = 1.0 a particle at u_in = 0.04 advances
            # 0.04 cells / frame, so the channel takes 800 frames to
            # cross. We pick dt so the particle covers ~1 cell per
            # frame: target_cross_frames frames to traverse nx.
            target_cross_frames = max(40, nx)
            dt_per_frame = nx / float(target_cross_frames * max(u_in, 1e-6))
            max_age = int(1.6 * target_cross_frames)
            n_frames_ad = max(int(1.4 * target_cross_frames), 90)

            # Animation snapshot schedule (ported from the gallery,
            # 2026-05-28). We capture ~28 keyframes during advection
            # so the user can scrub / play through the development of
            # the wake instead of seeing only the steady-state cloud.
            # The first snapshot lands at target_cross_frames // 4
            # (the warmup), giving the smoke time to fill the inflow
            # band before frame 0 of the animation -- an animation
            # that opens on an empty channel is just dead air.
            n_keyframes = 28
            warmup_frames_ad = target_cross_frames // 4
            snapshot_stride = max(1, (n_frames_ad - warmup_frames_ad) // n_keyframes)

            # Pre-cast the velocity field to float32 ONCE outside the
            # loop; the previous code did this per-step which was a
            # small but real waste at the new denser seeding.
            ux_f32 = ux.astype(np.float32, copy=False)
            uy_f32 = uy.astype(np.float32, copy=False)
            uz_f32 = uz.astype(np.float32, copy=False)

            snapshots_ad: list[tuple] = []   # (px, py, pz, speeds_at_capture)
            for i in range(n_frames_ad):
                seed = seed_inflow_particles(
                    n_per_row=1,
                    y_rows=y_rows_ad,
                    z_rows=z_rows_ad,
                    x=2.0,
                    rng=rng_smoke,
                )
                px, py, pz, age = step_smoke(
                    px, py, pz, age,
                    ux_f32, uy_f32, uz_f32,
                    body_mask=body_mask_3d,
                    dt=dt_per_frame,
                    n_substeps=4,
                    max_age=max_age,
                    inflow_seed_xyz=seed,
                )
                if i >= warmup_frames_ad and (
                    (i - warmup_frames_ad) % snapshot_stride == 0
                    or i == n_frames_ad - 1
                ):
                    snap_speeds = trilerp_3d(ux_f32, px, py, pz)
                    snapshots_ad.append(
                        (px.copy(), py.copy(), pz.copy(), snap_speeds.copy())
                    )

            if len(px) == 0:
                st.info(
                    "Particles all exited the domain before the viz "
                    "captured them; try a lower inflow speed or more "
                    "steps."
                )
            else:
                # The last snapshot (post-warmup, fully-developed) is
                # what we show initially; the slider scrubs back into
                # the earlier transient states. The snapshot tuple
                # carries (px, py, pz, speeds) so the colour data is
                # already paired with the positions -- no per-render
                # trilerp_3d call needed here (the loop above already
                # paid that cost at capture time).
                px, py, pz, sp = snapshots_ad[-1]

                # Phase A3 visual: Q-criterion isosurface overlay (the
                # first reusable consumer-viz primitive). Compute Q from
                # the velocity field, marching-cubes an isosurface at
                # the user-selected fraction of Q_max, render as a
                # translucent cyan Plotly Mesh3d behind the particles.
                # Cheap (~5 ms on this grid) so we always compute Q
                # when the checkbox is on; no caching gymnastics.
                q_col1, q_col2 = st.columns([1, 3])
                with q_col1:
                    show_q = st.checkbox(
                        "Q-criterion vortex shell",
                        value=False,
                        key="show_q_3d",
                        help=(
                            "Render the Q = level isosurface around "
                            "vortex tubes in the wake. Q = (1/2)(|Ω|² - "
                            "|S|²); positive Q means rotation dominates "
                            "strain. The shell carves out where the "
                            "flow swirls -- visible at moderate Re even "
                            "before full vortex shedding."
                        ),
                    )
                with q_col2:
                    q_level_pct = st.slider(
                        "Q threshold (% of max)", 1, 50, 10,
                        key="q_level_pct_3d",
                        disabled=not show_q,
                        help="Lower → bigger shell (catches weaker swirls). "
                             "Higher → only the most intense vortex cores.",
                    )

                # Marker constants matched to the gallery (2026-05-28
                # backport): size 2.4 + opacity 0.78 reads as a smoke
                # cloud rather than discrete dots; cmax = 1.6 * u_in
                # gives a touch more headroom than the channel inflow
                # itself so the plasma ramp doesn't saturate near the
                # body where the flow accelerates.
                _marker_size = 2.4
                _marker_opacity = 0.78
                _marker_cmax = float(u_in) * 1.6
                scene_traces = [
                    go.Scatter3d(
                        x=px, y=py, z=pz,
                        mode="markers",
                        name="smoke",
                        marker=dict(
                            size=_marker_size,
                            color=sp,
                            colorscale="Plasma",
                            cmin=0.0,
                            cmax=_marker_cmax,
                            colorbar=dict(
                                title="u<sub>x</sub>",
                                thickness=12,
                                len=0.6,
                            ),
                            opacity=_marker_opacity,
                        ),
                    )
                ]
                if show_q:
                    from src.lbm_3d_qcriterion import (
                        compute_q_field,
                        extract_q_isosurface,
                    )
                    Q = compute_q_field(ux, uy, uz, body=body_mask_3d)
                    q_max = float(Q.max())
                    if q_max > 0.0:
                        level = (q_level_pct / 100.0) * q_max
                        iso = extract_q_isosurface(Q, level=level)
                        if iso is not None:
                            verts, faces = iso
                            scene_traces.append(
                                go.Mesh3d(
                                    x=verts[:, 0],
                                    y=verts[:, 1],
                                    z=verts[:, 2],
                                    i=faces[:, 0],
                                    j=faces[:, 1],
                                    k=faces[:, 2],
                                    color="#22d3ee",            # cyan-400
                                    opacity=0.35,
                                    name=f"Q = {level:.2e}",
                                    flatshading=False,
                                    hoverinfo="name",
                                )
                            )
                        else:
                            st.caption(
                                f":material/info: No Q ≥ {level:.2e} "
                                f"region found. Lower the threshold or "
                                f"crank `u_in` / lower `nu` to develop "
                                f"the wake."
                            )
                    else:
                        st.caption(
                            ":material/info: Q ≤ 0 everywhere -- the flow "
                            "has no vortex structure at this Re yet. "
                            "Raise `u_in` or lower `nu` and re-run."
                        )
                if use_sphere:
                    # Render the sphere as a translucent slate-grey Surface
                    # so the viewer can see particles deflecting AROUND it
                    # rather than through empty space. Parametric mesh:
                    # 32 x 32 grid in (theta, phi). Surface3d uses one
                    # colour ramp so we hold it at a single neutral tone
                    # via colorscale=[[0, c], [1, c]] -- gives a uniform
                    # slate-grey body that doesn't compete with the
                    # plasma-coloured particles for attention.
                    _theta = np.linspace(0.0, 2.0 * np.pi, 33)
                    _phi = np.linspace(0.0, np.pi, 17)
                    _T, _P = np.meshgrid(_theta, _phi)
                    sph_x = sphere_cx + sphere_R * np.sin(_P) * np.cos(_T)
                    sph_y = sphere_cy + sphere_R * np.sin(_P) * np.sin(_T)
                    sph_z = sphere_cz + sphere_R * np.cos(_P)
                    scene_traces.append(
                        go.Surface(
                            x=sph_x, y=sph_y, z=sph_z,
                            showscale=False,
                            colorscale=[[0, "#475569"], [1, "#475569"]],
                            opacity=0.55,
                            lighting=dict(
                                ambient=0.55, diffuse=0.7,
                                specular=0.25, roughness=0.5,
                            ),
                            lightposition=dict(x=100, y=200, z=0),
                            name="sphere",
                            hoverinfo="skip",
                        )
                    )
                elif use_stl and stl_mask is not None:
                    # Render the uploaded body via marching cubes on the
                    # bool mask at level 0.5 -- same primitive the
                    # Q-criterion overlay uses, just on a different
                    # scalar field. Same translucent slate-grey palette
                    # as the sphere so the user reads "body" not "data".
                    # (D-9 Phase 2, 2026-05-28.)
                    from skimage.measure import marching_cubes
                    try:
                        body_verts, body_faces, _, _ = marching_cubes(
                            stl_mask.astype(np.float32),
                            level=0.5,
                            spacing=(1.0, 1.0, 1.0),
                        )
                        scene_traces.append(
                            go.Mesh3d(
                                x=body_verts[:, 0],
                                y=body_verts[:, 1],
                                z=body_verts[:, 2],
                                i=body_faces[:, 0],
                                j=body_faces[:, 1],
                                k=body_faces[:, 2],
                                color="#475569",         # same slate as sphere
                                opacity=0.55,
                                flatshading=True,
                                name="body (voxel)",
                                hoverinfo="name",
                                lighting=dict(
                                    ambient=0.55, diffuse=0.7,
                                    specular=0.25, roughness=0.5,
                                ),
                                lightposition=dict(x=100, y=200, z=0),
                            )
                        )
                    except (ValueError, RuntimeError):
                        # Empty mask or single-voxel artefact -- shouldn't
                        # happen since the uploader already gated on
                        # n_solid > 0, but be defensive.
                        st.caption(
                            ":material/info: Could not extract a body "
                            "surface from the voxel mask -- the smoke "
                            "still renders but the body outline is "
                            "hidden."
                        )
                # Build per-snapshot Frames. The smoke Scatter3d is
                # trace 0 in ``scene_traces``; every frame overrides
                # just that trace via ``traces=[0]``, leaving sphere
                # + Q-criterion (static across time since the velocity
                # field is steady) untouched. Marker template carries
                # everything except ``color`` so each frame can swap
                # the per-particle speed array without re-specifying
                # the colorscale / colorbar / opacity. (Backported
                # from the gallery, 2026-05-28.)
                _smoke_marker_template = dict(
                    size=_marker_size,
                    colorscale="Plasma",
                    cmin=0.0,
                    cmax=_marker_cmax,
                    opacity=_marker_opacity,
                    colorbar=dict(title="u<sub>x</sub>", thickness=12, len=0.6),
                )
                frames_ad = []
                for k, (snap_px, snap_py, snap_pz, snap_sp) in enumerate(snapshots_ad):
                    frame_marker = dict(_smoke_marker_template)
                    frame_marker["color"] = snap_sp
                    frames_ad.append(
                        go.Frame(
                            data=[
                                go.Scatter3d(
                                    x=snap_px, y=snap_py, z=snap_pz,
                                    mode="markers",
                                    marker=frame_marker,
                                    name="smoke",
                                )
                            ],
                            name=str(k),
                            traces=[0],
                        )
                    )
                slider_steps = [
                    dict(
                        method="animate",
                        args=[[str(k)], dict(
                            mode="immediate",
                            frame=dict(duration=0, redraw=True),
                            transition=dict(duration=0),
                        )],
                        label=str(k),
                    )
                    for k in range(len(frames_ad))
                ]

                smoke_fig = go.Figure(data=scene_traces, frames=frames_ad)
                smoke_fig.update_layout(
                    scene=dict(
                        xaxis=dict(title="x (streamwise)", range=[0, nx]),
                        yaxis=dict(title="y (wall-normal)", range=[0, ny]),
                        zaxis=dict(title="z (spanwise)", range=[0, nz]),
                        aspectmode="data",
                        bgcolor="#0a0a0a",
                        camera=dict(
                            eye=dict(x=1.4, y=1.1, z=0.85),
                        ),
                    ),
                    height=540,
                    margin=dict(l=0, r=0, t=20, b=60),
                    paper_bgcolor="#0a0a0a",
                    updatemenus=[dict(
                        type="buttons",
                        direction="left",
                        x=0.02, y=-0.04, xanchor="left", yanchor="top",
                        pad=dict(t=6, r=6, b=6, l=6),
                        bgcolor="rgba(15,23,42,0.85)",
                        bordercolor="#334155",
                        font=dict(color="#e2e8f0", size=12),
                        buttons=[
                            dict(
                                label="▶  Play",
                                method="animate",
                                args=[None, dict(
                                    mode="immediate",
                                    frame=dict(duration=70, redraw=True),
                                    transition=dict(duration=0),
                                    fromcurrent=True,
                                    loop=True,
                                )],
                            ),
                            dict(
                                label="⏸  Pause",
                                method="animate",
                                args=[[None], dict(
                                    mode="immediate",
                                    frame=dict(duration=0, redraw=False),
                                    transition=dict(duration=0),
                                )],
                            ),
                        ],
                    )],
                    sliders=[dict(
                        active=len(frames_ad) - 1,
                        x=0.14, y=-0.04, len=0.82,
                        xanchor="left", yanchor="top",
                        pad=dict(t=6, b=0),
                        bgcolor="rgba(15,23,42,0.7)",
                        bordercolor="#334155",
                        activebgcolor="#22d3ee",
                        font=dict(color="#94a3b8", size=10),
                        currentvalue=dict(
                            visible=True, prefix="frame ",
                            font=dict(color="#e2e8f0", size=11),
                        ),
                        steps=slider_steps,
                    )],
                )
                st.plotly_chart(smoke_fig, width="stretch")
                st.caption(
                    f"{len(px):,} live particles in the final snapshot of "
                    f"a {n_frames_ad}-step RK4 advection (4 substeps / "
                    f"frame). {len(snapshots_ad)} animation frames -- click "
                    f"**▶ Play** to watch the wake develop, drag the slider "
                    f"to scrub. Tests: [test_lbm_3d_smoke_particles.py]"
                    f"(https://github.com/devansh2003-dev/aerolab/blob/"
                    f"main/tests/test_lbm_3d_smoke_particles.py) -- 15 "
                    f"analytic-field gates including uniform-flow drift "
                    f"and 3D Poiseuille centerline (D-8 reviewer "
                    f"requirement, 2026-05-26)."
                )

    st.divider()
    st.caption(
        "Source: [src/lbm_3d.py](https://github.com/devansh2003-dev/aerolab/"
        "blob/main/src/lbm_3d.py), "
        "[src/lbm_3d_smoke_particles.py](https://github.com/devansh2003-dev/"
        "aerolab/blob/main/src/lbm_3d_smoke_particles.py) &middot; "
        "Tests: [tests/test_lbm_3d_smoke.py](https://github.com/"
        "devansh2003-dev/aerolab/blob/main/tests/test_lbm_3d_smoke.py), "
        "[tests/test_lbm_3d_smoke_particles.py](https://github.com/"
        "devansh2003-dev/aerolab/blob/main/tests/test_lbm_3d_smoke_particles.py)"
    )
    st.stop()


# ---------------------------------------------------------------------------
# 3D Gallery: pre-baked field replay (consumer-mode, Cloud-safe)
#
# Loads a .npz produced by scripts/bake_3d_field.py and renders streamlines
# through the steady velocity field as a Plotly Scatter3d line trace, with
# the body (sphere / cylinder) and a wireframe chamber overlay.
#
# Why compute streamlines server-side (this module) instead of via
# go.Streamtube? Streamtube does the streamline integration AND geometry
# tessellation in JavaScript on the browser's main thread, which blocks
# the UI thread for ~3-8 s per scene swap on Cloud-tier CPUs and prevents
# Streamlit's progress bar from updating in real time. Tracing in Python
# with numba-jitted trilerp_3d takes ~50-100 ms total, lets the progress
# bar tick visibly, and ships only ~12 k line vertices to the browser
# (instead of ~60 k triangles) so the WebGL render is essentially free.
# ---------------------------------------------------------------------------


def _trace_streamlines(
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    body_mask: np.ndarray,
    seeds_x: np.ndarray,
    seeds_y: np.ndarray,
    seeds_z: np.ndarray,
    *,
    dt: float = 12.0,
    max_steps: int = 400,
    min_speed: float = 1e-5,
    progress_callback=None,
):
    """Trace forward streamlines (RK2 midpoint) through a steady 3D
    velocity field. Vectorised over all alive seeds per step.

    Returns flat NumPy arrays (x, y, z, speed) with NaN separators
    between seeds -- ready to plot as a single ``go.Scatter3d`` trace
    with ``mode="lines"`` and per-vertex ``line.color``.

    Streamlines terminate on body contact, out-of-domain, or stagnation
    (``|u| < min_speed``). RK2 is second-order accurate (twice the work
    of Euler, visibly smoother paths) -- sufficient for visualisation
    even though it's not what the underlying solver used.

    Parameters
    ----------
    ux, uy, uz : (Nx, Ny, Nz) ndarray
        Steady velocity field components in lattice units.
    body_mask : (Nx, Ny, Nz) bool ndarray
        True where the cell is solid.
    seeds_x, seeds_y, seeds_z : 1D ndarray
        Seed positions in lattice coordinates.
    dt : float, default 12.0
        Integration step. Tuned so a typical inflow displacement
        (u ~ 0.04) covers ~0.5 lattice cell per step.
    max_steps : int, default 400
        Hard cap on integration steps per seed.
    min_speed : float, default 1e-5
        Stagnation threshold.
    progress_callback : callable, optional
        Called periodically as ``cb(fraction)`` where fraction in [0, 1].
    """
    from src.lbm_3d_smoke_particles import trilerp_3d

    seeds_x = np.asarray(seeds_x, dtype=np.float64)
    seeds_y = np.asarray(seeds_y, dtype=np.float64)
    seeds_z = np.asarray(seeds_z, dtype=np.float64)

    Nx, Ny, Nz = ux.shape
    n_seeds = len(seeds_x)
    if n_seeds == 0:
        empty = np.array([], dtype=np.float32)
        return empty, empty, empty, empty

    # Path storage. (n_seeds, max_steps + 1). NaN at slot k means
    # "this seed had already died by step k". Plotly's Scatter3d
    # treats NaN as a line break, which is what we want at the path
    # end for each seed in the final concatenated trace.
    paths_x = np.full((n_seeds, max_steps + 1), np.nan, dtype=np.float32)
    paths_y = np.full((n_seeds, max_steps + 1), np.nan, dtype=np.float32)
    paths_z = np.full((n_seeds, max_steps + 1), np.nan, dtype=np.float32)
    paths_speed = np.full((n_seeds, max_steps + 1), np.nan, dtype=np.float32)
    paths_x[:, 0] = seeds_x.astype(np.float32)
    paths_y[:, 0] = seeds_y.astype(np.float32)
    paths_z[:, 0] = seeds_z.astype(np.float32)

    # Initial seed speed (so the colorbar gradient looks coherent from
    # the very first vertex of each line).
    seed_u = trilerp_3d(ux, seeds_x, seeds_y, seeds_z)
    seed_v = trilerp_3d(uy, seeds_x, seeds_y, seeds_z)
    seed_w = trilerp_3d(uz, seeds_x, seeds_y, seeds_z)
    paths_speed[:, 0] = np.sqrt(
        seed_u ** 2 + seed_v ** 2 + seed_w ** 2
    ).astype(np.float32)

    cur_x = seeds_x.copy()
    cur_y = seeds_y.copy()
    cur_z = seeds_z.copy()
    alive = np.ones(n_seeds, dtype=bool)

    # Tick the progress callback ~16 times across the loop -- coarser
    # than per-step (which would saturate the WebSocket) but fine
    # enough for a smooth-feeling bar.
    progress_period = max(1, max_steps // 16)

    for step in range(max_steps):
        if not alive.any():
            break

        ai = np.where(alive)[0]
        ax = cur_x[ai]
        ay = cur_y[ai]
        az = cur_z[ai]

        # RK2 midpoint. k1 sample at the current position, half-step
        # forward to the midpoint, k2 sample there, full step using k2.
        k1_u = trilerp_3d(ux, ax, ay, az)
        k1_v = trilerp_3d(uy, ax, ay, az)
        k1_w = trilerp_3d(uz, ax, ay, az)

        mid_x = np.clip(ax + 0.5 * dt * k1_u, 0.5, Nx - 1.5)
        mid_y = np.clip(ay + 0.5 * dt * k1_v, 0.5, Ny - 1.5)
        mid_z = np.clip(az + 0.5 * dt * k1_w, 0.5, Nz - 1.5)

        k2_u = trilerp_3d(ux, mid_x, mid_y, mid_z)
        k2_v = trilerp_3d(uy, mid_x, mid_y, mid_z)
        k2_w = trilerp_3d(uz, mid_x, mid_y, mid_z)

        new_x = ax + dt * k2_u
        new_y = ay + dt * k2_v
        new_z = az + dt * k2_w
        speed = np.sqrt(k2_u * k2_u + k2_v * k2_v + k2_w * k2_w)

        in_bounds = (
            (new_x > 0.5) & (new_x < Nx - 0.5)
            & (new_y > 0.5) & (new_y < Ny - 0.5)
            & (new_z > 0.5) & (new_z < Nz - 0.5)
        )
        ix = np.clip(new_x.astype(np.int32), 0, Nx - 1)
        iy = np.clip(new_y.astype(np.int32), 0, Ny - 1)
        iz = np.clip(new_z.astype(np.int32), 0, Nz - 1)
        not_in_body = ~body_mask[ix, iy, iz]
        moving = speed > min_speed
        survives = in_bounds & not_in_body & moving

        surviving_global = ai[survives]
        dying_global = ai[~survives]

        cur_x[surviving_global] = new_x[survives]
        cur_y[surviving_global] = new_y[survives]
        cur_z[surviving_global] = new_z[survives]
        paths_x[surviving_global, step + 1] = new_x[survives].astype(np.float32)
        paths_y[surviving_global, step + 1] = new_y[survives].astype(np.float32)
        paths_z[surviving_global, step + 1] = new_z[survives].astype(np.float32)
        paths_speed[surviving_global, step + 1] = speed[survives].astype(np.float32)
        alive[dying_global] = False

        if progress_callback is not None and (
            step % progress_period == 0 or step == max_steps - 1
        ):
            progress_callback((step + 1) / max_steps)

    if progress_callback is not None:
        progress_callback(1.0)

    # Flatten with NaN separators. One Plotly trace, multiple
    # streamlines, NaN breaks the line between them.
    out_x, out_y, out_z, out_speed = [], [], [], []
    nan_sep = np.array([np.nan], dtype=np.float32)
    for s in range(n_seeds):
        valid_count = int(np.isfinite(paths_x[s]).sum())
        if valid_count < 2:
            continue
        out_x.append(paths_x[s, :valid_count])
        out_x.append(nan_sep)
        out_y.append(paths_y[s, :valid_count])
        out_y.append(nan_sep)
        out_z.append(paths_z[s, :valid_count])
        out_z.append(nan_sep)
        out_speed.append(paths_speed[s, :valid_count])
        out_speed.append(nan_sep)

    if not out_x:
        empty = np.array([], dtype=np.float32)
        return empty, empty, empty, empty

    return (
        np.concatenate(out_x),
        np.concatenate(out_y),
        np.concatenate(out_z),
        np.concatenate(out_speed),
    )


if view == "3D gallery (preview)":
    from pathlib import Path

    from src.baked_fields import list_baked_fields, load_baked_field

    baked_dir = Path("data/baked")
    available = list_baked_fields(baked_dir)

    if not available:
        # Empty-state is a Cloud-friendly fallback. The .npz scenes are
        # committed to the repo (data/baked/*.npz with a .gitignore
        # exception), so this branch only fires if a deploy missed those
        # files -- which is a transient build issue, not user-actionable.
        # Telling a Cloud visitor to "run python scripts/..." is wrong;
        # they have no shell. Just ask them to retry.
        st.markdown("# AeroLab 3D")
        st.info(
            ":material/hourglass_empty: Scenes are loading. "
            "Try again in a moment."
        )
        st.stop()

    # All 3D controls live in the sidebar -- mirrors the 2D playground.
    # The main page below is reserved for the visualisation; no widgets,
    # no metric strips, no engineering chrome above the fold.
    preset_options = {p.stem: p for p in available}
    with st.sidebar:
        st.markdown("### :material/auto_awesome: &nbsp; 3D scene")
        chosen_name = st.selectbox(
            "Scene",
            options=list(preset_options.keys()),
            index=0,
            key="gallery_preset_choice",
            label_visibility="collapsed",
            help=(
                "Pre-baked steady velocity fields. More scenes appear "
                "here automatically -- bake them with "
                "`scripts/bake_3d_field.py`."
            ),
        )
        st.divider()
        st.markdown("### :material/air: &nbsp; Streamlines")
        n_seeds = st.slider(
            "Density",
            min_value=12, max_value=50, value=24,
            key="n_seeds_gallery",
            help=(
                "Number of streamlines released at the inflow plane. "
                "Higher = denser smoke field but slower in-browser "
                "render. Above ~40 the WebGL frame rate may drop."
            ),
        )
        line_width = st.slider(
            "Line width",
            min_value=1, max_value=8, value=3,
            key="line_width_gallery",
            help=(
                "Pixel width of each streamline. WebGL line "
                "rendering is essentially free at any width -- this "
                "is purely a visual preference."
            ),
        )
        st.divider()
        st.markdown("### :material/visibility: &nbsp; Overlays")
        show_sphere = st.checkbox(
            "Body", value=True, key="show_body_gallery",
            help=(
                "Render the solid obstacle (sphere or cylinder, "
                "depending on the scene)."
            ),
        )
        show_box = st.checkbox(
            "Wind-tunnel chamber outline", value=True,
            key="show_box_gallery",
        )
        show_q = st.checkbox(
            "Q-criterion vortex shell", value=False,
            key="show_q_gallery",
            help=(
                "Q = (1/2)(|Ω|² - |S|²); positive Q means rotation "
                "dominates strain. The shell carves out where the "
                "flow swirls."
            ),
        )
        q_level_pct = st.slider(
            "Q threshold (% of max)",
            min_value=1, max_value=50, value=10,
            key="q_level_pct_gallery",
            disabled=not show_q,
        )

    # Progress bar: gives the user visible feedback during scene swap.
    # Each stage is fast in Python (<1 s each) but the browser-side
    # WebGL render after st.plotly_chart can take 2-5 s on a fresh
    # scene; that part we can't progress-bar (it's client-side). The
    # explicit ticks here at least signal "something is happening"
    # during the Python-side prep instead of a frozen-looking page.
    _gallery_prog = st.progress(0, text="Loading scene...")
    _gallery_prog.progress(10, text="Loading velocity field...")
    field = load_baked_field(preset_options[chosen_name])
    u_in_meta = float(field.meta.get("u_in", 0.05))
    Nx, Ny, Nz = field.Nx, field.Ny, field.Nz

    _gallery_prog.progress(25, text="Computing colormap range...")
    # Per-scene colormap normalisation. The previous cmax = u_in * 1.6
    # under-saturated the colormap on higher-Re scenes where peak
    # speed past the body is closer to 2.5 - 3 x u_in -- the Plasma
    # colorbar saturated yellow early and lost the speed gradient.
    # Use the 99.5th percentile of |u| over the fluid (robust to corner
    # artifacts from boundary conditions); floor at u_in * 1.2 so slow
    # scenes still get a visible gradient.
    _fluid_mask_flat = ~field.body
    if _fluid_mask_flat.any():
        _fluid_umag = np.sqrt(
            field.ux ** 2 + field.uy ** 2 + field.uz ** 2
        )[_fluid_mask_flat]
        _field_speed_99 = float(np.percentile(_fluid_umag, 99.5))
    else:
        _field_speed_99 = u_in_meta * 1.5
    cmax_speed = max(1.1 * _field_speed_99, u_in_meta * 1.2)

    # --- Main page: title + one-line caption + the visualisation. -----
    st.markdown(
        f"# AeroLab 3D &nbsp;<span style='color:#64748b;font-weight:400;"
        f"font-size:0.55em;letter-spacing:0.04em;text-transform:uppercase;'>"
        f"&middot; {chosen_name}</span>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Streamlines trace how the air wraps around the body. "
        "Colour shows speed (Plasma colormap: purple slow, yellow fast). "
        "Drag to rotate, scroll to zoom."
    )

    # --- Seed positions on the inflow plane ---------------------------
    # Regular (y, z) grid at x = 2 cells from the inlet. The density
    # slider sets the total seed count; we split it across y and z
    # proportional to the cross-section aspect so the grid isn't
    # squashed on non-square sections.
    _gallery_prog.progress(40, text="Building seed positions...")
    aspect = max(Nz / max(Ny, 1), 1e-3)
    n_y_seeds = max(2, int(round((n_seeds / aspect) ** 0.5)))
    n_z_seeds = max(2, int(round(n_seeds / n_y_seeds)))
    sy = np.linspace(3.0, float(Ny) - 4.0, n_y_seeds)
    sz = np.linspace(3.0, float(Nz) - 4.0, n_z_seeds)
    _SY, _SZ = np.meshgrid(sy, sz)
    seeds_x = np.full(_SY.size, 2.0, dtype=np.float64)
    seeds_y = _SY.flatten().astype(np.float64)
    seeds_z = _SZ.flatten().astype(np.float64)

    # --- Server-side streamline tracing -------------------------------
    # See _trace_streamlines docstring above for the rationale: the
    # alternative go.Streamtube does this integration in JavaScript on
    # the browser's main thread, freezing the UI on scene swap. Doing
    # it server-side with numba-jitted trilerp_3d gives us real
    # progress updates AND lets the browser just render lines.
    def _trace_prog(frac):
        pct = int(50 + 40 * frac)
        _gallery_prog.progress(
            pct,
            text=f"Tracing streamlines ({int(frac * 100)}%)...",
        )

    flat_x, flat_y, flat_z, flat_speed = _trace_streamlines(
        field.ux, field.uy, field.uz, field.body,
        seeds_x, seeds_y, seeds_z,
        progress_callback=_trace_prog,
    )

    _gallery_prog.progress(92, text="Composing scene...")
    scene_traces = [
        go.Scatter3d(
            x=flat_x, y=flat_y, z=flat_z,
            mode="lines",
            line=dict(
                color=flat_speed,
                colorscale="Plasma",
                cmin=0.0, cmax=cmax_speed,
                width=float(line_width),
                showscale=True,
                colorbar=dict(
                    title="speed",
                    thickness=10, len=0.45,
                    x=1.02, xanchor="left",
                ),
            ),
            name="streamlines",
            hoverinfo="skip",
            showlegend=False,
        )
    ]

    # Q-criterion shell (optional, controlled from sidebar).
    if show_q:
        from src.lbm_3d_qcriterion import compute_q_field, extract_q_isosurface
        Q = compute_q_field(field.ux, field.uy, field.uz, body=field.body)
        q_max_f = float(Q.max())
        if q_max_f > 0.0:
            level = (q_level_pct / 100.0) * q_max_f
            iso = extract_q_isosurface(Q, level=level)
            if iso is not None:
                verts, faces = iso
                scene_traces.append(
                    go.Mesh3d(
                        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                        color="#22d3ee", opacity=0.32,
                        flatshading=False,
                        name="Q shell",
                        hoverinfo="name",
                    )
                )

    # Body overlay (controlled from sidebar). Both sphere and
    # spanwise-cylinder presets render as studio-lit gradient surfaces
    # with fresnel rim for a solid-object look. Shared lighting +
    # colorscale block defined once below so the two surfaces look
    # like the same material under the same lights.
    _body_colorscale = [
        [0, "#64748b"],
        [0.5, "#cbd5e1"],
        [1, "#f1f5f9"],
    ]
    _body_lighting = dict(
        ambient=0.42, diffuse=0.85,
        specular=0.55, roughness=0.28,
        fresnel=0.45,
    )
    _body_lightpos = dict(x=2000, y=2500, z=1500)
    body_type = str(field.meta.get("body_type", ""))
    if show_sphere and body_type == "sphere":
        bp = field.meta.get("body_params", {})
        try:
            sphere_cx = float(bp["cx"])
            sphere_cy = float(bp["cy"])
            sphere_cz = float(bp["cz"])
            sphere_R = float(bp["R"])
            _theta = np.linspace(0.0, 2.0 * np.pi, 33)
            _phi = np.linspace(0.0, np.pi, 17)
            _T, _P = np.meshgrid(_theta, _phi)
            sph_x = sphere_cx + sphere_R * np.sin(_P) * np.cos(_T)
            sph_y = sphere_cy + sphere_R * np.sin(_P) * np.sin(_T)
            sph_z = sphere_cz + sphere_R * np.cos(_P)
            scene_traces.append(
                go.Surface(
                    x=sph_x, y=sph_y, z=sph_z,
                    showscale=False,
                    colorscale=_body_colorscale,
                    opacity=1.0,
                    lighting=_body_lighting,
                    lightposition=_body_lightpos,
                    name="sphere", hoverinfo="skip",
                )
            )
        except (KeyError, ValueError, TypeError):
            pass
    elif show_sphere and body_type == "cylinder":
        bp = field.meta.get("body_params", {})
        try:
            cyl_cx = float(bp["cx"])
            cyl_cy = float(bp["cy"])
            cyl_R = float(bp["R"])
            # Curved side surface: parametric (theta, z) grid.
            _theta = np.linspace(0.0, 2.0 * np.pi, 49)
            _zs = np.linspace(0.0, float(Nz), 21)
            _T, _ZG = np.meshgrid(_theta, _zs)
            cyl_x = cyl_cx + cyl_R * np.cos(_T)
            cyl_y = cyl_cy + cyl_R * np.sin(_T)
            cyl_z = _ZG
            scene_traces.append(
                go.Surface(
                    x=cyl_x, y=cyl_y, z=cyl_z,
                    showscale=False,
                    colorscale=_body_colorscale,
                    opacity=1.0,
                    lighting=_body_lighting,
                    lightposition=_body_lightpos,
                    name="cylinder", hoverinfo="skip",
                )
            )
            # End caps: filled disks at z=0 and z=Nz. Parametrise as
            # polar (r, theta) grids -- r from 0 to R, theta full
            # revolution. The disk fills the cylinder's open end so
            # the body reads as a closed solid object instead of a
            # tube you can see through.
            _r = np.linspace(0.0, cyl_R, 9)
            _R_grid, _T_grid = np.meshgrid(_r, _theta)
            cap_x = cyl_cx + _R_grid * np.cos(_T_grid)
            cap_y = cyl_cy + _R_grid * np.sin(_T_grid)
            for cap_z_val, cap_name in (
                (0.0, "cap_bottom"), (float(Nz), "cap_top"),
            ):
                cap_z = np.full_like(cap_x, cap_z_val)
                scene_traces.append(
                    go.Surface(
                        x=cap_x, y=cap_y, z=cap_z,
                        showscale=False,
                        colorscale=_body_colorscale,
                        opacity=1.0,
                        lighting=_body_lighting,
                        lightposition=_body_lightpos,
                        name=cap_name, hoverinfo="skip",
                    )
                )
        except (KeyError, ValueError, TypeError):
            pass

    # Wireframe box: 12 edges of the simulation domain as a single
    # Scatter3d in lines mode (None-separated segments).
    if show_box:
        _bx1, _by1, _bz1 = float(Nx), float(Ny), float(Nz)
        _box_pts = [
            (0.0, 0.0, 0.0), (_bx1, 0.0, 0.0), (_bx1, _by1, 0.0), (0.0, _by1, 0.0), (0.0, 0.0, 0.0),
            (None, None, None),
            (0.0, 0.0, _bz1), (_bx1, 0.0, _bz1), (_bx1, _by1, _bz1), (0.0, _by1, _bz1), (0.0, 0.0, _bz1),
            (None, None, None),
            (0.0, 0.0, 0.0), (0.0, 0.0, _bz1),
            (None, None, None),
            (_bx1, 0.0, 0.0), (_bx1, 0.0, _bz1),
            (None, None, None),
            (_bx1, _by1, 0.0), (_bx1, _by1, _bz1),
            (None, None, None),
            (0.0, _by1, 0.0), (0.0, _by1, _bz1),
        ]
        scene_traces.append(
            go.Scatter3d(
                x=[p[0] for p in _box_pts],
                y=[p[1] for p in _box_pts],
                z=[p[2] for p in _box_pts],
                mode="lines",
                line=dict(color="#475569", width=2),
                opacity=0.55, name="tunnel",
                hoverinfo="skip", showlegend=False,
            )
        )

    # Animation REMOVED (2026-05-29). The animated tracer overlay was
    # causing the page to hang on scene-dropdown change -- either the
    # particle advection JIT compile blocked the thread on Cloud, or
    # Plotly's per-frame Scatter3d data (28 frames x 3600 markers)
    # overwhelmed the browser's WebGL pipeline on a fresh scene load.
    # The static streamtubes already deliver the "streamlines wrapping
    # around the body" visual the user asked for -- their structural
    # form communicates flow direction without temporal animation.
    # Will re-introduce motion in a follow-up using a lighter-weight
    # approach (e.g. animated seed phase-shift on the streamtubes, or
    # a much smaller tracer count) once the static version is confirmed
    # stable on Cloud.

    fig = go.Figure(data=scene_traces)
    fig.update_layout(
        scene=dict(
            # Hide axis ticks/labels entirely -- the wireframe box
            # already provides the chamber outline, and the axis lines
            # were competing visually with the streamlines / body.
            xaxis=dict(visible=False, range=[-2.0, Nx + 2.0]),
            yaxis=dict(visible=False, range=[-2.0, Ny + 2.0]),
            zaxis=dict(visible=False, range=[-2.0, Nz + 2.0]),
            aspectmode="data",
            bgcolor="#0a0a0a",
            # Camera tightened from (1.5, 1.15, 0.9) so the body fills
            # more of the viewport on first load (user note: spheres
            # were looking lost inside the long chamber).
            camera=dict(eye=dict(x=1.15, y=0.9, z=0.65)),
        ),
        height=640,
        margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor="#0a0a0a",
        showlegend=False,
    )
    _gallery_prog.progress(
        100, text="Ready -- rendering in your browser."
    )
    _gallery_prog.empty()
    st.plotly_chart(fig, width="stretch")

    # Engineering details: collapsed, on the main page, below the
    # chart. Read-only display of metadata; all interactive controls
    # already live in the sidebar.
    _Re_est = (
        u_in_meta * 2.0 * float(field.meta.get("body_params", {}).get("R", 0.0))
        / max(float(field.meta.get("nu", 1.0)), 1e-9)
    )
    _solver = field.meta.get("solver_diag", {})
    _scheme = str(field.meta.get("scheme", "?")).upper()
    _outflow = str(field.meta.get("outflow_scheme", "guo"))
    _n_steps = int(field.meta.get("n_steps", 0))
    with st.expander(
        ":material/tune: &nbsp; **Engineering details**",
        expanded=False,
    ):
        st.caption(
            f"*Solver: {_scheme} collision, "
            f"{'Bouzidi' if field.meta.get('use_bouzidi') else 'half-way BB'} "
            f"wall BC, {_outflow} outflow, {_n_steps} steps.*"
        )
        diag_cols = st.columns(4)
        diag_cols[0].metric("Grid", f"{Nx} × {Ny} × {Nz}")
        diag_cols[1].metric(
            "Re (approx)", f"{_Re_est:.0f}",
            help="u_in · D / nu, using sphere diameter D = 2R from "
                 "the preset's body params.",
        )
        diag_cols[2].metric(
            "Mass drift", f"{_solver.get('mass_drift_rel', 0):.2%}",
            help="Mass change between the simulation's first and last step.",
        )
        diag_cols[3].metric(
            "Peak u_x", f"{_solver.get('u_peak', 0):.4f}",
            help="Maximum streamwise velocity (lattice units).",
        )
        st.markdown("##### Manifest")
        st.caption(
            f"Preset hash: `{field.meta.get('hash', '?')[:16]}...` "
            f"&middot; Schema v{field.meta.get('version', '?')} "
            f"&middot; Baked at {field.meta.get('ts_baked', '?')}"
        )
        st.json(field.meta, expanded=False)
        st.caption(
            "Source: [src/baked_fields.py](https://github.com/devansh2003-dev/"
            "aerolab/blob/main/src/baked_fields.py), "
            "[scripts/bake_3d_field.py](https://github.com/devansh2003-dev/"
            "aerolab/blob/main/scripts/bake_3d_field.py)"
        )
    st.stop()

# --- 2D validity thresholds (single source of truth) ---
# Cited in VALIDATION.md headline and surfaced in the UI as the Re-banner
# (above the velocity slider) and the inline pill (in the slider caption).
# These are the Williamson 1996 / ARFM 28 mode-A 3D-instability threshold
# (Re ~ 190 for the cylinder; we round to 200) and the rough envelope where
# the 2D approximation can still produce visually plausible wakes without
# claiming engineering accuracy. Above RE_UNPHYSICAL_2D the solver still
# converges, but the missing 3D modes mean the numbers do not represent
# any real flow.
RE_VALIDATED_MAX = 200
RE_EXPLORATORY_MAX = 800
RE_UNPHYSICAL_2D = RE_EXPLORATORY_MAX  # alias for clarity at call sites

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
        "<b style='color:#cbd5e1;'>Fast (ML surrogate)</b> &mdash; airfoil "
        "lift/drag numbers in &lt;1 s. Drag a slider, get a polar.<br>"
        "<b style='color:#cbd5e1;'>CFD (LBM solver)</b> &mdash; watch the air "
        "actually move around a shape. ~2.5 min on Cloud, ~30 s locally."
        "</div>",
        unsafe_allow_html=True,
    )
    # Internal option values are kept verbatim ("Fast (NeuralFoil)" /
    # "Real CFD (LBM)") so the `if mode == "Real CFD (LBM)"` branch below
    # and any session-state keys persist. Only the user-visible labels
    # change, via format_func: "(NeuralFoil)" -> "(ML surrogate)" so a
    # first-time visitor sees it is a NEURAL NETWORK, not a CFD speedup;
    # "Real CFD (LBM)" -> "CFD (LBM solver)" to drop the "Real" framing
    # that implied the other mode was not real, and put the algorithm
    # family up front. (Card #3 from the 2026-05-27 reviewer round.)
    _MODE_LABELS = {
        "Fast (NeuralFoil)": "Fast (ML surrogate)",
        "Real CFD (LBM)": "CFD (LBM solver)",
        "Validation": "Validation (benchmarks)",
    }
    mode = st.radio(
        "Simulation mode",
        ["Fast (NeuralFoil)", "Real CFD (LBM)", "Validation"],
        # Default to Real CFD: the user-facing "see air move" feature is
        # what makes AeroLab visually distinctive, and the curated gallery
        # cards give first-time visitors something compelling to click
        # without needing to know what NeuralFoil is.
        index=1,
        format_func=lambda v: _MODE_LABELS[v],
        label_visibility="collapsed",
        help=(
            "**Fast (ML surrogate)**: NeuralFoil neural-network prediction "
            "of lift / drag for NACA airfoils. Trained on XFoil / RANS data; "
            "not a live simulation.\n\n"
            "**CFD (LBM solver)**: full 2D Lattice Boltzmann simulation -- "
            "watch the air actually move.\n\n"
            "**Validation (benchmarks)**: read-only benchmark tables and "
            "bar charts vs Williamson 1996 / Okajima 1982. No solver runs "
            "here -- the data is committed in `data/validation/`."
        ),
    )
    st.divider()

# --- Cached LBM simulate+render wrapper (module level) ---
# Bluff-body reference data + blockage-correction helpers live in
# src/references.py (imported at the top of this file). The leading
# underscores on the aliases above match the convention the rest of
# this script uses for "in-app private" lookups.


def _cylinder_reference(re_value: int):
    """Backward-compat alias kept for tests / older importers."""
    return _textbook_reference("Cylinder", re_value)


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


# --- Validation mode: read-only benchmark tables + bar charts ---
# Card #8 from the 2026-05-27 review. Loads the committed
# `data/validation/results*.json` files and renders them with no solver
# work -- the math behind the numbers lives in VALIDATION.md.
if mode == "Validation":
    import json as _json_validation
    from pathlib import Path as _Path_validation

    st.title("Validation benchmarks")
    st.caption(
        ":material/menu_book: Read-only view of "
        "[`data/validation/results*.json`](https://github.com/devansh2003-dev/"
        "aerolab/tree/main/data/validation). The math, methodology, and "
        "10+ academic citations live in "
        "[VALIDATION.md](https://github.com/devansh2003-dev/"
        "aerolab/blob/main/VALIDATION.md)."
    )

    # The Resolved preset (D = 40, B = 10 %) is the headline data. We
    # lead with it so a first-time visitor sees the validated numbers
    # before scrolling into the cross-check tables.
    _v_root = _Path_validation(__file__).resolve().parent / "data" / "validation"
    _v_files = [
        ("Resolved preset (D = 40, B = 10 %) -- headline",
         _v_root / "results_resolved.json", True),
        ("Validation preset (D = 20, B = 5 %) -- cross-check",
         _v_root / "results_lowblockage.json", False),
        ("Standard preset (D = 28, B = 35 %) -- interactive UI / CI gate",
         _v_root / "results.json", False),
    ]

    for _title, _path, _is_headline in _v_files:
        if not _path.exists():
            continue
        try:
            _payload = _json_validation.loads(
                _path.read_text(encoding="utf-8")
            )
        except (_json_validation.JSONDecodeError, OSError) as _exc:
            st.warning(f"Could not read {_path.name}: {_exc}")
            continue
        _rows = _payload.get("results", [])
        if not _rows:
            continue

        st.markdown(f"### {_title}")

        # --- Compact table ---
        _table_rows = []
        for r in _rows:
            cd_corr = r.get("cd_corrected")
            cd_err = r.get("cd_error_pct")
            _table_rows.append({
                "Shape": r["shape"],
                "Re": int(r["re"]),
                "Cd raw": round(float(r.get("cd_raw", 0.0)), 3),
                "Cd corrected": (round(float(cd_corr), 3)
                                  if cd_corr is not None else None),
                "Cd ref": (round(float(r.get("cd_ref", 0.0)), 3)
                           if r.get("cd_ref") is not None else None),
                "Cd err %": (round(float(cd_err), 1)
                              if cd_err is not None else None),
                "St raw": (round(float(r.get("st_raw", 0.0)), 3)
                           if r.get("st_raw") is not None else None),
                "St cycles": (round(float(r.get("strouhal_n_cycles", 0.0)), 1)
                              if "strouhal_n_cycles" in r else None),
                "Cd pass": bool(r.get("cd_pass", False)),
            })
        st.dataframe(_table_rows, width="stretch", hide_index=True)

        # --- Per-shape bar chart: AeroLab vs reference ---
        # Cylinder + Square are the two benchmark shapes. We do one chart
        # per shape so the y-axis ranges are sane (Square Cd ~ 1.5 - 2.0,
        # Cylinder Cd ~ 1.0 - 1.4).
        for _shape in sorted({r["shape"] for r in _rows}):
            _shape_rows = [r for r in _rows if r["shape"] == _shape
                           and r.get("cd_ref") is not None]
            if not _shape_rows:
                continue
            _shape_rows.sort(key=lambda r: int(r["re"]))
            _re_labels = [f"Re={int(r['re'])}" for r in _shape_rows]
            _cd_aero = [
                float(r.get("cd_corrected") or r.get("cd_raw"))
                for r in _shape_rows
            ]
            _cd_ref_arr = [float(r["cd_ref"]) for r in _shape_rows]
            _fig = go.Figure()
            _fig.add_trace(go.Bar(
                x=_re_labels, y=_cd_aero,
                name="AeroLab Cd",
                marker_color="#10b981" if _is_headline else "#94a3b8",
            ))
            _fig.add_trace(go.Bar(
                x=_re_labels, y=_cd_ref_arr,
                name=f"Reference ({_shape})",
                marker_color="#3b82f6",
            ))
            _fig.update_layout(
                title=dict(
                    text=f"<span style='font-size:0.95rem;'>"
                         f"{_shape} Cd vs reference</span>",
                    x=0.0, xanchor="left",
                ),
                barmode="group",
                height=300,
                margin=dict(t=50, l=50, r=20, b=40),
                yaxis_title="Cd",
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                             xanchor="left", x=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(_fig, width="stretch")

        # Strouhal-record-quality notice. Every St row currently has
        # n_cycles < 20 (card #5). Surface that here so a reader who
        # only opens this tab still sees the FFT caveat.
        _any_insufficient = any(
            r.get("strouhal_insufficient_record") for r in _rows
        )
        if _any_insufficient:
            st.caption(
                ":material/info_outline: The St columns above are reported "
                "for completeness only. Every row was captured with "
                "fewer than 20 shedding cycles in the FFT window, so the "
                "St-axis bin width is wide vs the Williamson St(Re) range -- "
                "treat St as qualitative agreement, not a percent-error "
                "measurement. VALIDATION.md section 3.4 has the long-form "
                "discussion."
            )
        st.markdown("---")

    # Long-time stability appendix (card #10), surfaced if the JSON has
    # been generated by `scripts/long_time_stability_cylinder.py`. We
    # don't gate the rest of the page on this file existing.
    _lt_path = _v_root / "long_time_cylinder_re100.json"
    if _lt_path.exists():
        try:
            _lt = _json_validation.loads(_lt_path.read_text(encoding="utf-8"))
        except (_json_validation.JSONDecodeError, OSError) as _exc:
            st.warning(f"Could not read {_lt_path.name}: {_exc}")
            _lt = None
        if _lt is not None:
            st.markdown(
                "### Long-time behaviour (cylinder Re = 100, Standard preset)"
            )
            st.caption(
                "How the validated case behaves as we push the run length "
                "past the validated window. SciML's standing critique of "
                "low-cost solvers is that long-time extrapolation is "
                "where accumulated drift wins; this table is AeroLab's "
                "honest answer for the most-instrumented case."
            )
            _lt_table = []
            for r in _lt.get("results", []):
                _lt_table.append({
                    "t_end (D/U)": r.get("t_end_DU"),
                    "n_steps": r.get("n_steps"),
                    "Finished clean": r.get("finished_clean"),
                    "Mass drift (%)": r.get("mass_drift_pct"),
                    "u_peak (lattice)": r.get("u_peak_lattice"),
                    "Cd (last 50 D/U)": r.get("cd_mean"),
                    "St": r.get("strouhal"),
                })
            st.dataframe(_lt_table, width="stretch", hide_index=True)

    st.stop()


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
                # rotation_deg is the new canonical name for non-airfoil
                # shapes (the slider is labelled "Rotation", not "AoA");
                # aoa stays supported as a backward-compat alias so old
                # shared links keep working.
                _a = float(_qp.get("rotation_deg", _qp.get("aoa", "0")))
                if -45.0 <= _a <= 45.0:
                    # AoA / rotation slider step is 0.5; round.
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
        "Upload your own  (PNG, JPG, WEBP, etc.)": "Custom",
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
                    import hashlib as _hl

                    from src.custom_shape import extract_silhouette_from_image
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
                    ":material/info: Upload an image, draw your own shape, "
                    "or open the **Sample** tab for a built-in silhouette."
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
        _clamped_v = float(np.clip(_prev_v, 0.15, _v_max))
        st.session_state["lbm_velocity_slider"] = _clamped_v
        # If we just silently lowered the user's chosen speed (e.g.
        # they rotated the square past 5 deg and the cap dropped from
        # 1.50 to 0.60), tell them. External review flagged this as
        # "the user's speed just changes under them".
        if _prev_v > _v_max + 1e-6 and _prev_v > 0.16:
            st.toast(
                f":material/speed: Flow speed capped to {_v_max:.2f} m/s "
                f"(was {_prev_v:.2f}). At {shape_preset}"
                f"{f' / {_aoa_for_cap:+.0f}' + chr(176) if abs(_aoa_for_cap) >= 5 else ''} "
                f"the solver stays stable up to Re &le; {_re_cap}.",
                icon=":material/info:",
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

        # Inline validity pill (card #7) -- color reflects what region of
        # the 2D-LBM validation envelope the current Re lands in. Bluff
        # bodies only; airfoil shapes use a different reference frame
        # (NeuralFoil polars, chord Re ~ 1e5+) where the bluff-body 2D
        # ceiling does not apply.
        _is_bluff = shape_preset in ("Cylinder", "Square", "Ellipse", "Custom")
        if _is_bluff:
            if reynolds_target <= RE_VALIDATED_MAX:
                _pill_label, _pill_bg, _pill_fg = "validated", "#10b981", "#022c22"
            elif reynolds_target <= RE_EXPLORATORY_MAX:
                _pill_label, _pill_bg, _pill_fg = "exploratory", "#f59e0b", "#451a03"
            else:
                _pill_label, _pill_bg, _pill_fg = "unphysical 2D", "#ef4444", "#450a0a"
            _pill_html = (
                f"<span style='display:inline-block;padding:0.05rem 0.5rem;"
                f"margin-left:0.4rem;background:{_pill_bg};color:{_pill_fg};"
                f"border-radius:9999px;font-size:0.72rem;font-weight:600;"
                f"vertical-align:middle;'>{_pill_label}</span>"
            )
        else:
            _pill_html = ""
        st.markdown(
            f"<div style='color:#94a3b8;font-size:0.85rem;margin-top:-0.25rem;"
            f"margin-bottom:0.45rem;'>"
            f"{velocity_mps:.2f} m/s &nbsp;·&nbsp; Re &asymp; {reynolds_target} "
            f"&nbsp;·&nbsp; <b>{reg}</b> &mdash; air feels {reg_feel}{_pill_html}"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Re-ceiling banner (card #1) -- bluff-body only. The 2D-LBM
        # validation ceiling (Williamson mode-A, Re ~ 190 for cylinder) is
        # the physical limit, not a solver one; above it a strictly 2D
        # solver is a different problem, not "the same problem at higher
        # Re". Airfoil shapes are exempt because they reference a chord
        # Re envelope (~1e5+) where the bluff-body 2D ceiling does not
        # apply -- the NeuralFoil Fast mode handles that regime instead.
        if _is_bluff:
            if reynolds_target <= RE_VALIDATED_MAX:
                st.info(
                    f":material/verified: **Validated range** (Re &le; "
                    f"{RE_VALIDATED_MAX}). Cd and St are benchmarked against "
                    "Williamson 1996 (cylinder) and Okajima 1982 (square). "
                    "See [VALIDATION.md](https://github.com/devansh2003-dev/"
                    "aerolab/blob/main/VALIDATION.md) for the methodology."
                )
            elif reynolds_target <= RE_EXPLORATORY_MAX:
                st.warning(
                    f":material/warning: **Beyond the 2D validation "
                    f"ceiling** (Re &gt; {RE_VALIDATED_MAX}). Real flow "
                    "develops 3D instabilities (oblique shedding, mode-A/B "
                    "vortex dislocations) that a 2D solver cannot capture. "
                    "Results are exploratory visualisation, not validated "
                    "CFD."
                )
            else:
                st.error(
                    f":material/dangerous: **Far beyond 2D validity** "
                    f"(Re &gt; {RE_EXPLORATORY_MAX}). The numbers shown are "
                    "not trustworthy as engineering predictions; the wake "
                    "still animates because the solver is stable, not "
                    "because the physics is right."
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
        # Validation (700 x 400) and Resolved (1200 x 400) presets are
        # offline-only -- many-minute runs that ship via
        # scripts/validate_solver.py, not interactive Streamlit use. The
        # consumer scope (re-locked 2026-05-26) demands two presets in
        # the UI: Standard for first-time visitors, Detailed for the
        # screenshot run. Anything heavier must NOT leak into the radio.
        _DEV_ONLY_PRESETS = ("Validation", "Resolved")
        _ui_res_keys = [
            k for k in RESOLUTION_PRESETS
            if not k.startswith(_DEV_ONLY_PRESETS)
        ]
        st.session_state.setdefault("lbm_res_radio", _ui_res_keys[0])
        res_display = st.radio(
            "Resolution",
            _ui_res_keys,
            label_visibility="collapsed",
            horizontal=True,
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

        def _stash_displayed_inputs():
            """Cache the inputs that drove the currently-displayed result.

            Stash-then-restore lets the user tweak sidebar widgets WITHOUT
            destroying the visible result (external review 2026-05-25,
            item #14). When the new inputs diverge from the stash, the
            block below restores the stashed values for the simulation
            call and surfaces a stale banner -- the user retains visual
            continuity until they click Run again.
            """
            st.session_state["lbm_last_displayed_inputs"] = {
                "shape_preset": shape_preset,
                "shape_display": shape_display,
                "reynolds_target": reynolds_target,
                "aoa_deg": aoa_deg,
                "res_display": res_display,
                "custom_polygon": custom_polygon,
                "viz_mode": viz_mode,
                "polygon_key": _polygon_key,
            }

        if run_clicked:
            st.session_state["lbm_last_displayed_config"] = _current_config
            _stash_displayed_inputs()
        # If a gallery card was just clicked, the widget values have been
        # rewritten via session_state and we want the run to display
        # immediately (without the user clicking Run again).
        if st.session_state.pop("lbm_gallery_pending", False):
            st.session_state["lbm_last_displayed_config"] = _current_config
            _stash_displayed_inputs()
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
            _stash_displayed_inputs()
        _should_display_run = run_clicked or (
            st.session_state.get("lbm_last_displayed_config") == _current_config
        )
        # Stale-display fallback: when the user has tweaked sidebar inputs
        # AFTER a successful run, keep showing the previous result (cache
        # hit on the stashed config) so they retain visual continuity. A
        # banner above the run explains the inputs have diverged and that
        # clicking Run will refresh. Without this, every slider drag
        # collapses the result back to the gallery -- jarring, and the
        # reviewer's specific complaint (item #14, 2026-05-25).
        _stale_display = False
        if (
            not _should_display_run
            and st.session_state.get("lbm_last_displayed_inputs") is not None
        ):
            _stale_display = True
            _should_display_run = True
        if "Standard" in res_display:
            st.caption(":material/timer: ~30-40 s on your laptop "
                       "(plus a ~25 s first-time JIT compile). On the "
                       "free Streamlit Cloud tier expect a few minutes. "
                       "Revisits are instant (cached).")
        else:
            st.caption(":material/timer: ~90-100 s on your laptop "
                       "(plus a ~25 s first-time JIT compile). On the "
                       "free Streamlit Cloud tier expect ~6 min. "
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
            shape_disp, vel_mps, aoa, res, viz, _title, *_ = card
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
            # Clear any custom-shape flip toggle so a preset card doesn't
            # inherit a flip from a prior custom-shape session.
            st.session_state.pop("lbm_custom_flipped", None)
            # Immediate toast so the user sees feedback even before the
            # rerun completes (reviewer item #23, 2026-05-25: clicks on
            # the gallery cards feel unresponsive on the first interaction
            # because the JIT compile + rerun cycle has no visible
            # progress for ~500 ms).
            st.toast(
                f":material/play_arrow: Loading: {_title}",
                icon=":material/play_arrow:",
            )

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
    # Stale-display restore: when the sidebar widgets have drifted from
    # the displayed result, override the local widget vars with the
    # stashed inputs so the simulation call hits cache and the result
    # stays on screen. A banner (rendered just below) tells the user the
    # inputs have diverged and that Run will refresh.
    if _stale_display:
        _stash = st.session_state["lbm_last_displayed_inputs"]
        shape_preset = _stash["shape_preset"]
        shape_display = _stash["shape_display"]
        reynolds_target = _stash["reynolds_target"]
        aoa_deg = _stash["aoa_deg"]
        res_display = _stash["res_display"]
        res_cfg = RESOLUTION_PRESETS[res_display]
        custom_polygon = _stash.get("custom_polygon")
        viz_mode = _stash["viz_mode"]
        _polygon_key = _stash["polygon_key"]
        # Regime label was already computed in the sidebar from the
        # CURRENT (live-widget) reynolds_target. After the override, the
        # title block below would otherwise render "{shape_name} in
        # {reg}" with shape_name from the displayed run but reg from
        # the live slider -- e.g. "Cylinder in fully turbulent flow"
        # while the actual GIF is the Re=200 transitional run. Recompute
        # so title + image agree (reviewer follow-up 2026-05-26).
        reg, reg_feel = regime_label(reynolds_target)
        # Recompute _current_config from the restored locals so the
        # downstream Pin / snapshot / share-link blocks (which compare
        # the snapshot against _current_config) compare against what the
        # user is ACTUALLY looking at, not the current widget state.
        _current_config = (
            shape_preset, int(reynolds_target), float(aoa_deg),
            res_display, _polygon_key, viz_mode,
        )
        st.warning(
            ":material/sync: **Showing your last run.** The sidebar inputs "
            "have changed since this result was computed. Click **Run "
            "simulation** to refresh, or change the sidebar back to match."
        )
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
        # Force the snapshot to re-render in the CURRENT viz_mode so the
        # two panels share a colormap -- otherwise the snapshot might be
        # Velocity (orange/yellow) and the current run Vorticity
        # (red/blue), which can't be visually compared. External review
        # 2026-05-24 flagged this; we now match viz modes silently and
        # only warn if the snapshot's other (shape/Re/AoA) params differ
        # in a way the side-by-side is supposed to show off.
        snap_result = _cached_simulate_and_render(
            snap_shape, snap_re, snap_aoa, snap_res,
            custom_polygon=snap_polygon, viz_mode=viz_mode,
        )
        st.markdown("---")
        st.markdown("### Side-by-side comparison")
        st.caption(
            f"Snapshot on the left, current run on the right -- both "
            f"rendered in **{viz_mode}** mode so colormaps match. "
            f"Clear the snapshot below to return to single-run view."
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
    # When we already showed the side-by-side above, the standalone hero
    # animation is redundant -- it makes the page a long endless scroll.
    # Skip the heading + hero GIF; the action row, legend, and metric
    # strip still render below for both single- and comparison-mode runs.
    _showing_side_by_side = snapshot is not None and not snapshot_is_current
    if not _showing_side_by_side:
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
        if not _showing_side_by_side:
            # "[CFD simulation]" chip (card #3) so a screenshot of the GIF
            # cropped out of the app carries the mode label. The Fast-mode
            # Plotly figures have an equivalent "[ML surrogate]" prefix in
            # their layout.title; this is the LBM-side analog. Keep the
            # chip small so it doesn't compete with the animation.
            st.markdown(
                "<div style='color:#94a3b8;font-size:0.72rem;"
                "margin-bottom:0.35rem;'>"
                "<span style='display:inline-block;padding:0.05rem 0.5rem;"
                "background:rgba(59,130,246,0.12);color:#93c5fd;"
                "border:1px solid rgba(59,130,246,0.35);border-radius:9999px;"
                "font-weight:600;'>[CFD simulation &middot; LBM]</span></div>",
                unsafe_allow_html=True,
            )
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
                ":material/download:  Save GIF",
                data=gif_bytes,
                file_name=_gif_name,
                mime="image/gif",
                width="stretch",
                help="Save the animation locally. Filename encodes shape, Re, "
                     "and AoA so multiple runs don't collide.",
            )
        with action_cols[1]:
            _pin_label = (
                ":material/push_pin:  Pinned"
                if snapshot_is_current else
                ":material/push_pin:  Pin"
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
                    ":material/close:  Clear",
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
                ":material/share:  Share",
                width="stretch",
                disabled=_share_disabled,
                help=_share_help,
                key="share_link_btn",
            ):
                # Use rotation_deg for square + custom (where the angle is
                # geometric rotation, not aerodynamic angle of attack).
                # Backward-compat: still read `aoa` on the receiving end so
                # old shared links keep working.
                _is_aerodynamic_aoa = shape_preset in ("NACA 0012", "NACA 4412")
                _angle_param = "aoa" if _is_aerodynamic_aoa else "rotation_deg"
                # Clear the other angle param to keep URLs canonical.
                if _is_aerodynamic_aoa:
                    if "rotation_deg" in st.query_params:
                        del st.query_params["rotation_deg"]
                else:
                    if "aoa" in st.query_params:
                        del st.query_params["aoa"]
                st.query_params["shape"] = _SHAPE_DISPLAY_TO_QP[shape_preset]
                st.query_params["vel"] = f"{velocity_mps:.2f}"
                st.query_params[_angle_param] = f"{aoa_deg:.1f}"
                st.query_params["res"] = (
                    "standard" if "Standard" in res_display else "detailed"
                )
                st.query_params["viz"] = viz_mode
                # Stash the click so we render an expandable, copyable URL
                # block below the action row -- much better than a toast that
                # tells the user to dig in their address bar.
                st.session_state["_share_url_just_built"] = True

        # Show a copyable share-link block right after the click. Streamlit
        # doesn't expose a native clipboard API across hosts, so we render
        # the URL as a code block: users hit the built-in "copy" icon
        # st.code adds in the top-right corner of every code block. Much
        # better than the old toast that told them to dig in the address bar.
        if st.session_state.get("_share_url_just_built"):
            _qp_str = "&".join(f"{k}={v}" for k, v in st.query_params.to_dict().items())
            _share_url = f"https://aerolab-devansh.streamlit.app/?{_qp_str}"
            st.info(
                ":material/share: **Share link ready.** Copy and send -- "
                "opening this link reopens this exact run."
            )
            st.code(_share_url, language=None)
            # One-shot: clear the flag so the next rerun (e.g. slider drag)
            # doesn't re-render the link block.
            del st.session_state["_share_url_just_built"]

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
                    ":material/push_pin: **Pinned: this run.** "
                    "Change a parameter and click Run to see side-by-side."
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
            "Massless smoke tracers carried by the wind, **colored by "
            "speed**: dark purple = slow or recirculating, orange = "
            "freestream speed, yellow = accelerated."
        )
    with leg_cols[3]:
        st.markdown(
            _swatch.format(color="#1f2937") + "**Dark shape**",
            unsafe_allow_html=True,
        )
        st.markdown(
            "The object itself. Air can't flow through it, so it has "
            "to go around -- which is what creates everything you see "
            "in the wake."
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
                   f"{actual_n_frames}f @ {round(1000 / GIF_FRAME_MS)} fps",
                   help=f"{actual_n_frames} animation frames played at "
                        f"{round(1000 / GIF_FRAME_MS)} frames per second. "
                        f"Loops continuously in the viewer above.")

    # === Measured forces ===
    # Open by default so Cd/Cl/St are visible at a glance -- external
    # review 2026-05-24 flagged the "inverted hierarchy" of these key
    # numbers hidden in a collapsed expander while decorative colorbars
    # were always visible.
    with st.expander(
            ":material/insights: **Forces measured during this run** "
            "(drag, lift, vortex-shedding frequency)",
            expanded=True):
        _st_val = sim_result.get("strouhal", float("nan"))
        _cd_raw = float(sim_result["cd_mean"])
        # Compute the Allen-Vincenti corrected estimate so we can lead
        # with the free-stream-equivalent number instead of the raw
        # 35 %-blocked value (which would otherwise show a +130 % delta
        # against the textbook free-stream Cd and make the solver look
        # broken -- external review 2026-05-24).
        _ny = sim_result.get("lbm_ny", 80)
        _L = sim_result.get("char_length", 1.0)
        _cd_corr, _st_corr, _B, _K = _blockage_corrected(
            shape_preset, _cd_raw, _st_val, _L, _ny,
        )
        _blockage_pct = round(100.0 * _B)
        # Free-stream textbook (Williamson / Okajima) for the "vs textbook"
        # delta. Cylinder always shows; Square shows only when the body
        # presents a flat face (aoa ~ 0) since the table is broadside.
        _show_textbook = (
            shape_preset == "Cylinder"
            or (shape_preset == "Square" and abs(aoa_deg) < 5.0)
        )
        _cd_free, _st_free = (
            _freestream_reference(shape_preset, int(reynolds_target))
            if _show_textbook else (None, None)
        )
        # Compare the CORRECTED estimate to textbook, not the raw value.
        # delta_color="off" stays gray because high-blockage square Cd
        # can still legitimately read +15-25 %.
        _cd_delta = (
            f"{(_cd_corr if _cd_corr is not None else _cd_raw) - _cd_free:+.2f} "
            f"vs free-stream {_cd_free:.2f}"
            if (_cd_free is not None) else None
        )
        _st_delta = (
            f"{(_st_corr if _st_corr is not None else _st_val) - _st_free:+.3f} "
            f"vs free-stream {_st_free:.3f}"
            if (_st_free is not None and np.isfinite(_st_val)) else None
        )

        _force_cols = st.columns(3)
        with _force_cols[0]:
            if _cd_corr is not None:
                st.metric(
                    "Drag (Cd, corrected)",
                    f"{_cd_corr:.2f}",
                    delta=_cd_delta,
                    delta_color="off",
                    help=(
                        "**Blockage-corrected drag coefficient.** Allen-"
                        "Vincenti (1944) / Pope-Harper rescale of the raw "
                        "channel Cd to a free-stream-equivalent estimate. "
                        "At the Standard interactive preset the channel "
                        f"is {_blockage_pct} %-blocked, so this rescale "
                        "is ~ 2.6 x and absorbs both blockage and solver "
                        "error -- a small delta vs free-stream here is a "
                        "property of the correction, not a validation. "
                        "The headline validation runs at a separate "
                        "low-blockage preset (B = 5 %, see VALIDATION.md "
                        "section 3.2); the doc's section 3.5 explains "
                        "this distinction at length."
                    ),
                )
                st.caption(
                    f":gray[Raw (channel): **{_cd_raw:.2f}** &nbsp;|&nbsp; "
                    f"K = {_K:.2f} at B = {_B:.2f} &nbsp;|&nbsp; "
                    f"interactive estimate, not the validation claim]"
                )
                # Validation-status badge (card #2). Numbers come from the
                # VALIDATION.md headline (Resolved preset, D = 40, B = 10 %),
                # not from this Standard-preset interactive run -- the badge
                # is a TRUST label, not a per-run error. Inside the Re <= 200
                # band the literature reference is well-defined; outside it,
                # there is no 2D validation reference at all.
                if shape_preset == "Cylinder" and reynolds_target <= RE_VALIDATED_MAX:
                    _badge_html = (
                        "<span style='display:inline-block;padding:0.1rem 0.55rem;"
                        "background:rgba(16,185,129,0.15);color:#6ee7b7;"
                        "border:1px solid rgba(16,185,129,0.35);border-radius:9999px;"
                        "font-size:0.74rem;font-weight:600;'>"
                        ":material/check_circle: Validated &mdash; "
                        "Williamson 1996, &plusmn;5.6 % (max 10.2 %)</span>"
                    )
                elif shape_preset == "Square" and reynolds_target <= RE_VALIDATED_MAX:
                    _badge_html = (
                        "<span style='display:inline-block;padding:0.1rem 0.55rem;"
                        "background:rgba(16,185,129,0.15);color:#6ee7b7;"
                        "border:1px solid rgba(16,185,129,0.35);border-radius:9999px;"
                        "font-size:0.74rem;font-weight:600;'>"
                        ":material/check_circle: Validated &mdash; "
                        "Okajima 1982, &plusmn;4.5 % (max 5.1 %)</span>"
                    )
                elif reynolds_target > RE_VALIDATED_MAX:
                    _badge_html = (
                        "<span style='display:inline-block;padding:0.1rem 0.55rem;"
                        "background:rgba(245,158,11,0.15);color:#fbbf24;"
                        "border:1px solid rgba(245,158,11,0.35);border-radius:9999px;"
                        "font-size:0.74rem;font-weight:600;'>"
                        ":material/warning: Exploratory &mdash; "
                        f"Re &gt; {RE_VALIDATED_MAX}, no 2D validation reference</span>"
                    )
                else:
                    # Re <= 200 on a shape we don't have a tabulated
                    # reference for (e.g. Ellipse at AoA != 0). Honest
                    # label: the regime is the validated one, but no
                    # paper to compare against.
                    _badge_html = (
                        "<span style='display:inline-block;padding:0.1rem 0.55rem;"
                        "background:rgba(100,116,139,0.18);color:#cbd5e1;"
                        "border:1px solid rgba(148,163,184,0.35);border-radius:9999px;"
                        "font-size:0.74rem;font-weight:600;'>"
                        ":material/info: No literature reference at "
                        "this shape</span>"
                    )
                st.markdown(_badge_html, unsafe_allow_html=True)
            else:
                # Shapes without a tabulated K (Custom polygons,
                # airfoils outside the validation set): just show raw.
                st.metric(
                    "Drag (Cd, raw)",
                    f"{_cd_raw:.2f}",
                    help=(
                        "Raw measured drag coefficient in the simulation "
                        "channel. We don't ship a blockage-correction "
                        "constant for this shape, so this value is what "
                        "the solver measured directly (inflated by the "
                        f"{_blockage_pct} % channel blockage)."
                    ),
                )
                st.caption(":gray[Channel-confined; no correction applied.]")
                # No-reference badge (card #2). Custom polygons and
                # airfoils outside the validation set have no published
                # Cd to compare against, regardless of Re.
                st.markdown(
                    "<span style='display:inline-block;padding:0.1rem 0.55rem;"
                    "background:rgba(100,116,139,0.18);color:#cbd5e1;"
                    "border:1px solid rgba(148,163,184,0.35);border-radius:9999px;"
                    "font-size:0.74rem;font-weight:600;'>"
                    ":material/info: Exploratory &mdash; no 2D validation "
                    "reference for this shape</span>",
                    unsafe_allow_html=True,
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
                _st_shown = _st_corr if _st_corr is not None else _st_val
                st.metric(
                    "Shedding rhythm (St)",
                    f"{_st_shown:.3f}",
                    delta=_st_delta,
                    delta_color="off",
                    help=(
                        "**Strouhal number** (West-Apelt 1982 corrected "
                        "where applicable). How often vortices peel off "
                        "the back of the shape, scaled by size and speed. "
                        "Stays around 0.2 for a cylinder no matter the "
                        "wind speed -- which is exactly why telephone "
                        "wires can hum a steady musical note in a breeze. "
                        "Treat as a qualitative indicator: at our record "
                        "length the FFT bin spacing (~ 0.05) is wider "
                        "than the full Williamson St range, so the value "
                        "lands on a discrete bin rather than a continuous "
                        "St(Re) curve. See VALIDATION.md section 3.4."
                    ),
                )
                if _st_corr is not None:
                    st.caption(
                        f":gray[Raw (channel): **{_st_val:.3f}**]"
                    )
                else:
                    st.caption(
                        ":gray[How fast vortices peel off. ~0.2 for cylinders.]"
                    )
                # St record-quality caption (card #5). The FFT samples
                # frequencies at spacing 1/record_len; the St-axis bin
                # width and the captured cycle count are how a senior
                # reader judges whether the displayed St means anything.
                # n_cycles < 20 = the bin width is on the order of the
                # full Williamson St(Re) range -> any percent-error
                # against the reference is coincidence, not measurement.
                _st_bw = sim_result.get("strouhal_bin_width", float("nan"))
                _st_nc = sim_result.get("strouhal_n_cycles", float("nan"))
                if np.isfinite(_st_bw) and np.isfinite(_st_nc):
                    _bw_lbl = f"bin &plusmn; {_st_bw:.3f}"
                    _nc_lbl = f"{_st_nc:.1f} cycles"
                    if _st_nc < 20:
                        st.markdown(
                            "<div style='color:#fbbf24;font-size:0.78rem;'>"
                            f":material/info_outline: INSUFFICIENT RECORD &mdash; "
                            f"{_nc_lbl} captured ({_bw_lbl}). Bin width "
                            "is wide vs the Williamson range; treat as "
                            "qualitative.</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption(
                            f":gray[{_bw_lbl} &nbsp;·&nbsp; {_nc_lbl} captured]"
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

        # === Methodology line (single source of truth on what those
        # numbers mean) ===
        # The metric tiles already lead with the blockage-corrected Cd,
        # so we no longer need a paragraph explaining the raw-vs-corrected
        # gap. One concise caption with the citation is enough.
        if _show_textbook and _cd_free is not None and _cd_corr is not None:
            st.caption(
                f":material/public: Free-stream reference: "
                f"**Cd &asymp; {_cd_free:.2f}** (Williamson 1996 / "
                f"Okajima 1982). Our channel is **{_blockage_pct} %-"
                f"blocked**; the corrected Cd above applies the standard "
                f"Allen-Vincenti rescale with K = {_K:.2f}, recovering "
                f"the free-stream value within &plusmn; 15 % (cylinder) "
                f"or &plusmn; 25 % (square) across Re 100 - 1000. "
                f"Strouhal carries the West-Apelt 1982 correction but at "
                f"this blockage the channel-resonance mode limits its "
                f"predictive value -- see "
                f"[VALIDATION.md](https://github.com/devansh2003-dev/aerolab/blob/main/VALIDATION.md) "
                f"section 4.1."
            )

        # Compact verdict using the BLOCKAGE-CORRECTED Strouhal vs the
        # free-stream Williamson / Okajima reference. Green if within
        # 25 %, blue otherwise. This complements the delta chip on the
        # metric tile above with a plain-English physics interpretation.
        if _show_textbook and _st_free is not None and _st_corr is not None and np.isfinite(_st_val):
            _st_err_pct = abs(_st_corr - _st_free) / _st_free * 100
            _bias_text = {
                "Cylinder": (
                    "The corrected Cd above is the apples-to-apples "
                    "comparison with Williamson 1996; raw channel Cd is "
                    "elevated by the channel walls accelerating the flow."
                ),
                "Square": (
                    "Square Cd is geometry-locked at the corners; "
                    "the corrected value above is the apples-to-apples "
                    "comparison with Okajima 1982 (within +/- 25 %)."
                ),
            }[shape_preset]
            if _st_err_pct < 25:
                st.success(
                    f":material/check_circle: **Shedding physics match** "
                    f"-- corrected Strouhal within {_st_err_pct:.0f} % of "
                    f"Williamson / Okajima at Re={int(reynolds_target)}.  "
                    f"{_bias_text}"
                )
            else:
                st.info(
                    f":material/info: **Strouhal is {_st_err_pct:.0f} % "
                    f"off Williamson / Okajima** at "
                    f"Re={int(reynolds_target)}. At this blockage the "
                    f"channel-resonance shedding mode is hard to fully "
                    f"correct -- try a longer / Detailed run so the FFT "
                    f"has more cycles to lock onto.  {_bias_text}"
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
# Persistent ML-surrogate disclaimer (card #3). The numbers below are
# NeuralFoil predictions: a neural network trained on XFoil / RANS data,
# not a live simulation. A first-time visitor needs to know which mode is
# the surrogate and which is the solver before they read a Cd value here.
st.caption(
    ":material/psychology: **[ML surrogate]** NeuralFoil prediction -- a "
    "neural network trained on XFoil / RANS data, not a live simulation. "
    "Use **CFD (LBM solver)** in the sidebar for time-resolved flow fields."
)
st.caption(
    ":material/bolt: For full fluid simulation with visible streamlines and "
    "wake structure, switch to **CFD (LBM solver)** in the sidebar."
)


# `normalize_naca` and `thickness_camber` moved to src/airfoils.py during
# the D1 split -- they're imported at the top of this file. NeuralFoil
# model-size config lives below.

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

# Parse and dedupe airfoil names while preserving input order. Each token
# can fail validation independently -- we surface a per-token warning so
# 'naca4412, banana' shows the polar for naca4412 and explains why banana
# was skipped, instead of crashing the whole app.
seen = set()
airfoil_names = []
for raw in raw_names.split(","):
    if not raw.strip():
        continue
    try:
        n = normalize_naca(raw)
    except ValueError as exc:
        st.warning(f"Skipping {raw.strip()!r}: {exc}")
        continue
    if n not in seen:
        seen.add(n)
        airfoil_names.append(n)

if not airfoil_names:
    st.warning(
        "No valid airfoils to analyze. Try a 4- or 5-digit NACA code "
        "like `naca4412` or `naca23012`."
    )
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

    # Defensive: get_airfoil already raises on None coords, but a future
    # caller path could bypass it -- so we keep the guard cheap rather
    # than crash on af.coordinates[:, 0].
    if af.coordinates is None or len(af.coordinates) == 0:
        st.warning(f"Skipping {label}: no geometry coordinates available.")
        continue

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
    height=400,
    # "[ML surrogate]" prefix (card #3) so a screenshot taken out of
    # context still carries the mode label -- a first-time GitHub visitor
    # who opens an Issue with this image attached must be able to tell
    # which mode produced it.
    title=dict(
        text="<span style='font-size:0.78rem;color:#94a3b8;'>"
             "[ML surrogate &middot; NeuralFoil]</span>",
        x=0.0, xanchor="left", y=0.99, yanchor="top",
        font=dict(size=11),
    ),
    # Subplot titles sit at y~1.0 in paper coords; the legend has to clear
    # them, otherwise the "Airfoil shape" caption gets occluded by airfoil
    # name chips (reviewer-flagged collision, 2026-05-25).
    margin=dict(t=110, l=50, r=20, b=50),
    legend=dict(orientation="h", yanchor="bottom", y=1.18, xanchor="left", x=0),
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

st.subheader(f"Coefficients at &alpha; = {alpha:+.2f}&deg;, Re = {reynolds:.0e}",
             anchor=False)
st.caption(f"NeuralFoil model: **{nf_model_display}**")
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
    height=480,
    # "[ML surrogate]" prefix (card #3). See geom_fig.update_layout above
    # for the rationale.
    title=dict(
        text="<span style='font-size:0.78rem;color:#94a3b8;'>"
             "[ML surrogate &middot; NeuralFoil]</span>",
        x=0.0, xanchor="left", y=0.99, yanchor="top",
        font=dict(size=11),
    ),
    margin=dict(t=110, l=50, r=20, b=50),
    legend=dict(orientation="h", yanchor="bottom", y=1.18, xanchor="left", x=0),
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
