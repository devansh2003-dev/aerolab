"""AeroLab Streamlit app -- interactive airfoil polars with comparison mode.

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

st.title("AeroLab")
st.caption(
    "Interactive airfoil aerodynamics. Pick one or more NACA airfoils, sweep "
    "angle of attack, compare lift, drag, and the drag polar -- all powered by "
    "NeuralFoil."
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


# --- Sidebar inputs ---
with st.sidebar:
    st.header("Inputs")
    raw_names = st.text_input(
        "Airfoils (comma-separated)",
        value="4412, 0012",
        help='Examples: "4412" for one, "0012, 4412, 2412" to compare. NACA prefix optional.',
    )
    alpha = st.slider("Angle of attack alpha (deg)", -5.0, 15.0, 5.0, 0.25)
    reynolds = st.select_slider(
        "Reynolds number",
        options=[1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7],
        value=5e5,
        format_func=lambda x: f"{x:.0e}",
    )

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
# Caching by (name, Re) means slider drags on alpha don't re-invoke NeuralFoil for
# the polar sweep -- only the alpha-dependent point evaluation re-runs.
@st.cache_data(show_spinner=False)
def sweep_polar(name: str, Re: float):
    alphas = np.linspace(-5, 15, 81)  # 0.25 deg resolution for smooth curves
    aero = analyze_airfoil(name, alphas, Re)
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
    p = analyze_airfoil(name, alpha, float(reynolds))
    table_rows.append(
        {
            "Airfoil": name.upper(),
            "CL": round(p["CL"].item(), 4),
            "CD": round(p["CD"].item(), 4),
            "L/D": round(p["LD"].item(), 1),
        }
    )
    sweep_results[name] = sweep_polar(name, float(reynolds))

st.subheader(f"Coefficients at alpha = {alpha:+.2f} deg, Re = {reynolds:.0e}")
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
