"""AeroLab Streamlit app -- interactive airfoil polars with comparison mode.

Run from the project root:
    streamlit run app.py

The browser opens automatically at http://localhost:8501.
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.airfoils import analyze_airfoil

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


# --- Cached polar sweep ---
# Caching by (name, Re) means slider drags on alpha don't re-invoke NeuralFoil for the
# polar sweep -- only the alpha-dependent point evaluation re-runs (one cheap NN call).
@st.cache_data(show_spinner=False)
def sweep_polar(name: str, Re: float):
    alphas = np.linspace(-5, 15, 81)  # 0.25 deg resolution for smooth curves
    aero = analyze_airfoil(name, alphas, Re)
    return alphas, aero["CL"], aero["CD"], aero["LD"]


# --- Single-point predictions at the current alpha (for the comparison table) ---
table_rows = []
valid_names = []
for name in airfoil_names:
    try:
        p = analyze_airfoil(name, alpha, float(reynolds))
        table_rows.append(
            {
                "Airfoil": name.upper(),
                "CL": round(p["CL"].item(), 4),
                "CD": round(p["CD"].item(), 4),
                "L/D": round(p["LD"].item(), 1),
            }
        )
        valid_names.append(name)
    except Exception as e:
        st.warning(f"Skipping {name.upper()!r}: {e}")

if not valid_names:
    st.error("None of the requested airfoils could be analyzed.")
    st.stop()

st.subheader(f"Coefficients at alpha = {alpha:+.2f} deg, Re = {reynolds:.0e}")
st.dataframe(table_rows, width="stretch", hide_index=True)

# --- Plotly figure: 3 subplots with all valid airfoils overlaid ---
fig = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=("Lift curve", "Drag curve", "Drag polar"),
    horizontal_spacing=0.08,
)

# Plotly's default qualitative palette -- categorically distinct colors for overlays.
palette = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

for i, name in enumerate(valid_names):
    alphas_arr, cl, cd, _ = sweep_polar(name, float(reynolds))
    color = palette[i % len(palette)]
    label = name.upper()
    # legendgroup ties the three traces for one airfoil together: clicking the legend
    # entry hides/shows the airfoil in all three subplots simultaneously. showlegend
    # only on the first trace per group avoids three duplicate legend entries.
    common = dict(mode="lines", line=dict(color=color, width=2), legendgroup=label)
    fig.add_trace(
        go.Scatter(x=alphas_arr, y=cl, name=label, showlegend=True, **common),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=alphas_arr, y=cd, name=label, showlegend=False, **common),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(x=cd, y=cl, name=label, showlegend=False, **common),
        row=1, col=3,
    )

# Vertical alpha-tracker on the alpha-axis subplots only (drag polar is CD vs CL,
# alpha isn't a coordinate there).
fig.add_vline(x=alpha, line=dict(color="black", dash="dash", width=1), opacity=0.4, row=1, col=1)
fig.add_vline(x=alpha, line=dict(color="black", dash="dash", width=1), opacity=0.4, row=1, col=2)

# Axis labels per subplot.
fig.update_xaxes(title_text="alpha (deg)", row=1, col=1)
fig.update_yaxes(title_text="CL", row=1, col=1)
fig.update_xaxes(title_text="alpha (deg)", row=1, col=2)
fig.update_yaxes(title_text="CD", row=1, col=2)
fig.update_xaxes(title_text="CD", row=1, col=3)
fig.update_yaxes(title_text="CL", row=1, col=3)

fig.update_layout(
    height=440,
    margin=dict(t=60, l=50, r=20, b=50),
    legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
    hovermode="x unified",  # hover anywhere on a subplot to see all series at that x
)

st.plotly_chart(fig, width="stretch")

st.caption(
    "Click an airfoil name in the legend to toggle it across all three panels. "
    "Hover for exact values. Drag to zoom; double-click to reset. The polar sweep "
    "is cached per (airfoil, Re), so dragging the alpha slider only refreshes the "
    "tracker line and the table."
)
