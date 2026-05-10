"""AeroLab Streamlit app -- interactive airfoil polars with comparison mode.

Run from the project root:
    streamlit run app.py

The browser opens automatically at http://localhost:8501.
"""
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

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


# --- Cached single-point and polar-sweep helpers ---
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

# --- Plot: 3 panels with all valid airfoils overlaid ---
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

# Use a categorical colormap so overlapping curves are visually distinct.
colors = plt.cm.tab10(np.linspace(0, 1, 10))

for i, name in enumerate(valid_names):
    alphas_arr, cl, cd, _ = sweep_polar(name, float(reynolds))
    c = colors[i % len(colors)]
    label = name.upper()
    axes[0].plot(alphas_arr, cl, color=c, linewidth=1.8, label=label)
    axes[1].plot(alphas_arr, cd, color=c, linewidth=1.8, label=label)
    axes[2].plot(cd, cl, color=c, linewidth=1.8, label=label)

# Alpha-tracker line on the two alpha-axis plots (omit on the drag polar where
# alpha isn't a coordinate).
for ax in axes[:2]:
    ax.axvline(alpha, color="k", linestyle="--", alpha=0.4, linewidth=1)

axes[0].axhline(0, color="k", linewidth=0.5)
axes[0].set_xlabel("alpha (deg)")
axes[0].set_ylabel("CL")
axes[0].set_title("Lift curve")
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc="best", fontsize=9)

axes[1].set_xlabel("alpha (deg)")
axes[1].set_ylabel("CD")
axes[1].set_title("Drag curve")
axes[1].grid(True, alpha=0.3)
axes[1].legend(loc="best", fontsize=9)

axes[2].set_xlabel("CD")
axes[2].set_ylabel("CL")
axes[2].set_title("Drag polar")
axes[2].grid(True, alpha=0.3)
axes[2].legend(loc="best", fontsize=9)

fig.suptitle(f"Re = {reynolds:.0e}", fontsize=12)
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)  # prevent figure leak across Streamlit reruns

st.caption(
    "Dashed line marks the current alpha on the lift and drag curves. The polar "
    "and the underlying neural-net evaluation are cached per (airfoil, Re), so "
    "only the table refreshes when you drag the alpha slider."
)
