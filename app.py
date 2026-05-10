"""AeroLab Streamlit app -- interactive airfoil polars.

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
    "Interactive airfoil aerodynamics. Pick a NACA airfoil, sweep angle of attack, "
    "see lift, drag, and the drag polar -- all powered by NeuralFoil."
)

# --- Sidebar inputs ---
with st.sidebar:
    st.header("Inputs")
    raw_name = st.text_input(
        "Airfoil (NACA 4-digit)",
        value="4412",
        help='Examples: "4412", "0012", "2412". Prefix "naca" is optional.',
    )
    alpha = st.slider("Angle of attack alpha (deg)", -5.0, 15.0, 5.0, 0.25)
    reynolds = st.select_slider(
        "Reynolds number",
        options=[1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7],
        value=5e5,
        format_func=lambda x: f"{x:.0e}",
    )

# Normalize the airfoil name: accept "4412", "naca4412", "NACA 4412", etc.
naca_id = raw_name.strip().lower().replace("naca", "").replace(" ", "")
full_name = f"naca{naca_id}"

# --- Single-point prediction (for the metric cards + the highlighted dot) ---
try:
    point = analyze_airfoil(full_name, alpha, float(reynolds))
except Exception as e:
    # asb.Airfoil raises on invalid names; surface a friendly message instead of a traceback.
    st.error(f"Could not analyze airfoil {full_name!r}: {e}")
    st.stop()

cl_pt = point["CL"].item()
cd_pt = point["CD"].item()
ld_pt = point["LD"].item()

c1, c2, c3 = st.columns(3)
c1.metric("Lift coefficient C_L", f"{cl_pt:.4f}")
c2.metric("Drag coefficient C_D", f"{cd_pt:.4f}")
c3.metric("L/D", f"{ld_pt:.1f}")

# --- Cached polar sweep ---
# st.cache_data memoizes by argument values. The expensive NeuralFoil sweep only
# re-runs when (airfoil name, Re) changes -- not when the alpha slider moves.
@st.cache_data(show_spinner="Computing polar sweep...")
def sweep_polar(name: str, Re: float):
    alphas = np.linspace(-5, 15, 81)  # 0.25 deg resolution for smooth curves
    aero = analyze_airfoil(name, alphas, Re)
    return alphas, aero["CL"], aero["CD"], aero["LD"]


alphas, cl, cd, ld = sweep_polar(full_name, float(reynolds))

# --- Plot: 3-panel CL-alpha, CD-alpha, drag polar, with current point marked ---
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

axes[0].plot(alphas, cl, color="tab:blue", linewidth=1.8)
axes[0].axvline(alpha, color="k", linestyle="--", alpha=0.4, linewidth=1)
axes[0].scatter([alpha], [cl_pt], color="k", zorder=5)
axes[0].set_xlabel("alpha (deg)")
axes[0].set_ylabel("CL")
axes[0].set_title("Lift curve")
axes[0].grid(True, alpha=0.3)

axes[1].plot(alphas, cd, color="tab:red", linewidth=1.8)
axes[1].axvline(alpha, color="k", linestyle="--", alpha=0.4, linewidth=1)
axes[1].scatter([alpha], [cd_pt], color="k", zorder=5)
axes[1].set_xlabel("alpha (deg)")
axes[1].set_ylabel("CD")
axes[1].set_title("Drag curve")
axes[1].grid(True, alpha=0.3)

axes[2].plot(cd, cl, color="tab:green", linewidth=1.8)
axes[2].scatter([cd_pt], [cl_pt], color="k", zorder=5)
axes[2].set_xlabel("CD")
axes[2].set_ylabel("CL")
axes[2].set_title("Drag polar")
axes[2].grid(True, alpha=0.3)

fig.suptitle(f"{full_name.upper()}  |  Re = {reynolds:.0e}", fontsize=12)
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)  # prevent figure leak across Streamlit reruns

st.caption(
    "Black dot marks the current alpha. Drag the slider to see the working point "
    "move along the curves; the polar itself only re-computes when you change the "
    "airfoil or Reynolds number."
)
