"""Day 1 smoke test: compute CL, CD, and L/D for NACA 4412 at alpha=5 deg, Re=500k.

Run from the project root:
    python scripts/day1_test.py
"""
import sys
from pathlib import Path

# Add project root to sys.path so we can import from src/ without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.airfoils import analyze_airfoil

alpha_deg = 5.0          # angle of attack, degrees
reynolds = 500_000       # Reynolds number based on chord

aero = analyze_airfoil("naca4412", alpha_deg, reynolds)

# NeuralFoil returns 1-D length-1 numpy arrays for scalar inputs;
# .item() pulls out the Python scalar (NumPy 2.x rejects float() on non-0-d arrays).
cl = aero["CL"].item()
cd = aero["CD"].item()
ld = aero["LD"].item()

print(f"NACA 4412  |  alpha = {alpha_deg} deg  |  Re = {reynolds:,}")
print(f"  CL  = {cl:.4f}")
print(f"  CD  = {cd:.4f}")
print(f"  L/D = {ld:.2f}")
