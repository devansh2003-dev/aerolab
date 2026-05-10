"""Day 1 smoke test: compute CL, CD, and L/D for NACA 4412 at alpha=5 deg, Re=500k.

Run from the project root:
    python scripts/day1_test.py
"""
import aerosandbox as asb
import neuralfoil as nf

# asb.Airfoil("naca####") generates the 4-digit NACA coordinates from the formula --
# no .dat file needed.
airfoil = asb.Airfoil("naca4412")

alpha_deg = 5.0          # angle of attack, degrees
reynolds = 500_000       # Reynolds number based on chord

# NeuralFoil's neural net predicts XFOIL-quality polars in ~1 ms.
# model_size trades accuracy for speed; "xxxlarge" is the most accurate.
aero = nf.get_aero_from_airfoil(
    airfoil=airfoil,
    alpha=alpha_deg,
    Re=reynolds,
    model_size="xxxlarge",
)

# NeuralFoil returns 1-D length-1 numpy arrays for scalar inputs. NumPy 2.x no longer
# allows float() on non-0-d arrays, so use .item() to pull out the Python scalar.
cl = aero["CL"].item()
cd = aero["CD"].item()
ld = cl / cd

print(f"NACA 4412  |  alpha = {alpha_deg} deg  |  Re = {reynolds:,}")
print(f"  CL  = {cl:.4f}")
print(f"  CD  = {cd:.4f}")
print(f"  L/D = {ld:.2f}")
