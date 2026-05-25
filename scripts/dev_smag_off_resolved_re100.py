"""One-shot Smagorinsky-off experiment for Cylinder Re = 100 at the
Resolved preset (D = 40, B = 10 %).

The reviewer (2026-05-27) flagged the Resolved cylinder Re = 100
result of -10.2 % as a yellow flag: it is the most laminar case, it
is where the solver should be most accurate, and the error appeared
only when grid resolution was improved from D = 20 to D = 40. The
section 4.4 LES-at-laminar-Re note already suggests the Smagorinsky
sub-grid eddy viscosity is a candidate cause.

This script runs the same case with C_SMAG = C_SMAG_SQ = 0 so any
delta from the Resolved sweep is the LES contribution. Approach:
monkey-patch ``src.lbm.C_SMAG`` / ``C_SMAG_SQ`` BEFORE the first call
to the JIT-compiled MRT kernel. Numba captures module-level globals
at trace time, so as long as the patch lands before the first solver
call, the compiled kernel sees the new value. ``@njit`` in this
project uses no on-disk cache, so we do not need to clear one.

Runtime ~ 18-20 min single-thread on a laptop (matches the on-the-
record Resolved sweep timing per case).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Project root on sys.path so this works whether invoked as a script
# or via ``python -m scripts.dev_smag_off_resolved_re100``.
_PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ_ROOT))

import src.lbm as lbm_module  # noqa: E402

# Patch BEFORE importing simulate_and_render / run_case (those triggers
# the chain that lazy-compiles step_njit_mrt_with_force).
_ORIGINAL_C_SMAG = lbm_module.C_SMAG
_ORIGINAL_C_SMAG_SQ = lbm_module.C_SMAG_SQ
lbm_module.C_SMAG = 0.0
lbm_module.C_SMAG_SQ = 0.0
print(f"[smag-off] patched C_SMAG: {_ORIGINAL_C_SMAG} -> {lbm_module.C_SMAG}")
print(f"[smag-off] patched C_SMAG_SQ: {_ORIGINAL_C_SMAG_SQ} -> {lbm_module.C_SMAG_SQ}")

from scripts.validate_solver import run_case  # noqa: E402

print("[smag-off] running Cylinder Re=100 at Resolved (D=40, B=10%, n_frames=300) ...")
t0 = time.time()
r = run_case(
    "Cylinder", 100, aoa_deg=0.0,
    resolution="Resolved (1200 x 400)", n_frames=300,
)
elapsed = time.time() - t0

print()
print("=" * 70)
print("[smag-off] Cylinder Re=100, Resolved preset, C_SMAG = 0.0")
print("=" * 70)
print(f"  Cd_raw       = {r.cd_raw:.4f}")
print(f"  Cd_corrected = {r.cd_corrected:.4f}")
print(f"  Cd_ref       = {r.cd_ref}")
print(f"  Cd_error_pct = {r.cd_error_pct:+.2f} %")
print(f"  St_raw       = {r.st_raw:.4f}")
print(f"  St_corrected = {r.st_corrected:.4f}")
print(f"  runtime      = {elapsed:.1f} s")
print()
print("[reference] Smagorinsky-on at same case (from results_resolved.json):")
print("  Cd_raw=1.4965 Cd_corrected=1.1859 Cd_error_pct=-10.16 %")
print()
print("[interpretation] If Smag-off Cd_error_pct is significantly closer")
print("to zero, the LES eddy viscosity at laminar Re was the source of")
print("the -10 % bias. If it is still ~ -10 %, the bias is something else")
print("(K-correction over-rescale, body discretisation, wall BC).")
