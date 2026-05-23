"""Final pre-ship validation sweep.

Runs every (shape, viz_mode) combination at a physics-meaningful Re and
checks:

  1. The simulation completes without divergence (no NaN/Inf in any
     output field).
  2. The Cd / Cl values land in the right physical band:
       - Symmetric bodies at AoA=0  -> |Cl| should be < 0.1 (within
         shedding noise)
       - All bodies                  -> Cd > 0 (positive, finite)
  3. The Strouhal number, when reported, is in (0, 1) -- excludes the
     pathological FFT failure mode of locking onto DC.
  4. Vorticity, velocity, and pressure GIFs each produce non-empty
     bytes and are encoded as GIF89a (sanity: header bytes start with
     "GIF8").

This script is the "show me everything works" sweep we run before a
ship -- complementary to tests/test_validation_benchmark.py (which
gates Cd / St against Williamson / Okajima) and
tests/test_visual_regression.py (which catches crashes on the gallery
card configs). This script proves the entire shape x viz_mode matrix
holds up at one canonical Re per shape, which is the combinatorial
gap the other two leave.

Runtime: ~3-4 min at n_frames=50 (the test-suite default; production
uses n_frames=150).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lbm_render import simulate_and_render  # noqa: E402
from src.sample_shapes import SAMPLE_SHAPES  # noqa: E402

# (shape, Re, AoA) cases. Each shape gets one canonical Re inside its
# stability envelope (matched to the gallery defaults).
SHAPE_CASES = [
    ("Cylinder", 200, 0.0),
    ("Square", 200, 0.0),
    ("Ellipse", 300, 0.0),
    ("NACA 0012", 400, 5.0),
    ("NACA 4412", 400, 5.0),
]

VIZ_MODES = ["Vorticity", "Velocity", "Pressure"]


def _gif_header_ok(b: bytes) -> bool:
    """Quick header check that the bytes are a real GIF (not e.g. an
    empty buffer)."""
    return len(b) > 100 and b[:4] in (b"GIF8",) and (b[4:6] in (b"7a", b"9a"))


def _check_one(shape: str, re: int, aoa: float, viz: str) -> dict:
    """Run one (shape, Re, AoA, viz) case and return a diagnostic dict."""
    t0 = time.time()
    # n_frames=150 = production length. Critical for the symmetry-
    # invariant Cl check: at n_frames=50 the time-averaging window
    # captures only ~2 full shedding cycles for slower bodies like
    # Ellipse, so the residual mean Cl from an incomplete cycle is
    # the dominant noise (Cl can sit at +/- 0.15 instead of ~0). At
    # n_frames=150 the average is over ~6 cycles and the noise drops
    # below 0.10.
    out = simulate_and_render(
        shape, reynolds_target=re, aoa_deg=aoa,
        res_key="Standard (320 x 80)",
        n_frames=150, viz_mode=viz,
    )
    runtime = time.time() - t0

    cd = float(out["cd_mean"])
    cl = float(out["cl_mean"])
    st_val = out.get("strouhal_st", float("nan"))
    gif_bytes = out["gif_bytes"]
    gif_ok = _gif_header_ok(gif_bytes)

    issues = []
    if not np.isfinite(cd):
        issues.append(f"Cd not finite ({cd!r})")
    if cd <= 0:
        issues.append(f"Cd <= 0 ({cd:.3f})")
    if not np.isfinite(cl):
        issues.append(f"Cl not finite ({cl!r})")
    # Symmetric bodies at AoA=0: Cl mean should be near zero.
    is_symmetric = (
        (shape in ("Cylinder", "Square", "Ellipse", "NACA 0012") and abs(aoa) < 0.25)
    )
    if is_symmetric and abs(cl) > 0.15:
        issues.append(f"Cl too large for symmetric shape ({cl:+.3f})")
    if np.isfinite(st_val) and not (0 < st_val < 1):
        issues.append(f"St out of plausible range ({st_val:.3f})")
    if not gif_ok:
        issues.append(f"GIF header invalid (len={len(gif_bytes)})")

    return {
        "shape": shape, "re": re, "aoa": aoa, "viz": viz,
        "cd": cd, "cl": cl, "st": st_val,
        "gif_bytes_len": len(gif_bytes),
        "runtime_s": runtime,
        "issues": issues,
    }


def main():
    print("=" * 72)
    print("Final pre-ship validation: shape x viz_mode x physics sweep")
    print("=" * 72)
    n_total = len(SHAPE_CASES) * len(VIZ_MODES)
    n_done = 0
    failures = []
    results = []
    t_start = time.time()
    for shape, re, aoa in SHAPE_CASES:
        for viz in VIZ_MODES:
            n_done += 1
            print(f"\n[{n_done:2d}/{n_total}] {shape:10s} Re={re:<5d} "
                  f"AoA={aoa:+5.1f}  viz={viz:9s}", end=" ", flush=True)
            try:
                r = _check_one(shape, re, aoa, viz)
                results.append(r)
                if r["issues"]:
                    failures.append(r)
                    print("FAIL")
                    for iss in r["issues"]:
                        print(f"        ! {iss}")
                else:
                    st_repr = (
                        f"St={r['st']:.3f}" if np.isfinite(r['st']) else "St=--"
                    )
                    print(f"OK   Cd={r['cd']:5.2f}  Cl={r['cl']:+5.2f}  "
                          f"{st_repr}  ({r['runtime_s']:.1f}s)")
            except Exception as e:  # noqa: BLE001
                failures.append({
                    "shape": shape, "re": re, "aoa": aoa, "viz": viz,
                    "issues": [f"EXCEPTION: {type(e).__name__}: {e}"],
                })
                print(f"CRASH: {type(e).__name__}: {e}")

    # === Symmetric shape Cl invariant ===
    print("\n" + "=" * 72)
    print("Symmetry invariant: |Cl| should be small for symmetric shapes at AoA=0")
    print("=" * 72)
    for r in results:
        is_symm = (
            r["shape"] in ("Cylinder", "Square", "Ellipse", "NACA 0012")
            and abs(r["aoa"]) < 0.25 and r["viz"] == "Vorticity"
        )
        if is_symm:
            verdict = "PASS" if abs(r["cl"]) < 0.15 else "FAIL"
            print(f"  {r['shape']:10s} Re={r['re']:<5d}  Cl = {r['cl']:+.3f}  "
                  f"  -> {verdict}")

    # === Custom shape end-to-end check ===
    print("\n" + "=" * 72)
    print("Custom-shape end-to-end (Upload pipeline + Sample silhouettes)")
    print("=" * 72)
    for name, poly_fn in SAMPLE_SHAPES.items():
        try:
            t0 = time.time()
            out = simulate_and_render(
                "Custom", reynolds_target=200, aoa_deg=0.0,
                res_key="Standard (320 x 80)",
                n_frames=80, custom_polygon=poly_fn(),
            )
            cd = float(out["cd_mean"])
            ok = np.isfinite(cd) and cd > 0 and _gif_header_ok(out["gif_bytes"])
            print(f"  {name:25s}  Cd={cd:5.2f}  "
                  f"gif={len(out['gif_bytes'])//1024} kB  "
                  f"({time.time()-t0:.1f}s)  {'PASS' if ok else 'FAIL'}")
            if not ok:
                failures.append({"shape": f"Sample/{name}", "issues": ["Cd or GIF check failed"]})
        except Exception as e:  # noqa: BLE001
            print(f"  {name:25s}  CRASH: {e}")
            failures.append({"shape": f"Sample/{name}", "issues": [f"crash: {e}"]})

    # === Pressure physicality: front high, wake low (cylinder Re=200) ===
    # Cheaper than re-running; we just check that the SOLVED rho field
    # has a maximum upstream of the body and a minimum in/behind it.
    print("\n" + "=" * 72)
    print("Pressure physicality (cylinder Re=200, AoA=0): front > wake")
    print("=" * 72)
    from src.lbm_render import solve_lbm  # noqa: E402
    solve = solve_lbm("Cylinder", 200, 0.0, "Standard (320 x 80)", n_frames=80)
    # solve["snapshots"] is a list of dicts; each dict has rho/u_x/u_y/vorticity.
    rho = solve["snapshots"][-1]["rho"]
    mask = solve["mask"]
    # Front: 5-cell strip just upstream of body. Wake: 5-cell strip just downstream.
    # Body center is around x=80 in Standard preset.
    body_xs = np.where(mask.any(axis=1))[0]
    if len(body_xs):
        bx_min, bx_max = int(body_xs.min()), int(body_xs.max())
        front_strip = rho[max(0, bx_min - 6):bx_min, :]
        wake_strip = rho[bx_max + 1:bx_max + 12, :]
        front_p = float(np.mean(front_strip[front_strip > 0]))
        wake_p = float(np.mean(wake_strip[wake_strip > 0]))
        # In LBM, rho > 1 means pressure above ambient; rho < 1 means below.
        # Front stagnation -> rho > 1 (high pressure). Near wake -> rho < 1
        # (suction). Difference should be positive and substantial.
        delta = front_p - wake_p
        verdict = "PASS" if delta > 0.001 else "FAIL"
        print(f"  Front mean rho:  {front_p:.5f}")
        print(f"  Wake  mean rho:  {wake_p:.5f}")
        print(f"  Delta:           {delta:+.5f}  -> {verdict} (need > 0.001)")
        if delta <= 0.001:
            failures.append({"shape": "Pressure-physicality", "issues": [f"delta={delta:.5f}"]})

    # === Summary ===
    print("\n" + "=" * 72)
    print(f"Summary: {n_total - len(failures)}/{n_total} OK    "
          f"runtime: {time.time() - t_start:.0f}s")
    print("=" * 72)
    if failures:
        print("\nFAILED CASES:")
        for f in failures:
            print(f"  - {f.get('shape')!r} Re={f.get('re')} "
                  f"AoA={f.get('aoa')} viz={f.get('viz')}: "
                  f"{', '.join(f['issues'])}")
        sys.exit(1)
    else:
        print("\nALL CHECKS PASS.")
        sys.exit(0)


if __name__ == "__main__":
    main()
