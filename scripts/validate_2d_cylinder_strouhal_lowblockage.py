"""2D cylinder Re = 100 Strouhal cross-check at the Validation preset.

Closes item #10 from the v0.6.5.1 senior re-audit:

    > Strouhal is OpenFOAM <-> Williamson only. AeroLab's own St is
    > n/a in the (VALIDATION.md sec 8.4) table -- so St is not
    > cross-validated against AeroLab.

The existing Validation-preset run in ``data/validation/results_lowblockage.json``
extracts a Strouhal but flags ``strouhal_insufficient_record: True``
(4 cycles in the FFT window). Williamson's St = 0.166 means a
shedding period of 1 / 0.166 ~ 6.0 D/U; on a record with only 4
cycles the FFT bin width is ~0.04 in St units, comparable to the
expected error -- not a useful number.

This script re-runs the SAME geometry (Validation preset:
700 x 400, D = 20, B = 5 %, cylinder Re = 100) for long enough to
get >= 20 shedding cycles in the FFT window. With St ~ 0.16 - 0.18
and U/D = 0.005 (lattice u_in = 0.1 / D = 20), one D/U is 200 LBM
steps and one shedding period is ~ 1100 - 1250 steps. To collect
~ 20 cycles with a ~ 30 D/U startup transient skipped, we need:

    total_steps >= 30 D/U (startup) + 20 cycles * 1250 steps/cycle
                 = 6 000 + 25 000 = 31 000 steps
                 ~ 885 frames at STEPS_PER_FRAME = 35

We use n_frames = 900 = 31 500 steps = ~ 158 D/U. The first 50 D/U
are dropped as startup; the FFT window is the remaining ~ 108 D/U,
giving ~ 17 - 20 saturated cycles depending on the realised St.

Outputs:
    data/validation/cylinder_re100_strouhal_lowblockage.json
    (includes cl_history / cd_history time series so future re-extract
    with a different FFT window does not require re-running solve_lbm)

Runtime: **~ 7.4 hours** on a 4-core laptop the first time, which is
much longer than expected. solve_lbm retains every per-frame
snapshot in memory (900 frames x 700 x 400 x 3 channels x float64
~ 5 - 6 GB), and on a 16 GB laptop with other apps open this forces
the OS to swap and the wall-time balloons by ~ 30 x. If you need to
re-run, drop n_frames or refactor solve_lbm to stream snapshots to
disk -- the cl_history extraction itself fits in ~ 250 kB so the
snapshot retention is what kills it.

The committed JSON already carries the per-step time series, so a
re-extract with different windowing does NOT require a fresh bake.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.lbm_render import RESOLUTION_PRESETS, STEPS_PER_FRAME, U_INFLOW, solve_lbm  # noqa: E402

RES_KEY = "Validation (700 x 400)"
SHAPE = "Cylinder"
RE = 100
N_FRAMES = 900  # 31 500 LBM steps

# Reference values for the cross-check table.
WILLIAMSON_ST = 0.166
OPENFOAM_ST = 0.1600

OUT_PATH = (
    _PROJECT_ROOT
    / "data"
    / "validation"
    / "cylinder_re100_strouhal_lowblockage.json"
)


def _strouhal_from_fft(
    cl: np.ndarray,
    steps_per_sample: int,
    u_in: float,
    char_length: float,
    skip_startup_steps: int,
) -> dict:
    """Extract Strouhal from the lift coefficient time series via FFT.

    Returns a dict with the peak frequency in (1 / step), the
    Strouhal number, the FFT bin width (in St units), and the number
    of cycles in the FFT window -- the latter is the standard
    "insufficient record" gate: < 20 cycles means the FFT bin width
    is comparable to typical St-error, and the headline should not
    quote a percent number off the peak.
    """
    cl = np.asarray(cl, dtype=np.float64)
    n_total = len(cl)
    n_skip = max(0, skip_startup_steps // steps_per_sample)
    tail = cl[n_skip:]
    if len(tail) < 4:
        raise ValueError(
            f"Tail too short for FFT: n_total={n_total}, n_skip={n_skip}, "
            f"steps_per_sample={steps_per_sample}, skip_startup_steps="
            f"{skip_startup_steps}. Increase n_frames."
        )

    # Remove the DC + linear trend so the spectral peak is the shedding mode.
    tail_detrended = tail - tail.mean()
    n = len(tail_detrended)
    # Sampling frequency is 1 sample per `steps_per_sample` LBM steps.
    # In dimensionless (D/U) time, dt_per_sample = steps_per_sample * u_in / D.
    dt_per_sample = steps_per_sample * u_in / char_length
    freqs = np.fft.rfftfreq(n, d=dt_per_sample)  # 1 / (D/U) = St directly
    spec = np.abs(np.fft.rfft(tail_detrended))
    spec[0] = 0.0
    peak_idx = int(np.argmax(spec))
    st = float(freqs[peak_idx])

    # FFT bin width in St units (= 1 / total_window_in_D_per_U).
    window_du = n * dt_per_sample
    bin_width_st = 1.0 / window_du if window_du > 0 else float("inf")

    # Cycle count = window length / shedding period.
    n_cycles = st * window_du if st > 0 else 0.0

    return {
        "n_total_samples": int(n_total),
        "n_skip_samples": int(n_skip),
        "n_window_samples": int(n),
        "window_du": float(window_du),
        "dt_per_sample_du": float(dt_per_sample),
        "strouhal": float(st),
        "fft_bin_width_st": float(bin_width_st),
        "strouhal_n_cycles": float(n_cycles),
    }


def main() -> int:
    res_cfg = RESOLUTION_PRESETS[RES_KEY]
    char_length = float(res_cfg["cylinder_D"])
    blockage = char_length / float(res_cfg["Ny"])
    n_steps = N_FRAMES * STEPS_PER_FRAME
    t_end_du = n_steps * U_INFLOW / char_length

    print(f"Running {SHAPE} Re={RE} at {RES_KEY}")
    print(f"  D = {int(char_length)} lattice units, B = {blockage*100:.2f} %")
    print(f"  n_frames = {N_FRAMES}, n_steps = {n_steps}, t_end = {t_end_du:.1f} D/U")
    print(f"  expected wall-time: ~10-15 min on a 4-core laptop")
    print()

    t0 = time.perf_counter()
    out = solve_lbm(
        shape_preset=SHAPE,
        reynolds_target=RE,
        aoa_deg=0.0,
        res_key=RES_KEY,
        n_frames=N_FRAMES,
    )
    wall_time = time.perf_counter() - t0
    print(f"  solve_lbm wall-time: {wall_time:.1f} s")
    print()

    cl_history = np.asarray(out["cl_history"], dtype=np.float64)
    cd_history = np.asarray(out["cd_history"], dtype=np.float64)
    char_length_out = float(out["char_length"])

    # Strouhal: drop the first 50 D/U as the startup transient. At
    # u_in=0.1 / D=20, 50 D/U = 10 000 LBM steps.
    #
    # IMPORTANT: solve_lbm's cl_history has ONE ENTRY PER LBM STEP
    # (verified empirically: len(cl_history) == n_steps, not n_frames).
    # An earlier revision of this script passed steps_per_sample=
    # STEPS_PER_FRAME and recorded St = 0.00513 = the true St / 35.
    # The correct cadence is one sample per step.
    skip_startup_steps = int(50.0 * char_length_out / U_INFLOW)
    st_info = _strouhal_from_fft(
        cl=cl_history,
        steps_per_sample=1,
        u_in=U_INFLOW,
        char_length=char_length_out,
        skip_startup_steps=skip_startup_steps,
    )

    # Cd mean on the SAME tail window so the headline number is
    # window-consistent with Strouhal. cd_history / cl_history are
    # per-step (mirror cl_history above).
    n_skip = skip_startup_steps
    cd_tail = cd_history[n_skip:]
    cd_mean_tail = float(cd_tail.mean())
    cd_std_tail = float(cd_tail.std())
    cl_std_tail = float(cl_history[n_skip:].std())

    err_st_williamson = 100.0 * (st_info["strouhal"] - WILLIAMSON_ST) / WILLIAMSON_ST
    err_st_openfoam = 100.0 * (st_info["strouhal"] - OPENFOAM_ST) / OPENFOAM_ST

    payload = {
        "_provenance": {
            "script": "scripts/validate_2d_cylinder_strouhal_lowblockage.py",
            "purpose": (
                "Close VALIDATION.md sec 8.4 audit item #10: produce an "
                "AeroLab Strouhal at the Validation preset (D=20, B=5%) "
                "with >= 20 shedding cycles, so the three-way table "
                "(AeroLab / OpenFOAM / Williamson) reports St on all "
                "three columns instead of n/a on AeroLab."
            ),
            "shape": SHAPE,
            "reynolds": RE,
            "resolution_preset": RES_KEY,
            "n_frames": N_FRAMES,
            "n_steps": n_steps,
            "char_length": char_length_out,
            "blockage": blockage,
            "u_inflow_lattice": U_INFLOW,
            "wall_time_s": round(wall_time, 1),
            "skip_startup_du": 50.0,
            "skip_startup_steps": skip_startup_steps,
        },
        # cl/cd histories serialized so a future re-extract (different
        # window, different FFT scheme, longer skip) does not require
        # a fresh 7-hour bake. ~ 31 500 float64s each ~ 250 kB / array.
        "time_series": {
            "n_samples_per_step": 1,
            "cl_history": cl_history.tolist(),
            "cd_history": cd_history.tolist(),
        },
        "strouhal_extraction": st_info,
        "cd_tail_window": {
            "n_skip_frames": n_skip_frames,
            "n_tail_frames": int(len(cd_tail)),
            "cd_mean": cd_mean_tail,
            "cd_std": cd_std_tail,
            "cl_std": cl_std_tail,
        },
        "references": {
            "williamson_1996_st": WILLIAMSON_ST,
            "openfoam_v0_6_5_st": OPENFOAM_ST,
        },
        "errors_pct": {
            "st_vs_williamson_pct": round(err_st_williamson, 2),
            "st_vs_openfoam_pct": round(err_st_openfoam, 2),
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # Summary print.
    print("=" * 72)
    print(f"  AeroLab St (Validation preset, D=20, B={blockage*100:.1f} %):")
    print(f"    St = {st_info['strouhal']:.4f}  "
          f"({st_info['strouhal_n_cycles']:.1f} cycles in FFT window)")
    print(f"    FFT bin width = +/- {st_info['fft_bin_width_st']:.4f} in St units")
    print(f"    vs Williamson 1996 (0.166): {err_st_williamson:+.2f} %")
    print(f"    vs OpenFOAM v0.6.5 (0.160): {err_st_openfoam:+.2f} %")
    print(f"  AeroLab Cd_mean (same tail window): {cd_mean_tail:.4f}")
    print("=" * 72)
    print(f"\nwrote {OUT_PATH.relative_to(_PROJECT_ROOT)}")

    if st_info["strouhal_n_cycles"] < 20:
        print(
            f"\nWARNING: only {st_info['strouhal_n_cycles']:.1f} cycles in the "
            "FFT window -- the auditor's '>= 20 cycles' bar is not met. "
            "Increase N_FRAMES at the top of this script and re-run."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
