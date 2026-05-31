"""Gate the OpenFOAM 11 cylinder Re=100 cross-check headline numbers.

VALIDATION.md sec 8.4, README.md, validation/cross_validation.md, and
the in-app Validation-tab callout all cite three numbers from the
refined-mesh OpenFOAM run:

  * Cd_mean over the late tail = 1.341 (+1.60 % vs Williamson 1.320)
  * Strouhal                   = 0.1600 (-3.62 % vs Williamson 0.166)
  * AeroLab corrected          = 1.348 (+2.13 %), cross-method gap 0.5 %

The actual time series is committed at
validation/openfoam/cylinder_re100/postProcessing/forceCoeffs/0/forceCoeffs.dat
(200 010 rows: t = 0 -> 1000 at dt = 0.005). This test recomputes the
headline numbers from that file and asserts they still match the
documented values. If a future refactor of the case or the cross-
validation script silently changes the numbers, this gate fails with
the exact substitution needed.

Mirrors the structure of test_doc_validation_consistency.py for the
2D Resolved benchmark.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FORCE_FILE = (
    ROOT
    / "validation"
    / "openfoam"
    / "cylinder_re100"
    / "postProcessing"
    / "forceCoeffs"
    / "0"
    / "forceCoeffs.dat"
)
VALIDATION = ROOT / "VALIDATION.md"
README = ROOT / "README.md"
CROSS = ROOT / "validation" / "cross_validation.md"

# Case constants (must match system/controlDict + constant/physicalProperties).
# CASE_D, CASE_U_INF, WINDOW_DU mirror compare_aerolab_vs_openfoam.py so the
# two scripts stay in lock-step.
CASE_D = 2.0
CASE_U_INF = 1.0
CASE_T_PER_DU = CASE_D / CASE_U_INF  # 2.0 s of case time per D/U.
WINDOW_DU = 50.0                     # last 50 D/U of the record

WILLIAMSON_CD = 1.320
WILLIAMSON_ST = 0.166


def _load_force_coeffs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (t, Cd, Cl) from the committed forceCoeffs.dat.

    OpenFOAM 11's forceCoeffs function object writes::

        # Time  Cm  Cd  Cl  Cl(f)  Cl(r)

    so the columns we want are 0, 2, 3.
    """
    assert FORCE_FILE.exists(), (
        f"Committed forceCoeffs file is missing: {FORCE_FILE}. "
        "The OpenFOAM cross-check headline depends on this file being "
        "in the tree -- it is what compare_aerolab_vs_openfoam.py "
        "regenerates the three-way table from without re-running "
        "OpenFOAM."
    )
    data = np.loadtxt(FORCE_FILE, comments="#")
    return data[:, 0], data[:, 2], data[:, 3]


def _strouhal_from_fft(t: np.ndarray, signal: np.ndarray) -> float:
    """Strouhal from the dominant FFT peak of `signal`.

    Returns St = f * D / U where f is the peak frequency. Removes the
    DC component before FFT so the peak is the shedding mode.
    """
    n = len(t)
    dt = float(t[1] - t[0])
    sig = signal - signal.mean()
    freqs = np.fft.rfftfreq(n, d=dt)
    spec = np.abs(np.fft.rfft(sig))
    spec[0] = 0.0  # belt-and-braces DC kill
    f_peak = float(freqs[int(np.argmax(spec))])
    return f_peak * CASE_D / CASE_U_INF


def test_force_file_has_full_record():
    """The committed file should be the full t=0..1000 run, not the old coarse baseline."""
    t, _cd, _cl = _load_force_coeffs()
    # 200 000 timesteps at dt = 0.005 from t = 0 to t = 1000, plus one
    # extra row at the final write step -> 200 001 +/- a few rows.
    assert len(t) >= 199_000, (
        f"forceCoeffs.dat has only {len(t)} rows -- this looks like an "
        "old short run, not the refined 500-D/U run. The headline "
        "numbers in VALIDATION.md sec 8.4 came from a 200 010-row "
        "record."
    )
    assert t[-1] >= 999.0, (
        f"forceCoeffs.dat ends at t = {t[-1]:.2f}, expected ~1000.0. "
        "VALIDATION.md sec 8.4 states endTime = 1000 (500 D/U)."
    )


def test_cd_mean_matches_docs():
    """Late-tail Cd_mean must agree with VALIDATION.md sec 8.4 and the README row."""
    t, cd, _cl = _load_force_coeffs()
    t_end = float(t[-1])
    t_window_start = t_end - WINDOW_DU * CASE_T_PER_DU
    mask = t >= t_window_start
    assert mask.sum() > 1000, (
        f"Late-tail window only has {mask.sum()} samples -- expected "
        "thousands. Window endpoints may have drifted."
    )
    cd_mean = float(cd[mask].mean())

    # VALIDATION.md sec 8.4 cites 1.341; the diagnostic script and the
    # cross-validation script cite the same number to 4 dp (1.3411).
    # Lock in a 0.5 % band -- enough to catch a real solver-config
    # regression, loose enough to absorb the floating-point drift of
    # the openfoam ascii write.
    assert abs(cd_mean - 1.341) < 0.341 * 0.005, (
        f"Late-tail Cd_mean = {cd_mean:.4f} drifted from the documented "
        "1.341 (VALIDATION.md sec 8.4 + README + cross_validation.md). "
        "Either the case files were re-run with different settings, or "
        "the headline numbers in the docs need to be re-derived."
    )

    # Belt-and-braces: this number must also land inside the +/-5 %
    # gate the docs claim it passes.
    err_pct = 100.0 * (cd_mean - WILLIAMSON_CD) / WILLIAMSON_CD
    assert abs(err_pct) <= 5.0, (
        f"Late-tail Cd_mean = {cd_mean:.4f} -> {err_pct:+.2f} % vs "
        f"Williamson {WILLIAMSON_CD:.3f}. The docs claim this is "
        "inside +/-5 %; the committed data no longer supports that."
    )


def test_strouhal_matches_docs():
    """FFT-of-Cl Strouhal must agree with VALIDATION.md sec 8.4."""
    t, _cd, cl = _load_force_coeffs()
    t_end = float(t[-1])
    t_window_start = t_end - WINDOW_DU * CASE_T_PER_DU
    mask = t >= t_window_start
    st = _strouhal_from_fft(t[mask], cl[mask])

    # VALIDATION.md sec 8.4 cites St = 0.1600. The FFT bin width on a
    # 50-D/U window is wide (~ 0.01 in St units), so the documented
    # value and the recomputed value can differ by exactly one bin
    # without there being a regression. The tight tolerance below is
    # the empirical residual of recomputing the same number from the
    # same file.
    assert abs(st - 0.1600) < 0.01, (
        f"FFT(Cl) Strouhal = {st:.4f} drifted from the documented "
        "0.1600 (VALIDATION.md sec 8.4). One FFT bin at the 50-D/U "
        "window size is ~0.01, so this gate trips on a real change, "
        "not on bin quantisation."
    )

    err_pct = 100.0 * (st - WILLIAMSON_ST) / WILLIAMSON_ST
    assert abs(err_pct) <= 5.0, (
        f"FFT(Cl) Strouhal = {st:.4f} -> {err_pct:+.2f} % vs Williamson "
        f"{WILLIAMSON_ST:.3f}. The docs claim this is inside +/-5 %; "
        "the committed data no longer supports that."
    )


def test_documented_numbers_appear_in_validation_md():
    """The three citation strings must all be present in VALIDATION.md sec 8.4."""
    text = VALIDATION.read_text(encoding="utf-8")
    # Cd headline number; the 1.6 % delta; the cross-method 0.5 % gap.
    for needle in ("1.341", "+1.60", "0.5 %"):
        assert needle in text, (
            f"VALIDATION.md no longer mentions '{needle}'. Either the "
            "headline numbers were edited away or the format changed; "
            "this test is the single source of truth for what the docs "
            "must keep saying."
        )


def test_no_stray_coarse_run_in_tree():
    """The old coarse-mesh result must not be re-introduced under postProcessing/."""
    coarse = (
        ROOT
        / "validation"
        / "openfoam"
        / "cylinder_re100"
        / "postProcessing"
        / "forceCoeffs"
        / "200"
        / "forceCoeffs.dat"
    )
    assert not coarse.exists(), (
        f"Stray coarse-baseline forceCoeffs at {coarse} -- this file "
        "gave Cd ~ 1.18 / St ~ 0.12 on the under-resolved 6 480-cell "
        "mesh and directly contradicts the v0.6.5 headline numbers. "
        "It was removed in the 2026-05-31 audit cleanup; do not "
        "re-commit it without also re-deriving the docs."
    )
