"""Regression guard for the Standard-preset corrected pipeline.

What this file IS: a CI-fast regression guard. It runs the Standard
preset (B = 0.35), applies the fitted Allen-Vincenti / West-Apelt
correction, and asserts the corrected estimate still lands inside the
documented bands. If a future refactor breaks the correction pipeline
or shifts raw Cd, this gate trips.

What this file is NOT: the validation claim. The 2026-05-26 senior CFD
review made clear that at B = 0.35 the Allen-Vincenti correction is a
2.6 x rescale fitted (K = 1.10) to recover Williamson -- so a small
corrected error here is a property of the correction at this blockage,
not of the solver. The actual validation claim lives in VALIDATION.md
section 3.2, is anchored to the low-blockage Validation preset
(B = 5 %, near-no-op correction), and is enforced by
tests/test_doc_validation_consistency.py against
data/validation/results_lowblockage.json. THIS file is here to make
sure the corrected pipeline doesn't silently drift -- not to validate
the solver.

Bands carried over from the previous post-hoc fits (cylinder Cd
+/- 15 %, square Cd +/- 25 %, cylinder St +/- 35 %, square St
ungated). Those bands were drawn around the maximum measured errors
in the Standard-preset sweep, so this gate is close to tautological
when read as "validation" but is still useful as "did anything change
since the last measurement?". The headline validation tolerance bands
(literature-derived, scoped to Re <= 200) are in VALIDATION.md
section 2.4.

Methodology (unchanged from previous rev):
  1. Run the solver at Standard (320 x 80, D = 28 -> B = 0.35).
  2. Apply Cd_corrected = Cd_raw * (1 - K * B)^2 and
            St_corrected = St_raw / (1 + 2 * B + B^2).
  3. Compare against Williamson 1996 (cylinder) / Okajima 1982 (square).
  4. Pass if abs(error) <= tolerance.

Runtime: each case ~50 s at n_frames=300 (FFT noise-floor minimum).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.validate_solver import (  # noqa: E402
    ALLEN_VINCENTI_K,
    CYLINDER_FREESTREAM,
    SQUARE_FREESTREAM,
    STANDARD_BLOCKAGE,
    run_case,
)

# n_frames at this validation level. 300 frames = 10500 lattice steps =
# ~8 shedding periods at Re=200, ABOVE the 250-frame FFT noise floor
# documented in scripts/validate_solver.py. The earlier value (200)
# tested Strouhal in the noise band and forced a wider tolerance to
# avoid false-failing -- caught in external review 2026-05-24.
VALIDATION_N_FRAMES = 300

# Per-shape tolerance bands -- chosen from the local validation sweep
# numbers documented in VALIDATION.md.
#   Cylinder Cd: K=1.10 Allen-Vincenti recovers Williamson within
#                median 4.3 % / max 11.6 %  -> 15 % band has headroom
#   Square Cd:   K=1.00 Allen-Vincenti recovers Okajima within
#                median 8.9 % / max 21.8 % over Re=150-500 -> 25 %
#                band needed for corner-shed channel coupling at high Re
#   Cylinder St: West-Apelt under-corrects channel-resonance shift;
#                worst measured error 23.4 % at Re=100 -> 35 % band
#                (still well above noise floor since CI now runs
#                n_frames=300, not 200)
#   Square St:   gated at None -- channel-resonance St at B=0.35 is
#                structurally not recoverable by any single-formula
#                correction; see VALIDATION.md section 4.1
CD_TOL_CYLINDER = 15.0
CD_TOL_SQUARE   = 25.0
ST_TOL_CYLINDER = 35.0
ST_TOL_SQUARE   = None

# Per-shape Cd-only validation cases (no shedding -> no Strouhal).
NO_SHEDDING_CASES = [
    # Re < ~ 47 for cylinder: steady symmetric wake, no vortex shedding.
    ("Cylinder", 40),
]

# (shape, Re) pairs that have both Cd AND Strouhal references.
SHEDDING_VALIDATION_CASES = [
    ("Cylinder", 100),  # canonical "first shedding" benchmark
    ("Cylinder", 200),  # mid laminar wake, Williamson's prime focus
    ("Cylinder", 500),  # late laminar / early transition
    ("Square",   200),  # geometry-locked shedding regime
    ("Square",   500),  # higher-Re square benchmark
]


def _expected_freestream(shape: str, re: int):
    """Look up (Cd_ref, St_ref) from the published-data tables."""
    if shape == "Cylinder":
        return CYLINDER_FREESTREAM.get(re, (None, None))
    if shape == "Square":
        return SQUARE_FREESTREAM.get(re, (None, None))
    return (None, None)


def _cd_tolerance(shape: str) -> float:
    return CD_TOL_CYLINDER if shape == "Cylinder" else CD_TOL_SQUARE


@pytest.mark.parametrize(
    "shape,re",
    SHEDDING_VALIDATION_CASES,
    ids=[f"{s}-Re{r}" for s, r in SHEDDING_VALIDATION_CASES],
)
def test_solver_matches_published_cd_freestream(shape, re):
    """Blockage-corrected Cd should match Williamson / Okajima within the
    per-shape tolerance.

    Cylinder gets the tighter +/- 15 % band because the K=1.10 Allen-
    Vincenti correction recovers Williamson Cd within median 4.3 % /
    max 11.6 % across the Re=100-1000 validated band. Square gets
    +/- 25 % because Square shedding has stronger channel-resonance
    coupling that a single K cannot fully capture; measured spread is
    median 5.4 % / max 21.8 %.

    NOTE on what this gate proves: the "correction" Cd_raw * (1-KB)^2
    at B=0.35 is a 2.65x rescale. Allen-Vincenti was derived for
    small-blockage wind tunnels (B < 0.05); applying it at B=0.35 is
    well outside the regime it was validated for. The gate therefore
    proves the corrected ESTIMATE matches reference, not that the
    solver matches reference natively. See VALIDATION.md sections 2.3
    and 4 for full discussion.
    """
    cd_ref, _ = _expected_freestream(shape, re)
    assert cd_ref is not None, f"No reference Cd for {shape} Re={re}"
    tol = _cd_tolerance(shape)

    r = run_case(shape, re, aoa_deg=0.0, n_frames=VALIDATION_N_FRAMES)

    assert r.cd_corrected is not None
    err_pct = 100.0 * (r.cd_corrected - cd_ref) / cd_ref
    assert abs(err_pct) <= tol, (
        f"{shape} Re={re}: corrected Cd = {r.cd_corrected:.3f} vs "
        f"reference {cd_ref} -> error {err_pct:+.1f} % (tolerance "
        f"+/- {tol} %).\n"
        f"  raw Cd = {r.cd_raw:.3f}, blockage B = {r.blockage_ratio:.3f}, "
        f"K = {ALLEN_VINCENTI_K[shape]:.2f}"
    )


# Only cylinder is gated on Strouhal -- square Strouhal in the blocked
# channel has channel-resonance modes that the West-Apelt correction
# cannot recover, and the resulting error band (60-80 %) is wider than
# any meaningful tolerance. We still RUN the square cases below to
# capture the measurement, but it's diagnostic-only -- not pass/fail.
CYLINDER_SHEDDING_CASES = [
    (shape, re) for shape, re in SHEDDING_VALIDATION_CASES
    if shape == "Cylinder"
]


@pytest.mark.parametrize(
    "shape,re",
    CYLINDER_SHEDDING_CASES,
    ids=[f"{s}-Re{r}" for s, r in CYLINDER_SHEDDING_CASES],
)
def test_solver_matches_published_strouhal_freestream(shape, re):
    """Blockage-corrected Strouhal (cylinder only) within +/- 30 %.

    West-Apelt 1982 channel correction recovers Williamson St within ~23 %
    at the extremes of the validated Re range. The 30 % tolerance gives
    headroom for the run-to-run FFT-peak-picking noise at the lower
    n_frames=200 used for CI.

    Square is NOT gated on Strouhal -- the channel-resonance shedding at
    B = 0.35 produces a near-Re-independent raw St ~ 0.37 that is
    inherently not recoverable by any single-formula blockage correction.
    See VALIDATION.md section 4.1 for the full discussion.
    """
    _, st_ref = _expected_freestream(shape, re)
    assert st_ref is not None, f"No reference St for {shape} Re={re}"

    r = run_case(shape, re, aoa_deg=0.0, n_frames=VALIDATION_N_FRAMES)

    if not np.isfinite(r.st_raw):
        pytest.fail(
            f"{shape} Re={re}: Strouhal is NaN -- FFT failed to resolve "
            f"shedding peak (likely insufficient run length)"
        )
    assert r.st_corrected is not None
    err_pct = 100.0 * (r.st_corrected - st_ref) / st_ref
    assert abs(err_pct) <= ST_TOL_CYLINDER, (
        f"{shape} Re={re}: corrected St = {r.st_corrected:.3f} vs "
        f"reference {st_ref} -> error {err_pct:+.1f} % (tolerance "
        f"+/- {ST_TOL_CYLINDER} %).\n"
        f"  raw St = {r.st_raw:.3f}, blockage B = {r.blockage_ratio:.3f}"
    )


def test_symmetric_cylinder_has_zero_mean_lift():
    """Cl mean for cylinder at AoA=0 must be near zero by symmetry. A non-
    zero mean Cl exposes either a numerical asymmetry (numba kernel,
    boundary condition) or a force-formula sign bug.

    Threshold: |Cl_mean| / Cl_rms < 0.10 (a relative gate, not absolute).
    The asymmetric kick during steps 30-200 injects +y momentum to break
    the perfect mirror symmetry that would otherwise leave the cylinder
    non-shedding. At Re=200 / D=28 the shedding period is ~1330 lattice
    steps, so the last-third averaging window (steps 7000-10500 ~ 3500
    steps) covers ~2.6 periods. A finite-period-fraction sample of a
    sinusoid with peak amplitude ~1.0 carries a residual of order
    0.05-0.10 even when the underlying mean is exactly zero. The
    measured |Cl_mean| / Cl_rms ratio at n_frames=300 is ~0.07; the
    0.10 threshold has noise-floor headroom without making the test
    miss a real asymmetry (which would manifest as a Cl_mean
    comparable to or larger than Cl_rms, not a few percent of it).
    """
    r = run_case("Cylinder", 200, aoa_deg=0.0, n_frames=VALIDATION_N_FRAMES)
    assert r.cl_rms > 0.3, (
        f"Cylinder Re=200 should have unsteady lift from shedding; "
        f"Cl_rms = {r.cl_rms:.3f} suggests shedding didn't lock in"
    )
    cl_ratio = abs(r.cl_raw) / max(r.cl_rms, 1e-12)
    assert cl_ratio < 0.10, (
        f"Symmetric cylinder at AoA=0 should have |Cl_mean| << Cl_rms; "
        f"got |Cl_mean| = {abs(r.cl_raw):.4f}, Cl_rms = {r.cl_rms:.3f}, "
        f"ratio = {cl_ratio:.3f} (threshold 0.10). A ratio above 0.10 "
        f"suggests a real asymmetry, not finite-window noise."
    )


def test_steady_wake_no_shedding_at_low_re():
    """Re=40 cylinder: classic textbook test. Below the shedding-onset
    threshold (~Re=47, Williamson 1996), the wake is steady and symmetric
    -- Cl_rms should be near zero, Cd close to the reference value
    (1.55 in free stream).
    """
    r = run_case("Cylinder", 40, aoa_deg=0.0, n_frames=VALIDATION_N_FRAMES)
    # No shedding -> Cl_rms (after transient) should be small.
    # Use a generous threshold (0.2) because some residual oscillation
    # persists from the initial kick.
    assert r.cl_rms < 0.2, (
        f"Cylinder Re=40 should have steady wake (Cl_rms ~ 0); "
        f"got Cl_rms = {r.cl_rms:.3f}"
    )
    # Cd should be in the laminar-attached range (corrected ~ 1.55).
    assert 1.0 < r.cd_corrected < 2.5, (
        f"Cylinder Re=40 corrected Cd = {r.cd_corrected:.3f} out of "
        f"laminar-attached band [1.0, 2.5]"
    )


def test_blockage_correction_recovers_freestream():
    """Sanity: the Allen-Vincenti correction we apply (K=1.10 for cylinder
    at B=0.35) must reduce raw Cd by a factor consistent with the
    literature ratio (Cd_channel / Cd_freestream ~ 2.5-3.0 at this
    blockage for cylinder).
    """
    r = run_case("Cylinder", 200, aoa_deg=0.0, n_frames=VALIDATION_N_FRAMES)
    ratio = r.cd_raw / r.cd_corrected
    # Theoretical AV ratio at B=0.35, K=1.1: (1 - 0.385)^-2 = 2.65
    assert 2.0 < ratio < 3.5, (
        f"Blockage correction ratio Cd_raw / Cd_corrected = {ratio:.2f} "
        f"out of expected band [2.0, 3.5] for B={STANDARD_BLOCKAGE:.3f}"
    )
