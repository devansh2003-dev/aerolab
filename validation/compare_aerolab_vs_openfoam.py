"""Three-way comparison: AeroLab vs OpenFOAM vs Williamson 1996.

Card #6 from the 2026-05-27 reviewer round. Produces
`validation/cross_validation.md` (committed) and prints the same table
to stdout. Reads:

  - `validation/openfoam/cylinder_re100/postProcessing/forceCoeffs/0/forceCoeffs.dat`
    (produced by `./Allrun` in the OpenFOAM case directory)
  - `data/validation/results_lowblockage.json` for AeroLab's Cd at
    cylinder Re = 100 (Validation preset, D = 20, B = 5 %)
  - `src/references.py:CYLINDER_FREESTREAM` for Williamson 1996

The OpenFOAM solve is documented in
`validation/openfoam/cylinder_re100/README.md`. The compare script
is split out from the Allrun driver so it can run in CI off the
committed `forceCoeffs.dat` even when OpenFOAM itself isn't
installed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.references import CYLINDER_FREESTREAM_CD as CYLINDER_FREESTREAM  # noqa: E402

FORCE_COEFFS_PATH = (
    _ROOT / "validation" / "openfoam" / "cylinder_re100"
    / "postProcessing" / "forceCoeffs" / "0" / "forceCoeffs.dat"
)
AEROLAB_RESULTS_PATH = (
    _ROOT / "data" / "validation" / "results_lowblockage.json"
)
OUT_PATH = _ROOT / "validation" / "cross_validation.md"

RE_OF_INTEREST = 100
WINDOW_DU = 50.0  # average Cd over the last 50 D/U so window-equivalent


def _read_openfoam_forcecoeffs(path: Path) -> tuple[list[float], list[float]] | None:
    """Parse OpenFOAM's forceCoeffs.dat into (times, cds). Returns None if
    the file is absent -- typical pre-solve state."""
    if not path.exists():
        return None
    times: list[float] = []
    cds: list[float] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        # forceCoeffs columns vary by OpenFOAM version; the consistent
        # ones across versions are:
        #     time   Cm   Cd   Cl   Cl(f)   Cl(r)
        # We read columns 0 and 2.
        if len(parts) < 3:
            continue
        try:
            times.append(float(parts[0]))
            cds.append(float(parts[2]))
        except ValueError:
            continue
    return times, cds


def _aerolab_cd(re: int) -> tuple[float | None, float | None]:
    if not AEROLAB_RESULTS_PATH.exists():
        return None, None
    data = json.loads(AEROLAB_RESULTS_PATH.read_text(encoding="utf-8"))
    for r in data.get("results", []):
        if r.get("shape") == "Cylinder" and int(r.get("re", -1)) == re:
            return (
                float(r.get("cd_corrected")) if r.get("cd_corrected") is not None else None,
                float(r.get("cd_raw")) if r.get("cd_raw") is not None else None,
            )
    return None, None


def _williamson_cd(re: int) -> float | None:
    # CYLINDER_FREESTREAM_CD is {Re -> Cd_freestream}; scalar value not tuple.
    entry = CYLINDER_FREESTREAM.get(re)
    if entry is None:
        return None
    return float(entry)


def _pct_dev(value: float | None, ref: float | None) -> str:
    if value is None or ref is None:
        return "n/a"
    return f"{100.0 * (value - ref) / ref:+.2f} %"


def main() -> int:
    williamson_cd = _williamson_cd(RE_OF_INTEREST)
    aero_corr, aero_raw = _aerolab_cd(RE_OF_INTEREST)

    of_block = _read_openfoam_forcecoeffs(FORCE_COEFFS_PATH)
    if of_block is None:
        of_cd: float | None = None
        of_note = (
            "OpenFOAM `forceCoeffs.dat` not present "
            f"({FORCE_COEFFS_PATH.relative_to(_ROOT)}). "
            "Run `./Allrun` in the case directory first."
        )
    else:
        times, cds = of_block
        # Average Cd over the last WINDOW_DU. The OpenFOAM time axis is in
        # seconds; t_DU = t_sec * U / D and the case is configured at
        # U = 1, D = 0.01 (Re = 100, nu = 1e-4) -> t_sec = t_DU * 0.01.
        # So the last 50 D/U is the last 0.5 s of the time series.
        if not times:
            of_cd = None
            of_note = "forceCoeffs.dat present but no parseable rows."
        else:
            t_end = max(times)
            t_window_start = t_end - WINDOW_DU * 0.01
            cd_tail = [
                cd for t, cd in zip(times, cds) if t >= t_window_start
            ]
            of_cd = sum(cd_tail) / len(cd_tail) if cd_tail else None
            of_note = (
                f"{len(cd_tail)} samples in last {WINDOW_DU:g} D/U "
                f"(t_end = {t_end:.3f} s)"
            )

    rows = [
        ("AeroLab (Cd corrected, D=20)", aero_corr, williamson_cd),
        ("AeroLab (Cd raw, D=20)",      aero_raw,  williamson_cd),
        ("OpenFOAM pisoFoam (laminar)", of_cd,     williamson_cd),
        ("Williamson 1996 ARFM 28",     williamson_cd, williamson_cd),
    ]

    lines = [
        "# Three-way cross-validation: cylinder Re = 100",
        "",
        "Sources:",
        "",
        f"- AeroLab: `data/validation/results_lowblockage.json` "
        f"(Validation preset, D = 20, B = 5 %)",
        f"- OpenFOAM: `validation/openfoam/cylinder_re100/` "
        f"(pisoFoam, 2D laminar). Notes: {of_note}",
        f"- Williamson 1996 ARFM 28: from `src/references.py:CYLINDER_FREESTREAM`",
        "",
        "| Source | Cd | Deviation from Williamson |",
        "|--------|----|----------------------------|",
    ]
    for label, value, ref in rows:
        val_s = f"{value:.3f}" if value is not None else "n/a"
        dev_s = _pct_dev(value, ref) if "Williamson" not in label else "0 (reference)"
        lines.append(f"| {label} | {val_s} | {dev_s} |")
    lines.append("")
    if of_cd is None:
        lines.append(
            "**Status: pending OpenFOAM solve.** Run "
            "`validation/openfoam/cylinder_re100/Allrun` on a Linux / WSL / "
            "macOS box with OpenFOAM >= 11, then re-run this script."
        )

    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {OUT_PATH.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
