"""Cylinder Cd-vs-Re validation chart: AeroLab points over Williamson 1996.

The "over-the-curve" validation plot in the spirit of the canonical
sphere/cylinder drag charts (cf. Morrison 2013 for spheres). Instead of
a flattering single-config line, this draws the *honest multi-resolution*
story straight from the committed validation JSONs:

  * Williamson 1996 free-stream Cd(Re) as the published reference line.
  * AeroLab Resolved (D = 40, B = 10 %) corrected Cd -- the headline
    configuration (Re = 100, 200, 500), which sits on the reference line
    at Re = 200 (+1.0 %) and diverges past the 2D limit.
  * AeroLab Validation (D = 20, B = 5 %) corrected Cd -- the coarser-grid
    cross-check (Re = 100 - 1000), which visibly drifts above the line as
    Re climbs, showing both the grid sensitivity and the 2D-vs-3D ceiling.
  * A shaded "validated band" at Re <= 200 (Williamson mode-A 3D
    transition), past which a strictly-2D solver is a different problem
    and we no longer claim a percent-error tolerance.

Reads ONLY committed data -- no solver runs. Reference values come from
``src.references.CYLINDER_FREESTREAM_CD`` so the line cannot drift out of
sync with the rest of the validation pipeline.

Usage
-----
    python scripts/plot_cd_vs_re.py

Writes ``data/validation/cylinder_cd_vs_re.png`` (referenced from README
and VALIDATION.md).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless -- no display needed on CI / Windows
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.references import CYLINDER_FREESTREAM_CD  # noqa: E402

_DATA = _ROOT / "data" / "validation"
RESOLVED_JSON = _DATA / "results_resolved.json"
LOWBLOCK_JSON = _DATA / "results_lowblockage.json"
OUT_PNG = _DATA / "cylinder_cd_vs_re.png"

# Physics ceiling: Williamson mode-A 3D instability. A strictly-2D solver
# is structurally a different problem above this Re, so we shade Re <= 200
# as the band where the comparison is honest.
VALIDATED_RE_MAX = 200

# Cap the reference line at the top of the validation sweep so we are not
# extrapolating the published curve past where we measured.
REF_RE_MAX = 1000


def _load_cylinder(json_path: Path) -> tuple[list[float], list[float], list[float]]:
    """Return (Re, Cd_corrected, Cd_error_pct) for the Cylinder rows."""
    rows = json.loads(json_path.read_text())["results"]
    cyl = sorted(
        (r for r in rows if r["shape"] == "Cylinder"),
        key=lambda r: r["re"],
    )
    re = [float(r["re"]) for r in cyl]
    cd = [float(r["cd_corrected"]) for r in cyl]
    err = [float(r["cd_error_pct"]) for r in cyl]
    return re, cd, err


def _reference_curve() -> tuple[np.ndarray, np.ndarray]:
    """Dense Williamson 1996 free-stream Cd(Re) line, Re = 40 - 1000."""
    keys = sorted(k for k in CYLINDER_FREESTREAM_CD if k <= REF_RE_MAX)
    vals = [CYLINDER_FREESTREAM_CD[k] for k in keys]
    re_dense = np.logspace(np.log10(keys[0]), np.log10(keys[-1]), 400)
    cd_dense = np.interp(re_dense, keys, vals)
    return re_dense, cd_dense


def main() -> Path:
    res_re, res_cd, res_err = _load_cylinder(RESOLVED_JSON)
    val_re, val_cd, _ = _load_cylinder(LOWBLOCK_JSON)
    ref_re, ref_cd = _reference_curve()

    fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=150)

    # --- validated band shading (drawn first so it sits behind everything)
    ax.axvspan(
        ref_re[0], VALIDATED_RE_MAX,
        color="#2e7d32", alpha=0.07, zorder=0,
    )
    ax.axvline(VALIDATED_RE_MAX, color="#2e7d32", lw=1.0, ls=":", alpha=0.6, zorder=1)

    # --- published reference line
    ax.plot(
        ref_re, ref_cd,
        color="#222222", lw=2.0, zorder=2,
        label="Williamson 1996  (free-stream $C_d$, experiment)",
    )

    # --- AeroLab Validation (D = 20) -- coarser grid, faded
    ax.plot(
        val_re, val_cd,
        marker="s", ms=7, mfc="white", mec="#c2691f", mew=1.6,
        ls="--", lw=1.2, color="#c2691f", alpha=0.85, zorder=3,
        label="AeroLab — Validation (D=20, B=5%), corrected",
    )

    # --- AeroLab Resolved (D = 40) -- headline
    ax.plot(
        res_re, res_cd,
        marker="o", ms=10, mfc="#1565c0", mec="#0b315e", mew=1.4,
        ls="none", zorder=5,
        label="AeroLab — Resolved (D=40, B=10%), corrected  [headline]",
    )

    # annotate each headline point with its signed error vs Williamson
    for re_v, cd_v, err_v in zip(res_re, res_cd, res_err):
        ax.annotate(
            f"{err_v:+.1f}%",
            xy=(re_v, cd_v),
            xytext=(0, 12), textcoords="offset points",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#0b315e", zorder=6,
        )

    # --- band labels (along the bottom so they clear the legend + data)
    ax.text(
        np.sqrt(ref_re[0] * VALIDATED_RE_MAX), 0.035,
        "validated band  (Re ≤ 200)",
        ha="center", va="bottom", fontsize=9, color="#2e7d32",
        transform=ax.get_xaxis_transform(),
    )
    ax.text(
        np.sqrt(VALIDATED_RE_MAX * REF_RE_MAX), 0.86,
        "2D limit — Williamson mode-A\n3D transition (reported, not claimed)",
        ha="center", va="bottom", fontsize=8.5, color="#777777",
        transform=ax.get_xaxis_transform(),
    )

    # --- axes cosmetics
    ax.set_xscale("log")
    ax.set_xlim(ref_re[0], REF_RE_MAX)
    ax.set_ylim(0.80, 1.70)
    ax.set_xlabel("Reynolds number  Re  =  U·D / ν", fontsize=11)
    ax.set_ylabel("Drag coefficient  $C_d$", fontsize=11)
    ax.set_title(
        "AeroLab cylinder $C_d$ vs Re — verified & validated against "
        "published experiment",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)

    # log-x minor ticks read as bare integers, not powers
    from matplotlib.ticker import ScalarFormatter

    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks([40, 100, 200, 300, 500, 1000])
    ax.set_xticklabels(["40", "100", "200", "300", "500", "1000"])

    fig.text(
        0.5, 0.012,
        "2D D2Q9 MRT-LES Lattice Boltzmann · Allen-Vincenti blockage "
        "correction (K=1.10) · verified vs OpenFOAM 11 at Re=100 (±1.6%)",
        ha="center", va="bottom", fontsize=7.5, color="#666666",
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)

    print(f"Resolved (D=40) cylinder points : "
          + ", ".join(f"Re={int(r)} Cd={c:.3f} ({e:+.1f}%)"
                      for r, c, e in zip(res_re, res_cd, res_err)))
    print(f"Validation (D=20) cylinder points: "
          + ", ".join(f"Re={int(r)} Cd={c:.3f}" for r, c in zip(val_re, val_cd)))
    print(f"-> {OUT_PNG}")
    return OUT_PNG


if __name__ == "__main__":
    main()
