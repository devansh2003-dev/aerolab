"""Single-source-of-truth gate for headline validation numbers.

Validation numbers appear in three places:
  * data/validation/results.json -- machine-readable, authored by
    `scripts/validate_solver.py`. This is the source of truth.
  * README.md headline table
  * VALIDATION.md tolerance + per-case tables

This test recomputes the headline aggregate stats (median / max abs
percent error per shape) from results.json and asserts that they
match the numbers quoted in README and VALIDATION. If they drift,
the test fails with the exact substitution needed -- so re-running
the validation sweep automatically gates README freshness.

External reviewer flagged this on 2026-05-24 as the highest-leverage
fix: prevents future "stale headline numbers" drift permanently.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_JSON = ROOT / "data" / "validation" / "results.json"
README = ROOT / "README.md"
VALIDATION = ROOT / "VALIDATION.md"


# Documented validated-band cutoffs (VALIDATION.md §3.3).
# Cases below these Re values are at the shedding-onset boundary where
# the Allen-Vincenti correction over-corrects (wake still attached or
# weakly shedding) -- they appear in the per-case table but are
# excluded from the headline aggregate so the median isn't dragged by
# an out-of-regime correction artifact.
VALIDATED_BAND_RE_MIN = {"Cylinder": 100, "Square": 150}


def _abs_errs(results, shape, key):
    """Sorted abs(percent error) list for one shape + metric, restricted
    to the documented validated band per shape."""
    re_min = VALIDATED_BAND_RE_MIN[shape]
    return sorted(
        abs(r[key]) for r in results
        if r["shape"] == shape
        and r["re"] >= re_min
        and r.get(key) is not None
    )


def _stats(errs):
    """Return (median, max) of abs error list, both as floats."""
    if not errs:
        return float("nan"), float("nan")
    return float(median(errs)), float(max(errs))


def _round1(x):
    """Match the 1-decimal display format used in README / VALIDATION."""
    return round(x, 1)


def _load_stats():
    """Recompute headline aggregates from results.json."""
    data = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    results = data["results"]
    cyl_cd_med, cyl_cd_max = _stats(_abs_errs(results, "Cylinder", "cd_error_pct"))
    sqr_cd_med, sqr_cd_max = _stats(_abs_errs(results, "Square", "cd_error_pct"))
    cyl_st_med, cyl_st_max = _stats(_abs_errs(results, "Cylinder", "st_error_pct"))
    return {
        "cylinder_cd_median": _round1(cyl_cd_med),
        "cylinder_cd_max":    _round1(cyl_cd_max),
        "square_cd_median":   _round1(sqr_cd_med),
        "square_cd_max":      _round1(sqr_cd_max),
        "cylinder_st_median": _round1(cyl_st_med),
        "cylinder_st_max":    _round1(cyl_st_max),
    }


# Headline row regex: matches the markdown table cells in README /
# VALIDATION. Captures (median, max) as floats. The tolerance band cell
# is allowed but not consumed -- it lives further along the row and is
# audited separately via the tolerance-band tests in
# tests/test_validation_benchmark.py.
_ROW = re.compile(
    r"\|\s*{label}\s*\|"                  # row label
    r"\s*\**\s*(\d+\.\d+)\s*%\**\s*\|"    # median %
    r"\s*\**\s*(\d+\.\d+)\s*%\**\s*\|",   # max %
)


def _find_row(text, label):
    """Return (median, max) declared in a markdown table row labelled
    ``label`` (e.g. 'Cylinder Cd'). Raises AssertionError if not
    found -- means the headline table is missing or mis-formatted."""
    rx = re.compile(_ROW.pattern.format(label=re.escape(label)))
    m = rx.search(text)
    assert m, (
        f"Could not parse {label!r} headline row. Expected a markdown row "
        f"like `| {label} | **X.Y %** | **Z.W %** |...`."
    )
    return float(m.group(1)), float(m.group(2))


def _assert_match(declared, computed, where, label, kind, tol=0.2):
    """Allow 0.2 pp slack for rounding (results.json holds full precision,
    README shows one decimal -- so 11.55 -> 11.6 vs 11.54 -> 11.5 should
    not trip an error)."""
    assert abs(declared - computed) <= tol, (
        f"{where} declares {label} {kind} = {declared} %, but "
        f"data/validation/results.json computes {computed} %. "
        f"Re-run `python scripts/validate_solver.py` and update "
        f"{where} to match, or vice versa."
    )


def test_results_json_exists():
    """The headline JSON must exist -- everything below depends on it."""
    assert RESULTS_JSON.exists(), (
        f"{RESULTS_JSON.relative_to(ROOT)} not found. Re-run "
        f"`python scripts/validate_solver.py` to regenerate it."
    )


def test_readme_cylinder_cd_matches_results_json():
    stats = _load_stats()
    declared = _find_row(README.read_text(encoding="utf-8"), "Cylinder Cd")
    _assert_match(declared[0], stats["cylinder_cd_median"], "README.md",
                  "Cylinder Cd", "median")
    _assert_match(declared[1], stats["cylinder_cd_max"], "README.md",
                  "Cylinder Cd", "max")


def test_readme_square_cd_matches_results_json():
    stats = _load_stats()
    declared = _find_row(README.read_text(encoding="utf-8"), "Square Cd")
    _assert_match(declared[0], stats["square_cd_median"], "README.md",
                  "Square Cd", "median")
    _assert_match(declared[1], stats["square_cd_max"], "README.md",
                  "Square Cd", "max")


def test_readme_cylinder_st_matches_results_json():
    stats = _load_stats()
    declared = _find_row(README.read_text(encoding="utf-8"), "Cylinder St")
    _assert_match(declared[0], stats["cylinder_st_median"], "README.md",
                  "Cylinder St", "median")
    _assert_match(declared[1], stats["cylinder_st_max"], "README.md",
                  "Cylinder St", "max")


def test_validation_tolerance_table_matches_results_json():
    """VALIDATION.md §2.4 quotes the SAME max-error figures as part of
    its tolerance-band justification. They must match results.json too."""
    stats = _load_stats()
    text = VALIDATION.read_text(encoding="utf-8")

    # Each row's justification cell contains the "max measured ..." figure.
    # Rows are single-line so .*? is bounded; no need for DOTALL.
    for label, key in (
        ("Cylinder Cd", "cylinder_cd_max"),
        ("Square Cd",   "square_cd_max"),
        ("Cylinder St", "cylinder_st_max"),
    ):
        m = re.search(
            rf"{re.escape(label)}.*?max measured(?: error)?\s*(\d+\.\d+)\s*%",
            text,
        )
        assert m, (
            f"VALIDATION.md §2.4 {label} row missing 'max measured' figure. "
            f"Expected something like 'max measured X.Y %'."
        )
        _assert_match(float(m.group(1)), stats[key], "VALIDATION.md",
                      label, "max-measured")
