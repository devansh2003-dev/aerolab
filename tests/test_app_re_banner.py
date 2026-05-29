"""Drift gate for the 2D Re-validity thresholds surfaced in the UI.

The 2D-LBM validation envelope is anchored at Re = 200 (Williamson 1996
mode-A 3D-instability threshold for the cylinder). The Re-banner above
the velocity slider and the inline validity pill in the slider caption
both colour-tier the current Re against these thresholds; VALIDATION.md
quotes the same 200 boundary in its headline.

The three constants are defined module-level in `app.py` so this test
can find them without importing Streamlit (the import would trigger a
`st.set_page_config` call outside a ScriptRunContext and fail in the
test runner). We parse the source instead.

If anyone moves the thresholds without also revising VALIDATION.md
§headline + the colour-tier copy in the banner, this drift gate fails.
"""
from __future__ import annotations

import ast
from pathlib import Path


def _module_level_int_assignments(source: str) -> dict[str, int]:
    """Return {name: value} for top-level `NAME = <int literal>` lines."""
    tree = ast.parse(source)
    found: dict[str, int] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        # `RE_UNPHYSICAL_2D = RE_EXPLORATORY_MAX` is an alias, not an int
        # literal -- resolve it after the literal pass.
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, int):
            found[target.id] = node.value.value
        elif isinstance(node.value, ast.Name) and node.value.id in found:
            found[target.id] = found[node.value.id]
    return found


def test_re_thresholds_present_and_correct():
    src = (Path(__file__).resolve().parents[1] / "app.py").read_text(
        encoding="utf-8"
    )
    assigns = _module_level_int_assignments(src)

    assert assigns.get("RE_VALIDATED_MAX") == 200, (
        "RE_VALIDATED_MAX must be 200 (Williamson 1996 mode-A 3D-instability "
        "threshold; VALIDATION.md headline anchors here)."
    )
    assert assigns.get("RE_EXPLORATORY_MAX") == 800, (
        "RE_EXPLORATORY_MAX must be 800 (rough envelope above which a 2D "
        "solver no longer produces visually plausible wakes)."
    )
    assert assigns.get("RE_UNPHYSICAL_2D") == 800, (
        "RE_UNPHYSICAL_2D must alias RE_EXPLORATORY_MAX (== 800)."
    )


def test_banner_text_cites_williamson_and_okajima():
    """The Re <= 200 banner names both reference papers so a screenshot
    of the green-banner state stands on its own."""
    src = (Path(__file__).resolve().parents[1] / "app.py").read_text(
        encoding="utf-8"
    )
    assert "Williamson 1996" in src
    assert "Okajima 1982" in src
