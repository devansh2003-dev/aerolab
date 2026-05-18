"""One-shot helper: regenerate tests/baselines/canonical_cylinder_re400_f50.png.

Reads data/inspect_canonical.png (produced by dev_save_frame.py with the
default canonical config), downscales to 128x40 grayscale, saves as
the baseline that test_visual_regression.py compares against.

Run this script when an intentional visual change lands (e.g. new
colormap, body-patch tweak). Don't run it casually -- the whole point
of the regression test is to flag unintentional drift.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

CANONICAL_PNG = "data/inspect_canonical.png"
BASELINE_PNG = "tests/baselines/canonical_cylinder_re400_f50.png"
BASELINE_SIZE = (128, 40)


def main():
    if not os.path.exists(CANONICAL_PNG):
        print(f"missing {CANONICAL_PNG} -- run scripts/dev_save_frame.py first")
        raise SystemExit(1)
    os.makedirs(os.path.dirname(BASELINE_PNG), exist_ok=True)
    img = Image.open(CANONICAL_PNG).convert("L").resize(BASELINE_SIZE)
    img.save(BASELINE_PNG, optimize=True)
    print(f"baseline written: {BASELINE_PNG}")
    print(f"  size: {os.path.getsize(BASELINE_PNG)} bytes")
    print(f"  shape: {BASELINE_SIZE}")


if __name__ == "__main__":
    main()
