"""Measure the actual delivered GIF size for each shape x resolution.

Streamlit Cloud's free tier doesn't have a hard bandwidth cap that's
public, but practical considerations apply:

  * a 10 MB GIF takes 10+ s to download on a typical 8 Mbit/s mobile
    connection,
  * Slack / Discord previews break above ~10 MB,
  * caching twelve of these in Streamlit's in-memory @cache_data adds
    up if they're each multi-MB.

This script runs ``simulate_and_render`` for every (shape, resolution)
combination at Re=400 (a moderately busy wake -- more wake structure
means more color diversity means bigger quantized palettes) and prints a
markdown table of sizes.

It writes nothing to disk; the GIF bytes are produced in-memory and
their length is the reported size.

Usage::

    python scripts/dev_measure_gif_size.py

Wall time: ~15-25 min total for all 8 combinations on a modern laptop.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lbm_render import RESOLUTION_PRESETS, simulate_and_render

SHAPES = ["Cylinder", "Ellipse", "Square", "NACA 0012", "NACA 4412"]
RES_KEYS = list(RESOLUTION_PRESETS.keys())
REYNOLDS = 400
AOA = 15


def _human_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


def main():
    rows = []
    print(f"Running {len(SHAPES) * len(RES_KEYS)} combinations at Re={REYNOLDS}, AoA={AOA}°...")
    print()

    for res_key in RES_KEYS:
        for shape in SHAPES:
            aoa_for_shape = 0 if shape == "Cylinder" else AOA
            label = f"{shape:<10} | {res_key}"
            print(f"  {label} ...", end="", flush=True)
            t = time.perf_counter()
            out = simulate_and_render(shape, REYNOLDS, aoa_for_shape, res_key)
            elapsed = time.perf_counter() - t
            gif_bytes = out["gif_bytes"]
            size = len(gif_bytes)
            print(f" {_human_bytes(size):>10}  ({elapsed:.1f} s)")
            rows.append((shape, res_key, size, elapsed))

    print()
    print("| Shape     | Resolution            | GIF size  | Sim time |")
    print("|-----------|-----------------------|-----------|----------|")
    for shape, res_key, size, elapsed in rows:
        print(f"| {shape:<9} | {res_key:<21} | {_human_bytes(size):>9} | {elapsed:>6.1f} s |")
    print()
    biggest = max(rows, key=lambda r: r[2])
    print(f"Largest: {biggest[0]} @ {biggest[1]} = {_human_bytes(biggest[2])}")
    print(f"Total bytes across all 10 GIFs: {_human_bytes(sum(r[2] for r in rows))}")
    print()
    print("Sanity thresholds:")
    print("  * any single GIF > 8 MB  -> consider lowering palette or frame count")
    print("  * Detailed median > 5 MB -> consider raising Standard's frame count")
    print("  * Standard median > 2 MB -> investigate palette diversity in render")


if __name__ == "__main__":
    main()
