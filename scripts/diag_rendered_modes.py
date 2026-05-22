"""Save frame 25 of the full render pipeline for each viz mode on a
cylinder + square + NACA, so we can see exactly what ships in the GIF
after the W6.1 Pressure (temporal-average) + Velocity (bipolar) fixes.

Output: data/render_<shape>_<mode>_frame25.png
"""
import io
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lbm_render import simulate_and_render

OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)


def grab(shape, re, aoa, mode, n_frames=51, label=None):
    label = label or f"{shape}_Re{re}_AoA{int(aoa)}_{mode}".replace(" ", "")
    print(f"running {label} ...", flush=True)
    out = simulate_and_render(
        shape, re, aoa, "Standard (320 x 80)",
        n_frames=n_frames, viz_mode=mode,
    )
    gif = Image.open(io.BytesIO(out["gif_bytes"]))
    gif.seek(min(25, n_frames - 1))
    frame = gif.convert("RGB")
    fp = OUT / f"render_{label}.png"
    frame.save(fp)
    print(f"  saved {fp}")


if __name__ == "__main__":
    for mode in ("Vorticity", "Velocity", "Pressure"):
        grab("Cylinder", 200, 0.0, mode)
    for mode in ("Velocity", "Pressure"):
        grab("Square", 200, 0.0, mode)
    for mode in ("Velocity", "Pressure"):
        grab("NACA 4412", 600, 15.0, mode)
