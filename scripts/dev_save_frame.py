"""Save a single canonical frame from simulate_and_render as a PNG.

Bridges the visual-QA gap: an AI agent / reviewer who can't watch
animated GIFs can still inspect a static PNG via the Read tool (which
renders images inline) or by opening the file. Captures middle-of-loop
frames where the wake is fully developed.

Two modes:

  * default     -- run a fixed canonical config (cylinder Re=400 AoA=0
                   Standard) and save frame 50 as data/inspect_canonical.png.
                   Use this as the baseline for "did anything visually
                   change?" checks.
  * --shape X --re N --aoa A --res K --frame F
                   override any of the above. Use this when investigating
                   a specific shape/regime regression.

Usage::

    python scripts/dev_save_frame.py
    python scripts/dev_save_frame.py --shape Ellipse --re 1000 --aoa 30 --frame 100
"""
import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from src.lbm_render import RESOLUTION_PRESETS, simulate_and_render

DEFAULT_CONFIG = dict(
    shape="Cylinder",
    re=400,
    aoa=0.0,
    res="Standard (320 x 100)",
    frame=50,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape", default=DEFAULT_CONFIG["shape"],
                   choices=["Cylinder", "Square", "Ellipse", "NACA 0012", "NACA 4412"])
    p.add_argument("--re", type=float, default=DEFAULT_CONFIG["re"])
    p.add_argument("--aoa", type=float, default=DEFAULT_CONFIG["aoa"])
    p.add_argument("--res", default=DEFAULT_CONFIG["res"],
                   choices=list(RESOLUTION_PRESETS.keys()))
    p.add_argument("--frame", type=int, default=DEFAULT_CONFIG["frame"],
                   help="Which frame index to extract (0 = first, default 50).")
    p.add_argument("--n_frames", type=int, default=None,
                   help="Total frames to simulate (defaults to preset value). "
                        "Reduce to speed up if you only need an early frame.")
    p.add_argument("--out", default=None,
                   help="Output PNG path. Defaults to data/inspect_<shape>.png "
                        "(canonical) or data/inspect_<shape>_re<re>_aoa<aoa>.png "
                        "(non-canonical).")
    args = p.parse_args()

    is_canonical = (
        args.shape == DEFAULT_CONFIG["shape"]
        and args.re == DEFAULT_CONFIG["re"]
        and args.aoa == DEFAULT_CONFIG["aoa"]
        and args.res == DEFAULT_CONFIG["res"]
        and args.frame == DEFAULT_CONFIG["frame"]
    )
    if args.out:
        out_path = Path(args.out)
    elif is_canonical:
        out_path = Path("data/inspect_canonical.png")
    else:
        slug = (
            f"{args.shape.lower().replace(' ', '_')}"
            f"_re{int(args.re)}_aoa{int(args.aoa)}_f{args.frame}"
        )
        out_path = Path(f"data/inspect_{slug}.png")
    out_path.parent.mkdir(exist_ok=True)

    n_frames = args.n_frames if args.n_frames is not None else args.frame + 1
    if n_frames < args.frame + 1:
        raise SystemExit(f"--n_frames {n_frames} < frame index {args.frame} + 1")

    print(f"Simulating {args.shape} Re={args.re} AoA={args.aoa}° res={args.res}")
    print(f"  ({n_frames} frames; extracting frame {args.frame})")
    out = simulate_and_render(
        args.shape, args.re, args.aoa, args.res, n_frames=n_frames,
    )

    gif = Image.open(io.BytesIO(out["gif_bytes"]))
    gif.seek(args.frame)
    rgb = gif.convert("RGB")
    rgb.save(out_path, format="PNG", optimize=True)
    print(f"Saved frame {args.frame} ({rgb.size[0]}x{rgb.size[1]}) -> {out_path}")
    print(f"  ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
