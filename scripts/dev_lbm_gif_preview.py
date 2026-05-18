"""Dev preview: render LBM GIFs for all 5 presets outside Streamlit.

Mirrors the simulate+render+GIF pipeline used in the Streamlit LBM mode
(app.py). Writes:

    data/lbm_preview_cylinder.gif
    data/lbm_preview_square.gif
    data/lbm_preview_ellipse.gif
    data/lbm_preview_naca_0012.gif
    data/lbm_preview_naca_4412.gif

Plus a vorticity colorbar PNG and a speed colorbar PNG per preset
(data/lbm_preview_<name>_vort_cbar.png and ..._speed_cbar.png).

Hot loop is hand-tuned for speed:
  - single Figure reused across frames (no per-frame create/close)
  - imshow.set_data() instead of pcolormesh
  - canvas.buffer_rgba() instead of savefig() -- no PNG round-trip
  - global v_clip from final frame (consistent color scale across animation)
  - smooth analytic body outline overlaid on top of the voxelized mask
  - bidirectional streamline integration with wake seeds so vortex spirals
    show up instead of being missed by inflow-only seeding

Run from project root:
    python scripts/dev_lbm_gif_preview.py
"""
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.patches import Polygon
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = [
    "Segoe UI Variable", "Segoe UI", "Inter", "SF Pro Text",
    "Helvetica Neue", "Arial", "DejaVu Sans",
]

from src.lbm import CS2, equilibrium, macroscopic, step_njit_mrt_with_force
from src.shapes import (
    cylinder_mask, ellipse_mask, naca4_airfoil_mask, naca4_outline_xy,
    square_mask,
)

# --- Constants (must match app.py LBM branch) ---
LBM_NX, LBM_NY = 320, 100
WARMUP_STEPS = 1500
N_FRAMES = 75
STEPS_PER_FRAME = 40
N_STEPS = WARMUP_STEPS + N_FRAMES * STEPS_PER_FRAME
U_INFLOW = 0.1
BODY_X = 70
CY_CENTER = LBM_NY // 2

INFLOW_DIRS = np.array([1, 5, 8], dtype=np.int32)
OUTFLOW_DIRS = np.array([3, 6, 7], dtype=np.int32)

KICK_START, KICK_END = 100, 500
KICK_AMPLITUDE = 0.005
KICK_Y_OFFSET = 2

BG_COLOR = "#0a0a0a"
BODY_COLOR = "#1f2937"
BODY_OUTLINE_MARGIN = 1.5
STREAMLINE_WIDTH = 1.4
SPEED_CMAP = LinearSegmentedColormap.from_list(
    "aerolab_speed", ["#22d3ee", "#ff5e8a", "#fde047"],
)
SPEED_CLIP_FACTOR = 2.0
VORT_ALPHA_MAX = 0.7
VORT_CLIP_FACTOR = 1.5
VORT_CLIP_PERCENTILE = 92
VORT_BLUR_SIGMA_BASE = 1.0
VORT_BLUR_SIGMA_RE_SCALE = 1.6
VORT_UPSAMPLE = 1
WALL_FADE_CELLS = 14
STREAM_BLUR_SIGMA = 0.8
TEXT_COLOR = "#f5f5f5"
GIF_FRAME_MS = 67

FIG_W_IN, FIG_H_IN, FIG_DPI = 10.0, 3.0, 90


def expand_outline(xs, ys, margin):
    cx_poly = float(np.mean(xs))
    cy_poly = float(np.mean(ys))
    dx = xs - cx_poly
    dy = ys - cy_poly
    r = np.sqrt(dx * dx + dy * dy)
    r_safe = np.where(r > 1e-6, r, 1.0)
    return xs + margin * dx / r_safe, ys + margin * dy / r_safe


def build_preset(preset_name, aoa_deg=0.0):
    """Returns (mask, char_length, kick_x, label, body_xs, body_ys)."""
    aoa_rad = np.deg2rad(aoa_deg)
    cos_a = np.cos(aoa_rad)
    sin_a = np.sin(aoa_rad)
    if preset_name == "Cylinder":
        D = 20
        r = D / 2
        t = np.linspace(0.0, 2 * np.pi, 200)
        return (
            cylinder_mask(LBM_NX, LBM_NY, cx=BODY_X, cy=CY_CENTER, radius=r),
            D, BODY_X + D, "Cylinder",
            BODY_X + r * np.cos(t), CY_CENTER + r * np.sin(t),
        )
    if preset_name == "Square":
        side = 20
        s = side / 2
        xs_local = np.array([-s, s, s, -s, -s])
        ys_local = np.array([-s, -s, s, s, -s])
        return (
            square_mask(LBM_NX, LBM_NY, cx=BODY_X, cy=CY_CENTER, side=side,
                         aoa_deg=aoa_deg),
            side, BODY_X + int(side * 1.5),
            f"Square  ·  {aoa_deg:+.1f}° rotation" if abs(aoa_deg) >= 0.25 else "Square",
            BODY_X + cos_a * xs_local + sin_a * ys_local,
            CY_CENTER + (-sin_a) * xs_local + cos_a * ys_local,
        )
    if preset_name == "Ellipse":
        a, b = 22, 11
        t = np.linspace(0.0, 2 * np.pi, 200)
        xs_local = a * np.cos(t)
        ys_local = b * np.sin(t)
        return (
            ellipse_mask(LBM_NX, LBM_NY, cx=BODY_X, cy=CY_CENTER, a=a, b=b,
                          aoa_deg=aoa_deg),
            2 * b, BODY_X + a + 10,
            f"Ellipse  ·  {aoa_deg:+.1f}° rotation" if abs(aoa_deg) >= 0.25 else "Ellipse",
            BODY_X + cos_a * xs_local + sin_a * ys_local,
            CY_CENTER + (-sin_a) * xs_local + cos_a * ys_local,
        )
    chord = 44
    naca_code = preset_name.split()[1]
    poly_x, poly_y = naca4_outline_xy(naca_code)
    gx = BODY_X + chord * (poly_x * cos_a + poly_y * sin_a)
    gy = CY_CENTER + chord * (-poly_x * sin_a + poly_y * cos_a)
    return (
        naca4_airfoil_mask(
            LBM_NX, LBM_NY, cx=BODY_X, cy=CY_CENTER,
            chord=chord, naca_code=naca_code, aoa_deg=aoa_deg,
        ),
        chord, BODY_X + chord + 10,
        f"{preset_name}  ·  {aoa_deg:+.1f}° wing tilt",
        gx, gy,
    )


def _render_horizontal_cbar(scalar_mappable, ticks, tick_labels):
    fig, ax = plt.subplots(figsize=(8.5, 0.55), dpi=FIG_DPI, facecolor=BG_COLOR)
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.55, top=0.95)
    cbar = fig.colorbar(scalar_mappable, cax=ax, orientation="horizontal")
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)
    cbar.ax.xaxis.set_tick_params(color=TEXT_COLOR, labelcolor=TEXT_COLOR,
                                   labelsize=9, length=0, pad=4)
    cbar.outline.set_visible(False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG_COLOR, dpi=FIG_DPI)
    plt.close(fig)
    return buf.getvalue()


def simulate_and_render(preset_name, reynolds, aoa_deg=0.0):
    """Returns (gif_bytes, vort_cbar_png, speed_cbar_png, tau, timings)."""
    mask, char_length, kick_x, label, body_xs, body_ys = build_preset(
        preset_name, aoa_deg,
    )
    nu = U_INFLOW * char_length / reynolds
    tau = nu / CS2 + 0.5

    # Alpha-modulated RdBu_r capped at 70% opacity (wake reads as a wash).
    _rdbu = plt.get_cmap("RdBu_r")(np.linspace(0.0, 1.0, 256))
    _alpha_t = np.abs(np.linspace(-1.0, 1.0, 256))
    _rdbu[:, 3] = VORT_ALPHA_MAX * _alpha_t ** 1.4
    vorticity_cmap = ListedColormap(_rdbu, name="rdbu_alpha70")
    vorticity_cmap.set_bad((0.0, 0.0, 0.0, 0.0))
    body_xs, body_ys = expand_outline(body_xs, body_ys, BODY_OUTLINE_MARGIN)

    ds = 2
    xs_ds = np.arange(0, LBM_NX, ds)
    ys_ds = np.arange(0, LBM_NY, ds)

    # Inflow column only, forward-only integration -> smooth deflected
    # streamlines past the body. Rotation is shown by the heatmap; we don't
    # add wake seeds because bidirectional integration through chaotic
    # recirculation creates frame-to-frame jitter.
    n_inflow = max(LBM_NY // 12, 8)
    inflow_y = np.linspace(4, LBM_NY - 5, n_inflow)
    stream_seeds = np.column_stack([np.full(n_inflow, 3.0), inflow_y])

    # --- Phase 1: simulate, store snapshots (vorticity + velocity) ---
    rho0 = np.ones((LBM_NX, LBM_NY))
    u0 = np.zeros((2, LBM_NX, LBM_NY))
    u0[0] = U_INFLOW
    f = equilibrium(rho0, u0)
    f_inflow_eq = equilibrium(1.0, np.array([U_INFLOW, 0.0]))
    kick_y = CY_CENTER + KICK_Y_OFFSET

    snapshots = []
    step_counter = 0
    t_sim = time.perf_counter()
    # Warmup phase: develop wake before recording so frame 0 is fully shed.
    for _ in range(WARMUP_STEPS):
        f, _Fx, _Fy = step_njit_mrt_with_force(
            f, tau, mask, f_inflow_eq, INFLOW_DIRS, OUTFLOW_DIRS,
        )
        if KICK_START <= step_counter < KICK_END:
            f[2, kick_x, kick_y] += KICK_AMPLITUDE
            f[4, kick_x, kick_y] -= KICK_AMPLITUDE
        step_counter += 1

    for _ in range(N_FRAMES):
        for _ in range(STEPS_PER_FRAME):
            f, _Fx, _Fy = step_njit_mrt_with_force(
                f, tau, mask, f_inflow_eq, INFLOW_DIRS, OUTFLOW_DIRS,
            )
            step_counter += 1

        _, u = macroscopic(f)
        dv_dx = np.zeros_like(u[1])
        du_dy = np.zeros_like(u[0])
        dv_dx[1:-1, :] = (u[1, 2:, :] - u[1, :-2, :]) / 2
        du_dy[:, 1:-1] = (u[0, :, 2:] - u[0, :, :-2]) / 2
        vorticity = dv_dx - du_dy
        snapshots.append({
            "vorticity": vorticity.astype(np.float32),
            "u_x": u[0].astype(np.float32),
            "u_y": u[1].astype(np.float32),
            "step": step_counter,
        })
    t_sim = time.perf_counter() - t_sim

    # --- Phase 2: blended v_clip (75th percentile + U/L floor), u_clip = 2*U ---
    last_vort_fluid = np.where(mask, np.nan, snapshots[-1]["vorticity"])
    v_clip = max(
        float(np.nanpercentile(np.abs(last_vort_fluid), VORT_CLIP_PERCENTILE)),
        VORT_CLIP_FACTOR * U_INFLOW / max(char_length, 1.0),
    )
    blur_sigma = VORT_BLUR_SIGMA_BASE + VORT_BLUR_SIGMA_RE_SCALE * np.log10(
        max(reynolds / 100.0, 1.0)
    )
    fade_hires = WALL_FADE_CELLS * VORT_UPSAMPLE
    ny_hi = LBM_NY * VORT_UPSAMPLE
    y_hi = np.arange(ny_hi)
    t_edge = np.clip(np.minimum(y_hi, ny_hi - 1 - y_hi) / fade_hires, 0.0, 1.0)
    wall_fade = (t_edge * t_edge * (3.0 - 2.0 * t_edge))[None, :]
    u_clip = SPEED_CLIP_FACTOR * U_INFLOW
    speed_cmap = SPEED_CMAP
    speed_norm = Normalize(vmin=0.0, vmax=u_clip)

    # --- Phase 3: render frames into PIL Images, reusing one figure ---
    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), dpi=FIG_DPI,
                            facecolor=BG_COLOR)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.99)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, LBM_NX - 1)
    ax.set_ylim(0, LBM_NY - 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    im = ax.imshow(
        np.zeros((LBM_NY * VORT_UPSAMPLE, LBM_NX * VORT_UPSAMPLE)),
        cmap=vorticity_cmap, origin="lower",
        extent=[0, LBM_NX - 1, 0, LBM_NY - 1],
        aspect="equal", interpolation="bicubic",
        interpolation_stage="rgba",
        vmin=-v_clip, vmax=v_clip,
    )
    title_text = ax.text(
        0.012, 0.93, "", transform=ax.transAxes,
        ha="left", va="top",
        color=TEXT_COLOR, fontsize=10.5, fontweight="600",
        bbox=dict(
            boxstyle="round,pad=0.45,rounding_size=0.35",
            facecolor=(0.094, 0.110, 0.165, 0.78),
            edgecolor=(0.482, 0.541, 0.643, 0.65),
            linewidth=0.6,
        ),
        zorder=25,
    )

    body_patch = Polygon(
        np.column_stack([body_xs, body_ys]),
        closed=True, facecolor=BODY_COLOR, edgecolor="none",
        antialiased=True, zorder=10,
    )
    ax.add_patch(body_patch)

    gif_frames = []
    t_render = time.perf_counter()
    for i, snap in enumerate(snapshots):
        vort_clipped = np.clip(snap["vorticity"], -3.0 * v_clip, 3.0 * v_clip)
        vort_smooth = gaussian_filter(vort_clipped, sigma=blur_sigma) * wall_fade
        u_x_blurred = gaussian_filter(snap["u_x"], sigma=STREAM_BLUR_SIGMA)
        u_y_blurred = gaussian_filter(snap["u_y"], sigma=STREAM_BLUR_SIGMA)
        u_x_plot = np.where(mask, 0.0, u_x_blurred)
        u_y_plot = np.where(mask, 0.0, u_y_blurred)
        u_mag = np.sqrt(u_x_blurred ** 2 + u_y_blurred ** 2)
        u_mag = np.where(mask, 0.0, u_mag)

        im.set_data(vort_smooth.T)
        title_text.set_text(f"{label}  ·  Re = {reynolds}")

        for col in list(ax.collections):
            col.remove()
        for patch in list(ax.patches):
            if patch is not body_patch:
                patch.remove()

        ax.streamplot(
            xs_ds, ys_ds,
            u_x_plot[::ds, ::ds].T, u_y_plot[::ds, ::ds].T,
            start_points=stream_seeds,
            integration_direction="forward",
            density=2.0,
            color=u_mag[::ds, ::ds].T,
            cmap=speed_cmap, norm=speed_norm,
            linewidth=STREAMLINE_WIDTH, arrowsize=0.9,
        )

        fig.canvas.draw()
        img_rgba = np.asarray(fig.canvas.buffer_rgba())
        img_p = Image.fromarray(img_rgba[..., :3]).quantize(
            colors=128, method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.FLOYDSTEINBERG,
        )
        gif_frames.append(img_p)

    plt.close(fig)
    t_render = time.perf_counter() - t_render

    # --- Phase 4: encode GIF + both colorbar PNGs ---
    t_encode = time.perf_counter()
    gif_buf = io.BytesIO()
    gif_frames[0].save(
        gif_buf, format="GIF",
        save_all=True, append_images=gif_frames[1:],
        duration=GIF_FRAME_MS, loop=0, optimize=True, disposal=2,
    )
    vort_cbar_png = _render_horizontal_cbar(
        cm.ScalarMappable(norm=Normalize(vmin=-v_clip, vmax=v_clip),
                          cmap=vorticity_cmap),
        ticks=[-v_clip, 0.0, v_clip],
        tick_labels=["Clockwise spin", "No rotation", "Anti-clockwise spin"],
    )
    speed_cbar_png = _render_horizontal_cbar(
        cm.ScalarMappable(norm=speed_norm, cmap=speed_cmap),
        ticks=[0.0, U_INFLOW, u_clip],
        tick_labels=["Stalled (slow)", "Inflow speed", "Accelerated (fast)"],
    )
    t_encode = time.perf_counter() - t_encode

    return (gif_buf.getvalue(), vort_cbar_png, speed_cbar_png, tau,
            {"sim": t_sim, "render": t_render, "encode": t_encode})


def slugify(name):
    return name.lower().replace(" ", "_")


def main():
    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(exist_ok=True)

    print("=" * 76)
    print("LBM GIF preview generator v3 (smooth body + speed-coloured streams)")
    print(f"  Grid {LBM_NX}x{LBM_NY}, {N_STEPS} steps over {N_FRAMES} frames")
    print(f"  Figure {FIG_W_IN}x{FIG_H_IN} in @ {FIG_DPI} dpi  "
          f"(= {int(FIG_W_IN * FIG_DPI)}x{int(FIG_H_IN * FIG_DPI)} px)")
    print(f"  Streamlines: cyan->pink->yellow, bidirectional, inflow + wake seeds")
    print(f"  Playback: {GIF_FRAME_MS} ms/frame ({1000 / GIF_FRAME_MS:.0f} fps)")
    print("=" * 76)

    presets = [
        ("Cylinder",  100, 0.0),
        ("Square",    150, 0.0),
        ("Ellipse",   100, 0.0),
        ("NACA 0012", 200, 5.0),
        ("NACA 4412", 200, 5.0),
    ]

    t_global = time.perf_counter()
    results = []
    for preset, Re, aoa in presets:
        print(f"\n[{preset}, Re={Re}, AoA={aoa:+.1f}]")
        gif_bytes, vort_cbar, speed_cbar, tau, timings = simulate_and_render(
            preset, Re, aoa,
        )

        slug = slugify(preset)
        gif_path = out_dir / f"lbm_preview_{slug}.gif"
        vort_cbar_path = out_dir / f"lbm_preview_{slug}_vort_cbar.png"
        speed_cbar_path = out_dir / f"lbm_preview_{slug}_speed_cbar.png"
        gif_path.write_bytes(gif_bytes)
        vort_cbar_path.write_bytes(vort_cbar)
        speed_cbar_path.write_bytes(speed_cbar)

        size_kb = len(gif_bytes) / 1024
        total = timings["sim"] + timings["render"] + timings["encode"]
        print(f"  tau={tau:.3f}  sim={timings['sim']:.1f}s  "
              f"render={timings['render']:.1f}s  encode={timings['encode']:.1f}s  "
              f"total={total:.1f}s  size={size_kb:.0f} KB")
        print(f"  -> {gif_path.relative_to(out_dir.parent)}")
        print(f"  -> {vort_cbar_path.relative_to(out_dir.parent)}")
        print(f"  -> {speed_cbar_path.relative_to(out_dir.parent)}")
        results.append({
            "preset": preset, "Re": Re, "tau": tau,
            "total_s": total, "size_kb": size_kb,
        })

    t_total = time.perf_counter() - t_global
    print(f"\n{'=' * 76}")
    print(f"All {len(presets)} GIFs generated in {t_total:.1f}s "
          f"(first preset includes Numba JIT compile)")
    print(f"{'=' * 76}\n")

    print(f"{'preset':<14} {'Re':>4}  {'tau':>6}  {'total':>7}  {'size':>8}")
    for r in results:
        print(f"{r['preset']:<14} {r['Re']:>4}  {r['tau']:>6.3f}  "
              f"{r['total_s']:>6.1f}s  {r['size_kb']:>6.0f} KB")


if __name__ == "__main__":
    main()
