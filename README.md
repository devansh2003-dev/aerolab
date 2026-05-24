# AeroLab

> A browser-based aerodynamics playground. Try the hosted demo with no signup, or `pip install` and run it locally.

![NACA 4412 airfoil at Re=600, +15° AoA](assets/hero_naca4412_re600_aoa15.gif)

Drop in a 2D shape (cylinder, square, ellipse, NACA 4-digit airfoil, **upload your own image**, or **sketch one in the browser**), set wind speed, watch the wake develop. Two modes:

- **Fast (NeuralFoil)** — instant ML polar predictions for airfoils.
- **Real CFD (LBM)** — full Lattice Boltzmann simulation rendered as an animated GIF, on any shape you can sketch.

CPU-only, free to use, mobile-friendly.

**Live demo:** [aerolab-devansh.streamlit.app](https://aerolab-devansh.streamlit.app/) (no install, no signup)

## Validation

[![tests](https://github.com/devansh2003-dev/aerolab/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/devansh2003-dev/aerolab/actions/workflows/tests.yml)
[![benchmarks](https://img.shields.io/badge/benchmarks-Williamson%201996%20%2B%20Okajima%201982-success)](VALIDATION.md)

The 2D D2Q9 MRT-LES solver has been benchmarked against published experimental data from peer-reviewed fluid-dynamics literature. The Standard interactive preset runs in a 35 %-blocked channel (chosen for solve speed + GIF size); raw drag is inflated ~2.5–3× by wall acceleration, and an Allen-Vincenti / West-Apelt blockage correction is applied before comparison to the free-stream reference. **The "validation" is therefore of the corrected estimate, not the raw solver output.** Full 14-case sweep (Re 100 – 1000):

| Quantity     | Median error | Max error | Tolerance band | Reference source                          |
|--------------|--------------|-----------|----------------|-------------------------------------------|
| Cylinder Cd  | **4.3 %**    | **11.6 %**| ± 15 %         | Williamson 1996 ARFM 28; Norberg 1994     |
| Square Cd    | **8.9 %**    | **21.8 %**| ± 25 %         | Okajima 1982 JFM 123; Sohankar 1998       |
| Cylinder St  | **15.2 %**   | **23.4 %**| ± 35 % †       | Williamson 1996                           |

† **Cylinder St is qualitative, not parallel-quality.** Vortex shedding is present in the right ballpark — the dominant FFT peak is within ~15–25 % of Williamson 1996 across Re = 100 – 1000. At 35 % blockage the confined-channel shedding mode flattens the raw St(Re) curve far more than any single-formula correction (West-Apelt 1982 applied here) can recover, so we report it as a sanity check, not a validation row on a par with the Cd numbers. Square St is **not** quoted at all for the same reason. See [VALIDATION.md §4.1](VALIDATION.md) for the full caveat.

Mass conservation is at **machine precision** (drift ≈ 3 × 10⁻¹³ over 5000 steps in a closed box) and **0.84 %** of throughflow in the open channel — this 0.84 % is the documented Zou-He tradeoff (prescribed velocity OR exact mass conservation, not both), not a "leak" to be hidden. Continuous validation runs on every commit via `tests/test_validation_benchmark.py`.

See [VALIDATION.md](VALIDATION.md) for full methodology, results table, limitations, and 14 academic citations.

## Performance

| Mode                    | Local (4+ cores) | Cloud (1 vCPU) |
|-------------------------|------------------|-----------------|
| Fast (NeuralFoil)       | instant          | instant         |
| CFD Standard (320×80)   | ~40 s            | ~3.3 min        |
| CFD Detailed (960×240)  | ~100 s           | ~6 min          |

First CFD click also pays a ~25 s JIT compile (~40 s on Cloud), hidden behind a startup spinner. For fast iteration use Fast mode; Real CFD is for watching the simulation, not iterating quickly.

## Quickstart

Requires Python 3.11 (Numba wheels).

```powershell
# Pick your env manager
conda create -n aerolab python=3.11 -y && conda activate aerolab
# or: py -3.11 -m venv .venv && .venv\Scripts\Activate.ps1
# or: uv venv --python 3.11 .venv && .venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install -r requirements-dev.txt   # pytest, for the test suite
streamlit run app.py
```

```powershell
pytest -q                                # 199 unit tests (+11 validation-benchmark gates), ~70 s warm
python scripts/dev_validate_cfd.py       # 4 physics gates + 3 diagnostics, ~90 s
```

## Features

**Real CFD (LBM mode)**
- **6 shape options:** cylinder, square, ellipse, NACA 0012, NACA 4412, **and "Custom"** (Upload an image / Draw in-browser / pick a built-in Sample)
- **3 ways to ship a custom shape:** drop a PNG / JPG (silhouette extraction), click-to-draw a polygon on the in-browser canvas (click vertices, click the green start dot to close), or pick a bundled sample (fish, car profile, building cross-section)
- **3 viz modes:** Vorticity (rotation), Velocity (|u|), and Pressure — switchable without re-solving
- Velocity slider 0.15–4.5 m/s (mapped to Re 50–1500 via Re = U·L/ν), per-shape AoA / rotation, two grid presets
- MRT collision + Smagorinsky LES, Zou-He inflow/outflow, Bouzidi interpolated bounce-back for built-in shapes (halfway BB for custom uploads — Bouzidi q-field for arbitrary polygons is on the roadmap)
- Side-by-side pinned comparison (works for custom shapes too), GIF download with parameter + shape encoded filenames
- Speed-coloured RK4 streaklines (plasma) + smooth body outline + flow/scale annotations baked in

**Fast (NeuralFoil)**
- Instant lift/drag/polar for NACA airfoils, three model sizes (xxxlarge / medium / xsmall)

## Status

**Day 16 of a 12-week build (started 2026-05-09). Phases 1 + 2 closed; Phase 3 in progress.**

Phase 1 (solver core, W1–4) shipped Day 5, expanded Days 6–14 with originally-Phase-2/3 work (MRT, Bouzidi, Zou-He, Mei momentum exchange). Phase 2 (W5 image upload, W6 multi-viz, W7 side-by-side compare, W8 gallery, plus a Phase-2.5 click-to-draw canvas) all shipped. Currently in Phase 3 polish: NeuralFoil ✅, validation against Williamson 1996 + Okajima 1982 ✅, plain-English UX overhaul ✅. 3D wing and OpenFOAM cross-validation remain optional stretch goals.

| Phase | Weeks | Deliverable | Status |
|------:|------:|------|------|
| 1 — Solver core | 1–4 | LBM works, validated, deployed with 5+ shapes | ✅ Day 5; expanded to D14 |
| 2 — Shape freedom | 5–8 | **Image upload + silhouette extraction**, multi-viz, side-by-side, GIF export, gallery | **Upload ✅**, sample silhouettes ✅, **click-to-draw canvas ✅**, side-by-side ✅, GIF ✅, gallery ✅, multi-viz (vorticity / velocity / pressure) ✅. |
| 3 — Polish + 3D | 9–12 | NeuralFoil ✅, optional AeroSandbox+AVL 3D, OpenFOAM cross-validation, launch | NeuralFoil + Cloud deploy ✅; 3D + OpenFOAM pending |

## Validation

`scripts/dev_validate_cfd.py` runs 8000 steps of cylinder Re=100 on the MRT + Zou-He + Bouzidi path.

**Local physics gates (pass/fail) — 4/4 pass:**
- median `|div(u)|` ≈ 1e-4 (incompressibility)
- mean `|u|` inside body ≈ 3e-3 (no-slip)
- mass-flux variation across x-slices ≈ 3 % (continuity)
- mass drift per 1k steps ≈ 0.1 % (Zou-He outflow conservation)

**Global diagnostics (reported, not gated):**
- Cd ≈ 1.9 (Richardson-extrapolated h→0, vs textbook 1.4 — residual gap is honest 20 % channel blockage)
- Strouhal grid-converges Standard↔Detailed to ~3 % (was ~240 % pre-Bouzidi)

**NACA 0012 polar (Re_c=200, 8 angles −5° to +15°):** lift curve is portfolio-grade (CL(0°)=−1e-4, antisymmetric to 4 decimals, slope ~0.048/deg). **Drag curve is non-physical** — chord=40 cells discretization + BGK-τ artifact stack. Use lift only. Trustworthy polar needs chord ≥ 80 cells.

**Phase 2 W5 gate:** 3 real-world bundled silhouettes (fish, car profile, building cross-section) run end-to-end at Re=200 Standard without NaN — verified by `test_phase2_w5_gate_sample_silhouettes_run_clean[*]` in the test suite.

Artifacts in `data/`: `cylinder_convergence.png`, `validation_grid_convergence.png`, `naca0012_aoa_polar.png`.

## What this solver isn't

Shares the *collision-rule family* (MRT + Smagorinsky LES) with industrial LBM solvers (PowerFLOW, Palabos, waLBerla). That's like sharing "has four wheels" with an F1 car. We don't have:

- GPU acceleration → Re envelope tops at 1500 in 2D (industrial: Re ≥ 10⁶ on GPU clusters)
- Adaptive mesh refinement → uniform 240×80 or 720×240
- Wall-function turbulence → we resolve the boundary layer directly (only feasible at low Re)
- Cumulant collision, multi-block, automatic time-stepping, 3D, OpenFOAM/Fluent cross-validation
- Bouzidi q-field for arbitrary uploaded polygons (built-ins have it; custom uploads use halfway BB)

Every choice on the production hot path is textbook-correct for built-in shapes. The *envelope* (Re, dimensionality, scope) is firmly academic-tutorial.

## Stack

- **Solver:** NumPy reference + Numba `@njit(fastmath=True)` fused-step (collide + force + bounce-back + stream + Zou-He + Bouzidi in one function)
- **UI:** Streamlit
- **Silhouette extraction:** scikit-image (Otsu threshold + find_contours + Douglas-Peucker)
- **Viz:** matplotlib (LBM render), Plotly (airfoil polars), Pillow (GIF assembly + polygon rasterization), `scipy.ndimage.gaussian_filter` (smoothing)
- **ML mode:** [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil) + [AeroSandbox](https://github.com/peterdsharpe/AeroSandbox)
- **Hosting:** Streamlit Community Cloud (auto-redeploys on push to `main`)

> The deployed version may lag `main` by 1–2 commits. If something looks off, `streamlit run app.py` locally is authoritative.

## Repo layout

```
aerolab/
├── app.py                          # Streamlit entry — dual mode
├── src/
│   ├── lbm.py                      # D2Q9 solver: BGK + MRT, Zou-He, Bouzidi
│   ├── lbm_render.py               # simulate_and_render: sim + streaklines + GIF
│   ├── forces.py                   # Ladd 1994 + Mei 2002 momentum exchange
│   ├── shapes.py                   # cylinder, square, ellipse, NACA 4-digit + q-fields
│   ├── custom_shape.py             # Upload silhouette extraction + polygon rasterization
│   ├── sample_shapes.py            # Bundled fish / car / building polygons
│   └── airfoils.py                 # NeuralFoil/AeroSandbox wrapper
├── scripts/
│   ├── lid_cavity_smoke.py         # cavity benchmark
│   ├── week1_cylinder.py           # cylinder Re=100 reference run
│   ├── week1_cylinder_sweep.py     # 5-config Cd convergence study
│   ├── naca0012_aoa_polar.py       # 8-angle airfoil polar
│   ├── dev_validate_cfd.py         # 4 physics gates + 3 diagnostics
│   └── dev_grid_convergence.py     # Std vs Detailed + Richardson extrapolation
└── tests/                          # 186 unit tests (incl. Phase 2 W5 gate)
```

## License

MIT — see [LICENSE](LICENSE). PRs welcome.
