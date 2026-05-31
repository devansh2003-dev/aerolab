# AeroLab

> A browser-based aerodynamics playground. Try the hosted demo with no signup, or `pip install` and run it locally.

![NACA 4412 airfoil at Re=600, +15° AoA](assets/hero_naca4412_re600_aoa15.gif)

Drop in a 2D shape (cylinder, square, ellipse, NACA 4-digit airfoil, **upload your own image**, or **sketch one in the browser**), set wind speed, watch the wake develop. Two modes — one is a **neural-network surrogate**, the other is a **live simulation**, and the difference matters when you read the numbers:

- **Fast (ML surrogate)** — NeuralFoil neural-network prediction of lift / drag polars for NACA airfoils. Trained on XFoil / RANS data; not a live simulation. Good for sweeping cases in &lt;1 s each.
- **CFD (LBM solver)** — full 2D Lattice Boltzmann simulation rendered as an animated GIF, on any shape you can sketch. Validated against Williamson 1996 / Okajima 1982 in the laminar-shedding band; see [VALIDATION.md](VALIDATION.md).

CPU-only, free to use, mobile-friendly.

### About the two solver modes

**Fast (ML surrogate).** Uses [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil), a neural network trained on XFoil + RANS polars. Inference is ~1 ms per (airfoil, alpha, Re) tuple — so a 81-alpha polar lands in &lt;1 s. **Do not trust it for** anything off the NACA-4/5 + Re ≈ 10⁵–10⁷ training distribution: custom shapes, separated flow well into stall, or transonic / compressible effects.

**CFD (LBM solver).** Real 2D Lattice Boltzmann simulation — D2Q9 MRT + Smagorinsky LES, Zou-He inflow / outflow, Bouzidi interpolated bounce-back on built-in shapes. ~30 s per case locally, ~2.5 min on the 1-vCPU Cloud container. **Do not trust the Cd numbers above Re ≈ 200** for bluff bodies: real flow becomes 3D (Williamson mode-A vortex dislocations) and a strictly 2D solver cannot capture that. The validated band is Re ≤ 200; see [VALIDATION.md](VALIDATION.md).

**Live demo:** [aerolab-devansh.streamlit.app](https://aerolab-devansh.streamlit.app/) (no install, no signup)

## Validation

[![tests](https://github.com/devansh2003-dev/aerolab/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/devansh2003-dev/aerolab/actions/workflows/tests.yml)
[![benchmarks](https://img.shields.io/badge/benchmarks-Williamson%201996%20%2B%20Okajima%201982-success)](VALIDATION.md)

The 2D D2Q9 MRT-LES solver has been benchmarked against Williamson 1996 (cylinder) and Okajima 1982 (square). Two rounds of senior CFD review (2026-05-26, 2026-05-27) scoped "validated" to the regime where the comparison is honest. The current headline comes from the **Resolved preset** (D = 40, B = 10 %), the configuration that simultaneously meets the Mei-Luo-Shyy 1999 D ≥ 40 literature guideline for 2D-LBM Cd and keeps blockage low enough that the Allen-Vincenti correction is small:

| Quantity                  | Re band   | Median error | Max error | Independent reference            |
|---------------------------|-----------|--------------|-----------|----------------------------------|
| Cylinder Cd (corrected)   | 100 – 200 | **5.6 %**    | **10.2 %**| Williamson 1996 ARFM 28          |
| Square Cd (raw, see note) | 150 – 200 | **4.5 %**    | **5.1 %** | Okajima 1982 JFM 123; Sohankar 1998 |

**Why the square row is uncorrected.** The Resolved sweep exposed that the K = 1.00 Allen-Vincenti correction for the square (fitted at the Standard preset B = 0.35) *over-corrects* at low blockage. At D = 40 / B = 10 % the AV-corrected square Cd is off by ~15 %, while the **raw** measurement is within 5 % of Okajima. The honest reading is that the correction is calibrated at the wrong blockage; the raw measurement IS the solver result at this preset. K-recalibration is roadmapped. See [VALIDATION.md §3.2](VALIDATION.md) for the analysis.

Re = 200 is the Williamson mode-A 3D-instability threshold -- above it a 2D solver is structurally a different problem, not "the same problem at higher Re", so the band boundary is set by physics. Above Re = 200 we report numbers without claiming them: the Resolved cylinder Cd error rises to +24.7 % at Re = 500 (vs +32.9 % at the D = 20 Validation preset), confirming the high-Re tail is the 2D approximation breaking down, not grid resolution. Strouhal is reported as a qualitative match across the full Re range (raw FFT peak in the published ballpark) but **not** as a percent-error result -- the FFT bin width at our record length is wide enough that a percent-error figure on a near-flat reference is coincidence, not measurement. See [VALIDATION.md §3.4](VALIDATION.md) for the detail.

The Standard interactive preset (35 % blockage) is the user-facing convenience; its corrected Cd numbers look excellent (4.3 % cyl / 8.9 % sq median across Re 100 – 1000) because the 2.6 × Allen-Vincenti rescale absorbs both blockage and any other systematic the solver carries. We keep those numbers in the doc for transparency, but a senior reader should read them as a property of the correction, not of the solver -- the validation claim is anchored to the Resolved sweep above. See [VALIDATION.md §3.6](VALIDATION.md).

Mass conservation: **machine precision** in a closed box (drift ≈ 3 × 10⁻¹³ over 5000 steps), **0.84 %** of throughflow in the open channel (the documented Zou-He BC tradeoff, not a leak). On every push, `tests/test_doc_validation_consistency.py` gates the headline numbers against the committed `results_resolved.json` and `tests/test_validation_benchmark.py` runs the Standard-preset regression guard.

See [VALIDATION.md](VALIDATION.md) for the full methodology, the K-flaw analysis for the square correction, the FFT-bin-quantisation argument behind the Strouhal demotion, the LES-bias-at-low-Re note, and 14 academic citations.

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

**Day 20 of a 12-week build (started 2026-05-09). Phases 1 + 2 closed; Phase 3 in progress.**

Phase 1 (solver core, W1–4) shipped Day 5, expanded Days 6–14 with originally-Phase-2/3 work (MRT, Bouzidi, Zou-He, Mei momentum exchange). Phase 2 (W5 image upload, W6 multi-viz, W7 side-by-side compare, W8 gallery, plus a Phase-2.5 click-to-draw canvas) all shipped. Phase 3 closed: NeuralFoil ✅, validation against Williamson 1996 + Okajima 1982 ✅, plain-English UX overhaul ✅, **3D gallery (pre-baked preview) ✅**, **OpenFOAM 11 cylinder Re=100 cross-check ✅** (Cd within +1.6 %, St within −3.6 % of Williamson — both inside the reviewer's ±5 % gate; see [VALIDATION.md §8.4](VALIDATION.md)).

The **3D gallery** is a pre-baked field replay (Cloud-safe). The D3Q19 TRT kernel runs offline (one-off bake per scene, ~20 s on a laptop); the deployed app loads the saved .npz and renders interactive streamlines + body. 10 scenes ship: sphere, cylinder, cube, NACA 0012, NACA 4412 — each at Re ∈ {40, 100}; NACA wings additionally at AoA ∈ {0°, 10°}. Real-time 3D solving stays local-only (D3Q19 populations are too big for the 1-vCPU Cloud worker). See [VALIDATION.md §8](VALIDATION.md) for what 3D is and isn't validated against.

| Phase | Weeks | Deliverable | Status |
|------:|------:|------|------|
| 1 — Solver core | 1–4 | LBM works, validated, deployed with 5+ shapes | ✅ Day 5; expanded to D14 |
| 2 — Shape freedom | 5–8 | **Image upload + silhouette extraction**, multi-viz, side-by-side, GIF export, gallery | **Upload ✅**, sample silhouettes ✅, **click-to-draw canvas ✅**, side-by-side ✅, GIF ✅, gallery ✅, multi-viz (vorticity / velocity / pressure) ✅. |
| 3 — Polish + 3D | 9–12 | NeuralFoil, 3D gallery, OpenFOAM cross-validation, launch | NeuralFoil ✅, Cloud deploy ✅, **3D gallery (D3Q19 TRT, 10 pre-baked scenes) ✅**, **OpenFOAM 11 cylinder Re=100 cross-check ✅** (Cd +1.6 %, St −3.6 % vs Williamson) |

## Solver diagnostics

Complementary to the headline validation above -- these are
solver-correctness checks (conservation laws, regression smoke
tests) that run alongside the Williamson / Okajima comparison, not
on top of it.

`scripts/dev_validate_cfd.py` runs 8000 steps of cylinder Re=100 on the MRT + Zou-He + Bouzidi path.

**Local physics gates (pass/fail) — 4/4 pass:**
- median `|div(u)|` ≈ 1e-4 (incompressibility)
- mean `|u|` inside body ≈ 3e-3 (no-slip)
- mass-flux variation across x-slices ≈ 3 % (continuity)
- mass drift per 1k steps ≈ 0.1 % (Zou-He outflow conservation)

**Grid convergence:** Strouhal converges Standard ↔ Detailed to ~3 %
(was ~240 % pre-Bouzidi). Cylinder Cd at Re = 100 grid-converges
within a few percent at the Validation preset (D = 20) -- see
[VALIDATION.md §3.2](VALIDATION.md). The Cd quoted in the headline
table is from that low-blockage sweep, not from Richardson
extrapolation of the Standard preset.

**NACA 0012 polar (Re_c=200, 8 angles −5° to +15°):** lift curve is portfolio-grade (CL(0°)=−1e-4, antisymmetric to 4 decimals, slope ~0.048/deg). **Drag curve is non-physical** — chord=40 cells discretization + BGK-τ artifact stack. Use lift only. Trustworthy polar needs chord ≥ 80 cells.

**Phase 2 W5 gate:** 3 real-world bundled silhouettes (fish, car profile, building cross-section) run end-to-end at Re=200 Standard without NaN — verified by `test_phase2_w5_gate_sample_silhouettes_run_clean[*]` in the test suite.

Artifacts in `data/`: `cylinder_convergence.png`, `validation_grid_convergence.png`, `naca0012_aoa_polar.png`.

## What this solver isn't

Shares the *collision-rule family* (MRT + Smagorinsky LES in 2D, TRT in 3D) with industrial LBM solvers (PowerFLOW, Palabos, waLBerla). That's like sharing "has four wheels" with an F1 car. We don't have:

- GPU acceleration → Re envelope tops at 1500 in 2D (industrial: Re ≥ 10⁶ on GPU clusters)
- Adaptive mesh refinement → uniform 320×80 (Standard) or 960×240 (Detailed); offline 700×400 (Validation) and 1200×400 (Resolved)
- Wall-function turbulence → we resolve the boundary layer directly (only feasible at low Re)
- Cumulant collision, multi-block, automatic time-stepping (OpenFOAM 11 cross-validation now lands within ±5 % at cylinder Re=100; Fluent cross-validation not run)
- Bouzidi q-field for arbitrary uploaded polygons (built-ins have it; custom uploads use halfway BB)
- **Live 3D solve on Cloud.** The 3D D3Q19 TRT kernel runs offline (~20 s per scene on a laptop); the hosted app replays the saved velocity field. 3D Re tops out at 100 in the shipped bakes (Re=200 BGK/TRT diverged at our grid resolution — tau ≈ 0.512). **No percent-level 3D drag validation yet** — the one quantitative run (sphere Re=100 vs Clift-Grace-Weber 1978) lands at Cd = 1.57 / +44 % above reference. A low-blockage cross-check refuted the original "blockage dominates" hypothesis; the residual error is now attributed to the simplified Ladd 1994 momentum exchange + D = 20 grid resolution (see [VALIDATION.md §8.3 / §8.3.1](VALIDATION.md)).

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
└── tests/                          # ~200 unit tests + 11 validation-benchmark gates
```

## License

MIT — see [LICENSE](LICENSE). PRs welcome.
