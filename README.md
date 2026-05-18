# AeroLab

> An open-source, browser-based aerodynamics playground.
> No install, no license, no 40-tab tutorial.

![AeroLab — cylinder wake at Re=400, von Kármán vortex shedding visualised with plasma-coloured streaklines on a vorticity heatmap](assets/hero_cylinder_re400.gif)

Drop in a 2D shape (cylinder, square, ellipse, NACA airfoil), set the angle of attack and Reynolds number, watch the wake develop. Two modes:

- **Fast (NeuralFoil)** — instant ML polar predictions for airfoils. Drag a slider, get lift / drag / drag-polar numbers in under a second.
- **Real CFD (LBM)** — full Lattice Boltzmann simulation rendered as an animated GIF. Watch the air actually move. ~12 s for a Standard-resolution run locally, ~50 s for Detailed. Streamlit Cloud is ~5× slower (~1 min and ~3 min respectively — see "Honest expectations" below).

Built CPU-only with a custom Lattice Boltzmann solver, free to use, mobile-friendly.

## Live demo

**[aerolab-devansh.streamlit.app](https://aerolab-devansh.streamlit.app/)**

> ⚠ **The deployed version may lag `main` by 1–2 commits.** Streamlit Cloud auto-redeploys on push, but the URL is only verified after each push. If something looks off, the local instance (`streamlit run app.py`) is authoritative.

> **Honest expectations:** Fast mode is instant. Real CFD mode runs a real LBM simulation. **Local (Numba parallel JIT, 4+ cores):** ~12 s Standard, ~50 s Detailed per click. **Streamlit Cloud free tier (1-vCPU shared CPU, no parallel benefit):** ~1 min Standard, ~3 min Detailed. The first CFD click also pays a ~25 s JIT compile (~40 s on Cloud) which we try to hide behind a startup spinner. If you want to play with sliders fast, use Fast mode — Real CFD is for watching the simulation, not iterating quickly.

### What's working

- **Real CFD (LBM mode):** 5 shape presets (cylinder, square, ellipse, NACA 0012, NACA 4412), Reynolds 50–1500, AoA / rotation per-shape, two grid presets (Standard 240×80, Detailed 720×240). MRT collision + Smagorinsky LES, Zou-He inflow/outflow BCs, Bouzidi interpolated wall correction. Side-by-side comparison, GIF download with parameter-encoded filenames.
- **Fast mode:** NeuralFoil polar sweeps for NACA airfoils with three model sizes (xxxlarge / medium / xsmall).
- **106 unit tests** covering physics invariants, JIT/reference equivalence, geometric correctness, visual-regression snapshot. `pytest -q` runs in ~21 s warm (the visual-regression test runs a 51-frame canonical cylinder pipeline).
- **Validation:** lid-driven cavity benchmark hits the Ghia 1982 reference, cylinder Re=100 produces clean von Kármán shedding, full validation audit in `scripts/dev_validate_cfd.py`.

### What this solver isn't

AeroLab shares the *collision-rule family* (MRT + Smagorinsky LES) with industrial LBM solvers like PowerFLOW, ProLB, and M-Star, and with academic codes like Palabos and waLBerla. Sharing the collision rule is like sharing "has four wheels" with a Formula 1 car. Industrial CFD adds layers we don't have:

- **No GPU acceleration.** Pure CPU + Numba parallel-over-x. Re envelope tops out at 1500 in 2D on a single-process box; industrial solvers run Re ≥ 10⁶ on GPU clusters.
- **No adaptive mesh refinement.** Uniform 240×80 or 720×240 grid. Real solvers refine near walls and shear layers automatically.
- **No wall-function turbulence model.** We resolve the boundary layer directly, which is only feasible at low Re. Wall functions are what lets industrial codes skip resolving the viscous sublayer.
- **No cumulant collision, no multi-block, no automatic time-stepping, no 3D.** Each is a multi-month addition.
- **No cross-validation against OpenFOAM / Fluent / Star-CCM+.** We compare against textbook numbers (Strouhal 0.165, Cd 1.4 from 1980s reference tables) — not a contemporary co-run. That cross-comparison is roadmapped as Phase 3 work.
- **No 30-year industrial validation library.** We have a handful of canonical cases (lid-driven cavity, cylinder Re=100, NACA 0012 polar) with the gaps documented honestly in the Validation section.

This is a serious academic-style toy solver, not an industrial tool. The honest framing is: every choice on the production hot path (MRT, Smagorinsky, Bouzidi, Zou-He, momentum exchange) is textbook-correct, but the *envelope* (Re, dimensionality, physics scope) is firmly in the academic-tutorial range.

### Project status

**Day 14 of a 12-week build. Honest accounting:**

Phase 1 (solver core, weeks 1–4) shipped on Day 5 with the originally-planned BGK path, then expanded in days 6–14 with work that was originally Phase 2/3 scope: MRT collision (Phase 2 W6), Bouzidi interpolated bounce-back, Zou-He BCs, Mei momentum-exchange (post-launch physics audit), plus UI polish (snapshot comparison, GIF download, demo gallery, sidebar onboarding) and repo hygiene (LICENSE, pyproject.toml, CONTRIBUTING).

**Phase 2's headline deliverable — image/SVG upload with silhouette extraction — has not started.** The "shape freedom" promise (load any 2D image, run CFD on its silhouette) is the feature that meaningfully differentiates AeroLab from "another LBM demo," and it's still vapourware. Multi-viz modes (pressure, streamline density, velocity magnitude) also pending.

The schedule isn't slipping in calendar time; it's pivoting in *scope*. The solver became more rigorous (Bouzidi + Zou-He are not undergrad-tutorial features), but the upload feature that was supposed to land in weeks 5–8 is still ahead.

### What's in the box

- **D2Q9 Lattice Boltzmann solver** ([src/lbm.py](src/lbm.py)). Two collision operators:
  - **BGK** with halfway bounce-back (pure-NumPy reference + parallel-JIT fused step). Bit-equivalent at atol=1e-10 across the JIT/reference paths, verified by two multistep equivalence tests.
  - **MRT (multi-relaxation-time)** with Smagorinsky LES sub-grid eddy viscosity (the production hot path). `C_SMAG = 0.17` (Lilly 1967 theoretical value, verified across a 9-shape × 4-Re stability survey). Stable from Re=50 up through Re=1500 on every shape preset including sharp-edged bluff bodies. References: Lallemand & Luo (2000), d'Humières et al. (2002), Lilly (1967) for the LES constant.
- **Boundary conditions:**
  - **Zou & He (1997)** velocity-inflow + pressure-outflow at the inlet/outlet (replaces the older equilibrium-inflow + zero-gradient-outflow pair). Cuts long-run mass drift from ~3 %/1 k steps to ~0.1 %/1 k steps.
  - **Bouzidi-Firdaouss-Lallemand (2001)** interpolated bounce-back at the body surface (replaces halfway bounce-back). Wall now lives at its analytic location, not at the nearest lattice node. Cuts the Cd overshoot from ~89 % above textbook to ~37 %, and reduces Standard-vs-Detailed Strouhal disagreement from ~240 % to ~3 % (grid converged).
  - **Mei-Yu-Shyy-Luo (2002) Bouzidi-aware momentum exchange** for the body force calculation.
  - **Bounce-back top/bottom walls** with a smoothstep alpha-fade in the displayed heatmap.
- **Shape mask + q-field library** in [src/shapes.py](src/shapes.py): cylinder, square, ellipse, NACA 4-digit airfoil. Square and ellipse take an `aoa_deg` rotation; NACA airfoils take wing tilt. All four ship analytic q-fields (sub-cell wall-distance per link) for Bouzidi. Polygon-segment intersection for the airfoil, quadratic line-shape for cylinder/ellipse, 4-face linear intersection for the square.
- **Momentum-exchange force calculation** (Ladd 1994 + Mei 2002) in [src/forces.py](src/forces.py) and inline in the JIT step.
- **Streamlit LBM mode** in [app.py](app.py):
  - 240×80 ("Standard", 60 frames × 35 steps = 2100 LBM steps) or 720×240 ("Detailed", 100 frames × 35 steps = 3500 LBM steps) grid, Reynolds slider 50–1500, rotation sliders per-shape. Frame counts tuned so the wake captures >1 full vortex-shedding period on each preset while keeping per-click wait time bearable on Streamlit Cloud's 1-vCPU shared CPU (~1 min Standard, ~3 min Detailed on Cloud; ~12 s and ~50 s locally with Numba parallel).
  - **Vorticity heatmap** (RdBu_r diverging, alpha-modulated, capped at 90 % opacity) with a vertical wall fade.
  - **Speed-coloured streaklines** on a perceptually-uniform `plasma` colormap. RK4-advected particles seeded from the inflow + from a wake band downstream of the body (with body-interior rejection), so the wake region stays populated even after particles convect out.
  - **Smooth analytic body outline** overlaid on the LBM mask (70 % alpha so the boundary layer is visible, thin slate edge stroke).
  - **In-figure annotations**: flow-direction arrow (top-left), body-size scale bar (bottom-right) — baked into the GIF so screenshots stay self-explanatory.
  - **Side-by-side comparison**: pin a run via session state, change parameters, run again — old run + new run shown together.
  - **GIF download** button with parameter-encoded filenames (`aerolab_naca_0012_re1000_aoa5_detailed.gif`).
  - Two plain-English colorbars below the GIF + an HTML-swatch legend (works in light/dark themes and screenshots).
- **106 unit tests** covering physics invariants, JIT/reference equivalence, force-calc sanity, per-shape geometric invariants, q-field correctness for Bouzidi, end-to-end visual regression snapshot. Run with `pytest -v`.
- **Lid-driven cavity** benchmark — vortex center lands at the Ghia et al. (1982) reference position.
- **Cylinder Re=100** — clean von Kármán vortex shedding.

### Validation — Phase 1 findings

The Phase 1 gate had two physical tests (Strouhal and Cd at Re=100). Strouhal passes cleanly. Cd has documented wall-treatment biases that we characterized via a 5-config convergence sweep.

**Cylinder Re=100 convergence sweep (Mach × resolution axes, halfway bounce-back, BGK):**

- **Strouhal** passes within 3.0–9.1 % of textbook 0.165 across all 5 configs.
- **Time-averaged Cd** ranges 2.04–2.49 vs textbook 1.4. The Mach axis went the *opposite* direction from the naive compressibility prediction — Cd *grew* as Ma decreased — which is the signature of the **BGK-τ wall-correction artifact** (He, Zou, Luo & Dembo 1997; Cornubert et al. 1991): halfway bounce-back's effective wall drifts inward as τ → 0.5, shrinking the effective cylinder area and inflating Cd against the nominal diameter.
- The discretization axis at fixed Mach behaved correctly (Cd decreased as D grew).
- **The structural fixes shipped early.** MRT collision was originally scheduled for Phase 2 W6 and pulled forward to Day 5. Bouzidi + Zou-He BCs landed during the post-launch physics audit and recover most of the remaining Cd overshoot (~37 % above textbook in the corrected sweep; the residual ~20 % is honest channel blockage at 20 % occupancy).

Convergence artifact: `data/cylinder_convergence.png` + `.csv` (from `scripts/week1_cylinder_sweep.py`; ~90 min full run). Grid-convergence study with Richardson extrapolation: `data/validation_grid_convergence.png` (from `scripts/dev_grid_convergence.py`; ~6 min).

**NACA 0012 AoA polar at Re_c=200 (bonus, 8 angles from -5° to +15°):**

- **Lift curve is portfolio-grade.** CL(0°) = −1e-4 (perfect symmetric airfoil), CL(−α) = −CL(α) to four decimals, monotonic and roughly linear with slope ~0.048/deg (about 44 % of thin-airfoil theory, consistent with Re=200 viscous effects).
- **Drag curve is non-physical** — non-monotonic with a peak at α=±5° and falling Cd at higher AoA. Two-part cause: (a) chord=40 cells means max thickness is only 4.8 cells at α=0, so discretization error per unit wetted area is much higher at α=0 than at α=15°; (b) the BGK-τ artifact stacks on top. Honest report: use the lift curve only.
- **No vortex shedding** at this Re (laminar attached wake), so the Strouhal column in the output CSV is meaningless.
- To produce a quantitatively trustworthy airfoil polar in the future: bump chord ≥ 80 cells (4× compute) and re-run with the new MRT + Bouzidi path.

Artifact: `data/naca0012_aoa_polar.png` + `.csv` (from `scripts/naca0012_aoa_polar.py`; ~52 min full run).

### CFD validation

`scripts/dev_validate_cfd.py` is the honest physics audit. It runs 8000 steps of cylinder Re=100 on the production MRT + Zou-He + Bouzidi path and reports two distinct things:

**Local physics gates (pass/fail).** These prove the solver is doing Navier-Stokes correctly cell-by-cell:

- median `|div(u)|` ≈ 1e-4 (incompressibility)
- mean `|u|` inside the body ≈ 3e-3 (no-slip enforced)
- mass-flux variation across x-slices ≈ 3 % (continuity holds)
- mass drift per 1k steps ≈ 0.1 % (conservation; Zou-He pressure outflow lets the domain breathe)

**4/4 local gates pass.**

**Global flow diagnostics (reported, NOT gated).** These compare against free-stream textbook for cylinder Re=100. Bouzidi shrunk the gap significantly versus the original halfway-BB numbers, but a residual offset remains due to 20 % channel blockage and a domain that's only ~12D downstream:

- measured Cd ≈ 1.9 (Richardson-extrapolated h→0, vs textbook 1.4 — remaining gap is honest blockage)
- measured Strouhal grid-converges across Standard and Detailed to within ~3 % (was ~240 % apart pre-Bouzidi)
- far-field `u_x` ≈ 0.086 (vs U=0.1 — wake still recovering at outflow)

The honest framing: our solver is doing Navier-Stokes correctly, the boundary conditions are textbook (Zou-He + Bouzidi), and the residual gap to free-stream textbook is geometric (channel blockage, finite downstream length) — not numerical.

## 12-Week Roadmap

| Phase | Weeks | Deliverable | Status |
|------:|------:|------|------|
| **1 — Solver core** | 1–4  | LBM solver works, validated, deployed with 5+ pre-set 2D shapes | ✅ Shipped Day 5; expanded Days 6–14 with MRT, Bouzidi, Zou-He, Mei (originally Phase 2/3) |
| **2 — Shape freedom** | 5–8  | **Image / SVG upload with silhouette extraction (the headline feature)**, multiple viz modes (pressure / velocity magnitude / streamline density), side-by-side comparison, GIF export, demo gallery | Side-by-side ✅, GIF download ✅, demo gallery ✅ (all originally Phase 2 "polish" deliverables). **Image upload + multi-viz: not started.** |
| **3 — Polish + optional 3D** | 9–12 | NeuralFoil instant mode (✅ already shipped Phase 0), optional 3D wing module via AeroSandbox+AVL, OpenFOAM cross-validation, public launch | NeuralFoil + Streamlit Cloud deploy ✅; 3D module + OpenFOAM co-run pending |

Validation gates per phase are non-negotiable. Phase 1's Strouhal gate passed. The Cd gate is now within blockage-explained range after the Bouzidi + Zou-He upgrade. **Phase 2's gate is the image-upload demo working end-to-end on three real-world silhouettes (e.g. car profile, fish, building cross-section) — not yet attempted.**

## Stack

- **Language:** Python 3.11
- **Solver:** NumPy reference path + Numba `@njit(parallel=True, fastmath=True)` fused-step (collide + force + bounce-back + stream + Zou-He BCs + Bouzidi correction in one function, parallel over x). JIT cache is pre-warmed at app startup via [src/warmup.py](src/warmup.py).
- **UI:** Streamlit (mobile-friendly, free hosting)
- **Heatmap smoothing + streamline pre-blur:** `scipy.ndimage.gaussian_filter`
- **Visualization:** matplotlib for the LBM render (alpha-modulated RdBu_r, plasma streakline cmap, smooth analytic body patches, flow + scale annotations); Plotly for the interactive airfoil polars
- **GIF assembly:** Pillow (`quantize` + `save_all` + `append_images`) for the LBM animation
- **Optional ML mode:** [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil) + [AeroSandbox](https://github.com/peterdsharpe/AeroSandbox) for instant airfoil predictions (xxxlarge / medium / xsmall model selector in the sidebar)
- **Hosting:** Streamlit Community Cloud (free tier; auto-redeploys on push to `main`)

## Local setup

Pick the environment manager you already have. The project requires Python 3.11 (some deps don't yet have 3.12 wheels for Numba).

### conda (maintainer's default)

```powershell
conda create -n aerolab python=3.11 -y
conda activate aerolab
pip install -r requirements.txt
pip install -r requirements-dev.txt   # pytest, for running the test suite
```

### venv (no conda)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1            # PowerShell. On bash: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### uv (fastest)

```powershell
uv venv --python 3.11 .venv
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt
```

### Run + test

```powershell
# Smoke tests
pytest -q                                # 106 unit tests, ~21 s warm
python scripts/day1_test.py              # NeuralFoil airfoil prediction
python scripts/lid_cavity_smoke.py       # LBM cavity benchmark, ~25 s
python scripts/week1_cylinder.py         # LBM cylinder + vortex shedding, ~180 s
python scripts/shape_gallery.py          # 5-shape wake comparison, ~3 min

# Long-running validation experiments
python scripts/week1_cylinder_sweep.py   # 5-config convergence sweep, ~90 min
python scripts/naca0012_aoa_polar.py     # 8-angle airfoil polar, ~52 min

# CFD physics validation (4 local gates + 3 honest diagnostics, ~90 s)
python scripts/dev_validate_cfd.py       # writes data/validation_cfd_full.png

# Grid convergence study with Richardson extrapolation (~6 min)
python scripts/dev_grid_convergence.py   # writes data/validation_grid_convergence.png

# Run the deployed app locally (first click ~30 s warm; first run cold ~70 s with JIT)
streamlit run app.py
```

## Repo layout

```
aerolab/
├── README.md
├── requirements.txt                # deploy deps (streamlit, numpy, numba, scipy, ...)
├── requirements-dev.txt            # pytest only
├── .gitignore
├── app.py                          # Streamlit entry — dual-mode (Fast NeuralFoil / Real CFD LBM)
├── src/
│   ├── __init__.py
│   ├── lbm.py                      # D2Q9 solver: BGK + MRT (Smagorinsky LES, C_SMAG=0.17), Zou-He BCs, Bouzidi correction
│   ├── lbm_render.py               # simulate_and_render: full sim + RK4 streaklines + GIF assembly
│   ├── warmup.py                   # pre-compile JIT step variants on a tiny grid at app startup
│   ├── forces.py                   # momentum-exchange force calc (Ladd 1994 + Mei 2002 Bouzidi-aware)
│   ├── shapes.py                   # cylinder, square, ellipse, NACA 4-digit masks + analytic q-fields
│   └── airfoils.py                 # NeuralFoil/AeroSandbox wrapper (Fast Mode in the app)
├── scripts/
│   ├── day1_test.py                # single-point NeuralFoil smoke test
│   ├── polar_sweep.py              # alpha-sweep NeuralFoil polar generator
│   ├── lid_cavity_smoke.py         # lid-driven cavity (LBM benchmark)
│   ├── week1_cylinder.py           # cylinder at Re=100: wake + Cd time series + Strouhal
│   ├── week1_cylinder_sweep.py     # 5-config Cd convergence study (Mach × D axes)
│   ├── shape_gallery.py            # 5-shape wake comparison
│   ├── naca0012_aoa_polar.py       # NACA 0012 lift+drag polar across 8 AoAs
│   ├── dev_validate_cfd.py         # 4 local-physics gates + 3 honest global diagnostics
│   └── dev_grid_convergence.py     # Std vs Detailed Cd/St + Richardson extrapolation
├── tests/
│   ├── test_lbm.py                 # 33 invariants for the solver + JIT-vs-reference equivalence (incl. Zou-He, Bouzidi)
│   ├── test_lbm_render.py          # 15 invariants for the GIF pipeline
│   ├── test_forces.py              # 6 invariants for momentum-exchange force calc
│   └── test_shapes.py              # 42 invariants across the 4 mask + q-field generators
└── data/                           # output artifacts: PNGs, CSVs, GIFs (git-ignored)
```

## License

MIT — see [LICENSE](LICENSE). Free to fork, use, modify, redistribute. Pull requests welcome.
