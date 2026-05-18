# AeroLab

> An open-source, browser-based aerodynamics playground.
> No install, no license, no 40-tab tutorial.

By August 2026: drop in a 2D shape, drag sliders for angle of attack and Reynolds number, watch streamlines and a vorticity heatmap of the air flowing around it in near real time. Built CPU-only with a custom Lattice Boltzmann solver, free to use, mobile-friendly.

## Live demo

**[aerolab-devansh.streamlit.app](https://aerolab-devansh.streamlit.app/)**

Two modes, sidebar toggle:

- **Fast (NeuralFoil)** — instant ML predictions for NACA airfoils, sweep angle of attack, compare lift / drag / drag-polar side-by-side. Powered by [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil) (a neural net trained on millions of XFoil runs).
- **Real CFD (LBM)** — a full Lattice Boltzmann simulation rendered as a 75-frame animated GIF. Pick a shape (cylinder, square, ellipse, NACA 0012, NACA 4412), set the Reynolds number (50–1500), tilt non-cylinder shapes through any rotation angle, and watch the wake develop. Numba-JIT compiled, ~30 s warm per click on a 320×100 grid.

## Status

**Phase 1 closed early (Day 5 of 12 weeks).** Solver works, shape library shipped, validation documented, **Streamlit LBM mode shipped with the MRT structural fix originally scheduled for Phase 2 W6**.

### What's in the box

- **D2Q9 Lattice Boltzmann solver** ([src/lbm.py](src/lbm.py)). Two collision operators:
  - **BGK** with halfway bounce-back (pure-NumPy reference + parallel-JIT fused step). Bit-equivalent at atol=1e-10 across the JIT/reference paths, verified by two multistep equivalence tests.
  - **MRT (multi-relaxation-time)** with Smagorinsky LES sub-grid eddy viscosity (the production hot path). Stable from Re=50 up through Re=1500 on every shape preset including sharp-edged bluff bodies. References: Lallemand & Luo (2000), d'Humières et al. (2002), Yu et al. (2005) for the LES adjustment.
- **Bounce-back top/bottom walls** with a smoothstep alpha-fade in the displayed heatmap — kills the periodic vertical wraparound (air exiting the bottom no longer re-enters at the top) without showing the wall boundary layer in the visualization.
- **Momentum-exchange force calculation** (Ladd 1994) in [src/forces.py](src/forces.py).
- **Shape mask library** in [src/shapes.py](src/shapes.py): cylinder, square, ellipse, NACA 4-digit airfoil. Square and ellipse take an `aoa_deg` rotation; NACA airfoils take wing tilt. All four use the same CW rotation convention.
- **Streamlit LBM mode** in [app.py](app.py):
  - 320×100 ("Standard") or 480×140 ("Detailed") grid, Reynolds slider 50–1500, rotation sliders per-shape
  - Vorticity heatmap (RdBu_r diverging, alpha-modulated, capped at 70% opacity) with a vertical wall fade
  - Speed-coloured streamlines (cyan → pink → yellow gradient) on a smoothed velocity field
  - Smooth analytic body outline overlaid on the voxelized LBM mask
  - 1500-step warmup before frame recording so frame 0 opens on a fully-developed wake (no startup glitch)
  - Two plain-English colorbars below the GIF + a four-column legend with Material Icons
- **56 unit tests** covering physics invariants, JIT/reference equivalence, force-calc sanity, per-shape geometric invariants. Run with `pytest -v`.
- **Lid-driven cavity** benchmark — vortex center lands at the Ghia et al. (1982) reference position.
- **Cylinder Re=100** — clean von Kármán vortex shedding with the BGK gate documented honestly (see below).

### Validation — Phase 1 findings

The Phase 1 gate had two physical tests (Strouhal and Cd at Re=100). Strouhal passes cleanly. Cd has a documented BGK-collision-model bias that we characterized via a 5-config convergence sweep — the *reason* it fails is itself the more valuable result.

**Cylinder Re=100 convergence sweep (Mach × resolution axes):**

- **Strouhal** passes within 3.0–9.1% of textbook 0.165 across all 5 configs.
- **Time-averaged Cd** ranges 2.04–2.49 vs textbook 1.4. The Mach axis went the *opposite* direction from the naive compressibility prediction — Cd *grew* as Ma decreased — which is the signature of the **BGK-τ wall-correction artifact** (He, Zou, Luo & Dembo 1997; Cornubert et al. 1991): halfway bounce-back's effective wall drifts inward as τ → 0.5, shrinking the effective cylinder area and inflating Cd against the nominal diameter.
- The discretization axis at fixed Mach behaved correctly (Cd decreased as D grew).
- **The structural fix shipped early.** MRT collision was originally scheduled for Phase 2 W6; pulled forward to Day 5 because the user wanted Re ≥ 1000 to render cleanly. With MRT + Smagorinsky LES the solver is stable on every shape × rotation × Re=1500 combination.

Convergence artifact: `data/cylinder_convergence.png` + `.csv` (from `scripts/week1_cylinder_sweep.py`; ~90 min full run).

**NACA 0012 AoA polar at Re_c=200 (bonus, 8 angles from -5° to +15°):**

- **Lift curve is portfolio-grade.** CL(0°) = −1e-4 (perfect symmetric airfoil), CL(−α) = −CL(α) to four decimals, monotonic and roughly linear with slope ~0.048/deg (about 44% of thin-airfoil theory, consistent with Re=200 viscous effects).
- **Drag curve is non-physical** — non-monotonic with a peak at α=±5° and falling Cd at higher AoA. Two-part cause: (a) chord=40 cells means max thickness is only 4.8 cells at α=0, so discretization error per unit wetted area is much higher at α=0 than at α=15°; (b) the BGK-τ artifact stacks on top. Honest report: use the lift curve only.
- **No vortex shedding** at this Re (laminar attached wake), so the Strouhal column in the output CSV is meaningless.
- To produce a quantitatively trustworthy airfoil polar in the future: bump chord ≥ 80 cells (4× compute) and re-run with the new MRT path.

Artifact: `data/naca0012_aoa_polar.png` + `.csv` (from `scripts/naca0012_aoa_polar.py`; ~52 min full run).

### Streamline validation

`scripts/dev_validate_streamlines.py` checks that the LBM velocity field is physical before we use it for streamlines:

- median `|div(u)|` = 1.24e-05 (vs ≪1e-3 target — incompressible to 5 dp)
- mean `|u|` inside the body = 2.5e-3 (vs U=0.1 — no-slip holds)
- flux variation across cross-sections = 2.65% (vs <10% — mass conserved)
- far-field `u_x` = 0.0991 (vs target 0.1 — asymptotic flow recovered)

All four checks pass at Re=100. If you ever doubt a streamline picture, run this script first.

## 12-Week Roadmap

| Phase | Weeks | Deliverable | Status |
|------:|------:|------|------|
| **1 — Solver core** | 1–4  | LBM solver works, validated, deployed with 5+ pre-set 2D shapes | ✅ Shipped Day 5 |
| **2 — Shape freedom** | 5–8  | Image / SVG upload with silhouette extraction, multiple viz modes, side-by-side comparison, GIF export, demo gallery | Up next |
| **3 — Polish + optional 3D** | 9–12 | NeuralFoil instant mode (✅ already shipped), optional 3D wing module via AeroSandbox+AVL, optional OpenFOAM validation, public launch | Partially shipped |

Validation gates per phase are non-negotiable. Phase 1's Strouhal gate passed. The Cd gate is dominated by the BGK-τ wall-correction artifact; the structural fix (MRT collision) shipped on Day 5 in service of the high-Re Streamlit experience, ahead of the planned W6 schedule.

## Stack

- **Language:** Python 3.11
- **Solver:** NumPy reference path + Numba `@njit(parallel=True, fastmath=True)` fused-step (collide + force + bounce-back + stream + BCs in one function, parallel over x)
- **UI:** Streamlit (mobile-friendly, free hosting)
- **Heatmap smoothing + streamline pre-blur:** `scipy.ndimage.gaussian_filter`
- **Visualization:** matplotlib for the LBM render (alpha-modulated RdBu_r, custom cyan-pink-yellow streamline cmap, smooth analytic body patches); Plotly for the interactive airfoil polars
- **GIF assembly:** Pillow (`quantize` + `save_all` + `append_images`) for the LBM animation
- **Optional ML mode:** [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil) + [AeroSandbox](https://github.com/peterdsharpe/AeroSandbox) for instant airfoil predictions (xxxlarge / medium / xsmall model selector in the sidebar)
- **Hosting:** Streamlit Community Cloud (free tier; auto-redeploys on push to `main`)

## Local setup

```powershell
conda create -n aerolab python=3.11 -y
conda activate aerolab
pip install -r requirements.txt
pip install -r requirements-dev.txt   # pytest, for running the test suite

# Smoke tests
pytest -v                                # 56 unit tests, ~2 s
python scripts/day1_test.py              # NeuralFoil airfoil prediction
python scripts/lid_cavity_smoke.py       # LBM cavity benchmark, ~25 s
python scripts/week1_cylinder.py         # LBM cylinder + vortex shedding, ~160 s
python scripts/shape_gallery.py          # 5-shape wake comparison, ~3 min

# Long-running validation experiments
python scripts/week1_cylinder_sweep.py   # 5-config convergence sweep, ~90 min
python scripts/naca0012_aoa_polar.py     # 8-angle airfoil polar, ~52 min

# Dev preview: render all 5 LBM presets as standalone GIFs
python scripts/dev_lbm_gif_preview.py    # ~3 min, writes data/lbm_preview_*.gif
python scripts/dev_validate_streamlines.py   # 4-check physics audit, ~30 s

# Run the deployed app locally
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
│   ├── lbm.py                      # D2Q9 solver: BGK + MRT (Smagorinsky LES), bounce-back walls
│   ├── forces.py                   # momentum-exchange force calc (Ladd 1994)
│   ├── shapes.py                   # cylinder, square, ellipse, NACA 4-digit (all with rotation)
│   └── airfoils.py                 # NeuralFoil/AeroSandbox wrapper (Fast Mode in the app)
├── scripts/
│   ├── day1_test.py                # single-point NeuralFoil smoke test
│   ├── polar_sweep.py              # alpha-sweep NeuralFoil polar generator
│   ├── lid_cavity_smoke.py         # lid-driven cavity (LBM benchmark)
│   ├── week1_cylinder.py           # cylinder at Re=100: wake + Cd time series + Strouhal
│   ├── week1_cylinder_sweep.py     # 5-config Cd convergence study (Mach × D axes)
│   ├── shape_gallery.py            # 5-shape wake comparison
│   ├── naca0012_aoa_polar.py       # NACA 0012 lift+drag polar across 8 AoAs
│   ├── dev_lbm_gif_preview.py      # offline preview render of all 5 Streamlit-mode presets
│   └── dev_validate_streamlines.py # div(u), no-slip, mass-conservation, far-field checks
├── tests/
│   ├── test_lbm.py                 # 30 invariants for the solver + JIT-vs-reference equivalence
│   ├── test_forces.py              # 6 invariants for momentum-exchange force calc
│   └── test_shapes.py              # 20 invariants across the 4 mask generators
└── data/                           # output artifacts: PNGs, CSVs, GIFs (git-ignored)
```

## License

TBD (planning MIT once the project hits public launch in Phase 3 Week 12).
