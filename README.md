# AeroLab

> Browser-based aerodynamics playground. No install, no signup.

![AeroLab — cylinder wake at Re=400, von Kármán shedding visualised with plasma streaklines on a vorticity heatmap](assets/hero_cylinder_re400.gif)

Drop in a 2D shape (cylinder, square, ellipse, NACA 4-digit airfoil), set angle of attack and Reynolds number, watch the wake develop. Two modes:

- **Fast (NeuralFoil)** — instant ML polar predictions for airfoils.
- **Real CFD (LBM)** — full Lattice Boltzmann simulation rendered as a GIF.

CPU-only, free to use, mobile-friendly.

**Live demo:** [aerolab-devansh.streamlit.app](https://aerolab-devansh.streamlit.app/)

## Performance

| Mode                    | Local (4+ cores) | Cloud (1 vCPU) |
|-------------------------|------------------|-----------------|
| Fast (NeuralFoil)       | instant          | instant         |
| CFD Standard (240×80)   | ~12 s            | ~1 min          |
| CFD Detailed (720×240)  | ~50 s            | ~3 min          |

First CFD click also pays a ~25 s JIT compile (~40 s on Cloud), hidden behind a startup spinner. For fast iteration use Fast mode; Real CFD is for watching.

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
pytest -q                                # 106 unit tests, ~21 s warm
python scripts/dev_validate_cfd.py       # 4 physics gates + 3 diagnostics, ~90 s
```

## Features

**Real CFD (LBM mode)**
- 5 shape presets: cylinder, square, ellipse, NACA 0012, NACA 4412
- Reynolds 50–1500, per-shape AoA / rotation, two grid presets
- MRT collision + Smagorinsky LES, Zou-He inflow/outflow, Bouzidi interpolated bounce-back
- Side-by-side pinned comparison, GIF download with parameter-encoded filenames
- Vorticity heatmap (RdBu_r) + speed-coloured RK4 streaklines (plasma) + smooth body outline + flow/scale annotations baked in

**Fast (NeuralFoil)**
- Instant lift/drag/polar for NACA airfoils, three model sizes (xxxlarge / medium / xsmall)

## Status

**Day 14 of a 12-week build.**

Phase 1 (solver core, W1–4) shipped Day 5, then expanded Days 6–14 with originally-Phase-2/3 work: MRT, Bouzidi, Zou-He, Mei momentum exchange, UI polish, repo hygiene.

**Phase 2's headline — image/SVG upload with silhouette extraction — has not started.** Multi-viz modes (pressure, streamline density) also pending. Scope is pivoting, not slipping: solver got more rigorous, but the upload feature that differentiates AeroLab from "another LBM demo" is still ahead.

| Phase | Weeks | Deliverable | Status |
|------:|------:|------|------|
| 1 — Solver core | 1–4 | LBM works, validated, deployed with 5+ shapes | ✅ Day 5; expanded to D14 |
| 2 — Shape freedom | 5–8 | **Image/SVG upload + silhouette extraction**, multi-viz, side-by-side, GIF export, gallery | Side-by-side ✅, GIF ✅, gallery ✅. **Upload + multi-viz: not started.** |
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

Artifacts in `data/`: `cylinder_convergence.png`, `validation_grid_convergence.png`, `naca0012_aoa_polar.png`.

## What this solver isn't

Shares the *collision-rule family* (MRT + Smagorinsky LES) with industrial LBM solvers (PowerFLOW, Palabos, waLBerla). That's like sharing "has four wheels" with an F1 car. We don't have:

- GPU acceleration → Re envelope tops at 1500 in 2D (industrial: Re ≥ 10⁶ on GPU clusters)
- Adaptive mesh refinement → uniform 240×80 or 720×240
- Wall-function turbulence → we resolve the boundary layer directly (only feasible at low Re)
- Cumulant collision, multi-block, automatic time-stepping, 3D, OpenFOAM/Fluent cross-validation

Every choice on the production hot path is textbook-correct. The *envelope* is firmly academic-tutorial.

## Stack

- **Solver:** NumPy reference + Numba `@njit(fastmath=True)` fused-step (collide + force + bounce-back + stream + Zou-He + Bouzidi in one function)
- **UI:** Streamlit
- **Viz:** matplotlib (LBM render), Plotly (airfoil polars), Pillow (GIF assembly), `scipy.ndimage.gaussian_filter` (smoothing)
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
│   ├── airfoils.py                 # NeuralFoil/AeroSandbox wrapper
│   └── warmup.py                   # JIT pre-warm
├── scripts/
│   ├── lid_cavity_smoke.py         # cavity benchmark
│   ├── week1_cylinder.py           # cylinder Re=100 reference run
│   ├── week1_cylinder_sweep.py     # 5-config Cd convergence study
│   ├── naca0012_aoa_polar.py       # 8-angle airfoil polar
│   ├── dev_validate_cfd.py         # 4 physics gates + 3 diagnostics
│   └── dev_grid_convergence.py     # Std vs Detailed + Richardson extrapolation
└── tests/                          # 106 unit tests
```

## License

MIT — see [LICENSE](LICENSE). PRs welcome.
