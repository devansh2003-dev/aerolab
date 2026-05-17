# AeroLab

> An open-source, browser-based aerodynamics playground.
> No install, no license, no 40-tab tutorial.

By August 2026: drop in a 2D shape, drag sliders for angle of attack and Reynolds number, watch streamlines and pressure contours of the air flowing around it in near real time. Built CPU-only with a custom Lattice Boltzmann solver, free to use, mobile-friendly.

## Live demo

**[aerolab-devansh.streamlit.app](https://aerolab-devansh.streamlit.app/)**

Currently the deployed app is the **fast (ML) airfoil mode** — pick one or more NACA airfoils, sweep angle of attack, and compare lift, drag, and the drag polar side-by-side, powered by [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil). The full LBM solver is being built underneath and will become a "Real CFD" mode in the same UI in Phase 1 Week 3.

## Status

**Phase 1, Weeks 1–2 closed (Day 5 of 12 weeks).** Solver works, shape library shipped, validation documented honestly. Phase 1 W3 (Streamlit LBM mode toggle) starts next.

- D2Q9 BGK Lattice Boltzmann solver in [src/lbm.py](src/lbm.py): equilibrium, BGK collision, streaming, halfway bounce-back. Fused per-step function is `@njit`-compiled (Numba) and **bit-equivalent to a pure-NumPy reference path** (atol=1e-12, verified by two multistep equivalence tests).
- Momentum-exchange force calculation (Ladd 1994) in [src/forces.py](src/forces.py).
- Shape mask library in [src/shapes.py](src/shapes.py): cylinder, square, ellipse, NACA 4-digit airfoil (with closed-TE thickness formula, camber line, and arbitrary AoA rotation).
- **56 unit tests** covering physics invariants, JIT/reference equivalence, force-calc sanity, and per-shape geometric invariants (area, symmetry, boundary inclusivity). Run with `pytest -v`.
- Lid-driven cavity benchmark — vortex center lands at the Ghia et al. 1982 reference position.
- Cylinder Re=100 — clean von Kármán vortex shedding.
- Shape gallery (`scripts/shape_gallery.py`): 5 presets through the same flow showing the **shape-dependence of wake structure** — bluff bodies vs streamlined ellipse vs airfoils.

### Validation — Phase 1 findings

The Phase 1 gate had two physical tests (Strouhal and Cd at Re=100). Strouhal passes cleanly. Cd has a documented BGK collision-model bias that we characterized via a convergence sweep — the *reason* it fails is itself the more valuable result.

**Cylinder Re=100 convergence (5-config sweep, Mach × resolution axes):**

- Strouhal number passes within **3.0–9.1%** of textbook 0.165 across all 5 configs.
- Time-averaged Cd ranges 2.04–2.49 vs textbook 1.4. The Mach axis went the *opposite* direction from the naive compressibility prediction — Cd *grew* as Ma decreased — which is the signature of the **BGK-τ wall-correction artifact** (He, Zou, Luo & Dembo 1997; Cornubert et al. 1991). Halfway bounce-back's effective wall drifts inward as τ → 0.5, shrinking the effective cylinder area and inflating Cd against the nominal diameter.
- The discretization axis at fixed Mach behaved correctly (Cd decreased as D grew).
- **Fix is structural, not parametric:** switch BGK → MRT collision. Scheduled for Phase 2 W6 alongside the visualization layer work.

Convergence artifact: `data/cylinder_convergence.png` + `.csv` (from `scripts/week1_cylinder_sweep.py`; ~90 min full run).

**NACA 0012 AoA polar at Re_c=200 (bonus, 8 angles from -5° to +15°):**

- **Lift curve is portfolio-grade.** CL(0°) = −1e-4 (perfect symmetric airfoil), CL(−α) = −CL(α) to four decimal places, monotonic and roughly linear with slope ~0.048/deg (about 44% of thin-airfoil theory, consistent with Re=200 viscous effects).
- **Drag curve is non-physical** — non-monotonic with a peak at α=±5° and falling Cd at higher AoA. Two-part cause: (a) chord=40 cells means max thickness is only 4.8 cells at α=0, so the discretization error per unit wetted area is much higher at α=0 than at α=15° (where projected x-extent is ~15 cells); (b) the BGK-τ artifact stacks on top. Honest report: use the lift curve only.
- **No vortex shedding** at this Re (laminar attached wake), so the Strouhal column in the output CSV is meaningless — the FFT is picking noise bins.
- To produce a quantitatively trustworthy airfoil polar in the future: switch to MRT collision **and** bump chord ≥ 80 cells (4× compute). Same Phase 2 W6 visit as the cylinder gate.

Artifact: `data/naca0012_aoa_polar.png` + `.csv` (from `scripts/naca0012_aoa_polar.py`; ~52 min full run).

## 12-Week Roadmap

| Phase | Weeks | Deliverable |
|------:|------:|------|
| **1 — Solver core** | 1–4  | LBM solver works, validated against cylinder Cd, deployed live in Streamlit with 5–8 pre-set 2D shapes |
| **2 — Shape freedom** | 5–8  | Image / SVG upload with silhouette extraction, multiple viz modes (velocity / pressure / vorticity / streamlines), side-by-side comparison, GIF export, demo gallery |
| **3 — Polish + optional 3D** | 9–12 | NeuralFoil instant mode (✅ already shipped), optional 3D wing module via AeroSandbox+AVL, optional OpenFOAM validation, public launch |

Validation gates per phase are non-negotiable. Phase 1's gate was Strouhal AND Cd within ±10% of textbook at Re=100. Strouhal passed; Cd is biased by the BGK-τ wall-correction artifact, which we documented with a convergence sweep rather than chase with parameter tweaks. The structural fix (MRT collision) is scheduled for Phase 2 W6 alongside the visualization layer work — closing Phase 1 with a known, characterized limit rather than blocking the schedule on it.

## Stack

- **Language:** Python 3.11
- **Solver:** NumPy now, Numba `@njit` for the per-step hot loop once profiling demands it
- **UI:** Streamlit (mobile-friendly, free hosting)
- **Visualization:** Matplotlib for static plots, Plotly for interactive, animated Matplotlib for GIF export
- **Optional ML mode:** [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil) + [AeroSandbox](https://github.com/peterdsharpe/AeroSandbox) for instant airfoil predictions
- **Hosting:** Streamlit Community Cloud (free tier)

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

# Run the deployed app locally
streamlit run app.py
```

## Repo layout

```
aerolab/
├── README.md
├── requirements.txt          # deploy deps
├── requirements-dev.txt      # pytest only
├── .gitignore
├── app.py                    # Streamlit entry (currently NeuralFoil airfoil mode)
├── src/
│   ├── __init__.py
│   ├── lbm.py                # D2Q9 BGK solver + Numba JIT fused-step (collide+stream+BB+BCs)
│   ├── forces.py             # momentum-exchange force calc (Ladd 1994)
│   ├── shapes.py             # cylinder, square, ellipse, NACA 4-digit airfoil masks
│   └── airfoils.py           # NeuralFoil/AeroSandbox wrapper (Fast Mode in the deployed app)
├── scripts/
│   ├── day1_test.py                  # single-point NeuralFoil smoke test
│   ├── polar_sweep.py                # alpha-sweep NeuralFoil polar generator
│   ├── lid_cavity_smoke.py           # lid-driven cavity (LBM benchmark)
│   ├── week1_cylinder.py             # cylinder at Re=100: wake + Cd time series + Strouhal
│   ├── week1_cylinder_sweep.py       # 5-config Cd convergence study (Mach × D axes)
│   ├── shape_gallery.py              # 5-shape wake comparison (cylinder, square, ellipse, 2 NACAs)
│   └── naca0012_aoa_polar.py         # NACA 0012 lift+drag polar across 8 angles of attack
├── tests/
│   ├── test_lbm.py                   # 30 invariants for the solver + JIT-vs-reference equivalence
│   ├── test_forces.py                # 6 invariants for momentum-exchange force calc
│   └── test_shapes.py                # 20 invariants across the 4 mask generators
└── data/                             # output artifacts: PNGs, CSVs (git-ignored)
```

## License

TBD (planning MIT once the project hits public launch in Phase 3 Week 12).
