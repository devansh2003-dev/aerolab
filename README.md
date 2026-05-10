# AeroLab

> An open-source, browser-based aerodynamics playground.
> No install, no license, no 40-tab tutorial.

By August 2026: drop in a 2D shape, drag sliders for angle of attack and Reynolds number, watch streamlines and pressure contours of the air flowing around it in near real time. Built CPU-only with a custom Lattice Boltzmann solver, free to use, mobile-friendly.

## Live demo

**[aerolab-devansh.streamlit.app](https://aerolab-devansh.streamlit.app/)**

Currently the deployed app is the **fast (ML) airfoil mode** — pick one or more NACA airfoils, sweep angle of attack, and compare lift, drag, and the drag polar side-by-side, powered by [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil). The full LBM solver is being built underneath and will become a "Real CFD" mode in the same UI in Phase 1 Week 3.

## Status

**Phase 1, Week 1 — solver core complete (Day 3 of 12 weeks).**

- D2Q9 BGK Lattice Boltzmann solver in [src/lbm.py](src/lbm.py): equilibrium, BGK collision, streaming, halfway bounce-back
- 33 unit tests covering physics invariants (mass/momentum conservation, lattice symmetry, relaxation rate)
- Lid-driven cavity benchmark validates qualitatively against Ghia et al. 1982
- Cylinder simulation at Re=100 produces correct **von Kármán vortex shedding**
- *Pending Phase 1 acceptance gate:* time-averaged Cd within ±10% of textbook value (~1.4 at Re=100). Force calculation via momentum exchange is the next session's first task.

## 12-Week Roadmap

| Phase | Weeks | Deliverable |
|------:|------:|------|
| **1 — Solver core** | 1–4  | LBM solver works, validated against cylinder Cd, deployed live in Streamlit with 5–8 pre-set 2D shapes |
| **2 — Shape freedom** | 5–8  | Image / SVG upload with silhouette extraction, multiple viz modes (velocity / pressure / vorticity / streamlines), side-by-side comparison, GIF export, demo gallery |
| **3 — Polish + optional 3D** | 9–12 | NeuralFoil instant mode (✅ already shipped), optional 3D wing module via AeroSandbox+AVL, optional OpenFOAM validation, public launch |

Validation gates per phase are non-negotiable. Phase 1 doesn't end until cylinder Cd matches the textbook within ±10%.

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
pytest -v                                # 33 unit tests, ~1 s
python scripts/day1_test.py              # NeuralFoil airfoil prediction
python scripts/lid_cavity_smoke.py       # LBM cavity benchmark, ~25 s
python scripts/week1_cylinder.py         # LBM cylinder + vortex shedding, ~160 s

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
│   ├── lbm.py                # D2Q9 BGK solver: equilibrium, collide, stream, bounce-back
│   ├── shapes.py             # cylinder_mask (more shapes coming Week 2)
│   └── airfoils.py           # NeuralFoil/AeroSandbox wrapper for the deployed app
├── scripts/
│   ├── day1_test.py          # single-point NeuralFoil smoke test
│   ├── polar_sweep.py        # alpha-sweep polar generator
│   ├── lid_cavity_smoke.py   # lid-driven cavity (LBM benchmark)
│   └── week1_cylinder.py     # cylinder in channel, visual wake
├── tests/
│   ├── test_lbm.py           # 28 invariants for the solver
│   └── test_shapes.py        # 5 invariants for the shape masks
└── data/                     # output artifacts (git-ignored)
```

## License

TBD (planning MIT once the project hits public launch in Phase 3 Week 12).
