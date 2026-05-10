# AeroLab

> An open-source, browser-based aerodynamics playground.
> No install, no license, no 40-tab tutorial.

Pick an airfoil, drag a slider for angle of attack, and instantly see lift, drag, and pressure characteristics in your browser.

## Live demo

**[aerolab-devansh.streamlit.app](https://aerolab-devansh.streamlit.app/)**

Type one or more NACA airfoils, sweep angle of attack, and compare lift, drag, and the drag polar side-by-side.

## Status

**Day 2** — deployed live on Streamlit Community Cloud. Comparison mode shipped (overlay multiple airfoils), reusable `analyze_airfoil()` helper in `src/airfoils.py`, polar sweep script with PNG export.

## 12-Week Roadmap

| Month | Weeks | Goal |
|------:|------:|------|
| 1 | 1–4  | Airfoil playground deployed live: NeuralFoil predictions, comparison mode, polar plots |
| 2 | 5–8  | Custom 2D Lattice Boltzmann solver in Python + Numba for arbitrary 2D shapes |
| 3 | 9–12 | 3D wing design via AeroSandbox + AVL, mission-to-design workflow, OpenFOAM validation pipeline |

## Stack

- **Language:** Python 3.11
- **UI:** Streamlit
- **Aerodynamics:** AeroSandbox, NeuralFoil
- **Performance:** Numba (JIT for the LBM solver)
- **Plotting:** Matplotlib, Plotly
- **Hosting:** Streamlit Community Cloud (free)

## Local setup

```powershell
conda create -n aerolab python=3.11 -y
conda activate aerolab
pip install -r requirements.txt
python scripts/day1_test.py
```

## Repo layout

```
aerolab/
├── README.md
├── requirements.txt
├── .gitignore
├── src/             # analysis + viz modules (grows over time)
├── scripts/         # one-off test scripts and experiments
└── data/            # cached results, airfoil files (git-ignored)
```

## License

TBD (planning MIT once the repo goes public).
