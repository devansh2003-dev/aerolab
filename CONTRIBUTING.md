# Contributing to AeroLab

PRs and issues welcome. This is a learning project as much as a tool, so the bar is "does it move things forward" rather than perfection.

## Setup

```powershell
conda create -n aerolab python=3.11 -y
conda activate aerolab
pip install -r requirements.txt
pip install -r requirements-dev.txt
pytest -q          # should pass 106 tests in ~21 s warm
streamlit run app.py
```

Tested on Python 3.11 + Windows 11 (the maintainer's environment). Should work on Linux/macOS but cold-cache JIT compile times will differ.

## What kind of PRs are wanted

- **Bug fixes** with a regression test in `tests/`. New tests next to existing tests for the same module.
- **Physics improvements** — better BCs, validation cases, additional shape generators. Include the citation + a unit test asserting the expected invariant.
- **UX improvements** to `app.py` — clearer copy, better defaults, mobile fixes. Include before/after screenshots in the PR description.
- **Performance work** — Numba tuning, vectorisation, AOT compile experiments. Include a measurement script or benchmark numbers in the PR description.
- **Docs / README clarity** — always welcome.

## What's out of scope (right now)

- 3D solver work — Phase 3, not yet started. Open an issue to discuss before opening a PR.
- Cloud-hosting alternatives (AWS, Vercel, etc.) — the Streamlit Community Cloud deployment is the production target.
- GPU paths — explicit project constraint (CPU-only, free-tier deployable).

## Code style

- Match the surrounding code. The codebase mostly uses 4-space indent, no formatter enforced.
- Comments are sparse. Only add one when the WHY is non-obvious (a hidden invariant, a citation, a workaround for a specific bug). Don't restate what the code does.
- New physics constants need a citation and ideally a `scripts/dev_*.py` validation script demonstrating the choice.

## Running the validation suite

For solver-touching PRs, run at least the fast validation:

```powershell
python scripts/dev_validate_cfd.py        # ~90 s, 4 local-physics gates + 3 global diagnostics
```

For BC or wall-treatment changes, also run:

```powershell
python scripts/dev_grid_convergence.py    # ~6 min, Richardson extrapolation
```

For changes that could affect long-run stability:

```powershell
python scripts/week1_cylinder_sweep.py    # ~90 min, 5-config Cd convergence sweep
```

Include the relevant artifact (PNG/CSV under `data/`) in the PR description, not in the repo (data/ is gitignored).

## Deploying

The live demo at [aerolab-devansh.streamlit.app](https://aerolab-devansh.streamlit.app/) is hosted on Streamlit Community Cloud, which **auto-redeploys on every push to `main`**. The deploy workflow:

1. **Run tests locally** — `pytest -q`. Don't push red.
2. **Smoke-test locally** — `streamlit run app.py`, click through both modes, verify the CFD mode runs cleanly on a Standard-grid cylinder Re=400 AoA=0 and on a Detailed-grid NACA 4412 Re=1000 AoA=10.
3. **Commit + push to `main`**. Streamlit Cloud picks up the change within 1–3 minutes.
4. **Smoke-test the deployed URL** on the same two canonical configs from step 2. The first CFD click on a fresh Cloud container pays a ~40 s JIT compile cost (no separate warmup step — it's amortized into the first run). If the first click takes much longer than that, check the Cloud logs.
5. **Only then** consider the deploy good. If something is off, push a revert before opening any PR / pinging anyone about the live demo.

## Filing issues

Include:
- What you tried (full command, shape preset, Re, AoA, resolution if it's a CFD-mode bug)
- What you expected
- What you got (error message + traceback, or a screenshot for visual issues)
- Your platform (OS, Python version, GPU/no-GPU)

## Maintainer

This is a solo project run by [@devansh2003-dev](https://github.com/devansh2003-dev). PR review turnaround is best-effort — expect a few days, not a few hours.
