# Changelog

All notable changes to AeroLab. Dates are absolute; versions follow [SemVer](https://semver.org/).

## [0.2.1] — 2026-05-19

**Wake-streakline polish.** Particle visualization now reads correctly from the very first frame and trails off gradually on the Detailed preset.

### Changed
- Both presets bumped from 60 / 100 frames to **150 frames** (5250 LBM steps each). Standard ~12 s → ~30 s locally; Detailed ~50 s → ~75 s locally. Cloud roughly 2.5× longer (free tier is 1 vCPU).
- `MAX_AGE` 60 → 100. Particles live longer, so the trail past `wake_x_max` extends gradually instead of cutting off — visible mostly on Detailed where the channel is 3× longer.
- Wake-spawn box extends to `LBM_NX − 30` (fixed cell buffer) instead of `0.78 × LBM_NX` (proportional). On Standard this barely moves; on Detailed it eliminates a 150-cell empty zone past the body.
- Detailed `gif_palette` 128 → 96 to absorb the +50% frame count without ballooning GIF size.
- Pin button no longer drops the GIF after click. Tracked the last-displayed config in session state so the post-run controls survive Streamlit's rerun.

### Fixed
- Wake particles spawned at frame 0 produced a visual artifact ("particles teleport behind the body"). Wake-spawn count now ramps from 0 to full over the frames it physically takes an inflow particle to reach the wake region.

## [0.2.0] — 2026-05-19

**Cloud-readiness pass + public-release polish.** Production solver is now MRT + Bouzidi + Zou-He end-to-end; Streamlit Cloud deploys cleanly; docs and code-comments are aligned with what actually ships.

### Added
- MRT collision + Smagorinsky LES (`C_SMAG=0.17`, Lilly 1967) as the production hot path; stable Re=50–1500 across all 5 shape presets.
- Bouzidi-Firdaouss-Lallemand interpolated bounce-back at the body surface (replaces halfway BB). Cd overshoot vs textbook cut from ~89% to ~37%; Standard↔Detailed Strouhal disagreement from ~240% to ~3% (grid-converged).
- Zou & He velocity-inflow + pressure-outflow BCs. Long-run mass drift cut from ~3%/1k steps to ~0.1%/1k steps.
- Mei-Yu-Shyy-Luo Bouzidi-aware momentum exchange for body-force calc.
- 4 shape presets beyond cylinder (square, ellipse, NACA 0012, NACA 4412) with analytic q-fields for Bouzidi.
- Side-by-side pinned comparison, GIF download with parameter-encoded filenames, demo gallery script.
- Visual-regression test (`tests/test_visual_regression.py`) — canonical cylinder Re=400 frame fingerprint.
- Validation infrastructure: `dev_validate_cfd.py` (4 physics gates + 3 diagnostics), `dev_grid_convergence.py` (Richardson extrapolation), `week1_cylinder_sweep.py` (5-config Cd convergence), `naca0012_aoa_polar.py` (8-AoA polar).
- LICENSE, CONTRIBUTING.md, pyproject.toml, devcontainer, GitHub Actions test workflow.
- Hero GIF (cylinder Re=400) via Git LFS.

### Changed
- Standard resolution preset 320×100 → 240×80 (44% fewer cells) for Cloud free-tier wall-time. Body sizes scaled so blockage stays ~20%. Detailed unchanged.
- `STEPS_PER_FRAME` 50 → 35; Standard `n_frames` 100 → 60; render DPI 100 → 88. Cloud wait time roughly halved.
- README rewritten to 135 lines, scannable structure with performance table.
- Pinned `NUMBA_NUM_THREADS=16` at line 1 of `app.py` to match Cloud's post-init reset (avoids `reload_config` mismatch crash).

### Removed
- `parallel=True` / `cache=True` on the JIT step functions (Streamlit Cloud env conflict). `prange` aliased to `range`. Local loses 2-3× speedup; Cloud was already serial.
- `src/warmup.py` (dead since the Numba-thread debug saga — first user click now amortizes the JIT compile).
- Stale halfway-BB sharp-corner warning in `app.py` (Bouzidi now active for rotated squares/ellipses).

### Fixed
- BGK-τ wall-correction artifact characterized via 5-config convergence sweep (He–Zou–Luo–Dembo 1997; Cornubert et al. 1991). Documented in Validation section; structural fixes shipped via MRT + Bouzidi.

## [0.1.0] — 2026-05-05

Initial Phase 1 ship: D2Q9 BGK + halfway bounce-back, Streamlit dual-mode app (NeuralFoil Fast + LBM Real CFD), cylinder Re=100 von Kármán validation, lid-driven cavity benchmark against Ghia 1982.
