# Changelog

All notable changes to AeroLab. Dates are absolute; versions follow [SemVer](https://semver.org/).

## [0.5.0] — 2026-05-23

**Custom polygon-drawer + plain-English help overhaul** — the major pre-ship revamp before sending to senior engineers for review.

### Added
- **Click-to-place polygon drawer.** Replaces `streamlit-drawable-canvas` (which forces double-click / right-click to close) with a hand-rolled custom Streamlit component in `components/polygon_drawer/`. The UX is the standard polygon-drawing gesture: click to drop each vertex, the green start dot lights up once you have 3+ vertices, click it to close. Rubber-band preview line follows the cursor between clicks; Undo / Clear toolbar; Ctrl+Z keyboard shortcut.
- **`vertices_to_polygon()`** helper in `src/custom_shape.py` adapts the JSON vertex list from the new component into the (N, 2) polygon_xy contract the LBM pipeline already consumes. 7 new unit tests cover happy path + 5 edge cases (too few vertices, zero extent, NaN coordinates, malformed dicts, wrong type).
- **End-to-end pre-ship validation** via `scripts/final_ship_validation.py` — sweeps every (shape × viz_mode) combination at production n_frames=150, checks Cd > 0, |Cl| < 0.15 for symmetric bodies, Strouhal in (0, 1), pressure stagnation delta > 0.001 at cylinder front. 15/15 pass.
- **Plain-English captions** under every Cd / Cl / Strouhal metric tile ("Air resistance. Lower = sleeker.", etc.) so non-CFD users have an anchor.

### Changed
- **Help text overhaul throughout.** Cd / Cl / Strouhal / Re / velocity / resolution / model-quality help all rewritten to lead with intuition, not formula. Velocity slider: "slow = syrupy, fast = chaotic"; formula footnoted. Strouhal: "how often vortex pairs peel off — ~0.2 for a cylinder, which is why telephone wires can hum a steady musical note". Fast-mode Reynolds help anchors values to flight regimes (hand-glider → sailplane → light aircraft → jet airliner).
- **Strouhal "—" dash** for short runs now explains "Run was too short to spot the rhythm — try Detailed mode" inline, instead of looking like an error code.
- **JIT compile freeze** (~20 s on first click) surfaces an honest "Warming up the solver (first run takes ~20 s while the compiler does its thing — later runs are instant)" message. Session-state flag switches to a plain "Simulating the flow..." for subsequent runs.
- **Upload tab help** tightened: 10-format laundry list replaced with "most common image formats" + left-facing-shape orientation hint.
- **Building cross-section sample polygon** rotated 90° CW so wind blows across the long axis as the README always intended. The upright polygon (1:3.2 aspect) produced ~75 % vertical blockage and unphysical Cd ≈ 33; the rotated polygon gives 24 % blockage and Cd ≈ 2.6.

### Fixed
- **Polygon drawer iframe was silently 404'ing.** The `from components import polygon_drawer` import sat inside the Draw-tab `with` block, so `declare_component` never ran on a fresh page load — the iframe URL returned "Component not found" while Streamlit's catch-all served the main-app HTML with HTTP 200 (which masked the failure in smoke tests). Import moved to module top; verified via Playwright that click-place + click-start-to-close → polygon committed to session state.

### Docs
- **README:** removed "no install, no signup" claim that contradicted the Quickstart's pip dance. Drawable canvas now listed as ✅ shipped (no longer "pending"). Pressure viz no longer listed as pending. Test count 129 → 186.

## [0.4.2] — 2026-05-23

**Stability sweep + default-mode flip** — every AoA-extreme divergence the gallery cards or sliders could reach, mapped and walled off.

### Fixed
- **Diamond gallery card crashed at Re=433** (Square AoA=45°). Re=600 at AoA=45° hit a stability cliff at frame ~31 / 150 — the diagonal blockage jumps from 35 % to ~50 % and the corner-shed shear layers thin out below the LBM stability margin. Lowered Diamond card to Re=200; added AoA-aware Re cap (`Square |AoA| ≥ 5 → Re ≤ 200`).
- **Four other hidden divergences** uncovered by an exhaustive AoA × Re local stability sweep:
  - Ellipse AoA=25° Re=600 → Ellipse slider tightened to ±20°, `|AoA| > 15 → Re ≤ 400`.
  - NACA AoA=45° → NACA slider tightened to ±25°.
  - Square AoA=9.5° Re=600 → Square broadside band tightened from `|AoA| < 10` to `|AoA| < 5`.
  - Square AoA=24° Re=1000 → Square broadside cap lowered 1000 → 500.

### Changed
- **Default mode flipped to Real CFD** (`index=1`). The "see air actually move" feature is what makes AeroLab visually distinctive; the curated gallery cards give first-time visitors something compelling to click without needing to know what NeuralFoil is.
- **Brick gallery card** lowered from vel 1.80 (Re=600) to vel 1.50 (Re=500) — matches the tightened Square broadside cap.
- **Visual-regression test** `n_frames` bumped 12 → 50. At 12 frames (~420 lattice steps) the test was missing divergences that hit at step ~1000-1500 in production.

## [0.4.1] — 2026-05-23

**Rigorous solver validation** against canonical 2D bluff-body data — the answer to "how accurate is your CFD?" for senior-engineer / professor scrutiny.

### Added
- **`VALIDATION.md`** (444 lines): full methodology, 14-case Cd / Strouhal sweep against **Williamson 1996 ARFM 28** (cylinder) and **Okajima 1982 JFM 123** (square), Allen-Vincenti / West-Apelt blockage corrections with cited K-factor ranges, 5 numerically-bounded limitations sections, 7 pre-empted reviewer Q&As, 14 academic citations.
- **`scripts/validate_solver.py`** — full Re sweep harness producing `data/validation/results.{json,md}`. CI-machine-readable and human-readable.
- **`scripts/validate_conservation.py`** — closed-box mass-drift diagnostic (machine precision target) + open-channel mass-balance diagnostic (< 2 % of throughflow target). Both pass.
- **`tests/test_validation_benchmark.py`** — per-shape Cd validation gate, Cylinder-only Strouhal gate, symmetry-invariant Cl gate, no-shedding-at-low-Re gate, blockage-correction-ratio gate. 11 tests, runs on every CI push.
- **Inline textbook-comparison delta chips** on the Cd / Strouhal metric tiles, plus a "free-stream reference" `st.info` callout with the Williamson / Okajima number after Allen-Vincenti correction.

### Validated bands (full 14-case sweep)
| Quantity | Median error | Max error | Tolerance band |
|---|---|---|---|
| Cylinder Cd (Re 100–1000) | 4.3 % | 11.6 % | ± 15 % |
| Square Cd (Re 150–500) | 5.4 % | 21.8 % | ± 25 % |
| Cylinder St (Re 100–1000) | 12.6 % | 23.4 % | ± 35 % |

Square Strouhal is *diagnostic-only*: confined-channel resonance at B=0.35 locks raw St ≈ 0.37 across Re, structurally uncorrectable by any single-formula blockage correction.

## [0.4.0] — 2026-05-23

**Blockage-honest Cd reporting + per-shape Re caps + wake polish.**

### Added
- **Free-stream reference Cd / St callout.** Standard runs at 35 % channel blockage; that inflates raw Cd by ~25-40 % vs wind-tunnel free-stream. The app now surfaces BOTH numbers — the raw measurement AND the blockage-corrected free-stream estimate (Williamson / Okajima) — so the user walks away with the number they would cite in a report.
- **Per-shape Re ceilings** in the velocity slider. Cylinder 1500, Square 1000, Ellipse 1200, NACA 1500. Tied to the local stability sweep numbers; the slider can never produce a configuration that diverges.

### Changed
- **Wake-spawn point** pulled closer to the body (+6 → +3 cells past trailing edge) so the streakline mass enters the shedding region earlier — visible difference at Standard preset where the channel is short.

## [0.3.2] — 2026-05-23

**Solver hardened against high-Re bluff-body divergence.**

### Added
- **End-of-frame `np.isfinite(f).all()` blow-up guard** in the LBM main loop. Surfaces a polite "Simulation diverged at frame N of M" error instead of an opaque `ZeroDivisionError` from the `@njit` macroscopic step.
- **Pre-flight mask validation** in `solve_lbm` — rejects degenerate / thread-like polygons (single-row, isolated cells) before the JIT path can crash on them.
- **`rho_safe` / `rho_int_safe`** clamps at every macroscopic division site in `src/lbm.py`. Combined with `@njit(error_model='numpy')`, numerical underflow now produces `inf` / `NaN` instead of a Python exception that breaks out of the JIT'd inner loop.

## [0.3.1] — 2026-05-23

**Visible version marker + Cloud-deploy fixes.**

### Added
- Version string in the AeroLab wordmark header so users can tell the deployed version at a glance.

### Fixed
- CI pythonpath import resolution for tests run as `pytest` from the repo root.
- `streamlit-drawable-canvas` defensive import — if the wheel ever breaks against a minor Streamlit version bump on Cloud, the rest of the app still works and the Draw tab surfaces a polite fallback.
- `ZeroDivisionError` in the Zou-He outflow boundary for shapes that ran off the channel edge.

## [0.3.0] — 2026-05-21

**Phase 2 W5 shipped: custom shape upload — the headline differentiator.**

### Added
- **Upload PNG/JPG** as a 6th shape option in the LBM sidebar. Otsu threshold + connected-component extraction + Douglas-Peucker simplification (1 % of shorter dim tolerance) via scikit-image. Inline error messages for sanity-gate failures (image too small, shape touches edge, low contrast, area < 2 % or > 85 %).
- **Three bundled sample silhouettes** (fish, car profile, building cross-section) via "Try a sample" buttons. Closes the Phase 2 W5 gate per README ("end-to-end on three real-world silhouettes").
- **Live silhouette preview** on the LBM grid before clicking Run — shows orientation + scale + AoA-rotated outline.
- **Pin / Clear snapshot** now supports custom shapes — polygon array stashed in session state alongside the snapshot tuple.
- **Velocity (m/s) slider** replaces the Reynolds number slider as the user-facing input. Re = U·L/ν with L = 5 mm (fountain-pen scale) in standard air; slider 0.15–4.5 m/s maps to Re 50–1500. Re displayed alongside so the educational angle stays.
- **Cd-accuracy `st.info` card** post-run for custom shapes, explaining the halfway-BB caveat.
- Differentiated GIF filenames for custom shapes — sample name (e.g., `aerolab_custom_fish_re200.gif`) or polygon hash for uploads.
- `src/custom_shape.py` (silhouette extraction + polygon rasterization) and `src/sample_shapes.py` (parametric polygons for the bundled samples).
- 23 new tests (`tests/test_custom_shape.py`): synthetic-image extraction, sanity-gate rejections, rasterization, rotation, end-to-end simulate_and_render with Custom polygon, Phase 2 W5 gate tests for the 3 bundled samples. **Total: 129 passing (was 106 at 0.2.0).**

### Changed
- Hero in README is now an MP4 (`assets/hero_cylinder_re400.mp4`, LFS-tracked) instead of the 1.1 MB GIF — smaller bandwidth footprint per render and higher visual quality.
- `custom_extent` per-preset: 30 cells (Standard), 80 cells (Detailed). Reduces aliasing on thin features like a fish tail.
- Pin button no longer disabled for custom shapes.

### Dependencies
- Added `scikit-image==0.26.0` (silhouette extraction) and `streamlit-drawable-canvas==0.9.3` (planned for the hand-draw feature, Day 3 of Phase 2 W5). Pinned in both `requirements.txt` and `pyproject.toml`.

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
