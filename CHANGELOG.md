# Changelog

All notable changes to AeroLab. Dates are absolute; versions follow [SemVer](https://semver.org/).

## [0.6.2] — 2026-05-29

**3D gallery polish + sphere Cd error-budget falsification.**

Two threads:

1. **3D viewer fixes**: streamlines no longer clip into the body (mesh now follows the snapped baked AoA, not the continuous slider), streamlines no longer cut off mid-flow on slow regions (`dt` 12 → 8, `max_steps` 400 → 700, 3-iteration binary-search body-collision snap), pressure became actual gauge pressure with a symmetric diverging colormap, velocity color now sampled at the polyline vertex.
2. **Sphere Cd low-blockage cross-check**: ran the experiment §8.3 of VALIDATION.md proposed. Halving blockage from 42 % to 25 % was supposed to drop Cd from 1.57 toward ~1.30. It didn't — Cd ticked UP to 1.65. The blockage hypothesis is falsified. Revised error budget points at simplified Ladd 1994 momentum exchange + D = 20 grid resolution as the prime suspects instead.

### Added

- **`scripts/validate_3d_sphere_cd_lowblock.py`** — same physics as the high-blockage validation but on a 160 × 80 × 80 grid (blockage 42 % → 25 %, D unchanged at 20). 1100 s solve time.
- **`tests/test_validation_3d_sphere_cd_lowblock.py`** (8 gates) — same physics-validity checks as the high-blockage gate plus two cross-check gates: `test_blockage_is_not_dominant_bias` (locks in that the two Cd values land within 15 % of each other, i.e. blockage is NOT the dominant gap) and `test_blockage_is_lower_than_shipped_bake` (sanity check on grid).
- **VALIDATION.md §8.3.1** — full low-blockage results table side-by-side with the high-blockage measurement, narrative of what the experiment was supposed to show vs what it actually showed, revised error budget.
- **`data/validation_3d_sphere_re100_lowblock.json`** — measured result, committed.

### Changed

- **3D body mesh (cube + NACA)** rendered at the snapped baked AoA, not the continuous slider value. With the mesh disagreeing with the flow field by up to 8°, streamlines were cutting visibly through the rotated body. Aligning the mesh to `aoa_actual` eliminates the clipping. The slider caption already announces the snap, so the discrete jump is honest.
- **`_trace_streamlines` defaults** in `app.py`: `dt` 12.0 → 8.0 (each step now covers ~0.32 cell instead of ~0.48; polyline tracks curvature more closely and doesn't chord-cut through curved bodies), `max_steps` 400 → 700 (streamlines reach the outlet even through low-speed wake regions where each step covers far less than 0.32 cell).
- **Body-collision snap** upgraded from a single bisection to a 3-iteration binary search; endpoint now lands within ~`dt·|u| / 8` (~0.04 cell at `u_in=0.04`) of the analytic wall instead of the previous ~0.12 cell.
- **3D Pressure mode** now plots **gauge pressure** `p = c_s²·(ρ − ρ_ref)` with `c_s² = 1/3` (D3Q19) and `ρ_ref = median(ρ_fluid)`. Colorbar is symmetric about zero, so RdBu_r reads as **red = stagnation, blue = suction, white = freestream** instead of the previous near-flat raw-density slice (Δρ ≈ 5e-3 was being percentile-stretched into a monochrome lump). The caption and colorbar title updated accordingly.
- **3D Velocity mode** colors via vertex-trilerp of `|u|` over the field, matching the Vorticity / Pressure pattern. Previously it reused the RK2 midpoint speed which lagged the geometry by half a step.
- **VALIDATION.md §8.3** — added a status callout up top noting the budget below was falsified by §8.3.1; the original budget is kept as a record of the prediction the cross-check refuted.
- **VALIDATION.md §8.7 priority list** — "Low-blockage sphere Cd sweep" moved to the **Closed** section with a note explaining why (hypothesis refuted, not confirmed). "Mei-Yu-Shyy-Luo 2002 Bouzidi-aware momentum exchange + D ≥ 40 sphere bake" promoted to item #1.

## [0.6.1] — 2026-05-29

**Sphere Re=100 drag validation — first quantitative 3D Cd comparison.**

Closes review item **V1** (one canonical 3D drag number, the gap that converted the 3D side from "no quantitative validation" to "validated for one canonical case"). The measured Cd lands recognisably in the published band with a systematic positive bias dominated by 42 % blockage — physical, signed correctly, axisymmetric forces vanish to 10⁻⁴, and the error budget is explainable and documented.

### Added

- **`src/forces_3d.py`** — 3D D3Q19 analogue of `src/forces.py`. Ladd 1994 momentum exchange with two API entry points: `momentum_exchange_force_3d(f_post_collision, body)` for use inside a step kernel, and `momentum_exchange_force_3d_post_stream(f_post_stream, body)` for the natural exit state of `run_channel_smoke_trt` (reads the opposite-direction slot at each fluid cell, accounting for the halfway bounce-back reflection that happens during the step). Plus `drag_coefficient_3d` for Cd from F_drag, ρ_ref, U_ref, A_proj.
- **`scripts/validate_3d_sphere_cd.py`** — runs the sphere preset for 2 500 steps (5 D/u, past startup), computes Cd via post-stream momentum exchange, compares to Clift-Grace-Weber 1978 (Cd ≈ 1.09), writes JSON. Measured Cd = **1.57** (raw); error +44 % vs free-stream reference, dominated by the 42 % blockage bias documented in VALIDATION.md §8.3.
- **`tests/test_validation_3d_sphere_cd.py`** (6 gates) — reads the committed JSON and asserts: Cd inside the tolerance band, drag positive (downstream), Cd in the broad physical envelope [0.4, 3.0], `|F_lift|/|F_drag|` < 5 %, `|F_side|/|F_drag|` < 5 %, mass drift < 1 %, advective times ≥ 4 D/u. All pass.
- **VALIDATION.md §8.3** — full results table, error budget broken down (blockage / halfway BB / grid resolution / finite advective time), reproduction commands, and an explicit "why the test passes a +44 % error" subsection so reviewers see the answer to the obvious follow-up question.
- **`data/validation_3d_sphere_re100.json`** — the measured result, committed (unlike most of `data/*` which is gitignored). The JSON keeps VALIDATION.md's "see the numbers I cite" promise: anyone can read the source of the published 1.57.

### Changed

- **`run_channel_smoke_trt`** now accepts `return_populations=True` to return the final population array `f` for downstream force / Cd computations. Backwards-compatible: the bake script and existing callers continue to receive the legacy 5-tuple.
- **VALIDATION.md §8.4–8.7** renumbered. Previous §8.3 ("What we do NOT validate") becomes §8.4 with the sphere Cd item removed (now §8.3); the priority list in §8.7 swaps "Sphere Re=100 Cd" for "Low-blockage sphere Cd sweep" as the next step (close the systematic gap, not the canonical comparison — that one is now made).

## [0.6.0] — 2026-05-29

**3D gallery (preview) ships.** A pre-baked D3Q19 TRT field replay running alongside the 2D playground via the sidebar's "Solver tab" radio. Cloud-safe: the kernel runs offline (~10–25 s per scene on a laptop), the saved `.npz` snapshots ship in `data/baked/`, the hosted app loads them and renders interactive 3D streamlines + a solid body mesh. No live 3D solve in the browser.

### Added

- **3D solver core** (`src/lbm_3d_bouzidi.py`, `src/lbm_3d_trt.py`, `src/lbm_3d_qcriterion.py`, `src/lbm_3d_smoke_particles.py`, `src/baked_fields.py`). D3Q19 with TRT collision (Λ = 3/16 magic parameter), Bouzidi interpolated bounce-back with analytic q-field for spheres, voxelised wall links for everything else, Guo NEEM inflow and regularised Latt-Chopard outflow. Validated against the Taylor-Green vortex decay rate to ±2 % vs the analytic 4νk² exponent (`tests/test_phase1_gate_trt_tgv_decay_rate`, gated in CI).
- **10 pre-baked 3D scenes**, all at u_in = 0.04 lattice units, regularised outflow:
  - Sphere, cylinder (spanwise z), cube at Re ∈ {40, 100}
  - NACA 0012 (symmetric) and NACA 4412 (cambered) at Re ∈ {40, 100} × AoA ∈ {0°, 10°}
- **3D gallery UI** (`if view == "3D gallery (preview)":` block in `app.py`). Sidebar mirrors the 2D playground layout exactly: Simulation setup heading, First-time expander, Shape selectbox, Flow speed slider (0.10–4.50 m/s, same `L = 5 mm` convention as 2D), AoA slider (NACA only), Color picker (Velocity / Vorticity / Pressure), top-down body silhouette preview, Streamlines (density / thickness / animate flow), Overlays (body / wind-tunnel chamber / Q-criterion shell).
- **Server-side RK2 streamline tracing** (`_trace_streamlines` in `app.py`). Vectorised numba-jitted trilinear interpolation; ~50–150 ms per scene for 25–96 seeds × 400 steps. Replaces the earlier Plotly `go.Streamtube` (which did the integration in JavaScript on the browser main thread and blocked the UI for 3–8 s on scene-swap).
- **Body-collision midpoint snap** in the tracer so streamlines visibly graze the body surface instead of stopping ~0.5 lattice cells short.
- **Animated growing streamlines.** Each cycle: head holds at outflow while tail sweeps from inflow forward (drain), then tail stays at inflow while head sweeps to outflow (grow). 60 frames at 80 ms per frame with 60 ms linear transition; per-streamline phase stagger so streaks don't all march in lockstep. Camera rotation stays live during playback via `uirevision` tokens on layout + scene (Plotly preserves view-state across redraws bound to the same token).
- **Curated 3D gallery cards** below the chart: *How a wing lifts* (NACA 4412 + pressure coloring), *Wing at zero AoA*, *Where the air spins* (cylinder + vorticity), *Bluff cube*, *Almost stopped (creep)*, *Sphere wake*.
- **3D bake script** (`scripts/bake_3d_field.py`) with `--preset <name> --out data/baked`. Each `.npz` carries the preset config, body mask, float16-quantised velocity / density arrays, and a SHA-256 hash of the (config + mask + final field) for reproducibility.
- **`naca_outline()`** and **`make_naca_mask()`** in `src/lbm_3d_bouzidi.py`. Analytic NACA 4-digit thickness + camber line, AoA-rotated about the chord midpoint, used by both the bake-time mask voxeliser and the runtime body-mesh renderer.
- **3D validation section** in `VALIDATION.md` (§8). What 3D is and isn't validated against, baked-scene parameters table, blockage and advective-times disclosure, reproduction commands.
- **Hero chip becomes contextual.** "3D gallery · preview, in sidebar →" while in 2D view, "3D gallery · preview · live" (green) while in 3D — no stale pointer.

### Changed

- **Velocity → Re mapping in the 3D playground uses the same `L = 5 mm` convention as 2D.** Previously a 0.5 mm characteristic length compressed the slider into the {40, 100} bake range, which produced a confusing mismatch ("in 2D this was Re 500 at 1.5 m/s, why is 3D calling it Re 50?"). Now 1.5 m/s reads as nominal Re 500 in both solvers; the 3D readout shows BOTH the nominal Re from the slider AND the snapped baked Re below it whenever they differ, so users see the gap honestly.
- **2D Resolution radio labels** now state the trade-off, not just the grid size. "Standard (320 x 80) — faster, ~40 s local" and "Detailed (960 x 240) — sharper wake, ~100 s local".
- **2D Solver tab radio** (sidebar) gates 2D vs 3D. The earlier developer-only "3D dev bench (local)" view was removed — 987 lines of dead code path retired.

### Fixed

- **3D NACA preview crashed when an older `lbm_3d_bouzidi.py` was on the deploy.** Refactored the preview and 3D body render to call `naca_outline()` with positional-only arguments and apply the chord-midpoint rotation inline — no longer depends on the deployed module exposing the `aoa_deg` kwarg.
- **3D shape selector tooltip** previously said "NACA wings + custom-upload support are queued for the next release" even after NACA shipped. Rewritten to reflect what's live (NACA 0012 / 4412 at AoA = 0° and 10°); only Custom STL/PNG upload is now flagged as deferred.
- **3D animation rotation was locked during playback** in an interim build (used `redraw=False` to preserve camera, but Plotly's 3D Scatter geometry updates require `redraw=True` to actually paint). Restored `redraw=True` and added the `uirevision` tokens so camera state survives redraws.
- **Filename parser** for 3D bake files used a `[a-zA-Z]+` regex that silently dropped shapes whose names contain digits (`naca0012`, `naca4412`). Replaced with an `rfind('_re')` / `rfind('_aoa')` split that handles arbitrary `<shape>[_aoa<deg>]_re<N>` patterns.

### Docs

- **README "Status"** updated to reflect the shipped 3D gallery. Phase 3 row in the roadmap table now shows "3D gallery (D3Q19 TRT, 10 pre-baked scenes) ✅" alongside the OpenFOAM-pending mark.
- **README "What this solver isn't"** clarified — Live 3D solve on Cloud explicitly called out as outside scope; the offline-bake / Cloud-replay split documented.
- **VALIDATION.md** gains §8 covering 3D bake parameters, the TGV decay-gate validation, and an explicit "what we do NOT validate in 3D" list (no Cd, no Strouhal, no NACA polar, no OpenFOAM cross-validation).

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
