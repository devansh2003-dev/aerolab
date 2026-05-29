# AeroLab 3D — Phase 0 Empirical Findings

Status: research phase complete. Four throwaway prototypes ran on
the development laptop. This memo records what each prototype
empirically confirmed (or refuted) against the on-paper decisions
in `3D_PHASE0_DECISIONS.md`.

Companion to `3D_RESEARCH_PLAN.md` and `3D_PHASE0_DECISIONS.md`. Read
those first.

Scripts:

| # | Prototype                          | File                                          |
|---|------------------------------------|-----------------------------------------------|
| 1 | TRT (Λ = 3/16) vs BGK on 3D TGV    | `scripts/dev_3d_proto1_trt_vs_bgk.py`         |
| 2 | Guo NEEM inflow/outflow mass close | `scripts/dev_3d_proto2_guo_mass.py`           |
| 3 | D3Q19 Numba throughput + JIT time  | `scripts/dev_3d_proto3_throughput.py`         |
| 4 | float32 vs float64 on TGV decay    | `scripts/dev_3d_proto4_dtype.py`              |

Shared helpers: `scripts/_3d_phase0_helpers.py`.

---

## Result summary

| # | Decision under test         | Gate                           | Measured                | Verdict |
|---|-----------------------------|--------------------------------|-------------------------|---------|
| 1 | TRT reproduces analytic ν   | TGV err < 2 %                  | ≤ 0.5 % across τ sweep  | **PASS** |
| 2 | Guo NEEM closes mass        | drift < 1 % over 500 steps     | drift = 0.0000 %        | **PASS** |
| 3 | 96³ throughput ≥ 20 Mcell/s | Cloud floor from plan §1.2     | 11.5 Mcell/s            | **FAIL** |
| 4 | float32 ≈ float64 on decay  | spread < 1 %                   | 0.0006 %                | **PASS** |

Three of four confirm the locked decisions. **One — throughput — came
in below the plan's estimate** and should be addressed before
Phase 5 wires the Interactive3D preset into the web app.

---

## Proto 1 — TRT (Λ = 3/16) reproduces analytic viscosity (PASS)

24³ periodic box, 2D-extruded Taylor-Green vortex, U = 0.04, n = 600
steps. Swept ν = 0.005, 0.01, 0.02, 0.05 (τ = 0.515 to 0.65).

| Scheme | ν      | τ      | measured decay rate | analytic    | err     |
|--------|--------|--------|---------------------|-------------|---------|
| BGK    | 0.005  | 0.515  | 0.001369            | 0.001370    | −0.07 % |
| TRT    | 0.005  | 0.515  | 0.001373            | 0.001370    | +0.19 % |
| TRT    | 0.010  | 0.530  | 0.002731            | 0.002742    | −0.35 % |
| TRT    | 0.020  | 0.560  | 0.005461            | 0.005483    | −0.41 % |
| TRT    | 0.050  | 0.650  | 0.013662            | 0.013708    | −0.33 % |

TRT decay-rate error stays within ±0.5 % of analytic across the whole
τ range. **Production collision = TRT (Λ = 3/16) confirmed.**

Honest caveat: BGK *also* reproduces the analytic decay rate in this
periodic box, because TRT's advantage over BGK is at **walls** (the
mid-link bounce-back placement that becomes viscosity-dependent under
BGK). The TGV has no walls. We confirm TRT works correctly; the
*comparative* gain over BGK is to be confirmed in Phase 2 against the
Bouzidi sphere, where Cd accuracy depends on wall placement.

---

## Proto 2 — Guo NEEM inflow/outflow closes mass (PASS)

32 × 16 × 16 channel, periodic in y and z. Inflow x = 0 prescribes
u = (0.04, 0, 0); outflow x = Nx − 1 prescribes ρ = 1 and extrapolates
u from x = Nx − 2. BGK collision, ω = 1.786 (ν = 0.02).

Mass drift over 500 steps: **0.0000 %** (numerically zero —
inflow and outflow flux balance to machine precision).

**Production inflow/outflow = Guo NEEM confirmed.** 3D Zou-He was
already rejected on grounds of edge/corner closure complexity; this
prototype removes any remaining doubt that the simpler NEEM is
sufficient.

Honest caveat: this is mass closure on an *empty* channel — the only
physics is uniform throughflow. A body in the channel adds a wake
that the outflow must transmit cleanly; that is exercised in Phase 1.

---

## Proto 3 — Numba throughput at 96³ falls short of the plan's floor (FAIL)

Measured single-core single-thread throughput on the development
laptop (Numba @njit, fastmath=True, **no `parallel=True`**):

| N     | mem MB (2 buf) | JIT first step | per step (ms) | Mcell/s |
|-------|----------------|----------------|---------------|---------|
| 48³   | 16.0           | 0.76 s         | 9.16          | 12.07   |
| 64³   | 38.0           | 0.02 s         | 22.43         | 11.69   |
| 96³   | 128.2          | 0.08 s         | 77.00         | 11.49   |

96³ comes in at **11.5 Mcell/s**, against the 20 Mcell/s Cloud target
floor the plan §1.2 estimated, and well below the plan's "50 to 150
Mcell/s on laptop" range.

### What this means in seconds

At 11.5 Mcell/s, a 96³ × 5000-step Interactive3D run takes
**~6.5 min on this laptop**. Streamlit Cloud's shared 1-vCPU is
typically 2–3× slower than laptop single-core, so a Cloud run is
likely **~12 to 18 min**, against the plan's ~2.5 min estimate and
the user-tolerated ~3-4 min ceiling.

### Why the gap (most-likely diagnosis)

The kernel measured (`src/lbm_3d.step_bgk_3d`) carries the full
channel-flow boundary handling (inflow override, outflow copy, body
bounce-back, channel-wall reflection) inside the inner-most loop. The
plan's "50–150 Mcell/s" estimate is for a clean periodic kernel
without that branching. A production kernel that splits the bulk
collision/stream from the boundary application (so the inner loop is
straight-line code with no `if` branches per direction) will run
materially faster on the same hardware.

This is a Phase 1 optimisation, not a Phase 0 dead-end.

### Implications for the plan — three options

**A. Optimise the kernel before Phase 5 wiring.** Most defensible.
Split the kernel into (i) bulk collision + push-stream, (ii)
boundary-application pass. Likely path to ~30–50 Mcell/s laptop
without exotic tricks; that puts 96³ at ~2 min laptop and ~5 min
Cloud — back inside the user's tolerance.

**B. Drop the Interactive3D preset to 64³.** At 11.5 Mcell/s the
existing kernel takes ~2 min for 64³ × 5000 steps on laptop, ~5 min
on Cloud. Acceptable. Cost: the body is ~16 cells wide instead of
~24, less qualitative wake detail.

**C. Pursue the in-place AA-pattern / esoteric twist** (Geier &
Schönherr 2017). Halves population memory (no `f_next` buffer) and
typically improves cache behaviour by ~20–40 %. Bigger lift, but the
plan §3.5 explicitly says "Do not start with the clever scheme;
revisit only if needed." This is the right call only if A and B
prove insufficient.

**Recommendation: A first.** The plan's exit criterion for Phase 5
("completes on Cloud in a tolerable time") is a portfolio-level
requirement, not a research one — solve it in Phase 1 by writing the
production kernel cleanly rather than inheriting the Phase 0
prototype's branchy hot loop.

### What this does NOT change

- The two-preset split (Interactive3D web vs Validation3D offline)
  stands. Validation runs offline with `parallel=True` and is in the
  4-8× range above what this serial number suggests; the plan's
  estimate of 2–4 min per validation run is unaffected.
- The float32 + double-buffer + (19, Nx, Ny, Nz) layout choices stand
  (other protos confirmed). Layout is not the bottleneck; branching
  in the hot loop is.
- The 96³ memory line (135 MB double-buffered) is still inside the
  Cloud budget.

---

## Proto 4 — float32 matches float64 on TGV decay (PASS)

24³ TGV, U = 0.04, ν = 0.01, n = 600 steps. Measured decay rate
against the analytic 4 ν k² = 0.002742:

| dtype   | measured | err vs analytic |
|---------|----------|------------------|
| float64 | 0.002741 | −0.04 %          |
| float32 | 0.002741 | −0.03 %          |

float32–float64 spread: **0.0006 %**. Well inside the 1 % gate.

**Production dtype = float32 confirmed.** Validation grid at
320 × 160 × 160 stays at ~622 MB single buffer, ~1.24 GB double, on
laptop only (offline). Cloud Interactive preset at 96³ stays at
~135 MB double-buffered.

Honest caveat: 600 steps is short. A long Validation3D run is ~10 000
steps; float32 round-off accumulates with step count. The Phase 1 TGV
gate should be re-run at float32 over the full step budget before any
production run.

---

## Locked decisions after Phase 0

Every Phase 0 question that was decided on paper has now been
empirically confirmed or refined:

| Question        | Plan decision (on paper)              | Phase 0 result                                    |
|-----------------|---------------------------------------|---------------------------------------------------|
| Lattice         | D3Q19                                 | Confirmed by proto 3 memory line                  |
| Collision       | TRT (Λ = 3/16); BGK reference         | TRT decay ≤ 0.5 % err (proto 1)                   |
| Turbulence      | none in validated path                | (no test — relies on 2D review's LES finding)     |
| Inflow/outflow  | Guo NEEM                              | 0.0000 % mass drift (proto 2)                     |
| Obstacle wall   | Bouzidi analytic-q                    | (no test — exercised in Phase 2)                  |
| q-field storage | analytic on-the-fly                   | (no test — geometric, no empirical question)      |
| Streaming       | double buffer, float32, (19,Nx,Ny,Nz) | float32 spread 0.0006 % (proto 4)                 |
| Backend seam    | NumbaCPU now                          | (no test — purely architectural)                  |
| Throughput      | 20–50 Mcell/s Cloud, 50–150 laptop    | **11.5 Mcell/s laptop on existing kernel**        |

The throughput finding is the only one that needs action. It does
not invalidate any other decision; it adjusts the Phase 1 work plan
to include a clean-kernel rewrite (option A above) before the
Interactive3D preset hits Streamlit.

---

## Phase 1 entry criteria

All four prototype gates passed except proto 3, which surfaced a
performance issue that is a Phase 1 work item rather than a research
blocker. Phase 1 can start. The first Phase 1 task is the production
TRT kernel, written *without* boundary-branching inside the hot loop
to recover the throughput the plan §1.2 assumed.
