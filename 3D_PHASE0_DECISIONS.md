# AeroLab 3D Solver: Phase 0 Decision Memo

Status: **Phase 0 + Phase 1 closed**. Phase 0 prototypes confirmed
the locked decisions (see `3D_PHASE0_FINDINGS.md`); Phase 1
production TRT kernel met the TGV decay-rate gate within ±0.25 %.
This memo records every locked technical decision; new entries
(D-8, D-9, D-10) reflect the 2026-05-26 product re-scope from
"validation tool" to "consumer 3D visualisation product with
validation as the backend credibility layer."

Companion to `3D_RESEARCH_PLAN.md`. Read that first for the budget
analysis, the revised phase ordering, and the rendering plan.

---

## Scope decisions (locked, revised 2026-05-26)

The original v1 was "validation-first": sphere + spanwise-periodic
cylinder, no uploads. The revised scope adds the consumer-product
layer that always sat behind AeroLab's purpose — see
`3D_RESEARCH_PLAN.md` section 0 for the framing.

**v1 user-facing geometry:** the same gallery / Upload split that
the 2D mode already has, scaled to 3D:

- **Built-in gallery shapes** (3D analogues of `src/sample_shapes.py`):
  a sphere, a simplified car, a building, a wing. **Pre-baked
  velocity fields** ship with the repo so the user clicks → result
  is instantaneous, no live LBM solve.
- **Upload** (3D analogue of `src/custom_shape.py`): STL / OBJ via
  `trimesh`, voxelised onto the LBM grid, solved live, the user
  waits a few minutes the first time. Pre-baking is impossible by
  definition for a user upload.

**v1 backend geometry (the credibility layer, never user-facing):**
sphere + spanwise-periodic cylinder, validated offline against
Schiller-Naumann (sphere) and Williamson (cylinder). These are how
`VALIDATION.md` proves the solver is real. They do not appear in
the gallery.

**Timeline:** still ~8 to 10 weeks but reordered. The web app's
gallery + smoke-particle viz + upload pipeline are now the
"product-blocking" front-half work; the sphere + cylinder
validation is the "credibility-blocking" back-half work. See
`3D_RESEARCH_PLAN.md` section 4 for the revised phase ordering.

**2D K-recalibration:** still carried as 2D backlog, not done
before 3D.

---

## D-1. Lattice: D3Q19 (locked, no prototype needed)

D3Q19 is the standard incompressible-flow lattice and the right
memory/accuracy point for this project. D3Q15 is rejected for known
anisotropy and instability; D3Q27 costs 42% more memory for advantages
(Galilean invariance at higher Mach) the laminar low-Mach band does not
need.

The 19 discrete velocities, grouped:

- 1 rest vector: (0,0,0).
- 6 face vectors, |c|^2 = 1: (+/-1,0,0), (0,+/-1,0), (0,0,+/-1).
- 12 edge vectors, |c|^2 = 2: all (+/-1,+/-1,0), (+/-1,0,+/-1),
  (0,+/-1,+/-1).

Weights: w_rest = 1/3; w_face = 1/18 (six of them); w_edge = 1/36
(twelve of them). Check: 1/3 + 6*(1/18) + 12*(1/36) = 1/3 + 1/3 + 1/3
= 1. Speed of sound squared cs^2 = 1/3, same as D2Q9.

Each direction needs its opposite index (for bounce-back) and the
exact ordering becomes the single source of truth for the kernel, the
streaming shifts and the q-field, exactly as `LATTICE_VELOCITIES` /
`OPPOSITE` are in the 2D `lbm.py`.

Memory confirmed (from plan section 1.1): 96 cubed, float32,
double-buffered = 135 MB. Safe inside the Cloud budget.

---

## D-2. Collision: TRT for production, BGK as reference path (locked on
literature; one prototype confirms stability)

### Why TRT

Two-relaxation-time collision (Ginzburg, Verhaeghe, d'Humieres 2008)
splits each population into a symmetric and an antisymmetric part and
relaxes them with two separate rates. It costs almost the same as BGK,
is far simpler than the 19-moment D3Q19 MRT, and has one property that
directly serves this project: with the right parameter the bounce-back
wall sits at the exact mid-link location independent of viscosity,
which removes a viscosity-dependent error from Cd, the quantity you
validate.

### The TRT scheme

For each direction i with opposite ii:

- Symmetric / antisymmetric split of the populations:
  f_i^+ = (f_i + f_ii) / 2,  f_i^- = (f_i - f_ii) / 2.
- Same split of the equilibrium e_i (the standard second-order D3Q19
  equilibrium): e_i^+ = (e_i + e_ii)/2,  e_i^- = (e_i - e_ii)/2.
- Collision:
  f_i^post = f_i - s_plus * (f_i^+ - e_i^+) - s_minus * (f_i^- - e_i^-).

Relaxation rates:

- s_plus sets the kinematic viscosity:
  nu = cs^2 * (1/s_plus - 1/2),  i.e. 1/s_plus = 3*nu + 1/2.
  This is identical to the 2D tau relation; s_plus = 1/tau.
- s_minus is free. It is fixed through the "magic parameter"
  Lambda = (1/s_plus - 1/2) * (1/s_minus - 1/2).
- Choose **Lambda = 3/16**. Then the halfway bounce-back wall is
  located exactly mid-link regardless of viscosity. Solving for
  s_minus: 1/s_minus - 1/2 = Lambda / (1/s_plus - 1/2) = (3/16)/(3*nu)
  = 1/(16*nu), so s_minus = 1 / ( 1/(16*nu) + 1/2 ).

BGK is the special case s_plus = s_minus = 1/tau. Implement it as a
one-line variant of the same kernel: it becomes the **reference path**
for the JIT-vs-reference equivalence test, exactly as the 2D code keeps
a pure-NumPy BGK reference.

### What is not carried over

No Smagorinsky LES in the validated path. The 2D review established it
injects spurious eddy viscosity at laminar Re. For the 3D validated
band (Re <= ~300, laminar) the solver is TRT, full stop. If the
interactive web toy is later pushed to higher Re for visual drama, an
LES term can be made opt-in there and clearly labelled, but it never
touches a validation number.

### Empirical confirmation (first prototype, on your go)

A throwaway script: run TRT and BGK on a 3D lid-driven cavity and on
the Taylor-Green vortex, sweep tau toward 0.5, and confirm TRT stays
stable where BGK diverges, and that TRT with Lambda = 3/16 reproduces
the analytic TGV decay rate. Expected outcome: TRT confirmed. If TRT
somehow proves marginal at the top of the Re band, the fallback is
D3Q19 MRT, which is more work but well documented (d'Humieres 2002).

---

## D-3. Inflow and outflow: Guo non-equilibrium extrapolation (locked
on literature; one prototype confirms mass closure)

Full 3D Zou-He has awkward edge and corner closures and many unknown
populations. Do not port it. Use the non-equilibrium extrapolation
method (Guo, Zheng, Shi 2002):

At a boundary node b with interior neighbour n:

- Decompose any population as equilibrium plus non-equilibrium:
  f_i = e_i + f_i^neq.
- Prescribe the macroscopics at b: velocity u_b at the inflow,
  density rho_b = 1 at the outflow. The unknown macroscopic at each
  face (rho at inflow, u at outflow) is extrapolated from n.
- Set the boundary equilibrium e_i(rho_b, u_b) from the prescribed
  values, and copy the non-equilibrium part from the neighbour:
  f_i(b) = e_i(rho_b, u_b) + f_i^neq(n)
         = e_i(rho_b, u_b) + ( f_i(n) - e_i(rho_n, u_n) ).

This is second-order, robust, has no per-direction unknown-population
algebra, and generalises to 3D edges and corners without special
cases. It is the standard pragmatic 3D inflow/outflow.

Lateral faces: for the sphere validation preset, free-slip or periodic
to minimise blockage interference; no-slip walls are acceptable in the
interactive preset. For the cylinder, spanwise periodic (see D-8).

Empirical confirmation (first prototype): channel flow with the body
removed, measure inflow-minus-outflow mass imbalance, expect it well
under 1% as in 2D. The regularized BC (Latt & Chopard 2006) is the
fallback if Guo proves noisy.

---

## D-4. Obstacle boundary: Bouzidi interpolated bounce-back with
analytic q (locked, no prototype needed)

### Sphere q derivation

A wall link runs from a fluid cell x_f along lattice direction c_i to
a solid neighbour. The interpolated bounce-back needs the wall
fraction q, the distance from x_f to the analytic surface along c_i,
normalised to |c_i|.

For a sphere of radius R centred at x_c, let d = x_f - x_c. The wall
crossing satisfies |d + q*c_i|^2 = R^2, which expands to a quadratic
in q:

  A*q^2 + B*q + C = 0,
  A = |c_i|^2   (1 for face directions, 2 for edge directions),
  B = 2 * (d . c_i),
  C = |d|^2 - R^2.

q is the smaller positive root. This is the **exact same structure**
as the existing 2D `cylinder_q_field` in `src/shapes.py`; the 3D
sphere version is a direct transcription with a third coordinate. The
Bouzidi linear-interpolation formula (the q >= 0.5 and q < 0.5
branches) carries over unchanged from the 2D kernel.

The spanwise-periodic cylinder reuses the existing 2D cylinder q
expression in the cross-flow plane and is uniform along the span.

### Force

Momentum exchange, a direct 3D generalisation of `src/forces.py`:
sum 2 * c_i * f_i over wall links (with the Bouzidi-aware correction
for q != 0.5). Drag is the streamwise component, lift the transverse.

---

## D-5. q-field storage: analytic for sphere/cylinder, sparse-list for uploads

A dense (Nx, Ny, Nz, 19) q-field is as large as the population array
itself (159 MB at 128 cubed) and is forbidden by the memory budget.

**For sphere and cylinder** (the credibility-layer validation
geometries): q is computed inside the kernel from the surface
equation when a wall link is detected. Zero storage. The Bouzidi
quadratic-root formula from section D-4 evaluates per cell at
trace-time cost.

**For uploaded meshes** (the product layer, added 2026-05-26): the
surface is a triangle soup with no closed-form q. The sparse
wall-link list is the right storage scheme — a 1-D array of
`(cell_x, cell_y, cell_z, direction_i, q)` tuples for every fluid
cell that has at least one solid neighbour. The wall-link count
scales as the body's surface area in cells, i.e. O(L²) for a body
of linear size L, which on a 96³ grid with a body filling ~half the
domain is a few tens of thousands of entries. Tiny next to the 67 MB
population array. The kernel reads the list once per step and applies
Bouzidi interpolation per entry. See D-9 for the upload pipeline.

A simpler full-way bounce-back (q assumed = 0.5) is the first
implementation for uploads; Bouzidi interpolation against the
voxelised mesh is the v1.1 follow-on. Full-way is what 2D's
`custom_shape.py` uses for arbitrary polygons; the same accuracy /
simplicity trade-off carries over.

---

## D-6. Streaming and memory layout: double buffer first (locked for
Phase 1)

Two arrays, f and f_new, float32. At 96 cubed this is 135 MB, safely
inside budget, and it keeps the kernel simple and easy to test. The
in-place AA-pattern / esoteric twist (Geier and Schonherr 2017) halves
population memory and is the route to a 128 cubed interactive preset,
but it complicates the kernel and the tests. Defer it: implement
double buffering, and revisit the in-place scheme in Phase 1 only if
128 cubed turns out to be wanted.

Array layout: (Nx, Ny, Nz, 19) AoS (array of structures), confirmed
by the Phase 1 throughput pass. The original plan named SoA layout
(19, Nx, Ny, Nz) as the default matching 2D's `(9, Nx, Ny)`, but
proto 3 measured throughput collapsing from ~12 Mcell/s at 48³ to
~3 Mcell/s at 96³ — the cache-spillover signature of the per-cell
strided read pattern that SoA forces (19 cache lines per cell, each
Nx·Ny·Nz·4 bytes apart). Switching to AoS made per-cell reads
unit-stride (one cache line for all 19 populations) and lifted the
96³ throughput ~1.3× — modest but in the predicted direction. The
remaining performance gap is in the equilibrium duplication (see
`3D_PHASE0_FINDINGS.md` and Phase 1.5 backlog), not the layout.

---

## D-7. Backend seam: thin interface, Numba CPU now (locked)

A minimal abstraction, deliberately not more:

- `Solver3DBackend` with `allocate(grid, params) -> state`,
  `step(state) -> state`, `download_macroscopic(state) -> (rho, u)`.
- State arrays are ndarray-like (numpy now; cupy is a near drop-in
  later).
- Geometry, masks and the analytic-q logic are host-side numpy,
  computed once.
- The time loop, boundary-condition application order and snapshot
  cadence live in backend-agnostic orchestration code.
- `NumbaCPUBackend` is implemented now, with a serial variant (Cloud)
  and a `parallel=True` prange variant (offline validation). The
  parallel variant is safe offline; the 2D code disabled it only
  because of a Cloud-specific NUMBA_NUM_THREADS error.
- The per-cell collision math is one small, heavily tested, commented
  reference function. A future GPU kernel is a transcription of it,
  not a redesign. Do not abstract beyond this.

---

## D-8. Smoke-particle advection (the v1 viz, locked 2026-05-26)

The user-facing 3D viz is **smoke streamers**, not Q-criterion
isosurfaces. Rationale and trade-offs are in
`3D_RESEARCH_PLAN.md` section 7; the decision and the algorithm
live here.

**Algorithm.** Once the LBM solve has produced a steady velocity
field `u(x, y, z) = (ux, uy, uz)`, integrate massless tracer
particles through it with RK4:

  p(t + dt) = p(t) + dt * RK4(u, p(t)),

with trilinear interpolation of u at the particle position. dt is
chosen so a particle moves <~ 0.5 cell per step at the inflow
speed; smaller particles look smoother but cost more. ~300-500
particles per scene, lifetime ~ Lx / u_in, continuous re-seeding
just upstream of the body so the streamlines stay alive.

**Why this is cheap.** The expensive work (the LBM solve) produced
a static (ux, uy, uz) array. Advecting 500 particles for 100 frames
is ~50 ms in pure numpy. The user can spin the camera, change the
seed pattern, replay — all free.

**Why this is the right viz for the audience.** Everyone has seen
a wind tunnel smoke test. Nobody outside CFD knows what a
Q-criterion isosurface is. The smoke particles ARE the 3D analogue
of the 2D streaklines AeroLab already ships.

**Empirical confirmation** (Phase A prototype): pick a pre-baked
field, advect ~300 particles, render with Plotly Scatter3d, confirm
the visual matches the 2D streakline output of the same case
projected onto a plane.

---

## D-9. Mesh-upload pipeline (the new product piece, locked 2026-05-26)

**v1 supports user-uploaded 3D meshes**, lifted out of the
"deferred" pile in the original Phase 0 memo because the consumer
product needs it. The 2D mode already ships custom-shape upload
(`src/custom_shape.py` + `src/sample_shapes.py`); 3D is the natural
extension.

**New dependency: `trimesh`** (the standard Python library for STL,
OBJ, PLY, etc.; pure-Python, no compiled binary on Cloud). Adds
~5 MB to the Cloud image, acceptable.

**Pipeline** (all host-side numpy, runs once on user upload):

  1. **Load** via `trimesh.load(filename)`. Accept STL (ASCII +
     binary), OBJ, PLY, GLB.
  2. **Repair** if non-watertight: `trimesh.repair.fill_holes()`,
     `trimesh.repair.fix_normals()`. If still bad, warn the user;
     do not refuse the upload (consumer product).
  3. **Centre + rescale** to a fixed fraction of the LBM domain
     (target ~ D / Lx = 0.3 for the streamwise bounding-box
     dimension, matching the 2D blockage convention).
  4. **Voxelise** via `trimesh.voxelized(pitch=lattice_spacing)`.
     Returns a `VoxelGrid` that exposes `.matrix` as a boolean
     numpy array of solid cells.
  5. **Build wall-link list**: for every fluid cell at the
     boundary, enumerate the 18 (or fewer) directions whose
     neighbour is solid, record (cell, direction, q=0.5) into the
     sparse list per D-5.
  6. **Hand to the solver** with the same `Solver3DBackend`
     interface as gallery shapes.

**Robustness layer** for what users actually upload (the 2D
silhouette extractor taught this lesson — most of that module is
validation code, not algorithm):

- Non-watertight meshes: auto-repair, on failure proceed anyway
  with the holey mask and note in the UI that the result may be
  approximate.
- Wrong scale (object is 0.001 m or 50 m): bounding-box normalise
  on load.
- Wrong orientation (user uploaded a car pointed up): offer a
  "flip / rotate" button in the UI; default to auto-aligning the
  longest axis to the wind direction.
- Way too many faces: `trimesh.simplify_quadric_decimation()` to
  a target face budget if the mesh would take >2 minutes to
  voxelise.

**Honest UX expectation** (the reviewer flagged this and it is
worth signing in blood): **uploaded meshes pay the full LBM solve
cost. ~3-5 minutes the first time on Cloud, then instant playback.**
The "first run is slow, then it is yours to play with" framing must
be in the UI from day one. Promising instant on an upload is the
broken-product trap.

---

## D-10. Pre-bake / replay architecture (the product-enables decision, locked 2026-05-26)

**Built-in gallery shapes ship as pre-computed velocity fields.**
The expensive LBM solve runs ONCE per gallery shape, offline, on
the developer machine. The result — a (3, Nx, Ny, Nz) `(ux, uy,
uz)` array — is compressed and committed to the repo. The Streamlit
app loads it on demand. The user never waits for an LBM solve for
a gallery shape.

**Storage cost.** Per shape, `3 * 96³ * 4 = 10.6 MB` raw, ~3 MB with
`np.savez_compressed` and float16 quantisation. Four to six gallery
shapes = ~15-20 MB. Acceptable in the repo; Git LFS for `*.npz` if
it grows.

**Solve-time cost.** Each gallery shape costs ~5-20 minutes offline
on the development laptop (96³ × ~5000 steps with the Phase 1 TRT
kernel at ~2 Mcell/s, less with the Phase 1.5 named-scalars
rewrite). Run once, never again unless the kernel changes.

**Two-stage architecture this enables:**

  Solve stage (offline, expensive, run by developer):
    geometry + LBM solve  ->  steady velocity field
                          ->  np.savez_compressed to data/3d_fields/

  Replay stage (runtime, cheap, run on every user interaction):
    load .npz  ->  smoke-particle advection through field
              ->  Plotly Scatter3d frames
              ->  smooth interactive scene

  Upload stage (runtime, expensive only once per upload):
    user .stl  ->  voxelise + solve  ->  field  ->  same replay path

**Cache the upload solve.** Hash the uploaded mesh, store the
solved field in `st.cache_data` keyed on the hash. Same-user
re-upload of the same file is instant.

**Validation files** (sphere, cylinder, TGV) stay in a different
directory (`data/validation/`) — they are not gallery candidates
and they ship raw JSON, not compressed fields.

---

## Phase 1 verification: 3D Taylor-Green vortex — PASSED 2026-05-28

The Phase 1 exit gate was "the viscosity is provably correct" via
the decaying Taylor-Green vortex in a periodic box. The 2D-extruded
TGV initial condition (uniform in z) has the analytic KE decay rate
4 ν k², and the Phase 1 production TRT kernel reproduces it.

**Measured** (`scripts/dev_3d_phase1_tgv_gate.py`, AoS and SoA
variants, N = 32, U = 0.04, n_steps = 800):

  TRT err = +0.16 % to -0.24 % across ν in {0.005, 0.01, 0.02}.
  BGK err = -0.01 % to -0.04 %.

Both inside the ±2 % gate. The collision operator, streaming, and
viscosity relation are correct. Phase 2 entry criterion met.

The full 3D TGV (a genuinely three-dimensional initial condition)
is still listed for Phase 2 as a qualitative check.

---

## What needed the first code — the Phase 0 prototypes (now done)

Everything in D-1 through D-7 was decided on paper. Four throwaway
prototype scripts confirmed the choices empirically before Phase 1
production code; results in `3D_PHASE0_FINDINGS.md`.

1. TRT vs BGK stability sweep on a 3D cavity and the TGV near
   tau -> 0.5.
2. Guo inflow/outflow mass-conservation check on empty channel flow.
3. Numba D3Q19 kernel throughput and JIT-compile-time measurement, to
   confirm the plan's compute budget (section 1.2).
4. float32 vs float64 accuracy spot check on the TGV decay rate.

All four shipped 2026-05-28 (commit `17566bb`) and gave PASS / PASS
/ PARTIAL / PASS verdicts; the PARTIAL on throughput drove the
Phase 1 AoS layout switch and the deferred Phase 1.5 named-scalars
rewrite. Phase 0 closed.

The next prototypes (per the revised plan section 4 in
`3D_RESEARCH_PLAN.md`) are Phase A scaffolding for the consumer
product, not more Phase 0 work:

  * Smoke-particle RK4 advection through a known velocity field.
  * Pre-bake driver: solve a sphere field offline, save to
    `data/3d_fields/sphere.npz`, reload and replay.
  * Plotly Scatter3d render of the advected particles inside
    Streamlit; confirm the rotate / zoom interaction works.
  * trimesh load + voxelise of a sample STL onto an LBM grid.

---

## Summary: Phase 0 decisions (revised 2026-05-26 for the product re-scope)

| Question         | Decision                                                          |
|------------------|-------------------------------------------------------------------|
| Lattice          | D3Q19                                                             |
| Collision        | TRT (Λ = 3/16); BGK as reference path                             |
| Turbulence       | none in the validated path (no LES)                               |
| Inflow/outflow   | Guo non-equilibrium extrapolation                                 |
| Obstacle wall    | Bouzidi interpolated bounce-back                                  |
| q-field storage  | analytic for sphere/cylinder; sparse wall-link list for uploads   |
| Streaming        | double buffer, float32, **(Nx, Ny, Nz, 19) AoS** (Phase 1 confirmed) |
| Backend          | thin seam, NumbaCPU (serial + parallel) now                       |
| **Viz**          | **smoke-particle RK4 advection rendered via Plotly Scatter3d**    |
| **Mesh upload**  | **trimesh + voxelise + sparse wall-link list (v1, full-way BB)**  |
| **Architecture** | **pre-bake gallery fields, replay at runtime; uploads pay once**  |
| Phase 1 gate     | 2D-extruded TGV decay rate within ~2 % — **PASSED 2026-05-28**    |

Phase 0 + Phase 1 closed. The next coding actions sit under the
revised Phase A in `3D_RESEARCH_PLAN.md` section 4: smoke-particle
advection prototype, the pre-bake driver, and the Plotly 3D viz.
No code is written until the user gives the explicit go-ahead.
