# AeroLab 3D Solver: Phase 0 Decision Memo

Status: research complete, no code written. This memo resolves every
open question from `3D_RESEARCH_PLAN.md` section 3 with the technical
detail needed to start Phase 1. Where a choice still wants empirical
confirmation, that is called out as the first prototype task.

Companion to `3D_RESEARCH_PLAN.md`. Read that first for the budget
analysis and phasing.

---

## Scope decisions (locked)

- **v1 geometry:** sphere and spanwise-periodic cylinder only. Both
  have analytic surfaces, so both get analytic q-fields and no mesh
  voxelisation is needed. Arbitrary 3D uploads are deferred.
- **Timeline:** validation-first, ~8 to 10 weeks. Phase 4 (cylinder)
  is the protected milestone.
- **2D K-recalibration:** carried as 2D backlog, not done before 3D.

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

## D-5. q-field storage: analytic on-the-fly (locked, no prototype)

A dense (Nx, Ny, Nz, 19) q-field is as large as the population array
itself (159 MB at 128 cubed) and is forbidden by the memory budget.
Because v1 geometry is sphere and cylinder only, both with analytic
surfaces, q is computed inside the kernel from the surface equation
when a wall link is detected. Zero storage. A sparse wall-link list
(cell index, direction, q) remains the general mechanism if arbitrary
geometry is ever added, but v1 does not need it.

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

Array layout: store f as shape (19, Nx, Ny, Nz) so each direction is
contiguous, matching the 2D (9, Nx, Ny) convention. Confirm cache
behaviour in the throughput prototype; if it disappoints, the
alternative is (Nx, Ny, Nz, 19) "array of structures".

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

## Phase 1 verification: 3D Taylor-Green vortex (the gate)

The Phase 1 exit gate is "the viscosity is provably correct." The
cleanest rigorous test is a decaying Taylor-Green vortex in a periodic
box.

Initialise the 2D Taylor-Green vortex extruded through a thin periodic
3D box (uniform in z):

  u_x = -U * cos(k*x) * sin(k*y),
  u_y =  U * sin(k*x) * cos(k*y),
  u_z = 0,

with k = 2*pi / L. The analytic solution decays as
u(t) = u(0) * exp(-2 * nu * k^2 * t), so the total kinetic energy
decays as exp(-4 * nu * k^2 * t).

Procedure: initialise, run a few thousand steps, fit a straight line
to ln(KE) versus t, extract the decay rate, and compare to the
analytic 4 * nu * k^2 using the nu set by s_plus. **Gate: agreement
within ~2%.** This single test proves the collision operator,
streaming, and the viscosity relation are all correct, and it is
prototype-cheap.

The full 3D Taylor-Green vortex (a genuinely three-dimensional initial
condition) is run afterwards as a qualitative check that the solver
behaves in 3D, but the 2D-extruded case is the one with a clean
analytic decay rate to gate on.

---

## What still needs the first code (the Phase 0 prototypes)

Everything above is decided on paper. Four throwaway prototype scripts,
roughly 100 to 250 lines total, disposable, confirm the choices
empirically before Phase 1 production code:

1. TRT vs BGK stability sweep on a 3D cavity and the TGV near
   tau -> 0.5.
2. Guo inflow/outflow mass-conservation check on empty channel flow.
3. Numba D3Q19 kernel throughput and JIT-compile-time measurement, to
   confirm the plan's compute budget (section 1.2).
4. float32 vs float64 accuracy spot check on the TGV decay rate.

These are the first lines of code in the 3D effort. Per your
"research only, do not implement first" instruction, they are not
written yet. They are the natural next action and they are cheap.

---

## Summary: Phase 0 is decided

| Question        | Decision                                            |
|-----------------|-----------------------------------------------------|
| Lattice         | D3Q19                                               |
| Collision       | TRT (Lambda = 3/16); BGK as reference path          |
| Turbulence      | none in the validated path (no LES)                 |
| Inflow/outflow  | Guo non-equilibrium extrapolation                   |
| Obstacle wall   | Bouzidi interpolated bounce-back, analytic q        |
| q-field storage | analytic on-the-fly, zero storage                   |
| Streaming       | double buffer, float32, (19,Nx,Ny,Nz) layout        |
| Backend         | thin seam, NumbaCPU (serial + parallel) now         |
| Phase 1 gate    | 2D-extruded Taylor-Green decay rate within ~2%      |

Nothing above requires a decision from you. The next action is the
four confirmation prototypes, then Phase 1. No code will be written
until you give the explicit go-ahead.
