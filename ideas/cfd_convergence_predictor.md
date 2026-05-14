# CFD Convergence Predictor

**Idea source:** Conversation with Samar (ex-ISRO High Performance Computing Lab, former CFD analyst) at GSTCE 2026, Marina Bay Sands, May 14, 2026.

**Status:** Parked. Detailed notes only. Not pursuing until AeroLab Phase 3 ships (August 2026).

---

## The Problem (in Samar's own words, paraphrased)

At ISRO HPC Lab, Samar and his colleagues ran CFD simulations on shared compute. Time slots on the cluster were limited and contested. A typical pattern:

1. Engineer sets up a CFD case in the evening
2. Submits the job, leaves work
3. Returns the next morning expecting results
4. Discovers the simulation diverged at hour 3
5. Slot is gone, no results, has to debug the setup and requeue
6. Repeat

The frustration was not just the wasted compute. It was the wasted human time, the schedule slip, the contested resource being burned, and the inability to diagnose what went wrong without rerunning.

This is not an ISRO-specific problem. It is universal across:
- National labs (NASA, ESA, ISRO, JAXA, DLR)
- Aerospace primes (Boeing, Airbus, Rolls-Royce, GE Aerospace, Lockheed)
- F1 and motorsport teams
- HVAC, automotive, biomedical, wind energy, any field that uses CFD seriously
- Academic research groups with shared cluster time

## The Proposed Solution

An **agentic AI system** that ingests a CFD case setup *before the run* and predicts:
1. Will this case converge within reasonable iterations? (Yes / No / Likely problematic)
2. If divergence is predicted, **why**? Which specific feature is the likely culprit?
3. What are the suggested fixes? (Refine mesh in region X, adjust under-relaxation factor, change turbulence model, etc.)

The "agentic" framing means the system is not just a classifier. It reasons over multiple inputs, generates explanations, and can suggest corrective actions. Modern LLM-tool-use architectures fit this naturally.

## Inputs The Agent Would Analyze

Based on Samar's description and standard CFD practice, the agent would ingest:

- **Mesh quality metrics:** aspect ratio distribution, skewness, orthogonality, expansion ratio, y+ values at walls
- **Geometry features:** sharp edges, thin features, regions of high curvature, characteristic length scales
- **Boundary conditions:** type and consistency at inlets/outlets/walls, mass flow balance, pressure gradient implications
- **Boundary layer setup:** first cell height vs. expected y+, growth rate, number of prism layers
- **Solver setup:** time step, CFL number, under-relaxation factors, turbulence model choice vs. flow regime
- **Physics regime:** Reynolds number, Mach number, expected separation, shock presence
- **Initial conditions:** are they physically reasonable for the problem
- **Historical convergence data:** if a similar case has been run before, what happened

## Output

For each predicted outcome:

- **Confidence score:** how sure the model is
- **Failure mode prediction:** if divergence is likely, what type (mass imbalance, residual oscillation, numerical instability, etc.)
- **Root cause identification:** specific feature that is likely responsible
- **Suggested mitigations:** ranked list of changes to try
- **Estimated additional setup time:** so the engineer can decide whether to fix or just submit and gamble

## Why This Is Massively Useful

- **Saves compute cost:** a single divergent run on industrial CFD can cost hundreds to thousands of dollars
- **Saves human time:** engineers stop losing whole days to failed runs
- **Saves schedule:** projects do not slip because of repeated failed simulations
- **Reduces queue contention:** in shared HPC environments, fewer failed slots means more useful science gets done
- **Democratizes CFD:** less experienced users get expert-level setup review before submitting
- **Generates training signal over time:** every prediction creates more labeled data for the model

## Market Size (rough estimate)

- ANSYS Fluent has ~30,000+ commercial seats globally
- OpenFOAM has 100,000+ users
- Star-CCM+ has 10,000+ seats
- Estimated 15-40% of industrial CFD runs diverge or fail to converge on first attempt
- Even a conservative valuation (saving 1 engineer-day per week per heavy CFD user) implies a multi-hundred-million-dollar productivity opportunity globally

## Why I Am Not Pursuing This Now

1. **AeroLab is committed through August 2026.** Splitting focus would produce two half-finished projects instead of one good one.
2. **The new project requires labeled training data I do not yet have access to.** Building this requires either (a) a partnership with a research group or industrial lab that can share anonymized CFD run telemetry, or (b) generating synthetic divergence/convergence data via OpenFOAM, which is itself a multi-month project.
3. **The relationship with Samar is more valuable preserved than burned.** A premature cold approach to him with a half-baked prototype would damage the relationship. Better to come back in 6 months with thoughtful framing.
4. **AeroLab itself produces relevant skills and infrastructure** for the convergence predictor: deep CFD understanding, OpenFOAM workflow, Python ML stack, validation discipline. AeroLab is preparation, not competition with this idea.

## What I Will Do Between Now and August

- **Phase 3 of AeroLab will add a "convergence telemetry" feature** that logs OpenFOAM validation runs (whether they converged, residual trajectories, mesh quality at runtime). This creates a small but growing dataset that could later seed this project.
- **Maintain the relationship with Samar.** Annual or quarterly check-ins via LinkedIn. Update him when AeroLab ships.
- **Read background literature.** Look for academic papers on ML-based CFD convergence prediction, anomaly detection in numerical solvers, and agentic systems for engineering workflows. Build a small reading list over time.
- **Watch for early signals.** If anyone else publishes a serious tool in this space before I revisit it, reassess whether the opportunity still exists or whether I should pivot the angle.

## Next Review Date

**September 2026.** After AeroLab Phase 3 ships and I have decompressed from the summer build. At that point, decide whether to:
- Start the convergence predictor as my next major project
- Pursue it as a research collaboration with Samar / NTU MAE / a research lab
- Park it longer if other higher-priority opportunities have emerged

## Related Reading (to populate over time)

- (to fill in as I find papers)
- Look for: "machine learning CFD convergence", "neural network turbulence model selection", "anomaly detection numerical solver", "agentic AI engineering simulation"

## Contacts

- **Samar [last name]:** ex-ISRO HPC Lab, met at GSTCE 2026. Connection request sent May 15. Reserved future conversation about this idea once AeroLab ships.

---

*This document was created the same evening the conversation happened (May 14, 2026), while the context was fresh. It exists to preserve the idea, the source, and the reasoning for the parking decision, so that future-me has a real artifact to work from when the time comes to revisit.*
