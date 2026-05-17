# MHX documentation

MHX is a validation-first, differentiable JAX framework for magnetic
reconnection and magnetohydrodynamics.

The active package is under `src/mhx/`.

## Active scope boundaries

The current repository supports deterministic FAST validation, seed-robust QI,
fitted latent-ODE reproducibility, timing/readiness reports, nonlinear
reduced-MHD media examples such as double-Harris, Orszag--Tang, decaying
turbulence, and forced turbulent reconnection, plus short validation-grade
Rutherford executor chunks. Long nonlinear Rutherford/plasmoid production
campaigns are still gated as future `production` claims and must pass duration,
checkpoint-restart, convergence, seed/QI, and artifact-hash checks before being
used as paper evidence.

## Evidence guide

Start with these pages when checking validation evidence and scope boundaries:

- [Evidence map](reviewer_evidence.md) links claims to source files,
  tests, reproduction commands, artifacts, and explicit limitations.
- [Physics validation](validation.md) contains the equations, citations,
  still-figure gallery, and numerical gates that are intentionally kept out of
  the README.
- [Benchmark command index](benchmarks.md) lists validation, scaffold,
  comparison, campaign, and neural-ODE commands with expected output families.
- [Validation movies](media.md) separates solver-generated movies from theory
  schematics and keeps the literature anchors next to the visuals.
- [Publication checklist](publication_checklist.md) states which figures are
  ready as validation evidence and which remain roadmap or production-only.

```{toctree}
:maxdepth: 2

install
tutorial
quickstart
reviewer_evidence
validation
benchmarks
architecture
reduced_mhd
diagnostics
output_schema
performance
long_run_evidence
seed_robust_qi
neural_ode_reproducibility
time_windows
campaigns
campaign_runner
publication_checklist
paper_plan
media
audit
api_policy
model_assembly
plugins
release
migration
literature
api
```
