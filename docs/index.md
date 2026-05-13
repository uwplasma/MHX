# MHX documentation

MHX is being rebuilt as a validation-first, differentiable JAX framework for
magnetic reconnection and magnetohydrodynamics.

The active package is under `src/mhx/`. The previous implementation is archived
under `legacy/old_mhx/`.

## Active scope boundaries

The current repository supports deterministic FAST validation, seed-robust QI
checks, short validation-grade campaign artifacts, and production campaign
planning with checkpoint/resume metadata. Long nonlinear Rutherford/plasmoid
production campaigns are still gated as future `production` claims and must pass
duration, checkpoint-restart, convergence, and artifact-hash checks before being
used as paper evidence.

## Reviewer shortcuts

Start with these pages when checking validation evidence:

- [Reviewer evidence map](reviewer_evidence.md) links claims to source files,
  tests, reproduction commands, artifacts, and explicit limitations.
- [Physics validation](validation.md) contains the equations, citations,
  still-figure gallery, and numerical gates that used to be too detailed for
  the README.
- [Benchmark command index](benchmarks.md) lists the validation, scaffold,
  comparison, campaign, and neural-ODE commands with expected output families.
- [Validation movies](media.md) separates solver-generated movies from theory
  schematics and keeps the literature anchors next to the visuals.
- [Publication checklist](publication_checklist.md) states which figures are
  ready as validation evidence and which remain roadmap or production-only.

```{toctree}
:maxdepth: 2

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
