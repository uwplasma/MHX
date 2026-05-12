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

```{toctree}
:maxdepth: 2

quickstart
architecture
reduced_mhd
diagnostics
output_schema
benchmarks
performance
validation
reviewer_evidence
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
