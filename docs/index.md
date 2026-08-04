# MHX documentation

MHX runs differentiable, two-dimensional reduced-MHD models in JAX. MHX owns
the plasma model. SOLVAX owns the general numerical solvers.

Start with these pages:

1. [Install MHX](install.md).
2. [Run a model from Python](quickstart.md).
3. [Read the reduced-MHD equations](reduced_mhd.md).
4. [Check the output format](output_schema.md).
5. [Check each validation limit](validation.md).

The package source is under `src/mhx/`. Beginner scripts are under
`examples/gallery/`. Benchmark commands produce review records and publication
checks.

MHX currently supports periodic two-dimensional reduced MHD. It does not solve
full three-dimensional MHD. Read each artifact manifest before you use a
validation run as research evidence.

```{toctree}
:maxdepth: 2

install
tutorial
quickstart
writing_style
reviewer_evidence
validation
benchmarks
architecture
reduced_mhd
diagnostics
output_schema
performance
long_run_evidence
nonlinear_campaign_evidence
seed_robust_qi
neural_ode_reproducibility
time_windows
campaigns
campaign_runner
publication_checklist
paper_plan
paper_pipeline
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
