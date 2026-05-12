# Production campaign templates

MHX separates **campaign planning artifacts** from completed production
simulations. A template can prove that a proposed run is long enough for its
declared growth-rate window, but it is not itself a nonlinear reconnection
result.

## Rutherford island template

For nonlinear island growth, the template enforces

$$
t_\mathrm{end}\ge s_f\frac{N_e}{\gamma},
$$

with the default Harris-sheet benchmark value $\gamma=1.31\times10^{-2}$,
$N_e=10$, and nonlinear safety factor $s_f=3$. The resulting default final time
is $t_\mathrm{end}\simeq2290.1$, which leaves one ten-e-fold linear-growth
window plus additional time for island-width tracking.

Generate the template without launching the expensive simulation:

```bash
mhx campaign rutherford-template \
  --outdir outputs/campaigns/rutherford_template \
  --nx 128 --ny 128 \
  --dt 0.1 \
  --target-saved-frames 400
```

Expected files:

- `outputs/campaigns/rutherford_template/campaign.json`
- `outputs/campaigns/rutherford_template/campaign_config.toml`
- `outputs/campaigns/rutherford_template/duration_assessment.json`
- `outputs/campaigns/rutherford_template/validation.json`
- `outputs/campaigns/rutherford_template/manifest.json`

The manifest has `claim_level = "production_template"`. This is intentionally
different from `claim_level = "production"`; a reviewer should not interpret the
template as evidence of Rutherford growth until the actual run, convergence
suite, histories, figures, and movies have been generated.

## Required production outputs

A completed Rutherford campaign should add, at minimum:

- reconnected flux $\psi_1(t)$;
- island width $W(t)=4\sqrt{|\psi_1|/|B_y'(0)|}$;
- current-sheet length, thickness, and aspect ratio;
- reconnection-rate proxy $E_\mathrm{rec}$;
- magnetic/kinetic/total energy and dissipation-budget residual;
- fixed-color flux and current movies;
- a resolution sweep and a time-step sweep.

The template records these requirements in `campaign.json` so the paper
pipeline can fail closed if a later production run omits them.

## FAST validation runner

The FAST runner exercises the same Rutherford-campaign diagnostic vocabulary on
a tiny nonlinear reduced-MHD trajectory. It is deterministic for the requested
seed list and writes validation artifacts only:

```python
from mhx.benchmarks import run_rutherford_campaign_fast

run_rutherford_campaign_fast(
    "outputs/campaigns/rutherford_fast",
    seeds=[0, 1, 2],
    shape=(16, 16),
    dt=1.0e-2,
    steps=20,
)
```

Expected files:

- `outputs/campaigns/rutherford_fast/rutherford_fast_histories.npz`
- `outputs/campaigns/rutherford_fast/diagnostics.json`
- `outputs/campaigns/rutherford_fast/validation.json`
- `outputs/campaigns/rutherford_fast/campaign_template.json`
- `outputs/campaigns/rutherford_fast/manifest.json`
- `outputs/campaigns/rutherford_fast/figures/rutherford_fast_histories.png`

The history schema stores `time`, `seed`, `reconnected_flux`,
`rutherford_island_width`, `reconnection_rate_proxy`, magnetic/kinetic/total
energy, magnetic-divergence error, and a current-density proxy. Gates check that
the run is finite, short relative to the production template, energy growth is
within tolerance, magnetic divergence is bounded, and the manifest uses
`claim_level = "validation"` or `claim_level = "smoke"`. It must not be used as
evidence of production Rutherford island growth.

The operational details and production acceptance criteria are documented in
[campaign_runner.md](campaign_runner.md). In short: the FAST runner proves the
schema, diagnostic names, plot path, and seed determinism. A production runner
must additionally prove duration, convergence, budget closure, fixed-color
movies, and reproducibility.

## Production planning and resume scaffold

The production scaffold extends the template into an operational bundle with
walltime chunking, checkpoint cadence, resume metadata, and a runbook:

```bash
mhx campaign rutherford-plan-production \
  --outdir outputs/campaigns/rutherford_production_plan \
  --nx 128 --ny 128 \
  --dt 0.1 \
  --target-saved-frames 400
```

Expected additional files:

- `outputs/campaigns/rutherford_production_plan/campaign_plan.json`
- `outputs/campaigns/rutherford_production_plan/runbook.md`
- `outputs/campaigns/rutherford_production_plan/job_array.json`
- `outputs/campaigns/rutherford_production_plan/checkpoints/checkpoint_index.json`

The checkpoint index is intentionally empty at planning time. A long-run
executor registers restartable state files with
`mhx.campaigns.write_checkpoint_metadata(...)`; the command
`mhx campaign rutherford-resume-plan <run-dir>` then chooses the latest valid
checkpoint by verifying artifact hashes.

A laptop-safe example that writes the same planning bundle is available at
`examples/make_rutherford_production_plan.py`.

## Source links

- Campaign template implementation:
  [`src/mhx/benchmarks/campaigns.py`](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/campaigns.py)
- FAST campaign runner:
  [`src/mhx/benchmarks/campaign_runner.py`](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/campaign_runner.py)
- Production campaign scaffold:
  [`src/mhx/campaigns/production.py`](https://github.com/uwplasma/MHX/blob/main/src/mhx/campaigns/production.py)
- Duration guard:
  [`src/mhx/benchmarks/duration_policy.py`](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/duration_policy.py)
- CLI entrypoint:
  [`src/mhx/cli/main.py`](https://github.com/uwplasma/MHX/blob/main/src/mhx/cli/main.py)
- Tests:
  [`tests/test_campaign_templates.py`](https://github.com/uwplasma/MHX/blob/main/tests/test_campaign_templates.py)
- FAST runner tests:
  [`tests/test_campaign_runner.py`](https://github.com/uwplasma/MHX/blob/main/tests/test_campaign_runner.py)
- Production scaffold tests:
  [`tests/test_production_campaign_scaffold.py`](https://github.com/uwplasma/MHX/blob/main/tests/test_production_campaign_scaffold.py)
