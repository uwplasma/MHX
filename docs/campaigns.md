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

## Source links

- Campaign template implementation: `src/mhx/benchmarks/campaigns.py`
- Duration guard: `src/mhx/benchmarks/duration_policy.py`
- CLI entrypoint: `src/mhx/cli/main.py`
- Tests: `tests/test_campaign_templates.py`
