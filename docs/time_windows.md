# Simulation time windows

MHX now treats simulation duration as a first-class validation object. The rule
for any linear-growth or nonlinear-island production claim is

$$
t_\mathrm{end} \ge s_f\frac{N_e}{\gamma},
$$

where $N_e$ is the required number of e-folds, $\gamma$ is the relevant growth
rate, and $s_f$ is a safety factor. The default production policy uses
$N_e=10$ and $s_f=1$ for linear-growth observation; the Rutherford/island
template uses $s_f=3$ to leave room for a resolved linear phase plus nonlinear
tracking.

## Duration policy command

```bash
mhx benchmark duration-policy --outdir outputs/benchmarks/duration_policy
```

Expected files:

- `outputs/benchmarks/duration_policy/duration_policy.json`
- `outputs/benchmarks/duration_policy/duration_policy.md`
- `outputs/benchmarks/duration_policy/validation.json`
- `outputs/benchmarks/duration_policy/manifest.json`

The policy intentionally separates **intended-scope sufficiency** from
**nonlinear-claim sufficiency**:

- FAST smoke, linear operator replay, nonlinear energy-budget, and
  differentiability gates may be long enough for their engineering/validation
  purpose while remaining invalid as nonlinear reconnection claims.
- Future production templates must satisfy the e-fold rule, otherwise the
  policy validation fails.

## Long campaign template

The next nonlinear result should start from a generated campaign template rather
than a smoke-test config:

```bash
mhx campaign rutherford-template \
  --outdir outputs/campaigns/rutherford_template \
  --nx 128 --ny 128 \
  --dt 0.1 \
  --target-saved-frames 400
```

This command does not run the expensive simulation. It writes a
`campaign_config.toml`, a `duration_assessment.json`, and a manifest with
`claim_level = "production_template"`. The generated config uses
$t_\mathrm{end}\approx2290.1$ for the default Harris growth-rate anchor, so it
is long enough for a ten-e-fold linear phase plus nonlinear island tracking.

## Current duration classification

| Workflow | Current/default time | Intended scope | Nonlinear claim status |
| --- | ---: | --- | --- |
| `linear_tearing_fast` | $t=0.1$ | smoke/IO/plotting | not a nonlinear claim |
| `linear_tearing_timedomain` | $t=80$ | exact eigenmode replay and growth-fit plumbing | not a nonlinear claim |
| `nonlinear_energy_budget` | $t=0.8$ | nonlinear conservation/dissipation identity | not a nonlinear island/plasmoid claim |
| `future_harris_linear_growth_campaign` | $t\approx763.4$ | production linear growth | long enough for 10 e-folds |
| `future_rutherford_island_campaign` | $t\approx2290.1$ | production island tracking | long enough with safety factor 3 |

The direct Harris benchmark anchor is $\gamma\simeq0.0131$, so ten e-folds
require $10/0.0131\approx763.4$. This is why the current nonlinear budget run
is explicitly labeled a code-validity gate rather than a Rutherford or plasmoid
simulation.

## Python guard for future workflows

Future production runners should call the guard before launching expensive jobs:

```python
from mhx.benchmarks import require_duration_for_claim

require_duration_for_claim(
    name="rutherford_campaign",
    purpose="nonlinear island-width growth",
    t_end=800.0,
    growth_rate=1.31e-2,
    required_efolds=10.0,
)
```

If `t_end` is too short, the guard raises a `ValueError` with the required
minimum. Tests cover the short-run failure path and the future-template pass
path.

## What remains for nonlinear publication plots

A duration that satisfies the e-fold rule is necessary but not sufficient.
Publication-grade nonlinear plots also need:

- resolution/time-step convergence;
- fixed color limits for flux and current movies;
- reconnected-flux and island-width histories;
- current-sheet length/thickness and reconnection-rate proxies;
- magnetic/kinetic/total energy and integrated dissipation residuals;
- artifact manifests with code commit, API version, and dependency set.

## Practical duration labels

Use these labels consistently in docs, figures, and manifests:

| Label | Duration status | Allowed use |
| --- | --- | --- |
| `short_validation` | Shorter than the relevant e-fold window. | Operator, IO, schema, differentiability, and energy-budget checks. |
| `fast_validation` | A short non-CI validation run below the README media minimum. | Local smoke/plotting checks only; not README or release media. |
| `ci_fast` | Explicitly bounded CI run, currently `t_end=10` for double-Harris media plumbing. | CI artifact generation and schema/movie checks. |
| `readme_release_media` | Longer validation media run, currently `t_end=120` for the Harris-sheet README contour movies and at least `t_end=100` for the validation preset. | README/release morphology media with validation claim level, not production physics. |
| `linear_window` | At least $N_e/\gamma$. | Linear growth-rate measurement if the mode remains in the linear regime. |
| `nonlinear_window` | At least $s_fN_e/\gamma$ with $s_f>1$. | Candidate island-growth or plasmoid campaign, still subject to convergence. |
| `overresolved_window` | Longer than the nonlinear window and accompanied by convergence checks. | Preferred for production paper figures. |

The label should be stored in the run notes or manifest claim scope. If the
label is `short_validation`, `fast_validation`, or `ci_fast`, figures should not
use wording such as "Rutherford phase", "plasmoid onset", or "nonlinear
saturation". The `readme_release_media` label permits public visual previews,
but it is still a validation-media label rather than a production reconnection
claim.
