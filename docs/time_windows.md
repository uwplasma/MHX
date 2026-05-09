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
