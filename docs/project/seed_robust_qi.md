# Seed-robust QI lane

The seed-robust quantitative indicator (QI) lane measures whether FAST
reduced-MHD trajectories are stable under tiny stochastic initial-condition
perturbations. It is a reproducibility and sensitivity gate, not a production
uncertainty-quantification study.

## What the lane checks

For a deterministic base ensemble seed, MHX expands a fixed list of child seeds,
adds zero-mean, unit-RMS-scaled perturbations to the cosine-tearing initial
condition, evolves each FAST trajectory, and records:

- `gamma_fit`
- `final_total_energy`
- `final_magnetic_energy`
- `final_kinetic_energy`
- `final_magnetic_divergence_linf`

For each metric the lane writes the ensemble samples, mean, sample standard
deviation, coefficient of variation (CV), minimum, maximum, and pass/fail gate.

The reported coefficient of variation is

$$
\mathrm{CV}(m)=
\frac{\operatorname{std}_s[m_s]}
     {\max(|\operatorname{mean}_s[m_s]|,\epsilon_m)},
$$

where $s$ indexes seeds and $\epsilon_m$ is the metric-specific floor used to
avoid meaningless ratios when the physical quantity is expected to be near
zero. Gates are applied to CV, absolute mean, or both depending on the metric.

## Physics-motivated gates

The default gates are deliberately conservative for FAST validation:

- Growth/decay fits should be seed robust at the few-percent CV level.
- Magnetic and total energy should be insensitive to tiny perturbations at the
  `1e-3` CV level.
- Kinetic energy may be near zero in the short linear FAST run, so it is gated
  by both CV and absolute mean.
- `B_perp = (∂_y ψ, -∂_x ψ)` is analytically solenoidal, so spectral magnetic
  divergence should remain near roundoff.

These gates complement the FKR, current-sheet, energy-budget, and duration
validation lanes by checking that stochastic seed choice does not dominate the
reported FAST trajectory diagnostics.

The seed perturbation itself is deliberately smooth and tiny. It is not a
surrogate for broadband turbulent noise, kinetic particle noise, or a physical
uncertainty model. Its purpose is narrower: catch fragile diagnostics and
accidental seed dependence before larger campaigns are launched.

## Artifacts

The single-amplitude writer is available as a Python API:

```python
from mhx.benchmarks import write_seed_robust_qi_validation

write_seed_robust_qi_validation("outputs/benchmarks/seed_robust_qi")
```

or from the CLI:

```bash
mhx benchmark seed-robust-qi \
  --outdir outputs/benchmarks/seed_robust_qi \
  --seeds 0,1,2,3 \
  --nx 16 --ny 16 \
  --t-end 0.12
```

Expected files:

- `diagnostics.json`
- `validation.json`
- `ensemble.npz`
- `figures/qi_summary.png` when `matplotlib` is available
- `manifest.json`

## Amplitude-sweep QI

The stronger reviewer-facing gate keeps the seed list fixed and sweeps the
seed-noise amplitude `epsilon`.  It validates two questions:

1. At each `epsilon`, are the fitted growth rate, energies, and divergence
   metrics insensitive to seed choice?
2. As `epsilon` increases through a tiny admissible range, do metric means stay
   close to the zero-noise baseline?

```bash
mhx benchmark seed-robust-qi-sweep \
  --outdir outputs/benchmarks/seed_robust_qi_sweep \
  --seeds 0,1,2,3 \
  --amplitudes 0,1e-9,1e-8 \
  --nx 16 --ny 16 --steps 12
```

Expected files:

- `diagnostics.json`
- `validation.json`
- `sweep.npz`
- `figures/qi_sweep_cv.png`
- `figures/qi_sweep_mean_drift.png`
- `manifest.json`

The NPZ stores a metric cube with shape
`(n_amplitudes, n_seeds, n_metrics)`.  The JSON diagnostics store
`metric_cv_max` and `metric_relative_mean_drift_max` so reviewers can audit
whether failures come from seed spread or amplitude drift.

Pure helpers are also exposed for tests and downstream rollout:
`generate_seed_ensemble`, `seeded_perturbation`,
`make_seeded_initial_state`, `compute_metric_statistics`, and
`default_seed_robust_qi_gates`.

## Claim boundary

The manifest is `claim_level = "validation"`. The QI lane can support claims
that FAST diagnostics are stable under tiny smooth seed perturbations and,
for the sweep command, under a documented perturbation-amplitude range for the
tested configuration. It cannot support production uncertainty quantification,
plasmoid-count statistics, or turbulent ensemble convergence. Those require
larger ensembles, long-duration campaign manifests, and convergence sweeps.

For a production nonlinear campaign, use the QI lane as a secondary check after
the duration and convergence gates pass. A production QI bundle should archive:

- the exact seed list;
- the base configuration and perturbation amplitude;
- all scalar samples, not only means;
- failure thresholds chosen before the run;
- the figure and JSON manifest in the production artifact directory.

## Source links

- QI implementation:
  [`src/mhx/benchmarks/seed_robust_qi.py`](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/seed_robust_qi.py)
- CLI entrypoint:
  [`src/mhx/cli/main.py`](https://github.com/uwplasma/MHX/blob/main/src/mhx/cli/main.py)
- Validation-suite integration:
  [`src/mhx/benchmarks/suite.py`](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/suite.py)
- Tests:
  [`tests/test_seed_robust_qi.py`](https://github.com/uwplasma/MHX/blob/main/tests/test_seed_robust_qi.py)

## Review checklist

Before citing a seed-robust QI result, verify:

1. `validation.json` has `passed = true`.
2. `manifest.json` records `claim_level = "validation"` or a justified
   production claim after longer campaigns.
3. `ensemble.npz` contains the full metric samples.
4. `figures/qi_summary.png` matches the archived JSON values.
5. The seed list and perturbation amplitude are written in the command or
   config bundle.
