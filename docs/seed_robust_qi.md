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

## Artifacts

The writer is available as a Python API:

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

Pure helpers are also exposed for tests and downstream rollout:
`generate_seed_ensemble`, `seeded_perturbation`,
`make_seeded_initial_state`, `compute_metric_statistics`, and
`default_seed_robust_qi_gates`.

## Claim boundary

The manifest is `claim_level = "validation"`. The QI lane can support claims
that FAST diagnostics are stable under tiny smooth seed perturbations for the
tested configuration. It cannot support production uncertainty quantification,
plasmoid-count statistics, or turbulent ensemble convergence. Those require
larger ensembles, long-duration campaign manifests, and convergence sweeps.

## Source links

- QI implementation: `src/mhx/benchmarks/seed_robust_qi.py`
- CLI entrypoint: `src/mhx/cli/main.py`
- Validation-suite integration: `src/mhx/benchmarks/suite.py`
- Tests: `tests/test_seed_robust_qi.py`
