# Neural-ODE reproducibility lane

The current neural-ODE lane freezes the dataset and evaluation contract before
adding trainable neural dynamics.  This is deliberate: a neural-ODE claim is not
credible unless the data split, baselines, metrics, and calibration checks are
stable and reproducible first.

## Dataset contract

Generate the FAST deterministic dataset with:

```bash
mhx neural-ode dataset \
  --outdir outputs/neural_ode/seed_qi_fast \
  --seeds 0,1,2,3,4,5 \
  --nx 16 --ny 16 \
  --steps 24 \
  --dt 1e-2
```

Expected files:

- `dataset.npz`
- `splits.json`
- `baseline_metrics.json`
- `calibration.json`
- `experiment_spec.json`
- `validation.json`
- `figures/dataset_targets.png`
- `figures/baseline_rmse.png`
- `figures/calibration_coverage.png`
- `manifest.json`

The dataset arrays are:

| Array | Shape | Meaning |
| --- | --- | --- |
| `seeds` | `(n_seed,)` | Deterministic sample identifiers. |
| `times` | `(n_time,)` | Saved simulation times. |
| `features` | `(n_seed, n_time, n_feature)` | Diagnostic histories used as model inputs. |
| `targets` | `(n_seed, n_time, n_target)` | Forecast targets selected from the feature tensor. |

Default features are mode amplitude, magnetic energy, kinetic energy, total
energy, magnetic-divergence error, $\|\psi\|_2$, and $\|\omega\|_2$.  Default
targets are mode amplitude, total energy, and magnetic-divergence error.

## Baselines

The lane evaluates no-training baselines:

- persistence: $\hat y(t)=y(t_\mathrm{obs})$;
- linear-prefix extrapolation: fit a two-point slope from the observed prefix;
- train-mean history: use the mean target history over training seeds.

For each baseline and split, MHX writes MAE, RMSE, maximum absolute error, and
target-wise scores:

$$
\mathrm{MAE}=\langle |y-\hat y|\rangle,\qquad
\mathrm{RMSE}=\sqrt{\langle (y-\hat y)^2\rangle}.
$$

The calibration file estimates train residual standard deviations and reports
empirical one- and two-sigma coverage on train/validation/test splits.  These
checks are not a probabilistic model; they are a minimum benchmark a later
trainable latent or neural ODE must beat.

## Claim boundary

The manifest is `claim_level = "validation"`.  The current lane supports claims
that the dataset/split/baseline/calibration contract is deterministic and
schema-valid.  It does **not** yet support claims that a neural ODE predicts
nonlinear reconnection better than physics solvers or simple baselines.

`validation.json` uses schema `mhx.neural_ode.reproducibility.gates.v1` and
gates four prerequisites together: the source seed-QI validation passed, the
split manifest is disjoint and complete, all baseline arrays are finite, and
the calibration report was generated from the same target tensor.

## Source links

- [Dataset and baseline implementation](https://github.com/uwplasma/MHX/blob/main/src/mhx/neural_ode/reproducibility.py)
- [Public exports](https://github.com/uwplasma/MHX/blob/main/src/mhx/neural_ode/__init__.py)
- [CLI entrypoint](https://github.com/uwplasma/MHX/blob/main/src/mhx/cli/main.py)
- [Example script](https://github.com/uwplasma/MHX/blob/main/examples/make_neural_ode_reproducibility.py)
- [Tests](https://github.com/uwplasma/MHX/blob/main/tests/test_neural_ode_reproducibility.py)
