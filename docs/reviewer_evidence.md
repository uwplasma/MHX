# Reviewer evidence map

This page is the reviewer-facing map from claims to evidence. It is designed to
make MHX hard to overclaim: every scientific statement should point to a
command, artifact schema, validation gate, source implementation, and explicit
claim boundary.

## Evidence standard

A result is reviewer-ready only when all of the following are true:

1. The claim has a declared `claim_level` in a `manifest.json`.
2. The command sequence that generated the result is documented.
3. The output directory contains checksummed artifacts through
   `mhx artifact-manifest`.
4. The plotted quantity is defined in a public API page or source-linked
   implementation.
5. The validation gate has a numerical tolerance and a failing test.
6. The limitation of the gate is written next to the result.

This is stricter than a passing smoke test. A smoke run can prove that IO,
plotting, and diagnostics execute. It cannot prove nonlinear reconnection
physics.

## Claim levels

| Claim level | Allowed statement | Disallowed statement |
| --- | --- | --- |
| `smoke` | The command runs, writes schema-valid outputs, and produces finite diagnostics. | The simulation reproduces a physical regime. |
| `validation` | A specific operator, diagnostic, scaling formula, or FAST sensitivity gate passed a documented test. | The result generalizes outside the tested regime. |
| `production_template` | The generated plan is long enough and complete enough to launch a production campaign. | A nonlinear production result has been obtained. |
| `production` | A long simulation, convergence suite, seed/QI check, and artifact manifest support the stated physics claim. | Any claim outside the documented duration, resolution, and model assumptions. |

The source of truth for these labels is the output schema documentation and the
manifest writer paths:

- [Output schema](output_schema.md)
- [Artifact manifest implementation](https://github.com/uwplasma/MHX/blob/main/src/mhx/io/manifest.py)
- [Reduced-MHD run writer](https://github.com/uwplasma/MHX/blob/main/src/mhx/cli/main.py)
- [Validation-suite writer](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/suite.py)

## Gate taxonomy

| Gate | Physics content | Current claim | Source and tests |
| --- | --- | --- | --- |
| Spectral identities | Fourier derivatives, Laplacian signs, inverse-Laplacian gauge handling. | Validation for smooth periodic grids. | [operators.py](https://github.com/uwplasma/MHX/blob/main/src/mhx/numerics/spectral/operators.py), [test_spectral.py](https://github.com/uwplasma/MHX/blob/main/tests/test_spectral.py) |
| Exact resistive decay | $\partial_t\psi=\eta\nabla^2\psi$ gives $\psi_k(t)=\psi_k(0)e^{-\eta k^2t}$. | Linear induction validation. | [decay.py](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/decay.py), [test_resistive_decay_validation.py](https://github.com/uwplasma/MHX/blob/main/tests/test_resistive_decay_validation.py) |
| FKR/scaling scaffolds | Constant-$\psi$ tearing, Sweet-Parker plasmoid, and ideal-tearing exponents. | Analytic target validation, not solver recovery. | [scaling.py](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/scaling.py), [test_reconnection_scaling_validation.py](https://github.com/uwplasma/MHX/blob/main/tests/test_reconnection_scaling_validation.py) |
| Harris $\Delta'$ | Outer-region Harris tearing ODE and analytic matching. | Numerical outer matching validation. | [fkr.py](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/fkr.py), [test_fkr_window_validation.py](https://github.com/uwplasma/MHX/blob/main/tests/test_fkr_window_validation.py) |
| Direct Harris eigenvalue | Dense finite-difference reduced-MHD tearing eigenproblem at the published reference case. | Single-case tearing eigenvalue validation. | [eigenvalue.py](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/eigenvalue.py), [test_linear_tearing_eigenvalue_validation.py](https://github.com/uwplasma/MHX/blob/main/tests/test_linear_tearing_eigenvalue_validation.py) |
| Finite-domain dispersion/layer | Growth sign, eigenpair residuals, and eigenfunction localization over small scans. | FAST branch and shape validation. | [tearing_eigen.py](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/tearing_eigen.py), [test_linear_tearing_eigenvalue_validation.py](https://github.com/uwplasma/MHX/blob/main/tests/test_linear_tearing_eigenvalue_validation.py) |
| Time-domain replay | RK4 integration of a known linear eigenmode and refit of $\gamma$. | Growth-fit plumbing validation. | [tearing_eigen.py](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/tearing_eigen.py), [test_linear_tearing_eigenvalue_validation.py](https://github.com/uwplasma/MHX/blob/main/tests/test_linear_tearing_eigenvalue_validation.py) |
| Nonlinear energy budget | Reduced-MHD identity $dE/dt=-\eta\langle j^2\rangle-\nu\langle\omega^2\rangle$. | Nonlinear conservation/dissipation validation. | [nonlinear.py](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/nonlinear.py), [test_nonlinear_energy_budget_validation.py](https://github.com/uwplasma/MHX/blob/main/tests/test_nonlinear_energy_budget_validation.py) |
| Nonlinear duration policy | $t_\mathrm{end}\ge s_fN_e/\gamma$ before linear-growth or island claims. | Claim-boundary validation. | [duration_policy.py](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/duration_policy.py), [test_duration_policy.py](https://github.com/uwplasma/MHX/blob/main/tests/test_duration_policy.py) |
| Seed-robust QI | Metric stability under deterministic tiny seed perturbations. | FAST sensitivity validation, not production UQ. | [seed_robust_qi.py](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/seed_robust_qi.py), [test_seed_robust_qi.py](https://github.com/uwplasma/MHX/blob/main/tests/test_seed_robust_qi.py) |
| Seed-robust QI sweep | Common-seed perturbation-amplitude sweep with seed-spread and mean-drift gates. | FAST sensitivity validation, not production UQ. | [seed_robust_qi.py](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/seed_robust_qi.py), [test_seed_robust_qi.py](https://github.com/uwplasma/MHX/blob/main/tests/test_seed_robust_qi.py) |
| FAST Rutherford runner | Island-width/reconnection-rate vocabulary on a tiny nonlinear trajectory. | Schema and diagnostic validation only. | [campaign_runner.py](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/campaign_runner.py), [test_campaign_runner.py](https://github.com/uwplasma/MHX/blob/main/tests/test_campaign_runner.py) |
| Readiness report | Release-vs-publication gate assembled from validation-suite artifacts. | Public release evidence, not production physics. | [readiness.py](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/readiness.py), [test_readiness_report.py](https://github.com/uwplasma/MHX/blob/main/tests/test_readiness_report.py) |

## Reviewer reproduction sequence

The minimal evidence bundle is generated by:

```bash
python -m pip install -e ".[dev,docs]"
python tools/check_legacy_imports.py
python -m ruff check src tests examples tools
mhx validate all --outdir outputs/reviewer/validation_suite
mhx benchmark timing --outdir outputs/reviewer/timing --repeats 3 --warmups 1
mhx benchmark seed-robust-qi --outdir outputs/reviewer/seed_robust_qi
mhx benchmark seed-robust-qi-sweep --outdir outputs/reviewer/seed_robust_qi_sweep
mhx campaign rutherford-template --outdir outputs/reviewer/rutherford_template
mhx campaign rutherford-run-fast --outdir outputs/reviewer/rutherford_fast
mhx validate readiness --suite outputs/reviewer/validation_suite --outdir outputs/reviewer/readiness
mhx artifact-manifest outputs/reviewer
sphinx-build -W -b html docs docs/_build/html
```

The resulting evidence bundle should contain:

- `outputs/reviewer/validation_suite/validation_suite.json`
- `outputs/reviewer/validation_suite/artifact_manifest.json`
- `outputs/reviewer/timing/timing.json`
- `outputs/reviewer/seed_robust_qi/validation.json`
- `outputs/reviewer/seed_robust_qi_sweep/validation.json`
- `outputs/reviewer/rutherford_template/duration_assessment.json`
- `outputs/reviewer/rutherford_fast/rutherford_fast_histories.npz`
- `outputs/reviewer/readiness/readiness.json`
- `outputs/reviewer/artifact_manifest.json`

## What is strong today

The current repository can defend:

- exact linear diffusion and spectral-operator identities;
- one published Harris tearing eigenvalue anchor;
- finite-domain tearing branch sign/residual checks;
- eigenfunction-layer localization in a FAST scan;
- time-domain replay of known linear eigenmodes;
- nonlinear reduced-MHD energy-budget consistency;
- explicit duration gates that prevent short nonlinear CI runs from being
  overclaimed;
- deterministic seed-robust QI for short FAST metrics;
- common-seed amplitude-sweep QI with drift and spread gates;
- schema-valid campaign artifacts and publication-pipeline scaffolding.

## What is not yet strong

The current repository should not claim:

- calibrated FKR/Coppi dispersion from the nonlinear PDE solver;
- Rutherford algebraic island growth;
- Sweet-Parker plasmoid-chain formation;
- production scaling on large grids;
- neural-ODE predictive superiority;
- inverse-design superiority over grid search.

Those are not wording issues; they require new long runs, convergence suites,
baselines, and figures.

## Escalation path from validation to production

To promote a nonlinear result from `validation` to `production`, run this
checklist in order:

1. Generate a duration-guarded campaign template.
2. Run the production simulation with fixed seed, x64/JIT settings, and
   archived config.
3. Repeat at no fewer than two grid resolutions and two time steps.
4. Run the seed-robust QI lane on the same diagnostic family.
5. Verify the nonlinear energy budget and magnetic-divergence diagnostics.
6. Generate flux/current movies with fixed color limits.
7. Write a recursive artifact manifest and include the git commit.
8. Update the claim table in this page and in [paper_plan.md](paper_plan.md).

If any step fails, the claim remains `validation` or `production_template`.
