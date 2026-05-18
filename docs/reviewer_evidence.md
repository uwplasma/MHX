# Reviewer evidence map

This page is the reviewer-facing map from claims to evidence. It is designed to
make MHX hard to overclaim: every scientific statement should point to a
command, artifact schema, validation gate, source implementation, and explicit
claim boundary.

## Fast entry points

- [Physics validation](validation.md) is the primary figure gallery: it keeps
  the equations, citations, tolerances, expected files, source links, and still
  figures for each validation gate.
- [Benchmarks](benchmarks.md) is the command index for tests, validation
  scaffolds, comparison lanes, neural-ODE bundles, and campaign examples.
- [Validation media](media.md) carries literature-anchored GIFs and separates
  solver output from schematic targets.
- [Long-run evidence](long_run_evidence.md) records longer nonlinear runs with
  skeptical interpretations and explicit non-claims.
- [Publication checklist](publication_checklist.md) states which still figures
  and movies are ready as validation evidence and which are production-only.

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
| `production` | A long simulation, convergence suite, seed/QI check, artifact manifest, and passing promotion-readiness report support the stated physics claim. | Any claim outside the documented duration, resolution, promotion report, and model assumptions. |

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
| Double-Harris nonlinear growth | Dense unstable periodic double-Harris eigenmode grows in the full nonlinear solver. | Small-grid instability-path validation, not Rutherford/plasmoid production. | [current_sheet.py](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/current_sheet.py), [test_current_sheet_eigenvalue_validation.py](https://github.com/uwplasma/MHX/blob/main/tests/test_current_sheet_eigenvalue_validation.py) |
| Seeded double-Harris long run | Scalable base-vs-seeded nonlinear replay with early growth, dominant reconnecting-flux response, Rutherford-width proxy, X/O counts, dissipative energy, current-density histories, and optional movies. | Bounded nonlinear validation, not convergence-backed Rutherford/plasmoid production. | [current_sheet.py](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/current_sheet.py), [test_current_sheet_eigenvalue_validation.py](https://github.com/uwplasma/MHX/blob/main/tests/test_current_sheet_eigenvalue_validation.py) |
| Seeded double-Harris convergence | Same replay swept over tiny resolution and time-step cases with spread gates. | FAST convergence scaffold; production claims still require larger sweeps. | [current_sheet.py](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/current_sheet.py), [test_current_sheet_eigenvalue_validation.py](https://github.com/uwplasma/MHX/blob/main/tests/test_current_sheet_eigenvalue_validation.py) |
| Nonlinear energy budget | Reduced-MHD identity $dE/dt=-\eta\langle j^2\rangle-\nu\langle\omega^2\rangle$. | Nonlinear conservation/dissipation validation. | [nonlinear.py](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/nonlinear.py), [test_nonlinear_energy_budget_validation.py](https://github.com/uwplasma/MHX/blob/main/tests/test_nonlinear_energy_budget_validation.py) |
| Orszag--Tang vortex | Reduced-MHD nonlinear roll-up, high-$k$ transfer, energy decay, and divergence preservation. | Nonlinear morphology validation, not compressible shock validation. | [orszag_tang.py](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/orszag_tang.py), [test_orszag_tang_validation.py](https://github.com/uwplasma/MHX/blob/main/tests/test_orszag_tang_validation.py) |
| Decaying turbulence | Deterministic broadband reduced-MHD current filamentation and high-$k$ transfer. | Turbulence media validation, not converged statistics. | [turbulence.py](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/turbulence.py), [test_turbulence_validation.py](https://github.com/uwplasma/MHX/blob/main/tests/test_turbulence_validation.py) |
| Forced turbulent reconnection | Periodic current sheet with deterministic broadband perturbations, forcing, critical-point-aware reconnection proxy, and fallback counter. | Pedagogical 2-D proxy validation, not 3-D fast-reconnection evidence. | [turbulence.py](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/turbulence.py), [critical_points.py](https://github.com/uwplasma/MHX/blob/main/src/mhx/diagnostics/critical_points.py), [test_turbulence_validation.py](https://github.com/uwplasma/MHX/blob/main/tests/test_turbulence_validation.py) |
| X/O critical points | $|\nabla\psi|$ minima classified by Hessian determinant, with optional sub-cell Newton refinement and frame-to-frame ID tracking. | Visualization and proxy diagnostic validation; not separatrix-event or bifurcation tracking. | [critical_points.py](https://github.com/uwplasma/MHX/blob/main/src/mhx/diagnostics/critical_points.py), [test_critical_points.py](https://github.com/uwplasma/MHX/blob/main/tests/test_critical_points.py) |
| Nonlinear duration policy | $t_\mathrm{end}\ge s_fN_e/\gamma$ before linear-growth or island claims. | Claim-boundary validation. | [duration_policy.py](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/duration_policy.py), [test_duration_policy.py](https://github.com/uwplasma/MHX/blob/main/tests/test_duration_policy.py) |
| Seed-robust QI | Metric stability under deterministic tiny seed perturbations. | FAST sensitivity validation, not production UQ. | [seed_robust_qi.py](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/seed_robust_qi.py), [test_seed_robust_qi.py](https://github.com/uwplasma/MHX/blob/main/tests/test_seed_robust_qi.py) |
| Seed-robust QI sweep | Common-seed perturbation-amplitude sweep with seed-spread and mean-drift gates. | FAST sensitivity validation, not production UQ. | [seed_robust_qi.py](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/seed_robust_qi.py), [test_seed_robust_qi.py](https://github.com/uwplasma/MHX/blob/main/tests/test_seed_robust_qi.py) |
| FAST Rutherford runner | Island-width/reconnection-rate vocabulary on a tiny nonlinear trajectory. | Schema and diagnostic validation only. | [campaign_runner.py](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/campaign_runner.py), [test_campaign_runner.py](https://github.com/uwplasma/MHX/blob/main/tests/test_campaign_runner.py) |
| Rutherford production executor | Restartable reduced-MHD chunks with checkpoint state, histories, resume plans, figures, and hashes. | Execution-path validation; production physics requires a completed target plus a passing promotion report. | [production.py](https://github.com/uwplasma/MHX/blob/main/src/mhx/campaigns/production.py), [test_production_campaign.py](https://github.com/uwplasma/MHX/blob/main/tests/test_production_campaign.py) |
| Rutherford promotion gate | Machine-readable target-completion, convergence, seed-QI, movie, response-amplification, current-sheet geometry, X/O-count, energy-budget, and divergence checks. | Boundary between validation execution bundles and production nonlinear claims. | [production.py](https://github.com/uwplasma/MHX/blob/main/src/mhx/campaigns/production.py), [test_production_campaign.py](https://github.com/uwplasma/MHX/blob/main/tests/test_production_campaign.py) |
| Fitted latent ODE | Frozen FAST seed-QI dataset, deterministic random-feature ODE fit, test metrics, and baseline comparison. | Neural-ODE workflow validation; not production surrogate evidence. | [reproducibility.py](https://github.com/uwplasma/MHX/blob/main/src/mhx/neural_ode/reproducibility.py), [test_neural_ode_reproducibility.py](https://github.com/uwplasma/MHX/blob/main/tests/test_neural_ode_reproducibility.py) |
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
mhx benchmark double-harris-growth --outdir outputs/reviewer/double_harris_growth
mhx benchmark orszag-tang --outdir outputs/reviewer/orszag_tang --movies
mhx benchmark decaying-turbulence --outdir outputs/reviewer/decaying_turbulence --movies
mhx benchmark forced-turbulent-reconnection --outdir outputs/reviewer/forced_reconnection --movies
mhx neural-ode train --outdir outputs/reviewer/neural_ode_latent_fit
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
- `outputs/reviewer/double_harris_growth/validation.json`
- `outputs/reviewer/neural_ode_latent_fit/validation.json`
- `outputs/reviewer/neural_ode_latent_fit/latent_ode_metrics.json`
- `outputs/reviewer/neural_ode_latent_fit/failure_modes.json`
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
- small-grid nonlinear growth of an unstable periodic double-Harris sheet;
- nonlinear reduced-MHD energy-budget consistency;
- nonlinear Orszag--Tang and turbulence media with finite-field, high-$k$, and
  energy/proxy gates;
- explicit duration gates that prevent short nonlinear CI runs from being
  overclaimed;
- deterministic seed-robust QI for short FAST metrics;
- common-seed amplitude-sweep QI with drift and spread gates;
- schema-valid campaign artifacts and publication-pipeline scaffolding.
- FAST latent-ODE dataset/baseline/calibration and fitted-model workflow
  validation.

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
6. Run `mhx campaign rutherford-promotion-check` and require the response,
   convergence, seed-QI, geometry, movie, energy, and divergence gates to pass.
7. Generate flux/current movies with fixed color limits.
8. Write a recursive artifact manifest and include the git commit.
9. Update the claim table in this page and in [paper_plan.md](paper_plan.md).

If any step fails, the claim remains `validation` or `production_template`.
