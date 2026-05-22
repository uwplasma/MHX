# Publication plot checklist

This page defines what counts as a publication-ready MHX figure or movie. It is
not a style guide only; it is a scientific evidence checklist. A polished plot
with a weak gate should not appear as a physics result.

## Global figure rules

Every figure included in the paper-style documentation should have:

- a command that regenerates it;
- an output file path and schema;
- a source-code link for the computed quantity;
- a test that fails if the underlying gate fails;
- axis labels with physical or normalized units;
- a caption that states the claim level;
- a limitation sentence when the result is analytic, FAST, or validation-only;
- fixed color ranges for movies or multi-panel image comparisons;
- a manifest entry with SHA-256 checksum;
- a passing `python examples/tools/verify_paper_artifacts.py` check before
  reviewer handoff.

## Current figure readiness

| Figure family | Files | Claim level | Paper readiness |
| --- | --- | --- | --- |
| Exact resistive decay | `decay_amplitude.png`, `decay_energy.png`, `decay_relative_error.png` | `validation` | Ready as a linear diffusion/operator gate. |
| Reconnection scaling formulas | `fkr_scaling.png`, `plasmoid_scaling.png`, `ideal_tearing_scaling.png` | `validation` | Ready only as analytic theory targets. |
| FKR window and growth assembly | `fkr_constant_psi_window.png`, `fkr_growth_rate.png` | `validation` | Ready as asymptotic-target scaffolding, not direct PDE recovery. |
| Harris $\Delta'$ and eigenvalue | `harris_delta_prime.png`, `linear_tearing_eigenvalue.png` | `validation` | Ready as tearing-specific validation for documented regimes. |
| Dispersion/layer/time replay | `linear_tearing_dispersion.png`, `linear_tearing_layer.png`, `linear_tearing_timedomain.png` | `validation` | Ready as FAST branch, shape, and growth-fit gates. |
| Periodic current sheet | `periodic_current_sheet_spectrum.png`, `periodic_current_sheet_timedomain.png`, `periodic_current_sheet_nonlinear_bridge.png` | `validation` | Ready as operator and differentiability gates. |
| Periodic double-Harris growth | `periodic_double_harris_nonlinear_growth.png` | `validation` | Ready as small-grid instability-path evidence; not a Rutherford/plasmoid production claim. |
| Seeded double-Harris long run | `periodic_double_harris_seeded_long_run.png`, optional flux/current GIFs, residual-flux GIFs | `validation` | Ready as bounded nonlinear evidence with early growth, response amplification, X/O counts, and dissipative energy; reviewer media should include residual flux or proxy histories, and production claims still need seed/aspect-ratio/Lundquist sweeps. |
| Seeded double-Harris convergence | `periodic_double_harris_convergence.png` | `validation` | Ready as convergence-backed validation evidence, including the medium GPU-assisted `32/48/64` sweep and latest bounded `64/96/128` GPU gate; still not production plasmoid/Rutherford evidence. |
| Seeded double-Harris parameter sweep | `periodic_double_harris_parameter_sweep.png` | `validation` | Ready as finite-response robustness evidence across mode/width/resistivity sweeps; still not a scaling-law or production reconnection claim. |
| Seeded double-Harris promotion report | `promotion_matrix.png`, `promotion_readiness.json` | `validation` only | Ready as the boundary between single-run media and convergence-backed validation evidence; not a production physics gate. |
| Nonlinear energy budget | `nonlinear_energy_budget.png` | `validation` | Ready as nonlinear conservation/dissipation evidence. |
| Orszag--Tang vortex media | `orszag_tang_summary.png`, optional flux/current/vorticity GIFs | `validation` | Ready as nonlinear reduced-MHD morphology and high-$k$ transfer evidence, not full-MHD shock validation. |
| Decaying turbulence media | `decaying_mhd_turbulence_summary.png`, optional flux/current GIFs | `validation` | Ready as deterministic reduced-MHD turbulence morphology and current-filament evidence; not turbulence-statistics evidence. |
| Forced turbulent reconnection media | `forced_turbulent_reconnection_summary.png`, optional flux/current GIFs, `readiness/promotion_matrix.png` | `validation` | Ready as a pedagogical forced current-sheet/reconnection-proxy example with a validation-only readiness gate; not 3-D fast-reconnection evidence. |
| Nonlinear duration audit | `nonlinear_duration_audit.png` | `validation` | Ready as an overclaim-prevention figure. |
| Seed-robust QI | `qi_summary.png` | `validation` | Ready as FAST seed-sensitivity evidence after generation in the evidence bundle. |
| Seed-robust QI sweep | `qi_sweep_cv.png`, `qi_sweep_mean_drift.png` | `validation` | Ready as FAST perturbation-amplitude sensitivity evidence. |
| Rutherford FAST runner | `rutherford_fast_histories.png` | `validation` | Ready as schema/diagnostic demonstration only. |
| Readiness report | `readiness_matrix.png` | `validation` | Ready as a release-vs-paper claim boundary figure. |
| Rutherford production executor | `production_histories.png`, `current_sheet_aspect_ratio.png`, optional fixed-scale GIFs | `validation` for partial chunks | Ready as restart/resume execution evidence; not yet a converged nonlinear physics claim. |
| Rutherford promotion gate | `promotion_matrix.png`, `promotion_readiness.json` | `production` only if every gate passes | Ready as the machine-readable boundary between validation chunks and reviewer-facing production claims. |
| Rutherford production physics | long-run histories, convergence figures, seed-QI evidence, fixed-scale movies, response diagnostics, and promotion report | `production` only after full target completion and a passing `rutherford-promotion-check` report | Not ready. Requires long run plus convergence, seed-QI, geometry/X-O counts, positive reconnecting-flux/island-width response, and explicit production-claim promotion. |
| Plasmoid production | not generated | none | Not ready. Requires long thin sheet, secondary islands, and convergence. |
| Neural ODE dataset/baselines | `dataset_targets.png`, `baseline_rmse.png`, `calibration_coverage.png` | `validation` | Ready as reproducibility protocol. |
| Fitted latent ODE | `latent_ode_predictions.png`, `latent_ode_rmse_comparison.png` | `validation` | Ready as FAST train/test evidence; not yet production surrogate evidence. |

## Validation figures already shipped

The strongest current tearing-specific result is the direct Harris-sheet
eigenvalue gate:

```{image} _static/validation/linear_tearing_eigenvalue/linear_tearing_eigenvalue.png
:alt: Direct Harris-sheet tearing eigenvalue gate
:width: 760px
```

The strongest current nonlinear code-validity result is the energy-budget gate:

```{image} _static/validation/nonlinear_energy_budget/nonlinear_energy_budget.png
:alt: Nonlinear reduced-MHD energy budget gate
:width: 760px
```

The strongest current nonlinear instability-path result is the double-Harris
growth gate:

```{image} _static/validation/periodic_double_harris_nonlinear_growth/periodic_double_harris_nonlinear_growth.png
:alt: Periodic double-Harris nonlinear growth gate
:width: 760px
```

The strongest current scalable nonlinear replay is the seeded double-Harris
long-run gate:

```{image} _static/validation/periodic_double_harris_seeded_long_run/figures/periodic_double_harris_seeded_long_run.png
:alt: Seeded periodic double-Harris nonlinear long run
:width: 760px
```

The current nonlinear evidence boundary is the double-Harris promotion report,
the medium GPU-assisted resolution/time-step validation sweep, and the latest
bounded GPU validation gate:

```{image} _static/validation/periodic_double_harris_convergence/periodic_double_harris_convergence.png
:alt: Seeded periodic double-Harris convergence evidence
:width: 760px
```

```{image} _static/validation/long_runs/double_harris_convergence_gpu_n32_48_64_t16/periodic_double_harris_convergence.png
:alt: Medium GPU-assisted double-Harris convergence sweep
:width: 760px
```

The most important claim-boundary figure is the nonlinear duration audit:

```{image} _static/validation/nonlinear_duration_audit/nonlinear_duration_audit.png
:alt: Nonlinear duration audit
:width: 760px
```

These claim-boundary figures should be shown together in reviewer discussions:
one tearing benchmark, one nonlinear growth-path gate, one seeded long-run
response plot, one convergence-backed validation sweep, one nonlinear identity,
and one explicit warning that short nonlinear runs are not island/plasmoid
evidence.

The latest GPU bundle,
`outputs/campaigns/gpu_nonlinear_20260522_085049/double_harris_long_n128_t160`,
is ready only as validation media: the local summary reports
`gate_ready = true`, `production_claim_ready = false`, and
`claim_level_if_gates_pass = "validation"`. It passes duration, convergence,
X/O-count, reconnecting-flux, island-width, fixed-scale movie, and manifest
gates with `reconnected_flux_amplification = 8.356`,
`island_width_amplification = 2.891`, `max_x_point_count = 4`, and
`max_o_point_count = 2`. The remaining production blockers are the explicit
validation-only promotion declaration, lack of Rutherford/Sweet-Parker/plasmoid
physics target, and missing production-scale seed, width/aspect, Lundquist,
duration, and independent production-claim promotion evidence.

## Nonlinear plot acceptance gates

Before adding a nonlinear island or plasmoid plot to the paper set, require:

1. `duration_assessment.json` passes with the declared growth rate.
2. `histories.npz` contains reconnected flux, island width, reconnection rate,
   energy terms, divergence error, and current-sheet geometry.
3. `validation.json` includes pass/fail checks for duration, finite values,
   energy-budget residual, divergence, and convergence.
4. The plotted interval includes the linear phase and the nonlinear phase being
   claimed.
5. Resolution and time-step comparison curves appear on the same axes or in a
   companion panel.
6. Movies show flux and current with fixed color limits and a clear timestamp.
7. Seeded double-Harris media include residual flux
   $\psi_\mathrm{perturbed}-\psi_\mathrm{base}$ or proxy histories next to raw
   total-field movies.
8. `mhx campaign rutherford-promotion-check` passes with the convergence and
   seed-QI bundle paths used for the paper artifact.
9. The promotion report shows peak/initial reconnecting-flux and island-width
   amplification above the declared thresholds; purely dissipative duration
   tests remain validation artifacts.

## Neural-ODE figure acceptance gates

Neural-ODE figures should not be added as novelty claims until the experiment
has:

- deterministic dataset generation from solver trajectories;
- train/validation/test splits that are saved in the manifest;
- at least one non-neural baseline such as persistence or AR(1);
- metric plots for MAE/MSE on $f_\mathrm{kin}$ and $C_\mathrm{plasmoid}$;
- calibration or uncertainty diagnostics if probabilistic outputs are used;
- failure-case examples where the surrogate extrapolates poorly;
- source-linked training and evaluation code.

The current FAST latent-ODE artifacts satisfy these gates as a validation
workflow and belong in the methods/reproducibility section. They should not be
used as novelty claims about production reconnection surrogates until the same
protocol is run on production-quality nonlinear trajectories and remains
superior to baselines on held-out regimes.

## Movie rules

Movies are effective for reviewer communication but dangerous for overclaiming.
Every movie should state one of:

- `validation movie`: generated from a validated benchmark or operator replay;
- `theory schematic`: generated from formulas, not from MHX nonlinear PDE data;
- `production movie`: generated from a long production run with convergence and
  manifest evidence.

Double-Harris public media has an extra residual-flux rule: raw total-flux
movies are acceptable QA artifacts, but reviewer/publication media should show
the seeded response with $\Delta\psi=\psi_\mathrm{perturbed}-\psi_\mathrm{base}$,
reconnection-proxy histories, or a gate-summary table. A visually clear
residual-flux movie still inherits the manifest claim level; it does not turn a
validation promotion into a production claim.

The README currently ships validation/theory movies only:

- [README media docs](media.md)
- [media generator](https://github.com/uwplasma/MHX/blob/main/examples/make_readme_media.py)
- [media tests](https://github.com/uwplasma/MHX/blob/main/tests/test_readme_media.py)

## Source map for publication plots

- [Paper artifact verifier](https://github.com/uwplasma/MHX/blob/main/examples/tools/verify_paper_artifacts.py)
- [Validation media generator](https://github.com/uwplasma/MHX/blob/main/examples/make_validation_media.py)
- [README media generator](https://github.com/uwplasma/MHX/blob/main/examples/make_readme_media.py)
- [Figure manifest](figures/manifest.toml)
- [Double-Harris validation and promotion checks](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/current_sheet.py)
- [Forced turbulent-reconnection readiness](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/turbulence.py)
- [Reduced-MHD plotting helpers](https://github.com/uwplasma/MHX/blob/main/src/mhx/plotting/reduced_mhd.py)
- [Artifact manifests](https://github.com/uwplasma/MHX/blob/main/src/mhx/io/manifest.py)
- [Docs-link tests](https://github.com/uwplasma/MHX/blob/main/tests/test_docs_links.py)
