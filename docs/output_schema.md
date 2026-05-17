# Output schema

Every active MHX run writes a JSON manifest plus schema-versioned data files.
Artifacts also record `api_version = "v1"` so loaders can enforce the public API
contract selected by `MHX_API_VERSION`.

## Run directory

`mhx run examples/linear_tearing.toml --outdir outputs/smoke` writes:

- `config_effective.json`: JSON serialization of the effective config.
- `diagnostics.json`: scalar JSON diagnostics.
- `trajectory.npz`: compressed trajectory arrays using schema
  `mhx.reduced_mhd.trajectory.v1`.
- `manifest.json`: file list, SHA-256 hashes, and claim metadata.

## Manifest claim levels

Every v1 `manifest.json` contains:

| Key | Meaning |
| --- | --- |
| `schema` | Manifest schema, currently `mhx.manifest.v1`. |
| `api_version` | Public API version selected by `MHX_API_VERSION`, currently `v1`. |
| `created_utc` | ISO-8601 creation timestamp in UTC. |
| `mhx_version` | Package version that wrote the manifest. |
| `claim_level` | One of `unspecified`, `smoke`, `validation`, `production_template`, or `production`. |
| `claim_scope` | Human-readable boundary explaining what the artifact can and cannot support. |
| `config` | JSON-serializable run or benchmark configuration. |
| `outputs` | Mapping from stable output names to run-directory-relative file paths. |
| `hashes` | SHA-256 hashes for output files that existed when the manifest was written. |

`mhx artifact-manifest` also collects nested `manifest.json` claim levels in a
top-level `claim_levels` mapping. This lets CI and reviewers check whether a
figure directory mixes smoke, validation, production-template, or production
artifacts.

## `trajectory.npz` keys

The reduced-MHD v1 trajectory file contains:

| Key | Meaning |
| --- | --- |
| `schema` | Schema string, currently `mhx.reduced_mhd.trajectory.v1`. |
| `api_version` | Public API version string, currently `v1`. |
| `mhx_version` | Package version that wrote the file. |
| `time` | Saved times. |
| `psi` | Saved magnetic flux arrays with shape `(n_save, nx, ny)`. |
| `omega` | Saved vorticity arrays with shape `(n_save, nx, ny)`. |
| `config_json` | JSON-encoded effective run config. |
| `diagnostics_json` | JSON-encoded scalar diagnostics. |

Important scalar diagnostics include `equilibrium`, `equilibrium_parameters`,
`physics_plugin_modules`, `physics_plugin_entry_point_groups`, `physics_terms`,
`diagnostic_plugin_modules`, `diagnostic_plugin_entry_point_groups`,
`diagnostic_quantities`, `diagnostic_mode`, `fit_time_window`,
`fit_sample_count`, `gamma_fit`, and `final_magnetic_divergence_linf`. These
fields are saved so model assembly, extension modules, entry-point plugin
discovery, growth-rate plots, divergence checks, and comparisons can be audited.

## `diagnostics.json` keys

The default reduced-MHD run uses the diagnostic registry entries `energy`,
`mode_growth`, and `divergence_error`. The default scalar output includes:

| Key | Meaning |
| --- | --- |
| `diagnostic_quantities` | Diagnostic registry names evaluated for this run. |
| `diagnostic_plugin_modules` | Importable modules used to register custom diagnostics. |
| `diagnostic_plugin_entry_point_groups` | Installed entry-point groups used to register diagnostics. |
| `physics_plugin_modules` | Importable modules used to register custom RHS physics terms. |
| `physics_plugin_entry_point_groups` | Installed entry-point groups used to register RHS physics terms. |
| `initial_total_energy` | Initial reduced-MHD total energy. |
| `final_total_energy` | Final reduced-MHD total energy. |
| `final_magnetic_energy` | Final mean magnetic perturbation energy. |
| `final_kinetic_energy` | Final mean kinetic energy. |
| `diagnostic_mode` | Fourier mode used by the mode-growth diagnostic. |
| `fit_time_window` | Inclusive time window used for the exponential fit. |
| `fit_sample_count` | Number of saved samples used in the fit. |
| `initial_mode_amplitude` | Initial normalized magnetic-flux Fourier amplitude. |
| `final_mode_amplitude` | Final normalized magnetic-flux Fourier amplitude. |
| `gamma_fit` | Least-squares exponential growth/decay rate. |
| `final_magnetic_divergence_linf` | Final spectral $\|\nabla\cdot B_\perp\|_\infty$ check. |

## Figures

Figures are regenerated from saved data:

```bash
mhx figures outputs/smoke --gif
```

Expected files:

- `outputs/smoke/figures/energy_history.png`
- `outputs/smoke/figures/flux_final.png`
- `outputs/smoke/figures/mode_amplitude.png`
- `outputs/smoke/figures/flux_movie.gif`

## Reports

Run summaries are regenerated from saved outputs:

```bash
mhx report outputs/smoke
```

Expected files:

- `outputs/smoke/report.json`
- `outputs/smoke/report.md`

`report.json` also includes `additional_scalar_diagnostics`, a dictionary of
plugin-provided scalar metrics not part of the core reduced-MHD diagnostic
schema. `report.md` renders the same values in an `Additional scalar
diagnostics` table. Reports also include `diagnostic_metadata` when MHX can
reconstruct the selected diagnostic registry from `config_effective.json`; any
import or registry reconstruction failures are recorded in `warnings`.
Diagnostics may also provide optional figure hooks. `mhx figures` writes those
figures under `figures/diagnostics/`. `mhx report` dispatches the same hooks and
records a `diagnostic_figures` list in `report.json`, with each entry containing
a stable figure `key` and run-relative `path`.

## Artifact manifests

For reproducible figure/report diffs, write a recursive checksum manifest:

```bash
mhx artifact-manifest outputs/smoke
```

This writes `outputs/smoke/artifact_manifest.json` with schema
`mhx.artifacts.v1`, API version, file paths, byte sizes, and SHA-256 hashes.

## Validation-suite outputs

`mhx validate all --outdir outputs/validation_suite` executes the active FAST
validation cases, including double-Harris, turbulence, seed-QI, fitted latent
ODE, and restartable Rutherford execution chunks. Important closed-lane schemas
include:

### Orszag--Tang vortex

`mhx benchmark orszag-tang --outdir outputs/benchmarks/orszag_tang_vortex --movies`
writes:

| File | Schema / contents |
| --- | --- |
| `orszag_tang_vortex.npz` | `mhx.validation.orszag_tang_vortex.v1`; keys `time`, `psi`, `omega`, `current_density`, `magnetic_energy`, `kinetic_energy`, `total_energy`, `current_linf`, `vorticity_linf`, `current_high_k_fraction`, `vorticity_high_k_fraction`, and scalar gate summaries. |
| `diagnostics.json` | Scalar run controls, energy/drop diagnostics, high-$k$ growth, and divergence summary. |
| `validation.json` | `mhx.validation.orszag_tang_vortex.gates.v1`; finite-array, monotone-energy, dissipation, high-$k$, and divergence gates. |
| `figures/orszag_tang_summary.png` | Energy, high-$k$, final-current, and final-vorticity summary. |
| `figures/orszag_tang_*.gif` | Optional flux, current, and vorticity movies when `--movies` is set. |

### Decaying turbulence and forced turbulent reconnection

`mhx benchmark decaying-turbulence --outdir outputs/benchmarks/decaying_mhd_turbulence --movies`
and
`mhx benchmark forced-turbulent-reconnection --outdir outputs/benchmarks/forced_turbulent_reconnection --movies`
write:

| File | Schema / contents |
| --- | --- |
| `decaying_mhd_turbulence.npz` | `mhx.validation.decaying_mhd_turbulence.v1`; keys `time`, `psi`, `omega`, `current_density`, `magnetic_energy`, `kinetic_energy`, `total_energy`, `current_linf`, `vorticity_linf`, `current_high_k_fraction`, and `magnetic_divergence_linf`. |
| `forced_turbulent_reconnection.npz` | `mhx.validation.forced_turbulent_reconnection.v1`; same core keys plus `reconnection_proxy` and `reconnection_rate_proxy`. |
| `diagnostics.json` | Scalar controls, energy/current diagnostics, high-$k$ summaries, and reconnection-proxy statistics. |
| `validation.json` | Finite-array, energy, current, high-$k$, and reconnection-proxy gates. |
| `figures/*_summary.png` | Energy, current/reconnection, final current, and final flux summary. |
| `figures/*_flux.gif`, `figures/*_current.gif` | Optional fixed-scale movies when `--movies` is set. |

### Rutherford production execution

`mhx campaign rutherford-execute <run-dir>` writes:

| File | Schema / contents |
| --- | --- |
| `production_history.npz` | `mhx.campaign.rutherford_history.v1`; keys `step`, `time`, `reconnected_flux`, `rutherford_island_width`, `reconnection_rate_proxy`, `magnetic_energy`, `kinetic_energy`, `total_energy`, `dissipation_budget_residual`, `magnetic_divergence_linf`, `current_density_linf`, `current_sheet_length`, `current_sheet_thickness`, `current_sheet_aspect_ratio`, `x_point_count`, and `o_point_count`. |
| `checkpoints/state_step_*.npz` | `mhx.campaign.rutherford_state.v1`; keys `step`, `time`, `psi`, `omega`. |
| `checkpoints/checkpoint_index.json` | Ordered checkpoint metadata for resume and audit. |
| `resume_plan.json` | Next-step metadata for chunked continuation. |
| `diagnostics.json` | `mhx.campaign.rutherford_execution.v1`; start/end step, target step, run controls, `allow_production_claim`, `production_promotion_report_ready`, divergence and energy-growth summaries. |
| `validation.json` | `mhx.campaign.rutherford_execution.gates.v1`; finite-history, checkpoint, energy, divergence, and movie gates. |
| `figures/*.png` | Island-width, reconnected-flux, energy-budget, and `current_sheet_aspect_ratio.png` quick-look figures. |
| `figures/*.gif` | Optional fixed-scale flux and current movies. |
| `artifact_manifest.json` | Recursive hashes for chunk outputs, including the final `manifest.json`. The top-level `manifest.json` records this file as an output but does not rely on mutually recursive hashes. |
| `manifest.json` | Top-level claim metadata and output hashes. |

`mhx campaign rutherford-promotion-check <run-dir>` writes a separate promotion
bundle under `<run-dir>/promotion/` by default:

| File | Schema / contents |
| --- | --- |
| `promotion_readiness.json` | `mhx.campaign.rutherford_promotion.v1`; target-completion, convergence, seed-QI, movie, geometry, energy, and divergence promotion diagnostics. |
| `validation.json` | `mhx.campaign.rutherford_promotion.gates.v1`; pass/fail checks for every promotion gate. |
| `figures/promotion_matrix.png` | Reviewer-readable pass/fail matrix. |
| `artifact_manifest.json` | Recursive hashes for the promotion bundle. |
| `manifest.json` | `claim_level = production` only when all promotion checks pass; otherwise `claim_level = validation`. |

### Fitted neural ODE

`mhx neural-ode train --outdir outputs/neural_ode/latent_ode_fast` writes:

| File | Schema / contents |
| --- | --- |
| `latent_ode_model.json` | `mhx.neural_ode.latent_model.v1`; random-feature weights, ridge coefficient matrix, target names, and training controls. |
| `latent_ode_metrics.json` | `mhx.neural_ode.latent_metrics.v1`; train/validation/test MAE/RMSE and ratio to the best baseline. |
| `latent_ode_predictions.npz` | Prediction tensor, target tensor, times, seeds, and target names. |
| `failure_modes.json` | `mhx.neural_ode.failure_modes.v1`; seed-extrapolation and late-vs-early forecast-drift probes. |
| `dataset.npz` | Frozen deterministic FAST dataset when the train command generated it locally. |
| `splits.json` | Train/validation/test seed split metadata. |
| `baseline_metrics.json` | Persistence/linear baseline metrics used in comparisons. |
| `calibration.json` | Calibration and coverage diagnostics for the deterministic fit. |
| `experiment_spec.json` | Reproducible experiment controls. |
| `validation.json` | `mhx.neural_ode.training.gates.v1`; finite-model, finite-prediction, split, held-out forecast, and failure-mode reporting gates. |
| `manifest.json` | Top-level validation manifest. |

The suite also writes:

- `validation_suite.json`: schema `mhx.validation.suite.v1`, aggregate pass/fail
  status, `jax_enable_x64`, case list, per-case validation schemas, checks, and
  relative paths.
- `validation_suite.md`: reviewer-readable pass/fail summary table.
- `artifact_manifest.json`: recursive checksum manifest for every generated
  validation artifact.
- `manifest.json`: top-level manifest for the suite summary files.
- one subdirectory per case, for example `resistive_decay/`,
  `harris_delta_prime/`,
  `linear_tearing_eigenvalue/`,
  `linear_tearing_dispersion/`,
  `linear_tearing_layer/`,
  `linear_tearing_timedomain/`,
  `cosine_equilibrium_linearization/`,
  `periodic_current_sheet_eigenvalue/`,
  `periodic_current_sheet_timedomain/`,
  `periodic_current_sheet_nonlinear_bridge/`,
  `periodic_double_harris_nonlinear_growth/`,
  `periodic_double_harris_convergence/`,
  `nonlinear_energy_budget/`,
  `orszag_tang_vortex/`,
  `decaying_mhd_turbulence/`,
  `forced_turbulent_reconnection/`,
  `nonlinear_duration_audit/`,
  `seed_robust_qi/`,
  `seed_robust_qi_sweep/`,
  `neural_ode_reproducibility/`,
  `neural_ode_latent_fit/`,
  `rutherford_production_execution/`,
  `duration_policy/`,
  `diffusion_eigenvalue/`,
  `power_iteration/`, and `arnoldi/`.

Each validation-suite case includes a `claim_level` copied from its nested
manifest. Most cases are `validation`; the short linear-tearing smoke run is
explicitly `smoke`.

## Campaign-template outputs

`mhx campaign rutherford-template --outdir outputs/campaigns/rutherford_template`
writes:

- `campaign.json`: schema `mhx.campaign.rutherford_template.v1`, generated
  config, duration assessment, required production outputs, and claim boundary.
- `campaign_config.toml`: long-run TOML template with
  `physics.model = "reduced_mhd_nonlinear_tearing_campaign"`.
- `duration_assessment.json`: serialized duration guard for the chosen
  $\gamma$, e-fold count, and safety factor.
- `validation.json`: pass/fail checks for duration, non-FAST resolution,
  saved-frame count, and required runtime diagnostics.
- `manifest.json`: top-level claim metadata with
  `claim_level = "production_template"`.

## FAST Rutherford runner outputs

`mhx campaign rutherford-run-fast --outdir outputs/campaigns/rutherford_fast`
writes validation-grade campaign artifacts:

- `rutherford_fast_histories.npz`: schema
  `mhx.validation.rutherford_campaign_run.v1`, with `time`, `seed`,
  `reconnected_flux`, `rutherford_island_width`, `reconnection_rate_proxy`,
  `magnetic_energy`, `kinetic_energy`, `total_energy`,
  `magnetic_divergence_linf`, and `current_density_linf`.
- `diagnostics.json`: scalar pass/fail diagnostics and run metadata.
- `validation.json`: finite-value, duration, energy-growth, divergence, and
  claim-level checks.
- `campaign_template.json`: copy of the production-template requirements used
  to interpret the FAST run.
- `figures/rutherford_fast_histories.png`: quick-look histories.
- `manifest.json`: top-level claim metadata with
  `claim_level = "validation"` unless explicitly configured otherwise.

The schema intentionally mirrors the future long Rutherford campaign history
keys, but the FAST runner is not a production nonlinear result.

## Seed-robust QI outputs

`mhx benchmark seed-robust-qi --outdir outputs/benchmarks/seed_robust_qi`
writes:

- `diagnostics.json`: schema `mhx.validation.seed_robust_qi.v1`,
  including sample means, standard deviations, CVs, and gate results.
- `validation.json`: schema `mhx.validation.seed_robust_qi.gates.v1`,
  including per-metric pass/fail checks.
- `ensemble.npz`: seed list and full metric samples for `gamma_fit`,
  final energies, and magnetic-divergence error.
- `figures/qi_summary.png`: compact metric stability summary when plotting
  dependencies are available.
- `manifest.json`: top-level claim metadata with `claim_level = "validation"`.

## Exact-decay validation outputs

`mhx benchmark decay --outdir outputs/benchmarks/resistive_decay` writes:

- `diagnostics.json`: scalar decay-rate and error diagnostics with schema
  `mhx.validation.resistive_decay.v1`.
- `validation.json`: pass/fail physics gates and thresholds.
- `decay_history.npz`: time, numerical/exact amplitude, numerical/exact energy,
  and relative-error arrays.
- `figures/decay_amplitude.png`, `figures/decay_energy.png`, and
  `figures/decay_relative_error.png`.

## Reconnection scaling validation outputs

`mhx benchmark scaling --outdir outputs/benchmarks/reconnection_scaling` writes:

- `diagnostics.json`: fitted and expected log-log slopes with schema
  `mhx.validation.reconnection_scaling.v1`.
- `validation.json`: pass/fail slope gates and tolerances.
- `scaling_history.npz`: Lundquist samples and analytic scaling arrays.
- `figures/fkr_scaling.png`, `figures/plasmoid_scaling.png`, and
  `figures/ideal_tearing_scaling.png`.

## FKR-window validation outputs

`mhx benchmark fkr-window --outdir outputs/benchmarks/fkr_window` writes:

- `diagnostics.json`: fixed-$S_a$ FKR constant-$\psi$ regime-window diagnostics
  with schema `mhx.validation.fkr_window.v1`.
- `validation.json`: pass/fail gates for positive $\Delta'a$, thin inner layer,
  and $\Delta'\delta$.
- `fkr_window.npz`: sampled `ka`, `gamma_tau_a`, `inner_width_a`,
  `delta_prime_a`, and `constant_psi_product` arrays.
- `figures/fkr_constant_psi_window.png`: publication-style regime-window plot.

## FKR growth-rate validation outputs

`mhx benchmark fkr-growth --outdir outputs/benchmarks/fkr_growth_rate` writes:

- `diagnostics.json`: asymptotic FKR growth-rate diagnostics with schema
  `mhx.validation.fkr_growth_rate.v1`.
- `validation.json`: pass/fail gates for the $S_a^{-3/5}$ slope,
  $(\Delta'a)^{4/5}$ response, numerical-$\Delta'$ propagation error, and the
  constant-$\psi$ window.
- `fkr_growth_rate.npz`: sampled `lundquist`, `gamma_vs_lundquist`, `ka`,
  `numerical_delta_prime_a`, `gamma_vs_delta_prime`, and
  `gamma_relative_error` arrays.
- `figures/fkr_growth_rate.png`: growth-rate scaling and error-gate plot.

## Harris Delta-prime validation outputs

`mhx benchmark harris-delta-prime --outdir outputs/benchmarks/harris_delta_prime`
writes:

- `diagnostics.json`: numerical Harris outer-region matching diagnostics with
  schema `mhx.validation.harris_delta_prime.v1`.
- `validation.json`: pass/fail gates for finite positive $\Delta'a$,
  monotonicity, and relative error against the analytic Harris formula.
- `harris_delta_prime.npz`: sampled `ka`, numerical and analytic
  `delta_prime_a`, and relative-error arrays.
- `figures/harris_delta_prime.png`: numerical-vs-analytic matching plot and
  relative-error gate.

## Direct Harris-sheet tearing eigenvalue outputs

`mhx benchmark linear-tearing-eigenvalue --outdir outputs/benchmarks/linear_tearing_eigenvalue`
writes:

- `diagnostics.json`: direct 1D Harris-sheet finite-difference tearing
  eigenvalue diagnostics with schema `mhx.validation.linear_tearing_eigenvalue.v1`.
- `validation.json`: pass/fail gates for selected growth-rate error,
  extrapolated growth-rate error, real eigenvalue, eigen-residual, grid
  convergence, and tearing parity.
- `linear_tearing_eigenvalue.npz`: grid counts, `dx`, finite-grid growth rates,
  fitted second-order growth rates, extrapolated/reference growth rates,
  stable-control wavenumber/max-real-part/residual, selected spectrum
  real/imaginary parts, normalized flux eigenfunction, and normalized imaginary
  stream-function eigenfunction.
- `figures/linear_tearing_eigenvalue.png`: grid extrapolation, selected
  spectrum, and tearing-parity plot.

## Finite-domain tearing dispersion outputs

`mhx benchmark linear-tearing-dispersion --outdir outputs/benchmarks/linear_tearing_dispersion`
writes:

- `diagnostics.json`: finite-domain Harris-sheet tearing dispersion diagnostics
  with schema `mhx.validation.linear_tearing_dispersion.v1`.
- `validation.json`: pass/fail gates for finite eigenvalues, positive growth in
  the sampled unstable band, no positive growth in stable controls, residuals,
  and the $ka=0.5$ reference point.
- `linear_tearing_dispersion.npz`: `wavenumber`, `growth_rate`,
  `eigenvalue_imag`, `residual_norm`, reference-point metadata, and boolean
  masks for unstable samples and stable controls.
- `figures/linear_tearing_dispersion.png`: growth branch, oscillatory stable
  controls, and residual gate.

## Harris tearing eigenfunction-layer outputs

`mhx benchmark linear-tearing-layer --outdir outputs/benchmarks/linear_tearing_layer`
writes:

- `diagnostics.json`: FAST Harris eigenfunction-shape diagnostics with schema
  `mhx.validation.linear_tearing_layer.v1`.
- `validation.json`: pass/fail gates for positive growth, monotonic flow-layer
  narrowing with $S$, outer-flux width stability, broad fitted-slope ranges,
  and dense eigen-residuals.
- `linear_tearing_layer.npz`: sampled `lundquist`, `growth_rate`,
  `stream_half_width`, `current_half_width`, `flux_half_width`,
  `residual_norm`, fitted slopes, flux-width spread, and normalized reference
  eigenfunction profiles.
- `figures/linear_tearing_layer.png`: growth trend, layer-width trend,
  reference eigenfunction localization, and residual gate.

## Harris tearing time-domain replay outputs

`mhx benchmark linear-tearing-timedomain --outdir outputs/benchmarks/linear_tearing_timedomain`
writes:

- `diagnostics.json`: linear time-domain replay diagnostics with schema
  `mhx.validation.linear_tearing_timedomain.v1`.
- `validation.json`: pass/fail gates for fitted growth-rate error, RK4
  amplitude error against $\exp(\gamma t)$, final eigenmode alignment, and the
  selected dense-eigenpair residual.
- `linear_tearing_timedomain.npz`: `time`, `amplitude`, `exact_amplitude`,
  `relative_amplitude_error`, `fitted_growth_rate`, `expected_growth_rate`,
  `relative_growth_error`, `final_mode_alignment`, selected eigenvalue parts,
  and selected residual.
- `figures/linear_tearing_timedomain.png`: semilog amplitude replay,
  fitted-growth plot, and relative-amplitude-error gate.

## Linearized-RHS validation outputs

`mhx benchmark linearized-rhs --outdir outputs/benchmarks/linearized_rhs`
writes:

- `diagnostics.json`: JVP/finite-difference linearized-RHS consistency metrics
  with schema `mhx.validation.linearized_rhs.v1`.
- `validation.json`: pass/fail gates for relative JVP consistency errors.
- `linearized_rhs.npz`: saved JVP and finite-difference `psi`/`omega` arrays.
- `figures/linearized_rhs_errors.png`: relative-error plot with the configured
  gate.

## Reduced-MHD linear eigenmode outputs

`mhx benchmark reduced-mhd-eigenmode --outdir outputs/benchmarks/reduced_mhd_eigenmode`
writes:

- `diagnostics.json`: zero-state linear eigenmode diagnostics with schema
  `mhx.validation.reduced_mhd_linear_eigenmode.v1`.
- `validation.json`: pass/fail gates for resistive $\psi$ and viscous $\omega$
  Fourier diffusion eigenvalues and residuals.
- `reduced_mhd_linear_eigenmode.npz`: saved eigenfunctions, flattened operator
  actions, expected eigenvalues, and measured Rayleigh quotients.
- `figures/reduced_mhd_linear_eigenmode_errors.png`: eigenvalue/residual error
  plot.

## Cosine-equilibrium linearization outputs

`mhx benchmark cosine-equilibrium-linearization --outdir outputs/benchmarks/cosine_equilibrium_linearization`
writes:

- `diagnostics.json`: analytic nonzero-equilibrium linearized-RHS diagnostics
  with schema `mhx.validation.cosine_equilibrium_linearization.v1`.
- `validation.json`: pass/fail gates for flow-to-flux and magnetic-tension
  coupling errors around $\psi_0=A\cos y$.
- `cosine_equilibrium_linearization.npz`: numerical and analytic JVP arrays for
  the selected flow and flux perturbations.
- `figures/cosine_equilibrium_linearization_errors.png`: relative-error plot.

## Periodic current-sheet eigenvalue outputs

`mhx benchmark current-sheet-eigenvalue --outdir outputs/benchmarks/periodic_current_sheet_eigenvalue`
writes:

- `diagnostics.json`: tiny dense-spectrum diagnostics with schema
  `mhx.validation.periodic_current_sheet_eigenvalue.v1`.
- `validation.json`: pass/fail gates for gauge residuals, non-gauge damping,
  spurious positive growth, and selected dense-eigenpair residual.
- `periodic_current_sheet_eigenvalue.npz`: dense operator matrix, eigenvalue
  real/imaginary parts, selected eigenvector real/imaginary parts, residual,
  and leading-real-part diagnostics.
- `figures/periodic_current_sheet_spectrum.png`: complex spectrum and
  selected-eigenpair residual plot.

## Periodic current-sheet time-domain outputs

`mhx benchmark current-sheet-timedomain --outdir outputs/benchmarks/periodic_current_sheet_timedomain`
writes:

- `diagnostics.json`: RK4 eigenmode replay diagnostics with schema
  `mhx.validation.periodic_current_sheet_timedomain.v1`.
- `validation.json`: pass/fail gates for spectrum consistency, real decaying
  mode selection, dense-eigenpair residual, full-state replay error, and fitted
  decay-rate error.
- `periodic_current_sheet_timedomain.npz`: saved `time`, numerical and exact
  amplitudes, relative state errors, selected/fitted decay rates, residuals,
  and initial/final `psi` and `omega` fields.
- `figures/periodic_current_sheet_timedomain.png`: semilog amplitude replay and
  relative full-state error against the configured gate.

## Nonlinear current-sheet bridge outputs

`mhx benchmark current-sheet-nonlinear-bridge --outdir outputs/benchmarks/periodic_current_sheet_nonlinear_bridge`
writes:

- `diagnostics.json`: nonlinear trajectory-map differentiability diagnostics
  with schema `mhx.validation.periodic_current_sheet_nonlinear_bridge.v1`.
- `validation.json`: pass/fail gates for finite positive errors, monotonic
  centered-difference convergence, second-order slope, finest-error tolerance,
  and nonzero tangent norm.
- `periodic_current_sheet_nonlinear_bridge.npz`: saved `epsilon`,
  `relative_error`, fitted `convergence_order`, tangent norm, perturbation
  fields, and final tangent fields.
- `figures/periodic_current_sheet_nonlinear_bridge.png`: log-log convergence
  plot against an $O(\epsilon^2)$ guide and the configured slope/error gates.

## Periodic double-Harris nonlinear-growth outputs

`mhx benchmark double-harris-growth --outdir outputs/benchmarks/periodic_double_harris_nonlinear_growth`
writes:

- `diagnostics.json`: nonlinear growth diagnostics with schema
  `mhx.validation.periodic_double_harris_nonlinear_growth.v1`.
- `validation.json`: pass/fail gates for a finite dense spectrum/history,
  positive unstable eigenvalue, small eigen-residual, nonlinear perturbation
  growth factor, positive fitted growth, and fitted-vs-eigenvalue error.
- `periodic_double_harris_nonlinear_growth.npz`: saved dense eigenvalue
  spectrum, selected eigenvector/eigenvalue, time, perturbation norm, expected
  frozen-linear norm, fitted growth rate, growth factor, and base/perturbed
  final fields.
- `figures/periodic_double_harris_nonlinear_growth.png`: semilog nonlinear
  growth history with fitted and frozen-linear references plus initial/final
  flux panels.

## Periodic double-Harris seeded long-run outputs

`mhx benchmark double-harris-long-run --outdir outputs/benchmarks/periodic_double_harris_seeded_long_run`
writes:

- `diagnostics.json`: scalable seeded nonlinear replay diagnostics with schema
  `mhx.validation.periodic_double_harris_seeded_long_run.v1`.
- `validation.json`: pass/fail gates for finite histories, full-duration
  completion, sample count, early growth, visible amplification, and
  dissipative total-energy behavior.
- `periodic_double_harris_seeded_long_run.npz`: saved time, normalized
  perturbed-minus-base norm, magnetic/kinetic/total energy, peak current
  density, fitted early growth rate, base/perturbed trajectories, and initial
  states.
- `figures/periodic_double_harris_seeded_long_run.png`: early-growth,
  energy, current-density, flux, and perturbed-minus-base morphology panels.
- Optional `figures/periodic_double_harris_flux.gif` and
  `figures/periodic_double_harris_current.gif` when `--movies` is supplied.

## Periodic double-Harris convergence outputs

`mhx benchmark double-harris-convergence --outdir outputs/benchmarks/periodic_double_harris_convergence`
writes:

- `diagnostics.json`: convergence scaffold diagnostics with schema
  `mhx.validation.periodic_double_harris_convergence.v1`, including
  `resolutions`, `dt_values`, thresholds, per-case metrics, and spread
  statistics.
- `validation.json`: pass/fail gates for finite case metrics, successful
  subcases, positive early growth, dissipative energy, and bounded
  resolution/time-step spread.
- `periodic_double_harris_convergence.npz`: stable keys `schema`, `case_kind`,
  `resolution`, `dt`, `samples`, `fitted_early_growth_rate`,
  `early_growth_factor`, `max_growth_factor`, `relative_energy_increase`,
  `max_current_density_linf`, and `max_kinetic_energy`.
- `figures/periodic_double_harris_convergence.png`: resolution/time-step
  `gamma_fit`, nonlinear amplification, energy, and peak-current panels.

## Nonlinear energy-budget outputs

`mhx benchmark nonlinear-energy-budget --outdir outputs/benchmarks/nonlinear_energy_budget`
writes:

- `diagnostics.json`: nonlinear reduced-MHD budget diagnostics with schema
  `mhx.validation.nonlinear_energy_budget.v1`.
- `validation.json`: pass/fail gates for finite arrays, nontrivial nonlinear
  RHS activity, nonincreasing total energy, integrated budget residual, net
  dissipation, and positive dissipation.
- `nonlinear_energy_budget.npz`: saved `time`, magnetic/kinetic/total energy,
  resistive and viscous dissipation channels, cumulative dissipation,
  `budget_residual`, `relative_budget_residual`, nonlinear-RHS ratio, initial
  fields, and final fields.
- `figures/nonlinear_energy_budget.png`: energy-vs-integrated-dissipation,
  relative budget residual, dissipation channels, and initial/final flux
  contours.

## Nonlinear duration-audit outputs

`mhx benchmark nonlinear-duration-audit --outdir outputs/benchmarks/nonlinear_duration_audit`
writes:

- `diagnostics.json`: runtime-window audit diagnostics with schema
  `mhx.validation.nonlinear_duration_audit.v1`.
- `validation.json`: pass/fail gates confirming that current FAST nonlinear
  runs are explicitly flagged as too short for nonlinear island/plasmoid
  claims and that production target windows are recorded.
- `nonlinear_duration_audit.npz`: saved `current_case_names`,
  `current_end_times`, `target_names`, `target_end_times`,
  `plasmoid_lundquist`, `plasmoid_efold_times`, `harris_growth_rate`, and
  `requested_linear_efolds`.
- `figures/nonlinear_duration_audit.png`: log-time comparison of current FAST
  run durations, target nonlinear windows, and plasmoid linear e-fold estimates.

## Duration-policy outputs

`mhx benchmark duration-policy --outdir outputs/benchmarks/duration_policy`
writes:

- `duration_policy.json`: schema `mhx.duration_policy.v1`, current and future
  duration assessments, growth rates, required e-folds, required final times,
  observed e-folds, `sufficient_for_intended_scope`,
  `sufficient_for_production_claim`, `sufficient_for_nonlinear_claim`, and
  claim-boundary actions.
- `duration_policy.md`: reviewer-readable table generated from the same JSON.
- `validation.json`: pass/fail gates confirming that current short runs are
  validation-only and future production templates are long enough for their
  declared e-fold targets.
- `manifest.json`: hashes for all duration-policy artifacts.

## Diffusion eigenvalue validation outputs

`mhx benchmark diffusion-eigenvalue --outdir outputs/benchmarks/diffusion_eigenvalue`
writes:

- `diagnostics.json`: Rayleigh-quotient and eigen-residual diagnostics with
  schema `mhx.validation.diffusion_eigenvalue.v1`.
- `validation.json`: pass/fail gates for eigenvalue and residual errors.
- `diffusion_eigenvalue.npz`: saved eigenfunction, operator action, expected
  eigenvalue, and measured Rayleigh quotient.
- `figures/diffusion_eigenvalue_errors.png`: eigenvalue/residual error plot.

## Power-iteration validation outputs

`mhx benchmark power-iteration --outdir outputs/benchmarks/power_iteration`
writes:

- `diagnostics.json`: dominant-eigenpair diagnostics with schema
  `mhx.validation.power_iteration.v1`.
- `validation.json`: pass/fail gates for dominant eigenvalue and residual
  errors.
- `power_iteration_history.npz`: saved iteration index, Rayleigh-quotient
  history, residual history, and expected dominant eigenvalue.
- `figures/power_iteration_history.png`: convergence history plot.

## Arnoldi validation outputs

`mhx benchmark arnoldi --outdir outputs/benchmarks/arnoldi` writes:

- `diagnostics.json`: Krylov Ritz-spectrum diagnostics with schema
  `mhx.validation.arnoldi.v1`.
- `validation.json`: pass/fail gates for fixture Ritz-value, imaginary-part,
  and residual-estimate errors.
- `arnoldi_spectrum.npz`: expected eigenvalues, measured Ritz values, Arnoldi
  residual estimates, and the projected Hessenberg matrix.
- `figures/arnoldi_ritz_values.png`: complex-plane Ritz spectrum and residual
  estimate plot.

## Timing benchmark outputs

`mhx benchmark timing --outdir outputs/benchmarks/timing` writes:

- `timing.json`: schema `mhx.benchmark.timing.v1`, raw repeat durations,
  median/min/max summaries, Python `tracemalloc` peaks, case details, and
  environment metadata.
- `timing.md`: reviewer-readable Markdown timing table.
- `figures/timing_summary.png`: wall-clock and Python-allocation summary plot.
- `manifest.json`: schema `mhx.manifest.v1`, output paths, and SHA-256 hashes.

Absolute timings are machine-dependent. The schema is meant for artifact
comparison across commits on the same runner class, not as a universal
performance guarantee.

## Neural-ODE reproducibility outputs

`mhx neural-ode dataset --outdir outputs/neural_ode/seed_qi_fast` writes a
deterministic no-training bundle for future neural-ODE work:

| File | Schema | Meaning |
| --- | --- | --- |
| `dataset.npz` | `mhx.neural_ode.dataset.v1` | Seed-QI trajectory tensor, target tensor, times, seeds, and feature names. |
| `splits.json` | `mhx.neural_ode.splits.v1` | Disjoint train/validation/test seed IDs and split checks. |
| `baseline_metrics.json` | `mhx.neural_ode.baselines.v1` | Persistence, linear-prefix, and train-mean MAE/RMSE scores. |
| `calibration.json` | `mhx.neural_ode.calibration.v1` | Empirical 1-sigma and 2-sigma residual coverage. |
| `experiment_spec.json` | `mhx.neural_ode.experiment_spec.v1` | Reviewer-facing experiment contract and future neural-ODE requirements. |
| `validation.json` | `mhx.neural_ode.reproducibility.gates.v1` | Gate summary tying seed-QI validation, splits, baselines, and calibration together. |
| `manifest.json` | `mhx.manifest.v1` | Hashes and `claim_level = "validation"`. |

`dataset.npz` keys are `schema`, `seeds`, `times`, `features`, `targets`,
`feature_names`, `target_names`, `diagnostics_json`, and
`source_diagnostics_json`. The v1 array convention is
`features[n_seed, n_time, n_feature]` and
`targets[n_seed, n_time, n_target]`.

## Benchmark catalog outputs

`mhx benchmark catalog --outdir outputs/benchmarks/catalog` writes:

- `benchmark_catalog.json`: schema `mhx.benchmark_catalog.v1`, active FAST
  validation entries, commands, output schemas, expected files, and literature
  anchors.
- `benchmark_catalog.md`: reviewer-readable table generated from the same
  catalog entries.
- `manifest.json`: hashes for the generated catalog files.
