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

Every v1 manifest contains:

| Key | Meaning |
| --- | --- |
| `claim_level` | One of `unspecified`, `smoke`, `validation`, `production_template`, or `production`. |
| `claim_scope` | Human-readable boundary explaining what the artifact can and cannot support. |

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
validation gates and writes:

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
  `nonlinear_energy_budget/`,
  `nonlinear_duration_audit/`,
  `duration_policy/`, and `arnoldi/`.

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

## Benchmark catalog outputs

`mhx benchmark catalog --outdir outputs/benchmarks/catalog` writes:

- `benchmark_catalog.json`: schema `mhx.benchmark_catalog.v1`, active FAST
  validation entries, commands, output schemas, expected files, and literature
  anchors.
- `benchmark_catalog.md`: reviewer-readable table generated from the same
  catalog entries.
- `manifest.json`: hashes for the generated catalog files.
