# Output schema

Every active MHX run writes a JSON manifest plus schema-versioned data files.

## Run directory

`mhx run examples/linear_tearing.toml --outdir outputs/smoke` writes:

- `config_effective.json`: JSON serialization of the effective config.
- `diagnostics.json`: scalar JSON diagnostics.
- `trajectory.npz`: compressed trajectory arrays using schema
  `mhx.reduced_mhd.trajectory.v1`.
- `manifest.json`: file list and SHA-256 hashes.

## `trajectory.npz` keys

The reduced-MHD v1 trajectory file contains:

| Key | Meaning |
| --- | --- |
| `schema` | Schema string, currently `mhx.reduced_mhd.trajectory.v1`. |
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

## Artifact manifests

For reproducible figure/report diffs, write a recursive checksum manifest:

```bash
mhx artifact-manifest outputs/smoke
```

This writes `outputs/smoke/artifact_manifest.json` with schema
`mhx.artifacts.v1`, file paths, byte sizes, and SHA-256 hashes.

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
