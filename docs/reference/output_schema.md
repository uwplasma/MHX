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

## Benchmark outputs

Each `mhx benchmark` command writes its own folder with `diagnostics.json`,
`validation.json` (the gate checks and pass/fail status), `manifest.json`, and a
`figures/` folder. The [CLI reference](cli.md) lists the files each command
writes.
