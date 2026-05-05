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
`physics_terms`, `diagnostic_mode`, `fit_time_window`, `fit_sample_count`, and
`gamma_fit`. These fields are saved so model assembly, growth-rate plots, and
comparisons can be audited.

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

## Artifact manifests

For reproducible figure/report diffs, write a recursive checksum manifest:

```bash
mhx artifact-manifest outputs/smoke
```

This writes `outputs/smoke/artifact_manifest.json` with schema
`mhx.artifacts.v1`, file paths, byte sizes, and SHA-256 hashes.
