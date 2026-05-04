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

## Figures

Figures are regenerated from saved data:

```bash
mhx figures outputs/smoke
```

Expected files:

- `outputs/smoke/figures/energy_history.png`
- `outputs/smoke/figures/flux_final.png`

