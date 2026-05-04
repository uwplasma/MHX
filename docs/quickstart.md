# Quickstart

Install the active package in editable mode:

```bash
python -m pip install -e ".[dev,docs]"
```

Check the installed version:

```bash
mhx version
```

Run the first deterministic smoke workflow:

```bash
mhx run examples/linear_tearing.toml --outdir outputs/smoke
```

The command writes:

- `config_effective.json`
- `diagnostics.json`
- `manifest.json`

The current smoke run validates the JAX spectral derivative path on a periodic
Cartesian mesh. It is deliberately small and deterministic; it is not yet the
full tearing benchmark.

