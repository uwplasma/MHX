# Examples

Start in `examples/gallery`. Those seven scripts show the public Python API. Each
script keeps its settings at the top and writes fields under `outputs/gallery`.

| Script | What it shows |
| --- | --- |
| `gallery/01_reconnection.py` | Double-Harris reconnection |
| `gallery/02_tearing_mode.py` | Cosine current sheet |
| `gallery/03_implicit_step.py` | SOLVAX backward Euler |
| `gallery/04_cpu_parallel.py` | Four-device CPU sharding |
| `gallery/05_gpu_parallel.py` | Multi-GPU sharding |
| `gallery/06_strong_scaling.py` | Fixed-ensemble strong scaling |
| `gallery/07_multi_process.py` | Multi-process ensemble |

Run the first script from the repository root:

```bash
python examples/gallery/01_reconnection.py
```

Each gallery script uses this order:

1. Choose an output path.
2. Create `mhx.Simulation`.
3. Run it.
4. Print and plot the result.
5. Save the fields and metadata.

## Recorded config runs

The TOML files support command-line runs with an explicit config record:

```bash
mhx run examples/linear_tearing.toml \
  --outdir outputs/examples/linear_tearing
```

Use the TOML path for campaign automation. Use the gallery path when you learn
or change the Python API.

## Validation drivers

Files named `publication_*.py` and `make_*.py` regenerate review artifacts.
They call benchmark and campaign modules, so they contain more controls than
the gallery. The test suite runs their small settings.

`examples/tools/verify_paper_artifacts.py` checks saved paper data.
`examples/plugin_template/` is a package template for an external physics or
diagnostics plugin.
