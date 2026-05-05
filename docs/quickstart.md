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

Inspect the registered model pieces used by TOML configs:

```bash
mhx physics equilibria
mhx physics list
mhx diagnostics list
```

The command writes:

- `config_effective.json`
- `diagnostics.json`
- `manifest.json`
- `trajectory.npz`

Regenerate figures from the saved run:

```bash
mhx figures outputs/smoke --gif
```

Expected figures:

- `outputs/smoke/figures/energy_history.png`
- `outputs/smoke/figures/flux_final.png`
- `outputs/smoke/figures/mode_amplitude.png`
- `outputs/smoke/figures/flux_movie.gif`

Create a reviewer-readable summary:

```bash
mhx report outputs/smoke
mhx artifact-manifest outputs/smoke
```

Run the same workflow through the benchmark command group:

```bash
mhx benchmark run --config examples/linear_tearing.toml --outdir outputs/benchmarks/linear_tearing_fast --gif
mhx benchmark validate outputs/benchmarks/linear_tearing_fast
mhx benchmark decay --outdir outputs/benchmarks/resistive_decay
mhx benchmark scaling --outdir outputs/benchmarks/reconnection_scaling
mhx benchmark fkr-window --outdir outputs/benchmarks/fkr_window
mhx benchmark linearized-rhs --outdir outputs/benchmarks/linearized_rhs
mhx benchmark diffusion-eigenvalue --outdir outputs/benchmarks/diffusion_eigenvalue
mhx benchmark timing --outdir outputs/benchmarks/timing --repeats 3 --warmups 1
```

The smoke run validates the JAX spectral derivative path on a periodic
Cartesian mesh. The exact-decay benchmark adds a physics gate for
$\psi_k(t)=\psi_k(0)\exp(-\eta |k|^2t)$. These are deliberately small and
deterministic; they are not yet the full tearing benchmark. The timing
benchmark records wall-clock and Python-allocation summaries for comparing
changes on the same machine or CI runner.

Try a local extension module without editing MHX source:

```bash
mhx run examples/linear_tearing_plugin_demo.toml --outdir outputs/plugin_demo
mhx figures outputs/plugin_demo --gif
mhx report outputs/plugin_demo
mhx physics list-with-plugins --plugin-module examples.local_extension_plugin
mhx diagnostics list-with-plugins --plugin-module examples.local_extension_plugin
mhx physics lint example_flux_drive --plugin-module examples.local_extension_plugin
mhx diagnostics lint final_flux_l2 --plugin-module examples.local_extension_plugin
```

This demo registers a toy flux-drive physics term and a `final_flux_l2`
diagnostic from `examples/local_extension_plugin.py`. Installed plugin packages
use the same registries through `--entry-point-group mhx.physics` and
`--entry-point-group mhx.diagnostics`. The installable package skeleton in
`examples/plugin_template/` shows the recommended external-repository layout.
