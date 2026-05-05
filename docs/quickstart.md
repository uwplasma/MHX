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
```

The smoke run validates the JAX spectral derivative path on a periodic
Cartesian mesh. The exact-decay benchmark adds a physics gate for
$\psi_k(t)=\psi_k(0)\exp(-\eta |k|^2t)$. These are deliberately small and
deterministic; they are not yet the full tearing benchmark.
