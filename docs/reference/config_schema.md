# TOML configuration

The TOML path assembles a model from explicit, auditable pieces: mesh and
time controls, an equilibrium, base coefficients, optional physics terms, and
diagnostics. Saved run directories include `config_effective.json`,
`diagnostics.json`, and `manifest.json`, so every figure traces back to the
exact assembled model.

Write a starter file with `mhx init <path>`, then run it with
`mhx run <path> --outdir <dir>`. The active model is
`reduced_mhd_linear_tearing`. It evolves the
[reduced-MHD equations](../physics/reduced_mhd.md) with optional plugin
sources $(S_\psi, S_\omega)$.

## The physics table

```toml
[physics]
model = "reduced_mhd_linear_tearing"
equilibrium = "cosine_tearing"
resistivity = 0.001
viscosity = 0.001
plugin_modules = []
plugin_entry_point_groups = []
rhs_terms = ["hyper_resistivity", "vorticity_drag"]

[physics.equilibrium_parameters]
perturbation_amplitude = 0.001

[physics.term_parameters.hyper_resistivity]
eta4 = 1e-5
nu4 = 1e-5

[physics.term_parameters.vorticity_drag]
rate = 0.01
```

Use `mhx physics equilibria` to list equilibria and `mhx physics list` to
list terms. `mhx physics lint <name>` validates a registered term against
the active API. The [model page](../physics/reduced_mhd.md#built-in-equilibria)
documents the equilibrium formulas and the
[term table](../physics/reduced_mhd.md#optional-physics-terms).

`plugin_modules` accepts importable Python modules that expose
`register_physics(registry)` or `register_diagnostics(registry)`. The demo
`examples/linear_tearing_plugin_demo.toml` assembles a local physics term and
a local diagnostic entirely from TOML.

`plugin_entry_point_groups` accepts installed package entry-point groups,
usually `mhx.physics` and `mhx.diagnostics`. The run manifest records the
discovery groups beside the selected names, so third-party plugin runs stay
auditable.

## The diagnostics table

```toml
[diagnostics]
quantities = ["energy", "mode_growth", "divergence_error"]
plugin_modules = []
plugin_entry_point_groups = []
mode = [1, 1]
fit_time_window = [0.02, 0.1]
```

`mode_growth` tracks the Fourier amplitude
$A_{m,n}(t) = |\hat\psi_{m,n}(t)|$ of the configured mode and fits
$A(t) \approx A_0 e^{\gamma t}$ by least squares on $\log A$ inside
`fit_time_window`. MHX records `fit_time_window` and `fit_sample_count` in
`diagnostics.json`, so growth-rate comparisons stay auditable. Do not read
`gamma_fit` as a tearing growth rate unless the equilibrium, fit window, and
parameter regime match a [validated gate](../validation/linear_tearing.md).

Use `mhx diagnostics list` for names and output keys, and
[add diagnostics](../how_to/add_diagnostics.md) for the registry.

## Reproducible command

```bash
mhx run examples/linear_tearing_hyper.toml --outdir outputs/linear_tearing_hyper
mhx figures outputs/linear_tearing_hyper --gif
mhx report outputs/linear_tearing_hyper
```

Expected model-audit fields in `outputs/linear_tearing_hyper/diagnostics.json`
include `equilibrium`, `equilibrium_parameters`, and `physics_terms`. The
[output schema](output_schema.md) documents every file a run writes.
