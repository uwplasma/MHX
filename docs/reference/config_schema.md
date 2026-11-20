# TOML configuration

The TOML path assembles a model from explicit, auditable pieces: mesh and
time controls, an equilibrium, base coefficients, optional physics terms, and
diagnostics. Saved run directories include `config_effective.json`,
`diagnostics.json`, and `manifest.json`, so every figure traces back to the
exact assembled model.

Write a starter file with `mhx init <path>`, then run it with
`mhx run <path> --outdir <dir>`. The same file drives the Python API:
`mhx.Simulation.from_config(mhx.load_config(path)).run()` does what
`mhx run` does for the `reduced_mhd`, `mhd3d` and `compressible` models.

## Models

`physics.model` selects the equations.

| Model | Equations | Mesh | Integrators |
| --- | --- | --- | --- |
| `reduced_mhd_linear_tearing` | [reduced MHD](../physics/reduced_mhd.md), lightweight smoke workflow with diagnostics plugins | 2 entries | `rk4`, `backward_euler` |
| `reduced_mhd` | reduced MHD through `mhx.Simulation` | 2 entries | `rk4`, `backward_euler` |
| `mhd3d` | [3D incompressible MHD](../physics/mhd3d.md) | 3 entries | `if_rk3` (`rk4` selects it), `etdrk4` |
| `compressible` | [subsonic compressible MHD](../physics/compressible.md) | 3 entries | `if_rk3` (`rk4` selects it), `etdrk4` |

The mesh has two or three entries to match the model. `mesh.lower` and
`mesh.upper` default to `0` and `2*pi` on every axis, and `Simulation` runs
need `mesh.periodic` to be all true. `time.t0` must be `0` for
`Simulation` runs, and `time.t1 - time.t0` must be a whole number of `time.dt`
steps.

```toml
[mesh]
shape = [64, 64, 64]

[time]
t1 = 2.0
dt = 2.0e-3
save_every = 100

[physics]
model = "mhd3d"
equilibrium = "orszag_tang_3d"
resistivity = 5.0e-3
viscosity = 5.0e-3
guide_field = [0.0, 0.0, 0.0]

[physics.equilibrium_parameters]
beta = 0.8

[numerics]
device_count = 2
```

The `guide_field`, `dissipation_order`, `sound_speed` and `bulk_viscosity`
keys apply to the 3D models (`dissipation_order` is the power of the Laplacian
in the dissipation, and `sound_speed` and `bulk_viscosity` apply to
`compressible`). `numerics.device_count` shards one run across that many
devices. A run
through `Simulation` prints its setup and timings, then writes
`config_effective.json`, `diagnostics.json`, `trajectory.npz`, `summary.png`
and `manifest.json`. The `[diagnostics]` table below applies to the
`reduced_mhd_linear_tearing` model only; `mhx figures` and `mhx report` read
its output format.

## Shipped examples

| File | Model | What it runs |
| --- | --- | --- |
| `examples/linear_tearing.toml` and variants | `reduced_mhd_linear_tearing` | smoke runs with hyper terms and plugins |
| `examples/toml/double_harris.toml` | `reduced_mhd` | periodic double-Harris sheet seeded with one mode |
| `examples/toml/double_harris_hyper.toml` | `reduced_mhd` | the same sheet with hyper-resistivity and hyper-viscosity |
| `examples/toml/double_harris_sharded.toml` | `reduced_mhd` | one run split across two devices |
| `examples/toml/orszag_tang_2d.toml` | `reduced_mhd` | the 2D Orszag--Tang vortex |
| `examples/toml/orszag_tang_3d.toml` | `mhd3d` | the 3D Orszag--Tang vortex |
| `examples/toml/taylor_green_3d.toml` | `mhd3d` | Taylor--Green vortex, insulating class |
| `examples/toml/alfven_collision_3d.toml` | `mhd3d` | counterpropagating Alfven waves in a guide field |
| `examples/toml/compressible_orszag_tang_3d.toml` | `compressible` | subsonic compressible Orszag--Tang in a thin box |

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

Use `mhx physics equilibria` to list equilibria, both the two-dimensional
ones and the three-dimensional builders (`orszag_tang_3d`, `taylor_green_i`,
`taylor_green_a`, `taylor_green_c`, `abc_flow`, `alfven_wave_collision`,
`cp_alfven`, `single_mode_3d`), and `mhx physics list` to list terms. An
unknown key in `[physics.equilibrium_parameters]` is an error that names the
valid keys, and list-valued parameters such as `mode = [1, 0, 2]` are
supported. `mhx physics lint <name>` validates a registered term against
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
parameter regime match a [validated gate](../validation/index.md).

Use `mhx diagnostics list` for names and output keys, and
[add diagnostics](../how_to/extend.md) for the registry.

## Reproducible command

```bash
mhx run examples/linear_tearing_hyper.toml --outdir outputs/linear_tearing_hyper
mhx figures outputs/linear_tearing_hyper --gif
mhx report outputs/linear_tearing_hyper
```

Expected model-audit fields in `outputs/linear_tearing_hyper/diagnostics.json`
include `equilibrium`, `equilibrium_parameters`, and `physics_terms`. The
[output schema](output_schema.md) documents every file a run writes.
