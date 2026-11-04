# Extend the physics and diagnostics

Add an equilibrium or a right-hand-side term without editing MHX source.
MHX exposes two small composability layers:

- named equilibria that build initial reduced-MHD states;
- versioned physics terms that add right-hand-side extensions.

The active physics-term API string is `mhx.physics.v1`.

List built-in equilibria and terms:

```bash
mhx physics equilibria
mhx physics list
mhx physics lint hyper_resistivity
mhx physics list-with-plugins --plugin-module examples.local_extension_plugin
mhx physics list-with-plugins --entry-point-group mhx.physics
```

## Equilibrium contract

An equilibrium implements:

```python
def initial_state(grid: CartesianGrid) -> ReducedMHDState:
    ...
```

The built-in `cosine_tearing` equilibrium uses

$$
\psi(x,y,0)=\cos(y) + \epsilon\cos(x)\cos(y), \qquad \omega(x,y,0)=0,
$$

on the configured periodic domain. The `zero` equilibrium is useful for plugin
and IO tests.

Select equilibria from TOML:

```toml
[physics]
equilibrium = "cosine_tearing"

[physics.equilibrium_parameters]
perturbation_amplitude = 1e-3
```

## Term contract

A reduced-MHD physics term implements:

```python
def rhs_addition(
    state: ReducedMHDState,
    params: ReducedMHDParams,
    *,
    lengths: tuple[float, float],
) -> ReducedMHDState:
    ...
```

The returned `ReducedMHDState` is added to the base reduced-MHD RHS. Terms must
not mutate inputs. They should be JAX-compatible and should avoid hidden global
state.

## Local plugin modules

Users can now load local extension modules directly from TOML. A physics plugin
module must expose:

```python
def register_physics(registry):
    registry.register("my_term", my_factory)
```

The factory receives the TOML parameter mapping and returns an object satisfying
the `PhysicsTerm` protocol. `examples/local_extension_plugin.py` is a minimal
working example. It registers `example_flux_drive`, a small deterministic
`cos(x)cos(y)` source for extension tutorials:

```toml
[physics]
plugin_modules = ["examples.local_extension_plugin"]
rhs_terms = ["example_flux_drive"]

[physics.term_parameters.example_flux_drive]
amplitude = 1e-5
```

Run the end-to-end demo:

```bash
mhx run examples/linear_tearing_plugin_demo.toml --outdir outputs/plugin_demo
mhx figures outputs/plugin_demo --gif
mhx report outputs/plugin_demo
```

The saved `diagnostics.json` records `physics_plugin_modules` and
`physics_terms`, so reviewer artifacts show exactly which extensions were active.

## Installed package entry points

Third-party packages can expose plugins without requiring users to place modules
on `PYTHONPATH`. MHX recognizes Python package entry-point groups:

- `mhx.physics`: entry points resolving to `register(registry)` hooks or modules
  exposing `register_physics(registry)`;
- `mhx.diagnostics`: entry points resolving to `register(registry)` hooks or
  modules exposing `register_diagnostics(registry)`.

A package can declare entry points in its `pyproject.toml`:

```toml
[project.entry-points."mhx.physics"]
my_hall_term = "my_mhx_plugin.physics:register_physics"

[project.entry-points."mhx.diagnostics"]
my_metric = "my_mhx_plugin.diagnostics:register_diagnostics"
```

Use them from TOML by opting into the groups:

```toml
[physics]
plugin_entry_point_groups = ["mhx.physics"]
rhs_terms = ["my_hall_term"]

[diagnostics]
plugin_entry_point_groups = ["mhx.diagnostics"]
quantities = ["energy", "mode_growth", "my_metric"]
```

The CLI can inspect and lint installed plugins:

```bash
mhx physics list-with-plugins --entry-point-group mhx.physics
mhx physics lint my_hall_term --entry-point-group mhx.physics
mhx diagnostics list-with-plugins --entry-point-group mhx.diagnostics
mhx diagnostics lint my_metric --entry-point-group mhx.diagnostics
```

## Template package layout

`examples/plugin_template/` is a minimal third-party package skeleton:

```text
examples/plugin_template/
  pyproject.toml
  src/mhx_example_plugin/
    physics.py
    diagnostics.py
  tests/test_plugin.py
```

The template declares both entry-point groups and can be installed in editable
mode:

```bash
cd examples/plugin_template
python -m pip install -e .
mhx physics lint template_flux_sink --entry-point-group mhx.physics
mhx diagnostics lint template_final_psi_mean --entry-point-group mhx.diagnostics
```

Use it as the starting point for an external repository. Keep extension tests
next to the plugin package; they should validate shapes, finite RHS additions,
metadata, sign conventions, and at least one FAST MHX run when practical.

## Built-in terms

`hyper_resistivity` adds

$$
\partial_t \psi \leftarrow \partial_t\psi - \eta_4 \nabla^4\psi,
\qquad
\partial_t \omega \leftarrow \partial_t\omega - \nu_4 \nabla^4\omega.
$$

`vorticity_drag` adds

$$
\partial_t \omega \leftarrow \partial_t\omega - \alpha\omega.
$$

`electron_pressure_tensor` is a reduced-state pressure-tensor closure inspired
by the $z$ component of generalized Ohm's law,

$$
E_z + [\phi,\psi] = \eta j_z
 - {1\over n_e e}\left(\partial_x P_{exz}+\partial_y P_{eyz}\right),
\qquad j_z=-\nabla^2\psi.
$$

The implemented toy closure models the pressure-divergence contribution as
anisotropic current smoothing:

$$
\partial_t\psi \leftarrow \partial_t\psi
 + \chi_x\partial_{xx}j_z + \chi_y\partial_{yy}j_z.
$$

`toy_hall_ohm` adds a reduced-state Hall-like bracket

$$
\partial_t\psi \leftarrow \partial_t\psi + d_i [j_z,\psi].
$$

This bracket is included to exercise two-fluid plugin wiring. It is not a
closed Hall-MHD model because the current reduced-MHD state does not evolve the
extra out-of-plane magnetic/velocity fields needed by standard reduced Hall
MHD closures.

The motivation is the generalized Ohm's-law structure used in GEM/Hall
reconnection studies, including the
[GEM reconnection challenge](https://www.mendeley.com/catalogue/92e8f29f-a6d8-3c8d-a0fa-b24bf4cb8c88/),
[Hall-term reconnection comparisons](https://ftp.bartol.udel.edu/whm/GEM/GEM-reconnection/shayEA-JGR-106-3759-2001.pdf),
and modern
[Ohm's-law reviews](https://arxiv.org/abs/2406.00875). MHX currently exposes
toy reduced closures first so the extension API is testable before making
stronger physics claims.

These terms are intentionally simple extension examples. They are useful for
testing the plugin path and for toy regularization/two-fluid experiments; they
are not yet validated extended-MHD reconnection models.

## TOML assembly

Users can assemble terms from config:

```toml
[physics]
equilibrium = "cosine_tearing"
rhs_terms = ["hyper_resistivity", "vorticity_drag", "electron_pressure_tensor"]

[physics.equilibrium_parameters]
perturbation_amplitude = 1e-3

[physics.term_parameters.hyper_resistivity]
eta4 = 1e-5
nu4 = 1e-5

[physics.term_parameters.vorticity_drag]
rate = 0.01

[physics.term_parameters.electron_pressure_tensor]
chi_x = 1e-5
chi_y = 2e-5
```

The examples `examples/linear_tearing_hyper.toml` and
`examples/linear_tearing_twofluid_toy.toml` demonstrate this path.

## Adding a term

For in-tree terms:

1. Add a dataclass implementing `rhs_addition`.
2. Register it in `default_physics_registry()`.
3. Add unit tests for sign, shape, finite values, and config assembly.
4. Add docs with equations and limitations.
5. Run `mhx physics lint <name>`, `pytest`, and `ruff`.

Config-loaded local modules are the current stable third-party extension path.
Package entry-point discovery is now available for installed plugins. Local
`plugin_modules` remain the simplest development path because they work from a
single source tree without packaging.

## Adding an equilibrium

For in-tree equilibria:

1. Add a dataclass implementing `initial_state(grid)`.
2. Register it in `default_equilibrium_registry()`.
3. Add a small TOML example if the parameters are user-facing.
4. Add tests for shape, finite values, deterministic output, and CLI listing.
5. Run `mhx physics equilibria`, `pytest`, `ruff`, and a FAST run.

## Add diagnostics

MHX diagnostics are centralized in `mhx.diagnostics` so runs, reports,
benchmarks, and future scan/inverse-design workflows use the same definitions.
TOML configs select diagnostics through `[diagnostics].quantities`:

```toml
[diagnostics]
quantities = ["energy", "mode_growth", "divergence_error"]
mode = [1, 1]
fit_time_window = [0.02, 0.1]
```

Inspect registered diagnostics from the CLI:

```bash
mhx diagnostics list
mhx diagnostics list-with-plugins --plugin-module examples.local_extension_plugin
mhx diagnostics list-with-plugins --entry-point-group mhx.diagnostics
```

### Built-in diagnostics

| Name | Output keys |
| --- | --- |
| `energy` | `initial_total_energy`, `final_total_energy`, `final_magnetic_energy`, `final_kinetic_energy` |
| `mode_growth` | `diagnostic_mode`, `fit_time_window`, `fit_sample_count`, `initial_mode_amplitude`, `final_mode_amplitude`, `gamma_fit` |
| `divergence_error` | `final_magnetic_divergence_linf` |

### Energy definitions

For the current 2D reduced-MHD state variables $\psi$ and $\omega=\nabla^2\phi$,
MHX computes

$$
E_B = \frac{1}{2}\langle |\nabla\psi|^2\rangle,
\qquad
E_K = \frac{1}{2}\langle |\nabla\phi|^2\rangle,
\qquad
E = E_B + E_K.
$$

The trajectory diagnostic reports final magnetic/kinetic energy and initial/final
total energy. The benchmark validation report uses these fields to gate
unphysical total-energy growth in the current dissipative FAST benchmark.

### Mode growth

For a configured Fourier mode $(k_x,k_y)$, MHX stores

$$
A_k(t) = |\hat\psi_{k_x,k_y}(t)|.
$$

The fitted growth/decay rate is the least-squares slope of
$\log A_k(t)$ against time over the configured inclusive window:

$$
\gamma_{\mathrm{fit}}
=
\frac{\sum_i (t_i-\bar t)(\log A_i-\overline{\log A})}
       {\sum_i (t_i-\bar t)^2}.
$$

This scalar is a reproducibility diagnostic for individual saved trajectories.
It is not, by itself, a calibrated FKR eigenmode claim. Calibrated tearing
checks live in the separate Harris eigenvalue, dispersion, layer, and
time-domain replay gates documented on [validation.md](../validation/index.md).

### Reconnected flux and island-width proxies

MHX exposes low-level island diagnostics as pure functions rather than default
run-summary fields, because a credible nonlinear island claim also needs a
resolved separatrix, a calibrated equilibrium shear, and a long enough time
window. For a real single-harmonic perturbation

$$
\psi_1(x,y,t)=\tilde\psi_1(t)\cos(k\cdot x),
$$

`mode_amplitude` returns $|\hat\psi_k|=|\tilde\psi_1|/2$. The helper
`reconnected_flux_amplitude(state, mode=...)` returns $2|\hat\psi_k|$, so the
cosine amplitude is recovered exactly for single-mode tests. Given local
magnetic shear $B_y'(0)$, the Rutherford full-width proxy is

$$
W = 4\sqrt{\frac{|\tilde\psi_1|}{|B_y'(0)|}}.
$$

Use `island_width_from_mode(state, mode=..., magnetic_shear=...)` for quick
post-processing, then gate any publication claim with the nonlinear duration
audit and a resolution/time-step convergence study. The unit tests verify that
the proxy recovers a known cosine amplitude and rejects nonpositive shear.

### Divergence error

The reduced-MHD perpendicular magnetic field is represented as

$$
B_\perp = (\partial_y\psi,\,-\partial_x\psi),
$$

so analytically $\nabla\cdot B_\perp=0$. MHX reports the final spectral
consistency check

$$
\|\nabla\cdot B_\perp\|_\infty
=
\|\partial_x\partial_y\psi - \partial_y\partial_x\psi\|_\infty.
$$

For smooth periodic fields this should be near roundoff. The unit tests include
a spectral-zero gate for this diagnostic.

### Flux critical points

README media and turbulent-reconnection diagnostics use magnetic-flux critical
points from
`mhx.diagnostics.detect_flux_critical_points` and
`mhx.diagnostics.critical_points_by_kind`. The detector finds local minima of
$|\nabla\psi|$ and classifies them with the Hessian determinant:

$$
D = \psi_{xx}\psi_{yy} - \psi_{xy}^2 .
$$

Points with $D<0$ are marked as X-points; points with $D>0$ are marked as
O-points. Passing `refine=True` applies one local quadratic Newton correction,
which reduces grid-locking for smooth fields:

$$
\delta x = -H(\psi)^{-1}\nabla\psi,\qquad
x_\mathrm{refined}=x_\mathrm{grid}+\delta x .
$$

`mhx.diagnostics.track_critical_points` then greedily links detected points
across frames by nearest neighbor and point kind. This is still a validation and
visualization diagnostic rather than a full topological event solver: it does
not model separatrix reconnection events, mergers, or bifurcations.

Source links:

- [Critical-point implementation](https://github.com/uwplasma/MHX/blob/main/src/mhx/diagnostics/critical_points.py)
- [Critical-point tests](https://github.com/uwplasma/MHX/blob/main/tests/test_critical_points.py)

### Python extension API

Use `DiagnosticSpec` and `DiagnosticsRegistry` for new diagnostics:

```python
from mhx.diagnostics import DiagnosticSpec, default_diagnostics_registry

def compute_my_metric(context):
    return {"my_metric": 0.0}

registry = default_diagnostics_registry()
registry.register(
    DiagnosticSpec(
        name="my_metric",
        description="Example user metric.",
        output_keys=("my_metric",),
        compute=compute_my_metric,
    )
)
```

The callable receives a `DiagnosticContext` with the saved trajectory, initial
state, domain lengths, diagnostic Fourier mode, and fit-time window.

Diagnostics can also provide optional report figures by passing a `figure`
callable. The hook receives the same `DiagnosticContext`, the scalar diagnostic
dictionary, and an output directory. It should write deterministic files and
return a mapping from figure key to path:

```python
from pathlib import Path

def plot_my_metric(context, diagnostics, figure_dir: Path):
    path = figure_dir / "my_metric.png"
    # write figure to path
    return {"my_metric_history": path}

registry.register(
    DiagnosticSpec(
        name="my_metric",
        description="Example metric with a report figure.",
        output_keys=("my_metric",),
        compute=compute_my_metric,
        figure=plot_my_metric,
    )
)
```

### Config-loaded diagnostics plugins

A local module can expose diagnostics without modifying MHX source code:

```python
from mhx.diagnostics import DiagnosticSpec

def compute_metric(context):
    return {"my_metric": 0.0}

def register_diagnostics(registry):
    registry.register(
        DiagnosticSpec(
            name="my_metric",
            description="Example metric.",
            output_keys=("my_metric",),
            compute=compute_metric,
        )
    )
```

Then enable it in TOML:

```toml
[diagnostics]
quantities = ["energy", "mode_growth", "my_metric"]
plugin_modules = ["my_project.mhx_diagnostics"]
```

Installed packages can alternatively expose diagnostics through the
`mhx.diagnostics` entry-point group:

```toml
[diagnostics]
quantities = ["energy", "mode_growth", "my_metric"]
plugin_entry_point_groups = ["mhx.diagnostics"]
```

The demo `examples/linear_tearing_plugin_demo.toml` loads
`examples.local_extension_plugin`, which registers `final_flux_l2`. The output
schema records `diagnostic_plugin_modules` and `diagnostic_quantities` to keep
extension-derived figures auditable.

`mhx report <run-dir>` now includes an **Additional scalar diagnostics** section
for scalar keys that come from plugin diagnostics, for example
`final_flux_l2` in the local plugin demo.

Reports also reconstruct diagnostic registry metadata from
`config_effective.json`. When plugin modules or entry points are still importable
at report time, `report.json` contains `diagnostic_metadata`, and `report.md`
renders a table of selected diagnostic descriptions and output keys. If a plugin
cannot be imported, the report still writes but records a warning.

If a selected diagnostic defines a figure hook, `mhx figures <run-dir>` writes
the files under `figures/diagnostics/`. `mhx report <run-dir>` dispatches the
same hooks and records them in `report.json` as `diagnostic_figures`. The local
plugin example writes
`figures/diagnostics/final_flux_l2_history.png`, which is checked in CI as part
of the FAST artifact pipeline.

### Source links

- [Diagnostics implementation](https://github.com/uwplasma/MHX/blob/main/src/mhx/diagnostics/reduced_mhd.py)
- [Diagnostic figure dispatch](https://github.com/uwplasma/MHX/blob/main/src/mhx/plotting/diagnostics.py)
- [Diagnostics tests](https://github.com/uwplasma/MHX/blob/main/tests/test_reduced_mhd.py)
- [Plugin-module tests](https://github.com/uwplasma/MHX/blob/main/tests/test_plugin_modules.py)
- [Local plugin example](https://github.com/uwplasma/MHX/blob/main/examples/local_extension_plugin.py)
- [Run integration](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/tearing.py)
