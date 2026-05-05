# Physics plugins

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
