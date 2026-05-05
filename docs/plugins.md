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

These are intentionally simple extension examples. They are useful for testing
the plugin path and for toy regularization experiments; they are not yet a
validated extended-MHD model.

## TOML assembly

Users can assemble terms from config:

```toml
[physics]
equilibrium = "cosine_tearing"
rhs_terms = ["hyper_resistivity", "vorticity_drag"]

[physics.equilibrium_parameters]
perturbation_amplitude = 1e-3

[physics.term_parameters.hyper_resistivity]
eta4 = 1e-5
nu4 = 1e-5

[physics.term_parameters.vorticity_drag]
rate = 0.01
```

The example `examples/linear_tearing_hyper.toml` demonstrates this path.

## Adding a term

For in-tree terms:

1. Add a dataclass implementing `rhs_addition`.
2. Register it in `default_physics_registry()`.
3. Add unit tests for sign, shape, finite values, and config assembly.
4. Add docs with equations and limitations.
5. Run `mhx physics lint <name>`, `pytest`, and `ruff`.

Third-party plugin discovery is not enabled yet. The current stable surface is
the protocol, metadata fields, and config-driven in-tree registry.

## Adding an equilibrium

For in-tree equilibria:

1. Add a dataclass implementing `initial_state(grid)`.
2. Register it in `default_equilibrium_registry()`.
3. Add a small TOML example if the parameters are user-facing.
4. Add tests for shape, finite values, deterministic output, and CLI listing.
5. Run `mhx physics equilibria`, `pytest`, `ruff`, and a FAST run.
