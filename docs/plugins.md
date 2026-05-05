# Physics plugins

MHX exposes a small versioned physics-term API for additive right-hand-side
extensions. The active API string is `mhx.physics.v1`.

List built-in terms:

```bash
mhx physics list
mhx physics lint hyper_resistivity
```

## Contract

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
rhs_terms = ["hyper_resistivity", "vorticity_drag"]

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

