# Model assembly

MHX run configs assemble a reduced-MHD model from explicit, auditable pieces:

1. mesh and time controls;
2. an equilibrium builder;
3. base resistive-viscous reduced-MHD coefficients;
4. optional RHS physics terms;
5. diagnostic definitions.

The current active model is `reduced_mhd_linear_tearing`. It evolves

$$
\partial_t \psi = -[\phi,\psi] + \eta\nabla^2\psi + S_\psi,
$$

$$
\partial_t \omega = -[\phi,\omega] + [\psi,\nabla^2\psi]
                  + \nu\nabla^2\omega + S_\omega,
$$

with $\nabla^2\phi=\omega$. The optional source terms
$(S_\psi,S_\omega)$ are the sum of configured physics plugins.

## TOML structure

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

This is intentionally explicit. Saved run directories include
`config_effective.json`, `diagnostics.json`, and `manifest.json`, so a figure or
benchmark report can be traced back to the exact assembled model.

Diagnostics are assembled the same way:

```toml
[diagnostics]
quantities = ["energy", "mode_growth", "divergence_error"]
plugin_modules = []
plugin_entry_point_groups = []
mode = [1, 1]
fit_time_window = [0.02, 0.1]
```

Use `mhx diagnostics list` to inspect names and output keys. The built-in
registry is documented in [Diagnostics registry](diagnostics.md).

`plugin_modules` accepts importable Python modules exposing
`register_physics(registry)` or `register_diagnostics(registry)`. The demo
`examples/linear_tearing_plugin_demo.toml` shows a local physics term and a local
diagnostic assembled entirely from TOML.

`plugin_entry_point_groups` accepts installed Python package entry-point groups,
usually `mhx.physics` and `mhx.diagnostics`. This is the reviewer-facing path
for third-party plugins because the run manifest records the exact discovery
groups in addition to the selected term and diagnostic names.

## Built-in equilibria

Use `mhx physics equilibria` to inspect available equilibria.

`cosine_tearing` initializes a periodic tearing smoke problem:

$$
\psi(x,y,0)=\cos(y)+\epsilon\cos(x)\cos(y), \qquad \omega(x,y,0)=0.
$$

`zero` initializes $\psi=\omega=0$ and is primarily for plugin tests and IO
validation.

## Built-in RHS terms

Use `mhx physics list` to inspect available terms and
`mhx physics lint <name>` to validate a registered term against the active API.

The current built-ins are:

- `hyper_resistivity`: fourth-order damping of $\psi$ and $\omega$.
- `vorticity_drag`: linear damping of $\omega$.
- `electron_pressure_tensor`: anisotropic current-smoothing closure motivated
  by the pressure-divergence term in generalized Ohm's law.
- `toy_hall_ohm`: reduced-state Hall-like bracket for exercising two-fluid
  plugin wiring.

These are deliberately simple terms. They test the extension path without
claiming validated Hall, anisotropic-pressure, or two-fluid reconnection
physics. The pressure-tensor and Hall examples are useful templates for API
extension, but reviewer-facing physics claims require additional evolved fields
and validation against Hall/two-fluid benchmarks.

## Reproducible command

```bash
mhx run examples/linear_tearing_hyper.toml --outdir outputs/linear_tearing_hyper
mhx figures outputs/linear_tearing_hyper --gif
mhx report outputs/linear_tearing_hyper
```

Expected model-audit fields in `outputs/linear_tearing_hyper/diagnostics.json`
include `equilibrium`, `equilibrium_parameters`, and `physics_terms`.
