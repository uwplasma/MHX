# API reference

Start a Python run with three public objects:

```python
import mhx

simulation = mhx.Simulation(
    equilibrium=mhx.PeriodicDoubleHarrisEquilibrium(),
)
result = simulation.run()
result.print_summary()
```

Use `Simulation` for settings, an equilibrium object for initial fields, and
`SimulationResult` for output.

The TOML API remains available for recorded runs:

```python
manifest = mhx.run("examples/linear_tearing.toml", outdir="outputs/config_run")
config = mhx.load_config("examples/linear_tearing.toml")
```

## Core: simulation, ensembles, config, grids

Build a `Simulation`, run it, and read the `SimulationResult`. `run_ensemble`
batches independent cases. `RunConfig` and `load_config` support the TOML path.
`CartesianGrid` holds the periodic domain.

```{eval-rst}
.. automodule:: mhx.simulation
   :members:

.. automodule:: mhx.ensemble
   :members:

.. automodule:: mhx.config
   :members:

.. automodule:: mhx.grids
   :members:
```

## Physics: equilibria, terms, equations

Equilibria build initial fields on a grid. Physics terms add optional
right-hand-side contributions through the plugin registry. The equations module
holds the reduced-MHD right-hand side and its spectral form.

```{eval-rst}
.. automodule:: mhx.physics
   :members:

.. automodule:: mhx.physics.equilibria
   :members:

.. automodule:: mhx.physics.terms
   :members:

.. automodule:: mhx.equations.reduced_mhd
   :members:
   :exclude-members: Array
```

## Numerics: spectral operators, time integrators, devices

Spectral operators compute Fourier derivatives, the inverse Laplacian, and
dealiasing filters. Time integrators advance a state with RK4 or backward Euler.
The parallel module builds device meshes and shards fields.

```{eval-rst}
.. automodule:: mhx.numerics.spectral
   :members:

.. automodule:: mhx.time_integrators
   :members:

.. automodule:: mhx.parallel
   :members:
```

## Diagnostics

Diagnostics are tested functions over states and trajectories: energies, mode
amplitudes, divergence checks, and flux critical points.

```{eval-rst}
.. automodule:: mhx.diagnostics
   :members:
```

## Benchmarks

These modules back the `mhx benchmark` commands. Most users reach them through
the CLI. The [CLI reference](cli.md) documents the commands themselves.

```{eval-rst}
.. automodule:: mhx.benchmarks
   :members:

.. automodule:: mhx.benchmarks.theory
   :members:

.. automodule:: mhx.benchmarks.decay
   :members:
   :exclude-members: Array

.. automodule:: mhx.benchmarks.scaling
   :members:
```

## Input, output, and versioning

The IO module reads and writes trajectory NPZ files and run manifests. The
versioning module records the public API version in every artifact.

```{eval-rst}
.. automodule:: mhx.io
   :members:

.. automodule:: mhx.versioning
   :members:
```
