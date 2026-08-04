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

The TOML API remains available for recorded campaigns:

```python
manifest = mhx.run("examples/linear_tearing.toml", outdir="outputs/config_run")
config = mhx.load_config("examples/linear_tearing.toml")
```

The generated reference below lists public arguments and return values.

```{eval-rst}
.. automodule:: mhx.simulation
   :members:

.. automodule:: mhx.ensemble
   :members:

.. automodule:: mhx.parallel
   :members:

.. automodule:: mhx.config
   :members:

.. automodule:: mhx.grids
   :members:

.. automodule:: mhx.numerics.spectral
   :members:

.. automodule:: mhx.equations.reduced_mhd
   :members:
   :exclude-members: Array

.. automodule:: mhx.diagnostics
   :members:

.. automodule:: mhx.benchmarks
   :members:

.. automodule:: mhx.benchmarks.theory
   :members:

.. automodule:: mhx.benchmarks.decay
   :members:
   :exclude-members: Array

.. automodule:: mhx.benchmarks.scaling
   :members:

.. automodule:: mhx.benchmarks.timing
   :members:

.. automodule:: mhx.neural_ode
   :members:

.. automodule:: mhx.neural_ode.reproducibility
   :members:

.. automodule:: mhx.campaigns
   :members:

.. automodule:: mhx.campaigns.production
   :members:

.. automodule:: mhx.physics
   :members:

.. automodule:: mhx.physics.equilibria
   :members:

.. automodule:: mhx.physics.terms
   :members:

.. automodule:: mhx.io
   :members:

.. automodule:: mhx.versioning
   :members:
```
