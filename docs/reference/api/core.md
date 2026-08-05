# Core: simulation, ensembles, config, grids

The objects on this page cover the main run path: build a `Simulation`, run
it, and read the `SimulationResult`. `run_ensemble` batches independent cases.
`RunConfig` and `load_config` support the TOML path. `CartesianGrid` holds the
periodic domain.

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
