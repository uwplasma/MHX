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

The reference splits by layer:

```{toctree}
:maxdepth: 1

core
physics
numerics
diagnostics
benchmarks
io
```
