# Run your first model

This guide runs a seeded periodic current sheet. It prints the result, writes a
figure, and saves each retained field.

## 1. Install MHX

From the repository root, run:

```bash
python -m pip install -e .
```

The base install includes SOLVAX and the plotting tools.

## 2. Create the simulation

Create a file named `first_run.py`:

```python
from pathlib import Path

import mhx

output = Path("outputs/first_run")

simulation = mhx.Simulation(
    shape=(64, 64),
    equilibrium=mhx.PeriodicDoubleHarrisEquilibrium(
        width=0.4,
        perturbation_amplitude=4.0e-3,
        perturbation_mode=(2, 1),
    ),
    resistivity=5.0e-3,
    viscosity=5.0e-3,
    dt=2.0e-2,
    t_end=2.0,
    save_every=10,
)
```

`shape` sets the global periodic grid. Resistivity diffuses magnetic flux.
Viscosity diffuses vorticity. The time settings produce 100 RK4 steps and save
10 states.

## 3. Run and inspect it

Add these lines:

```python
result = simulation.run()
result.print_summary()
```

The first call compiles and runs the JAX program. The second call prints timing,
energy, and divergence data.

Access the arrays directly when you need them:

```python
times = result.trajectory.times
flux_history = result.trajectory.states.psi
vorticity_history = result.trajectory.states.omega
final_flux = result.final_state.psi
```

The field history uses the shape `(saved_time, nx, ny)`.

## 4. Plot and save it

Add these lines:

```python
figure = result.plot(output / "summary.png")
data = result.save(output)

print(f"Figure: {figure}")
print(f"Data:   {data}")
```

The NPZ file contains time, flux, vorticity, settings, diagnostics, MHX
version, and API version.

## 5. Run the file

Run:

```bash
python first_run.py
```

Open `outputs/first_run/summary.png`. Load the saved fields with:

```python
from mhx.io import read_reduced_mhd_trajectory_npz

trajectory, diagnostics = read_reduced_mhd_trajectory_npz(
    "outputs/first_run/trajectory.npz"
)
```

## Use an implicit step

Set the integrator before you run:

```python
simulation = mhx.Simulation(
    shape=(32, 32),
    integrator="backward_euler",
    dt=1.0e-2,
    t_end=5.0e-2,
)
```

SOLVAX does the Newton and GMRES work. After the run, check:

```python
print(result.diagnostics["implicit_converged"])
print(result.diagnostics["implicit_linear_converged"])
```

Both values must be `True` before you use the result.

## Use several CPU devices

Run the CPU example:

```bash
python examples/gallery/04_cpu_parallel.py
```

The script creates four logical CPU devices before JAX starts. It splits the
first grid axis across those devices.

For other models, set `device_count` on `mhx.Simulation`. Make sure that
`shape[0]` divides evenly by that count.

## Continue

Use the [example gallery](https://github.com/uwplasma/MHX/tree/main/examples/gallery)
for tearing, implicit steps, CPU sharding, GPU sharding, and scaling. Use
[`mhx --help`](benchmarks.md) when you need validation or campaign commands.
