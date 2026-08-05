# Run your first model

This guide runs a seeded periodic current sheet, prints the result, writes a
figure, and saves the field history. It takes under a minute on a laptop CPU.

## 1. Create the simulation

Install MHX ([installation](install.md)), then create `first_run.py`:

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

Each setting has a physical meaning:

- `shape` sets the periodic grid. The domain is a $2\pi \times 2\pi$ square.
- The equilibrium builds two opposite [Harris current sheets](../physics/reduced_mhd.md#periodic-double-harris-sheet)
  and seeds them with a small flux perturbation.
- `resistivity=5.0e-3` sets the Lundquist number $S = 200$. Resistivity
  diffuses magnetic flux, viscosity diffuses vorticity.
- The time settings produce 100 RK4 steps and keep every tenth state.

## 2. Run and inspect it

Add these lines and run the file:

```python
result = simulation.run()
result.print_summary()
```

The first call compiles the whole time loop with JAX, then runs it. The
summary table reports timing, energy, and the divergence check:

```text
│ Compile time   │      0.276 s │
│ Run time       │      0.065 s │
│ Initial energy │ 3.726865e-01 │
│ Final energy   │ 3.625520e-01 │
│ Max |div B|    │    0.000e+00 │
```

Energy decreases because the run is resistive. The divergence is exactly zero
because the [spectral method](../physics/spectral_method.md) evolves the flux
function, not the field components.

Access the arrays directly when you need them:

```python
times = result.trajectory.times
flux_history = result.trajectory.states.psi
vorticity_history = result.trajectory.states.omega
final_flux = result.final_state.psi
```

The field history has shape `(saved_time, nx, ny)`.

## 3. Plot and save it

```python
figure = result.plot(output / "summary.png")
data = result.save(output)
```

`plot` writes a four-panel figure: initial flux, final flux, final current
density, and the energy history. `save` writes one compressed NPZ file with
the fields, settings, diagnostics, and version metadata. Reload it later
with:

```python
from mhx.io import read_reduced_mhd_trajectory_npz

trajectory, diagnostics = read_reduced_mhd_trajectory_npz(
    "outputs/first_run/trajectory.npz"
)
```

## 4. See the physics

A `t_end=2.0` run barely moves the fields, because tearing growth is slow at
$S=200$. To watch the island grow, extend the run and isolate the island
flux. Subtracting the $y$-averaged profile removes both the equilibrium and
its slow resistive spreading:

```python
import numpy as np

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
    t_end=40.0,
    save_every=100,
)
result = simulation.run()

flux = np.asarray(result.trajectory.states.psi)
island = flux - flux.mean(axis=2, keepdims=True)
```

The total flux is dominated by the current sheets, so plots of `psi` alone
look frozen. The island flux shows the reconnecting mode directly: its peak
grows from about `0.006` to `0.013` over this run.
[Make your first movie](first_movie.md) animates it.

## Continue

- [Make your first movie](first_movie.md)
- [Run from a TOML config](../how_to/run_from_toml.md)
- [The reduced-MHD model](../physics/reduced_mhd.md)
- [Differentiate a run](../physics/differentiability.md)
