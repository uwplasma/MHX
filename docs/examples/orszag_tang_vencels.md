# Orszag--Tang against the Vencels kinetic result

Run the incompressible reduced-MHD Orszag--Tang vortex and compare its energy
history with the kinetic reference data in `examples/vencelsdata.csv`. Then fit
the resistivity and viscosity to that reference by autodiff.

## Set up

```bash
python -m pip install -e ".[research]"
```

Build the vortex on a 10 x 10 periodic box and evolve it with RK4:

```python
import jax

from mhx.benchmarks.orszag_tang import orszag_tang_initial_state
from mhx.equations.reduced_mhd import reduced_mhd_rhs
from mhx.grids import CartesianGrid
from mhx.state import ReducedMHDParams
from mhx.time_integrators import evolve_rk4

jax.config.update("jax_enable_x64", True)

grid = CartesianGrid(shape=(64, 64), lower=(0.0, 0.0), upper=(10.0, 10.0))
state0 = orszag_tang_initial_state(grid)
params = ReducedMHDParams(resistivity=1.0e-2, viscosity=1.0e-2)


def rhs(state):
    return reduced_mhd_rhs(state, params, lengths=grid.lengths)


trajectory = evolve_rk4(state0, rhs, dt=5.0e-3, steps=400, save_every=20)
print(trajectory.states.psi.shape)
```

## Run

The comparison script runs the vortex, normalizes the energies by the initial
total energy `W0`, and draws them against the Vencels curves:

```bash
python examples/orszag_tang/vencels_kinetic_comparison.py \
  --nx 32 --ny 32 --t-end 2 --outdir outputs/orszag_tang_kinetic
```

It writes `kinetic_comparison.png` and `manifest.json` to `--outdir`.

Fit `eta` and `nu` so the fluid energy curves match the reference. The
optimizer differentiates through the whole solve:

```bash
python examples/orszag_tang/autodiff_optimization.py \
  --nx 32 --ny 32 --t-end 0.2 --save-every 20 --opt-steps 2 \
  --outdir outputs/orszag_tang_opt
```

Each step prints the loss and the current `eta` and `nu`.

## Scale up

Raise `--nx`, `--ny`, `--t-end` and `--opt-steps`. The defaults are 256 x 256
and 15 optimization steps. A high-resolution comparison used `--nx 1024 --ny 1024
--dt 5e-4 --t-end 30`. Keep `dt` below the explicit diffusion limit, about
`2 / (eta * 2 * (pi * N / 10)**2)`.
