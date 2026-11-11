# Orszag--Tang against the Vencels kinetic result

Run the incompressible reduced-MHD Orszag--Tang vortex and compare its energy
history with the kinetic reference data that ships with the package in
`mhx/benchmarks/data/vencelsdata.csv`. Then fit
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

Open `examples/orszag_tang/vencels_kinetic_comparison.py`, edit the constants
under "Input parameters" if you want a different grid or end time, and run it
from the repository root:

```bash
python examples/orszag_tang/vencels_kinetic_comparison.py
```

The script runs the vortex, normalizes the energies by the initial total energy
`W0`, and draws them against the Vencels curves. With the defaults (24 x 24
grid, `T_END = 30`) it takes seconds and writes `kinetic_comparison.png` and
`manifest.json` to `OUTDIR`.

Fit `eta` and `nu` so the fluid energy curves match the reference with
`examples/orszag_tang/autodiff_optimization.py`. The optimizer differentiates
through the whole solve; each step prints the loss and the current `eta` and
`nu`. For a quick look set `NX = 32`, `NY = 32`, `T_END = 0.2`, `SAVE_EVERY = 20`
and `OPT_STEPS = 2` before running:

```bash
python examples/orszag_tang/autodiff_optimization.py
```

## Scale up

In `vencels_kinetic_comparison.py` raise `NX`, `NY` and `T_END`; a
high-resolution comparison used `NX = NY = 1024`, `DT = 5e-4` and `T_END = 30`.
In `autodiff_optimization.py` the defaults are 256 x 256 and `OPT_STEPS = 15`;
`T_END` must be a multiple of `DT * SAVE_EVERY`. Keep `DT` below the explicit
diffusion limit, about `2 / (eta * 2 * (pi * N / 10)**2)`.
