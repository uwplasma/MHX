# Autodiff against central differences

Differentiate a reduced-MHD solve with respect to the resistivity, and check
the result against central differences at several step sizes.

## Set up

Everything comes from `mhx`. Enable float64 first. Float32 finite differences
carry percent-level noise and cannot confirm a derivative.

```python
import math

import jax
import jax.numpy as jnp

from mhx.equations.reduced_mhd import reduced_mhd_rhs_spectral, to_spectral_state
from mhx.grids import CartesianGrid
from mhx.physics import CosineTearingEquilibrium
from mhx.state import ReducedMHDParams
from mhx.time_integrators import evolve_rk4

jax.config.update("jax_enable_x64", True)

grid = CartesianGrid(
    shape=(32, 32), lower=(0.0, 0.0), upper=(2.0 * math.pi, 2.0 * math.pi)
)
state0 = to_spectral_state(CosineTearingEquilibrium().initial_state(grid))
```

Define a scalar loss. Here it is the final magnetic energy proxy after a short
RK4 solve:

```python
def loss(resistivity):
    params = ReducedMHDParams(resistivity=resistivity, viscosity=5.0e-3)

    def rhs(state):
        return reduced_mhd_rhs_spectral(state, params, lengths=grid.lengths)

    trajectory = evolve_rk4(state0, rhs, dt=2.0e-2, steps=20, save_every=20)
    return jnp.sum(jnp.abs(trajectory.states.psi[-1]) ** 2)
```

## Run

`jax.grad` differentiates through the whole solve. Compare it with central
differences over a range of `eps`:

```python
eta = jnp.asarray(5.0e-3)
value, gradient = jax.value_and_grad(loss)(eta)

for eps in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
    fd = (loss(eta + eps) - loss(eta - eps)) / (2.0 * eps)
    print(f"eps={eps:.0e}  |FD - AD| = {abs(fd - gradient):.3e}")
```

The error falls as `eps` shrinks, because the truncation error is proportional
to `eps**2`. It then rises again once round-off in `loss` dominates. Autodiff has
no step size to tune.

The repository ships this as `examples/gallery/08_gradient.py`:

```bash
python examples/gallery/08_gradient.py
```

## Scale up

Seed a single tearing mode instead of the cosine sheet and differentiate the
growth rate with respect to `eta`. Each extra parameter costs one more
finite-difference pair per gradient, while the reverse-mode gradient stays at one
backward pass.
