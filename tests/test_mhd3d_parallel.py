"""Sharded-transform contract: parity, distributed gradients, no gathers.

The checks run in a subprocess because the device count must be set before
JAX initializes. One process runs all three assertions to pay the import
cost once.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = """
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jax.sharding import Mesh

from mhx.equations import mhd3d
from mhx.numerics.spectral.pfft import pfft3, pifft3, shard_physical
from mhx.physics.equilibria3d import OrszagTang3DEquilibrium
from mhx.state.mhd3d import MHD3DParams, MHD3DState
from mhx.time_integrators.low_storage import evolve_if_rk3

assert jax.device_count() == 4, jax.devices()
mesh = Mesh(jax.devices(), axis_names=("x",))
shape = (16, 16, 16)
lengths = (2.0 * jnp.pi,) * 3

velocity, magnetic = OrszagTang3DEquilibrium().initial_fields(shape)
k = mhd3d.wavevectors(shape, lengths)

# 1. Forward parity: sharded transform equals the single-device transform.
single = jnp.fft.rfftn(velocity, axes=(-3, -2, -1))
sharded = pfft3(shard_physical(velocity, mesh), mesh=mesh)
parity = float(jnp.max(jnp.abs(single - sharded)))
assert parity < 1.0e-12 * float(jnp.max(jnp.abs(single))), parity
round_trip = pifft3(sharded, shape=shape, mesh=mesh)
rt_error = float(jnp.max(jnp.abs(round_trip - velocity)))
assert rt_error < 1.0e-12, rt_error

# 2. Gradient parity: a loss through the sharded pipeline has the same
# derivative as the single-device pipeline.
mask = mhd3d.two_thirds_mask_rfft(shape)


def loss(viscosity, use_mesh):
    active = mesh if use_mesh else None
    params = MHD3DParams(viscosity=viscosity, resistivity=5.0e-3)
    state0 = MHD3DState(
        v_hat=mhd3d.project(pfft3(velocity, mesh=active), k),
        b_hat=mhd3d.project(pfft3(magnetic, mesh=active), k),
    )
    decay = mhd3d.decay_rates(params, k)

    def nonlinear(state):
        return mhd3d.mhd3d_nonlinear(
            state, params, shape=shape, k=k, mask=mask, mesh=active
        )

    trajectory = evolve_if_rk3(
        state0, nonlinear, decay, dt=5.0e-3, steps=4, save_every=4
    )
    final = jax.tree.map(lambda leaf: leaf[-1], trajectory.states)
    return mhd3d.energies(final, shape=shape)["total"]


nu = jnp.asarray(2.0e-2)
grad_single = jax.grad(lambda v: loss(v, False))(nu)
grad_sharded = jax.grad(lambda v: loss(v, True))(nu)
gap = abs(float(grad_single - grad_sharded)) / abs(float(grad_single))
assert gap < 1.0e-10, gap

# 3. No field-sized all-gather in the compiled sharded transform, forward
# or backward.
def fwd(field):
    return jnp.sum(jnp.abs(pfft3(field, mesh=mesh)) ** 2)


sharded_field = shard_physical(velocity, mesh)
for name, fn in (("forward", fwd), ("grad", jax.grad(fwd))):
    hlo = jax.jit(fn).lower(sharded_field).compile().as_text()
    assert "all-gather" not in hlo, f"{name} compiles to an all-gather"
    assert "all-to-all" in hlo, f"{name} lost the transpose collective"

print("PARALLEL-OK")
"""


def test_sharded_transform_contract() -> None:
    env = dict(os.environ)
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    env["JAX_PLATFORM_NAME"] = "cpu"
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert "PARALLEL-OK" in result.stdout
