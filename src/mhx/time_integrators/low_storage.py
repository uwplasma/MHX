"""Low-storage RK3 with exact integrating factors for diagonal dissipation.

The scheme is Williamson's two-register RK3 applied to the transformed
variable :math:`w(t) = e^{-L(t - t_n)} u(t)`, where ``L`` is the diagonal
dissipation operator. The transform integrates the stiff linear part
exactly, so only the advective time scale limits the step, and a run with
zero nonlinearity reproduces the analytic decay to round-off.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import jax
import jax.numpy as jnp

from mhx.state.mhd3d import MHD3DState, MHD3DTrajectory

StateT = TypeVar("StateT")

# Williamson (1980) two-register coefficients, Wray's RK3 variant.
_A = (0.0, -5.0 / 9.0, -153.0 / 128.0)
_B = (1.0 / 3.0, 15.0 / 16.0, 8.0 / 15.0)
_C = (0.0, 1.0 / 3.0, 3.0 / 4.0)


def _scale(state: MHD3DState, factor: MHD3DState) -> MHD3DState:
    return jax.tree.map(lambda leaf, f: leaf * f, state, factor)


def _exp_factor(decay: MHD3DState, time: float | jax.Array) -> MHD3DState:
    return jax.tree.map(lambda rate: jnp.exp(-rate * time), decay)


def if_rk3_step(
    state: MHD3DState,
    nonlinear: Callable[[MHD3DState], MHD3DState],
    decay: MHD3DState,
    dt: float,
) -> MHD3DState:
    """Advance one step of integrating-factor Williamson RK3.

    Args:
        state: Spectral state at the step start.
        nonlinear: Nonadvective-stiffness-free right-hand side.
        decay: State-shaped nonnegative decay rates, ``nu k^2`` and
            ``eta k^2`` leaves.
        dt: Positive time step.

    Returns:
        The state one step later. Pure dissipation is integrated exactly.
    """
    w = state
    q = jax.tree.map(jnp.zeros_like, state)
    for a_i, b_i, c_i in zip(_A, _B, _C, strict=True):
        forward = _exp_factor(decay, c_i * dt)
        backward = jax.tree.map(lambda rate, c=c_i: jnp.exp(rate * c * dt), decay)
        rhs = _scale(nonlinear(_scale(w, forward)), backward)
        q = jax.tree.map(lambda qq, rr, a=a_i: a * qq + dt * rr, q, rhs)
        w = jax.tree.map(lambda ww, qq, b=b_i: ww + b * qq, w, q)
    return _scale(w, _exp_factor(decay, dt))


def evolve_if_rk3(
    state0: MHD3DState,
    nonlinear: Callable[[MHD3DState], MHD3DState],
    decay: MHD3DState,
    *,
    dt: float,
    steps: int,
    save_every: int = 1,
    t0: float = 0.0,
) -> MHD3DTrajectory:
    """Evolve with :func:`if_rk3_step` and retain every ``save_every`` step.

    The final state is always included, matching :func:`mhx.time_integrators
    .evolve_rk4`. Returned times are absolute.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if save_every < 1:
        raise ValueError("save_every must be at least 1")

    chunks, remainder = divmod(steps, save_every)

    def advance(state: MHD3DState, count: int) -> MHD3DState:
        def body(carry: MHD3DState, _: None) -> tuple[MHD3DState, None]:
            return if_rk3_step(carry, nonlinear, decay, dt), None

        advanced, _ = jax.lax.scan(body, state, None, length=count)
        return advanced

    def chunk_body(carry: MHD3DState, _: None) -> tuple[MHD3DState, MHD3DState]:
        advanced = advance(carry, save_every)
        return advanced, advanced

    state_final, saved = jax.lax.scan(chunk_body, state0, None, length=chunks)
    times = t0 + dt * save_every * jnp.arange(1, chunks + 1)
    if remainder:
        state_final = advance(state_final, remainder)
        saved = jax.tree.map(
            lambda tail, last: jnp.concatenate([tail, last[None]], axis=0),
            saved,
            state_final,
        )
        times = jnp.concatenate([times, jnp.asarray([t0 + dt * steps])])
    return MHD3DTrajectory(times=times, states=saved)
