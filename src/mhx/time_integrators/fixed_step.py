"""Fixed-step differentiable time integrators."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import jax
import jax.numpy as jnp

from mhx.state import ReducedMHDTrajectory

StateT = TypeVar("StateT")


def _tree_add(left: StateT, right: StateT) -> StateT:
    return jax.tree_util.tree_map(lambda x, y: x + y, left, right)


def _tree_scale(scale: float, tree: StateT) -> StateT:
    return jax.tree_util.tree_map(lambda value: scale * value, tree)


def _tree_add_scaled(left: StateT, scale: float, right: StateT) -> StateT:
    return _tree_add(left, _tree_scale(scale, right))


def rk4_step(state: StateT, rhs: Callable[[StateT], StateT], dt: float) -> StateT:
    """Advance one Runge-Kutta 4 step for a PyTree state."""
    k1 = rhs(state)
    k2 = rhs(_tree_add_scaled(state, 0.5 * dt, k1))
    k3 = rhs(_tree_add_scaled(state, 0.5 * dt, k2))
    k4 = rhs(_tree_add_scaled(state, dt, k3))
    increment = jax.tree_util.tree_map(
        lambda a, b, c, d: (dt / 6.0) * (a + 2.0 * b + 2.0 * c + d),
        k1,
        k2,
        k3,
        k4,
    )
    return _tree_add(state, increment)


def evolve_rk4(
    state0: StateT,
    rhs: Callable[[StateT], StateT],
    *,
    dt: float,
    steps: int,
    save_every: int = 1,
) -> ReducedMHDTrajectory:
    """Evolve a state with RK4 and save every ``save_every`` steps."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if save_every < 1:
        raise ValueError("save_every must be >= 1")
    stride = min(save_every, steps)

    def scan_step(carry: StateT, step_index: Any) -> tuple[StateT, StateT]:
        del step_index
        next_state = rk4_step(carry, rhs, dt)
        return next_state, next_state

    _, states = jax.lax.scan(scan_step, state0, jnp.arange(steps))
    saved_states = jax.tree_util.tree_map(lambda values: values[stride - 1 :: stride], states)
    times = dt * jnp.arange(stride, steps + 1, stride)
    return ReducedMHDTrajectory(times=times, states=saved_states)
