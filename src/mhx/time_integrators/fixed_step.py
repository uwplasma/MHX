"""Fixed-step differentiable time integrators."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple, TypeVar

import jax
import jax.numpy as jnp
from jaxtyping import Array

StateT = TypeVar("StateT")


class Trajectory(NamedTuple):
    """Saved samples for a fixed-step PyTree state trajectory."""

    times: Array
    states: Any


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
) -> Trajectory:
    """Evolve a state with RK4 and save every ``save_every`` steps.

    The implementation advances ``save_every`` internal steps per saved sample,
    so long runs store only the returned trajectory rather than every internal
    RK4 step. If ``steps`` is not an exact multiple of ``save_every``, the
    returned trajectory matches the historical API by omitting the unsaved tail.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if save_every < 1:
        raise ValueError("save_every must be >= 1")
    stride = min(save_every, steps)
    saved_count = steps // stride

    def inner_step(carry: StateT, step_index: Any) -> tuple[StateT, None]:
        del step_index
        return rk4_step(carry, rhs, dt), None

    def saved_step(carry: StateT, save_index: Any) -> tuple[StateT, StateT]:
        del save_index
        next_state, _ = jax.lax.scan(inner_step, carry, None, length=stride)
        return next_state, next_state

    _, saved_states = jax.lax.scan(saved_step, state0, jnp.arange(saved_count))
    times = dt * stride * jnp.arange(1, saved_count + 1)
    return Trajectory(times=times, states=saved_states)
