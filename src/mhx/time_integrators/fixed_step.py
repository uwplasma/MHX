"""Fixed-step differentiable time integrators."""

from __future__ import annotations

import math
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
    """Advance one classical fourth-order Runge--Kutta step.

    Args:
        state: Current state as any JAX PyTree.
        rhs: Function that returns the time derivative of ``state``.
        dt: Positive time-step size.

    Returns:
        The state after one step.
    """
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
    t0: float = 0.0,
    checkpoint: bool = True,
) -> ReducedMHDTrajectory:
    """Evolve a state with RK4 and retain selected steps.

    The implementation advances ``save_every`` internal steps per saved sample,
    so long runs store only the returned trajectory rather than every internal
    RK4 step. Each save interval runs as checkpointed blocks of about
    ``sqrt(save_every)`` steps, so reverse-mode memory grows with the square
    root of the interval instead of linearly, at the cost of one extra forward
    pass in the gradient (a plain gradient costs about 2.5 forward passes, a
    checkpointed one about 3.7). Pass ``checkpoint=False`` when the full tape
    fits in memory. The final state is always included, even when
    ``steps`` is not an exact multiple of ``save_every``. Returned times are
    absolute and start after ``t0``.

    Args:
        state0: State at ``t0``.
        rhs: Function that returns the time derivative.
        dt: Positive fixed step.
        steps: Number of time steps.
        save_every: Retain a state after this many steps.
        t0: Initial time.
        checkpoint: Recompute blocks in reverse mode instead of storing every step.

    Returns:
        Saved times and states.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if save_every < 1:
        raise ValueError("save_every must be >= 1")
    stride = min(save_every, steps)
    full_saved_count = steps // stride
    remainder = steps % stride

    def inner_step(carry: StateT, step_index: Any) -> tuple[StateT, None]:
        del step_index
        return rk4_step(carry, rhs, dt), None

    def advance(carry: StateT, length: int) -> StateT:
        if not checkpoint:
            return jax.lax.scan(inner_step, carry, None, length=length)[0]
        block = math.isqrt(length)
        blocks, tail = divmod(length, block)

        @jax.checkpoint
        def block_step(state: StateT, block_index: Any) -> tuple[StateT, None]:
            del block_index
            return jax.lax.scan(inner_step, state, None, length=block)[0], None

        carry = jax.lax.scan(block_step, carry, None, length=blocks)[0]
        return jax.lax.scan(inner_step, carry, None, length=tail)[0] if tail else carry

    def saved_step(carry: StateT, save_index: Any) -> tuple[StateT, StateT]:
        del save_index
        next_state = advance(carry, stride)
        return next_state, next_state

    final_state, saved_states = jax.lax.scan(
        saved_step,
        state0,
        jnp.arange(full_saved_count),
    )
    step_numbers = stride * jnp.arange(1, full_saved_count + 1)
    if remainder:
        final_state = advance(final_state, remainder)
        saved_states = jax.tree.map(
            lambda saved, final: jnp.concatenate((saved, final[None, ...]), axis=0),
            saved_states,
            final_state,
        )
        step_numbers = jnp.concatenate(
            (step_numbers, jnp.asarray([steps], dtype=step_numbers.dtype))
        )
    times = t0 + dt * step_numbers
    return ReducedMHDTrajectory(times=times, states=saved_states)
