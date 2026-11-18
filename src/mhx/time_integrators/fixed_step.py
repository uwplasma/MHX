"""Fixed-step differentiable time integrators."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, TypeVar

import jax
import jax.numpy as jnp

from mhx.state import ReducedMHDTrajectory

StateT = TypeVar("StateT")


def checkpointed_steps(
    step: Callable[[StateT], StateT], state: StateT, count: int, checkpoint: bool = True
) -> StateT:
    """Apply ``step`` ``count`` times in ``jax.checkpoint`` blocks of about sqrt(count).

    Reverse mode stores one state per block and recomputes a block when it
    needs it, so gradient memory grows as ``sqrt(count)`` instead of
    linearly, for about one extra forward pass (a plain gradient costs about
    2.5 forward passes, a checkpointed one about 3.7). ``checkpoint=False``
    stores every step instead, for gradients whose tape fits in memory.
    """

    def body(carry: StateT, _: Any) -> tuple[StateT, None]:
        return step(carry), None

    if not checkpoint:
        return jax.lax.scan(body, state, None, length=count)[0]

    block = math.isqrt(count)
    blocks, tail = divmod(count, block)

    @jax.checkpoint
    def block_body(carry: StateT, _: Any) -> tuple[StateT, None]:
        return jax.lax.scan(body, carry, None, length=block)[0], None

    state = jax.lax.scan(block_body, state, None, length=blocks)[0]
    return jax.lax.scan(body, state, None, length=tail)[0] if tail else state


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
) -> ReducedMHDTrajectory:
    """Evolve a state with RK4 and retain selected steps.

    The implementation advances ``save_every`` internal steps per saved sample,
    so long runs store only the returned trajectory rather than every internal
    RK4 step. The final state is always included, even when ``steps`` is not an
    exact multiple of ``save_every``. Returned times are absolute and start
    after ``t0``.

    Args:
        state0: State at ``t0``.
        rhs: Function that returns the time derivative.
        dt: Positive fixed step.
        steps: Number of time steps.
        save_every: Retain a state after this many steps.
        t0: Initial time.

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

    def saved_step(carry: StateT, save_index: Any) -> tuple[StateT, StateT]:
        del save_index
        next_state, _ = jax.lax.scan(inner_step, carry, None, length=stride)
        return next_state, next_state

    final_state, saved_states = jax.lax.scan(
        saved_step,
        state0,
        jnp.arange(full_saved_count),
    )
    step_numbers = stride * jnp.arange(1, full_saved_count + 1)
    if remainder:
        final_state, _ = jax.lax.scan(inner_step, final_state, None, length=remainder)
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
