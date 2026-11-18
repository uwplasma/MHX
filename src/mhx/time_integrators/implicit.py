"""Implicit fixed-step integration with SOLVAX."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple, TypeVar

import jax
import jax.numpy as jnp
import solvax
from jaxtyping import Array

from mhx.state import ReducedMHDTrajectory

StateT = TypeVar("StateT")


class ImplicitTrajectoryResult(NamedTuple):
    """Saved implicit trajectory and per-save-interval solver diagnostics."""

    trajectory: ReducedMHDTrajectory
    residual_norms: Array
    nonlinear_iterations: Array
    linear_iterations: Array
    converged: Array
    linear_converged: Array


def backward_euler_step(
    state: StateT,
    rhs: Callable[[StateT], StateT],
    dt: float,
    *,
    preconditioner: Callable[[StateT], StateT] | None = None,
    rtol: float = 1.0e-9,
    atol: float = 1.0e-11,
    max_steps: int = 20,
    linear_restart: int = 30,
    linear_max_restarts: int = 10,
) -> solvax.NewtonKrylovSolution:
    """Advance one backward-Euler step with Jacobian-free Newton–Krylov."""
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    def residual(candidate: StateT) -> StateT:
        candidate_rhs = rhs(candidate)
        return jax.tree.map(
            lambda next_value, old_value, derivative: next_value - old_value - dt * derivative,
            candidate,
            state,
            candidate_rhs,
        )

    return solvax.newton_krylov(
        residual,
        state,
        precond=preconditioner,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        linear_restart=linear_restart,
        linear_max_restarts=linear_max_restarts,
    )


def evolve_backward_euler(
    state0: StateT,
    rhs: Callable[[StateT], StateT],
    *,
    dt: float,
    steps: int,
    save_every: int = 1,
    t0: float = 0.0,
    preconditioner: Callable[[StateT], StateT] | None = None,
    rtol: float = 1.0e-9,
    atol: float = 1.0e-11,
    max_steps: int = 20,
    linear_restart: int = 30,
    linear_max_restarts: int = 10,
) -> ImplicitTrajectoryResult:
    """Evolve a PyTree with backward Euler and retain solver diagnostics."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if save_every < 1:
        raise ValueError("save_every must be >= 1")
    stride = min(save_every, steps)
    full_saved_count = steps // stride
    remainder = steps % stride

    def inner_step(
        carry: StateT,
        step_index: Array,
    ) -> tuple[StateT, solvax.NewtonKrylovSolution]:
        del step_index
        solution = backward_euler_step(
            carry,
            rhs,
            dt,
            preconditioner=preconditioner,
            rtol=rtol,
            atol=atol,
            max_steps=max_steps,
            linear_restart=linear_restart,
            linear_max_restarts=linear_max_restarts,
        )
        return solution.x, solution

    def saved_step(
        carry: StateT,
        save_index: Array,
    ) -> tuple[StateT, tuple[StateT, Array, Array, Array, Array, Array]]:
        del save_index
        next_state, interval = jax.lax.scan(inner_step, carry, None, length=stride)
        return next_state, (
            next_state,
            jnp.max(interval.residual_norm),
            jnp.sum(interval.newton_iterations),
            jnp.sum(interval.linear_iterations),
            jnp.all(interval.converged),
            jnp.all(interval.linear_converged),
        )

    final_state, saved = jax.lax.scan(
        saved_step,
        state0,
        jnp.arange(full_saved_count),
    )
    (
        saved_states,
        residual_norms,
        nonlinear_iterations,
        linear_iterations,
        converged,
        linear_converged,
    ) = saved
    step_numbers = stride * jnp.arange(1, full_saved_count + 1)

    if remainder:
        final_state, interval = jax.lax.scan(inner_step, final_state, None, length=remainder)
        saved_states = jax.tree.map(
            lambda states, final: jnp.concatenate((states, final[None, ...]), axis=0),
            saved_states,
            final_state,
        )
        residual_norms = jnp.concatenate((residual_norms, jnp.max(interval.residual_norm)[None]))
        nonlinear_iterations = jnp.concatenate(
            (
                nonlinear_iterations,
                jnp.sum(interval.newton_iterations)[None],
            )
        )
        linear_iterations = jnp.concatenate(
            (linear_iterations, jnp.sum(interval.linear_iterations)[None])
        )
        converged = jnp.concatenate((converged, jnp.all(interval.converged)[None]))
        linear_converged = jnp.concatenate(
            (linear_converged, jnp.all(interval.linear_converged)[None])
        )
        step_numbers = jnp.concatenate(
            (step_numbers, jnp.asarray([steps], dtype=step_numbers.dtype))
        )

    trajectory = ReducedMHDTrajectory(
        times=t0 + dt * step_numbers,
        states=saved_states,
    )
    return ImplicitTrajectoryResult(
        trajectory=trajectory,
        residual_norms=residual_norms,
        nonlinear_iterations=nonlinear_iterations,
        linear_iterations=linear_iterations,
        converged=converged,
        linear_converged=linear_converged,
    )
