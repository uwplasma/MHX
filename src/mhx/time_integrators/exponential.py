"""ETDRK4 for diagonal linear operators, the validation cross-check stepper.

Cox and Matthews (2002) fourth-order exponential time differencing, with
the phi functions evaluated stably through ``expm1`` because the linear
operator is diagonal. Montanelli and Bootland (2020) find ETDRK4 hard to
beat for periodic stiff PDEs. MHX uses it to cross-check the production
low-storage IF-RK3 on the wave gates: two independent steppers agreeing to
their respective orders is a stronger statement than either alone.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from mhx.state.mhd3d import MHD3DState, MHD3DTrajectory


def _phi_factors(decay_leaf: jax.Array, dt: float) -> tuple[jax.Array, ...]:
    """Return ``(E, E2, Q, f1, f2, f3)`` for one diagonal decay-rate leaf.

    The Cox--Matthews coefficients for ``L = -decay`` with ``z = L dt``
    suffer catastrophic cancellation near ``z = 0`` (they divide an
    O(eps)-cancelled numerator by ``z^3``). Kassam and Trefethen (2005)
    evaluate them as means over a contour around each ``z``; with 32
    points on the unit circle the result is accurate to round-off for the
    whole ``z <= 0`` range used here, at setup-only cost.
    """
    z = (-decay_leaf * dt).astype(jnp.result_type(float))
    exp_z = jnp.exp(z)
    exp_half = jnp.exp(0.5 * z)

    points = 32
    theta = jnp.pi * (jnp.arange(1, points + 1) - 0.5) / points
    circle = jnp.exp(1j * theta)
    zc = z[..., None] + circle
    zc3 = zc * zc * zc
    exp_zc = jnp.exp(zc)

    def contour_mean(values: jax.Array) -> jax.Array:
        return jnp.real(jnp.mean(values, axis=-1))

    q = dt * contour_mean((jnp.exp(zc / 2.0) - 1.0) / zc)
    f1 = dt * contour_mean(
        (-4.0 - zc + exp_zc * (4.0 - 3.0 * zc + zc * zc)) / zc3
    )
    f2 = dt * contour_mean((2.0 + zc + exp_zc * (-2.0 + zc)) / zc3)
    f3 = dt * contour_mean(
        (-4.0 - 3.0 * zc - zc * zc + exp_zc * (4.0 - zc)) / zc3
    )

    return exp_z, exp_half, q, f1, f2, f3


def etdrk4_step(
    state: MHD3DState,
    nonlinear: Callable[[MHD3DState], MHD3DState],
    decay: MHD3DState,
    dt: float,
) -> MHD3DState:
    """Advance one Cox--Matthews ETDRK4 step with diagonal dissipation."""
    factors_v = _phi_factors(decay.v_hat, dt)
    factors_b = _phi_factors(decay.b_hat, dt)

    def pick(index: int) -> MHD3DState:
        return MHD3DState(v_hat=factors_v[index], b_hat=factors_b[index])

    exp_full, exp_half, phi_half, f1, f2, f3 = (pick(i) for i in range(6))

    def mul(left: MHD3DState, right: MHD3DState) -> MHD3DState:
        return jax.tree.map(jnp.multiply, left, right)

    def add(left: MHD3DState, right: MHD3DState) -> MHD3DState:
        return jax.tree.map(jnp.add, left, right)

    n_state = nonlinear(state)
    a = add(mul(exp_half, state), mul(phi_half, n_state))
    n_a = nonlinear(a)
    b = add(mul(exp_half, state), mul(phi_half, n_a))
    n_b = nonlinear(b)
    c = add(mul(exp_half, a), mul(phi_half, jax.tree.map(lambda x, y: 2.0 * x - y, n_b, n_state)))
    n_c = nonlinear(c)

    term1 = mul(f1, n_state)
    term2 = mul(f2, jax.tree.map(lambda x, y: 2.0 * (x + y), n_a, n_b))
    term3 = mul(f3, n_c)
    return add(mul(exp_full, state), add(term1, add(term2, term3)))


def evolve_etdrk4(
    state0: MHD3DState,
    nonlinear: Callable[[MHD3DState], MHD3DState],
    decay: MHD3DState,
    *,
    dt: float,
    steps: int,
    save_every: int = 1,
    t0: float = 0.0,
) -> MHD3DTrajectory:
    """Evolve with :func:`etdrk4_step`, mirroring ``evolve_if_rk3``."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if save_every < 1:
        raise ValueError("save_every must be at least 1")

    chunks, remainder = divmod(steps, save_every)

    def advance(state: MHD3DState, count: int) -> MHD3DState:
        def body(carry: MHD3DState, _: None) -> tuple[MHD3DState, None]:
            return etdrk4_step(carry, nonlinear, decay, dt), None

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
