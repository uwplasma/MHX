"""Batched Fourier evaluation and Newton refinement on the active JAX device."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from mhx.diagnostics.critical_points import PeriodicFluxInterpolator


@jax.jit
def _evaluate(coefficients, kx, ky, lower, lengths, mean, positions):
    coordinates = (positions - lower) % lengths
    ex = jnp.exp(1j * coordinates[:, :1] * kx)
    ey = jnp.exp(1j * coordinates[:, 1:] * ky)
    # Contract one axis at a time instead of constructing a full 2-D basis
    # for each point. The large contraction becomes a batched matrix product.
    intermediate = jnp.einsum("bi,kij->kbj", ex, coefficients)
    values = jnp.einsum("kbj,bj->bk", intermediate, ey).real
    return values.at[:, 0].add(mean)


@partial(jax.jit, static_argnames=("max_iterations",))
def _refine(
    coefficients,
    kx,
    ky,
    lower,
    lengths,
    mean,
    positions,
    spacing,
    fraction,
    tolerance,
    max_iterations,
):
    def evaluate(points):
        return _evaluate(coefficients, kx, ky, lower, lengths, mean, points)

    def condition(state):
        iteration, _, _, pending = state
        return (iteration < max_iterations) & pending.any()

    def step(state):
        iteration, points, counts, pending = state
        samples = evaluate(points)
        gx, gy, hxx, hxy, hyy = [samples[:, i] for i in range(1, 6)]
        determinant = hxx * hyy - hxy**2
        safe = jnp.where(determinant != 0, determinant, 1.0)
        delta = jnp.stack(((-hyy * gx + hxy * gy) / safe, (hxy * gx - hxx * gy) / safe), axis=1)
        residual = jnp.hypot(gx, gy)
        converged = (residual == 0) | (
            (residual <= tolerance) & (jnp.max(jnp.abs(delta), axis=1) <= 1e-8 * spacing.min())
        )
        pending = pending & ~converged & (determinant != 0) & jnp.isfinite(delta).all(axis=1)
        delta = jnp.clip(delta, -fraction * spacing, fraction * spacing)
        updated = lower + (points + delta - lower) % lengths
        points = jnp.where(pending[:, None], updated, points)
        return iteration + 1, points, counts + pending.astype(jnp.int32), pending

    _, positions, counts, _ = jax.lax.while_loop(
        condition,
        step,
        (0, positions, jnp.zeros(len(positions), jnp.int32), jnp.ones(len(positions), bool)),
    )
    # Independently evaluate the final field, rather than reporting a linearized residual.
    return positions, counts, evaluate(positions)


class JaxPeriodicFluxInterpolator(PeriodicFluxInterpolator):
    """Same Fourier interpolant as NumPy, evaluated in bounded device batches."""

    batch_size = 64

    def __init__(self, values, lengths=(2.0 * np.pi,) * 2, lower=(0.0, 0.0)):
        if not jax.config.x64_enabled:
            raise ValueError("JAX Fourier refinement requires enable_x64=True")
        super().__init__(values, lengths, lower)
        self.device_arguments = tuple(
            jnp.asarray(a)
            for a in (
                self.coefficients,
                self.kx,
                self.ky,
                self.lower,
                self.lengths,
                self.mean,
            )
        )

    def sample_values(self, positions):
        arguments = (self.device_arguments[0][:1], *self.device_arguments[1:])
        return self._batches(positions, lambda batch: _evaluate(*arguments, batch))[:, 0]

    def _batches(self, positions, operation):
        positions = np.asarray(positions, dtype=float).reshape((-1, 2))
        chunks = []
        for start in range(0, len(positions), self.batch_size):
            batch = positions[start : start + self.batch_size]
            # A fixed shape reuses the compiled kernels for every frame and tail batch.
            padded = np.pad(batch, ((0, self.batch_size - len(batch)), (0, 0)), mode="edge")
            chunks.append(np.asarray(operation(jnp.asarray(padded)))[: len(batch)])
        return np.concatenate(chunks) if chunks else np.empty((0, 1))

    def refine_many(self, positions, *, spacing, fraction, tolerance, max_iterations):
        positions = np.asarray(positions, dtype=float).reshape((-1, 2))
        results = []
        for start in range(0, len(positions), self.batch_size):
            batch = positions[start : start + self.batch_size]
            padded = np.pad(batch, ((0, self.batch_size - len(batch)), (0, 0)), mode="edge")
            result = _refine(
                *self.device_arguments,
                jnp.asarray(padded),
                jnp.asarray(spacing),
                fraction,
                tolerance,
                max_iterations,
            )
            results.append(tuple(np.asarray(a)[: len(batch)] for a in result))
        if not results:
            return np.empty((0, 2)), np.empty(0, int), np.empty((0, 6))
        return tuple(np.concatenate([result[i] for result in results]) for i in range(3))
